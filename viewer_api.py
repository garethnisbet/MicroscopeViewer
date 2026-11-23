"""Remote API for a running ImageVisualiser.

The viewer process starts a small line-oriented JSON server::

    from image_visualiser import launch
    v = launch()
    v.serve()                       # 127.0.0.1:8765

Any other Python process then reads its Hough and fit parameters::

    from viewer_api import connect
    with connect() as remote:
        remote.hough_results                # HoughResult
        remote.hough_params()               # detector settings
        remote.fit_results["top"]           # FitResult
        remote.fit_params()["top"]          # the fitted Gaussians

and sets them::

        remote.set_hough_params(param2=28, min_dist=40)
        remote.set_fit_params(method="Gradient", n_gaussians=2)
        remote.run_fit("both")               # waits for the result

Every setter returns the viewer's resulting parameters, so the caller
sees what actually took effect.  Serve with ``allow_writes=False`` to
publish a viewer read-only.

It also takes frames and hands back images, which is what a scan needs::

        remote.grab()                        # one camera frame, analysed
        remote.wait_for_frame(frames=2)      # ...or wait on the live stream
        snap = remote.snapshot(include_image=True)
        snap["frame_id"], snap["hough"], snap["image"]

``frame_id`` counts every image the viewer loads, so results and the
image they came from can be tied to a moment after, say, a stage move.

The client half needs only the standard library and NumPy -- importing
it does not pull in PyQt5, OpenCV or SciPy -- so it can run in a bare
environment, a different virtualenv, or on another machine.

Requests are answered on the viewer's Qt main thread, so a reply is a
consistent snapshot and the GUI is never touched from a socket thread.
The corollary is that the viewer's event loop must be running -- under
``launch()`` in IPython, or ``app.exec_()`` in a script.  Without one,
calls fail with :class:`ViewerAPIError` after the client timeout.
"""

from __future__ import annotations

import base64
import json
import socket
import socketserver
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from results import HoughResult, FitResult

__all__ = [
    "serve",
    "stop_serving",
    "server_port",
    "connect",
    "RemoteViewer",
    "ViewerAPIError",
    "DEFAULT_PORT",
]

DEFAULT_PORT = 8765
_ENCODING = "utf-8"


class ViewerAPIError(RuntimeError):
    """Raised when a remote call fails or the viewer cannot answer."""


# ---------------------------------------------------------------------------
# Main-thread dispatch
# ---------------------------------------------------------------------------

class _MainThreadDispatcher:
    """Runs callables on the Qt main thread and returns their result.

    Socket handlers run on their own threads, and Qt widgets may only be
    touched from the thread that owns them.  Each handler therefore
    queues a callable here and blocks; a ``QTimer`` on the main thread
    drains the queue, so every reply is also a snapshot taken between
    GUI events rather than midway through one.
    """

    def __init__(self, interval_ms: int = 20):
        self._queue = deque()
        self._timer = None
        self._interval_ms = interval_ms

    def start(self):
        from PyQt5 import QtCore
        if self._timer is not None:
            return
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._drain)
        self._timer.start(self._interval_ms)

    def stop(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._drain()  # release anyone still waiting

    def _drain(self):
        while self._queue:
            fn, box, done = self._queue.popleft()
            try:
                box["result"] = fn()
            except Exception as exc:              # returned to the caller
                box["error"] = exc
            finally:
                done.set()

    def call(self, fn, timeout: float = 5.0):
        box: dict = {}
        done = threading.Event()
        self._queue.append((fn, box, done))
        if not done.wait(timeout):
            raise ViewerAPIError(
                f"viewer did not respond within {timeout:g}s -- is its Qt "
                f"event loop running?"
            )
        if "error" in box:
            raise box["error"]
        return box["result"]


# ---------------------------------------------------------------------------
# Server side (runs inside the viewer process)
# ---------------------------------------------------------------------------

_registry = None    # callable() -> {name: viewer}, set by serve()


def _viewer_registry() -> dict:
    """Return the open viewers.

    ``serve()`` is handed the registry by the module that owns it.  That
    matters when the viewer is run as a script: importing
    ``image_visualiser`` from here would then build a *second* copy of
    the module, with its own empty registry, and no viewer would ever be
    found.
    """
    if _registry is not None:
        return _registry()
    from image_visualiser import get_viewers
    return get_viewers()


def _resolve_viewer(name: Optional[str]):
    viewers = _viewer_registry()
    if not viewers:
        raise ViewerAPIError("no viewers are open")
    if name is None:
        if len(viewers) > 1:
            raise ViewerAPIError(
                f"{len(viewers)} viewers are open -- name one of: "
                f"{', '.join(sorted(viewers))}"
            )
        return next(iter(viewers.values()))
    try:
        return viewers[name]
    except KeyError:
        raise ViewerAPIError(
            f"no viewer named {name!r}; open: {', '.join(sorted(viewers))}"
        ) from None


def _encode_array(a) -> dict:
    """Pack an array into JSON: dtype, shape and base64 bytes.

    The viewer's working image is float64 even when it came off an 8-bit
    camera, which would cost eight bytes a pixel on the wire.  When every
    value is a whole number in 0..255 it travels as uint8 and is cast
    back on arrival, so the client still gets exactly what the viewer
    holds.
    """
    a = np.asarray(a)
    wire = a
    if (a.dtype.kind == "f" and a.size and a.min() >= 0 and a.max() <= 255
            and np.array_equal(a, np.rint(a))):
        wire = a.astype(np.uint8)
    wire = np.ascontiguousarray(wire)
    return dict(dtype=a.dtype.str, wire_dtype=wire.dtype.str,
                shape=list(a.shape),
                data=base64.b64encode(wire.tobytes()).decode("ascii"))


def _decode_array(d: dict) -> np.ndarray:
    """Rebuild an array packed by :func:`_encode_array`."""
    wire = np.frombuffer(base64.b64decode(d["data"]),
                         dtype=np.dtype(d["wire_dtype"]))
    return wire.reshape(d["shape"]).astype(np.dtype(d["dtype"]))


def _frame_info(v) -> dict:
    rows, cols = (v._grey.shape if v._grey is not None else (0, 0))
    return dict(
        frame_id=int(v.frame_id),
        frame_time=v.frame_time,
        streaming=bool(v._streaming),
        image_shape=[int(rows), int(cols)],
    )


def _image_payload(v, raw: bool) -> Optional[dict]:
    """The greyscale working image, or with *raw* the image as loaded."""
    img = v._raw if raw else v._grey
    return None if img is None else _encode_array(img)


def _snapshot(v, include_curve: bool, include_image: bool = False,
              raw: bool = False) -> dict:
    hough = v.hough_results
    snap = dict(
        name=v.name,
        **_frame_info(v),
        hough=None if hough is None else hough.to_dict(),
        hough_params=v.hough_params(),
        fits={d: (None if r is None else r.to_dict(include_curve))
              for d, r in v.fit_results.items()},
        fit_params=v.fit_params(),
        fit_busy=bool(v.fit_busy),
    )
    if include_image:
        snap["image"] = _image_payload(v, raw)
    return snap


_FIT_DIRECTIONS = {None: (), "both": ("top", "right"),
                   "top": ("top",), "right": ("right",)}


def _measure(name: Optional[str], args: dict,
             dispatcher: _MainThreadDispatcher, timeout: float) -> dict:
    """A snapshot whose fits are computed for its own frame.

    The frame -- results, projections, image -- is copied on the main
    thread in one go; the fits then run here, on the connection's
    thread.  A slow minimiser therefore neither freezes the GUI (and
    with it every other API call) nor races the next stream frame, which
    would clear the viewer's own fit results.  The fits are returned,
    not drawn.
    """
    fit = args.get("fit")
    if fit not in _FIT_DIRECTIONS:
        raise ViewerAPIError(
            f"fit must be 'top', 'right', 'both' or None, not {fit!r}")

    def on_main():
        v = _resolve_viewer(name)
        snap = _snapshot(v, False, bool(args.get("include_image")),
                         bool(args.get("raw")))
        projections = {"top": v.vertical_proj, "right": v.horizontal_proj}
        copies = {d: (None if projections[d] is None
                      else np.array(projections[d], dtype=np.float64))
                  for d in _FIT_DIRECTIONS[fit]}
        # The viewer's own module, which is __main__ when run as a script.
        return snap, copies, sys.modules[type(v).__module__]

    snap, projections, module = dispatcher.call(on_main, timeout=timeout)
    method = snap["fit_params"]["method"]
    n = snap["fit_params"]["n_gaussians"]
    fits = {}
    for direction, y in projections.items():
        if y is None:
            fits[direction] = None
            continue
        x = np.arange(len(y), dtype=np.float64)
        t0 = time.perf_counter()
        y_fit, popt, msg = module.fit_gaussians(x, y, n, method)
        fits[direction] = module.make_fit_result(
            direction, method, n, y, y_fit, popt, msg,
            (time.perf_counter() - t0) * 1000).to_dict(include_curve=False)
    snap["fits"] = fits
    return {"snapshot": snap}


#: Operations that change the viewer; refused when ``allow_writes`` is off.
WRITE_OPS = frozenset({
    "set_hough_params", "set_fit_params", "run_fit", "cancel_fit",
    "clear_fits", "set_scan_roi", "grab", "start_stream", "stop_stream",
})


def _handle(request: dict, dispatcher: _MainThreadDispatcher,
            timeout: float, allow_writes: bool = True) -> dict:
    op = request.get("op")
    name = request.get("viewer")
    args = request.get("args") or {}

    if op == "ping":
        return {"pong": True, "call_timeout": timeout,
                "allow_writes": allow_writes}

    if op == "list":
        return {"viewers": sorted(_viewer_registry())}

    if op in WRITE_OPS and not allow_writes:
        raise ViewerAPIError(
            f"{op!r} is refused: this viewer is served read-only"
        )

    if op == "measure":
        return _measure(name, args, dispatcher, timeout)

    def on_main():
        v = _resolve_viewer(name)

        # ---- writes ----
        if op == "set_hough_params":
            return {"hough_params": v.set_hough_params(**args)}
        if op == "set_fit_params":
            return {"fit_params": v.set_fit_params(
                method=args.get("method"),
                n_gaussians=args.get("n_gaussians"),
                live_fit=args.get("live_fit"))}
        if op == "run_fit":
            v.run_fit(args.get("direction", "both"))
            return {"fit_busy": bool(v.fit_busy)}
        if op == "cancel_fit":
            v.cancel_fit()
            return {"fit_busy": bool(v.fit_busy)}
        if op == "clear_fits":
            v.clear_fits()
            return {"fit_params": v.fit_params()}
        if op == "set_scan_roi":
            return {"bounds": v.set_scan_roi(args.get("bounds"))}
        if op == "grab":
            if v._streaming:
                raise ViewerAPIError(
                    "the viewer is streaming, which holds the camera -- "
                    "stop the stream to grab, or wait for a stream frame")
            before = v.frame_id
            v._on_snapshot()     # not snapshot(): no nested event processing
            if v.frame_id == before:
                raise ViewerAPIError("could not grab a frame from the camera")
            return {"frame": _frame_info(v)}
        if op == "start_stream":
            v.start_stream()
            if not v._streaming:
                raise ViewerAPIError("could not open the camera to stream")
            return {"frame": _frame_info(v)}
        if op == "stop_stream":
            v.stop_stream()
            return {"frame": _frame_info(v)}

        # ---- reads ----
        if op == "frame":
            return {"frame": _frame_info(v)}
        if op == "image":
            return {"frame": _frame_info(v),
                    "image": _image_payload(v, bool(args.get("raw")))}
        if op == "fit_busy":
            return {"fit_busy": bool(v.fit_busy)}
        if op == "hough":
            h = v.hough_results
            return {"hough": None if h is None else h.to_dict()}
        if op == "hough_params":
            return {"hough_params": v.hough_params()}
        if op == "fit":
            include_curve = bool(args.get("include_curve", True))
            direction = args.get("direction")
            if direction is None:
                return {"fits": {
                    d: (None if r is None else r.to_dict(include_curve))
                    for d, r in v.fit_results.items()}}
            if direction not in v.fit_results:
                raise ViewerAPIError(
                    f"unknown direction {direction!r}; "
                    f"valid: {', '.join(v.fit_results)}"
                )
            r = v.fit_results[direction]
            return {"fit": None if r is None else r.to_dict(include_curve)}
        if op == "fit_params":
            return {"fit_params": v.fit_params()}
        if op == "snapshot":
            return {"snapshot": _snapshot(
                v, bool(args.get("include_curve", False)),
                bool(args.get("include_image", False)),
                bool(args.get("raw", False)))}
        raise ViewerAPIError(f"unknown op {op!r}")

    return dispatcher.call(on_main, timeout=timeout)


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        server = self.server
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line.decode(_ENCODING))
                reply = {"ok": True,
                         "result": _handle(request, server.dispatcher,
                                           server.call_timeout,
                                           server.allow_writes)}
            except Exception as exc:
                reply = {"ok": False,
                         "error": f"{type(exc).__name__}: {exc}"}
            try:
                self.wfile.write(
                    (json.dumps(reply) + "\n").encode(_ENCODING))
                self.wfile.flush()
            except OSError:
                return  # client hung up


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


_server: Optional[_Server] = None
_server_thread: Optional[threading.Thread] = None
_dispatcher: Optional[_MainThreadDispatcher] = None


def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1",
          call_timeout: float = 5.0, allow_writes: bool = True,
          registry=None) -> int:
    """Start the API server in this process and return its port.

    All viewers open in the process are served; the client picks one by
    name.  Pass ``port=0`` to let the OS choose a free port.  Calling
    this again while a server is running is a no-op that returns the
    running port.

    *allow_writes* controls whether clients may change the viewer;
    ``False`` serves it read-only.

    *host* defaults to loopback, so the API is not reachable off this
    machine.  Binding it to ``"0.0.0.0"`` hands the viewer to anything
    that can reach the port -- there is no authentication, and with
    writes enabled that includes control of its parameters.

    *registry* is a zero-argument callable returning ``{name: viewer}``;
    it defaults to ``image_visualiser.get_viewers``.  Pass it explicitly
    from the module that holds the viewers, as
    :meth:`ImageVisualiser.serve` does.
    """
    global _server, _server_thread, _dispatcher, _registry
    if registry is not None:
        _registry = registry
    if _server is not None:
        return _server.server_address[1]

    _dispatcher = _MainThreadDispatcher()
    _dispatcher.start()

    _server = _Server((host, port), _RequestHandler)
    _server.dispatcher = _dispatcher
    _server.call_timeout = call_timeout
    _server.allow_writes = bool(allow_writes)
    _server_thread = threading.Thread(
        target=_server.serve_forever, name="viewer-api", daemon=True)
    _server_thread.start()
    return _server.server_address[1]


def stop_serving():
    """Stop the API server, if one is running."""
    global _server, _server_thread, _dispatcher, _registry
    if _server is None:
        _registry = None
        return
    _server.shutdown()
    _server.server_close()
    if _server_thread is not None:
        _server_thread.join(timeout=2.0)
    if _dispatcher is not None:
        _dispatcher.stop()
    _server, _server_thread, _dispatcher, _registry = None, None, None, None


def server_port() -> Optional[int]:
    """Return the port the API is serving on, or ``None`` if stopped."""
    return None if _server is None else _server.server_address[1]


# ---------------------------------------------------------------------------
# Client side (runs in the other process)
# ---------------------------------------------------------------------------

class RemoteViewer:
    """Client for a viewer served by :func:`serve`.

    The accessors mirror the local viewer's, so code written against
    :class:`~image_visualiser.ImageVisualiser` reads the same::

        viewer.hough_results.circles        # local
        remote.hough_results.circles        # remote

    Each access is a round trip, so grab :meth:`snapshot` instead when
    several values must describe the same frame.

    Setters return the viewer's resulting parameters, so the caller sees
    what actually took effect.  They raise :class:`ViewerAPIError` when
    the server was started with ``allow_writes=False`` -- check
    :attr:`writable` to know in advance.

    Usable as a context manager; safe to share between threads.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 name: Optional[str] = None, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.name = name
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._rfile = None
        self._read_timeout = timeout
        self._lock = threading.Lock()
        # Fail fast if nothing is listening, and learn how long the viewer
        # may take.  Waiting out the server's own limit means a stalled
        # viewer reports *why* it stalled instead of the socket timing out
        # first and blaming the connection.
        pong = self._call("ping")
        self.writable = bool(pong.get("allow_writes", False))
        self._read_timeout = max(timeout, pong["call_timeout"] + 1.0)
        with self._lock:
            if self._sock is not None:
                self._sock.settimeout(self._read_timeout)

    # ---- transport ----

    def _open(self):
        # Connect impatiently, then read patiently.
        self._sock = socket.create_connection(
            (self.host, self.port), timeout=self.timeout)
        self._sock.settimeout(self._read_timeout)
        self._rfile = self._sock.makefile("rb")

    def _call(self, op: str, _read_timeout: Optional[float] = None, **args):
        """One request.  *_read_timeout* extends the wait for a reply to
        calls that do real work away from the viewer's main thread."""
        request = json.dumps(
            {"op": op, "viewer": self.name, "args": args}) + "\n"
        with self._lock:
            for attempt in (0, 1):   # one reconnect for a dropped socket
                try:
                    if self._sock is None:
                        self._open()
                    if _read_timeout is not None:
                        self._sock.settimeout(
                            max(self._read_timeout, _read_timeout))
                    self._sock.sendall(request.encode(_ENCODING))
                    line = self._rfile.readline()
                    if not line:
                        raise ConnectionError("connection closed by viewer")
                    if _read_timeout is not None:
                        self._sock.settimeout(self._read_timeout)
                    break
                except (OSError, ConnectionError) as exc:
                    self._drop()
                    if attempt:
                        raise ViewerAPIError(
                            f"cannot reach viewer API at "
                            f"{self.host}:{self.port} -- {exc}"
                        ) from exc
        reply = json.loads(line.decode(_ENCODING))
        if not reply.get("ok"):
            raise ViewerAPIError(reply.get("error", "unknown error"))
        return reply["result"]

    def _drop(self):
        for obj in (self._rfile, self._sock):
            try:
                if obj is not None:
                    obj.close()
            except OSError:
                pass
        self._rfile, self._sock = None, None

    def close(self):
        """Close the connection.  Further calls reconnect on demand."""
        with self._lock:
            self._drop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        who = self.name or "<only viewer>"
        mode = "" if self.writable else ", read-only"
        return f"RemoteViewer({who!r} at {self.host}:{self.port}{mode})"

    # ---- reads ----

    def list_viewers(self) -> list:
        """Names of every viewer open in the served process."""
        return self._call("list")["viewers"]

    @property
    def hough_results(self) -> Optional[HoughResult]:
        """The viewer's current :class:`HoughResult`, or ``None``."""
        d = self._call("hough")["hough"]
        return None if d is None else HoughResult.from_dict(d)

    def hough_params(self) -> dict:
        """Current detector settings, including whether it is enabled."""
        return self._call("hough_params")["hough_params"]

    @property
    def fit_results(self) -> dict:
        """``{"top": FitResult|None, "right": FitResult|None}``."""
        fits = self._call("fit")["fits"]
        return {d: (None if r is None else FitResult.from_dict(r))
                for d, r in fits.items()}

    def fit(self, direction: str,
            include_curve: bool = True) -> Optional[FitResult]:
        """One direction's :class:`FitResult`, or ``None`` if unfitted.

        Set *include_curve* false to leave out ``y_fit``, which carries
        one value per pixel and dominates the transfer size.
        """
        d = self._call("fit", direction=direction,
                       include_curve=include_curve)["fit"]
        return None if d is None else FitResult.from_dict(d)

    def fit_params(self) -> dict:
        """Fitted Gaussians per direction, plus the solver settings.

        ``{"method", "n_gaussians", "top": {...}|None, "right": ...}``,
        where each direction gives ``offset``, ``gaussians`` and
        ``r_squared`` for the fit that actually ran.
        """
        return self._call("fit_params")["fit_params"]

    def snapshot(self, include_curve: bool = False,
                 include_image: bool = False, raw: bool = False) -> dict:
        """Everything above in one round trip, from a single moment.

        Keys: ``name``, ``frame_id``, ``frame_time``, ``image_shape``,
        ``streaming``, ``hough``, ``hough_params``, ``fits``,
        ``fit_params``, ``fit_busy``.  ``hough`` and the ``fits``
        entries are rebuilt into result objects.

        *include_image* adds ``image``, the array the results were
        measured on -- greyscale, or the frame as loaded with *raw*.
        """
        snap = self._call("snapshot", include_curve=include_curve,
                          include_image=include_image, raw=raw)["snapshot"]
        return self._rebuild_snapshot(snap)

    def measure(self, fit: Optional[str] = None, include_image: bool = False,
                raw: bool = False, timeout: float = 120.0) -> dict:
        """A :meth:`snapshot` whose fits are computed for its own frame.

        The frame's results and image are copied in one step, then the
        viewer fits that frame's projections -- *fit* is ``"top"``,
        ``"right"``, ``"both"`` or ``None`` -- with its current method
        and Gaussian count.  ``fits`` holds only the directions asked
        for.  The fits run off the viewer's GUI thread, so they neither
        freeze it nor get cleared by the next stream frame; they are not
        drawn in the viewer.  *timeout* bounds the fitting.
        """
        snap = self._call("measure", _read_timeout=timeout, fit=fit,
                          include_image=include_image, raw=raw)["snapshot"]
        return self._rebuild_snapshot(snap)

    @staticmethod
    def _rebuild_snapshot(snap: dict) -> dict:
        if snap["hough"] is not None:
            snap["hough"] = HoughResult.from_dict(snap["hough"])
        snap["fits"] = {d: (None if r is None else FitResult.from_dict(r))
                        for d, r in snap["fits"].items()}
        if snap.get("image") is not None:
            snap["image"] = _decode_array(snap["image"])
        return snap

    def frame_info(self) -> dict:
        """``frame_id``, ``frame_time``, ``streaming`` and ``image_shape``.

        ``frame_id`` goes up by one for every image the viewer loads --
        a stream frame, a grab or ``set_image`` -- so comparing it before
        and after tells you whether a new frame has arrived.
        """
        return self._call("frame")["frame"]

    @property
    def frame_id(self) -> int:
        """Count of images the viewer has loaded."""
        return self.frame_info()["frame_id"]

    def get_image(self, raw: bool = False) -> Optional[np.ndarray]:
        """The viewer's current image, or ``None``.

        By default the greyscale working image that detection and fits
        run on, as the local ``get_image()`` returns it; *raw* gives the
        image as loaded, colour included.  Use
        ``snapshot(include_image=True)`` to get an image together with
        the results measured on it.
        """
        d = self._call("image", raw=raw)["image"]
        return None if d is None else _decode_array(d)

    def wait_for_frame(self, frames: int = 1, after: Optional[int] = None,
                       timeout: float = 10.0, poll: float = 0.01) -> dict:
        """Block until *frames* new images have arrived; return :meth:`frame_info`.

        Counts from frame *after* when given, otherwise from the frame
        current at the call.  New frames come from the stream, so this
        raises :class:`ViewerAPIError` on timeout rather than waiting on
        a viewer that is not streaming.
        """
        info = self.frame_info()
        start = info["frame_id"] if after is None else after
        deadline = time.monotonic() + timeout
        while info["frame_id"] < start + frames:
            if time.monotonic() > deadline:
                hint = "" if info["streaming"] else " -- it is not streaming"
                raise ViewerAPIError(
                    f"viewer loaded no new frame within {timeout:g}s{hint}")
            time.sleep(poll)
            info = self.frame_info()
        return info

    @property
    def fit_busy(self) -> bool:
        """True while the viewer is running or queuing a fit."""
        return self._call("fit_busy")["fit_busy"]

    # ---- writes ----

    def set_hough_params(self, **params) -> dict:
        """Set Hough parameters; return the viewer's resulting settings.

        Keywords: ``enabled``, ``min_radius``, ``max_radius``,
        ``param1``, ``param2``, ``min_dist``, ``stabilise``,
        ``hold_frames``.  The whole batch is applied before detection
        re-runs, so one call means one detection pass and the viewer
        never detects against a half-applied combination.

        An unknown name or an out-of-range value raises
        :class:`ViewerAPIError` and leaves the viewer untouched.
        """
        return self._call("set_hough_params", **params)["hough_params"]

    def enable_hough(self, **params) -> dict:
        """Turn detection on, optionally setting parameters at the same time."""
        return self.set_hough_params(enabled=True, **params)

    def disable_hough(self) -> dict:
        """Turn detection off and clear the overlay."""
        return self.set_hough_params(enabled=False)

    def set_scan_roi(self, bounds=None):
        """Show a scan ROI as ``(x0, y0, x1, y1)`` pixels, or hide with None.

        This overlay is separate from the viewer's interactive projection ROI.
        """
        values = None if bounds is None else list(bounds)
        return self._call("set_scan_roi", bounds=values)["bounds"]

    def set_fit_params(self, method: Optional[str] = None,
                       n_gaussians: Optional[int] = None,
                       live_fit: Optional[bool] = None) -> dict:
        """Set the minimiser and/or Gaussian count.

        These apply to the *next* fit -- follow with :meth:`run_fit`.
        *method* is matched case-insensitively; the valid names are in
        ``fit_params()["method"]``'s combo, e.g. ``"Nelder-Mead"``,
        ``"Diff. Evolution"``, ``"Gradient"``, ``"Genetic Alg."``.

        *live_fit* switches Live Fit, which fits both projections of
        every stream frame as it arrives; a snapshot's fits then
        describe its own frame.
        """
        return self._call("set_fit_params", method=method,
                          n_gaussians=n_gaussians,
                          live_fit=live_fit)["fit_params"]

    def grab(self) -> dict:
        """Take one camera frame, load it and analyse it; return :meth:`frame_info`.

        Detection runs on the new frame before this returns; follow with
        :meth:`run_fit` for fits.  Raises :class:`ViewerAPIError` if the
        camera cannot be read, or while the viewer is streaming (the
        stream holds the camera).
        """
        return self._call("grab")["frame"]

    def start_stream(self) -> dict:
        """Start the live camera stream; return :meth:`frame_info`."""
        return self._call("start_stream")["frame"]

    def stop_stream(self) -> dict:
        """Stop the live camera stream; return :meth:`frame_info`."""
        return self._call("stop_stream")["frame"]

    def run_fit(self, direction: str = "both", wait: bool = True,
                timeout: float = 120.0, poll: float = 0.05) -> dict:
        """Run a Gaussian fit and, by default, wait for it to finish.

        *direction* is ``"top"``, ``"right"`` or ``"both"``.  Fits run
        in a worker thread, so this polls :attr:`fit_busy` rather than
        holding the viewer.  Returns :meth:`fit_params`, which carries
        the fitted values once the fit lands.

        With ``wait=False`` it returns as soon as the fit is started,
        and ``fit_params()`` still holds the *previous* result until the
        new one arrives.  A fit that outlasts *timeout* raises
        :class:`ViewerAPIError` and keeps running -- :meth:`cancel_fit`
        stops it.
        """
        self._call("run_fit", direction=direction)
        if not wait:
            return self.fit_params()
        deadline = time.monotonic() + timeout
        while self.fit_busy:
            if time.monotonic() > deadline:
                raise ViewerAPIError(
                    f"fit did not finish within {timeout:g}s (it is still "
                    f"running; call cancel_fit() to stop it)"
                )
            time.sleep(poll)
        return self.fit_params()

    def cancel_fit(self) -> bool:
        """Cancel a running fit and drop anything queued behind it."""
        return self._call("cancel_fit")["fit_busy"]

    def clear_fits(self) -> dict:
        """Remove the fit curves and reset the stored results."""
        return self._call("clear_fits")["fit_params"]


def connect(host: str = "127.0.0.1", port: int = DEFAULT_PORT,
            name: Optional[str] = None,
            timeout: float = 5.0) -> RemoteViewer:
    """Connect to a served viewer and return a :class:`RemoteViewer`.

    *name* selects among several viewers in the served process; it may
    be omitted when only one is open.  Raises :class:`ViewerAPIError`
    if nothing is listening.  The returned client's ``writable``
    attribute says whether the server accepts writes.
    """
    return RemoteViewer(host=host, port=port, name=name, timeout=timeout)
