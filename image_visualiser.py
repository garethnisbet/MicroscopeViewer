"""
Image Visualiser with Principal Direction Integrations (pyqtgraph)

Displays an image with its horizontal and vertical projections
(sum along rows and columns). Includes webcam snapshot/streaming,
Gaussian fitting with selectable minimisers, and Hough circle detection.

IPython usage::

    from image_visualiser import launch
    viewer = launch()

    viewer.set_image(my_array)
    viewer.enable_hough(min_radius=20, max_radius=80)
    viewer.run_fit("both")
    viewer.hough_results          # HoughResult (.circles is (N,3) array)
    viewer.fit_results["top"]     # FitResult

Driving the viewer from another Python process::

    viewer.serve()                # in the viewer process -> port 8765

    from viewer_api import connect          # in the other process
    with connect() as remote:
        remote.hough_results
        remote.set_hough_params(param2=28, min_dist=40)
        remote.set_fit_params(method="Gradient", n_gaussians=2)
        remote.run_fit("both")

Scripted usage::

    from image_visualiser import ImageVisualiser
    app = QtWidgets.QApplication(sys.argv)
    viewer = ImageVisualiser()
    viewer.hough_changed.connect(callback)
    viewer.show()
    app.exec_()
"""

from __future__ import annotations

import os
import sys
import time
import threading
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize, differential_evolution, least_squares

# Import cv2 first so it can set its env vars, then clear the Qt plugin path
# it injects — otherwise it overrides PyQt5's system plugins and Qt aborts.
import cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

# Re-exported: the result types are defined without GUI dependencies so a
# remote client can rebuild them (see viewer_api).
from results import HoughResult, FitResult

__all__ = [
    "ImageVisualiser",
    "HoughResult",
    "FitResult",
    "fit_gaussians",
    "make_fit_result",
    "launch",
    "get_viewer",
    "get_viewers",
]


# ---------------------------------------------------------------------------
# Instance registry
# ---------------------------------------------------------------------------

_viewers: dict[str, "ImageVisualiser"] = {}


# ---------------------------------------------------------------------------
# Cancellation support
# ---------------------------------------------------------------------------

class FitCancelled(Exception):
    pass


# ---------------------------------------------------------------------------
# Gaussian model helpers
# ---------------------------------------------------------------------------

def _multi_gaussian(x, params):
    """params: [offset, amp1, mu1, sigma1, amp2, mu2, sigma2, ...]"""
    offset = params[0]
    y = np.full_like(x, offset, dtype=np.float64)
    for i in range(1, len(params), 3):
        amp, mu, sigma = params[i], params[i + 1], params[i + 2]
        y += amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y


def _residuals(params, x, y):
    return _multi_gaussian(x, params) - y


def _cost(params, x, y):
    return np.sum(_residuals(params, x, y) ** 2)


def _initial_guess(x, y, n_gaussians):
    offset = np.median(y)
    y_shifted = y - offset
    peak = np.max(y_shifted)
    span = x[-1] - x[0]
    sigma_guess = span / (4 * max(n_gaussians, 1))
    p0 = [offset]
    for k in range(n_gaussians):
        frac = (k + 1) / (n_gaussians + 1)
        p0 += [peak / n_gaussians, x[0] + frac * span, sigma_guess]
    return np.array(p0, dtype=np.float64)


def _bounds_for(x, y, n_gaussians):
    y_range = np.ptp(y)
    lo = [np.min(y) - y_range]
    hi = [np.max(y) + y_range]
    for _ in range(n_gaussians):
        lo += [0, x[0], 1.0]
        hi += [np.max(y) * 2, x[-1], (x[-1] - x[0]) / 2]
    return list(zip(lo, hi))


def fit_gaussians(x, y, n_gaussians, method, cancel_event=None):
    p0 = _initial_guess(x, y, n_gaussians)
    t0 = time.perf_counter()

    def _check():
        if cancel_event is not None and cancel_event.is_set():
            raise FitCancelled()

    def cost(params, x, y):
        _check()
        return _cost(params, x, y)

    def residuals(params, x, y):
        _check()
        return _residuals(params, x, y)

    if method == "Nelder-Mead":
        res = minimize(cost, p0, args=(x, y), method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
        popt = res.x
        dt = time.perf_counter() - t0
        msg = f"Nelder-Mead: nit={res.nit}, {dt*1000:.1f} ms"

    elif method == "Diff. Evolution":
        bounds = _bounds_for(x, y, n_gaussians)
        res = differential_evolution(cost, bounds, args=(x, y),
                                     maxiter=1000, seed=42, tol=1e-10)
        popt = res.x
        dt = time.perf_counter() - t0
        msg = f"Diff. Evol.: nit={res.nit}, {dt*1000:.1f} ms"

    elif method == "Gradient":
        res = least_squares(residuals, p0, args=(x, y), method="trf",
                            max_nfev=20000)
        popt = res.x
        dt = time.perf_counter() - t0
        msg = f"Least-Sq (trf): nfev={res.nfev}, {dt*1000:.1f} ms"

    elif method == "Genetic Alg.":
        bounds = _bounds_for(x, y, n_gaussians)
        n_params = len(p0)
        pop_size = max(50, 10 * n_params)
        n_gen = 200
        mut_rate = 0.8
        cross_rate = 0.7
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Initialise population — seed one member with the heuristic guess
        pop = rng.uniform(lo, hi, size=(pop_size, n_params))
        pop[0] = np.clip(p0, lo, hi)
        fitness = np.array([cost(ind, x, y) for ind in pop])
        nfev = pop_size

        for gen in range(n_gen):
            _check()
            # Tournament selection + crossover + mutation
            new_pop = np.empty_like(pop)
            for j in range(pop_size):
                # Tournament select two parents (k=3)
                idx = rng.choice(pop_size, size=3, replace=False)
                p1 = pop[idx[np.argmin(fitness[idx])]]
                idx = rng.choice(pop_size, size=3, replace=False)
                p2 = pop[idx[np.argmin(fitness[idx])]]

                # Uniform crossover
                mask = rng.random(n_params) < cross_rate
                child = np.where(mask, p1, p2)

                # Gaussian mutation
                if rng.random() < mut_rate:
                    delta = rng.normal(0, 0.1 * (hi - lo))
                    child = np.clip(child + delta, lo, hi)

                new_pop[j] = child

            # Elitism — keep the best
            best_idx = np.argmin(fitness)
            new_fitness = np.array([cost(ind, x, y) for ind in new_pop])
            nfev += pop_size
            worst_new = np.argmax(new_fitness)
            if fitness[best_idx] < new_fitness[worst_new]:
                new_pop[worst_new] = pop[best_idx]
                new_fitness[worst_new] = fitness[best_idx]

            pop = new_pop
            fitness = new_fitness

        popt = pop[np.argmin(fitness)]
        dt = time.perf_counter() - t0
        msg = f"GA: gen={n_gen}, pop={pop_size}, {dt*1000:.1f} ms"

    else:
        raise ValueError(f"Unknown method: {method}")

    y_fit = _multi_gaussian(x, popt)
    return y_fit, popt, msg


def make_fit_result(direction, method, n_gaussians, y, y_fit, popt, msg,
                    elapsed_ms) -> FitResult:
    """Package a :func:`fit_gaussians` result, with its R², as a FitResult."""
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    gaussians = [{"amplitude": float(popt[1 + i * 3]),
                  "mean": float(popt[2 + i * 3]),
                  "sigma": float(popt[3 + i * 3])}
                 for i in range(n_gaussians)]
    return FitResult(
        direction=direction,
        method=method,
        n_gaussians=n_gaussians,
        offset=float(popt[0]),
        gaussians=gaussians,
        r_squared=r2,
        elapsed_ms=elapsed_ms,
        popt=np.array(popt, dtype=np.float64),
        y_fit=np.array(y_fit, dtype=np.float64),
        message=msg,
    )


# ---------------------------------------------------------------------------
# Background fit worker
# ---------------------------------------------------------------------------

class FitWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)

    def __init__(self, x, y, n_gaussians, method, direction, cancel_event):
        super().__init__()
        self.x = x
        self.y = y
        self.n_gaussians = n_gaussians
        self.method = method
        self.direction = direction
        self.cancel_event = cancel_event
        self.t_start = time.perf_counter()

    def run(self):
        try:
            y_fit, popt, msg = fit_gaussians(
                self.x, self.y, self.n_gaussians, self.method, self.cancel_event
            )
            self.finished.emit({
                "y_fit": y_fit, "popt": popt, "msg": msg,
                "direction": self.direction, "n": self.n_gaussians,
                "x": self.x, "y": self.y, "t_start": self.t_start, "error": None,
            })
        except FitCancelled:
            self.finished.emit({"error": "cancelled"})
        except Exception as e:
            self.finished.emit({"error": str(e)})


# ---------------------------------------------------------------------------
# Main visualiser
# ---------------------------------------------------------------------------

class ImageVisualiser(QtWidgets.QMainWindow):
    """Image viewer with projections, Gaussian fitting, and Hough circles.

    Signals
    -------
    hough_changed(HoughResult)
        Emitted when Hough circle detection runs.  The payload is a
        :class:`HoughResult` containing the circles array and parameters.
    fit_changed(FitResult)
        Emitted each time a Gaussian fit completes on either axis.
    """

    hough_changed = QtCore.pyqtSignal(object)
    fit_changed = QtCore.pyqtSignal(object)
    crosshair_moved = QtCore.pyqtSignal(float, float)
    frame_grabbed = QtCore.pyqtSignal(object)

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or f"viewer-{id(self)}"
        self.setWindowTitle(f"Image Visualiser — {self.name}")
        self.resize(1100, 750)

        if self.name in _viewers:
            raise ValueError(
                f"A viewer named {self.name!r} already exists. "
                f"Close it first or choose a different name."
            )
        _viewers[self.name] = self

        # Public result state
        self.hough_results: Optional[HoughResult] = None
        self.fit_results: dict[str, Optional[FitResult]] = {
            "top": None,
            "right": None,
        }

        # Internal state
        self._raw = None   # original image as passed to set_image
        self._grey = None  # current greyscale working image
        self.vertical_proj = None
        self.horizontal_proj = None
        self._streaming = False
        self._stream_cap = None
        self._fit_worker = None
        self._fit_cancel = None
        self._fit_pending = []
        self._cameras = self._detect_cameras()
        self._hough_circles_items = []  # circle overlay curves
        self._scan_roi_bounds = None     # tracked scan ROI: x0, y0, x1, y1
        self._hough_tracks = []         # persistent circle tracks
        self._hough_fresh = True        # next run is the first after a reset
        self._hough_norm = None         # smoothed (lo, hi) contrast limits
        self._view_shape = None         # image shape the view was last framed to
        self.frame_id = 0               # bumped for every image loaded
        self.frame_time = None          # time.time() the latest image arrived

        self._build_ui()
        self._set_image(self._test_pattern())

    # ---- Public API ----

    @staticmethod
    def _flush():
        """Process pending Qt events so the display updates immediately."""
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()

    def set_image(self, img: np.ndarray, keep_fits: bool = False):
        """Load a new image (greyscale or BGR/RGB ndarray)."""
        self._set_image(img, keep_fits=keep_fits)
        self._flush()

    def set_scan_roi(self, bounds=None):
        """Show the scan tracker ROI as ``(x0, y0, x1, y1)`` image pixels.

        Pass ``None`` to hide it. This overlay is independent of the
        interactive projection ROI and does not affect image analysis.
        """
        if bounds is None:
            self._scan_roi_bounds = None
        else:
            if len(bounds) != 4:
                raise ValueError("scan ROI needs (x0, y0, x1, y1)")
            values = tuple(float(value) for value in bounds)
            if not all(np.isfinite(value) for value in values):
                raise ValueError("scan ROI coordinates must be finite")
            x0, y0, x1, y1 = values
            if x1 <= x0 or y1 <= y0:
                raise ValueError("scan ROI must have positive width and height")
            self._scan_roi_bounds = values
        self._update_scan_roi_overlay()
        self._flush()
        return self._scan_roi_bounds

    def _update_scan_roi_overlay(self):
        bounds = self._scan_roi_bounds
        if bounds is None:
            self.scan_roi_item.hide()
            self.scan_roi_label.hide()
            return
        x0, y0, x1, y1 = bounds
        self.scan_roi_item.setData(
            [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
        self.scan_roi_label.setPos(x0, y0)
        self.scan_roi_item.show()
        self.scan_roi_label.show()

    def get_image(self) -> Optional[np.ndarray]:
        """Return the current greyscale image, or None."""
        return self._grey.copy() if self._grey is not None else None

    # ---- Hough parameters ----

    @property
    def _hough_widgets(self) -> dict:
        """Parameter name -> the control holding it."""
        return {
            "enabled": self.chk_hough,
            "min_radius": self.spin_min_r,
            "max_radius": self.spin_max_r,
            "param1": self.spin_hough_p1,
            "param2": self.spin_hough_p2,
            "min_dist": self.spin_min_dist,
            "stabilise": self.chk_hough_stable,
            "hold_frames": self.spin_hough_hold,
        }

    def hough_params(self) -> dict:
        """Return the current Hough settings as a dict."""
        return {
            key: (w.isChecked() if isinstance(w, QtWidgets.QCheckBox)
                  else w.value())
            for key, w in self._hough_widgets.items()
        }

    def set_hough_params(self, **kwargs) -> dict:
        """Set any of the Hough parameters and return the resulting dict.

        Keyword arguments: enabled, min_radius, max_radius, param1,
        param2, min_dist, stabilise, hold_frames.

        Every parameter is applied before detection re-runs, so setting
        several at once costs one pass rather than one per value and no
        intermediate combination is ever detected against.  A value
        outside its control's range raises ``ValueError`` rather than
        being silently clamped.
        """
        widgets = self._hough_widgets
        unknown = set(kwargs) - set(widgets)
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) {', '.join(map(repr, sorted(unknown)))}; "
                f"valid: {', '.join(widgets)}"
            )

        # Validate everything before touching the GUI, so a bad value in
        # the batch cannot leave the controls half-applied.
        pending = []
        for key, val in kwargs.items():
            w = widgets[key]
            if isinstance(w, QtWidgets.QCheckBox):
                pending.append((w.setChecked, bool(val)))
            else:
                ival = int(val)
                if not w.minimum() <= ival <= w.maximum():
                    raise ValueError(
                        f"{key}={val!r} is outside the valid range "
                        f"{w.minimum()}..{w.maximum()}"
                    )
                pending.append((w.setValue, ival))

        blocked = [(w, w.signalsBlocked()) for w in widgets.values()]
        try:
            for w, _ in blocked:
                w.blockSignals(True)
            for setter, val in pending:
                setter(val)
        finally:
            for w, was_blocked in blocked:
                w.blockSignals(was_blocked)

        # One refresh for the whole batch.
        if self.chk_hough.isChecked():
            self._reset_hough_tracks()
            self._run_hough()
        else:
            self._clear_hough()
        return self.hough_params()

    def enable_hough(self, **kwargs) -> dict:
        """Enable Hough circle detection, optionally setting parameters.

        Keyword arguments are those of :meth:`set_hough_params`.
        Returns the resulting parameter dict.
        """
        params = self.set_hough_params(**dict(kwargs, enabled=True))
        self._flush()
        return params

    def disable_hough(self) -> dict:
        """Disable Hough circle detection."""
        params = self.set_hough_params(enabled=False)
        self._flush()
        return params

    # ---- Fit parameters ----

    def fit_params(self) -> dict:
        """Return the fit settings, plus the parameters of any fit that ran.

        ``method`` and ``n_gaussians`` are the current settings, which
        apply to the *next* fit.  Each direction key carries the
        ``offset``, ``gaussians`` and ``r_squared`` of the fit actually
        held for it, along with the settings it was run under, or
        ``None`` if it has not been fitted.
        """
        out = {
            "method": self.combo_method.currentText(),
            "n_gaussians": self.spin_ngauss.value(),
            "live_fit": self.chk_live_fit.isChecked(),
        }
        for direction, res in self.fit_results.items():
            out[direction] = None if res is None else {
                "offset": float(res.offset),
                "gaussians": [{k: float(x) for k, x in g.items()}
                              for g in res.gaussians],
                "r_squared": float(res.r_squared),
                "method": res.method,
                "n_gaussians": int(res.n_gaussians),
            }
        return out

    def fit_methods(self) -> List[str]:
        """Return the available minimiser names."""
        return [self.combo_method.itemText(i)
                for i in range(self.combo_method.count())]

    def set_fit_params(self, method: Optional[str] = None,
                       n_gaussians: Optional[int] = None,
                       live_fit: Optional[bool] = None) -> dict:
        """Set the minimiser and/or Gaussian count; return the new settings.

        These take effect on the next fit -- call :meth:`run_fit` to
        apply them.  *method* is matched case-insensitively against
        :meth:`fit_methods`.  *live_fit* ticks Live Fit, which fits both
        projections of every stream frame as it arrives, so the fits
        always describe the frame on screen.  Returns :meth:`fit_params`.
        """
        if live_fit is not None:
            self.chk_live_fit.setChecked(bool(live_fit))

        if method is not None:
            methods = self.fit_methods()
            match = next((m for m in methods if m.lower() == method.lower()),
                         None)
            if match is None:
                raise ValueError(
                    f"Unknown method {method!r}; valid: {', '.join(methods)}"
                )
            self.combo_method.setCurrentText(match)

        if n_gaussians is not None:
            n = int(n_gaussians)
            lo, hi = self.spin_ngauss.minimum(), self.spin_ngauss.maximum()
            if not lo <= n <= hi:
                raise ValueError(
                    f"n_gaussians={n_gaussians!r} is outside the valid "
                    f"range {lo}..{hi}"
                )
            self.spin_ngauss.setValue(n)

        return self.fit_params()

    @property
    def fit_busy(self) -> bool:
        """True while a fit is running or queued."""
        return self._fit_worker is not None or bool(self._fit_pending)

    def cancel_fit(self):
        """Cancel a running fit and drop anything queued behind it."""
        self._on_cancel_fit()

    def serve(self, port: int = 8765, host: str = "127.0.0.1",
              allow_writes: bool = True) -> int:
        """Start the remote API and return the port it uses.

        Other Python processes can then read and set this viewer's Hough
        and fit parameters via ``viewer_api.connect()``.  One server
        covers every viewer in the process, so calling this on a second
        viewer just returns the running port; clients pick a viewer by
        name.

        Pass ``allow_writes=False`` to serve the viewer read-only.

        Requires a running Qt event loop, which ``launch()`` sets up.
        Pass ``port=0`` for an OS-assigned port.  See
        :func:`viewer_api.serve` for the host/exposure caveat.
        """
        from viewer_api import serve as _serve
        # Hand over this module's registry: run as a script, importing
        # image_visualiser from viewer_api would find a different one.
        return _serve(port=port, host=host, allow_writes=allow_writes,
                      registry=get_viewers)

    @staticmethod
    def stop_serving():
        """Stop the remote API server, if one is running."""
        from viewer_api import stop_serving as _stop
        _stop()

    def run_fit(self, direction: str = "both"):
        """Run a Gaussian fit. *direction*: ``"top"``, ``"right"``, or ``"both"``."""
        if direction == "both":
            self._on_fit_both()
        elif direction == "top":
            self._on_fit_top()
        elif direction == "right":
            self._on_fit_right()
        else:
            raise ValueError(f"direction must be 'top', 'right', or 'both'")
        self._flush()

    def clear_fits(self):
        """Remove all fit curves and reset fit_results."""
        self._on_clear_fits()
        self._flush()

    def grab_frame(self) -> Optional[np.ndarray]:
        """Grab a single frame from the selected camera and return it.

        Returns the frame as an RGB uint8 numpy array, or ``None`` on
        failure.  The frame is **not** loaded into the viewer — use
        :meth:`snapshot` to grab *and* display, or feed the returned
        array to :meth:`set_image` yourself.
        """
        cap = cv2.VideoCapture(self._selected_camera, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def snapshot(self):
        """Grab a single frame, display it, and emit ``frame_grabbed``."""
        self._on_snapshot()
        self._flush()

    @property
    def channel(self) -> str:
        """The currently selected channel name (e.g. ``"Greyscale"``, ``"Red"``)."""
        return self.combo_channel.currentText() if self.combo_channel.isVisible() else "Greyscale"

    @channel.setter
    def channel(self, value: str):
        idx = self.combo_channel.findText(value)
        if idx < 0:
            available = [self.combo_channel.itemText(i)
                         for i in range(self.combo_channel.count())]
            raise ValueError(
                f"Unknown channel {value!r}; available: {available}"
            )
        self.combo_channel.setCurrentIndex(idx)
        self._flush()

    def start_stream(self):
        """Start live camera streaming."""
        if not self._streaming:
            self._start_stream()

    def stop_stream(self):
        """Stop live camera streaming."""
        if self._streaming:
            self._stop_stream()

    # ---- UI construction ----

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(4)

        # --- Plot area (left) using GraphicsLayoutWidget ---
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.graphics.ci.layout.setSpacing(2)
        main_layout.addWidget(self.graphics, stretch=5)

        # Top projection plot
        self.plot_top = self.graphics.addPlot(row=0, col=0)
        self.plot_top.setLabel("left", "\u2211 along y")
        self.plot_top.hideAxis("bottom")
        self.curve_top = self.plot_top.plot(pen=pg.mkPen("steelblue", width=1))
        self.fill_top = pg.FillBetweenItem(
            self.curve_top,
            self.plot_top.plot([], []),
            brush=pg.mkBrush(70, 130, 180, 80),
        )
        self.plot_top.addItem(self.fill_top)
        self.fit_curves_top = []

        # Spacer top-right
        self.graphics.addItem(pg.GraphicsWidget(), row=0, col=1)

        # Image
        self.plot_image = self.graphics.addPlot(row=1, col=0)
        self.plot_image.setLabel("bottom", "x (pixels)")
        self.plot_image.setLabel("left", "y (pixels)")
        self.plot_image.setAspectLocked(True)
        self.image_item = pg.ImageItem()
        self.plot_image.addItem(self.image_item)

        # Scan tracker ROI, separate from the draggable projection ROI below.
        scan_roi_pen = pg.mkPen("#00e5ff", width=2, style=QtCore.Qt.DashLine)
        self.scan_roi_item = pg.PlotCurveItem(pen=scan_roi_pen)
        self.plot_image.addItem(self.scan_roi_item)
        self.scan_roi_label = pg.TextItem(
            "Scan ROI", color="#00e5ff", anchor=(0, 1),
            fill=pg.mkBrush(0, 0, 0, 150))
        self.plot_image.addItem(self.scan_roi_label)
        self._update_scan_roi_overlay()

        # ROI overlay (initially hidden)
        self.roi = pg.ROI(
            pos=[100, 100], size=[200, 200],
            pen=pg.mkPen("lime", width=2),
            hoverPen=pg.mkPen("yellow", width=2),
            handlePen=pg.mkPen("white", width=1),
            handleHoverPen=pg.mkPen("yellow", width=2),
        )
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.roi.addScaleHandle([1, 0], [0, 1])
        self.roi.addScaleHandle([0, 1], [1, 0])
        self.plot_image.addItem(self.roi)
        self.roi.hide()
        self.roi.sigRegionChanged.connect(self._on_roi_changed)

        # Crosshairs (initially hidden)
        crosshair_pen = pg.mkPen("cyan", width=1, style=QtCore.Qt.DashLine)
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=True, pen=crosshair_pen,
                                           hoverPen=pg.mkPen("yellow", width=2))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=True, pen=crosshair_pen,
                                           hoverPen=pg.mkPen("yellow", width=2))
        self.plot_image.addItem(self.crosshair_v)
        self.plot_image.addItem(self.crosshair_h)
        self.crosshair_v.hide()
        self.crosshair_h.hide()
        self.crosshair_v.sigPositionChanged.connect(self._on_crosshair_moved)
        self.crosshair_h.sigPositionChanged.connect(self._on_crosshair_moved)

        # Right projection plot
        self.plot_right = self.graphics.addPlot(row=1, col=1)
        self.plot_right.setLabel("bottom", "\u2211 along x")
        self.plot_right.hideAxis("left")
        self.curve_right = self.plot_right.plot(pen=pg.mkPen("coral", width=1))
        self.fill_right = pg.FillBetweenItem(
            self.plot_right.plot([], []),
            self.curve_right,
            brush=pg.mkBrush(255, 127, 80, 80),
        )
        self.plot_right.addItem(self.fill_right)
        self.fit_curves_right = []

        # Link axes
        self.plot_top.setXLink(self.plot_image)
        self.plot_right.setYLink(self.plot_image)
        self.plot_right.invertY(True)

        # Give the image row/col most of the space
        ci = self.graphics.ci
        ci.layout.setRowStretchFactor(0, 1)   # top projection: small
        ci.layout.setRowStretchFactor(1, 5)   # image row: large
        ci.layout.setColumnStretchFactor(0, 5)  # image col: large
        ci.layout.setColumnStretchFactor(1, 1)  # right projection: small

        # --- Control panel (right) ---
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(170)
        panel.setStyleSheet("""
            QWidget { background: #2b2b2b; color: #e0e0e0; }
            QPushButton {
                background: #3c3f41; color: #e0e0e0; border: 1px solid #555;
                padding: 4px; border-radius: 3px;
            }
            QPushButton:hover { background: #4c5052; }
            QPushButton:pressed { background: #2a2a2a; }
            QPushButton:checked { background: #cc4444; color: white; }
            QComboBox {
                background: #3c3f41; color: #e0e0e0; border: 1px solid #555;
                padding: 3px; border-radius: 3px;
            }
            QComboBox QAbstractItemView { background: #3c3f41; color: #e0e0e0; }
            QSpinBox {
                background: #3c3f41; color: #e0e0e0; border: 1px solid #555;
                padding: 3px; border-radius: 3px;
            }
            QCheckBox { color: #e0e0e0; spacing: 5px; }
            QCheckBox::indicator {
                width: 14px; height: 14px; border: 1px solid #555;
                border-radius: 2px; background: #3c3f41;
            }
            QCheckBox::indicator:checked { background: #5599ff; }
            QLabel { color: #e0e0e0; }
            QFrame[frameShape="4"] { color: #555; }
        """)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(6, 6, 6, 6)
        panel_layout.setSpacing(4)
        main_layout.addWidget(panel)

        # Camera selector
        panel_layout.addWidget(QtWidgets.QLabel("<b>Camera</b>"))
        self.combo_camera = QtWidgets.QComboBox()
        for idx, name in self._cameras:
            self.combo_camera.addItem(name, idx)
        self.combo_camera.currentIndexChanged.connect(self._on_camera_changed)
        panel_layout.addWidget(self.combo_camera)

        self.btn_refresh_cameras = QtWidgets.QPushButton("Refresh")
        self.btn_refresh_cameras.clicked.connect(self._on_refresh_cameras)
        panel_layout.addWidget(self.btn_refresh_cameras)

        panel_layout.addSpacing(5)

        # Snapshot / Stream buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_snapshot = QtWidgets.QPushButton("Snapshot")
        self.btn_snapshot.clicked.connect(self._on_snapshot)
        btn_row.addWidget(self.btn_snapshot)

        self.btn_stream = QtWidgets.QPushButton("Stream")
        self.btn_stream.setCheckable(True)
        self.btn_stream.clicked.connect(self._on_stream_toggle)
        btn_row.addWidget(self.btn_stream)
        panel_layout.addLayout(btn_row)

        self.btn_save = QtWidgets.QPushButton("Save Image")
        self.btn_save.clicked.connect(self._on_save_image)
        panel_layout.addWidget(self.btn_save)

        # Channel selector (hidden for greyscale images)
        self._channel_label = QtWidgets.QLabel("<b>Channel</b>")
        self._channel_label.hide()
        panel_layout.addWidget(self._channel_label)
        self.combo_channel = QtWidgets.QComboBox()
        self.combo_channel.hide()
        self.combo_channel.currentIndexChanged.connect(self._on_channel_changed)
        panel_layout.addWidget(self.combo_channel)

        panel_layout.addSpacing(10)
        panel_layout.addWidget(self._make_separator())

        self.btn_reset_view = QtWidgets.QPushButton("Reset View")
        self.btn_reset_view.setToolTip("Zoom back out to the whole image")
        self.btn_reset_view.clicked.connect(self._reset_view)
        panel_layout.addWidget(self.btn_reset_view)

        panel_layout.addSpacing(5)

        # ROI toggle
        self.chk_roi = QtWidgets.QCheckBox("Region of Interest")
        self.chk_roi.toggled.connect(self._on_roi_toggle)
        panel_layout.addWidget(self.chk_roi)

        self.btn_roi_reset = QtWidgets.QPushButton("Reset ROI")
        self.btn_roi_reset.clicked.connect(self._on_roi_reset)
        self.btn_roi_reset.setEnabled(False)
        panel_layout.addWidget(self.btn_roi_reset)

        panel_layout.addSpacing(5)

        self.chk_negative = QtWidgets.QCheckBox("Negative")
        self.chk_negative.toggled.connect(self._on_negative_toggle)
        panel_layout.addWidget(self.chk_negative)

        self.chk_crosshair = QtWidgets.QCheckBox("Crosshairs")
        self.chk_crosshair.toggled.connect(self._on_crosshair_toggle)
        panel_layout.addWidget(self.chk_crosshair)

        self.label_crosshair = QtWidgets.QLabel("")
        self.label_crosshair.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #66e0e0; "
            "background: #1e1e1e; border: 1px solid #444; padding: 3px;"
        )
        self.label_crosshair.hide()
        panel_layout.addWidget(self.label_crosshair)

        panel_layout.addSpacing(10)
        panel_layout.addWidget(self._make_separator())

        # Hough circle detection
        panel_layout.addWidget(QtWidgets.QLabel("<b>Hough Circles</b>"))

        self.chk_hough = QtWidgets.QCheckBox("Enable")
        self.chk_hough.toggled.connect(self._on_hough_toggle)
        panel_layout.addWidget(self.chk_hough)

        hough_grid = QtWidgets.QGridLayout()
        hough_grid.setSpacing(2)

        hough_grid.addWidget(QtWidgets.QLabel("Min Radius"), 0, 0)
        self.spin_min_r = QtWidgets.QSpinBox()
        self.spin_min_r.setRange(0, 2000)
        self.spin_min_r.setValue(10)
        self.spin_min_r.valueChanged.connect(self._on_hough_param_changed)
        hough_grid.addWidget(self.spin_min_r, 0, 1)

        hough_grid.addWidget(QtWidgets.QLabel("Max Radius"), 1, 0)
        self.spin_max_r = QtWidgets.QSpinBox()
        self.spin_max_r.setRange(0, 2000)
        self.spin_max_r.setValue(100)
        self.spin_max_r.valueChanged.connect(self._on_hough_param_changed)
        hough_grid.addWidget(self.spin_max_r, 1, 1)

        hough_grid.addWidget(QtWidgets.QLabel("Param1"), 2, 0)
        self.spin_hough_p1 = QtWidgets.QSpinBox()
        self.spin_hough_p1.setRange(1, 500)
        self.spin_hough_p1.setValue(100)
        self.spin_hough_p1.setToolTip("Canny edge upper threshold")
        self.spin_hough_p1.valueChanged.connect(self._on_hough_param_changed)
        hough_grid.addWidget(self.spin_hough_p1, 2, 1)

        hough_grid.addWidget(QtWidgets.QLabel("Param2"), 3, 0)
        self.spin_hough_p2 = QtWidgets.QSpinBox()
        self.spin_hough_p2.setRange(1, 500)
        self.spin_hough_p2.setValue(30)
        self.spin_hough_p2.setToolTip("Accumulator threshold (lower = more circles)")
        self.spin_hough_p2.valueChanged.connect(self._on_hough_param_changed)
        hough_grid.addWidget(self.spin_hough_p2, 3, 1)

        hough_grid.addWidget(QtWidgets.QLabel("Min Dist"), 4, 0)
        self.spin_min_dist = QtWidgets.QSpinBox()
        self.spin_min_dist.setRange(1, 1000)
        self.spin_min_dist.setValue(50)
        self.spin_min_dist.setToolTip("Min distance between circle centres")
        self.spin_min_dist.valueChanged.connect(self._on_hough_param_changed)
        hough_grid.addWidget(self.spin_min_dist, 4, 1)

        self.chk_hough_stable = QtWidgets.QCheckBox("Stabilise")
        self.chk_hough_stable.setChecked(True)
        self.chk_hough_stable.setToolTip(
            "Track circles between frames so ones scoring near the "
            "accumulator threshold stop flickering on and off"
        )
        self.chk_hough_stable.toggled.connect(self._on_hough_param_changed)
        hough_grid.addWidget(self.chk_hough_stable, 5, 0, 1, 2)

        hough_grid.addWidget(QtWidgets.QLabel("Hold Frames"), 6, 0)
        self.spin_hough_hold = QtWidgets.QSpinBox()
        self.spin_hough_hold.setRange(0, 60)
        self.spin_hough_hold.setValue(3)
        self.spin_hough_hold.setToolTip(
            "Frames a circle is kept after it stops being detected"
        )
        self.spin_hough_hold.valueChanged.connect(self._on_hough_param_changed)
        hough_grid.addWidget(self.spin_hough_hold, 6, 1)

        panel_layout.addLayout(hough_grid)

        panel_layout.addSpacing(10)
        panel_layout.addWidget(self._make_separator())

        # Minimiser selector
        panel_layout.addWidget(QtWidgets.QLabel("<b>Minimiser</b>"))
        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItems(["Nelder-Mead", "Diff. Evolution", "Gradient", "Genetic Alg."])
        panel_layout.addWidget(self.combo_method)

        panel_layout.addSpacing(5)

        # Number of Gaussians
        panel_layout.addWidget(QtWidgets.QLabel("<b># Gaussians</b>"))
        self.spin_ngauss = QtWidgets.QSpinBox()
        self.spin_ngauss.setRange(1, 20)
        self.spin_ngauss.setValue(1)
        panel_layout.addWidget(self.spin_ngauss)

        panel_layout.addSpacing(10)
        panel_layout.addWidget(self._make_separator())

        # Fit buttons
        self.btn_fit_top = QtWidgets.QPushButton("Fit Vertical (top)")
        self.btn_fit_top.clicked.connect(self._on_fit_top)
        panel_layout.addWidget(self.btn_fit_top)

        self.btn_fit_right = QtWidgets.QPushButton("Fit Horiz. (right)")
        self.btn_fit_right.clicked.connect(self._on_fit_right)
        panel_layout.addWidget(self.btn_fit_right)

        self.btn_fit_both = QtWidgets.QPushButton("Fit Both")
        self.btn_fit_both.clicked.connect(self._on_fit_both)
        panel_layout.addWidget(self.btn_fit_both)

        self.btn_clear = QtWidgets.QPushButton("Clear Fits")
        self.btn_clear.clicked.connect(self._on_clear_fits)
        panel_layout.addWidget(self.btn_clear)

        self.btn_cancel_fit = QtWidgets.QPushButton("Cancel Fit")
        self.btn_cancel_fit.clicked.connect(self._on_cancel_fit)
        self.btn_cancel_fit.hide()
        panel_layout.addWidget(self.btn_cancel_fit)

        self.chk_live_fit = QtWidgets.QCheckBox("Live Fit (stream)")
        self.chk_live_fit.setToolTip("Automatically fit every stream frame")
        panel_layout.addWidget(self.chk_live_fit)

        panel_layout.addSpacing(10)
        panel_layout.addWidget(self._make_separator())

        # Status label
        self.label_status = QtWidgets.QLabel("")
        self.label_status.setWordWrap(True)
        self.label_status.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #e0e0e0; "
            "background: #1e1e1e; border: 1px solid #444; padding: 4px;"
        )
        self.label_status.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.label_status.setMinimumHeight(100)
        panel_layout.addWidget(self.label_status, stretch=1)

        # Stream timer
        self._stream_timer = QtCore.QTimer(self)
        self._stream_timer.timeout.connect(self._stream_frame)

    @staticmethod
    def _make_separator():
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    # ---- Image handling ----

    def _test_pattern(self):
        y, x = np.mgrid[-1:1:480j, -1:1:640j]
        g1 = 200 * np.exp(-((x - 0.3) ** 2 + (y + 0.2) ** 2) / 0.08)
        g2 = 150 * np.exp(-((x + 0.4) ** 2 + (y - 0.3) ** 2) / 0.12)
        pattern = g1 + g2 + np.random.normal(0, 8, x.shape)
        return np.clip(pattern, 0, 255).astype(np.uint8)

    def _set_image(self, img, keep_fits=False):
        self._raw = img.copy()
        self.frame_id += 1
        self.frame_time = time.time()

        # Update channel selector visibility
        if img.ndim == 3:
            n_ch = img.shape[2]
            self.combo_channel.blockSignals(True)
            prev = self.combo_channel.currentText()
            self.combo_channel.clear()
            if n_ch == 3:
                self.combo_channel.addItems(["Colour", "Greyscale", "Red", "Green", "Blue"])
            elif n_ch == 4:
                self.combo_channel.addItems(["Colour", "Greyscale", "Red", "Green", "Blue", "Alpha"])
            else:
                self.combo_channel.addItems(
                    ["Colour", "Greyscale"] + [f"Ch {i}" for i in range(n_ch)]
                )
            # Restore previous selection if it still exists
            idx = self.combo_channel.findText(prev)
            self.combo_channel.setCurrentIndex(max(idx, 0))
            self.combo_channel.blockSignals(False)
            self._channel_label.show()
            self.combo_channel.show()
        else:
            self._channel_label.hide()
            self.combo_channel.hide()

        self._apply_channel(keep_fits=keep_fits)

    def _apply_channel(self, keep_fits=False):
        """Derive greyscale from the selected channel and refresh the display."""
        img = self._raw
        if img is None:
            return

        is_colour = False
        if img.ndim == 3:
            ch_name = self.combo_channel.currentText()
            grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
            if ch_name == "Colour":
                is_colour = True
            elif ch_name == "Red":
                grey = img[:, :, 0].astype(np.float64)
            elif ch_name == "Green":
                grey = img[:, :, 1].astype(np.float64)
            elif ch_name == "Blue":
                grey = img[:, :, 2].astype(np.float64)
            elif ch_name == "Alpha":
                grey = img[:, :, 3].astype(np.float64)
            elif ch_name.startswith("Ch "):
                grey = img[:, :, int(ch_name.split()[-1])].astype(np.float64)
        else:
            grey = img.astype(np.float64)

        self._grey = grey
        rows, cols = grey.shape

        if not keep_fits:
            self._clear_fits()

        if is_colour:
            display = img.astype(np.float64)
            if self.chk_negative.isChecked():
                display = 255.0 - display
            self.image_item.setImage(np.transpose(display, (1, 0, 2)))
        else:
            display = grey
            if self.chk_negative.isChecked():
                display = grey.max() - grey
            self.image_item.setImage(display.T, levels=(display.min(), display.max()))

        self.image_item.setRect(QtCore.QRectF(0, 0, cols, rows))
        # Only frame the image when its size changes, so a zoom/pan set by the
        # user survives the stream of new frames.
        if self._view_shape != (rows, cols):
            self._view_shape = (rows, cols)
            self.plot_image.setRange(xRange=(0, cols), yRange=(0, rows), padding=0)
            self.plot_image.invertY(True)

        self._update_projections()
        if self.chk_hough.isChecked():
            self._run_hough()
        if self.chk_crosshair.isChecked():
            self._update_crosshair_label(self.crosshair_v.value(),
                                         self.crosshair_h.value())

    def _reset_view(self):
        """Frame the whole image again after the user has zoomed or panned."""
        if self._grey is None:
            return
        rows, cols = self._grey.shape
        self._view_shape = (rows, cols)
        self.plot_image.setRange(xRange=(0, cols), yRange=(0, rows), padding=0)
        self.plot_image.invertY(True)

    def _get_roi_bounds(self):
        """Return (y0, y1, x0, x1) pixel indices clamped to image, or None."""
        if self._grey is None:
            return None
        rows, cols = self._grey.shape
        pos = self.roi.pos()
        size = self.roi.size()
        x0 = int(max(0, pos.x()))
        y0 = int(max(0, pos.y()))
        x1 = int(min(cols, pos.x() + size.x()))
        y1 = int(min(rows, pos.y() + size.y()))
        if x1 <= x0 or y1 <= y0:
            return None
        return y0, y1, x0, x1

    def _update_projections(self):
        if self._grey is None:
            return
        grey = self._grey
        if self.chk_negative.isChecked():
            grey = grey.max() - grey
        rows, cols = grey.shape

        if self.chk_roi.isChecked():
            bounds = self._get_roi_bounds()
            if bounds:
                y0, y1, x0, x1 = bounds
                roi_data = grey[y0:y1, x0:x1]
                # Vertical: sum within ROI rows, place in full-width array
                self.vertical_proj = np.zeros(cols)
                self.vertical_proj[x0:x1] = roi_data.sum(axis=0)
                # Horizontal: sum within ROI cols, place in full-height array
                self.horizontal_proj = np.zeros(rows)
                self.horizontal_proj[y0:y1] = roi_data.sum(axis=1)
            else:
                self.vertical_proj = grey.sum(axis=0)
                self.horizontal_proj = grey.sum(axis=1)
        else:
            self.vertical_proj = grey.sum(axis=0)
            self.horizontal_proj = grey.sum(axis=1)

        # Top projection
        xv = np.arange(cols, dtype=np.float64)
        self.curve_top.setData(xv, self.vertical_proj)
        self.fill_top.setCurves(
            self.curve_top,
            pg.PlotDataItem(xv, np.zeros(cols)),
        )

        # Right projection
        yr = np.arange(rows, dtype=np.float64)
        self.curve_right.setData(self.horizontal_proj, yr)
        self.fill_right.setCurves(
            pg.PlotDataItem(np.zeros(rows), yr),
            self.curve_right,
        )

    # ---- ROI ----

    def _on_roi_toggle(self, checked):
        if checked:
            # Position ROI at centre quarter of image
            if self._grey is not None:
                rows, cols = self._grey.shape
                self.roi.setPos([cols * 0.25, rows * 0.25])
                self.roi.setSize([cols * 0.5, rows * 0.5])
            self.roi.show()
            self.btn_roi_reset.setEnabled(True)
        else:
            self.roi.hide()
            self.btn_roi_reset.setEnabled(False)
        self._update_projections()

    def _on_roi_changed(self):
        if self.chk_roi.isChecked():
            self._update_projections()

    def _on_roi_reset(self):
        if self._grey is not None:
            rows, cols = self._grey.shape
            self.roi.setPos([cols * 0.25, rows * 0.25])
            self.roi.setSize([cols * 0.5, rows * 0.5])

    def _on_negative_toggle(self, checked):
        if self._raw is not None:
            self._apply_channel(keep_fits=True)

    def _on_channel_changed(self, _index):
        self._apply_channel(keep_fits=False)

    # ---- Crosshairs ----

    @property
    def crosshair_position(self) -> Optional[tuple]:
        """Return ``(x, y)`` pixel position of the crosshairs, or None if hidden."""
        if not self.chk_crosshair.isChecked():
            return None
        return (self.crosshair_v.value(), self.crosshair_h.value())

    def set_crosshair_position(self, x: float, y: float):
        """Move the crosshairs to *(x, y)* in pixel coordinates."""
        self.crosshair_v.setValue(x)
        self.crosshair_h.setValue(y)

    def _on_crosshair_toggle(self, checked):
        if checked:
            if self._grey is not None:
                rows, cols = self._grey.shape
                self.crosshair_v.setValue(cols / 2)
                self.crosshair_h.setValue(rows / 2)
            self.crosshair_v.show()
            self.crosshair_h.show()
            self.label_crosshair.show()
            self._on_crosshair_moved()
        else:
            self.crosshair_v.hide()
            self.crosshair_h.hide()
            self.label_crosshair.hide()

    def _on_crosshair_moved(self, _line=None):
        x = self.crosshair_v.value()
        y = self.crosshair_h.value()
        self._update_crosshair_label(x, y)
        self.crosshair_moved.emit(x, y)

    def _update_crosshair_label(self, x, y):
        """Show the crosshair pixel position (and value under it) in the panel."""
        text = "X {:8.2f}\nY {:8.2f}".format(x, y)
        if self._grey is not None:
            rows, cols = self._grey.shape
            col, row = int(round(x)), int(round(y))
            if 0 <= col < cols and 0 <= row < rows:
                text += "\nI {:8.4g}".format(float(self._grey[row, col]))
        self.label_crosshair.setText(text)

    # ---- Hough circle detection ----

    def _on_hough_toggle(self, checked):
        if checked:
            self._reset_hough_tracks()
            self._run_hough()
        else:
            self._clear_hough()

    def _on_hough_param_changed(self, _value=None):
        """Re-run detection after a control changes, discarding old tracks."""
        self._reset_hough_tracks()
        self._run_hough()

    def _reset_hough_tracks(self):
        """Drop the temporal state so the next run starts from scratch."""
        self._hough_tracks = []
        self._hough_fresh = True
        self._hough_norm = None

    def _normalise_for_hough(self, grey):
        """Map the working image onto 0-255 with temporally eased limits.

        HoughCircles runs Canny internally, and Canny's thresholds are
        absolute.  Clipping the working image straight to 0-255 therefore
        ties edge strength to the absolute signal level, so gain drift, a
        16-bit source, or a mostly-dark frame flips edges on and off
        between near-identical frames.  Stretching between robust
        percentiles makes param1 mean the same thing every frame, and
        easing the limits keeps the percentiles' own noise out of it.
        """
        lo, hi = (float(v) for v in np.percentile(grey, (1.0, 99.0)))
        if hi - lo < 1e-9:
            lo, hi = float(grey.min()), float(grey.max())
        if hi - lo < 1e-9:
            hi = lo + 1.0

        prev = self._hough_norm
        if prev is not None:
            a = 0.3
            lo = a * lo + (1.0 - a) * prev[0]
            hi = a * hi + (1.0 - a) * prev[1]
        self._hough_norm = (lo, hi)

        scaled = (grey.astype(np.float64) - lo) * (255.0 / (hi - lo))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def _update_hough_tracks(self, detections):
        """Fold this frame's detections into persistent tracks.

        The accumulator is thresholded hard at *param2*, so a circle
        scoring near that threshold is detected or missed on essentially
        no change in the image.  Confirming a circle over two frames
        before drawing it, holding it through a few missed frames, and
        smoothing its centre and radius turns that binary flicker into a
        stable overlay.  Returns the confirmed circles as an ``(N, 3)``
        int array.
        """
        confirm = 2
        hold = self.spin_hough_hold.value()
        alpha = 0.4
        # The first run after a reset shows immediately, so changing a
        # parameter on a still image gives an answer straight away.  After
        # that a circle has to survive *confirm* frames, which is what
        # keeps a one-frame false positive off the display.
        birth_hits = confirm if self._hough_fresh else 1
        self._hough_fresh = False

        for tr in self._hough_tracks:
            tr["matched"] = False

        for cx, cy, r in detections:
            tol = max(0.5 * r, 0.5 * self.spin_min_dist.value())
            best, best_d = None, None
            for tr in self._hough_tracks:
                if tr["matched"]:
                    continue
                d = float(np.hypot(cx - tr["cx"], cy - tr["cy"]))
                if d <= tol and (best_d is None or d < best_d):
                    best, best_d = tr, d
            if best is None:
                self._hough_tracks.append(dict(
                    cx=float(cx), cy=float(cy), r=float(r),
                    hits=birth_hits, misses=0, matched=True,
                ))
            else:
                best["cx"] += alpha * (cx - best["cx"])
                best["cy"] += alpha * (cy - best["cy"])
                best["r"] += alpha * (r - best["r"])
                best["hits"] += 1
                best["misses"] = 0
                best["matched"] = True

        kept = []
        for tr in self._hough_tracks:
            if not tr["matched"]:
                tr["misses"] += 1
                if tr["misses"] > hold:
                    continue
            kept.append(tr)
        self._hough_tracks = kept

        stable = [(tr["cx"], tr["cy"], tr["r"])
                  for tr in kept if tr["hits"] >= confirm]
        if not stable:
            return np.empty((0, 3), dtype=int)
        return np.round(np.array(stable)).astype(int)

    def _run_hough(self):
        if not self.chk_hough.isChecked() or self._grey is None:
            return
        self._clear_hough()

        img8 = self._normalise_for_hough(self._grey)
        blurred = cv2.GaussianBlur(img8, (9, 9), 2)

        raw = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.spin_min_dist.value(),
            param1=self.spin_hough_p1.value(),
            param2=self.spin_hough_p2.value(),
            minRadius=self.spin_min_r.value(),
            maxRadius=self.spin_max_r.value(),
        )

        if raw is None:
            raw_circles = np.empty((0, 3), dtype=int)
        else:
            raw_circles = np.round(raw[0]).astype(int)

        stabilised = self.chk_hough_stable.isChecked()
        if stabilised:
            circles = self._update_hough_tracks(raw_circles)
        else:
            self._hough_tracks = []
            self._hough_fresh = True
            circles = raw_circles

        self.hough_results = HoughResult(
            circles=circles.copy(),
            param1=self.spin_hough_p1.value(),
            param2=self.spin_hough_p2.value(),
            min_dist=self.spin_min_dist.value(),
            min_radius=self.spin_min_r.value(),
            max_radius=self.spin_max_r.value(),
            stabilised=stabilised,
            raw_circles=raw_circles.copy(),
        )
        self.hough_changed.emit(self.hough_results)

        if len(circles) == 0:
            self.label_status.setText("Hough: no circles found")
            return

        theta = np.linspace(0, 2 * np.pi, 120)
        for cx, cy, r in circles:
            xs = cx + r * np.cos(theta)
            ys = cy + r * np.sin(theta)
            curve = self.plot_image.plot(
                xs, ys, pen=pg.mkPen("lime", width=2)
            )
            self._hough_circles_items.append(curve)
            # Centre marker
            cross_size = max(3, r * 0.1)
            cx_h = self.plot_image.plot(
                [cx - cross_size, cx + cross_size], [cy, cy],
                pen=pg.mkPen("lime", width=1),
            )
            cx_v = self.plot_image.plot(
                [cx, cx], [cy - cross_size, cy + cross_size],
                pen=pg.mkPen("lime", width=1),
            )
            self._hough_circles_items.extend([cx_h, cx_v])

        lines = [f"Hough: {len(circles)} circle(s)"]
        for i, (cx, cy, r) in enumerate(circles):
            lines.append(f"  C{i+1}: centre=({cx},{cy}) r={r}")
        self.label_status.setText("\n".join(lines))

    def _clear_hough(self):
        for item in self._hough_circles_items:
            self.plot_image.removeItem(item)
        self._hough_circles_items.clear()
        self.hough_results = None

    # ---- Fit logic ----

    def _clear_fits(self):
        for c in self.fit_curves_top:
            self.plot_top.removeItem(c)
        for c in self.fit_curves_right:
            self.plot_right.removeItem(c)
        self.fit_curves_top.clear()
        self.fit_curves_right.clear()
        self.fit_results = {"top": None, "right": None}

    def _plot_fit_result(self, direction, x, y, y_fit, popt, n, msg, t_start):
        """Plot a completed fit result onto the appropriate projection plot."""
        if direction == "top":
            plot = self.plot_top
            store = self.fit_curves_top
        else:
            plot = self.plot_right
            store = self.fit_curves_right

        is_top = direction == "top"
        pen_fit = pg.mkPen("r", width=2, style=QtCore.Qt.DashLine)
        xy = (x, y_fit) if is_top else (y_fit, x)

        if store:
            store[0].setData(*xy)
        else:
            store.append(plot.plot(*xy, pen=pen_fit))

        colours = ["#e6c300", "#ff66ff", "#00cc99", "#ff8800", "#6666ff"]
        comp_idx = 1
        for i in range(n if n > 1 else 0):
            comp = [0.0] + list(popt[1 + i * 3: 1 + (i + 1) * 3])
            yc = _multi_gaussian(x, comp)
            xy_c = (x, yc) if is_top else (yc, x)
            pen_c = pg.mkPen(colours[i % len(colours)], width=1, style=QtCore.Qt.DotLine)
            if comp_idx < len(store):
                store[comp_idx].setData(*xy_c)
            else:
                store.append(plot.plot(*xy_c, pen=pen_c))
            comp_idx += 1

        while len(store) > comp_idx:
            plot.removeItem(store.pop())

        dt_total = (time.perf_counter() - t_start) * 1000
        result = make_fit_result(direction, self.combo_method.currentText(),
                                 n, y, y_fit, popt, msg, dt_total)
        r2, gaussians = result.r_squared, result.gaussians
        self.fit_results[direction] = result
        self.fit_changed.emit(result)

        lines = [msg, f"total: {dt_total:.1f} ms", f"R\u00b2 = {r2:.6f}"]
        for i, g in enumerate(gaussians):
            lines.append(
                f"G{i+1}: A={g['amplitude']:.0f}  "
                f"\u03bc={g['mean']:.1f}  \u03c3={g['sigma']:.1f}"
            )
        lines.append(f"offset={popt[0]:.0f}")
        self.label_status.setText("\n".join(lines))

    def _do_fit(self, direction):
        """Synchronous fit — used by live streaming."""
        proj = self.vertical_proj if direction == "top" else self.horizontal_proj
        if proj is None:
            return
        n = self.spin_ngauss.value()
        method = self.combo_method.currentText()
        x = np.arange(len(proj), dtype=np.float64)
        t0 = time.perf_counter()
        try:
            y_fit, popt, msg = fit_gaussians(x, proj, n, method)
        except Exception as e:
            self.label_status.setText(f"Fit error: {e}")
            return
        self._plot_fit_result(direction, x, proj, y_fit, popt, n, msg, t0)

    # ---- Async fit (button-driven) ----

    def _set_fit_running(self, running):
        self.btn_fit_top.setEnabled(not running)
        self.btn_fit_right.setEnabled(not running)
        self.btn_fit_both.setEnabled(not running)
        self.btn_cancel_fit.setVisible(running)

    def _start_fit_worker(self, direction):
        proj = self.vertical_proj if direction == "top" else self.horizontal_proj
        if proj is None:
            if self._fit_pending:
                self._start_fit_worker(self._fit_pending.pop(0))
            else:
                self._set_fit_running(False)
            return

        n = self.spin_ngauss.value()
        method = self.combo_method.currentText()
        x = np.arange(len(proj), dtype=np.float64)

        self._fit_cancel = threading.Event()
        self._fit_worker = FitWorker(x, proj, n, method, direction, self._fit_cancel)
        self._fit_worker.finished.connect(self._on_fit_finished)
        self._set_fit_running(True)
        self.label_status.setText(f"Fitting {direction}...")
        self._fit_worker.start()

    def _on_fit_finished(self, result):
        self._fit_worker = None
        self._fit_cancel = None

        if result.get("error") == "cancelled":
            self.label_status.setText("Fit cancelled")
            self._fit_pending.clear()
            self._set_fit_running(False)
            return
        if result.get("error"):
            self.label_status.setText(f"Fit error: {result['error']}")
            self._fit_pending.clear()
            self._set_fit_running(False)
            return

        self._plot_fit_result(
            result["direction"], result["x"], result["y"], result["y_fit"],
            result["popt"], result["n"], result["msg"], result["t_start"],
        )

        if self._fit_pending:
            self._start_fit_worker(self._fit_pending.pop(0))
        else:
            self._set_fit_running(False)

    def _on_cancel_fit(self):
        if self._fit_cancel is not None:
            self._fit_cancel.set()
        self._fit_pending.clear()

    def _on_fit_top(self):
        for c in self.fit_curves_top:
            self.plot_top.removeItem(c)
        self.fit_curves_top.clear()
        self._fit_pending.clear()
        self._start_fit_worker("top")

    def _on_fit_right(self):
        for c in self.fit_curves_right:
            self.plot_right.removeItem(c)
        self.fit_curves_right.clear()
        self._fit_pending.clear()
        self._start_fit_worker("right")

    def _on_fit_both(self):
        self._clear_fits()
        self._fit_pending = ["right"]
        self._start_fit_worker("top")

    def _on_clear_fits(self):
        self._clear_fits()
        self.label_status.setText("")

    def closeEvent(self, event):
        if self._fit_cancel is not None:
            self._fit_cancel.set()
        if self._fit_worker is not None:
            self._fit_worker.wait()
        self._stop_stream()
        _viewers.pop(self.name, None)
        super().closeEvent(event)

    # ---- Camera detection ----

    @staticmethod
    def _detect_cameras(max_check=8):
        """Probe video devices and return list of (index, label) for those that open."""
        found = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                name = cap.getBackendName()
                found.append((i, f"Camera {i} ({name})"))
                cap.release()
        if not found:
            found.append((0, "Camera 0"))
        return found

    @property
    def _selected_camera(self):
        return self.combo_camera.currentData()

    def _on_camera_changed(self):
        if self._streaming:
            self._stop_stream()
            self._start_stream()

    def _on_refresh_cameras(self):
        was_streaming = self._streaming
        if was_streaming:
            self._stop_stream()

        prev_idx = self.combo_camera.currentData()
        self._cameras = self._detect_cameras()

        self.combo_camera.blockSignals(True)
        self.combo_camera.clear()
        select = 0
        for i, (idx, name) in enumerate(self._cameras):
            self.combo_camera.addItem(name, idx)
            if idx == prev_idx:
                select = i
        self.combo_camera.setCurrentIndex(select)
        self.combo_camera.blockSignals(False)

        self.label_status.setText(f"Found {len(self._cameras)} camera(s)")
        if was_streaming:
            self._start_stream()

    # ---- Webcam ----

    def _on_save_image(self):
        if self._raw is None:
            self.label_status.setText("No image to save")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        if not path:
            return
        img = self._raw
        if img.ndim == 3:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
        else:
            cv2.imwrite(path, img.astype(np.uint8))
        self.label_status.setText(f"Saved: {os.path.basename(path)}")

    def _on_snapshot(self):
        frame = self.grab_frame()
        if frame is None:
            self.label_status.setText("Error: could not grab frame")
            return
        self._set_image(frame)
        self.frame_grabbed.emit(frame)

    def _on_stream_toggle(self, checked):
        if checked:
            self._start_stream()
        else:
            self._stop_stream()

    def _start_stream(self):
        self._stream_cap = cv2.VideoCapture(self._selected_camera, cv2.CAP_V4L2)
        if not self._stream_cap.isOpened():
            self.label_status.setText("Error: could not open webcam")
            self._stream_cap = None
            self.btn_stream.setChecked(False)
            return
        self._streaming = True
        self.btn_stream.setText("Stop")
        self._stream_timer.start(33)  # ~30 fps

    def _stop_stream(self):
        self._streaming = False
        self._stream_timer.stop()
        if self._stream_cap is not None:
            self._stream_cap.release()
            self._stream_cap = None
        self.btn_stream.setText("Stream")
        self.btn_stream.setChecked(False)

    def _stream_frame(self):
        if not self._streaming or self._stream_cap is None:
            return
        ret, raw = self._stream_cap.read()
        if not ret:
            self._stop_stream()
            self.label_status.setText("Stream: lost webcam feed")
            return
        frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        live = self.chk_live_fit.isChecked()
        self._set_image(frame, keep_fits=live)
        self.frame_grabbed.emit(frame)
        if live:
            self._do_fit("top")
            self._do_fit("right")


def _install_qt_inputhook(ipy) -> bool:
    """Give the IPython prompt a Qt event loop hook.  True if one is active.

    ``%gui qt5`` goes through IPython's own Qt loader, which only accepts
    PyQt5 when ``PyQt5.QtSvg`` is importable too.  On systems where the
    distro splits QtSvg into a separate package the magic fails and the
    windows never get to process events, so fall back to a small polling
    hook of our own.
    """
    if getattr(ipy, "active_eventloop", None):
        return True

    try:
        ipy.run_line_magic("gui", "qt5")
    except Exception:
        pass
    if getattr(ipy, "active_eventloop", None):
        return True

    try:
        from IPython.terminal.pt_inputhooks import register
    except ImportError:
        return False

    def _hook(context):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        while not context.input_is_ready():
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)
            time.sleep(0.005)

    register("qt5-pyqt5", _hook)
    try:
        ipy.enable_gui("qt5-pyqt5")
    except Exception:
        return False
    return bool(getattr(ipy, "active_eventloop", None))


def launch(name: Optional[str] = None) -> ImageVisualiser:
    """Create and show an ImageVisualiser, setting up the Qt event loop.

    Parameters
    ----------
    name : str, optional
        Identifier for this viewer instance.  Must be unique among open
        viewers.  If omitted a default is generated.

    In IPython / Jupyter this enables ``%gui qt5`` so the window stays
    responsive while the prompt remains interactive.  In a plain Python
    script the caller should follow up with ``app.exec_()``.

    Returns the :class:`ImageVisualiser` instance.
    """
    # Enable Qt event-loop integration in IPython if available
    try:
        ipy = get_ipython()          # type: ignore[name-defined]  # noqa: F821
    except NameError:
        ipy = None                   # not in IPython
    if ipy is not None and not _install_qt_inputhook(ipy):
        print("Warning: no Qt event loop is running — the viewer window will "
              "not appear or respond.  Run '%gui qt5' at the prompt, or call "
              "QtWidgets.QApplication.instance().exec_().")

    # Create QApplication if one doesn't exist yet
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    viewer = ImageVisualiser(name=name)
    viewer.show()
    return viewer


def get_viewer(name: str) -> ImageVisualiser:
    """Return an open viewer by *name*, or raise ``KeyError``."""
    return _viewers[name]


def get_viewers() -> dict[str, ImageVisualiser]:
    """Return a dict of all open viewer instances keyed by name."""
    return dict(_viewers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--serve", nargs="?", type=int, const=8765, metavar="PORT",
        help="start the remote API on PORT (default 8765) so other "
             "processes can read and set the Hough and fit parameters",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="interface for --serve (default 127.0.0.1, loopback only)",
    )
    parser.add_argument(
        "--read-only", action="store_true",
        help="serve the viewer without write access",
    )
    opts = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv[:1])
    win = ImageVisualiser()
    win.show()
    if opts.serve is not None:
        port = win.serve(port=opts.serve, host=opts.host,
                         allow_writes=not opts.read_only)
        print(f"viewer API listening on {opts.host}:{port}"
              f"{' (read-only)' if opts.read_only else ''}")
    sys.exit(app.exec_())
