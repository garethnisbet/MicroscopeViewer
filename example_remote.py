"""Read and set a running viewer's Hough and fit parameters remotely.

Start a viewer with the API switched on::

    python3 image_visualiser.py --serve

then, in a separate terminal::

    python3 example_remote.py                    # report once
    python3 example_remote.py --watch 1.0        # poll
    python3 example_remote.py --set param2=28 --set min_dist=40
    python3 example_remote.py --method Gradient --gaussians 2 --fit both

This script imports only ``viewer_api`` and NumPy -- no PyQt5, OpenCV or
SciPy -- so it runs in a bare environment or on another machine (point
``--host`` at the viewer and serve it on a reachable interface).
"""

import argparse
import time

from viewer_api import connect, ViewerAPIError, DEFAULT_PORT


def _parse(value: str):
    """Turn a command-line value into a bool or int."""
    low = value.lower()
    if low in ("true", "on", "yes"):
        return True
    if low in ("false", "off", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        raise SystemExit(f"cannot read {value!r} as a number or true/false")


def report(remote):
    """Print one consistent snapshot of the viewer's state."""
    snap = remote.snapshot()
    rows, cols = snap["image_shape"]
    print(f"viewer {snap['name']!r}: {cols}x{rows}"
          f"{', streaming' if snap['streaming'] else ''}")

    hp = snap["hough_params"]
    print(f"  hough  : {'on' if hp['enabled'] else 'off'}  "
          f"r=[{hp['min_radius']}..{hp['max_radius']}] "
          f"p1={hp['param1']} p2={hp['param2']} min_dist={hp['min_dist']} "
          f"stabilise={hp['stabilise']} hold={hp['hold_frames']}")

    hough = snap["hough"]
    if hough is None:
        print("           (detection disabled)")
    elif len(hough) == 0:
        print("           no circles")
    else:
        for i, (cx, cy, r) in enumerate(hough.circles, 1):
            print(f"    C{i}: centre=({cx},{cy}) r={r}")
        if hough.stabilised and len(hough.raw_circles) != len(hough):
            print(f"           ({len(hough.raw_circles)} raw this frame, "
                  f"{len(hough)} after stabilisation)")

    fp = snap["fit_params"]
    print(f"  fit    : {fp['method']}, {fp['n_gaussians']} gaussian(s)")
    for direction in ("top", "right"):
        fit = fp[direction]
        if fit is None:
            print(f"    {direction:5}: not fitted")
            continue
        print(f"    {direction:5}: offset={fit['offset']:.2f} "
              f"r2={fit['r_squared']:.4f}")
        for g in fit["gaussians"]:
            print(f"           amp={g['amplitude']:9.2f} "
                  f"mean={g['mean']:7.2f} sigma={g['sigma']:6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--viewer", help="viewer name (needed if several are open)")
    ap.add_argument("--watch", type=float, metavar="SECONDS",
                    help="poll this often instead of reporting once")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    dest="settings",
                    help="set a Hough parameter; repeatable, and all of them "
                         "are applied in one pass")
    ap.add_argument("--method", help="minimiser for the next fit")
    ap.add_argument("--gaussians", type=int, help="number of Gaussians to fit")
    ap.add_argument("--fit", choices=("top", "right", "both"),
                    help="run a fit and wait for it")
    opts = ap.parse_args()

    try:
        remote = connect(host=opts.host, port=opts.port, name=opts.viewer)
    except ViewerAPIError as exc:
        raise SystemExit(f"{exc}\nIs the viewer running with --serve?")

    writes = bool(opts.settings or opts.method or opts.gaussians or opts.fit)
    if writes and not remote.writable:
        raise SystemExit("this viewer is served read-only")

    with remote:
        print(remote, "\nopen viewers:", ", ".join(remote.list_viewers()), "\n")

        if opts.settings:
            params = {}
            for item in opts.settings:
                if "=" not in item:
                    raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
                key, _, value = item.partition("=")
                params[key.strip()] = _parse(value.strip())
            # One call, so the viewer re-detects once rather than per value.
            remote.set_hough_params(**params)
            print("set:", ", ".join(f"{k}={v}" for k, v in params.items()))

        if opts.method is not None or opts.gaussians is not None:
            fp = remote.set_fit_params(method=opts.method,
                                       n_gaussians=opts.gaussians)
            print(f"fit settings: {fp['method']}, "
                  f"{fp['n_gaussians']} gaussian(s)")

        if opts.fit:
            print(f"fitting {opts.fit}...")
            remote.run_fit(opts.fit)

        if opts.settings or opts.method or opts.gaussians or opts.fit:
            print()

        if opts.watch is None:
            report(remote)
            return
        try:
            while True:
                report(remote)
                print()
                time.sleep(opts.watch)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
