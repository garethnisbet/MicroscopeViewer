"""
Example: use ImageVisualiser interactively from IPython.

Start IPython from the CameraViewer directory, then run::

    %run example.py

This opens two named viewer windows and returns you to the IPython
prompt.  The ``v1`` and ``v2`` variables (and ``get_viewer()``) are
available for interactive use.

Alternatively, from a bare IPython session::

    from image_visualiser import launch, get_viewer, get_viewers
    v1 = launch("cam-left")
    v2 = launch("cam-right")
    get_viewer("cam-left").set_image(img)
"""

import numpy as np

from image_visualiser import launch, get_viewer, get_viewers, HoughResult, FitResult


# ---------------------------------------------------------------------------
# Synthetic test image
# ---------------------------------------------------------------------------

def make_circle_image(width=640, height=480):
    """Return a uint8 greyscale image with three bright circles."""
    y, x = np.mgrid[0:height, 0:width]
    img = (40 * (x / width) + 20 * (y / height)).astype(np.float64)
    for cx, cy, r, bright in [(200, 150, 60, 180),
                               (420, 300, 45, 220),
                               (130, 350, 30, 200)]:
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        img += np.exp(-0.5 * ((dist - r) / 2.5) ** 2) * bright
        img += np.clip(1.0 - dist / r, 0, 1) * bright * 0.3
    img += np.random.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Signal callbacks — print results as they arrive
# ---------------------------------------------------------------------------

def on_hough(result):
    if len(result) == 0:
        print("[hough] No circles detected")
        return
    print(f"[hough] {len(result)} circle(s):")
    for i, (cx, cy, r) in enumerate(result.circles):
        print(f"  C{i+1}: centre=({cx}, {cy})  r={r}")


def on_fit(result):
    axis = "vertical" if result.direction == "top" else "horizontal"
    print(f"[fit] {axis} — {result.method}, "
          f"R²={result.r_squared:.6f}, {result.elapsed_ms:.1f} ms")
    for i, g in enumerate(result.gaussians):
        print(f"  G{i+1}: A={g['amplitude']:.1f}  "
              f"μ={g['mean']:.1f}  σ={g['sigma']:.1f}")


def on_crosshair(x, y):
    print(f"[crosshair] x={x:.1f}  y={y:.1f}")


# ---------------------------------------------------------------------------
# Launch two named viewers
# ---------------------------------------------------------------------------

v1 = launch("primary")
v1.hough_changed.connect(on_hough)
v1.fit_changed.connect(on_fit)
v1.crosshair_moved.connect(on_crosshair)
v1.set_image(make_circle_image())
v1.enable_hough(min_radius=20, max_radius=80, param2=25)

v2 = launch("secondary")
v2.set_image(make_circle_image())

print(f"""
──────────────────────────────────────────────────
 Two viewers are running: {list(get_viewers())}
 Variables: v1, v2

 Try:
   v1.hough_results
   v1.run_fit("both")
   v1.crosshair_position
   v1.set_crosshair_position(200, 150)

   v2.set_image(make_circle_image())

   get_viewer("primary").fit_results["top"]
   get_viewers()
──────────────────────────────────────────────────
""")

# Run under plain ``python example.py`` as well: without IPython there is no
# event loop behind the prompt, so block in the Qt one instead.
try:
    get_ipython()          # type: ignore[name-defined]  # noqa: F821
except NameError:
    from PyQt5 import QtWidgets
    QtWidgets.QApplication.instance().exec_()
