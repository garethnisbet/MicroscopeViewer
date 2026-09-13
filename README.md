# Image Visualiser

A PyQt5/pyqtgraph application that displays a greyscale image alongside its principal-direction projections (sum along rows and columns) and provides interactive Gaussian curve fitting, Hough circle detection, and a Python API for embedding in other programs or driving interactively from IPython.

[DeltabotController](https://github.com/garethnisbet/DeltabotController) uses the [remote API](#remote-api) as the detector for stage scans: it moves a flexure stage and records circle positions, fitted peak centres and images at every point.

## Running

### Standalone

```bash
python3 image_visualiser.py
python3 image_visualiser.py --serve        # also expose the remote API
```

`--serve` lets other processes read and set the viewer's parameters; add `--read-only` to publish it without write access.

### IPython (interactive)

```python
from image_visualiser import launch
viewer = launch()
```

`launch()` enables `%gui qt5` automatically so the window stays responsive while the IPython prompt remains available. See [API](#python-api) below.

### Example script

```bash
# In IPython:
%run example.py
```

Opens the viewer with a synthetic circle image, enables Hough detection, and connects print callbacks. The `viewer` variable is left in the IPython namespace for interactive use.

`example_remote.py` is the other half: it reads and sets a running viewer's parameters from a separate process. See [Remote API](#remote-api).

### Dependencies

- Python 3
- NumPy
- SciPy
- OpenCV (`cv2`)
- PyQt5
- pyqtgraph

A process driving a viewer over the [remote API](#remote-api) needs just Python 3 and NumPy.

## Features

### Image Display

The central view shows a greyscale image with a locked aspect ratio. On startup a synthetic test pattern containing two Gaussian blobs with additive noise is generated. Images can also be acquired from a connected webcam.

Two projection plots surround the image:

- **Top plot** -- the vertical projection (sum of each column, i.e. summed along the y-axis).
- **Right plot** -- the horizontal projection (sum of each row, i.e. summed along the x-axis).

The projection axes are linked to the image axes so that panning and zooming stay synchronised.

### Camera Input

- **Snapshot** -- captures a single frame from the webcam.
- **Stream** -- opens a live feed at approximately 30 fps. Press again to stop.
- **Camera selector** -- a dropdown lists all detected video devices. Switching cameras while streaming automatically restarts the stream on the new device.
- **Refresh** -- re-probes `/dev/video*` and updates the camera list.

### View and Zoom

Scroll or right-drag on the image to zoom and pan. The view is **kept across frames**, so a zoomed-in region stays put while streaming; it re-frames automatically only when the image size changes. **Reset View** zooms back out to the whole image. The projection plots are axis-linked to the image, so they follow the zoom.

### Region of Interest (ROI)

Enabling the **Region of Interest** checkbox overlays a draggable, resizable rectangle on the image. When active, the projections are computed only over the pixels inside the ROI. The **Reset ROI** button returns it to the default centre-quarter position.

### Negative

The **Negative** checkbox inverts the greyscale image (maps each pixel value `v` to `max - v`) and recomputes the projections accordingly.

### Crosshairs

The **Crosshairs** checkbox overlays a draggable vertical and horizontal line, starting at the image centre. A readout below the checkbox shows the crosshair position in pixel coordinates (`X`, `Y`) and the greyscale value `I` under the intersection, updating live as you drag and as new frames arrive. The position is also available from code as `viewer.crosshair_position` (or `None` when the crosshairs are off), settable with `set_crosshair_position(x, y)`, and emitted on the `crosshair_moved` signal.

### Hough Circle Detection

Enabling the **Hough Circles** checkbox runs `cv2.HoughCircles` (gradient method) on the current image. Detected circles are drawn as lime-green outlines with crosshair centre markers.

Before detection the image is stretched onto 0-255 between its 1st and 99th percentiles, then blurred with a 9×9 Gaussian (σ=2). The stretch matters because `HoughCircles` runs Canny internally and Canny's thresholds are absolute: without it, `Param1` means something different at every brightness level, so low-contrast, dim, or non-8-bit images lose their edges and gain drift makes edges come and go. The percentile limits are themselves eased frame to frame (EMA, α=0.3) so their own noise does not reach the detector.

Controls:

- **Min Radius / Max Radius** -- filter circles by size (pixels).
- **Param1** -- Canny edge detector upper threshold.
- **Param2** -- accumulator threshold; lower values detect more circles.
- **Min Dist** -- minimum distance between circle centres.
- **Stabilise** -- track circles between frames (see below). On by default.
- **Hold Frames** -- how many missed frames a circle survives before it is dropped.

Circles re-detect automatically when any parameter changes, when a new image is loaded, or during streaming.

#### Stabilisation

The Hough accumulator is thresholded hard at `Param2`, so a circle scoring near that threshold is found on one frame and missed on the next even when the image barely changes -- the overlay flickers. **Stabilise** matches each frame's detections against persistent tracks (matched on centre distance, within half the radius or half **Min Dist**, whichever is larger) and:

- requires a circle on two consecutive frames before drawing it, which rejects one-frame false positives;
- holds a circle for **Hold Frames** frames after it stops being detected, which bridges dropouts;
- smooths each centre and radius with an EMA (α=0.4), which removes pixel jitter.

Tracks are discarded whenever a Hough control changes or detection is toggled, so an edit to a parameter still shows its result on the very next frame. `HoughResult.circles` holds the stabilised circles and `HoughResult.raw_circles` this frame's unfiltered detections, so a callback can see both.

Turn **Stabilise** off to get the previous frame-independent behaviour.

### Gaussian Fitting

The projections can be fitted with a sum of Gaussian functions:

```
f(x) = offset + sum_i [ A_i * exp(-0.5 * ((x - mu_i) / sigma_i)^2) ]
```

Controls:

- **# Gaussians** -- number of Gaussian components (1--20).
- **Fit Vertical (top)** -- fits the top projection.
- **Fit Horiz. (right)** -- fits the right projection.
- **Fit Both** -- fits both projections.
- **Clear Fits** -- removes all fit curves.
- **Cancel Fit** -- aborts a running fit.
- **Live Fit (stream)** -- when checked, both projections are re-fitted on every stream frame.

The total fit is drawn as a red dashed line. When multiple Gaussians are used, individual components are shown as coloured dotted lines. Fit parameters (amplitude, mean, sigma, offset) and timing are reported in the status panel.

### Minimisers

Four optimisation methods are available for fitting. Each minimises the sum-of-squared-residuals between the projection data and the multi-Gaussian model.

#### Nelder-Mead

A derivative-free simplex method (`scipy.optimize.minimize` with `method="Nelder-Mead"`). It maintains a simplex of N+1 points in parameter space (where N is the number of parameters) and iteratively reflects, expands, or contracts the worst vertex to move downhill. It requires no gradient information, making it robust for noisy or non-smooth cost surfaces, but convergence can be slow in high dimensions. The implementation uses up to 20,000 iterations with tight absolute tolerances on both the parameter values and the function value.

#### Differential Evolution

A global, population-based stochastic optimiser (`scipy.optimize.differential_evolution`). It maintains a population of candidate solutions and creates trial vectors by combining differences between randomly selected members. Each trial replaces its parent if it achieves a lower cost. Because it samples broadly across the bounded parameter space it is much less likely to get trapped in local minima than Nelder-Mead, at the expense of more function evaluations. The implementation runs up to 1,000 generations with a fixed random seed for reproducibility.

#### Gradient (Least-Squares)

A gradient-based Trust Region Reflective method (`scipy.optimize.least_squares` with `method="trf"`). It operates on the vector of residuals rather than the scalar cost, which allows it to exploit the structure of least-squares problems via an approximate Jacobian. It converges quickly when the initial guess is reasonable and the cost surface is smooth, but can settle in a local minimum if started far from the global optimum. The implementation allows up to 20,000 function evaluations.

#### Genetic Algorithm

A custom evolutionary algorithm implemented from scratch. It works as follows:

1. **Initialisation** -- a population of candidate solutions (at least 50, or 10 times the number of parameters) is drawn uniformly within parameter bounds. The heuristic initial guess is seeded as the first member.
2. **Selection** -- two parents are chosen per offspring via tournament selection with a tournament size of 3.
3. **Crossover** -- a uniform crossover with a 70% swap probability creates a child from the two parents.
4. **Mutation** -- with 80% probability, Gaussian noise scaled to 10% of the parameter range is added and the result is clamped to bounds.
5. **Elitism** -- the single best individual from the previous generation replaces the worst of the new generation, ensuring the best solution is never lost.

Steps 2--5 repeat for 200 generations. Like Differential Evolution it searches globally, but the hand-rolled implementation gives direct control over the evolutionary operators. It is the slowest method for small problems but can be competitive when the cost landscape has many local minima.

### Choosing a Minimiser

| Method | Type | Strengths | Weaknesses |
|---|---|---|---|
| Nelder-Mead | Local, derivative-free | Simple, no bounds needed | Slow in high dimensions, local minima |
| Diff. Evolution | Global, population-based | Escapes local minima, bounded | Many function evaluations |
| Gradient | Local, Jacobian-based | Fast convergence near solution | Needs good initial guess, local minima |
| Genetic Alg. | Global, evolutionary | Full control, global search | Slowest, stochastic |

For a single well-separated Gaussian, any method works well. For multiple overlapping Gaussians, Differential Evolution or the Genetic Algorithm are more reliable at finding the global optimum, while Gradient is the fastest when the initial guess is close.

## Python API

The module can be imported and used programmatically. All public classes and functions are available via `from image_visualiser import ...`.

### `launch() -> ImageVisualiser`

Convenience function that creates a `QApplication` (if needed), enables `%gui qt5` when running inside IPython, and returns a visible `ImageVisualiser` instance.

```python
from image_visualiser import launch
viewer = launch()
```

### Methods

| Method | Description |
|---|---|
| `set_image(img, keep_fits=False)` | Load a numpy array (greyscale or colour). |
| `get_image()` | Return the current greyscale image as a numpy array, or `None`. |
| `hough_params()` | Return the current detector settings as a dict. |
| `set_hough_params(**kwargs)` | Set any of `enabled`, `min_radius`, `max_radius`, `param1`, `param2`, `min_dist`, `stabilise`, `hold_frames`; returns the resulting dict. |
| `enable_hough(**kwargs)` | Enable Hough circle detection, optionally setting the same parameters. |
| `disable_hough()` | Disable Hough circle detection. |
| `fit_params()` | Fit settings (including `live_fit`), plus the parameters of any fit that has run. |
| `set_fit_params(method, n_gaussians, live_fit)` | Set the minimiser and/or Gaussian count for the next fit, and/or tick **Live Fit**. |
| `frame_id` | Count of images loaded (stream frames, snapshots, `set_image`); goes up by one per image (attribute). |
| `frame_time` | `time.time()` when the latest image was loaded (attribute). |
| `fit_methods()` | List the available minimiser names. |
| `run_fit(direction)` | Run a Gaussian fit. `direction`: `"top"`, `"right"`, or `"both"`. |
| `fit_busy` | `True` while a fit is running or queued (property). |
| `cancel_fit()` | Cancel a running fit and drop anything queued behind it. |
| `clear_fits()` | Remove all fit curves and reset results. |
| `snapshot()` | Grab a single frame from the selected camera. |
| `start_stream()` | Start live camera streaming. |
| `stop_stream()` | Stop live camera streaming. |
| `serve(port=8765, host="127.0.0.1", allow_writes=True)` | Start the [remote API](#remote-api); returns the port. |
| `stop_serving()` | Stop the remote API server. |

### Signals

| Signal | Payload | Emitted when |
|---|---|---|
| `hough_changed` | `HoughResult` | Hough detection runs (empty when no circles found). |
| `fit_changed` | `FitResult` | A Gaussian fit completes on either axis. |

### Result Properties

| Property | Type | Description |
|---|---|---|
| `hough_results` | `HoughResult` or `None` | Detected circles and detection parameters. `None` when detection is disabled. |
| `fit_results` | `dict` | `{"top": FitResult or None, "right": FitResult or None}` |

### Data Classes

#### `HoughResult`

| Field | Type | Description |
|---|---|---|
| `circles` | `np.ndarray` | `(N, 3)` int array with columns `[cx, cy, radius]`. |
| `param1` | `int` | Canny upper threshold used. |
| `param2` | `int` | Accumulator threshold used. |
| `min_dist` | `int` | Minimum centre distance used. |
| `min_radius` | `int` | Minimum radius filter used. |
| `max_radius` | `int` | Maximum radius filter used. |
| `stabilised` | `bool` | Whether temporal stabilisation was applied to `circles`. |
| `raw_circles` | `np.ndarray` | `(N, 3)` int array of this frame's unstabilised detections. Equal to `circles` when stabilisation is off. |

Supports `len()` (number of circles) and iteration over rows of `circles`.

#### `FitResult`

| Field | Type | Description |
|---|---|---|
| `direction` | `str` | `"top"` or `"right"`. |
| `method` | `str` | Name of the minimiser used. |
| `n_gaussians` | `int` | Number of Gaussian components. |
| `offset` | `float` | Fitted baseline offset. |
| `gaussians` | `list[dict]` | Per-component dicts with keys `"amplitude"`, `"mean"`, `"sigma"`. |
| `r_squared` | `float` | Coefficient of determination (R²). |
| `elapsed_ms` | `float` | Total fit time in milliseconds. |
| `popt` | `np.ndarray` | Raw parameter vector `[offset, A1, μ1, σ1, ...]`. |
| `y_fit` | `np.ndarray` | Fitted curve values. |
| `message` | `str` | Solver summary string. |

## Remote API

`viewer_api` lets another Python process read **and set** a running viewer's Hough and fit parameters over a loopback socket.

### Serving

```python
from image_visualiser import launch
viewer = launch()
viewer.serve()                     # -> 8765
viewer.serve(port=0)               # OS-assigned port; returns it
viewer.serve(allow_writes=False)   # publish read-only
```

or from the shell:

```bash
python3 image_visualiser.py --serve
python3 image_visualiser.py --serve 9000
python3 image_visualiser.py --serve --read-only
```

One server covers every viewer in the process, so calling `serve()` on a second viewer just returns the running port -- clients choose a viewer by name. `viewer.stop_serving()` shuts it down.

Requests are answered on the viewer's Qt main thread, so each reply is a consistent snapshot, writes land like any other GUI change, and the widgets are never touched from a socket thread. The consequence is that **the viewer's event loop must be running** -- via `launch()` in IPython, or `app.exec_()` in a script. Without one, calls fail with `ViewerAPIError` explaining exactly that.

The server binds to `127.0.0.1`, so it is not reachable off the machine. There is no authentication: binding elsewhere (`serve(host="0.0.0.0")`) exposes the viewer to anything that can reach the port, and with writes enabled that means control of its parameters, not just visibility of them. Serve read-only, or keep it on loopback.

### Connecting

```python
from viewer_api import connect

with connect() as remote:                    # 127.0.0.1:8765
    remote.hough_results.circles             # same call as on a local viewer
    remote.set_hough_params(param2=28, min_dist=40)
```

`connect(host, port, name, timeout)` returns a `RemoteViewer`. Pass `name` when several viewers are open; omit it when there is only one. Its `writable` attribute says whether the server accepts writes. Importing `viewer_api` pulls in only the standard library and NumPy -- no PyQt5, OpenCV or SciPy -- so a client can run in a bare environment, a different virtualenv, or on another machine.

### Reading

| Member | Returns | Description |
|---|---|---|
| `hough_results` | `HoughResult` or `None` | The viewer's current detection result. Property, mirroring the local attribute. |
| `hough_params()` | `dict` | `enabled`, `min_radius`, `max_radius`, `param1`, `param2`, `min_dist`, `stabilise`, `hold_frames`. |
| `fit_results` | `dict` | `{"top": FitResult or None, "right": ...}`. Property. |
| `fit(direction, include_curve=True)` | `FitResult` or `None` | One direction. `include_curve=False` drops `y_fit`, which is one value per pixel and dominates the transfer size. |
| `fit_params()` | `dict` | `method` and `n_gaussians` (the settings for the *next* fit), plus per direction the `offset`, `gaussians` and `r_squared` of the fit that ran. |
| `fit_busy` | `bool` | `True` while a fit is running or queued. Property. |
| `snapshot(include_curve=False, include_image=False, raw=False)` | `dict` | Everything above from a single moment: `name`, `frame_id`, `frame_time`, `image_shape`, `streaming`, `hough`, `hough_params`, `fits`, `fit_params`, `fit_busy`, and with `include_image` the `image` those results were measured on (greyscale, or as loaded with `raw`). |
| `measure(fit=None, include_image=False, raw=False, timeout=120.0)` | `dict` | A `snapshot()` whose `fits` are computed for its own frame: the frame's results, projections and image are copied in one step, then `fit` (`"top"`, `"right"`, `"both"`) is fitted with the current method and Gaussian count, off the GUI thread. The fits are returned, not drawn. |
| `frame_info()` | `dict` | `frame_id`, `frame_time`, `streaming`, `image_shape`. |
| `frame_id` | `int` | Count of images the viewer has loaded. Property. |
| `get_image(raw=False)` | `ndarray` or `None` | The greyscale working image, or with `raw` the image as loaded (colour included). |
| `wait_for_frame(frames=1, after=None, timeout=10.0)` | `dict` | Block until `frames` new images have arrived after frame `after` (default: now); returns `frame_info()`. |
| `list_viewers()` | `list` | Names of the viewers open in the served process. |
| `close()` | | Close the connection. Later calls reconnect on demand. |

Results arrive as the same `HoughResult` and `FitResult` objects the local API returns, so code reads identically either side of the socket. Each attribute access is one round trip, so use `snapshot()` when several values must describe the same frame.

### Writing

| Member | Returns | Description |
|---|---|---|
| `set_hough_params(**kwargs)` | `dict` | Set any of `enabled`, `min_radius`, `max_radius`, `param1`, `param2`, `min_dist`, `stabilise`, `hold_frames`. |
| `enable_hough(**kwargs)` | `dict` | Turn detection on, optionally setting parameters at the same time. |
| `disable_hough()` | `dict` | Turn detection off and clear the overlay. |
| `set_fit_params(method, n_gaussians, live_fit)` | `dict` | Set the minimiser and/or Gaussian count for the next fit. `method` is matched case-insensitively. `live_fit` switches Live Fit, which fits every stream frame as it arrives. |
| `grab()` | `dict` | Take one camera frame, load it and run detection on it; returns `frame_info()`. Refused while streaming (the stream holds the camera). |
| `start_stream()` / `stop_stream()` | `dict` | Start or stop the live stream; returns `frame_info()`. |
| `run_fit(direction="both", wait=True, timeout=120.0)` | `dict` | Run a fit and, by default, wait for it; returns `fit_params()`. |
| `cancel_fit()` | `bool` | Cancel a running fit and drop anything queued behind it. |
| `clear_fits()` | `dict` | Remove the fit curves and reset the stored results. |

Every setter returns the viewer's *resulting* parameters, so the caller sees what actually took effect rather than assuming.

**One call, one detection pass.** `set_hough_params` applies the whole batch before detection re-runs, so a six-parameter change costs one pass, not six, and the viewer never detects against a half-applied combination. (The same batching now applies to a local `enable_hough()`.)

**Failed writes change nothing.** An unknown parameter name, or a value outside its control's range, raises `ViewerAPIError` with the valid range and leaves every control untouched -- values are validated before any of them are applied. Out-of-range values are rejected rather than silently clamped.

**Fits are asynchronous.** They run in the viewer's worker thread, so `run_fit` polls `fit_busy` rather than holding the viewer or a server thread. Pass `wait=False` to return as soon as it starts -- `fit_params()` then still holds the previous result until the new one lands. A fit that outlasts `timeout` raises `ViewerAPIError` and keeps running; `cancel_fit()` stops it.

**Tying results to a frame.** `frame_id` goes up by one for every image, and `snapshot()` and `measure()` return it with the results, so a caller can tell that a measurement came from a frame taken after some event -- a stage move, say. Not streaming: `grab()`, then `measure()`. Streaming: `wait_for_frame()`, then `measure()`. Prefer `measure()` to `run_fit()` + `snapshot()` for this: while streaming, the next frame clears the viewer's fit results, and Live Fit runs the minimiser on the GUI thread, where a slow fit (seconds, for some data and methods) stalls every API call past its timeout. Stabilised circles need a circle on two consecutive frames and are smoothed over several, so after a sudden change read `hough.raw_circles`.

Images travel base64-encoded with their dtype and shape. The greyscale image is float64, but when it holds whole numbers in 0..255 it is sent as uint8 and cast back, so the client gets exactly what the viewer holds at an eighth of the size.

When the server was started with `allow_writes=False`, every one of these raises `ViewerAPIError` ("served read-only") and reads carry on working.

A `RemoteViewer` is safe to share between threads, and all failures raise `ViewerAPIError`.

Both result classes live in `results.py`, which depends only on NumPy, and gained `to_dict()` / `from_dict()` for the wire format. `from image_visualiser import HoughResult, FitResult` still works.

### Example

```bash
python3 image_visualiser.py --serve                    # terminal 1

python3 example_remote.py                              # terminal 2
python3 example_remote.py --watch 1.0                  # poll once a second
python3 example_remote.py --set param2=28 --set min_dist=40
python3 example_remote.py --method Gradient --gaussians 2 --fit both
```

```
set: param2=28, min_dist=40
fit settings: Gradient, 2 gaussian(s)
fitting both...

viewer 'demo': 320x240
  hough  : on  r=[15..60] p1=100 p2=28 min_dist=40 stabilise=True hold=3
    C1: centre=(160,120) r=37
    C2: centre=(80,60) r=19
  fit    : Gradient, 2 gaussian(s)
    top  : offset=-218.65 r2=0.6169
           amp=  3092.00 mean= 122.74 sigma= 54.38
           amp=  5589.76 mean= 197.47 sigma=  3.16
```

### IPython Example Session

```python
from image_visualiser import launch
import numpy as np

viewer = launch()

# Load an image
img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
viewer.set_image(img)

# Detect circles
viewer.enable_hough(min_radius=10, max_radius=100, param2=30)
viewer.hough_results
# HoughResult(2 circles, r=[10..100], p1=100, p2=30)
viewer.hough_results.circles
# array([[320, 240,  55],
#        [150, 300,  42]])

# Fit projections
viewer.run_fit("both")
viewer.fit_results["top"]
# FitResult(direction='top', method='Nelder-Mead', r_squared=0.998, ...)
viewer.fit_results["top"].gaussians
# [{'amplitude': 1234.0, 'mean': 312.5, 'sigma': 45.2}]

# Connect callbacks
viewer.hough_changed.connect(lambda r: print(f"{len(r)} circles\n{r.circles}"))
viewer.fit_changed.connect(lambda r: print(f"{r.direction} R²={r.r_squared:.4f}"))

# Camera
viewer.snapshot()
viewer.start_stream()
viewer.stop_stream()
```
