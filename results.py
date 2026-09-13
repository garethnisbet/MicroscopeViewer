"""Result data classes shared by the viewer and its remote API.

Kept free of PyQt, OpenCV and SciPy so a client process can import the
types over the wire without the viewer's GUI dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

__all__ = ["HoughResult", "FitResult"]


@dataclass
class HoughResult:
    """Result of a Hough circle detection run.

    Attributes
    ----------
    circles : np.ndarray
        ``(N, 3)`` int array with columns ``[cx, cy, radius]``.
        Shape ``(0, 3)`` when no circles are found.
    param1 : int
        Canny upper threshold used.
    param2 : int
        Accumulator threshold used.
    min_dist : int
        Minimum centre distance used.
    min_radius : int
        Minimum radius used.
    max_radius : int
        Maximum radius used.
    stabilised : bool
        Whether temporal stabilisation was applied to ``circles``.
    raw_circles : np.ndarray
        ``(N, 3)`` int array of this frame's unstabilised detections,
        straight out of ``cv2.HoughCircles``.  Equal to ``circles``
        when stabilisation is off.
    """
    circles: np.ndarray
    param1: int
    param2: int
    min_dist: int
    min_radius: int
    max_radius: int
    stabilised: bool = False
    raw_circles: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=int)
    )

    def __len__(self):
        return len(self.circles)

    def __iter__(self):
        return iter(self.circles)

    def __repr__(self):
        return (f"HoughResult({len(self.circles)} circles, "
                f"r=[{self.min_radius}..{self.max_radius}], "
                f"p1={self.param1}, p2={self.param2})")

    def to_dict(self) -> dict:
        """Return a JSON-safe dict (arrays become nested lists)."""
        return dict(
            circles=np.asarray(self.circles).tolist(),
            param1=int(self.param1),
            param2=int(self.param2),
            min_dist=int(self.min_dist),
            min_radius=int(self.min_radius),
            max_radius=int(self.max_radius),
            stabilised=bool(self.stabilised),
            raw_circles=np.asarray(self.raw_circles).tolist(),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "HoughResult":
        """Rebuild from :meth:`to_dict` output."""
        def arr(v):
            a = np.asarray(v, dtype=int)
            return a.reshape(0, 3) if a.size == 0 else a
        return cls(
            circles=arr(d["circles"]),
            param1=d["param1"],
            param2=d["param2"],
            min_dist=d["min_dist"],
            min_radius=d["min_radius"],
            max_radius=d["max_radius"],
            stabilised=d.get("stabilised", False),
            raw_circles=arr(d.get("raw_circles", [])),
        )


@dataclass
class FitResult:
    """Result of a Gaussian fit on one projection axis."""
    direction: str              # "top" or "right"
    method: str
    n_gaussians: int
    offset: float
    gaussians: List[dict]       # [{"amplitude", "mean", "sigma"}, ...]
    r_squared: float
    elapsed_ms: float
    popt: np.ndarray            # raw parameter vector
    y_fit: np.ndarray           # fitted curve values
    message: str                # solver summary string

    def to_dict(self, include_curve: bool = True) -> dict:
        """Return a JSON-safe dict.

        *include_curve* keeps ``y_fit``, which is one value per pixel
        along the projection and dominates the payload size; drop it
        when only the fitted parameters are wanted.
        """
        return dict(
            direction=self.direction,
            method=self.method,
            n_gaussians=int(self.n_gaussians),
            offset=float(self.offset),
            gaussians=[{k: float(v) for k, v in g.items()}
                       for g in self.gaussians],
            r_squared=float(self.r_squared),
            elapsed_ms=float(self.elapsed_ms),
            popt=np.asarray(self.popt, dtype=float).tolist(),
            y_fit=(np.asarray(self.y_fit, dtype=float).tolist()
                   if include_curve else None),
            message=self.message,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "FitResult":
        """Rebuild from :meth:`to_dict` output."""
        y_fit = d.get("y_fit")
        return cls(
            direction=d["direction"],
            method=d["method"],
            n_gaussians=d["n_gaussians"],
            offset=d["offset"],
            gaussians=list(d["gaussians"]),
            r_squared=d["r_squared"],
            elapsed_ms=d["elapsed_ms"],
            popt=np.asarray(d["popt"], dtype=float),
            y_fit=(np.empty(0) if y_fit is None
                   else np.asarray(y_fit, dtype=float)),
            message=d["message"],
        )
