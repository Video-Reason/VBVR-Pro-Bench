"""
Multi-object tracking utilities backed by Norfair.

Used by evaluators that need per-object trajectories across the full video
(not just first/last frame), e.g. G-5 multi_object_placement, where:
- objects of different colors may overlap during transit
- object identity must be preserved to score path directness, fidelity, and
  hallucination (objects appearing/disappearing mid-sequence)

Design:
- One Norfair Tracker *per color*. Color is treated as a strong identity
  prior — within a synthetic VBVR scene two objects of identical color are
  rare, so per-color tracking sidesteps cross-color ID swaps when shapes
  occlude one another.
- Detection reuses HSV color masks; ranges are deliberately broader than the
  legacy detector to cover magenta and orange (which appear in G-5 GT data).
- Kalman filter survives brief full occlusion via ``hit_counter_max``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from norfair import Detection, Tracker
    _NORFAIR_AVAILABLE = True
except ImportError:
    _NORFAIR_AVAILABLE = False


# HSV ranges. Each color may have multiple ranges (red wraps the hue circle).
# Saturation/Value floors are intentionally generous to catch the soft pastel
# fills used in VBVR synthetic data.
COLOR_RANGES: Dict[str, List[Tuple[List[int], List[int]]]] = {
    'red':     [([0, 80, 80],   [10, 255, 255]),
                ([160, 80, 80], [180, 255, 255])],
    'orange':  [([11, 80, 80],  [22, 255, 255])],
    'yellow':  [([23, 80, 80],  [34, 255, 255])],
    'green':   [([35, 60, 60],  [85, 255, 255])],
    'cyan':    [([86, 60, 60],  [99, 255, 255])],
    'blue':    [([100, 60, 60], [130, 255, 255])],
    'purple':  [([131, 50, 50], [149, 255, 255])],
    'magenta': [([150, 50, 80], [170, 255, 255])],
}


@dataclass
class Detection2D:
    color: str
    center: Tuple[float, float]
    area: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class TrackPoint:
    frame_idx: int
    center: Tuple[float, float]
    area: float
    detected: bool  # False when only Kalman prediction (no actual detection)


@dataclass
class Tracklet:
    track_id: int
    color: str
    points: List[TrackPoint] = field(default_factory=list)

    @property
    def first(self) -> TrackPoint:
        return self.points[0]

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]

    @property
    def last_detected(self) -> Optional[TrackPoint]:
        """Last point that came from a real detection (not Kalman prediction).
        Returns None if the track never had any actual detection."""
        for p in reversed(self.points):
            if p.detected:
                return p
        return None

    @property
    def first_detected(self) -> Optional[TrackPoint]:
        for p in self.points:
            if p.detected:
                return p
        return None

    def trajectory(self) -> np.ndarray:
        """Nx2 array of centers in temporal order."""
        return np.array([p.center for p in self.points], dtype=float)

    def path_length(self) -> float:
        traj = self.trajectory()
        if len(traj) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))

    def displacement(self) -> float:
        if len(self.points) < 2:
            return 0.0
        a = np.array(self.first.center)
        b = np.array(self.last.center)
        return float(np.linalg.norm(b - a))

    def directness(self) -> float:
        """1.0 = perfect straight line; smaller = more meandering."""
        path = self.path_length()
        if path < 1e-6:
            return 1.0
        return self.displacement() / path

    def area_stability(self) -> float:
        """1 - normalized std of areas (capped at 1.0). High = stable size."""
        areas = np.array([p.area for p in self.points if p.detected and p.area > 0])
        if len(areas) < 2:
            return 1.0
        cv = float(np.std(areas) / (np.mean(areas) + 1e-6))
        return max(0.0, 1.0 - cv)

    def speed_uniformity(self) -> float:
        """1 - coefficient of variation of per-frame speeds. 1.0 = uniform speed."""
        traj = self.trajectory()
        if len(traj) < 3:
            return 1.0
        speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        mean = float(np.mean(speeds))
        if mean < 1e-6:
            return 1.0  # stationary — trivially "uniform"
        cv = float(np.std(speeds) / mean)
        return max(0.0, 1.0 - cv)


def detect_colored_blobs(
    frame: np.ndarray,
    min_area: float = 300.0,
    max_area: float = float('inf'),
    colors: Optional[Iterable[str]] = None,
    morph_kernel: int = 3,
) -> List[Detection2D]:
    """Detect colored blobs in a BGR frame using HSV masks.

    Parameters
    ----------
    frame : BGR uint8 image
    min_area, max_area : area filter (pixels)
    colors : restrict to a subset of COLOR_RANGES keys; None = all
    morph_kernel : size of opening kernel to suppress salt noise; 0 disables

    Returns
    -------
    List of Detection2D (one per blob, multiple per color allowed).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if colors is None:
        colors = COLOR_RANGES.keys()

    out: List[Detection2D] = []
    kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        if morph_kernel > 0
        else None
    )

    for color in colors:
        ranges = COLOR_RANGES.get(color)
        if not ranges:
            continue
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        if kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            x, y, w, h = cv2.boundingRect(cnt)
            out.append(Detection2D(color=color, center=(cx, cy), area=area, bbox=(x, y, w, h)))
    return out


def is_star_shape(contour: np.ndarray, eps_factor: float = 0.02) -> bool:
    """Heuristic: star markers approximate to ≥8 vertices."""
    approx = cv2.approxPolyDP(contour, eps_factor * cv2.arcLength(contour, True), True)
    return len(approx) >= 8


def detect_star_markers(
    frame: np.ndarray,
    min_area: float = 50.0,
    max_area: float = 2500.0,
) -> List[Detection2D]:
    """Detect star-shaped markers (small, many-vertex contours)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out: List[Detection2D] = []

    for color, ranges in COLOR_RANGES.items():
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            if not is_star_shape(cnt):
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            x, y, w, h = cv2.boundingRect(cnt)
            out.append(Detection2D(color=color, center=(cx, cy), area=area, bbox=(x, y, w, h)))
    return out


class MultiColorTracker:
    """Wraps one Norfair tracker per color.

    Per-color tracking eliminates cross-color ID swaps during occlusion.
    Within a single color, Norfair's Kalman + Hungarian assignment handles
    the rare case of two same-color objects.
    """

    def __init__(
        self,
        distance_threshold: float = 80.0,
        hit_counter_max: int = 8,
        initialization_delay: int = 1,
    ) -> None:
        if not _NORFAIR_AVAILABLE:
            raise ImportError(
                "norfair is required for MultiColorTracker. "
                "Install with: uv pip install norfair"
            )
        self.distance_threshold = distance_threshold
        self.hit_counter_max = hit_counter_max
        self.initialization_delay = initialization_delay
        self._trackers: Dict[str, Tracker] = {}
        # global_id -> Tracklet
        self.tracks: Dict[int, Tracklet] = {}
        # (color, norfair_id) -> global_id
        self._id_map: Dict[Tuple[str, int], int] = {}
        self._next_global_id = 0

    def _get_tracker(self, color: str) -> Tracker:
        if color not in self._trackers:
            self._trackers[color] = Tracker(
                distance_function='euclidean',
                distance_threshold=self.distance_threshold,
                hit_counter_max=self.hit_counter_max,
                initialization_delay=self.initialization_delay,
            )
        return self._trackers[color]

    def update(self, frame_idx: int, detections: List[Detection2D]) -> None:
        # Group detections by color
        by_color: Dict[str, List[Detection2D]] = {}
        for d in detections:
            by_color.setdefault(d.color, []).append(d)

        # Always step every existing tracker so dormant tracks age out
        all_colors = set(self._trackers.keys()) | set(by_color.keys())
        for color in all_colors:
            tracker = self._get_tracker(color)
            color_dets = by_color.get(color, [])
            # Tag each detection with the current frame_idx so we can later
            # tell whether obj.last_detection came from THIS frame (truly
            # detected) or a prior frame (Kalman coasting).
            norfair_dets = [
                Detection(
                    points=np.array([d.center]),
                    data={'area': d.area, 'frame_idx': frame_idx},
                )
                for d in color_dets
            ]
            tracked_objs = tracker.update(detections=norfair_dets)

            for obj in tracked_objs:
                key = (color, obj.id)
                if key not in self._id_map:
                    self._id_map[key] = self._next_global_id
                    self.tracks[self._next_global_id] = Tracklet(
                        track_id=self._next_global_id, color=color
                    )
                    self._next_global_id += 1
                gid = self._id_map[key]

                center = tuple(obj.estimate[0])
                detected_now = (
                    obj.last_detection is not None
                    and obj.last_detection.data.get('frame_idx') == frame_idx
                )
                area = (
                    obj.last_detection.data.get('area', 0.0)
                    if detected_now and obj.last_detection is not None
                    else 0.0
                )
                self.tracks[gid].points.append(
                    TrackPoint(frame_idx=frame_idx, center=center, area=area, detected=detected_now)
                )

    def active_tracklets(self, min_length: int = 2) -> List[Tracklet]:
        return [t for t in self.tracks.values() if len(t.points) >= min_length]


def track_video(
    frames: List[np.ndarray],
    *,
    min_area: float = 300.0,
    max_area: float = float('inf'),
    colors: Optional[Iterable[str]] = None,
    distance_threshold: float = 80.0,
    hit_counter_max: int = 8,
    initialization_delay: int = 1,
) -> MultiColorTracker:
    """Convenience: detect + track a sequence of BGR frames in one call."""
    tracker = MultiColorTracker(
        distance_threshold=distance_threshold,
        hit_counter_max=hit_counter_max,
        initialization_delay=initialization_delay,
    )
    for i, frame in enumerate(frames):
        dets = detect_colored_blobs(frame, min_area=min_area, max_area=max_area, colors=colors)
        tracker.update(i, dets)
    return tracker


def _bbox_contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]) -> bool:
    """True if ``inner`` bbox sits inside ``outer`` (no tolerance)."""
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ix >= ox and iy >= oy and ix + iw <= ox + ow and iy + ih <= oy + oh


def find_bordered_object(
    frame: np.ndarray,
    border_color: str = 'green',
    min_border_area: float = 500.0,
    min_interior_area: float = 500.0,
    min_interior_ratio: float = 0.20,
) -> Optional[Tuple[Detection2D, Detection2D]]:
    """Find an object visually "marked" by a colored ring border.

    Returns ``(border, interior)`` where ``interior`` is a non-border-color
    blob whose bbox sits inside the border's bbox and whose area is at least
    ``min_interior_ratio`` of the border-bbox area.

    Distinguishes a ring-shaped marker from a solid-filled shape of the same
    color: a solid green rectangle has no qualifying non-green object inside,
    while a green ring around a purple square does. If multiple borders match,
    the one with the largest interior object wins.
    """
    borders = detect_colored_blobs(frame, min_area=min_border_area, colors=[border_color])
    if not borders:
        return None

    other_colors = [c for c in COLOR_RANGES.keys() if c != border_color]
    candidates = detect_colored_blobs(frame, min_area=min_interior_area, colors=other_colors)

    best: Optional[Tuple[Detection2D, Detection2D]] = None
    best_area = -1.0
    for border in borders:
        _, _, bw, bh = border.bbox
        border_bbox_area = float(bw * bh)
        if border_bbox_area <= 0:
            continue
        for obj in candidates:
            if not _bbox_contains(border.bbox, obj.bbox):
                continue
            if obj.area / border_bbox_area < min_interior_ratio:
                continue
            if obj.area > best_area:
                best = (border, obj)
                best_area = obj.area
    return best


def find_star_marked_object(
    frame: np.ndarray,
    star_color: str = 'red',
    star_min_area: float = 50.0,
    star_max_area: float = 2500.0,
    host_min_area: float = 500.0,
) -> Optional[Tuple[Detection2D, Detection2D]]:
    """Find the object hosting a small star marker of ``star_color``.

    Returns ``(host, star)``. Two-stage:
    1. Try ``detect_star_markers`` (external star-shaped contours) and match
       each star to the enclosing non-star-colored host.
    2. Fallback: if the star is the same color as its host (so it has no
       external contour — e.g., a red star inside a red circle), pick the
       smallest host-sized blob of that color as host and synthesize a
       star-at-center. Callers that need a real star location should treat
       this as best-effort.
    """
    all_hosts = detect_colored_blobs(frame, min_area=host_min_area)
    stars = detect_star_markers(frame, min_area=star_min_area, max_area=star_max_area)

    # Stage 1: star has its own contour (different color from host)
    for star in stars:
        for host in all_hosts:
            if host.color == star.color:
                continue
            if _bbox_contains(host.bbox, star.bbox):
                return (host, star)

    # Stage 2: star absorbed into same-color host. Find the smallest blob of
    # star_color — stars are small vs. typical objects.
    same_color_blobs = [h for h in all_hosts if h.color == star_color]
    if not same_color_blobs:
        return None
    host = min(same_color_blobs, key=lambda d: d.area)
    synth_star = Detection2D(
        color=star_color, center=host.center, area=0.0, bbox=host.bbox
    )
    return (host, synth_star)


def per_color_hungarian(
    gen_centers_by_color: Dict[str, List[Tuple[float, float]]],
    gt_centers_by_color: Dict[str, List[Tuple[float, float]]],
) -> Dict[Tuple[str, int], Tuple[int, float]]:
    """Per-color Hungarian matching on 2D centers.

    Returns ``(color, gt_local_idx) -> (gen_local_idx, distance)`` where the
    indices are positions within each per-color list (not the original lists).
    Only matches within the same color. Callers own the index translation back
    to global indices.
    """
    from scipy.optimize import linear_sum_assignment

    result: Dict[Tuple[str, int], Tuple[int, float]] = {}
    for color, gt_centers in gt_centers_by_color.items():
        gen_centers = gen_centers_by_color.get(color, [])
        if not gen_centers or not gt_centers:
            continue
        cost = np.zeros((len(gen_centers), len(gt_centers)))
        for i, gc in enumerate(gen_centers):
            for j, tc in enumerate(gt_centers):
                cost[i, j] = float(np.hypot(gc[0] - tc[0], gc[1] - tc[1]))
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            result[(color, int(c))] = (int(r), float(cost[r, c]))
    return result


def _mean_center_distance(a: Tracklet, b: Tracklet) -> float:
    """Mean distance between center points of two tracklets on the frames
    where both have *detected* points. Returns inf if they never co-occur.
    """
    a_by_frame = {p.frame_idx: p.center for p in a.points if p.detected}
    b_by_frame = {p.frame_idx: p.center for p in b.points if p.detected}
    common = a_by_frame.keys() & b_by_frame.keys()
    if not common:
        return float('inf')
    dists = [
        float(np.hypot(a_by_frame[f][0] - b_by_frame[f][0],
                       a_by_frame[f][1] - b_by_frame[f][1]))
        for f in common
    ]
    return float(np.mean(dists))


def count_moved_tracklets(
    tracklets: List[Tracklet],
    motion_tolerance: float = 20.0,
    exclude_ids: Optional[Iterable[int]] = None,
    reference_tracklet: Optional[Tracklet] = None,
    attach_tolerance: float = 40.0,
) -> int:
    """Count tracklets whose first→last displacement exceeds ``motion_tolerance``.

    Used to penalize "only the marked object should move" tasks: pass in the
    marked tracklet's id via ``exclude_ids``.

    ``reference_tracklet`` + ``attach_tolerance`` ignore tracklets that ride
    along with the reference — an annotation ring drawn on top of the marked
    object, or an HSV "shadow" detection of the same physical blob under a
    neighboring color range (e.g. pink matching both ``red`` and ``magenta``).
    Any tracklet whose mean co-occurring center distance to the reference is
    under ``attach_tolerance`` is treated as attached and skipped.
    """
    excluded = set(exclude_ids) if exclude_ids is not None else set()
    n = 0
    for t in tracklets:
        if t.track_id in excluded:
            continue
        if reference_tracklet is not None and t.track_id != reference_tracklet.track_id:
            if _mean_center_distance(t, reference_tracklet) < attach_tolerance:
                continue
        if t.displacement() > motion_tolerance:
            n += 1
    return n
