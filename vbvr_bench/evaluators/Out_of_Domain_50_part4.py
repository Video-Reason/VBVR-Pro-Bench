"""
Specific evaluators for Out-of-Domain_50 tasks (Part 4).
"""

import numpy as np
import cv2
from collections import deque
from itertools import permutations
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import denoise_contour
from ..utils import normalize_frame_size, compute_ssim, safe_distance, COLOR_BOUNDS
from .Out_of_Domain_50_part1 import SeparateObjectsNoSpinEvaluator as _ShapeDetectBase
from .utils import maze
from .utils.tracking import track_video


# ---------------------------------------------------------------------------
# O-39 helper functions (variable-size maze grid)
# ---------------------------------------------------------------------------

def _maze_bfs(
    start: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    grid_size: int = 15,
) -> Dict[Tuple[int, int], int]:
    """BFS on a maze grid.  Returns {(row, col): distance_from_start}."""
    dist: Dict[Tuple[int, int], int] = {start: 0}
    q: deque = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = (r + dr, c + dc)
            if (
                0 <= nb[0] < grid_size
                and 0 <= nb[1] < grid_size
                and nb not in dist
                and nb not in walls
            ):
                dist[nb] = dist[(r, c)] + 1
                q.append(nb)
    return dist


def _maze_optimal_cell_set(
    start: Tuple[int, int],
    end: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    grid_size: int = 15,
) -> Optional[Tuple[FrozenSet[Tuple[int, int]], Dict[Tuple[int, int], int], int]]:
    """All cells on *some* shortest path from start to end.

    Returns (optimal_cells, dist_from_start, shortest_distance) or None.
    """
    dist_s = _maze_bfs(start, walls, grid_size)
    if end not in dist_s:
        return None
    dist_e = _maze_bfs(end, walls, grid_size)
    shortest = dist_s[end]
    optimal = frozenset(
        c for c in dist_s if c in dist_e and dist_s[c] + dist_e[c] == shortest
    )
    return optimal, dist_s, shortest


def _maze_cell_center_px(
    cell: Tuple[int, int], frame_shape: Tuple[int, ...], grid_size: int = 15,
) -> Tuple[int, int]:
    """(row, col) -> pixel (x, y) at cell centre."""
    h, w = frame_shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    return (cell[1] * cell_w + cell_w // 2, cell[0] * cell_h + cell_h // 2)


def _maze_pixel_to_cell(
    px: int, py: int, frame_shape: Tuple[int, ...], grid_size: int = 15,
) -> Tuple[int, int]:
    """Pixel (x, y) -> (row, col)."""
    h, w = frame_shape[:2]
    return (
        min(max(py * grid_size // h, 0), grid_size - 1),
        min(max(px * grid_size // w, 0), grid_size - 1),
    )



class SymbolDeletionEvaluator(BaseEvaluator):
    """
    O-5: Symbol deletion evaluator.
    
    Evaluator for Symbol Deletion Task.

    Task: Delete the symbol inside the red bounding box while keeping everything else exactly the same.

    Scoring (Total 100% comparing final_frame vs gt_final_frame):
    - delete_accuracy (60%): The region INSIDE the red box must be empty (match GT final).
    - keep_accuracy   (20%): The foreground region OUTSIDE the red box must be intact.
    - bg_consistency  (20%): The background (outside the main Y-range) must be unchanged.
    """

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate symbol deletion task."""
        if len(video_frames) < 1:
            return 0.0

        final_frame = video_frames[-1]
        if gt_final_frame is None:
            gt_final_frame = gt_frames[-1]
        h, w = gt_final_frame.shape[:2]

        if final_frame.shape[:2] != (h, w):
            final_frame = cv2.resize(final_frame, (w, h))

        try:
            g_gray = cv2.cvtColor(gt_final_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            base_err = float(np.mean(cv2.absdiff(gt_final_frame, final_frame)))
            best_err, best_img = base_err, None
            for sc in (0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20):
                if sc == 1.0:
                    cand0 = final_frame
                else:
                    M0 = np.float32([[sc, 0, (1 - sc) * w / 2.0], [0, sc, (1 - sc) * h / 2.0]])
                    cand0 = cv2.warpAffine(final_frame, M0, (w, h), borderValue=(255, 255, 255))
                c_gray = cv2.cvtColor(cand0, cv2.COLOR_BGR2GRAY).astype(np.float32)
                (dx, dy), _ = cv2.phaseCorrelate(g_gray, c_gray)
                if float(np.hypot(dx, dy)) >= 0.25 * max(h, w):
                    continue
                cand = cv2.warpAffine(
                    cand0, np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h),
                    borderValue=(255, 255, 255),
                )
                err = float(np.mean(cv2.absdiff(gt_final_frame, cand)))
                if err < best_err:
                    best_err, best_img = err, cand
            if best_img is not None:
                final_frame = best_img
        except cv2.error:
            pass

        outer_bbox, inner_bbox = self.find_red_hollow_box(gt_final_frame, COLOR_BOUNDS['red'])
        
        if outer_bbox is None:
            return 0.0 

        y_min, y_max = self.get_foreground_y_range(gt_final_frame, bg_color=(255, 255, 255))

        delete_mask = np.zeros((h, w), dtype=bool)
        x_in, y_in, w_in, h_in = inner_bbox
        delete_mask[y_in:y_in+h_in, x_in:x_in+w_in] = True

        keep_mask = np.zeros((h, w), dtype=bool)
        keep_mask[y_min:y_max, :] = True
        x_out, y_out, w_out, h_out = outer_bbox
        keep_mask[y_out:y_out+h_out, x_out:x_out+w_out] = False 

        bg_mask = np.ones((h, w), dtype=bool)
        bg_mask[y_min:y_max, :] = False

        delete_score = self.score_masked_similarity(gt_final_frame, final_frame, delete_mask,low_tolerance=0.02,expand_ratio=10)

        keep_score = self.score_masked_similarity(gt_final_frame, final_frame, keep_mask,low_tolerance=0.02,expand_ratio=10)

        bg_score = self.score_masked_similarity(gt_final_frame, final_frame, bg_mask,low_tolerance=0.02,expand_ratio=5)

        score = (delete_score * 0.6) + (keep_score * 0.2) + (bg_score * 0.2)

        self._last_task_details = {
            'delete_score': delete_score,
            'keep_score': keep_score,
            'bg_score': bg_score,
            'y_range': (y_min, y_max),
            'red_box': outer_bbox
        }
        
        return float(score)

    def find_red_hollow_box(self, image: np.ndarray, red_bounds: tuple) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        if image is None or image.size == 0:
            return None, None

        h, w = image.shape[:2]

        bounds = np.asarray(red_bounds)
        if bounds.shape == (2, 3):
            lower_bgr = np.array([bounds[0, 2], bounds[0, 1], bounds[0, 0]], dtype=np.uint8)
            upper_bgr = np.array([bounds[1, 2], bounds[1, 1], bounds[1, 0]], dtype=np.uint8)
        else:
            lower_bgr = np.array(bounds[0], dtype=np.uint8)
            upper_bgr = np.array(bounds[1], dtype=np.uint8)

        mask = cv2.inRange(image, lower_bgr, upper_bgr)

        # Close small gaps in the red outline
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        best = None
        best_area = 0.0

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 200:  # filter noise
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue

            rect_area = float(bw * bh)
            if rect_area <= 0:
                continue

            aspect = bw / float(bh)
            # Keep rectangle-like shapes
            if aspect < 0.5 or aspect > 2.0:
                continue

            # Prefer large contours that also occupy a reasonable part of their bounding rect
            extent = area / rect_area
            if extent < 0.05:
                continue

            if area > best_area:
                best_area = area
                best = (x, y, bw, bh)

        if best is None:
            return None, None

        x, y, bw, bh = best
        # Estimate border thickness by shrinking a small fraction of box size
        shrink = max(2, int(round(min(bw, bh) * 0.06)))
        shrink = min(shrink, max(0, (bw - 1) // 2), max(0, (bh - 1) // 2))

        x_in = int(max(0, x + shrink))
        y_in = int(max(0, y + shrink))
        w_in = int(max(1, min(w - x_in, bw - 2 * shrink)))
        h_in = int(max(1, min(h - y_in, bh - 2 * shrink)))

        outer_bbox = (int(x), int(y), int(bw), int(bh))
        inner_bbox = (x_in, y_in, w_in, h_in)
        return outer_bbox, inner_bbox

    def get_foreground_y_range(self, image: np.ndarray, bg_color=(255, 255, 255)) -> Tuple[int, int]:
        if image is None or image.size == 0:
            return 0, 0

        h, w = image.shape[:2]
        bg = np.array(bg_color, dtype=image.dtype).reshape((1, 1, -1))

        if image.ndim == 2:
            # grayscale fallback
            non_bg = image != bg_color[0]
        else:
            # any channel differs from background means foreground
            non_bg = np.any(image != bg, axis=2)

        ys = np.where(non_bg)[0]
        if ys.size == 0:
            return 0, h

        y_min = int(ys.min())
        y_max = int(ys.max()) + 1  # make it half-open [y_min, y_max)

        # Small padding for robustness
        pad = 2
        y_min = max(0, y_min - pad)
        y_max = min(h, y_max + pad)
        if y_max <= y_min:
            return 0, h
        return y_min, y_max

    def score_masked_similarity(self, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, low_tolerance: float, expand_ratio: float) -> float:
        if img1 is None or img2 is None or mask is None:
            return 0.0

        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Ensure mask is boolean and correct shape
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        if mask.shape != img1.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        if img1.ndim == 2:
            p1 = img1[mask].astype(np.float32)
            p2 = img2[mask].astype(np.float32)
            if p1.size == 0:
                return 1.0
            diff = np.abs(p1 - p2)
            mean_diff = float(np.mean(diff))
        else:
            p1 = img1[mask].astype(np.float32)
            p2 = img2[mask].astype(np.float32)
            if p1.size == 0:
                return 1.0
            diff = np.abs(p1 - p2)  # (N, 3)
            mean_diff = float(np.mean(diff))  
        if mean_diff/255.0 < low_tolerance:
            return 1.0
        else:
            score = 1.0 - (mean_diff / 255.0) * expand_ratio
            if score < 0.0:
                return 0.0
            return score



class GeometricTransformationEvaluator(BaseEvaluator):
    """
    O-6: planar rotation around a marked pivot.

    The old scorer was a loose weighted sum over centre variance, final angle,
    final position, and shape area. That allows obvious cheats:

    - teleporting straight to the target pose
    - drifting on the wrong radius around the pivot
    - getting partial credit from one strong term while missing the core
      "rotate this shape around this pivot into that outline" contract

    We instead score the two essential requirements multiplicatively:

        score = final_pose × orbital_motion

    Where:
    - ``final_pose`` checks that the final filled shape overlaps the GT final
      shape at the right position / size
    - ``orbital_motion`` checks that the detected centre trace stays on the GT
      orbit and that its travelled path length matches the GT process rather
      than a straight teleport chord
    """

    MIN_SHAPE_AREA = 800.0
    PIVOT_MIN_AREA = 20.0
    PIVOT_MAX_AREA = 600.0
    CENTER_FULL_PX = 2.0
    CENTER_HALF_PX = 35.0
    FINAL_IOU_SATURATION = 0.80
    AREA_RATIO_SATURATION = 0.85
    RADIUS_STABILITY_FRAC = 0.40
    RADIUS_JITTER_TOL_PX = 1.0
    RADIUS_MATCH_TOL_PX = 0.5
    PATH_LENGTH_TOL_PX = 1.0
    ACTIVE_STEP_PX = 1.5
    ACTIVE_COVERAGE_REF_FRAC = 0.50
    MIN_ACTIVE_STEPS_FOR_FULL_COVERAGE = 1.0
    MAX_STEP_TOL_FRAC = 0.50
    INTERLEAVE_STEP_TOL_FRAC = 1.5
    MAX_STEP_TOL_MIN_PX = 20.0
    TRACK_DISTANCE_THRESHOLD = 90.0
    MIN_TRACK_COVERAGE = 0.35

    def _detect_main_shape(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Return the dominant filled shape in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 30, 40], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.MIN_SHAPE_AREA]
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        center = (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
        return {
            "center": center,
            "area": float(cv2.contourArea(contour)),
            "angle": float(cv2.minAreaRect(contour)[2]),
            "contour": contour,
        }

    def _detect_pivot(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect the small outlined pivot marker."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark = cv2.inRange(gray, 0, 120)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.PIVOT_MIN_AREA or area > self.PIVOT_MAX_AREA:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.45:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center = (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
            candidates.append((circularity, area, center))

        if not candidates:
            return None
        _, _, center = max(candidates, key=lambda item: (item[0], -abs(item[1] - 120.0)))
        return center

    def _contour_iou(
        self,
        contour_a: np.ndarray,
        contour_b: np.ndarray,
        frame_shape: Tuple[int, int, int],
    ) -> float:
        ax, ay, aw, ah = cv2.boundingRect(contour_a)
        bx, by, bw, bh = cv2.boundingRect(contour_b)
        x1 = max(0, min(ax, bx) - 5)
        y1 = max(0, min(ay, by) - 5)
        x2 = min(frame_shape[1], max(ax + aw, bx + bw) + 5)
        y2 = min(frame_shape[0], max(ay + ah, by + bh) + 5)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        mask_a = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        mask_b = np.zeros_like(mask_a)
        offset_a = np.array([[[x1, y1]]], dtype=contour_a.dtype)
        offset_b = np.array([[[x1, y1]]], dtype=contour_b.dtype)
        cv2.drawContours(mask_a, [contour_a - offset_a], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask_b, [contour_b - offset_b], -1, 255, thickness=cv2.FILLED)

        intersection = int(np.logical_and(mask_a > 0, mask_b > 0).sum())
        union = int(np.logical_or(mask_a > 0, mask_b > 0).sum())
        return float(intersection / union) if union > 0 else 0.0

    @staticmethod
    def _angle_span(a0: float, a1: float) -> float:
        delta = abs(a1 - a0) % (2.0 * np.pi)
        return min(delta, 2.0 * np.pi - delta)

    def _trace_shape_centers(
        self, video_frames: Sequence[np.ndarray],
    ) -> List[Tuple[float, float]]:
        trace: List[Tuple[float, float]] = []
        for frame in video_frames:
            shape = self._detect_main_shape(frame)
            if shape is not None:
                trace.append(shape["center"])
        return trace

    @staticmethod
    def _path_length(trace: Sequence[Tuple[float, float]]) -> float:
        if len(trace) < 2:
            return 0.0
        return float(sum(
            safe_distance(trace[i - 1], trace[i]) for i in range(1, len(trace))
        ))

    def _motion_profile(self, trace: Sequence[Tuple[float, float]]) -> Dict[str, float]:
        if len(trace) < 2:
            return {
                "path_length": 0.0,
                "active_steps": 0.0,
                "max_step": 0.0,
            }

        steps = [
            safe_distance(trace[i - 1], trace[i]) for i in range(1, len(trace))
        ]
        return {
            "path_length": float(sum(steps)),
            "active_steps": float(sum(step >= self.ACTIVE_STEP_PX for step in steps)),
            "max_step": float(max(steps) if steps else 0.0),
        }

    def _tracked_shape_centers(
        self,
        video_frames: Sequence[np.ndarray],
    ) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
        """Return the main moving shape trajectory, preferring the shared tracker."""
        frames = list(video_frames)
        if len(frames) < 2:
            return [], {"trace_source": "none", "n_tracklets": 0}

        first_shape = self._detect_main_shape(frames[0])
        try:
            tracker = track_video(
                frames,
                min_area=self.MIN_SHAPE_AREA,
                distance_threshold=self.TRACK_DISTANCE_THRESHOLD,
                hit_counter_max=8,
            )
            tracklets = tracker.active_tracklets(min_length=2)
        except Exception:
            tracklets = []

        if tracklets:
            def track_score(tracklet: Any) -> Tuple[int, float, float]:
                detected = [p for p in tracklet.points if p.detected]
                first_detected = tracklet.first_detected or tracklet.first
                if first_shape is None:
                    first_cost = 0.0
                else:
                    first_cost = safe_distance(first_detected.center, first_shape["center"])
                return (len(detected), -first_cost, tracklet.area_stability())

            best = max(tracklets, key=track_score)
            detected_points = [p.center for p in best.points if p.detected]
            min_points = max(3, int(round(len(frames) * self.MIN_TRACK_COVERAGE)))
            if len(detected_points) >= min_points:
                circle = self._fit_circle(detected_points)
                if circle is not None:
                    _, radius, radius_std = circle
                    if radius_std <= max(radius * 0.25, 2.0):
                        return detected_points, {
                            "trace_source": "tracker",
                            "track_color": best.color,
                            "track_id": best.track_id,
                            "n_tracklets": len(tracklets),
                            "detected_points": len(detected_points),
                            "track_fit_std": float(radius_std),
                        }

        contour_trace = self._trace_shape_centers(frames)
        return contour_trace, {
            "trace_source": "contour",
            "n_tracklets": len(tracklets),
            "detected_points": len(contour_trace),
        }

    @staticmethod
    def _fit_circle(
        trace: Sequence[Tuple[float, float]],
    ) -> Optional[Tuple[Tuple[float, float], float, float]]:
        """Least-squares circle fit for the GT object trajectory."""
        if len(trace) < 3:
            return None

        pts = np.asarray(trace, dtype=np.float64)
        x = pts[:, 0]
        y = pts[:, 1]
        a = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(pts))])
        b = x * x + y * y
        try:
            cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        radius_sq = float(c + cx * cx + cy * cy)
        if radius_sq <= 1.0:
            return None
        radius = float(np.sqrt(radius_sq))
        radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        radius_std = float(np.std(radii))
        return (float(cx), float(cy)), radius, radius_std

    def _reference_orbit(
        self,
        gt_frames: Sequence[np.ndarray],
        gt_start_center: Tuple[float, float],
        gt_final_center: Tuple[float, float],
        marker_pivot: Optional[Tuple[float, float]],
    ) -> Optional[Dict[str, Any]]:
        gt_trace, trace_details = self._tracked_shape_centers(gt_frames)
        circle = self._fit_circle(gt_trace)
        if circle is not None:
            center, radius, radius_std = circle
            if radius > 1.0 and radius_std <= max(radius * 0.25, 2.0):
                return {
                    "center": center,
                    "radius": radius,
                    **self._motion_profile(gt_trace),
                    "source": "gt_trace_fit",
                    "fit_std": radius_std,
                    "gt_trace_len": len(gt_trace),
                    **trace_details,
                }

        if marker_pivot is None:
            return None

        radius = safe_distance(gt_start_center, marker_pivot)
        expected_angle = self._angle_span(
            float(np.arctan2(gt_start_center[1] - marker_pivot[1], gt_start_center[0] - marker_pivot[0])),
            float(np.arctan2(gt_final_center[1] - marker_pivot[1], gt_final_center[0] - marker_pivot[0])),
        )
        return {
            "center": marker_pivot,
            "radius": radius,
            "path_length": radius * expected_angle,
            "active_steps": 0.0,
            "max_step": 0.0,
            "source": "marker_pivot",
            "fit_std": 0.0,
            "gt_trace_len": len(gt_trace),
            **trace_details,
        }

    def _orbital_motion_score(
        self,
        trace: Sequence[Tuple[float, float]],
        reference_orbit: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        if len(trace) < 2:
            return 0.0, {
                "expected_radius": 0.0,
                "mean_radius": 0.0,
                "radius_match": 0.0,
                "radius_stability": 0.0,
                "path_length": 0.0,
                "expected_arc_length": 0.0,
                "path_score": 0.0,
            }

        orbit_center = reference_orbit["center"]
        expected_radius = float(reference_orbit["radius"])
        expected_arc_length = float(reference_orbit["path_length"])

        radii = [safe_distance(center, orbit_center) for center in trace]
        mean_radius = float(np.mean(radii)) if radii else 0.0
        std_radius = float(np.std(radii)) if radii else 0.0
        motion = self._motion_profile(trace)
        path_length = motion["path_length"]
        active_steps = motion["active_steps"]
        max_step = motion["max_step"]

        radius_delta = abs(mean_radius - expected_radius)
        if expected_radius <= 0:
            radius_match = 0.0
        elif radius_delta <= self.RADIUS_MATCH_TOL_PX:
            radius_match = 1.0
        else:
            radius_match = min(mean_radius, expected_radius) / max(
                mean_radius, expected_radius, 1.0,
            )
        radius_jitter = max(0.0, std_radius - self.RADIUS_JITTER_TOL_PX)
        _jitter_scale = max(expected_radius * self.RADIUS_STABILITY_FRAC, 1.0)
        radius_stability = _jitter_scale / max(_jitter_scale, radius_jitter)
        if expected_arc_length > 1.0 and path_length > 1.0:
            if path_length + self.PATH_LENGTH_TOL_PX >= expected_arc_length:
                path_score = 1.0
            else:
                path_score = min(1.0, path_length / expected_arc_length)
        else:
            path_score = 0.0

        expected_active_steps = float(reference_orbit.get("active_steps", 0.0))
        coverage_target = max(
            self.MIN_ACTIVE_STEPS_FOR_FULL_COVERAGE,
            expected_active_steps * self.ACTIVE_COVERAGE_REF_FRAC,
        )
        max_step_tolerance = max(
            self.MAX_STEP_TOL_MIN_PX,
            expected_radius * self.MAX_STEP_TOL_FRAC,
        )
        step_continuity = (
            1.0 if max_step <= max_step_tolerance
            else max_step_tolerance / max(max_step, 1.0)
        )
        if (
            expected_active_steps <= 2.0
            and path_score >= 0.95
            and max_step <= self.ACTIVE_STEP_PX
        ):
            motion_coverage = 1.0
        else:
            motion_coverage = min(1.0, active_steps / coverage_target) if coverage_target > 0 else 0.0
        motion_distribution = motion_coverage * step_continuity

        total = radius_match * radius_stability * path_score * motion_distribution
        return total, {
            "expected_radius": float(expected_radius),
            "mean_radius": float(mean_radius),
            "radius_delta": float(radius_delta),
            "radius_match": float(radius_match),
            "radius_std": float(std_radius),
            "radius_jitter": float(radius_jitter),
            "radius_stability": float(radius_stability),
            "path_length": float(path_length),
            "expected_arc_length": float(expected_arc_length),
            "path_score": float(path_score),
            "active_steps": float(active_steps),
            "expected_active_steps": float(expected_active_steps),
            "motion_coverage": float(motion_coverage),
            "max_step": float(max_step),
            "max_step_tolerance": float(max_step_tolerance),
            "step_continuity": float(step_continuity),
            "motion_distribution": float(motion_distribution),
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2 or gt_first_frame is None or gt_final_frame is None:
            return 0.0

        gt_start_shape = self._detect_main_shape(gt_first_frame)
        gt_final_shape = self._detect_main_shape(gt_final_frame)
        marker_pivot = self._detect_pivot(gt_first_frame)
        pred_final_shape = self._detect_main_shape(video_frames[-1])
        reference_orbit = (
            self._reference_orbit(
                gt_frames if gt_frames else [gt_first_frame, gt_final_frame],
                gt_start_shape["center"] if gt_start_shape is not None else (0.0, 0.0),
                gt_final_shape["center"] if gt_final_shape is not None else (0.0, 0.0),
                marker_pivot,
            )
            if gt_start_shape is not None and gt_final_shape is not None
            else None
        )

        if gt_start_shape is None or gt_final_shape is None or reference_orbit is None or pred_final_shape is None:
            self._last_task_details = {
                "error": "detection_failed",
                "gt_start_shape": gt_start_shape is not None,
                "gt_final_shape": gt_final_shape is not None,
                "pivot_found": marker_pivot is not None,
                "reference_orbit_found": reference_orbit is not None,
                "pred_final_shape": pred_final_shape is not None,
            }
            return 0.0

        final_iou = self._contour_iou(
            pred_final_shape["contour"],
            gt_final_shape["contour"],
            video_frames[-1].shape,
        )
        area_ratio = min(pred_final_shape["area"], gt_final_shape["area"]) / max(
            pred_final_shape["area"], gt_final_shape["area"], 1.0,
        )
        center_dist = safe_distance(pred_final_shape["center"], gt_final_shape["center"])

        iou_score = min(1.0, final_iou / self.FINAL_IOU_SATURATION)
        area_score = min(1.0, area_ratio / self.AREA_RATIO_SATURATION)
        if center_dist <= self.CENTER_FULL_PX:
            center_score = 1.0
        else:
            center_score = max(
                0.0,
                1.0 - (center_dist - self.CENTER_FULL_PX) / (2.0 * self.CENTER_HALF_PX),
            )
        final_pose = (iou_score + area_score + center_score) / 3

        trace, pred_trace_details = self._tracked_shape_centers(video_frames)
        orbital_motion, orbital_details = self._orbital_motion_score(
            trace,
            reference_orbit,
        )

        total = final_pose * (0.4 + 0.6 * orbital_motion)
        self._last_task_details = {
            "final_iou": round(float(final_iou), 4),
            "area_ratio": round(float(area_ratio), 4),
            "center_dist_px": round(float(center_dist), 2),
            "iou_score": round(float(iou_score), 4),
            "area_score": round(float(area_score), 4),
            "center_score": round(float(center_score), 4),
            "final_pose": round(float(final_pose), 4),
            "trace_len": len(trace),
            "trace_source": pred_trace_details.get("trace_source"),
            "track_color": pred_trace_details.get("track_color"),
            "n_tracklets": pred_trace_details.get("n_tracklets"),
            "marker_pivot": (
                [round(float(marker_pivot[0]), 2), round(float(marker_pivot[1]), 2)]
                if marker_pivot is not None else None
            ),
            "orbit_source": reference_orbit["source"],
            "orbit_center": [
                round(float(reference_orbit["center"][0]), 2),
                round(float(reference_orbit["center"][1]), 2),
            ],
            "gt_orbit_fit_std": round(float(reference_orbit["fit_std"]), 4),
            "gt_trace_len": int(reference_orbit["gt_trace_len"]),
            "expected_radius": round(float(orbital_details["expected_radius"]), 2),
            "mean_radius": round(float(orbital_details["mean_radius"]), 2),
            "radius_delta": round(float(orbital_details["radius_delta"]), 4),
            "radius_match": round(float(orbital_details["radius_match"]), 4),
            "radius_std": round(float(orbital_details["radius_std"]), 4),
            "radius_jitter": round(float(orbital_details["radius_jitter"]), 4),
            "radius_stability": round(float(orbital_details["radius_stability"]), 4),
            "path_length": round(float(orbital_details["path_length"]), 2),
            "expected_arc_length": round(float(orbital_details["expected_arc_length"]), 2),
            "path_score": round(float(orbital_details["path_score"]), 4),
            "active_steps": round(float(orbital_details["active_steps"]), 2),
            "expected_active_steps": round(float(orbital_details["expected_active_steps"]), 2),
            "motion_coverage": round(float(orbital_details["motion_coverage"]), 4),
            "max_step": round(float(orbital_details["max_step"]), 2),
            "max_step_tolerance": round(float(orbital_details["max_step_tolerance"]), 2),
            "step_continuity": round(float(orbital_details["step_continuity"]), 4),
            "motion_distribution": round(float(orbital_details["motion_distribution"]), 4),
            "orbital_motion": round(float(orbital_motion), 4),
            "score": round(float(total), 4),
        }
        return total

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        if not pred_images or input_frame is None or gt_final_frame is None:
            self._last_task_details = {"error": "missing_frames"}
            return 0.0

        gt_first_frame = input_frame
        gt_start_shape = self._detect_main_shape(gt_first_frame)
        gt_final_shape = self._detect_main_shape(gt_final_frame)
        marker_pivot = self._detect_pivot(gt_first_frame)
        pred_final_shape = self._detect_main_shape(pred_images[-1])
        reference_orbit = (
            self._reference_orbit(
                gt_images if gt_images else [gt_first_frame, gt_final_frame],
                gt_start_shape["center"] if gt_start_shape is not None else (0.0, 0.0),
                gt_final_shape["center"] if gt_final_shape is not None else (0.0, 0.0),
                marker_pivot,
            )
            if gt_start_shape is not None and gt_final_shape is not None
            else None
        )
        if gt_start_shape is None or gt_final_shape is None or pred_final_shape is None:
            self._last_task_details = {"error": "detection_failed"}
            return 0.0

        # final_pose: identical to the video path
        final_iou = self._contour_iou(
            pred_final_shape["contour"], gt_final_shape["contour"], pred_images[-1].shape,
        )
        area_ratio = min(pred_final_shape["area"], gt_final_shape["area"]) / max(
            pred_final_shape["area"], gt_final_shape["area"], 1.0,
        )
        center_dist = safe_distance(pred_final_shape["center"], gt_final_shape["center"])
        iou_score = min(1.0, final_iou / self.FINAL_IOU_SATURATION)
        area_score = min(1.0, area_ratio / self.AREA_RATIO_SATURATION)
        if center_dist <= self.CENTER_FULL_PX:
            center_score = 1.0
        else:
            center_score = max(
                0.0, 1.0 - (center_dist - self.CENTER_FULL_PX) / (2.0 * self.CENTER_HALF_PX),
            )
        final_pose = (iou_score + area_score + center_score) / 3

        # Orbital motion over the keyframe sequence.
        pred_seq = [f for f in ([input_frame] + list(pred_images)) if f is not None]
        pred_centers, _pred_trace_details = self._tracked_shape_centers(pred_seq)

        if reference_orbit is None or len(pred_centers) < 2:
            orbital, orbital_details = 0.0, {}
        else:
            orbital, orbital_details = self._orbital_motion_score(
                pred_centers, reference_orbit,
            )

            gt_max_step = float(reference_orbit.get("max_step") or 0.0)
            pred_max_step = float(orbital_details.get("max_step") or 0.0)
            if gt_max_step > 1.0:
                tol = gt_max_step * self.INTERLEAVE_STEP_TOL_FRAC
                step_continuity = (1.0 if pred_max_step <= tol
                                   else tol / max(pred_max_step, 1.0))
                motion_distribution = (
                    float(orbital_details.get("motion_coverage", 0.0)) * step_continuity
                )
                orbital = (
                    float(orbital_details.get("radius_match", 0.0))
                    * float(orbital_details.get("radius_stability", 0.0))
                    * float(orbital_details.get("path_score", 0.0))
                    * motion_distribution
                )
                orbital_details = {
                    **orbital_details,
                    "step_continuity": step_continuity,
                    "motion_distribution": motion_distribution,
                    "step_tolerance": tol,
                }

        total = final_pose * (0.4 + 0.6 * orbital)
        self._last_task_details = {
            "mode": "interleave",
            "final_pose": round(float(final_pose), 4),
            "final_iou": round(float(final_iou), 4),
            "n_pred_centers": len(pred_centers),
            "expected_radius": (round(float(reference_orbit["radius"]), 2)
                                if reference_orbit else None),
            "radius_match": round(float(orbital_details.get("radius_match", 0.0)), 4),
            "radius_stability": round(float(orbital_details.get("radius_stability", 0.0)), 4),
            "pred_path_length": round(float(orbital_details.get("path_length", 0.0)), 2),
            "expected_arc_length": round(float(orbital_details.get("expected_arc_length", 0.0)), 2),
            "path_score": round(float(orbital_details.get("path_score", 0.0)), 4),
            "motion_coverage": round(float(orbital_details.get("motion_coverage", 0.0)), 4),
            "step_continuity": round(float(orbital_details.get("step_continuity", 0.0)), 4),
            "motion_distribution": round(float(orbital_details.get("motion_distribution", 0.0)), 4),
            "orbital_motion": round(float(orbital), 4),
            "score": round(float(total), 4),
        }
        return float(total)


class ShapeScalingAnalogyEvaluator(BaseEvaluator):
    """
    O-9: Shape scaling analogy evaluator.

    Dimensions:
        - completion (60%): focuses on the bottom-right quadrant of the final frame.
          Extracts the largest shape and evaluates its structural features:
          shape, size, color, position against the GT final frame.
        - foreground_preservation (25%): compare first vs generated final on
          foreground while excluding the changed shape region.
        - background_preservation (15%): compare first vs generated final on
          background region.
    """

    TASK_WEIGHTS = {
        "completion": 0.60,
        "foreground_preservation": 0.25,
        "background_preservation": 0.15,
    }

    SCALING_SHAPE_FEATURE_WEIGHTS = {
        "shape": 0.30,
        "size": 0.40,
        "color": 0.15,
        "position": 0.15,
    }

    def _pixel_similarity(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        mask: Optional[np.ndarray] = None,
        strictness: float = 2.0,
        min_cutoff: float = 0.3,
    ) -> float:
        """Return normalized pixel similarity in [0, 1] between two BGR frames."""
        a = frame_a.astype(np.float32)
        b = frame_b.astype(np.float32)

        if mask is not None and np.any(mask > 0):
            pixels_a = a[mask > 0]
            pixels_b = b[mask > 0]
        else:
            pixels_a = a.reshape(-1, 3)
            pixels_b = b.reshape(-1, 3)

        if pixels_a.size == 0 or pixels_b.size == 0:
            return 0.0

        pixel_distances = np.linalg.norm(pixels_a - pixels_b, axis=1)
        mean_dist = float(np.mean(pixel_distances))
        max_dist = float(np.sqrt(3.0 * (255.0 ** 2)))
        base_sim = max(0.0, 1.0 - (mean_dist / max_dist))
        final_sim = float(max(0.0, min(1.0, base_sim ** strictness)))
        return final_sim if final_sim >= min_cutoff else 0.0

    @staticmethod
    def _frame_masks(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (foreground_mask, background_mask) using near-white background."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_not(bg_mask)
        return fg_mask, bg_mask

    @staticmethod
    def _shape_change_mask(gt_first: np.ndarray, gt_last: np.ndarray) -> np.ndarray:
        """Return the GT first/final pixel difference mask."""
        diff = cv2.absdiff(gt_first, gt_last)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, change_mask = cv2.threshold(diff_gray, 18, 255, cv2.THRESH_BINARY)
        return change_mask

    def _extract_quadrant_shape_features(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Extracts features (contour, area, centroid, color) of the largest shape
        in the bottom-right quadrant of the frame.
        """
        h, w = frame.shape[:2]
        quadrant = frame[h // 2:, w // 2:]
        qh, qw = quadrant.shape[:2]
        if qh == 0 or qw == 0:
            return None

        gray = cv2.cvtColor(quadrant, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < 100:
            return None

        M = cv2.moments(largest_contour)
        if M["m00"] <= 0:
            return None
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        contour_mask = np.zeros((qh, qw), dtype=np.uint8)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=-1)

        mean_bgr = np.array(cv2.mean(quadrant, mask=contour_mask)[:3], dtype=np.float32)
        perimeter = float(cv2.arcLength(largest_contour, True))
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        approx = cv2.approxPolyDP(largest_contour, 0.015 * perimeter if perimeter > 0 else 0.0, True)
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter + 1e-6))

        return {
            "contour": largest_contour,
            "area": float(area),
            "area_ratio": float(area / float(qh * qw)),
            "centroid": (float(cx / qw), float(cy / qh)),
            "mean_bgr": mean_bgr,
            "bbox_aspect_ratio": float(bw / max(bh, 1)),
            "bbox_extent": float(area / max(float(bw * bh), 1.0)),
            "vertex_count": int(len(approx)),
            "circularity": max(0.0, min(1.0, circularity)),
        }

    def _compute_scaling_completion_score(
        self,
        gt_shape_features: Optional[Dict],
        pred_shape_features: Optional[Dict],
    ) -> Tuple[float, Dict[str, float]]:
        """Compute completion and sub-scores from largest-shape feature comparison for scaling tasks."""
        if gt_shape_features is None or pred_shape_features is None:
            return 0.0, {
                "shape": 0.0,
                "size": 0.0,
                "color": 0.0,
                "position": 0.0,
                "shape_contour": 0.0,
                "shape_vertex": 0.0,
            }

        match_score = cv2.matchShapes(
            denoise_contour(gt_shape_features["contour"]),
            denoise_contour(pred_shape_features["contour"]),
            cv2.CONTOURS_MATCH_I1,
            0.0,
        )
        shape_score_from_contour = float(np.exp(-4.0 * max(0.0, match_score)))

        gt_vertex_count = gt_shape_features["vertex_count"]
        pred_vertex_count = pred_shape_features["vertex_count"]
        if gt_vertex_count == pred_vertex_count:
            vertex_score = 1.0
        elif abs(gt_vertex_count - pred_vertex_count) <= 1:
            vertex_score = 0.3
        else:
            vertex_score = 0.0

        shape_score = float(0.5 * shape_score_from_contour + 0.5 * vertex_score)

        shape_gate = min(1.0, shape_score / 0.6)
        if shape_gate <= 0.0:
            return 0.0, {
                "shape": shape_score,
                "size": 0.0,
                "color": 0.0,
                "position": 0.0,
                "shape_contour": shape_score_from_contour,
                "shape_vertex": vertex_score,
            }

        # 2. size similarity
        area_ratio = min(gt_shape_features["area"], pred_shape_features["area"]) / max(
            gt_shape_features["area"], pred_shape_features["area"], 1e-6
        )
        extent_ratio = min(gt_shape_features["bbox_extent"], pred_shape_features["bbox_extent"]) / max(
            gt_shape_features["bbox_extent"], pred_shape_features["bbox_extent"], 1e-6
        )
        size_ratio = float(0.80 * area_ratio + 0.20 * extent_ratio)

        size_score = size_ratio if size_ratio >= 0.75 else 0.0

        # 3. color similarity
        color_dist = float(np.linalg.norm(gt_shape_features["mean_bgr"] - pred_shape_features["mean_bgr"]))
        color_score = float(max(0.0, 1.0 - color_dist / np.sqrt(3.0 * (255.0 ** 2))))

        # 4. position similarity
        gt_cx, gt_cy = gt_shape_features["centroid"]
        pred_cx, pred_cy = pred_shape_features["centroid"]
        position_dist = float(np.sqrt((gt_cx - pred_cx) ** 2 + (gt_cy - pred_cy) ** 2))
        position_score = float(max(0.0, 1.0 - position_dist / np.sqrt(2.0)))

        completion = shape_gate * (
            self.SCALING_SHAPE_FEATURE_WEIGHTS["shape"] * shape_score
            + self.SCALING_SHAPE_FEATURE_WEIGHTS["size"] * size_score
            + self.SCALING_SHAPE_FEATURE_WEIGHTS["color"] * color_score
            + self.SCALING_SHAPE_FEATURE_WEIGHTS["position"] * position_score
        )
        # for scaling task, accuracy penalty is applied to size ratio.
        completion = float(max(0.0, min(1.0, completion))) * size_ratio
        completion_details = {
            "shape": shape_score,
            "size": size_score,
            "color": color_score,
            "position": position_score,
            "shape_contour": shape_score_from_contour,
            "shape_vertex": vertex_score,
        }
        return completion, completion_details


    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate O-9 with final-frame completion and preservation metrics."""
        scores: Dict[str, float] = {}

        if len(video_frames) < 2 or gt_first_frame is None or gt_final_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        if last_frame.shape[:2] != gt_final_frame.shape[:2]:
            first_frame = normalize_frame_size(first_frame, gt_final_frame)
            last_frame = normalize_frame_size(last_frame, gt_final_frame)
        gt_first, gt_last = gt_first_frame, gt_final_frame

        # 1) completion (60%): compare largest shape in the bottom-right quadrant (2x2 split).
        gt_shape_features = self._extract_quadrant_shape_features(gt_last)
        pred_shape_features = self._extract_quadrant_shape_features(last_frame)
        completion_score, completion_details = self._compute_scaling_completion_score(gt_shape_features, pred_shape_features)
        scores["completion"] = completion_score

        # 2) foreground_preservation (25%): compare first vs final on non-background and non-changed region.
        change_mask = self._shape_change_mask(gt_first, gt_last)
        first_fg, first_bg = self._frame_masks(first_frame)
        fg_compare_mask = cv2.bitwise_and(first_fg, cv2.bitwise_not(change_mask))
        scores["foreground_preservation"] = self._pixel_similarity(first_frame, last_frame, fg_compare_mask)

        # 3) background_preservation (15%): compare first vs final on stable background region.
        bg_compare_mask = cv2.bitwise_and(first_bg, cv2.bitwise_not(change_mask))
        scores["background_preservation"] = self._pixel_similarity(first_frame, last_frame, bg_compare_mask, strictness=3.0, min_cutoff=0.6)

        self._last_task_details = {
            **scores,
            "completion_details": completion_details,
        }

        total = sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
        total *= min(1.0, scores["completion"] / 0.5) * min(1.0, scores["foreground_preservation"] / 0.5)
        return float(total)


class ShapeColorThenMoveEvaluator(BaseEvaluator):
    """
    O-11: Shape color then move evaluator.

    Dimensions:
        - completion (60%): split frame into 3 columns, use GT targets as anchors,
          and match generated targets by nearest centroid.
          - first shape (col 2): color task.
          - second shape (col 3): move task.
        - foreground_preservation (25%): compare first vs generated final on foreground.
        - background_preservation (15%): compare first vs generated final on background.
    """

    TASK_WEIGHTS = {
        "completion": 0.60,
        "foreground_preservation": 0.25,
        "background_preservation": 0.15,
    }

    # first shape: color task
    FIRST_SHAPE_WEIGHTS = {
        "shape": 0.25,
        "size": 0.20,
        "color": 0.40,
        "position": 0.15,
    }

    # second shape: move task
    SECOND_SHAPE_WEIGHTS = {
        "shape": 0.25,
        "size": 0.20,
        "color": 0.15,
        "position": 0.40,
    }

    def _pixel_similarity(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        mask: Optional[np.ndarray] = None,
        strictness: float = 2.0,
        min_cutoff: float = 0.3,
    ) -> float:
        """Return normalized pixel similarity in [0, 1] between two BGR frames."""
        a = frame_a.astype(np.float32)
        b = frame_b.astype(np.float32)

        if mask is not None and np.any(mask > 0):
            pixels_a = a[mask > 0]
            pixels_b = b[mask > 0]
        else:
            pixels_a = a.reshape(-1, 3)
            pixels_b = b.reshape(-1, 3)

        if pixels_a.size == 0 or pixels_b.size == 0:
            return 0.0

        pixel_distances = np.linalg.norm(pixels_a - pixels_b, axis=1)
        mean_dist = float(np.mean(pixel_distances))
        max_dist = float(np.sqrt(3.0 * (255.0 ** 2)))
        base_sim = max(0.0, 1.0 - (mean_dist / max_dist))
        final_sim = float(max(0.0, min(1.0, base_sim ** strictness)))
        return final_sim if final_sim >= min_cutoff else 0.0

    @staticmethod
    def _frame_masks(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (foreground_mask, background_mask) using near-white background."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_not(bg_mask)
        return fg_mask, bg_mask

    @staticmethod
    def _shape_change_mask(gt_first: np.ndarray, gt_last: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(gt_first, gt_last)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, change_mask = cv2.threshold(diff_gray, 18, 255, cv2.THRESH_BINARY)
        return change_mask

    def _extract_cell_shape_features(self, frame: np.ndarray, row_idx: int, col_idx: int) -> Optional[Dict]:
        """Extract largest shape features from one 2x3 cell in final frame."""
        h, w = frame.shape[:2]
        y0 = int(round(row_idx * h / 2.0))
        y1 = int(round((row_idx + 1) * h / 2.0))
        x0 = int(round(col_idx * w / 3.0))
        x1 = int(round((col_idx + 1) * w / 3.0))

        cell = frame[y0:y1, x0:x1]
        ch, cw = cell.shape[:2]
        if ch == 0 or cw == 0:
            return None

        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < 100:
            return None

        M = cv2.moments(largest_contour)
        if M["m00"] <= 0:
            return None
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        contour_mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=-1)

        mean_bgr = np.array(cv2.mean(cell, mask=contour_mask)[:3], dtype=np.float32)
        perimeter = float(cv2.arcLength(largest_contour, True))
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        approx = cv2.approxPolyDP(largest_contour, 0.015 * perimeter if perimeter > 0 else 0.0, True)

        return {
            "contour": largest_contour,
            "area": float(area),
            "centroid": (float(cx / cw), float(cy / ch)),
            "mean_bgr": mean_bgr,
            "bbox_extent": float(area / max(float(bw * bh), 1.0)),
            "vertex_count": int(len(approx)),
        }

    def _extract_target_shape_features(
        self,
        frame: np.ndarray,
        col_idx: int,
        total_cols: int = 3,
        reference_centroid: Optional[Tuple[float, float]] = None
    ) -> Optional[Dict]:
        """
        Extract target shape features for one column.
        When reference_centroid exists, choose contour closest to reference.
        Otherwise choose the bottom-most valid contour in this column.
        """
        h, w = frame.shape[:2]
        col_w = w // total_cols
        start_x = col_idx * col_w
        end_x = (col_idx + 1) * col_w if col_idx < total_cols - 1 else w

        roi = frame[:, start_x:end_x]
        rh, rw = roi.shape[:2]
        if rh == 0 or rw == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        if not valid_contours:
            return None

        contour_info: List[Dict[str, float | np.ndarray]] = []
        for cnt in valid_contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx = float(x + bw / 2.0)
                cy = float(y + bh / 2.0)

            contour_info.append({
                "contour": cnt,
                "local_cy": cy,
                "global_cx": cx + start_x,
                "global_cy": cy,
            })

        if reference_centroid is not None:
            ref_x = reference_centroid[0] * w
            ref_y = reference_centroid[1] * h

            def _dist_sq(item: Dict[str, float | np.ndarray]) -> float:
                dx = float(item["global_cx"]) - ref_x
                dy = float(item["global_cy"]) - ref_y
                return dx * dx + dy * dy

            target_info = min(contour_info, key=_dist_sq)
        else:
            target_info = max(contour_info, key=lambda item: float(item["local_cy"]))

        target_contour = target_info["contour"]
        global_cx = float(target_info["global_cx"])
        global_cy = float(target_info["global_cy"])

        area = float(cv2.contourArea(target_contour))
        contour_mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.drawContours(contour_mask, [target_contour], -1, 255, thickness=-1)

        mean_bgr = np.array(cv2.mean(roi, mask=contour_mask)[:3], dtype=np.float32)
        perimeter = float(cv2.arcLength(target_contour, True))
        x, y, bw, bh = cv2.boundingRect(target_contour)
        approx = cv2.approxPolyDP(target_contour, 0.015 * perimeter if perimeter > 0 else 0.0, True)

        return {
            "contour": target_contour,
            "area": area,
            "centroid": (float(global_cx / w), float(global_cy / h)),
            "mean_bgr": mean_bgr,
            "bbox_extent": float(area / max(float(bw * bh), 1.0)),
            "vertex_count": int(len(approx)),
        }

    def _compute_shape_score(
        self,
        gt_shape_features: Optional[Dict],
        pred_shape_features: Optional[Dict],
        weights: Dict[str, float],
        color_task: bool = False,
        position_task: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute shape score and sub-scores from shape feature comparison."""
        if gt_shape_features is None or pred_shape_features is None:
            return 0.0, {
                "shape": 0.0,
                "size": 0.0,
                "color": 0.0,
                "position": 0.0,
                "shape_contour": 0.0,
                "shape_vertex": 0.0,
            }

        # 1.1 shape contour similarity
        match_score = cv2.matchShapes(
            denoise_contour(gt_shape_features["contour"]),
            denoise_contour(pred_shape_features["contour"]),
            cv2.CONTOURS_MATCH_I1,
            0.0,
        )
        shape_score_from_contour = float(np.exp(-4.0 * max(0.0, match_score)))

        # 1.2 shape vertex count similarity
        gt_vertex_count = gt_shape_features["vertex_count"]
        pred_vertex_count = pred_shape_features["vertex_count"]
        if gt_vertex_count == pred_vertex_count:
            vertex_score = 1.0
        elif abs(gt_vertex_count - pred_vertex_count) <= 1:
            vertex_score = 0.3
        else:
            vertex_score = 0.0
        
        shape_score = float(0.5 * shape_score_from_contour + 0.5 * vertex_score)

        shape_gate = min(1.0, shape_score / 0.6)
        if shape_gate <= 0.0:
            return 0.0, {
                "shape": shape_score,
                "size": 0.0,
                "color": 0.0,
                "position": 0.0,
                "shape_contour": shape_score_from_contour,
                "shape_vertex": vertex_score,
            }

        # 2. size similarity
        area_ratio = min(gt_shape_features["area"], pred_shape_features["area"]) / max(
            gt_shape_features["area"], pred_shape_features["area"], 1e-6
        )
        extent_ratio = min(gt_shape_features["bbox_extent"], pred_shape_features["bbox_extent"]) / max(
            gt_shape_features["bbox_extent"], pred_shape_features["bbox_extent"], 1e-6
        )
        size_score = float(0.80 * area_ratio + 0.20 * extent_ratio)

        # 3. color similarity
        color_dist = float(np.linalg.norm(gt_shape_features["mean_bgr"] - pred_shape_features["mean_bgr"]))
        color_ratio = float(max(0.0, 1.0 - color_dist / np.sqrt(3.0 * (255.0 ** 2))))

        if color_task:
            color_score = color_ratio if color_ratio >= 0.75 else 0.0
        else:
            color_score = color_ratio if color_ratio >= 0.75 else 0.0

        # 4. position similarity
        gt_cx, gt_cy = gt_shape_features["centroid"]
        pred_cx, pred_cy = pred_shape_features["centroid"]
        position_dist = float(abs(gt_cy - pred_cy)) * 10.0
        if position_dist > 1:
            position_score = 0.0
        else:
            position_score = 1.0 - position_dist / 1.0

        if position_task and position_score < 0.5:
            return 0.0, {
                "shape": shape_score,
                "size": size_score,
                "color": color_score,
                "position": position_score,
                "shape_contour": shape_score_from_contour,
                "shape_vertex": vertex_score,
            }

        total_score = shape_gate * (
            weights["shape"] * shape_score
            + weights["size"] * size_score
            + weights["color"] * color_score
            + weights["position"] * position_score
        )

        # for color task, accuracy penalty is applied to color ratio.
        if color_task:
            total_score = total_score * color_ratio

        return float(max(0.0, min(1.0, total_score))), {
            "shape": shape_score,
            "size": size_score,
            "color": color_score,
            "position": position_score,
            "shape_contour": shape_score_from_contour,
            "shape_vertex": vertex_score,
        }

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate O-11 with final-frame completion and preservation metrics."""
        scores: Dict[str, float] = {}

        if len(video_frames) < 2 or gt_first_frame is None or gt_final_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        if last_frame.shape[:2] != gt_final_frame.shape[:2]:
            first_frame = normalize_frame_size(first_frame, gt_final_frame)
            last_frame = normalize_frame_size(last_frame, gt_final_frame)
        gt_first, gt_last = gt_first_frame, gt_final_frame

        # 1) completion (60%): first shape from fixed 2x3 cell, second shape keeps anchor-based matching.
        gt_first_shape_features = self._extract_cell_shape_features(gt_last, row_idx=1, col_idx=1)
        gt_second_shape_features = self._extract_target_shape_features(gt_last, col_idx=2)

        pred_first_shape_features = self._extract_cell_shape_features(last_frame, row_idx=1, col_idx=1)
        pred_second_shape_features = self._extract_target_shape_features(
            last_frame,
            col_idx=2,
            reference_centroid=gt_second_shape_features["centroid"] if gt_second_shape_features else None,
        )

        first_shape_score, first_shape_details = self._compute_shape_score(
            gt_first_shape_features,
            pred_first_shape_features,
            weights=self.FIRST_SHAPE_WEIGHTS,
            color_task=True,
            position_task=False,
        )
        second_shape_score, second_shape_details = self._compute_shape_score(
            gt_second_shape_features,
            pred_second_shape_features,
            weights=self.SECOND_SHAPE_WEIGHTS,
            color_task=False,
            position_task=True,
        )
        scores["completion"] = 0.5 * first_shape_score + 0.5 * second_shape_score

        # 2) foreground_preservation (25%): compare first vs final on non-background and non-changed region.
        change_mask = self._shape_change_mask(gt_first, gt_last)
        first_fg, first_bg = self._frame_masks(first_frame)
        fg_compare_mask = cv2.bitwise_and(first_fg, cv2.bitwise_not(change_mask))
        scores["foreground_preservation"] = self._pixel_similarity(first_frame, last_frame, fg_compare_mask)

        # 3) background_preservation (15%): compare first vs final on stable background region.
        bg_compare_mask = cv2.bitwise_and(first_bg, cv2.bitwise_not(change_mask))
        scores["background_preservation"] = self._pixel_similarity(first_frame, last_frame, bg_compare_mask, strictness=3.0, min_cutoff=0.6)

        self._last_task_details = {
            **scores,
            "completion_details": {
                "first_shape": first_shape_details,
                "second_shape": second_shape_details,
            },
        }
        return float(sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS))
    
class ConstructionStackEvaluator(BaseEvaluator):
    """
    O-22: Construction stack (block stacking) evaluator.

    Pipeline:
    1. Per-frame block detection over the whole frame; classify each block by
       x against the semantic midline (W/2) into LEFT (current) vs RIGHT
       (target) side. Tall contours are split into stacked blocks by an
       expected block-height ratio and each band is hue-sampled individually
       to recover per-block colors even when stacked blocks merge into one
       contour.
    2. With solution metadata, process scoring follows persistent completed
       stack states. A state is eligible only when all blocks are grounded and
       it persists for at least three frames; this excludes in-flight states
       and one/two-frame false landings while a block crosses another slot.
       Older samples without metadata retain the legacy lift→flight→place
       movement-validity fallback.
    3. Main score = sequence + count match between gen-final LEFT stack
       (current side) and the target reference. Target reference is GT
       final-frame right side when available, else gen final-frame right
       side (NOT first-frame right, since the right column is sometimes
       animated and only reaches the true target in the final frame).
    4. Final score = main_score * (0.4 + 0.6 * process_score)
       - 0.05 if right stack drifted across frames (vs final right)
       - 0.1 if background SSIM (with stacks masked) is below threshold.
    """

    # Detector
    EXPECTED_BLOCK_H_RATIO = 0.083   # ~40 px in 480-tall frame
    DETECT_SAT_THRESH = 60
    DETECT_VAL_THRESH = 60
    DETECT_MIN_AREA = 400
    DETECT_MIN_W = 12
    DETECT_MIN_H = 12

    # Detector baseline resolution the absolute pixel cutoffs were tuned on.
    DETECT_BASE_W = 640
    DETECT_BASE_H = 480

    # Stack clustering
    STACK_X_CLUSTER_FRAC = 0.7        # block widths tolerance for grouping
    FLOOR_Y_FRAC = 0.72               # bottom block center must be >= this fraction of H
    INTERLEAVE_FLOOR_Y_FRAC = 0.72

    # Matching
    HUE_MATCH_THRESH = 28

    # State change smoothing
    STATE_PERSIST_FRAMES = 2
    VIDEO_STABLE_STATE_FRAMES = 3

    # Deductions
    TARGET_ALL_FRAMES_MIN = 0.88
    BG_SSIM_THRESHOLD = 0.72

    # Canonical color names
    @staticmethod
    def _canonical_color(hue: int) -> str:
        h = int(hue) % 180
        if h < 8 or h >= 170: return 'R'
        if h < 18: return 'O'
        if h < 35: return 'Y'
        if h < 80: return 'G'
        if h < 100: return 'C'
        if h < 135: return 'B'
        return 'M'

    def _load_solution_metadata(self, eval_info: Dict) -> Optional[Dict]:
        """Load O-22's authoritative initial state and move sequence."""
        import json as _json
        import os as _os

        meta_path = eval_info.get('metafile_path')
        if isinstance(meta_path, (list, tuple)):
            meta_path = next(
                (path for path in meta_path if path and _os.path.exists(path)),
                None,
            )
        if not (meta_path and _os.path.exists(meta_path)):
            meta_path = _os.path.join(eval_info.get('gt_path', ''), 'metadata.json')
        if not _os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, encoding='utf-8') as handle:
                params = (_json.load(handle).get('parameters') or {})
        except (OSError, ValueError, TypeError):
            return None

        initial_state = params.get('initial_state')
        solution = params.get('solution')
        palette = params.get('block_palette_rgb')
        layout = params.get('layout') or {}
        if not initial_state or not solution or not palette:
            return None

        try:
            color_codes: Dict[int, str] = {}
            for block_id, rgb in enumerate(palette):
                rgb_pixel = np.asarray([[rgb]], dtype=np.uint8)
                hue = int(cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0, 0, 0])
                color_codes[block_id] = self._canonical_color(hue)

            state_ids = [list(map(int, stack)) for stack in initial_state]
            expected_states: List[Tuple[Tuple[str, ...], ...]] = []
            normalized_solution: List[Tuple[int, int]] = []
            for move in solution:
                source_idx, dest_idx = int(move[0]), int(move[1])
                if (
                    source_idx < 0 or source_idx >= len(state_ids)
                    or dest_idx < 0 or dest_idx >= len(state_ids)
                    or not state_ids[source_idx]
                ):
                    return None
                block_id = state_ids[source_idx].pop()
                state_ids[dest_idx].append(block_id)
                normalized_solution.append((source_idx, dest_idx))
                expected_states.append(tuple(
                    tuple(color_codes[bid] for bid in stack)
                    for stack in state_ids
                ))

            initial_codes = tuple(
                tuple(color_codes[int(bid)] for bid in stack)
                for stack in initial_state
            )
        except (KeyError, TypeError, ValueError, IndexError):
            return None

        return {
            'initial_state': initial_codes,
            'expected_states': expected_states,
            'solution': normalized_solution,
            'optimal_moves': len(normalized_solution),
            'num_stacks': int(layout.get('num_stacks', len(initial_codes))),
            'layout_width': float(layout.get('width', 1024.0)),
            'section_width': float(layout.get('section_width', 512.0)),
            'metafile_path': meta_path,
        }

    @staticmethod
    def _slot_centers(solution_meta: Dict, frame_width: int) -> List[float]:
        num_stacks = max(1, int(solution_meta['num_stacks']))
        scale = float(frame_width) / max(float(solution_meta['layout_width']), 1.0)
        section_width = float(solution_meta['section_width']) * scale
        return [
            section_width * (idx + 1) / (num_stacks + 1)
            for idx in range(num_stacks)
        ]

    def _slotted_left_state(
        self,
        frame_state: Dict,
        solution_meta: Dict,
        frame_width: int,
    ) -> Tuple[Tuple[str, ...], ...]:
        """Restore empty stack slots omitted by ``_frame_state``."""
        centers = self._slot_centers(solution_meta, frame_width)
        slots: List[Tuple[str, ...]] = [tuple() for _ in centers]
        occupied_slots: Set[int] = set()
        for stack, x_center in zip(
            frame_state.get('L_grounded', []),
            frame_state.get('L_x', []),
        ):
            slot_idx = min(
                range(len(centers)),
                key=lambda idx: abs(float(x_center) - centers[idx]),
            )
            if slot_idx in occupied_slots:
                # Two detected stacks in one semantic slot cannot be a legal state.
                slots[slot_idx] = tuple(list(slots[slot_idx]) + ['?'])
            else:
                slots[slot_idx] = tuple(stack)
                occupied_slots.add(slot_idx)
        return tuple(slots)

    def _score_interleave_solution(
        self,
        per_frame_state: Sequence[Dict],
        solution_meta: Dict,
        frame_width: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """Score the ordered, one-completed-move-per-image state sequence."""
        expected_states = solution_meta['expected_states']
        current_state = solution_meta['initial_state']
        matched_moves = 0
        invalid_changes = 0
        duplicate_states = 0
        observed_states: List[Tuple[Tuple[str, ...], ...]] = []

        for frame_state in per_frame_state[1:]:
            observed = self._slotted_left_state(
                frame_state, solution_meta, frame_width,
            )
            observed_states.append(observed)
            if observed == current_state:
                duplicate_states += 1
                continue
            if (
                matched_moves < len(expected_states)
                and observed == expected_states[matched_moves]
            ):
                current_state = observed
                matched_moves += 1
            else:
                invalid_changes += 1

        optimal_moves = max(int(solution_meta['optimal_moves']), 1)
        coverage = matched_moves / optimal_moves
        validity = (
            matched_moves / (matched_moves + invalid_changes)
            if matched_moves + invalid_changes > 0 else 0.0
        )
        process_score = coverage * validity
        return float(process_score), {
            'matched_solution_moves': matched_moves,
            'optimal_moves': optimal_moves,
            'coverage': round(float(coverage), 4),
            'validity': round(float(validity), 4),
            'invalid_state_changes': invalid_changes,
            'duplicate_states': duplicate_states,
            'expected_states': [
                [list(stack) for stack in state] for state in expected_states
            ],
            'observed_states': [
                [list(stack) for stack in state] for state in observed_states
            ],
        }

    def _score_video_solution(
        self,
        movements: Sequence[Dict],
        solution_meta: Dict,
        frame_width: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compare completed dense-video moves with metadata's move order."""
        expected_moves = solution_meta['solution']
        centers = self._slot_centers(solution_meta, frame_width)
        matched_moves = 0
        invalid_moves = 0
        detected_moves: List[Optional[Tuple[int, int]]] = []

        for movement in movements:
            detected: Optional[Tuple[int, int]] = None
            if (
                movement.get('valid')
                and movement.get('kind') == 'transit'
                and movement.get('source_side') == 'L'
                and movement.get('dest_side') == 'L'
                and movement.get('source_x') is not None
                and movement.get('dest_x') is not None
            ):
                source_idx = min(
                    range(len(centers)),
                    key=lambda idx: abs(float(movement['source_x']) - centers[idx]),
                )
                dest_idx = min(
                    range(len(centers)),
                    key=lambda idx: abs(float(movement['dest_x']) - centers[idx]),
                )
                detected = (source_idx, dest_idx)
            detected_moves.append(detected)

            if (
                detected is not None
                and matched_moves < len(expected_moves)
                and detected == expected_moves[matched_moves]
            ):
                matched_moves += 1
            else:
                invalid_moves += 1

        optimal_moves = max(int(solution_meta['optimal_moves']), 1)
        coverage = matched_moves / optimal_moves
        validity = (
            matched_moves / (matched_moves + invalid_moves)
            if matched_moves + invalid_moves > 0 else 0.0
        )
        process_score = coverage * validity
        return float(process_score), {
            'matched_solution_moves': matched_moves,
            'optimal_moves': optimal_moves,
            'coverage': round(float(coverage), 4),
            'validity': round(float(validity), 4),
            'invalid_moves': invalid_moves,
            'expected_moves': [list(move) for move in expected_moves],
            'detected_moves': [
                list(move) if move is not None else None for move in detected_moves
            ],
        }

    def _score_video_state_solution(
        self,
        per_frame_state: Sequence[Dict],
        solution_meta: Dict,
        frame_width: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """Match persistent completed stack states to the metadata solution.

        A moving block can pass close enough to an intermediate slot to look
        grounded for one or two frames.  Treating every such observation as a
        completed source->destination movement splits one physical move into
        several false moves.  Completed states instead have all blocks present
        and persist; transient in-flight states do not.
        """
        observed = [
            self._slotted_left_state(state, solution_meta, frame_width)
            for state in per_frame_state
        ]
        runs: List[Dict[str, Any]] = []
        for frame_idx, state in enumerate(observed):
            if runs and runs[-1]['state'] == state:
                runs[-1]['end_frame'] = frame_idx
                runs[-1]['duration'] += 1
            else:
                runs.append({
                    'state': state,
                    'start_frame': frame_idx,
                    'end_frame': frame_idx,
                    'duration': 1,
                })

        initial_state = solution_meta['initial_state']
        expected_states = solution_meta['expected_states']
        expected_blocks = sum(len(stack) for stack in initial_state)
        stable_runs = [
            run for run in runs
            if (
                run['duration'] >= self.VIDEO_STABLE_STATE_FRAMES
                and sum(len(stack) for stack in run['state']) == expected_blocks
            )
        ]

        current_state = initial_state
        matched_moves = 0
        invalid_states = 0
        duplicate_states = 0
        accepted_states = []
        rejected_states = []
        for run in stable_runs:
            state = run['state']
            if state == current_state:
                duplicate_states += 1
                continue
            if (
                matched_moves < len(expected_states)
                and state == expected_states[matched_moves]
            ):
                current_state = state
                matched_moves += 1
                accepted_states.append({
                    'move': matched_moves,
                    'frame_range': [run['start_frame'], run['end_frame']],
                    'state': [list(stack) for stack in state],
                })
            else:
                invalid_states += 1
                rejected_states.append({
                    'frame_range': [run['start_frame'], run['end_frame']],
                    'state': [list(stack) for stack in state],
                })

        optimal_moves = max(int(solution_meta['optimal_moves']), 1)
        coverage = matched_moves / optimal_moves
        validity = (
            matched_moves / (matched_moves + invalid_states)
            if matched_moves + invalid_states > 0 else 0.0
        )
        process_score = coverage * validity
        return float(process_score), {
            'mode': 'persistent_complete_states',
            'matched_solution_moves': matched_moves,
            'optimal_moves': optimal_moves,
            'coverage': round(float(coverage), 4),
            'validity': round(float(validity), 4),
            'invalid_stable_states': invalid_states,
            'duplicate_stable_states': duplicate_states,
            'stable_state_frames': self.VIDEO_STABLE_STATE_FRAMES,
            'accepted_states': accepted_states,
            'rejected_states': rejected_states,
        }

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0

        scores: Dict = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        H, W = first_frame.shape[:2]

        # Semantic midline divides LEFT (current) from RIGHT (target)
        side_split = W // 2

        # Per-frame state: grounded stacks on each side
        per_frame_state: List[Dict] = []
        for f in video_frames:
            per_frame_state.append(self._frame_state(f, side_split))

        # Final and reference stacks (lists of stacks, left-to-right)
        final_state = per_frame_state[-1]
        final_current_stacks = final_state['L_grounded']     # list of bottom→top tuples
        final_target_gen_stacks = final_state['R_grounded']

        # Target reference: GT final right (preferred) else gen final right.
        gt_target_stacks: List[Tuple[str, ...]] = []
        if gt_final_frame is not None and gt_final_frame.shape[:2] == (H, W):
            gt_state = self._frame_state(gt_final_frame, side_split)
            gt_target_stacks = gt_state['R_grounded']
        target_ref_stacks = gt_target_stacks if gt_target_stacks else final_target_gen_stacks

        scores['side_split'] = side_split
        scores['target_stack_count'] = len(target_ref_stacks)
        scores['target_stacks_left_to_right'] = [list(s) for s in target_ref_stacks]
        scores['target_total_blocks'] = sum(len(s) for s in target_ref_stacks)
        scores['final_result_stack_count'] = len(final_current_stacks)
        scores['final_result_stacks_left_to_right'] = [list(s) for s in final_current_stacks]
        scores['final_result_total_blocks'] = sum(len(s) for s in final_current_stacks)
        scores['gen_final_right_stacks'] = [list(s) for s in final_target_gen_stacks]

        # 1) Main score: gen final-frame LEFT stacks vs target reference stacks
        main_score = self._match_stacks(target_ref_stacks, final_current_stacks)
        scores['main_score'] = main_score

        # 2) Movement validity multiplier
        movements = self._detect_movements(per_frame_state)
        n_total = len(movements)
        n_valid = sum(1 for m in movements if m['valid'])
        solution_meta = self._load_solution_metadata(eval_info)
        if solution_meta is not None:
            mv_ratio, solution_details = self._score_video_state_solution(
                per_frame_state, solution_meta, W,
            )
        else:
            mv_ratio = (n_valid / n_total) if n_total > 0 else 0.0
            solution_details = {
                'mode': 'legacy_valid_over_detected',
                'metadata_available': False,
            }
        scores['movement_validity_ratio'] = mv_ratio
        scores['solution_process'] = solution_details
        scores['movements_total'] = n_total
        scores['movements_valid'] = n_valid
        scores['movements_detail'] = movements

        # Completion and process are separate dimensions.
        score = main_score * (0.4 + 0.6 * mv_ratio)

        # 3) Target preserved across all frames (-0.05): right side should
        #    end at the final right state; check stability across frames.
        min_tp, mean_tp = self._target_preservation_all_frames_v2(
            per_frame_state, final_target_gen_stacks
        )
        scores['target_preservation_min_frames'] = min_tp
        scores['target_preservation_mean_frames'] = mean_tp
        deduct_target = 0.05 if min_tp < self.TARGET_ALL_FRAMES_MIN else 0.0
        scores['deduction_target_frames'] = deduct_target

        # 4) Background consistency with stacks masked
        bg_score, bg_detail = self._background_consistency_masked_stacks(
            video_frames, gt_frames, gt_first_frame, gt_final_frame,
            side_split, H, W,
        )
        scores['background_consistency'] = bg_score
        scores['background_detail'] = bg_detail
        deduct_bg = 0.1 if bg_score < self.BG_SSIM_THRESHOLD else 0.0
        scores['deduction_background'] = deduct_bg

        score = score - deduct_target - deduct_bg
        score = max(0.0, min(1.0, score))
        scores['final_score'] = score
        scores['score_formula'] = 'main_score * (0.4 + 0.6 * process_score) - deductions'
        self._last_task_details = scores
        return score

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Interleave: validate one completed metadata move per output image."""
        if not pred_images or input_frame is None:
            self._last_task_details = {"error": "no_input_or_pred"}
            return 0.0
        video_frames = [input_frame] + pred_images
        if len(video_frames) < 2:
            self._last_task_details = {"error": "too_few_frames"}
            return 0.0
        H, W = input_frame.shape[:2]
        side_split = W // 2
        gt_final = gt_images[-1] if gt_images else gt_final_frame

        saved_floor = self.FLOOR_Y_FRAC
        self.FLOOR_Y_FRAC = self.INTERLEAVE_FLOOR_Y_FRAC
        try:
            per_frame_state = [self._frame_state(f, side_split) for f in video_frames]
            final_state = per_frame_state[-1]
            final_current_stacks = final_state['L_grounded']
            final_target_gen_stacks = final_state['R_grounded']

            gt_target_stacks: List[Tuple[str, ...]] = []
            if gt_final is not None and gt_final.shape[:2] == (H, W):
                gt_target_stacks = self._frame_state(gt_final, side_split)['R_grounded']
            target_ref_stacks = gt_target_stacks if gt_target_stacks else final_target_gen_stacks

            main_score = self._match_stacks(target_ref_stacks, final_current_stacks)

            min_tp, mean_tp = self._target_preservation_all_frames_v2(
                per_frame_state, final_target_gen_stacks,
            )
            deduct_target = 0.05 if min_tp < self.TARGET_ALL_FRAMES_MIN else 0.0
            bg_score, bg_detail = self._background_consistency_masked_stacks(
                video_frames, gt_images, input_frame, gt_final, side_split, H, W,
            )
            deduct_bg = 0.1 if bg_score < self.BG_SSIM_THRESHOLD else 0.0

            solution_meta = self._load_solution_metadata(eval_info)
            if solution_meta is not None:
                process_score, process_details = self._score_interleave_solution(
                    per_frame_state, solution_meta, W,
                )
            else:
                process_score = 1.0
                process_details = {
                    'mode': 'legacy_final_only',
                    'metadata_available': False,
                }
        finally:
            self.FLOOR_Y_FRAC = saved_floor

        score = max(0.0, min(
            1.0,
            main_score * (0.4 + 0.6 * process_score) - deduct_target - deduct_bg,
        ))
        self._last_task_details = {
            'side_split': side_split,
            'target_stack_count': len(target_ref_stacks),
            'target_stacks_left_to_right': [list(s) for s in target_ref_stacks],
            'final_result_stack_count': len(final_current_stacks),
            'final_result_stacks_left_to_right': [list(s) for s in final_current_stacks],
            'main_score': main_score,
            'process_score': process_score,
            'solution_process': process_details,
            'deduction_target_frames': deduct_target,
            'deduction_background': deduct_bg,
            'background_consistency': bg_score,
            'final_score': score,
            'score_formula': 'main_score * (0.4 + 0.6 * process_score) - deductions',
            'note': 'interleave: metadata-ordered completed-move state machine',
        }
        return float(score)

    # ---------- block detection ----------
    def _detect_all_blocks_fine_region(self, region: np.ndarray) -> List[Dict]:
        """Detect blocks, splitting tall contours into stacked sub-blocks by
        an expected block height (derived from the region height). Each
        sub-block's hue is sampled from its own central band, so adjacent
        stacked blocks with different colors are recovered as separate
        blocks even when their contours merge."""
        if region.size == 0:
            return []
        H, W = region.shape[:2]
        block_h_est = max(8, int(round(H * self.EXPECTED_BLOCK_H_RATIO)))
        area_scale = (H * W) / float(self.DETECT_BASE_H * self.DETECT_BASE_W)
        lin_scale = ((H / float(self.DETECT_BASE_H)) +
                     (W / float(self.DETECT_BASE_W))) / 2.0
        min_area = self.DETECT_MIN_AREA * area_scale
        min_w = self.DETECT_MIN_W * lin_scale
        min_h = self.DETECT_MIN_H * lin_scale
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] > self.DETECT_SAT_THRESH) &
                (hsv[:, :, 2] > self.DETECT_VAL_THRESH)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blocks: List[Dict] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < min_w or h < min_h:
                continue
            n = max(1, int(round(h / block_h_est)))
            for i in range(n):
                y0 = int(y + i * h / n)
                y1 = int(y + (i + 1) * h / n)
                pad_y = max(2, (y1 - y0) // 5)
                pad_x = max(2, w // 5)
                patch = hsv[y0 + pad_y:y1 - pad_y, x + pad_x:x + w - pad_x]
                if patch.size == 0:
                    continue
                m = patch[:, :, 1] > self.DETECT_SAT_THRESH
                if not m.any():
                    continue
                hue = int(np.median(patch[:, :, 0][m]))
                blocks.append({
                    'x': x + w // 2,
                    'y': (y0 + y1) // 2,
                    'hue': hue,
                    'color': self._canonical_color(hue),
                    'bw': w,
                    'bh': y1 - y0,
                    'area': float(w * (y1 - y0)),
                })
        return blocks

    # ---------- stack assembly ----------
    def _cluster_into_stacks_xtol(self, blocks: List[Dict]) -> List[List[Dict]]:
        if not blocks:
            return []
        bw_ref = max(b['bw'] for b in blocks)
        tol = max(8.0, bw_ref * self.STACK_X_CLUSTER_FRAC)
        sorted_x = sorted(blocks, key=lambda b: b['x'])
        stacks: List[List[Dict]] = [[sorted_x[0]]]
        for b in sorted_x[1:]:
            ref = float(np.mean([bb['x'] for bb in stacks[-1]]))
            if abs(b['x'] - ref) < tol:
                stacks[-1].append(b)
            else:
                stacks.append([b])
        return [sorted(s, key=lambda b: -b['y']) for s in stacks]  # bottom→top

    def _is_grounded(self, stack: List[Dict], H: int) -> bool:
        """A stack is grounded iff its bottom block sits near the floor."""
        if not stack:
            return False
        bottom_cy = stack[0]['y']
        return bottom_cy >= self.FLOOR_Y_FRAC * H

    def _frame_state(self, frame: np.ndarray, side_split: int) -> Dict:
        """Return grounded stacks on left/right sides and any in-flight blocks."""
        H, W = frame.shape[:2]
        blocks = self._detect_all_blocks_fine_region(frame)
        left_b = [b for b in blocks if b['x'] < side_split]
        right_b = [b for b in blocks if b['x'] >= side_split]
        L_all = self._cluster_into_stacks_xtol(left_b)
        R_all = self._cluster_into_stacks_xtol(right_b)
        L_grounded = [s for s in L_all if self._is_grounded(s, H)]
        L_flying = [s for s in L_all if not self._is_grounded(s, H)]
        R_grounded = [s for s in R_all if self._is_grounded(s, H)]
        R_flying = [s for s in R_all if not self._is_grounded(s, H)]
        return {
            'L_grounded': [tuple(b['color'] for b in s) for s in L_grounded],
            'L_x': [int(np.mean([b['x'] for b in s])) for s in L_grounded],
            'R_grounded': [tuple(b['color'] for b in s) for s in R_grounded],
            'R_x': [int(np.mean([b['x'] for b in s])) for s in R_grounded],
            'L_flying_colors': [tuple(b['color'] for b in s) for s in L_flying],
            'R_flying_colors': [tuple(b['color'] for b in s) for s in R_flying],
        }

    # ---------- matching ----------
    def _match_stacks(
        self,
        target: List[Tuple[str, ...]],
        result: List[Tuple[str, ...]],
    ) -> float:
        """Compare two lists of stacks (each already ordered left-to-right by
        x, each stack bottom→top). Score combines:
          - stack-count agreement (paired by position)
          - per-stack height + color sequence match
        Returns a score in [0, 1]."""
        if not target:
            return 0.0
        n_t = len(target)
        n_r = len(result)
        # Pair stacks by position; if counts differ, unpaired stacks score 0.
        n_pairs = min(n_t, n_r)
        total_blocks = sum(len(s) for s in target)
        if total_blocks == 0:
            return 0.0
        matched_blocks = 0
        for i in range(n_pairs):
            t, r = target[i], result[i]
            if len(t) == len(r):
                matched_blocks += sum(1 for a, b in zip(t, r) if a == b)
            # if heights differ, count zero blocks matched for this stack
        # Penalize stack-count mismatch on top of block-level matching.
        block_ratio = matched_blocks / total_blocks
        count_ratio = n_pairs / max(n_t, n_r)
        # Combine: require both to be high for a high score.
        combined = block_ratio * count_ratio
        if combined >= 0.999:
            return 1.0
        if combined >= 0.7:
            return 0.6
        if combined >= 0.5:
            return 0.3
        if combined > 0.0:
            return 0.1
        return 0.0

    # ---------- movement detection ----------
    def _state_signature(self, state: Dict) -> Tuple[Tuple[str, ...], ...]:
        """Multiset-like signature for grounded stacks (L + R) used to detect
        meaningful state changes (ignoring in-flight transients)."""
        L = tuple(sorted(state['L_grounded']))
        R = tuple(sorted(state['R_grounded']))
        return (L, R)

    def _smoothed_change_indices(self, states: List[Dict], K: int) -> List[int]:
        sigs = [self._state_signature(s) for s in states]
        n = len(sigs)
        if n == 0:
            return []
        idx = [0]
        i = 1
        while i < n:
            if sigs[i] != sigs[idx[-1]]:
                j = i
                while j + 1 < n and sigs[j + 1] == sigs[i]:
                    j += 1
                if (j - i + 1) >= K:
                    idx.append(i)
                    i = j + 1
                    continue
            i += 1
        return idx

    def _stack_diff(
        self,
        prev: List[Tuple[str, ...]],
        prev_x: List[int],
        curr: List[Tuple[str, ...]],
        curr_x: List[int],
        side: str,
    ) -> List[Dict]:
        """Pair stacks by closest x, emit ADD_TOP / REMOVE_TOP / REPLACE_TOP /
        APPEAR / VANISH events.
        """
        prev_p = list(zip(prev_x, prev))
        curr_p = list(zip(curr_x, curr))
        used = set()
        events: List[Dict] = []
        for px, pc in prev_p:
            best = -1
            best_d = 1e9
            for j, (cx, cc) in enumerate(curr_p):
                if j in used:
                    continue
                d = abs(px - cx)
                if d < best_d:
                    best_d = d
                    best = j
            # rough block-width tolerance: assume ~60px
            if best >= 0 and best_d < 50:
                used.add(best)
                cx, cc = curr_p[best]
                if pc == cc:
                    continue
                n_p = len(pc); n_c = len(cc)
                if n_c == n_p + 1 and pc == cc[:n_p]:
                    events.append({'kind': 'ADD_TOP', 'side': side, 'x': cx,
                                   'color': cc[-1]})
                elif n_c == n_p - 1 and cc == pc[:n_c]:
                    events.append({'kind': 'REMOVE_TOP', 'side': side, 'x': px,
                                   'color': pc[-1]})
                elif n_c == n_p and n_p >= 1 and pc[:n_p - 1] == cc[:n_c - 1]:
                    events.append({'kind': 'REPLACE_TOP', 'side': side, 'x': cx,
                                   'color_from': pc[-1], 'color_to': cc[-1]})
                else:
                    events.append({'kind': 'OTHER', 'side': side,
                                   'from': pc, 'to': cc, 'x': cx})
            else:
                if len(pc) == 1:
                    events.append({'kind': 'REMOVE_TOP', 'side': side, 'x': px,
                                   'color': pc[-1]})
                else:
                    events.append({'kind': 'VANISH', 'side': side, 'x': px,
                                   'colors': pc})
        for j, (cx, cc) in enumerate(curr_p):
            if j in used:
                continue
            if len(cc) == 1:
                events.append({'kind': 'ADD_TOP', 'side': side, 'x': cx,
                               'color': cc[-1]})
            else:
                events.append({'kind': 'APPEAR', 'side': side, 'x': cx,
                               'colors': cc})
        return events

    def _detect_movements(self, states: List[Dict]) -> List[Dict]:
        """Track grounded-stack state changes, pair lift/place events into
        complete movements, classify each as valid/invalid by source-top /
        dest-top / color-preserved rules."""
        change_idx = self._smoothed_change_indices(states, self.STATE_PERSIST_FRAMES)
        if len(change_idx) < 2:
            return []

        all_events: List[Tuple[int, Dict]] = []
        for k in range(1, len(change_idx)):
            i_prev, i_curr = change_idx[k - 1], change_idx[k]
            sp = states[i_prev]; sc = states[i_curr]
            evs = self._stack_diff(sp['L_grounded'], sp['L_x'],
                                   sc['L_grounded'], sc['L_x'], 'L')
            evs += self._stack_diff(sp['R_grounded'], sp['R_x'],
                                    sc['R_grounded'], sc['R_x'], 'R')
            for e in evs:
                e['frame'] = i_curr
                all_events.append((i_curr, e))

        # Pair lifts with subsequent placements by color. 
        _kind_order = {'REMOVE_TOP': 0, 'VANISH': 1, 'REPLACE_TOP': 2,
                       'ADD_TOP': 3, 'APPEAR': 4, 'OTHER': 5}
        in_transit: List[Dict] = []
        movements: List[Dict] = []
        for fi, e in sorted(all_events,
                            key=lambda x: (x[0], _kind_order.get(x[1]['kind'], 9))):
            k = e['kind']
            if k == 'REMOVE_TOP':
                in_transit.append({'color': e['color'], 'side': e['side'],
                                   'frame': fi, 'x': e['x']})
            elif k == 'ADD_TOP':
                # Pair LIFO: a placement is most likely the most recently
                # lifted block of the same color.
                m_idx = -1
                for j in range(len(in_transit) - 1, -1, -1):
                    if in_transit[j]['color'] == e['color']:
                        m_idx = j; break
                if m_idx >= 0:
                    src = in_transit.pop(m_idx)
                    movements.append({
                        'kind': 'transit',
                        'source_side': src['side'], 'source_frame': src['frame'],
                        'source_x': src['x'],
                        'dest_side': e['side'], 'dest_frame': fi,
                        'dest_x': e['x'],
                        'color': e['color'],
                        'valid': True,
                        'reason': 'source=top, dest=top, color preserved',
                    })
                else:
                    movements.append({
                        'kind': 'appear',
                        'dest_side': e['side'], 'dest_frame': fi,
                        'color': e['color'],
                        'valid': False,
                        'reason': 'block appeared on top without matching prior lift',
                    })
            elif k == 'REPLACE_TOP':
                movements.append({
                    'kind': 'replace',
                    'dest_side': e['side'], 'dest_frame': fi,
                    'color_from': e['color_from'], 'color_to': e['color_to'],
                    'valid': False,
                    'reason': f"top color changed ({e['color_from']}->{e['color_to']})",
                })
            elif k == 'APPEAR':
                movements.append({
                    'kind': 'stack_appear',
                    'dest_side': e['side'], 'dest_frame': fi,
                    'colors': list(e['colors']),
                    'valid': False,
                    'reason': 'new stack appeared on the ground without lift',
                })
            elif k == 'VANISH':
                movements.append({
                    'kind': 'stack_vanish',
                    'source_side': e['side'], 'source_frame': fi,
                    'colors': list(e['colors']),
                    'valid': False,
                    'reason': 'stack disappeared without place',
                })
            else:
                movements.append({'kind': k, 'detail': e, 'frame': fi,
                                  'valid': False, 'reason': 'irregular change'})

        for t in in_transit:
            movements.append({
                'kind': 'lost',
                'source_side': t['side'], 'source_frame': t['frame'],
                'color': t['color'],
                'valid': False,
                'reason': 'lifted block never landed',
            })
        return movements

    # ---------- target preservation ----------
    def _stack_list_match_ratio(
        self,
        a: List[Tuple[str, ...]],
        b: List[Tuple[str, ...]],
    ) -> float:
        """Per-block matching ratio over all paired stacks (by left-to-right
        position). Returns 0..1; reaches 1.0 only when stack count, heights,
        and colors all match."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        n_pairs = min(len(a), len(b))
        total = sum(len(s) for s in a)
        if total == 0:
            return 0.0
        matched = 0
        for i in range(n_pairs):
            ta, tb = a[i], b[i]
            if len(ta) == len(tb):
                matched += sum(1 for x, y in zip(ta, tb) if x == y)
        block_ratio = matched / total
        count_ratio = n_pairs / max(len(a), len(b))
        return block_ratio * count_ratio

    def _target_preservation_all_frames_v2(
        self, per_frame_state: List[Dict],
        reference_right_stacks: List[Tuple[str, ...]],
    ) -> Tuple[float, float]:
        if not reference_right_stacks:
            return (1.0, 1.0)
        vals = []
        for s in per_frame_state:
            vals.append(self._stack_list_match_ratio(
                reference_right_stacks, s['R_grounded']))
        return (float(min(vals)), float(np.mean(vals)))

    def _stack_pixels_mask(self, frame: np.ndarray) -> np.ndarray:
        """Binary mask (255 = stack pixels) covering all saturated regions."""
        if frame.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 30)).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        return mask

    def _background_consistency_masked_stacks(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        side_split: int,
        h: int,
        w: int,
    ) -> Tuple[float, Dict]:
        """Mask saturated (stack) pixels via a GT-derived mask; compare the
        remaining (background) pixels between generated and GT frames when
        available, else fall back to temporal stability on the generated
        video only."""
        ref = gt_first_frame
        if ref is None or ref.shape[:2] != (h, w):
            ref = gt_final_frame
        if ref is None or ref.shape[:2] != (h, w):
            return (1.0, {'skipped': True, 'reason': 'no_gt_shape_match'})

        stack_mask = self._stack_pixels_mask(ref)
        bg_mask = cv2.bitwise_not(stack_mask)
        if not np.any(bg_mask > 0):
            return (1.0, {'skipped': True, 'reason': 'empty_background'})

        if gt_frames and len(gt_frames) > 0:
            n = min(len(video_frames), len(gt_frames))
            ssims: List[float] = []
            for i in range(n):
                g = video_frames[i]
                gt = gt_frames[i]
                if g.shape[:2] != (h, w):
                    g = normalize_frame_size(g, ref)
                if gt.shape[:2] != (h, w):
                    gt = normalize_frame_size(gt, ref)
                crop_g = cv2.bitwise_and(g, g, mask=bg_mask)
                crop_gt = cv2.bitwise_and(gt, gt, mask=bg_mask)
                ssims.append(compute_ssim(crop_g, crop_gt))
            mean_ssim = float(np.mean(ssims)) if ssims else 0.0
            return (mean_ssim, {'frame_pairs': n, 'mean_ssim_masked': mean_ssim})

        # Fallback: temporal stability of masked background on generated only
        ssims_fb: List[float] = []
        for i in range(len(video_frames) - 1):
            g0 = video_frames[i]
            g1 = video_frames[i + 1]
            c0 = cv2.bitwise_and(g0, g0, mask=bg_mask)
            c1 = cv2.bitwise_and(g1, g1, mask=bg_mask)
            ssims_fb.append(compute_ssim(c0, c1))
        mean_fb = float(np.mean(ssims_fb)) if ssims_fb else 1.0
        return (mean_fb, {'fallback': 'gen_temporal_only', 'mean_ssim_masked': mean_fb})


class MoveObjectsToTargetEvaluator(BaseEvaluator):
    """
    O-27: move the pink and blue balls into their matching target rings.

    The legacy scorer only compared coarse colour centroids in the final frame
    and a mid-frame "did both move?" check.  That misses the core contract:

    - each coloured ball should land on its own coloured target
    - both balls should actually travel there
    - they should move at roughly the same time

    We therefore track the pink and blue balls across the whole clip and
    score:

        score = mean(per_ball_score) x synchronization

    where each ball uses a source-anchored path-functional transport term:
    maximum prefix progress, positive-vs-negative projected variation, lateral
    corridor deviation, and endpoint landing.
    """
    
    COLOR_RANGES = {
        "pink": [
            (
                np.array([135, 35, 80], dtype=np.uint8),
                np.array([179, 255, 255], dtype=np.uint8),
            ),
        ],
        "blue": [
            (
                np.array([90, 35, 60], dtype=np.uint8),
                np.array([130, 255, 255], dtype=np.uint8),
            ),
        ],
    }
    COLOR_ORDER = ("pink", "blue")
    MIN_COMPONENT_AREA = 250.0
    LANDING_HALF_PX = 45.0
    MOVE_START_FRAC = 0.10
    MOVE_END_FRAC = 0.90
    MOTION_TOL_PX = 4.0
    PROGRESS_SATURATION = 0.90
    PROGRESS_ONLY_FLOOR = 0.25
    PATH_QUALITY_FLOOR = 0.5
    FORWARD_RATIO_SATURATION = 0.97
    SOURCE_ANCHOR_FULL_PX = 12.0
    SOURCE_ANCHOR_DROP_PX = 80.0
    SOURCE_ANCHOR_RELATIVE_SATURATION = 0.80
    LATERAL_FULL_PX = 5.0
    LATERAL_DROP_PX = 60.0
    ENDPOINT_FULL_PX = 5.0
    ENDPOINT_DROP_PX = 120.0
    VISIBILITY_SATURATION = 0.90
    TELEPORT_PX_FRAC = 0.25
    TELEPORT_EXPECTED_FRAC = 0.50
    TELEPORT_PENALTY_BASE = 0.25
    MAX_TELEPORT_EVENTS = 6
    MOTION_DURATION_FULL_FRAC = 0.15
    MOTION_DURATION_MIN_FRAMES = 2.0

    def _color_mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.COLOR_RANGES[color]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

    def _extract_color_candidates(
        self, frame: np.ndarray, color: str,
    ) -> List[Dict[str, float]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._color_mask(hsv, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Dict[str, float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.MIN_COMPONENT_AREA:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            perimeter = float(cv2.arcLength(contour, True))
            circularity = (
                4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
            )
            candidates.append({
                "center": (
                    float(moments["m10"] / moments["m00"]),
                    float(moments["m01"] / moments["m00"]),
                ),
                "area": area,
                "fill_ratio": float(area / max(w * h, 1)),
                "circularity": circularity,
            })
        return candidates

    def _pick_reference_ball(
        self,
        frame: np.ndarray,
        color: str,
        prefer_side: Optional[str] = None,
    ) -> Optional[Dict[str, float]]:
        candidates = self._extract_color_candidates(frame, color)
        if not candidates:
            return None
        if prefer_side == "left":
            return min(
                candidates,
                key=lambda cand: (cand["center"][0], -cand["fill_ratio"], -cand["area"]),
            )
        if prefer_side == "right":
            return max(
                candidates,
                key=lambda cand: (cand["center"][0], cand["fill_ratio"], cand["area"]),
            )
        return max(
            candidates,
            key=lambda cand: (
                cand["fill_ratio"],
                cand["circularity"],
                cand["area"],
            ),
        )

    def _track_ball(
        self,
        video_frames: Sequence[np.ndarray],
        color: str,
        source: Dict[str, float],
    ) -> List[Optional[Dict[str, float]]]:
        trace: List[Optional[Dict[str, float]]] = []
        last_center = tuple(source["center"])
        source_area = float(source["area"])

        for frame in video_frames:
            candidates = self._extract_color_candidates(frame, color)
            if not candidates:
                trace.append(None)
                continue
            best = min(
                candidates,
                key=lambda cand: (
                    safe_distance(cand["center"], last_center)
                    + 40.0 * abs(cand["area"] - source_area) / max(source_area, 1.0)
                    - 20.0 * cand["fill_ratio"]
                ),
            )
            trace.append(best)
            last_center = tuple(best["center"])
        return trace

    def _extract_rgb_candidates(self, frame, rgb, tol=70.0):
        """Like _extract_color_candidates but detects an arbitrary RGB (VBVR-Pro
        renders per-sample object colours read from metadata)."""
        target_bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
        d = np.linalg.norm(frame.astype(np.float32) - target_bgr, axis=2)
        mask = (d < tol).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.MIN_COMPONENT_AREA:
                continue
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            perim = float(cv2.arcLength(contour, True))
            circ = 4.0 * np.pi * area / (perim * perim) if perim > 0 else 0.0
            out.append({
                "center": (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])),
                "area": area, "fill_ratio": float(area / max(w * h, 1)), "circularity": circ,
            })
        return out

    def _track_ball_rgb(self, video_frames, rgb, source):
        trace = []
        last_center = tuple(source["center"])
        source_area = float(source["area"])
        for frame in video_frames:
            candidates = self._extract_rgb_candidates(frame, rgb)
            if not candidates:
                trace.append(None)
                continue
            best = min(candidates, key=lambda c: (
                safe_distance(c["center"], last_center)
                + 40.0 * abs(c["area"] - source_area) / max(source_area, 1.0)
                - 20.0 * c["fill_ratio"]))
            trace.append(best)
            last_center = tuple(best["center"])
        return trace

    def _eval_v2_video(self, video_frames, eval_info):
        """VBVR-Pro video path: read per-sample object colours + start/target from
        metadata and track by RGB. Returns None when not a v2 sample (→ v1 path)."""
        import json as _json, os as _os
        mp = eval_info.get("metafile_path")
        if isinstance(mp, (list, tuple)):
            mp = next((p for p in mp if p and _os.path.exists(p)), None)
        if not (mp and _os.path.exists(mp)):
            gp = eval_info.get("gt_path", "")
            mp = _os.path.join(gp, "metadata.json")
        if not _os.path.exists(mp):
            return None
        try:
            meta = _json.load(open(mp))
        except Exception:
            return None
        params = meta.get("parameters") or {}
        sgt = meta.get("semantic_ground_truth") or {}
        objs = sgt.get("objects") or []
        colors = [o.get("color") for o in objs]
        starts = [o.get("initial_center") for o in objs]
        targets = [o.get("target_center") for o in objs]
        if not (colors and starts and targets) or any(v is None for v in colors + starts + targets):
            colors = params.get("object_colors") or []
            starts = params.get("object_starts") or []
            targets = params.get("target_positions") or []
        if not (colors and starts and targets) or not (len(colors) == len(starts) == len(targets)):
            return None 
        canvas = params.get("canvas") or sgt.get("canvas") or {}
        fh, fw = video_frames[0].shape[:2]
        sx = fw / float(canvas.get("width", fw))
        sy = fh / float(canvas.get("height", fh))
        per_scores, per_details = [], {}
        for i in range(len(colors)):
            rgb = colors[i]
            sc = (starts[i][0] * sx, starts[i][1] * sy)
            tg = (targets[i][0] * sx, targets[i][1] * sy)
            c0 = self._extract_rgb_candidates(video_frames[0], rgb)
            src = min(c0, key=lambda c: safe_distance(c["center"], sc)) if c0 else {
                "center": sc, "area": self.MIN_COMPONENT_AREA, "fill_ratio": 1.0}
            trace = self._track_ball_rgb(video_frames, rgb, src)
            score, details = self._score_ball_trace(trace, sc, tg, video_frames[0].shape)
            per_scores.append(score)
            per_details[f"obj{i}"] = details
        if not per_scores:
            return None
        movement = float(np.mean(per_scores))
        sync = self._synchronization_score(per_details, len(video_frames))
        total = movement * (0.8 + 0.2 * sync)
        self._last_task_details = {"schema": "v2", "movement": round(movement, 4),
                                   "synchronization": round(sync, 4),
                                   "score": round(total, 4), "per_color": per_details}
        return total

    def _movement_window(
        self,
        trace: Sequence[Optional[Dict[str, float]]],
        start_center: Tuple[float, float],
        target_center: Tuple[float, float],
        expected_distance: float,
    ) -> Tuple[Optional[int], Optional[int]]:
        detected = [(idx, point) for idx, point in enumerate(trace) if point is not None]
        if len(detected) < 2 or expected_distance <= self.MOTION_TOL_PX:
            return None, None

        start_threshold = expected_distance * self.MOVE_START_FRAC
        end_threshold = expected_distance * self.MOVE_END_FRAC
        source = np.array(start_center, dtype=float)
        target = np.array(target_center, dtype=float)
        unit = (target - source) / expected_distance
        cumulative_projected = 0.0
        max_projected = 0.0
        prev_center = source
        start_idx: Optional[int] = None
        end_idx: Optional[int] = None
        for frame_idx, point in detected:
            curr_center = np.array(point["center"], dtype=float)
            cumulative_projected += float(np.dot(curr_center - prev_center, unit))
            max_projected = max(max_projected, cumulative_projected)
            if start_idx is None and max_projected >= start_threshold:
                start_idx = frame_idx
            if end_idx is None and max_projected >= end_threshold:
                end_idx = frame_idx
            prev_center = curr_center
        return start_idx, end_idx

    def _endpoint_score(self, final_dist: float) -> float:
        if final_dist <= self.ENDPOINT_FULL_PX:
            return 1.0
        return max(
            0.0,
            1.0 - (final_dist - self.ENDPOINT_FULL_PX) / self.ENDPOINT_DROP_PX,
        )

    def _source_anchor_score(self, start_dist: float, expected_distance: float) -> float:
        if start_dist <= self.SOURCE_ANCHOR_FULL_PX:
            return 1.0
        fixed_score = max(
            0.0,
            1.0 - (
                (start_dist - self.SOURCE_ANCHOR_FULL_PX)
                / self.SOURCE_ANCHOR_DROP_PX
            ),
        )
        if expected_distance <= 1e-6:
            return fixed_score
        relative_score = max(
            0.0,
            1.0 - (
                start_dist
                / (self.SOURCE_ANCHOR_RELATIVE_SATURATION * expected_distance)
            ),
        )
        return min(fixed_score, relative_score)

    def _trace_path_functional(
        self,
        trace: Sequence[Optional[Dict[str, float]]],
        start_center: Tuple[float, float],
        target_center: Tuple[float, float],
    ) -> Tuple[float, Dict[str, Any]]:
        expected_distance = safe_distance(start_center, target_center)
        details: Dict[str, Any] = {
            "path_score": 0.0,
            "source_anchor_dist_px": None,
            "source_anchor_score": 0.0,
            "progress_score": 0.0,
            "projected_path_integral_px": 0.0,
            "max_projected_progress_px": 0.0,
            "positive_projected_px": 0.0,
            "negative_projected_px": 0.0,
            "forward_ratio": 0.0,
            "forward_score": 0.0,
            "lateral_p90_px": 0.0,
            "lateral_score": 0.0,
        }

        detected = [point for point in trace if point is not None]
        if not detected:
            return 0.0, details
        if expected_distance <= self.MOTION_TOL_PX:
            details.update({
                "path_score": 1.0,
                "source_anchor_dist_px": 0.0,
                "source_anchor_score": 1.0,
                "progress_score": 1.0,
                "forward_ratio": 1.0,
                "forward_score": 1.0,
                "lateral_score": 1.0,
            })
            return 1.0, details

        source = np.array(start_center, dtype=float)
        target = np.array(target_center, dtype=float)
        unit = (target - source) / expected_distance
        lateral_unit = np.array([-unit[1], unit[0]], dtype=float)

        first_center = np.array(detected[0]["center"], dtype=float)
        source_anchor_dist = float(np.linalg.norm(first_center - source))
        source_anchor = self._source_anchor_score(source_anchor_dist, expected_distance)

        prev_center = first_center
        projected_integral = 0.0
        max_projected = 0.0
        positive = 0.0
        negative = 0.0
        lateral_values = []
        for point in detected:
            curr_center = np.array(point["center"], dtype=float)
            lateral_values.append(abs(float(np.dot(curr_center - source, lateral_unit))))
        for point in detected[1:]:
            curr_center = np.array(point["center"], dtype=float)
            projected_step = float(np.dot(curr_center - prev_center, unit))
            projected_integral += projected_step
            max_projected = max(max_projected, projected_integral)
            if projected_step >= 0.0:
                positive += projected_step
            else:
                negative += -projected_step
            prev_center = curr_center

        progress_px = float(np.clip(max_projected, 0.0, expected_distance))
        progress_score = min(
            1.0,
            progress_px / max(self.PROGRESS_SATURATION * expected_distance, 1e-6),
        )
        variation = positive + negative
        forward_ratio = positive / variation if variation > 1e-6 else 0.0
        forward_score = min(1.0, forward_ratio / self.FORWARD_RATIO_SATURATION)

        lateral_p90 = float(np.percentile(lateral_values, 90)) if lateral_values else 0.0
        if lateral_p90 <= self.LATERAL_FULL_PX:
            lateral_score = 1.0
        else:
            lateral_score = max(
                0.0,
                1.0 - (lateral_p90 - self.LATERAL_FULL_PX) / self.LATERAL_DROP_PX,
            )

        path_score = source_anchor * float(progress_score) * forward_score * lateral_score
        details.update({
            "path_score": round(float(path_score), 4),
            "source_anchor_dist_px": round(float(source_anchor_dist), 2),
            "source_anchor_score": round(float(source_anchor), 4),
            "progress_score": round(float(progress_score), 4),
            "projected_path_integral_px": round(float(projected_integral), 2),
            "max_projected_progress_px": round(float(progress_px), 2),
            "positive_projected_px": round(float(positive), 2),
            "negative_projected_px": round(float(negative), 2),
            "forward_ratio": round(float(forward_ratio), 4),
            "forward_score": round(float(forward_score), 4),
            "lateral_p90_px": round(float(lateral_p90), 2),
            "lateral_score": round(float(lateral_score), 4),
        })
        return path_score, details

    def _translation_score(
        self,
        path_score: float,
        endpoint_score: float,
        expected_distance: float,
    ) -> float:
        if expected_distance <= self.MOTION_TOL_PX:
            return float(endpoint_score)
        endpoint_weight = (
            self.PROGRESS_ONLY_FLOOR
            + (1.0 - self.PROGRESS_ONLY_FLOOR) * float(endpoint_score)
        )
        path_weight = (
            self.PATH_QUALITY_FLOOR
            + (1.0 - self.PATH_QUALITY_FLOOR) * float(path_score)
        )
        return path_weight * endpoint_weight

    def _teleport_events(
        self,
        trace: Sequence[Optional[Dict[str, float]]],
        expected_distance: float,
        frame_shape: Optional[Tuple[int, ...]],
    ) -> int:
        detected = [(idx, point) for idx, point in enumerate(trace) if point is not None]
        if len(detected) <= 2:
            return 0
        frame_limit = 0.0
        if frame_shape is not None and len(frame_shape) >= 2:
            frame_limit = self.TELEPORT_PX_FRAC * min(frame_shape[:2])
        per_frame_limit = max(
            frame_limit,
            self.TELEPORT_EXPECTED_FRAC * expected_distance,
        )
        events = 0
        for (prev_idx, prev), (curr_idx, curr) in zip(detected, detected[1:]):
            frame_gap = max(1, curr_idx - prev_idx)
            if safe_distance(prev["center"], curr["center"]) > per_frame_limit * frame_gap:
                events += 1
        return events

    def _motion_duration_score(
        self,
        move_start_frame: Optional[int],
        move_end_frame: Optional[int],
        frame_count: int,
        require_process: bool = True,
    ) -> float:
        if not require_process:
            return 1.0
        if move_start_frame is None or move_end_frame is None:
            return 0.0
        duration = max(0.0, float(move_end_frame - move_start_frame))
        full_duration = max(
            self.MOTION_DURATION_MIN_FRAMES,
            self.MOTION_DURATION_FULL_FRAC * max(frame_count - 1, 1),
        )
        return min(1.0, duration / full_duration)

    def _score_ball_trace(
        self,
        trace: Sequence[Optional[Dict[str, float]]],
        start_center: Tuple[float, float],
        target_center: Tuple[float, float],
        frame_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        detected = [(idx, point) for idx, point in enumerate(trace) if point is not None]
        if not detected:
            return 0.0, {
                "trace_detected": 0,
                "coverage": 0.0,
                "visibility": 0.0,
                "landing": 0.0,
                "endpoint_score": 0.0,
                "progress": 0.0,
                "progress_score": 0.0,
                "path_score": 0.0,
                "translation_score": 0.0,
                "source_anchor_dist_px": None,
                "source_anchor_score": 0.0,
                "forward_score": 0.0,
                "lateral_score": 0.0,
                "path_length": 0.0,
                "expected_distance": 0.0,
                "teleport_events": 0,
                "teleport_penalty": 1.0,
                "motion_duration_score": 0.0,
                "move_start_frame": None,
                "move_end_frame": None,
            }

        last = detected[-1][1]
        expected_distance = safe_distance(start_center, target_center)
        path_length = float(sum(
            safe_distance(detected[i - 1][1]["center"], detected[i][1]["center"])
            for i in range(1, len(detected))
        ))
        final_dist = safe_distance(last["center"], target_center)
        endpoint_score = self._endpoint_score(final_dist)
        path_score, path_details = self._trace_path_functional(
            trace,
            start_center,
            target_center,
        )

        coverage = len(detected) / max(len(trace), 1)
        visibility = min(1.0, coverage / self.VISIBILITY_SATURATION)
        translation_score = self._translation_score(
            path_score,
            endpoint_score,
            expected_distance,
        )
        teleport_events = self._teleport_events(
            trace,
            expected_distance,
            frame_shape,
        )
        teleport_penalty = self.TELEPORT_PENALTY_BASE ** min(
            teleport_events,
            self.MAX_TELEPORT_EVENTS,
        )
        move_start_frame, move_end_frame = self._movement_window(
            trace,
            start_center,
            target_center,
            expected_distance,
        )
        require_motion_process = (
            endpoint_score >= 0.8
            and float(path_details["progress_score"]) >= 0.9
        )
        motion_duration_score = self._motion_duration_score(
            move_start_frame,
            move_end_frame,
            len(trace),
            require_process=require_motion_process,
        )
        score = (
            translation_score
            * visibility
            * teleport_penalty
            * motion_duration_score
        )
        return score, {
            "trace_detected": len(detected),
            "coverage": round(float(coverage), 4),
            "visibility": round(float(visibility), 4),
            "landing": round(float(endpoint_score), 4),
            "endpoint_score": round(float(endpoint_score), 4),
            "progress": path_details["progress_score"],
            "progress_score": path_details["progress_score"],
            "path_score": path_details["path_score"],
            "translation_score": round(float(translation_score), 4),
            "source_anchor_dist_px": path_details["source_anchor_dist_px"],
            "source_anchor_score": path_details["source_anchor_score"],
            "projected_path_integral_px": path_details["projected_path_integral_px"],
            "max_projected_progress_px": path_details["max_projected_progress_px"],
            "positive_projected_px": path_details["positive_projected_px"],
            "negative_projected_px": path_details["negative_projected_px"],
            "forward_ratio": path_details["forward_ratio"],
            "forward_score": path_details["forward_score"],
            "lateral_p90_px": path_details["lateral_p90_px"],
            "lateral_score": path_details["lateral_score"],
            "path_length": round(float(path_length), 2),
            "expected_distance": round(float(expected_distance), 2),
            "final_dist_px": round(float(final_dist), 2),
            "teleport_events": teleport_events,
            "teleport_penalty": round(float(teleport_penalty), 4),
            "motion_duration_score": round(float(motion_duration_score), 4),
            "move_start_frame": move_start_frame,
            "move_end_frame": move_end_frame,
            "start_center": [round(float(start_center[0]), 2), round(float(start_center[1]), 2)],
            "target_center": [round(float(target_center[0]), 2), round(float(target_center[1]), 2)],
            "last_center": [round(float(last["center"][0]), 2), round(float(last["center"][1]), 2)],
        }

    def _synchronization_score(
        self,
        per_color: Dict[str, Dict[str, Any]],
        frame_count: int,
    ) -> float:
        if len(per_color) < 2:
            return 0.0
        windows = [
            (details["move_start_frame"], details["move_end_frame"])
            for details in per_color.values()
        ]
        if any(start is None or end is None for start, end in windows):
            return 0.0

        start_gap = abs(int(windows[0][0]) - int(windows[1][0]))
        end_gap = abs(int(windows[0][1]) - int(windows[1][1]))
        sync_window = max(2.0, 0.2 * max(frame_count - 1, 1))
        start_sync = max(0.0, 1.0 - start_gap / sync_window)
        end_sync = max(0.0, 1.0 - end_gap / sync_window)
        return float(np.sqrt(start_sync * end_sync))
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        del gt_frames
        if len(video_frames) < 2 or gt_first_frame is None or gt_final_frame is None:
            self._last_task_details = {"error": "missing_frames"}
            return 0.0

        v2 = self._eval_v2_video(video_frames, eval_info)
        if v2 is not None:
            return v2

        reference_pairs: Dict[str, Tuple[Dict[str, float], Dict[str, float]]] = {}
        for color in self.COLOR_ORDER:
            start_ball = self._pick_reference_ball(
                gt_first_frame, color, prefer_side="left",
            )
            target_ball = self._pick_reference_ball(
                gt_final_frame, color, prefer_side="right",
            )
            if start_ball is not None and target_ball is not None:
                reference_pairs[color] = (start_ball, target_ball)

        if len(reference_pairs) != len(self.COLOR_ORDER):
            self._last_task_details = {
                "error": "reference_detection_failed",
                "detected_colors": sorted(reference_pairs.keys()),
            }
            return 0.0

        per_color_scores: List[float] = []
        per_color_details: Dict[str, Dict[str, Any]] = {}
        for color, (source, target) in reference_pairs.items():
            trace = self._track_ball(video_frames, color, source)
            score, details = self._score_ball_trace(
                trace,
                tuple(source["center"]),
                tuple(target["center"]),
                video_frames[0].shape,
            )
            per_color_scores.append(score)
            per_color_details[color] = details

        movement = float(np.mean(per_color_scores)) if per_color_scores else 0.0
        synchronization = self._synchronization_score(
            per_color_details,
            len(video_frames),
        )
        total = movement * (0.8 + 0.2 * synchronization)

        self._last_task_details = {
            "movement": round(float(movement), 4),
            "synchronization": round(float(synchronization), 4),
            "score": round(float(total), 4),
            "per_color": per_color_details,
        }
        return total

class MazePathfindingEvaluator(BaseEvaluator):
    """
    O-39: Maze pathfinding evaluator.

    Multiplicative scoring, same family as G-15 / G-16 / G-18:

      score = proximity × coverage
            × (1 − 0.30 × continuity_penalty)
            × 0.5^num_wall_hit_cells

    The task is a 15×15 cell maze with black walls and white pathways.
    A green circle marks the start, a red flag marks the end.  The video
    visualises the solution path — typically yellow dots accumulating,
    but also green lines or moving coloured balls depending on the model.

    We compute the optimal cell set (all cells on any shortest path
    through the maze) via BFS, then score the predicted video against
    it.  Wall cells stepped on carry a G-15-style ``0.5^n`` penalty —
    drawing through walls is a hard failure mode for this task.
    """

    GRID_SIZE = 15
    MAX_PENALTY = 0.70
    JITTER_TOL = 0.05
    PENALTY_FLOOR = 0.05
    EXTRA_AGENT_PENALTY = 0.20
    COVERAGE_GAP_THRESHOLD = 2

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cell_size(frame: np.ndarray, grid_size: int = 15) -> float:
        return max(frame.shape[:2]) / float(grid_size)

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _detect_maze_structure(
        self, frame: np.ndarray, grid_size: int = 15,
    ) -> Tuple[Set[Tuple[int, int]], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Detect walls, start cell, and end cell from a maze frame.

        Returns (walls, start_cell, end_cell) where walls is a set of
        (row, col) cells that are walls (black), and start/end are the
        cells containing the green circle / red flag.
        """
        h, w = frame.shape[:2]
        cell_h, cell_w = h / grid_size, w / grid_size
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        walls: Set[Tuple[int, int]] = set()
        for r in range(grid_size):
            for c in range(grid_size):
                y1 = int(r * cell_h + cell_h * 0.2)
                y2 = int((r + 1) * cell_h - cell_h * 0.2)
                x1 = int(c * cell_w + cell_w * 0.2)
                x2 = int((c + 1) * cell_w - cell_w * 0.2)
                region = gray[y1:y2, x1:x2]
                if np.mean(region) < 80:
                    walls.add((r, c))

        # Green start marker
        green_mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
        start_cell = self._marker_to_cell(green_mask, frame.shape, grid_size)

        # Red end marker
        red_mask = (
            cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
            | cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        )
        end_cell = self._marker_to_cell(red_mask, frame.shape, grid_size)

        # Ensure start/end cells are not in walls
        if start_cell is not None:
            walls.discard(start_cell)
        if end_cell is not None:
            walls.discard(end_cell)

        return walls, start_cell, end_cell

    @staticmethod
    def _marker_to_cell(
        mask: np.ndarray,
        frame_shape: Tuple[int, ...],
        grid_size: int = 15,
    ) -> Optional[Tuple[int, int]]:
        """Convert a colour mask to the grid cell containing its centroid."""
        coords = np.where(mask > 0)
        if len(coords[0]) < 10:
            return None
        cy = float(np.mean(coords[0]))
        cx = float(np.mean(coords[1]))
        h, w = frame_shape[:2]
        r = min(max(int(cy * grid_size / h), 0), grid_size - 1)
        c = min(max(int(cx * grid_size / w), 0), grid_size - 1)
        return (r, c)

    def _detect_all_agents(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect yellow path-marker dots (hue 20-35)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([40, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        agents: List[Tuple[int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            agents.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))
        return agents

    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect the leading (newest) yellow dot — the one farthest from start."""
        agents = self._detect_all_agents(frame)
        return agents[0] if agents else None

    def _detect_newest_dot(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
    ) -> Optional[Tuple[int, int]]:
        """Detect the newest yellow dot by diffing with previous frame."""
        if prev_frame is None:
            agents = self._detect_all_agents(frame)
            return agents[0] if agents else None
        diff = cv2.absdiff(frame, prev_frame)
        hsv_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_diff, np.array([0, 20, 20]), np.array([180, 255, 255]))
        coords = np.where(mask > 0)
        if len(coords[0]) < 10:
            return None
        return (int(np.mean(coords[1])), int(np.mean(coords[0])))

    def _detect_agents_broad(
        self,
        frame: np.ndarray,
        ref_frame: Optional[np.ndarray] = None,
        walls: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """Detect agents via yellow-dot detection + diff-based fallback.

        Returns pixel (x, y) positions.  Yellow detection is tried first;
        if it finds nothing and *ref_frame* is provided, diff-based
        detection kicks in so that any-colour moving objects are found.
        """
        agents = self._detect_all_agents(frame)
        if agents or ref_frame is None:
            return agents
        changed = self._detect_changed_cells(
            frame, ref_frame, self.GRID_SIZE, walls or set(),
        )
        return [
            _maze_cell_center_px(c, frame.shape, self.GRID_SIZE)
            for c in changed
        ]

    # ------------------------------------------------------------------
    # GT path extraction (fallback)
    # ------------------------------------------------------------------

    def _extract_gt_path(self, gt_frames: List[np.ndarray]) -> List[Tuple[int, int]]:
        """Extract path by tracking new yellow dots across GT frames."""
        path: List[Tuple[int, int]] = []
        prev_dots: Set[Tuple[int, int]] = set()  # cells already seen
        grid_size = self.GRID_SIZE
        for frame in gt_frames:
            dots = self._detect_all_agents(frame)
            for dx, dy in dots:
                cell = _maze_pixel_to_cell(dx, dy, frame.shape, grid_size)
                if cell not in prev_dots:
                    prev_dots.add(cell)
                    px = _maze_cell_center_px(cell, frame.shape, grid_size)
                    path.append(px)
        return path

    # ------------------------------------------------------------------
    # Scoring (same scaffold as G-18)
    # ------------------------------------------------------------------

    def _score_path_correctness(
        self,
        video_frames: List[np.ndarray],
        gt_path: List[Tuple[int, int]],
        cell: float,
        ref_frame: Optional[np.ndarray] = None,
        walls: Optional[Set[Tuple[int, int]]] = None,
        optimal_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> float:
        """Per-frame cell-level proximity.

        For each frame, converts detected agents to grid cells and
        checks what fraction sits on the optimal cell set.  Frames
        with no detectable agent are skipped (the discontinuity
        penalty already handles gaps).
        """
        if not gt_path or optimal_cells is None:
            return 0.0
        grid_size = self.GRID_SIZE
        frame_scores: List[float] = []
        for frame in video_frames:
            agents = self._detect_agents_broad(frame, ref_frame, walls)
            if not agents:
                continue
            frame_shape = frame.shape
            detected_cells = {
                _maze_pixel_to_cell(ax, ay, frame_shape, grid_size)
                for ax, ay in agents
            }
            on_path = len(detected_cells & optimal_cells)
            frame_scores.append(
                on_path / len(detected_cells) if detected_cells else 0.0,
            )
        return sum(frame_scores) / len(frame_scores) if frame_scores else 0.0

    def _score_coverage_completion(
        self,
        video_frames: List[np.ndarray],
        ref_points: List[Tuple[int, int]],
        start_cell: Tuple[int, int],
        end_cell: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        cell: float,
        ref_frame: Optional[np.ndarray] = None,
    ) -> float:
        """Coverage = fraction of optimal-path cells visited, with
        start/end connectivity penalty.

        Collects all cells visited across every frame, then measures
        what share of the optimal cell set was reached.  A small penalty
        is applied if the path doesn't connect to start or end.
        """
        if not video_frames or not ref_points:
            return 0.0
        frame_shape = video_frames[0].shape
        grid_size = self.GRID_SIZE

        # Collect optimal cells from ref_points (pixel coords → cells)
        optimal_cells: Set[Tuple[int, int]] = set()
        for px, py in ref_points:
            optimal_cells.add(
                _maze_pixel_to_cell(px, py, frame_shape, grid_size),
            )

        # Collect all visited cells across frames.  Start and end cells
        # are always counted as visited because they are part of the maze
        # definition (green marker / red flag) and may not produce a
        # detection diff.
        visited: Set[Tuple[int, int]] = {start_cell, end_cell}
        for frame in video_frames:
            agents = self._detect_agents_broad(frame, ref_frame, walls)
            for ax, ay in agents:
                visited.add(
                    _maze_pixel_to_cell(ax, ay, frame_shape, grid_size),
                )

        # Coverage: fraction of optimal cells visited
        hit = len(visited & optimal_cells)
        return hit / len(optimal_cells) if optimal_cells else 0.0

    def _discontinuity_penalty(
        self,
        video_frames: List[np.ndarray],
        cell: float,
        ref_frame: Optional[np.ndarray] = None,
        walls: Optional[Set[Tuple[int, int]]] = None,
    ) -> float:
        """Continuity penalty based on agent-count progression.

        For a maze, dots accumulate monotonically. We penalise:
        - Frames with no dots at all (disappearance)
        - Backward jumps in dot count (dots disappearing)
        """
        n = len(video_frames)
        if n < 2:
            return 1.0
        dot_counts: List[int] = []
        no_dot_count = 0
        for frame in video_frames:
            dots = self._detect_agents_broad(frame, ref_frame, walls)
            if not dots:
                no_dot_count += 1
            dot_counts.append(len(dots))

        disappear_rate = no_dot_count / n

        # Backward jumps: dot count decreases
        backward = 0
        transitions = 0
        for i in range(1, len(dot_counts)):
            if dot_counts[i - 1] > 0 and dot_counts[i] > 0:
                transitions += 1
                if dot_counts[i] < dot_counts[i - 1] - 1:
                    backward += 1
        backward_rate = backward / max(transitions, 1)

        penalty = 0.5 * disappear_rate + 0.5 * backward_rate
        if penalty < self.PENALTY_FLOOR:
            penalty = 0.0
        return min(1.0, penalty)

    # ------------------------------------------------------------------
    # Optimal path info
    # ------------------------------------------------------------------

    def _compute_optimal_path_info(
        self, frame: np.ndarray,
    ) -> Optional[Tuple[
        List[Tuple[int, int]],
        Dict[int, List[Tuple[int, int]]],
        int,
        Tuple[int, int],
        Tuple[int, int],
        Set[Tuple[int, int]],
        FrozenSet[Tuple[int, int]],
    ]]:
        """Detect maze structure and compute optimal cell set.

        Returns (all_points, by_dist, shortest, start_cell, end_cell,
        walls, optimal_cells) or None.
        """
        grid_size = self.GRID_SIZE
        walls, start_cell, end_cell = self._detect_maze_structure(frame, grid_size)
        if start_cell is None or end_cell is None:
            return None

        result = _maze_optimal_cell_set(start_cell, end_cell, walls, grid_size)
        if result is None:
            return None
        optimal_cells, dist_start, shortest = result

        all_points: List[Tuple[int, int]] = []
        by_dist: Dict[int, List[Tuple[int, int]]] = {}
        for c in optimal_cells:
            px = _maze_cell_center_px(c, frame.shape, grid_size)
            all_points.append(px)
            by_dist.setdefault(dist_start[c], []).append(px)

        return all_points, by_dist, shortest, start_cell, end_cell, walls, optimal_cells

    # ------------------------------------------------------------------
    # Video evaluation
    # ------------------------------------------------------------------
    WALL_HIT_INNER_MARGIN = 0.2  # keep inner 60% (20% margin each side)
    WALL_HIT_INNER_PIXEL_FRAC = 0.05  # ≥5% of inner area must be saturated

    def _wall_interior_touched(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray,
        cell: Tuple[int, int],
    ) -> bool:
        """True iff the wall cell's inner core contains saturated content
        that differs from the reference frame.

        Mirrors ``_detect_changed_cells`` (absdiff > 25 AND saturation ≥
        60) but measures only the inner 60% × 60% of the cell, ignoring
        anti-aliasing spill from a neighbour cell's outline.
        """
        h, w = frame.shape[:2]
        cell_h, cell_w = h / self.GRID_SIZE, w / self.GRID_SIZE
        m = self.WALL_HIT_INNER_MARGIN
        r, c = cell
        y1 = int(r * cell_h + cell_h * m)
        y2 = int((r + 1) * cell_h - cell_h * m)
        x1 = int(c * cell_w + cell_w * m)
        x2 = int((c + 1) * cell_w - cell_w * m)
        if y2 <= y1 or x2 <= x1:
            return False

        roi = frame[y1:y2, x1:x2]
        ref_roi = ref_frame[y1:y2, x1:x2]

        diff_gray = cv2.cvtColor(
            cv2.absdiff(roi, ref_roi), cv2.COLOR_BGR2GRAY,
        )
        _, diff_mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat_mask = cv2.inRange(
            hsv[:, :, 1],
            np.array([60], dtype=np.uint8),
            np.array([255], dtype=np.uint8),
        )
        mask = cv2.bitwise_and(diff_mask, sat_mask)

        inner_area = (y2 - y1) * (x2 - x1)
        min_pixels = max(3, int(inner_area * self.WALL_HIT_INNER_PIXEL_FRAC))
        return int(np.count_nonzero(mask)) >= min_pixels

    def _yellow_dot_in_wall_interior(
        self,
        frame: np.ndarray,
        cell: Tuple[int, int],
    ) -> bool:
        """True iff a yellow blob's bbox lies inside the wall cell's core."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([40, 255, 255]))
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        h, w = frame.shape[:2]
        cell_h, cell_w = h / self.GRID_SIZE, w / self.GRID_SIZE
        m = self.WALL_HIT_INNER_MARGIN
        r, c = cell
        y1 = int(r * cell_h + cell_h * m)
        y2 = int((r + 1) * cell_h - cell_h * m)
        x1 = int(c * cell_w + cell_w * m)
        x2 = int((c + 1) * cell_w - cell_w * m)
        for cnt in contours:
            if cv2.contourArea(cnt) < 30:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if y1 <= cy < y2 and x1 <= cx < x2:
                return True
        return False

    def _count_wall_hit_cells(
        self,
        video_frames: List[np.ndarray],
        ref_frame: Optional[np.ndarray],
        walls: Set[Tuple[int, int]],
    ) -> Set[Tuple[int, int]]:
        """Distinct wall cells whose *interior* ever hosts a coloured
        agent / line / dot.

        Strategy:
        1. Run the fast diff-based ``_detect_changed_cells`` to narrow
           down wall candidates (cell has *some* saturated diff).
        2. For each candidate, confirm with ``_wall_interior_touched`` —
           only the inner 60% × 60% of the cell counts.  This filters out
           outline-style renderings whose edge highlights bleed into
           adjacent wall cells without actually entering the wall.
        3. Explicit yellow dots in the wall's interior also count.
        """
        if not walls:
            return set()
        hit: Set[Tuple[int, int]] = set()
        for frame in video_frames:
            # Yellow dots centred inside a wall → hit.
            pending_yellow_cells = {
                _maze_pixel_to_cell(ax, ay, frame.shape, self.GRID_SIZE)
                for ax, ay in self._detect_all_agents(frame)
            } & walls
            for cell in pending_yellow_cells - hit:
                if self._yellow_dot_in_wall_interior(frame, cell):
                    hit.add(cell)

            if ref_frame is None:
                continue

            # Diff-based: fast screen, then strict inner-core confirm.
            candidates = self._detect_changed_cells(
                frame, ref_frame, self.GRID_SIZE, set(),
            ) & walls
            for cell in candidates - hit:
                if self._wall_interior_touched(frame, ref_frame, cell):
                    hit.add(cell)
        return hit

    @staticmethod
    def _count_wall_crossings(
        wall_hit_cells: Set[Tuple[int, int]],
    ) -> int:
        """Group touched wall cells into 4-connected components.

        A single crossing that traverses many adjacent wall cells (e.g. a
        line drawn straight through a thick wall block) counts as *one*
        crossing, not N.  Separate wall regions touched elsewhere add
        additional crossings.
        """
        if not wall_hit_cells:
            return 0
        remaining = set(wall_hit_cells)
        crossings = 0
        while remaining:
            seed = next(iter(remaining))
            stack = [seed]
            while stack:
                c = stack.pop()
                if c not in remaining:
                    continue
                remaining.discard(c)
                r, col = c
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nb = (r + dr, col + dc)
                    if nb in remaining:
                        stack.append(nb)
            crossings += 1
        return crossings

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """score = proximity × coverage × cont_factor × 0.5^wall_hits"""
        if not video_frames or gt_final_frame is None:
            return 0.0

        grid_frame = gt_frames[0] if gt_frames else gt_final_frame
        cell = self._cell_size(grid_frame, self.GRID_SIZE)

        ref_frame = video_frames[0] if video_frames else None

        opt = self._compute_optimal_path_info(grid_frame)
        walls: Set[Tuple[int, int]] = set()
        if opt is not None:
            all_pts, _bd, _td, start_cell, end_cell, walls, oc = opt
            proximity = self._score_path_correctness(
                video_frames, all_pts, cell, ref_frame, walls, oc,
            )
            coverage = self._score_coverage_completion(
                video_frames, all_pts, start_cell, end_cell, walls, cell,
                ref_frame,
            )
        else:
            gt_path = self._extract_gt_path(gt_frames)
            if gt_path:
                frame_shape = video_frames[0].shape
                fallback_oc: Set[Tuple[int, int]] = {
                    _maze_pixel_to_cell(px, py, frame_shape, self.GRID_SIZE)
                    for px, py in gt_path
                }
                sc = _maze_pixel_to_cell(
                    gt_path[0][0], gt_path[0][1], frame_shape, self.GRID_SIZE,
                )
                ec = _maze_pixel_to_cell(
                    gt_path[-1][0], gt_path[-1][1], frame_shape, self.GRID_SIZE,
                )
                proximity = self._score_path_correctness(
                    video_frames, gt_path, cell, ref_frame,
                    optimal_cells=frozenset(fallback_oc),
                )
                coverage = self._score_coverage_completion(
                    video_frames, gt_path, sc, ec, set(), cell, ref_frame,
                )
            else:
                proximity = 0.0
                coverage = 0.0

        penalty = self._discontinuity_penalty(video_frames, cell, ref_frame,
                                              walls if opt else None)
        continuity_factor = 1.0 - self.MAX_PENALTY * penalty

        wall_hit_cells = self._count_wall_hit_cells(
            video_frames, ref_frame, walls,
        )
        num_wall_crossings = self._count_wall_crossings(wall_hit_cells)
        wall_multiplier = 0.5 ** num_wall_crossings

        score = ((proximity + coverage + continuity_factor) / 3.0) * (0.4 + 0.6 * wall_multiplier)

        self._last_task_details = {
            "proximity": round(float(proximity), 4),
            "coverage": round(float(coverage), 4),
            "continuity_penalty": round(float(penalty), 4),
            "continuity_factor": round(float(continuity_factor), 4),
            "wall_hit_cells": [list(c) for c in sorted(wall_hit_cells)],
            "num_wall_hit_cells": len(wall_hit_cells),
            "num_wall_crossings": num_wall_crossings,
            "wall_multiplier": round(float(wall_multiplier), 6),
            "final_score": round(float(score), 4),
        }
        return score

    # ------------------------------------------------------------------
    # Interleave evaluation
    # ------------------------------------------------------------------

    def _detect_changed_cells(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray,
        grid_size: int,
        walls: Set[Tuple[int, int]],
    ) -> Set[Tuple[int, int]]:
        """Detect cells that changed significantly compared to *ref_frame*.

        Uses per-cell scanning so that long continuous lines (e.g. a drawn
        path) are correctly split into individual cells, rather than
        collapsing to a single contour centroid.

        **Saturation-gated**: a changed pixel only counts when the *current*
        frame's saturation at that pixel is high.  This filters out
        greyscale wall re-rendering / anti-aliasing drift (which has zero
        saturation) while keeping every coloured agent style (yellow dots,
        green lines, colour balls — all highly saturated).

        **Walls are not skipped**: wall cells are returned too, so the
        caller can compute an explicit wall-hit penalty.  The ``walls``
        argument is accepted for API compatibility but intentionally
        unused here.
        """
        del walls  # kept for API compat; caller derives wall hits elsewhere

        diff = cv2.absdiff(frame, ref_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_mask = cv2.inRange(
            hsv[:, :, 1], np.array([60], dtype=np.uint8),
            np.array([255], dtype=np.uint8),
        )
        mask = cv2.bitwise_and(diff_mask, sat_mask)

        h, w = frame.shape[:2]
        cell_h, cell_w = h / grid_size, w / grid_size
        min_pixels = max(5, int(cell_h * cell_w * 0.01))

        cells: Set[Tuple[int, int]] = set()
        for r in range(grid_size):
            for c in range(grid_size):
                y1 = int(r * cell_h)
                y2 = int((r + 1) * cell_h)
                x1 = int(c * cell_w)
                x2 = int((c + 1) * cell_w)
                if np.count_nonzero(mask[y1:y2, x1:x2]) >= min_pixels:
                    cells.add((r, c))
        return cells

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Interleave: cell-based scoring aligned with the video method.

        Image-mode path connectivity/efficiency stands in for video continuity;
        the final aggregation and wall-penalty floor mirror the video branch.
        """
        if not pred_images or input_frame is None:
            self._last_task_details = {"error": "no_input_or_pred"}
            return 0.0

        grid_size = self.GRID_SIZE
        walls, start_cell, end_cell = self._detect_maze_structure(
            input_frame, grid_size,
        )
        # Fallback: try GT frames if the input lacks a readable start/end
        if start_cell is None or end_cell is None:
            for src in (gt_images or []):
                walls, start_cell, end_cell = self._detect_maze_structure(
                    src, grid_size,
                )
                if start_cell is not None and end_cell is not None:
                    break
        if start_cell is None or end_cell is None:
            self._last_task_details = {"error": "no_start_or_end"}
            return 0.0

        opt = _maze_optimal_cell_set(start_cell, end_cell, walls, grid_size)
        if opt is None:
            self._last_task_details = {"error": "no_optimal_path"}
            return 0.0
        optimal_cells, dist_start, shortest = opt

        counts = maze.cell_draw_counts(
            pred_images, input_frame, grid_size=grid_size,
        )
        drawn = set(counts)
        walk = maze.simulate_walk_through_drawn(
            drawn=drawn, start=start_cell, end=end_cell,
            grid_size=grid_size,
            allow_cells={start_cell, end_cell},
        )
        optimal_set = set(optimal_cells)
        on_path = drawn & optimal_set
        total_pixels = sum(counts.values())
        on_path_pixels = sum(n for c, n in counts.items() if c in optimal_set)
        proximity = on_path_pixels / total_pixels if total_pixels else 0.0
        path_length = shortest + 1
        coverage = min(1.0, len(on_path) / max(path_length, 1))
        length_factor = min(1.0, path_length / len(drawn)) if drawn else 0.0
        continuity_factor = length_factor if walk else 0.0

        wall_hit_cells = set(drawn) & set(walls)
        remaining = set(wall_hit_cells)
        num_wall_crossings = 0
        while remaining:
            num_wall_crossings += 1
            stack = [remaining.pop()]
            while stack:
                r, c = stack.pop()
                for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if nb in remaining:
                        remaining.remove(nb)
                        stack.append(nb)
        wall_multiplier = 0.5 ** num_wall_crossings

        score = (
            (proximity + coverage + continuity_factor) / 3.0
            * (0.4 + 0.6 * wall_multiplier)
        )
        details = {
            "walk_reachable": bool(walk),
            "walk": [list(c) for c in (walk or [])[:60]],
            "num_drawn_cells": len(drawn),
            "num_optimal_cells": len(optimal_set),
            "num_on_path": len(on_path),
            "proximity": round(float(proximity), 4),
            "coverage": round(float(coverage), 4),
            "length_factor": round(float(length_factor), 4),
            "continuity_factor": round(float(continuity_factor), 4),
            "wall_hit_cells": [list(c) for c in sorted(wall_hit_cells)],
            "num_wall_hit_cells": len(wall_hit_cells),
            "num_wall_crossings": num_wall_crossings,
            "wall_multiplier": round(float(wall_multiplier), 6),
            "final_score": round(float(score), 4),
            "score_breakdown": {
                "formula": (
                    "mean(proximity, coverage, continuity)"
                    " × (0.4 + 0.6 × wall_multiplier)"
                ),
            },
        }
        details.update({
            "start_cell": list(start_cell),
            "end_cell": list(end_cell),
            "total_shortest_distance": shortest,
        })
        self._last_task_details = details
        return score


class ObjectSubtractionEvaluator(BaseEvaluator):
    """
    O-43: Object subtraction - remove specific objects, keep others.

    Evaluation:
    1. Kept objects correctness (40%): remaining shapes match GT
    2. Removed objects deletion (40%): deleted shapes are gone
    3. Background preservation (20%): background stays clean
    """

    TASK_WEIGHTS = {
        'completion': 0.80,
        'background_preservation': 0.20,
    }

    def _detect_fg_mask(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        corners = [frame[2, 2], frame[2, w-3], frame[h-3, 2], frame[h-3, w-3]]
        bg_color = np.mean(corners, axis=0)
        diff = np.sqrt(np.sum((frame.astype(float) - bg_color.astype(float)) ** 2, axis=2))
        binary = (diff > 30).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    def _detect_changed_region(self, gt_first: np.ndarray, gt_last: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(gt_first, gt_last)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    def _pixel_diff_score(self, frame1: np.ndarray, frame2: np.ndarray,
                          mask: np.ndarray,
                          thresholds: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)) -> Tuple[float, Dict]:
        mask_pixels = int((mask > 0).sum())
        if mask_pixels == 0:
            return 1.0, {'ratio': 0.0, 'changed_px': 0, 'total_px': 0}
        diff = cv2.absdiff(frame1, frame2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed = int((gray_diff[mask > 0] > 20).sum())
        ratio = float(changed) / mask_pixels
        t1, t2, t3, t4 = thresholds
        if ratio < t1:
            score = 1.0
        elif ratio < t2:
            score = 1.0 - (ratio - t1) / (t2 - t1) * 0.3
        elif ratio < t3:
            score = 0.7 - (ratio - t2) / (t3 - t2) * 0.4
        elif ratio < t4:
            score = 0.3 - (ratio - t3) / (t4 - t3) * 0.3
        else:
            score = 0.0
        return score, {'ratio': round(ratio, 6), 'changed_px': changed, 'total_px': mask_pixels}

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        gen_first = video_frames[0]
        gen_last = video_frames[-1]
        gt_first = gt_first_frame
        gt_last = gt_final_frame

        if gen_first.shape != gt_first.shape:
            gen_first = normalize_frame_size(gen_first, gt_first)
        if gen_last.shape != gt_last.shape:
            gen_last = normalize_frame_size(gen_last, gt_last)

        # Kept region: objects that remain in GT last
        kept_mask = self._detect_fg_mask(gt_last)
        # Removed region: objects in GT first that are gone in GT last
        changed_mask = self._detect_changed_region(gt_first, gt_last)
        fg_first = self._detect_fg_mask(gt_first)
        all_fg = cv2.bitwise_or(fg_first, kept_mask)

        kernel = np.ones((5, 5), np.uint8)
        kept_dilated = cv2.dilate(kept_mask, kernel, iterations=1)
        changed_dilated = cv2.dilate(changed_mask, kernel, iterations=1)
        all_fg_dilated = cv2.dilate(all_fg, kernel, iterations=1)

        # 1. Kept objects correctness: gt_last vs gen_last in kept region (erode edges)
        kept_score, kept_details = self._pixel_diff_score(
            gt_last, gen_last, cv2.erode(kept_mask, kernel, iterations=1),
            thresholds=(0.15, 0.25, 0.35, 0.60))

        # 2. Removed objects deletion: gt_last vs gen_last in changed region (erode edges)
        removed_score, removed_details = self._pixel_diff_score(
            gt_last, gen_last, cv2.erode(changed_mask, kernel, iterations=1),
            thresholds=(0.15, 0.25, 0.35, 0.60))

        # 3. Background preservation (20%): gen_first vs gen_last outside all fg
        bg_mask = cv2.bitwise_not(all_fg_dilated)
        bg_score, bg_details = self._pixel_diff_score(
            gen_first, gen_last, bg_mask, thresholds=(0.01, 0.025, 0.05, 0.10))

        scores = {
            'completion': round(kept_score * removed_score, 4),
            'background_preservation': round(bg_score, 4),
        }
        self._last_task_details = {
            **scores,
            'kept_objects': round(kept_score, 4),
            'removed_objects': round(removed_score, 4),
            **{f'kept_{k}': v for k, v in kept_details.items()},
            **{f'removed_{k}': v for k, v in removed_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class ShapeSorterEvaluator(BaseEvaluator):
    """
    O-46: drag each coloured shape into its matching outline on the right.

    The legacy scorer only counted coloured blobs on the left/right halves.
    That misses the actual task contract:

    - each source shape should end on its paired outline
    - the left tray should be emptied
    - the animation should move the shapes one at a time rather than all at
      once

    We reuse the robust filled/outline detector from OOD Part 1 and split the
    score into final sorting quality and process quality:

        score = final_layout x process_score
        final_layout = mean(final_pose) x left_clear x count_penalty
        process_score = geometric(transport_process, one_at_a_time)
    """

    ALIGNMENT_HALF_PX = 65.0
    ALIGNMENT_FULL_PX = 2.0
    AREA_RATIO_SATURATION = 0.93
    COVERAGE_SATURATION = 0.90
    STEP_MOVE_TOL_PX = 5.0
    SIMULTANEOUS_RATE_TOL = 0.08
    O46_MIN_SHAPE_AREA_FRAC = 0.0008
    SHAPE_PROCESS_WEIGHTS = {
        "travel": 0.65,
        "coverage": 0.35,
    }
    SCENE_PROCESS_WEIGHTS = {
        "transport_process": 0.35,
        "one_at_a_time": 0.65,
    }

    def _shape_helper(self) -> _ShapeDetectBase:
        return _ShapeDetectBase(device=self.device, task_name=self.task_name)

    def _o46_min_area(self, frame: np.ndarray) -> float:
        return max(300.0, frame.shape[0] * frame.shape[1] * self.O46_MIN_SHAPE_AREA_FRAC)

    @staticmethod
    def _o46_shape_type(contour: np.ndarray) -> str:
        area = cv2.contourArea(contour)
        perim = cv2.arcLength(contour, True)
        if perim <= 0.0:
            return "polygon"
        approx = cv2.approxPolyDP(contour, 0.04 * perim, True)
        vertices = len(approx)
        circularity = 4 * np.pi * area / (perim * perim)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        if solidity < 0.70 and vertices >= 8:
            return "star"
        if vertices == 3:
            return "triangle"
        if vertices == 4:
            return "square"
        if circularity >= 0.86:
            return "circle"
        if vertices in (5, 6, 7):
            return "hexagon"
        return "polygon"

    def _o46_shape_record(
        self,
        frame: np.ndarray,
        contour: np.ndarray,
        is_outline: bool,
    ) -> Optional[Dict[str, Any]]:
        area = cv2.contourArea(contour)
        if area < self._o46_min_area(frame):
            return None
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        perim = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perim * perim) if perim > 0 else 0.0
        approx = cv2.approxPolyDP(contour, 0.04 * perim, True) if perim > 0 else []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        region_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
        interior_mask = cv2.erode(region_mask, np.ones((5, 5), np.uint8))
        interior_hsv = hsv[interior_mask > 0]
        fill_hue = float(np.median(interior_hsv[:, 0])) if len(interior_hsv) else 0.0
        fill_sat = float(np.median(interior_hsv[:, 1])) if len(interior_hsv) else 0.0
        fill_val = float(np.median(interior_hsv[:, 2])) if len(interior_hsv) else 0.0

        return {
            "center": (cx, cy),
            "area": float(area),
            "angle": float(cv2.minAreaRect(contour)[2]),
            "vertices": len(approx),
            "circularity": float(circularity),
            "shape": self._o46_shape_type(contour),
            "is_outline": is_outline,
            "color_dist": 0.0,
            "fill_hue": fill_hue,
            "fill_sat": fill_sat,
            "fill_val": fill_val,
            "contour": contour,
        }

    def _dedup_o46_shapes(self, shapes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for shape in sorted(shapes, key=lambda item: -item["area"]):
            if any(safe_distance(shape["center"], other["center"]) < 20.0 for other in kept):
                continue
            kept.append(shape)
        return kept

    def _detect_o46_filled_shapes(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if frame is None:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 45, 70], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes: List[Dict[str, Any]] = []
        for contour in contours:
            record = self._o46_shape_record(frame, contour, is_outline=False)
            if record is not None:
                shapes.append(record)
        return self._dedup_o46_shapes(shapes)

    def _detect_o46_outline_targets(
        self,
        frame: np.ndarray,
        divider_x: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, 180)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes: List[Dict[str, Any]] = []
        height, width = frame.shape[:2]
        right_min_x = divider_x if divider_x is not None else width / 2.0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 20 or h < 20:
                continue
            if h > 0.5 * height and w < 0.05 * width:
                continue
            record = self._o46_shape_record(frame, contour, is_outline=True)
            if record is None:
                continue
            if record["center"][0] <= right_min_x:
                continue
            shapes.append(record)
        return self._dedup_o46_shapes(shapes)

    @staticmethod
    def _o46_hue_delta(a: float, b: float) -> float:
        delta = abs(float(a) - float(b))
        return min(delta, 180.0 - delta)

    @staticmethod
    def _weighted_geometric_score(
        scores: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        total_weight = sum(float(w) for w in weights.values() if w > 0)
        if total_weight <= 0:
            return 0.0
        accum = 0.0
        for name, weight in weights.items():
            if weight <= 0:
                continue
            value = max(0.0, min(1.0, float(scores.get(name, 0.0))))
            if value <= 0.0:
                return 0.0
            accum += float(weight) * np.log(value)
        return float(np.exp(accum / total_weight))

    def _o46_alignment_score(self, final_dist: float) -> float:
        if final_dist <= self.ALIGNMENT_FULL_PX:
            return 1.0
        return max(
            0.0,
            1.0 - (final_dist - self.ALIGNMENT_FULL_PX) / (2.0 * self.ALIGNMENT_HALF_PX),
        )

    @staticmethod
    def _saturated_ratio(value: float, saturation: float) -> float:
        if saturation <= 0.0:
            return 0.0
        return min(1.0, max(0.0, float(value)) / saturation)

    def _combine_scene_score(self, final_result: float, process_score: float) -> float:
        final_result = max(0.0, min(1.0, float(final_result)))
        process_score = max(0.0, min(1.0, float(process_score)))
        return float(0.6 * final_result + 0.4 * process_score)

    def _match_o46_final_shape(
        self,
        final_filled: List[Dict[str, Any]],
        source: Dict[str, Any],
        target: Dict[str, Any],
        used_final_ids: Set[int],
    ) -> Optional[Dict[str, Any]]:
        candidates = [shape for shape in final_filled if id(shape) not in used_final_ids]
        if not candidates:
            return None
        same_shape = [shape for shape in candidates if shape["shape"] == source["shape"]]
        pool = same_shape if same_shape else candidates

        def match_cost(candidate: Dict[str, Any]) -> float:
            target_dist = safe_distance(candidate["center"], target["center"])
            hue = self._o46_hue_delta(candidate["fill_hue"], source["fill_hue"])
            sat = abs(float(candidate["fill_sat"]) - float(source["fill_sat"]))
            area = abs(float(candidate["area"]) - float(source["area"])) / max(
                float(candidate["area"]), float(source["area"]), 1.0,
            )
            shape = 0.0 if candidate["shape"] == source["shape"] else 250.0
            return target_dist + 1.5 * hue + 0.15 * sat + 60.0 * area + shape

        return min(pool, key=match_cost)

    def _reference_pairs(
        self, gt_first_frame: np.ndarray,
    ) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], float]:
        helper = self._shape_helper()
        divider_x = gt_first_frame.shape[1] / 2.0
        filled = [
            shape for shape in self._detect_o46_filled_shapes(gt_first_frame)
            if shape["center"][0] < divider_x
        ]
        outlines = self._detect_o46_outline_targets(gt_first_frame, divider_x)

        return helper._pair_filled_with_targets(filled, outlines), divider_x

    @staticmethod
    def _first_last_detected(
        trace: Sequence[Optional[Dict[str, float]]],
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        first = next((point for point in trace if point is not None), None)
        last = next((point for point in reversed(trace) if point is not None), None)
        return first, last

    def _score_final_layout(
        self,
        video_frames: List[np.ndarray],
        pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
        divider_x: float,
    ) -> Tuple[float, float, Dict[str, Any], List[List[Optional[Dict[str, float]]]]]:
        helper = self._shape_helper()
        final_frame = video_frames[-1]
        final_filled = self._detect_o46_filled_shapes(final_frame)

        source_area_total = sum(source["area"] for source, _ in pairs)
        left_area_total = sum(
            shape["area"] for shape in final_filled if shape["center"][0] < divider_x
        )
        left_clear = max(0.0, 1.0 - left_area_total / max(source_area_total, 1.0))
        count_penalty = min(len(final_filled), len(pairs)) / max(
            len(final_filled), len(pairs), 1,
        )

        traces: List[List[Optional[Dict[str, float]]]] = []
        per_shape: List[Dict[str, Any]] = []
        final_pose_scores: List[float] = []
        process_scores: List[float] = []
        used_final_ids: Set[int] = set()

        for source, target in pairs:
            trace = helper._track_filled_shape(video_frames, source)
            traces.append(trace)
            _, traced_last = self._first_last_detected(trace)
            coverage = sum(point is not None for point in trace) / max(len(trace), 1)

            final_shape = self._match_o46_final_shape(
                final_filled, source, target, used_final_ids,
            )
            if final_shape is not None:
                used_final_ids.add(id(final_shape))
            else:
                final_shape = traced_last

            if final_shape is None:
                final_pose_scores.append(0.0)
                process_scores.append(0.0)
                per_shape.append({
                    "shape": source["shape"],
                    "alignment": 0.0,
                    "final_pose": 0.0,
                    "travel": 0.0,
                    "process_score": 0.0,
                    "area_ratio": 0.0,
                    "area_score": 0.0,
                    "coverage": round(float(coverage), 4),
                    "coverage_score": round(float(
                        self._saturated_ratio(coverage, self.COVERAGE_SATURATION),
                    ), 4),
                    "final_distance_px": None,
                    "final_center": None,
                    "target_center": [
                        round(float(target["center"][0]), 2),
                        round(float(target["center"][1]), 2),
                    ],
                })
                continue

            final_dist = safe_distance(final_shape["center"], target["center"])
            alignment = self._o46_alignment_score(final_dist)
            expected_distance = safe_distance(source["center"], target["center"])
            travel = min(
                1.0,
                safe_distance(final_shape["center"], source["center"]) / max(expected_distance, 1.0),
            )
            area_ratio = min(final_shape["area"], source["area"]) / max(
                final_shape["area"], source["area"], 1.0,
            )
            area_score = self._saturated_ratio(area_ratio, self.AREA_RATIO_SATURATION)
            coverage_score = self._saturated_ratio(coverage, self.COVERAGE_SATURATION)
            final_pose = alignment * area_score
            process_score_i = self._weighted_geometric_score(
                {
                    "travel": travel,
                    "coverage": coverage_score,
                },
                self.SHAPE_PROCESS_WEIGHTS,
            )
            final_pose_scores.append(final_pose)
            process_scores.append(process_score_i)
            per_shape.append({
                "shape": source["shape"],
                "alignment": round(float(alignment), 4),
                "final_pose": round(float(final_pose), 4),
                "travel": round(float(travel), 4),
                "process_score": round(float(process_score_i), 4),
                "area_ratio": round(float(area_ratio), 4),
                "area_score": round(float(area_score), 4),
                "coverage": round(float(coverage), 4),
                "coverage_score": round(float(coverage_score), 4),
                "final_distance_px": round(float(final_dist), 2),
                "final_center": [
                    round(float(final_shape["center"][0]), 2),
                    round(float(final_shape["center"][1]), 2),
                ],
                "target_center": [
                    round(float(target["center"][0]), 2),
                    round(float(target["center"][1]), 2),
                ],
            })

        final_pose_score = float(np.mean(final_pose_scores)) if final_pose_scores else 0.0
        transport_process = float(np.mean(process_scores)) if process_scores else 0.0
        final_layout = final_pose_score * left_clear * count_penalty
        return final_layout, transport_process, {
            "alignment_score": round(float(final_pose_score), 4),
            "final_pose_score": round(float(final_pose_score), 4),
            "transport_process": round(float(transport_process), 4),
            "left_clear": round(float(left_clear), 4),
            "count_penalty": round(float(count_penalty), 4),
            "final_filled_count": len(final_filled),
            "expected_count": len(pairs),
            "per_shape": per_shape,
        }, traces

    def _one_at_a_time_score(
        self,
        traces: Sequence[Sequence[Optional[Dict[str, float]]]],
    ) -> Tuple[float, Dict[str, Any]]:
        if not traces:
            return 0.0, {
                "active_frames": 0,
                "simultaneous_frames": 0,
                "peak_movers": 0,
            }

        frame_count = max(len(trace) for trace in traces)
        moving_counts: List[int] = []
        for frame_idx in range(1, frame_count):
            movers = 0
            for trace in traces:
                if frame_idx >= len(trace):
                    continue
                prev = trace[frame_idx - 1]
                curr = trace[frame_idx]
                if prev is None or curr is None:
                    continue
                if safe_distance(prev["center"], curr["center"]) > self.STEP_MOVE_TOL_PX:
                    movers += 1
            moving_counts.append(movers)

        active_frames = sum(count > 0 for count in moving_counts)
        simultaneous_frames = sum(count > 1 for count in moving_counts)
        peak_movers = max(moving_counts) if moving_counts else 0
        if active_frames == 0:
            return 0.0, {
                "active_frames": 0,
                "simultaneous_frames": simultaneous_frames,
                "peak_movers": peak_movers,
            }

        multi_rate = simultaneous_frames / active_frames
        excess_multi_rate = max(
            0.0,
            (multi_rate - self.SIMULTANEOUS_RATE_TOL) / max(1.0 - self.SIMULTANEOUS_RATE_TOL, 1e-6),
        )
        peak_penalty = (
            1.0
            if multi_rate <= self.SIMULTANEOUS_RATE_TOL
            else 0.8 ** max(0, peak_movers - 1)
        )
        score = peak_penalty * max(0.25, 1.0 - excess_multi_rate)
        return score, {
            "active_frames": active_frames,
            "simultaneous_frames": simultaneous_frames,
            "peak_movers": peak_movers,
            "multi_rate": round(float(multi_rate), 4),
            "simultaneous_rate_tolerance": round(float(self.SIMULTANEOUS_RATE_TOL), 4),
            "excess_multi_rate": round(float(excess_multi_rate), 4),
            "score": round(float(score), 4),
        }

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        del gt_frames, gt_final_frame, eval_info
        if len(video_frames) < 2 or gt_first_frame is None:
            self._last_task_details = {"error": "missing_frames"}
            return 0.0

        pairs, divider_x = self._reference_pairs(gt_first_frame)
        if not pairs:
            self._last_task_details = {"error": "reference_detection_failed"}
            return 0.0

        final_layout, transport_process, layout_details, traces = self._score_final_layout(
            video_frames,
            pairs,
            divider_x,
        )
        one_at_a_time, timing_details = self._one_at_a_time_score(traces)
        process_score = self._weighted_geometric_score(
            {
                "transport_process": transport_process,
                "one_at_a_time": one_at_a_time,
            },
            self.SCENE_PROCESS_WEIGHTS,
        )
        total = self._combine_scene_score(
            final_layout, process_score * min(1.0, final_layout / 0.2)
        )

        self._last_task_details = {
            "final_layout": round(float(final_layout), 4),
            "transport_process": round(float(transport_process), 4),
            "one_at_a_time": round(float(one_at_a_time), 4),
            "process_score": round(float(process_score), 4),
            "score_formula": "multiplicative_final_layout_and_geometric_process",
            "shape_process_weights": self.SHAPE_PROCESS_WEIGHTS,
            "scene_process_weights": self.SCENE_PROCESS_WEIGHTS,
            "score": round(float(total), 4),
            "layout": layout_details,
            "timing": timing_details,
        }
        return total


class SymmetryCompletionEvaluator(BaseEvaluator):
    """
    O-49: Symmetry completion - fill in missing cells to complete symmetry.

    Evaluation:
    1. Generated cells correctness (60%): new cells match GT
    2. Existing cells preservation (20%): original cells preserved
    3. Background preservation (20%): background stays clean
    """

    TASK_WEIGHTS = {
        'completion': 0.70,
        'consistency': 0.30,
    }

    def _detect_grid_masks(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Detect grid components using flood fill from corners.
        Returns: (grid_area, filled_mask, blank_mask, outer_bg)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        gray_copy = gray.copy()
        cv2.floodFill(gray_copy, mask, (0, 0), 128, loDiff=15, upDiff=15)
        outer_bg = (gray_copy == 128).astype(np.uint8) * 255
        grid_area = cv2.bitwise_not(outer_bg)
        sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        filled_mask = cv2.bitwise_and(
            grid_area, (((gray < 100) | (sat > 60)).astype(np.uint8)) * 255)
        blank_mask = cv2.bitwise_and(grid_area, (gray > 230).astype(np.uint8) * 255)
        return grid_area, filled_mask, blank_mask, outer_bg

    def _detect_fg_mask(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        corners = [frame[2, 2], frame[2, w-3], frame[h-3, 2], frame[h-3, w-3]]
        bg_color = np.mean(corners, axis=0)
        diff = np.sqrt(np.sum((frame.astype(float) - bg_color.astype(float)) ** 2, axis=2))
        binary = (diff > 30).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    def _detect_changed_region(self, gt_first: np.ndarray, gt_last: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(gt_first, gt_last)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    def _pixel_diff_score(self, frame1: np.ndarray, frame2: np.ndarray,
                          mask: np.ndarray,
                          thresholds: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)) -> Tuple[float, Dict]:
        mask_pixels = int((mask > 0).sum())
        if mask_pixels == 0:
            return 1.0, {'ratio': 0.0, 'changed_px': 0, 'total_px': 0}
        diff = cv2.absdiff(frame1, frame2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed = int((gray_diff[mask > 0] > 20).sum())
        ratio = float(changed) / mask_pixels
        t1, t2, t3, t4 = thresholds
        if ratio < t1:
            score = 1.0
        elif ratio < t2:
            score = 1.0 - (ratio - t1) / (t2 - t1) * 0.3
        elif ratio < t3:
            score = 0.7 - (ratio - t2) / (t3 - t2) * 0.4
        elif ratio < t4:
            score = 0.3 - (ratio - t3) / (t4 - t3) * 0.3
        else:
            score = 0.0
        return score, {'ratio': round(ratio, 6), 'changed_px': changed, 'total_px': mask_pixels}

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        gen_first = video_frames[0]
        gen_last = video_frames[-1]
        gt_first = gt_first_frame
        gt_last = gt_final_frame

        if gen_first.shape != gt_first.shape:
            gen_first = normalize_frame_size(gen_first, gt_first)
        if gen_last.shape != gt_last.shape:
            gen_last = normalize_frame_size(gen_last, gt_last)

        # Detect grid components from GT last using flood fill
        grid_area, gt_filled, gt_blank, outer_bg = self._detect_grid_masks(gt_last)
        # New cells = filled in GT last but not in GT first
        _, gt_first_filled, _, _ = self._detect_grid_masks(gt_first)
        kernel = np.ones((5, 5), np.uint8)
        new_cells_mask = cv2.bitwise_and(gt_filled, cv2.bitwise_not(cv2.dilate(gt_first_filled, kernel, iterations=1)))
        new_cells_mask = cv2.morphologyEx(new_cells_mask, cv2.MORPH_OPEN, kernel)
        # Existing filled cells = filled in both GT first and GT last
        existing_mask = cv2.bitwise_and(gt_filled, gt_first_filled)

        # 1. Generated cells (40%): new filled cells match GT
        new_score, new_details = self._pixel_diff_score(
            gt_last, gen_last, cv2.erode(new_cells_mask, kernel, iterations=1),
            thresholds=(0.1, 0.2, 0.30, 0.50))

        # 2. Blank preservation (30%): blank cells in GT should stay blank in gen
        blank_score, blank_details = self._pixel_diff_score(
            gt_last, gen_last, gt_blank, thresholds=(0.1, 0.2, 0.30, 0.50))

        # 3. Existing cells preservation (20%): already filled cells match GT
        existing_score, existing_details = self._pixel_diff_score(
            gt_last, gen_last, cv2.erode(existing_mask, kernel, iterations=1),
            thresholds=(0.1, 0.2, 0.30, 0.50))

        # 4. Background preservation (10%): outer background unchanged (erode to avoid grid border)
        outer_bg_eroded = cv2.erode(outer_bg, kernel, iterations=2)
        bg_score, bg_details = self._pixel_diff_score(
            gt_last, gen_last, outer_bg_eroded, thresholds=(0.015, 0.025, 0.05, 0.10))


        scores = {
            'completion': round(new_score * blank_score, 4),
            'consistency': round((existing_score + bg_score) / 2, 4),
        }
        self._last_task_details = {
            **scores,
            'generated_cells': round(new_score, 4),
            'blank_preservation': round(blank_score, 4),
            'existing_preservation': round(existing_score, 4),
            'background_preservation': round(bg_score, 4),
            **{f'new_{k}': v for k, v in new_details.items()},
            **{f'blank_{k}': v for k, v in blank_details.items()},
            **{f'exist_{k}': v for k, v in existing_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }
        total = float((scores['completion']) * (0.6 + 0.4 * scores['consistency']))
        if bg_score < 0.2:
            total = min(total, 0.15)
        return total


# Export all Part 3 evaluators
OUT_OF_DOMAIN_50_EVALUATORS_PART4 = {
    'O-5_symbol_deletion_data-generator': SymbolDeletionEvaluator,
    'O-6_2d_geometric_transformation_data-generator': GeometricTransformationEvaluator,
    'O-9_shape_scaling_data-generator': ShapeScalingAnalogyEvaluator,
    'O-11_shape_color_then_move_data-generator': ShapeColorThenMoveEvaluator,
    'O-22_construction_stack_data-generator': ConstructionStackEvaluator,
    'O-27_move_2_object_to_2_target_data-generator': MoveObjectsToTargetEvaluator,
    'O-39_maze_data-generator': MazePathfindingEvaluator,
    'O-43_object_subtraction_data-generator': ObjectSubtractionEvaluator,
    'O-46_shape_sorter_data-generator': ShapeSorterEvaluator,
    'O-49_symmetry_completion_data-generator': SymmetryCompletionEvaluator,
}
