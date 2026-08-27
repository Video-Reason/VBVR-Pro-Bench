"""
Specific evaluators for Out-of-Domain_50 tasks (Part 1).
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Set, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, normalize_frame_size, safe_distance, \
    detect_closed_contours_by_color, match_contours, COLOR_BOUNDS, \
    score_background_similarity, score_foreground_similarity, calculate_list_length_penalty
from ..utils import CircleSelectionProcessor, threshold_score
from .utils import maze
import os
import json
import shutil

class SeparateObjectsNoSpinEvaluator(BaseEvaluator):
    """
    G-24: Separate objects (no spin).

    Each filled shape on the left must slide horizontally to its matching
    dashed/solid target outline on the right, keeping its orientation.

    Scoring is a hybrid arithmetic/geometric rubric:

        object_i = weighted_geo(
            motion_quality_i, teleport_score_i, rotation_score_i,
            weights = 0.50, 0.15, 0.10,
        )
        process_score = arithmetic_mean_i(object_i)
        non_alignment = weighted_geo(
            process_score, bg_preservation, weights = 0.75, 0.10,
        )
        score = 0.15 * alignment + 0.85 * non_alignment * alignment_gate
    """

    # Pixel thresholds (calibrated against 1024×1024 GT).
    MIN_SHAPE_AREA = 1500          # smallest enclosed region treated as a shape
    DARK_THRESHOLD = 180           # gray ≤ this = border / fill stroke pixel
    BORDER_DILATE_KERNEL = 15      # dilate dark mask to bridge dashed gaps
    BG_COLOR_DIST_TOL = 35.0       # interior within this BGR distance of page bg → outline
    BG_SAT_MIN = 25                # "non-background" saturation lower bound
    BG_VAL_MAX = 220               # "non-background" value upper bound (catches grey)
    FILL_EROSION_KERNEL = 5        # erode non-bg mask to drop dash speckles
    DEDUP_RADIUS_PX = 40.0         # merge detections whose centres are closer than this
    ALIGNMENT_FULL_PX = 5.0        # within this endpoint error, give full alignment credit
    ALIGNMENT_HALF_PX = 60.0       # distance at which alignment = 0.5
    ROT_DELTA_DEG = 35.0           # rotation-robust deviation threshold
    TELEPORT_PX_FRAC = 0.25        # frame-to-frame jump > this × min(H,W) = teleport
    MAX_ROT_VIOLATIONS = 6         # cap so one wobble run can't zero the score
    MAX_TELEPORT_EVENTS = 6
    PATH_PROGRESS_FULL_RATIO = 0.90
    PATH_OVERSHOOT_FREE_RATIO = 0.05
    PATH_OVERSHOOT_DROP_RATIO = 0.30
    PATH_BACKTRACK_FREE_RATIO = 0.10
    PATH_BACKTRACK_DROP_RATIO = 0.90
    PATH_LATERAL_FULL_PX = 10.0
    PATH_LATERAL_DROP_PX = 120.0
    INTERMEDIATE_PROGRESS_MIN = 0.25
    INTERMEDIATE_PROGRESS_MAX = 0.75
    INTERMEDIATE_PROGRESS_EPS = 0.01
    INTERMEDIATE_TERMINAL_SEPARATION_RATIO = 0.08
    SHAPE_TRANSLATION_IOU_FULL = 0.80
    SHAPE_TRANSLATION_IOU_DROP = 0.45
    BG_MASK_DILATE_PX = 12
    BG_NOISE_FLOOR = 5.0
    BG_SATURATION_THRESHOLD = 0.98
    BG_BAD_PIXEL_THRESHOLD = 20.0
    BG_BAD_PIXEL_TOLERANCE = 0.10
    SCORE_WEIGHTS = {
        "alignment": 0.40,
        "motion_quality": 0.25,
        "teleport_score": 0.15,
        "rotation_score": 0.10,
        "bg_preservation": 0.10,
    }
    ALIGNMENT_GATE_FLOOR = 0.25

    SYMMETRY_PERIOD = {
        "circle": 0.0,
        "triangle": 60.0,
        "square": 0.0,
        "rectangle": 180.0,
        "pentagon": 36.0,
        "hexagon": 0.0,
        "polygon": 180.0,
    }

    # ------------------------------------------------------------------
    # Shape detection
    # ------------------------------------------------------------------

    def _detect_shapes(self, frame: np.ndarray) -> List[Dict]:
        """Return all shapes in the frame (both filled and outline targets).
        """
        if frame is None:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bg_color = self._estimate_bg_color(frame)

        candidates: List[Dict] = []
        candidates.extend(self._detect_enclosed_white_regions(frame, gray, hsv, bg_color))
        candidates.extend(self._detect_filled_blobs(frame, hsv, bg_color))

        return self._dedup_candidates(candidates)

    @staticmethod
    def _estimate_bg_color(frame: np.ndarray) -> np.ndarray:
        """Median of a small corner patch — the page background."""
        patch = frame[5:30, 5:30].reshape(-1, 3)
        return np.median(patch, axis=0)

    def _detect_enclosed_white_regions(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        hsv: np.ndarray,
        bg_color: np.ndarray,
    ) -> List[Dict]:
        """Find shapes whose interior is isolated from the page background.

        Works on every outline style observed in the samples (dashed, thin
        solid) and also catches bright pastel fills (yellow, cream) whose
        interior survives the dark-pixel threshold.
        """
        dark = cv2.inRange(gray, 0, self.DARK_THRESHOLD)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.BORDER_DILATE_KERNEL, self.BORDER_DILATE_KERNEL),
        )
        dark_dilated = cv2.dilate(dark, kernel)
        white = cv2.bitwise_not(dark_dilated)
        n_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(
            white, connectivity=8,
        )
        if n_comp <= 1:
            return []

        bg_idx = int(
            max(range(1, n_comp), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        )

        results: List[Dict] = []
        for i in range(1, n_comp):
            if i == bg_idx:
                continue
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.MIN_SHAPE_AREA:
                continue

            region_mask = (labels == i).astype(np.uint8) * 255
            interior_mask = cv2.erode(region_mask, np.ones((7, 7), np.uint8))
            if int(interior_mask.sum()) < 100:
                continue

            cx, cy = centroids[i]
            interior_bgr = frame[interior_mask > 0].reshape(-1, 3)
            median_bgr = np.median(interior_bgr, axis=0)
            color_dist = float(np.sqrt(((median_bgr - bg_color) ** 2).sum()))
            is_outline = color_dist < self.BG_COLOR_DIST_TOL

            interior_hsv = hsv[interior_mask > 0]
            fill_hue = float(np.median(interior_hsv[:, 0])) if len(interior_hsv) else 0.0
            fill_sat = float(np.median(interior_hsv[:, 1])) if len(interior_hsv) else 0.0
            fill_val = float(np.median(interior_hsv[:, 2])) if len(interior_hsv) else 0.0

            results.append(self._shape_record(
                region_mask,
                area,
                (float(cx), float(cy)),
                is_outline,
                color_dist,
                fill_hue, fill_sat, fill_val,
            ))
        return results

    def _detect_filled_blobs(
        self, frame: np.ndarray, hsv: np.ndarray, bg_color: np.ndarray,
    ) -> List[Dict]:
        """Catch every filled shape regardless of saturation.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sat_mask = cv2.inRange(
            hsv,
            np.array([0, self.BG_SAT_MIN, 0], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        dark_mask = cv2.inRange(gray, 0, self.BG_VAL_MAX)
        non_bg = cv2.bitwise_or(sat_mask, dark_mask)
        non_bg = cv2.erode(
            non_bg,
            np.ones((self.FILL_EROSION_KERNEL, self.FILL_EROSION_KERNEL), np.uint8),
        )
        contours, _ = cv2.findContours(
            non_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        results: List[Dict] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_SHAPE_AREA:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            interior_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(interior_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            interior_mask = cv2.erode(interior_mask, np.ones((7, 7), np.uint8))
            if int(interior_mask.sum()) == 0:
                continue
            interior_bgr = frame[interior_mask > 0].reshape(-1, 3)
            median_bgr = np.median(interior_bgr, axis=0)
            color_dist = float(np.sqrt(((median_bgr - bg_color) ** 2).sum()))

            interior_hsv = hsv[interior_mask > 0]
            fill_hue = float(np.median(interior_hsv[:, 0])) if len(interior_hsv) else 0.0
            fill_sat = float(np.median(interior_hsv[:, 1])) if len(interior_hsv) else 0.0
            fill_val = float(np.median(interior_hsv[:, 2])) if len(interior_hsv) else 0.0

            results.append(self._shape_record(
                interior_mask,
                int(area),
                (cx, cy),
                is_outline=False,
                color_dist=color_dist,
                fill_hue=fill_hue,
                fill_sat=fill_sat,
                fill_val=fill_val,
            ))
        return results

    def _shape_record(
        self,
        region_mask: np.ndarray,
        area: int,
        center: Tuple[float, float],
        is_outline: bool,
        color_dist: float,
        fill_hue: float,
        fill_sat: float,
        fill_val: float,
    ) -> Dict[str, Any]:
        """Build the per-shape record from a filled region mask."""
        contours, _ = cv2.findContours(
            region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cnt = max(contours, key=cv2.contourArea) if contours else None
        perim = cv2.arcLength(cnt, True) if cnt is not None else 0.0
        circularity = (
            4 * np.pi * area / (perim * perim) if perim > 0 else 0.0
        )
        approx = (
            cv2.approxPolyDP(cnt, 0.04 * perim, True) if cnt is not None else []
        )
        vertices = len(approx)
        shape_type = self._classify_shape(vertices, circularity)
        rect_angle = float(cv2.minAreaRect(cnt)[2]) if cnt is not None else 0.0

        return {
            "center": center,
            "area": float(area),
            "angle": rect_angle,
            "vertices": vertices,
            "circularity": float(circularity),
            "shape": shape_type,
            "is_outline": is_outline,
            "color_dist": color_dist,
            "fill_hue": fill_hue,
            "fill_sat": fill_sat,
            "fill_val": fill_val,
            "contour": cnt,
        }

    def _dedup_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Merge candidates whose centres fall within ``DEDUP_RADIUS_PX``.
        """
        kept: List[Dict] = []
        for cand in sorted(candidates, key=lambda s: -s["area"]):
            merged = False
            for k in kept:
                if safe_distance(cand["center"], k["center"]) < self.DEDUP_RADIUS_PX:
                    if not cand["is_outline"]:
                        k["is_outline"] = False
                        k["fill_hue"] = cand["fill_hue"]
                        k["fill_sat"] = cand["fill_sat"]
                        k["fill_val"] = cand["fill_val"]
                    merged = True
                    break
            if not merged:
                kept.append(cand)
        return kept

    @staticmethod
    def _classify_shape(vertices: int, circularity: float) -> str:
        """Vertex-first classification with circularity to separate hex/circle.
        """
        if vertices == 3:
            return "triangle"
        if vertices == 4:
            return "square"
        if circularity >= 0.85:
            return "circle"
        if vertices in (5, 6):
            return "hexagon"
        if vertices == 7:
            return "hexagon"
        return "polygon"

    def _pair_filled_with_targets(
        self, filled: List[Dict], outlines: List[Dict],
    ) -> List[Tuple[Dict, Dict]]:
        """Greedy pairing by shape type, breaking ties by nearest y-coordinate.

        Each filled shape is matched to an outline of the same classified
        type when possible; otherwise the nearest-y outline of any type is
        used as a fallback.  Every outline is consumed at most once.
        """
        pairs: List[Tuple[Dict, Dict]] = []
        remaining = list(outlines)
        for fs in filled:
            same_type = [o for o in remaining if o["shape"] == fs["shape"]]
            pool = same_type if same_type else remaining
            if not pool:
                continue
            best = min(pool, key=lambda o: abs(o["center"][1] - fs["center"][1]))
            pairs.append((fs, best))
            remaining.remove(best)
        return pairs

    def _symmetry_period(self, shape_type: str) -> float:
        return self.SYMMETRY_PERIOD.get(shape_type, 180.0)

    def _angle_delta(self, a: float, b: float, period: float) -> float:
        """Smallest angular distance between ``a`` and ``b`` modulo ``period``.

        ``period == 0`` means the shape has no stable orientation (e.g.
        circles); we return 0 so rotation checks are a no-op.
        """
        if period <= 0:
            return 0.0
        d = abs(a - b) % period
        return min(d, period - d)

    def _track_filled_shape(
        self, video_frames: List[np.ndarray], source: Dict,
    ) -> List[Optional[Dict]]:
        """Track one filled shape across ``video_frames`` by colour + position.
        """
        hue0, sat0, val0 = source["fill_hue"], source["fill_sat"], source["fill_val"]
        area0 = source["area"]
        last_center = tuple(source["center"])

        trace: List[Optional[Dict]] = []
        for frame in video_frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if sat0 >= 40:
                # Saturated fill — hue window is reliable.
                hue_lo = int(hue0 - 15)
                hue_hi = int(hue0 + 15)
                sat_min = int(max(30, sat0 * 0.5))
                val_min = int(max(40, val0 * 0.4))
                if hue_lo < 0:
                    mask = cv2.bitwise_or(
                        cv2.inRange(
                            hsv,
                            np.array([0, sat_min, val_min], dtype=np.uint8),
                            np.array([hue_hi, 255, 255], dtype=np.uint8),
                        ),
                        cv2.inRange(
                            hsv,
                            np.array([180 + hue_lo, sat_min, val_min], dtype=np.uint8),
                            np.array([179, 255, 255], dtype=np.uint8),
                        ),
                    )
                elif hue_hi > 179:
                    mask = cv2.bitwise_or(
                        cv2.inRange(
                            hsv,
                            np.array([hue_lo, sat_min, val_min], dtype=np.uint8),
                            np.array([179, 255, 255], dtype=np.uint8),
                        ),
                        cv2.inRange(
                            hsv,
                            np.array([0, sat_min, val_min], dtype=np.uint8),
                            np.array([hue_hi - 180, 255, 255], dtype=np.uint8),
                        ),
                    )
                else:
                    mask = cv2.inRange(
                        hsv,
                        np.array([hue_lo, sat_min, val_min], dtype=np.uint8),
                        np.array([hue_hi, 255, 255], dtype=np.uint8),
                    )
            else:
                v_lo = int(max(30, val0 - 50))
                v_hi = int(min(255, val0 + 50))
                s_hi = int(max(40, sat0 + 15))
                mask = cv2.inRange(
                    hsv,
                    np.array([0, 0, v_lo], np.uint8),
                    np.array([179, s_hi, v_hi], np.uint8),
                )
                bg = cv2.inRange(
                    hsv,
                    np.array([0, 0, 240], np.uint8),
                    np.array([179, 15, 255], np.uint8),
                )
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(bg))

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            candidates = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < max(self.MIN_SHAPE_AREA * 0.5, 500):
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx_c = M["m10"] / M["m00"]
                cy_c = M["m01"] / M["m00"]
                pos_dist = safe_distance((cx_c, cy_c), last_center)
                area_penalty = abs(area - area0) / max(area0, 1.0)
                # Position dominates (shapes translate slowly); area is a
                # tiebreaker when two candidates are similarly close.
                candidates.append((pos_dist + 100.0 * area_penalty, c))
            best_cnt = (
                min(candidates, key=lambda t: t[0])[1] if candidates else None
            )
            if best_cnt is None:
                trace.append(None)
                continue
            M = cv2.moments(best_cnt)
            if M["m00"] == 0:
                trace.append(None)
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            rect_angle = float(cv2.minAreaRect(best_cnt)[2])
            trace.append({
                "center": (cx, cy),
                "angle": rect_angle,
                "area": float(cv2.contourArea(best_cnt)),
                "contour": best_cnt,
            })
            last_center = (cx, cy)
        return trace

    def _alignment_score(self, final_dist: float) -> float:
        if final_dist <= self.ALIGNMENT_FULL_PX:
            return 1.0
        return max(
            0.0,
            1.0
            - (final_dist - self.ALIGNMENT_FULL_PX)
            / (2.0 * self.ALIGNMENT_HALF_PX),
        )

    def _weighted_geometric_score(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        total_weight = sum(float(w) for w in weights.values() if w > 0)
        if total_weight <= 0:
            return 0.0
        log_sum = 0.0
        for name, weight in weights.items():
            if weight <= 0:
                continue
            score = max(0.0, min(1.0, float(scores.get(name, 0.0))))
            if score <= 0.0:
                return 0.0
            log_sum += (float(weight) / total_weight) * float(np.log(score))
        return float(np.exp(log_sum))

    def _alignment_gate(self, alignment: float) -> float:
        alignment = max(0.0, min(1.0, float(alignment)))
        return self.ALIGNMENT_GATE_FLOOR + (1.0 - self.ALIGNMENT_GATE_FLOOR) * alignment

    def _combine_scene_score(self, alignment: float, non_alignment_score: float) -> float:
        """Combine final alignment with process/background quality."""
        alignment = max(0.0, min(1.0, float(alignment)))
        non_alignment_score = max(0.0, min(1.0, float(non_alignment_score)))
        alignment_weight = self.SCORE_WEIGHTS["alignment"]
        gated_non_alignment = non_alignment_score * self._alignment_gate(alignment)
        return float(max(
            0.0,
            min(
                1.0,
                alignment_weight * alignment
                + (1.0 - alignment_weight) * gated_non_alignment,
            ),
        ))

    def _translated_contour_iou(
        self,
        source: Dict[str, Any],
        point: Dict[str, Any],
        frame_shape: Tuple[int, int, int],
    ) -> Optional[float]:
        source_contour = source.get("contour")
        point_contour = point.get("contour")
        if source_contour is None or point_contour is None:
            return None

        delta = (
            np.array(point["center"], dtype=float)
            - np.array(source["center"], dtype=float)
        )
        moved = source_contour.astype(np.float32) + delta.reshape(1, 1, 2)
        moved = np.round(moved).astype(np.int32)
        current = point_contour.astype(np.int32)

        ax, ay, aw, ah = cv2.boundingRect(moved)
        bx, by, bw, bh = cv2.boundingRect(current)
        x1 = max(0, min(ax, bx) - 5)
        y1 = max(0, min(ay, by) - 5)
        x2 = min(frame_shape[1], max(ax + aw, bx + bw) + 5)
        y2 = min(frame_shape[0], max(ay + ah, by + bh) + 5)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        moved_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        current_mask = np.zeros_like(moved_mask)
        offset = np.array([[[x1, y1]]], dtype=np.int32)
        cv2.drawContours(moved_mask, [moved - offset], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(current_mask, [current - offset], -1, 255, thickness=cv2.FILLED)

        inter = int(np.logical_and(moved_mask > 0, current_mask > 0).sum())
        union = int(np.logical_or(moved_mask > 0, current_mask > 0).sum())
        return float(inter / union) if union > 0 else 0.0

    def _trace_motion_quality(
        self,
        trace: List[Optional[Dict[str, Any]]],
        source: Dict[str, Any],
        target: Dict[str, Any],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[float, Dict[str, Any]]:
        """Score G-24 process quality: horizontal path + rigid no-spin shape."""
        expected_disp = float(safe_distance(source["center"], target["center"]))
        check_shape_iou = self._symmetry_period(source["shape"]) > 0
        details = {
            "used": False,
            "expected_disp_px": round(expected_disp, 4),
            "max_projected_progress_px": 0.0,
            "progress_ratio": 0.0,
            "progress_score": 0.0,
            "overshoot_px": 0.0,
            "overshoot_ratio": 0.0,
            "overshoot_score": 1.0,
            "positive_projected_px": 0.0,
            "negative_projected_px": 0.0,
            "forward_ratio": 1.0,
            "backtrack_ratio": 0.0,
            "forward_score": 1.0,
            "lateral_p90_px": 0.0,
            "lateral_max_px": 0.0,
            "lateral_score": 1.0,
            "shape_translation_iou_p10": None,
            "shape_score": 1.0,
            "shape_check": (
                "contour_translation_iou"
                if check_shape_iou else "skipped_symmetric_shape"
            ),
            "motion_quality": 1.0,
        }
        if expected_disp <= 1e-6:
            return 1.0, details

        source_center = np.array(source["center"], dtype=float)
        target_center = np.array(target["center"], dtype=float)
        unit = (target_center - source_center) / expected_disp
        lateral_unit = np.array([-unit[1], unit[0]], dtype=float)

        prev_center = source_center
        positive = 0.0
        negative = 0.0
        max_projected_progress = 0.0
        lateral_values = [0.0]
        shape_ious: List[float] = []
        saw_detection = False
        for point in trace:
            if point is None:
                continue
            curr_center = np.array(point["center"], dtype=float)
            projected_progress = float(np.dot(curr_center - source_center, unit))
            max_projected_progress = max(max_projected_progress, projected_progress)
            projected_step = float(np.dot(curr_center - prev_center, unit))
            if projected_step >= 0:
                positive += projected_step
            else:
                negative += -projected_step
            lateral_values.append(
                abs(float(np.dot(curr_center - source_center, lateral_unit)))
            )

            if check_shape_iou:
                shape_iou = self._translated_contour_iou(source, point, frame_shape)
                if shape_iou is not None:
                    shape_ious.append(shape_iou)

            prev_center = curr_center
            saw_detection = True

        if not saw_detection:
            details.update({
                "used": True,
                "forward_ratio": 0.0,
                "forward_score": 0.0,
                "motion_quality": 0.0,
            })
            return 0.0, details

        denom = positive + negative
        progress_ratio = max(
            0.0,
            min(1.0, max_projected_progress / expected_disp),
        )
        progress_score = min(
            1.0,
            progress_ratio / max(self.PATH_PROGRESS_FULL_RATIO, 1e-6),
        )
        overshoot_px = max(0.0, max_projected_progress - expected_disp)
        overshoot_ratio = overshoot_px / expected_disp
        if overshoot_ratio <= self.PATH_OVERSHOOT_FREE_RATIO:
            overshoot_score = 1.0
        else:
            overshoot_score = max(
                0.0,
                1.0
                - (overshoot_ratio - self.PATH_OVERSHOOT_FREE_RATIO)
                / self.PATH_OVERSHOOT_DROP_RATIO,
            )
        forward_ratio = positive / denom if denom > 1e-6 else 0.0
        backtrack_ratio = negative / denom if denom > 1e-6 else 0.0
        if backtrack_ratio <= self.PATH_BACKTRACK_FREE_RATIO:
            forward_score = 1.0
        else:
            forward_score = max(
                0.0,
                1.0 - (
                    (backtrack_ratio - self.PATH_BACKTRACK_FREE_RATIO)
                    / self.PATH_BACKTRACK_DROP_RATIO
                ),
            )

        lateral_p90 = float(np.percentile(lateral_values, 90)) if lateral_values else 0.0
        lateral_max = float(max(lateral_values)) if lateral_values else 0.0
        if lateral_p90 <= self.PATH_LATERAL_FULL_PX:
            lateral_score = 1.0
        else:
            lateral_score = max(
                0.0,
                1.0
                - (lateral_p90 - self.PATH_LATERAL_FULL_PX)
                / self.PATH_LATERAL_DROP_PX,
            )

        shape_iou_p10: Optional[float] = None
        shape_score = 1.0
        if shape_ious:
            shape_iou_p10 = float(np.percentile(shape_ious, 10))
            if shape_iou_p10 < self.SHAPE_TRANSLATION_IOU_FULL:
                shape_score = max(
                    0.0,
                    (shape_iou_p10 - self.SHAPE_TRANSLATION_IOU_DROP)
                    / (self.SHAPE_TRANSLATION_IOU_FULL - self.SHAPE_TRANSLATION_IOU_DROP),
                )

        motion_quality = (
            progress_score
            * overshoot_score
            * forward_score
            * lateral_score
            * shape_score
        )
        details.update({
            "used": True,
            "max_projected_progress_px": round(float(max_projected_progress), 4),
            "progress_ratio": round(float(progress_ratio), 4),
            "progress_score": round(float(progress_score), 4),
            "overshoot_px": round(float(overshoot_px), 4),
            "overshoot_ratio": round(float(overshoot_ratio), 4),
            "overshoot_score": round(float(overshoot_score), 4),
            "positive_projected_px": round(float(positive), 4),
            "negative_projected_px": round(float(negative), 4),
            "forward_ratio": round(float(forward_ratio), 4),
            "backtrack_ratio": round(float(backtrack_ratio), 4),
            "forward_score": round(float(forward_score), 4),
            "lateral_p90_px": round(float(lateral_p90), 4),
            "lateral_max_px": round(float(lateral_max), 4),
            "lateral_score": round(float(lateral_score), 4),
            "shape_translation_iou_p10": (
                None if shape_iou_p10 is None else round(float(shape_iou_p10), 4)
            ),
            "shape_score": round(float(shape_score), 4),
            "motion_quality": round(float(motion_quality), 4),
        })
        return motion_quality, details

    def _intermediate_progress_score(
        self,
        trace: List[Optional[Dict[str, Any]]],
        source: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Require broad, visible progress without matching GT timestamps.

        A moved object only needs to appear once anywhere in the middle 25%--75%
        of its source-to-target segment.  This deliberately does not require a
        particular GT frame, speed, or exact midpoint.  The lateral allowance
        reuses the existing path tolerance so a slightly curved/noisy trace is
        accepted, while an unrelated same-colour object off the path is not.
        """
        source_center = np.array(source["center"], dtype=float)
        target_center = np.array(target["center"], dtype=float)
        expected_disp = float(np.linalg.norm(target_center - source_center))
        details: Dict[str, Any] = {
            "progress_range": [
                self.INTERMEDIATE_PROGRESS_MIN,
                self.INTERMEDIATE_PROGRESS_MAX,
            ],
            "expected_disp_px": round(expected_disp, 4),
            "observed_internal_progress": [],
            "has_intermediate_state": False,
        }
        if expected_disp <= 1e-6:
            return 1.0, details

        unit = (target_center - source_center) / expected_disp
        lateral_unit = np.array([-unit[1], unit[0]], dtype=float)
        progress_lo = self.INTERMEDIATE_PROGRESS_MIN - self.INTERMEDIATE_PROGRESS_EPS
        progress_hi = self.INTERMEDIATE_PROGRESS_MAX + self.INTERMEDIATE_PROGRESS_EPS
        observations: List[Dict[str, float]] = []
        found = False
        terminal = next((point for point in reversed(trace) if point is not None), None)
        terminal_center = (
            np.array(terminal["center"], dtype=float) if terminal is not None else None
        )
        min_terminal_separation = (
            self.INTERMEDIATE_TERMINAL_SEPARATION_RATIO * expected_disp
        )

        for point in trace[1:-1]:
            if point is None:
                continue
            center = np.array(point["center"], dtype=float)
            delta = center - source_center
            progress = float(np.dot(delta, unit) / expected_disp)
            lateral_px = abs(float(np.dot(delta, lateral_unit)))
            in_band = progress_lo <= progress <= progress_hi
            on_path = lateral_px <= self.PATH_LATERAL_DROP_PX
            terminal_separation = (
                float(np.linalg.norm(center - terminal_center))
                if terminal_center is not None else 0.0
            )
            distinct_from_terminal = terminal_separation >= min_terminal_separation
            observations.append({
                "progress": round(progress, 4),
                "lateral_px": round(lateral_px, 4),
                "terminal_separation_px": round(terminal_separation, 4),
                "accepted": bool(in_band and on_path and distinct_from_terminal),
            })
            found = found or (in_band and on_path and distinct_from_terminal)

        details["observed_internal_progress"] = observations
        details["has_intermediate_state"] = bool(found)
        return float(found), details

    def _draw_background_exclude_contour(
        self, mask: np.ndarray, contour: Optional[np.ndarray],
    ) -> None:
        if contour is None:
            return
        cv2.drawContours(
            mask,
            [contour.astype(np.int32)],
            -1,
            255,
            thickness=cv2.FILLED,
        )

    def _background_preservation_for_traces(
        self,
        video_frames: List[np.ndarray],
        reference_frame: Optional[np.ndarray],
        shape_traces: List[Tuple[Dict[str, Any], List[Optional[Dict[str, Any]]]]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Score static background preservation outside legitimate object motion.
        """
        details = {
            "frame_min": 0.0,
            "frame_mean": 0.0,
            "exclude_fraction_mean": 0.0,
            "noise_floor": self.BG_NOISE_FLOOR,
            "saturation_threshold": self.BG_SATURATION_THRESHOLD,
            "bad_pixel_threshold": self.BG_BAD_PIXEL_THRESHOLD,
            "bad_pixel_tolerance": self.BG_BAD_PIXEL_TOLERANCE,
        }
        if not video_frames or reference_frame is None:
            return 0.0, details

        H, W = reference_frame.shape[:2]
        source_mask = np.zeros((H, W), dtype=np.uint8)
        for source, _trace in shape_traces:
            self._draw_background_exclude_contour(source_mask, source.get("contour"))

        if self.BG_MASK_DILATE_PX > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.BG_MASK_DILATE_PX, self.BG_MASK_DILATE_PX),
            )
        else:
            kernel = None

        scores: List[float] = []
        exclude_fracs: List[float] = []
        for idx, frame in enumerate(video_frames):
            exclude = source_mask.copy()
            for _source, trace in shape_traces:
                if idx >= len(trace):
                    continue
                point = trace[idx]
                if point is None:
                    continue
                self._draw_background_exclude_contour(exclude, point.get("contour"))
            if kernel is not None and np.any(exclude):
                exclude = cv2.dilate(exclude, kernel)
            exclude_fracs.append(float(np.mean(exclude > 0)))
            scores.append(maze.background_preservation_image(
                frame,
                reference_frame,
                exclude_mask=exclude,
                noise_floor=self.BG_NOISE_FLOOR,
                saturation_threshold=self.BG_SATURATION_THRESHOLD,
                bad_pixel_threshold=self.BG_BAD_PIXEL_THRESHOLD,
                bad_pixel_tolerance=self.BG_BAD_PIXEL_TOLERANCE,
            ))

        if not scores:
            return 0.0, details
        score = float(np.mean(scores))
        if score >= self.BG_SATURATION_THRESHOLD:
            score = 1.0
        details.update({
            "frame_min": round(float(min(scores)), 4),
            "frame_mean": round(float(np.mean(scores)), 4),
            "exclude_fraction_mean": round(float(np.mean(exclude_fracs)), 4)
            if exclude_fracs else 0.0,
        })
        return score, details

    def _final_candidates_for_source(
        self, frame: np.ndarray, source: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return final-frame candidates compatible with one source colour.
        """
        hue0, sat0, val0 = source["fill_hue"], source["fill_sat"], source["fill_val"]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if sat0 >= 40:
            hue_lo = int(hue0 - 15)
            hue_hi = int(hue0 + 15)
            sat_min = int(max(30, sat0 * 0.5))
            val_min = int(max(40, val0 * 0.4))
            if hue_lo < 0:
                mask = cv2.bitwise_or(
                    cv2.inRange(
                        hsv,
                        np.array([0, sat_min, val_min], dtype=np.uint8),
                        np.array([hue_hi, 255, 255], dtype=np.uint8),
                    ),
                    cv2.inRange(
                        hsv,
                        np.array([180 + hue_lo, sat_min, val_min], dtype=np.uint8),
                        np.array([179, 255, 255], dtype=np.uint8),
                    ),
                )
            elif hue_hi > 179:
                mask = cv2.bitwise_or(
                    cv2.inRange(
                        hsv,
                        np.array([hue_lo, sat_min, val_min], dtype=np.uint8),
                        np.array([179, 255, 255], dtype=np.uint8),
                    ),
                    cv2.inRange(
                        hsv,
                        np.array([0, sat_min, val_min], dtype=np.uint8),
                        np.array([hue_hi - 180, 255, 255], dtype=np.uint8),
                    ),
                )
            else:
                mask = cv2.inRange(
                    hsv,
                    np.array([hue_lo, sat_min, val_min], dtype=np.uint8),
                    np.array([hue_hi, 255, 255], dtype=np.uint8),
                )
        else:
            v_lo = int(max(30, val0 - 50))
            v_hi = int(min(255, val0 + 50))
            s_hi = int(max(40, sat0 + 15))
            mask = cv2.inRange(
                hsv,
                np.array([0, 0, v_lo], np.uint8),
                np.array([179, s_hi, v_hi], np.uint8),
            )
            bg = cv2.inRange(
                hsv,
                np.array([0, 0, 240], np.uint8),
                np.array([179, 15, 255], np.uint8),
            )
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(bg))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        candidates: List[Dict[str, Any]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(self.MIN_SHAPE_AREA * 0.5, 500):
                continue
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            region_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
            interior = cv2.erode(region_mask, np.ones((7, 7), np.uint8))
            if int(interior.sum()) == 0:
                continue
            interior_hsv = hsv[interior > 0]
            fill_hue = float(np.median(interior_hsv[:, 0])) if len(interior_hsv) else 0.0
            fill_sat = float(np.median(interior_hsv[:, 1])) if len(interior_hsv) else 0.0
            fill_val = float(np.median(interior_hsv[:, 2])) if len(interior_hsv) else 0.0

            perim = cv2.arcLength(contour, True)
            circularity = (
                4 * np.pi * area / (perim * perim) if perim > 0 else 0.0
            )
            approx = cv2.approxPolyDP(contour, 0.04 * perim, True) if perim > 0 else []
            shape_type = self._classify_shape(len(approx), circularity)
            candidates.append({
                "center": (cx, cy),
                "area": float(area),
                "angle": float(cv2.minAreaRect(contour)[2]),
                "vertices": len(approx),
                "circularity": float(circularity),
                "shape": shape_type,
                "is_outline": False,
                "fill_hue": fill_hue,
                "fill_sat": fill_sat,
                "fill_val": fill_val,
                "contour": contour,
            })
        return candidates

    # ------------------------------------------------------------------
    # Task-specific entry point
    # ------------------------------------------------------------------

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        if len(video_frames) < 2 or gt_first_frame is None:
            self._last_task_details = {"error": "missing frames"}
            return 0.0

        # Reference: filled shapes + target outlines from the GT first frame.
        gt_shapes = self._detect_shapes(gt_first_frame)
        gt_filled = [s for s in gt_shapes if not s["is_outline"]]
        gt_outlines = [s for s in gt_shapes if s["is_outline"]]
        if not gt_filled or not gt_outlines:
            self._last_task_details = {
                "error": "could not detect filled+outline in gt_first_frame",
                "gt_filled": len(gt_filled),
                "gt_outlines": len(gt_outlines),
            }
            return 0.0

        pairs = self._pair_filled_with_targets(gt_filled, gt_outlines)
        if not pairs:
            self._last_task_details = {"error": "no filled→outline pairs"}
            return 0.0

        H, W = video_frames[0].shape[:2]
        teleport_px = self.TELEPORT_PX_FRAC * min(H, W)

        per_shape: List[Dict[str, Any]] = []
        alignments: List[float] = []
        horizontals: List[float] = []
        motion_qualities: List[float] = []
        intermediate_scores: List[float] = []
        process_scores: List[float] = []
        rotation_scores: List[float] = []
        teleport_scores: List[float] = []
        shape_traces: List[Tuple[Dict[str, Any], List[Optional[Dict[str, Any]]]]] = []
        total_rot_violations = 0
        total_teleport_events = 0

        for filled, target in pairs:
            trace = self._track_filled_shape(video_frames, filled)
            shape_traces.append((filled, trace))
            period = self._symmetry_period(filled["shape"])

            # ---- alignment: last detected centre vs target centre ----
            last = next((p for p in reversed(trace) if p is not None), None)
            if last is None:
                align = 0.0
                final_dist = float("inf")
            else:
                final_dist = safe_distance(last["center"], target["center"])
                align = self._alignment_score(final_dist)
            alignments.append(align)

            # ---- horizontal-motion ratio ----
            first = next((p for p in trace if p is not None), None)
            if first is None or last is None or first is last:
                horizontals.append(0.0)
            else:
                dx = abs(last["center"][0] - first["center"][0])
                dy = abs(last["center"][1] - first["center"][1])
                denom = dx + dy
                horizontals.append(dx / denom if denom > 1.0 else 0.0)

            motion_quality, motion_details = self._trace_motion_quality(
                trace, filled, target, video_frames[0].shape,
            )
            motion_qualities.append(motion_quality)
            intermediate_score, intermediate_details = self._intermediate_progress_score(
                trace, filled, target,
            )
            intermediate_scores.append(intermediate_score)

            # ---- teleport count (per-frame position deltas) ----
            teleport_events = 0
            prev = None
            for p in trace:
                if p is None:
                    continue
                if prev is not None:
                    step_px = safe_distance(p["center"], prev["center"])
                    if step_px > teleport_px:
                        teleport_events += 1
                prev = p

            # ---- rotation: 90th-percentile deviation from the trace median ----
            rot_events = 0
            if period > 0:
                angles = [p["angle"] for p in trace if p is not None]
                if angles:
                    median_angle = float(np.median(angles))
                    deviations = [
                        self._angle_delta(a, median_angle, period)
                        for a in angles
                    ]
                    robust_dev = float(np.percentile(deviations, 90))
                    if robust_dev > self.ROT_DELTA_DEG:
                        rot_events = 1
            total_rot_violations += rot_events
            total_teleport_events += teleport_events
            shape_rotation_score = 0.5 ** min(rot_events, self.MAX_ROT_VIOLATIONS)
            shape_teleport_score = 0.5 ** min(
                teleport_events, self.MAX_TELEPORT_EVENTS,
            )
            shape_process_scores = {
                "motion_quality": motion_quality,
                "teleport_score": shape_teleport_score,
                "rotation_score": shape_rotation_score,
            }
            process_weights = {
                name: weight
                for name, weight in self.SCORE_WEIGHTS.items()
                if name not in ("alignment", "bg_preservation")
            }
            process_score_i = self._weighted_geometric_score(
                shape_process_scores, process_weights,
            )
            process_scores.append(process_score_i)
            rotation_scores.append(shape_rotation_score)
            teleport_scores.append(shape_teleport_score)

            per_shape.append({
                "shape": filled["shape"],
                "target_center": target["center"],
                "final_center": last["center"] if last else None,
                "final_distance_px": round(final_dist, 2)
                if last is not None else None,
                "alignment": round(align, 4),
                "horizontal_ratio": round(horizontals[-1], 4),
                "motion_quality": round(float(motion_quality), 4),
                "motion_check": motion_details,
                "intermediate_progress": round(float(intermediate_score), 4),
                "intermediate_check": intermediate_details,
                "process_score": round(float(process_score_i), 4),
                "object_score": round(float(process_score_i), 4),
                "rotation_score": round(float(shape_rotation_score), 4),
                "teleport_score": round(float(shape_teleport_score), 4),
                "rot_events": rot_events,
                "teleport_events": teleport_events,
                "trace_coverage": sum(1 for p in trace if p is not None) / len(trace),
            })

        alignment = float(np.mean(alignments)) if alignments else 0.0
        horizontal_ratio = float(np.mean(horizontals)) if horizontals else 0.0
        motion_quality = float(np.mean(motion_qualities)) if motion_qualities else 0.0
        intermediate_coverage = (
            float(np.mean(intermediate_scores)) if intermediate_scores else 0.0
        )
        process_score = float(np.mean(process_scores)) if process_scores else 0.0
        rotation_score = float(np.mean(rotation_scores)) if rotation_scores else 0.0
        teleport_score = float(np.mean(teleport_scores)) if teleport_scores else 0.0

        rot_penalty = 0.5 ** min(total_rot_violations, self.MAX_ROT_VIOLATIONS)
        teleport_penalty = 0.5 ** min(total_teleport_events, self.MAX_TELEPORT_EVENTS)
        bg_preservation, bg_details = self._background_preservation_for_traces(
            video_frames, gt_first_frame, shape_traces,
        )

        process_weight = sum(
            weight for name, weight in self.SCORE_WEIGHTS.items()
            if name not in ("alignment", "bg_preservation")
        )
        alignment_weight = self.SCORE_WEIGHTS["alignment"]
        non_alignment_score = self._weighted_geometric_score(
            {
                "process_score": process_score,
                "bg_preservation": bg_preservation,
            },
            {
                "process_score": process_weight,
                "bg_preservation": self.SCORE_WEIGHTS["bg_preservation"],
            },
        )
        alignment_gate = self._alignment_gate(alignment)
        score = self._combine_scene_score(alignment, non_alignment_score)
        score *= intermediate_coverage

        self._last_task_details = {
            "alignment": round(alignment, 4),
            "alignment_gate": round(float(alignment_gate), 4),
            "alignment_gate_floor": self.ALIGNMENT_GATE_FLOOR,
            "horizontal_ratio": round(horizontal_ratio, 4),
            "motion_quality": round(motion_quality, 4),
            "intermediate_coverage": round(intermediate_coverage, 4),
            "process_score": round(process_score, 4),
            "object_score": round(process_score, 4),
            "rotation_violations": total_rot_violations,
            "teleport_events": total_teleport_events,
            "score_formula": (
                "intermediate_coverage_x_"
                "linear_alignment_plus_alignment_gated_geometric_process_bg"
            ),
            "component_weights": self.SCORE_WEIGHTS,
            "process_component_weights": {
                name: weight
                for name, weight in self.SCORE_WEIGHTS.items()
                if name not in ("alignment", "bg_preservation")
            },
            "final_geometric_weights": {
                "process_score": round(float(process_weight), 4),
                "bg_preservation": self.SCORE_WEIGHTS["bg_preservation"],
            },
            "alignment_weight": alignment_weight,
            "non_alignment_weight": round(float(1.0 - alignment_weight), 4),
            "non_alignment_score": round(float(non_alignment_score), 4),
            "rotation_score": round(rotation_score, 4),
            "teleport_score": round(teleport_score, 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "background_preservation": bg_details,
            "rot_penalty": round(rot_penalty, 4),
            "teleport_penalty": round(teleport_penalty, 4),
            "num_pairs": len(pairs),
            "per_shape": per_shape,
        }
        return score

    # ------------------------------------------------------------------
    # Interleave scoring
    # ------------------------------------------------------------------

    def _hue_delta(self, a: float, b: float) -> float:
        d = abs(float(a) - float(b))
        return min(d, 180.0 - d)

    def _interleave_final_match_cost(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> float:
        """Cost for matching a final-frame filled shape to a source/target pair.

        Interleave outputs can be a single final image, so colour+position
        tracking from the source location is not reliable.  The final object
        should instead be the candidate that both lands near the target and
        preserves the source identity.
        """
        target_dist = (
            safe_distance(target["center"], candidate["center"])
            / self.ALIGNMENT_HALF_PX
        )
        hue = self._hue_delta(source["fill_hue"], candidate["fill_hue"]) / 18.0
        sat = abs(float(source["fill_sat"]) - float(candidate["fill_sat"])) / 80.0
        val = abs(float(source["fill_val"]) - float(candidate["fill_val"])) / 80.0
        area = abs(float(source["area"]) - float(candidate["area"])) / max(
            float(source["area"]), float(candidate["area"]), 1.0,
        )
        shape_penalty = 0.0 if source["shape"] == candidate["shape"] else 0.75
        return (
            2.0 * target_dist
            + hue
            + 0.5 * sat
            + 0.25 * val
            + 0.5 * area
            + shape_penalty
        )

    def _interleave_rotation_event(
        self,
        seq: List[np.ndarray],
        source: Dict[str, Any],
        final_shape: Dict[str, Any],
    ) -> Tuple[int, Dict[str, Any]]:
        """Return a no-spin violation for interleave outputs.

        With enough predicted frames, reuse the video trace signal.  For a
        sparse or single-image interleave output, fall back to endpoint
        orientation: it cannot prove "no mid-way spin", but it should catch
        outputs that end rotated.
        """
        period = self._symmetry_period(source["shape"])
        if period <= 0:
            return 0, {"mode": "skipped", "angle_delta": 0.0}

        if len(seq) >= 4:
            trace = self._track_filled_shape(seq, source)
            angles = [point["angle"] for point in trace if point is not None]
            if len(angles) >= 3:
                median_angle = float(np.median(angles))
                deviations = [
                    self._angle_delta(angle, median_angle, period)
                    for angle in angles
                ]
                robust_dev = float(np.percentile(deviations, 90))
                return int(robust_dev > self.ROT_DELTA_DEG), {
                    "mode": "trace",
                    "robust_deviation": round(robust_dev, 4),
                }

        angle_delta = self._angle_delta(final_shape["angle"], source["angle"], period)
        return int(angle_delta > self.ROT_DELTA_DEG), {
            "mode": "endpoint",
            "angle_delta": round(float(angle_delta), 4),
        }

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        del gt_images, gt_final_frame, eval_info
        if not pred_images or input_frame is None:
            self._last_task_details = {"error": "missing interleave frames"}
            return 0.0

        H, W = input_frame.shape[:2]
        seq: List[np.ndarray] = [input_frame]
        for pred in pred_images:
            seq.append(cv2.resize(pred, (W, H)) if pred.shape[:2] != (H, W) else pred)

        gt_shapes = self._detect_shapes(seq[0])
        gt_filled = [shape for shape in gt_shapes if not shape["is_outline"]]
        gt_outlines = [shape for shape in gt_shapes if shape["is_outline"]]
        if not gt_filled or not gt_outlines:
            self._last_task_details = {
                "error": "could not detect filled+outline in input_frame",
                "gt_filled": len(gt_filled),
                "gt_outlines": len(gt_outlines),
            }
            return 0.0

        pairs = self._pair_filled_with_targets(gt_filled, gt_outlines)
        if not pairs:
            self._last_task_details = {"error": "no filled->outline pairs"}
            return 0.0

        final_shapes = self._detect_shapes(seq[-1])
        final_filled = [shape for shape in final_shapes if not shape["is_outline"]]

        alignments: List[float] = []
        horizontals: List[float] = []
        intermediate_scores: List[float] = []
        total_rot_violations = 0
        per_shape: List[Dict[str, Any]] = []

        for source, target in pairs:
            trace = self._track_filled_shape(seq, source)
            intermediate_score, intermediate_details = self._intermediate_progress_score(
                trace, source, target,
            )
            intermediate_scores.append(intermediate_score)
            candidates = self._final_candidates_for_source(seq[-1], source)
            final_shape = (
                min(
                    candidates,
                    key=lambda candidate: self._interleave_final_match_cost(
                        source, target, candidate,
                    ),
                )
                if candidates else None
            )
            if final_shape is None:
                alignments.append(0.0)
                horizontals.append(0.0)
                per_shape.append({
                    "shape": source["shape"],
                    "target_center": target["center"],
                    "final_center": None,
                    "final_distance_px": None,
                    "alignment": 0.0,
                    "horizontal_ratio": 0.0,
                    "intermediate_progress": round(float(intermediate_score), 4),
                    "intermediate_check": intermediate_details,
                    "rot_events": 0,
                    "rotation_check": {"mode": "missing_final"},
                    "num_final_candidates": 0,
                })
                continue

            final_dist = safe_distance(final_shape["center"], target["center"])
            alignment = self._alignment_score(final_dist)
            alignments.append(alignment)

            dx = abs(final_shape["center"][0] - source["center"][0])
            dy = abs(final_shape["center"][1] - source["center"][1])
            denom = dx + dy
            horizontal = dx / denom if denom > 1.0 else 0.0
            horizontals.append(horizontal)

            rot_events, rotation_details = self._interleave_rotation_event(
                seq, source, final_shape,
            )
            total_rot_violations += rot_events

            per_shape.append({
                "shape": source["shape"],
                "target_center": target["center"],
                "final_center": final_shape["center"],
                "final_distance_px": round(final_dist, 2),
                "alignment": round(float(alignment), 4),
                "horizontal_ratio": round(float(horizontal), 4),
                "intermediate_progress": round(float(intermediate_score), 4),
                "intermediate_check": intermediate_details,
                "rot_events": rot_events,
                "rotation_check": rotation_details,
                "num_final_candidates": len(candidates),
            })

        alignment = float(np.mean(alignments)) if alignments else 0.0
        horizontal_ratio = float(np.mean(horizontals)) if horizontals else 0.0
        intermediate_coverage = (
            float(np.mean(intermediate_scores)) if intermediate_scores else 0.0
        )
        rot_penalty = 0.5 ** min(total_rot_violations, self.MAX_ROT_VIOLATIONS)
        n_hallucinated = max(0, len(final_filled) - len(pairs))
        extra_object_penalty = len(pairs) / max(len(pairs) + n_hallucinated, 1)
        score = (
            alignment
            * horizontal_ratio
            * rot_penalty
            * extra_object_penalty
            * intermediate_coverage
        )
        score = float(max(0.0, min(1.0, score)))

        self._last_task_details = {
            "mode": "interleave_final_matching",
            "alignment": round(alignment, 4),
            "horizontal_ratio": round(horizontal_ratio, 4),
            "intermediate_coverage": round(intermediate_coverage, 4),
            "rotation_violations": total_rot_violations,
            "rot_penalty": round(rot_penalty, 4),
            "teleport_events": 0,
            "teleport_penalty": 1.0,
            "extra_object_penalty": round(float(extra_object_penalty), 4),
            "num_pairs": len(pairs),
            "n_final_filled": len(final_filled),
            "n_hallucinated": n_hallucinated,
            "per_shape": per_shape,
        }
        # Charge for frames the model never showed (no-op when it matches GT's count).
        return float(score)



class MultipleKeysForOneDoorEvaluator(BaseEvaluator):
    """
    G-47: Multi-key maze evaluator.

    Task: the green circular agent collects every coloured diamond key
    (any order, minimising total distance), then reaches the red
    hollow-square door while avoiding black walls.  Same multiplicative
    scoring family as G-15 / G-16 / G-18:

      score = proximity
            × (1 − 0.30 × continuity_penalty)
            × 0.5^num_missed_keys          # G-16-style required-cell penalty
            × 0.5^num_wall_hit_cells       # G-15-style obstacle penalty
            × coverage
            × min(1, shortest / path_length)  # back-and-forth / detour cap

    Path optimality is modelled in the combined state space
    ``(cell, visited_key_mask)`` via :func:`maze.grid_state_bfs`, so the
    agent is free to pick any visit order as long as the total tour is
    minimum length.
    """

    MAX_PENALTY = 0.70
    JITTER_TOL = 0.05
    PENALTY_FLOOR = 0.05
    EXTRA_AGENT_PENALTY = 0.20
    COVERAGE_GAP_THRESHOLD = 2
    DISAPPEAR_RATE_CAP = 1.0

    KEY_VISIT_TOLERANCE = 1

    AGENT_HSV_LOWER = (35, 100, 100)
    AGENT_HSV_UPPER = (85, 255, 255)
    BG_DYNAMIC_DIFF_THRESHOLD = 20
    BG_DYNAMIC_DILATE_KERNEL = 5
    BG_NOISE_FLOOR = 2.0
    BG_BAD_PIXEL_THRESHOLD = 20.0
    BG_BAD_PIXEL_TOLERANCE = 0.10
    _grid_size: int = 13

    # ------------------------------------------------------------------
    # Grid-structure inference
    # ------------------------------------------------------------------

    def _infer_grid_size(self, frame: np.ndarray) -> int:
        """Infer ``grid_size`` from the outer black-border thickness.

        G-47 mazes use a uniform cell side that equals the outer-wall
        thickness.  Scan several columns for the first non-black pixel
        from the top, take the median height, then
        ``grid_size = round(H / border_top)``.  Clamped to ``[8, 32]`` and
        falls back to 13 (the observed default) on ambiguous input.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        border_tops: List[int] = []
        step = max(1, w // 8)
        for x in range(step, w, step):
            for y in range(h // 2):
                if gray[y, x] > 80:
                    if y >= 5:  # ignore anti-aliasing at the edge
                        border_tops.append(y)
                    break

        if not border_tops:
            return 13

        med = int(np.median(border_tops))
        if med < 10 or med > h // 3:
            return 13

        return max(8, min(32, int(round(h / med))))

    def _detect_wall_cells(
        self, frame: np.ndarray, grid_size: int,
    ) -> Set[Tuple[int, int]]:
        """Cells whose inner core is ≥70% black are walls."""
        h, w = frame.shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        walls: Set[Tuple[int, int]] = set()
        for r in range(grid_size):
            for c in range(grid_size):
                y1 = r * cell_h + cell_h // 4
                y2 = (r + 1) * cell_h - cell_h // 4
                x1 = c * cell_w + cell_w // 4
                x2 = (c + 1) * cell_w - cell_w // 4
                if y2 <= y1 or x2 <= x1:
                    continue
                inner = gray[y1:y2, x1:x2]
                if inner.size == 0:
                    continue
                black_ratio = float(np.mean(inner < 60))
                if black_ratio > 0.7:
                    walls.add((r, c))
        return walls

    def _gt_dynamic_exclude_mask(
        self,
        gt_frames: List[np.ndarray],
        landmark_frame: np.ndarray,
        dynamic_cells: Set[Tuple[int, int]],
        grid_size: int,
    ) -> Optional[np.ndarray]:
        """Mask regions that legitimately change in the GT sequence.
        """
        if not gt_frames or not dynamic_cells:
            return None

        h, w = landmark_frame.shape[:2]
        acc = np.zeros((h, w), dtype=np.uint8)
        cell_mask = np.zeros((h, w), dtype=np.uint8)
        for r, c in dynamic_cells:
            if r < 0 or c < 0 or r >= grid_size or c >= grid_size:
                continue
            y1 = int(round(r * h / grid_size))
            y2 = int(round((r + 1) * h / grid_size))
            x1 = int(round(c * w / grid_size))
            x2 = int(round((c + 1) * w / grid_size))
            cell_mask[y1:y2, x1:x2] = 255

        ref = landmark_frame
        for frame in gt_frames:
            if frame is None:
                continue
            f = frame
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            diff = np.abs(f.astype(np.float32) - ref.astype(np.float32))
            if diff.ndim == 3:
                diff = diff.mean(axis=2)
            changed = (diff > self.BG_DYNAMIC_DIFF_THRESHOLD).astype(np.uint8) * 255
            acc = cv2.bitwise_or(acc, cv2.bitwise_and(changed, cell_mask))

        if self.BG_DYNAMIC_DILATE_KERNEL > 1 and np.any(acc):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.BG_DYNAMIC_DILATE_KERNEL, self.BG_DYNAMIC_DILATE_KERNEL),
            )
            acc = cv2.dilate(acc, kernel, iterations=1)
        return acc if np.any(acc) else None

    # ------------------------------------------------------------------
    # Landmark detection (agent / door / keys)
    # ------------------------------------------------------------------

    def _agent_area_bounds(self, frame: np.ndarray) -> Tuple[float, float]:
        cell_px = max(frame.shape[:2]) / float(self._grid_size)
        cell_area = cell_px ** 2
        return (max(100.0, cell_area * 0.08), cell_area * 2.0)

    def _detect_all_agents(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Centroids of green blobs within cell-area bounds."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, np.array(self.AGENT_HSV_LOWER), np.array(self.AGENT_HSV_UPPER),
        )
        min_area, max_area = self._agent_area_bounds(frame)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        agents: List[Tuple[int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            agents.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        return agents

    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        agents = self._detect_all_agents(frame)
        return agents[0] if agents else None

    def _detect_door_cell(
        self, frame: np.ndarray, grid_size: int,
    ) -> Optional[Tuple[int, int]]:
        """Detect the door as a hollow saturated square of any colour.

        Video renders the door in red; the interleave dataset renders it in
        blue. Rather than hardcode a colour, we scan every saturated non-
        green blob and pick the one with the thinnest outline (lowest
        fill-ratio inside a roughly-square bbox). Keys are filled diamonds
        (fill_ratio >= 0.35) and are filtered out by the hollow threshold.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_mask = cv2.inRange(
            hsv, np.array([0, 100, 80]), np.array([180, 255, 255]),
        )
        # Exclude green (agent).
        green_mask = cv2.inRange(
            hsv, np.array([40, 80, 80]), np.array([90, 255, 255]),
        )
        door_mask = cv2.bitwise_and(sat_mask, cv2.bitwise_not(green_mask))

        cell_px = max(frame.shape[:2]) / float(grid_size)
        cell_area = cell_px ** 2

        contours, _ = cv2.findContours(
            door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        hollow_candidates: List[Tuple[float, float, int, int]] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            bbox_area = bw * bh
            if bbox_area < cell_area * 0.15 or bbox_area > cell_area * 4.0:
                continue
            aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0
            if aspect < 0.6:
                continue  # not square-ish (skip long lines, etc.)
            inside = door_mask[y:y + bh, x:x + bw]
            pixels = int(np.sum(inside > 0))
            fill_ratio = pixels / bbox_area if bbox_area else 0.0
            if fill_ratio < 0.30:  # clearly hollow outline
                # Rank by squareness (aspect) with a secondary hollow-ness key
                hollow_candidates.append((aspect, 1.0 - fill_ratio, x + bw // 2, y + bh // 2))

        if hollow_candidates:
            hollow_candidates.sort(key=lambda t: (-t[0], -t[1]))
            _, _, cx, cy = hollow_candidates[0]
            return maze.pixel_to_cell(cx, cy, frame.shape, grid_size)
        return None

    def _detect_key_cells(
        self, frame: np.ndarray, grid_size: int,
    ) -> List[Tuple[int, int]]:
        """Colour-agnostic key detection (any saturated, filled, non-green,
        non-red blob with diamond-like fill ratio)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        sat_mask = cv2.inRange(
            hsv, np.array([0, 100, 100]), np.array([180, 255, 255]),
        )
        green_mask = cv2.inRange(
            hsv, np.array(self.AGENT_HSV_LOWER), np.array(self.AGENT_HSV_UPPER),
        )
        red_mask = (
            cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            | cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        )
        key_mask = cv2.bitwise_and(
            sat_mask, cv2.bitwise_not(cv2.bitwise_or(green_mask, red_mask)),
        )

        cell_px = max(frame.shape[:2]) / float(grid_size)
        cell_area = cell_px ** 2

        contours, _ = cv2.findContours(
            key_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cells: Set[Tuple[int, int]] = set()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cell_area * 0.08 or area > cell_area * 2.5:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            bbox_area = bw * bh
            if bbox_area == 0 or (area / bbox_area) < 0.35:
                continue  # thin outline → skip
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cells.add(maze.pixel_to_cell(cx, cy, frame.shape, grid_size))

        return sorted(cells)

    # ------------------------------------------------------------------
    # Cell-based proximity (stricter than pixel-based — GT scores 1.0)
    # ------------------------------------------------------------------

    def _score_cell_proximity(
        self,
        video_frames: List[np.ndarray],
        optimal_cells: Set[Tuple[int, int]],
        grid_size: int,
    ) -> float:
        """Per-frame proximity: 1.0 if agent's cell is on the optimal tour,
        else linearly decays by cell-Manhattan distance.

        Uses cells instead of pixels because in-cell pixel offsets are
        rendering artefacts, not path quality.  Extra blobs more than one
        cell off the optimal tour are penalised the same way as in
        :func:`maze.score_proximity`.
        """
        if not optimal_cells or not video_frames:
            return 0.0

        opt_arr = np.array(list(optimal_cells))  # (N, 2)
        max_cells = 2.0  # matches maze.score_proximity's max_distance_cells

        frame_scores: List[float] = []
        for frame in video_frames:
            blobs = self._detect_all_agents(frame)
            if not blobs:
                frame_scores.append(0.0)
                continue

            cell_dists: List[int] = []
            for ax, ay in blobs:
                cell = maze.pixel_to_cell(ax, ay, frame.shape, grid_size)
                d = int(
                    np.min(
                        np.abs(opt_arr[:, 0] - cell[0])
                        + np.abs(opt_arr[:, 1] - cell[1])
                    )
                )
                cell_dists.append(d)

            cell_dists.sort()
            best = cell_dists[0]
            base = max(0.0, 1.0 - best / max_cells)

            n_hallucinated = sum(1 for d in cell_dists[1:] if d > 1)
            extra_pen = min(1.0, n_hallucinated * self.EXTRA_AGENT_PENALTY)
            frame_scores.append(base * (1.0 - extra_pen))

        return sum(frame_scores) / len(frame_scores) if frame_scores else 0.0

    # ------------------------------------------------------------------
    # Optimal tour: state-space BFS with required keys + wall obstacles
    # ------------------------------------------------------------------

    def _compute_optimal_path_info(
        self,
        frame: np.ndarray,
        grid_size: int,
    ) -> Optional[Tuple[
        List[Tuple[int, int]],
        Dict[int, List[Tuple[int, int]]],
        int,
        Tuple[int, int],
        Tuple[int, int],
        Set[Tuple[int, int]],
        List[Tuple[int, int]],
        Set[Tuple[int, int]],
    ]]:
        """Return ``(ref_points, by_dist, shortest, start_cell, door_cell,
        walls, key_cells, optimal_cells)`` or ``None`` when detection or
        BFS fails."""
        agent = self._detect_agent(frame)
        if agent is None:
            return None
        start_cell = maze.pixel_to_cell(
            agent[0], agent[1], frame.shape, grid_size,
        )

        door_cell = self._detect_door_cell(frame, grid_size)
        if door_cell is None:
            return None

        walls = self._detect_wall_cells(frame, grid_size)
        key_cells = self._detect_key_cells(frame, grid_size)
        # Landmarks sit on corridor cells — make sure mis-thresholded walls
        # never mask them out.
        for c in (start_cell, door_cell, *key_cells):
            walls.discard(c)

        if not key_cells:
            # No keys detected: degrade to plain shortest-path (G-18).
            result = maze.optimal_cell_set(
                start_cell, door_cell, walls, grid_size,
            )
            if result is None:
                return None
            optimal_cells, dist_s, shortest = result
            all_points: List[Tuple[int, int]] = []
            by_dist: Dict[int, List[Tuple[int, int]]] = {}
            for c in optimal_cells:
                px = maze.cell_center_px(c, frame.shape, grid_size)
                all_points.append(px)
                by_dist.setdefault(dist_s[c], []).append(px)
            return (
                all_points, by_dist, shortest,
                start_cell, door_cell, walls, [],
                set(optimal_cells),
            )

        dist_s, required_bits, all_mask, _start_state, goal_state = (
            maze.grid_state_bfs(
                start=start_cell, end=door_cell,
                required_cells=key_cells, obstacles=walls,
                grid_size=grid_size,
            )
        )
        if goal_state not in dist_s:
            return None

        dist_e = maze.grid_state_reverse_bfs(
            end=door_cell,
            required_bits=required_bits,
            all_mask=all_mask,
            obstacles=walls,
            grid_size=grid_size,
        )

        shortest = dist_s[goal_state]
        by_dist_cells: Dict[int, Set[Tuple[int, int]]] = {}
        optimal_cells: Set[Tuple[int, int]] = set()
        for state, d in dist_s.items():
            if state in dist_e and d + dist_e[state] == shortest:
                cell = state[0]
                by_dist_cells.setdefault(d, set()).add(cell)
                optimal_cells.add(cell)

        all_points = []
        by_dist = {}
        seen: Set[Tuple[int, int]] = set()
        for d, cells_here in by_dist_cells.items():
            pts = [
                maze.cell_center_px(c, frame.shape, grid_size)
                for c in sorted(cells_here)
            ]
            by_dist[d] = pts
            for pt in pts:
                if pt not in seen:
                    seen.add(pt)
                    all_points.append(pt)

        return (
            all_points, by_dist, shortest,
            start_cell, door_cell, walls, list(key_cells),
            optimal_cells,
        )

    # ------------------------------------------------------------------
    # Video evaluation
    # ------------------------------------------------------------------

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        if not video_frames or gt_final_frame is None:
            return 0.0

        landmark_frame = (
            gt_first_frame
            if gt_first_frame is not None
            else (gt_frames[0] if gt_frames else video_frames[0])
        )

        grid_size = self._infer_grid_size(landmark_frame)
        self._grid_size = grid_size
        cell = maze.cell_size(landmark_frame, grid_size)

        opt = self._compute_optimal_path_info(landmark_frame, grid_size)
        if opt is None:
            self._last_task_details = {
                "error": "optimal_path_infeasible",
                "grid_size": grid_size,
            }
            return 0.0

        (
            ref_points, _by_dist, shortest,
            start_cell, door_cell, walls, key_cells, optimal_cells,
        ) = opt

        proximity = self._score_cell_proximity(
            video_frames, optimal_cells, grid_size,
        )

        actual_hops, bfs_fill = maze.decode_cell_hops(
            video_frames, ref_points, start_cell, door_cell, walls,
            self._detect_all_agents,
            gap_threshold=self.COVERAGE_GAP_THRESHOLD,
            grid_size=grid_size,
        )
        total_hops = actual_hops + bfs_fill
        coverage = actual_hops / total_hops if total_hops > 0 else 0.0
        length_factor = (
            min(1.0, shortest / total_hops) if total_hops > 0 else 0.0
        )

        strict_single = maze.make_strict_single_detector(
            self._detect_all_agents, ref_points,
        )
        cont_penalty = maze.discontinuity_penalty(
            video_frames, cell, strict_single,
            disappear_cap=self.DISAPPEAR_RATE_CAP,
            penalty_floor=self.PENALTY_FLOOR,
            cell_based=True,
            grid_size=grid_size,
        )
        continuity_factor = 1.0 - self.MAX_PENALTY * cont_penalty

        visited_cells = {
            c for c in maze.best_blob_cells(
                video_frames, self._detect_all_agents, ref_points,
                grid_size=grid_size,
            ) if c is not None
        }
        tol = self.KEY_VISIT_TOLERANCE
        def _near(k):
            if not visited_cells:
                return False
            return min(
                abs(k[0] - v[0]) + abs(k[1] - v[1]) for v in visited_cells
            ) <= tol
        visited_keys = [k for k in key_cells if _near(k)]
        missed_keys = [k for k in key_cells if not _near(k)]
        key_multiplier = 0.5 ** len(missed_keys)

        hit_report = maze.obstacle_hit_report(
            video_frames, walls, self._detect_all_agents,
            grid_size=grid_size,
            reference_points=ref_points,
        )
        num_wall_hit_cells = len(hit_report["hit_cells"])
        wall_multiplier = 0.5 ** num_wall_hit_cells

        score_without_coverage = (
            proximity * continuity_factor * key_multiplier * wall_multiplier
        )
        task_score = score_without_coverage * coverage * length_factor

        dynamic_cells = set(key_cells)
        dynamic_cells.add(door_cell)
        bg_dynamic_exclude_mask = self._gt_dynamic_exclude_mask(
            gt_frames, landmark_frame, dynamic_cells, grid_size,
        )
        bg_preservation = maze.background_preservation_frames(
            video_frames, landmark_frame,
            detector=self._detect_all_agents,
            base_exclude_mask=bg_dynamic_exclude_mask,
            grid_size=grid_size,
            noise_floor=self.BG_NOISE_FLOOR,
            bad_pixel_threshold=self.BG_BAD_PIXEL_THRESHOLD,
            bad_pixel_tolerance=self.BG_BAD_PIXEL_TOLERANCE,
        )
        bg_dynamic_exclude_fraction = (
            float(np.mean(bg_dynamic_exclude_mask > 0))
            if bg_dynamic_exclude_mask is not None else 0.0
        )
        # Background preservation is a penalty multiplier
        final_score = task_score * (0.6 + 0.4 * bg_preservation)

        self._last_task_details = {
            "grid_size": grid_size,
            "start_cell": list(start_cell),
            "door_cell": list(door_cell),
            "key_cells": [list(k) for k in key_cells],
            "num_walls": len(walls),
            "shortest_distance": shortest,
            "proximity": round(float(proximity), 4),
            "coverage": round(float(coverage), 4),
            "actual_hops": int(actual_hops),
            "bfs_fill_hops": int(bfs_fill),
            "total_hops": int(total_hops),
            "length_factor": round(float(length_factor), 4),
            "continuity_penalty": round(float(cont_penalty), 4),
            "continuity_factor": round(float(continuity_factor), 4),
            "visited_keys": [list(k) for k in visited_keys],
            "missed_keys": [list(k) for k in missed_keys],
            "num_missed_keys": len(missed_keys),
            "key_visit_tolerance": self.KEY_VISIT_TOLERANCE,
            "key_multiplier": round(float(key_multiplier), 6),
            "wall_hit_cells": hit_report["hit_cells"],
            "num_wall_hit_cells": num_wall_hit_cells,
            "wall_hit_frames": hit_report["hit_frames"],
            "wall_hit_per_cell": hit_report["per_cell_frames"],
            "wall_multiplier": round(float(wall_multiplier), 6),
            "score_without_coverage": round(float(score_without_coverage), 4),
            "task_score": round(float(task_score), 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "bg_dynamic_exclude_fraction": round(float(bg_dynamic_exclude_fraction), 4),
            "bg_noise_floor": self.BG_NOISE_FLOOR,
            "bg_bad_pixel_threshold": self.BG_BAD_PIXEL_THRESHOLD,
            "bg_bad_pixel_tolerance": self.BG_BAD_PIXEL_TOLERANCE,
            "final_score": round(float(final_score), 4),
            "score_breakdown": {
                "formula": (
                    "0.8 × task_score + 0.2 × bg_preservation; "
                    "task_score = proximity × continuity × 0.5^missed_keys"
                    " × 0.5^wall_hits × coverage × length_factor"
                ),
                "proximity": round(float(proximity), 4),
                "continuity_factor": round(float(continuity_factor), 4),
                "key_multiplier": round(float(key_multiplier), 6),
                "wall_multiplier": round(float(wall_multiplier), 6),
                "coverage": round(float(coverage), 4),
                "length_factor": round(float(length_factor), 4),
                "task_score": round(float(task_score), 4),
                "bg_preservation": round(float(bg_preservation), 4),
                "final": round(float(final_score), 4),
            },
        }
        return final_score

    # ------------------------------------------------------------------
    # Interleave evaluation
    # ------------------------------------------------------------------

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Interleave: cell-based scoring aligned with the video method.

        score = proximity x coverage x 0.5^num_missed_keys x 0.5^num_wall_hits
        """
        if not pred_images or input_frame is None:
            self._last_task_details = {"error": "no_input_or_pred"}
            return 0.0

        grid_size = self._infer_grid_size(input_frame)
        opt = self._compute_optimal_path_info(input_frame, grid_size)
        if opt is None:
            self._last_task_details = {
                "error": "detection_failed",
                "grid_size": grid_size,
            }
            return 0.0
        (
            ref_points, by_dist, shortest,
            start_cell, door_cell, walls, key_cells, optimal_cells,
        ) = opt

        counts = maze.cell_draw_counts(
            pred_images, input_frame, grid_size=grid_size,
        )
        drawn = set(counts)
        # Landmark cells are coloured in the input; diff may miss them.
        landmarks = {start_cell, door_cell, *key_cells}
        drawn = drawn | (landmarks & set(optimal_cells))

        # G-47 interleave is agent-trajectory (each "drawn" cell is an
        # isolated agent-blob stamp, not part of one continuous line), so
        # cell-4-adjacency is the right connectivity semantic.  Skip the
        # pixel-component check that the line-drawing tasks use.
        walk = maze.simulate_walk_through_drawn(
            drawn=drawn, start=start_cell, end=door_cell,
            required=set(key_cells),
            grid_size=grid_size,
            allow_cells=landmarks,
        )

        gt_drawn = maze.cells_from_pred_diff(
            gt_images, input_frame, grid_size=grid_size,
        )
        gt_drawn = gt_drawn | (landmarks & set(optimal_cells)) if gt_drawn else set(optimal_cells)

        task_score, details = maze.score_interleave_walk(
            walk=walk, drawn=drawn, optimal_cells=gt_drawn,
            required_cells=list(key_cells),
            wall_cells=set(walls),
            path_length=len(gt_drawn),
            draw_counts=counts,
        )
        pred_mask = maze.pred_diff_mask(pred_images, input_frame)
        bg_preservation = maze.background_preservation_image(
            pred_images[-1], input_frame,
            exclude_mask=pred_mask,
            noise_floor=self.BG_NOISE_FLOOR,
            bad_pixel_threshold=self.BG_BAD_PIXEL_THRESHOLD,
            bad_pixel_tolerance=self.BG_BAD_PIXEL_TOLERANCE,
        )
        # Background preservation is a penalty multiplier
        final_score = task_score * (0.6 + 0.4 * bg_preservation)
        details.update({
            "grid_size": grid_size,
            "start_cell": list(start_cell),
            "door_cell": list(door_cell),
            "key_cells": [list(c) for c in key_cells],
            "total_shortest_distance": shortest,
            "task_score": round(float(task_score), 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "bg_noise_floor": self.BG_NOISE_FLOOR,
            "bg_bad_pixel_threshold": self.BG_BAD_PIXEL_THRESHOLD,
            "bg_bad_pixel_tolerance": self.BG_BAD_PIXEL_TOLERANCE,
            "final_score": round(float(final_score), 4),
            "score_breakdown": {
                "formula": "0.8 × task_score + 0.2 × bg_preservation",
                "task_score": round(float(task_score), 4),
                "bg_preservation": round(float(bg_preservation), 4),
                "final": round(float(final_score), 4),
            },
        })
        self._last_task_details = details
        return final_score

class ConnectingColorEvaluator(BaseEvaluator):
    """
    G-54: Connecting color evaluator.

    Task: Input image has colored shapes (any shape) on white background.
    Same-color shapes should be connected by same-color curves in the output.

    Evaluates:
    - correct_connections (80%): Same-color objects connected by same-color curves
    - consistency (20%): object_preservation * background_preservation
    """

    TASK_WEIGHTS = {
        'correct_connections': 0.80,
        'consistency': 0.20
    }

    def _detect_objects(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        h, w = frame.shape[:2]
        corners = [frame[5, 5], frame[5, w-5], frame[h-5, 5], frame[h-5, w-5]]
        bg_color = np.mean(corners, axis=0).astype(np.float32)

        dist = np.sqrt(np.sum((frame.astype(np.float32) - bg_color) ** 2, axis=2))
        binary = (dist > 30).astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        objects = []
        fg_mask = np.zeros((h, w), dtype=np.uint8)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            temp = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(temp, [contour], -1, 255, cv2.FILLED)
            mean_bgr = cv2.mean(frame, mask=temp)[:3]
            mean_hsv = cv2.mean(hsv, mask=temp)[:3]

            cv2.drawContours(fg_mask, [contour], -1, 255, cv2.FILLED)
            x, y, bw, bh = cv2.boundingRect(contour)

            objects.append({
                'center': (cx, cy),
                'area': area,
                'bbox': (x, y, bw, bh),
                'contour': contour,
                'mean_bgr': mean_bgr,
                'mean_hsv': mean_hsv,
            })

        # Pair objects by HSV distance (objects come in pairs of same color)
        objects = self._assign_color_pairs(objects)

        return objects, fg_mask

    def _assign_color_pairs(self, objects: List[Dict]) -> List[Dict]:
        n = len(objects)
        if n < 2:
            for i, obj in enumerate(objects):
                obj['color_label'] = f'color_{i}'
            return objects

        used = [False] * n
        pair_id = 0
        pairs = []
        for i in range(n):
            if used[i]:
                continue
            best_j = -1
            best_dist = float('inf')
            for j in range(i + 1, n):
                if used[j]:
                    continue
                hi, si, vi = objects[i]['mean_hsv']
                hj, sj, vj = objects[j]['mean_hsv']
                h_diff = min(abs(hi - hj), 180 - abs(hi - hj))
                dist = np.sqrt(h_diff**2 + (si - sj)**2 + (vi - vj)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j >= 0:
                label = f'color_{pair_id}'
                objects[i]['color_label'] = label
                objects[best_j]['color_label'] = label
                used[i] = True
                used[best_j] = True
                pair_id += 1

        for i in range(n):
            if not used[i]:
                objects[i]['color_label'] = f'color_{pair_id}'
                pair_id += 1

        return objects

    def _pixel_diff_score(self, frame1: np.ndarray, frame2: np.ndarray,
                          mask: np.ndarray,
                          thresholds: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)) -> Tuple[float, Dict]:
        mask_pixels = int((mask > 0).sum())
        if mask_pixels == 0:
            return 1.0, {'ratio': 0.0, 'changed_px': 0, 'total_px': 0}
        diff = cv2.absdiff(frame1, frame2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed = (gray_diff[mask > 0] > 20).sum()
        ratio = changed / mask_pixels
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
        return score, {'ratio': float(ratio), 'changed_px': int(changed), 'total_px': mask_pixels}
    
    def _evaluate_connections(self, gt_first: np.ndarray, gt_last: np.ndarray,
                              gen_last: np.ndarray,
                              gt_objects: List[Dict], fg_mask: np.ndarray) -> Tuple[float, Dict, Optional[np.ndarray]]:
        """Evaluate connections using color-based connected components on gen_last.

        1. For each color in gt_objects, create color mask on gen_last
        2. Find connected components on each color mask
        3. Same-color object pair is connected if both fall in the same component
        4. Also returns curve_mask (changed pixels outside objects) for background scoring
        """
        h, w = gt_first.shape[:2]
        hsv_gen = cv2.cvtColor(gen_last, cv2.COLOR_BGR2HSV)

        # GT curve pixels: new pixels in gt_last vs gt_first, outside objects
        gt_diff = cv2.absdiff(gt_first, gt_last)
        gt_gray_diff = cv2.cvtColor(gt_diff, cv2.COLOR_BGR2GRAY)
        gt_curve_mask = (gt_gray_diff > 20).astype(np.uint8) * 255
        gt_curve_mask[fg_mask > 0] = 0
        gt_curve_pixels = cv2.countNonZero(gt_curve_mask)

        objects_by_color = {}
        for obj in gt_objects:
            objects_by_color.setdefault(obj['color_label'], []).append(obj)

        # Collect same-color pairs
        expected = 0
        color_pairs = []
        for color, objs in objects_by_color.items():
            if len(objs) >= 2:
                for i in range(len(objs)):
                    for j in range(i + 1, len(objs)):
                        color_pairs.append((color, objs[i], objs[j]))
                        expected += 1

        if expected == 0:
            return 0.0, {'error': 'no_same_color_pairs'}, None

        # Compute mean H for each color group
        color_mean_hsv = {}
        for color, objs in objects_by_color.items():
            color_mean_hsv[color] = (
                np.mean([o['mean_hsv'][0] for o in objs]),
                np.mean([o['mean_hsv'][1] for o in objs]),
            )

        # Find min hue gap between different color groups to limit tolerance
        colors = list(color_mean_hsv.keys())
        min_h_gap = 180
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                hi = color_mean_hsv[colors[i]][0]
                hj = color_mean_hsv[colors[j]][0]
                gap = min(abs(hi - hj), 180 - abs(hi - hj))
                min_h_gap = min(min_h_gap, gap)
        h_tol = max(5, min(15, min_h_gap / 2 - 1))

        # For each color pair, create mask from the pair's mean HSV
        kernel = np.ones((5, 5), np.uint8)
        color_label_maps = {}
        color_masks = {}
        for color in objects_by_color:
            mean_h, mean_s = color_mean_hsv[color]
            color_mask = self._get_color_mask_from_hsv(hsv_gen, mean_h, mean_s, h_tol=h_tol)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            _, label_map = cv2.connectedComponents(color_mask)
            color_label_maps[color] = label_map
            color_masks[color] = color_mask

        # Check each pair
        correct = 0
        correct_labels = {}  # color -> set of correctly connected component labels
        pair_results = []
        for color, obj1, obj2 in color_pairs:
            if color not in color_label_maps:
                pair_results.append({'color': color, 'connected': False, 'reason': 'no_color_mask'})
                continue

            label_map = color_label_maps[color]

            # Get labels near each object center
            labels1 = self._get_labels_near(label_map, obj1['center'], radius=10)
            labels2 = self._get_labels_near(label_map, obj2['center'], radius=10)
            labels1.discard(0)  # Remove background
            labels2.discard(0)

            # Connected if they share a component label
            shared = labels1 & labels2
            connected = len(shared) > 0
            if connected:
                correct += 1
                # Record the connected component labels for building correct_curve_mask
                correct_labels.setdefault(color, set()).update(shared)

            pair_results.append({
                'color': color,
                'obj1': obj1['center'],
                'obj2': obj2['center'],
                'labels1': list(labels1),
                'labels2': list(labels2),
                'connected': connected,
            })

        # Check cross-color connections (penalty): different-color objects in same region
        cross_count = 0
        colors_list = list(objects_by_color.keys())
        for i in range(len(colors_list)):
            for j in range(i + 1, len(colors_list)):
                c1, c2 = colors_list[i], colors_list[j]
                # Merge both color masks and check if objects from different colors connect
                merged = cv2.bitwise_or(color_masks[c1], color_masks[c2])
                merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)
                _, merged_labels = cv2.connectedComponents(merged)
                for o1 in objects_by_color[c1]:
                    for o2 in objects_by_color[c2]:
                        l1 = self._get_labels_near(merged_labels, o1['center'], radius=10)
                        l2 = self._get_labels_near(merged_labels, o2['center'], radius=10)
                        l1.discard(0)
                        l2.discard(0)
                        if l1 & l2:
                            cross_count += 1

        # Penalty for cross-color connections
        cross_penalty = min(1.0, cross_count * 0.2)

        # Build correct_curve_mask: only correctly connected components, outside objects
        correct_curve_mask = np.zeros((h, w), dtype=np.uint8)
        for color, labels in correct_labels.items():
            label_map = color_label_maps[color]
            for lbl in labels:
                correct_curve_mask[label_map == lbl] = 255
        # Remove object regions from curve mask
        correct_curve_mask[fg_mask > 0] = 0

        # Pixel count penalty: gen curve pixels should not be much more than GT
        gen_curve_pixels = cv2.countNonZero(correct_curve_mask)
        if gt_curve_pixels > 0:
            pixel_ratio = gen_curve_pixels / gt_curve_pixels
            # ratio ~1.0 is ideal; penalize if >2x GT
            if pixel_ratio > 5.0:
                pixel_penalty = 0.4
            elif pixel_ratio > 3.0:
                pixel_penalty = 0.2
            else:
                pixel_penalty = 0.0
        else:
            pixel_penalty = 0.0

        pixel_factor = max(0.0, 1.0 - pixel_penalty)
        score = max(0.0, (correct / expected) * (1 - cross_penalty) * pixel_factor)
        details = {
            'expected': expected, 'correct': correct,
            'cross_connections': cross_count, 'cross_penalty': cross_penalty,
            'gt_curve_pixels': gt_curve_pixels, 'gen_curve_pixels': gen_curve_pixels,
            'pixel_penalty': pixel_penalty,
            'pairs': pair_results,
        }
        return score, details, correct_curve_mask

    @staticmethod
    def _get_color_mask_from_hsv(hsv: np.ndarray, target_h: float, target_s: float,
                                  h_tol: float = 15, s_min: float = 40) -> np.ndarray:
        """Create binary mask for pixels matching a target HSV color.
        Uses the object's actual mean H/S to define the range dynamically."""
        # Hue tolerance (circular), cast to int for cv2.inRange
        h_low = int(target_h - h_tol)
        h_high = int(target_h + h_tol)
        s_low = int(max(s_min, target_s - 60))

        if h_low < 0:
            # Hue wraps around 0 (e.g. red)
            mask1 = cv2.inRange(hsv, np.array([0, s_low, 60]), np.array([h_high, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([180 + h_low, s_low, 60]), np.array([180, 255, 255]))
            return mask1 | mask2
        elif h_high > 180:
            mask1 = cv2.inRange(hsv, np.array([h_low, s_low, 60]), np.array([180, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([0, s_low, 60]), np.array([h_high - 180, 255, 255]))
            return mask1 | mask2
        else:
            return cv2.inRange(hsv, np.array([h_low, s_low, 60]), np.array([h_high, 255, 255]))

    @staticmethod
    def _get_labels_near(label_map: np.ndarray, center: Tuple[int, int], radius: int = 10) -> set:
        """Get set of connected component labels near a point."""
        h, w = label_map.shape
        cx, cy = center
        y_min, y_max = max(0, cy - radius), min(h, cy + radius)
        x_min, x_max = max(0, cx - radius), min(w, cx + radius)
        region = label_map[y_min:y_max, x_min:x_max]
        return set(region.flatten())

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        scores = {}

        if len(video_frames) < 2 or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        gt_first = gt_first_frame
        gt_last = gt_final_frame

        if last_frame.shape != gt_last.shape:
            last_frame = normalize_frame_size(last_frame, gt_last)
        if first_frame.shape != gt_first.shape:
            first_frame = normalize_frame_size(first_frame, gt_first)

        gt_objects, fg_mask = self._detect_objects(gt_first)


        if len(gt_objects) < 2:
            self._last_task_details = {'error': 'not_enough_objects', 'num_objects': len(gt_objects)}
            return 0.0

        kernel = np.ones((5, 5), np.uint8)
        fg_mask_dilated = cv2.dilate(fg_mask, kernel, iterations=1)

        # 2. Object preservation (20%): pixel diff in object regions
        # Check this FIRST — if objects are destroyed, everything else is meaningless
        scores['object_preservation'], obj_details = self._pixel_diff_score(
            gt_first, last_frame, fg_mask_dilated, thresholds=(0.1, 0.2, 0.30, 0.50)
        )

        # 2. Correct connections (60%):
        scores['correct_connections'], conn_details, curve_mask = self._evaluate_connections(
            gt_first, gt_last, last_frame, gt_objects, fg_mask_dilated
        )

        # 3. Background preservation (20%): exclude objects AND correct curves
        bg_mask = cv2.bitwise_not(fg_mask_dilated)
        if curve_mask is not None:
            bg_mask[curve_mask > 0] = 0  # Correct curves should not penalize background
        scores['background_preservation'], bg_details = self._pixel_diff_score(
            gt_first, last_frame, bg_mask, thresholds=(0.015, 0.025, 0.035, 0.05)
        )
        
        scores['consistency'] = (scores['object_preservation'] + scores['background_preservation']) / 2

        self._last_task_details = {
            **scores,
            'num_objects': len(gt_objects),
            'object_colors': [o['color_label'] for o in gt_objects],
            **{f'obj_{k}': v for k, v in obj_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
            **{f'conn_{k}': v for k, v in conn_details.items() if not isinstance(v, (list, dict))},
        }
        return float((scores['correct_connections']) * (0.6 + 0.4 * scores['consistency']))


class SelectNextFigureAlternatingEvaluator(BaseEvaluator):
    """
    G-135: Select next figure in small-big alternating sequence.
    
    Rule-based evaluation:
    - Pattern recognition (40%): Identify "small-big-small" pattern in existing sequence
    - Selection correctness (35%): Next should be "big" - largest candidate selected
    - Marking accuracy (15%): Red circle marks exactly one figure
    - Animation quality (10%): Circle appears with smooth expansion
    """
    
    TASK_WEIGHTS = {
        'pattern_recognition': 0.40,
        'selection_correctness': 0.35,
        'marking_accuracy': 0.15,
        'animation_quality': 0.10
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
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        # 1. Pattern recognition (40%)
        # Rule: Check if pattern analysis shows alternating sizes
        scores['pattern_recognition'] = self._evaluate_pattern_recognition(
            first_frame, final_frame
        )
        
        # 2. Selection correctness (35%)
        # Rule: Red circle should mark the largest candidate figure
        scores['selection_correctness'] = self._evaluate_selection(
            first_frame, final_frame
        )
        
        # 3. Marking accuracy (15%)
        # Rule: Exactly one red circle marking
        scores['marking_accuracy'] = self._evaluate_marking(final_frame, first_frame)
        
        # 4. Animation quality (10%)
        # Rule: Circle should expand smoothly
        scores['animation_quality'] = self._evaluate_animation(video_frames)

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate alternating figure selection for interleaved image generation.

        Interleave outputs a single frame with the selected figure marked.
        Video version:
          - pattern_recognition (40%): alternating size pattern — first/last frame
          - selection_correctness (35%): correct candidate marked — first/last frame
          - marking_accuracy (15%): exactly one red circle — last frame
          - animation_quality (10%): circle expands smoothly — needs multiple frames
        Interleave version:
          - pattern_recognition (50%): reuse, absorb animation_quality weight
          - selection_correctness (35%): reuse
          - marking_accuracy (15%): reuse
        """
        INTERLEAVE_WEIGHTS = {
            'pattern_recognition': 0.50,
            'selection_correctness': 0.35,
            'marking_accuracy': 0.15,
        }

        if not pred_images or input_frame is None:
            return 0.0

        scores = {}
        last_frame = pred_images[-1]
        first_frame = input_frame

        scores['pattern_recognition'] = self._evaluate_pattern_recognition(first_frame, last_frame)
        scores['selection_correctness'] = self._evaluate_selection(first_frame, last_frame)
        scores['marking_accuracy'] = self._evaluate_marking(last_frame, first_frame)

        self._last_task_details = scores
        return sum(scores[k] * INTERLEAVE_WEIGHTS[k] for k in INTERLEAVE_WEIGHTS)


    def _evaluate_pattern_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if alternating pattern is understood."""
        # Detect existing shapes
        shapes = self._detect_shapes_with_sizes(first_frame)
        
        if len(shapes) < 3:
            return 0.5
        
        # Only consider sequence shapes (top half) for pattern recognition
        h = first_frame.shape[0]
        sequence_shapes = [s for s in shapes if s[1] < h // 2]
        
        if len(sequence_shapes) < 3:
            return 0.5
        
        # Sort by x-position (left to right sequence)
        shapes_sorted = sorted(sequence_shapes, key=lambda s: s[0])
        sizes = [s[2] for s in shapes_sorted]
        
        if len(sizes) < 3:
            return 0.5
        
        # Check for alternating pattern: small-big-small or big-small-big
        is_alternating = True
        for i in range(len(sizes) - 2):
            if sizes[i] < sizes[i+1] > sizes[i+2] or sizes[i] > sizes[i+1] < sizes[i+2]:
                continue
            else:
                is_alternating = False
                break
        
        if is_alternating:
            return 1.0
        return 0.5
    
    def _evaluate_selection(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the correct candidate is marked based on pattern."""
        # Detect red circle marking (new markings only)
        circles = self._detect_red_circles(final_frame, first_frame)
        
        if len(circles) == 0:
            return 0.0
        
        marked_pos = circles[0][:2]  # Get position of marked item
        
        # Detect all shapes
        all_shapes = self._detect_shapes_with_sizes(first_frame)
        
        if len(all_shapes) == 0:
            return 0.5
        
        # Separate into sequence (top half) and candidates (bottom half)
        h = first_frame.shape[0]
        sequence_shapes = sorted([s for s in all_shapes if s[1] < h // 2], key=lambda s: s[0])
        candidate_shapes = [s for s in all_shapes if s[1] >= h // 2]
        
        if len(candidate_shapes) == 0:
            return 0.5
        
        # Find which candidate is marked
        marked_candidate = None
        min_dist = float('inf')
        for cand in candidate_shapes:
            dist = np.sqrt((cand[0] - marked_pos[0])**2 + (cand[1] - marked_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                marked_candidate = cand
        
        if marked_candidate is None or min_dist > 100:
            return 0.3
        
        # Determine expected size based on alternating pattern
        if len(sequence_shapes) >= 2:
            sizes = [s[2] for s in sequence_shapes]
            
            # Calculate threshold as mean of min and max sizes
            min_size = min(sizes)
            max_size = max(sizes)
            threshold = (min_size + max_size) / 2
            
            # Classify each shape as small or big
            pattern = ['small' if s <= threshold else 'big' for s in sizes]
            last_type = pattern[-1]
            
            # Determine expected next type
            expected_type = 'big' if last_type == 'small' else 'small'
            
            # Check if marked candidate matches expected type
            marked_type = 'small' if marked_candidate[2] <= threshold else 'big'
            
            if marked_type == expected_type:
                return 1.0
            else:
                return 0.5
        
        # Fallback: check if marked candidate is among the larger ones
        candidate_sizes = [c[2] for c in candidate_shapes]
        if marked_candidate[2] >= np.median(candidate_sizes):
            return 0.8
        return 0.5
    
    def _evaluate_marking(
        self, 
        final_frame: np.ndarray, 
        first_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Evaluate red circle marking quality."""
        circles = self._detect_red_circles(final_frame, first_frame)
        
        if len(circles) == 0:
            return 0.0
        elif len(circles) == 1:
            return 1.0  # Correct number of markings
        else:
            return max(0.3, 1.0 - 0.2 * (len(circles) - 1))  # Penalty for multiple
    
    def _evaluate_animation(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate animation smoothness."""
        if len(video_frames) < 3:
            return 0.5
        
        # Check for smooth circle expansion
        circle_sizes = []
        for frame in video_frames[len(video_frames)//2:]:
            circles = self._detect_red_circles(frame)
            if circles:
                circle_sizes.append(circles[0][2] if len(circles[0]) > 2 else 30)
        
        if len(circle_sizes) < 2:
            return 0.5
        
        # Check if sizes increase smoothly
        increases = sum(1 for i in range(1, len(circle_sizes)) 
                       if circle_sizes[i] >= circle_sizes[i-1] * 0.95)
        smoothness = increases / (len(circle_sizes) - 1)
        
        return smoothness
    
    def _detect_shapes_with_sizes(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect shapes with their (x, y, area)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:  # Lower threshold to detect smaller shapes
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy, area))
        
        return shapes
    
    def _detect_candidate_shapes(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect candidate shapes (typically in bottom portion)."""
        h, w = frame.shape[:2]
        
        # Focus on bottom half or right portion where candidates usually are
        bottom_region = frame[h//2:, :]
        
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"]) + h//2  # Adjust for cropped region
                    candidates.append((cx, cy, area))
        
        return candidates
    
    def _detect_red_circles(
        self, 
        frame: np.ndarray, 
        first_frame: Optional[np.ndarray] = None
    ) -> List[Tuple[int, int, int]]:
        """Detect red circles in the frame (new markings only if first_frame provided)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # If first_frame provided, only detect NEW red regions (markings)
        if first_frame is not None:
            hsv_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
            mask_first = cv2.inRange(hsv_first, lower_red1, upper_red1) | cv2.inRange(hsv_first, lower_red2, upper_red2)
            # Only keep red regions that are new (not in first frame)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_first))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circles.append((int(x), int(y), int(radius)))
        
        return circles

class LocateAllCorrectPointsEvaluator(BaseEvaluator):
    """
    G-136: Locate all correct points evaluator.
    """
    
    TASK_WEIGHTS = {
        'consistency_score': 0.40,
        'match_score': 0.60
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate locate all correct points task."""
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        use_metafile = True # decide for each task
        meta_file_path = eval_info.get('metafile_path') if use_metafile else None
        debug_dir = eval_info.get('debug_dir')
        if debug_dir is not None:
            shutil.rmtree(debug_dir, ignore_errors=True)
            os.makedirs(debug_dir, exist_ok=True)
        circle_selection_processor = CircleSelectionProcessor(
            meta_file_path=meta_file_path,
            circle_color='red',
            circle_fill_max_ratio=0.9,
            circle_hsv_tolerance=(10, 80, 80), # special tolerance = 10, because background similar to red
            foreground_hsv_delta_tolerance=(15.0, 150.0, 150.0),  
            background_hsv_delta_tolerance=(15.0, 150.0, 150.0),
            foreground_enlarge_pixels=5 # special enlarge = 5, because foreground shapes are small
        )
        circle_selection_info = circle_selection_processor.process(gt_first_frame, gt_final_frame, last_frame, debug_dir=debug_dir)
        
        scores = {}
        background_consistency_score = threshold_score(
            circle_selection_info['background_change_ratio'],
            [(0.05, 1.0), (0.2, 0.0)]
        )
        foreground_consistency_score = threshold_score(
            circle_selection_info['foreground_change_ratio'],
            [(0.25, 1.0), (0.5, 0.0)]
        )
        circle_area_penalty_score = threshold_score(
            circle_selection_info['circle_color_mask_ratio'],
            [(0.3, 1.0), (0.5, 0.0)]
        )
        scores['consistency_score'] = (background_consistency_score + foreground_consistency_score + circle_area_penalty_score) / 3

        per_shape_scores = [0.0 for _ in range(len(circle_selection_info['is_target_shape']))]
        circle_size_penalty_list = []
        circle_ratio_list = []
        for circle_id, per_shape_overlap_ratio in enumerate(circle_selection_info['circle_vs_shape_overlap']):
            circle_area = circle_selection_info['pred_circles'][circle_id]['area']
            approx_ratio = float(np.sqrt(circle_area) / gt_first_frame.shape[0])
            circle_ratio_list.append(approx_ratio)
            circle_match_score = 0.0
            for shape_id in range(len(per_shape_overlap_ratio)):
                shape_inclusion_score = threshold_score(
                    per_shape_overlap_ratio[shape_id],
                    [(0.9, 0.0), (1.0, 1.0)]
                )
                circle_match_score = max(circle_match_score, shape_inclusion_score)
                per_shape_scores[shape_id] = max(per_shape_scores[shape_id], shape_inclusion_score)
            if circle_match_score > 0.0:
                circle_size_penalty = threshold_score(
                    approx_ratio,
                    [(0.07, 0.0), (0.12, 1.0)]
                )
            else:
                circle_size_penalty = threshold_score(
                    approx_ratio,
                    [(0.0, 0.0), (0.02, 1.0)]
                )
            circle_size_penalty_list.append(circle_size_penalty)
        
        circle_size_penalty_score = float(sum(circle_size_penalty_list))
        selection_score = float(np.mean(np.array(per_shape_scores)))
        scores['match_score'] = max(0, selection_score * (1.0 - circle_size_penalty_score))
        task_score = scores['match_score']
        total_score = task_score * (0.6 + 0.4 * scores['consistency_score'])

        if len(circle_selection_info['pred_circles']) == 0:
            total_score = min(total_score, 0.1)

        if debug_dir is not None:
            circle_selection_info.pop('pred_circles')
            circle_selection_info.pop('foreground_shapes')
            circle_selection_info.pop('pred_foreground_shapes')
            debug_info = {
                # all info from circle_selection_info
                **circle_selection_info,
                # all info from scores
                'background_consistency_score': background_consistency_score,
                'foreground_consistency_score': foreground_consistency_score,
                'circle_area_penalty_score': circle_area_penalty_score,
                'circle_ratio_list': circle_ratio_list,
                'circle_size_penalty_list': circle_size_penalty_list,
                'circle_size_penalty_score': circle_size_penalty_score,
                'per_shape_scores': per_shape_scores,
                "selection_score": selection_score,
                **scores,
                "total_score": total_score
            }
            with open(os.path.join(debug_dir, "debug_info.json"), "w") as f:
                json.dump(debug_info, f)
        self._last_task_details = {
            'background_change_ratio': circle_selection_info['background_change_ratio'],
            'foreground_change_ratio': circle_selection_info['foreground_change_ratio'],
            'circle_color_mask_ratio': circle_selection_info['circle_color_mask_ratio'],
            'background_consistency_score': background_consistency_score,
            'foreground_consistency_score': foreground_consistency_score,
            'circle_area_penalty_score': circle_area_penalty_score,
            'circle_ratio_list': circle_ratio_list,
            'circle_size_penalty_list': circle_size_penalty_list,
            'circle_size_penalty_score': circle_size_penalty_score,
            'per_shape_scores': per_shape_scores,
            'selection_score': selection_score,
            'consistency_score': scores['consistency_score'],
            'match_score': scores['match_score'],
            'total_score': total_score,
        }
        return total_score


class LocateTopmostFigureEvaluator(BaseEvaluator):
    """
    G-140: Locate topmost (unobscured) figure in overlapping shapes.

    Scoring:
    - accuracy         (60%): IoU-based matching of red contours vs GT
    - back_consistency (20%): white background similarity between final and GT frames
    - fore_consistency (20%): non-white non-red foreground similarity
    """

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: Optional[List[np.ndarray]],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: np.ndarray,
        eval_info: Dict
    ) -> float:
        final_frame = video_frames[-1]
        canvas_size = (final_frame.shape[0], final_frame.shape[1])

        # Resize GT if needed
        if final_frame.shape != gt_final_frame.shape:
            gt_final_frame = cv2.resize(
                gt_final_frame, (final_frame.shape[1], final_frame.shape[0])
            )

        gt_contours  = detect_closed_contours_by_color(gt_final_frame, COLOR_BOUNDS['red'])
        gen_contours = detect_closed_contours_by_color(
            final_frame, COLOR_BOUNDS['red'],
            hull_fallback=True,
            ref_area=max((cv2.contourArea(c) for c in gt_contours), default=None),
        )

        match_results = match_contours(
            gt_contours, gen_contours,
            iou_threshold=0.1,
            canvas_size=canvas_size
        )

        valid_ious = [iou for iou in match_results if iou is not None]
        gen_centroids = []
        for c in gen_contours:
            m = cv2.moments(c)
            if m['m00'] > 0:
                gen_centroids.append((m['m10'] / m['m00'], m['m01'] / m['m00']))
        per_gt_scores = []
        n_contained = 0
        for gi, gt_cnt in enumerate(gt_contours):
            iou = match_results[gi] if gi < len(match_results) else None
            base = float(iou) if iou is not None else 0.0
            if any(cv2.pointPolygonTest(gt_cnt, (float(gx), float(gy)), False) >= 0 for (gx, gy) in gen_centroids):
                n_contained += 1
                per_gt_scores.append(max(base, 0.9))
            else:
                per_gt_scores.append(base)
        accuracy = float(np.mean(per_gt_scores)) if per_gt_scores else 0.0
        accuracy = accuracy * calculate_list_length_penalty(len(gt_contours), max(len(valid_ious), n_contained), len(gen_contours))
        back_consistency = score_background_similarity(gt_final_frame, final_frame)

        fore_consistency = score_foreground_similarity(
            gt_final_frame, final_frame, COLOR_BOUNDS['red']
        )

        consistency = 0.5 * back_consistency + 0.5 * fore_consistency
        score = accuracy * (0.6 + 0.4 * consistency)
        self._last_task_details = {
            'accuracy': accuracy,
            'back_consistency': back_consistency,
            'fore_consistency': fore_consistency,
        }
        return score


class SelectFigureOutOfDomainEvaluator(BaseEvaluator):
    """
    G-160, G-247: Choose from figures.
    G-218: identify largest angle in triangle.
    """
   
    TASK_WEIGHTS = {
        'consistency_score': 0.30,
        'match_score': 0.70
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        use_metafile = True # decide for each task
        meta_file_path = eval_info.get('metafile_path') if use_metafile else None
        debug_dir = eval_info.get('debug_dir')
        if debug_dir is not None:
            shutil.rmtree(debug_dir, ignore_errors=True)
            os.makedirs(debug_dir, exist_ok=True)
        circle_selection_processor = CircleSelectionProcessor(
            meta_file_path=meta_file_path,
            circle_color='red',
            circle_fill_max_ratio=0.9,
            circle_hsv_tolerance=(15, 80, 80),
            foreground_hsv_delta_tolerance=(15.0, 150.0, 150.0),
            background_hsv_delta_tolerance=(15.0, 150.0, 150.0),
            consistency_forground_remove_bg="white",
            foreground_enlarge_pixels=0
        )
        circle_selection_info = circle_selection_processor.process(gt_first_frame, gt_final_frame, last_frame, debug_dir=debug_dir)
        
        scores = {}
        background_consistency_score = threshold_score(
            circle_selection_info['background_change_ratio'],
            [(0.015, 1.0), (0.03, 0.0)]
        )
        foreground_consistency_score = threshold_score(
            circle_selection_info['foreground_change_ratio'],
            [(0.55, 1.0), (0.7, 0.0)]
        )
        circle_area_penalty_score = threshold_score(
            circle_selection_info['circle_color_mask_ratio'],
            [(0.075, 1.0), (0.1, 0.0)]
        )
        scores['consistency_score'] = (background_consistency_score + foreground_consistency_score + circle_area_penalty_score) / 3

        per_shape_scores = [0.0 for _ in range(len(circle_selection_info['is_target_shape']))]
        ambiguous_circles_count = 0
        selected_threshold = 0.6
        ignored_threshold = 0.25
        for per_shape_overlap_ratio in circle_selection_info['circle_vs_shape_overlap']:
            ambiguous_shapes_count = sum(1 for ratio in per_shape_overlap_ratio if ratio > ignored_threshold and ratio < selected_threshold)
            if ambiguous_shapes_count > 0:
                ambiguous_circles_count += 1
                continue
            for shape_id in range(len(per_shape_overlap_ratio)):
                if per_shape_overlap_ratio[shape_id] >= selected_threshold:
                    per_shape_scores[shape_id] = threshold_score(
                        per_shape_overlap_ratio[shape_id],
                        [(0.6, 0.0), (0.7, 1.0)]
                    )
        num_circles = len(circle_selection_info['circle_vs_shape_overlap'])
        if num_circles > 0:
            ambiguous_circles_ratio = ambiguous_circles_count / num_circles
        else:
            ambiguous_circles_ratio = 0.0
        ambiguous_score = threshold_score(
            ambiguous_circles_ratio,
            [(0.2, 0.0), (0.6, 1.0)]
        )
        num_target_shapes = sum(circle_selection_info['is_target_shape'])
        correct_match_score = 0.0
        wrong_match_score = 0.0
        for shape_id in range(len(per_shape_scores)):
            if circle_selection_info['is_target_shape'][shape_id] == 1:
                correct_match_score += per_shape_scores[shape_id] / num_target_shapes
            else:
                wrong_match_score = max(wrong_match_score, per_shape_scores[shape_id])
        scores['match_score'] = max(0, (correct_match_score - wrong_match_score) * (0.5 + 0.5 * background_consistency_score) * (0.5 + 0.5 * foreground_consistency_score) * (0.4 + 0.6 * circle_area_penalty_score) - ambiguous_score)
        task_score = scores['match_score']
        total_score = task_score * (0.6 + 0.4 * scores['consistency_score'])
        
        if debug_dir is not None:
            circle_selection_info.pop('pred_circles')
            circle_selection_info.pop('foreground_shapes')
            circle_selection_info.pop('pred_foreground_shapes')
            debug_info = {
                # all info from circle_selection_info
                **circle_selection_info,
                # all info from scores
                'background_consistency_score': background_consistency_score,
                'foreground_consistency_score': foreground_consistency_score,
                'circle_area_penalty_score': circle_area_penalty_score,
                'ambiguous_circles_count': ambiguous_circles_count,
                'correct_match_score': correct_match_score,
                'wrong_match_score': wrong_match_score,
                **scores,
                "total_score": total_score
            }
            with open(os.path.join(debug_dir, "debug_info.json"), "w") as f:
                json.dump(debug_info, f)
        self._last_task_details = {
            'background_change_ratio': circle_selection_info['background_change_ratio'],
            'foreground_change_ratio': circle_selection_info['foreground_change_ratio'],
            'circle_color_mask_ratio': circle_selection_info['circle_color_mask_ratio'],
            'background_consistency_score': background_consistency_score,
            'foreground_consistency_score': foreground_consistency_score,
            'circle_area_penalty_score': circle_area_penalty_score,
            'per_shape_scores': per_shape_scores,
            'ambiguous_circles_count': ambiguous_circles_count,
            'ambiguous_circles_ratio': ambiguous_circles_ratio,
            'ambiguous_score': ambiguous_score,
            'num_target_shapes': num_target_shapes,
            'correct_match_score': correct_match_score,
            'wrong_match_score': wrong_match_score,
            'consistency_score': scores['consistency_score'],
            'match_score': scores['match_score'],
            'total_score': total_score,
        }
        return total_score


class OutlineShapeEvaluator(BaseEvaluator):
    """
    G-161: Mark the second largest shape.
    """
    TASK_WEIGHTS = {
        'consistency_score': 0.20,
        'match_score': 0.50,
        'shape_score': 0.30
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        use_metafile = False # decide for each task
        meta_file_path = eval_info.get('metafile_path') if use_metafile else None
        debug_dir = eval_info.get('debug_dir')
        if debug_dir is not None:
            shutil.rmtree(debug_dir, ignore_errors=True)
            os.makedirs(debug_dir, exist_ok=True)
        circle_selection_processor = CircleSelectionProcessor(
            meta_file_path=meta_file_path,
            circle_color='red',
            circle_fill_max_ratio=0.9,
            circle_hsv_tolerance=(15, 80, 80),
            foreground_hsv_delta_tolerance=(15.0, 150.0, 150.0),
            background_hsv_delta_tolerance=(15.0, 150.0, 150.0),
            foreground_enlarge_pixels=20
        )
        circle_selection_info = circle_selection_processor.process(gt_first_frame, gt_final_frame, last_frame, debug_dir=debug_dir)
        
        scores = {}
        background_consistency_score = threshold_score(
            circle_selection_info['background_change_ratio'],
            [(0.05, 1.0), (0.2, 0.0)]
        )
        foreground_consistency_score = threshold_score(
            circle_selection_info['foreground_change_ratio'],
            [(0.1, 1.0), (0.3, 0.0)]
        )
        circle_area_penalty_score = threshold_score(
            circle_selection_info['circle_color_mask_ratio'],
            [(0.3, 1.0), (0.5, 0.0)]
        )
        scores['consistency_score'] = (background_consistency_score + foreground_consistency_score + circle_area_penalty_score) / 3

        per_shape_scores = [0.0 for _ in range(len(circle_selection_info['is_target_shape']))]
        ambiguous_circles_count = 0
        selected_threshold = 0.5
        ignored_threshold = 0.25

        target_shape_score = 0.0
        circle_shapes = [circle_selection_info['pred_circles'][circle_id]['type'] for circle_id in range(len(circle_selection_info['pred_circles']))]
        shape_shapes = [circle_selection_info['foreground_shapes'][shape_id]['type'] for shape_id in range(len(circle_selection_info['foreground_shapes']))]
        for circle_id, per_shape_overlap_ratio in enumerate(circle_selection_info['circle_vs_shape_overlap']):
            ambiguous_shapes_count = sum(1 for ratio in per_shape_overlap_ratio if ratio > ignored_threshold and ratio < selected_threshold)
            if ambiguous_shapes_count > 0:
                ambiguous_circles_count += 1
                continue
            for shape_id in range(len(per_shape_overlap_ratio)):
                if per_shape_overlap_ratio[shape_id] >= selected_threshold:
                    cur_match_score = threshold_score(
                        per_shape_overlap_ratio[shape_id],
                        [(0.5, 0.0), (0.7, 1.0)]
                    )
                    per_shape_scores[shape_id] = max(per_shape_scores[shape_id], cur_match_score)
                    if circle_selection_info['is_target_shape'][shape_id] == 1 and circle_shapes[circle_id] == shape_shapes[shape_id]:
                        target_shape_score = max(target_shape_score, cur_match_score)
        num_circles = len(circle_selection_info['circle_vs_shape_overlap'])
        if num_circles > 0:
            ambiguous_circles_ratio = ambiguous_circles_count / num_circles
        else:
            ambiguous_circles_ratio = 0.0
        ambiguous_score = threshold_score(
            ambiguous_circles_ratio,
            [(0.2, 0.0), (0.6, 1.0)]
        )
        num_target_shapes = sum(circle_selection_info['is_target_shape'])
        correct_match_score = 0.0
        wrong_match_score = 0.0
        for shape_id in range(len(per_shape_scores)):
            if circle_selection_info['is_target_shape'][shape_id] == 1:
                correct_match_score += per_shape_scores[shape_id] / num_target_shapes
            else:
                wrong_match_score = max(wrong_match_score, per_shape_scores[shape_id])
        scores['match_score'] = max(0, (correct_match_score - wrong_match_score) * (0.5 + 0.5 * foreground_consistency_score) * (0.4 + 0.6 * circle_area_penalty_score) - ambiguous_score)
        scores['shape_score'] = target_shape_score
        task_score = 0.625 * scores['match_score'] + 0.375 * scores['shape_score']
        total_score = task_score * (0.6 + 0.4 * scores['consistency_score'])
        
        if debug_dir is not None:
            circle_selection_info.pop('pred_circles')
            circle_selection_info.pop('foreground_shapes')
            circle_selection_info.pop('pred_foreground_shapes')
            debug_info = {
                # all info from circle_selection_info
                **circle_selection_info,
                # all info from scores
                'background_consistency_score': background_consistency_score,
                'foreground_consistency_score': foreground_consistency_score,
                'circle_area_penalty_score': circle_area_penalty_score,
                'ambiguous_circles_count': ambiguous_circles_count,
                'correct_match_score': correct_match_score,
                'wrong_match_score': wrong_match_score,
                'circle_shapes': circle_shapes,
                'shape_shapes': shape_shapes,
                'target_shape_score': target_shape_score,
                **scores,
                "total_score": total_score
            }
            with open(os.path.join(debug_dir, "debug_info.json"), "w") as f:
                json.dump(debug_info, f)
        self._last_task_details = {
            'background_change_ratio': circle_selection_info['background_change_ratio'],
            'foreground_change_ratio': circle_selection_info['foreground_change_ratio'],
            'circle_color_mask_ratio': circle_selection_info['circle_color_mask_ratio'],
            'background_consistency_score': background_consistency_score,
            'foreground_consistency_score': foreground_consistency_score,
            'circle_area_penalty_score': circle_area_penalty_score,
            'per_shape_scores': per_shape_scores,
            'ambiguous_circles_count': ambiguous_circles_count,
            'ambiguous_circles_ratio': ambiguous_circles_ratio,
            'ambiguous_score': ambiguous_score,
            'num_target_shapes': num_target_shapes,
            'correct_match_score': correct_match_score,
            'wrong_match_score': wrong_match_score,
            'circle_shapes': circle_shapes,
            'shape_shapes': shape_shapes,
            'target_shape_score': target_shape_score,
            'consistency_score': scores['consistency_score'],
            'match_score': scores['match_score'],
            'shape_score': scores['shape_score'],
            'total_score': total_score,
        }
        return total_score


class SelectLineEvaluator(BaseEvaluator):
    """
    G-167: Select the longest polygon side.
    G-212: find incorrect arrow direction.
    """
    
    TASK_WEIGHTS = {
        'consistency_score': 0.30,
        'match_score': 0.70
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate select next figure increasing task."""
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        use_metafile = True # decide for each task
        meta_file_path = eval_info.get('metafile_path') if use_metafile else None
        debug_dir = eval_info.get('debug_dir')
        if debug_dir is not None:
            shutil.rmtree(debug_dir, ignore_errors=True)
            os.makedirs(debug_dir, exist_ok=True)
        circle_selection_processor = CircleSelectionProcessor(
            meta_file_path=meta_file_path,
            circle_color='red',
            circle_fill_max_ratio=0.9,
            circle_hsv_tolerance=(15, 80, 80),
            foreground_hsv_delta_tolerance=(15.0, 150.0, 150.0),
            background_hsv_delta_tolerance=(15.0, 150.0, 150.0),
            consistency_forground_remove_bg="white",
            foreground_enlarge_pixels=0
        )
        circle_selection_info = circle_selection_processor.process(gt_first_frame, gt_final_frame, last_frame, debug_dir=debug_dir)

        scores = {}
        background_consistency_score = threshold_score(
            circle_selection_info['background_change_ratio'],
            [(0.02, 1.0), (0.05, 0.0)]
        )
        foreground_consistency_score = threshold_score(
            circle_selection_info['foreground_change_ratio'],
            [(0.5, 1.0), (0.8, 0.0)]
        )
        circle_area_penalty_score = threshold_score(
            circle_selection_info['circle_color_mask_ratio'],
            [(0.1, 1.0), (0.2, 0.0)]
        )
        scores['consistency_score'] = (background_consistency_score + foreground_consistency_score + circle_area_penalty_score) / 3

        per_shape_scores = [0.0 for _ in range(len(circle_selection_info['is_target_shape']))]
        selected_ratio_threshold = 3
        selected_area_threshold = 0.01
        ambiguous_circles_count = 0
        for per_shape_overlap_ratio in circle_selection_info['circle_vs_shape_overlap']:
            max_ratio_shape_id = -1
            max_ratio = 0.0
            for shape_id in range(len(per_shape_overlap_ratio)):
                if per_shape_overlap_ratio[shape_id] > max_ratio and per_shape_overlap_ratio[shape_id] > selected_area_threshold:
                    max_ratio = per_shape_overlap_ratio[shape_id]
                    max_ratio_shape_id = shape_id
            if max_ratio_shape_id != -1:
                min_to_others_ratio = float('inf')
                for shape_id in range(len(per_shape_overlap_ratio)):
                    if shape_id != max_ratio_shape_id and per_shape_overlap_ratio[shape_id] > 0.0:
                        min_to_others_ratio = min(min_to_others_ratio, max_ratio / per_shape_overlap_ratio[shape_id])
                if min_to_others_ratio > selected_ratio_threshold:
                    per_shape_scores[max_ratio_shape_id] = 1.0
                else:
                    ambiguous_circles_count += 1
            else:
                ambiguous_circles_count += 1
        num_circles = len(circle_selection_info['circle_vs_shape_overlap'])
        if num_circles > 0:
            ambiguous_circles_ratio = ambiguous_circles_count / num_circles
        else:
            ambiguous_circles_ratio = 0.0
        ambiguous_score = threshold_score(
            ambiguous_circles_ratio,
            [(0.0, 0.0), (0.4, 1.0)]
        )
        num_target_shapes = sum(circle_selection_info['is_target_shape'])
        correct_match_score = 0.0
        wrong_match_score = 0.0
        for shape_id in range(len(per_shape_scores)):
            if circle_selection_info['is_target_shape'][shape_id] == 1:
                correct_match_score += per_shape_scores[shape_id] / num_target_shapes
            else:
                wrong_match_score = max(wrong_match_score, per_shape_scores[shape_id])
        scores['match_score'] = max(0, (correct_match_score - wrong_match_score) - ambiguous_score)
        task_score = scores['match_score']
        total_score = task_score * (0.6 + 0.4 * scores['consistency_score'])
        
        if debug_dir is not None:
            circle_selection_info.pop('pred_circles')
            circle_selection_info.pop('foreground_shapes')
            circle_selection_info.pop('pred_foreground_shapes')
            debug_info = {
                # all info from circle_selection_info
                **circle_selection_info,
                # all info from scores
                'background_consistency_score': background_consistency_score,
                'foreground_consistency_score': foreground_consistency_score,
                'circle_area_penalty_score': circle_area_penalty_score,
                'ambiguous_circles_count': ambiguous_circles_count,
                'per_shape_scores': per_shape_scores,
                'correct_match_score': correct_match_score,
                'wrong_match_score': wrong_match_score,
                **scores,
                "total_score": total_score
            }
            with open(os.path.join(debug_dir, "debug_info.json"), "w") as f:
                json.dump(debug_info, f)
        self._last_task_details = {
            'background_change_ratio': circle_selection_info['background_change_ratio'],
            'foreground_change_ratio': circle_selection_info['foreground_change_ratio'],
            'circle_color_mask_ratio': circle_selection_info['circle_color_mask_ratio'],
            'background_consistency_score': background_consistency_score,
            'foreground_consistency_score': foreground_consistency_score,
            'circle_area_penalty_score': circle_area_penalty_score,
            'per_shape_scores': per_shape_scores,
            'ambiguous_circles_count': ambiguous_circles_count,
            'ambiguous_circles_ratio': ambiguous_circles_ratio,
            'ambiguous_score': ambiguous_score,
            'num_target_shapes': num_target_shapes,
            'correct_match_score': correct_match_score,
            'wrong_match_score': wrong_match_score,
            'consistency_score': scores['consistency_score'],
            'match_score': scores['match_score'],
            'total_score': total_score,
        }
        return total_score


# Export all evaluators
OUT_OF_DOMAIN_50_EVALUATORS = {
    'G-24_separate_objects_no_spin_data-generator': SeparateObjectsNoSpinEvaluator,
    'G-47_multiple_keys_for_one_door_data-generator': MultipleKeysForOneDoorEvaluator,
    'G-54_connecting_color_data-generator': ConnectingColorEvaluator,
    'G-135_select_next_figure_small_large_alternating_sequence_data-generator': SelectNextFigureAlternatingEvaluator,
    'G-136_locate_point_in_overlapping_area_data-generator': LocateAllCorrectPointsEvaluator,
    'G-140_locate_topmost_unobscured_figure_data-generator': LocateTopmostFigureEvaluator,
    'G-160_circle_largest_numerical_value_data-generator': SelectFigureOutOfDomainEvaluator,
    'G-247_identify_chinese_character_data-generator': SelectFigureOutOfDomainEvaluator,
    'G-218_identify_largest_angle_in_triangle_data-generator': SelectFigureOutOfDomainEvaluator,
    'G-212_find_incorrect_arrow_direction_data-generator': SelectLineEvaluator,
    'G-167_select_longest_polygon_side_data-generator': SelectLineEvaluator,
    'G-161_mark_second_largest_shape_data-generator': OutlineShapeEvaluator,
}
