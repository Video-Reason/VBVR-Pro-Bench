"""
Specific evaluators for In-Domain_50 tasks (Part 1).
"""

import numpy as np
import cv2
import os as _os
import json as _json
from collections import deque
from itertools import permutations
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Any
from .base_evaluator import BaseEvaluator
from .Out_of_Domain_50_part1 import SeparateObjectsNoSpinEvaluator as _ShapeMotionBase
from .utils import maze
from .utils.tracking import (
    COLOR_RANGES,
    MultiColorTracker,
    count_moved_tracklets,
    detect_colored_blobs,
    detect_star_markers,
    find_bordered_object,
    find_star_marked_object,
    per_color_hungarian,
    track_video,
)
from ..utils import compute_optical_flow, safe_distance, normalize_frame_size, compute_ssim, \
    detect_closed_contours_by_color, match_contours, COLOR_BOUNDS, \
    score_background_similarity, score_foreground_similarity, \
    extract_patterns_from_gray_bg, find_patterns_in_image, cluster_patterns_by_shape, calculate_list_length_penalty

class StableSortEvaluator(BaseEvaluator):
    """
    G-3: Stable sort objects evaluator.

    Scoring:
    - arrangement     (60%): horizontal alignment (0.3) + grouping by shape (0.3)
                              + within-group size order (0.4)
    - fore_consistency (20%): all reference patterns found in final frame
    - back_consistency (20%): non-pattern area is gray
    """

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
        final_frame = video_frames[-1]

        # Fallbacks
        if gt_first_frame is None and gt_frames:
            gt_first_frame = gt_frames[0]
        if gt_final_frame is None and gt_frames:
            gt_final_frame = gt_frames[-1]

        if gt_final_frame is None:
            return 0.0

        # Resize if needed
        if gt_final_frame.shape[:2] != final_frame.shape[:2]:
            gt_final_frame = cv2.resize(
                gt_final_frame, (final_frame.shape[1], final_frame.shape[0])
            )

        # 1. Extract reference patterns from gt_final_frame
        ref_patterns = extract_patterns_from_gray_bg(gt_final_frame, min_area=500)
        if len(ref_patterns) < 2:
            return 0.0

        # 2. Find matching patterns in final_frame
        matched = find_patterns_in_image(
            gt_final_frame, ref_patterns, final_frame,
            match_threshold=0.8,
        )

        # 2b. Extract patterns directly from final_frame for extra-pattern detection
        gen_patterns = extract_patterns_from_gray_bg(final_frame, min_area=500)

        # 3. Determine cluster assignments using shape clustering
        pat_to_cluster: Dict[int, int] = {}
        groups = cluster_patterns_by_shape(ref_patterns, n_clusters=2)
        for cluster_id, group in enumerate(groups):
            for pat in group:
                for idx, rp in enumerate(ref_patterns):
                    if rp is pat:
                        pat_to_cluster[idx] = cluster_id
                        break

        # Tag each matched item with its cluster_id
        for m in matched:
            m['cluster_id'] = pat_to_cluster.get(m['pattern_idx'], -1)


        # 4. Scores
        fore_consistency = self._evaluate_fore_consistency(ref_patterns, matched, gen_patterns)
        arrangement = self._evaluate_arrangement(matched, ref_patterns, final_frame.shape)
        back_consistency = self._evaluate_back_consistency(final_frame, matched)

        # Consistency is a penalty multiplier
        consistency = 0.5 * fore_consistency + 0.5 * back_consistency
        score = arrangement * (0.6 + 0.4 * consistency)
        self._last_task_details = {
            'arrangement': arrangement,
            'fore_consistency': fore_consistency,
            'back_consistency': back_consistency,
            'consistency': consistency,
        }
        return score

    def _evaluate_fore_consistency(
        self, ref_patterns: List[Dict], matched: List[Dict], gen_patterns: List[Dict]
    ) -> float:
        n_ref = len(ref_patterns)
        if n_ref == 0:
            return 1.0

        # Part 1: missing patterns (matched < ref)
        missing = n_ref - min(len(matched), n_ref)
        if missing == 0:
            score = 1.0
        elif missing == 1:
            score = 0.5
        else:
            score = 0.0

        # Part 2: extra patterns (gen > matched)
        extra = max(0, len(gen_patterns) - len(matched))
        score -= extra * 0.5

        return max(0.0, score)

    def _evaluate_arrangement(self, matched: List[Dict], ref_patterns: List[Dict], frame_shape: Tuple) -> float:
        """
        0.3 horizontal alignment + 0.3 grouping by shape + 0.4 within-group size order.
        """
        if len(matched) < 2:
            return 0.0

        # If matched count != ref_patterns count, grouping and size_order are 0
        if len(matched) != len(ref_patterns):
            # Only calculate horizontal alignment
            h = frame_shape[0]
            sorted_m = sorted(matched, key=lambda m: m['center'][0])

            center_ys = [m['center'][1] for m in sorted_m]
            avg_bbox_h = float(np.mean([m['bbox'][3] for m in sorted_m])) + 1e-6
            avg_y = float(np.mean(center_ys))
            max_dev = max(abs(y - avg_y) for y in center_ys)
            if max_dev > 0.2 * avg_bbox_h:
                horiz_score = 0.3
            else:
                y_std = float(np.std(center_ys))
                rel_std = max(0.0, y_std / avg_bbox_h - 0.10)
                horiz_score = 1.0 / (1.0 + (rel_std * 5) ** 2)

            grouping_score = 0.0
            size_order_score = 0.0
            return 0.3 * horiz_score + 0.3 * grouping_score + 0.4 * size_order_score

        h = frame_shape[0]
        sorted_m = sorted(matched, key=lambda m: m['center'][0])

        center_ys = [m['center'][1] for m in sorted_m]
        avg_bbox_h = float(np.mean([m['bbox'][3] for m in sorted_m])) + 1e-6
        avg_y = float(np.mean(center_ys))
        max_dev = max(abs(y - avg_y) for y in center_ys)
        if max_dev > 0.2 * avg_bbox_h:
            horiz_score = 0.3
        else:
            y_std = float(np.std(center_ys))
            rel_std = max(0.0, y_std / avg_bbox_h - 0.10)
            horiz_score = 1.0 / (1.0 + (rel_std * 5) ** 2)

        cluster_ids = [m['cluster_id'] for m in sorted_m]
        transitions = sum(
            1 for i in range(len(cluster_ids) - 1)
            if cluster_ids[i] != cluster_ids[i + 1]
        )
        grouped = (transitions <= 1)
        grouping_score = 1.0 if grouped else 0.0

        if not grouped:
            size_order_score = 0.0
        else:
            # Split matched into per-cluster lists (already sorted by x)
            cluster_groups: Dict[int, List[Dict]] = {}
            for m in sorted_m:
                cid = m['cluster_id']
                cluster_groups.setdefault(cid, []).append(m)

            group_scores = []
            for cid, members in cluster_groups.items():
                if len(members) < 2:
                    group_scores.append(1.0)
                    continue
                areas = [m['ref_area'] for m in members]
                n = len(areas)
                wrong_adj = sum(
                    1 for i in range(n - 1) if areas[i] > areas[i + 1]
                )
                wrong_ratio = wrong_adj / (n - 1)
                score = (1.0 - wrong_ratio) ** 2
                group_scores.append(score)

            size_order_score = float(np.mean(group_scores)) if group_scores else 0.0
        return 0.3 * horiz_score + 0.3 * grouping_score + 0.4 * size_order_score

    def _evaluate_back_consistency(self, frame: np.ndarray, matched: List[Dict]) -> float:
        h, w = frame.shape[:2]
        fg_mask = np.zeros((h, w), dtype=np.uint8)

        for m in matched:
            x, y, bw, bh = m['bbox']
            x1 = max(0, x - 4)
            y1 = max(0, y - 4)
            x2 = min(w, x + bw + 4)
            y2 = min(h, y + bh + 4)
            fg_mask[y1:y2, x1:x2] = 255

        bg_pixels = frame[fg_mask == 0]
        if len(bg_pixels) == 0:
            return 1.0

        b = bg_pixels[:, 0].astype(np.int32)
        g = bg_pixels[:, 1].astype(np.int32)
        r = bg_pixels[:, 2].astype(np.int32)
        ch_min = np.minimum(np.minimum(b, g), r)
        ch_max = np.maximum(np.maximum(b, g), r)
        non_gray_ratio = float(np.mean(~((ch_min >= 20) & (ch_max <= 245) & ((ch_max - ch_min) < 60))))

        # exp(-k * 0.05) = 0.5  =>  k = ln(2) / 0.05 ≈ 13.86
        k = np.log(2) / 0.05
        return float(max(0.0, min(1.0, np.exp(-k * non_gray_ratio))))

class MultiObjectPlacementEvaluator(BaseEvaluator):
    """
    G-5: Multi-object placement evaluator.

    Tracking-based, per-object scoring:

    1. Norfair multi-color tracker follows each colored object across the
       whole video; identity is preserved through occlusion.
    2. Each GT object gets ONE score in [0, 1] combining placement accuracy,
       path directness, and fidelity. Unmatched GT objects score 0.
       Hallucinated extra tracklets are added as 0-score "ghosts" so they
       dilute the mean (penalizes phantom objects).
    3. Final score = mean(per-object scores) * star_penalty
       where star_penalty = 0.5 ** n_moved_stars
       — moving stars is treated as a hard violation: 1 moved star halves
       the score, 2 moved stars quarter it, etc. Stars are allowed to vanish
       (e.g. occluded/erased on placement) without penalty.
    """

    OBJECT_MIN_AREA = 2000.0     # objects are large filled shapes
    STAR_MIN_AREA = 50.0
    STAR_MAX_AREA = 2500.0
    TRACK_DIST_THRESHOLD = 80.0  # px between adjacent frames
    TRACK_HIT_MAX = 8            # survive ~0.8s full occlusion at 10 fps
    TRACK_INIT_DELAY = 1
    PLACEMENT_RANGE = 80.0       # px; endpoint full-credit radius
    STAR_MOTION_TOLERANCE = 20.0
    STAR_PENALTY_BASE = 0.5      # per moved star: total *= 0.5
    # Per-issue discount factor and pass thresholds
    ISSUE_PENALTY = 0.8
    DIRECTNESS_THRESHOLD = 0.95
    UNIFORMITY_THRESHOLD = 0.65
    AREA_STAB_THRESHOLD  = 0.75
    COMPLETENESS_THRESHOLD = 0.95
    COVERAGE_MIN_TRAVEL = 40.0   # px; skip coverage check when origin≈target
    COVERAGE_SATURATION = 0.9    # coverage ≥ this → treat as 1.0 (tracker init delay slack)
    PROGRESS_ONLY_FLOOR = 0.25
    PATH_QUALITY_FLOOR = 0.5     # final placement / demonstrated path each contribute 50%
    FRAME_COMPLETENESS_FLOOR = 0.7  # interleave only: credit kept when the model
                                     # returns fewer frames than GT demonstrates
    ENDPOINT_DROP_RANGE = 120.0
    SOURCE_ANCHOR_FULL_RANGE = 60.0
    SOURCE_ANCHOR_DROP_RANGE = 160.0
    SOURCE_ANCHOR_RELATIVE_SATURATION = 0.90
    SOURCE_ANCHOR_RELATIVE_FULL_FRAC = 0.10
    PATH_LATERAL_FULL_PX = 20.0
    PATH_LATERAL_DROP_PX = 60.0
    TELEPORT_PX_FRAC = 0.25
    TELEPORT_PENALTY_BASE = 0.25
    MAX_TELEPORT_EVENTS = 6

    def _track_objects(self, frames: List[np.ndarray]) -> MultiColorTracker:
        tracker = MultiColorTracker(
            distance_threshold=self.TRACK_DIST_THRESHOLD,
            hit_counter_max=self.TRACK_HIT_MAX,
            initialization_delay=self.TRACK_INIT_DELAY,
        )
        for i, frame in enumerate(frames):
            dets = detect_colored_blobs(frame, min_area=self.OBJECT_MIN_AREA)
            tracker.update(i, dets)
        return tracker

    _per_color_hungarian = staticmethod(per_color_hungarian)

    def _tracklet_detected_centers(self, tracklet: Any) -> List[Tuple[float, float]]:
        """Actual detection centers only, excluding Kalman-only predictions."""
        return [(float(p.center[0]), float(p.center[1])) for p in tracklet.points if p.detected]

    def _count_teleport_events(
        self,
        centers: Sequence[Tuple[float, float]],
        teleport_px: float,
    ) -> int:
        n = 0
        prev: Optional[Tuple[float, float]] = None
        for center in centers:
            if prev is not None and safe_distance(center, prev) > teleport_px:
                n += 1
            prev = center
        return n

    def _endpoint_score(self, end_dist: float) -> float:
        """Soft endpoint term with the old placement radius as full credit."""
        if end_dist <= self.PLACEMENT_RANGE:
            return 1.0
        return max(0.0, 1.0 - (float(end_dist) - self.PLACEMENT_RANGE) / self.ENDPOINT_DROP_RANGE)

    def _source_anchor_score(self, start_dist: float, expected_disp: Optional[float] = None) -> float:
        if start_dist <= self.SOURCE_ANCHOR_FULL_RANGE:
            fixed_score = 1.0
        else:
            fixed_score = max(
                0.0,
                1.0 - (
                    (float(start_dist) - self.SOURCE_ANCHOR_FULL_RANGE)
                    / self.SOURCE_ANCHOR_DROP_RANGE
                ),
            )
        if expected_disp is None or expected_disp <= 1e-6:
            return fixed_score
        relative_full_range = min(
            self.SOURCE_ANCHOR_FULL_RANGE,
            self.SOURCE_ANCHOR_RELATIVE_FULL_FRAC * float(expected_disp),
        )
        relative_zero_range = max(
            relative_full_range + 1.0,
            self.SOURCE_ANCHOR_RELATIVE_SATURATION * float(expected_disp),
        )
        if start_dist <= relative_full_range:
            relative_score = 1.0
        else:
            relative_score = max(
                0.0,
                1.0 - (
                    (float(start_dist) - relative_full_range)
                    / (relative_zero_range - relative_full_range)
                ),
            )
        return min(fixed_score, relative_score)

    def _path_functional_score(
        self,
        centers: Sequence[Tuple[float, float]],
        origin: Tuple[float, float],
        target: Tuple[float, float],
    ) -> Tuple[float, Dict[str, Any]]:
        """Source-anchored path-functional transport score.

        The signed projected integral itself telescopes to endpoint projection.
        This score uses the non-linear pieces that retain process information:
        source anchoring, max-prefix projected progress, forward/backward
        variation, and p90 lateral deviation from the source->target corridor.
        """
        origin_arr = np.array(origin, dtype=float)
        target_arr = np.array(target, dtype=float)
        expected_disp = float(np.linalg.norm(target_arr - origin_arr))
        details: Dict[str, Any] = {
            "used": False,
            "expected_disp_px": round(expected_disp, 4),
            "source_anchor_dist_px": None,
            "source_anchor_score": 0.0,
            "projected_path_integral_px": 0.0,
            "max_projected_progress_px": 0.0,
            "progress_score": 0.0,
            "positive_projected_px": 0.0,
            "negative_projected_px": 0.0,
            "forward_ratio": 0.0,
            "forward_score": 0.0,
            "lateral_p90_px": 0.0,
            "lateral_score": 1.0,
            "path_score": 0.0,
        }

        if expected_disp <= self.COVERAGE_MIN_TRAVEL:
            details.update({
                "used": False,
                "source_anchor_dist_px": 0.0,
                "source_anchor_score": 1.0,
                "progress_score": 1.0,
                "forward_ratio": 1.0,
                "forward_score": 1.0,
                "path_score": 1.0,
            })
            return 1.0, details

        clean_centers = [np.array(c, dtype=float) for c in centers]
        if len(clean_centers) < 2:
            if clean_centers:
                start_dist = float(np.linalg.norm(clean_centers[0] - origin_arr))
                details["source_anchor_dist_px"] = round(start_dist, 4)
                details["source_anchor_score"] = round(
                    self._source_anchor_score(start_dist, expected_disp), 4,
                )
            return 0.0, details

        unit = (target_arr - origin_arr) / expected_disp
        lateral_unit = np.array([-unit[1], unit[0]], dtype=float)

        start_dist = float(np.linalg.norm(clean_centers[0] - origin_arr))
        source_anchor = self._source_anchor_score(start_dist, expected_disp)

        projected_path_integral = 0.0
        max_projected_progress = 0.0
        positive = 0.0
        negative = 0.0
        lateral_values = []
        prev = clean_centers[0]
        for curr in clean_centers:
            lateral_values.append(abs(float(np.dot(curr - origin_arr, lateral_unit))))
        for curr in clean_centers[1:]:
            projected_step = float(np.dot(curr - prev, unit))
            projected_path_integral += projected_step
            max_projected_progress = max(max_projected_progress, projected_path_integral)
            if projected_step >= 0:
                positive += projected_step
            else:
                negative += -projected_step
            prev = curr

        progress_px = float(np.clip(max_projected_progress, 0.0, expected_disp))
        progress_score = min(1.0, (progress_px / expected_disp) / self.COVERAGE_SATURATION)

        denom = positive + negative
        forward_ratio = positive / denom if denom > 1e-6 else 0.0
        forward_score = forward_ratio

        lateral_p90 = float(np.percentile(lateral_values, 90)) if lateral_values else 0.0
        if lateral_p90 <= self.PATH_LATERAL_FULL_PX:
            lateral_score = 1.0
        else:
            lateral_score = max(
                0.0,
                1.0 - (lateral_p90 - self.PATH_LATERAL_FULL_PX) / self.PATH_LATERAL_DROP_PX,
            )

        path_score = source_anchor * progress_score * forward_score * lateral_score
        details.update({
            "used": True,
            "source_anchor_dist_px": round(float(start_dist), 4),
            "source_anchor_score": round(float(source_anchor), 4),
            "projected_path_integral_px": round(float(projected_path_integral), 4),
            "max_projected_progress_px": round(float(progress_px), 4),
            "progress_score": round(float(progress_score), 4),
            "positive_projected_px": round(float(positive), 4),
            "negative_projected_px": round(float(negative), 4),
            "forward_ratio": round(float(forward_ratio), 4),
            "forward_score": round(float(forward_score), 4),
            "lateral_p90_px": round(float(lateral_p90), 4),
            "lateral_score": round(float(lateral_score), 4),
            "path_score": round(float(path_score), 4),
        })
        return path_score, details

    def _transport_score(self, path_score: float, endpoint_score: float, expected_disp: float) -> float:
        if expected_disp <= self.COVERAGE_MIN_TRAVEL:
            return float(endpoint_score)

        path_weight = self.PATH_QUALITY_FLOOR + ((1.0 - self.PATH_QUALITY_FLOOR) * float(path_score))

        return float(endpoint_score) * path_weight

    def _per_object_scores(
        self,
        last_frame: np.ndarray,
        tracklets: List,
        gt_objects: List[Dict],
        n_frames: int,
        first_frame: Optional[np.ndarray] = None,
        allow_single_step_path: bool = False,
    ) -> Tuple[List[float], int, int, List[Dict[str, Any]]]:
        """Returns (per_object_scores, n_matched, n_hallucinated, details).

        Strategy:
        - Placement uses actual last-frame detections (not tracker beliefs),
          so a glitched final frame correctly counts against the score.
        - Tracklets supply path / fidelity for objects that have one.
        - Each GT object scored independently; final = mean across all.
        - Hallucinated last-frame objects (not matched to any GT) added as
          0-score ghosts to the mean.
        """
        n_gt = len(gt_objects)

        # 1. Detect actual last-frame objects  
        last_dets = detect_colored_blobs(last_frame, min_area=self.OBJECT_MIN_AREA)
        n_last = len(last_dets)

        if n_gt == 0:
            return [0.0] * max(1, n_last), 0, n_last, []

        # 2. Group by color for two independent Hungarian matchings.
        last_by_color: Dict[str, List[Tuple[float, float]]] = {}
        last_idx_by_color: Dict[str, List[int]] = {}
        for i, d in enumerate(last_dets):
            last_by_color.setdefault(d.color, []).append(d.center)
            last_idx_by_color.setdefault(d.color, []).append(i)

        gt_by_color: Dict[str, List[Tuple[float, float]]] = {}
        gt_idx_by_color: Dict[str, List[int]] = {}
        for j, g in enumerate(gt_objects):
            gt_by_color.setdefault(g['color'], []).append(g['center'])
            gt_idx_by_color.setdefault(g['color'], []).append(j)

        track_by_color: Dict[str, List[Tuple[float, float]]] = {}
        track_idx_by_color: Dict[str, List[int]] = {}
        for k, t in enumerate(tracklets):
            ld = t.last_detected
            if ld is None:
                continue
            track_by_color.setdefault(t.color, []).append(ld.center)
            track_idx_by_color.setdefault(t.color, []).append(k)

        # 3a. Match last-frame detections to GT objects (PLACEMENT).
        last_to_gt = self._per_color_hungarian(last_by_color, gt_by_color)
        # 3b. Match last-frame detections to tracklets (LINK for path/fidelity).
        last_to_track = self._per_color_hungarian(last_by_color, track_by_color)
 
        gt_to_last: Dict[int, Tuple[int, float]] = {}
        for (color, gt_local), (last_local, dist) in last_to_gt.items():
            gt_global = gt_idx_by_color[color][gt_local]
            last_global = last_idx_by_color[color][last_local]
            gt_to_last[gt_global] = (last_global, dist)

        last_to_track_global: Dict[int, int] = {}
        for (color, track_local), (last_local, _) in last_to_track.items():
            last_global = last_idx_by_color[color][last_local]
            track_global = track_idx_by_color[color][track_local]
            last_to_track_global[last_global] = track_global

        first_by_color: Dict[str, List[Tuple[float, float]]] = {}
        first_idx_by_color: Dict[str, List[int]] = {}
        first_dets: List = []
        if first_frame is not None:
            first_dets = detect_colored_blobs(first_frame, min_area=self.OBJECT_MIN_AREA)
            for i, d in enumerate(first_dets):
                first_by_color.setdefault(d.color, []).append(d.center)
                first_idx_by_color.setdefault(d.color, []).append(i)
        first_to_gt = self._per_color_hungarian(first_by_color, gt_by_color)
        # gt_idx -> first-frame origin position for that GT object
        gt_origin: Dict[int, Tuple[float, float]] = {}
        for (color, gt_local), (first_local, _) in first_to_gt.items():
            gt_global = gt_idx_by_color[color][gt_local]
            first_global = first_idx_by_color[color][first_local]
            gt_origin[gt_global] = first_dets[first_global].center

        # 4. Score each GT object: source-anchored transport, then quality discounts.
        per_object: List[float] = []
        per_object_details: List[Dict[str, Any]] = []
        matched_last_indices: set = set()
        n_matched = 0
        max_track_len = max(n_frames - self.TRACK_INIT_DELAY, 1)
        for j, g in enumerate(gt_objects):
            if j not in gt_to_last:
                per_object.append(0.0)  # No same-color object in last frame
                per_object_details.append({
                    "gt_index": j,
                    "color": g["color"],
                    "matched": False,
                    "score": 0.0,
                    "reason": "no_same_color_last_detection",
                })
                continue
            last_idx, end_dist = gt_to_last[j]
            matched_last_indices.add(last_idx)
            n_matched += 1

            origin = gt_origin.get(j)
            if origin is None:
                per_object.append(0.0)
                per_object_details.append({
                    "gt_index": j,
                    "color": g["color"],
                    "matched": True,
                    "score": 0.0,
                    "reason": "no_source_origin",
                    "endpoint_dist_px": round(float(end_dist), 4),
                })
                continue

            endpoint_score = self._endpoint_score(end_dist)
            expected_disp = float(np.hypot(
                origin[0] - g['center'][0], origin[1] - g['center'][1]
            ))
            track_idx = last_to_track_global.get(last_idx)
            quality_discount = 1.0
            teleport_events = 0
            teleport_penalty = 1.0
            path_details: Dict[str, Any]

            has_process_evidence = n_frames >= 3

            def _no_process_evidence() -> Tuple[float, Dict[str, Any]]:
                _, det = self._path_functional_score([], origin, g["center"])
                det["path_score"] = 0.0
                det["no_process_evidence"] = True
                return 0.0, det

            if not has_process_evidence:
                path_score, path_details = _no_process_evidence()
            elif track_idx is None:
                if allow_single_step_path:
                    centers = [origin, last_dets[last_idx].center]
                    path_score, path_details = self._path_functional_score(
                        centers, origin, g["center"],
                    )
                else:
                    path_score, path_details = self._path_functional_score(
                        [], origin, g["center"],
                    )
                    quality_discount = 0.0
            else:
                t = tracklets[track_idx]
                centers = self._tracklet_detected_centers(t)
                if allow_single_step_path and len(centers) < 2:
                    centers = [origin, last_dets[last_idx].center]
                path_score, path_details = self._path_functional_score(
                    centers, origin, g["center"],
                )

                if not allow_single_step_path:
                    teleport_px = self.TELEPORT_PX_FRAC * min(last_frame.shape[:2])
                    teleport_events = self._count_teleport_events(centers, teleport_px)
                    teleport_penalty = self.TELEPORT_PENALTY_BASE ** min(
                        teleport_events, self.MAX_TELEPORT_EVENTS,
                    )
                    quality_discount *= teleport_penalty
                if t.directness() < self.DIRECTNESS_THRESHOLD:
                    quality_discount *= self.ISSUE_PENALTY
                if t.speed_uniformity() < self.UNIFORMITY_THRESHOLD:
                    quality_discount *= self.ISSUE_PENALTY
                if t.area_stability() < self.AREA_STAB_THRESHOLD:
                    quality_discount *= self.ISSUE_PENALTY
                completeness = min(1.0, len(t.points) / max_track_len)
                if completeness < self.COMPLETENESS_THRESHOLD and not allow_single_step_path:
                    quality_discount *= self.ISSUE_PENALTY

            transport_score = self._transport_score(path_score, endpoint_score, expected_disp)
            score = transport_score * quality_discount
            per_object.append(score)
            per_object_details.append({
                "gt_index": j,
                "color": g["color"],
                "matched": True,
                "endpoint_dist_px": round(float(end_dist), 4),
                "endpoint_score": round(float(endpoint_score), 4),
                "expected_disp_px": round(float(expected_disp), 4),
                "transport_score": round(float(transport_score), 4),
                "quality_discount": round(float(quality_discount), 4),
                "teleport_events": teleport_events,
                "teleport_penalty": round(float(teleport_penalty), 4),
                "score": round(float(score), 4),
                **path_details,
            })

        # 5. Hallucinated objects in last frame (not matched to any GT).
        n_hallucinated = n_last - len(matched_last_indices)
        per_object.extend([0.0] * n_hallucinated)
        for _ in range(n_hallucinated):
            per_object_details.append({
                "matched": False,
                "score": 0.0,
                "reason": "hallucinated_last_frame_object",
            })

        return per_object, n_matched, n_hallucinated, per_object_details

    def _count_moved_stars(
        self,
        frames: List[np.ndarray],
        reference_frame: Optional[np.ndarray] = None,
    ) -> int:
        """Count first-frame stars whose displacement during their visible
        presence exceeded STAR_MOTION_TOLERANCE. Stars allowed to vanish."""
        if not frames:
            return 0
        star_reference = reference_frame if reference_frame is not None else frames[0]
        if star_reference.shape[:2] != frames[0].shape[:2]:
            star_reference = cv2.resize(star_reference, (frames[0].shape[1], frames[0].shape[0]))

        per_frame_stars = [
            detect_star_markers(f, min_area=self.STAR_MIN_AREA, max_area=self.STAR_MAX_AREA)
            for f in frames
        ]
        first_stars = detect_star_markers(
            star_reference,
            min_area=self.STAR_MIN_AREA,
            max_area=self.STAR_MAX_AREA,
        )
        if not first_stars:
            return 0  # No stars to evaluate

        moved = 0
        for fs in first_stars:
            for stars in per_frame_stars[1:]:
                same_color = [s for s in stars if s.color == fs.color]
                if not same_color:
                    continue  # vanished in this frame — OK
                nearest = min(same_color, key=lambda s: safe_distance(s.center, fs.center))
                if safe_distance(nearest.center, fs.center) >= self.STAR_MOTION_TOLERANCE:
                    moved += 1
                    break  # this star confirmed moved; count once
        return moved

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        if len(video_frames) < 2 or gt_final_frame is None:
            return 0.0

        # Track every object across the video
        tracker = self._track_objects(video_frames)
        tracklets = tracker.active_tracklets(min_length=2)

        # GT objects from final frame (large blobs only)
        gt_dets = detect_colored_blobs(gt_final_frame, min_area=self.OBJECT_MIN_AREA)
        gt_objects = [{'color': d.color, 'center': d.center, 'area': d.area} for d in gt_dets]

        per_obj, n_matched, n_halluc, per_object_details = self._per_object_scores(
            video_frames[-1], tracklets, gt_objects, len(video_frames),
            first_frame=video_frames[0],
        )
        object_score = float(np.mean(per_obj)) if per_obj else 0.0

        n_moved_stars = self._count_moved_stars(
            video_frames,
            reference_frame=gt_first_frame if gt_first_frame is not None else video_frames[0],
        )
        star_penalty = self.STAR_PENALTY_BASE ** n_moved_stars

        total = object_score * (0.4 + 0.6 * star_penalty)

        self._last_task_details = {
            'per_object_scores': [round(s, 3) for s in per_obj],
            'object_score_avg': round(object_score, 3),
            'n_gt_objects': len(gt_objects),
            'n_tracklets': len(tracklets),
            'n_matched': n_matched,
            'n_hallucinated': n_halluc,
            'n_moved_stars': n_moved_stars,
            'star_penalty': round(star_penalty, 3),
            'per_object_details': per_object_details,
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
        """Interleave version: treat ``[input_frame, *pred_images]`` as a
        short video and apply the same per-object tracking score."""
        if not pred_images or gt_final_frame is None:
            return 0.0

        # Normalize frame sizes — input_frame and pred_images may differ.
        H, W = gt_final_frame.shape[:2]
        seq: List[np.ndarray] = []
        if input_frame is not None:
            seq.append(
                cv2.resize(input_frame, (W, H))
                if input_frame.shape[:2] != (H, W) else input_frame
            )
        for p in pred_images:
            seq.append(cv2.resize(p, (W, H)) if p.shape[:2] != (H, W) else p)

        tracker = self._track_objects(seq)
        tracklets = tracker.active_tracklets(min_length=2)

        gt_dets = detect_colored_blobs(gt_final_frame, min_area=self.OBJECT_MIN_AREA)
        gt_objects = [{'color': d.color, 'center': d.center, 'area': d.area} for d in gt_dets]

        per_obj, n_matched, n_halluc, per_object_details = self._per_object_scores(
            seq[-1], tracklets, gt_objects, len(seq),
            first_frame=seq[0],
            allow_single_step_path=True,
        )
        object_score = float(np.mean(per_obj)) if per_obj else 0.0

        n_moved_stars = self._count_moved_stars(seq, reference_frame=seq[0])
        star_penalty = self.STAR_PENALTY_BASE ** n_moved_stars

        total = object_score * (0.4 + 0.6 * star_penalty)

        self._last_task_details = {
            'per_object_scores': [round(s, 3) for s in per_obj],
            'object_score_avg': round(object_score, 3),
            'n_gt_objects': len(gt_objects),
            'n_tracklets': len(tracklets),
            'n_matched': n_matched,
            'n_hallucinated': n_halluc,
            'n_moved_stars': n_moved_stars,
            'star_penalty': round(star_penalty, 3),
            'n_pred_frames': len(pred_images),
            'n_expected_frames': max(1, len(gt_images) - 1),
            'per_object_details': per_object_details,
        }
        return total


class TrackObjectMovementEvaluator(BaseEvaluator):
    """
    G-8: Track object movement evaluator.

    Norfair-tracking based. The first frame identifies *which* colored object
    is "marked" (surrounded by a green border ring); we then follow that
    object's interior color across the video and score:

        score = movement × target_alignment × only_one_moved

    The prompt says the marked object moves horizontally until it is directly
    below the red-star-marked target. Movement credit is still driven by the
    marked object's trajectory over time, but final placement must align with
    the target's x-coordinate. We also keep a shape-preservation gate so
    centroid drift caused by stretching / smearing does not count as a
    successful move.

    Sub-scores in [0, 1]:
    - horizontal_motion : min(1, x_range / MIN_HORIZONTAL_EXCURSION)
    - vertical_stability : max(0, 1 - y_range / MAX_VERTICAL_EXCURSION)
    - shape_preservation : translated IoU × area preservation × aspect preservation
    - target_alignment : final marked-object x near the red-star target x
    - only_one_moved : 0.5 ^ n_others_that_moved — hard penalty per bystander

    `movement = horizontal_motion × vertical_stability × shape_preservation`.
    This rejects mostly vertical motion and cases where the centroid moved
    only because the shape was stretched / deformed in place; target alignment
    then rejects horizontal motion that does not stop below the specified
    target.
    """

    # Multiplicative thresholds
    MIN_HORIZONTAL_EXCURSION = 40.0  # px — horizontal range for full credit
    MAX_VERTICAL_EXCURSION = 60.0  # px — y-range at/above this zeroes direction credit
    VERTICAL_JITTER_TOL = 0.5      # px — ignore codec/tracker sub-pixel y jitter
    BYSTANDER_MOTION_TOL = 20.0   # px — below this counts as stationary
    ATTACH_TOLERANCE = 60.0       # px — tracklets within this mean distance
                                  # of the marked object are treated as
                                  # attachments (annotation ring, HSV shadow)
    OTHER_PENALTY_BASE = 0.5      # per moved bystander: total *= 0.5
    TARGET_ALIGNMENT_FULL = 30.0   # px — final x gap for full target credit
    TARGET_ALIGNMENT_ZERO = 120.0  # px — final x gap where target credit is zero
    DETECTION_GATE_RATIO = 0.5     # marked object must be findable in >= 50% of
                                   # frames for full credit; below that the run is
                                   # scaled down, since there is nothing to grade
    INTERMEDIATE_PROGRESS_MIN = 0.25
    INTERMEDIATE_PROGRESS_MAX = 0.75

    # Detection / tracking config (1024x1024 frames, 10 fps, 6s clips)
    OBJECT_MIN_AREA = 2000.0
    BORDER_MIN_AREA = 500.0
    TRACK_DIST_THRESHOLD = 80.0
    TRACK_HIT_MAX = 8
    TRACK_INIT_DELAY = 1
    SHAPE_IOU_SATURATION = 0.75
    TRANSLATION_LIFT_SATURATION = 0.20
    SHAPE_RATIO_SATURATION = 0.75
    SHAPE_MORPH_KERNEL = 3

    def _marked_reference(self, first_frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Interior blob of the green-bordered object, or None."""
        pair = find_bordered_object(
            first_frame,
            border_color='green',
            min_border_area=self.BORDER_MIN_AREA,
            min_interior_area=500.0,
        )
        if pair is None:
            return None
        border, interior = pair
        return {
            'border_bbox': border.bbox,
            'color': interior.color,
            'center': interior.center,
            'area': interior.area,
            'bbox': interior.bbox,
        }

    def _target_reference(self, reference_frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Red-star-marked target object, or the GT bordered-object fallback."""
        pair = find_star_marked_object(reference_frame, star_color='red')
        if pair is not None:
            host, star = pair
            return {
                'source': 'red_star_host',
                'color': host.color,
                'center': host.center,
                'bbox': host.bbox,
                'star_center': star.center,
                'star_bbox': star.bbox,
            }

        # Fallback for GT final frames where the final marked-object location
        # encodes the same destination but star detection is unavailable.
        bordered = find_bordered_object(
            reference_frame,
            border_color='green',
            min_border_area=self.BORDER_MIN_AREA,
            min_interior_area=500.0,
        )
        if bordered is None:
            return None
        _, interior = bordered
        return {
            'source': 'gt_final_bordered_object',
            'color': interior.color,
            'center': interior.center,
            'bbox': interior.bbox,
            'star_center': None,
            'star_bbox': None,
        }

    def _pick_marked_tracklet(
        self,
        tracker: MultiColorTracker,
        color: str,
        reference_center: Tuple[float, float],
        reference_area: float,
    ):
        """Pick the same-color tracklet that starts closest to the marked blob."""
        candidates = [t for t in tracker.active_tracklets(min_length=2) if t.color == color]
        if not candidates:
            return None

        def _candidate_key(tracklet):
            first_detected = tracklet.first_detected
            if first_detected is None:
                return (float('inf'), float('inf'), float('inf'))
            position_cost = safe_distance(first_detected.center, reference_center)
            area_penalty = abs(first_detected.area - reference_area) / max(reference_area, 1.0)
            coverage_bonus = -sum(1 for p in tracklet.points if p.detected)
            return (position_cost + 40.0 * area_penalty, position_cost, coverage_bonus)

        return min(candidates, key=_candidate_key)

    def _color_mask(self, frame: np.ndarray, color: str) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in COLOR_RANGES.get(color, []):
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        if self.SHAPE_MORPH_KERNEL > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.SHAPE_MORPH_KERNEL, self.SHAPE_MORPH_KERNEL),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _extract_shape(
        self,
        frame: np.ndarray,
        color: str,
        reference_center: Tuple[float, float],
        reference_area: float,
    ) -> Optional[Dict[str, Any]]:
        """Recover the actual contour for one colored object instance."""
        mask = self._color_mask(frame, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = max(500.0, reference_area * 0.4)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            cx = float(moments['m10'] / moments['m00'])
            cy = float(moments['m01'] / moments['m00'])
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
            position_cost = safe_distance((cx, cy), reference_center)
            area_penalty = abs(area - reference_area) / max(reference_area, 1.0)
            candidates.append((
                position_cost + 100.0 * area_penalty,
                {
                    'center': (cx, cy),
                    'area': area,
                    'bbox': (x, y, w, h),
                    'aspect_ratio': float(aspect_ratio),
                    'contour': contour,
                },
            ))

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

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

    def _translated_contour(self, contour: np.ndarray, dx: float, dy: float) -> np.ndarray:
        delta = np.array([[[dx, dy]]], dtype=np.float32)
        shifted = contour.astype(np.float32) + delta
        return np.round(shifted).astype(contour.dtype)

    def _shape_translation_score(
        self,
        start_shape: Optional[Dict[str, Any]],
        end_shape: Optional[Dict[str, Any]],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[float, Dict[str, float]]:
        if start_shape is None or end_shape is None:
            return 0.0, {
                'translated_iou': 0.0,
                'raw_iou': 0.0,
                'translation_lift': 0.0,
                'area_ratio': 0.0,
                'aspect_ratio_similarity': 0.0,
                'iou_score': 0.0,
                'lift_score': 0.0,
                'area_score': 0.0,
                'aspect_score': 0.0,
            }

        dx = end_shape['center'][0] - start_shape['center'][0]
        dy = end_shape['center'][1] - start_shape['center'][1]
        shifted_contour = self._translated_contour(start_shape['contour'], dx, dy)

        translated_iou = self._contour_iou(shifted_contour, end_shape['contour'], frame_shape)
        raw_iou = self._contour_iou(start_shape['contour'], end_shape['contour'], frame_shape)
        translation_lift = max(0.0, translated_iou - raw_iou)

        area_ratio = min(start_shape['area'], end_shape['area']) / max(
            start_shape['area'], end_shape['area'], 1.0,
        )
        aspect_ratio_similarity = min(
            start_shape['aspect_ratio'],
            end_shape['aspect_ratio'],
        ) / max(start_shape['aspect_ratio'], end_shape['aspect_ratio'], 1e-6)

        iou_score = min(1.0, translated_iou / self.SHAPE_IOU_SATURATION)
        lift_score = min(1.0, translation_lift / self.TRANSLATION_LIFT_SATURATION)
        area_score = min(1.0, area_ratio / self.SHAPE_RATIO_SATURATION)
        aspect_score = min(1.0, aspect_ratio_similarity / self.SHAPE_RATIO_SATURATION)
        total = iou_score * lift_score * area_score * aspect_score
        return total, {
            'translated_iou': float(translated_iou),
            'raw_iou': float(raw_iou),
            'translation_lift': float(translation_lift),
            'area_ratio': float(area_ratio),
            'aspect_ratio_similarity': float(aspect_ratio_similarity),
            'iou_score': float(iou_score),
            'lift_score': float(lift_score),
            'area_score': float(area_score),
            'aspect_score': float(aspect_score),
        }

    def _trajectory_motion_score(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
    ) -> Tuple[float, Dict[str, float]]:
        if len(xs) >= 2:
            horizontal_excursion = max(float(x) for x in xs) - min(float(x) for x in xs)
        else:
            horizontal_excursion = 0.0
        if len(ys) >= 2:
            vertical_excursion = max(float(y) for y in ys) - min(float(y) for y in ys)
        else:
            vertical_excursion = 0.0
        horizontal_motion = min(
            1.0,
            horizontal_excursion / self.MIN_HORIZONTAL_EXCURSION,
        )
        if vertical_excursion <= self.VERTICAL_JITTER_TOL:
            vertical_stability = 1.0
        else:
            vertical_stability = max(
                0.0,
                1.0 - vertical_excursion / self.MAX_VERTICAL_EXCURSION,
            )
        return horizontal_motion * vertical_stability, {
            'horizontal_excursion': horizontal_excursion,
            'vertical_excursion': vertical_excursion,
            'horizontal_motion': float(horizontal_motion),
            'vertical_stability': float(vertical_stability),
            'vertical_jitter_tolerance': float(self.VERTICAL_JITTER_TOL),
        }

    def _target_alignment_score(
        self,
        final_center: Tuple[float, float],
        target_center: Tuple[float, float],
    ) -> Tuple[float, Dict[str, float]]:
        x_gap = abs(float(final_center[0]) - float(target_center[0]))
        if x_gap <= self.TARGET_ALIGNMENT_FULL:
            score = 1.0
        elif x_gap >= self.TARGET_ALIGNMENT_ZERO:
            score = 0.0
        else:
            span = self.TARGET_ALIGNMENT_ZERO - self.TARGET_ALIGNMENT_FULL
            score = 1.0 - (x_gap - self.TARGET_ALIGNMENT_FULL) / max(span, 1e-6)
        return float(score), {
            'target_x_gap': float(x_gap),
            'target_alignment_full_px': float(self.TARGET_ALIGNMENT_FULL),
            'target_alignment_zero_px': float(self.TARGET_ALIGNMENT_ZERO),
        }

    def _intermediate_progress_score(
        self,
        centers: Sequence[Tuple[float, float]],
        start_center: Tuple[float, float],
        final_center: Tuple[float, float],
    ) -> Tuple[float, Dict[str, Any]]:
        """Require motion through the broad middle half, not an exact midpoint."""
        dx = float(final_center[0] - start_center[0])
        if abs(dx) < 1.0:
            return 0.0, {
                'progress_values': [],
                'middle_hits': 0,
                'progress_min': self.INTERMEDIATE_PROGRESS_MIN,
                'progress_max': self.INTERMEDIATE_PROGRESS_MAX,
            }

        progress_values = [
            float((center[0] - start_center[0]) / dx)
            for center in centers
        ]
        middle_hits = sum(
            self.INTERMEDIATE_PROGRESS_MIN <= progress <= self.INTERMEDIATE_PROGRESS_MAX
            for progress in progress_values
        )
        return float(middle_hits > 0), {
            'progress_values': [round(value, 4) for value in progress_values],
            'middle_hits': int(middle_hits),
            'progress_min': self.INTERMEDIATE_PROGRESS_MIN,
            'progress_max': self.INTERMEDIATE_PROGRESS_MAX,
        }

    def _endpoint_bystander_moved_count(
        self,
        first: np.ndarray,
        last: np.ndarray,
        marked_ref: Dict[str, Any],
        end_ref: Dict[str, Any],
    ) -> int:
        def _without_marked_attachment(frame: np.ndarray, ref: Dict[str, Any]) -> List[Any]:
            blobs = detect_colored_blobs(frame, min_area=self.OBJECT_MIN_AREA)
            return [
                blob for blob in blobs
                if safe_distance(blob.center, ref['center']) >= self.ATTACH_TOLERANCE
            ]

        first_blobs = _without_marked_attachment(first, marked_ref)
        last_blobs = _without_marked_attachment(last, end_ref)
        used_last: Set[int] = set()
        moved = 0
        for blob in first_blobs:
            candidates = [
                (safe_distance(blob.center, other.center), idx)
                for idx, other in enumerate(last_blobs)
                if idx not in used_last and other.color == blob.color
            ]
            if not candidates:
                moved += 1
                continue
            dist, idx = min(candidates, key=lambda item: item[0])
            used_last.add(idx)
            if dist > self.BYSTANDER_MOTION_TOL:
                moved += 1
        return moved

    def _score_endpoint_fallback(
        self,
        video_frames: List[np.ndarray],
        marked_ref: Dict[str, Any],
        target_ref: Dict[str, Any],
        reason: str,
    ) -> float:
        first = video_frames[0]
        last = video_frames[-1]
        end_ref = self._marked_reference(last)
        if end_ref is None:
            self._last_task_details = {
                'error': 'endpoint_fallback_no_green_bordered_object_in_final_frame',
                'fallback_reason': reason,
                'marked_color': marked_ref['color'],
            }
            return 0.0
        if end_ref['color'] != marked_ref['color']:
            self._last_task_details = {
                'error': 'endpoint_fallback_marked_color_changed',
                'fallback_reason': reason,
                'marked_color': marked_ref['color'],
                'final_marked_color': end_ref['color'],
            }
            return 0.0

        start_center = marked_ref['center']
        final_center_ref = end_ref['center']
        dx_signed = float(final_center_ref[0] - start_center[0])
        dy_signed = float(final_center_ref[1] - start_center[1])
        displacement = float(np.hypot(dx_signed, dy_signed))

        start_shape = self._extract_shape(
            first,
            marked_ref['color'],
            marked_ref['center'],
            marked_ref['area'],
        )
        end_shape = self._extract_shape(
            last,
            marked_ref['color'],
            end_ref['center'],
            end_ref['area'],
        )
        process_motion, trajectory_motion_details = self._trajectory_motion_score(
            [start_center[0], final_center_ref[0]],
            [start_center[1], final_center_ref[1]],
        )
        intermediate_centers: List[Tuple[float, float]] = []
        for frame in video_frames[1:-1]:
            frame_ref = self._marked_reference(frame)
            if frame_ref is not None and frame_ref['color'] == marked_ref['color']:
                intermediate_centers.append(frame_ref['center'])
        intermediate_progress, intermediate_details = self._intermediate_progress_score(
            intermediate_centers,
            start_center,
            final_center_ref,
        )
        rigid_motion, rigid_motion_details = self._shape_translation_score(
            start_shape,
            end_shape,
            first.shape,
        )
        shape_preservation = (
            rigid_motion_details['iou_score']
            * rigid_motion_details['area_score']
            * rigid_motion_details['aspect_score']
        )
        final_center = end_shape['center'] if end_shape is not None else final_center_ref
        target_alignment, target_alignment_details = self._target_alignment_score(
            final_center,
            target_ref['center'],
        )
        process_score = process_motion * intermediate_progress
        n_others_moved = self._endpoint_bystander_moved_count(
            first,
            last,
            marked_ref,
            end_ref,
        )
        only_one_moved = self.OTHER_PENALTY_BASE ** n_others_moved

        completion = (
            shape_preservation
            * (0.1 + 0.9 * target_alignment)
            * (0.2 + 0.8 * only_one_moved)
        )
        total = completion * (0.4 + 0.6 * process_score)

        self._last_task_details = {
            'mode': 'endpoint_fallback',
            'fallback_reason': reason,
            'marked_color': marked_ref['color'],
            'target_color': target_ref['color'],
            'target_reference_source': target_ref['source'],
            'target_center': tuple(round(float(v), 2) for v in target_ref['center']),
            'final_center': tuple(round(float(v), 2) for v in final_center),
            'dx_signed': round(dx_signed, 2),
            'dy_signed': round(dy_signed, 2),
            'displacement': round(displacement, 2),
            'start_shape_found': start_shape is not None,
            'end_shape_found': end_shape is not None,
            'detected_frames': 2,
            'total_frames': len(video_frames),
            'n_others_moved': n_others_moved,
            'horizontal_excursion': round(trajectory_motion_details['horizontal_excursion'], 2),
            'vertical_excursion': round(trajectory_motion_details['vertical_excursion'], 2),
            'horizontal_motion': round(trajectory_motion_details['horizontal_motion'], 3),
            'vertical_stability': round(trajectory_motion_details['vertical_stability'], 3),
            'shape_preservation': round(shape_preservation, 3),
            'target_alignment': round(target_alignment, 3),
            'target_x_gap': round(target_alignment_details['target_x_gap'], 2),
            'rigid_motion_reference': round(rigid_motion, 3),
            'completion': round(completion, 3),
            'process_score': round(process_score, 3),
            'intermediate_progress': round(intermediate_progress, 3),
            'intermediate_progress_details': intermediate_details,
            'only_one_moved': round(only_one_moved, 3),
            'translated_iou': round(rigid_motion_details['translated_iou'], 3),
            'raw_iou': round(rigid_motion_details['raw_iou'], 3),
            'translation_lift': round(rigid_motion_details['translation_lift'], 3),
            'area_ratio': round(rigid_motion_details['area_ratio'], 3),
            'aspect_ratio_similarity': round(rigid_motion_details['aspect_ratio_similarity'], 3),
        }
        return total

    def _score(
        self,
        video_frames: List[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
    ) -> float:
        if len(video_frames) < 2 or gt_final_frame is None:
            return 0.0

        first = video_frames[0]
        last = video_frames[-1]
        marked_ref = self._marked_reference(first)
        if marked_ref is None:
            self._last_task_details = {'error': 'no_green_bordered_object_in_first_frame'}
            return 0.0
        marked_color = marked_ref['color']

        target_ref = self._target_reference(gt_final_frame)
        if target_ref is None:
            self._last_task_details = {'error': 'no_target_reference_in_gt_final_frame'}
            return 0.0

        tracker = track_video(
            video_frames,
            min_area=self.OBJECT_MIN_AREA,
            distance_threshold=self.TRACK_DIST_THRESHOLD,
            hit_counter_max=self.TRACK_HIT_MAX,
            initialization_delay=self.TRACK_INIT_DELAY,
        )
        tracklets = tracker.active_tracklets(min_length=2)
        marked = self._pick_marked_tracklet(
            tracker,
            marked_color,
            marked_ref['center'],
            marked_ref['area'],
        )
        if marked is None:
            return self._score_endpoint_fallback(
                video_frames,
                marked_ref,
                target_ref,
                reason='marked_object_not_tracked',
            )

        detected = [p for p in marked.points if p.detected]
        if len(detected) < 2:
            return self._score_endpoint_fallback(
                video_frames,
                marked_ref,
                target_ref,
                reason='marked_tracklet_has_less_than_two_detections',
            )
        xs = [p.center[0] for p in detected]
        ys = [p.center[1] for p in detected]
        if len(xs) >= 2:
            dx_signed = xs[-1] - xs[0]
            dy_signed = ys[-1] - ys[0]
            displacement = float(np.hypot(dx_signed, dy_signed))
        else:
            dx_signed = 0.0
            dy_signed = 0.0
            displacement = 0.0

        start_shape = self._extract_shape(
            first,
            marked_color,
            marked_ref['center'],
            marked_ref['area'],
        )
        last_detected = marked.last_detected
        end_reference_center = (
            last_detected.center if last_detected is not None else marked.last.center
        )
        end_reference_area = (
            last_detected.area if last_detected is not None and last_detected.area > 0 else marked_ref['area']
        )
        end_shape = self._extract_shape(
            last,
            marked_color,
            end_reference_center,
            end_reference_area,
        )

        process_motion, trajectory_motion_details = self._trajectory_motion_score(
            xs,
            ys,
        )
        intermediate_progress, intermediate_details = self._intermediate_progress_score(
            [point.center for point in detected[1:-1]],
            detected[0].center,
            detected[-1].center,
        )
        rigid_motion, rigid_motion_details = self._shape_translation_score(
            start_shape,
            end_shape,
            first.shape,
        )
        shape_preservation = (
            rigid_motion_details['iou_score']
            * rigid_motion_details['area_score']
            * rigid_motion_details['aspect_score']
        )
        final_center = end_shape['center'] if end_shape is not None else end_reference_center
        target_alignment, target_alignment_details = self._target_alignment_score(
            final_center,
            target_ref['center'],
        )
        process_score = process_motion * intermediate_progress

        n_others_moved = count_moved_tracklets(
            tracklets,
            motion_tolerance=self.BYSTANDER_MOTION_TOL,
            exclude_ids=[marked.track_id],
            reference_tracklet=marked,
            attach_tolerance=self.ATTACH_TOLERANCE,
        )
        only_one_moved = self.OTHER_PENALTY_BASE ** n_others_moved

        detection_ratio = len(detected) / max(len(video_frames), 1)
        detection_gate = min(1.0, detection_ratio / self.DETECTION_GATE_RATIO)
        completion = (
            shape_preservation
            * (0.1 + 0.9 * target_alignment)
            * (0.2 + 0.8 * only_one_moved)
            * detection_gate
        )
        total = completion * (0.4 + 0.6 * process_score)

        self._last_task_details = {
            'marked_color': marked_color,
            'detection_gate': round(float(detection_gate), 4),
            'target_color': target_ref['color'],
            'target_reference_source': target_ref['source'],
            'target_center': tuple(round(float(v), 2) for v in target_ref['center']),
            'final_center': tuple(round(float(v), 2) for v in final_center),
            'dx_signed': round(float(dx_signed), 2),
            'dy_signed': round(float(dy_signed), 2),
            'displacement': round(displacement, 2),
            'start_shape_found': start_shape is not None,
            'end_shape_found': end_shape is not None,
            'detected_frames': len(detected),
            'total_frames': len(video_frames),
            'n_others_moved': n_others_moved,
            'horizontal_excursion': round(trajectory_motion_details['horizontal_excursion'], 2),
            'vertical_excursion': round(trajectory_motion_details['vertical_excursion'], 2),
            'horizontal_motion': round(trajectory_motion_details['horizontal_motion'], 3),
            'vertical_stability': round(trajectory_motion_details['vertical_stability'], 3),
            'shape_preservation': round(shape_preservation, 3),
            'target_alignment': round(target_alignment, 3),
            'target_x_gap': round(target_alignment_details['target_x_gap'], 2),
            'rigid_motion_reference': round(rigid_motion, 3),
            'completion': round(completion, 3),
            'process_score': round(process_score, 3),
            'intermediate_progress': round(intermediate_progress, 3),
            'intermediate_progress_details': intermediate_details,
            'only_one_moved': round(only_one_moved, 3),
            'translated_iou': round(rigid_motion_details['translated_iou'], 3),
            'raw_iou': round(rigid_motion_details['raw_iou'], 3),
            'translation_lift': round(rigid_motion_details['translation_lift'], 3),
            'area_ratio': round(rigid_motion_details['area_ratio'], 3),
            'aspect_ratio_similarity': round(rigid_motion_details['aspect_ratio_similarity'], 3),
        }
        return total

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        return self._score(video_frames, gt_final_frame)

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        if not pred_images or gt_final_frame is None:
            return 0.0
        H, W = gt_final_frame.shape[:2]
        seq: List[np.ndarray] = []
        if input_frame is not None:
            seq.append(
                cv2.resize(input_frame, (W, H)) if input_frame.shape[:2] != (H, W) else input_frame
            )
        for p in pred_images:
            seq.append(cv2.resize(p, (W, H)) if p.shape[:2] != (H, W) else p)
        # (H, W) is gt_final_frame's own size, so no resize of GT is possible here.
        gt_final = gt_final_frame
        return self._score(seq, gt_final)

class IdentifyObjectsInRegionEvaluator(BaseEvaluator):
    """
    G-9: Identify objects in region evaluator.

    Scoring:
    - accuracy        (60%): IoU-based matching of green contours vs GT
    - back_consistency (20%): white background similarity between final and GT frames
    - fore_consistency (20%): non-white non-green foreground similarity
    """

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: np.ndarray,
        eval_info: Dict
    ) -> float:
        """Evaluate identify objects in region task."""
        if len(video_frames) < 1:
            return 0.0

        final_frame = video_frames[-1]
        canvas_size = (final_frame.shape[0], final_frame.shape[1])

        # Compare against the GT video's own last frame (same encoding domain);
        # PNG-vs-video loses thin green contours to mp4 compression.
        gt_ref = gt_frames[-1] if gt_frames else gt_final_frame
        if gt_ref.shape[:2] != final_frame.shape[:2]:
            gt_ref = cv2.resize(
                gt_ref, (final_frame.shape[1], final_frame.shape[0])
            )

        gt_contours  = detect_closed_contours_by_color(gt_ref, COLOR_BOUNDS['green'], max_fill_ratio=0.5)
        gen_contours = detect_closed_contours_by_color(
            final_frame, COLOR_BOUNDS['green'], max_fill_ratio=0.5,
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

        def _inner_color(img, contour):
            m = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(m, [contour], -1, 255, -1)
            cv2.drawContours(m, [contour], -1, 0, 14) 
            px = img[m > 0]
            if px.size == 0:
                return None
            gray = px.mean(axis=1)
            px = px[(gray > 55) & (gray < 240)]  
            if px.size == 0:
                return None
            gpx = px[:, 1].astype(np.int16)
            px = px[(gpx - px[:, 0].astype(np.int16) < 60) | (gpx - px[:, 2].astype(np.int16) < 60)]
            return np.median(px, axis=0) if px.size else None

        gt_inner = [_inner_color(gt_ref, c) for c in gt_contours]
        gen_inner = [_inner_color(final_frame, c) for c in gen_contours]
        per_gt_scores = []
        n_contained = 0
        for gi, gt_cnt in enumerate(gt_contours):
            iou = match_results[gi] if gi < len(match_results) else None
            base = float(iou) if iou is not None else 0.0
            contained = any(
                cv2.pointPolygonTest(gt_cnt, (float(gx), float(gy)), False) >= 0
                for (gx, gy) in gen_centroids
            )
            if not contained and gi < len(gt_inner) and gt_inner[gi] is not None:
                contained = any(
                    gc is not None and float(np.linalg.norm(gt_inner[gi] - gc)) <= 60.0
                    for gc in gen_inner
                )
            if contained:
                n_contained += 1
                per_gt_scores.append(max(base, 0.9))
            else:
                per_gt_scores.append(base)
        accuracy = float(np.mean(per_gt_scores)) if per_gt_scores else 0.0
        accuracy = accuracy * calculate_list_length_penalty(len(gt_contours), max(len(valid_ious), n_contained), len(gen_contours))

        back_consistency = score_background_similarity(gt_ref, final_frame)

        fore_consistency = score_foreground_similarity(
            gt_ref, final_frame, COLOR_BOUNDS['green']
        )

        consistency = 0.5 * back_consistency + 0.5 * fore_consistency
        score = accuracy * (0.6 + 0.4 * consistency)

        self._last_task_details = {
            'accuracy': accuracy,
            'back_consistency': back_consistency,
            'fore_consistency': fore_consistency,
        }
        return score

# ---------------------------------------------------------------------------
# Grid path utilities — re-exported from utils.maze for backward compatibility
# ---------------------------------------------------------------------------

_grid_bfs = maze.grid_bfs
_optimal_cell_set = maze.optimal_cell_set
_grid_state_bfs = maze.grid_state_bfs
_grid_state_reverse_bfs = maze.grid_state_reverse_bfs
_cell_center_px = maze.cell_center_px
_pixel_to_cell = maze.pixel_to_cell


def _meta_path_pixels(eval_info, frame_shape):
    """VBVR-Pro: the GT's actual optimal path is in metadata (avoids pixel
    block/obstacle under-detection). Returns the path as pixel centres, or
    None for v1 (no metadata) -> caller falls back to detection."""
    import json as _json, os as _os
    mp = eval_info.get("metafile_path")
    if isinstance(mp, (list, tuple)):
        mp = next((p for p in mp if p and _os.path.exists(p)), None)
    if not (mp and _os.path.exists(mp)):
        mp = _os.path.join(eval_info.get("gt_path", ""), "metadata.json")
    if not _os.path.exists(mp):
        return None
    try:
        meta = _json.load(open(mp))
    except Exception:
        return None
    sgt = meta.get("semantic_ground_truth") or {}
    params = meta.get("parameters") or {}
    path = sgt.get("path") or params.get("path")
    if not path:
        return None
    grid = sgt.get("grid") or {}
    gs = grid.get("rows") or grid.get("cols") or params.get("grid_size") or 10
    h, w = frame_shape[:2]
    cw, ch = w / float(gs), h / float(gs)
    return [(int((c + 0.5) * cw), int((r + 0.5) * ch)) for c, r in path]


def _meta_coverage(video_frames, ref_px, detector, frame_shape):
    """Expected-position coverage: fraction of the GT meta-path cells visited.
    Each frame, tracks the blob nearest where the agent SHOULD be along the GT
    path at that time (frame fraction -> path index). The expected position
    moves, so it follows the real agent and ignores static visited-cell marks
    (which a 'nearest-previous' tracker would glue onto)."""
    if not video_frames or not ref_px:
        return 1.0
    path_set = {maze.pixel_to_cell(x, y, frame_shape) for x, y in ref_px}
    if not path_set:
        return 1.0
    n, m = len(video_frames), len(ref_px)
    visited = set()
    for fi, frame in enumerate(video_frames):
        blobs = detector(frame)
        if not blobs:
            continue
        exp = ref_px[min(m - 1, int(fi / max(1, n - 1) * (m - 1)))]
        bx, by = min(blobs, key=lambda b: abs(b[0] - exp[0]) + abs(b[1] - exp[1]))
        visited.add(maze.pixel_to_cell(bx, by, frame_shape))
    return len(visited & path_set) / len(path_set)


def _make_expected_path_detector(detector, ref_px, n_frames):
    """Stateful frame->pixel: the blob nearest where the agent SHOULD be along the
    GT path at the current frame. Call once per frame in order (matches
    discontinuity_penalty's list-comprehension). Follows the moving agent and
    ignores static visited-cell marks that glue a nearest-previous tracker."""
    m = len(ref_px)
    state = {"i": -1}

    def _single(frame):
        state["i"] += 1
        blobs = detector(frame)
        if not blobs:
            return None
        exp = ref_px[min(m - 1, int(state["i"] / max(1, n_frames - 1) * (m - 1)))]
        return min(blobs, key=lambda b: abs(b[0] - exp[0]) + abs(b[1] - exp[1]))

    return _single
_detect_grid_structure = maze.detect_grid_colors


class GridNumberSequenceEvaluator(BaseEvaluator):
    """
    G-13: Grid number sequence evaluator.

    Scoring (multiplicative, same family as G-15/G-16/G-18):
      segment_score = proximity × coverage   (within each start→wp / wp→wp / wp→end segment)
      base          = mean(segment_scores)
      final_score   = base × (1 − MAX_PENALTY × discontinuity)

    - proximity: Per-frame Manhattan distance from detected agent(s) to the
      nearest point on the segment's optimal cell set.
    - coverage: Fraction of the segment's shortest-path cells visited by the
      best-blob trajectory.
    - discontinuity: Animation-quality penalty ∈ [0, 1]; deducts at most
      MAX_PENALTY (70%).
    """

    MAX_PENALTY = 0.70  # continuity issues deduct up to 70%
    JITTER_TOL = 0.05  # distances < 5% of cell_size treated as on-path
    PENALTY_FLOOR = 0.05  # penalty below this is detection noise, clamped to 0
    EXTRA_AGENT_PENALTY = 0.20  # multiplicative penalty per extra agent blob
    COVERAGE_GAP_THRESHOLD = 2  # grid-cell Manhattan dist; > this = gap (fill)

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    AGENT_HSV_LOWER = (10, 100, 100)
    AGENT_HSV_UPPER = (25, 255, 255)
    AGENT_MIN_AREA = 50

    def _detect_all_agents(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect all orange blobs and return their centroids."""
        return maze.detect_color_centroids(
            frame, self.AGENT_HSV_LOWER, self.AGENT_HSV_UPPER,
            min_area=self.AGENT_MIN_AREA,
        )

    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect the first orange blob (single agent)."""
        agents = self._detect_all_agents(frame)
        return agents[0] if agents else None

    def _detect_endpoint(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red endpoint (largest red blob, hue wraps around 0/180)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = (cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                    | cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255])))
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None
        return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

    @staticmethod
    def _cell_size(frame: np.ndarray) -> float:
        return maze.cell_size(frame)

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return maze.manhattan(a, b)

    # ------------------------------------------------------------------
    # Core: extract GT path
    # ------------------------------------------------------------------

    def _extract_gt_path(self, gt_frames: List[np.ndarray]) -> List[Tuple[int, int]]:
        """Extract the ground-truth trajectory as a list of agent positions."""
        return maze.extract_trajectory(gt_frames, self._detect_all_agents)

    # ------------------------------------------------------------------
    # Metric 1: path correctness
    # ------------------------------------------------------------------

    def _score_path_correctness(
        self,
        video_frames: List[np.ndarray],
        gt_path: List[Tuple[int, int]],
        cell: float,
        optimal_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> float:
        """Per-frame proximity with best-blob selection and extra-blob penalty."""
        return maze.score_proximity(
            video_frames, gt_path, cell, self._detect_all_agents,
            jitter_tol=self.JITTER_TOL,
            extra_agent_penalty=self.EXTRA_AGENT_PENALTY,
            optimal_cells=optimal_cells,
        )

    def _score_coverage_completion(
        self,
        video_frames: List[np.ndarray],
        ref_points: List[Tuple[int, int]],
        start_cell: Tuple[int, int],
        end_cell: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        cell: float,
    ) -> float:
        """Coverage via path completion: actual / (actual + fill)."""
        return maze.score_coverage_completion(
            video_frames, ref_points, start_cell, end_cell, obstacles, cell,
            self._detect_all_agents,
            gap_threshold=self.COVERAGE_GAP_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # Multi-path: optimal cell set (G-13)
    # ------------------------------------------------------------------

    def _read_waypoint_order(
        self,
        frame: np.ndarray,
        yellow_cells: List[Tuple[int, int]],
        gt_frames: Optional[List[np.ndarray]] = None,
    ) -> List[Tuple[int, int]]:
        """Return yellow waypoint cells sorted by the digit printed on them.

        Primary method: OCR via ddddocr on each yellow cell's crop.

        Fallback 1: infer order from GT agent trajectory.
        Fallback 2: sort by digit bbox width (least reliable).
        """
        if not yellow_cells:
            return []

        h, w = frame.shape[:2]
        cell_h, cell_w = h // 10, w // 10

        # --- Primary: OCR with ddddocr ---
        try:
            import ddddocr
            ocr = ddddocr.DdddOcr(show_ad=False)
            ocr_results: List[Tuple[str, Tuple[int, int]]] = []
            for yc in yellow_cells:
                r, c = yc
                roi = frame[r * cell_h : (r + 1) * cell_h,
                            c * cell_w : (c + 1) * cell_w]
                _, buf = cv2.imencode('.png', roi)
                digit = ocr.classification(buf.tobytes())
                ocr_results.append((digit, yc))
            ocr_results.sort(key=lambda x: x[0])
            return [yc for _, yc in ocr_results]
        except Exception:
            pass

        # --- Fallback 1: infer order from GT trajectory ---
        if gt_frames:
            wp_set = set(yellow_cells)
            ordered: List[Tuple[int, int]] = []
            seen: Set[Tuple[int, int]] = set()
            for gf in gt_frames:
                agent = self._detect_agent(gf)
                if agent is None:
                    continue
                ac = _pixel_to_cell(agent[0], agent[1], gf.shape)
                if ac in wp_set and ac not in seen:
                    seen.add(ac)
                    ordered.append(ac)
                    if len(ordered) == len(yellow_cells):
                        break
            for yc in yellow_cells:
                if yc not in seen:
                    ordered.append(yc)
            return ordered

        # --- Fallback 2: bbox-width heuristic ---
        cell_widths: List[Tuple[int, Tuple[int, int]]] = []
        for yc in yellow_cells:
            r, c = yc
            y1 = r * cell_h + cell_h // 5
            x1 = c * cell_w + cell_w // 5
            y2 = (r + 1) * cell_h - cell_h // 5
            x2 = (c + 1) * cell_w - cell_w // 5
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

            coords = np.where(thresh > 0)
            if len(coords[0]) == 0:
                cell_widths.append((0, yc))
                continue
            bbox_w = int(coords[1].max() - coords[1].min() + 1)
            cell_widths.append((bbox_w, yc))

        cell_widths.sort(key=lambda x: x[0])
        return [cell for _, cell in cell_widths]

    def _compute_optimal_path_info(
        self,
        frame: np.ndarray,
        gt_frames: List[np.ndarray],
    ) -> Optional[Tuple[List[Tuple[int, int]], Dict[int, List[Tuple[int, int]]], int]]:
        """Compute optimal-cell-set path info from the grid structure.

        For G-13 the route is start -> wp1 -> wp2 -> ... -> end.
        Waypoint visit order is read from the digit on each yellow cell
        (narrowest bbox = "1", widest = "3").

        Returns ``(all_optimal_points, optimal_by_dist, total_distance,
        start_cell, end_cell, obstacles)`` or ``None`` when grid
        detection fails.
        """
        grid = _detect_grid_structure(frame)
        obstacles: Set[Tuple[int, int]] = set()  # G-13 has no obstacles

        # Start = cell containing the agent
        agent = self._detect_agent(frame)
        if agent is None:
            return None
        start_cell = _pixel_to_cell(agent[0], agent[1], frame.shape)

        # End = red cell
        end_cells = grid["red"]
        if not end_cells:
            return None
        end_cell = end_cells[0]

        # Waypoints = yellow cells, ordered by reading the digit
        waypoint_cells = grid["yellow"]
        ordered_wps = self._read_waypoint_order(frame, waypoint_cells)

        # Segments: start -> wp1 -> wp2 -> ... -> end
        nodes = [start_cell] + ordered_wps + [end_cell]

        all_points: List[Tuple[int, int]] = []
        by_dist: Dict[int, List[Tuple[int, int]]] = {}
        cum = 0

        for i in range(len(nodes) - 1):
            result = _optimal_cell_set(nodes[i], nodes[i + 1], obstacles)
            if result is None:
                return None
            seg_opt, seg_dist, seg_len = result
            for c in seg_opt:
                px = _cell_center_px(c, frame.shape)
                all_points.append(px)
                by_dist.setdefault(cum + seg_dist[c], []).append(px)
            cum += seg_len

        return all_points, by_dist, cum, start_cell, end_cell, obstacles

    # ------------------------------------------------------------------
    # Metric 2: movement continuity
    # ------------------------------------------------------------------

    DISAPPEAR_RATE_CAP = 1.0  # G-18 lowers this to 0.5 for same-hue occlusion

    def _discontinuity_penalty(
        self,
        video_frames: List[np.ndarray],
        cell: float,
    ) -> float:
        """Cell-based discontinuity penalty in [0, 1].  0 = smooth, 1 = worst."""
        return maze.discontinuity_penalty(
            video_frames, cell, self._detect_agent,
            disappear_cap=self.DISAPPEAR_RATE_CAP,
            penalty_floor=self.PENALTY_FLOOR,
            cell_based=True,
        )

    # ------------------------------------------------------------------
    # Segment helpers
    # ------------------------------------------------------------------

    def _score_segment(
        self,
        video_frames: List[np.ndarray],
        seg_optimal: FrozenSet[Tuple[int, int]],
        seg_points: List[Tuple[int, int]],
        seg_shortest: int,
        dest_cell: Tuple[int, int],
        start_cell: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        cell: float,
    ) -> Tuple[float, bool]:
        """Score a single segment and return (score, reached_dest).

        score = proximity × coverage (within this segment) — matches the
        multiplicative philosophy of G-15 / G-16 / G-18.
        """
        if not video_frames or not seg_points:
            return 0.0, False

        # Strict tracking: per-frame best blob relative to this segment's
        # reference points — avoids phantom blobs inflating the visited set.
        cells_per_frame = maze.best_blob_cells(
            video_frames, self._detect_all_agents, seg_points,
        )
        visited: Set[Tuple[int, int]] = {start_cell, dest_cell}
        reached = False
        for cell_here in cells_per_frame:
            if cell_here is None:
                continue
            visited.add(cell_here)
            if cell_here == dest_cell:
                reached = True

        proximity = maze.score_proximity(
            video_frames, seg_points, cell, self._detect_all_agents,
            jitter_tol=self.JITTER_TOL,
            extra_agent_penalty=0.0,  # segment scoring stays single-agent
            optimal_cells=seg_optimal,  # cell-based: smooth agent between optimal cells stays on-path
        )

        hit = len(visited & seg_optimal)
        denom = seg_shortest + 1
        coverage = min(1.0, hit / denom) if denom > 0 else 0.0

        return proximity * coverage, reached

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
        """Segment-by-segment scoring with optimal cell sets.

        Segments: start → wp1, wp1 → wp2, ..., wpN → end.
        Each segment is scored by proximity × coverage against its own
        optimal cell set (multiplicative, matches G-15 / G-16 / G-18).
        If the agent never reaches waypoint N, all subsequent segments
        score 0.
        Final score = mean(segment scores) × (1 − 0.7 × continuity).
        """
        if not video_frames or gt_final_frame is None:
            return 0.0

        cell = self._cell_size(video_frames[0])
        grid_frame = gt_frames[0] if gt_frames else video_frames[0]
        frame_shape = video_frames[0].shape

        # --- detect waypoints ---
        grid = _detect_grid_structure(grid_frame)
        agent = self._detect_agent(grid_frame)
        if agent is None:
            return 0.0
        start_cell = _pixel_to_cell(agent[0], agent[1], grid_frame.shape)

        end_cells = grid["red"]
        if not end_cells:
            return 0.0
        end_cell = end_cells[0]

        obstacles: Set[Tuple[int, int]] = set()  # G-13 has no obstacles
        waypoint_cells = self._read_waypoint_order(grid_frame, grid["yellow"], gt_frames)
        nodes = [start_cell] + waypoint_cells + [end_cell]

        # --- compute per-segment optimal cell sets ---
        SegInfo = Tuple[FrozenSet[Tuple[int, int]], List[Tuple[int, int]], int]
        seg_info: List[Optional[SegInfo]] = []
        for i in range(len(nodes) - 1):
            result = _optimal_cell_set(nodes[i], nodes[i + 1], obstacles)
            if result is None:
                seg_info.append(None)
            else:
                seg_opt, seg_dist, seg_len = result
                seg_pts = [_cell_center_px(c, frame_shape) for c in seg_opt]
                seg_info.append((seg_opt, seg_pts, seg_len))

        # --- split video frames by waypoint visits ---
        wp_frame_idx: List[Optional[int]] = [0]
        for wi in range(1, len(nodes)):
            found = None
            search_from = wp_frame_idx[-1] or 0
            for fi in range(search_from, len(video_frames)):
                a = self._detect_agent(video_frames[fi])
                if a is not None:
                    ac = _pixel_to_cell(a[0], a[1], frame_shape)
                    if ac == nodes[wi]:
                        found = fi
                        break
            wp_frame_idx.append(found)

        # --- score each segment ---
        seg_scores: List[float] = []
        seg_reached: List[bool] = []
        gate_open = True
        for i, info in enumerate(seg_info):
            if not gate_open or info is None:
                seg_scores.append(0.0)
                seg_reached.append(False)
                if info is None:
                    gate_open = False
                continue

            seg_opt, seg_pts, seg_len = info
            f_start = wp_frame_idx[i] if wp_frame_idx[i] is not None else 0
            f_end = (wp_frame_idx[i + 1] if wp_frame_idx[i + 1] is not None
                     else len(video_frames) - 1)
            seg_frames = video_frames[f_start : f_end + 1]
            if not seg_frames:
                seg_frames = video_frames

            score, reached = self._score_segment(
                seg_frames, seg_opt, seg_pts, seg_len, nodes[i + 1],
                nodes[i], obstacles, cell,
            )
            seg_scores.append(score)
            seg_reached.append(reached)
            if not reached:
                gate_open = False

        base = sum(seg_scores) / len(seg_scores) if seg_scores else 0.0

        # Strict continuity: track the blob closest to *any* segment ref
        # point, not an arbitrary first blob.
        all_ref_points: List[Tuple[int, int]] = []
        for info in seg_info:
            if info is not None:
                _, seg_pts, _ = info
                all_ref_points.extend(seg_pts)
        strict_single = maze.make_strict_single_detector(
            self._detect_all_agents, all_ref_points,
        )
        penalty = maze.discontinuity_penalty(
            video_frames, cell, strict_single,
            disappear_cap=self.DISAPPEAR_RATE_CAP,
            penalty_floor=self.PENALTY_FLOOR,
            cell_based=True,
        )
        continuity_factor = 1.0 - self.MAX_PENALTY * penalty
        task_score = base * continuity_factor

        bg_preservation = maze.background_preservation_frames(
            video_frames, grid_frame,
            detector=self._detect_all_agents,
        )
        # Background preservation is a penalty multiplier
        final_score = task_score * (0.6 + 0.4 * bg_preservation)

        self._last_task_details = {
            "start_cell": list(start_cell),
            "end_cell": list(end_cell),
            "waypoint_cells": [list(c) for c in waypoint_cells],
            "nodes": [list(c) for c in nodes],
            "waypoint_frame_idx": wp_frame_idx,
            "segment_scores": [round(float(s), 4) for s in seg_scores],
            "segment_reached": seg_reached,
            "mean_segment_score": round(float(base), 4),
            "continuity_penalty": round(float(penalty), 4),
            "continuity_factor": round(float(continuity_factor), 4),
            "task_score": round(float(task_score), 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "final_score": round(float(final_score), 4),
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
        """Interleave: walk-gated scoring aligned with the video method.

        - optimal = union of every segment shortest-path cell set
          (start → wp1 → ... → wp_n → end)
        - drawn   = cells with saturated diff vs ``input_frame``
        - score   = proximity × coverage × line_length_factor
                    × 0.5^num_missed_waypoints
          (no walls in G-13; continuity dropped — no animation)
        """
        if not pred_images or input_frame is None:
            self._last_task_details = {"error": "no_input_or_pred"}
            return 0.0

        mp = eval_info.get("metafile_path")
        if isinstance(mp, (list, tuple)):
            mp = next((p for p in mp if p and _os.path.exists(p)), None)
        if not mp or not _os.path.exists(mp):
            mp = _os.path.join(eval_info.get("gt_path", ""), "metadata.json")
        try:
            with open(mp) as f:
                params = _json.load(f).get("parameters", {})
        except Exception:
            self._last_task_details = {"error": "metadata_unavailable"}
            return 0.0
        if "start" not in params or "end" not in params:
            self._last_task_details = {"error": "no_start_end_in_meta"}
            return 0.0
        # metadata stores [col, row]; evaluator cells are (row, col).
        start_cell = (params["start"][1], params["start"][0])
        end_cell = (params["end"][1], params["end"][0])
        waypoints = [
            (x["position"][1], x["position"][0])
            for x in sorted(params.get("number_positions", []),
                            key=lambda z: z["number"])
        ]
        nodes = [start_cell] + waypoints + [end_cell]

        optimal_cells: Set[Tuple[int, int]] = set()
        total_shortest = 0
        for i in range(len(nodes) - 1):
            r = _optimal_cell_set(nodes[i], nodes[i + 1], set())
            if r is None:
                continue
            seg_opt, _d, seg_len = r
            optimal_cells |= set(seg_opt)
            total_shortest += seg_len
        if not optimal_cells:
            self._last_task_details = {"error": "no_optimal_path"}
            return 0.0

        counts = maze.cell_draw_counts(pred_images, input_frame)
        drawn = set(counts)
        pred_mask = maze.pred_diff_mask(pred_images, input_frame)
        cell_components = maze.cell_pixel_components(pred_mask)
        walk = maze.simulate_walk_through_drawn(
            drawn=drawn, start=start_cell, end=end_cell,
            waypoints=waypoints,
            cell_components=cell_components,
        )
        gt_drawn = maze.cells_from_pred_diff(gt_images, input_frame)
        ref_optimal = gt_drawn if gt_drawn else optimal_cells
        task_score, details = maze.score_interleave_walk(
            walk=walk, drawn=drawn, optimal_cells=ref_optimal,
            required_cells=waypoints + [end_cell],
            path_length=len(ref_optimal),
            draw_counts=counts,
        )
        bg_preservation = maze.background_preservation_image(
            pred_images[-1], input_frame, exclude_mask=pred_mask,
        )
        # Background preservation is a penalty multiplier
        final_score = task_score * (0.6 + 0.4 * bg_preservation)
        details.update({
            "start_cell": list(start_cell),
            "end_cell": list(end_cell),
            "waypoint_cells": [list(c) for c in waypoints],
            "total_shortest_distance": total_shortest,
            "task_score": round(float(task_score), 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "final_score": round(float(final_score), 4),
        })
        self._last_task_details = details
        return final_score


class GridAvoidObstaclesEvaluator(BaseEvaluator):
    """
    G-15: Grid avoid obstacles evaluator.

      base = 0.5 × proximity + 0.5 × coverage
      score = base × (1 − 0.7 × discontinuity) × (0.5 if hit obstacle else 1)

    Stepping on any obstacle cell at any point halves the final score.
    """

    MAX_PENALTY = 0.70
    JITTER_TOL = 0.05
    PENALTY_FLOOR = 0.05
    EXTRA_AGENT_PENALTY = 0.20
    COVERAGE_GAP_THRESHOLD = 2
    DISAPPEAR_RATE_CAP = 1.0

    # Wider hue range than G-13 to cover "yellow/orange" agents.
    AGENT_HSV_LOWER = (10, 100, 100)
    AGENT_HSV_UPPER = (35, 255, 255)
    AGENT_MIN_AREA = 50

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_all_agents(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect all yellow/orange agent blobs and return their centroids."""
        return maze.detect_color_centroids(
            frame, self.AGENT_HSV_LOWER, self.AGENT_HSV_UPPER,
            min_area=self.AGENT_MIN_AREA,
        )

    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        agents = self._detect_all_agents(frame)
        return agents[0] if agents else None

    @staticmethod
    def _cell_size(frame: np.ndarray) -> float:
        return maze.cell_size(frame)

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return maze.manhattan(a, b)

    # ------------------------------------------------------------------
    # Core: extract GT path
    # ------------------------------------------------------------------

    def _extract_gt_path(self, gt_frames: List[np.ndarray]) -> List[Tuple[int, int]]:
        return maze.extract_trajectory(gt_frames, self._detect_all_agents)

    # ------------------------------------------------------------------
    # Metric 1: path correctness
    # ------------------------------------------------------------------

    def _score_path_correctness(
        self,
        video_frames: List[np.ndarray],
        gt_path: List[Tuple[int, int]],
        cell: float,
        optimal_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> float:
        return maze.score_proximity(
            video_frames, gt_path, cell, self._detect_all_agents,
            jitter_tol=self.JITTER_TOL,
            extra_agent_penalty=self.EXTRA_AGENT_PENALTY,
            optimal_cells=optimal_cells,
        )

    def _score_coverage_completion(
        self,
        video_frames: List[np.ndarray],
        ref_points: List[Tuple[int, int]],
        start_cell: Tuple[int, int],
        end_cell: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        cell: float,
    ) -> float:
        return maze.score_coverage_completion(
            video_frames, ref_points, start_cell, end_cell, obstacles, cell,
            self._detect_all_agents,
            gap_threshold=self.COVERAGE_GAP_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # Multi-path: optimal cell set (G-15)
    # ------------------------------------------------------------------

    def _compute_optimal_path_info(
        self,
        frame: np.ndarray,
    ) -> Optional[Tuple[List[Tuple[int, int]], Dict[int, List[Tuple[int, int]]], int]]:
        """Compute optimal-cell-set path info from the grid structure.

        For G-15 the route is start -> end avoiding obstacles.

        Returns ``(all_optimal_points, optimal_by_dist, total_distance)``
        or ``None`` when grid detection fails.
        """
        grid = _detect_grid_structure(frame)

        # Start = blue cell. In real samples the yellow agent often occludes most
        # of the blue fill, so fall back to the agent's cell when blue is not
        # detectable on the first frame.
        start_cells = grid["blue"]
        end_cells = grid["red"]
        obstacles: Set[Tuple[int, int]] = set(grid["obstacle"])

        if not start_cells:
            agent = self._detect_agent(frame)
            if agent is not None:
                start_cells = [_pixel_to_cell(agent[0], agent[1], frame.shape)]

        if not start_cells or not end_cells:
            return None
        start_cell = start_cells[0]
        end_cell = end_cells[0]

        result = _optimal_cell_set(start_cell, end_cell, obstacles)
        if result is None:
            return None
        optimal_cells, dist_start, shortest = result

        all_points: List[Tuple[int, int]] = []
        by_dist: Dict[int, List[Tuple[int, int]]] = {}
        for c in optimal_cells:
            px = _cell_center_px(c, frame.shape)
            all_points.append(px)
            by_dist.setdefault(dist_start[c], []).append(px)

        return all_points, by_dist, shortest, start_cell, end_cell, obstacles

    # ------------------------------------------------------------------
    # Metric 2: movement continuity
    # ------------------------------------------------------------------

    def _discontinuity_penalty(
        self,
        video_frames: List[np.ndarray],
        cell: float,
    ) -> float:
        return maze.discontinuity_penalty(
            video_frames, cell, self._detect_agent,
            disappear_cap=self.DISAPPEAR_RATE_CAP,
            penalty_floor=self.PENALTY_FLOOR,
            cell_based=True,
        )

    def _obstacle_hit(
        self,
        video_frames: List[np.ndarray],
        obstacle_cells: Set[Tuple[int, int]],
    ) -> bool:
        return maze.any_agent_on_obstacle(
            video_frames, obstacle_cells, self._detect_all_agents,
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
        """Score path quality with continuity and obstacle-hit penalties."""
        if not video_frames or gt_final_frame is None:
            return 0.0

        cell = self._cell_size(video_frames[0])
        frame_shape = video_frames[0].shape
        grid_frame = gt_frames[0] if gt_frames else gt_final_frame
        obstacle_cells = set(_detect_grid_structure(grid_frame)["obstacle"])

        mode = "gt_path"
        start_cell: Optional[Tuple[int, int]] = None
        end_cell: Optional[Tuple[int, int]] = None
        shortest = None
        ref_points: List[Tuple[int, int]] = []

        opt = self._compute_optimal_path_info(grid_frame)
        if opt is not None:
            mode = "optimal_cell_set"
            all_pts, _by_dist, shortest, start_cell, end_cell, obstacles = opt
            meta_px = _meta_path_pixels(eval_info, frame_shape)
            if meta_px:  # VBVR-Pro: use the GT's actual path (pixel obstacle-detection under-counts)
                all_pts = meta_px
            ref_points = all_pts
            optimal_cells_set = {maze.pixel_to_cell(x, y, frame_shape) for x, y in all_pts}
            proximity = self._score_path_correctness(video_frames, all_pts, cell, optimal_cells=optimal_cells_set)
            if meta_px:
                coverage = _meta_coverage(video_frames, meta_px, self._detect_all_agents, frame_shape)
            else:
                coverage = self._score_coverage_completion(
                    video_frames, all_pts, start_cell, end_cell, obstacles, cell,
                )
        else:
            gt_path = self._extract_gt_path(gt_frames)
            ref_points = gt_path
            proximity = self._score_path_correctness(video_frames, gt_path, cell)
            if gt_path:
                start_cell = _pixel_to_cell(gt_path[0][0], gt_path[0][1], frame_shape)
                end_cell = _pixel_to_cell(gt_path[-1][0], gt_path[-1][1], frame_shape)
                coverage = self._score_coverage_completion(
                    video_frames, gt_path, start_cell, end_cell, obstacle_cells, cell,
                )
            else:
                coverage = 0.0

        strict_single = maze.make_strict_single_detector(
            self._detect_all_agents, ref_points,
        )
        continuity_penalty = maze.discontinuity_penalty(
            video_frames, cell, strict_single,
            disappear_cap=self.DISAPPEAR_RATE_CAP,
            penalty_floor=self.PENALTY_FLOOR,
            cell_based=True,
        )
        continuity_factor = 1.0 - self.MAX_PENALTY * continuity_penalty

        hit_report = maze.obstacle_hit_report(
            video_frames, obstacle_cells, self._detect_all_agents,
            reference_points=ref_points,
        )
        # Each distinct obstacle cell the agent steps on halves the score.
        #   0 hits  -> x1      1 hit  -> x0.5     2 hits -> x0.25     ...
        num_hit_cells = len(hit_report["hit_cells"])
        obstacle_multiplier = 0.5 ** num_hit_cells

        score_without_coverage = proximity * continuity_factor * obstacle_multiplier
        _fs = video_frames[-1].shape
        reached_end = False
        if end_cell is not None:
            for (ax, ay) in self._detect_all_agents(video_frames[-1]):
                fc = maze.pixel_to_cell(ax, ay, _fs)
                if abs(fc[0] - end_cell[0]) + abs(fc[1] - end_cell[1]) <= 1:
                    reached_end = True
                    break
        endpoint_factor = 1.0 if reached_end else 0.35
        _n_last = len(self._detect_all_agents(video_frames[-1]))
        trail_factor = 1.0 if _n_last <= 3 else max(0.2, 1.0 - 0.15 * (_n_last - 3))
        obstacle_keep_factor = 1.0
        if obstacle_cells:
            try:
                _gen_grid = _detect_grid_structure(video_frames[-1])
                _gen_obs = set(_gen_grid["obstacle"])
                if _gen_obs:
                    _kept = sum(1 for c in obstacle_cells if c in _gen_obs)
                    _ratio = _kept / float(len(obstacle_cells))
                    obstacle_keep_factor = 0.3 + 0.7 * _ratio
            except Exception:
                pass
        task_score = (score_without_coverage * coverage * endpoint_factor
                      * trail_factor * obstacle_keep_factor)

        bg_preservation = maze.background_preservation_frames(
            video_frames, grid_frame,
            detector=self._detect_all_agents,
        )
        final_score = task_score * (0.6 + 0.4 * bg_preservation)

        self._last_task_details = {
            "mode": mode,
            "start_cell": list(start_cell) if start_cell is not None else None,
            "end_cell": list(end_cell) if end_cell is not None else None,
            "shortest_distance": shortest,
            "obstacle_cells": [list(c) for c in sorted(obstacle_cells)],
            "num_obstacles_detected": len(obstacle_cells),
            "proximity": round(float(proximity), 4),
            "coverage": round(float(coverage), 4),
            "continuity_penalty": round(float(continuity_penalty), 4),
            "continuity_factor": round(float(continuity_factor), 4),
            "obstacle_hit": hit_report["hit"],
            "obstacle_hit_cells": hit_report["hit_cells"],
            "num_obstacle_hit_cells": num_hit_cells,
            "obstacle_hit_frames": hit_report["hit_frames"],
            "obstacle_per_cell_frames": hit_report["per_cell_frames"],
            "obstacle_multiplier": round(float(obstacle_multiplier), 6),
            "score_without_coverage": round(float(score_without_coverage), 4),
            "task_score": round(float(task_score), 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "final_score": round(float(final_score), 4),
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

        score = proximity × coverage × 0.5^num_wall_hits
        """
        if not pred_images or input_frame is None:
            self._last_task_details = {"error": "no_input_or_pred"}
            return 0.0

        opt = self._compute_optimal_path_info(input_frame)
        if opt is None:
            self._last_task_details = {"error": "grid_detection_failed"}
            return 0.0
        all_points, _by_dist, shortest, start_cell, end_cell, obstacles = opt
        optimal_cells = {_pixel_to_cell(x, y, input_frame.shape) for x, y in all_points}

        counts = maze.cell_draw_counts(pred_images, input_frame)
        drawn = set(counts)
        pred_mask = maze.pred_diff_mask(pred_images, input_frame)
        cell_components = maze.cell_pixel_components(pred_mask)
        walk = maze.simulate_walk_through_drawn(
            drawn=drawn, start=start_cell, end=end_cell,
            cell_components=cell_components,
        )
        task_score, details = maze.score_interleave_walk(
            walk=walk, drawn=drawn, optimal_cells=optimal_cells,
            wall_cells=obstacles,
            path_length=shortest + 1,
            draw_counts=counts,
        )
        bg_preservation = maze.background_preservation_image(
            pred_images[-1], input_frame, exclude_mask=pred_mask,
        )
        # Background preservation is a penalty multiplier
        final_score = task_score * (0.6 + 0.4 * bg_preservation)
        details.update({
            "start_cell": list(start_cell),
            "end_cell": list(end_cell),
            "total_shortest_distance": shortest,
            "task_score": round(float(task_score), 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "final_score": round(float(final_score), 4),
        })
        self._last_task_details = details
        return final_score


class GridGoThroughBlockEvaluator(GridNumberSequenceEvaluator):
    """
    G-16: Grid go through block evaluator.

    Scoring formula (same multiplicative shape as G-15):

      score = proximity
            × (1 − 0.7 × discontinuity)
            × 0.5^num_missed_blocks
            × coverage

    The agent must reach the red end square via a shortest path that visits
    every blue block first.  Since the visit order is not specified, we
    compute shortest paths in a state space over
    ``(grid_cell, visited_blocks_mask)`` and score against the union of all
    optimal states rather than a single GT trajectory.

    The **block miss penalty** halves the score for every required block
    the agent never stood on — mirroring G-15's obstacle-hit halving so
    the two tasks share one penalty philosophy.
    """

    def _detect_required_cells(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Return block cells that must be visited before the end square."""
        return sorted(set(_detect_grid_structure(frame)["blue"]))

    def _compute_optimal_path_info(
        self,
        frame: np.ndarray,
    ) -> Optional[Tuple[List[Tuple[int, int]], Dict[int, List[Tuple[int, int]]], int]]:
        """Compute optimal-cell-set path info in ``(cell, visited_blocks)`` space."""
        grid = _detect_grid_structure(frame)
        start_cells = grid["green"]
        end_cells = grid["red"]
        required_cells = self._detect_required_cells(frame)
        obstacles: Set[Tuple[int, int]] = set()

        if not start_cells:
            agent = self._detect_agent(frame)
            if agent is not None:
                start_cells = [_pixel_to_cell(agent[0], agent[1], frame.shape)]

        if not start_cells or not end_cells:
            return None

        start_cell = start_cells[0]
        end_cell = end_cells[0]

        dist_s, required_bits, all_mask, _, goal_state = _grid_state_bfs(
            start=start_cell,
            end=end_cell,
            required_cells=required_cells,
            obstacles=obstacles,
        )
        if goal_state not in dist_s:
            return None

        dist_e = _grid_state_reverse_bfs(
            end=end_cell,
            required_bits=required_bits,
            all_mask=all_mask,
            obstacles=obstacles,
        )

        shortest = dist_s[goal_state]
        by_dist_cells: Dict[int, Set[Tuple[int, int]]] = {}

        for state, d in dist_s.items():
            if state in dist_e and d + dist_e[state] == shortest:
                cell = state[0]
                by_dist_cells.setdefault(d, set()).add(cell)

        by_dist: Dict[int, List[Tuple[int, int]]] = {}
        all_points: List[Tuple[int, int]] = []
        seen_points: Set[Tuple[int, int]] = set()

        for d, cells in by_dist_cells.items():
            pts = [_cell_center_px(cell, frame.shape) for cell in sorted(cells)]
            by_dist[d] = pts
            for pt in pts:
                if pt not in seen_points:
                    seen_points.add(pt)
                    all_points.append(pt)

        return all_points, by_dist, shortest, start_cell, end_cell, obstacles

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Score path quality with GT-grid-defined optimal-state-set routing.

        score = proximity × (1 − 0.7 × continuity)
              × 0.5^num_missed_blocks × coverage
        """
        if not video_frames or gt_final_frame is None:
            return 0.0

        cell = self._cell_size(video_frames[0])
        frame_shape = video_frames[0].shape
        grid_frame = gt_frames[0] if gt_frames else gt_final_frame

        mode = "gt_path"
        start_cell: Optional[Tuple[int, int]] = None
        end_cell: Optional[Tuple[int, int]] = None
        total_dist = None
        ref_points: List[Tuple[int, int]] = []

        opt = self._compute_optimal_path_info(grid_frame)
        if opt is not None:
            mode = "optimal_cell_set"
            all_pts, _by_dist, total_dist, start_cell, end_cell, obstacles = opt
            meta_px = _meta_path_pixels(eval_info, frame_shape)
            if meta_px:  # VBVR-Pro: use the GT's actual block-visiting path (block-detection under-counts)
                all_pts = meta_px
                start_cell = maze.pixel_to_cell(meta_px[0][0], meta_px[0][1], frame_shape)
                end_cell = maze.pixel_to_cell(meta_px[-1][0], meta_px[-1][1], frame_shape)
                obstacles = set()
            ref_points = all_pts
            optimal_cells_set = {maze.pixel_to_cell(x, y, frame_shape) for x, y in all_pts}
            proximity = self._score_path_correctness(video_frames, all_pts, cell, optimal_cells=optimal_cells_set)
            if meta_px:
                coverage = _meta_coverage(video_frames, meta_px, self._detect_all_agents, frame_shape)
            else:
                coverage = self._score_coverage_completion(
                    video_frames, all_pts, start_cell, end_cell, obstacles, cell,
                )
        else:
            gt_path = self._extract_gt_path(gt_frames)
            ref_points = gt_path
            proximity = self._score_path_correctness(video_frames, gt_path, cell)
            if gt_path:
                start_cell = _pixel_to_cell(gt_path[0][0], gt_path[0][1], frame_shape)
                end_cell = _pixel_to_cell(gt_path[-1][0], gt_path[-1][1], frame_shape)
                coverage = self._score_coverage_completion(
                    video_frames, gt_path, start_cell, end_cell, set(), cell,
                )
            else:
                coverage = 0.0

        # Strict mode: continuity tracks the ref-closest blob, not agents[0].
        if meta_px:
            strict_single = _make_expected_path_detector(
                self._detect_all_agents, meta_px, len(video_frames),
            )
        else:
            strict_single = maze.make_strict_single_detector(
                self._detect_all_agents, ref_points,
            )
        cont_penalty = maze.discontinuity_penalty(
            video_frames, cell, strict_single,
            disappear_cap=self.DISAPPEAR_RATE_CAP,
            penalty_floor=self.PENALTY_FLOOR,
            cell_based=True,
        )
        continuity_factor = 1.0 - self.MAX_PENALTY * cont_penalty

        required_cells = self._detect_required_cells(grid_frame)
        if meta_px:
            _vdet = _make_expected_path_detector(self._detect_all_agents, meta_px, len(video_frames))
            visited = {maze.pixel_to_cell(p[0], p[1], frame_shape)
                       for p in (_vdet(f) for f in video_frames) if p is not None}
        else:
            visited = {
                cell for cell in maze.best_blob_cells(
                    video_frames, self._detect_all_agents, ref_points,
                ) if cell is not None
            }
        visited_required = [c for c in required_cells if c in visited]
        missed_required = [c for c in required_cells if c not in visited]

        # Each missed required block halves the score:
        #   0 missed -> x1      1 -> x0.5     2 -> x0.25     3 -> x0.125 ...
        num_missed_blocks = len(missed_required)
        block_multiplier = 0.5 ** num_missed_blocks

        # Multiplicative structure (same as G-15): proximity sets the base
        # scale; continuity / block-miss / coverage each scale it down.
        score_without_coverage = proximity * continuity_factor * block_multiplier
        # Require reaching the destination (see G-15/G-18). GT unchanged.
        _fs = video_frames[-1].shape
        reached_end = False
        if end_cell is not None:
            for (ax, ay) in self._detect_all_agents(video_frames[-1]):
                fc = maze.pixel_to_cell(ax, ay, _fs)
                if abs(fc[0] - end_cell[0]) + abs(fc[1] - end_cell[1]) <= 1:
                    reached_end = True
                    break
        endpoint_factor = 1.0 if reached_end else 0.35
        _n_last = len(self._detect_all_agents(video_frames[-1]))
        trail_factor = 1.0 if _n_last <= 3 else max(0.2, 1.0 - 0.15 * (_n_last - 3))
        task_score = score_without_coverage * coverage * endpoint_factor * trail_factor

        bg_preservation = maze.background_preservation_frames(
            video_frames, grid_frame,
            detector=self._detect_all_agents,
        )
        # Background preservation is a penalty multiplier
        final_score = task_score * (0.6 + 0.4 * bg_preservation)

        self._last_task_details = {
            "mode": mode,
            "start_cell": list(start_cell) if start_cell is not None else None,
            "end_cell": list(end_cell) if end_cell is not None else None,
            "total_shortest_distance": total_dist,
            "required_blocks": [list(c) for c in required_cells],
            "visited_required": [list(c) for c in visited_required],
            "missed_required": [list(c) for c in missed_required],
            "num_missed_blocks": num_missed_blocks,
            "block_multiplier": round(float(block_multiplier), 6),
            "proximity": round(float(proximity), 4),
            "coverage": round(float(coverage), 4),
            "continuity_penalty": round(float(cont_penalty), 4),
            "continuity_factor": round(float(continuity_factor), 4),
            "score_without_coverage": round(float(score_without_coverage), 4),
            "task_score": round(float(task_score), 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "final_score": round(float(final_score), 4),
        }
        return final_score

    # ------------------------------------------------------------------
    # Interleave evaluation (overrides G-13's inherited method)
    # ------------------------------------------------------------------

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Interleave: cell-based scoring against the GT-drawn path.

        Required cells in G-16 include multiple colours (blue blocks, yellow
        highlights, pink / purple "obstacles" the task actually wants the
        agent to visit). ``_detect_required_cells`` only picks up blue, so
        ``optimal_cells`` from ``_compute_optimal_path_info`` under-counts on
        most samples and GT scores below 1.

        Using the GT-drawn cells as the reference side-steps the incomplete
        required-cell detector: GT scores 1.0 by construction, and a model
        is scored by how many cells of GT's actual path it covered.
        """
        if not pred_images or input_frame is None or not gt_images:
            self._last_task_details = {"error": "no_input_pred_or_gt"}
            return 0.0

        grid = _detect_grid_structure(input_frame)
        start_list = grid.get("green") or []
        end_list = grid.get("red") or []
        if not start_list or not end_list:
            agent = self._detect_agent(input_frame)
            if agent is None or not end_list:
                self._last_task_details = {"error": "no_start_or_end"}
                return 0.0
            start_cell = _pixel_to_cell(agent[0], agent[1], input_frame.shape)
            end_cell = end_list[0]
        else:
            start_cell = start_list[0]
            end_cell = end_list[0]

        H, W = input_frame.shape[:2]
        cell_h, cell_w = H // 10, W // 10
        hsv_in = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)
        landmark_cells: Set[Tuple[int, int]] = set()
        for r in range(10):
            for c in range(10):
                if (r, c) == start_cell or (r, c) == end_cell:
                    continue
                y1 = r * cell_h + cell_h // 5
                y2 = (r + 1) * cell_h - cell_h // 5
                x1 = c * cell_w + cell_w // 5
                x2 = (c + 1) * cell_w - cell_w // 5
                if y2 <= y1 or x2 <= x1:
                    continue
                sat = hsv_in[y1:y2, x1:x2, 1]
                if int(np.sum(sat > 40)) >= (y2 - y1) * (x2 - x1) * 0.40:
                    landmark_cells.add((r, c))
        required_blocks: List[Tuple[int, int]] = sorted(landmark_cells)

        counts = maze.cell_draw_counts(
            pred_images, input_frame, inner_margin=0.0, inner_frac=0.02,
        )
        drawn = set(counts)
        pred_mask = maze.pred_diff_mask(pred_images, input_frame)
        cell_components = maze.cell_pixel_components(pred_mask)
        walk = maze.simulate_walk_through_drawn(
            drawn=drawn, start=start_cell, end=end_cell,
            required=set(required_blocks),
            cell_components=cell_components,
        )

        # Reference for proximity/coverage: GT drawn cells give a valid path
        # through all landmarks on this sample (detector-agnostic). 
        gt_drawn = maze.cells_from_pred_diff(
            gt_images, input_frame, inner_margin=0.0, inner_frac=0.02,
        )
        if not gt_drawn:
            gt_drawn = set(walk or []) | set(required_blocks) | {start_cell, end_cell}

        task_score, details = maze.score_interleave_walk(
            walk=walk, drawn=drawn, optimal_cells=gt_drawn,
            required_cells=required_blocks,
            path_length=len(gt_drawn),
            draw_counts=counts,
        )
        bg_preservation = maze.background_preservation_image(
            pred_images[-1], input_frame, exclude_mask=pred_mask,
        )
        # Background preservation is a penalty multiplier
        final_score = task_score * (0.6 + 0.4 * bg_preservation)
        details.update({
            "start_cell": list(start_cell),
            "end_cell": list(end_cell),
            "required_blocks": [list(c) for c in required_blocks],
            "task_score": round(float(task_score), 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "final_score": round(float(final_score), 4),
        })
        self._last_task_details = details
        return final_score


class GridShortestPathEvaluator(BaseEvaluator):
    """
    G-18: Grid shortest path evaluator.

    Multiplicative scoring, same family as G-13/G-15/G-16:

      score = proximity × (1 − MAX_PENALTY × continuity_penalty) × coverage

    The task is a direct shortest path from a coloured start cell to a
    coloured end cell on a 10×10 grid with no obstacles.  Start/end/agent
    colours vary across samples, so detection is generic: any
    high-saturation *circular* blob is treated as the agent, and the two
    coloured cells are identified as start (contains agent) and end.
    """

    MAX_PENALTY = 0.70
    JITTER_TOL = 0.05
    PENALTY_FLOOR = 0.05
    EXTRA_AGENT_PENALTY = 0.20
    COVERAGE_GAP_THRESHOLD = 2
    MIN_AGENT_AREA = 200  # filter tiny boundary artifacts
    # Agent hue can collide with same-hue coloured cells — cap the
    # disappearance-rate contribution to the discontinuity penalty.
    DISAPPEAR_RATE_CAP = 0.5

    # ------------------------------------------------------------------
    # Detection helpers (generic colour — works for any agent/cell hue)
    # ------------------------------------------------------------------

    _agent_hue: Optional[int] = None  # set by _evaluate_task_specific
    _ref_frame: Optional[np.ndarray] = None  # first frame for diff-based fallback

    def _detect_all_agents(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect the agent via hue + area filtering, with diff fallback.

        When ``_agent_hue`` is set, only saturated blobs matching that
        hue (±15) with area < 60% of cell area are returned.  This
        avoids false positives from coloured cells whose hue differs
        from the agent's.

        If hue-based detection fails and ``_ref_frame`` is set, falls
        back to frame-differencing: compares against the reference
        (first) frame to find the moving agent blob.
        """
        agents = self._detect_agents_by_hue(frame)
        if not agents and self._ref_frame is not None:
            agents = self._detect_agents_by_diff(frame)
        return agents

    def _detect_agents_by_hue(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Hue-based agent detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cell_area = (max(frame.shape[:2]) / 10.0) ** 2

        if self._agent_hue is not None:
            hue = self._agent_hue
            lo, hi = hue - 15, hue + 15
            if lo < 0:
                mask = (cv2.inRange(hsv, np.array([lo + 180, 80, 80]), np.array([180, 255, 255]))
                        | cv2.inRange(hsv, np.array([0, 80, 80]), np.array([hi, 255, 255])))
            elif hi > 180:
                mask = (cv2.inRange(hsv, np.array([lo, 80, 80]), np.array([180, 255, 255]))
                        | cv2.inRange(hsv, np.array([0, 80, 80]), np.array([hi - 180, 255, 255])))
            else:
                mask = cv2.inRange(hsv, np.array([lo, 80, 80]), np.array([hi, 255, 255]))
        else:
            mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        agents: List[Tuple[int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AGENT_AREA or area > cell_area * 0.6:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            agents.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))
        return agents

    def _detect_agents_by_diff(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Diff-based fallback: find agent by comparing to reference frame.

        Returns centroids of changed regions that are agent-sized.
        Filters out static coloured cell positions.
        """
        ref = self._ref_frame
        if ref is None:
            return []
        if ref.shape != frame.shape:
            ref = cv2.resize(ref, (frame.shape[1], frame.shape[0]))
        diff = cv2.absdiff(frame, ref)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        cell_area = (max(frame.shape[:2]) / 10.0) ** 2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        agents: List[Tuple[int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AGENT_AREA or area > cell_area * 0.8:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # Check this blob has saturation (is the agent, not noise)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blob_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
            mean_sat = float(hsv[:, :, 1][blob_mask > 0].mean())
            if mean_sat <= 60:
                continue
            if self._agent_hue is not None:
                hues = hsv[:, :, 0][blob_mask > 0].astype(np.int32)
                d = np.abs(hues - self._agent_hue)
                d = np.minimum(d, 180 - d)  # circular hue distance
                if float(np.median(d)) > 20:
                    continue
            agents.append((cx, cy))
        return agents

    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        agents = self._detect_all_agents(frame)
        return agents[0] if agents else None

    def _detect_agent_hue_from_gt(
        self,
        gt_frames: List[np.ndarray],
        start_cell: Tuple[int, int],
    ) -> Optional[int]:
        """Infer agent hue from GT middle frames (small blob on white cell).

        Falls back to the centre pixel of *start_cell* on frame 0 when
        the agent never separates from coloured cells (e.g. distance=1).
        """
        cell_area = (max(gt_frames[0].shape[:2]) / 10.0) ** 2
        # Primary: find a small blob in middle-third of GT video
        for fi in range(len(gt_frames) // 3, 2 * len(gt_frames) // 3):
            hsv = cv2.cvtColor(gt_frames[fi], cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 50 < area < cell_area * 0.6:
                    blob_mask = np.zeros(gt_frames[fi].shape[:2], dtype=np.uint8)
                    cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
                    return int(np.median(hsv[:, :, 0][blob_mask > 0]))
        # Fallback: centre pixel of start cell on first frame
        h, w = gt_frames[0].shape[:2]
        cell_h, cell_w = h // 10, w // 10
        r, c = start_cell
        cy, cx = r * cell_h + cell_h // 2, c * cell_w + cell_w // 2
        return int(cv2.cvtColor(gt_frames[0], cv2.COLOR_BGR2HSV)[cy, cx, 0])

    @staticmethod
    def _cell_size(frame: np.ndarray) -> float:
        return maze.cell_size(frame)

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return maze.manhattan(a, b)

    # ------------------------------------------------------------------
    # Grid structure detection (generic coloured cells)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_colored_cells(
        frame: np.ndarray, grid_size: int = 10,
    ) -> List[Tuple[int, int]]:
        """Return (row, col) of cells whose inner region has high saturation.

        G-18 cannot rely on named colours because start/end cells pick a random
        hue per sample, so detection is colour-agnostic — pure saturation.
        """
        h, w = frame.shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cells: List[Tuple[int, int]] = []
        for r in range(grid_size):
            for c in range(grid_size):
                y1 = r * cell_h + cell_h // 5
                x1 = c * cell_w + cell_w // 5
                y2 = (r + 1) * cell_h - cell_h // 5
                x2 = (c + 1) * cell_w - cell_w // 5
                roi = hsv[y1:y2, x1:x2]
                if roi[:, :, 1].mean() > 80:
                    cells.append((r, c))
        return cells

    # ------------------------------------------------------------------
    # GT path extraction
    # ------------------------------------------------------------------

    def _extract_gt_path(self, gt_frames: List[np.ndarray]) -> List[Tuple[int, int]]:
        return maze.extract_trajectory(gt_frames, self._detect_all_agents)

    # ------------------------------------------------------------------
    # Scoring (shared helpers with skip-on-undetected + capped disappearance)
    # ------------------------------------------------------------------

    def _score_path_correctness(
        self,
        video_frames: List[np.ndarray],
        gt_path: List[Tuple[int, int]],
        cell: float,
        optimal_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> float:
        """Skip-undetected variant: same-hue occlusion frames don't count as 0.

        When ``optimal_cells`` is passed, proximity uses cell-membership
        instead of pixel distance to reference cell *centres* — a
        smoothly-animated agent between two adjacent optimal cells is
        still on-path and should score 1.0.
        """
        return maze.score_proximity(
            video_frames, gt_path, cell, self._detect_all_agents,
            jitter_tol=self.JITTER_TOL,
            extra_agent_penalty=self.EXTRA_AGENT_PENALTY,
            skip_undetected=True,
            optimal_cells=optimal_cells,
        )

    def _score_coverage_completion(
        self,
        video_frames: List[np.ndarray],
        ref_points: List[Tuple[int, int]],
        start_cell: Tuple[int, int],
        end_cell: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        cell: float,
    ) -> float:
        return maze.score_coverage_completion(
            video_frames, ref_points, start_cell, end_cell, obstacles, cell,
            self._detect_all_agents,
            gap_threshold=self.COVERAGE_GAP_THRESHOLD,
        )

    def _discontinuity_penalty(self, video_frames: List[np.ndarray], cell: float) -> float:
        return maze.discontinuity_penalty(
            video_frames, cell, self._detect_agent,
            disappear_cap=self.DISAPPEAR_RATE_CAP,
            penalty_floor=self.PENALTY_FLOOR,
            cell_based=True,
        )

    # ------------------------------------------------------------------
    # Optimal path info
    # ------------------------------------------------------------------

    def _detect_agent_position_in_frame(
        self,
        frame: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """Detect the agent (small saturated circular blob) and return its cell."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cell_area = (max(frame.shape[:2]) / 10.0) ** 2
        mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AGENT_AREA or area > cell_area * 0.6:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return _pixel_to_cell(cx, cy, frame.shape)
        return None

    def _compute_optimal_path_info(
        self,
        frame: np.ndarray,
        gt_frames: Optional[List[np.ndarray]] = None,
    ) -> Optional[Tuple[List[Tuple[int, int]], Dict[int, List[Tuple[int, int]]], int,
                        Tuple[int, int], Tuple[int, int], Set[Tuple[int, int]]]]:
        """Detect start/end cells from GT trajectory and compute optimal cell set.

        Uses the GT video to reliably determine start (agent position in
        first frames) and end (agent position in last frames) cells.
        Falls back to hue-variance detection when GT is unavailable.
        """
        obstacles: Set[Tuple[int, int]] = set()

        # -- Primary: use GT trajectory to determine start/end cells ------
        if gt_frames and len(gt_frames) >= 2:
            colored = list(self._detect_colored_cells(frame))

            if len(colored) >= 2:
                # Collect agent positions from middle GT frames (agent is on
                # white cells → easy to detect as a small saturated blob).
                mid_start = len(gt_frames) // 4
                mid_end = 3 * len(gt_frames) // 4
                mid_positions: List[Tuple[int, int]] = []
                for f in gt_frames[mid_start:mid_end]:
                    pos = self._detect_agent_position_in_frame(f)
                    if pos is not None:
                        mid_positions.append(pos)

                if mid_positions:
                    # The colored cell closer to early mid-positions is start;
                    # the one closer to late mid-positions is end.
                    early_pos = mid_positions[0]
                    late_pos = mid_positions[-1]

                    def _cell_dist(a: Tuple[int,int], b: Tuple[int,int]) -> int:
                        return abs(a[0]-b[0]) + abs(a[1]-b[1])

                    # Pick start: colored cell nearest to early agent position
                    colored_sorted_by_early = sorted(colored, key=lambda c: _cell_dist(c, early_pos))
                    start_cell = colored_sorted_by_early[0]
                    # Pick end: colored cell nearest to late agent position (excluding start)
                    remaining = [c for c in colored if c != start_cell]
                    if remaining:
                        end_cell = min(remaining, key=lambda c: _cell_dist(c, late_pos))
                    else:
                        end_cell = colored_sorted_by_early[-1]

                    if start_cell != end_cell:
                        result = _optimal_cell_set(start_cell, end_cell, obstacles)
                        if result is not None:
                            optimal_cells, dist_start, shortest = result
                            all_points: List[Tuple[int, int]] = []
                            by_dist: Dict[int, List[Tuple[int, int]]] = {}
                            for c in optimal_cells:
                                px = _cell_center_px(c, frame.shape)
                                all_points.append(px)
                                by_dist.setdefault(dist_start[c], []).append(px)
                            return all_points, by_dist, shortest, start_cell, end_cell, obstacles

        # -- Fallback: hue-variance on the grid frame ---------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))
        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs: List[Tuple[float, Tuple[int, int], Tuple[int, int]]] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            blob_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
            hues = hsv[:, :, 0][blob_mask > 0]
            hue_std = float(np.std(hues))
            cell = _pixel_to_cell(cx, cy, frame.shape)
            blobs.append((hue_std, (cx, cy), cell))

        if len(blobs) < 2:
            return None

        blobs.sort(key=lambda x: x[0], reverse=True)
        start_cell = blobs[0][2]
        end_cell = blobs[-1][2]
        if start_cell == end_cell:
            return None

        result = _optimal_cell_set(start_cell, end_cell, obstacles)
        if result is None:
            return None
        optimal_cells, dist_start, shortest = result

        all_points = []
        by_dist = {}
        for c in optimal_cells:
            px = _cell_center_px(c, frame.shape)
            all_points.append(px)
            by_dist.setdefault(dist_start[c], []).append(px)

        return all_points, by_dist, shortest, start_cell, end_cell, obstacles

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
        """score = proximity × (1 − 0.7 × continuity) × coverage"""
        if not video_frames or gt_final_frame is None:
            return 0.0

        cell = self._cell_size(video_frames[0])
        frame_shape = video_frames[0].shape
        grid_frame = gt_frames[0] if gt_frames else gt_final_frame

        mode = "gt_path"
        start_cell: Optional[Tuple[int, int]] = None
        end_cell: Optional[Tuple[int, int]] = None
        shortest = None
        ref_points: List[Tuple[int, int]] = []

        opt = self._compute_optimal_path_info(grid_frame, gt_frames)
        self._ref_frame = video_frames[0]
        try:
            if opt is not None:
                mode = "optimal_cell_set"
                all_pts, _bd, shortest, start_cell, end_cell, obstacles = opt
                ref_points = all_pts
                optimal_cells_set = {
                    maze.pixel_to_cell(x, y, frame_shape) for x, y in all_pts
                }
                self._agent_hue = self._detect_agent_hue_from_gt(
                    gt_frames, start_cell,
                )
                proximity = self._score_path_correctness(
                    video_frames, all_pts, cell,
                    optimal_cells=optimal_cells_set,
                )
                coverage = self._score_coverage_completion(
                    video_frames, all_pts, start_cell, end_cell, obstacles, cell,
                )
            else:
                gt_path = self._extract_gt_path(gt_frames)
                ref_points = gt_path
                proximity = self._score_path_correctness(
                    video_frames, gt_path, cell,
                )
                if gt_path:
                    start_cell = _pixel_to_cell(gt_path[0][0], gt_path[0][1], frame_shape)
                    end_cell = _pixel_to_cell(gt_path[-1][0], gt_path[-1][1], frame_shape)
                    coverage = self._score_coverage_completion(
                        video_frames, gt_path, start_cell, end_cell, set(), cell,
                    )
                else:
                    coverage = 0.0

            strict_single = maze.make_strict_single_detector(
                self._detect_all_agents, ref_points,
            )
            penalty = maze.discontinuity_penalty(
                video_frames, cell, strict_single,
                disappear_cap=self.DISAPPEAR_RATE_CAP,
                penalty_floor=self.PENALTY_FLOOR,
                cell_based=True,
                trim_edge_gaps=True,
            )
            continuity_factor = 1.0 - self.MAX_PENALTY * penalty
            score_without_coverage = proximity * continuity_factor
            # Reaching the destination is required
            reached_end = False
            if end_cell is not None:
                for (ax, ay) in self._detect_all_agents(video_frames[-1]):
                    fc = maze.pixel_to_cell(ax, ay, frame_shape)
                    if abs(fc[0] - end_cell[0]) + abs(fc[1] - end_cell[1]) <= 1:
                        reached_end = True
                        break
            endpoint_factor = 1.0 if reached_end else 0.35
            _n_last = len(self._detect_all_agents(video_frames[-1]))
            trail_factor = 1.0 if _n_last <= 3 else max(0.2, 1.0 - 0.15 * (_n_last - 3))
            task_score = score_without_coverage * coverage * endpoint_factor * trail_factor

            bg_preservation = maze.background_preservation_frames(
                video_frames, grid_frame,
                detector=self._detect_all_agents,
            )
            # Background preservation is a penalty multiplier
            final_score = task_score * (0.6 + 0.4 * bg_preservation)

            self._last_task_details = {
                "mode": mode,
                "start_cell": list(start_cell) if start_cell is not None else None,
                "end_cell": list(end_cell) if end_cell is not None else None,
                "shortest_distance": shortest,
                "agent_hue": self._agent_hue,
                "proximity": round(float(proximity), 4),
                "coverage": round(float(coverage), 4),
                "continuity_penalty": round(float(penalty), 4),
                "continuity_factor": round(float(continuity_factor), 4),
                "score_without_coverage": round(float(score_without_coverage), 4),
                "task_score": round(float(task_score), 4),
                "bg_preservation": round(float(bg_preservation), 4),
                "final_score": round(float(final_score), 4),
            }
            return final_score
        finally:
            self._agent_hue = None
            self._ref_frame = None

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

        score = proximity × coverage
        (G-18 has no walls / required cells; continuity dropped.)
        """
        if not pred_images or input_frame is None:
            self._last_task_details = {"error": "no_input_or_pred"}
            return 0.0

        opt = self._compute_optimal_path_info(input_frame, gt_images)
        if opt is None:
            # GT-trajectory fallback: derive optimal cells from GT images.
            gt_path = self._extract_gt_path(gt_images) if gt_images else []
            if not gt_path:
                self._last_task_details = {"error": "grid_detection_failed"}
                return 0.0
            optimal_cells = {
                _pixel_to_cell(x, y, input_frame.shape) for x, y in gt_path
            }
            start_cell = _pixel_to_cell(gt_path[0][0], gt_path[0][1], input_frame.shape)
            end_cell = _pixel_to_cell(gt_path[-1][0], gt_path[-1][1], input_frame.shape)
            shortest = len(optimal_cells) - 1 if optimal_cells else None
            mode = "gt_path"
        else:
            all_points, _by_dist, shortest, start_cell, end_cell, _obstacles = opt
            optimal_cells = {
                _pixel_to_cell(x, y, input_frame.shape) for x, y in all_points
            }
            mode = "optimal_cell_set"

        counts = maze.cell_draw_counts(pred_images, input_frame)
        drawn = set(counts)
        pred_mask = maze.pred_diff_mask(pred_images, input_frame)
        cell_components = maze.cell_pixel_components(pred_mask)
        walk = maze.simulate_walk_through_drawn(
            drawn=drawn, start=start_cell, end=end_cell,
            cell_components=cell_components,
        )
        task_score, details = maze.score_interleave_walk(
            walk=walk, drawn=drawn, optimal_cells=optimal_cells,
            path_length=(shortest + 1) if shortest is not None else None,
            draw_counts=counts,
        )
        bg_preservation = maze.background_preservation_image(
            pred_images[-1], input_frame, exclude_mask=pred_mask,
        )
        # Background preservation is a penalty multiplier
        final_score = task_score * (0.6 + 0.4 * bg_preservation)
        details.update({
            "mode": mode,
            "start_cell": list(start_cell),
            "end_cell": list(end_cell),
            "shortest_distance": shortest,
            "task_score": round(float(task_score), 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "final_score": round(float(final_score), 4),
        })
        self._last_task_details = details
        return final_score


class MultipleOcclusionsVerticalEvaluator(BaseEvaluator):
    """
    G-21: Multiple occlusions vertical evaluator.

    Dimensions:
        - mask_path_vadility (50%): Gray mask moves downward, exits frame bottom, and remains shape/size/color-consistent.
        - occlusion_correctness (30%): During overlap with the object band, target objects are actually occluded.
        - elements_preservation (20%): Foreground and background stay stable outside occlusion regions.
    """
    
    TASK_WEIGHTS = {
        'mask_path_vadility': 0.50,
        'occlusion_correctness': 0.30,
        'elements_preservation': 0.20,
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
    
    def _detect_mask(
        self,
        frame: np.ndarray,
    ) -> Optional[Dict]:
        """Detect rectangular mask and extract features."""
        # gray mask
        b, g, r = frame[..., 0].astype(np.int16), frame[..., 1].astype(np.int16), frame[..., 2].astype(np.int16)
        max_c = np.maximum(np.maximum(b, g), r)
        min_c = np.minimum(np.minimum(b, g), r)
        is_gray = (max_c - min_c) <= 15
        mean_c = (b + g + r) // 3
        in_range = (mean_c >= 110) & (mean_c <= 235)
        mask_bin = (is_gray & in_range).astype(np.uint8) * 255

        best_cnt, best_score = None, 0.0
        best_box = (0, 0, 0, 0)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area < 100:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            rect_area = cw * ch
            if rect_area == 0:
                continue
            fill_ratio = cnt_area / rect_area
            if fill_ratio < 0.80:
                continue
            score = cnt_area * fill_ratio
            if score > best_score:
                best_score = score
                best_cnt = cnt
                best_box = (x, y, cw, ch)

        if best_cnt is None:
            return None

        x, y, cw, ch = best_box
        cnt_area = cv2.contourArea(best_cnt)
        approx = cv2.approxPolyDP(best_cnt, 0.04 * cv2.arcLength(best_cnt, True), True)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + cw), min(frame.shape[0], y + ch)
        frame_crop = frame[y1:y2, x1:x2]
        return {
            'bbox': (x, y, cw, ch),
            'contour': best_cnt,
            'vertex_count': len(approx),
            'area': cnt_area,
            'bbox_extent': cnt_area / max(cw * ch, 1),
            'mean_bgr': frame_crop.mean(axis=(0, 1)) if frame_crop.size > 0 else np.zeros(3),
        }

    def _track_mask(
        self,
        video_frames: List[np.ndarray],
    ) -> List[Optional[Dict]]:
        """Track mask across all frames."""
        masks_info = []
        for frame_index, frame in enumerate(video_frames):
            mask_feature = self._detect_mask(frame)
            masks_info.append({
                'frame_index': frame_index,
                'mask_feature': mask_feature,
            })
        return masks_info

    def _detect_colored_objects(
        self, 
        frame: np.ndarray,
        search_region: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Dict]:
        """Detect colored objects."""
        if search_region:
            rx, ry, rw, rh = search_region
            roi = frame[ry:ry + rh, rx:rx + rw]
        else:
            rx, ry = 0, 0
            roi = frame

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00']) + rx
            cy = int(M['m01'] / M['m00']) + ry
            x, y, bw, bh = cv2.boundingRect(cnt)
            objects.append({'center': (cx, cy), 'bbox': (x + rx, y + ry, bw, bh), 'area': area})
        return objects

    def _evaluate_mask_path_vadility(
        self,
        masks_info: List[Optional[Dict]],
        frame_height: int,
        frame_width: int,
    ) -> float:
        """Evaluate mask path vadility. mask_features: per-frame _detect_mask() results."""
        if not masks_info:
            return 0.0

        # a) Downward motion: y-center monotonically non-decreasing; x-center should stay fixed
        centers = []
        for mask_info in masks_info:
            if mask_info['mask_feature'] is None:
                continue
            bx, by, bw, bh = mask_info['mask_feature']['bbox']
            centers.append((bx + bw / 2, by + bh / 2))
        if len(centers) >= 2:
            # Vertical: each y should be >= previous y (allow ±5px jitter)
            monotonic_pairs = sum(
                1 for k in range(1, len(centers)) if centers[k][1] >= centers[k - 1][1] - 5
            )
            vertical_score = monotonic_pairs / (len(centers) - 1)
            # Horizontal: x deviation from mean should be small (penalise drift)
            x_values = [c[0] for c in centers]
            x_mean = float(np.mean(x_values))
            x_tolerance = frame_width * 0.05  # 5% of frame width
            x_stable = [1.0 if abs(x - x_mean) <= x_tolerance else max(0.0, 1.0 - (abs(x - x_mean) - x_tolerance) / x_tolerance) for x in x_values]
            horizontal_score = float(np.mean(x_stable))
            direction_score = 0.7 * vertical_score + 0.3 * horizontal_score

            # Non-decreasing is not the same as moving
            first_feat = masks_info[0]['mask_feature']
            ref_height = float(first_feat['bbox'][3]) if first_feat is not None else 0.0
            first_y = float(centers[0][1])
            downward_displacement = max(0.0, max(c[1] for c in centers) - first_y)
            required_displacement = max(
                ref_height,
                float(frame_height) - first_y - ref_height / 2.0,
                1.0,
            )
            displacement_score = min(
                1.0, downward_displacement / required_displacement,
            )
            motion_score = direction_score * displacement_score
        else:
            vertical_score = 0.0
            horizontal_score = 0.0
            direction_score = 0.0
            downward_displacement = 0.0
            required_displacement = 1.0
            displacement_score = 0.0
            motion_score = 0.0

        # b) Exit: mask not detected in last frame
        last_mask_info = masks_info[-1]
        ref_area = masks_info[0]['mask_feature']['area']
        if last_mask_info['mask_feature'] is None:
            exit_score = 1.0
        else:
            last_area = float(last_mask_info['mask_feature']['area'])
            exit_score = max(0.0, 1.0 - last_area / ref_area)

        # c) shape/size/color should stay consistent across frames before exiting
        ref_feat = masks_info[0]['mask_feature']
        consistency_scores: List[float] = []
        if ref_feat is None:
            consistency_score = 0.0
        else:
            ref_mask_height = ref_feat['bbox'][3]
            for mask_info in masks_info:
                feat = mask_info['mask_feature']
                # Stop once the mask has started exiting the frame
                if feat is not None and feat['bbox'][1] + ref_mask_height > frame_height:
                    break
                # Mask not detected but hasn't exited yet -> penalise with 0
                if feat is None:
                    consistency_scores.append(0.0)
                    continue

                # 1.1 shape contour similarity
                match_score = cv2.matchShapes(
                    ref_feat['contour'], feat['contour'], cv2.CONTOURS_MATCH_I1, 0.0,
                )
                shape_score_from_contour = float(np.exp(-4.0 * max(0.0, match_score)))

                # 1.2 shape vertex count similarity
                if ref_feat['vertex_count'] == feat['vertex_count']:
                    vertex_score = 1.0
                elif abs(ref_feat['vertex_count'] - feat['vertex_count']) <= 1:
                    vertex_score = 0.3
                else:
                    vertex_score = 0.0

                shape_score = float(0.5 * shape_score_from_contour + 0.5 * vertex_score)

                # 1.3 shape size similarity
                area_ratio = min(ref_feat['area'], feat['area']) / max(ref_feat['area'], feat['area'], 1e-6)
                extent_ratio = min(ref_feat['bbox_extent'], feat['bbox_extent']) / max(ref_feat['bbox_extent'], feat['bbox_extent'], 1e-6)
                size_ratio = float(0.80 * area_ratio + 0.20 * extent_ratio)
                grow_ratio = float(feat['area']) / max(float(ref_feat['area']), 1e-6)
                if grow_ratio >= 0.6:
                    size_score = 1.0
                else:
                    size_score = grow_ratio / 0.6

                # 1.4 shape color similarity
                color_dist = float(np.linalg.norm(ref_feat['mean_bgr'] - feat['mean_bgr']))
                color_score = float(max(0.0, 1.0 - color_dist / np.sqrt(3.0 * (255.0 ** 2))))

                consistency_scores.append(0.3 * shape_score + 0.4 * size_score + 0.3 * color_score)

            consistency_score = float(np.mean(consistency_scores)) if consistency_scores else 0.0

        score = motion_score * exit_score * consistency_score
        details = {
            'motion': motion_score,
            'direction': direction_score,
            'vertical': vertical_score,
            'horizontal': horizontal_score,
            'downward_displacement': downward_displacement,
            'required_displacement': required_displacement,
            'displacement': displacement_score,
            'exit': exit_score,
            'consistency': consistency_score,
        }
        return score, details

    def _evaluate_occlusion_correctness(
        self,
        masks_info: List[Optional[Dict]],
        initial_objects: List[Dict],
        video_frames: List[np.ndarray],
    ) -> float:
        """Evaluate occlusion correctness."""
        if not initial_objects or not masks_info or not video_frames:
            return 0.0, {'max_occlusion': 0.0}

        obj_y_min = min(o['bbox'][1] for o in initial_objects)
        obj_y_max = max(o['bbox'][1] + o['bbox'][3] for o in initial_objects)
        n_objects = len(initial_objects)

        def _obj_area(o):
            a = o.get('area')
            return float(a) if a else float(o['bbox'][2] * o['bbox'][3])

        initial_total_area = sum(_obj_area(o) for o in initial_objects)

        max_occlusion = 0.0
        mask_reached_objects = False
        for frame, mask_info in zip(video_frames, masks_info):
            feat = mask_info['mask_feature']
            if feat is None:
                continue
            _, my, _, mh = feat['bbox']
            overlap = max(0, min(my + mh, obj_y_max) - max(my, obj_y_min))
            if overlap <= 0:
                continue
            mask_reached_objects = True
            visible = self._detect_colored_objects(frame)
            if initial_total_area > 0:
                visible_area = sum(_obj_area(v) for v in visible)
                occlusion = max(0.0, 1.0 - visible_area / initial_total_area)
            else:
                occlusion = max(0.0, 1.0 - len(visible) / n_objects)
            max_occlusion = max(max_occlusion, occlusion)

        # mask never reached the object area → task not completed
        if not mask_reached_objects:
            return 0.0, {'max_occlusion': 0.0}

        return max_occlusion, {'max_occlusion': max_occlusion}

    def _evaluate_elements_preservation(
        self,
        gt_first: np.ndarray,
        video_frames: List[np.ndarray],
        initial_objects: List[Dict],
        masks_info: List[Optional[Dict]],
    ) -> float:
        """Evaluate elements preservation."""
        if not initial_objects or not masks_info:
            return 0.0

        first_frame = video_frames[0]
        h, w = first_frame.shape[:2]
        obj_y_min = min(o['bbox'][1] for o in initial_objects)
        obj_y_max = max(o['bbox'][1] + o['bbox'][3] for o in initial_objects)

        # Frames were aligned to GT upstream, so gt_first already matches (h, w);
        # the guard stays only for direct calls that bypass that path.
        gt_first_ref = gt_first if gt_first.shape[:2] == (h, w) else cv2.resize(gt_first, (w, h))

        # Initial mask bbox (from frame 0), used to exclude from background
        init_mask_feat = masks_info[0]['mask_feature']

        fg_scores: List[float] = []
        bg_scores: List[float] = []

        for frame, mask_info in zip(video_frames[1:], masks_info[1:]):
            feat = mask_info['mask_feature']
            if feat is not None:
                _, my, _, mh = feat['bbox']
                mask_obj_overlap = max(0, min(my + mh, obj_y_max) - max(my, obj_y_min))
            else:
                mask_obj_overlap = 0

            # 1. Foreground: when mask is NOT over the object band, compare object strip to first frame
            if mask_obj_overlap == 0:
                strip_ref = first_frame[obj_y_min:obj_y_max, :]
                strip_cur = frame[obj_y_min:obj_y_max, :]
                if strip_ref.shape == strip_cur.shape and strip_ref.size > 0:
                    fg_scores.append(self._pixel_similarity(strip_ref, strip_cur))

            # 2. Background: compare against gt_first, excluding all foreground regions
            #    background = frame − initial objects − initial mask − current objects − current mask
            bg_mask = np.ones((h, w), dtype=bool)

            # Exclude initial object strip
            bg_mask[obj_y_min:obj_y_max, :] = False

            # Exclude initial mask region
            if init_mask_feat is not None:
                bx, by, bww, bhh = init_mask_feat['bbox']
                bg_mask[max(0, by):min(h, by + bhh), max(0, bx):min(w, bx + bww)] = False

            # Exclude current mask region
            if feat is not None:
                bx, by, bww, bhh = feat['bbox']
                bg_mask[max(0, by):min(h, by + bhh), max(0, bx):min(w, bx + bww)] = False

            # Exclude current frame's detected objects
            for obj in self._detect_colored_objects(frame):
                ox, oy, ow, oh = obj['bbox']
                bg_mask[max(0, oy):min(h, oy + oh), max(0, ox):min(w, ox + ow)] = False

            if bg_mask.any():
                bg_scores.append(self._pixel_similarity(
                    gt_first_ref, frame, mask=bg_mask.astype(np.uint8), strictness=3.0, min_cutoff=0.5,
                ))

        fg_score = float(np.mean(fg_scores)) if fg_scores else 0.5
        bg_score = float(np.mean(bg_scores)) if bg_scores else 0.5
        score = 0.6 * fg_score + 0.4 * bg_score
        return score, {'fg': fg_score, 'bg': bg_score}

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate multiple occlusions task."""
        scores: Dict[str, float] = {}

        if len(video_frames) < 2 or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        h, w = first_frame.shape[:2]

        # Normalize frame size (handles padding removal + resize)
        gt_frames = [normalize_frame_size(f, last_frame) if f.shape[:2] != last_frame.shape[:2] else f for f in gt_frames] if gt_frames else gt_frames
        gt_first = gt_frames[0]

        # detect colored objects in bottom-2/3 of first frame
        initial_objects = self._detect_colored_objects(
            gt_first, search_region=(0, h // 3, w, h - h // 3)
        )

        # detect masks across all frames
        masks_info = self._track_mask(video_frames)

        # 1) mask_path_vadility (50%)
        scores['mask_path_vadility'], mpv_d = self._evaluate_mask_path_vadility(masks_info, h, w)

        # 2) occlusion_correctness (30%)
        scores['occlusion_correctness'], occ_d = self._evaluate_occlusion_correctness(
            masks_info, initial_objects, video_frames,
        )

        # 3) elements_preservation (20%)
        scores['elements_preservation'], pres_d = self._evaluate_elements_preservation(
            gt_first, video_frames, initial_objects, masks_info,
        )

        total = sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

        lf_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        degenerate = float((lf_gray < 30).mean()) > 0.6 or float(lf_gray.std()) < 3.0
        if degenerate:
            total = min(total, 0.1)

        self._last_task_details = {
            **scores,
            'mask_path_vadility.motion':       mpv_d['motion'],
            'mask_path_vadility.direction':    mpv_d['direction'],
            'mask_path_vadility.displacement': mpv_d['displacement'],
            'mask_path_vadility.exit':         mpv_d['exit'],
            'mask_path_vadility.consistency':  mpv_d['consistency'],
            'occlusion_correctness.max':       occ_d['max_occlusion'],
            'elements_preservation.fg':        pres_d['fg'],
            'elements_preservation.bg':        pres_d['bg'],
            'degenerate_final_frame':          degenerate,
        }
        return total

    def _colored_object_area(self, frame: np.ndarray, y0: int, y1: int) -> int:
        """Saturated (colored object) pixel count inside the object y-band.

        The gray mask and background are low-saturation, so when the mask
        covers part of an object those pixels drop out of this count.
        """
        strip = frame[max(0, y0):max(0, y1)]
        if strip.size == 0:
            return 0
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        return int(np.sum(hsv[:, :, 1] > 60))

    def _evaluate_occlusion_correctness_interleave(
        self,
        masks_info: List[Optional[Dict]],
        initial_objects: List[Dict],
        frames: List[np.ndarray],
    ) -> Tuple[float, Dict]:
        """Interleave occlusion: credit PARTIAL occlusion.
        """
        FULL_CREDIT_COVERAGE = 0.4
        if not initial_objects or not masks_info or not frames:
            return 0.0, {'max_occlusion': 0.0}

        obj_y_min = min(o['bbox'][1] for o in initial_objects)
        obj_y_max = max(o['bbox'][1] + o['bbox'][3] for o in initial_objects)
        orig_area = self._colored_object_area(frames[0], obj_y_min, obj_y_max)
        if orig_area <= 0:
            return 0.0, {'max_occlusion': 0.0}

        max_cov = 0.0
        mask_reached_objects = False
        for frame, mask_info in zip(frames, masks_info):
            feat = mask_info['mask_feature'] if mask_info else None
            if feat is None:
                continue
            _, my, _, mh = feat['bbox']
            if max(0, min(my + mh, obj_y_max) - max(my, obj_y_min)) <= 0:
                continue
            mask_reached_objects = True
            vis_area = self._colored_object_area(frame, obj_y_min, obj_y_max)
            max_cov = max(max_cov, max(0.0, 1.0 - vis_area / orig_area))

        if not mask_reached_objects:
            return 0.0, {'max_occlusion': 0.0}
        return min(1.0, max_cov / FULL_CREDIT_COVERAGE), {'max_occlusion': round(max_cov, 3)}

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Interleave: same 3 weighted dims as video; only occlusion_correctness
        is redefined to credit partial occlusion (the full-occlusion frame is
        not captured by the sparse image sampling). Video path is untouched.
        """
        if not pred_images or input_frame is None:
            self._last_task_details = {"error": "no_input_or_pred"}
            return 0.0

        gt_first = gt_images[0] if gt_images else input_frame
        if input_frame.shape[:2] != gt_first.shape[:2]:
            input_frame = normalize_frame_size(input_frame, gt_first)
            pred_images = [normalize_frame_size(f, gt_first)
                           if f.shape[:2] != gt_first.shape[:2] else f for f in pred_images]

        pred_frames = [input_frame] + pred_images
        h, w = input_frame.shape[:2]
        if len(pred_frames) < 2:
            self._last_task_details = {"error": "too_few_frames"}
            return 0.0
        initial_objects = self._detect_colored_objects(
            gt_first, search_region=(0, h // 3, w, h - h // 3),
        )
        masks_info = self._track_mask(pred_frames)

        scores: Dict[str, float] = {}
        scores['mask_path_vadility'], mpv_d = self._evaluate_mask_path_vadility(
            masks_info, h, w,
        )
        scores['occlusion_correctness'], occ_d = self._evaluate_occlusion_correctness_interleave(
            masks_info, initial_objects, pred_frames,
        )
        scores['elements_preservation'], pres_d = self._evaluate_elements_preservation(
            gt_first, pred_frames, initial_objects, masks_info,
        )

        self._last_task_details = {
            **scores,
            'mask_path_vadility.motion':       mpv_d['motion'],
            'mask_path_vadility.direction':    mpv_d['direction'],
            'mask_path_vadility.displacement': mpv_d['displacement'],
            'mask_path_vadility.exit':         mpv_d['exit'],
            'mask_path_vadility.consistency':  mpv_d['consistency'],
            'occlusion_correctness.max':       occ_d['max_occlusion'],
            'elements_preservation.fg':        pres_d['fg'],
            'elements_preservation.bg':        pres_d['bg'],
        }
        total = sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
        lf_gray = cv2.cvtColor(pred_frames[-1], cv2.COLOR_BGR2GRAY)
        if float((lf_gray < 30).mean()) > 0.6 or float(lf_gray.std()) < 3.0:
            total = min(total, 0.1)
        return total


class SeparateObjectsSpinningEvaluator(_ShapeMotionBase):
    """
    G-25: Separate objects with spinning evaluator.

    Rewritten in the same per-object family as G-5:
    - pair GT source objects (first frame) with GT target objects (final frame)
      by visual identity
    - track each source object across the whole prediction
    - score each GT object from the tracked object's actual last-frame pose,
      plus travel / visibility / order quality
    - add 0-score ghosts for hallucinated extra objects in the last frame
    """

    PLACEMENT_RANGE = 60.0
    POSE_IOU_SATURATION = 0.85
    POSE_CENTER_SNAP_PX = 1.0
    POSE_IOU_SNAP = 0.80
    COVERAGE_MIN_TRAVEL = 40.0
    COVERAGE_SATURATION = 0.90
    PROGRESS_ONLY_FLOOR = 0.25
    VISIBILITY_SATURATION = 0.90
    TELEPORT_PX_FRAC = 0.25
    TELEPORT_PENALTY_BASE = 0.25
    MAX_TELEPORT_EVENTS = 6
    PATH_BACKTRACK_FREE_RATIO = 0.10
    PATH_BACKTRACK_DROP_RATIO = 0.90
    PATH_OVERSHOOT_FREE_RATIO = 0.05
    PATH_OVERSHOOT_DROP_RATIO = 0.30
    PATH_LATERAL_FULL_PX = 10.0
    PATH_LATERAL_DROP_PX = 120.0
    ROTATION_NEEDED_DEG = 8.0
    ORDER_MOVE_ONSET_PX = 25.0
    ORDER_ANGLE_TOL = 6.0
    POST_MOVE_ANGLE_TOL = 12.0
    ORDER_WINDOW = 3
    ORDER_SLACK_FRAMES = 1
    INTERMEDIATE_PROGRESS_MIN = 0.15
    INTERMEDIATE_PROGRESS_MAX = 0.85
    INTERMEDIATE_PROGRESS_SEPARATION = 0.15
    REQUIRED_INTERMEDIATE_STATES = 2
    SCENE_MILESTONE_SATURATION = 0.80
    ROTATION_STATIONARY_SATURATION = 0.80
    ROTATION_SOURCE_CHANGE_THRESHOLD = 15
    ROTATION_SOURCE_CHANGE_SATURATION = 0.02
    STABILITY_SATURATION = 0.85
    CLEANLINESS_P10_SATURATION = 0.80
    CLEANLINESS_STABILITY_SATURATION = 0.85
    ALIGNMENT_GATE_FLOOR = 0.25
    SCORE_WEIGHTS = {
        "alignment": 0.15,
        "motion_quality": 0.35,
        "sequence_quality": 0.25,
        "teleport_score": 0.15,
        "bg_preservation": 0.10,
    }
    SEQUENCE_COMPONENT_WEIGHTS = {
        "rotate_before_move": 0.10,
        "no_rotation_during_after_move": 0.08,
        "rotation_cleanliness": 0.04,
        "visibility": 0.03,
    }

    SYMMETRY_PERIOD = {
        "circle": 0.0,
        "triangle": 60.0,
        "square": 90.0,
        "rectangle": 180.0,
        "pentagon": 36.0,
        "hexagon": 60.0,
        "polygon": 180.0,
    }

    def _hue_delta(self, a: float, b: float) -> float:
        d = abs(float(a) - float(b))
        return min(d, 180.0 - d)

    def _shape_identity_cost(self, a: Dict, b: Dict) -> float:
        """Soft identity cost for GT source ↔ GT target pairing."""
        hue = self._hue_delta(a["fill_hue"], b["fill_hue"]) / 18.0
        sat = abs(float(a["fill_sat"]) - float(b["fill_sat"])) / 80.0
        val = abs(float(a["fill_val"]) - float(b["fill_val"])) / 80.0
        area = abs(float(a["area"]) - float(b["area"])) / max(
            float(a["area"]), float(b["area"]), 1.0,
        )
        dy = abs(float(a["center"][1]) - float(b["center"][1])) / 120.0
        shape_penalty = 0.0 if a["shape"] == b["shape"] else 0.5
        return hue + 0.5 * sat + 0.25 * val + 0.5 * area + 0.25 * dy + shape_penalty

    def _target_last_cost(self, target: Dict, candidate: Dict) -> float:
        """Cost for GT target ↔ predicted last-frame detection matching."""
        dist = safe_distance(target["center"], candidate["center"]) / self.PLACEMENT_RANGE
        hue = self._hue_delta(target["fill_hue"], candidate["fill_hue"]) / 18.0
        sat = abs(float(target["fill_sat"]) - float(candidate["fill_sat"])) / 80.0
        val = abs(float(target["fill_val"]) - float(candidate["fill_val"])) / 80.0
        area = abs(float(target["area"]) - float(candidate["area"])) / max(
            float(target["area"]), float(candidate["area"]), 1.0,
        )
        shape_penalty = 0.0 if target["shape"] == candidate["shape"] else 0.5
        return 1.5 * dist + hue + 0.5 * sat + 0.25 * val + 0.5 * area + shape_penalty

    def _best_assignment(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Brute-force rectangular assignment.

        G-25 scenes contain only a handful of objects (2-4 in the GT set), so
        permutations are simpler than pulling in a Hungarian dependency.
        """
        n_rows, n_cols = cost_matrix.shape
        if n_rows == 0 or n_cols == 0:
            return []

        if n_rows <= n_cols:
            best_cost = float("inf")
            best_cols: Optional[Tuple[int, ...]] = None
            for cols in permutations(range(n_cols), n_rows):
                total = float(sum(cost_matrix[r, c] for r, c in enumerate(cols)))
                if total < best_cost:
                    best_cost = total
                    best_cols = cols
            return [] if best_cols is None else [(r, c) for r, c in enumerate(best_cols)]

        best_cost = float("inf")
        best_rows: Optional[Tuple[int, ...]] = None
        for rows in permutations(range(n_rows), n_cols):
            total = float(sum(cost_matrix[r, c] for c, r in enumerate(rows)))
            if total < best_cost:
                best_cost = total
                best_rows = rows
        return [] if best_rows is None else [(r, c) for c, r in enumerate(best_rows)]

    def _pair_sources_to_targets(
        self, sources: List[Dict], targets: List[Dict],
    ) -> List[Tuple[int, int]]:
        if not sources or not targets:
            return []
        cost = np.zeros((len(sources), len(targets)), dtype=float)
        for i, src in enumerate(sources):
            for j, tgt in enumerate(targets):
                cost[i, j] = self._shape_identity_cost(src, tgt)
        return self._best_assignment(cost)

    def _match_targets_to_last(
        self, targets: List[Dict], last_shapes: List[Dict],
    ) -> Dict[int, int]:
        if not targets or not last_shapes:
            return {}
        cost = np.zeros((len(targets), len(last_shapes)), dtype=float)
        for i, target in enumerate(targets):
            for j, cand in enumerate(last_shapes):
                cost[i, j] = self._target_last_cost(target, cand)
        return {target_idx: last_idx for target_idx, last_idx in self._best_assignment(cost)}

    def _track_spinning_shape(
        self, video_frames: List[np.ndarray], source: Dict,
    ) -> List[Optional[Dict]]:
        """Per-frame trace of one source object with the actual contour kept.

        G-25 uses the same colour+position tracker as G-24.  Keeping this as a
        thin wrapper avoids diverging on practical fixes such as HSV red
        hue-wrap handling and low-saturation pastel/background separation.
        """
        return self._track_filled_shape(video_frames, source)

    def _pose_iou(self, pred_shape: Dict, gt_shape: Dict, frame_shape: Tuple[int, int, int]) -> float:
        if pred_shape.get("contour") is None or gt_shape.get("contour") is None:
            return 0.0

        ax, ay, aw, ah = cv2.boundingRect(pred_shape["contour"])
        bx, by, bw, bh = cv2.boundingRect(gt_shape["contour"])
        x1 = max(0, min(ax, bx) - 5)
        y1 = max(0, min(ay, by) - 5)
        x2 = min(frame_shape[1], max(ax + aw, bx + bw) + 5)
        y2 = min(frame_shape[0], max(ay + ah, by + bh) + 5)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        pred_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        gt_mask = np.zeros_like(pred_mask)
        offset = np.array([[[x1, y1]]], dtype=pred_shape["contour"].dtype)
        cv2.drawContours(pred_mask, [pred_shape["contour"] - offset], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(gt_mask, [gt_shape["contour"] - offset], -1, 255, thickness=cv2.FILLED)

        inter = int(np.logical_and(pred_mask > 0, gt_mask > 0).sum())
        union = int(np.logical_or(pred_mask > 0, gt_mask > 0).sum())
        return float(inter / union) if union > 0 else 0.0

    def _pose_score(self, final_dist: float, pose_iou: float) -> float:
        """Score final pose while tolerating sub-pixel GT-copy contour drift."""
        pose_score = min(1.0, float(pose_iou) / self.POSE_IOU_SATURATION)
        if (
            final_dist <= self.POSE_CENTER_SNAP_PX
            and pose_iou >= self.POSE_IOU_SNAP
        ):
            return 1.0
        return pose_score

    def _trace_visibility(self, trace: List[Optional[Dict]]) -> float:
        if not trace:
            return 0.0
        detected = sum(1 for p in trace if p is not None)
        ratio = detected / len(trace)
        return min(1.0, ratio / self.VISIBILITY_SATURATION)

    def _trace_displacement(self, trace: List[Optional[Dict]]) -> float:
        detected = [p for p in trace if p is not None]
        if len(detected) < 2:
            return 0.0
        return float(safe_distance(detected[0]["center"], detected[-1]["center"]))

    def _translation_milestone_score(
        self, trace: List[Optional[Dict]], source: Dict, target: Dict,
    ) -> Tuple[float, Dict[str, Any]]:
        """Evidence that translation occupied distinct intermediate states.

        Positions are measured only as relative progress on the source-target
        axis.  This deliberately accepts any two well-separated intermediate
        locations instead of requiring the generated frames to copy GT's exact
        one-third/two-thirds samples.
        """
        expected_disp = float(safe_distance(source["center"], target["center"]))
        details: Dict[str, Any] = {
            "used": expected_disp > self.COVERAGE_MIN_TRAVEL,
            "candidate_progress": [],
            "distinct_progress": [],
            "distinct_count": 0,
            "score": 1.0,
        }
        if expected_disp <= self.COVERAGE_MIN_TRAVEL:
            return 1.0, details

        source_center = np.array(source["center"], dtype=float)
        target_center = np.array(target["center"], dtype=float)
        unit = (target_center - source_center) / expected_disp
        candidates: List[float] = []
        for point in trace[1:-1]:
            if point is None:
                continue
            center = np.array(point["center"], dtype=float)
            progress = float(np.dot(center - source_center, unit) / expected_disp)
            if self.INTERMEDIATE_PROGRESS_MIN <= progress <= self.INTERMEDIATE_PROGRESS_MAX:
                candidates.append(progress)

        distinct: List[float] = []
        for progress in sorted(candidates):
            if not distinct or progress - distinct[-1] >= self.INTERMEDIATE_PROGRESS_SEPARATION:
                distinct.append(progress)
        score = min(1.0, len(distinct) / max(self.REQUIRED_INTERMEDIATE_STATES, 1))
        details.update({
            "candidate_progress": [round(float(v), 4) for v in candidates],
            "distinct_progress": [round(float(v), 4) for v in distinct],
            "distinct_count": len(distinct),
            "score": round(float(score), 4),
        })
        return float(score), details

    def _rotation_phase_evidence(
        self,
        video_frames: List[np.ndarray],
        gt_first_frame: np.ndarray,
        shape_traces: List[Tuple[Dict[str, Any], List[Optional[Dict[str, Any]]]]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Scene-level evidence for an in-place rotation phase.

        Angle estimates are unreliable for symmetric polygons in sparse image
        renders.  Instead, require an interior frame where most object centers
        are still near their sources while the source-object pixels have
        visibly changed.  A repeated input has no pixel change; a translation
        frame has insufficient stationary objects.
        """
        rotatable_sources = [
            source for source, _ in shape_traces
            if self._symmetry_period(source["shape"]) > 0
        ]
        if not rotatable_sources:
            return 1.0, {
                "required": False,
                "score": 1.0,
                "best_frame_idx": None,
                "best_stationary_ratio": 1.0,
                "best_changed_ratio": 0.0,
            }
        if len(video_frames) < 3:
            return 0.0, {
                "required": True,
                "score": 0.0,
                "best_frame_idx": None,
                "best_stationary_ratio": 0.0,
                "best_changed_ratio": 0.0,
            }

        source_mask = np.zeros(gt_first_frame.shape[:2], dtype=np.uint8)
        for source in rotatable_sources:
            contour = source.get("contour")
            if contour is not None:
                cv2.drawContours(source_mask, [contour], -1, 255, thickness=cv2.FILLED)
        source_mask = cv2.dilate(source_mask, np.ones((5, 5), np.uint8), iterations=1)
        if not np.any(source_mask > 0):
            return 0.0, {
                "required": True,
                "score": 0.0,
                "best_frame_idx": None,
                "best_stationary_ratio": 0.0,
                "best_changed_ratio": 0.0,
            }

        best_score = 0.0
        best_frame_idx: Optional[int] = None
        best_stationary_ratio = 0.0
        best_changed_ratio = 0.0
        baseline = video_frames[0]
        if baseline.shape != gt_first_frame.shape:
            baseline = normalize_frame_size(baseline, gt_first_frame)
        for frame_idx in range(1, len(video_frames) - 1):
            stationary = 0
            detected = 0
            for source, trace in shape_traces:
                if frame_idx >= len(trace) or trace[frame_idx] is None:
                    continue
                detected += 1
                if safe_distance(trace[frame_idx]["center"], source["center"]) <= self.ORDER_MOVE_ONSET_PX:
                    stationary += 1
            stationary_ratio = stationary / detected if detected else 0.0

            frame = video_frames[frame_idx]
            if frame.shape != gt_first_frame.shape:
                frame = normalize_frame_size(frame, gt_first_frame)
            gray_diff = cv2.cvtColor(cv2.absdiff(baseline, frame), cv2.COLOR_BGR2GRAY)
            changed_ratio = float(
                np.mean(gray_diff[source_mask > 0] > self.ROTATION_SOURCE_CHANGE_THRESHOLD)
            )
            stationary_score = min(
                1.0,
                stationary_ratio / max(self.ROTATION_STATIONARY_SATURATION, 1e-6),
            )
            change_score = min(
                1.0,
                changed_ratio / max(self.ROTATION_SOURCE_CHANGE_SATURATION, 1e-6),
            )
            score = stationary_score * change_score
            if score > best_score:
                best_score = score
                best_frame_idx = frame_idx
                best_stationary_ratio = stationary_ratio
                best_changed_ratio = changed_ratio

        return float(best_score), {
            "required": True,
            "score": round(float(best_score), 4),
            "best_frame_idx": best_frame_idx,
            "best_stationary_ratio": round(float(best_stationary_ratio), 4),
            "best_changed_ratio": round(float(best_changed_ratio), 4),
        }

    def _trace_path_progress(
        self, trace: List[Optional[Dict]], source: Dict, target: Dict,
    ) -> Tuple[float, Dict[str, Any]]:
        """Score how far the trace advanced along the GT source->target axis.

        Uses a projected path integral over tracker steps, then takes the
        maximum accumulated forward progress reached during the clip. This
        gives credit to "moved halfway and stopped" while avoiding extra
        reward for back-and-forth dithering.
        """
        expected_disp = float(safe_distance(source["center"], target["center"]))
        details = {
            "used": False,
            "expected_disp_px": round(expected_disp, 4),
            "projected_path_integral_px": 0.0,
            "max_projected_progress_px": 0.0,
            "clipped_progress_px": 0.0,
            "progress_ratio": 0.0,
            "progress_score": 0.0,
            "overshoot_px": 0.0,
            "overshoot_ratio": 0.0,
        }
        if expected_disp <= 1e-6:
            details["used"] = True
            details["projected_path_integral_px"] = 0.0
            details["max_projected_progress_px"] = 0.0
            details["clipped_progress_px"] = 0.0
            details["progress_ratio"] = 1.0
            details["progress_score"] = 1.0
            details["overshoot_px"] = 0.0
            details["overshoot_ratio"] = 0.0
            return 1.0, details

        if expected_disp <= self.COVERAGE_MIN_TRAVEL:
            details["used"] = False
            details["progress_ratio"] = 1.0
            details["progress_score"] = 1.0
            return 1.0, details

        unit = np.array(
            [
                (float(target["center"][0]) - float(source["center"][0])) / expected_disp,
                (float(target["center"][1]) - float(source["center"][1])) / expected_disp,
            ],
            dtype=float,
        )
        prev_center = (float(source["center"][0]), float(source["center"][1]))
        projected_path_integral = 0.0
        max_projected_progress = 0.0
        saw_detection = False
        for point in trace:
            if point is None:
                continue
            curr_center = (float(point["center"][0]), float(point["center"][1]))
            step = float(np.dot(np.array(curr_center, dtype=float) - np.array(prev_center, dtype=float), unit))
            projected_path_integral += step
            max_projected_progress = max(max_projected_progress, projected_path_integral)
            prev_center = curr_center
            saw_detection = True

        if not saw_detection:
            return 0.0, details

        progress_px = float(np.clip(max_projected_progress, 0.0, expected_disp))
        progress_ratio = progress_px / expected_disp
        overshoot_px = max(0.0, max_projected_progress - expected_disp)
        overshoot_ratio = overshoot_px / expected_disp
        progress_score = min(
            1.0,
            progress_ratio / self.COVERAGE_SATURATION,
        )
        details = {
            "used": True,
            "expected_disp_px": round(expected_disp, 4),
            "projected_path_integral_px": round(float(projected_path_integral), 4),
            "max_projected_progress_px": round(float(max_projected_progress), 4),
            "clipped_progress_px": round(float(progress_px), 4),
            "progress_ratio": round(float(progress_ratio), 4),
            "progress_score": round(float(progress_score), 4),
            "overshoot_px": round(float(overshoot_px), 4),
            "overshoot_ratio": round(float(overshoot_ratio), 4),
        }
        return progress_score, details

    def _trace_path_functional(
        self, trace: List[Optional[Dict]], source: Dict, target: Dict,
    ) -> Tuple[float, Dict[str, Any]]:
        """Path-quality score for translation.

        The signed projected path integral is equal to endpoint projection, so
        this combines non-linear path functionals instead:
        max-prefix progress, forward/backward variation, and lateral deviation
        from the source->target corridor.
        """
        progress_score, details = self._trace_path_progress(trace, source, target)
        expected_disp = float(safe_distance(source["center"], target["center"]))
        details.update({
            "path_score": round(float(progress_score), 4),
            "overshoot_score": 1.0,
            "positive_projected_px": 0.0,
            "negative_projected_px": 0.0,
            "forward_ratio": 1.0,
            "backtrack_ratio": 0.0,
            "forward_score": 1.0,
            "lateral_p90_px": 0.0,
            "lateral_score": 1.0,
        })

        if expected_disp <= self.COVERAGE_MIN_TRAVEL:
            details["path_score"] = 1.0
            return 1.0, details

        source_center = np.array(source["center"], dtype=float)
        target_center = np.array(target["center"], dtype=float)
        unit = (target_center - source_center) / expected_disp
        lateral_unit = np.array([-unit[1], unit[0]], dtype=float)

        prev_center = source_center
        positive = 0.0
        negative = 0.0
        lateral_values = [0.0]
        saw_detection = False
        for point in trace:
            if point is None:
                continue
            curr_center = np.array(point["center"], dtype=float)
            projected_step = float(np.dot(curr_center - prev_center, unit))
            if projected_step >= 0:
                positive += projected_step
            else:
                negative += -projected_step
            lateral_values.append(abs(float(np.dot(curr_center - source_center, lateral_unit))))
            prev_center = curr_center
            saw_detection = True

        if not saw_detection:
            details.update({
                "path_score": 0.0,
                "overshoot_score": 1.0,
                "forward_ratio": 0.0,
                "forward_score": 0.0,
            })
            return 0.0, details

        max_projected_progress = float(details.get("max_projected_progress_px", 0.0))
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

        denom = positive + negative
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
        if lateral_p90 <= self.PATH_LATERAL_FULL_PX:
            lateral_score = 1.0
        else:
            lateral_score = max(
                0.0,
                1.0 - (lateral_p90 - self.PATH_LATERAL_FULL_PX) / self.PATH_LATERAL_DROP_PX,
            )

        path_score = float(progress_score) * overshoot_score * forward_score * lateral_score
        details.update({
            "path_score": round(float(path_score), 4),
            "overshoot_px": round(float(overshoot_px), 4),
            "overshoot_ratio": round(float(overshoot_ratio), 4),
            "overshoot_score": round(float(overshoot_score), 4),
            "positive_projected_px": round(float(positive), 4),
            "negative_projected_px": round(float(negative), 4),
            "forward_ratio": round(float(forward_ratio), 4),
            "backtrack_ratio": round(float(backtrack_ratio), 4),
            "forward_score": round(float(forward_score), 4),
            "lateral_p90_px": round(float(lateral_p90), 4),
            "lateral_score": round(float(lateral_score), 4),
        })
        return path_score, details

    def _translation_score(
        self, path_progress: float, pose_score: float, expected_disp: float,
    ) -> float:
        if expected_disp <= self.COVERAGE_MIN_TRAVEL:
            return float(pose_score)
        pose_weight = self.PROGRESS_ONLY_FLOOR + (1.0 - self.PROGRESS_ONLY_FLOOR) * float(pose_score)
        return float(path_progress) * pose_weight

    def _alignment_gate(self, alignment: float) -> float:
        alignment = max(0.0, min(1.0, float(alignment)))
        return self.ALIGNMENT_GATE_FLOOR + (1.0 - self.ALIGNMENT_GATE_FLOOR) * alignment

    def _combine_scene_score(self, alignment: float, non_alignment_score: float) -> float:
        """Combine final pose with process/background.

        G-25 still gives partial credit for real spin+transport attempts, but
        the prompt requires ending aligned with the targets.  Gate the process
        portion by scene-level alignment so videos that move objects around but
        miss every target cannot receive a high final score.
        """
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

    def _trace_teleports(
        self, trace: List[Optional[Dict]], teleport_px: float,
    ) -> int:
        n = 0
        prev: Optional[Dict] = None
        for point in trace:
            if point is None:
                continue
            if prev is not None:
                if safe_distance(point["center"], prev["center"]) > teleport_px:
                    n += 1
            prev = point
        return n

    def _rotation_stage_scores(
        self, trace: List[Optional[Dict]], source: Dict, target: Dict,
    ) -> Tuple[float, float, Optional[int], Optional[int], bool]:
        period = self._symmetry_period(target["shape"])
        initial_delta = self._angle_delta(source["angle"], target["angle"], period)
        if period <= 0 or initial_delta < self.ROTATION_NEEDED_DEG:
            return 1.0, 1.0, None, None, False

        detected = [p for p in trace if p is not None]
        if len(detected) < max(2, self.ORDER_WINDOW):
            return 1.0, 1.0, None, None, False

        start_center = detected[0]["center"]
        move_start = next(
            (
                idx for idx, point in enumerate(detected)
                if safe_distance(point["center"], start_center) >= self.ORDER_MOVE_ONSET_PX
            ),
            None,
        )
        if move_start is None:
            return 1.0, 1.0, None, None, False

        angle_errors = [
            self._angle_delta(point["angle"], target["angle"], period)
            for point in detected
        ]
        rotation_done = None
        for idx in range(len(angle_errors) - self.ORDER_WINDOW + 1):
            window = angle_errors[idx: idx + self.ORDER_WINDOW]
            if max(window) <= self.ORDER_ANGLE_TOL:
                rotation_done = idx
                break

        if rotation_done is None:
            return 1.0, 1.0, move_start, None, False

        rotate_before = 1.0 if rotation_done <= move_start + self.ORDER_SLACK_FRAMES else 0.0
        post_move_errors = angle_errors[move_start:]
        if not post_move_errors:
            no_rotation_after = 1.0
        else:
            ok_ratio = sum(err <= self.POST_MOVE_ANGLE_TOL for err in post_move_errors) / len(post_move_errors)
            no_rotation_after = min(1.0, ok_ratio / self.STABILITY_SATURATION)
        return rotate_before, no_rotation_after, move_start, rotation_done, True

    def _rotation_cleanliness_score(
        self, trace: List[Optional[Dict]], source: Dict, target: Dict, move_start_idx: Optional[int],
    ) -> Tuple[float, Dict[str, Any]]:
        period = self._symmetry_period(target["shape"])
        initial_delta = self._angle_delta(source["angle"], target["angle"], period)
        details = {
            "used": False,
            "p10_area_ratio": None,
            "p90_area_ratio": None,
            "absolute_score": None,
            "stability_score": None,
        }
        if period <= 0 or initial_delta < self.ROTATION_NEEDED_DEG:
            return 1.0, details

        detected = [p for p in trace if p is not None]
        if move_start_idx is None or move_start_idx < 2 or len(detected) < 3:
            return 1.0, details

        phase = detected[:move_start_idx]
        if len(phase) < 2:
            return 1.0, details

        ref_area = max(float(source["area"]), 1.0)
        area_ratios = np.array([float(point["area"]) / ref_area for point in phase], dtype=float)
        p10 = float(np.percentile(area_ratios, 10))
        p90 = float(np.percentile(area_ratios, 90))
        absolute_score = min(1.0, p10 / self.CLEANLINESS_P10_SATURATION)
        stability_score = min(
            1.0,
            (p10 / max(p90, 1e-6)) / self.CLEANLINESS_STABILITY_SATURATION,
        )
        details = {
            "used": True,
            "p10_area_ratio": p10,
            "p90_area_ratio": p90,
            "absolute_score": absolute_score,
            "stability_score": stability_score,
        }
        return absolute_score * stability_score, details

    def _score_sequence(
        self,
        video_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
    ) -> float:
        if len(video_frames) < 2 or gt_first_frame is None or gt_final_frame is None:
            self._last_task_details = {"error": "missing frames"}
            return 0.0

        gt_sources = [s for s in self._detect_shapes(gt_first_frame) if not s["is_outline"]]
        gt_targets = [s for s in self._detect_shapes(gt_final_frame) if not s["is_outline"]]
        if not gt_sources or not gt_targets:
            self._last_task_details = {
                "error": "could not detect GT source/target objects",
                "gt_sources": len(gt_sources),
                "gt_targets": len(gt_targets),
            }
            return 0.0

        source_target_pairs = self._pair_sources_to_targets(gt_sources, gt_targets)
        last_shapes = [s for s in self._detect_shapes(video_frames[-1]) if not s["is_outline"]]
        target_to_last = self._match_targets_to_last(gt_targets, last_shapes)
        matched_last_indices = set(target_to_last.values())
        n_hallucinated = max(0, len(last_shapes) - len(matched_last_indices))

        H, W = video_frames[0].shape[:2]
        teleport_px = self.TELEPORT_PX_FRAC * min(H, W)

        process_component_weights = {
            name: weight
            for name, weight in self.SCORE_WEIGHTS.items()
            if name not in ("alignment", "bg_preservation")
        }

        alignments: List[float] = []
        motion_scores: List[float] = []
        sequence_scores: List[float] = []
        teleport_scores: List[float] = []
        process_scores: List[float] = []
        translation_milestone_scores: List[float] = []
        shape_traces: List[Tuple[Dict[str, Any], List[Optional[Dict[str, Any]]]]] = []
        per_object_details: List[Dict[str, Any]] = []
        rubric_rotate_scores: List[float] = []
        rubric_stability_scores: List[float] = []
        rubric_cleanliness_scores: List[float] = []
        rubric_visibility_scores: List[float] = []
        for source_idx, target_idx in source_target_pairs:
            source = gt_sources[source_idx]
            target = gt_targets[target_idx]
            trace = self._track_spinning_shape(video_frames, source)
            shape_traces.append((source, trace))
            traced_last = trace[-1]
            final_match_idx = target_to_last.get(target_idx)
            final_actual = (
                last_shapes[final_match_idx]
                if final_match_idx is not None and final_match_idx < len(last_shapes)
                else traced_last
            )
            visibility = self._trace_visibility(trace)
            displacement = self._trace_displacement(trace)
            expected_disp = safe_distance(source["center"], target["center"])
            travel = 1.0
            if expected_disp > self.COVERAGE_MIN_TRAVEL:
                travel = min(1.0, (displacement / expected_disp) / self.COVERAGE_SATURATION)
            path_score, progress_details = self._trace_path_functional(
                trace, source, target,
            )
            path_progress = float(progress_details["progress_score"])
            translation_milestone, translation_milestone_details = self._translation_milestone_score(
                trace, source, target,
            )
            translation_milestone_scores.append(translation_milestone)

            teleport_events = self._trace_teleports(trace, teleport_px)
            teleport_score = self.TELEPORT_PENALTY_BASE ** min(
                teleport_events, self.MAX_TELEPORT_EVENTS,
            )
            rotate_before_score, no_rotation_after_score, move_start_idx, rotation_done_idx, rotation_signal_used = self._rotation_stage_scores(
                trace, source, target,
            )
            rotation_cleanliness, cleanliness_details = self._rotation_cleanliness_score(
                trace, source, target, move_start_idx,
            )
            rubric_rotate_scores.append(rotate_before_score)
            rubric_stability_scores.append(no_rotation_after_score)
            rubric_cleanliness_scores.append(rotation_cleanliness)
            rubric_visibility_scores.append(visibility)

            sequence_score = self._weighted_geometric_score(
                {
                    "rotate_before_move": rotate_before_score,
                    "no_rotation_during_after_move": no_rotation_after_score,
                    "rotation_cleanliness": rotation_cleanliness,
                    "visibility": visibility,
                },
                self.SEQUENCE_COMPONENT_WEIGHTS,
            )

            if final_actual is None:
                final_dist = None
                pose_iou = 0.0
                pose_score = 0.0
                final_center = None
            else:
                final_dist = safe_distance(final_actual["center"], target["center"])
                pose_iou = self._pose_iou(final_actual, target, video_frames[-1].shape)
                pose_score = self._pose_score(final_dist, pose_iou)
                final_center = tuple(round(v, 2) for v in final_actual["center"])

            alignment_score = pose_score
            motion_quality = float(path_score)
            process_score_i = self._weighted_geometric_score(
                {
                    "motion_quality": motion_quality,
                    "sequence_quality": sequence_score,
                    "teleport_score": teleport_score,
                },
                process_component_weights,
            )
            translation_score = self._translation_score(
                path_score, pose_score, expected_disp,
            )

            alignments.append(alignment_score)
            motion_scores.append(motion_quality)
            sequence_scores.append(sequence_score)
            teleport_scores.append(teleport_score)
            process_scores.append(process_score_i)

            per_object_details.append({
                "source_shape": source["shape"],
                "target_shape": target["shape"],
                "target_center": tuple(round(v, 2) for v in target["center"]),
                "final_center": final_center,
                "final_pose_source": "final_frame_assignment" if final_match_idx is not None else "trace_fallback",
                "final_distance_px": None if final_dist is None else round(float(final_dist), 2),
                "alignment": round(float(alignment_score), 4),
                "pose_iou": round(float(pose_iou), 4),
                "pose_score": round(float(pose_score), 4),
                "translation_score": round(float(translation_score), 4),
                "motion_quality": round(float(motion_quality), 4),
                "visibility": round(float(visibility), 4),
                "travel_coverage": round(float(travel), 4),
                "path_progress": round(float(path_progress), 4),
                "path_score": round(float(path_score), 4),
                "translation_milestone_score": round(float(translation_milestone), 4),
                "translation_milestone_used": translation_milestone_details["used"],
                "translation_candidate_progress": translation_milestone_details["candidate_progress"],
                "translation_distinct_progress": translation_milestone_details["distinct_progress"],
                "translation_distinct_count": translation_milestone_details["distinct_count"],
                "path_progress_signal_used": progress_details["used"],
                "expected_disp_px": progress_details["expected_disp_px"],
                "projected_path_integral_px": progress_details["projected_path_integral_px"],
                "max_projected_progress_px": progress_details["max_projected_progress_px"],
                "clipped_progress_px": progress_details.get("clipped_progress_px"),
                "progress_ratio": progress_details.get("progress_ratio"),
                "overshoot_px": progress_details.get("overshoot_px"),
                "overshoot_ratio": progress_details.get("overshoot_ratio"),
                "overshoot_score": progress_details.get("overshoot_score"),
                "positive_projected_px": progress_details["positive_projected_px"],
                "negative_projected_px": progress_details["negative_projected_px"],
                "forward_ratio": progress_details["forward_ratio"],
                "backtrack_ratio": progress_details["backtrack_ratio"],
                "forward_score": progress_details["forward_score"],
                "lateral_p90_px": progress_details["lateral_p90_px"],
                "lateral_score": progress_details["lateral_score"],
                "teleport_events": teleport_events,
                "teleport_score": round(float(teleport_score), 4),
                "teleport_penalty": round(float(teleport_score), 4),
                "rotate_before_move": round(float(rotate_before_score), 4),
                "no_rotation_during_after_move": round(float(no_rotation_after_score), 4),
                "rotation_cleanliness": round(float(rotation_cleanliness), 4),
                "sequence_quality": round(float(sequence_score), 4),
                "rotation_signal_used": rotation_signal_used,
                "cleanliness_signal_used": cleanliness_details["used"],
                "cleanliness_p10_area_ratio": None if cleanliness_details["p10_area_ratio"] is None else round(float(cleanliness_details["p10_area_ratio"]), 4),
                "cleanliness_p90_area_ratio": None if cleanliness_details["p90_area_ratio"] is None else round(float(cleanliness_details["p90_area_ratio"]), 4),
                "cleanliness_absolute_score": None if cleanliness_details["absolute_score"] is None else round(float(cleanliness_details["absolute_score"]), 4),
                "cleanliness_stability_score": None if cleanliness_details["stability_score"] is None else round(float(cleanliness_details["stability_score"]), 4),
                "rubric": {
                    "alignment": round(float(alignment_score), 4),
                    "motion_quality": round(float(motion_quality), 4),
                    "sequence_quality": round(float(sequence_score), 4),
                    "teleport_score": round(float(teleport_score), 4),
                    "rotate_before_move": round(float(rotate_before_score), 4),
                    "no_rotation_during_after_move": round(float(no_rotation_after_score), 4),
                    "rotation_cleanliness": round(float(rotation_cleanliness), 4),
                    "visibility": round(float(visibility), 4),
                },
                "move_start_idx": move_start_idx,
                "rotation_done_idx": rotation_done_idx,
                "process_score": round(float(process_score_i), 4),
                "score": round(float(process_score_i), 4),
            })

        alignments_with_extras = alignments + [0.0] * n_hallucinated
        motion_scores_with_extras = motion_scores + [0.0] * n_hallucinated
        sequence_scores_with_extras = sequence_scores + [0.0] * n_hallucinated
        teleport_scores_with_extras = teleport_scores + [0.0] * n_hallucinated
        process_scores_with_extras = process_scores + [0.0] * n_hallucinated
        translation_milestones_with_extras = translation_milestone_scores + [0.0] * n_hallucinated

        alignment = float(np.mean(alignments_with_extras)) if alignments_with_extras else 0.0
        motion_quality = float(np.mean(motion_scores_with_extras)) if motion_scores_with_extras else 0.0
        sequence_quality = float(np.mean(sequence_scores_with_extras)) if sequence_scores_with_extras else 0.0
        teleport_score = float(np.mean(teleport_scores_with_extras)) if teleport_scores_with_extras else 0.0
        raw_process_score = float(np.mean(process_scores_with_extras)) if process_scores_with_extras else 0.0
        mean_translation_milestone = (
            float(np.mean(translation_milestones_with_extras))
            if translation_milestones_with_extras else 0.0
        )
        translation_process_evidence = min(
            1.0,
            mean_translation_milestone / max(self.SCENE_MILESTONE_SATURATION, 1e-6),
        )
        rotation_phase_evidence, rotation_phase_details = self._rotation_phase_evidence(
            video_frames, gt_first_frame, shape_traces,
        )
        process_score = (
            raw_process_score
            * translation_process_evidence
            * rotation_phase_evidence
        )
        bg_preservation, bg_details = self._background_preservation_for_traces(
            video_frames, gt_first_frame, shape_traces,
        )

        alignment_weight = self.SCORE_WEIGHTS["alignment"]
        process_weight = sum(process_component_weights.values())
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
        total = self._combine_scene_score(alignment, non_alignment_score)
        rubric_avg = {
            "alignment": round(float(alignment), 4),
            "motion_quality": round(float(motion_quality), 4),
            "sequence_quality": round(float(sequence_quality), 4),
            "teleport_score": round(float(teleport_score), 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "rotate_before_move": round(float(np.mean(rubric_rotate_scores)), 4) if rubric_rotate_scores else 0.0,
            "no_rotation_during_after_move": round(float(np.mean(rubric_stability_scores)), 4) if rubric_stability_scores else 0.0,
            "rotation_cleanliness": round(float(np.mean(rubric_cleanliness_scores)), 4) if rubric_cleanliness_scores else 0.0,
            "visibility": round(float(np.mean(rubric_visibility_scores)), 4) if rubric_visibility_scores else 0.0,
        }

        self._last_task_details = {
            "alignment": round(float(alignment), 4),
            "motion_quality": round(float(motion_quality), 4),
            "sequence_quality": round(float(sequence_quality), 4),
            "teleport_score": round(float(teleport_score), 4),
            "process_score": round(float(process_score), 4),
            "raw_process_score": round(float(raw_process_score), 4),
            "translation_process_evidence": round(float(translation_process_evidence), 4),
            "rotation_phase_evidence": round(float(rotation_phase_evidence), 4),
            "rotation_phase_details": rotation_phase_details,
            "mean_translation_milestone": round(float(mean_translation_milestone), 4),
            "scene_milestone_saturation": self.SCENE_MILESTONE_SATURATION,
            "object_score_avg": round(float(process_score), 4),
            "non_alignment_score": round(float(non_alignment_score), 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "background_preservation": bg_details,
            "per_object_scores": [round(float(s), 4) for s in process_scores_with_extras],
            "per_object_process_scores": [round(float(s), 4) for s in process_scores_with_extras],
            "score_formula": "linear_alignment_plus_alignment_gated_geometric_process_bg",
            "component_weights": self.SCORE_WEIGHTS,
            "process_component_weights": process_component_weights,
            "sequence_component_weights": self.SEQUENCE_COMPONENT_WEIGHTS,
            "final_geometric_weights": {
                "process_score": round(float(process_weight), 4),
                "bg_preservation": self.SCORE_WEIGHTS["bg_preservation"],
            },
            "alignment_weight": alignment_weight,
            "non_alignment_weight": round(float(1.0 - alignment_weight), 4),
            "alignment_gate": round(float(alignment_gate), 4),
            "alignment_gate_floor": self.ALIGNMENT_GATE_FLOOR,
            "rubric_weights": self.SCORE_WEIGHTS,
            "rubric_avg": rubric_avg,
            "n_gt_objects": len(gt_sources),
            "n_gt_targets": len(gt_targets),
            "n_last_frame_objects": len(last_shapes),
            "n_hallucinated": n_hallucinated,
            "teleport_px": round(float(teleport_px), 2),
            "per_object": per_object_details,
        }
        return total

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        return self._score_sequence(video_frames, gt_first_frame, gt_final_frame)

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        if not pred_images or input_frame is None or gt_final_frame is None:
            return 0.0

        H, W = gt_final_frame.shape[:2]
        seq: List[np.ndarray] = []
        seq.append(
            cv2.resize(input_frame, (W, H))
            if input_frame.shape[:2] != (H, W) else input_frame
        )
        for pred in pred_images:
            seq.append(cv2.resize(pred, (W, H)) if pred.shape[:2] != (H, W) else pred)

        gt_first = seq[0]
        gt_final = gt_final_frame   # (H, W) is gt_final_frame's own size
        return self._score_sequence(seq, gt_first, gt_final)

# Mapping of task names to evaluators
IN_DOMAIN_50_EVALUATORS = {
    'G-3_stable_sort_data-generator': StableSortEvaluator,
    'G-5_multi_object_placement_data-generator': MultiObjectPlacementEvaluator,
    'G-8_track_object_movement_data-generator': TrackObjectMovementEvaluator,
    'G-9_identify_objects_in_region_data-generator': IdentifyObjectsInRegionEvaluator,
    'G-13_grid_number_sequence_data-generator': GridNumberSequenceEvaluator,
    'G-15_grid_avoid_obstacles_data-generator': GridAvoidObstaclesEvaluator,
    'G-16_grid_go_through_block_data-generator': GridGoThroughBlockEvaluator,
    'G-18_grid_shortest_path_data-generator': GridShortestPathEvaluator,
    'G-21_multiple_occlusions_vertical_data-generator': MultipleOcclusionsVerticalEvaluator,
    'G-25_seperate_object_spinning_data-generator': SeparateObjectsSpinningEvaluator,
}
