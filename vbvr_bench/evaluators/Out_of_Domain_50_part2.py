"""
Specific evaluators for Out-of-Domain_50 tasks (Part 2).
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import normalize_frame_size, threshold_score
from ..utils import CircleSelectionProcessor
from ..utils import compute_ssim, score_background_similarity, extract_patterns_from_white_bg, find_patterns_in_image
import os
import json
import shutil



class LocateSegmentIntersectionEvaluator(BaseEvaluator):
    """
    G-169: Locate intersection of segments evaluator.
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
            circle_hsv_tolerance=(12, 50, 50), # h tight (bg red-ish) but min_s/v lowered 80->50 to catch desaturated red circles
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
            [(0.25, 1.0), (0.35, 0.0)]
        )
        circle_area_penalty_score = threshold_score(
            circle_selection_info['circle_color_mask_ratio'],
            [(0.1, 1.0), (0.2, 0.0)]
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
                    [(0.4, 0.0), (0.5, 1.0)]
                )
                circle_match_score = max(circle_match_score, shape_inclusion_score)
                per_shape_scores[shape_id] = max(per_shape_scores[shape_id], shape_inclusion_score)
            if circle_match_score > 0.0:
                circle_size_penalty = threshold_score(
                    approx_ratio,
                    [(0.15, 0.0), (0.2, 1.0)]
                )
            else:
                circle_size_penalty = threshold_score(
                    approx_ratio,
                    [(0.0, 0.0), (0.04, 1.0)]
                )
            circle_size_penalty_list.append(circle_size_penalty)
        
        circle_size_penalty_score = float(sum(circle_size_penalty_list))
        selection_score = float(np.mean(np.array(per_shape_scores)))

        _TIGHT = 0.15  
        best = 0.0
        for ci, row in enumerate(circle_selection_info['circle_vs_shape_overlap']):
            ov = max(row) if row else 0.0
            if ov <= 0.0:
                continue
            ratio = circle_ratio_list[ci] if ci < len(circle_ratio_list) else 1.0
            size_factor = 0.5 + 0.5 * min(1.0, _TIGHT / max(ratio, 1e-6))
            best = max(best, ov * size_factor)
        n_circles_all = len(circle_size_penalty_list)
        precision = 1.0 / n_circles_all if n_circles_all > 1 else 1.0
        scores['match_score'] = max(0.0, min(selection_score, best) * precision)
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
            'consistency_score': scores['consistency_score'],
            'match_score': scores['match_score'],
            'background_consistency_score': background_consistency_score,
            'foreground_consistency_score': foreground_consistency_score,
            'circle_area_penalty_score': circle_area_penalty_score,
            'background_change_ratio': circle_selection_info['background_change_ratio'],
            'foreground_change_ratio': circle_selection_info['foreground_change_ratio'],
            'circle_color_mask_ratio': circle_selection_info['circle_color_mask_ratio'],
            'circle_ratio_list': circle_ratio_list,
            'circle_size_penalty_list': circle_size_penalty_list,
            'circle_size_penalty_score': circle_size_penalty_score,
            'per_shape_scores': per_shape_scores,
            'selection_score': selection_score,
            'total_score': total_score,
        }
        return total_score


class ArrangeCirclesByCircumferenceEvaluator(BaseEvaluator):
    """
    G-174: Arrange circles by circumference (large to small).

    Scoring:
    - arrangement    (60%): size order (0.5) + horizontal alignment (0.5)
    - fore_consistency (20%): pattern detection accuracy
    - back_consistency (20%): non-pattern regions are blank (white)
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

        # Handle missing GT frames
        if gt_final_frame is None and gt_frames:
            gt_final_frame = gt_frames[-1]
        if gt_first_frame is None and gt_frames:
            gt_first_frame = gt_frames[0]

        if gt_final_frame is None or gt_first_frame is None:
            return 0.0

        final_frame = video_frames[-1]

        # 1. Extract pattern templates from GT first frame
        gt_patterns = extract_patterns_from_white_bg(gt_first_frame)
        if not gt_patterns:
            return 0.0

        # 2. Extract patterns directly from final frame
        gen_patterns = extract_patterns_from_white_bg(final_frame)

        # 3. Find matching patterns in final frame (used for arrangement + back_consistency)
        matched_patterns = find_patterns_in_image(gt_first_frame, gt_patterns, final_frame, size_tolerance=0.15)
        # 4. Calculate fore_consistency: compare gt_patterns count vs gen_patterns count
        fore_consistency = self._evaluate_fore_consistency(gt_patterns, gen_patterns)

        # 5. Calculate arrangement (size order + horizontal alignment)
        arrangement = self._evaluate_arrangement(matched_patterns)
        if len(matched_patterns) < 0.6 * len(gt_patterns) and len(gen_patterns) >= 2:
            gen_arrangement = self._evaluate_arrangement(gen_patterns)
            count_match = (min(len(gen_patterns), len(gt_patterns))
                           / max(len(gen_patterns), len(gt_patterns), 1))
            arrangement = max(arrangement, gen_arrangement * count_match)

        # 6. Calculate back_consistency
        back_consistency = self._evaluate_back_consistency(final_frame, matched_patterns)

        # consistency is a penalty multiplier
        consistency = 0.5 * back_consistency + 0.5 * fore_consistency
        score = arrangement * (0.6 + 0.4 * consistency)
        self._last_task_details = {
            'arrangement': arrangement,
            'fore_consistency': fore_consistency,
            'back_consistency': back_consistency,
        }
        return score

    def _evaluate_fore_consistency(self, gt_patterns: List[Dict], matched_patterns: List[Dict]) -> float:
        """
        Calculate fore_consistency based on pattern detection accuracy.
        - All patterns found: 1.0
        - Missing/Extra 1 pattern: 0.5
        - Missing/Extra 2+ patterns: 0.0
        """
        n_gt = len(gt_patterns)
        n_matched = len(matched_patterns)

        missing = n_gt - n_matched
        if missing == 0:
            return 1.0
        elif missing == 1 or missing == -1:  # Allow one extra or one missing pattern for partial credit
            return 0.5
        else:
            return 0.0

    def _evaluate_arrangement(self, matched_patterns: List[Dict]) -> float:
        """
        Calculate arrangement score (60% weight):
        - Size order (0.5): large patterns must be to the right of small patterns
          (task G-174: large to small, so left-to-right order is descending in size)
        - Horizontal alignment (0.5): patterns should form a horizontal line (±5px tolerance)
        """
        if len(matched_patterns) < 2:
            return 0.0

        # Extract cx, cy, r from find_patterns_in_image output format:
        # 'center': (cx, cy), 'bbox': (x, y, w, h), 'ref_area': float
        def get_cx(p):
            return p['center'][0]

        def get_cy(p):
            return p['center'][1]

        def get_r(p):
            # Estimate radius from bbox dimensions
            _, _, bw, bh = p['bbox']
            return (bw + bh) / 4.0  # average of half-width and half-height

        # Sort by x position (left to right)
        sorted_by_x = sorted(matched_patterns, key=get_cx)

        radii = [get_r(p) for p in sorted_by_x]
        cy_list = [get_cy(p) for p in sorted_by_x]
        n = len(radii)

        # --- Part 1: Size order (0.5) ---
        # Task says "large to small" → left-to-right should be descending (large → small)
        # Violation: left pattern is smaller than right pattern
        violations = 0
        for i in range(n - 1):
            if radii[i] < radii[i + 1]:  # left is smaller than right — violation
                violations += 1

        size_order_score = max(0.0, 1.0 - (violations / (n - 1))) if n > 1 else 1.0

        # --- Part 2: Horizontal alignment (0.5) ---
        avg_y = np.mean(cy_list)
        tolerance = 5.0
        aligned_count = sum(1 for y in cy_list if abs(y - avg_y) <= tolerance)
        alignment_score = aligned_count / n if n > 0 else 0.0

        # Order gates alignment: circles neatly aligned but in the wrong size order
        # are not a half-success. GT (size_order_score=1) is unchanged.
        return size_order_score * (0.5 + 0.5 * alignment_score)

    def _evaluate_back_consistency(self, frame: np.ndarray, matched_patterns: List[Dict]) -> float:
        """
        Calculate back_consistency: non-pattern regions should be white.
        - All white: 1.0
        - Non-white ratio < 5%: score > 0.5
        - Non-white ratio >= 5%: score < 0.5
        """
        h, w = frame.shape[:2]

        pattern_mask = np.zeros((h, w), dtype=np.uint8)
        for pattern in matched_patterns:
            bx, by, bw, bh = pattern['bbox']
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(w, bx + bw)
            y2 = min(h, by + bh)
            pattern_mask[y1:y2, x1:x2] = 255

        # Get background pixels (non-pattern regions)
        bg_pixels = frame[pattern_mask == 0]
        if len(bg_pixels) == 0:
            return 1.0

        # White = all channels >= 240
        white_mask = np.all(bg_pixels >= 240, axis=1)
        white_ratio = np.mean(white_mask)
        non_white_ratio = 1.0 - white_ratio

        k = np.log(2) / 0.05
        score = np.exp(-k * non_white_ratio)

        return max(0.0, min(1.0, score))


class DrawMidpointPerpendicularEvaluator(BaseEvaluator):
    """
    G-189: Draw midpoint perpendicular line evaluator.

    Evaluation:
    - Red line correctness (60%): position, length, count, thickness
    - Object preservation (20%): dots + black lines unchanged (excluding correct red line)
    - Background preservation (20%): background clean (excluding dots, black lines, correct red line)
    """

    TASK_WEIGHTS = {
        'red_line': 0.60,
        'consistency': 0.40,
    }

    def _detect_red_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get binary mask of red pixels."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 80, 80])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 80, 80])
        upper2 = np.array([180, 255, 255])
        return cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    def _detect_horizontal_lines(self, frame: np.ndarray) -> List[int]:
        """Detect y positions of horizontal black lines."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Look for rows that are mostly dark (black line spans full width)
        row_means = np.mean(gray, axis=1)
        # Black lines: rows with mean < 50 (dark)
        dark_rows = np.where(row_means < 50)[0]
        if len(dark_rows) == 0:
            return []
        # Cluster dark rows into distinct lines
        lines_y = []
        prev = dark_rows[0]
        group = [prev]
        for r in dark_rows[1:]:
            if r - prev <= 3:
                group.append(r)
            else:
                lines_y.append(int(np.mean(group)))
                group = [r]
            prev = r
        lines_y.append(int(np.mean(group)))
        return sorted(lines_y)

    def _detect_fg_mask(self, frame: np.ndarray) -> np.ndarray:
        """Detect foreground (dots + black lines) by color distance from background."""
        h, w = frame.shape[:2]
        corners = [frame[2, 2], frame[2, w-3], frame[h-3, 2], frame[h-3, w-3]]
        bg_color = np.mean(corners, axis=0)
        diff = np.sqrt(np.sum((frame.astype(float) - bg_color.astype(float)) ** 2, axis=2))
        binary = (diff > 30).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
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
            score = 0.7
        elif ratio < t3:
            score = 0.5
        elif ratio < t4:
            score = 0.3
        else:
            score = 0.0
        return score, {'ratio': round(ratio, 6), 'changed_px': changed, 'total_px': mask_pixels}

    def _evaluate_red_line(self, gt_red_mask: np.ndarray, gen_red_mask: np.ndarray,
                           gt_first: np.ndarray, frame_h: int, frame_w: int,
                           initial_red_mask: Optional[np.ndarray] = None) -> Tuple[float, Dict]:
        """Evaluate red line using row-scanning on GT-defined corridor.

        1. Row coverage (40%): fraction of GT rows that have gen red in x window
        2. Position accuracy (30%): how close gen red center is to GT x (per row avg)
        3. Extra red penalty (30%): gen red outside x corridor → penalty
        """
        details = {}

        # --- Find GT red line position (filter out red dots) ---
        # Use connected components: keep only elongated components (line-like, h > w * 3)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_red_mask, connectivity=8)
        gt_line_mask = np.zeros_like(gt_red_mask)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 10:
                continue
            # Line: tall and narrow; Dot: roughly square
            if h > w * 3:
                gt_line_mask[labels == i] = 255

        gt_red_pixels = int((gt_line_mask > 0).sum())
        if gt_red_pixels == 0:
            details['error'] = 'no_gt_red_line'
            return 0.0, details

        ys, xs = np.where(gt_line_mask > 0)
        gt_x_min, gt_x_max = int(xs.min()), int(xs.max())
        gt_y_min, gt_y_max = int(ys.min()), int(ys.max())
        gt_x_center = (gt_x_min + gt_x_max) // 2
        gt_width = gt_x_max - gt_x_min + 1

        details['gt_x_center'] = gt_x_center
        details['gt_y_range'] = (gt_y_min, gt_y_max)

        hlines = self._detect_horizontal_lines(gt_first)
        if len(hlines) >= 2:
            line_y_top = hlines[0]
            line_y_bot = hlines[-1]
        else:
            line_y_top = gt_y_min
            line_y_bot = gt_y_max
        details['black_line_num'] = len(hlines)

        # --- X window: GT x ± tolerance (layered for position scoring) ---
        # Narrow window for strict position, wider for lenient
        narrow_tol = max(gt_width * 2, 8)   # ~8px: strict
        wide_tol = max(gt_width * 5, 20)    # ~20px: lenient

        narrow_x_min = max(0, gt_x_center - narrow_tol)
        narrow_x_max = min(frame_w - 1, gt_x_center + narrow_tol)
        wide_x_min = max(0, gt_x_center - wide_tol)
        wide_x_max = min(frame_w - 1, gt_x_center + wide_tol)

        # --- 1. Row coverage: scan GT y range, check gen red in wide window ---
        corridor = gen_red_mask[gt_y_min:gt_y_max+1, wide_x_min:wide_x_max+1]
        row_has_red = np.any(corridor > 0, axis=1)
        n_rows = len(row_has_red)
        row_coverage = float(row_has_red.sum()) / max(n_rows, 1)
        details['row_coverage'] = round(row_coverage, 4)
        details['covered_rows'] = int(row_has_red.sum())
        details['total_rows'] = n_rows

        # --- 2. Position accuracy: per covered row, how close is gen red center to GT x ---
        corridor_narrow = gen_red_mask[gt_y_min:gt_y_max+1, narrow_x_min:narrow_x_max+1]
        row_has_red_narrow = np.any(corridor_narrow > 0, axis=1)
        narrow_coverage = float(row_has_red_narrow.sum()) / max(n_rows, 1)
        # Position score: blend narrow and wide (narrow=strict, wide=lenient)
        position_score = narrow_coverage * 0.7 + row_coverage * 0.3
        details['narrow_coverage'] = round(narrow_coverage, 4)
        details['position_score'] = round(position_score, 4)

        # --- 3. Length penalty: red extending beyond GT y range within corridor ---
        # Check how much red is outside GT y range but inside the x corridor
        corridor_above = gen_red_mask[0:gt_y_min, wide_x_min:wide_x_max+1]
        corridor_below = gen_red_mask[gt_y_max+1:frame_h, wide_x_min:wide_x_max+1]
        gt_line_len = gt_y_max - gt_y_min + 1
        extra_above = int((corridor_above > 0).any(axis=1).sum())
        extra_below = int((corridor_below > 0).any(axis=1).sum())
        extra_len_ratio = (extra_above + extra_below) / max(gt_line_len, 1)
        # Mild penalty: up to 50% extra length → score drops to 0.5
        length_score = max(0.0, 1.0 - extra_len_ratio)
        details['extra_above_rows'] = extra_above
        details['extra_below_rows'] = extra_below
        details['extra_len_ratio'] = round(extra_len_ratio, 4)
        details['length_score'] = round(length_score, 4)

        # --- 4. Extra red penalty: newly-added red outside the x corridor ---
        novel_gen_red_mask = gen_red_mask
        initial_red_pixels = 0
        excluded_existing_red_pixels = 0
        if initial_red_mask is not None:
            if initial_red_mask.shape != gen_red_mask.shape:
                initial_red_mask = cv2.resize(
                    initial_red_mask,
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            initial_red_pixels = int((initial_red_mask > 0).sum())
            initial_red_guard = cv2.dilate(
                initial_red_mask,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
            novel_gen_red_mask = cv2.bitwise_and(
                gen_red_mask,
                cv2.bitwise_not(initial_red_guard),
            )
            excluded_existing_red_pixels = int(
                ((gen_red_mask > 0) & (novel_gen_red_mask == 0)).sum()
            )

        valid_mask = np.zeros_like(gen_red_mask)
        valid_mask[0:frame_h, wide_x_min:wide_x_max+1] = 255
        gen_red_total_raw = int((gen_red_mask > 0).sum())
        gen_red_total = int((novel_gen_red_mask > 0).sum())
        gen_red_outside = int(
            ((novel_gen_red_mask > 0) & (valid_mask == 0)).sum()
        )
        if gen_red_total > 0:
            overflow_ratio = gen_red_outside / gen_red_total
            overflow_score = max(0.0, 1.0 - overflow_ratio * 2.0)
        else:
            overflow_ratio = 0.0
            # Precision is undefined when no annotation was drawn.  Awarding
            # 1.0 here gave a completely unchanged input 30% of red-line credit.
            overflow_score = 0.0
        details['gen_red_total_px'] = gen_red_total
        details['gen_red_total_px_raw'] = gen_red_total_raw
        details['initial_red_px'] = initial_red_pixels
        details['excluded_existing_red_px'] = excluded_existing_red_pixels
        details['gen_red_outside_px'] = gen_red_outside
        details['overflow_ratio'] = round(overflow_ratio, 4)
        details['overflow_score'] = round(overflow_score, 4)

        placement = row_coverage * 0.25 + position_score * 0.45 + overflow_score * 0.30
        final = placement * (0.2 + 0.8 * length_score)

        details['valid_x_range'] = (wide_x_min, wide_x_max)
        details['valid_y_range'] = (line_y_top, line_y_bot)

        return round(final, 4), details

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        first_frame = video_frames[0] if len(video_frames) > 0 else None
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_first = gt_first_frame
        gt_last = gt_final_frame

        if last_frame is None or gt_last is None or gt_first is None:
            return 0.0

        # Normalize sizes: gen → GT size
        if last_frame.shape != gt_last.shape:
            last_frame = normalize_frame_size(last_frame, gt_last)
        if first_frame is not None and first_frame.shape != gt_first.shape:
            first_frame = normalize_frame_size(first_frame, gt_first)

        # --- Detect red masks ---
        gt_red_mask = self._detect_red_mask(gt_last)
        gen_red_mask = self._detect_red_mask(last_frame)
        initial_red_mask = (
            self._detect_red_mask(first_frame)
            if first_frame is not None else None
        )

        frame_h, frame_w = gt_last.shape[:2]

        # 1. Red line correctness (60%): coverage-based on GT region
        red_score, red_details = self._evaluate_red_line(
            gt_red_mask,
            gen_red_mask,
            gt_first,
            frame_h,
            frame_w,
            initial_red_mask=initial_red_mask,
        )

        # Foreground = dots + black lines from GT first frame (clean reference)
        # Diff is computed on gen_first vs gen_last, but mask positions from GT
        fg_mask = self._detect_fg_mask(gt_first)
        kernel = np.ones((5, 5), np.uint8)

        # Split fg into black lines vs dots by connected component size
        num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
        line_mask = np.zeros_like(fg_mask)
        dot_mask = np.zeros_like(fg_mask)
        line_area_threshold = frame_w * 0.3  # black lines span most of width
        for i in range(1, num_cc):
            x, y, w, h, area = cc_stats[i]
            if w > line_area_threshold:
                line_mask[cc_labels == i] = 255
            else:
                dot_mask[cc_labels == i] = 255

        line_mask_dilated = cv2.dilate(line_mask, kernel, iterations=1)
        # Dots are small — use minimal dilation to avoid ratio dilution
        dot_kernel = np.ones((3, 3), np.uint8)
        dot_mask_dilated = cv2.dilate(dot_mask, dot_kernel, iterations=1)
        fg_mask_dilated = cv2.bitwise_or(line_mask_dilated, dot_mask_dilated)

        # Build correct corridor mask for exclusion from preservation checks
        gt_red_dilated = cv2.dilate(gt_red_mask, kernel, iterations=2)
        correct_corridor = np.zeros((frame_h, frame_w), dtype=np.uint8)
        valid_x = red_details.get('valid_x_range')
        valid_y = red_details.get('valid_y_range')
        if valid_x is not None and valid_y is not None:
            correct_corridor[valid_y[0]:valid_y[1]+1, valid_x[0]:valid_x[1]+1] = 255
        red_exclude = cv2.bitwise_or(gt_red_dilated, correct_corridor)

        # 2. Object preservation (20%): separate line and dot scores, then average
        line_obj_mask = cv2.bitwise_and(line_mask_dilated, cv2.bitwise_not(red_exclude))
        line_score, line_details = self._pixel_diff_score(
            first_frame, last_frame, line_obj_mask, thresholds=(0.1, 0.2, 0.30, 0.50))
        dot_obj_mask = cv2.bitwise_and(dot_mask_dilated, cv2.bitwise_not(red_exclude))
        dot_score, dot_details = self._pixel_diff_score(
            first_frame, last_frame, dot_obj_mask, thresholds=(0.15, 0.25, 0.35, 0.50))
        obj_score = line_score * 0.4 + dot_score * 0.6
        obj_details = {
            'line_score': round(line_score, 4), 'line_ratio': line_details['ratio'],
            'line_px': line_details['total_px'],
            'dot_score': round(dot_score, 4), 'dot_ratio': dot_details['ratio'],
            'dot_px': dot_details['total_px'],
        }

        # 3. Background preservation (20%): compare gen_first vs gen_last outside fg and corridor
        exclude_mask = cv2.bitwise_or(fg_mask_dilated, red_exclude)
        bg_mask = cv2.bitwise_not(exclude_mask)
        bg_score, bg_details = self._pixel_diff_score(
            first_frame, last_frame, bg_mask, thresholds=(0.005, 0.01, 0.05, 0.10))

        scores = {
            'red_line': round(red_score, 4),
            'object_preservation': round(obj_score, 4),
            'background_preservation': round(bg_score, 4),
        }

        scores['consistency'] = (scores['object_preservation'] + scores['background_preservation']) / 2
        self._last_task_details = {
            **scores,
            **{f'red_{k}': v for k, v in red_details.items()},
            **{f'obj_{k}': v for k, v in obj_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }

        _rl = scores.get('red_line', 0.0)
        _cons = scores.get('consistency', 0.0)
        return (self.TASK_WEIGHTS['red_line'] * _rl * min(1.0, _cons / 0.5)
                + min(1.0, _rl / 0.5) * self.TASK_WEIGHTS['consistency'] * _cons)

class DrawNextSizedShapeEvaluator(BaseEvaluator):
    """
    G-193: Draw next sized shape in pattern.

    Dimensions:
        - completion (60%): split image vertically into 5 parts, detect the largest
          filled shape in the rightmost 1/5 for GT final and generated final, then
          score shape/size/color/position.
        - foreground_preservation (25%): compare first vs generated final on
          foreground region.
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
        """Return (foreground_mask, background_mask) based on near-white background."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_not(bg_mask)
        return fg_mask, bg_mask

    def _extract_right_region_shape_features(self, frame: np.ndarray) -> Optional[Dict]:
        """Extract features of the largest filled shape in the rightmost 1/5."""
        h, w = frame.shape[:2]
        x_start = w * 4 // 5

        roi = frame[:, x_start:]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            # Solidity: ratio of contour area to its convex hull area.
            # Filled shapes have high solidity (≥ 0.7); hollow outlines are low.
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < 0.7:
                continue
            if area > best_area:
                best_area = area
                best_cnt = cnt

        if best_cnt is None:
            return None

        shifted = best_cnt.copy()
        shifted[:, :, 0] += x_start

        area = cv2.contourArea(shifted)
        if area <= 0:
            return None

        M = cv2.moments(shifted)
        if M["m00"] <= 0:
            return None

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        filled_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(filled_mask, [shifted], -1, 255, thickness=-1)
        mean_bgr = np.array(cv2.mean(frame, mask=filled_mask)[:3], dtype=np.float32)

        perimeter = float(cv2.arcLength(shifted, True))
        x, y, bw, bh = cv2.boundingRect(shifted)
        approx = cv2.approxPolyDP(shifted, 0.015 * perimeter if perimeter > 0 else 0.0, True)

        return {
            "contour": shifted,
            "area": float(area),
            "centroid": (float(cx / max(w, 1)), float(cy / max(h, 1))),
            "mean_bgr": mean_bgr,
            "bbox_extent": float(area / max(float(bw * bh), 1.0)),
            "vertex_count": int(len(approx)),
            "mask": filled_mask,
        }

    def _compute_scaling_completion_score(
        self,
        gt_shape_features: Optional[Dict],
        pred_shape_features: Optional[Dict],
    ) -> Tuple[float, Dict[str, float]]:
        """Compute completion and sub-scores with O-9 aligned scoring."""
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
            gt_shape_features["contour"],
            pred_shape_features["contour"],
            cv2.CONTOURS_MATCH_I1,
            0.0,
        )
        # exp(-4 * matchShapes) is already the continuous contour similarity; clipping
        # everything below 0.5 to exactly 0 threw away how close the shape came.
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
        if shape_score < 0.6:
            return 0.0, {
                "shape": shape_score,
                "size": 0.0,
                "color": 0.0,
                "position": 0.0,
                "shape_contour": shape_score_from_contour,
                "shape_vertex": vertex_score,
            }

        area_ratio = min(gt_shape_features["area"], pred_shape_features["area"]) / max(
            gt_shape_features["area"], pred_shape_features["area"], 1e-6
        )
        extent_ratio = min(gt_shape_features["bbox_extent"], pred_shape_features["bbox_extent"]) / max(
            gt_shape_features["bbox_extent"], pred_shape_features["bbox_extent"], 1e-6
        )
        size_ratio = float(0.80 * area_ratio + 0.20 * extent_ratio)
        # ratio is already the measured agreement; the old >=0.75-else-0 cut discarded how close it came.
        size_score = size_ratio if size_ratio >= 0.75 else 0.0

        color_dist = float(np.linalg.norm(gt_shape_features["mean_bgr"] - pred_shape_features["mean_bgr"]))
        color_score = float(max(0.0, 1.0 - color_dist / np.sqrt(3.0 * (255.0 ** 2))))

        gt_cx, gt_cy = gt_shape_features["centroid"]
        pred_cx, pred_cy = pred_shape_features["centroid"]
        position_dist = float(np.sqrt((gt_cx - pred_cx) ** 2 + (gt_cy - pred_cy) ** 2))
        position_score = float(max(0.0, 1.0 - position_dist / np.sqrt(2.0)))

        completion = (
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
        """Evaluate draw-next-sized-shape task with completion/preservation dimensions."""
        if len(video_frames) < 2:
            return 0.0

        if gt_first_frame is None or gt_final_frame is None:
            return 0.0

        scores: Dict[str, float] = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        if last_frame.shape[:2] != gt_final_frame.shape[:2]:
            first_frame = normalize_frame_size(first_frame, gt_final_frame)
            last_frame = normalize_frame_size(last_frame, gt_final_frame)
        gt_first, gt_last = gt_first_frame, gt_final_frame

        # 1) completion (60%): rightmost 1/5 largest filled shape with O-9 aligned scoring.
        gt_shape_features = self._extract_right_region_shape_features(gt_last)
        pred_shape_features = self._extract_right_region_shape_features(last_frame)
        gt_completion_shape = gt_shape_features["mask"] if gt_shape_features is not None else np.zeros(last_frame.shape[:2], dtype=np.uint8)
        gen_completion_shape = pred_shape_features["mask"] if pred_shape_features is not None else np.zeros(last_frame.shape[:2], dtype=np.uint8)
        change_mask = cv2.bitwise_or(gt_completion_shape, gen_completion_shape)

        completion_score, completion_details = self._compute_scaling_completion_score(
            gt_shape_features,
            pred_shape_features,
        )
        scores["completion"] = completion_score

        # 2) foreground_preservation (25%): compare first/last frame consistency on foreground.
        first_fg, first_bg = self._frame_masks(first_frame)
        fg_compare_mask = cv2.bitwise_and(first_fg, cv2.bitwise_not(change_mask))
        scores["foreground_preservation"] = self._pixel_similarity(first_frame, last_frame, fg_compare_mask)

        # 3) background_preservation (15%): compare first/last frame consistency on background.
        bg_compare_mask = cv2.bitwise_and(first_bg, cv2.bitwise_not(change_mask))
        scores["background_preservation"] = self._pixel_similarity(first_frame, last_frame, bg_compare_mask, strictness=3.0, min_cutoff=0.6)

        self._last_task_details = {
            **scores,
            "completion_details": completion_details,
        }
        return float(sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS))

class MarkWavePeaksEvaluator(BaseEvaluator):
    """
    G-202: Mark wave peaks evaluator.
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
            circle_hsv_tolerance=(18, 50, 50),
            foreground_hsv_delta_tolerance=(15.0, 150.0, 150.0),  
            background_hsv_delta_tolerance=(15.0, 150.0, 150.0),
            consistency_forground_remove_bg="white",
            foreground_enlarge_pixels=0 # special enlarge = 5, because foreground shapes are small
        )
        circle_selection_info = circle_selection_processor.process(gt_first_frame, gt_final_frame, last_frame, debug_dir=debug_dir)
        
        scores = {}
        background_consistency_score = threshold_score(
            circle_selection_info['background_change_ratio'],
            [(0.03, 1.0), (0.1, 0.0)]
        )
        foreground_consistency_score = threshold_score(
            circle_selection_info['foreground_change_ratio'],
            [(0.7, 1.0), (0.85, 0.0)]
        )
        circle_area_penalty_score = threshold_score(
            circle_selection_info['circle_color_mask_ratio'],
            [(0.1, 1.0), (0.2, 0.0)]
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
                    [(0.4, 0.0), (0.5, 1.0)]
                )
                circle_match_score = max(circle_match_score, shape_inclusion_score)
                per_shape_scores[shape_id] = max(per_shape_scores[shape_id], shape_inclusion_score)
            if circle_match_score > 0.0:
                circle_size_penalty = threshold_score(
                    approx_ratio,
                    [(0.075, 0.0), (0.15, 1.0)]
                )
            else:
                circle_size_penalty = threshold_score(
                    approx_ratio,
                    [(0.0, 0.0), (0.02, 1.0)]
                )
            circle_size_penalty_list.append(circle_size_penalty)
        
        if len(circle_size_penalty_list) > 0:
            circle_size_penalty_score = float(np.mean(np.array(circle_size_penalty_list))) * 2
        else:
            circle_size_penalty_score = 0.0
        selection_score = float(np.mean(np.array(per_shape_scores)))
        scores['match_score'] = max(0, selection_score * (1.0 - circle_size_penalty_score))

        _m = scores['match_score']
        total_score = (self.TASK_WEIGHTS['match_score'] * _m
                       + min(1.0, _m / 0.5) * self.TASK_WEIGHTS['consistency_score'] * scores['consistency_score'])

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
            'consistency_score': scores['consistency_score'],
            'match_score': scores['match_score'],
            'background_consistency_score': background_consistency_score,
            'foreground_consistency_score': foreground_consistency_score,
            'circle_area_penalty_score': circle_area_penalty_score,
            'background_change_ratio': circle_selection_info['background_change_ratio'],
            'foreground_change_ratio': circle_selection_info['foreground_change_ratio'],
            'circle_color_mask_ratio': circle_selection_info['circle_color_mask_ratio'],
            'circle_ratio_list': circle_ratio_list,
            'circle_size_penalty_list': circle_size_penalty_list,
            'circle_size_penalty_score': circle_size_penalty_score,
            'per_shape_scores': per_shape_scores,
            'selection_score': selection_score,
            'total_score': total_score,
        }
        return total_score


# Export all evaluators
OUT_OF_DOMAIN_50_EVALUATORS_PART2 = {
    'G-169_locate_intersection_of_segments_data-generator': LocateSegmentIntersectionEvaluator,
    'G-174_arrange_circles_by_circumference_data-generator': ArrangeCirclesByCircumferenceEvaluator,
    'G-189_draw_midpoint_perpendicular_line_data-generator': DrawMidpointPerpendicularEvaluator,
    'G-193_draw_next_sized_shape_data-generator': DrawNextSizedShapeEvaluator,
    'G-202_mark_wave_peaks_data-generator': MarkWavePeaksEvaluator,
}
