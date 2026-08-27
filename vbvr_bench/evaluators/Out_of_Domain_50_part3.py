"""
Specific evaluators for Out-of-Domain_50 tasks (Part 3).
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import normalize_frame_size
from ..utils import CircleSelectionProcessor, threshold_score
from ..utils import (
    detect_closed_contours_by_color, match_contours, COLOR_BOUNDS,
    score_background_similarity, score_foreground_similarity,
    calculate_list_length_penalty,
)
import os
import json
import shutil


class OutlineInnermostSquareEvaluator(BaseEvaluator):
    """
    G-221: Outline innermost square evaluator.

    Scoring:
    - accuracy         (60%): IoU-based matching of blue contours vs GT
    - fore_consistency (40%): Consistency of non-blue foreground region between final and GT frames
                              (white pixels and blue pixels in GT are excluded from comparison)

    Note: back_consistency is omitted because this task has no white background.
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

        gen_contours = detect_closed_contours_by_color(final_frame, COLOR_BOUNDS['blue'],hollow_only=True)
        gt_contours  = detect_closed_contours_by_color(gt_final_frame, COLOR_BOUNDS['blue'],hollow_only=True)

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
                gt_mask_c = np.zeros(canvas_size, dtype=np.uint8)
                cv2.drawContours(gt_mask_c, [gt_cnt], -1, 255, -1)
                gen_mask_c = np.zeros(canvas_size, dtype=np.uint8)
                cv2.drawContours(gen_mask_c, gen_contours, -1, 255, -1)
                _ga_c = int((gt_mask_c > 0).sum())
                _gen_a = int((gen_mask_c > 0).sum())
                _cov_c = (int(((gt_mask_c > 0) & (gen_mask_c > 0)).sum()) / _ga_c) if _ga_c else 0.0
                _size_c = min(1.0, _ga_c / float(_gen_a)) if _gen_a > 0 else 0.0
                per_gt_scores.append(max(base, 0.9 * _cov_c * _size_c))
            else:
                per_gt_scores.append(base)
        accuracy = float(np.mean(per_gt_scores)) if per_gt_scores else 0.0
        accuracy = accuracy * calculate_list_length_penalty(len(gt_contours), max(len(valid_ious), n_contained), len(gen_contours))

        fore_consistency = score_foreground_similarity(
            gt_final_frame, final_frame, COLOR_BOUNDS['blue'], type='mse'
        )

        consistency = fore_consistency
        score = accuracy * (0.6 + 0.4 * consistency)
        self._last_task_details = {
            'accuracy': accuracy,
            'fore_consistency': fore_consistency,
        }
        return score


class MarkTangentPointEvaluator(BaseEvaluator):
    """
    G-222: Mark tangent point of circles evaluator.
    """
    NEW_CIRCLE_CHANGE_MIN = 0.05

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
            circle_color='black',
            circle_fill_max_ratio=0.9,
            circle_hsv_tolerance=(18, 50, 50),
            foreground_hsv_delta_tolerance=(15.0, 150.0, 150.0),  
            background_hsv_delta_tolerance=(15.0, 150.0, 150.0),
            foreground_enlarge_pixels=20
        )
        circle_selection_info = circle_selection_processor.process(gt_first_frame, gt_final_frame, last_frame, debug_dir=debug_dir)

        scoring_last = last_frame
        if scoring_last.shape != gt_first_frame.shape:
            scoring_last = normalize_frame_size(scoring_last, gt_first_frame)
        pixel_change = np.max(
            cv2.absdiff(gt_first_frame, scoring_last), axis=2,
        ) > 25
        circle_change_ratios = []
        eligible_circle_ids = []
        for circle_id, circle in enumerate(circle_selection_info['pred_circles']):
            contour = circle.get('contour')
            if contour is None:
                change_ratio = 0.0
            else:
                contour_mask = np.zeros(gt_first_frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                area = int((contour_mask > 0).sum())
                changed = int(((contour_mask > 0) & pixel_change).sum())
                change_ratio = changed / max(area, 1)
            circle_change_ratios.append(float(change_ratio))
            if change_ratio >= self.NEW_CIRCLE_CHANGE_MIN:
                eligible_circle_ids.append(circle_id)
        eligible_circle_id_set = set(eligible_circle_ids)
        
        scores = {}
        background_consistency_score = threshold_score(
            circle_selection_info['background_change_ratio'],
            [(0.05, 1.0), (0.2, 0.0)]
        )
        foreground_consistency_score = threshold_score(
            circle_selection_info['foreground_change_ratio'],
            [(0.15, 1.0), (0.25, 0.0)]
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
            if circle_id not in eligible_circle_id_set:
                continue
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
                    [(0.1, 0.0), (0.15, 1.0)]
                )
            else:
                circle_size_penalty = threshold_score(
                    approx_ratio,
                    [(0.0, 0.0), (0.05, 1.0)]
                )
            circle_size_penalty_list.append(circle_size_penalty)
        
        circle_size_penalty_score = float(sum(circle_size_penalty_list))
        selection_score = float(np.mean(np.array(per_shape_scores)))
        scores['match_score'] = max(0, selection_score * (1.0 - circle_size_penalty_score))
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
            'background_change_ratio': circle_selection_info['background_change_ratio'],
            'foreground_change_ratio': circle_selection_info['foreground_change_ratio'],
            'circle_color_mask_ratio': circle_selection_info['circle_color_mask_ratio'],
            'background_consistency_score': background_consistency_score,
            'foreground_consistency_score': foreground_consistency_score,
            'circle_area_penalty_score': circle_area_penalty_score,
            'circle_ratio_list': circle_ratio_list,
            'circle_size_penalty_list': circle_size_penalty_list,
            'circle_size_penalty_score': circle_size_penalty_score,
            'circle_change_ratios': [round(r, 4) for r in circle_change_ratios],
            'eligible_circle_ids': eligible_circle_ids,
            'new_circle_change_min': self.NEW_CIRCLE_CHANGE_MIN,
            'per_shape_scores': per_shape_scores,
            'selection_score': selection_score,
            'consistency_score': scores['consistency_score'],
            'match_score': scores['match_score'],
            'total_score': total_score
        }
        return total_score


class SelectLineBlackCircleEvaluator(BaseEvaluator):
    """
    G-223: Highlight horizontal lines evaluator.
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
            circle_color='black',
            circle_fill_max_ratio=0.9,
            circle_hsv_tolerance=(18, 50, 50),
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
            [(0.15, 1.0), (0.4, 0.0)]
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
        # Gate the match by scene preservation
        scores['match_score'] = scores['match_score'] * foreground_consistency_score
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
            'ambiguous_circles_count': ambiguous_circles_count,
            'ambiguous_circles_ratio': ambiguous_circles_ratio,
            'ambiguous_score': ambiguous_score,
            'num_circles': num_circles,
            'num_target_shapes': num_target_shapes,
            'per_shape_scores': per_shape_scores,
            'correct_match_score': correct_match_score,
            'wrong_match_score': wrong_match_score,
            'consistency_score': scores['consistency_score'],
            'match_score': scores['match_score'],
            'total_score': total_score
        }
        return total_score


class AddBordersToUnborderedEvaluator(BaseEvaluator):
    """
    G-240: Add borders to unbordered shapes evaluator.

    Scoring:
    - accuracy         (60%): IoU-based matching of black contours vs GT
    - back_consistency (20%): white background similarity between final and GT frames
    - fore_consistency (20%): non-white non-black foreground similarity
    """

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: [List[np.ndarray]],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        final_frame = video_frames[-1]
        canvas_size = (final_frame.shape[0], final_frame.shape[1])
        if gt_final_frame is None:
            gt_final_frame = gt_frames[-1]
        if gt_first_frame is None:
            gt_first_frame = gt_frames[0]

        # Resize GT if needed
        if final_frame.shape != gt_final_frame.shape:
            final_frame = cv2.resize(
                final_frame, (gt_final_frame.shape[1], gt_final_frame.shape[0])
            )

        gen_contours = detect_closed_contours_by_color(final_frame, COLOR_BOUNDS['black'])
        gt_contours  = detect_closed_contours_by_color(gt_final_frame, COLOR_BOUNDS['black'])
        ini_contours = detect_closed_contours_by_color(gt_first_frame, COLOR_BOUNDS['black'])

        match_results = match_contours(
            gt_contours, gen_contours,
            iou_threshold=0.1,
            canvas_size=canvas_size
        )

        valid_ious = [iou for iou in match_results if iou is not None]
        if not valid_ious:
            accuracy = 0.0
        elif len(valid_ious) == len(gt_contours) and len(gt_contours) == len(ini_contours):
            accuracy = 1.0
            valid_ious = []
        elif len(valid_ious) < len(ini_contours):
            accuracy = 0.0
        else:
            if len(ini_contours) > 0:
                valid_ious = sorted(valid_ious, reverse=True)[len(ini_contours):]
            accuracy = float(np.mean(valid_ious)) if valid_ious else 0.0
        accuracy = accuracy * calculate_list_length_penalty(len(gt_contours)-len(ini_contours), len(valid_ious), len(gen_contours)-len(ini_contours))

        back_consistency = score_background_similarity(gt_final_frame, final_frame, type='mse')

        fore_consistency = score_foreground_similarity(
            gt_final_frame, final_frame, COLOR_BOUNDS['black'], type='mse'
        )

        consistency = 0.5 * back_consistency + 0.5 * fore_consistency
        score = accuracy * (0.6 + 0.4 * consistency)
        self._last_task_details = {
            'accuracy': accuracy,
            'back_consistency': back_consistency,
            'fore_consistency': fore_consistency,
        }
        return score


class ColorTripleIntersectionEvaluator(BaseEvaluator):
    """
    G-250: Color Triple Intersection Red

    Task: In a Venn diagram with 3 circles, fill the triple intersection with red.

    Evaluation:
    1. Red region correctness (60%): IoU of red region vs GT, multiplied by coverage and precision
    2. Non-red region preservation (40%): regions outside correct red area should be unchanged
    """

    TASK_WEIGHTS = {
        'red_region': 0.60,
        'preservation': 0.40,
    }

    def _detect_red_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get binary mask of red pixels."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 80, 80])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 80, 80])
        upper2 = np.array([180, 255, 255])
        return cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

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

        # Normalize gen → GT size
        if gen_last.shape != gt_last.shape:
            gen_last = normalize_frame_size(gen_last, gt_last)
        if gen_first.shape != gt_first.shape:
            gen_first = normalize_frame_size(gen_first, gt_first)

        # --- 1. Red region correctness (60%) ---
        red_mask_gen = self._detect_red_mask(gen_last)
        red_mask_gt = self._detect_red_mask(gt_last)

        gt_area = int((red_mask_gt > 0).sum())
        gen_area = int((red_mask_gen > 0).sum())
        overlap = int(((red_mask_gen > 0) & (red_mask_gt > 0)).sum())
        union = int(((red_mask_gen > 0) | (red_mask_gt > 0)).sum())

        if gt_area == 0:
            red_score = 1.0 if gen_area == 0 else 0.0
            coverage, precision = 1.0, 1.0
        elif gen_area == 0:
            red_score = 0.0
            coverage, precision = 0.0, 0.0
        else:
            coverage = overlap / gt_area  # recall
            precision = overlap / gen_area

            # Tiered coverage score (how much GT red is covered)
            if coverage >= 0.9:
                cov_score = 1.0
            elif coverage >= 0.7:
                cov_score = 0.7 + (coverage - 0.7) / 0.2 * 0.3
            elif coverage >= 0.4:
                cov_score = 0.3 + (coverage - 0.4) / 0.3 * 0.4
            else:
                cov_score = coverage / 0.4 * 0.3

            # Tiered precision score (how much gen red is correct)
            if precision >= 0.85:
                prec_score = 1.0
            elif precision >= 0.6:
                prec_score = 0.7 + (precision - 0.6) / 0.25 * 0.3
            elif precision >= 0.3:
                prec_score = 0.4 + (precision - 0.3) / 0.3 * 0.3
            else:
                prec_score = precision / 0.3 * 0.4

            red_score = cov_score * prec_score

        red_details = {
            'gt_red_area': gt_area,
            'gen_red_area': gen_area,
            'overlap': overlap,
            'coverage': round(overlap / max(gt_area, 1), 4),
            'precision': round(overlap / max(gen_area, 1), 4),
            'iou': round(overlap / max(union, 1), 4),
        }

        # --- 2. Non-red region preservation (40%) ---
        kernel = np.ones((5, 5), np.uint8)
        gt_red_dilated = cv2.dilate(red_mask_gt, kernel, iterations=1)
        preserve_mask = cv2.bitwise_not(gt_red_dilated)

        preserve_score, preserve_details = self._pixel_diff_score(
            gen_first, gen_last, preserve_mask, thresholds=(0.005, 0.02, 0.05, 0.10))

        scores = {
            'red_region': round(red_score, 4),
            'preservation': round(preserve_score, 4),
        }
        self._last_task_details = {
            **scores,
            **{f'red_{k}': v for k, v in red_details.items()},
            **{f'preserve_{k}': v for k, v in preserve_details.items()},
        }
        # Preservation is a penalty multiplier
        return red_score * (0.6 + 0.4 * preserve_score)


class HighDensityLiquidEvaluator(BaseEvaluator):
    """
    G-273 Task: Objects fall into cups with different density liquids.
    Float in high-density (>= object), sink in low-density.

    Evaluation:
    Final state (60%):
      - object_position (40%): each object at correct float/sink y
      - scene_preservation (20%): background/cups/liquid unchanged
    Process (40%):
      - non_teleport (15%): objects pass through intermediate y-zones
      - one_per_column (15%): each cup has <=1 object per frame
      - scene_stability (10%): intermediate frames not corrupted
    """

    TASK_WEIGHTS = {
        'final_state': 0.60,
        'process': 0.40,
    }
    
    def _detect_gt_obj_info(self, gt_first: np.ndarray, gt_final: np.ndarray,
                            cup: Dict) -> Dict:
        """Detect object hue, pixel area, and height from GT first/final frame diff.
        Returns {'hue': int|None, 'area': int, 'height': int, 'mask': ndarray|None}.
        """
        cup_x, cup_y = cup['position']
        cup_w, cup_h = cup['size']
        fh, fw = gt_first.shape[:2]

        x1, x2 = max(0, cup_x), min(fw, cup_x + cup_w)
        y1, y2 = max(0, cup_y), min(fh, cup_y + cup_h)

        roi_first = gt_first[y1:y2, x1:x2].astype(np.float32)
        roi_final = gt_final[y1:y2, x1:x2]
        diff = np.abs(roi_final.astype(np.float32) - roi_first).max(axis=2)

        changed = diff > 25
        if changed.sum() < 30:
            return {'hue': None, 'area': 400, 'height': 20, 'mask': None}

        hsv_final = cv2.cvtColor(roi_final, cv2.COLOR_BGR2HSV)
        sat = hsv_final[:, :, 1]
        obj_mask = changed & (sat > 60)
        pixel_area = int(obj_mask.sum())
        hues = hsv_final[:, :, 0][obj_mask]
        if len(hues) < 20:
            return {'hue': None, 'area': 400, 'height': 20, 'mask': None}

        hue = int(np.median(hues))

        # Object height from bounding box of mask
        rows = np.any(obj_mask, axis=1)
        row_indices = np.where(rows)[0]
        obj_height = max(20, int(row_indices[-1] - row_indices[0] + 1)) if len(row_indices) > 0 else 20

        full_mask = np.zeros((fh, fw), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = obj_mask.astype(np.uint8) * 255

        return {'hue': hue, 'area': pixel_area, 'height': obj_height, 'mask': full_mask}

    def _detect_objects_in_column(self, frame: np.ndarray, cup: Dict,
                              gt_hue: int, obj_area: int) -> List[Dict]:
        """Detect object contours by hue within a cup's vertical strip."""
        cup_x, cup_y = cup['position']
        cup_w, cup_h = cup['size']
        fh, fw = frame.shape[:2]

        x1, x2 = max(0, cup_x), min(fw, cup_x + cup_w)
        y1, y2 = 0, min(fh, cup_y + cup_h)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_range = 20
        lo = np.array([max(0, gt_hue - h_range), 40, 40])
        hi = np.array([min(180, gt_hue + h_range), 255, 255])
        mask = cv2.inRange(hsv, lo, hi)

        if gt_hue < h_range:
            lo2 = np.array([180 + gt_hue - h_range, 40, 40])
            hi2 = np.array([180, 255, 255])
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
        elif gt_hue > 180 - h_range:
            lo2 = np.array([0, 40, 40])
            hi2 = np.array([gt_hue + h_range - 180, 255, 255])
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if obj_area > 0 and int((mask > 0).sum()) > obj_area * 3:
            vals = hsv[:, :, 2][mask > 0]
            if vals.size > 0:
                liquid_v = int(np.median(vals))
                v_sep = (np.abs(hsv[:, :, 2].astype(np.int16) - liquid_v) > 45).astype(np.uint8) * 255
                refined = cv2.bitwise_and(mask, v_sep)
                refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
                refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
                if int((refined > 0).sum()) >= obj_area * 0.3:
                    mask = refined

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(50, obj_area * 0.1)
        results = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00']) + x1
            cy = int(M['m01'] / M['m00']) + y1
            results.append({'cy': cy, 'cx': cx, 'area': int(area)})
        results.sort(key=lambda d: d['area'], reverse=True)
        return results

    def _should_float(self, obj: Dict, cup: Dict) -> bool:
        """True if this object should float (final_y near water_level)."""
        final_y = obj['final_position'][1]
        water_level = cup['water_level']
        return abs(final_y - water_level) < 100

    def _get_cup_colors(self, gt_first: np.ndarray, cup: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Get liquid and empty-cup BGR colors from GT first frame.
        Returns (liquid_color, empty_color).
        """
        cx, cy = cup['position']
        cw, ch = cup['size']
        wl = cup['water_level']
        fh, fw = gt_first.shape[:2]
        margin = max(5, cw // 20)
        x1, x2 = max(0, cx + margin), min(fw, cx + cw - margin)
        default = np.array([128, 128, 128], dtype=float)
        # Liquid color: below water_level
        ly1, ly2 = min(fh - 1, wl + 10), min(fh, cy + ch - 5)
        if ly2 > ly1 and x2 > x1:
            liquid_color = np.mean(gt_first[ly1:ly2, x1:x2].reshape(-1, 3), axis=0)
        else:
            liquid_color = default
        # Empty cup color: above water_level
        ey1, ey2 = max(0, cy + 10), max(0, wl - 10)
        if ey2 > ey1 and x2 > x1:
            empty_color = np.mean(gt_first[ey1:ey2, x1:x2].reshape(-1, 3), axis=0)
        else:
            empty_color = np.array([255, 255, 255], dtype=float)
        return liquid_color, empty_color

    def _detect_liquid_level(self, frame: np.ndarray, cup: Dict,
                             liquid_color: np.ndarray,
                             empty_color: np.ndarray) -> Optional[int]:
        """Detect liquid surface y in a cup. Pixel is liquid if closer to
        liquid_color than empty_color."""
        cx, cy = cup['position']
        cw, ch = cup['size']
        fh, fw = frame.shape[:2]
        margin = max(5, cw // 20)
        x1, x2 = max(0, cx + margin), min(fw, cx + cw - margin)
        y1, y2 = cy, min(fh, cy + ch)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        roi_f = roi.astype(float)
        dist_liq = np.sqrt(np.sum((roi_f - liquid_color) ** 2, axis=2))
        dist_emp = np.sqrt(np.sum((roi_f - empty_color) ** 2, axis=2))
        liquid_mask = ((dist_liq < dist_emp) & (dist_liq < 100)).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_CLOSE, kernel)
        liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(liquid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None
        _, by, _, _ = cv2.boundingRect(largest)
        return y1 + by

    def _score_object_position(self, gen_last: np.ndarray, objects: List[Dict],
                               cups: List[Dict],
                               gt_hues: List[Optional[int]],
                               obj_infos: List[Dict],
                               cup_colors: List[Tuple]) -> Tuple[float, List]:
        """Per-object: check gen y-position matches expected float/sink level."""
        scores = []
        details = []
        for i, (obj, cup) in enumerate(zip(objects, cups)):
            gt_hue = gt_hues[i]
            if gt_hue is None:
                scores.append(0.0)
                details.append(f'obj{i}:hue_not_detected')
                continue

            should_fl = self._should_float(obj, cup)
            water_level = cup['water_level']
            cup_bottom = cup['position'][1] + cup['size'][1]
            obj_h = obj_infos[i]['height']
            water_depth = max(obj_h, cup_bottom - water_level)
            tol = min(obj_h, water_depth * 0.3)

            liq_c, emp_c = cup_colors[i]
            gen_wl = self._detect_liquid_level(gen_last, cup, liq_c, emp_c)

            if should_fl:
                expected_y = gen_wl if gen_wl is not None else obj['final_position'][1]
            else:
                expected_y = obj['final_position'][1]

            dets = self._detect_objects_in_column(gen_last, cup, gt_hue, obj_infos[i]['area'])
            if not dets:
                scores.append(0.0)
                details.append(f'obj{i}:not_found exp={"float" if should_fl else "sink"}')
                continue

            best = min(dets, key=lambda d: abs(d['cy'] - expected_y))
            gen_y = best['cy']
            gen_area = best['area']
            gt_area = obj_infos[i]['area']
            dist = abs(gen_y - expected_y)
            # Area ratio penalty
            area_ratio = gen_area / max(1, gt_area)
            if 0.8 < area_ratio < 1.3:
                area_penalty = 1.0
            elif 0.5 < area_ratio < 2.0:
                area_penalty = 0.7
            else:
                area_penalty = 0.3
            # Penalize if multiple objects detected in this column
            if len(dets) == 1:
                multi_penalty = 1.0
            elif len(dets) == 2:
                multi_penalty = 0.6
            else:
                multi_penalty = 0.2

            if should_fl:
                # Float: object should be near gen liquid level
                if dist < tol * 0.8:
                    sc = 1.0
                elif dist < tol * 1.5:
                    sc = 0.5
                else:
                    sc = 0.0
            else:
                # Sink: object should be below water level and near cup bottom
                if gen_y > water_level + tol:
                    if dist < tol:
                        sc = 1.0
                    elif dist < tol * 2:
                        sc = 0.5
                    else:
                        sc = 0.0
                else:
                    sc = 0.0

            sc *= multi_penalty * area_penalty
            scores.append(sc)
            details.append(
                f'obj{i}:{"float" if should_fl else "sink"} '
                f'gen_y={gen_y} exp_y={expected_y} wl={water_level} gen_wl={gen_wl} '
                f'gen_area={gen_area} gt_area={gt_area} ar={area_ratio:.2f} '
                f'cup_bot={cup_bottom} wd={water_depth:.0f} obj_h={obj_h} '
                f'tol={tol:.0f} dist={dist:.0f} '
                f'n_det={len(dets)} multi_pen={multi_penalty} sc={sc:.2f}')

        return (float(np.mean(scores)) if scores else 0.0), details

    def _score_scene_preservation(self, gen_first: np.ndarray, gen_last: np.ndarray,
                                  gt_first: np.ndarray,
                                  cups: List[Dict], objects: List[Dict],
                                  obj_infos: List[Dict],
                                  cup_colors: List[Tuple] = None) -> Tuple[float, Dict]:
        """Check background + liquid level preserved between gen first and last."""
        h, w = gen_first.shape[:2]
        # --- Background check (outside cups) ---
        cup_mask = np.zeros((h, w), dtype=np.uint8)
        for cup in cups:
            cx, cy = cup['position']
            cw, ch = cup['size']
            cup_mask[0:min(h, cy + ch + 10), max(0, cx - 10):min(w, cx + cw + 10)] = 255
        for i, obj in enumerate(objects):
            ox, oy = obj['position']
            sz = obj_infos[i]['height']
            cup_mask[max(0, oy - sz):min(h, oy + sz),
                     max(0, ox - sz):min(w, ox + sz)] = 255

        bg_mask = cv2.bitwise_not(cup_mask)
        mask_px = int((bg_mask > 0).sum())
        if mask_px > 0:
            diff = cv2.absdiff(gen_first, gen_last)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            changed = int((gray[bg_mask > 0] > 20).sum())
            bg_ratio = changed / mask_px
        else:
            bg_ratio = 0.0

        if bg_ratio < 0.02:
            bg_sc = 1.0
        elif bg_ratio < 0.05:
            bg_sc = 1.0 - (bg_ratio - 0.02) / 0.03 * 0.3
        elif bg_ratio < 0.10:
            bg_sc = 0.7 - (bg_ratio - 0.05) / 0.05 * 0.4
        elif bg_ratio < 0.20:
            bg_sc = 0.3 - (bg_ratio - 0.10) / 0.10 * 0.3
        else:
            bg_sc = 0.0

        # --- Liquid level check (inside cups) ---
        if cup_colors is None:
            cup_colors = [self._get_cup_colors(gt_first, cup) for cup in cups]
        level_diffs = []
        gen_levels = []
        for i, cup in enumerate(cups):
            gt_wl = cup['water_level']
            liq_c, emp_c = cup_colors[i]
            gen_level = self._detect_liquid_level(gen_last, cup, liq_c, emp_c)
            gen_levels.append(gen_level)
            if gen_level is not None:
                level_diffs.append(gt_wl - gen_level)  # positive = level rose
            else:
                level_diffs.append(0)

        obj_h_avg = max(20, int(np.mean([info['height'] for info in obj_infos])))
        cup_liq_scores = []
        for i, d in enumerate(level_diffs):
            # If liquid not detected at all, liquid color likely changed → penalize
            if gen_levels[i] is None:
                cup_liq_scores.append(0.0)
                continue
            ad = abs(d)
            if ad < obj_h_avg * 0.5:
                cup_liq_scores.append(1.0)
            elif ad < obj_h_avg:
                cup_liq_scores.append(0.7)
            elif ad < obj_h_avg * 1.5:
                cup_liq_scores.append(0.4)
            else:
                cup_liq_scores.append(0.0)
        liquid_sc = float(np.mean(cup_liq_scores)) if cup_liq_scores else 1.0

        sc = bg_sc * liquid_sc
        gt_wls = [cup['water_level'] for cup in cups]
        return round(sc, 4), {
            'bg_ratio': round(bg_ratio, 4), 'bg_sc': round(bg_sc, 2),
            'liquid_sc': round(liquid_sc, 2),
            'cup_liq_scores': str([round(s, 2) for s in cup_liq_scores]),
            'gt_wls': str(gt_wls),
            'gen_levels': str(gen_levels),
            'level_diffs': str([round(d, 1) for d in level_diffs]),
        }


    def _analyze_process(self, video_frames: List[np.ndarray], objects: List[Dict],
                         cups: List[Dict],
                         gt_hues: List[Optional[int]],
                         obj_infos: List[Dict]) -> Tuple[Dict, Dict]:
        """Single pass through all frames: non-teleport + one-per-column."""
        n_objs = len(objects)
        obj_ys = [[] for _ in range(n_objs)]
        multi_violations = 0
        total_checks = 0

        obj_areas = [[] for _ in range(n_objs)]
        for frame in video_frames:
            for i, (obj, cup) in enumerate(zip(objects, cups)):
                gt_hue = gt_hues[i]
                if gt_hue is None:
                    continue
                dets = self._detect_objects_in_column(frame, cup, gt_hue, obj_infos[i]['area'])
                total_checks += 1
                if len(dets) > 1:
                    multi_violations += 1
                if dets:
                    best = max(dets, key=lambda d: d['area'])
                    obj_ys[i].append(best['cy'])
                    obj_areas[i].append(best['area'])

        # --- Non-teleport: check objects pass through intermediate y-zones ---
        tp_scores, tp_details = [], []
        for i, (obj, cup) in enumerate(zip(objects, cups)):
            init_y = obj['position'][1]
            final_y = obj['final_position'][1]
            obj_h = obj_infos[i]['height']
            travel = abs(final_y - init_y)
            water_level = cup['water_level']
            should_fl = self._should_float(obj, cup)

            if travel < obj_h:
                tp_scores.append(1.0)
                tp_details.append(f'obj{i}:short_travel')
                continue

            ys = obj_ys[i]
            if len(ys) < 2:
                sc = 0.0 if len(ys) == 0 else 0.3
                tp_scores.append(sc)
                tp_details.append(f'obj{i}:few_det({len(ys)}) sc={sc}')
                continue

            if should_fl:
                deepest_idx = int(np.argmax(ys))
                deepest_y = ys[deepest_idx]

                descent_span = max(float(obj_h), float(water_level - init_y))
                approach_lo = init_y + 0.20 * descent_span
                approach_hi = water_level + 0.15 * obj_h
                approach_indices = [
                    idx for idx, y in enumerate(ys)
                    if 0 < idx < deepest_idx and approach_lo <= y <= approach_hi
                ]
                approach = bool(approach_indices)

                submerged_threshold = max(water_level, final_y) + 0.50 * obj_h
                submerged = (
                    0 < deepest_idx < len(ys) - 1
                    and deepest_y >= submerged_threshold
                )

                settle_tol = max(float(obj_h), 0.12 * travel)
                recovery_indices = [
                    idx for idx in range(deepest_idx + 1, len(ys))
                    if (
                        deepest_y - ys[idx] >= 0.50 * obj_h
                        and abs(ys[idx] - final_y) <= settle_tol
                    )
                ]
                # Recovery earns credit only when all phases occur in order;
                recovery = bool(approach and submerged and recovery_indices)

                sc = (
                    0.25 * float(approach)
                    + 0.35 * float(submerged)
                    + 0.40 * float(recovery)
                )
                tp_scores.append(sc)
                tp_details.append(
                    f'obj{i}:float_phases approach={approach} '
                    f'submerged={submerged} recovery={recovery} '
                    f'deepest={deepest_y}@{deepest_idx} '
                    f'approach_band=[{approach_lo:.0f},{approach_hi:.0f}] '
                    f'submerge_y>={submerged_threshold:.0f} '
                    f'ys={ys} sc={sc:.2f}')
                continue

            y_lo = min(init_y, final_y)
            y_hi = max(init_y, final_y)
            zone_h = travel / 3
            zones = [False, False, False]
            for y in ys:
                if y_lo <= y < y_lo + zone_h:
                    zones[0] = True
                elif y_lo + zone_h <= y < y_lo + 2 * zone_h:
                    zones[1] = True
                elif y <= y_hi + obj_h:
                    zones[2] = True

            hit = sum(zones)
            if hit == 3:
                sc = 1.0
            elif hit == 2:
                sc = 0.6 if zones[1] else 0.2
            else:
                sc = 0.2 if hit == 1 else 0.0

            # Check actual movement range — if y barely changed, object didn't move
            y_range = max(ys) - min(ys)
            if y_range < travel * 0.3:
                sc *= 0.3  # object barely moved

            entered_water = any(y >= water_level for y in ys)

            tp_scores.append(sc)
            tp_details.append(
                f'obj{i}:zones={zones} entered={entered_water} '
                f'y_range={y_range:.0f} travel={travel:.0f} '
                f'{"float" if should_fl else "sink"} sc={sc:.2f}')

        non_tp = float(np.mean(tp_scores)) if tp_scores else 0.0

        # --- One-per-column ---
        if total_checks > 0:
            v_ratio = multi_violations / total_checks
            if v_ratio < 0.10:
                opc = 1.0
            elif v_ratio < 0.20:
                opc = 0.7
            elif v_ratio < 0.35:
                opc = 0.4
            else:
                opc = 0.0
        else:
            opc = 0.0
            v_ratio = 0.0

        # --- Area consistency: median detected area vs GT area ---
        area_scores = []
        area_ratios = []
        for i in range(n_objs):
            gt_area = obj_infos[i]['area']
            if not obj_areas[i]:
                area_scores.append(0.0)
                area_ratios.append(0.0)
                continue
            median_area = float(np.median(obj_areas[i]))
            ar = median_area / max(1, gt_area)
            area_ratios.append(round(ar, 2))
            if 0.8 < ar < 1.3:
                area_scores.append(1.0)
            elif 0.5 < ar <2.0:
                area_scores.append(0.7)
            else:
                area_scores.append(0.3)
        area_sc = float(np.mean(area_scores)) if area_scores else 1.0

        scores = {
            'non_teleport': round(non_tp, 4),
            'one_per_column': round(opc, 4),
            'area_consistency': round(area_sc, 4),
        }
        det = {
            'tp_details': tp_details,
            'opc_violations': multi_violations,
            'opc_total': total_checks,
            'opc_v_ratio': round(v_ratio, 4),
            'area_ratios': str(area_ratios),
        }
        return scores, det

    def _score_scene_stability(self, video_frames: List[np.ndarray],
                               gt_first: np.ndarray,
                               cups: List[Dict],
                               objects: List[Dict],
                               obj_infos: List[Dict]) -> Tuple[float, Dict]:
        """Sample ~10 intermediate frames, average preservation scores."""
        n = len(video_frames)
        if n < 3:
            return 1.0, {'n_sampled': 0}

        gen_first = video_frames[0]
        n_samples = min(10, n - 1)
        indices = [int(n * i / (n_samples + 1)) for i in range(1, n_samples + 1)]
        frame_scores = []
        for idx in indices:
            sc, _ = self._score_scene_preservation(
                gen_first, video_frames[idx], gt_first,
                cups, objects, obj_infos)
            frame_scores.append(sc)

        avg_sc = float(np.mean(frame_scores))
        return round(avg_sc, 4), {
            'n_sampled': len(indices),
            'min_sc': round(min(frame_scores), 4),
            'max_sc': round(max(frame_scores), 4),
        }
        

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if not video_frames or gt_first_frame is None or gt_final_frame is None:
            return 0.0

        if video_frames[0].shape != gt_first_frame.shape:
            video_frames = [normalize_frame_size(f, gt_first_frame) for f in video_frames]

        import json as _json, os as _os
        gt_path = eval_info.get('gt_path', '')
        meta_path = _os.path.join(gt_path, 'metadata.json')
        if not _os.path.exists(meta_path):
            self._last_task_details = {'error': 'metadata not found'}
            return 0.0
        with open(meta_path) as f:
            _meta = _json.load(f)
        _sgt = _meta.get('semantic_ground_truth') or {}
        _params = _meta.get('parameters') or {}
        cups = _sgt.get('cups') or _params.get('cups')
        objects_raw = _sgt.get('objects') or _params.get('objects')
        if cups is None or objects_raw is None:
            self._last_task_details = {'error': 'metadata cups/objects missing'}
            return 0.0
        objects = [
            {**o, 'position': o.get('position', o.get('initial_position'))}
            for o in objects_raw
        ]
        if len(objects) != len(cups):
            self._last_task_details = {'error': f'obj/cup mismatch {len(objects)}/{len(cups)}'}
            return 0.0

        # Detect object hue + area + height from GT first/final diff
        obj_infos = [self._detect_gt_obj_info(gt_first_frame, gt_final_frame, cup)
                     for cup in cups]
        gt_hues = [info['hue'] for info in obj_infos]


        # Get cup colors for liquid level detection
        cup_colors = [self._get_cup_colors(gt_first_frame, cup) for cup in cups]

        # Final state
        pos_sc, pos_det = self._score_object_position(
            video_frames[-1], objects, cups, gt_hues, obj_infos, cup_colors)
        pres_sc, pres_det = self._score_scene_preservation(
            video_frames[0], video_frames[-1], gt_first_frame,
            cups, objects, obj_infos)

        # Process
        proc_scores, proc_det = self._analyze_process(
            video_frames, objects, cups, gt_hues, obj_infos)
        stab_sc, stab_det = self._score_scene_stability(
            video_frames, gt_first_frame, cups, objects, obj_infos)

        # Final state = object_position * scene_preservation (penalty)
        final_sc = round(pos_sc * (0.5 + 0.5 * pres_sc), 4)

        # Process = non_teleport * penalties (opc, stability, area)
        non_tp = proc_scores['non_teleport']
        opc = proc_scores['one_per_column']
        area_c = proc_scores['area_consistency']
        proc_sc = round(non_tp * (0.1 + 0.3 * opc + 0.3 * stab_sc + 0.3 * area_c), 4)

        scores = {
            'final_state': final_sc,
            'process': proc_sc,
        }

        self._last_task_details = {
            **scores,
            'object_position': round(pos_sc, 4),
            'scene_preservation': pres_sc,
            'non_teleport': non_tp,
            'one_per_column': opc,
            'area_consistency': area_c,
            'area_ratios': proc_det.get('area_ratios', ''),
            'scene_stability': stab_sc,
            'n_objects': len(objects),
            'special_cup': _sgt.get('special_cup_index', _params.get('special_cup_index')),
            'gt_hues': str(gt_hues),
            'obj_areas': str([info['area'] for info in obj_infos]),
            'obj_heights': str([info['height'] for info in obj_infos]),
            'opc_v_ratio': proc_det['opc_v_ratio'],
            'tp_details': proc_det.get('tp_details', []),
            **{f'pres_{k}': v for k, v in pres_det.items()},
            **{f'stab_{k}': v for k, v in stab_det.items()},
        }
        for d in pos_det:
            k = d.split(':')[0]
            self._last_task_details[k] = d.split(':', 1)[1]

        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

class PigmentColorMixingEvaluator(BaseEvaluator):
    """
    O-2: Pigment color mixing (subtractive) evaluator.

    Task: Two colored circles with a box between them. The box should be
    filled with the mixed color of the two circles.

    Evaluation:
    1. Mixed color correctness (60%): color in mixing region matches GT
    2. Object preservation (20%): two circles unchanged
    3. Background preservation (20%): background clean
    """

    TASK_WEIGHTS = {
        'mixing_color': 0.60,
        'object_preservation': 0.20,
        'background_clean': 0.20,
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

    def _get_bg_color(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        corners = [frame[2, 2], frame[2, w-3], frame[h-3, 2], frame[h-3, w-3]]
        return np.mean(corners, axis=0).astype(float)

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

        kernel = np.ones((5, 5), np.uint8)
        bg_color = self._get_bg_color(gt_first)

        # Mixing region: GT changed area (the box that gets filled)
        mixing_mask = self._detect_changed_region(gt_first, gt_last)

        # Circle mask: foreground in GT first (the two circles + box outline)
        fg_first = self._detect_fg_mask(gt_first)
        # All foreground (circles + filled box)
        fg_last = self._detect_fg_mask(gt_last)
        all_fg = cv2.bitwise_or(fg_first, fg_last)

        # --- 1. Mixed color correctness (60%): GT mixing region → gen color ---
        # Note: do NOT exclude dark pixels - mixed color can be very dark
        interior_px = int((mixing_mask > 0).sum())
        if interior_px > 0:
            gt_colors = gt_last[mixing_mask > 0].astype(float)
            gen_colors = gen_last[mixing_mask > 0].astype(float)
            gt_mean = np.mean(gt_colors, axis=0)
            gen_mean = np.mean(gen_colors, axis=0)
            color_dist = np.sqrt(np.sum((gt_mean - gen_mean) ** 2))

            # Fill check
            gen_dist_to_bg = np.sqrt(np.sum((gen_colors - bg_color) ** 2, axis=1))
            fill_ratio = float((gen_dist_to_bg > 30).sum()) / len(gen_dist_to_bg)

            if color_dist < 30:
                color_score = 1.0
            elif color_dist < 60:
                color_score = 1.0 - (color_dist - 30) / 30 * 0.3
            elif color_dist < 100:
                color_score = 0.7 - (color_dist - 60) / 40 * 0.4
            elif color_dist < 150:
                color_score = 0.3 - (color_dist - 100) / 50 * 0.3
            else:
                color_score = 0.0

            # Uniformity
            gen_std = np.mean(np.std(gen_colors, axis=0))
            if gen_std < 15:
                uni = 1.0
            elif gen_std < 30:
                uni = 1.0 - (gen_std - 15) / 15 * 0.3
            elif gen_std < 50:
                uni = 0.7 - (gen_std - 30) / 20 * 0.4
            else:
                uni = max(0.0, 0.3 - (gen_std - 50) / 50 * 0.3)

            mixing_score = color_score * fill_ratio * uni
        else:
            mixing_score = 0.0
            color_dist = -1
            fill_ratio = 0.0
            gen_std = -1

        mixing_details = {
            'color_dist': round(float(color_dist), 2),
            'fill_ratio': round(fill_ratio, 4),
            'color_std': round(float(gen_std), 2),
            'mixing_area': interior_px,
        }

        # --- 2. Object preservation (20%): two circles should remain unchanged ---
        # Circle-only mask: foreground in first frame minus the mixing region
        mixing_dilated = cv2.dilate(mixing_mask, kernel, iterations=1)
        circle_only = cv2.bitwise_and(fg_first, cv2.bitwise_not(mixing_dilated))
        preservation_score, preservation_details = self._pixel_diff_score(
            gen_first, gen_last, circle_only)

        # --- 3. Background clean (20%): gen_first vs gen_last outside all fg ---
        all_fg_dilated = cv2.dilate(all_fg, kernel, iterations=1)
        bg_mask = cv2.bitwise_not(all_fg_dilated)
        bg_score, bg_details = self._pixel_diff_score(
            gen_first, gen_last, bg_mask, thresholds=(0.005, 0.02, 0.05, 0.10))

        scores = {
            'mixing_color': round(mixing_score, 4),
            'object_preservation': round(preservation_score, 4),
            'background_clean': round(bg_score, 4),
        }
        self._last_task_details = {
            **scores,
            **{f'mix_{k}': v for k, v in mixing_details.items()},
            **{f'pres_{k}': v for k, v in preservation_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }

        TAU = 0.4
        gate = min(1.0, scores['object_preservation'] / TAU)
        total = (scores['mixing_color'] * self.TASK_WEIGHTS['mixing_color'] * gate
                 + scores['object_preservation'] * self.TASK_WEIGHTS['object_preservation']
                 + scores['background_clean'] * self.TASK_WEIGHTS['background_clean'])
        return float(total)

# Export all evaluators
OUT_OF_DOMAIN_50_EVALUATORS_PART3 = {
    'G-221_outline_innermost_square_data-generator': OutlineInnermostSquareEvaluator,
    'G-222_mark_tangent_point_of_circles_data-generator': MarkTangentPointEvaluator,
    'G-223_highlight_horizontal_lines_data-generator': SelectLineBlackCircleEvaluator,
    'G-240_add_borders_to_unbordered_shapes_data-generator': AddBordersToUnborderedEvaluator,
    'G-250_color_triple_intersection_red_data-generator': ColorTripleIntersectionEvaluator,
    'G-273_high_density_liquid_data-generator': HighDensityLiquidEvaluator,
    'O-2_pigment_color_mixing_subtractive_data-generator': PigmentColorMixingEvaluator
}
