"""
Specific evaluators for In-Domain_50 tasks (Part 3).
"""

import os
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import denoise_contour
from ..utils import normalize_frame_size, compute_ssim, safe_distance, color_distance
from ..utils import CircleSelectionProcessor, threshold_score
import os
import json
import shutil

def _shape_transform_score(scores, weights, tau=0.5):
    """Weighted sum for shape-transform tasks, gated so BOTH cores must be present.

    These tasks require the transform to be correct (``completion``) AND the shape
    to stay intact through the animation (``foreground_preservation``). Multiply the
    sum by a near-zero ramp on each core so a destroyed shape or a wrong transform
    collapses the score. GT (both cores ~1) is unchanged.
    """
    total = float(sum(scores[k] * weights[k] for k in weights))
    if 'completion' not in scores or 'foreground_preservation' not in scores:
        return total
    gate = (min(1.0, scores['completion'] / tau)
            * min(1.0, scores['foreground_preservation'] / tau))
    return float(total * gate)


class SelectAllHollowPointsEvaluator(BaseEvaluator):
    """
    G-158: Select multiple shapes evaluator.
    """
    
    TASK_WEIGHTS = {
        'consistency_score': 0.20,
        'match_score': 0.80
    }
    
    @staticmethod
    def _match_by_enclosed_shape(pred_frame, pred_circles, num_target_shapes, num_wrong_shapes):
        if pred_frame is None or not pred_circles or num_target_shapes <= 0:
            return None
        hsv = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        white = (S < 40) & (V > 200)
        red = ((H < 12) | (H > 168)) & (S > 80) & (V > 60)
        n_hollow = n_solid = 0
        for c in pred_circles:
            m = np.zeros(pred_frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(m, [c['contour']], -1, 255, -1)
            m = cv2.erode(m, np.ones((9, 9), np.uint8))
            content = (m > 0) & (~white) & (~red)
            n = int(content.sum())
            if n < 200:
                continue              
            ys, xs = np.nonzero(content)
            bbox = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
            if n / max(bbox, 1) > 0.55:
                n_solid += 1
            else:
                n_hollow += 1
        if n_hollow + n_solid == 0:
            return None
        correct = min(1.0, n_hollow / float(num_target_shapes))
        wrong = (n_solid / float(num_wrong_shapes)) if num_wrong_shapes > 0 else 0.0
        score = max(0.0, correct - 1.4 * wrong)
        n_rings = n_hollow + n_solid
        if n_rings > num_target_shapes:    
            score *= num_target_shapes / float(n_rings)
        return float(score)

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
            foreground_hsv_delta_tolerance=(15.0, 100.0, 100.0),
            background_hsv_delta_tolerance=(15.0, 100.0, 100.0),
            foreground_enlarge_pixels=20
        )
        circle_selection_info = circle_selection_processor.process(gt_first_frame, gt_final_frame, last_frame, debug_dir=debug_dir)
        
        scores = {}
        background_consistency_score = threshold_score(
            circle_selection_info['background_change_ratio'],
            [(0.05, 1.0), (0.2, 0.0)]
        )

        foreground_shape_types = [s['type'] for s in circle_selection_info["foreground_shapes"]]
        foreground_shape_fill_ratios = [s['fill_ratio'] for s in circle_selection_info["foreground_shapes"]]
        pred_foreground_shape_types = [s['type'] if s is not None else 'null' for s in circle_selection_info["pred_foreground_shapes"]]
        pred_foreground_shape_fill_ratios = [s['fill_ratio'] if s is not None else 0.0 for s in circle_selection_info["pred_foreground_shapes"]]

        num_shapes = len(foreground_shape_types)
        foreground_mismatch_count = 0
        for shape_id in range(num_shapes):
            gt_shape_type = foreground_shape_types[shape_id]
            pred_shape_type = pred_foreground_shape_types[shape_id]
            gt_fill_ratio = foreground_shape_fill_ratios[shape_id]
            pred_fill_ratio = pred_foreground_shape_fill_ratios[shape_id]
            gt_is_hollow = gt_fill_ratio < 0.9
            pred_is_hollow = pred_fill_ratio < 0.9
            if not (gt_shape_type == pred_shape_type and gt_is_hollow == pred_is_hollow):
                foreground_mismatch_count += 1

        foreground_mismatch_ratio = (
            foreground_mismatch_count / num_shapes if num_shapes > 0 else 0.0
        )
        foreground_consistency_score = 1.0 - foreground_mismatch_ratio

        circle_area_penalty_score = threshold_score(
            circle_selection_info['circle_color_mask_ratio'],
            [(0.3, 1.0), (0.5, 0.0)]
        )
        scores['consistency_score'] = (background_consistency_score + foreground_consistency_score + circle_area_penalty_score) / 3

        per_shape_scores = [0.0 for _ in range(len(circle_selection_info['is_target_shape']))]
        ambiguous_circles_count = 0
        selected_threshold = 0.5
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
                        [(0.5, 0.0), (0.7, 1.0)]
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
        num_wrong_shapes = len(circle_selection_info['is_target_shape']) - num_target_shapes
        correct_match_score = 0.0
        wrong_match_score = 0.0
        for shape_id in range(len(per_shape_scores)):
            if circle_selection_info['is_target_shape'][shape_id] == 1:
                correct_match_score += per_shape_scores[shape_id] / num_target_shapes
            else:
                wrong_match_score += per_shape_scores[shape_id] / num_wrong_shapes
        wrong_match_score *= 1.4
        scores['match_score'] = max(0, (correct_match_score - wrong_match_score) * (0.5 + 0.5 * foreground_consistency_score) * (0.4 + 0.6 * circle_area_penalty_score) - ambiguous_score)
        # Penalise drawing more circles than there are target shapes. 
        if num_target_shapes > 0 and num_circles > num_target_shapes:
            scores['match_score'] *= num_target_shapes / num_circles

        fallback = self._match_by_enclosed_shape(
            last_frame, circle_selection_info['pred_circles'],
            num_target_shapes, num_wrong_shapes,
        )
        if fallback is not None:
            scores['match_score'] = max(scores['match_score'], fallback)

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
                'foreground_mismatch_count': foreground_mismatch_count,
                'foreground_mismatch_ratio': foreground_mismatch_ratio,
                'foreground_shape_types': foreground_shape_types,
                'pred_foreground_shape_types': pred_foreground_shape_types,
                'foreground_shape_fill_ratios': foreground_shape_fill_ratios,
                'pred_foreground_shape_fill_ratios': pred_foreground_shape_fill_ratios,
                'ambiguous_circles_count': ambiguous_circles_count,
                'correct_match_score': correct_match_score,
                'wrong_match_score': wrong_match_score,
                **scores,
                "total_score": total_score
            }
            with open(os.path.join(debug_dir, "debug_info.json"), "w") as f:
                json.dump(debug_info, f)
        self._last_task_details = {
            'background_consistency_score': background_consistency_score,
            'foreground_consistency_score': foreground_consistency_score,
            'circle_area_penalty_score': circle_area_penalty_score,
            'foreground_mismatch_count': foreground_mismatch_count,
            'foreground_mismatch_ratio': foreground_mismatch_ratio,
            'num_shapes': num_shapes,
            'ambiguous_circles_count': ambiguous_circles_count,
            'ambiguous_circles_ratio': ambiguous_circles_ratio,
            'ambiguous_score': ambiguous_score,
            'num_circles': num_circles,
            'num_target_shapes': num_target_shapes,
            'num_wrong_shapes': num_wrong_shapes,
            'correct_match_score': correct_match_score,
            'wrong_match_score': wrong_match_score,
            'background_change_ratio': circle_selection_info['background_change_ratio'],
            'circle_color_mask_ratio': circle_selection_info['circle_color_mask_ratio'],
            'consistency_score': scores['consistency_score'],
            'match_score': scores['match_score'],
            'total_score': total_score,
        }
        return total_score


class ConstructConcentricRingEvaluator(BaseEvaluator):
    """
    G-194: Construct concentric ring evaluator.

    Scoring:
    - arrangement      (60%): circles are concentric + centered on frame
    - fore_consistency (20%): all GT rings matched, no extra rings
    - back_consistency (20%): non-ring area is white
    """

    def _detect_rings(self, frame: np.ndarray) -> List[Dict]:
        """Detect ring-shaped circles using contour-based approach. Returns list of {center, radius, color_bgr}."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 10, 30)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Find circular contours
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            if circularity > 0.7:
                candidates.append([cx, cy, r])

        candidates.sort(key=lambda x: x[2])

        # Merge contours from same ring: each hollow ring produces multiple contours
        # (inner/outer edges). Group by similar radius + center.
        used = set()
        merged = []
        for i in range(len(candidates)):
            if i in used:
                continue
            group = [candidates[i]]
            used.add(i)
            for j in range(i + 1, len(candidates)):
                if j in used:
                    continue
                mean_r = float(np.mean([x[2] for x in group]))
                mean_cx = float(np.mean([x[0] for x in group]))
                mean_cy = float(np.mean([x[1] for x in group]))
                if (abs(candidates[j][2] - mean_r) < 20 and
                    abs(candidates[j][0] - mean_cx) < 15 and
                    abs(candidates[j][1] - mean_cy) < 15):
                    group.append(candidates[j])
                    used.add(j)
            mean_r = float(np.mean([x[2] for x in group]))
            mean_cx = float(np.mean([x[0] for x in group]))
            mean_cy = float(np.mean([x[1] for x in group]))
            merged.append({'center': (mean_cx, mean_cy), 'radius': mean_r})

        # Sample color at ring midpoint
        rings = []
        for m in merged:
            cx, cy, r = int(m['center'][0]), int(m['center'][1]), int(m['radius'])
            angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            samples = []
            # Sample at multiple radii around the detected radius to capture the ring color
            # This handles cases where the ring is thick and the exact radius may vary
            for radius_offset in [-4, 0, 4]:  # Sample at r-4, r, r+4
                for a in angles:
                    px = int(cx + (r + radius_offset) * np.cos(a))
                    py = int(cy + (r + radius_offset) * np.sin(a))
                    if 0 <= px < w and 0 <= py < h:
                        samples.append(frame[py, px].tolist())
            color_bgr = tuple(int(v) for v in np.mean(samples, axis=0)) if samples else (128, 128, 128)
            rings.append({'center': (cx, cy), 'radius': int(r), 'color_bgr': color_bgr})

        rings.sort(key=lambda x: x['radius'])
        return rings

    def _match_rings(self, gt_rings: List[Dict], pred_rings: List[Dict]) -> List[Optional[Dict]]:
        """One-to-one match each GT ring to a pred ring by radius+color. Returns list same length as gt_rings."""
        used = set()
        matched = []
        for gt_idx, gt in enumerate(gt_rings):
            best = None
            best_cost = float('inf')
            for i, pred in enumerate(pred_rings):
                if i in used:
                    continue
                r_diff = abs(pred['radius'] - gt['radius']) / max(gt['radius'], 1)
                c_diff = color_distance(pred['color_bgr'], gt['color_bgr'])
                if r_diff < 0.08 and c_diff < 150:
                    quality = 1.0 if c_diff <= 60 else 1.0 - (c_diff - 60) / 90.0
                    cost = r_diff + c_diff / 60.0   
                    if cost < best_cost:
                        best_cost = cost
                        best = (i, dict(pred, match_quality=quality))
            if best is not None:
                used.add(best[0])
                matched.append(best[1])
            else:
                matched.append(None)


        return matched

    def _evaluate_fore_consistency(self, gt_rings: List[Dict], pred_rings: List[Dict]) -> float:
        n_gt = len(gt_rings)
        n_pred = len(pred_rings)
        if n_gt == 0:
            return 1.0 if n_pred == 0 else max(0.0, 1.0 - n_pred * 0.5)
        if n_pred < n_gt:
            return 0.0
        matched = self._match_rings(gt_rings, pred_rings)
        all_matched = all(m is not None for m in matched)
        if not all_matched:
            return 0.0
        extra = n_pred - n_gt
        return max(0.0, 1.0 - extra * 0.5)

    def _evaluate_arrangement(self, matched_pred: List[Dict], frame_shape: Tuple) -> float:
        valid = [m for m in matched_pred if m is not None]
        if len(valid) < 2:
            return 0.0
        h, w = frame_shape[:2]
        ref_size = min(h, w)

        # Concentric check: max pairwise center distance
        centers = [m['center'] for m in valid]
        max_dist = 0.0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                max_dist = max(max_dist, safe_distance(centers[i], centers[j]))
        is_concentric = max_dist < 0.05 * ref_size

        # Center check: mean center vs frame center
        mean_cx = float(np.mean([c[0] for c in centers]))
        mean_cy = float(np.mean([c[1] for c in centers]))
        frame_center = (w / 2.0, h / 2.0)
        dist_to_center = safe_distance((mean_cx, mean_cy), frame_center)
        is_at_center = dist_to_center < 0.1 * ref_size
        base = 1.0 if (is_concentric and is_at_center) else (0.5 if is_concentric else 0.0)
        mean_q = float(np.mean([m.get('match_quality', 1.0) for m in valid]))
        return base * mean_q

    def _evaluate_back_consistency(self, frame: np.ndarray, matched_pred: List[Dict]) -> float:
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for m in matched_pred:
            if m is None:
                continue
            cx, cy = m['center']
            r = m['radius'] + 8
            cv2.circle(mask, (cx, cy), r, 255, -1)

        bg_pixels = frame[mask == 0]
        if len(bg_pixels) == 0:
            return 1.0
        white_ratio = float(np.mean(np.all(bg_pixels >= 240, axis=1)))
        return white_ratio

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if gt_final_frame is None and gt_frames:
            gt_final_frame = gt_frames[-1]
        if gt_first_frame is None and gt_frames:
            gt_first_frame = gt_frames[0]
        if not video_frames or gt_final_frame is None:
            return 0.0

        # Align the prediction to GT: GT is the reference answer, so its resolution
        # is the baseline.
        final_frame = video_frames[-1]
        if final_frame.shape[:2] != gt_final_frame.shape[:2]:
            final_frame = cv2.resize(final_frame, (gt_final_frame.shape[1], gt_final_frame.shape[0]))

        gt_rings = self._detect_rings(gt_final_frame)
        pred_rings = self._detect_rings(final_frame)
        fore_consistency = self._evaluate_fore_consistency(gt_rings, pred_rings)
        matched_pred = self._match_rings(gt_rings, pred_rings)

        arrangement = self._evaluate_arrangement(matched_pred, final_frame.shape)
        if arrangement == 0.0 and len(pred_rings) == len(gt_rings) >= 2:
            arrangement = 0.6 * self._evaluate_arrangement(pred_rings, final_frame.shape)
        back_consistency = self._evaluate_back_consistency(final_frame, matched_pred)

        consistency = 0.5 * fore_consistency + 0.5 * back_consistency
        score = arrangement * (0.6 + 0.4 * consistency)
        self._last_task_details = {
            'arrangement': arrangement,
            'fore_consistency': fore_consistency,
            'back_consistency': back_consistency,
        }
        return score


class ShapeOutlineFillEvaluator(BaseEvaluator):
    """
    O-10: Shape outline fill evaluator.

    Dimensions:
        - completion (60%): focuses on the bottom-right quadrant of the final frame.
          Extracts the largest shape and evaluates its structural features:
          shape, outline_style, size, color, position against the GT final frame.
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

    OUTLINE_SHAPE_FEATURE_WEIGHTS = {
        "shape": 0.30,
        "outline_style": 0.30,
        "size": 0.25,
        "color": 0.10,
        "position": 0.05,
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

    @staticmethod
    def _classify_outline_style(
        contour: np.ndarray,
        fg_mask: np.ndarray,
        quadrant_shape: Tuple[int, int],
    ) -> str:
        """
        Classify the outline style of the largest shape as one of:
          'thin_outline', 'thick_outline', 'filled'

        Strategy:
          1. Draw the filled shape mask and the contour (outline) mask.
          2. Compute fill_ratio = filled_pixels / total_shape_pixels.
             - High fill_ratio (~1.0) -> 'filled'
             - Low fill_ratio -> outline only; distinguish thin vs thick by
               the mean stroke width estimated from area / perimeter.
        """
        qh, qw = quadrant_shape
        filled_mask = np.zeros((qh, qw), dtype=np.uint8)
        cv2.drawContours(filled_mask, [contour], -1, 255, thickness=-1)

        fg_inside = cv2.bitwise_and(fg_mask, filled_mask)
        total_filled = int(np.sum(filled_mask > 0))
        actual_fg = int(np.sum(fg_inside > 0))

        if total_filled == 0:
            return "thin_outline"

        fill_ratio = actual_fg / total_filled

        if fill_ratio >= 0.80:
            return "filled"

        perimeter = float(cv2.arcLength(contour, True))
        stroke_area = actual_fg
        if perimeter > 0:
            estimated_stroke_width = stroke_area / perimeter
        else:
            estimated_stroke_width = 0.0

        if estimated_stroke_width >= 2.2:
            return "thick_outline"
        return "thin_outline"

    def _extract_quadrant_shape_features(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Extracts features (contour, outline_style, area, centroid, color) of the largest shape
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
        outline_style = self._classify_outline_style(largest_contour, fg_mask, (qh, qw))

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
            "outline_style": outline_style,
        }

    def _compute_outline_completion_score(
        self,
        gt_shape_features: Optional[Dict],
        pred_shape_features: Optional[Dict],
    ) -> Tuple[float, Dict[str, float]]:
        """Compute completion and sub-scores from largest-shape feature comparison for outline tasks."""
        if gt_shape_features is None or pred_shape_features is None:
            return 0.0, {
                "shape": 0.0,
                "outline_style": 0.0,
                "size": 0.0,
                "color": 0.0,
                "position": 0.0,
                "shape_contour": 0.0,
                "shape_vertex": 0.0,
            }

        # 1.1 shape contour similarity
        match_score = cv2.matchShapes(gt_shape_features["contour"], pred_shape_features["contour"], cv2.CONTOURS_MATCH_I1, 0.0)

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

        # If the shape score is less than 0.6, incorrect shape is generated.
        if shape_score < 0.6:
            return 0.0, {
                "shape": shape_score,
                "outline_style": 0.0,
                "size": 0.0,
                "color": 0.0,
                "position": 0.0,
                "shape_contour": shape_score_from_contour,
                "shape_vertex": vertex_score,
            }

        # 2. outline style similarity
        style_order = {"thin_outline": 0, "thick_outline": 1, "filled": 3}
        gt_style = gt_shape_features["outline_style"]
        pred_style = pred_shape_features["outline_style"]
        style_diff = abs(style_order.get(gt_style, 0) - style_order.get(pred_style, 0))
        if style_diff == 0:
            outline_style_score = 1.0
        elif style_diff == 1:
            outline_style_score = 0.4
        else:
            outline_style_score = 0.0

        # 3. size similarity
        area_ratio = min(gt_shape_features["area"], pred_shape_features["area"]) / max(
            gt_shape_features["area"], pred_shape_features["area"], 1e-6
        )
        extent_ratio = min(gt_shape_features["bbox_extent"], pred_shape_features["bbox_extent"]) / max(
            gt_shape_features["bbox_extent"], pred_shape_features["bbox_extent"], 1e-6
        )
        size_score = float(0.80 * area_ratio + 0.20 * extent_ratio)

        # 4. color similarity
        color_dist = float(np.linalg.norm(gt_shape_features["mean_bgr"] - pred_shape_features["mean_bgr"]))
        color_score = float(max(0.0, 1.0 - color_dist / np.sqrt(3.0 * (255.0 ** 2))))

        # 5. position similarity
        gt_cx, gt_cy = gt_shape_features["centroid"]
        pred_cx, pred_cy = pred_shape_features["centroid"]
        position_dist = float(np.sqrt((gt_cx - pred_cx) ** 2 + (gt_cy - pred_cy) ** 2))
        position_score = float(max(0.0, 1.0 - position_dist / np.sqrt(2.0)))

        completion = (
            self.OUTLINE_SHAPE_FEATURE_WEIGHTS["shape"] * shape_score
            + self.OUTLINE_SHAPE_FEATURE_WEIGHTS["outline_style"] * outline_style_score
            + self.OUTLINE_SHAPE_FEATURE_WEIGHTS["size"] * size_score
            + self.OUTLINE_SHAPE_FEATURE_WEIGHTS["color"] * color_score
            + self.OUTLINE_SHAPE_FEATURE_WEIGHTS["position"] * position_score
        )
        completion = float(max(0.0, min(1.0, completion)))
        completion_details = {
            "shape": shape_score,
            "outline_style": outline_style_score,
            "size": size_score,
            "color": color_score,
            "position": position_score,
            "shape_contour": shape_score_from_contour,
            "shape_vertex": vertex_score,
            "gt_outline_style": gt_style,
            "pred_outline_style": pred_style,
        }
        return completion, completion_details

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Evaluate O-10 with final-frame completion and preservation metrics."""
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
        completion_score, completion_details = self._compute_outline_completion_score(gt_shape_features, pred_shape_features)
        scores["completion"] = completion_score

        # 2) foreground_preservation (25%): compare first vs final on non-background and non-changed region.
        change_mask = self._shape_change_mask(gt_first, gt_last)
        first_fg, first_bg = self._frame_masks(first_frame)
        fg_compare_mask = cv2.bitwise_and(first_fg, cv2.bitwise_not(change_mask))
        scores["foreground_preservation"] = self._pixel_similarity(first_frame, last_frame, fg_compare_mask, strictness=3.0, min_cutoff=0.6)

        # 3) background_preservation (15%): compare first vs final on stable background region.
        bg_compare_mask = cv2.bitwise_and(first_bg, cv2.bitwise_not(change_mask))
        scores["background_preservation"] = self._pixel_similarity(first_frame, last_frame, bg_compare_mask, strictness=3.0, min_cutoff=0.6)

        self._last_task_details = {
            **scores,
            "completion_details": completion_details,
        }
        return _shape_transform_score(scores, self.TASK_WEIGHTS)


class ShapeColorThenScaleEvaluator(BaseEvaluator):
    """
    O-12: Shape color then scale evaluator.

    Dimensions:
        - completion (60%): split final frame into 2x3 cells; compare first shape
          (row=2,col=2) and second shape (row=2,col=3) using feature similarity:
          shape, size, color, position, each with 50% weight.
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

    # first shape: color task
    FIRST_SHAPE_WEIGHTS = {
        "shape": 0.25,
        "size": 0.20,
        "color": 0.40,
        "position": 0.15,
    }

    # second shape: scale task
    SECOND_SHAPE_WEIGHTS = {
        "shape": 0.25,
        "size": 0.40,
        "color": 0.20,
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
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter + 1e-6))

        return {
            "contour": largest_contour,
            "area": float(area),
            "area_ratio": float(area / float(ch * cw)),
            "centroid": (float(cx / cw), float(cy / ch)),
            "mean_bgr": mean_bgr,
            "bbox_aspect_ratio": float(bw / max(bh, 1)),
            "bbox_extent": float(area / max(float(bw * bh), 1.0)),
            "vertex_count": int(len(approx)),
            "circularity": max(0.0, min(1.0, circularity)),
        }

    def _compute_shape_score(
        self,
        gt_shape_features: Optional[Dict],
        pred_shape_features: Optional[Dict],
        weights: Dict[str, float],
        color_task: bool = False,
        size_task: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute shape score and sub-scores from shape feature comparison."""
        if gt_shape_features is None or pred_shape_features is None:
            return 0.0, {
                "shape": 0.0,
                "size": 0.0,
                "color": 0.0,
                "position": 0.0,
                "shape_contour": 0.0,
                "shape_hu": 0.0,
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
        size_ratio = float(0.80 * area_ratio + 0.20 * extent_ratio)
        
        if size_task:
            size_score = size_ratio if size_ratio >= 0.75 else 0.0
        else:
            size_score = size_ratio

        # 3. color similarity
        color_dist = float(np.linalg.norm(gt_shape_features["mean_bgr"] - pred_shape_features["mean_bgr"]))
        color_ratio = float(max(0.0, 1.0 - color_dist / np.sqrt(3.0 * (255.0 ** 2))))

        if color_task:
            color_score = color_ratio if color_ratio >= 0.75 else 0.0
        else:
            color_score = color_ratio
        
        # 4. position similarity
        gt_cx, gt_cy = gt_shape_features["centroid"]
        pred_cx, pred_cy = pred_shape_features["centroid"]
        position_dist = float(np.sqrt((gt_cx - pred_cx) ** 2 + (gt_cy - pred_cy) ** 2))
        position_score = float(max(0.0, 1.0 - position_dist / np.sqrt(2.0)))

        total_score = shape_gate * (
            weights["shape"] * shape_score
            + weights["size"] * size_score
            + weights["color"] * color_score
            + weights["position"] * position_score
        )

        if size_task:
            # for color task, accuracy penalty is applied to size ratio.
            total_score = total_score * size_ratio
        if color_task:
            # for size task, accuracy penalty is applied to color ratio.
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
        """Evaluate O-12 with final-frame completion and preservation metrics."""
        scores: Dict[str, float] = {}

        if len(video_frames) < 2 or gt_first_frame is None or gt_final_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        if last_frame.shape[:2] != gt_final_frame.shape[:2]:
            first_frame = normalize_frame_size(first_frame, gt_final_frame)
            last_frame = normalize_frame_size(last_frame, gt_final_frame)
        gt_first, gt_last = gt_first_frame, gt_final_frame

        # 1) completion (60%): split final frame into 3x3 grid.
        gt_first_shape_features = self._extract_cell_shape_features(gt_last, row_idx=1, col_idx=1)
        gt_second_shape_features = self._extract_cell_shape_features(gt_last, row_idx=1, col_idx=2)
        
        pred_first_shape_features = self._extract_cell_shape_features(last_frame, row_idx=1, col_idx=1)
        pred_second_shape_features = self._extract_cell_shape_features(last_frame, row_idx=1, col_idx=2)

        first_completion, first_completion_details = self._compute_shape_score(
            gt_first_shape_features,
            pred_first_shape_features,
            self.FIRST_SHAPE_WEIGHTS,
            color_task=True,
            size_task=False,
        )
        second_completion, second_completion_details = self._compute_shape_score(
            gt_second_shape_features,
            pred_second_shape_features,
            self.SECOND_SHAPE_WEIGHTS,
            color_task=False,
            size_task=True,
        )
        scores["completion"] = 0.5 * first_completion + 0.5 * second_completion

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
                "first_shape": first_completion_details,
                "second_shape": second_completion_details,
            },
        }
        return _shape_transform_score(scores, self.TASK_WEIGHTS)


class ShapeOutlineThenMoveEvaluator(BaseEvaluator):
    """
    O-13: Shape outline then move evaluator.

    Dimensions:
        - completion (60%): split frame into 3 columns, use GT targets as anchors.
          - first shape (col 2): outline task.
          - second shape (col 3): move task.
        - foreground_preservation (25%): compare first vs generated final on foreground.
        - background_preservation (15%): compare first vs generated final on background.
    """

    TASK_WEIGHTS = {
        "completion": 0.60,
        "foreground_preservation": 0.25,
        "background_preservation": 0.15,
    }

    # first shape: outline task
    FIRST_SHAPE_WEIGHTS = {
        "shape": 0.25,
        "outline_style": 0.40,
        "size": 0.20,
        "color": 0.10,
        "position": 0.05,
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

    @staticmethod
    def _classify_outline_style(
        contour: np.ndarray,
        fg_mask: np.ndarray,
        roi_shape: Tuple[int, int],
    ) -> str:
        """
        Classify the outline style of a shape as 'thin_outline', 'thick_outline', or 'filled'.

        Strategy:
          1. Draw the filled shape mask and compute fill_ratio = fg_pixels / total_filled_pixels.
             - High fill_ratio (~1.0) -> 'filled'
             - Low fill_ratio -> outline only; distinguish thin vs thick by estimated stroke width.
        """
        rh, rw = roi_shape
        filled_mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.drawContours(filled_mask, [contour], -1, 255, thickness=-1)

        fg_inside = cv2.bitwise_and(fg_mask, filled_mask)
        total_filled = int(np.sum(filled_mask > 0))
        actual_fg = int(np.sum(fg_inside > 0))

        if total_filled == 0:
            return "thin_outline"

        fill_ratio = actual_fg / total_filled
        if fill_ratio >= 0.80:
            return "filled"

        perimeter = float(cv2.arcLength(contour, True))
        estimated_stroke_width = actual_fg / perimeter if perimeter > 0 else 0.0
        return "thick_outline" if estimated_stroke_width >= 2.2 else "thin_outline"

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
        outline_style = self._classify_outline_style(largest_contour, fg_mask, (ch, cw))

        return {
            "contour": largest_contour,
            "area": float(area),
            "centroid": (float(cx / cw), float(cy / ch)),
            "mean_bgr": mean_bgr,
            "bbox_extent": float(area / max(float(bw * bh), 1.0)),
            "vertex_count": int(len(approx)),
            "outline_style": outline_style,
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
        outline_style = self._classify_outline_style(target_contour, fg_mask, (rh, rw))

        return {
            "contour": target_contour,
            "area": area,
            "centroid": (float(global_cx / w), float(global_cy / h)),
            "mean_bgr": mean_bgr,
            "bbox_extent": float(area / max(float(bw * bh), 1.0)),
            "vertex_count": int(len(approx)),
            "outline_style": outline_style,
        }

    def _compute_shape_score(
        self,
        gt_shape_features: Optional[Dict],
        pred_shape_features: Optional[Dict],
        weights: Dict[str, float],
        outline_task: bool = False,
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
        
        # 2. outline style similarity
        style_order = {"thin_outline": 0, "thick_outline": 1, "filled": 3}
        gt_style = gt_shape_features["outline_style"]
        pred_style = pred_shape_features["outline_style"]
        style_diff = abs(style_order.get(gt_style, 0) - style_order.get(pred_style, 0))
        if style_diff == 0:
            outline_style_score = 1.0
        elif style_diff == 1:
            outline_style_score = 0.4
        else:
            outline_style_score = 0.0

        # 3. size similarity
        area_ratio = min(gt_shape_features["area"], pred_shape_features["area"]) / max(
            gt_shape_features["area"], pred_shape_features["area"], 1e-6
        )
        extent_ratio = min(gt_shape_features["bbox_extent"], pred_shape_features["bbox_extent"]) / max(
            gt_shape_features["bbox_extent"], pred_shape_features["bbox_extent"], 1e-6
        )
        size_score = float(0.80 * area_ratio + 0.20 * extent_ratio)

        # 4. color similarity
        color_dist = float(np.linalg.norm(gt_shape_features["mean_bgr"] - pred_shape_features["mean_bgr"]))
        color_score = float(max(0.0, 1.0 - color_dist / np.sqrt(3.0 * (255.0 ** 2))))

        # 5. position similarity
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
                "outline_style": outline_style_score,
                "size": size_score,
                "color": color_score,
                "position": position_score,
                "shape_contour": shape_score_from_contour,
                "shape_vertex": vertex_score,
            }

        if outline_task:
            total_score = shape_gate * (
                weights["shape"] * shape_score
                + weights["outline_style"] * outline_style_score
                + weights["size"] * size_score
                + weights["color"] * color_score
                + weights["position"] * position_score
            )
        else:
            total_score = shape_gate * (
                weights["shape"] * shape_score
                + weights["size"] * size_score
                + weights["color"] * color_score
                + weights["position"] * position_score
            )
        return float(max(0.0, min(1.0, total_score))), {
            "shape": shape_score,
            "outline_style": outline_style_score,
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
        eval_info: Dict,
    ) -> float:
        """Evaluate O-13 with final-frame completion and preservation metrics."""
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
            gt_first_shape_features, pred_first_shape_features,
            weights=self.FIRST_SHAPE_WEIGHTS,
            outline_task=True,
            position_task=False
        )
        second_shape_score, second_shape_details = self._compute_shape_score(
            gt_second_shape_features, pred_second_shape_features,
            weights=self.SECOND_SHAPE_WEIGHTS,
            outline_task=False,
            position_task=True
        )
        scores["completion"] = 0.5 * first_shape_score + 0.5 * second_shape_score

        # 2) foreground_preservation (25%): compare first vs final on non-background and non-changed region.
        change_mask = self._shape_change_mask(gt_first, gt_last)
        first_fg, first_bg = self._frame_masks(first_frame)
        fg_compare_mask = cv2.bitwise_and(first_fg, cv2.bitwise_not(change_mask))
        scores["foreground_preservation"] = self._pixel_similarity(first_frame, last_frame, fg_compare_mask, min_cutoff=0.6)

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
        return _shape_transform_score(scores, self.TASK_WEIGHTS)


class ShapeScaleThenOutlineEvaluator(BaseEvaluator):
    """
    O-14: Shape scale then outline evaluator.

    Dimensions:
        - completion (60%): split final frame into a 2x3 grid.
          - first shape: row=2, col=2, scale task.
          - second shape: row=2, col=3, outline task.
          Final completion is the average of these two shape scores.
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

    # first shape: scale task
    FIRST_SHAPE_WEIGHTS = {
        "shape": 0.25,
        "size": 0.40,
        "color": 0.20,
        "position": 0.15,
    }

    # second shape: outline task
    SECOND_SHAPE_WEIGHTS = {
        "shape": 0.25,
        "outline_style": 0.40,
        "size": 0.20,
        "color": 0.10,
        "position": 0.05,
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

    @staticmethod
    def _classify_outline_style(
        contour: np.ndarray,
        fg_mask: np.ndarray,
        roi_shape: Tuple[int, int],
    ) -> str:
        """
        Classify the outline style of a shape as 'thin_outline', 'thick_outline', or 'filled'.

        Strategy:
          1. Draw the filled shape mask and compute fill_ratio = fg_pixels / total_filled_pixels.
             - High fill_ratio (~1.0) -> 'filled'
             - Low fill_ratio -> outline only; distinguish thin vs thick by estimated stroke width.
        """
        rh, rw = roi_shape
        filled_mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.drawContours(filled_mask, [contour], -1, 255, thickness=-1)

        fg_inside = cv2.bitwise_and(fg_mask, filled_mask)
        total_filled = int(np.sum(filled_mask > 0))
        actual_fg = int(np.sum(fg_inside > 0))

        if total_filled == 0:
            return "thin_outline"

        fill_ratio = actual_fg / total_filled
        if fill_ratio >= 0.80:
            return "filled"

        perimeter = float(cv2.arcLength(contour, True))
        estimated_stroke_width = actual_fg / perimeter if perimeter > 0 else 0.0
        return "thick_outline" if estimated_stroke_width >= 2.2 else "thin_outline"
    
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
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter + 1e-6))
        outline_style = self._classify_outline_style(largest_contour, fg_mask, (ch, cw))

        return {
            "contour": largest_contour,
            "area": float(area),
            "area_ratio": float(area / float(ch * cw)),
            "centroid": (float(cx / cw), float(cy / ch)),
            "mean_bgr": mean_bgr,
            "bbox_aspect_ratio": float(bw / max(bh, 1)),
            "bbox_extent": float(area / max(float(bw * bh), 1.0)),
            "vertex_count": int(len(approx)),
            "circularity": max(0.0, min(1.0, circularity)),
            "outline_style": outline_style,
        }

    def _compute_shape_score(
        self,
        gt_shape_features: Optional[Dict],
        pred_shape_features: Optional[Dict],
        weights: Dict[str, float],
        scale_task: bool = False,
        outline_task: bool = False,
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
        
        # 2. outline style similarity
        style_order = {"thin_outline": 0, "thick_outline": 1, "filled": 3}
        gt_style = gt_shape_features["outline_style"]
        pred_style = pred_shape_features["outline_style"]
        style_diff = abs(style_order.get(gt_style, 0) - style_order.get(pred_style, 0))
        if style_diff == 0:
            outline_style_score = 1.0
        elif style_diff == 1:
            outline_style_score = 0.4
        else:
            outline_style_score = 0.0

        # 3. size similarity
        area_ratio = min(gt_shape_features["area"], pred_shape_features["area"]) / max(
            gt_shape_features["area"], pred_shape_features["area"], 1e-6
        )
        extent_ratio = min(gt_shape_features["bbox_extent"], pred_shape_features["bbox_extent"]) / max(
            gt_shape_features["bbox_extent"], pred_shape_features["bbox_extent"], 1e-6
        )
        size_ratio = float(0.80 * area_ratio + 0.20 * extent_ratio)
        if scale_task:
            size_score = size_ratio if size_ratio >= 0.75 else 0.0
        else:
            size_score = size_ratio

        # 4. color similarity
        color_dist = float(np.linalg.norm(gt_shape_features["mean_bgr"] - pred_shape_features["mean_bgr"]))
        color_score = float(max(0.0, 1.0 - color_dist / np.sqrt(3.0 * (255.0 ** 2))))

        # 5. position similarity
        gt_cx, gt_cy = gt_shape_features["centroid"]
        pred_cx, pred_cy = pred_shape_features["centroid"]
        position_dist = float(np.sqrt((gt_cx - pred_cx) ** 2 + (gt_cy - pred_cy) ** 2))
        position_score = float(max(0.0, 1.0 - position_dist / np.sqrt(2.0)))

        if outline_task:
            total_score = shape_gate * (
                weights["shape"] * shape_score
                + weights["outline_style"] * outline_style_score
                + weights["size"] * size_score
                + weights["color"] * color_score
                + weights["position"] * position_score
            )
        else:
            total_score = shape_gate * (
                weights["shape"] * shape_score
                + weights["size"] * size_score
                + weights["color"] * color_score
                + weights["position"] * position_score
            )

        # for scale task, accuracy penalty is applied to size ratio.
        if scale_task:
            total_score = total_score * size_ratio

        return float(max(0.0, min(1.0, total_score))), {
            "shape": shape_score,
            "outline_style": outline_style_score,
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
        """Evaluate O-14 with final-frame completion and preservation metrics."""
        scores: Dict[str, float] = {}

        if len(video_frames) < 2 or gt_first_frame is None or gt_final_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        if last_frame.shape[:2] != gt_final_frame.shape[:2]:
            first_frame = normalize_frame_size(first_frame, gt_final_frame)
            last_frame = normalize_frame_size(last_frame, gt_final_frame)
        gt_first, gt_last = gt_first_frame, gt_final_frame

        # 1) completion (60%): split final frame into 3x3 grid.
        gt_first_shape_features = self._extract_cell_shape_features(gt_last, row_idx=1, col_idx=1)
        gt_second_shape_features = self._extract_cell_shape_features(gt_last, row_idx=1, col_idx=2)
        
        pred_first_shape_features = self._extract_cell_shape_features(last_frame, row_idx=1, col_idx=1)
        pred_second_shape_features = self._extract_cell_shape_features(last_frame, row_idx=1, col_idx=2)
        
        first_shape_score, first_shape_details = self._compute_shape_score(
            gt_first_shape_features, pred_first_shape_features, weights=self.FIRST_SHAPE_WEIGHTS, scale_task=True, outline_task=False
        )
        second_shape_score, second_shape_details = self._compute_shape_score(
            gt_second_shape_features, pred_second_shape_features, weights=self.SECOND_SHAPE_WEIGHTS, scale_task=False, outline_task=True
        )
        scores["completion"] = 0.5 * first_shape_score + 0.5 * second_shape_score

        # 2) foreground_preservation (25%): compare first vs final on non-background and non-changed region.
        change_mask = self._shape_change_mask(gt_first, gt_last)
        first_fg, first_bg = self._frame_masks(first_frame)
        fg_compare_mask = cv2.bitwise_and(first_fg, cv2.bitwise_not(change_mask))
        scores["foreground_preservation"] = self._pixel_similarity(first_frame, last_frame, fg_compare_mask, min_cutoff=0.6)

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
        return _shape_transform_score(scores, self.TASK_WEIGHTS)


class BallBounceEvaluator(BaseEvaluator):
    """
    O-15: Ball Bounces Given Time

    Scoring (video mode):
      physics_score  (40%): straight-line segments between bounces + reflection
                            law (angle_in ≈ angle_out) at boundary contacts
      trajectory_score (60%): time-cropped gen vs GT, both resampled uniformly in arc
                            length; mean chord distance mapped to [0,1] via a GT-scale
                            power law (calibrated on O-15 ``00000`` ref: GT≈1, Wan≈0.6,
                            bad models≈0), × path-length penalty (extra arc vs GT), then ×
                            ball-count penalty (2→×0.7, 3→×0.5, 4→×0.3, >4→×0).

    Post-scoring deduction:
      foreground_similarity: sample 1 frame per 10, erase ball regions from
      both gen and GT, measure pixel SSIM.  If < 0.7 → subtract 0.10 from
      final score (clamped to 0).
    """

    # ------------------------------------------------------------------ helpers

    def _detect_circles(self, gray: np.ndarray):
        """Return all circles found by HoughCircles, or []."""
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
            param1=50, param2=25, minRadius=4, maxRadius=60,
        )
        if circles is None:
            return []
        return np.round(circles[0]).astype(int).tolist()

    def _detect_ball_in_frame(self, frame: np.ndarray):
        """Return (x, y, r) of the most prominent circle, or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        circles = self._detect_circles(gray)
        if circles:
            return tuple(circles[0])          # (x, y, r)
        # fallback: largest dark contour
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M['m00'] > 0:
                cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
                r = int(np.sqrt(cv2.contourArea(c) / np.pi)) + 1
                return (cx, cy, r)
        return None

    def _count_balls_in_frame(self, frame: np.ndarray) -> int:
        """Count distinct ball-like circles in one frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        circles = self._detect_circles(gray)
        return max(1, len(circles)) if circles else 1

    def _track_positions(self, frames: List[np.ndarray]) -> List[Tuple[float, float]]:
        """Track the primary ball centre across frames."""
        pos = []
        for f in frames:
            det = self._detect_ball_in_frame(f)
            if det is not None:
                pos.append((float(det[0]), float(det[1])))
        return pos

    def _erase_ball(self, frame: np.ndarray, det) -> np.ndarray:
        """Return frame with ball region filled with local median colour."""
        out = frame.copy()
        if det is None:
            return out
        x, y, r = int(det[0]), int(det[1]), int(det[2]) + 4
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        out[mask > 0] = np.median(frame, axis=(0, 1)).astype(np.uint8)
        return out

    # ------------------------------------------------------------------ physics

    def _segment_linearity(self, pts: List[Tuple[float, float]]) -> float:
        """
        Score how well a list of consecutive positions forms a straight line.
        Uses the ratio of chord length to total arc length (1 = perfect line).
        """
        if len(pts) < 2:
            return 0.0
        chord = np.sqrt((pts[-1][0] - pts[0][0])**2 + (pts[-1][1] - pts[0][1])**2)
        arc = sum(np.sqrt((pts[i][0]-pts[i-1][0])**2 + (pts[i][1]-pts[i-1][1])**2)
                  for i in range(1, len(pts)))
        return chord / arc if arc > 1 else 0.0

    def _pixel_similarity(self, frame_a: np.ndarray, frame_b: np.ndarray,
                          strictness: float = 2.0, min_cutoff: float = 0.3) -> float:
        """Strict pixel-level similarity in [0, 1] using L2 distance with power scaling."""
        a = frame_a.astype(np.float32).reshape(-1, 3 if frame_a.ndim == 3 else 1)
        b = frame_b.astype(np.float32).reshape(-1, 3 if frame_b.ndim == 3 else 1)
        if len(a) == 0:
            return 0.0
        mean_dist = float(np.mean(np.linalg.norm(a - b, axis=1)))
        max_dist = float(np.sqrt(3.0 * 255.0 ** 2)) if frame_a.ndim == 3 else 255.0
        base_sim = max(0.0, 1.0 - mean_dist / max_dist)
        final_sim = float(max(0.0, min(1.0, base_sim ** strictness)))
        return final_sim if final_sim >= min_cutoff else 0.0

    def _detect_walls(self, first_frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect the 4 wall positions from the first frame using row/column
        edge-projection.  The playing area is typically enclosed by thick
        rectangular borders that produce strong edge peaks near each side.

        Algorithm:
          1. Compute Canny edges.
          2. Project (sum) edges along rows → horizontal wall candidates.
             Project along cols  → vertical wall candidates.
          3. For each axis, take the innermost strong peaks that are at
             least 5% from the frame edge (so we skip the image border itself).
          4. Fall back to frame boundary if detection fails.

        Returns (left, top, right, bottom) in pixel coordinates.
        """
        h, w = first_frame.shape[:2]
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) if first_frame.ndim == 3 else first_frame

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 20, 80)

        # ---- column projection → left/right walls ----
        col_proj = edges.sum(axis=0).astype(float)   # shape (w,)
        row_proj = edges.sum(axis=1).astype(float)   # shape (h,)

        def _wall_pair(proj, size, min_frac=0.05, peak_thresh_frac=0.25):
            """Find (near_start, near_end) peaks in a 1-D projection."""
            threshold = max(proj) * peak_thresh_frac
            margin = int(size * min_frac)
            # near-start: first strong peak after margin
            near_start = margin
            for i in range(margin, size // 2):
                if proj[i] >= threshold:
                    near_start = i
                    break
            # near-end: last strong peak before (size - margin)
            near_end = size - margin - 1
            for i in range(size - margin - 1, size // 2, -1):
                if proj[i] >= threshold:
                    near_end = i
                    break
            return near_start, near_end

        left,  right  = _wall_pair(col_proj, w)
        top,   bottom = _wall_pair(row_proj, h)

        # Sanity: walls must enclose at least 30% of the frame
        if (right - left) < 0.30 * w or (bottom - top) < 0.30 * h:
            return (0, 0, w, h)
        
        return (int(left), int(top), int(right), int(bottom))

    def _wall_contact_margin(self, walls: Tuple[int, int, int, int]) -> float:
        wall_l, wall_t, wall_r, wall_b = walls
        pw = max(int(wall_r) - int(wall_l), 1)
        ph = max(int(wall_b) - int(wall_t), 1)
        return max(pw, ph) * 0.15 + 8.0

    @staticmethod
    def _dedupe_consecutive_positions(
        positions: List[Tuple[float, float]], eps: float = 0.25,
    ) -> List[Tuple[float, float]]:
        """Drop consecutive duplicates / micro-stutter points from tracking."""
        if not positions:
            return positions
        out: List[Tuple[float, float]] = [positions[0]]
        for p in positions[1:]:
            if abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
                out.append(p)
        return out

    def _velocity_segment_reversal(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        walls: Tuple[int, int, int, int],
    ) -> bool:
        """True if chord p0→p1 and p1→p2 show a clear axis flip (wall bounce / corner)."""
        dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
        dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]
        n1 = float(np.hypot(dx1, dy1))
        n2 = float(np.hypot(dx2, dy2))
        wall_l, wall_t, wall_r, wall_b = walls
        play_diag = float(np.hypot(max(wall_r - wall_l, 1), max(wall_b - wall_t, 1)))
        min_leg = max(4.5, 0.009 * play_diag)
        if n1 < min_leg or n2 < min_leg:
            return False
        prod_th = (min_leg * 0.55) ** 2
        return bool(dx1 * dx2 < -prod_th or dy1 * dy2 < -prod_th)

    def _axis_direction_reversal_at(
        self,
        positions: List[Tuple[float, float]],
        i: int,
        walls: Tuple[int, int, int, int],
    ) -> bool:
        """Single-frame reversal, else 2-frame (i±2) when micro-steps hide the bounce."""
        if i < 1 or i + 1 >= len(positions):
            return False
        p0, p1, p2 = positions[i - 1], positions[i], positions[i + 1]
        dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
        dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]
        n1 = float(np.hypot(dx1, dy1))
        n2 = float(np.hypot(dx2, dy2))
        wall_l, wall_t, wall_r, wall_b = walls
        play_diag = float(np.hypot(max(wall_r - wall_l, 1), max(wall_b - wall_t, 1)))
        min_leg = max(4.5, 0.009 * play_diag)
        prod_th = (min_leg * 0.55) ** 2
        if n1 >= min_leg and n2 >= min_leg:
            if dx1 * dx2 < -prod_th or dy1 * dy2 < -prod_th:
                return True
        if i >= 2 and i + 2 < len(positions):
            return self._velocity_segment_reversal(
                positions[i - 2], positions[i], positions[i + 2], walls,
            )
        return False

    def _wall_for_point(self, bp: Tuple[float, float],
                        walls: Tuple[int, int, int, int]) -> Optional[str]:
        """Return the nearest wall the point is close to, or None.
        Picks the closest wall so corner points (near two walls) are classified correctly."""
        wall_l, wall_t, wall_r, wall_b = walls
        MARGIN = self._wall_contact_margin(walls)
        dists = {
            'left':  abs(bp[0] - wall_l),
            'right': abs(bp[0] - wall_r),
            'top':   abs(bp[1] - wall_t),
            'bot':   abs(bp[1] - wall_b),
        }
        nearest, dist = min(dists.items(), key=lambda kv: kv[1])
        return nearest if dist < MARGIN else None

    def _split_at_wall_bounces(self, positions: List[Tuple[float, float]],
                               walls: Tuple[int, int, int, int],
                               min_seg_frames: int = 4) -> List[List[Tuple[float, float]]]:
        """
        Split trajectory only at direction-reversals that occur near a wall.
        Segments shorter than min_seg_frames are merged into the longer neighbour —
        this handles corner bounces where two reversals fire within 1-2 frames.
        """
        if len(positions) < 3:
            return [positions]

        # Collect split indices at wall-proximate reversals
        split_at = []
        for i in range(1, len(positions) - 1):
            if not self._axis_direction_reversal_at(positions, i, walls):
                continue
            if self._wall_for_point(positions[i], walls) is not None:
                split_at.append(i)

        # Merge split points that are closer than min_seg_frames apart (corner bounces)
        merged = []
        for idx in split_at:
            if merged and idx - merged[-1] < min_seg_frames:
                # Replace previous with the midpoint index so the chord spans both walls
                merged[-1] = idx
            else:
                merged.append(idx)

        # Build segments
        segments = []
        seg_start = 0
        for idx in merged:
            segments.append(positions[seg_start:idx + 1])
            seg_start = idx
        segments.append(positions[seg_start:])
        return [s for s in segments if len(s) >= 2]

    def _reflection_score_at_bounces(self, positions: List[Tuple[float, float]],
                                     walls: Tuple[int, int, int, int]) -> float:
        """
        Score reflection law compliance at each wall bounce.
        Segments are already split only at wall-proximate reversals, so every
        inter-segment boundary is guaranteed to be near a wall — no need to
        re-check proximity here.
        """
        segs = self._split_at_wall_bounces(positions, walls)
        if len(segs) < 2:
            return 0

        scores = []
        eps = 1e-6
        for i in range(len(segs) - 1):
            s1, s2 = segs[i], segs[i+1]
            bp = s1[-1]
            wall = self._wall_for_point(bp, walls)
            if wall is None or len(s1) < 2 or len(s2) < 2:
                continue

            d_in  = (s1[-1][0] - s1[0][0], s1[-1][1] - s1[0][1])
            d_out = (s2[-1][0] - s2[0][0], s2[-1][1] - s2[0][1])

            if wall in ('left', 'right'):
                angle_in  = np.arctan2(abs(d_in[1]),  abs(d_in[0])  + eps)
                angle_out = np.arctan2(abs(d_out[1]), abs(d_out[0]) + eps)
            else:
                angle_in  = np.arctan2(abs(d_in[0]),  abs(d_in[1])  + eps)
                angle_out = np.arctan2(abs(d_out[0]), abs(d_out[1]) + eps)

            angle_diff = abs(angle_in - angle_out)
            scores.append(max(0.0, 1.0 - angle_diff / (np.pi / 4)))  # 45° → 0

        return float(np.mean(scores)) if scores else 1.0

    def _smoothness_penalty(self, positions: List[Tuple[float, float]]) -> float:
        """Multiplier in (0, 1]: 1 = perfectly smooth velocity profile."""
        if len(positions) < 3:
            return 0.0
        speeds = [np.hypot(positions[i][0] - positions[i-1][0],
                           positions[i][1] - positions[i-1][1])
                  for i in range(1, len(positions))]
        mean_s = np.mean(speeds)
        if mean_s < 1:
            return 0.8
        cv = np.std(speeds) / mean_s
        return max(0.5, 1.0 - cv / 2.0)

    def _count_bad_bounces(self, positions: List[Tuple[float, float]],
                           walls: Tuple[int, int, int, int]) -> int:
        """Count direction reversals that are NOT near any wall (non-physical bounces)."""
        bad = 0
        for i in range(1, len(positions) - 1):
            if not self._axis_direction_reversal_at(positions, i, walls):
                continue
            if self._wall_for_point(positions[i], walls) is None:
                bad += 1
        return bad

    def _physics_score(self, positions: List[Tuple[float, float]],
                       walls: Tuple[int, int, int, int]) -> Tuple[float, float, float, int]:
        """40% physics: linearity (50%) + reflection*bad_bounce_penalty (50%).
        Returns (score, linearity, reflection, bad_bounce_count)."""
        positions = self._dedupe_consecutive_positions(positions)
        if len(positions) < 3:
            return 0.0, 0.0, 0.0, 0

        segs = self._split_at_wall_bounces(positions, walls)
        lin_scores = [self._segment_linearity(s) for s in segs if len(s) >= 2]
        linearity = float(np.mean(lin_scores)) if lin_scores else 0.5

        reflection = self._reflection_score_at_bounces(positions, walls)

        bad_bounces = self._count_bad_bounces(positions, walls)
        bad_penalty = max(0.0, 1.0 - bad_bounces * 0.1)

        score = 0.5 * linearity + 0.5 * reflection * bad_penalty
        return score, linearity, reflection, bad_bounces

    # ------------------------------------------------------------------ trajectory similarity

    def _dtw_distance(self, seq_a: List[Tuple[float, float]],
                      seq_b: List[Tuple[float, float]]) -> float:
        n, m = len(seq_a), len(seq_b)
        if n == 0 or m == 0:
            return float('inf')
        dtw = np.full((n, m), np.inf)
        dtw[0, 0] = np.hypot(seq_a[0][0]-seq_b[0][0], seq_a[0][1]-seq_b[0][1])
        for i in range(1, n):
            dtw[i, 0] = dtw[i-1, 0] + np.hypot(seq_a[i][0]-seq_b[0][0], seq_a[i][1]-seq_b[0][1])
        for j in range(1, m):
            dtw[0, j] = dtw[0, j-1] + np.hypot(seq_a[0][0]-seq_b[j][0], seq_a[0][1]-seq_b[j][1])
        for i in range(1, n):
            for j in range(1, m):
                cost = np.hypot(seq_a[i][0]-seq_b[j][0], seq_a[i][1]-seq_b[j][1])
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
        return float(dtw[n-1, m-1])

    # ------------------------------------------------------------------ shape-aware DTW

    @staticmethod
    def _arc_resample(pts: np.ndarray, target_n: int) -> np.ndarray:
        """Resample a (N,2) trajectory to target_n points at uniform arc-length spacing."""
        if len(pts) < 2 or target_n < 2:
            return pts
        diffs = np.diff(pts, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cumlen[-1]
        if total < 1e-8:
            return np.tile(pts[0], (target_n, 1))
        targets = np.linspace(0.0, total, target_n)
        out = np.empty((target_n, 2), dtype=float)
        for k, t in enumerate(targets):
            idx = np.searchsorted(cumlen, t, side='right') - 1
            idx = min(idx, len(pts) - 2)
            seg = cumlen[idx + 1] - cumlen[idx]
            frac = (t - cumlen[idx]) / seg if seg > 1e-10 else 0.0
            out[k] = pts[idx] * (1 - frac) + pts[idx + 1] * frac
        return out

    @staticmethod
    def _unit_tangents(pts: np.ndarray) -> np.ndarray:
        """Return unit tangent vectors at each of the N points.
        Uses forward difference for all points, central for interior (smoother)."""
        n = len(pts)
        dirs = np.diff(pts, axis=0)  # (N-1, 2)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        unit = dirs / norms           # (N-1, 2)
        # Pad last tangent by repeating the previous one → (N, 2)
        return np.vstack([unit, unit[[-1]]])

    @staticmethod
    def _normalize_traj(pts: np.ndarray) -> Tuple[np.ndarray, float]:
        """Translate to centroid-zero, scale to unit bounding-box diagonal.
        Returns (normalized_pts, scale_factor) where scale_factor is the original
        bounding-box diagonal (used to restore physical distance units for scoring)."""
        centroid = pts.mean(axis=0)
        pts = pts - centroid
        bb = np.max(pts, axis=0) - np.min(pts, axis=0)
        scale = float(np.hypot(bb[0], bb[1]))
        if scale < 1e-6:
            scale = 1.0
        return pts / scale, scale

    def _shape_dtw(self, traj_a: np.ndarray, traj_b: np.ndarray,
                   alpha: float = 0.5, beta: float = 0.5) -> float:
        """Shape-aware DTW on normalised, arc-resampled trajectories.

        Per-cell distance = alpha * ||p_i - q_j|| + beta * (1 - cos(theta_ij))
        Final cost divided by (N+M) so longer sequences don't inflate the distance.
        """
        n, m = len(traj_a), len(traj_b)
        dirs_a = self._unit_tangents(traj_a)   # (n, 2)
        dirs_b = self._unit_tangents(traj_b)   # (m, 2)

        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                pos_d = float(np.linalg.norm(traj_a[i-1] - traj_b[j-1]))
                cos_sim = float(np.dot(dirs_a[i-1], dirs_b[j-1]))
                # Clamp numerical noise before using as distance
                dir_d = 1.0 - max(-1.0, min(1.0, cos_sim))
                d = alpha * pos_d + beta * dir_d
                dp[i, j] = d + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

        return float(dp[n, m]) / (n + m)

    @staticmethod
    def _resample_positions(pos: List[Tuple[float, float]], target_n: int) -> List[Tuple[float, float]]:
        """Thin wrapper used by _physics_score callers; kept for compatibility."""
        arr = np.array(pos, dtype=float)
        if len(arr) == target_n:
            return pos
        resampled = BallBounceEvaluator._arc_resample(arr, target_n)
        return [(float(p[0]), float(p[1])) for p in resampled]

    @staticmethod
    def _polyline_arc_length(pts: np.ndarray) -> float:
        """Total Euclidean length of a piecewise-linear path (pixels)."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape[0] < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    @staticmethod
    def _point_to_polyline_min_dist(p: np.ndarray, poly: np.ndarray) -> float:
        """Min distance from 2-vector p to polyline vertices/segments (N,2)."""
        if poly.shape[0] == 0:
            return float('inf')
        if poly.shape[0] == 1:
            return float(np.linalg.norm(p - poly[0]))
        a = poly[:-1]
        b = poly[1:]
        ab = b - a
        denom = np.sum(ab * ab, axis=1) + 1e-12
        ap = p - a
        t = np.clip(np.sum(ap * ab, axis=1) / denom, 0.0, 1.0)
        proj = a + (t[:, np.newaxis] * ab)
        return float(np.min(np.linalg.norm(p - proj, axis=1)))

    _TRAJ_ARC_SAMPLES = 256
    _TRAJ_CHORD_D0_FLOOR_PX = 240.0
    _TRAJ_CHORD_D0_REL_SPAN = 0.55
    _TRAJ_CHORD_POWER = 1.25

    def _mean_chord_distance_arc_resampled(
        self,
        gt_arr: np.ndarray,
        gen_arr: np.ndarray,
        n_samples: int,
    ) -> float:
        """Uniform arc-length resampling of both polylines, then mean Euclidean chord length."""
        gt_arr = np.asarray(gt_arr, dtype=float)
        gen_arr = np.asarray(gen_arr, dtype=float)
        if gt_arr.shape[0] < 2:
            return 0.0
        gt_s = self._arc_resample(gt_arr, n_samples)
        if gen_arr.shape[0] < 2:
            gen_s = np.tile(gen_arr[:1], (n_samples, 1))
        else:
            gen_s = self._arc_resample(gen_arr, n_samples)
        return float(np.mean(np.linalg.norm(gt_s - gen_s, axis=1)))

    def _trajectory_similarity(
        self,
        gen_pos: List[Tuple[float, float]],
        gt_pos: List[Tuple[float, float]],
        frame_diag: float,
    ) -> Tuple[float, float, float]:
        """Arc-aligned mean chord distance → coverage, × path-length penalty.

        **Time crop:** gen is truncated to the first ``len(gt_pos)`` samples so both paths
        describe the same nominal time span.

        **Coverage:** Resample GT and cropped gen uniformly by **arc length** to
        ``_TRAJ_ARC_SAMPLES`` points (straight GT segments become evenly spaced samples along
        the piecewise-linear path). Let ``d_mean`` be the mean Euclidean distance between paired
        samples. Let ``S`` be the GT bounding-box diagonal. Map
        ``coverage = clip(1 - (d_mean / D0)^p, 0, 1)`` with
        ``D0 = max(_TRAJ_CHORD_D0_FLOOR_PX, _TRAJ_CHORD_D0_REL_SPAN * S)`` and
        ``p = _TRAJ_CHORD_POWER``.  Identical paths give ``d_mean = 0`` → 1; distances well above
        ``D0`` → ~0.

        **Length penalty:** full-gen polyline length vs ``L_gt``; no penalty when gen
        travelled distance is at or below GT.

        Returns ``(score, coverage_fraction, length_penalty)`` with
        ``score = clip(coverage × penalty, 0, 1)``.
        """
        _ = frame_diag  # kept for API compatibility with callers
        self._traj_similarity_details = {}

        if not gen_pos or not gt_pos or len(gt_pos) < 2:
            self._traj_similarity_details = {
                'traj_note': 'missing gen/gt or GT path too short',
            }
            return 0.0, 0.0, 1.0

        n_gt = len(gt_pos)
        n_gen = len(gen_pos)
        gen_crop = gen_pos[:n_gt] if n_gen > n_gt else gen_pos
        gt_arr = np.asarray(gt_pos, dtype=float)
        gen_arr = np.asarray(gen_crop, dtype=float)
        gen_full_arr = np.asarray(gen_pos, dtype=float)

        n_s = int(self._TRAJ_ARC_SAMPLES)
        mean_chord = self._mean_chord_distance_arc_resampled(gt_arr, gen_arr, n_s)

        span_gt = float(np.linalg.norm(np.max(gt_arr, axis=0) - np.min(gt_arr, axis=0)))
        d0 = max(self._TRAJ_CHORD_D0_FLOOR_PX, self._TRAJ_CHORD_D0_REL_SPAN * span_gt)
        p = float(self._TRAJ_CHORD_POWER)
        if mean_chord <= 1e-9:
            coverage = 1.0
        else:
            coverage = float(np.clip(1.0 - (mean_chord / max(d0, 1e-6)) ** p, 0.0, 1.0))

        L_gt = self._polyline_arc_length(gt_arr)
        L_gen_full = self._polyline_arc_length(gen_full_arr)
        L_gen_crop = self._polyline_arc_length(gen_arr)

        if L_gen_full <= L_gt + 1e-6:
            length_penalty = 1.0
        else:
            length_penalty = max(0.0, 1.0 - (L_gen_full - L_gt) / max(L_gt, 1e-6))

        score = float(np.clip(coverage * length_penalty, 0.0, 1.0))

        self._traj_similarity_details = {
            'traj_coverage_def': (
                f'mean Euclidean distance between GT and cropped gen after uniform '
                f'arc-length resampling to {n_s} points; score = max(0, 1 - (mean / D0)^p)'
            ),
            'traj_mean_chord_px':    round(mean_chord, 4),
            'traj_D0_scale_px':      round(d0, 4),
            'traj_chord_power':      p,
            'traj_gt_bbox_diag_px':  round(span_gt, 4),
            'traj_L_gt_poly_px':     round(L_gt, 2),
            'traj_L_gen_full_poly_px': round(L_gen_full, 2),
            'traj_L_gen_cropped_poly_px': round(L_gen_crop, 2),
            'traj_len_penalty_def': (
                '1.0 if L_gen_full <= L_gt else max(0, 1 - (L_gen_full - L_gt) / L_gt)'
            ),
            'traj_n_gt_points':      n_gt,
            'traj_n_gen_points':     n_gen,
            'traj_extra_poly_px':    round(max(0.0, L_gen_full - L_gt), 2),
        }
        return score, coverage, length_penalty

    def _ball_count_penalty(self, frames: List[np.ndarray]) -> Tuple[float, int]:
        """
        Sample up to 10 evenly-spaced frames, take the median ball count.
        Returns (penalty_multiplier, median_count).
        """
        step = max(1, len(frames) // 10)
        counts = [self._count_balls_in_frame(frames[i])
                  for i in range(0, len(frames), step)]
        med = int(np.median(counts))
        penalty = {1: 1.0, 2: 0.7, 3: 0.5, 4: 0.3}.get(med, 0.0 if med > 4 else 1.0)
        return penalty, med


    # ------------------------------------------------------------------ foreground consistency

    def _foreground_similarity(self, gen_frames: List[np.ndarray],
                               gt_frames: List[np.ndarray]) -> float:
        """
        Sample 1 frame per 10, erase the detected ball from both gen and GT,
        then measure background consistency via two signals:
          1. pixel_similarity (L2, strictness=2): catches spatial changes
          2. mean-colour distance in HSV: catches global colour-shift (e.g. white→blue)

        The per-frame score is min(pixel_sim, colour_sim) so that either
        a spatial change OR a colour shift is enough to lower the score.
        The overall score is the *median* over sampled frames (not mean), so
        early frames with identical white backgrounds cannot mask large changes
        in later frames.
        """
        indices = list(range(1, min(len(gen_frames), len(gt_frames))))
        if not indices:
            return 0.0
        per_frame = []
        for i in indices:
            gf = gen_frames[i]
            gtf = gt_frames[i]
            # Align the prediction to GT, not GT to the prediction.
            if gf.shape != gtf.shape:
                gf = cv2.resize(gf, (gtf.shape[1], gtf.shape[0]))
            det_gen = self._detect_ball_in_frame(gf)
            det_gt  = self._detect_ball_in_frame(gtf)
            gf_c  = self._erase_ball(gf,  det_gen)
            gtf_c = self._erase_ball(gtf, det_gt)

            # Signal 1: pixel-level similarity
            pix_sim = self._pixel_similarity(gf_c, gtf_c, strictness=2.0, min_cutoff=0.0)

            # Signal 2: mean-colour distance in HSV (catches global colour drift)
            gf_hsv  = cv2.cvtColor(gf_c,  cv2.COLOR_BGR2HSV).astype(float)
            gtf_hsv = cv2.cvtColor(gtf_c, cv2.COLOR_BGR2HSV).astype(float)
            mean_gen = gf_hsv.mean(axis=(0, 1))    # [H, S, V]
            mean_gt  = gtf_hsv.mean(axis=(0, 1))
            # Hue is circular (0-180 in OpenCV); compute circular diff
            dh = abs(mean_gen[0] - mean_gt[0])
            dh = min(dh, 180 - dh)
            ds = abs(mean_gen[1] - mean_gt[1])   # Saturation 0-255
            dv = abs(mean_gen[2] - mean_gt[2])   # Value 0-255
            # Normalise each channel and combine
            colour_dist = (dh / 90.0 + ds / 255.0 + dv / 255.0) / 3.0
            colour_sim = max(0.0, 1.0 - colour_dist * 3.0)  # aggressive scale

            per_frame.append(min(pix_sim, colour_sim))

        # Median so that a minority of unchanged frames can't mask later drift
        return float(np.median(per_frame))

    # ------------------------------------------------------------------ main evaluate

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        if not video_frames or not gt_frames:
            return 0.0

        h, w = video_frames[0].shape[:2]
        frame_diag = np.hypot(w, h)

        # --- detect walls from the GT first frame (most reliable source) ---
        ref_frame = gt_first_frame if gt_first_frame is not None else gt_frames[0]
        walls = self._detect_walls(ref_frame)   # (left, top, right, bottom)
        wall_l, wall_t, wall_r, wall_b = walls

        # --- track positions ---
        gen_pos = self._track_positions(video_frames)
        gt_pos  = self._track_positions(gt_frames)

        # --- 40% physics ---
        if len(gen_pos) >= 3:
            physics, linearity, reflection, bad_bounces = self._physics_score(gen_pos, walls)
        else:
            physics, linearity, reflection, bad_bounces = 0.0, 0.0, 0.0, 0
        bad_bounce_penalty = max(0.0, 1.0 - bad_bounces * 0.1)

        # --- 60% trajectory (GT arc coverage + extra-length penalty) ---
        traj_raw, traj_coverage, traj_len_penalty = self._trajectory_similarity(
            gen_pos, gt_pos, frame_diag,
        )
        ball_mult, ball_count = self._ball_count_penalty(video_frames)
        smooth_mult = self._smoothness_penalty(gen_pos)
        trajectory  = traj_raw * ball_mult

        total = 0.40 * max(physics, traj_coverage) + 0.60 * trajectory

        # --- foreground consistency deduction ---
        fg_sim = self._foreground_similarity(video_frames, gt_frames)
        if fg_sim < 0.7:
            total = max(0.0, total - 0.10)

        traj_details = getattr(self, '_traj_similarity_details', None) or {}

        scores = {
            'physics':            round(physics, 4),
            'linearity':          round(linearity, 4),
            'reflection':         round(reflection, 4),
            'bad_bounces':        bad_bounces,
            'bad_bounce_penalty': round(bad_bounce_penalty, 2),
            'trajectory':         round(trajectory, 4),
            'traj_raw':           round(traj_raw, 4),
            'traj_coverage':      round(traj_coverage, 4),
            'traj_len_penalty':   round(traj_len_penalty, 4),
            'ball_count':         ball_count,
            'ball_penalty':       round(ball_mult, 2),
            'smooth_penalty':     round(smooth_mult, 4),
            'fg_similarity':      round(fg_sim, 4),
            'walls_ltrb':         f"{wall_l},{wall_t},{wall_r},{wall_b}",
            **traj_details,
        }
        self._last_task_details = scores
        return max(0.0, min(1.0, total))

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Interleave: a single frame with the bounce trajectory drawn as a line.

        Mirrors the VIDEO logic exactly — same _physics_score (40%) +
        _trajectory_similarity (60%) x ball-count penalty, same 40/60 weights.
        The ONLY difference is how the ball positions are obtained: instead of
        tracking the ball across frames, gen_pos is the centreline of the drawn
        trajectory line (colour-masked, then ordered/averaged along the GT
        polyline's arc length). gt_pos is the exact GT trajectory_polyline from
        metadata; walls come from the metadata bounce bounds.
        """
        import json as _json, os as _os
        if not pred_images or input_frame is None:
            self._last_task_details = {"error": "missing_frames"}
            return 0.0

        # --- GT trajectory + line colour + walls from metadata ---
        meta = None
        mp = eval_info.get("metafile_path")
        if isinstance(mp, (list, tuple)):
            mp = next((p for p in mp if p and _os.path.exists(p)), None)
        if not (mp and _os.path.exists(mp)):
            mp = _os.path.join(eval_info.get("gt_path", ""), "metadata.json")
        if _os.path.exists(mp):
            with open(mp) as f:
                meta = _json.load(f)
        if meta is None:
            self._last_task_details = {"error": "metadata not found"}
            return 0.0
        sgt = meta.get("semantic_ground_truth") or {}
        params = meta.get("parameters") or {}
        poly = sgt.get("trajectory_polyline") or params.get("trajectory") or []
        gt_pos = [(float(pt[0]), float(pt[1])) for pt in poly]
        if len(gt_pos) < 2:
            self._last_task_details = {"error": "no gt trajectory"}
            return 0.0
        traj_rgb = params.get("trajectory_rgb") or [255, 140, 0]
        bounds = params.get("bounds")
        if bounds and len(bounds) == 4:  # [xmin, xmax, ymin, ymax]
            walls = (int(bounds[0]), int(bounds[2]), int(bounds[1]), int(bounds[3]))  # l,t,r,b
        else:
            walls = self._detect_walls(input_frame)

        drawn = pred_images[-1]
        if drawn.shape[:2] != input_frame.shape[:2]:
            drawn = normalize_frame_size(drawn, input_frame)
        h, w = drawn.shape[:2]
        frame_diag = float(np.hypot(w, h))

        # gen_pos: centreline of the drawn trajectory line (replaces frame tracking)
        gen_pos = self._trajectory_centerline(drawn, traj_rgb, np.asarray(gt_pos, dtype=float))

        # --- 40% physics (same method as video) ---
        if len(gen_pos) >= 3:
            physics, linearity, reflection, bad_bounces = self._physics_score(gen_pos, walls)
        else:
            physics, linearity, reflection, bad_bounces = 0.0, 0.0, 0.0, 0

        # --- 60% trajectory (same method as video) ---
        traj_raw, traj_coverage, traj_len_penalty = self._trajectory_similarity(
            gen_pos, gt_pos, frame_diag,
        )
        ball_mult, ball_count = self._ball_count_penalty(pred_images)
        trajectory = traj_raw * ball_mult

        total = 0.40 * max(physics, traj_coverage) + 0.60 * trajectory

        # Mirror the video branch's foreground-consistency deduction.  The base
        # interleave path prepends input_frame to gt_images, so prepend it to the
        # prediction sequence as well before using the shared comparison helper.
        gen_frames = [input_frame] + list(pred_images)
        fg_sim = self._foreground_similarity(gen_frames, gt_images)
        if fg_sim < 0.7:
            total = max(0.0, total - 0.10)

        self._last_task_details = {
            'mode': 'interleave',
            'physics': round(physics, 4),
            'linearity': round(linearity, 4),
            'reflection': round(reflection, 4),
            'bad_bounces': bad_bounces,
            'trajectory': round(trajectory, 4),
            'traj_raw': round(traj_raw, 4),
            'traj_coverage': round(traj_coverage, 4),
            'traj_len_penalty': round(traj_len_penalty, 4),
            'ball_count': ball_count,
            'ball_penalty': round(ball_mult, 2),
            'fg_similarity': round(fg_sim, 4),
            'n_gen_pos': len(gen_pos),
            'n_gt_pos': len(gt_pos),
        }
        return max(0.0, min(1.0, total))

    def _trajectory_centerline(
        self, drawn: np.ndarray, traj_rgb, gt_arr: np.ndarray,
        nbins: int = 256, tol: float = 70.0,
    ) -> List[Tuple[float, float]]:
        """Extract the drawn trajectory line as an ordered centreline polyline.

        Colour-mask the line pixels, then bin them by their projection onto the
        GT polyline's arc length and average each bin. A static drawn line has no
        temporal direction, so the GT parameterisation is the only sensible
        ordering; a wrong prediction's pixels still project onto GT but land far
        from it, so _trajectory_similarity returns low coverage as intended.
        """
        target = np.array([traj_rgb[2], traj_rgb[1], traj_rgb[0]], dtype=np.float32)  # BGR
        dist = np.linalg.norm(drawn.astype(np.float32) - target, axis=2)
        ys, xs = np.where(dist < tol)
        if len(xs) < 2:
            return []
        cloud = np.stack([xs, ys], axis=1).astype(float)
        if len(cloud) > 20000:
            cloud = cloud[np.linspace(0, len(cloud) - 1, 20000).astype(int)]

        g = self._arc_resample(gt_arr, nbins)
        a = g[:-1]
        b = g[1:]
        ab = b - a
        denom = np.sum(ab * ab, axis=1) + 1e-9
        seglen = np.hypot(ab[:, 0], ab[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        total_len = cum[-1] if cum[-1] > 1e-9 else 1.0

        params = np.empty(len(cloud))
        for i, pt in enumerate(cloud):
            t = np.clip(np.sum((pt - a) * ab, axis=1) / denom, 0.0, 1.0)
            proj = a + t[:, None] * ab
            dd = np.hypot(proj[:, 0] - pt[0], proj[:, 1] - pt[1])
            si = int(np.argmin(dd))
            params[i] = cum[si] + t[si] * seglen[si]

        bins = np.clip((params / total_len * nbins).astype(int), 0, nbins - 1)
        out: List[Tuple[float, float]] = []
        for k in range(nbins):
            sel = cloud[bins == k]
            if len(sel):
                out.append((float(sel[:, 0].mean()), float(sel[:, 1].mean())))
        return out

    def _count_bounces_from_lines(self, mask: np.ndarray) -> int:
        """Count direction changes from trajectory line using HoughLinesP."""
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180,
                                threshold=30, minLineLength=20, maxLineGap=10)
        if lines is None or len(lines) < 2:
            return 0
        line_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            line_list.append({'mid': ((x1+x2)/2, (y1+y2)/2), 'angle': angle})
        line_list.sort(key=lambda l: l['mid'][0])
        bounces = 0
        for i in range(1, len(line_list)):
            diff = abs(line_list[i]['angle'] - line_list[i-1]['angle'])
            diff = min(diff, np.pi - diff)
            if diff > np.pi / 6:
                bounces += 1
        return bounces


class ColorAdditionEvaluator(BaseEvaluator):
    """
    O-16: Color Addition (Additive Color Mixing)

    Task: Two colored balls move toward each other and merge into one ball,
    showing additive color mixing. The merged ball color should match GT.

    Evaluation:
    1. Mixed color correctness (60%): merged ball color matches GT, uniform, properly filled
    2. Object preservation (20%): gen_first vs gen_last in non-ball background area
    3. Background preservation (20%): background stays clean
    """

    TASK_WEIGHTS = {
        'mixing_color': 0.60,
        'circle_removal': 0.20,
        'background_clean': 0.20,
    }

    def _detect_fg_mask(self, frame: np.ndarray) -> np.ndarray:
        """Detect foreground (balls) by color distance from background."""
        h, w = frame.shape[:2]
        corners = [frame[2, 2], frame[2, w-3], frame[h-3, 2], frame[h-3, w-3]]
        bg_color = np.mean(corners, axis=0)
        diff = np.sqrt(np.sum((frame.astype(float) - bg_color.astype(float)) ** 2, axis=2))
        binary = (diff > 30).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
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

        mixing_mask = self._detect_fg_mask(gt_last)
        fg_first_mask = self._detect_fg_mask(gt_first)
        all_fg = cv2.bitwise_or(fg_first_mask, mixing_mask)

        kernel = np.ones((5, 5), np.uint8)
        mixing_mask_dilated = cv2.dilate(mixing_mask, kernel, iterations=1)
        all_fg_dilated = cv2.dilate(all_fg, kernel, iterations=1)

        # --- 1. Mixed color correctness (60%) ---
        gt_last_gray = cv2.cvtColor(gt_last, cv2.COLOR_BGR2GRAY)
        gt_interior = cv2.bitwise_and(mixing_mask, (gt_last_gray > 20).astype(np.uint8) * 255)
        gt_interior_px = int((gt_interior > 0).sum())
        h, w = gt_first.shape[:2]
        corners = [gt_first[2, 2], gt_first[2, w-3], gt_first[h-3, 2], gt_first[h-3, w-3]]
        bg_color = np.mean(corners, axis=0).astype(float)

        if gt_interior_px > 0:
            gt_target_color = np.mean(gt_last[gt_interior > 0].astype(float), axis=0)
        else:
            gt_target_color = None

        gen_last_fg = self._detect_fg_mask(gen_last)
        gen_contours, _ = cv2.findContours(gen_last_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gen_objects = [c for c in gen_contours if cv2.contourArea(c) > 200]
        n_gen_objects = len(gen_objects)

        # GT merged ball center
        gt_moments = cv2.moments(mixing_mask)
        if gt_moments['m00'] > 0:
            gt_cx = int(gt_moments['m10'] / gt_moments['m00'])
            gt_cy = int(gt_moments['m01'] / gt_moments['m00'])
        else:
            gt_cx, gt_cy = w // 2, h // 2

        if gt_target_color is not None and n_gen_objects > 0:
            if n_gen_objects == 1:
                # Path A: gen merged into one object → use gen's own object region
                chosen_idx = 0
            else:
                # Path B: gen didn't merge → find closest object to GT merged ball center
                chosen_idx = 0
                best_dist_to_gt = float('inf')
                for i, cnt in enumerate(gen_objects):
                    M = cv2.moments(cnt)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        d = np.sqrt((cx - gt_cx)**2 + (cy - gt_cy)**2)
                        if d < best_dist_to_gt:
                            best_dist_to_gt = d
                            chosen_idx = i

            # Extract chosen object interior (erode to remove edge pixels)
            obj_mask = np.zeros(gen_last.shape[:2], dtype=np.uint8)
            cv2.drawContours(obj_mask, gen_objects, chosen_idx, 255, -1)
            erode_kernel = np.ones((7, 7), np.uint8)
            obj_interior = cv2.erode(obj_mask, erode_kernel, iterations=1)
            obj_px = int((obj_interior > 0).sum())

            if obj_px > 0:
                gen_colors = gen_last[obj_interior > 0].astype(float)
                gen_mean = np.mean(gen_colors, axis=0)
                color_dist = np.sqrt(np.sum((gen_mean - gt_target_color) ** 2))
                gen_std = np.max(np.std(gen_colors, axis=0))

                # Size score: gen ball area vs GT ball area
                size_ratio = obj_px / max(gt_interior_px, 1)
                if 0.8 <= size_ratio <= 1.2:
                    size_score = 1.0
                elif 0.6 <= size_ratio <= 1.4:
                    size_score = 1.0 - (abs(size_ratio - 1.0) - 0.2) / 0.2 * 0.3
                elif 0.5 <= size_ratio <= 2.0:
                    size_score = 0.7 - (abs(size_ratio - 1.0) - 0.4) / 0.6 * 0.4
                else:
                    size_score = max(0.0, 0.3 - (abs(size_ratio - 1.0) - 1.0) / 1.0 * 0.3)

                # Position score: gen center to GT center (normalized by image size)
                gen_M = cv2.moments(gen_objects[chosen_idx])
                if gen_M['m00'] > 0:
                    gen_cx = int(gen_M['m10'] / gen_M['m00'])
                    gen_cy = int(gen_M['m01'] / gen_M['m00'])
                else:
                    gen_cx, gen_cy = w // 2, h // 2
                center_dist = np.sqrt((gen_cx - gt_cx)**2 + (gen_cy - gt_cy)**2)
                norm_dist = center_dist / max(h, w)
                if norm_dist < 0.05:
                    position_score = 1.0
                elif norm_dist < 0.15:
                    position_score = 1.0 - (norm_dist - 0.05) / 0.10 * 0.3
                elif norm_dist < 0.30:
                    position_score = 0.7 - (norm_dist - 0.15) / 0.15 * 0.4
                else:
                    position_score = max(0.0, 0.3 - (norm_dist - 0.30) / 0.30 * 0.3)
            else:
                color_dist = 999.0
                gen_std = 999.0
                size_score = 0.0
                position_score = 0.0
                center_dist = -1

            # Color score: distance between gen and GT mean color
            if color_dist < 20:
                color_score = 1.0
            elif color_dist < 30:
                color_score = 1.0 - (color_dist - 20) / 10 * 0.3   # 15→1.0, 25→0.7
            elif color_dist < 50:
                color_score = 0.7 - (color_dist - 30) / 20 * 0.4   # 25→0.7, 40→0.3
            elif color_dist < 100:
                color_score = 0.3 - (color_dist - 50) / 50 * 0.3   # 40→0.3, 80→0.0
            else:
                color_score = 0.0

            # Uniformity: max channel std (higher = less uniform)
            if gen_std < 5:
                uniformity_score = 1.0
            elif gen_std < 15:
                uniformity_score = 1.0 - (gen_std - 5) / 10 * 0.3
            elif gen_std < 30:
                uniformity_score = 0.7 - (gen_std - 15) / 15 * 0.4
            else:
                uniformity_score = max(0.0, 0.3 - (gen_std - 30) / 30 * 0.3)

            mixing_score = color_score * size_score * uniformity_score * position_score
        else:
            mixing_score = 0.0
            color_dist = -1
            color_score = 0.0
            size_score = 0.0
            position_score = 0.0
            center_dist = -1
            gen_std = -1
            uniformity_score = 0.0

        mixing_details = {
            'color_dist': round(float(color_dist), 2),
            'color_score': round(color_score, 4),
            'size_score': round(size_score, 4),
            'position_score': round(position_score, 4),
            'center_dist': round(float(center_dist), 2) if center_dist != -1 else -1,
            'color_std': round(float(gen_std), 2),
            'uniformity_score': round(uniformity_score, 4),
            'n_gen_objects': n_gen_objects,
            'eval_path': 'gen_object' if n_gen_objects == 1 else 'closest_object',
        }



        # --- 2. Circle removal (20%): original balls should disappear ---
        circle_only = cv2.bitwise_and(fg_first_mask, cv2.bitwise_not(mixing_mask_dilated))
        circle_pixels = int((circle_only > 0).sum())
        h, w = gt_first.shape[:2]
        corners = [gt_first[2, 2], gt_first[2, w-3], gt_first[h-3, 2], gt_first[h-3, w-3]]
        bg_color_arr = np.mean(corners, axis=0).astype(float)
        if circle_pixels > 0:
            gen_at_circles = gen_last[circle_only > 0].astype(float)
            dist_to_bg = np.sqrt(np.sum((gen_at_circles - bg_color_arr) ** 2, axis=1))
            removal_ratio = float((dist_to_bg < 40).sum()) / len(dist_to_bg)
            if removal_ratio > 0.9:
                removal_score = 1.0
            elif removal_ratio > 0.7:
                removal_score = 0.7 + (removal_ratio - 0.7) / 0.2 * 0.3
            elif removal_ratio > 0.4:
                removal_score = 0.3 + (removal_ratio - 0.4) / 0.3 * 0.4
            else:
                removal_score = removal_ratio / 0.4 * 0.3
        else:
            removal_score = 1.0
            removal_ratio = 1.0

        # --- 3. Background clean (20%): gen_first vs gen_last outside all fg ---
        bg_mask = cv2.bitwise_not(all_fg_dilated)
        bg_score, bg_details = self._pixel_diff_score(
            gen_first, gen_last, bg_mask, thresholds=(0.005, 0.015, 0.025, 0.05))

        scores = {
            'mixing_color': round(mixing_score, 4),
            'circle_removal': round(removal_score, 4),
            'background_clean': round(bg_score, 4),
        }
        self._last_task_details = {
            **scores,
            **{f'mix_{k}': v for k, v in mixing_details.items()},
            'rm_removal_ratio': round(removal_ratio, 4),
            'rm_circle_px': circle_pixels,
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class _RayLineEvaluatorBase(BaseEvaluator):
    """
    Base evaluator for tasks that draw a colored ray line (refraction/reflection).

    Evaluation:
      score = red_line * (0.6 + 0.4 * consistency)

    """

    TASK_WEIGHTS = {
        'red_line': 1.00,
    }

    def _get_line_detection_roi(self, h: int, w: int,
                                eval_info: Optional[Dict] = None) -> Optional[Tuple[int, int, int, int]]:

        return None

    @staticmethod
    def _load_gt_meta(eval_info: Optional[Dict]) -> Dict:
        import os as _os, json as _json
        if not eval_info:
            return {}
        mp = eval_info.get("gt_meta_path") or eval_info.get("metadata_path")
        if isinstance(mp, (list, tuple)):
            mp = next((q for q in mp if q and _os.path.exists(q)), None)
        if not (mp and _os.path.exists(mp)):
            mp = _os.path.join(eval_info.get("gt_path", ""), "metadata.json")
        try:
            if mp and _os.path.exists(mp):
                with open(mp) as f:
                    return _json.load(f) or {}
        except Exception:
            pass
        return {}

    def _detect_new_colored_line(self, first_frame: np.ndarray, last_frame: np.ndarray) -> np.ndarray:
        """Detect new colorful pixels that appeared in last_frame but not in first_frame."""
        diff = cv2.absdiff(first_frame, last_frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed = (diff_gray > 30).astype(np.uint8) * 255
        hsv = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        colorful = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_and(changed, colorful)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _detect_fg_mask(self, frame: np.ndarray) -> np.ndarray:
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
        changed = int((gray_diff[mask > 0] > 35).sum())
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

    def _fit_line_angle(self, mask: np.ndarray) -> Tuple[Optional[float], int]:
        """Fit a line to mask pixels. Returns (angle_degrees, n_lines_detected)."""
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=30,
                                minLineLength=20, maxLineGap=10)
        if lines is None or len(lines) == 0:
            return None, 0
        best = max(lines, key=lambda l: np.hypot(l[0][2] - l[0][0], l[0][3] - l[0][1]))
        x1, y1, x2, y2 = best[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Normalize to [0, 180): eliminate direction ambiguity from endpoint order
        angle = angle % 180
        return angle, len(lines)

    def _evaluate_red_line(self, gen_line_mask: np.ndarray, gt_line_mask: np.ndarray,
                           frame_h: int, frame_w: int) -> Tuple[float, Dict]:
        """Evaluate red line: angle difference + existence check."""
        details = {}
        gt_red_pixels = int((gt_line_mask > 0).sum())
        gen_red_pixels = int((gen_line_mask > 0).sum())

        if gt_red_pixels == 0:
            details['error'] = 'no_gt_red_line'
            return 0.0, details

        # Check if gen has a red line at all
        if gen_red_pixels < 50:
            details['error'] = 'no_gen_red_line'
            return 0.0, details

        gt_angle, gt_n_lines = self._fit_line_angle(gt_line_mask)
        gen_angle, gen_n_lines = self._fit_line_angle(gen_line_mask)
        details['gt_angle'] = round(gt_angle, 2) if gt_angle is not None else None
        details['gen_angle'] = round(gen_angle, 2) if gen_angle is not None else None
        details['gt_n_lines'] = gt_n_lines
        details['gen_n_lines'] = gen_n_lines

        if gt_angle is None or gen_angle is None:
            details['error'] = 'line_fit_failed'
            return 0.2, details

        # Angle difference (handle wraparound in [0,180) space)
        angle_diff = abs(gt_angle - gen_angle)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff  # e.g. 5° vs 175° → diff=10°, not 170°
        details['angle_diff'] = round(angle_diff, 2)

        if angle_diff <= 10:
            angle_score = 1.0
        elif angle_diff <= 18:
            angle_score = 1.0 - (angle_diff - 10) / 8 * 0.3
        elif angle_diff <= 25:
            angle_score = 0.7 - (angle_diff - 18) / 7 * 0.4
        elif angle_diff <= 40:
            angle_score = 0.3 - (angle_diff - 25) / 15 * 0.3
        else:
            angle_score = 0.0
        details['angle_score'] = round(angle_score, 4)

        def _span(mask):
            ys, xs = np.nonzero(mask)
            if ys.size < 2:
                return 0.0
            pts = np.column_stack([xs, ys]).astype(np.float32)
            hull = cv2.convexHull(pts).reshape(-1, 2)
            best = 0.0
            for i in range(len(hull)):
                for j in range(i + 1, len(hull)):
                    best = max(best, float(np.hypot(*(hull[i] - hull[j]))))
            return best

        gt_len = _span(gt_line_mask)
        gen_len = _span(gen_line_mask)
        if gt_len <= 1.0 or gen_len <= 1.0:
            len_ratio = 0.0
        else:
            len_ratio = min(gt_len, gen_len) / max(gt_len, gen_len)
        details['gt_span_px'] = round(gt_len, 1)
        details['gen_span_px'] = round(gen_len, 1)

        gt_w = gt_red_pixels / max(gt_len, 1.0)
        gen_w = gen_red_pixels / max(gen_len, 1.0)
        w_ratio = min(gt_w, gen_w) / max(gt_w, gen_w, 1e-6)
        width_penalty = 1.0 if w_ratio >= 0.5 else 0.5 + w_ratio
        details['gt_width_px'] = round(gt_w, 2)
        details['gen_width_px'] = round(gen_w, 2)
        details['width_penalty'] = round(width_penalty, 4)
        if len_ratio >= 0.9:
            len_score = 1.0
        elif len_ratio >= 0.6:
            len_score = 0.7 + (len_ratio - 0.6) / 0.3 * 0.3
        elif len_ratio >= 0.3:
            len_score = 0.3 + (len_ratio - 0.3) / 0.3 * 0.4
        else:
            len_score = len_ratio / 0.3 * 0.3
        details['len_ratio'] = round(len_ratio, 4)
        details['len_score'] = round(len_score, 4)

        red_score = angle_score * len_score * (0.5 + 0.5 * width_penalty)
        return round(red_score, 4), details

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

        frame_h, frame_w = gt_last.shape[:2]

        # Detect new colored lines in both GT and gen (within ROI)
        roi = self._get_line_detection_roi(frame_h, frame_w, eval_info)
        if roi is not None:
            y1r, y2r, x1r, x2r = roi
            gt_line_mask_roi = self._detect_new_colored_line(gt_first[y1r:y2r, x1r:x2r], gt_last[y1r:y2r, x1r:x2r])
            gen_line_mask_roi = self._detect_new_colored_line(gen_first[y1r:y2r, x1r:x2r], gen_last[y1r:y2r, x1r:x2r])
            gt_line_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            gen_line_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            gt_line_mask[y1r:y2r, x1r:x2r] = gt_line_mask_roi
            gen_line_mask[y1r:y2r, x1r:x2r] = gen_line_mask_roi
        else:
            gt_line_mask = self._detect_new_colored_line(gt_first, gt_last)
            gen_line_mask = self._detect_new_colored_line(gen_first, gen_last)


        # 1. Line correctness (60%)
        line_score, line_details = self._evaluate_red_line(gen_line_mask, gt_line_mask, frame_h, frame_w)

        # Foreground masks
        fg_first = self._detect_fg_mask(gt_first)
        fg_last = self._detect_fg_mask(gt_last)
        all_fg = fg_last

        kernel = np.ones((5, 5), np.uint8)
        gt_line_dilated = cv2.dilate(gt_line_mask, kernel, iterations=2)
        gen_line_dilated = cv2.dilate(gen_line_mask, kernel, iterations=2)
        line_exclude = cv2.bitwise_or(gt_line_dilated, gen_line_dilated)

        # Foreground excluding new lines
        fg_no_line = cv2.bitwise_and(all_fg, cv2.bitwise_not(line_exclude))
        fg_no_line_eroded = cv2.dilate(fg_no_line, kernel, iterations=1)

        # 2. Foreground preservation (20%): gt_last vs gen_last, non-line fg
        fg_score, fg_details = self._pixel_diff_score(
            gt_last, gen_last, fg_no_line_eroded, thresholds=(0.2, 0.3, 0.4, 0.60))

        # 3. Background preservation (20%): gt_last vs gen_last, bg area
        all_fg_dilated = cv2.dilate(cv2.bitwise_or(all_fg, line_exclude), kernel, iterations=1)
        bg_mask = cv2.bitwise_not(all_fg_dilated)
        bg_score, bg_details = self._pixel_diff_score(
            gt_last, gen_last, bg_mask, thresholds=(0.008, 0.015, 0.035, 0.07))

        consistency = 0.3 * fg_score + 0.7 * bg_score
        scores = {
            'red_line': round(line_score, 4),
            'consistency': round(consistency, 4),
        }
        self._last_task_details = {
            **scores,
            'foreground_preservation': round(fg_score, 4),
            'background_preservation': round(bg_score, 4),
            **{f'line_{k}': v for k, v in line_details.items()},
            **{f'fg_{k}': v for k, v in fg_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }
        return float(line_score * (0.6 + 0.4 * consistency))


class GlassRefractionEvaluator(_RayLineEvaluatorBase):
    """O-18: Glass Refraction - draw refracted ray (red line) through glass interface."""

    def _get_line_detection_roi(self, h: int, w: int,
                                eval_info: Optional[Dict] = None) -> Optional[Tuple[int, int, int, int]]:
        params = (self._load_gt_meta(eval_info) or {}).get("parameters") or {}
        try:
            iy = int(round(float(params.get("interface_y_px", params.get("mirror_y_px", h / 2)))))
        except (TypeError, ValueError):
            iy = h // 2
        try:
            hx = int(round(float(params.get("hit_x_px", w / 2))))
        except (TypeError, ValueError):
            hx = w // 2
        iy = max(0, min(h - 1, iy))
        hx = max(0, min(w - 1, hx))
        return (iy, h, hx, w)


class MirrorReflectionEvaluator(_RayLineEvaluatorBase):
    """O-19: Mirror Reflection - draw reflected ray (red line) off mirror surface."""

    def _get_line_detection_roi(self, h: int, w: int,
                                eval_info: Optional[Dict] = None) -> Optional[Tuple[int, int, int, int]]:
        params = (self._load_gt_meta(eval_info) or {}).get("parameters") or {}
        try:
            my = int(round(float(params.get("mirror_y_px", h / 2))))
        except (TypeError, ValueError):
            my = h // 2
        try:
            hx = int(round(float(params.get("hit_x_px", w / 2))))
        except (TypeError, ValueError):
            hx = w // 2
        my = max(1, min(h, my))
        hx = max(0, min(w - 1, hx))
        return (0, my, hx, w)

# Export mapping for this batch
IN_DOMAIN_50_EVALUATORS_PART3 = {
    'G-158_identify_all_hollow_points_data-generator': SelectAllHollowPointsEvaluator,
    'G-194_construct_concentric_ring_data-generator': ConstructConcentricRingEvaluator,
    'O-10_shape_outline_fill_data-generator': ShapeOutlineFillEvaluator,
    'O-12_shape_color_then_scale_data-generator': ShapeColorThenScaleEvaluator,
    'O-13_shape_outline_then_move_data-generator': ShapeOutlineThenMoveEvaluator,
    'O-14_shape_scale_then_outline_data-generator': ShapeScaleThenOutlineEvaluator,
    'O-15_ball_bounces_given_time_data-generator': BallBounceEvaluator,
    'O-16_color_addition_data-generator': ColorAdditionEvaluator,
    'O-18_glass_refraction_data-generator': GlassRefractionEvaluator,
    'O-19_mirror_reflection_data-generator': MirrorReflectionEvaluator,
}
