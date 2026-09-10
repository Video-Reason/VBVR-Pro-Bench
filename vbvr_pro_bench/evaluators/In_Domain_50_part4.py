"""
Specific evaluators for In-Domain_50 tasks (Part 4).
"""

import os
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Sequence, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import normalize_frame_size, compute_ssim, safe_distance

class ConstructionBlueprintEvaluator(BaseEvaluator):
    """
    O-21: Construction Blueprint (Missing Piece)

    Task: Select the correct piece from 4 candidates to fill the highlighted
    gap (red dotted outline) in a block structure.

    Scoring:
    - Shape matching accuracy (70%, binary 0/1): the new block exactly fills the
      red-dotted gap (not white, matches GT shape) and no extra blocks are placed
      outside gt_blocks_bbox.
    - Correct option green (20%): the correct candidate option is marked green.
    - Other options red / other options green (10% combined): options before the
      correct (green) candidate should match GT's “wrong option” styling; if the
      correct answer is the leftmost option, that 10% uses other_options_green,
      set equal to correct_option_green.
    - Deduction: option damage and strong off-GT pattern deviations (gap-adjacent
      block-mask noise suppressed). First region -5 pts; each additional -10 pts.
    """

    TASK_WEIGHTS = {
        'shape_matching': 0.70,
        'correct_option_green': 0.20,
        'other_options_red': 0.10,
    }
    ROW_OPTION_WEIGHT = 0.10

    # ------------------------------------------------------------------ helpers

    def _get_red_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        return m1 | m2

    def _get_green_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

    def _get_blue_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))

    def _detect_gap_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the largest red-dotted gap bounding box in the first frame."""
        if len(frame.shape) != 3:
            return None
        red_mask = self._get_red_mask(frame)
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, close_k)
        red_mask = cv2.dilate(red_mask, dilate_k, iterations=2)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 200:
                return cv2.boundingRect(largest)
        return None

    def _resize_to(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        if frame.shape[:2] != (h, w):
            return cv2.resize(frame, (w, h))
        return frame

    def _detect_blue_block_regions(
        self,
        frame: np.ndarray,
        gap: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[Optional[Tuple[int,int,int,int]], List[Tuple[int,int,int,int]]]:
        """
        Detect gt_blocks and gt_option_blocks from gt_first_frame.

        Returns:
          gt_blocks_bbox   - bounding box covering ALL main-pattern blue blocks
                             (union of every blue contour that is not an option).
          gt_option_bboxes - list of bounding boxes of the 4 option blocks,
                             sorted left-to-right.

        Strategy:
          1. Find all significant blue contours.
          2. Identify the 4 options as a horizontal row of similarly-sized
             blocks, typically at the bottom. Require: at least 4 candidates
             with close y-centers and comparable areas.
          3. gt_blocks_bbox = bounding box of every remaining blue contour
             (the whole upper pattern, which may be several separate pieces).
          4. Fall back to a spatial split (upper 55% = pattern, lower 45% split
             evenly in 4) when blue detection is too sparse.
        """
        h, w = frame.shape[:2]
        blue_mask = self._get_blue_mask(frame)
        # Mild dilation: large kernels merge adjacent option blobs and widen boxes
        # past the light-gray option cells.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blue_mask = cv2.dilate(blue_mask, kernel)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 300]

        def _fallback():
            split_y = int(h * 0.55)
            gt_blocks_bbox = (0, 0, w, split_y)
            opt_w = w // 4
            gt_option_bboxes = [(i * opt_w, split_y, opt_w, h - split_y) for i in range(4)]
            return gt_blocks_bbox, self._clamp_option_bboxes_to_column_bands(
                gt_option_bboxes, w)

        if len(bboxes) < 4:
            return _fallback()

        # --- Identify the 4 option blocks ---
        def _overlaps_gap(b):
            if gap is None:
                return False
            bx, by, bw, bh = b
            gx, gy, gw, gh = gap
            return not (bx + bw <= gx or gx + gw <= bx or
                        by + bh <= gy or gy + gh <= by)

            # Candidates: must be below mid-frame and not overlap the gap.
        candidates = [b for b in bboxes
                      if (b[1] + b[3] // 2) > h * 0.45 and not _overlaps_gap(b)]

        option_bboxes: List[Tuple[int, int, int, int]] = []
        if len(candidates) >= 4:
            # Look for the largest group of candidates sharing a similar y-center
            # and size — these are the 4 options laid out in a row.
            cands = sorted(candidates, key=lambda b: b[1] + b[3] // 2)  # by y-center
            best_group: List[Tuple[int,int,int,int]] = []
            for i in range(len(cands)):
                ref_cy = cands[i][1] + cands[i][3] // 2
                ref_area = cands[i][2] * cands[i][3]
                group = [b for b in cands
                         if abs((b[1] + b[3] // 2) - ref_cy) <= max(h * 0.08, 30)
                         and 0.35 <= (b[2] * b[3]) / (ref_area + 1) <= 3.0]
                if len(group) >= 4 and len(group) > len(best_group):
                    best_group = group
            if len(best_group) >= 4:
                # Pick the 4 bottom-most in the group (options sit at the bottom)
                best_group.sort(key=lambda b: -(b[1] + b[3] // 2))
                option_bboxes = sorted(best_group[:4], key=lambda b: b[0])

        if len(option_bboxes) != 4:
            return _fallback()

        # --- gt_blocks_bbox = union of everything that is NOT an option ---
        option_set = set(option_bboxes)
        pattern_bboxes = [b for b in bboxes if b not in option_set]

        if not pattern_bboxes:
            # No pattern detected — fall back to upper region
            split_y = min(b[1] for b in option_bboxes)
            gt_blocks_bbox: Optional[Tuple[int, int, int, int]] = (0, 0, w, split_y)
        else:
            x0 = min(b[0] for b in pattern_bboxes)
            y0 = min(b[1] for b in pattern_bboxes)
            x1 = max(b[0] + b[2] for b in pattern_bboxes)
            y1 = max(b[1] + b[3] for b in pattern_bboxes)
            # Grow the union slightly to cover thin borders / anti-alias edges.
            pad = 4
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
            # Ensure the detected gap is inside gt_blocks_bbox.
            if gap is not None:
                gx, gy, gw, gh = gap
                x0 = min(x0, gx); y0 = min(y0, gy)
                x1 = max(x1, gx + gw); y1 = max(y1, gy + gh)
            gt_blocks_bbox = (x0, y0, x1 - x0, y1 - y0)

        option_bboxes = self._clamp_option_bboxes_to_column_bands(option_bboxes, w)
        return gt_blocks_bbox, option_bboxes

    def _clamp_option_bboxes_to_column_bands(
        self,
        option_bboxes: List[Tuple[int, int, int, int]],
        frame_w: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Keep each option ROI inside its horizontal slot: split at midpoints between
        adjacent blue centroids so dilated masks cannot bleed into neighbors.
        """
        if len(option_bboxes) != 4:
            return option_bboxes
        ob = sorted(option_bboxes, key=lambda b: b[0])
        cx = [b[0] + b[2] // 2 for b in ob]
        boundaries = [0] + [(cx[i] + cx[i + 1]) // 2 for i in range(3)] + [frame_w]
        out: List[Tuple[int, int, int, int]] = []
        for i, b in enumerate(ob):
            ox, oy, ow, oh = b
            xL, xR = boundaries[i], boundaries[i + 1]
            nx1 = max(xL, ox)
            nx2 = min(xR, ox + ow)
            if nx2 - nx1 < max(8, ow // 3):
                out.append((ox, oy, ow, oh))
            else:
                out.append((nx1, oy, nx2 - nx1, oh))
        return out

    def debug_detect_regions(
        self,
        frame: np.ndarray,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Visualize gap / gt_blocks / gt_option detection on a frame. Draws:
            red    = gap
            yellow = gt_blocks_bbox
            cyan   = each gt_option_bbox (numbered 1..4)
        Returns the detected values; optionally writes an annotated image.
        """
        gap = self._detect_gap_region(frame)
        gt_blocks_bbox, gt_option_bboxes = self._detect_blue_block_regions(frame, gap)
        vis = frame.copy()
        if gap is not None:
            gx, gy, gw, gh = gap
            cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 2)
            cv2.putText(vis, "gap", (gx, max(0, gy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if gt_blocks_bbox is not None:
            bx, by, bw, bh = gt_blocks_bbox
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            cv2.putText(vis, "gt_blocks", (bx, max(0, by - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for i, (ox, oy, ow, oh) in enumerate(gt_option_bboxes, 1):
            cv2.rectangle(vis, (ox, oy), (ox + ow, oy + oh), (255, 255, 0), 2)
            cv2.putText(vis, f"opt{i}", (ox, max(0, oy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if save_path is not None:
            cv2.imwrite(save_path, vis)
        return {
            "gap": gap,
            "gt_blocks_bbox": gt_blocks_bbox,
            "gt_option_bboxes": gt_option_bboxes,
            "visualization": vis,
        }

    def _is_mostly_white(self, region: np.ndarray, white_thresh: int = 240, ratio_thresh: float = 0.85) -> bool:
        """Return True if the region is predominantly white (nothing was drawn)."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        return float(np.sum(gray >= white_thresh)) / (gray.size + 1) >= ratio_thresh

    def _get_sorted_contour_centers(self, mask: np.ndarray, min_area: int = 200) -> List[Tuple[int, int, int, int]]:
        """Return bounding boxes sorted left-to-right by x centre."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > min_area]
        boxes.sort(key=lambda b: b[0] + b[2] // 2)
        return boxes

    def _green_bbox_for_correct_option(
        self,
        gt_green: np.ndarray,
        gt_option_bboxes: List[Tuple[int, int, int, int]],
        frame_h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Green UI in the pattern (checkmarks) shares the same HSV band as the
        selected option. Prefer the green blob whose centre lies inside an
        option tile; otherwise the lowest green region (option strip).
        """
        cnts, _ = cv2.findContours(gt_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 200]
        if not boxes:
            return None

        def _centre_in_option(b: Tuple[int, int, int, int]) -> bool:
            cx = b[0] + b[2] // 2
            cy = b[1] + b[3] // 2
            for ox, oy, ow, oh in gt_option_bboxes:
                if ox <= cx < ox + ow and oy <= cy < oy + oh:
                    return True
            return False

        in_strip = [b for b in boxes if _centre_in_option(b)]
        if in_strip:
            return max(in_strip, key=lambda b: (b[1] + b[3], b[2] * b[3]))
        lower = [b for b in boxes if (b[1] + b[3] // 2) > frame_h * 0.33]
        if lower:
            return max(lower, key=lambda b: b[1] + b[3])
        return max(boxes, key=lambda b: b[1] + b[3])

    def _option_crop_mean_absdiff(
        self,
        a: np.ndarray,
        b: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> float:
        ox, oy, ow, oh = roi
        ca = a[oy:oy + oh, ox:ox + ow].astype(float)
        cb = b[oy:oy + oh, ox:ox + ow].astype(float)
        if ca.size == 0:
            return 0.0
        return float(np.mean(np.abs(ca - cb)))

    # ------------------------------------------------------ scoring components

    def _score_shape_matching(
        self,
        gen_final: np.ndarray,
        gt_first: np.ndarray,
        gt_final: np.ndarray,
        gt_blocks_bbox: Optional[Tuple[int, int, int, int]],
        gt_option_bboxes: List[Tuple[int, int, int, int]],
        gap: Optional[Tuple[int, int, int, int]],
    ) -> float:
        """
        70% — binary 0 or 1.

        Conditions that must ALL be true for score = 1:
        1. The gap region in gen_final CHANGED from gt_first (something new was placed).
           This catches: blank/white fill, unchanged red-dotted gap, off-white backgrounds.
           Comparing against gt_first rather than checking for white handles all cases.
        2. The content drawn inside the gap matches gt_final (pixel diff < threshold).
        3. No new blocks were placed OUTSIDE gt_blocks_bbox (structural change check
           using block-presence maps, not raw pixel diff, to avoid false positives from
           color overlay changes like option markings).
        """
        h, w = gen_final.shape[:2]
        gt_first = self._resize_to(gt_first, h, w)
        gt_final = self._resize_to(gt_final, h, w)

        if gap is None:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            return 1.0 if diff < 30 else 0.0

        gx, gy, gw, gh = gap
        gen_gap_crop   = gen_final[gy:gy+gh, gx:gx+gw]
        gt_first_gap   = gt_first[gy:gy+gh, gx:gx+gw]
        gt_final_gap   = gt_final[gy:gy+gh, gx:gx+gw]

        def _nonbg_ratio(crop, bg_thresh=200):
            if crop.size == 0:
                return 0.0
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            return float(np.sum(gray < bg_thresh)) / float(gray.size + 1)

        # ---- Condition 1: gap was actually FILLED in gen_final ----
        first_diff = float(np.abs(gen_gap_crop.astype(float) - gt_first_gap.astype(float)).mean())
        final_diff = float(np.abs(gen_gap_crop.astype(float) - gt_final_gap.astype(float)).mean())

        # (b) gap unchanged — gen looks (almost) like gt_first
        if first_diff < 15:
            return 0.0

        # (a) gap blank — gen's block-content ratio is much lower than expected
        gen_fill   = _nonbg_ratio(gen_gap_crop)
        gtfin_fill = _nonbg_ratio(gt_final_gap)
        if gtfin_fill > 0.1 and gen_fill < gtfin_fill * 0.5:
            return 0.0

        # gen must be closer to gt_final than to gt_first
        if final_diff >= first_diff * 0.8:
            return 0.0

        # (c) absolute match threshold (margin for compression / codec noise)
        if final_diff >= 52:
            return 0.0

        # ---- Condition 2: no extra blocks placed OUTSIDE gt_blocks_bbox + options ----
        # The "outside" region = not the pattern and not the options row.
        if gt_blocks_bbox is not None:
            outside_mask = np.ones((h, w), dtype=np.uint8)
            bx, by, bw, bh = gt_blocks_bbox
            outside_mask[by:by+bh, bx:bx+bw] = 0
            for ox, oy, ow, oh in gt_option_bboxes:
                outside_mask[oy:oy+oh, ox:ox+ow] = 0

            def _block_presence(frame, bg_thresh=200):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                _, binary = cv2.threshold(gray, bg_thresh, 255, cv2.THRESH_BINARY_INV)
                return binary

            gen_p   = _block_presence(gen_final)
            first_p = _block_presence(gt_first)
            gt_p    = _block_presence(gt_final)

            # "New" presence in gen, outside, that isn't also new in gt.
            gen_new = cv2.bitwise_and(gen_p, cv2.bitwise_not(first_p))
            gt_new  = cv2.bitwise_and(gt_p,  cv2.bitwise_not(first_p))
            gen_new = cv2.bitwise_and(gen_new, gen_new, mask=outside_mask)
            gt_new  = cv2.bitwise_and(gt_new,  gt_new,  mask=outside_mask)

            # Dilate expected change to absorb minor placement drift
            dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            allowed  = cv2.dilate(gt_new, dilate_k)
            unexpected = cv2.bitwise_and(gen_new, cv2.bitwise_not(allowed))

            # Erode thin noise away, then count.
            erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            unexpected = cv2.erode(unexpected, erode_k)
            if float(np.sum(unexpected > 0)) > 800:
                return 0.0

        return 1.0

    def _score_option_selection(
        self,
        gen_final: np.ndarray,
        gt_first: np.ndarray,
        gt_final: np.ndarray,
        gt_option_bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[float, float, float, int]:
        """
        Returns (correct_option_green, other_options_red, other_options_green, correct_idx).

        Wrong-option styling is often coral/brown, not HSV-pure red; score using
        per-option ROI change vs gt_first. When correct_idx == 0 (no reds to the
        left), other_options_green equals correct_option_green for the row weight.
        """
        h, w = gen_final.shape[:2]
        gt_final = self._resize_to(gt_final, h, w)
        gt_first = self._resize_to(gt_first, h, w)

        gen_green = self._get_green_mask(gen_final)
        gt_green  = self._get_green_mask(gt_final)

        def _count_cnts(mask, min_area=200):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return len([c for c in cnts if cv2.contourArea(c) > min_area])

        gt_green_count = _count_cnts(gt_green)

        if gt_green_count == 0:
            correct_green = 1.0 if _count_cnts(gen_green) == 0 else 0.5
            return correct_green, 1.0, 1.0, -1

        overlap  = float(np.sum((gen_green > 0) & (gt_green > 0)))
        gen_px   = float(np.sum(gen_green > 0)) + 1.0
        gt_px    = float(np.sum(gt_green > 0)) + 1.0
        precision = overlap / gen_px
        recall    = overlap / gt_px
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        gen_green_count = _count_cnts(gen_green)
        extra_penalty = max(0.0, gen_green_count - gt_green_count) * 0.2
        correct_green = max(0.0, f1 - extra_penalty)

        gbox = self._green_bbox_for_correct_option(gt_green, gt_option_bboxes, h)
        if gbox is None:
            return correct_green, 1.0, 1.0, 0

        cx = gbox[0] + gbox[2] // 2
        cy = gbox[1] + gbox[3] // 2
        correct_idx = 0
        for i, (ox, oy, ow, oh) in enumerate(gt_option_bboxes):
            if ox <= cx < ox + ow and oy <= cy < oy + oh:
                correct_idx = i
                break

        t_low = 3.5
        match_eps = 0.35
        subtle_t_cutoff = 22.0

        def mark_strength(frame: np.ndarray, opt_i: int) -> float:
            roi = gt_option_bboxes[opt_i]
            return self._option_crop_mean_absdiff(frame, gt_first, roi)

        def gen_vs_gt_final_strength(opt_i: int) -> float:
            roi = gt_option_bboxes[opt_i]
            return self._option_crop_mean_absdiff(gen_final, gt_final, roi)

        if correct_idx == 0:
            other_red = 1.0
            other_green = correct_green
            return correct_green, other_red, other_green, correct_idx

        other_green = 1.0
        left_idx = list(range(correct_idx))
        eligible = [j for j in left_idx if mark_strength(gt_final, j) >= t_low]
        if not eligible:
            return correct_green, 1.0, other_green, correct_idx

        def _slot_red_hit(g_j: float, t_j: float, d_j: float) -> bool:
            if abs(g_j - t_j) < match_eps:
                return True
            if t_j < subtle_t_cutoff:
                d_cap = max(6.5, 1.15 * t_j + 1.5)
                # Require real change vs first frame, not only low |gen−gf| on static tiles.
                g_min = max(3.8, 0.65 * t_j)
                return (g_j >= g_min) and (d_j <= d_cap)
            lo, hi = 0.78 * t_j, 1.26 * t_j
            d_cap = max(16.0, 0.18 * t_j)
            return (lo <= g_j <= hi) and (d_j <= d_cap)

        hits = 0.0
        for j in eligible:
            t_j = mark_strength(gt_final, j)
            g_j = mark_strength(gen_final, j)
            d_j = gen_vs_gt_final_strength(j)
            if _slot_red_hit(g_j, t_j, d_j):
                hits += 1.0
        other_red = hits / float(len(eligible))
        return correct_green, other_red, other_green, correct_idx

    @staticmethod
    def _deduction_odd_kernel_size(
        dim_max: int,
        min_k: int,
        max_k: int,
        divisor: int,
    ) -> int:
        """Odd kernel length in [min_k, max_k], scaling with max(h, w)."""
        k = max(min_k, min(max_k, 2 * (dim_max // divisor) + 1))
        if k % 2 == 0:
            k = min(max_k, k + 1)
        return k

    def _score_deduction(
        self,
        gen_final: np.ndarray,
        gt_first: np.ndarray,
        gt_final: np.ndarray,
        gap: Optional[Tuple[int, int, int, int]],
        gt_blocks_bbox: Optional[Tuple[int, int, int, int]],
        gt_option_bboxes: List[Tuple[int, int, int, int]],
    ) -> float:
        """
        Deduction for UNEXPECTED block-mask changes vs gt_first that are not
        explained by gt_final (dilated), evaluated only on pixels that belong
        to the GT or generated block union (pattern + options, gap excluded).

        Morphology kernel sizes scale with frame resolution (not fixed 5×5 /
        15×15). Option-tile faults count only unexpected pixels that also lie on
        the GT ∪ gen block mask inside each option bbox.

        Regions are counted in two buckets: (1) each option tile with large
        unexpected mass (e.g. erased option), (2) at most one pattern-core fault
        when unexpected pixels are numerous and backed by strong RGB deviation
        from gt_final (suppresses gap-adjacent silhouette / codec noise).

        First region: -5 pts; each additional: -10 pts. Returns a value <= 0.
        """
        h, w = gen_final.shape[:2]
        gt_first = self._resize_to(gt_first, h, w)
        gt_final = self._resize_to(gt_final, h, w)
        dim_max = max(h, w)
        ref_area = 512.0 * 512.0
        area_scale = max(0.25, min(4.0, (h * w) / ref_area))

        def _block_presence(frame, bg_thresh=200):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            _, binary = cv2.threshold(gray, bg_thresh, 255, cv2.THRESH_BINARY_INV)
            return binary  # 255 = block present, 0 = background

        gen_p   = _block_presence(gen_final)
        first_p = _block_presence(gt_first)
        gt_p    = _block_presence(gt_final)

        # Only score where GT (first/final) or gen draws blocks — not bare canvas.
        block_union = cv2.bitwise_or(cv2.bitwise_or(first_p, gt_p), gen_p)
        edge_sz = self._deduction_odd_kernel_size(dim_max, min_k=5, max_k=35, divisor=48)
        edge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_sz, edge_sz))
        block_area = cv2.dilate(block_union, edge_k)
        block_bin = (block_area > 0).astype(np.uint8)

        # Check mask = pattern area + option row, minus (padded) gap, ∩ block_area.
        check_mask = np.zeros((h, w), dtype=np.uint8)
        if gt_blocks_bbox is not None:
            bx, by, bw, bh = gt_blocks_bbox
            check_mask[by:by+bh, bx:bx+bw] = 1
        for ox, oy, ow, oh in gt_option_bboxes:
            check_mask[oy:oy+oh, ox:ox+ow] = 1
        if gap is not None:
            gx, gy, gw, gh = gap
            # Pad the gap exclusion generously so the red-dotted border / any
            # overshoot of the fill block outside the detected gap rect does
            # NOT register as a spurious change.
            pad = max(12, gw // 4, gh // 4)
            gx0 = max(0, gx - pad); gy0 = max(0, gy - pad)
            gx1 = min(w, gx + gw + pad); gy1 = min(h, gy + gh + pad)
            check_mask[gy0:gy1, gx0:gx1] = 0

        check_mask = cv2.bitwise_and(check_mask, block_bin)

        if not np.any(check_mask):
            return 0.0

        gen_change = cv2.absdiff(gen_p, first_p)
        gt_change  = cv2.absdiff(gt_p,  first_p)

        allow_sz = self._deduction_odd_kernel_size(dim_max, min_k=9, max_k=37, divisor=56)
        dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (allow_sz, allow_sz))
        allowed  = cv2.dilate(gt_change, dilate_k)

        unexpected = cv2.bitwise_and(gen_change, cv2.bitwise_not(allowed))
        unexpected = cv2.bitwise_and(unexpected, unexpected, mask=check_mask)

        erode_sz = self._deduction_odd_kernel_size(dim_max, min_k=3, max_k=7, divisor=200)
        erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_sz, erode_sz))
        unexpected = cv2.erode(unexpected, erode_k)

        # Bucket faults: (1) each damaged option tile, (2) at most one pattern-core
        # fault when block-mask noise is backed by strong RGB deviation from GT.
        # Gap-adjacent silhouette noise usually has low mean RGB diff → ignored.
        opt_union = np.zeros((h, w), dtype=np.uint8)
        for ox, oy, ow, oh in gt_option_bboxes:
            opt_union[oy:oy + oh, ox:ox + ow] = 255

        rgb_diff = np.linalg.norm(
            gen_final.astype(np.float32) - gt_final.astype(np.float32), axis=2)

        changed_regions = 0
        opt_px_thr = int(max(350.0, 800.0 * area_scale))

        for ox, oy, ow, oh in gt_option_bboxes:
            u = unexpected[oy:oy + oh, ox:ox + ow]
            b = block_bin[oy:oy + oh, ox:ox + ow]
            n_opt_unex = int(np.sum((u > 0) & (b > 0)))
            if n_opt_unex > opt_px_thr:
                changed_regions += 1

        if gt_blocks_bbox is not None:
            bx, by, bw, bh = gt_blocks_bbox
            pat_mask = np.zeros((h, w), dtype=np.uint8)
            pat_mask[by:by + bh, bx:bx + bw] = 255
            if gap is not None:
                gx, gy, gw, gh = gap
                pad = max(12, gw // 4, gh // 4)
                gx0 = max(0, gx - pad)
                gy0 = max(0, gy - pad)
                gx1 = min(w, gx + gw + pad)
                gy1 = min(h, gy + gh + pad)
                pat_mask[gy0:gy1, gx0:gx1] = 0
            pat_mask = cv2.bitwise_and(pat_mask, cv2.bitwise_not(opt_union))
            pat_mask = cv2.bitwise_and(pat_mask, block_bin)
            pat_unex = cv2.bitwise_and(unexpected, unexpected, mask=pat_mask)
            n_pat = int(np.sum(pat_unex > 0))
            pat_px_thr = int(max(250.0, 450.0 * area_scale))
            if n_pat > pat_px_thr:
                ys, xs = np.where(pat_unex > 0)
                mean_rgb = float(np.mean(rgb_diff[ys, xs]))
                if mean_rgb > 218.0:
                    changed_regions += 1

        if changed_regions == 0:
            return 0.0
        deduction_pts = 5.0 + max(0, changed_regions - 1) * 10.0
        return -(deduction_pts / 100.0)

    # ---------------------------------------------------------------- evaluate

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate blueprint piece selection and placement."""

        if not video_frames or gt_final_frame is None:
            return 0.0

        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        gt_first = gt_first_frame if gt_first_frame is not None else video_frames[0]

        scores = {}

        gap = self._detect_gap_region(gt_first)
        gt_blocks_bbox, gt_option_bboxes = self._detect_blue_block_regions(gt_first, gap)

        scores['shape_matching'] = self._score_shape_matching(
            gen_final, gt_first, gt_final, gt_blocks_bbox, gt_option_bboxes, gap)
        cg, or_red, or_green, cidx = self._score_option_selection(
            gen_final, gt_first, gt_final, gt_option_bboxes)
        scores['correct_option_green'] = cg
        scores['other_options_red'] = or_red
        scores['other_options_green'] = or_green
        scores['correct_option_index'] = cidx

        if cidx < 0:
            row_score = 1.0
        elif cidx == 0:
            row_score = or_green
        else:
            row_score = or_red

        base_score = (
            self.TASK_WEIGHTS['shape_matching'] * scores['shape_matching']
            + self.TASK_WEIGHTS['correct_option_green'] * scores['correct_option_green']
            + self.ROW_OPTION_WEIGHT * row_score
        )

        deduction = self._score_deduction(
            gen_final, gt_first, gt_final, gap, gt_blocks_bbox, gt_option_bboxes)

        self._last_task_details = {**scores, 'deduction': deduction, 'row_score': row_score}
        return max(0.0, base_score + deduction)

    def _detect_green_filled_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect largest green filled region (kept for compatibility)."""
        if len(frame.shape) != 3:
            return None
        green_mask = self._get_green_mask(frame)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                return cv2.boundingRect(largest)
        return None


class DominoChainBranchEvaluator(BaseEvaluator):
    """
    O-23: Domino Chain Branch Path Prediction

    Y-shaped domino chain: START → Trunk → Branch A (up) + Branch B (down).
    Some samples have a gap (red X) stopping one branch.

    Detection: template matching from GT first frame templates.
    - Standing: original template matches well (score >= STANDING_THRESH)
    - Fallen: original template doesn't match, but rotated template does
              (confirms domino is still present, just tilted)

    Evaluation:
    1. Final State (50%): per-domino state correctness + position + background
    2. Process (50%): fall order left-to-right + completion
    """

    TASK_WEIGHTS = {
        'final_state': 0.50,
        'process': 0.50,
    }
    STANDING_THRESH = 0.6
    FALLEN_THRESH = 0.6
    ANGLE_TOL_DEG = 5.0
    ANGLE_DROP_DEG = 20.0
    ANGLE_SCORE_FLOOR = 0.5

    def _pixel_diff_score(self, frame1, frame2, mask, thresholds=(0.02, 0.05, 0.10, 0.20)):
        mask_pixels = int((mask > 0).sum())
        if mask_pixels == 0:
            return 1.0, {'ratio': 0.0, 'changed_px': 0, 'total_px': 0}
        diff = cv2.absdiff(frame1, frame2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
        changed = int((gray_diff[mask > 0] > 20).sum())
        ratio = float(changed) / mask_pixels
        t1, t2, t3, t4 = thresholds
        if ratio < t1: score = 1.0
        elif ratio < t2: score = 1.0 - (ratio - t1) / (t2 - t1) * 0.3
        elif ratio < t3: score = 0.7 - (ratio - t2) / (t3 - t2) * 0.4
        elif ratio < t4: score = 0.3 - (ratio - t3) / (t4 - t3) * 0.3
        else: score = 0.0
        return score, {'ratio': round(ratio, 6), 'changed_px': changed, 'total_px': mask_pixels}


    def _extract_dominos(self, frame, min_area=1000):
        """Detect domino regions in GT first frame. Returns list sorted by cx.
        Filters out legend text (bottom 15%) and non-rectangular shapes (X marks).
        """
        # Sample background color from corners
        corners = [frame[5, 5], frame[5, -5], frame[-5, 5], frame[-5, -5]]
        self._bg_color = tuple(int(v) for v in np.mean(corners, axis=0))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_img = frame.shape[0]
        dominos = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            # Solidity filter: exclude X marks
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and area / hull_area < 0.75:
                continue

            pad = 2
            y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
            dominos.append({
                'cx': cx, 'cy': cy,
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area,
                'template': gray[y1:y2, x1:x2].copy(),
                'color_template': frame[y1:y2, x1:x2].copy(),
            })
        dominos.sort(key=lambda d: d['cx'])
        return dominos


    def _rotate_template(self, template, angle_deg):
        """Rotate template (grayscale or BGR), padding with background color."""
        h, w = template.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle_deg, 1.0)
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        nw = int(h * sin_a + w * cos_a)
        nh = int(h * cos_a + w * sin_a)
        M[0, 2] += (nw - w) / 2
        M[1, 2] += (nh - h) / 2
        bg = getattr(self, '_bg_color', (240, 243, 245))
        return cv2.warpAffine(template, M, (nw, nh), borderMode=cv2.BORDER_CONSTANT, borderValue=bg)

    def _match_template_region(self, frame, template, center,
                               search_radius=150,
                               search_bounds=None,
                               scales=(0.85, 1.0, 1.15),
                               angles=(0,)):
        """Match template near *center* with search constraint.
        search_bounds: (left, right, top, bottom) asymmetric bounds from center.
                       If None, uses circular search_radius.
        Returns (score, (cx,cy), angle, scale)."""
        h_f, w_f = frame.shape[:2]
        cx, cy = center
        th, tw = template.shape[:2]
        max_dim = int(max(th, tw) * max(scales) * 1.5)

        if search_bounds is not None:
            sb_l, sb_r, sb_t, sb_b = search_bounds
            y1 = max(0, cy - sb_t - max_dim)
            y2 = min(h_f, cy + sb_b + max_dim)
            x1 = max(0, cx - sb_l - max_dim)
            x2 = min(w_f, cx + sb_r + max_dim)
        else:
            margin = search_radius + max_dim
            y1 = max(0, cy - margin)
            y2 = min(h_f, cy + margin)
            x1 = max(0, cx - margin)
            x2 = min(w_f, cx + margin)

        region = frame[y1:y2, x1:x2]
        rh, rw = region.shape[:2]
        rel_cx, rel_cy = cx - x1, cy - y1

        best = (-1.0, None, 0, 1.0)
        best_weighted = -1.0
        for angle in angles:
            rot = self._rotate_template(template, angle) if angle else template
            for scale in scales:
                sh = int(rot.shape[0] * scale)
                sw = int(rot.shape[1] * scale)
                if sh >= rh or sw >= rw or sh < 5 or sw < 5:
                    continue
                scaled = cv2.resize(rot, (sw, sh))
                result = cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED)
                res_h, res_w = result.shape[:2]
                x_coords = np.arange(res_w) + sw // 2
                y_coords = np.arange(res_h) + sh // 2
                xx, yy = np.meshgrid(x_coords, y_coords)

                if search_bounds is not None:
                    sb_l, sb_r, sb_t, sb_b = search_bounds
                    dmask = ((xx >= rel_cx - sb_l) & (xx <= rel_cx + sb_r) &
                             (yy >= rel_cy - sb_t) & (yy <= rel_cy + sb_b)
                             ).astype(np.uint8) * 255
                else:
                    dist = np.sqrt((xx.astype(float) - rel_cx) ** 2 +
                                   (yy.astype(float) - rel_cy) ** 2)
                    dmask = (dist <= search_radius).astype(np.uint8) * 255

                if dmask.sum() == 0:
                    continue
                # Apply proximity weighting to result map before finding max
                dist_map = np.sqrt((xx.astype(float) - rel_cx) ** 2 +
                                   (yy.astype(float) - rel_cy) ** 2)
                prox_map = np.maximum(0.3, 1.0 - 0.5 * dist_map / max(1, max_dim))
                # Crop prox_map to match result shape
                prox_crop = prox_map[:res_h, :res_w].astype(np.float32)
                weighted_result = result * prox_crop
                _, wv, _, wl = cv2.minMaxLoc(weighted_result, mask=dmask)
                if wv > best_weighted:
                    # Get raw score at the weighted-best position
                    raw_score = float(result[wl[1], wl[0]])
                    match_cx = wl[0] + sw // 2
                    match_cy = wl[1] + sh // 2
                    best = (raw_score, (match_cx + x1, match_cy + y1), angle, scale)
                    best_weighted = wv
        return best

    def _check_standing(self, frame, domino, search_radius=100):
        """Is domino still standing? Returns (score, position, scale)."""
        s, p, _, sc = self._match_template_region(
            frame, domino['color_template'],
            (domino['cx'], domino['cy']),
            search_radius=search_radius,
            scales=(0.85, 1.0, 1.15),
            angles=(0,))
        return s, p, sc

    def _check_fallen(self, frame, domino, scale_factor=1.2):
        """Try to locate fallen domino at various angles. """
        h = domino['h']
        sb_l = int(h / 2 * scale_factor)   # left: small
        sb_r = int(h * scale_factor)        # right: large (fall direction)
        sb_v = int(h / 2 * scale_factor)    # vertical: moderate
        s, p, a, sc = self._match_template_region(
            frame, domino['color_template'],
            (domino['cx'], domino['cy']),
            search_bounds=(sb_l, sb_r, sb_v, sb_v),
            scales=(0.85, 1.0, 1.15),
            angles=list(range(5, 110, 5)))
        return s, p, a, sc

    def _detect_domino_state(self, frame, domino):
        """Detect domino state in a single frame (BGR).
        Returns ('standing'|'fallen'|'missing', details_dict).
        """
        stand_s, stand_p, stand_sc = self._check_standing(frame, domino)
        if stand_s >= self.STANDING_THRESH:
            pos_dist = np.hypot(stand_p[0] - domino['cx'],
                                stand_p[1] - domino['cy']) if stand_p else 0
            return 'standing', {
                'stand_score': round(stand_s, 3),
                'stand_pos': stand_p,
                'scale': stand_sc,
                'pos_dist': round(pos_dist, 1),
            }

        # Not standing — check if fallen (rotated template matches)
        fall_s, fall_p, fall_a, fall_sc = self._check_fallen(frame, domino)
        if fall_s >= self.FALLEN_THRESH:
            pos_dist = np.hypot(fall_p[0] - domino['cx'],
                                fall_p[1] - domino['cy']) if fall_p else 0
            return 'fallen', {
                'stand_score': round(stand_s, 3),
                'fall_score': round(fall_s, 3),
                'fall_pos': fall_p,
                'fall_angle': fall_a,
                'scale': fall_sc,
                'pos_dist': round(pos_dist, 1),
            }

        # Neither standing nor fallen template matches
        return 'missing', {
            'stand_score': round(stand_s, 3),
            'fall_score': round(fall_s, 3),
        }

    def _detect_branches(self, dominos):
        trunk_cy = dominos[0]['cy']
        tol = dominos[0]['h'] * 0.5
        trunk, branch_a, branch_b = [], [], []
        for i, d in enumerate(dominos):
            if abs(d['cy'] - trunk_cy) <= tol:
                trunk.append(i)
            elif d['cy'] < trunk_cy:
                branch_a.append(i)
            else:
                branch_b.append(i)
        trunk.sort(key=lambda i: dominos[i]['cx'])
        branch_a.sort(key=lambda i: dominos[i]['cx'])
        branch_b.sort(key=lambda i: dominos[i]['cx'])
        branches = [b for b in [branch_a, branch_b] if b]
        return {'trunk': trunk, 'branches': branches}

    def _detect_frame_states(self, frame, dominos, target_indices, fi):
        """Detect all target dominos in a single frame. Thread-safe."""
        return [(di, fi, *self._detect_domino_state(frame, dominos[di]))
                for di in target_indices]

    def _build_state_timeline(self, video_frames, dominos, target_indices,
                              sample_step=None, num_workers=1):
        """For specified dominos only, detect state at sampled frames."""
        n = len(video_frames)
        if sample_step is None:
            sample_step = max(1, n // 20)
        samples = list(range(0, n, sample_step))
        if samples[-1] != n - 1:
            samples.append(n - 1)

        if not target_indices:
            return {i: [] for i in target_indices}, samples

        # Collect per-frame results (may be out of order if threaded)
        all_results = []
        num_workers = max(5, len(samples))
        if num_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {pool.submit(self._detect_frame_states,
                                       video_frames[fi], dominos,
                                       target_indices, fi): fi
                           for fi in samples}
                for fut in futures:
                    all_results.extend(fut.result())
        else:
            for fi in samples:
                all_results.extend(
                    self._detect_frame_states(video_frames[fi], dominos,
                                              target_indices, fi))

        # Sort by frame index and build timeline
        all_results.sort(key=lambda x: x[1])  # sort by fi
        timeline = {i: [] for i in target_indices}
        for di, fi, state, det in all_results:
            timeline[di].append((fi, state, det))
        return timeline, samples

    def _find_fall_frame(self, state_entries):
        if not state_entries:
            return None
        if state_entries[-1][1] == 'missing':
            return None

        last_stand_idx = None
        for idx in range(len(state_entries) - 1, -1, -1):
            if state_entries[idx][1] == 'standing':
                last_stand_idx = idx
                break

        if last_stand_idx is not None and last_stand_idx + 1 < len(state_entries):
            return state_entries[last_stand_idx + 1][0]  # frame_idx of next entry

        return None

    def _add_domino_to_mask(self, mask, domino, det, state):
        """Add a detected domino region to foreground mask using rotated rect.
        Scale is enlarged by 1.2 to fully cover the domino area.
        """
        scale = det.get('scale', 1.0) * 1.2
        w, h = domino['w'] * scale, domino['h'] * scale
        if state == 'standing':
            pos = det.get('stand_pos')
            if pos:
                rect = (pos, (w, h), 0)
                box = cv2.boxPoints(rect).astype(int)
                cv2.fillPoly(mask, [box], 255)
        elif state == 'fallen':
            pos = det.get('fall_pos')
            angle = det.get('fall_angle', 0)
            if pos:
                rect = (pos, (w, h), -angle)
                box = cv2.boxPoints(rect).astype(int)
                cv2.fillPoly(mask, [box], 255)

    def _score_final_state(self, gen_last, gt_last, dominos, should_fall,
                           gen_detections, gt_detections):
        """Final state: per-domino (state * position) + background.
        should_stand dominos (after gap) get 2x weight."""
        domino_scores = []
        domino_weights = []
        per_domino = []

        # Build foreground mask from gen + gt detected positions
        domino_mask = np.zeros(gen_last.shape[:2], np.uint8)

        for i, d in enumerate(dominos):
            state, det = gen_detections[i]
            gt_state, gt_det = gt_detections[i]

            self._add_domino_to_mask(domino_mask, d, gt_det, gt_state)

            # State score
            if should_fall[i]:
                if state == 'fallen':
                    gen_dx = det.get('fall_pos', (d['cx'],))[0] - d['cx']
                    if gen_dx < -15:
                        state_s = 0.0  # fell left, should fall right
                    else:
                        state_s = 1.0
                else:
                    state_s = 0.0
            else:
                state_s = 1.0 if state == 'standing' else 0.0

            angle_s = 1.0
            if should_fall[i] and state == 'fallen' and gt_state == 'fallen':
                gen_a, gt_a = det.get('fall_angle'), gt_det.get('fall_angle')
                if gen_a is not None and gt_a is not None:
                    dev = abs(float(gen_a) - float(gt_a))
                    angle_s = max(
                        self.ANGLE_SCORE_FLOOR,
                        1.0 - max(0.0, dev - self.ANGLE_TOL_DEG) / self.ANGLE_DROP_DEG,
                    )

            # Position score
            pos_dist = det.get('pos_dist', 0)
            max_shift = max(d['w'], d['h'])
            if state == 'missing':
                pos_s = 0.0
            elif pos_dist < max_shift * 0.8:
                pos_s = 1.0
            elif pos_dist < max_shift * 1.5:
                pos_s = 0.7
            elif pos_dist < max_shift * 2.5:
                pos_s = 0.4
            else:
                pos_s = 0.0

            if state_s == 1.0 and pos_s > 0.5:
                self._add_domino_to_mask(domino_mask, d, det, state)

            domino_scores.append(state_s * pos_s * angle_s)
            domino_weights.append(1.0)

            # Detail string: expect|state match_scores pos→matched_pos dist score
            e = 'F' if should_fall[i] else 'S'
            st = det.get('stand_score', -1)
            fl = det.get('fall_score', -1)
            if state == 'fallen':
                fp = det.get('fall_pos', (d['cx'], d['cy']))
                dx, dy = fp[0] - d['cx'], fp[1] - d['cy']
                a = det.get('fall_angle', -1)
                info = (f'{e}|fell stand_match={st:.3f} fall_match={fl:.3f} angle={a} '
                        f'orig=({d["cx"]},{d["cy"]}) detected=({fp[0]},{fp[1]}) '
                        f'offset=({dx:+d},{dy:+d}) dist={det.get("pos_dist",0):.0f} '
                        f'position_score={pos_s} state_score={state_s} final={state_s*pos_s:.2f}')
            elif state == 'standing':
                sp = det.get('stand_pos', (d['cx'], d['cy']))
                info = (f'{e}|stand stand_match={st:.3f} '
                        f'orig=({d["cx"]},{d["cy"]}) detected=({sp[0]},{sp[1]}) '
                        f'dist={det.get("pos_dist",0):.0f} '
                        f'position_score={pos_s} state_score={state_s} final={state_s*pos_s:.2f}')
            else:
                info = (f'{e}|missing stand_match={st:.3f} fall_match={fl:.3f} '
                        f'orig=({d["cx"]},{d["cy"]}) final=0.00')
            per_domino.append(f'd{i}|{info}')

        fall_scores = [s for s, f in zip(domino_scores, should_fall) if f]
        stand_scores = [s for s, f in zip(domino_scores, should_fall) if not f]
        if fall_scores and stand_scores:
            complete_sc = float(np.mean(fall_scores)) * (
                0.6 + 0.4 * float(np.mean(stand_scores))
            )
        elif domino_scores:
            complete_sc = float(np.average(domino_scores, weights=domino_weights))
        else:
            complete_sc = 0.0

        # Background: dilate foreground mask, rest is background
        kernel = np.ones((7, 17), np.uint8)
        domino_dilated = cv2.dilate(domino_mask, kernel, iterations=1)
        bg = cv2.bitwise_not(domino_dilated)
        bg_sc, bg_det = self._pixel_diff_score(gen_last, gt_last, bg)

        score = complete_sc * (0.5 + 0.5 * bg_sc)
        details = {
            'complete_score': round(complete_sc, 4),
            'bg_score': round(bg_sc, 4),
            'bg_changed_ratio': bg_det['ratio'],
            'per_domino': per_domino,
        }
        return round(score, 4), details

    def _chain_order_score(self, chain, fall_frames):
        """Check left-to-right fall order within a single chain.
        Returns (score, violations)."""
        fell = [(i, fall_frames[i]) for i in chain
                if i in fall_frames and fall_frames[i] is not None]
        if len(fell) < 2:
            return 0, 0
        times = [ff for _, ff in fell]
        violations = sum(1 for j in range(1, len(times)) if times[j] < times[j - 1] + 1)
        if violations == 0: sc = 1.0
        elif violations == 1: sc = 0.5
        elif violations == 2: sc = 0.2
        else: sc = max(0, 1.0 - violations / max(1, len(times) - 1))
        return sc, violations

    def _score_intermediate_motion_evidence(
        self, fall_indices, gen_detections, timeline,
    ):
        """Require a real in-progress collapse, without fixed GT milestones.

        The sparse image outputs are uniform simulation samples rather than one
        frame per fallen domino, so several dominos may advance in one observed
        frame.  A valid intermediate only needs to show that some target domino
        has started falling while the target chain has not yet reached its
        generated final configuration.  This definition works for both O-23's
        branching chain and O-24's linear chain.
        """
        observed_frames = sorted({
            fi
            for di in fall_indices
            for fi, _, _ in timeline.get(di, [])
        })
        intermediate_frames = observed_frames[1:-1]
        angle_tolerance = max(5.0, float(self.ANGLE_TOL_DEG))

        states_by_domino = {
            di: {fi: (state, det) for fi, state, det in timeline.get(di, [])}
            for di in fall_indices
        }
        candidates = []
        valid_frames = []
        for fi in intermediate_frames:
            started = []
            unfinished = []
            per_domino = []
            for di in fall_indices:
                state, det = states_by_domino.get(di, {}).get(fi, ('missing', {}))
                final_state, final_det = gen_detections[di]
                has_started = state == 'fallen'

                if state != final_state:
                    is_unfinished = True
                    angle_delta = None
                elif state == 'fallen':
                    cur_angle = det.get('fall_angle')
                    final_angle = final_det.get('fall_angle')
                    if cur_angle is None or final_angle is None:
                        is_unfinished = False
                        angle_delta = None
                    else:
                        angle_delta = abs(float(cur_angle) - float(final_angle))
                        is_unfinished = angle_delta > angle_tolerance
                else:
                    is_unfinished = False
                    angle_delta = None

                if has_started:
                    started.append(di)
                if is_unfinished:
                    unfinished.append(di)
                per_domino.append({
                    'domino': di,
                    'state': state,
                    'angle_delta_from_final': (
                        None if angle_delta is None else round(angle_delta, 2)
                    ),
                })

            is_in_progress = bool(started and unfinished)
            if is_in_progress:
                valid_frames.append(fi)
            candidates.append({
                'frame': fi,
                'started': started,
                'unfinished': unfinished,
                'is_in_progress': is_in_progress,
                'dominos': per_domino,
            })

        score = 1.0 if valid_frames else 0.0
        return score, {
            'score': score,
            'observed_frames': observed_frames,
            'intermediate_frames': intermediate_frames,
            'valid_frames': valid_frames,
            'angle_tolerance': angle_tolerance,
            'candidates': candidates,
        }

    def _score_process(self, dominos, should_fall, gen_detections, timeline,
                       complete_sc=0.0):
        """Process: order * complete_sc (from final_state) * gap_acc."""

        fall_indices = [i for i, f in enumerate(should_fall) if f]
        if not fall_indices:
            return 1.0, {'reason': 'nothing_to_fall'}

        stand_indices = [i for i, f in enumerate(should_fall) if not f]

        # Gap accuracy: should_stand that stayed standing
        if stand_indices:
            correct_stands = sum(1 for di in stand_indices if gen_detections[di][0] == 'standing')
            gap_acc = correct_stands / len(stand_indices)
        else:
            gap_acc = -1

        # Fall times for order check
        confirmed_fallen = [di for di in fall_indices if gen_detections[di][0] == 'fallen']
        fall_frames = {}
        for di in confirmed_fallen:
            fall_frames[di] = self._find_fall_frame(timeline[di])

        # Branch-aware order
        structure = self._detect_branches(dominos)
        trunk = structure['trunk']
        if structure['branches']:
            chains = [trunk + br for br in structure['branches']]
        else:
            chains = [trunk]
        chain_scores = []
        total_violations = 0
        for chain in chains:
            chain_fall = [i for i in chain if i in fall_indices]
            if len(chain_fall) < 2:
                chain_scores.append(0.0)
                continue
            sc, v = self._chain_order_score(chain_fall, fall_frames)
            chain_scores.append(sc)
            total_violations += v
        order_sc = float(np.mean(chain_scores)) if chain_scores else 0.0

        if gap_acc >= 0:
            base_score = order_sc * (0.2 + 0.4 * complete_sc + 0.4 * gap_acc)
        else:
            base_score = order_sc * (0.2 + 0.8 * complete_sc)

        motion_evidence, motion_details = self._score_intermediate_motion_evidence(
            fall_indices, gen_detections, timeline,
        )
        score = base_score * (0.2 + 0.8 * motion_evidence)

        # Fall info string
        fall_info = []
        for di in fall_indices:
            if di in fall_frames and fall_frames[di] is not None:
                fall_info.append(f'd{di}:f{fall_frames[di]}')
            else:
                fall_info.append(f'd{di}:{gen_detections[di][0]}')

        br_str = f'T={structure["trunk"]}'
        for bi, br in enumerate(structure['branches']):
            br_str += f' B{bi}={br}'

        details = {
            'order_score': round(order_sc, 4),
            'completion': round(complete_sc, 4),
            'gap_acc': round(gap_acc, 4),
            'violations': total_violations,
            'fall_times': ' '.join(fall_info),
            'branches': br_str,
            'base_score': round(base_score, 4),
            'motion_evidence': round(motion_evidence, 4),
            'motion_details': motion_details,
        }
        return round(score, 4), details

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
            video_frames = [normalize_frame_size(f, gt_first_frame)
                            for f in video_frames]

        import time

        t0 = time.time()
        dominos = self._extract_dominos(gt_first_frame)
        if not dominos:
            self._last_task_details = {'error': 'no dominos detected'}
            return 0.0

        # 2. Detect states on GT last and gen last (cache, avoid repeated calls)
        t1 = time.time()

        gt_detections = []  # (state, det) per domino on GT last
        gen_detections = []  # (state, det) per domino on gen last
        should_fall = []
        for d in dominos:
            gt_state, gt_det = self._detect_domino_state(gt_final_frame, d)
            gen_state, gen_det = self._detect_domino_state(video_frames[-1], d)
            gt_detections.append((gt_state, gt_det))
            gen_detections.append((gen_state, gen_det))
            should_fall.append(gt_state != 'standing')

        # 3. Build timeline only for confirmed-fallen dominos (saves time)
        t2 = time.time()
        confirmed_fallen = [i for i in range(len(dominos))
                            if should_fall[i] and gen_detections[i][0] == 'fallen']
        timeline, sample_frames = self._build_state_timeline(
            video_frames, dominos, confirmed_fallen)

        # 4. Score
        t3 = time.time()
        final_sc, f_det = self._score_final_state(
            video_frames[-1], gt_final_frame, dominos, should_fall,
            gen_detections, gt_detections)
        proc_sc, p_det = self._score_process(
            dominos, should_fall, gen_detections, timeline,
            complete_sc=f_det['complete_score'])

        scores = {'final_state': final_sc, 'process': proc_sc}

        # Debug output
        import os
        video_path = eval_info.get('video_path', '')
        parts = video_path.replace('\\', '/').split('/')
        vid_id = os.path.splitext(parts[-1])[0] if parts else 'unknown'
        model_name = parts[-4] if len(parts) >= 4 else 'unknown'

        self._last_task_details = {
            **scores,
            'final_complete_sc': f_det['complete_score'],
            'final_bg_sc': f_det['bg_score'],
            'final_bg_ratio': f_det['bg_changed_ratio'],
            'pro_order_sc': p_det['order_score'],
            'pro_completion': p_det['completion'],
            'pro_gap_acc': p_det['gap_acc'],
            'pro_motion_evidence': p_det['motion_evidence'],
            'pro_motion_details': p_det['motion_details'],
            'fall_order': p_det['fall_times'],
            'branches': p_det['branches'],
        }
        # Per-domino detail: each as separate key for HTML viewer
        for entry in f_det['per_domino']:
            # entry: "d0|F→stand .958 p=1.0"
            key, val = entry.split('|', 1)
            self._last_task_details[key] = val
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

class DominoChainGapEvaluator(DominoChainBranchEvaluator):
    """
    O-24: Domino Chain Gap Analysis

    Linear domino chain with a gap. Push first → dominos fall left-to-right.
    Chain stops at gap — dominos before gap fall, dominos after gap stay standing.

    Same template-matching detection as O-23, but linear chain (no Y-branch).
    1. Final state (50%): per-domino state correctness + background
    2. Process (50%): left-to-right fall order + gap accuracy
    """

    TASK_WEIGHTS = {
        'final_state': 0.50,
        'process': 0.50,
    }

    def _extract_dominos(self, frame, min_area=1000):
        """Override: remove ground line before detecting dominos.
        In O-24, dominos sit on a brown ground line and are connected to it.
        """
        corners = [frame[5, 5], frame[5, -5], frame[-5, 5], frame[-5, -5]]
        self._bg_color = tuple(int(v) for v in np.mean(corners, axis=0))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

        # Remove ground line: scan from bottom up, find where fg pixels
        # span >50% of width (ground line), zero out from there down
        h_img, w_img = frame.shape[:2]
        ground_top = h_img
        for row in range(h_img - 1, 0, -1):
            row_px = int((fg[row] > 0).sum())
            if row_px > w_img * 0.5:
                ground_top = row
            elif ground_top < h_img:
                break
        if ground_top < h_img:
            fg[ground_top:, :] = 0

        # Re-open to clean up
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

        # Now find domino contours
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img = frame.shape[0]
        dominos = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and area / hull_area < 0.75:
                continue

            pad = 2
            y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
            dominos.append({
                'cx': cx, 'cy': cy,
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area,
                'template': gray[y1:y2, x1:x2].copy(),
                'color_template': frame[y1:y2, x1:x2].copy(),
            })
        dominos.sort(key=lambda d: d['cx'])
        return dominos

    FALLEN_THRESH = 0.60  # Masked NCC gives high scores (0.66+) for correct matches

    @staticmethod
    def _masked_ncc(template, crop, thresh=230):
        """NCC only on domino body pixels (ignore white padding from rotation)."""
        gray_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        mask = gray_t < thresh
        if mask.sum() < 50:
            return -1.0
        t_vals = template[mask].astype(np.float64).flatten()
        c_vals = crop[mask].astype(np.float64).flatten()
        t_vals -= t_vals.mean()
        c_vals -= c_vals.mean()
        denom = np.sqrt((t_vals ** 2).sum() * (c_vals ** 2).sum())
        if denom < 1e-6:
            return -1.0
        return float((t_vals * c_vals).sum() / denom)

    def _chain_order_score(self, chain, fall_frames):
        """Override: O-24 dominos fall very quickly in succession,
        so allow same-frame falls (tolerance of 0 frames difference)."""
        fell = [(i, fall_frames[i]) for i in chain
                if i in fall_frames and fall_frames[i] is not None]
        if len(fell) < 2:
            return 0, 0
        times = [ff for _, ff in fell]
        # Allow same frame (>=) instead of requiring strict increase (+1)
        violations = sum(1 for j in range(1, len(times)) if times[j] < times[j - 1])
        if violations == 0: sc = 1.0
        elif violations == 1: sc = 0.5
        elif violations == 2: sc = 0.2
        else: sc = max(0, 1.0 - violations / max(1, len(times) - 1))
        return sc, violations

    def _check_fallen(self, frame, domino, scale_factor=1.2):
        """Override: bottom-center pivot rotation + masked NCC.
        O-24 dominos rotate around their bottom edge when falling.
        Masked NCC ignores white padding, only comparing domino body pixels,
        which handles overlap from adjacent fallen dominos."""
        tmpl = domino['color_template']
        th, tw = tmpl.shape[:2]
        cx, cy = domino['cx'], domino['cy']
        bot_y = cy + domino['h'] // 2  # bottom of domino

        best_score = -1.0
        best_pos = None
        best_angle = 0
        search_xy = 15

        for angle in range(15, 95, 5):
            pivot = (tw / 2, th - 1)  # bottom-center of template
            M = cv2.getRotationMatrix2D(pivot, -angle, 1.0)
            corners = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
            nc = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
            mn, mx = nc.min(axis=0), nc.max(axis=0)
            M[0, 2] -= mn[0]
            M[1, 2] -= mn[1]
            nw_r = int(mx[0] - mn[0]) + 1
            nh_r = int(mx[1] - mn[1]) + 1
            rot = cv2.warpAffine(tmpl, M, (nw_r, nh_r),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            rh, rw_t = rot.shape[:2]

            # Pivot position in rotated image
            piv_new = cv2.transform(
                np.array([[[float(tw) / 2, float(th) - 1]]], dtype=np.float32), M)[0][0]
            fx1_base = int(cx - piv_new[0])
            fy1_base = int(bot_y - piv_new[1])

            for dy in range(-search_xy, search_xy + 1, 5):
                for dx in range(-search_xy, search_xy + 1, 5):
                    ax1 = fx1_base + dx
                    ay1 = fy1_base + dy
                    ax2 = ax1 + rw_t
                    ay2 = ay1 + rh
                    if ax1 < 0 or ay1 < 0 or ax2 > frame.shape[1] or ay2 > frame.shape[0]:
                        continue
                    crop = frame[ay1:ay2, ax1:ax2]
                    if crop.shape != rot.shape:
                        continue
                    s = self._masked_ncc(rot, crop)
                    if s > best_score:
                        best_score = s
                        best_angle = angle
                        # Compute detected center position
                        ct = cv2.transform(
                            np.array([[[float(tw) / 2, float(th) / 2]]], dtype=np.float32), M)[0][0]
                        best_pos = (int(ax1 + ct[0]), int(ay1 + ct[1]))

        return best_score, best_pos, best_angle, 1.0

    def _score_process(self, dominos, should_fall, gen_detections, timeline,
                       complete_sc=0.0):
        """Process: linear left-to-right fall order, stops at gap."""
        fall_indices = [i for i, f in enumerate(should_fall) if f]
        if not fall_indices:
            return 1.0, {'reason': 'nothing_to_fall'}

        stand_indices = [i for i, f in enumerate(should_fall) if not f]

        # Gap accuracy: should_stand dominos that stayed standing
        if stand_indices:
            correct_stands = sum(1 for di in stand_indices
                                 if gen_detections[di][0] == 'standing')
            gap_acc = correct_stands / len(stand_indices)
        else:
            gap_acc = -1

        # Fall times for order check
        confirmed_fallen = [di for di in fall_indices
                            if gen_detections[di][0] == 'fallen']
        fall_frames = {}
        for di in confirmed_fallen:
            fall_frames[di] = self._find_fall_frame(timeline[di])

        # Linear chain order check (no branch detection)
        order_sc, violations = self._chain_order_score(fall_indices, fall_frames)

        if gap_acc >= 0:
            base_score = order_sc * (0.2 + 0.4 * complete_sc + 0.4 * gap_acc)
        else:
            base_score = order_sc * (0.2 + 0.8 * complete_sc)

        motion_evidence, motion_details = self._score_intermediate_motion_evidence(
            fall_indices, gen_detections, timeline,
        )
        score = base_score * (0.2 + 0.8 * motion_evidence)

        fall_info = []
        for di in fall_indices:
            if di in fall_frames and fall_frames[di] is not None:
                fall_info.append(f'd{di}:f{fall_frames[di]}')
            else:
                fall_info.append(f'd{di}:{gen_detections[di][0]}')

        details = {
            'order_score': round(order_sc, 4),
            'completion': round(complete_sc, 4),
            'gap_acc': round(gap_acc, 4),
            'violations': violations,
            'fall_times': ' '.join(fall_info),
            'base_score': round(base_score, 4),
            'motion_evidence': round(motion_evidence, 4),
            'motion_details': motion_details,
        }
        return round(score, 4), details

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
            video_frames = [normalize_frame_size(f, gt_first_frame)
                            for f in video_frames]

        import time

        t0 = time.time()
        dominos = self._extract_dominos(gt_first_frame)
        if not dominos:
            self._last_task_details = {'error': 'no dominos detected'}
            return 0.0

        t1 = time.time()
        gt_detections = []
        gen_detections = []
        should_fall = []
        for d in dominos:
            gt_state, gt_det = self._detect_domino_state(gt_final_frame, d)
            gen_state, gen_det = self._detect_domino_state(video_frames[-1], d)
            gt_detections.append((gt_state, gt_det))
            gen_detections.append((gen_state, gen_det))
            should_fall.append(gt_state != 'standing')

        t2 = time.time()
        confirmed_fallen = [i for i in range(len(dominos))
                            if should_fall[i] and gen_detections[i][0] == 'fallen']
        timeline, sample_frames = self._build_state_timeline(
            video_frames, dominos, confirmed_fallen)

        t3 = time.time()
        final_sc, f_det = self._score_final_state(
            video_frames[-1], gt_final_frame, dominos, should_fall,
            gen_detections, gt_detections)
        proc_sc, p_det = self._score_process(
            dominos, should_fall, gen_detections, timeline,
            complete_sc=f_det['complete_score'])

        scores = {'final_state': final_sc, 'process': proc_sc}

        # Debug output
        import os
        video_path = eval_info.get('video_path', '')
        parts = video_path.replace('\\', '/').split('/')
        vid_id = os.path.splitext(parts[-1])[0] if parts else 'unknown'
        model_name = parts[-4] if len(parts) >= 4 else 'unknown'

        self._last_task_details = {
            **scores,
            'final_complete_sc': f_det['complete_score'],
            'final_bg_sc': f_det['bg_score'],
            'final_bg_ratio': f_det['bg_changed_ratio'],
            'pro_order_sc': p_det['order_score'],
            'pro_completion': p_det['completion'],
            'pro_gap_acc': p_det['gap_acc'],
            'pro_motion_evidence': p_det['motion_evidence'],
            'pro_motion_details': p_det['motion_details'],
            'fall_order': p_det['fall_times'],
        }
        for entry in f_det['per_domino']:
            key, val = entry.split('|', 1)
            self._last_task_details[key] = val
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class LEGOConstructionEvaluator(BaseEvaluator):
    """
    O-25: LEGO Construction Assembly

    Task: Follow LEGO assembly instructions - assemble brick onto structure.
    First frame has instruction diagram + partial model, last frame has assembled result.

    Evaluation:
    1. Assembly correctness (60%): changed region (new brick + instruction removal) matches GT
    2. Structure preservation (20%): existing LEGO structure unchanged
    3. Background preservation (20%): background stays clean
    """

    TASK_WEIGHTS = {
        'assembly_correctness': 0.80,
        'consistency': 0.20
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

        # Changed region: instruction diagram disappears + new brick appears
        changed_mask = self._detect_changed_region(gt_first, gt_last)
        # Existing structure: foreground in GT last that didn't change
        fg_last = self._detect_fg_mask(gt_last)
        existing_mask = cv2.bitwise_and(fg_last, cv2.bitwise_not(changed_mask))
        # All foreground across first and last
        fg_first = self._detect_fg_mask(gt_first)
        all_fg = cv2.bitwise_or(fg_first, fg_last)

        kernel = np.ones((5, 5), np.uint8)
        changed_dilated = cv2.dilate(changed_mask, kernel, iterations=1)
        existing_eroded = cv2.erode(existing_mask, kernel, iterations=1)
        all_fg_dilated = cv2.dilate(all_fg, kernel, iterations=1)


        # 1. Assembly correctness (60%): gt_last vs gen_last in changed region
        changed_eroded = cv2.erode(changed_mask, kernel, iterations=1) if changed_mask.sum() > 0 else changed_mask
        assembly_score, assembly_details = self._pixel_diff_score(
            gt_last, gen_last, changed_eroded,
            thresholds=(0.15, 0.25, 0.35, 0.60))

        # 2. Structure preservation (20%): gen_first vs gen_last in existing structure
        struct_score, struct_details = self._pixel_diff_score(
            gt_last, gen_last, existing_eroded, thresholds=(0.15, 0.25, 0.35, 0.60))

        # 3. Background preservation (20%): gen_first vs gen_last outside all fg
        bg_mask = cv2.bitwise_not(all_fg_dilated)
        bg_score, bg_details = self._pixel_diff_score(
            gt_last, gen_last, bg_mask, thresholds=(0.005, 0.02, 0.05, 0.10))

        scores = {
            'assembly_correctness': round(assembly_score, 4),
            'consistency': round((struct_score + bg_score) / 2, 4),
        }
        self._last_task_details = {
            **scores,
            'structure_preservation': round(struct_score, 4),
            'background_preservation': round(bg_score, 4),
            **{f'asm_{k}': v for k, v in assembly_details.items()},
            **{f'struct_{k}': v for k, v in struct_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }
        _cons = scores['consistency']
        gated_assembly = scores['assembly_correctness'] * min(1.0, _cons / 0.5)
        return (self.TASK_WEIGHTS['assembly_correctness'] * gated_assembly
                + self.TASK_WEIGHTS['consistency'] * _cons)

class BallColorEvaluator(BaseEvaluator):
    """
    O-29: Ball Color (Cluster Merging)

    Task: Survivor cluster A moves toward other clusters and absorbs them one by one.
    All absorbed balls turn to A's color. Continue until only A remains.

    Evaluation:
    1. final_state (60%): last frame ball count correct (contour-based), text label check
    2. merge_process (40%): each merge step matches GT — correct color absorbed,
       correct ball counts before/after each merge (contour-based)
    """

    TASK_WEIGHTS = {
        'final_state': 0.60,
        'merge_process': 0.40,
    }

    MERGE_RATIO_THRESH = 0.15
    MERGE_STABLE_FRAMES = 2
    PROCESS_GATE_FLOOR = 0.20

    def __init__(self, device: str = 'cpu', task_name: str = ''):
        super().__init__(device, task_name)
        self._easyocr_reader = None

    def _combine_final_and_process(self, final_state: float,
                                   merge_process: float) -> float:
        """Combine O-29 scores while keeping process evidence mandatory.

        The old additive-only composition gave a perfect final frame 0.60 even
        when no merge was shown.  Keep the documented 60/40 balance, then gate
        it by process quality.  The small floor preserves limited credit for a
        correct final state without letting it pass as a complete multi-frame
        solution.  This helper is shared by video and interleave so aggregation
        stays aligned across settings.
        """
        final_state = float(max(0.0, min(1.0, final_state)))
        merge_process = float(max(0.0, min(1.0, merge_process)))
        weighted = (
            self.TASK_WEIGHTS['final_state'] * final_state
            + self.TASK_WEIGHTS['merge_process'] * merge_process
        )
        process_gate = self.PROCESS_GATE_FLOOR + (
            (1.0 - self.PROCESS_GATE_FLOOR) * merge_process
        )
        return float(weighted * process_gate)

    def _get_easyocr_reader(self):
        if self._easyocr_reader is None:
            import os, easyocr, torch
            self._easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), model_storage_directory=(os.environ.get('VBVR_EASYOCR_MODELS') or None))
        return self._easyocr_reader

    def _ocr_read_labels(self, frame: np.ndarray) -> Dict[str, str]:
        """Read text labels from frame. Returns dict of detected texts."""
        import re
        try:
            reader = self._get_easyocr_reader()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = reader.readtext(rgb)
            texts = {}
            for bbox, text, conf in results:
                if conf < 0.3:
                    continue
                text = text.strip()
                texts[text] = conf
            return texts
        except Exception:
            return {}

    def _ocr_extract_numbers(self, frame: np.ndarray) -> Dict[str, int]:
        """Extract label:number pairs like 'A:8', 'TOTAL=8' from frame."""
        import re
        texts = self._ocr_read_labels(frame)
        result = {}
        for text in texts:
            # Match patterns like "A:4", "B:3", "TOTAL=8", "TOTAL:8"
            for m in re.finditer(r'([A-Z]+)\s*[:=]\s*(\d+)', text, re.IGNORECASE):
                result[m.group(1).upper()] = int(m.group(2))
        return result

    def _get_ball_mask(self, frame: np.ndarray, sat_thresh: int = 80, val_thresh: int = 80) -> np.ndarray:
        """Get binary mask of all colored ball pixels (exclude gray bg and black text)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] > sat_thresh) & (hsv[:, :, 2] > val_thresh)).astype(np.uint8) * 255
        kernel3 = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)

    def _detect_color_from_mask(self, frame: np.ndarray, mask: np.ndarray) -> Optional[dict]:
        """Detect one color from masked ball pixels. Returns hue range dict or None."""
        if mask.sum() == 0:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_pixels = hsv[:, :, 0][mask > 0]
        if len(hue_pixels) == 0:
            return None

        # Circular mean for hue
        angles = hue_pixels.astype(float) * np.pi / 90.0
        mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        center = int((mean_angle * 90.0 / np.pi) % 180)

        # Determine hue spread from actual pixel distribution
        # Shift hues so center is at 90 to avoid wrapping issues
        shift = (90 - center) % 180
        shifted = (hue_pixels.astype(int) + shift) % 180
        low_shifted = int(np.percentile(shifted, 5))
        high_shifted = int(np.percentile(shifted, 95))
        # Tight range: based on actual spread + small padding
        half_w = max(5, (high_shifted - low_shifted) // 2 + 3)

        hue_low = (center - half_w) % 180
        hue_high = (center + half_w) % 180
        wrap = hue_low > hue_high  # wraps around 0/180

        return {
            'hue_center': center,
            'hue_low': hue_low,
            'hue_high': hue_high,
            'wrap': wrap,
        }

    def _mask_out_color(self, frame: np.ndarray, color: dict,
                         sat_thresh: int = 80, val_thresh: int = 80) -> np.ndarray:
        """Create mask that EXCLUDES pixels of the given color."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if color['wrap']:
            hue_match = (hsv[:, :, 0] >= color['hue_low']) | (hsv[:, :, 0] <= color['hue_high'])
        else:
            hue_match = (hsv[:, :, 0] >= color['hue_low']) & (hsv[:, :, 0] <= color['hue_high'])
        sat_val = (hsv[:, :, 1] > sat_thresh) & (hsv[:, :, 2] > val_thresh)
        color_mask = (hue_match & sat_val).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.dilate(color_mask, kernel, iterations=1)
        return cv2.bitwise_not(color_mask)

    def _count_balls_by_contour(self, frame: np.ndarray, mask: np.ndarray) -> Tuple[int, float]:
        """Count individual balls in masked region. Returns (n_balls, median_area)."""
        masked = cv2.bitwise_and(mask, mask)
        contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri ** 2)
            if circ < 0.5:
                continue
            areas.append(area)
        n = len(areas)
        med_area = float(np.median(areas)) if areas else 0.0
        return n, med_area

    def _detect_ball_info_from_gt(self, gt_frames: List[np.ndarray]) -> Tuple[List[dict], float]:
        """Detect ball colors by scanning GT video from LAST frame to FIRST.
        """
        if not gt_frames:
            return [], 0.0

        gt_first = gt_frames[0]
        gt_last = gt_frames[-1]

        # Step 1: last frame has only the survivor color (A)
        survivor_mask = self._get_ball_mask(gt_last)
        survivor_info = self._detect_color_from_mask(gt_last, survivor_mask)
        if survivor_info is None:
            return [], 0.0
        survivor_info['is_survivor'] = True
        survivor_info['name'] = f'survivor_{survivor_info["hue_center"]}'

        known_colors = [survivor_info]

        # Step 2: scan backwards, mask out known colors, detect new ones
        min_new_pixels = 100
        for t in range(len(gt_frames) - 2, -1, -1):
            frame = gt_frames[t]
            all_ball = self._get_ball_mask(frame)
            remaining = all_ball.copy()
            for kc in known_colors:
                exclude = self._mask_out_color(frame, kc)
                remaining = cv2.bitwise_and(remaining, exclude)

            if remaining.sum() / 255 < min_new_pixels:
                continue

            new_info = self._detect_color_from_mask(frame, remaining)
            if new_info is None:
                continue
            new_info['is_survivor'] = False
            new_info['name'] = f'color_{new_info["hue_center"]}'
            known_colors.append(new_info)

        # Step 3: count balls per color and avg_ball_area from first frame
        all_areas = []
        for color in known_colors:
            ball_mask = self._get_ball_mask(gt_first)
            # Isolate this color's pixels
            hsv = cv2.cvtColor(gt_first, cv2.COLOR_BGR2HSV)
            if color['wrap']:
                hue_match = (hsv[:, :, 0] >= color['hue_low']) | (hsv[:, :, 0] <= color['hue_high'])
            else:
                hue_match = (hsv[:, :, 0] >= color['hue_low']) & (hsv[:, :, 0] <= color['hue_high'])
            color_ball_mask = cv2.bitwise_and(
                ball_mask, (hue_match.astype(np.uint8) * 255))
            kernel3 = np.ones((3, 3), np.uint8)
            color_ball_mask = cv2.morphologyEx(color_ball_mask, cv2.MORPH_OPEN, kernel3)

            n_balls, med_area = self._count_balls_by_contour(gt_first, color_ball_mask)
            color['init_balls'] = n_balls
            color['init_mask'] = color_ball_mask
            if med_area > 0:
                all_areas.append(med_area)

        avg_ball_area = float(np.median(all_areas)) if all_areas else 0.0

        return known_colors, avg_ball_area


    def _make_color_mask(self, hsv: np.ndarray, color: dict,
                          sat_thresh: int = 70, val_thresh: int = 70) -> np.ndarray:
        sat_val = (hsv[:, :, 1] > sat_thresh) & (hsv[:, :, 2] > val_thresh)
        hue_low = color['hue_low']
        hue_high = color['hue_high']
        wrap = hue_low > hue_high
        if wrap:
            hue_mask = (hsv[:, :, 0] >= hue_low) | (hsv[:, :, 0] <= hue_high)
        else:
            hue_mask = (hsv[:, :, 0] >= hue_low) & (hsv[:, :, 0] <= hue_high)
        mask = (hue_mask & sat_val).astype(np.uint8) * 255
        kernel = np.ones((7, 7), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def _detect_occlusion_frames(self, frames: List[np.ndarray], colors: List[dict],
                                  is_gen: bool = False, min_overlap: int = 50) -> Dict[str, set]:
        """For each non-survivor color, find frames where its mask overlaps with survivor mask.
        Returns {color_name: set of occluded frame indices}.
        Also returns survivor occluded frames (where survivor overlaps with any non-merged color).
        """
        survivor_c = next(c for c in colors if c['is_survivor'])
        non_surv = [c for c in colors if not c['is_survivor']]
        kernel = np.ones((7, 7), np.uint8)
        sat_t, val_t = (50, 50) if is_gen else (70, 70)

        result = {c['name']: set() for c in non_surv}
        result['__survivor__'] = set()

        for t, frame in enumerate(frames):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            surv_mask = self._make_color_mask(hsv, survivor_c, sat_t, val_t)
            surv_dilated = cv2.dilate(surv_mask, kernel, iterations=2)

            for c in non_surv:
                c_mask = self._make_color_mask(hsv, c, sat_t, val_t)
                c_dilated = cv2.dilate(c_mask, kernel, iterations=2)
                overlap = cv2.bitwise_and(surv_dilated, c_dilated)
                if overlap.sum() / 255 > min_overlap:
                    result[c['name']].add(t)
                    result['__survivor__'].add(t)

        return result

    def _count_pixels_per_color(self, frame: np.ndarray, colors: List[dict],
                                 is_gen: bool = False) -> Dict[str, int]:
        """Pixel count per color in one frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_t = 50 if is_gen else 70
        val_t = 50 if is_gen else 70
        counts = {}
        for c in colors:
            mask = self._make_color_mask(hsv, c, sat_t, val_t)
            counts[c['name']] = int(mask.sum() / 255)
        return counts

    def _build_pixel_sequences(self, frames: List[np.ndarray], colors: List[dict],
                                is_gen: bool = False) -> Dict[str, List[int]]:
        """Per-color pixel count for every frame. Returns {name: [px_count_per_frame]}."""
        seqs = {c['name']: [] for c in colors}
        for frame in frames:
            counts = self._count_pixels_per_color(frame, colors, is_gen)
            for name, px in counts.items():
                seqs[name].append(px)
        return seqs

    def _compute_ratio_curves(self, pixel_seqs: Dict[str, List[int]],
                              gt_pixel_seqs: Dict[str, List[int]] = None) -> Dict[str, List[float]]:
        """Convert pixel counts to ratio relative to GT first frame."""
        ratios = {}
        for name, seq in pixel_seqs.items():
            if gt_pixel_seqs and name in gt_pixel_seqs:
                init = gt_pixel_seqs[name][0] if gt_pixel_seqs[name] else 0
            else:
                init = seq[0] if seq else 0
            if init > 0:
                ratios[name] = [px / init for px in seq]
            else:
                ratios[name] = [0.0] * len(seq)
        return ratios

    def _detect_merge_events(self, ratio_curves: Dict[str, List[float]],
                              other_names: List[str],
                              occlusion_frames: Dict[str, set] = None) -> Dict[str, Optional[float]]:
        """For each non-survivor color, find normalized time when merge permanently completes.
        Scan from end backwards: find last frame with ratio >= threshold, merge_frame = next frame.
        Occlusion frames are excluded from transition counting.
        """
        events = {}
        for name in other_names:
            curve = ratio_curves.get(name, [])
            n = len(curve)
            if n == 0:
                events[name] = None
                continue

            occluded = occlusion_frames.get(name, set()) if occlusion_frames else set()

            if curve[0] < self.MERGE_RATIO_THRESH:
                events[name] = None
                continue
            if curve[-1] >= self.MERGE_RATIO_THRESH:
                events[name] = None
                continue

            if n >= 3:
                smooth = [curve[0]] + [
                    (curve[t-1] + curve[t] + curve[t+1]) / 3 for t in range(1, n-1)
                ] + [curve[-1]]
            else:
                smooth = list(curve)

            # Count transitions, skipping occluded frames
            transitions = 0
            was_present = True
            for t in range(1, n):
                if t in occluded:
                    continue  # skip occluded frames
                is_present = smooth[t] >= self.MERGE_RATIO_THRESH
                if was_present and not is_present:
                    transitions += 1
                elif not was_present and is_present:
                    transitions += 1
                was_present = is_present

            if transitions > 2:
                events[name] = None
                continue

            for t in range(n - 1, -1, -1):
                if curve[t] >= self.MERGE_RATIO_THRESH:
                    events[name] = (t + 1) / max(n - 1, 1)
                    break

        return events

    def _count_balls_of_color(self, frame: np.ndarray, color: dict,
                               avg_ball_area: float = 0,
                               is_gen: bool = False) -> int:
        """Count individual balls of one color using contour detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_t, val_t = (50, 50) if is_gen else (70, 70)
        mask = self._make_color_mask(hsv, color, sat_t, val_t)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri ** 2)
            if circ < 0.5:
                continue
            if avg_ball_area > 0:
                ratio = area / avg_ball_area
                if ratio > 3:
                    continue  # way too big, not a valid ball
                else:
                    count += 1
            else:
                count += 1
        return count

    def _count_all_colors(self, frame: np.ndarray, colors: List[dict],
                           avg_ball_area: float = 0,
                           is_gen: bool = False) -> Dict[str, int]:
        """Count balls per color in one frame via contour detection."""
        return {c['name']: self._count_balls_of_color(frame, c, avg_ball_area, is_gen) for c in colors}


    def _score_final_state(self, gen_last: np.ndarray, gt_final: np.ndarray,
                            colors: List[dict],
                            gt_total: int, avg_ball_area: float = 0) -> Tuple[float, Dict]:
        gen_counts = self._count_all_colors(gen_last, colors, avg_ball_area, is_gen=True)

        survivor_c = next(c for c in colors if c['is_survivor'])
        gen_survivor_n = gen_counts[survivor_c['name']]

        # Survivor count match
        diff = abs(gen_survivor_n - gt_total)
        if diff == 0:
            survivor_score = 1.0
        elif diff == 1:
            survivor_score = 0.7
        elif diff == 2:
            survivor_score = 0.4
        else:
            survivor_score = max(0, 0.2 - (diff - 2) * 0.05)

        # Other colors should be 0
        other_scores = []
        for c in colors:
            if c['is_survivor']:
                continue
            n = gen_counts[c['name']]
            if n == 0:
                other_scores.append(1.0)
            elif n == 1:
                other_scores.append(0.5)
            elif n == 2:
                other_scores.append(0.2)
            else:
                other_scores.append(0.0)
        others_gone = float(np.mean(other_scores)) if other_scores else 1.0

        ball_mask = self._get_ball_mask(gen_last, sat_thresh=50, val_thresh=50)
        hsv_last = cv2.cvtColor(gen_last, cv2.COLOR_BGR2HSV)
        known_mask = np.zeros(gen_last.shape[:2], dtype=np.uint8)
        for c in colors:
            known_mask = cv2.bitwise_or(known_mask, self._make_color_mask(hsv_last, c, sat_thresh=50, val_thresh=50))
        unknown_pixels = int(cv2.bitwise_and(ball_mask, cv2.bitwise_not(known_mask)).sum() / 255)
        total_ball_pixels = int(ball_mask.sum() / 255)
        if total_ball_pixels > 0:
            unknown_ratio = max(0, unknown_pixels / total_ball_pixels - 0.30)  # 15% tolerance for edge pixels
            clean_score = max(0, 1.0 - unknown_ratio * 2.5)
        else:
            unknown_ratio = -1
            clean_score = 1.0

        # Check dispersion: survivor balls should form a cluster in final frame
        survivor_c = next(c for c in colors if c['is_survivor'])
        gt_disp = self._get_color_dispersion(gt_final, survivor_c, is_gen=False) if gt_final is not None else (0, 0)
        gen_disp = self._get_color_dispersion(gen_last, survivor_c, is_gen=True)
        disp_ratio = -1.0
        if gt_disp[0] > 0 and gen_disp[0] > 0:
            disp_ratio = max(gen_disp[0] / (gt_disp[0] + 1), gen_disp[1] / (gt_disp[1] + 1))
            if disp_ratio < 2:
                cluster_score = 1.0
            elif disp_ratio < 3:
                cluster_score = 0.5
            else:
                cluster_score = 0.0
        else:
            cluster_score = 1.0

        # OCR text label check (penalty)
        gt_labels = self._ocr_extract_numbers(gt_final) if gt_final is not None else {}
        gen_labels = self._ocr_extract_numbers(gen_last)
        if gt_labels:
            match_count = sum(1 for k in gt_labels if gen_labels.get(k) == gt_labels[k])
            text_score = match_count / len(gt_labels)
        else:
            text_score = 1.0

        penalty = others_gone * 0.5 + clean_score * 0.5
        score = (0.5 * survivor_score + 0.5 * cluster_score) * (0.6 + 0.4 * penalty) * (0.8 + 0.2 * text_score)

        details = {
            'gt_total': gt_total,
            'gen_survivor_count': gen_survivor_n,
            'survivor_score': round(survivor_score, 4),
            'others_gone': round(others_gone, 4),
            'unknown_ratio': round(unknown_ratio, 4),
            'clean_score': round(clean_score, 4),
            'gt_disp': f'{gt_disp[0]:.1f},{gt_disp[1]:.1f}',
            'gen_disp': f'{gen_disp[0]:.1f},{gen_disp[1]:.1f}',
            'disp_ratio': round(disp_ratio, 2) if 'disp_ratio' in dir() else -1,
            'cluster_score': round(cluster_score, 4),
            'text_score': round(text_score, 4),
            'gt_labels': str(gt_labels),
            'gen_labels': str(gen_labels),
            'gen_final_counts': ' '.join(f'{k}:{v}' for k, v in gen_counts.items() if v > 0),
        }
        return round(score, 4), details

    def _get_color_centroid(self, frame: np.ndarray, color: dict,
                             is_gen: bool = False) -> Optional[Tuple[float, float]]:
        """Get centroid (x, y) of a color's pixels in a frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_t, val_t = (50, 50) if is_gen else (70, 70)
        mask = self._make_color_mask(hsv, color, sat_t, val_t)
        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            return None
        return float(np.mean(xs)), float(np.mean(ys))

    def _get_color_dispersion(self, frame: np.ndarray, color: dict,
                               is_gen: bool = False) -> Tuple[float, float]:
        """Get std of a color's pixel positions (spread of cluster)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_t, val_t = (50, 50) if is_gen else (70, 70)
        mask = self._make_color_mask(hsv, color, sat_t, val_t)
        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            return 0.0, 0.0
        return float(np.std(xs)), float(np.std(ys))

    def _score_merge_process(self, gt_frames: List[np.ndarray],
                              video_frames: List[np.ndarray],
                              colors: List[dict],
                              gt_merge_events: Dict[str, Optional[float]],
                              gen_merge_events: Dict[str, Optional[float]],
                              other_names: List[str],
                              avg_ball_area: float = 0,
                              occlusion_frames: Dict[str, set] = None) -> Tuple[float, Dict]:
        """Per-merge evaluation: each GT merge scored on 5 checks, then averaged.
        """
        if not other_names:
            return 1.0, {'n_merges': 0}

        if len(video_frames) < 3:
            return 0.0, {'n_merges': len(other_names),
                         'error': 'no_intermediate_frames'}

        gt_merges = sorted(
            [(name, t) for name, t in gt_merge_events.items() if t is not None],
            key=lambda x: x[1]
        )
        gen_merges = sorted(
            [(name, t) for name, t in gen_merge_events.items() if t is not None],
            key=lambda x: x[1]
        )
        gt_merge_names = [n for n, _ in gt_merges]
        gen_merge_names = [n for n, _ in gen_merges]
        n_gt_merges = len(gt_merges)

        if n_gt_merges == 0:
            return 1.0, {'n_merges': 0}

        survivor_c = next(c for c in colors if c['is_survivor'])
        color_map = {c['name']: c for c in colors}
        gt_first = gt_frames[0]

        # Record initial centroids and dispersions from GT first frame
        init_centroids = {}
        init_dispersions = {}
        for c in colors:
            cent = self._get_color_centroid(gt_first, c)
            if cent:
                init_centroids[c['name']] = cent
            disp = self._get_color_dispersion(gt_first, c)
            init_dispersions[c['name']] = disp

        # Build gen order lookup: {color_name: position_in_gen_order}
        gen_order_map = {n: i for i, n in enumerate(gen_merge_names)}

        per_merge_scores = []
        per_merge_details = []

        # --- Pre-compute arrival in gen time order ---
        arrival_data = {}  # color_name -> score
        a_prev_pos = init_centroids.get(survivor_c['name'])
        for gp, (gname, g_t) in enumerate(gen_merges):
            target_pos = init_centroids.get(gname)
            arrival_s = 0.5
            gen_merge_idx = int(g_t * (len(video_frames) - 1))
            gen_a_pos = self._get_color_centroid(video_frames[gen_merge_idx], survivor_c, is_gen=True)
            if target_pos and gen_a_pos:
                final_dist = np.sqrt((gen_a_pos[0] - target_pos[0])**2 +
                                      (gen_a_pos[1] - target_pos[1])**2)
                if a_prev_pos:
                    init_dist = np.sqrt((a_prev_pos[0] - target_pos[0])**2 +
                                        (a_prev_pos[1] - target_pos[1])**2)
                else:
                    init_dist = 0
                if init_dist > 0:
                    r = final_dist / init_dist
                    arrival_s = 1.0 if r < 0.15 else (0.7 if r < 0.3 else (0.4 if r < 0.5 else 0.0))
                else:
                    arrival_s = 1.0 if final_dist < 30 else 0.0
                gen_after_pos = self._get_color_centroid(
                    video_frames[min(gen_merge_idx + 2, len(video_frames) - 1)], survivor_c, is_gen=True)
                a_prev_pos = gen_after_pos if gen_after_pos else gen_a_pos
            elif not gen_a_pos:
                arrival_s = 0.0
            arrival_data[gname] = arrival_s

        # --- Pre-compute merged_before in gen time order ---
        gen_merged_before = {}  # color_name -> set of names merged before it in gen time
        for gp, (gname, g_t) in enumerate(gen_merges):
            gen_merged_before[gname] = {gen_merges[j][0] for j in range(gp)}

        # --- Main scoring loop (GT order) ---
        for i, (name, gt_t) in enumerate(gt_merges):
            gen_t = gen_merge_events.get(name)

            # --- Check 1: did this color disappear? ---
            if gen_t is None:
                per_merge_scores.append(0.0)
                per_merge_details.append({'missing': True})
                continue

            # --- Check 2: order correct? ---
            gen_pos = gen_order_map.get(name, -1)
            pos_diff = abs(gen_pos - i)
            if pos_diff == 0:
                order_s = 1.0
            elif pos_diff >= 1:
                swapped_with = gt_merge_names[gen_pos] if 0 <= gen_pos < len(gt_merge_names) else None
                if swapped_with and color_map[swapped_with]['init_balls'] == color_map[name]['init_balls']:
                    order_s = 1.0
                elif pos_diff == 1:
                    order_s = 0.5
                else:
                    order_s = 0.0

            # --- Check 3: arrival (from pre-computed, gen time order) ---
            arrival_s = arrival_data.get(name, 0.5)

            # --- Check 4: survivor count correct after merge? ---
            gt_after_idx = min(int(gt_t * (len(gt_frames) - 1)) + 2, len(gt_frames) - 1)
            gen_after_idx = min(int(gen_t * (len(video_frames) - 1)) + 2, len(video_frames) - 1)
            gt_surv_n = self._count_balls_of_color(gt_frames[gt_after_idx], survivor_c, avg_ball_area)
            gen_surv_n = self._count_balls_of_color(video_frames[gen_after_idx], survivor_c, avg_ball_area, is_gen=True)
            d = abs(gt_surv_n - gen_surv_n)
            count_s = {0: 1.0, 1: 0.7, 2: 0.4}.get(d, max(0, 0.2 - (d - 2) * 0.05))

            # --- Check 5: other clusters stayed still during this merge? ---
            # Use gen time order for interval and merged_so_far
            prev_gen_t = gen_merges[gen_pos - 1][1] if gen_pos > 0 else 0.0
            start_idx = int(prev_gen_t * (len(video_frames) - 1)) + 2
            end_idx = int(gen_t * (len(video_frames) - 1)) - 2
            if end_idx > start_idx:
                sample_indices = [int(start_idx + k * (end_idx - start_idx) / 4) for k in range(5)]
            else:
                sample_indices = []

            merged_so_far = gen_merged_before.get(name, set())

            still_scores = []
            img_w = video_frames[0].shape[1]
            for idx in sample_indices:
                idx = min(idx, len(video_frames) - 1)
                for c in colors:
                    if c['is_survivor'] or c['name'] in merged_so_far:
                        continue
                    if occlusion_frames and idx in occlusion_frames.get(c['name'], set()):
                        continue
                    init_cent = init_centroids.get(c['name'])
                    init_disp = init_dispersions.get(c['name'], (0, 0))
                    if not init_cent:
                        continue
                    cent = self._get_color_centroid(video_frames[idx], c, is_gen=True)
                    if cent:
                        shift = np.sqrt((cent[0] - init_cent[0])**2 + (cent[1] - init_cent[1])**2)
                        shift_ratio = shift / img_w
                        if shift_ratio < 0.03:
                            still_scores.append(1.0)
                        elif shift_ratio < 0.08:
                            still_scores.append(0.5)
                        else:
                            still_scores.append(0.0)
                    cur_n = self._count_balls_of_color(video_frames[idx], c, avg_ball_area, is_gen=True)
                    init_n = c.get('init_balls', 0)
                    if init_n > 0:
                        cnt_d = abs(cur_n - init_n)
                        if cnt_d == 0:
                            still_scores.append(1.0)
                        elif cnt_d == 1:
                            still_scores.append(0.5)
                        else:
                            still_scores.append(0.0)
                    disp = self._get_color_dispersion(video_frames[idx], c, is_gen=True)
                    if disp[0] > 0 and init_disp[0] > 0:
                        disp_r = max(disp[0] / (init_disp[0] + 1), disp[1] / (init_disp[1] + 1))
                        if disp_r < 1.5:
                            still_scores.append(1.0)
                        elif disp_r < 2.5:
                            still_scores.append(0.5)
                        else:
                            still_scores.append(0.0)

            still_s = float(np.mean(still_scores)) if still_scores else 0.0

            # --- Combine 5 checks for this merge ---
            merge_score = order_s * 0.2 + arrival_s * 0.3 + count_s * 0.30 + still_s * 0.2
            per_merge_scores.append(merge_score)
            per_merge_details.append({
                'order': order_s, 'arrival': arrival_s,
                'count': count_s, 'gt_n': gt_surv_n, 'gen_n': gen_surv_n,
                'still': still_s,
            })

        # === Survivor count stability (separate check) ===
        gen_merge_times = [0.0] + [(gen_merge_events.get(n) or 1.0) for n, _ in gt_merges] + [1.0]

        surv_stab_scores = []
        for seg in range(len(gen_merge_times) - 1):
            t0 = gen_merge_times[seg]
            t1 = gen_merge_times[seg + 1]
            s_idx = int(t0 * (len(video_frames) - 1)) + 2
            e_idx = int(t1 * (len(video_frames) - 1)) - 2
            if e_idx <= s_idx:
                continue
            # Find first non-occluded frame for reference
            surv_occluded = occlusion_frames.get('__survivor__', set()) if occlusion_frames else set()
            ref_idx = s_idx
            while ref_idx in surv_occluded and ref_idx < e_idx:
                ref_idx += 1
            if ref_idx >= e_idx:
                continue
            ref_n = self._count_balls_of_color(video_frames[ref_idx], survivor_c, avg_ball_area, is_gen=True)
            if ref_n == 0:
                surv_stab_scores.append(0.0)
                continue
            # Sample 3 frames in the interval, skip occluded frames
            for k in range(1, 4):
                check_idx = min(int(s_idx + k * (e_idx - s_idx) / 4), len(video_frames) - 1)
                if check_idx in surv_occluded:
                    continue  # skip occluded frame
                check_n = self._count_balls_of_color(video_frames[check_idx], survivor_c, avg_ball_area, is_gen=True)
                d = abs(check_n - ref_n)
                if d == 0:
                    surv_stab_scores.append(1.0)
                elif d == 1:
                    surv_stab_scores.append(0.7)
                else:
                    surv_stab_scores.append(max(0, 0.3 - (d - 2) * 0.1))

        surv_stability = float(np.mean(surv_stab_scores)) if surv_stab_scores else 0.0

        # === Final score ===
        merge_avg = float(np.mean(per_merge_scores)) if per_merge_scores else 0.0
        score = merge_avg * 0.7 + surv_stability * 0.3

        details = {'gt_merges': n_gt_merges, 'gen_merges': len(gen_merges),
                    'surv_stability': round(surv_stability, 4)}
        for i, (name, gt_t) in enumerate(gt_merges):
            p = f'm{i+1}'
            md = per_merge_details[i]
            if md.get('missing'):
                details[p] = f'{name} missing'
            else:
                details[p] = f'{name} {round(per_merge_scores[i], 2)}'
                details[f'{p}_order'] = md['order']
                details[f'{p}_arrival'] = md['arrival']
                details[f'{p}_count'] = f"gt{md['gt_n']}/gen{md['gen_n']}={md['count']}"
                details[f'{p}_still'] = round(md['still'], 2)
        return round(score, 4), details


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

        if not gt_frames or len(gt_frames) < 2:
            gt_frames = [gt_first_frame, gt_final_frame]

        if video_frames[0].shape != gt_first_frame.shape:
            video_frames = [normalize_frame_size(f, gt_first_frame) for f in video_frames]

        # ---- 1. Detect colors & ball info from GT ----
        colors, avg_ball_area = self._detect_ball_info_from_gt(gt_frames)
        if not colors or avg_ball_area <= 0:
            self._last_task_details = {'error': 'no ball colors detected from GT'}
            return 0.0

        other_names = [c['name'] for c in colors if not c['is_survivor']]
        gt_total = sum(c['init_balls'] for c in colors)

        # ---- 2. Detect merge events via ratio curves ----
        gt_pixel_seqs = self._build_pixel_sequences(gt_frames, colors, is_gen=False)
        gen_pixel_seqs = self._build_pixel_sequences(video_frames, colors, is_gen=True)

        gt_ratio_curves = self._compute_ratio_curves(gt_pixel_seqs)
        gen_ratio_curves = self._compute_ratio_curves(gen_pixel_seqs, gt_pixel_seqs)

        # Detect occlusion frames for gen video
        gen_occlusion = self._detect_occlusion_frames(video_frames, colors, is_gen=True)

        gt_merge_events = self._detect_merge_events(gt_ratio_curves, other_names)
        gen_merge_events = self._detect_merge_events(gen_ratio_curves, other_names, gen_occlusion)

        # ---- 3. Score: final state (contour-based) ----
        final_score, final_details = self._score_final_state(
            video_frames[-1], gt_final_frame, colors, gt_total, avg_ball_area)

        # ---- 4. Score: merge process (contour-based) ----
        merge_score, merge_details = self._score_merge_process(
            gt_frames, video_frames, colors,
            gt_merge_events, gen_merge_events, other_names, avg_ball_area,
            occlusion_frames=gen_occlusion)

        scores = {
            'final_state': final_score,
            'merge_process': merge_score,
        }


        # Summarize occlusion info for detail
        occ_summary = {}
        for key, frames_set in gen_occlusion.items():
            if not frames_set:
                continue
            sorted_frames = sorted(frames_set)
            # Compress to ranges: [1,2,3,7,8] -> "1-3,7-8"
            ranges = []
            start = sorted_frames[0]
            end = start
            for f in sorted_frames[1:]:
                if f == end + 1:
                    end = f
                else:
                    ranges.append(f'{start}-{end}' if start != end else str(start))
                    start = end = f
            ranges.append(f'{start}-{end}' if start != end else str(start))
            occ_summary[key] = ','.join(ranges)

        self._last_task_details = {
            **scores,
            'gt_colors': str({c['name']: c['init_balls'] for c in colors}),
            'gt_total': gt_total,
            **{f'final_{k}': v for k, v in final_details.items()},
            **{f'merge_{k}': v for k, v in merge_details.items()},
            'occlusion': str(occ_summary) if occ_summary else 'none',
        }
        total = self._combine_final_and_process(
            scores['final_state'], scores['merge_process'],
        )
        self._last_task_details['process_gate_floor'] = self.PROCESS_GATE_FLOOR
        self._last_task_details['final_score'] = round(total, 4)
        return total

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Evaluate interleaved images with the video task logic.

        An interleave output is a sequence of still images.  Give every image a
        fixed dwell time so the video's +/-2-frame merge checks and stability
        windows remain meaningful, then reuse the video evaluator unchanged.
        Existing duplicate images are deliberately preserved: waiting at a
        state must not be mistaken for an additional merge event.
        """
        if not pred_images or input_frame is None or gt_final_frame is None:
            self._last_task_details = {"error": "no_input_or_pred"}
            return 0.0

        hold_frames = 10

        def expand_stills(images: Sequence[np.ndarray]) -> List[np.ndarray]:
            return [frame for image in images
                    for frame in [image] * hold_frames]

        # BaseEvaluator.evaluate_interleave has already prepended input_frame to
        # gt_images.  Predictions do not include it, so prepend it exactly once.
        gt_stills = (list(gt_images) if gt_images
                     else [input_frame, gt_final_frame])
        pred_stills = [input_frame] + list(pred_images)

        score = self._evaluate_task_specific(
            expand_stills(pred_stills),
            expand_stills(gt_stills),
            input_frame,
            gt_final_frame,
            eval_info,
        )
        self._last_task_details.update({
            'interleave_adapter': 'repeat_each_still_then_video_evaluator',
            'interleave_hold_frames': hold_frames,
            'interleave_pred_images': len(pred_images),
            'interleave_gt_images': max(0, len(gt_stills) - 1),
        })
        return score


class BookshelfEvaluator(BaseEvaluator):
    """
    O-30: Bookshelf - insert books (right side) one by one into correct positions among original books (left side).

    Evaluation:
    1. Final placement (40%): gt_last vs gen_last in changed region
    2. Sequential insertion (30%): books inserted gradually
    3. Original preservation (20%): gen_first vs gen_last in original book area
    4. Background clean (10%): gen_first vs gen_last outside all books
    """

    TASK_WEIGHTS = {
        'final_placement': 0.625,
        'sequential_insertion': 0.375,
    }

    def _detect_books_in_frame(self, frame: np.ndarray, shelf_top: int, x_min: int = 0, x_max: int = None) -> List[Dict]:
        """
        Detect individual books in a frame as a list of dicts with full metadata.
        Each book dict contains: x, y, w, h, cx, cy, area, height (=h), mean_bgr, mean_hue.
        """
        h, w = frame.shape[:2]
        if x_max is None:
            x_max = w
        kernel3 = np.ones((3, 3), np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Non-white mask (books are dark/colored, background is white)
        non_white = (gray < 220).astype(np.uint8) * 255
        non_white[shelf_top:, :] = 0          # exclude shelf
        non_white[:, :x_min] = 0              # exclude left of ROI
        non_white[:, x_max:] = 0              # exclude right of ROI
        non_white = cv2.morphologyEx(non_white, cv2.MORPH_OPEN, kernel3)

        contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        books = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > w * 0.8 or bh < 15:
                continue
            mask_cnt = np.zeros((h, w), np.uint8)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_bgr = cv2.mean(frame, mask=mask_cnt)[:3]
            mean_hue = cv2.mean(hsv[:, :, 0], mask=mask_cnt)[0]
            books.append({
                'x': x, 'y': y, 'w': bw, 'h': bh,
                'cx': x + bw // 2, 'cy': y + bh // 2,
                'area': int(area),
                'height': bh,          # book height = visual height in image
                'mean_bgr': tuple(float(v) for v in mean_bgr),
                'mean_hue': float(mean_hue),
            })
        books.sort(key=lambda b: b['x'])
        return books

    def _detect_books_from_gt(self, gt_first: np.ndarray, gt_last: np.ndarray):
        h, w = gt_first.shape[:2]
        kernel3 = np.ones((3, 3), np.uint8)

        # Step 1: find right-side books that disappeared: gt_last - gt_first
        diff = cv2.subtract(gt_last, gt_first)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_bin = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, kernel3)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, kernel3)

        # Use diff_bin contours only to get shelf_top and right_boundary
        contours, _ = cv2.findContours(diff_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rough_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > w * 0.6 or bh < 15:
                continue
            rough_boxes.append((x, y, bw, bh))

        if rough_boxes:
            shelf_top = max(y + bh for (x, y, bw, bh) in rough_boxes) - 2
            right_boundary = max(0, min(x for (x, y, bw, bh) in rough_boxes) - 10)
        else:
            shelf_top = int(h * 0.85)
            right_boundary = int(w * 0.6)

        # Detect actual shelf bottom: scan down from shelf_top to find where non-white ends
        gray_gt = cv2.cvtColor(gt_first, cv2.COLOR_BGR2GRAY)
        shelf_bottom = shelf_top
        for row in range(shelf_top, min(shelf_top + 60, h)):
            if (gray_gt[row, :] < 200).sum() > w * 0.3:  # row is mostly non-white = shelf bar
                shelf_bottom = row + 1
        shelf_bottom = shelf_bottom+ 2
        shelf_mask = np.zeros((h, w), np.uint8)
        shelf_mask[shelf_top:shelf_bottom, :] = 255

        # Step 2b: insertion zone on left side: gt_first - gt_last
        diff_rev = cv2.subtract(gt_first, gt_last)
        diff_rev_gray = cv2.cvtColor(diff_rev, cv2.COLOR_BGR2GRAY)
        _, insertion_zone_bin = cv2.threshold(diff_rev_gray, 30, 255, cv2.THRESH_BINARY)
        insertion_zone_bin = cv2.morphologyEx(insertion_zone_bin, cv2.MORPH_CLOSE, kernel3)
        insertion_zone_bin = cv2.morphologyEx(insertion_zone_bin, cv2.MORPH_OPEN, kernel3)
        insertion_zone_bin[:, right_boundary:] = 0
        insertion_zone_bin[shelf_top:, :] = 0
        insertion_zone_mask = insertion_zone_bin

        # Step 3: detect individual books with full metadata; build masks from bboxes
        right_books = self._detect_books_in_frame(gt_first, shelf_top,
                                                   x_min=right_boundary, x_max=w)
        left_books = self._detect_books_in_frame(gt_first, shelf_top,
                                                  x_min=0, x_max=right_boundary)
        insert_region_mask = np.zeros((h, w), np.uint8)
        for b in right_books:
            insert_region_mask[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']] = 255
        original_mask = np.zeros((h, w), np.uint8)
        for b in left_books:
            original_mask[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']] = 255

        # Extract individual gaps from insertion_zone_mask (each gap = one insertion slot)
        n_labels, labels = cv2.connectedComponents(insertion_zone_mask)
        gaps = []
        for label in range(1, n_labels):
            gap_mask = (labels == label).astype(np.uint8)
            if gap_mask.sum() < 50:
                continue
            ys, xs = np.where(gap_mask)
            gaps.append({
                'mask': gap_mask,
                'cx': int(xs.mean()),
                'height': int(ys.max() - ys.min()),
            })
        gaps.sort(key=lambda g: g['cx'])

        debug = {
            'diff_gray': diff_gray,
            'diff_bin': diff_bin,
            'diff_rev_gray': diff_rev_gray,
            'insertion_zone_mask': insertion_zone_mask,
            'gaps': gaps,
            'shelf_mask': shelf_mask,
            'shelf_top': shelf_top,
            'shelf_bottom': shelf_bottom,
            'insert_region_mask': insert_region_mask,
            'original_mask': original_mask,
            'right_boundary': right_boundary,
            'n_insert': len(right_books),
            'n_original': len(left_books),
            'right_books': right_books,
            'left_books': left_books,
        }
        return insert_region_mask, insertion_zone_mask, original_mask, shelf_mask, right_books, left_books, debug

    def _pixel_diff_score(self, frame1, frame2, mask, thresholds=(0.02, 0.05, 0.10, 0.20)):
        mask_pixels = int((mask > 0).sum())
        if mask_pixels == 0:
            return 1.0, {'ratio': 0.0, 'changed_px': 0, 'total_px': 0}
        diff = cv2.absdiff(frame1, frame2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed = int((gray_diff[mask > 0] > 20).sum())
        ratio = float(changed) / mask_pixels
        t1, t2, t3, t4 = thresholds
        if ratio < t1: score = 1.0
        elif ratio < t2: score = 1.0 - (ratio - t1) / (t2 - t1) * 0.3
        elif ratio < t3: score = 0.7 - (ratio - t2) / (t3 - t2) * 0.4
        elif ratio < t4: score = 0.3 - (ratio - t3) / (t4 - t3) * 0.3
        else: score = 0.0
        return score, {'ratio': round(ratio, 6), 'changed_px': changed, 'total_px': mask_pixels}

    def _compute_sequential_score(self, video_frames, gt_first, gaps: List[Dict],
                                   right_books: List[Dict]) -> Tuple[float, Dict]:
        n_insert = len(right_books)
        n_gaps = len(gaps)
        if n_insert == 0 or n_gaps == 0:
            return 0.0, {'n_insert': n_insert, 'n_gaps': n_gaps}

        def gap_filled(frame_gray, gap):
            region = frame_gray[gap['mask'] > 0]
            return float((region < 200).sum()) / len(region) > 0.8

        indices = list(range(len(video_frames)))

        filled_per_frame = []
        for idx in indices:
            f = video_frames[int(idx)]
            if f.shape != gt_first.shape:
                f = normalize_frame_size(f, gt_first)
            f_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            filled_per_frame.append([gap_filled(f_gray, g) for g in gaps])

        # Insertion events: first time each gap gets stably filled
        # A gap counts as filled only if it's also filled in the 2 frames before and 2 frames after
        insertion_events = []
        ever_filled = [False] * n_gaps
        n_frames = len(filled_per_frame)
        for i in range(len(filled_per_frame)):
            curr = filled_per_frame[i]
            newly = []
            for gi in range(n_gaps):
                if ever_filled[gi] or not curr[gi]:
                    continue
                # Check 2 frames before and 2 frames after; relax for last 2 frames
                if i >= n_frames - 2:
                    newly.append(gi)
                else:
                    neighbors = range(max(0, i - 2), min(n_frames, i + 3))
                    if all(filled_per_frame[j][gi] for j in neighbors):
                        newly.append(gi)
            if newly:
                insertion_events.append({
                    'frame_idx': int(indices[i]),
                    'n_added': len(newly),
                    'heights': [gaps[gi]['height'] for gi in newly],
                })
                for gi in newly:
                    ever_filled[gi] = True

        # Score 1: one at a time
        one_at_a_time = (sum(1 for e in insertion_events if e['n_added'] == 1) / len(insertion_events)
                         if insertion_events else 0.0)

        # Score 2: height order (monotonically increasing or decreasing)
        heights_in_order = []
        heights_in_idxes = []
        for e in insertion_events:
            heights_in_order.extend(e['heights'])
            heights_in_idxes.append(e['frame_idx'])
        if len(heights_in_order) >= 2:
            n_pairs = len(heights_in_order) - 1
            asc  = sum(1 for i in range(n_pairs) if heights_in_order[i+1] >= heights_in_order[i])
            desc = sum(1 for i in range(n_pairs) if heights_in_order[i+1] <= heights_in_order[i])
            height_order_score = max(asc, desc) / n_pairs
        else:
            height_order_score = 1.0

        n_filled_last = sum(filled_per_frame[-1])
        completion = min(n_filled_last / n_gaps, 1.0)
        score = round(completion * one_at_a_time * height_order_score, 4)
        filled_counts = [sum(f) for f in filled_per_frame]
        details = {
            'n_insert': n_insert,
            'n_gaps': n_gaps,
            'n_filled_last': n_filled_last,
            'n_insertion_events': len(insertion_events),
            'one_at_a_time': round(one_at_a_time, 4),
            'height_order_score': round(height_order_score, 4),
            'completion': round(completion, 4),
            "heights_in_idxes": str([round(h, 1) for h in heights_in_idxes]),
            'inserted_heights': str([round(h, 1) for h in heights_in_order]),
        }
        return score, details

    def _evaluate_task_specific(self, video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info):
        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        gen_first, gen_last = video_frames[0], video_frames[-1]
        gt_first, gt_last = gt_first_frame, gt_final_frame

        if gen_first.shape != gt_first.shape:
            gen_first = normalize_frame_size(gen_first, gt_first)
        if gen_last.shape != gt_last.shape:
            gen_last = normalize_frame_size(gen_last, gt_last)

        kernel = np.ones((5, 5), np.uint8)

        insert_region_mask, insertion_zone_mask, original_mask, shelf_mask, right_books, left_books, dbg = self._detect_books_from_gt(gt_first, gt_last)


        # --- 1. Final placement (40%): gt_last vs gen_last in insertion zone + right side should be empty ---
        insertion_zone_extended = insertion_zone_mask.copy()
        for col in range(insertion_zone_extended.shape[1]):
            filled_rows = np.where(insertion_zone_extended[:, col] > 0)[0]
            if len(filled_rows) > 0:
                insertion_zone_extended[0:filled_rows.max() + 1, col] = 255
        insertion_zone_eroded = cv2.erode(insertion_zone_extended, kernel, iterations=1) if insertion_zone_extended.sum() > 0 else insertion_zone_extended
        insert_region_eroded = cv2.erode(insert_region_mask, kernel, iterations=1) if insert_region_mask.sum() > 0 else insert_region_mask
        left_score, left_details = self._pixel_diff_score(
            gt_last, gen_last, insertion_zone_eroded, thresholds=(0.02, 0.05, 0.15, 0.3))
        right_score, right_details = self._pixel_diff_score(
            gt_last, gen_last, insert_region_eroded, thresholds=(0.01, 0.05, 0.15, 0.3))
        final_score = left_score * right_score
        final_details = {
            'left_ratio': left_details['ratio'], 'left_score': round(left_score, 4),
            'right_ratio': right_details['ratio'], 'right_score': round(right_score, 4),
        }

        # --- 2. Sequential insertion (30%) ---
        sequential_score, seq_details = self._compute_sequential_score(
            video_frames, gt_first, dbg['gaps'], right_books)
        n_insert = dbg['n_insert']
        zone_total = int((insertion_zone_mask > 0).sum())

        # --- 3. Original preservation (20%): gen_first vs gen_last in original book region ---
        orig_eroded = cv2.erode(original_mask, kernel, iterations=1) if original_mask.sum() > 0 else original_mask
        orig_score, orig_details = self._pixel_diff_score(
            gen_first, gen_last, orig_eroded, thresholds=(0.01, 0.05, 0.15, 0.3))

        # --- 4. Background clean (10%): gen_first vs gen_last outside all fg ---
        all_fg = cv2.bitwise_or(insertion_zone_mask, original_mask)
        all_fg = cv2.bitwise_or(all_fg, insert_region_mask)
        all_fg = cv2.bitwise_or(all_fg, shelf_mask)
        all_fg_dilated = cv2.dilate(all_fg, kernel, iterations=2)
        bg_mask = cv2.bitwise_not(all_fg_dilated)
        bg_score, bg_details = self._pixel_diff_score(
            gt_last, gen_last, bg_mask, thresholds=(0.005, 0.01, 0.025, 0.05))

        scores = {
            'final_placement': round(final_score, 4),
            'sequential_insertion': round(sequential_score, 4),
            'consistency': round((orig_score + bg_score) / 2, 4),
        }
        self._last_task_details = {
            **scores,
            'original_preservation': round(orig_score, 4),
            'background_clean': round(bg_score, 4),
            'n_insert': n_insert,
            'n_original': dbg['n_original'],
            **{f'seq_{k}': v for k, v in seq_details.items()},
            'shelf_top': dbg['shelf_top'],
            'right_boundary': dbg['right_boundary'],
            **{f'final_{k}': v for k, v in final_details.items()},
            **{f'orig_{k}': v for k, v in orig_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }

        return float((0.625 * scores['final_placement'] + 0.375 * scores['sequential_insertion']) * (0.6 + 0.4 * scores['consistency']))


    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if not pred_images or gt_final_frame is None or input_frame is None:
            return 0.0

        gt_first = input_frame
        gt_last = gt_final_frame
        gen_first = pred_images[0]
        gen_last = pred_images[-1]

        if gen_first.shape != gt_first.shape:
            gen_first = normalize_frame_size(gen_first, gt_first)
        if gen_last.shape != gt_last.shape:
            gen_last = normalize_frame_size(gen_last, gt_last)

        kernel = np.ones((5, 5), np.uint8)

        insert_region_mask, insertion_zone_mask, original_mask, shelf_mask, right_books, left_books, dbg = self._detect_books_from_gt(gt_first, gt_last)

        # --- 1. Final placement: same as video version ---
        insertion_zone_extended = insertion_zone_mask.copy()
        for col in range(insertion_zone_extended.shape[1]):
            filled_rows = np.where(insertion_zone_extended[:, col] > 0)[0]
            if len(filled_rows) > 0:
                insertion_zone_extended[0:filled_rows.max() + 1, col] = 255
        insertion_zone_eroded = cv2.erode(insertion_zone_extended, kernel, iterations=1) if insertion_zone_extended.sum() > 0 else insertion_zone_extended
        insert_region_eroded = cv2.erode(insert_region_mask, kernel, iterations=1) if insert_region_mask.sum() > 0 else insert_region_mask
        left_score, left_details = self._pixel_diff_score(
            gt_last, gen_last, insertion_zone_eroded, thresholds=(0.01, 0.05, 0.15, 0.3))
        right_score, right_details = self._pixel_diff_score(
            gt_last, gen_last, insert_region_eroded, thresholds=(0.01, 0.05, 0.15, 0.3))
        final_score = left_score * right_score

        # --- 2. Sequential insertion: directly check each frame, no neighbor validation ---
        gaps = dbg['gaps']
        n_gaps = len(gaps)
        n_insert = len(right_books)

        if n_insert == 0 or n_gaps == 0:
            sequential_score = 0.0
            seq_details = {'n_insert': n_insert, 'n_gaps': n_gaps}
        else:
            def gap_filled(frame_gray, gap):
                region = frame_gray[gap['mask'] > 0]
                return float((region < 200).sum()) / len(region) > 0.8

            filled_per_frame = []
            for f in pred_images:
                if f.shape != gt_first.shape:
                    f = normalize_frame_size(f, gt_first)
                f_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                filled_per_frame.append([gap_filled(f_gray, g) for g in gaps])

            # Insertion events: simple first-time detection, no neighbor check
            insertion_events = []
            ever_filled = [False] * n_gaps
            for i in range(len(filled_per_frame)):
                curr = filled_per_frame[i]
                newly = [gi for gi in range(n_gaps) if curr[gi] and not ever_filled[gi]]
                if newly:
                    insertion_events.append({
                        'frame_idx': i,
                        'n_added': len(newly),
                        'heights': [gaps[gi]['height'] for gi in newly],
                    })
                    for gi in newly:
                        ever_filled[gi] = True

            one_at_a_time = (sum(1 for e in insertion_events if e['n_added'] == 1) / len(insertion_events)
                             if insertion_events else 0.0)

            heights_in_order = []
            for e in insertion_events:
                heights_in_order.extend(e['heights'])
            if len(heights_in_order) >= 2:
                n_pairs = len(heights_in_order) - 1
                asc = sum(1 for i in range(n_pairs) if heights_in_order[i+1] >= heights_in_order[i])
                desc = sum(1 for i in range(n_pairs) if heights_in_order[i+1] <= heights_in_order[i])
                height_order_score = max(asc, desc) / n_pairs
            else:
                height_order_score = 1.0

            n_filled_last = sum(filled_per_frame[-1])
            completion = min(n_filled_last / n_gaps, 1.0)
            sequential_score = round(completion * one_at_a_time * height_order_score, 4)
            seq_details = {
                'n_insert': n_insert, 'n_gaps': n_gaps,
                'n_filled_last': n_filled_last,
                'n_insertion_events': len(insertion_events),
                'one_at_a_time': round(one_at_a_time, 4),
                'height_order_score': round(height_order_score, 4),
                'completion': round(completion, 4),
                'inserted_heights': str([round(h, 1) for h in heights_in_order]),
            }

        # --- 3. Consistency: original + background ---
        orig_eroded = cv2.erode(original_mask, kernel, iterations=1) if original_mask.sum() > 0 else original_mask
        orig_score, orig_details = self._pixel_diff_score(
            gen_first, gen_last, orig_eroded, thresholds=(0.01, 0.05, 0.15, 0.3))

        all_fg = cv2.bitwise_or(insertion_zone_mask, original_mask)
        all_fg = cv2.bitwise_or(all_fg, insert_region_mask)
        all_fg = cv2.bitwise_or(all_fg, shelf_mask)
        all_fg_dilated = cv2.dilate(all_fg, kernel, iterations=2)
        bg_mask = cv2.bitwise_not(all_fg_dilated)
        bg_score, bg_details = self._pixel_diff_score(
            gt_last, gen_last, bg_mask, thresholds=(0.005, 0.01, 0.025, 0.05))
        consistency_score = (orig_score + bg_score) / 2

        scores = {
            'final_placement': round(final_score, 4),
            'sequential_insertion': round(sequential_score, 4),
            'consistency': round(consistency_score, 4),
        }
        self._last_task_details = {
            **scores,
            'n_insert': dbg['n_insert'],
            'n_original': dbg['n_original'],
            **{f'seq_{k}': v for k, v in seq_details.items()},
            **{f'final_left_{k}': v for k, v in left_details.items()},
            **{f'final_right_{k}': v for k, v in right_details.items()},
            'orig_score': round(orig_score, 4),
            'bg_score': round(bg_score, 4),
        }

        return float((0.625 * scores['final_placement'] + 0.375 * scores['sequential_insertion']) * (0.6 + 0.4 * scores['consistency']))


class BallEatingEvaluator(BaseEvaluator):
    """
    O-31: Ball Eating (Greedy Algorithm)
    Black ball eats target balls one by one (smallest first that is ≤ its size),
    growing 1.4x each time. Target balls can be ANY color.

    Modeled after BallColorEvaluator with same level of rigor:
    - Per-ball pixel-presence curves (analogous to per-color pixel sequences)
    - Mask-based occlusion detection (actual pixel overlap, not position distance)
    - Backwards-scan eat detection (like merge detection)
    - Per-eat scoring with all 5 checks

    Evaluation:
    1. Final state (50%): all targets gone, black ball grown, no extra colors
    2. Eat process (50%): per-eat (order, arrival, growth, still)
    """

    TASK_WEIGHTS = {
        'final_state': 0.50,
        'eat_process': 0.50,
    }

    PRESENCE_THRESH = 0.85   # ball "present" if >= 85% of initial pixels remain
    STABLE_FRAMES = 2       # need N consecutive absent frames to confirm eat

    # ================ Detection helpers ================

    def _get_black_balls(self, frame: np.ndarray) -> list:
        """Detect all black balls: returns list of ((cx, cy), area), sorted by area desc."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri ** 2)
            if circ < 0.5:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            results.append(((cx, cy), area))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _get_black_ball(self, frame: np.ndarray) -> Optional[Tuple[Tuple[float, float], float]]:
        """Detect largest black ball: returns ((cx, cy), area) or None."""
        balls = self._get_black_balls(frame)
        return balls[0] if balls else None

    def _get_colored_mask(self, frame: np.ndarray, is_gen: bool = False) -> np.ndarray:
        """Get mask of all colored (non-black, non-gray, non-white) pixels."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_t, val_t = (50, 50) if is_gen else (70, 70)
        mask = ((hsv[:, :, 1] > sat_t) & (hsv[:, :, 2] > val_t)).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def _get_black_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get mask of dark (black ball) pixels."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    def _get_target_balls(self, frame: np.ndarray,
                           is_gen: bool = False) -> list:
        """Detect all non-black colored balls (any color)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colored_mask = self._get_colored_mask(frame, is_gen)
        contours, _ = cv2.findContours(colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        balls = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri ** 2)
            if circ < 0.5:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            ball_mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.drawContours(ball_mask, [cnt], -1, 255, -1)
            mean_hue = float(cv2.mean(hsv[:, :, 0], mask=ball_mask)[0])
            balls.append({
                'pos': (cx, cy), 'area': area, 'hue': mean_hue,
                'mask': ball_mask, 'radius': np.sqrt(area / np.pi),
            })
        return sorted(balls, key=lambda b: b['area'])  # smallest first


    def _build_ball_presence(self, frames: List[np.ndarray], gt_balls: list,
                              is_gen: bool = False) -> Dict[int, List[float]]:
        """For each GT ball, compute per-frame pixel presence ratio.
        Returns {ball_idx: [ratio_per_frame]}.
        """
        presence = {i: [] for i in range(len(gt_balls))}

        for t, frame in enumerate(frames):
            colored_mask = self._get_colored_mask(frame, is_gen)

            for i, ball in enumerate(gt_balls):
                init_px = int(ball['mask'].sum() / 255)
                if init_px <= 0:
                    presence[i].append(0.0)
                    continue

                # Count colored pixels within the ball's own mask region
                overlap = cv2.bitwise_and(colored_mask, ball['mask'])
                px_count = int(overlap.sum() / 255)
                ratio = px_count / init_px
                presence[i].append(ratio)

        return presence

    def _detect_occlusion_frames(self, frames: List[np.ndarray], gt_balls: list,
                                  is_gen: bool = False, min_overlap: int = 20) -> Dict[str, set]:
        """Detect per-ball occlusion: find actual colored balls in each frame,
        match to GT balls by hue+proximity, check overlap with black mask.
        Returns {'ball_0': set_of_frames, ..., '__black__': set_of_frames}.
        """
        kernel = np.ones((3, 3), np.uint8)
        h, w = frames[0].shape[:2]
        edge_margin = 15

        result = {f'ball_{i}': set() for i in range(len(gt_balls))}
        result['__black__'] = set()

        for t, frame in enumerate(frames):
            black_mask = self._get_black_mask(frame)
            black_d = cv2.dilate(black_mask, kernel, iterations=2)

            # Find actual colored balls in this frame
            cur_balls = self._get_target_balls(frame, is_gen)

            # Match each current ball to closest GT ball by hue + distance
            used_gt = set()
            for cb in cur_balls:
                best_i, best_score = -1, float('inf')
                for i, gb in enumerate(gt_balls):
                    if i in used_gt:
                        continue
                    hue_diff = min(abs(cb['hue'] - gb['hue']),
                                   180 - abs(cb['hue'] - gb['hue']))
                    dist = np.sqrt((cb['pos'][0] - gb['pos'][0])**2 +
                                   (cb['pos'][1] - gb['pos'][1])**2)
                    score = hue_diff * 2 + dist
                    if score < best_score and hue_diff < 30:
                        best_score = score
                        best_i = i
                if best_i < 0:
                    continue
                used_gt.add(best_i)

                # Check overlap between this ball's actual mask and dilated black mask
                ball_d = cv2.dilate(cb['mask'], kernel, iterations=2)
                overlap = cv2.bitwise_and(ball_d, black_d)
                overlap_px = int(overlap.sum() / 255)
                if overlap_px > min_overlap:
                    result[f'ball_{best_i}'].add(t)
                    result['__black__'].add(t)

            # Edge detection for black ball
            bys, bxs = np.where(black_mask > 0)
            if len(bxs) > 10:
                if (bxs.min() < edge_margin or bxs.max() > w - edge_margin or
                    bys.min() < edge_margin or bys.max() > h - edge_margin):
                    result['__black__'].add(t)

        return result

    # ================ Eat event detection (like BallColor merge detection) ================

    def _detect_eat_events(self, presence: Dict[int, List[float]],
                            gt_balls: list,
                            occlusion_frames: Dict[str, set] = None) -> List[Tuple[int, int]]:
        events = []
        for ball_idx in range(len(gt_balls)):
            curve = presence.get(ball_idx, [])
            n = len(curve)
            if n == 0:
                continue

            occluded = occlusion_frames.get(f'ball_{ball_idx}', set()) if occlusion_frames else set()

            if curve[0] < 0.9:
                continue

            if curve[-1] >= 0.1:
                continue

            last_present = -1
            for t in range(n - 1, -1, -1):
                # if t in occluded:
                #     continue
                if curve[t] >= self.PRESENCE_THRESH:
                    last_present = t
                    break

            if last_present >= 0:
                eat_time = last_present / max(n - 1, 1)
                events.append((eat_time, ball_idx, last_present))

        events.sort(key=lambda x: x[0])  # sort by normalized time
        return events

    def _score_final_state(self, gen_last: np.ndarray, gt_final: np.ndarray,
                            gt_n_targets: int) -> Tuple[float, Dict]:
        """Final state: all targets gone, black ball exists and grew, no extra colors."""
        remaining = len(self._get_target_balls(gen_last, is_gen=True))
        if remaining == 0:
            targets_score = 1.0
        else:
            targets_score = max(0, 1.0 - remaining / gt_n_targets)

        # Black ball should exist
        black = self._get_black_ball(gen_last)
        gt_black = self._get_black_ball(gt_final)
        if black and gt_black:
            gen_area, gt_area = black[1], gt_black[1]
            area_ratio = gen_area / gt_area if gt_area > 0 else 0
            if 0.8 < area_ratio and area_ratio < 1.2:
                black_score = 1.0
            elif 0.6 < area_ratio and area_ratio < 1.4:
                black_score = 0.7
            elif 0.3 < area_ratio and area_ratio < 1.8:
                black_score = 0.3
            else:
                black_score = 0.0
            # Penalize excess black regions (multiple balls or irregular shapes)
            total_black = int(self._get_black_mask(gen_last).sum() / 255)
            if gen_area > 0 and total_black > gen_area * 1.3:
                black_score *= max(0.3, gen_area / total_black)
        else:
            black_score = 0.0

        # Clean check: no colored pixels should remain (only black ball allowed)
        hsv = cv2.cvtColor(gen_last, cv2.COLOR_BGR2HSV)
        colored_mask = ((hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)).astype(np.uint8) * 255
        colored_pixels = int(colored_mask.sum() / 255)
        total_pixels = gen_last.shape[0] * gen_last.shape[1]
        unknown_ratio = colored_pixels / total_pixels if total_pixels > 0 else 0
        if unknown_ratio < 0.005:
            clean_score = 1.0
        elif unknown_ratio < 0.02:
            clean_score = 0.7
        elif unknown_ratio < 0.05:
            clean_score = 0.4
        else:
            clean_score = 0.0

        penalty = targets_score * 0.5 + clean_score * 0.5
        score = black_score * (0.5 + 0.5 * penalty)
        details = {
            'remaining_targets': remaining,
            'targets_score': round(targets_score, 4),
            'black_score': round(black_score, 4),
            'black_area_ratio': round(area_ratio, 3) if black and gt_black else -1,
            'unknown_ratio': round(unknown_ratio, 4),
            'clean_score': round(clean_score, 4),
        }
        return round(score, 4), details

    def _score_eat_process(self, gt_frames: List[np.ndarray],
                            video_frames: List[np.ndarray],
                            gt_balls: list,
                            gt_presence: Dict[int, List[float]],
                            gen_presence: Dict[int, List[float]],
                            occlusion_frames: Dict[str, set] = None) -> Tuple[float, Dict]:
        """Per-eat scoring using presence curves. Mirrors BallColor's _score_merge_process."""
        if not gt_balls:
            return 1.0, {'n_eats': 0}

        if len(video_frames) < 3:
            return 0.0, {'n_eats': len(self._detect_eat_events(gt_presence, gt_balls)),
                         'error': 'no_intermediate_frames'}

        gt_events = self._detect_eat_events(gt_presence, gt_balls)
        gen_events = self._detect_eat_events(gen_presence, gt_balls, occlusion_frames=occlusion_frames)

        event_frame_counts: Dict[int, int] = {}
        for _, _, event_frame in gen_events:
            event_frame_counts[event_frame] = event_frame_counts.get(event_frame, 0) + 1

        n_gt_eats = len(gt_events)
        if n_gt_eats == 0:
            return 1.0, {'n_eats': 0}

        # gen_order: {ball_idx: position_in_gen_order}
        gen_order = {ball_idx: pos for pos, (_, ball_idx, _) in enumerate(gen_events)}
        img_w = video_frames[0].shape[1]
        n_gen = len(video_frames)

        per_eat_scores = []
        self._growth_debug = []
        _growth_debug = self._growth_debug
        per_eat_details = []

        bb_init = self._get_black_ball(video_frames[0])
        black_prev_pos = bb_init[0] if bb_init else None

        # --- Group overlapping eats (eat_frame within 5 frames) ---
        OVERLAP_THRESH = 8
        eat_groups = []  # list of lists of gen_pos
        for gp in range(len(gen_events)):
            if not eat_groups or gen_events[gp][2] - gen_events[eat_groups[-1][0]][2] > OVERLAP_THRESH:
                eat_groups.append([gp])
            else:
                eat_groups[-1].append(gp)

        # --- Pre-compute growth per group (in gen time order) ---
        growth_data = {}  # gen_pos -> {before_area, after_area, before_idx, after_idx, group_size}
        init_bb_area = bb_init[1] if bb_init else 0
        prev_after_area = init_bb_area
        prev_after_idx = 0
        for group in eat_groups:
            first_gp = group[0]
            last_gp = group[-1]
            first_eat_idx = min(gen_events[first_gp][2], n_gen - 1)

            # before: use previous group's after, or initial
            if first_gp == 0:
                bef_idx = max(0, first_eat_idx - 3)
                bb_bef = self._get_black_ball(video_frames[bef_idx])
                bef_area = bb_bef[1] if bb_bef else 0
            else:
                bef_area = prev_after_area
                bef_idx = prev_after_idx

            # after: search from last eat in group to next group's first eat
            next_frame = n_gen
            if last_gp + 1 < len(gen_events):
                next_frame = gen_events[last_gp + 1][2]
            aft_end = min(next_frame, n_gen)
            max_area = 0
            aft_idx = first_eat_idx
            for fi in range(first_eat_idx, aft_end):
                bb = self._get_black_ball(video_frames[fi])
                if bb and bb[1] > max_area:
                    max_area = bb[1]
                    aft_idx = fi
            prev_after_area = max_area
            prev_after_idx = aft_idx

            # All events in this group share the same growth data
            for gp in group:
                growth_data[gp] = {
                    'before_area': bef_area, 'after_area': max_area,
                    'before_idx': bef_idx, 'after_idx': aft_idx,
                    'group_size': len(group),
                }

        # --- Pre-compute eaten_so_far per gen event (in gen time order) ---
        # For each gen_pos, collect ball_idxs eaten before it in gen time
        gen_eaten_before = {}  # gen_pos -> set of ball_idxs eaten before this in gen time
        for gp in range(len(gen_events)):
            gen_eaten_before[gp] = {gen_events[j][1] for j in range(gp)}

        # --- Arrival: process in gen time order to track black_prev_pos correctly ---
        arrival_data = {}  # gen_pos -> {arrival_s, detail, black_prev_pos_after}
        arr_black_prev_pos = bb_init[0] if bb_init else None
        for gp in range(len(gen_events)):
            g_eat_t, g_ball_idx, g_eat_frame = gen_events[gp]
            g_eat_idx = min(g_eat_frame, n_gen - 1)
            ball = gt_balls[g_ball_idx]
            target_pos = ball['pos']

            arrival_s = 0.0
            arr_detail = ''
            # Check eat_frame+1: the frame where ball just disappeared (black ball on target)
            arr_idx = min(g_eat_idx + 1, n_gen - 1)
            bb_at_eat = self._get_black_ball(video_frames[arr_idx])
            if not bb_at_eat:
                bb_at_eat = self._get_black_ball(video_frames[g_eat_idx])
            if bb_at_eat and arr_black_prev_pos:
                bb_pos = bb_at_eat[0]
                bb_radius = np.sqrt(bb_at_eat[1] / np.pi)
                target_radius = ball['radius']
                final_dist = np.sqrt((bb_pos[0] - target_pos[0])**2 + (bb_pos[1] - target_pos[1])**2)
                init_dist = np.sqrt((arr_black_prev_pos[0] - target_pos[0])**2 + (arr_black_prev_pos[1] - target_pos[1])**2)
                if final_dist < bb_radius + target_radius:
                    arrival_s = 1.0
                elif init_dist > 30:
                    r = final_dist / init_dist
                    arrival_s = 1.0 if r < 0.35 else (0.7 if r < 0.5 else (0.4 if r < 0.7 else 0.0))
                else:
                    arrival_s = 1.0 if final_dist < 30 else 0.0
                arr_detail = f'arr={arrival_s}(d={int(final_dist)}/{int(init_dist)},r={int(bb_radius)}+{int(target_radius)})'
                post_idx = min(g_eat_idx + 5, n_gen - 1)
                bb_after = self._get_black_ball(video_frames[post_idx])
                arr_black_prev_pos = bb_after[0] if bb_after else bb_pos
            elif bb_at_eat:
                arrival_s = 0.5
                arr_black_prev_pos = bb_at_eat[0]
                arr_detail = f'arr={arrival_s}(no_prev)'
            else:
                arr_detail = 'arr=0(no_bb)'

            arrival_data[gp] = {'score': arrival_s, 'detail': arr_detail}

        # --- Main scoring loop (GT order) ---
        for i, (gt_t, gt_ball_idx, gt_frame_idx) in enumerate(gt_events):
            ball = gt_balls[gt_ball_idx]
            target_pos = ball['pos']
            target_area = ball['area']
            detail_parts = [f'b{gt_ball_idx}(a={int(target_area)},h={int(ball["hue"])})']

            # --- Check 1: did this target disappear in gen? ---
            if gt_ball_idx not in gen_order:
                per_eat_scores.append(0.0)
                per_eat_details.append(f'b{gt_ball_idx}:missing')
                continue

            gen_pos = gen_order[gt_ball_idx]
            gen_eat_t, _, gen_eat_frame = gen_events[gen_pos]
            gen_eat_idx = min(gen_eat_frame, n_gen - 1)

            simultaneous_n = event_frame_counts.get(gen_eat_frame, 0)
            if simultaneous_n > 1:
                per_eat_scores.append(0.0)
                per_eat_details.append(
                    f'b{gt_ball_idx}:simultaneous@f{gen_eat_frame}'
                    f'(group={simultaneous_n})'
                )
                continue

            # --- Check 2: order ---
            pos_diff = abs(gen_pos - i)
            if pos_diff == 0:
                order_s = 1.0
            else:
                swapped_idx = gt_events[gen_pos][1] if gen_pos < len(gt_events) else -1
                max_area = max(gt_balls[swapped_idx]['area'], target_area) if swapped_idx >= 0 else 0
                if swapped_idx >= 0 and abs(gt_balls[swapped_idx]['area'] - target_area) < max_area * 0.15:
                    order_s = 1.0
                elif pos_diff == 1:
                    order_s = 0.3
                else:
                    order_s = 0.0
            detail_parts.append(f'ord={order_s}(gt{i}/gen{gen_pos})')

            # --- Check 3: arrival (from pre-computed, gen time order) ---
            arrival_s = arrival_data[gen_pos]['score']
            detail_parts.append(arrival_data[gen_pos]['detail'])

            # --- Check 4: growth (from pre-computed, gen time order) ---
            growth_s = 0.0
            growth_ratio = 0.0
            gd = growth_data[gen_pos]
            before_area = gd['before_area']
            max_bb_area = gd['after_area']
            _growth_debug.append((i, gt_ball_idx, gd['before_idx'], gd['after_idx']))

            if before_area > 0 and max_bb_area > 0:
                growth_ratio = max_bb_area / before_area
                # Expected ratio: ~2.0 per eat, ~2.0^n for grouped eats
                gs = gd['group_size']
                exp_lo = 1.5 ** gs  # lower bound
                exp_hi = 2.8 ** gs  # upper bound
                exp_lo2 = 1.2 ** gs
                exp_hi2 = 3.5 ** gs
                if exp_lo < growth_ratio < exp_hi:
                    growth_s = 1.0
                elif exp_lo2 < growth_ratio < exp_hi2:
                    growth_s = 0.5
                else:
                    growth_s = 0.0
            detail_parts.append(f'grow={growth_s}(r={growth_ratio:.2f},bef={int(before_area)},aft={int(max_bb_area)},tgt={int(target_area)})')

            # --- Check 5: still (gen time order for interval and eaten_so_far) ---
            # prev_eat_t: gen time of previous event in gen order
            prev_gen_eat_t = gen_events[gen_pos - 1][0] if gen_pos > 0 else 0.0
            eaten_so_far = gen_eaten_before[gen_pos]

            still_scores = []
            for ball_j in range(len(gt_balls)):
                if ball_j in eaten_so_far:
                    continue
                curve = gen_presence.get(ball_j, [])
                if not curve:
                    continue
                n_c = len(curve)
                s_f = int(prev_gen_eat_t * (n_c - 1)) + 2
                e_f = int(gen_eat_t * (n_c - 1)) - 2
                if e_f <= s_f:
                    continue
                occ_set = occlusion_frames.get(f'ball_{ball_j}', set()) if occlusion_frames else set()
                for k in range(s_f, e_f, max(1, (e_f - s_f) // 4)):
                    if k in occ_set:
                        continue
                    if curve[k] >= self.PRESENCE_THRESH:
                        still_scores.append(1.0)
                    elif curve[k] >= self.PRESENCE_THRESH * 0.5:
                        still_scores.append(0.5)
                    else:
                        still_scores.append(0.0)

            still_s = float(np.mean(still_scores)) if still_scores else 1.0
            detail_parts.append(f'stl={still_s:.2f}')

            # --- Combine ---
            eat_score = order_s * 0.25 + arrival_s * 0.25 + growth_s * 0.30 + still_s * 0.20
            per_eat_scores.append(eat_score)
            per_eat_details.append(' '.join(detail_parts))

        score = float(np.mean(per_eat_scores)) if per_eat_scores else 0.0
        details = {
            'gt_eats': n_gt_eats,
            'gen_eats': len(gen_events),
            'simultaneous_events': sum(
                count for count in event_frame_counts.values() if count > 1
            ),
        }
        for i in range(len(per_eat_details)):
            details[f'eat{i+1}'] = per_eat_details[i]
        return round(score, 4), details

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
        if not gt_frames or len(gt_frames) < 2:
            gt_frames = [gt_first_frame, gt_final_frame]
        if video_frames[0].shape != gt_first_frame.shape:
            video_frames = [normalize_frame_size(f, gt_first_frame) for f in video_frames]

        # ---- 1. Detect balls from GT first frame ----
        gt_balls = self._get_target_balls(gt_first_frame)
        if not gt_balls:
            self._last_task_details = {'error': 'no target balls detected'}
            return 0.0
        gt_black = self._get_black_ball(gt_first_frame)

        # ---- 2. Build per-ball presence curves (like BallColor pixel sequences) ----
        gt_presence = self._build_ball_presence(gt_frames, gt_balls, is_gen=False)
        gen_presence = self._build_ball_presence(video_frames, gt_balls, is_gen=True)

        # ---- 3. Detect occlusion (mask-based, like BallColor) ----
        gen_occlusion = self._detect_occlusion_frames(video_frames, gt_balls, is_gen=True)

        # ---- 4. Score ----
        final_score, final_details = self._score_final_state(
            video_frames[-1], gt_final_frame, len(gt_balls))
        eat_score, eat_details = self._score_eat_process(
            gt_frames, video_frames, gt_balls,
            gt_presence, gen_presence, occlusion_frames=gen_occlusion)

        scores = {'final_state': final_score, 'eat_process': eat_score}










        # Build detail strings
        gt_balls_info = ' | '.join(
            f'b{i}:({int(b["pos"][0])},{int(b["pos"][1])})a={int(b["area"])}h={int(b["hue"])}'
            for i, b in enumerate(gt_balls)
        )
        black_info = f'({int(gt_black[0][0])},{int(gt_black[0][1])})a={int(gt_black[1])}' if gt_black else 'none'

        # Build occlusion summary
        occ_summary = {}
        for key, occ_set in gen_occlusion.items():
            if occ_set:
                sorted_f = sorted(occ_set)
                ranges, start, prev = [], sorted_f[0], sorted_f[0]
                for f in sorted_f[1:]:
                    if f > prev + 1:
                        ranges.append(f'{start}-{prev}' if start != prev else str(start))
                        start = f
                    prev = f
                ranges.append(f'{start}-{prev}' if start != prev else str(start))
                occ_summary[key] = ','.join(ranges)

        self._last_task_details = {
            **scores,
            'n_targets': len(gt_balls),
            'gt_balls': gt_balls_info,
            'gt_black': black_info,
            'occlusion': occ_summary if occ_summary else 'none',
            **{f'final_{k}': v for k, v in final_details.items()},
            **{f'eat_{k}': v for k, v in eat_details.items()},
        }
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Give every still a dwell time, then reuse the video evaluator."""
        if not pred_images or input_frame is None or gt_final_frame is None:
            self._last_task_details = {'error': 'no_input_or_pred'}
            return 0.0

        hold_frames = 10

        def expand_stills(images: Sequence[np.ndarray]) -> List[np.ndarray]:
            return [frame for image in images
                    for frame in [image] * hold_frames]

        # BaseEvaluator.evaluate_interleave prepends the input to gt_images but
        # not to pred_images.
        gt_stills = (list(gt_images) if gt_images
                     else [input_frame, gt_final_frame])
        pred_stills = [input_frame] + list(pred_images)

        score = self._evaluate_task_specific(
            expand_stills(pred_stills),
            expand_stills(gt_stills),
            input_frame,
            gt_final_frame,
            eval_info,
        )
        self._last_task_details.update({
            'interleave_adapter': 'repeat_each_still_then_video_evaluator',
            'interleave_hold_frames': hold_frames,
            'interleave_pred_images': len(pred_images),
            'interleave_gt_images': max(0, len(gt_stills) - 1),
        })
        return score



class RollingBallEvaluator(BaseEvaluator):
    """
    O-32: Rolling Ball along blue dashed path.

    Evaluation:
    1. Final position correct (40%): ball ends at GT's final position
    2. Path following (30%): ball stays on the blue dashed line during movement
    3. Scene preservation (20%): blue path + scene not destroyed
    4. Background clean (10%): no extra objects generated
    """

    TASK_WEIGHTS = {
        'completion': 1.0,
    }
    CONSISTENCY_GATE_FLOOR = 0.60

    def _detect_ball_color(self, first_frame: np.ndarray, last_frame: np.ndarray) -> Optional[int]:
        """Detect ball color by finding the ball in last frame."""
        fg_first = self._detect_fg_mask(first_frame)
        fg_last = self._detect_fg_mask(last_frame)
        # New foreground in last frame = ball at its final position
        new_in_last = cv2.bitwise_and(fg_last, cv2.bitwise_not(fg_first))
        kernel = np.ones((5, 5), np.uint8)
        new_in_last = cv2.morphologyEx(new_in_last, cv2.MORPH_OPEN, kernel)

        hsv_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        colorful = cv2.inRange(hsv_last, np.array([0, 80, 80]), np.array([180, 255, 255]))
        ball_pixels = cv2.bitwise_and(new_in_last, colorful)

        if int((ball_pixels > 0).sum()) < 30:
            # Fallback: try new foreground in first_frame (ball at start)
            new_in_first = cv2.bitwise_and(fg_first, cv2.bitwise_not(fg_last))
            new_in_first = cv2.morphologyEx(new_in_first, cv2.MORPH_OPEN, kernel)
            hsv_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
            colorful_first = cv2.inRange(hsv_first, np.array([0, 80, 80]), np.array([180, 255, 255]))
            ball_pixels = cv2.bitwise_and(new_in_first, colorful_first)
            if int((ball_pixels > 0).sum()) < 30:
                return None
            ball_hues = hsv_first[:, :, 0][ball_pixels > 0]
        else:
            ball_hues = hsv_last[:, :, 0][ball_pixels > 0]

        if len(ball_hues) == 0:
            return None
        return int(np.median(ball_hues))

    def _detect_ball_mask(self, frame: np.ndarray, ball_hue: Optional[int] = None) -> np.ndarray:
        """Detect ball mask using known ball hue. If hue unknown, fallback to red."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if ball_hue is not None:
            hue_range = 15
            if ball_hue < hue_range:
                mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([ball_hue + hue_range, 255, 255]))
                mask |= cv2.inRange(hsv, np.array([180 - hue_range + ball_hue, 80, 80]), np.array([180, 255, 255]))
            elif ball_hue > 180 - hue_range:
                mask = cv2.inRange(hsv, np.array([ball_hue - hue_range, 80, 80]), np.array([180, 255, 255]))
                mask |= cv2.inRange(hsv, np.array([0, 80, 80]), np.array([hue_range - (180 - ball_hue), 255, 255]))
            else:
                mask = cv2.inRange(hsv, np.array([ball_hue - hue_range, 80, 80]),
                                   np.array([ball_hue + hue_range, 255, 255]))
        else:
            mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
            mask |= cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        return mask

    def _detect_ball(self, frame: np.ndarray, ball_hue: Optional[int] = None) -> Optional[Tuple[float, float]]:
        """Detect ball center. Returns (x, y) or None."""
        mask = self._detect_ball_mask(frame, ball_hue)
        if int((mask > 0).sum()) > 30:
            coords = np.where(mask > 0)
            return (float(np.mean(coords[1])), float(np.mean(coords[0])))
        return None

    def _detect_path_mask(self, first_frame: np.ndarray, last_frame: np.ndarray,
                          ball_hue: Optional[int] = None) -> np.ndarray:
        """Detect path: colored regions that don't change between first and last frame,
        plus the path endpoints hidden under the ball in each frame."""
        fg1 = self._detect_fg_mask(first_frame)
        fg2 = self._detect_fg_mask(last_frame)
        # Path = foreground present in both frames (ball position differs, so excluded)
        path = cv2.bitwise_and(fg1, fg2)

        ball1 = self._detect_ball_mask(first_frame, ball_hue)
        ball2 = self._detect_ball_mask(last_frame, ball_hue)
        kernel_small = np.ones((3, 3), np.uint8)
        ball1_dilated = cv2.dilate(ball1, kernel_small, iterations=2)
        ball2_dilated = cv2.dilate(ball2, kernel_small, iterations=2)
        # Path exposed in last_frame (was hidden by ball in first_frame)
        exposed_in_last = cv2.bitwise_and(fg2, ball1_dilated)
        exposed_in_last = cv2.bitwise_and(exposed_in_last, cv2.bitwise_not(ball2_dilated))
        # Path exposed in first_frame (will be hidden by ball in last_frame)
        exposed_in_first = cv2.bitwise_and(fg1, ball2_dilated)
        exposed_in_first = cv2.bitwise_and(exposed_in_first, cv2.bitwise_not(ball1_dilated))
        path = cv2.bitwise_or(path, cv2.bitwise_or(exposed_in_last, exposed_in_first))

        kernel = np.ones((5, 5), np.uint8)
        path = cv2.morphologyEx(path, cv2.MORPH_CLOSE, kernel)
        path = cv2.dilate(path, kernel, iterations=2)
        return path

    def _detect_fg_mask(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        corners = [frame[2, 2], frame[2, w-3], frame[h-3, 2], frame[h-3, w-3]]
        bg_color = np.mean(corners, axis=0)
        diff = np.sqrt(np.sum((frame.astype(float) - bg_color.astype(float)) ** 2, axis=2))
        binary = (diff > 30).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _pixel_diff_score(self, frame1, frame2, mask, thresholds=(0.02, 0.05, 0.10, 0.20)):
        mask_pixels = int((mask > 0).sum())
        if mask_pixels == 0:
            return 1.0, {'ratio': 0.0, 'changed_px': 0, 'total_px': 0}
        diff = cv2.absdiff(frame1, frame2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed = int((gray_diff[mask > 0] > 20).sum())
        ratio = float(changed) / mask_pixels
        t1, t2, t3, t4 = thresholds
        if ratio < t1: score = 1.0
        elif ratio < t2: score = 1.0 - (ratio - t1) / (t2 - t1) * 0.3
        elif ratio < t3: score = 0.7 - (ratio - t2) / (t3 - t2) * 0.4
        elif ratio < t4: score = 0.3 - (ratio - t3) / (t4 - t3) * 0.3
        else: score = 0.0
        return score, {'ratio': round(ratio, 6), 'changed_px': changed, 'total_px': mask_pixels}

    def _evaluate_task_specific(self, video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info):
        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        gen_first, gen_last = video_frames[0], video_frames[-1]
        gt_first, gt_last = gt_first_frame, gt_final_frame

        if gen_first.shape != gt_first.shape:
            gen_first = normalize_frame_size(gen_first, gt_first)
        if gen_last.shape != gt_last.shape:
            gen_last = normalize_frame_size(gen_last, gt_last)

        frame_h, frame_w = gt_first.shape[:2]

        gt_ball_hue = self._detect_ball_color(gt_first, gt_last)
        gen_ball_hue = self._detect_ball_color(gen_first, gen_last)

        # 1. Final position (40%): ball at GT's final position, scaled by size ratio
        gt_ball = self._detect_ball(gt_last, gt_ball_hue)
        gen_ball = self._detect_ball(gen_last, gt_ball_hue)
        gt_ball_mask = self._detect_ball_mask(gt_last, gt_ball_hue)
        gen_ball_mask = self._detect_ball_mask(gen_last, gt_ball_hue)
        gt_ball_pixels = int((gt_ball_mask > 0).sum())
        gen_ball_pixels = int((gen_ball_mask > 0).sum())
        if gt_ball_pixels > 0 and gen_ball_pixels > 0:
            raw_ratio = gen_ball_pixels / gt_ball_pixels
            if 0.6 <= raw_ratio <= 1.8:
                size_ratio = 1.0
            elif 0.3 <= raw_ratio <= 2.5:
                size_ratio = 0.7
            elif 0.1 <= raw_ratio <= 4:
                size_ratio = 0.4
            else:
                size_ratio = 0.1
        else:
            size_ratio = 0.0 if gt_ball_pixels > 0 else 1.0
            
        if gt_ball is not None and gen_ball is not None:
            dist = np.sqrt((gt_ball[0] - gen_ball[0])**2 + (gt_ball[1] - gen_ball[1])**2)
            norm_dist = dist / max(frame_h, frame_w)
            if norm_dist < 0.03:
                pos_score = 1.0
            elif norm_dist < 0.08:
                pos_score = 1.0 - (norm_dist - 0.03) / 0.05 * 0.3
            elif norm_dist < 0.15:
                pos_score = 0.7 - (norm_dist - 0.08) / 0.07 * 0.4
            else:
                pos_score = max(0.0, 0.3 - (norm_dist - 0.15) / 0.15 * 0.3)
            pos_score *= size_ratio
        elif gen_ball is not None:
            pos_score = 0.1 * size_ratio
        else:
            pos_score = 0.0

        # 2. Path following (30%): ball stays on path and moves rightward
        path_mask = self._detect_path_mask(gt_first, gt_last, gt_ball_hue)
        use_hue = gt_ball_hue
        on_path_count = 0
        detected_count = 0
        ball_positions = []  # list of (x, y)
        for idx in range(len(video_frames)):
            f = video_frames[idx]
            if f.shape != gt_first.shape:
                f = normalize_frame_size(f, gt_first)
            ball_pos = self._detect_ball(f, use_hue)
            if ball_pos is not None:
                detected_count += 1
                ball_positions.append(ball_pos)
                bx, by = int(ball_pos[0]), int(ball_pos[1])
                r = 15
                y1, y2 = max(0, by-r), min(frame_h, by+r)
                x1, x2 = max(0, bx-r), min(frame_w, bx+r)
                if int((path_mask[y1:y2, x1:x2] > 0).sum()) > 0:
                    on_path_count += 1
        on_path_ratio = on_path_count / max(len(video_frames), 1)

        # Direction score: ball x-coordinate should monotonically increase (move right)
        direction_score = 0.0
        if len(ball_positions) >= 2:
            rightward_count = 0
            for i in range(1, len(ball_positions)):
                if ball_positions[i][0] >= ball_positions[i-1][0] - 1:  # 10px tolerance: right, still, or slight left ok
                    rightward_count += 1
            direction_score = rightward_count / (len(ball_positions) - 1)
        elif len(ball_positions) == 1:
            direction_score = 0.5

        gt_start_ball = self._detect_ball(gt_first, gt_ball_hue)
        distinct_intermediate_positions: List[Tuple[float, float]] = []
        intermediate_margin = 0.0
        if gt_start_ball is not None and gt_ball is not None:
            travel_distance = float(np.hypot(
                gt_ball[0] - gt_start_ball[0],
                gt_ball[1] - gt_start_ball[1],
            ))
            intermediate_margin = max(10.0, travel_distance * 0.15)
            for pos in ball_positions:
                from_start = float(np.hypot(
                    pos[0] - gt_start_ball[0], pos[1] - gt_start_ball[1],
                ))
                from_final = float(np.hypot(
                    pos[0] - gt_ball[0], pos[1] - gt_ball[1],
                ))
                if (from_start < intermediate_margin
                        or from_final < intermediate_margin):
                    continue
                if all(
                    np.hypot(pos[0] - prev[0], pos[1] - prev[1])
                    >= intermediate_margin
                    for prev in distinct_intermediate_positions
                ):
                    distinct_intermediate_positions.append(pos)

        intermediate_position_score = (
            1.0 if len(distinct_intermediate_positions) >= 2 else 0.0
        )
        path_score = (
            on_path_ratio * direction_score * intermediate_position_score
        )


        # 3. Scene preservation (20%): blue path not destroyed
        fg_first = self._detect_fg_mask(gt_first)
        kernel = np.ones((5, 5), np.uint8)
        # Exclude ball areas using detected ball hue
        ball_mask_gt = self._detect_ball_mask(gt_first, gt_ball_hue)
        ball_mask_gt |= self._detect_ball_mask(gt_last, gt_ball_hue)
        ball_mask_gen = self._detect_ball_mask(gen_first, use_hue)
        ball_mask_gen |= self._detect_ball_mask(gen_last, use_hue)
        ball_exclude = cv2.dilate(cv2.bitwise_or(ball_mask_gt, ball_mask_gen), kernel, iterations=2)
        scene_mask = cv2.bitwise_and(fg_first, cv2.bitwise_not(ball_exclude))
        scene_score, scene_details = self._pixel_diff_score(
            gen_first, gen_last, scene_mask, thresholds=(0.25, 0.4, 0.5, 0.70))

        # 4. Background clean (10%)
        all_fg = cv2.bitwise_or(self._detect_fg_mask(gt_first), self._detect_fg_mask(gt_last))
        all_fg_dilated = cv2.dilate(all_fg, kernel, iterations=1)
        bg_mask = cv2.bitwise_not(all_fg_dilated)
        bg_score, bg_details = self._pixel_diff_score(
            gen_first, gen_last, bg_mask, thresholds=(0.005, 0.02, 0.05, 0.10))

        scores = {
            'completion': round(pos_score * (0.2 + 0.8 * path_score), 4),
            "consistency": round((scene_score + bg_score) / 2, 4),
        }
        self._last_task_details = {
            **scores,
            'scene_preservation': round(scene_score, 4),
            'background_clean': round(bg_score, 4),
            "pos_score": round(pos_score, 4),
            'pos_size_ratio': round(size_ratio, 4),
            'gt_ball_hue': gt_ball_hue, 'gen_ball_hue': gen_ball_hue,
            'on_path_ratio': round(on_path_ratio, 4),
            'direction_score': round(direction_score, 4),
            'intermediate_position_score': intermediate_position_score,
            'distinct_intermediate_positions': len(distinct_intermediate_positions),
            'intermediate_position_margin': round(intermediate_margin, 2),
            'on_path': on_path_count, 'detected': detected_count,
            **{f'scene_{k}': v for k, v in scene_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }
        consistency_gate = self.CONSISTENCY_GATE_FLOOR + (
            (1.0 - self.CONSISTENCY_GATE_FLOOR) * scores['consistency']
        )
        self._last_task_details['consistency_gate'] = round(
            consistency_gate, 4,
        )
        return float(scores['completion'] * consistency_gate)



class CountingObjectEvaluator(BaseEvaluator):
    """
    O-33: Counting Objects

    Evaluation:
    1. Count correctness (60%): OCR the number from gen, compare with GT
    2. Object preservation (20%): shapes not destroyed
    3. Background preservation (20%): background clean

    Supports two OCR backends: 'easyocr' and 'pytesseract'.
    Set via ocr_backend parameter. Falls back to pixel matching if neither available.
    """

    TASK_WEIGHTS = {
        'count_correctness': 0.80,
        'consistency': 0.20,
    }

    def __init__(self, device: str = 'cpu', task_name: str = '', ocr_backend: str = 'easyocr'):
        super().__init__(device, task_name)
        self._ocr_backend = ocr_backend
        self._easyocr_reader = None
        self._paddleocr_reader = None

    def _get_easyocr_reader(self):
        if self._easyocr_reader is None:
            import os, easyocr, torch
            self._easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), model_storage_directory=(os.environ.get('VBVR_EASYOCR_MODELS') or None))
        return self._easyocr_reader

    def _get_paddleocr_reader(self):
        if self._paddleocr_reader is None:
            from paddleocr import PaddleOCR
            self._paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        return self._paddleocr_reader

    def _parse_ocr_number(self, text: str) -> Optional[int]:
        """Try to extract a count/total number from OCR text."""
        import re
        text = text.strip()
        match = re.search(r'(?:count|total)\s*[:;,.]\s*(\d+)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r'^(\d+)$', text)
        if match:
            return int(match.group(1))
        return None

    def _ocr_extract_number_paddleocr(self, frame: np.ndarray) -> Tuple[int, List, float]:
        """Extract count number using PaddleOCR. Returns (number, matched_bbox, confidence)."""
        try:
            reader = self._get_paddleocr_reader()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = reader.ocr(rgb, cls=True)
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, conf) = line[0], line[1]
                    if conf < 0.3:
                        continue
                    num = self._parse_ocr_number(text)
                    if num is not None:
                        return num, [bbox], conf
            return -1, [], 0.0
        except Exception:
            return -1, [], 0.0

    def _ocr_extract_number_easyocr(self, frame: np.ndarray) -> Tuple[int, List, float]:
        """Extract count number using easyocr. Returns (number, matched_bbox, confidence)."""
        try:
            reader = self._get_easyocr_reader()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = reader.readtext(rgb)
            for bbox, text, conf in results:
                if conf < 0.3:
                    continue
                num = self._parse_ocr_number(text)
                if num is not None:
                    return num, [bbox], conf
            return -1, [], 0.0
        except Exception:
            return -1, [], 0.0

    def _ocr_extract_number(self, frame: np.ndarray) -> Tuple[int, List, float]:
        """Extract number using configured OCR backend."""
        if self._ocr_backend == 'easyocr':
            return self._ocr_extract_number_easyocr(frame)
        else:
            return self._ocr_extract_number_paddleocr(frame)

    def _bboxes_to_mask(self, bboxes: List, h: int, w: int, padding: int = 0) -> np.ndarray:
        """Convert easyocr bboxes (list of 4-point polygons) to binary mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        for bbox in bboxes:
            pts = np.array(bbox, dtype=np.int32)
            if padding > 0:
                pts[:, 0] = np.clip(pts[:, 0] + np.where(pts[:, 0] < np.mean(pts[:, 0]), -padding, padding), 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1] + np.where(pts[:, 1] < np.mean(pts[:, 1]), -padding, padding), 0, h - 1)
            cv2.fillPoly(mask, [pts], 255)
        return mask

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

        # --- 1. Count correctness (60%): OCR the number ---
        gt_count, gt_bboxes, gt_conf = self._ocr_extract_number(gt_last)
        gen_count, gen_bboxes, gen_conf = self._ocr_extract_number(gen_last)

        if gt_count != -1 and gen_count != -1:
            if gen_count == gt_count:
                count_score = 1.0
            elif abs(gen_count - gt_count) == 1:
                count_score = 0.3
            elif abs(gen_count - gt_count) <= 2:
                count_score = 0.1
            else:
                count_score = 0.0
        elif gen_count == -1:
            count_score = 0.0
        else:
            count_score = 0.0

        count_details = {
            'gt_count': gt_count,
            'gen_count': gen_count,
            'gt_conf': round(gt_conf, 4),
            'gen_conf': round(gen_conf, 4),
            'ocr_backend': self._ocr_backend,
        }

        # --- Text exclusion mask from OCR bboxes ---
        gt_text_mask = self._bboxes_to_mask(gt_bboxes, frame_h, frame_w, padding=10)
        gen_text_mask = self._bboxes_to_mask(gen_bboxes, frame_h, frame_w)
        kernel = np.ones((7, 7), np.uint8)
        text_exclude = cv2.dilate(
            cv2.bitwise_or(gt_text_mask, gen_text_mask), kernel, iterations=2)

        # --- Foreground: colored objects on GT first frame (non-white regions) ---
        fg_mask = self._detect_fg_mask(gt_first)
        fg_no_text = cv2.bitwise_and(fg_mask, cv2.bitwise_not(text_exclude))
        fg_no_text = cv2.dilate(fg_no_text, kernel, iterations=1)

        # --- Background: everything except foreground and text ---
        bg_mask = cv2.bitwise_not(cv2.bitwise_or(cv2.dilate(fg_mask, kernel, iterations=1), text_exclude))

        # --- 2. Foreground preservation: gt_last vs gen_last on object regions ---
        fg_score, fg_details = self._pixel_diff_score(
            gt_last, gen_last, fg_no_text, thresholds=(0.05, 0.1, 0.25, 0.50))

        # --- 3. Background preservation: gt_last vs gen_last on bg regions ---
        bg_score, bg_details = self._pixel_diff_score(
            gt_last, gen_last, bg_mask, thresholds=(0.005, 0.02, 0.05, 0.10))

        consistency = (fg_score + bg_score) / 2

        scores = {
            'count_correctness': round(count_score, 4),
            'consistency': round(consistency, 4),
        }
        self._last_task_details = {
            **scores,
            'foreground_preservation': round(fg_score, 4),
            'background_preservation': round(bg_score, 4),
            **{f'count_{k}': v for k, v in count_details.items()},
            **{f'fg_{k}': v for k, v in fg_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }

        return float((scores['count_correctness']) * (0.6 + 0.4 * scores['consistency']))


class DotToDotEvaluator(BaseEvaluator):
    """
    O-34: Dot to Dot

    Task: Connect numbered circles (any color) in numerical order (1→2→3→...→N)
    with red line segments.

    Final score (clamped to [0, 1]):
        final = connection_completeness
              * connection_order_penalty
              * numerical_consistency_penalty
              - consistency_penalty

    where:
        connection_completeness =
            correct_red_connections / max(N - 1, detected_lines_in_final_frame)
        connection_order_penalty =
            LCS(detected_new_connection_sequence, expected_sequence) / (N - 1)
        numerical_consistency_penalty =
            max(0, 1 - 0.05 * num_circles_whose_number_changed)
        consistency_penalty = 0.10 if (masked-region similarity between gt_final
            and gen_final, with circles and red lines excluded) < 0.7 else 0.0
    """

    _ocr_reader = None
    _ocr_reader_failed = False

    @classmethod
    def _get_ocr_reader(cls):
        if cls._ocr_reader is not None:
            return cls._ocr_reader
        if cls._ocr_reader_failed:
            return None
        try:
            import easyocr, torch
            cls._ocr_reader = easyocr.Reader(
                ['en'], gpu=torch.cuda.is_available(), verbose=False)
        except Exception:
            cls._ocr_reader_failed = True
            cls._ocr_reader = None
        return cls._ocr_reader

    # ---------------------------------------------------------------- helpers

    def _red_mask(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) != 3:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 90, 70]), np.array([12, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 90, 70]), np.array([180, 255, 255]))
        return m1 | m2

    def _detect_circles(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored disk dots via Hough Gradient on grayscale.
        """
        if len(frame.shape) != 3:
            return []
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        min_r = max(15, min(h, w) // 50)
        max_r = max(min_r + 1, min(h, w) // 8)
        raw = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(20, int(min(h, w) * 0.05)),
            param1=80, param2=30,
            minRadius=min_r, maxRadius=max_r,
        )
        if raw is None:
            return []
        out: List[Dict] = []
        for (cx, cy, r) in np.round(raw[0]).astype(int):
            if r < min_r or not (0 <= cx < w and 0 <= cy < h):
                continue
            # Interior saturation: high for a coloured disk, low for
            # white-background digit loops.
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (int(cx), int(cy)), max(2, int(r * 0.6)), 255, -1)
            sat = hsv[..., 1][mask > 0]
            if sat.size == 0 or float(sat.mean()) < 60.0:
                continue
            out.append({'center': (int(cx), int(cy)), 'radius': int(r)})
        # Drop radius outliers — real dots are all the same size.
        if len(out) >= 3:
            radii = sorted(c['radius'] for c in out)
            median = radii[len(radii) // 2]
            out = [c for c in out if 0.7 * median <= c['radius'] <= 1.5 * median]

        out.sort(key=lambda c: -c['radius'])
        deduped: List[Dict] = []
        for c in out:
            cx, cy = c['center']
            r = c['radius']
            dup = False
            for k in deduped:
                kx, ky = k['center']
                if (cx - kx) ** 2 + (cy - ky) ** 2 < (0.9 * (r + k['radius'])) ** 2:
                    dup = True
                    break
            if not dup:
                deduped.append(c)
        return deduped

    def _ocr_frame_digits(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int]]]:
        """OCR the entire frame; return [(digit, (cx, cy))] for every digit token found.

        Centroids are mapped back to the original frame coordinate space.
        """
        reader = self._get_ocr_reader()
        if reader is None:
            return []
        h, w = frame.shape[:2]
        target = 1200
        scale = 1.0
        ocr_img = frame
        if max(h, w) < target:
            scale = target / max(h, w)
            ocr_img = cv2.resize(
                frame, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        try:
            results = reader.readtext(
                ocr_img, allowlist='0123456789', paragraph=False,
            )
        except Exception:
            return []
        out: List[Tuple[int, Tuple[int, int]]] = []
        for bbox, text, _conf in results:
            digits = ''.join(c for c in text if c.isdigit())
            if not digits:
                continue
            try:
                n = int(digits)
            except ValueError:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            cx = int((sum(xs) / len(xs)) / scale)
            cy = int((sum(ys) / len(ys)) / scale)
            out.append((n, (cx, cy)))
        return out

    def _assign_digits_to_circles(
        self,
        result: List[Optional[int]],
        detections: List[Tuple[int, Tuple[int, int]]],
        circles: List[Dict],
    ) -> None:
        for digit, (tx, ty) in detections:
            best_i, best_d = None, float('inf')
            for i, c in enumerate(circles):
                if result[i] is not None:
                    continue
                cx, cy = c['center']
                d = (tx - cx) ** 2 + (ty - cy) ** 2
                if d < (c['radius'] * 1.6) ** 2 and d < best_d:
                    best_d = d
                    best_i = i
            if best_i is not None:
                result[best_i] = digit

    def _ocr_single_circle(
        self, frame: np.ndarray, circle: Dict,
    ) -> Optional[int]:
        """Crop a tight box around one circle and OCR it in isolation.

        EasyOCR routinely misses thin digits (e.g. "1") in whole-frame mode.
        Running it on a small upscaled crop recovers them with high confidence.
        """
        reader = self._get_ocr_reader()
        if reader is None:
            return None
        h, w = frame.shape[:2]
        cx, cy = circle['center']
        r = circle['radius']
        pad = int(r * 1.2)
        y0, y1 = max(0, cy - pad), min(h, cy + pad)
        x0, x1 = max(0, cx - pad), min(w, cx + pad)
        if y1 - y0 < 4 or x1 - x0 < 4:
            return None
        crop = frame[y0:y1, x0:x1]

        big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
        try:
            results = reader.readtext(big, allowlist='0123456789', paragraph=False)
        except Exception:
            return None
        # Prefer the highest-confidence single-digit token in the central area.
        best, best_conf = None, 0.0
        ch, cw = big.shape[:2]
        for bbox, text, conf in results:
            digits = ''.join(c for c in text if c.isdigit())
            if not digits:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            tx, ty = sum(xs) / len(xs), sum(ys) / len(ys)
            # Center proximity bonus — neighboring-circle bleed lives at the edge.
            cx_b, cy_b = cw / 2.0, ch / 2.0
            edge_dist = ((tx - cx_b) ** 2 + (ty - cy_b) ** 2) ** 0.5
            if edge_dist > 0.6 * min(cw, ch):
                continue
            try:
                n = int(digits)
            except ValueError:
                continue
            score = float(conf)
            if score > best_conf:
                best_conf = score
                best = n
        if best is None or best_conf < 0.4:
            return None
        return best

    def _fill_by_elimination(
        self, result: List[Optional[int]], N: int,
    ) -> None:
        assigned = {n for n in result if n is not None}
        missing_slots = [i for i, n in enumerate(result) if n is None]
        missing_vals = [v for v in range(1, N + 1) if v not in assigned]
        if len(missing_slots) == 1 and len(missing_vals) == 1:
            result[missing_slots[0]] = missing_vals[0]

    def _circle_numbers(
        self, frame: np.ndarray, circles: List[Dict],
    ) -> List[Optional[int]]:
        result: List[Optional[int]] = [None] * len(circles)
        if not circles:
            return result
        self._assign_digits_to_circles(result, self._ocr_frame_digits(frame), circles)
        if any(n is None for n in result):
            red = self._red_mask(frame)
            if red.any():
                cleaned = frame.copy()
                dilated = cv2.dilate(red, np.ones((5, 5), np.uint8), iterations=1)
                cleaned[dilated > 0] = (240, 240, 240)
                self._assign_digits_to_circles(
                    result, self._ocr_frame_digits(cleaned), circles,
                )
        N = len(circles)
        valid_range = set(range(1, N + 1))
        for i, n in enumerate(result):
            if n is None:
                guess = self._ocr_single_circle(frame, circles[i])
                if guess is not None and guess in valid_range:
                    # Avoid duplicate-number collisions with already-assigned circles.
                    if guess not in {x for j, x in enumerate(result) if j != i and x is not None}:
                        result[i] = guess
        self._fill_by_elimination(result, N)
        return result

    def _detected_connections(
        self, frame: np.ndarray, circles: List[Dict], threshold: float = 0.7,
    ) -> set:
        if len(circles) < 2:
            return set()
        red = self._red_mask(frame)
        if not red.any():
            return set()

        h, w = red.shape
        red_clean = red.copy()
        for c in circles:
            cv2.circle(
                red_clean, c['center'],
                int(c['radius'] * 1.15) + 3, 0, -1,
            )
        if not red_clean.any():
            return set()
        red_close = cv2.morphologyEx(
            red_clean, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8),
        )

        touch_thresh = max(15, int(min(h, w) * 0.04))
        min_size = max(20, int(min(h, w) * 0.015))

        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
            red_close, connectivity=8,
        )
        conns: set = set()
        for cid in range(1, n_comp):
            if stats[cid, cv2.CC_STAT_AREA] < min_size:
                continue
            ys, xs = np.where(labels == cid)
            if xs.size == 0:
                continue
            touched: List[int] = []
            for idx, c in enumerate(circles):
                cx, cy = c['center']
                ds = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
                if float(np.abs(ds - c['radius']).min()) <= touch_thresh:
                    touched.append(idx)
            if len(touched) < 2:
                continue
            if len(touched) == 2:
                conns.add(frozenset((touched[0], touched[1])))
                continue
            sub_pairs = self._pair_component_endpoints(
                labels == cid, circles, touched, touch_thresh,
            )
            for p in sub_pairs:
                conns.add(p)
        return conns

    @staticmethod
    def _endpoint_direction(
        skel: np.ndarray, y0: int, x0: int, steps: int = 10,
    ) -> Tuple[float, float]:
        """Walk along the skeleton from (y0, x0) and return the unit direction
        toward where the line is heading (a few pixels in)."""
        h, w = skel.shape
        visited = {(y0, x0)}
        cur = (y0, x0)
        for _ in range(steps):
            found = None
            cy, cx = cur
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < h and 0 <= nx < w
                        and skel[ny, nx] and (ny, nx) not in visited
                    ):
                        found = (ny, nx)
                        break
                if found:
                    break
            if found is None:
                break
            visited.add(found)
            cur = found
        dy = cur[0] - y0
        dx = cur[1] - x0
        norm = (dx * dx + dy * dy) ** 0.5
        if norm == 0:
            return (0.0, 0.0)
        return (dx / norm, dy / norm)

    @staticmethod
    def _best_matching(vectors: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Pair indices 0..K-1 to minimise sum of pairwise dot products.

        Anti-parallel direction vectors → most-negative dot product → identifies
        the two endpoints that belong to a single straight-through line.
        """
        n = len(vectors)
        if n < 2:
            return []

        def dot(a: int, b: int) -> float:
            return vectors[a][0] * vectors[b][0] + vectors[a][1] * vectors[b][1]

        if n - (n % 2) > 10:
            remaining = list(range(n - (n % 2)))
            greedy: List[Tuple[int, int]] = []
            while len(remaining) >= 2:
                bi, bj, bs = 0, 1, float('inf')
                for i in range(len(remaining)):
                    for j in range(i + 1, len(remaining)):
                        s = dot(remaining[i], remaining[j])
                        if s < bs:
                            bs, bi, bj = s, i, j
                greedy.append((remaining[bi], remaining[bj]))
                for k in sorted((bi, bj), reverse=True):
                    remaining.pop(k)
            return greedy

        def matchings(indices: List[int]):
            if len(indices) < 2:
                yield []
                return
            first = indices[0]
            for i in range(1, len(indices)):
                pair = (first, indices[i])
                rest = indices[1:i] + indices[i + 1:]
                for sub in matchings(rest):
                    yield [pair] + sub

        best, best_score = None, float('inf')
        for m in matchings(list(range(n - (n % 2)))):
            s = 0.0
            for a, b in m:
                s += vectors[a][0] * vectors[b][0] + vectors[a][1] * vectors[b][1]
            if s < best_score:
                best_score = s
                best = m
        return best or []

    def _pair_component_endpoints(
        self,
        comp_mask: np.ndarray,
        circles: List[Dict],
        touched_ids: List[int],
        touch_thresh: int,
    ) -> List[frozenset]:
        """Split a multi-touch component by pairing skeleton endpoints.
        """
        try:
            from skimage.morphology import skeletonize
        except Exception:
            ordered = list(touched_ids)
            return [
                frozenset((ordered[i], ordered[i + 1]))
                for i in range(len(ordered) - 1)
            ]
        skel = skeletonize(comp_mask.astype(bool)).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        nbr = cv2.filter2D(skel, -1, kernel)
        ep_yx = np.argwhere((skel == 1) & (nbr == 1))
        endpoints: List[Tuple[int, int, int]] = []  # (circle_idx, y, x)
        for (y, x) in ep_yx:
            best_idx, best_d = -1, float('inf')
            for idx in touched_ids:
                c = circles[idx]
                cx, cy = c['center']
                d = abs(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 - c['radius'])
                if d < best_d:
                    best_d = d
                    best_idx = idx
            if best_d <= touch_thresh:
                endpoints.append((best_idx, int(y), int(x)))
        if len(endpoints) < 2:
            return []
        vectors = [self._endpoint_direction(skel, y, x) for (_, y, x) in endpoints]
        matching = self._best_matching(vectors)
        pairs: List[frozenset] = []
        for a, b in matching:
            ci = endpoints[a][0]
            cj = endpoints[b][0]
            if ci != cj:
                pairs.append(frozenset((ci, cj)))
        return pairs

    def _mask_circles_and_red(
        self, frame: np.ndarray, circles: List[Dict],
    ) -> np.ndarray:
        """uint8 mask: 255 where kept, 0 inside any circle (with margin) or on red lines."""
        h, w = frame.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        for c in circles:
            cv2.circle(mask, c['center'], int(c['radius'] * 1.4) + 2, 0, -1)
        red = self._red_mask(frame)
        red_dil = cv2.dilate(red, np.ones((5, 5), np.uint8), iterations=1)
        mask[red_dil > 0] = 0
        return mask

    @staticmethod
    def _masked_similarity(
        a: np.ndarray, b: np.ndarray, mask: np.ndarray,
    ) -> float:
        if mask is None or not mask.any():
            return 1.0
        if a.shape != b.shape:
            return 0.0
        pa = a.astype(np.float32)[mask > 0]
        pb = b.astype(np.float32)[mask > 0]
        if pa.size == 0:
            return 1.0
        dist = np.linalg.norm(pa - pb, axis=1)
        max_dist = float(np.sqrt(3.0 * (255.0 ** 2)))
        return float(max(0.0, 1.0 - float(np.mean(dist)) / max_dist))

    @staticmethod
    def _lcs_length(seq_a: List, seq_b: List) -> int:
        n, m = len(seq_a), len(seq_b)
        if n == 0 or m == 0:
            return 0
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n):
            for j in range(m):
                if seq_a[i] == seq_b[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[n][m]

    @staticmethod
    def _match_circles_to_gt(
        gen_circles: List[Dict], gt_circles: List[Dict],
    ) -> List[Optional[int]]:
        """Map each gen circle to nearest GT circle index (within ~3 GT radii)."""
        out: List[Optional[int]] = []
        for gc in gen_circles:
            gx, gy = gc['center']
            best_i, best_d = None, float('inf')
            for i, gt in enumerate(gt_circles):
                tx, ty = gt['center']
                d = (gx - tx) ** 2 + (gy - ty) ** 2
                limit = (max(gt['radius'], gc['radius']) * 3) ** 2
                if d <= limit and d < best_d:
                    best_d = d
                    best_i = i
            out.append(best_i)
        return out

    @staticmethod
    def _merge_canonical_circles(
        primary: List[Dict], extras: List[Dict],
    ) -> List[Dict]:
        """Append circles from `extras` that don't overlap any circle in `primary`."""
        merged = list(primary)
        for c in extras:
            cx, cy = c['center']
            r = c['radius']
            dup = False
            for k in merged:
                kx, ky = k['center']
                if (cx - kx) ** 2 + (cy - ky) ** 2 < (0.9 * (r + k['radius'])) ** 2:
                    dup = True
                    break
            if not dup:
                merged.append(c)
        return merged

    @staticmethod
    def _metadata_connection_reference(
        eval_info: Dict, frame_shape: Tuple[int, ...],
    ) -> Tuple[List[Dict], List[frozenset]]:
        """Build the ordered dot layout and expected segments from metadata.
        """
        import json

        metadata_path = eval_info.get('metadata_path', '')
        if not metadata_path:
            gt_path = eval_info.get('gt_path', '')
            if gt_path:
                base = gt_path if os.path.isdir(gt_path) else os.path.dirname(gt_path)
                metadata_path = os.path.join(base, 'metadata.json')
        if not metadata_path or not os.path.isfile(metadata_path):
            return [], []

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except (OSError, ValueError, TypeError):
            return [], []

        semantic = meta.get('semantic_ground_truth') or {}
        params = meta.get('parameters') or {}
        order = semantic.get('connection_order') or params.get('connection_order')
        if not isinstance(order, list) or len(order) < 2:
            return [], []

        dot_by_index: Dict[int, Dict] = {}
        objects = semantic.get('objects') or []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            idx = obj.get('index')
            center = obj.get('center')
            if isinstance(idx, int) and isinstance(center, (list, tuple)) and len(center) >= 2:
                dot_by_index[idx] = {
                    'center': center,
                    'radius': obj.get('radius', params.get('dot_radius', 45)),
                }

        if not dot_by_index:
            points = params.get('points') or []
            if not isinstance(points, list):
                return [], []
            for idx, center in enumerate(points):
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    dot_by_index[idx] = {
                        'center': center,
                        'radius': params.get('dot_radius', 45),
                    }

        canvas = (meta.get('generic_declarative_render') or {}).get('canvas') or {}
        frame_h, frame_w = frame_shape[:2]
        source_w = float(canvas.get('width') or frame_w)
        source_h = float(canvas.get('height') or frame_h)
        if source_w <= 0 or source_h <= 0:
            return [], []
        scale_x = frame_w / source_w
        scale_y = frame_h / source_h
        radius_scale = min(scale_x, scale_y)

        circles: List[Dict] = []
        try:
            ordered_indices = [int(idx) for idx in order]
        except (TypeError, ValueError):
            return [], []
        if len(set(ordered_indices)) != len(ordered_indices):
            return [], []
        for idx in ordered_indices:
            dot = dot_by_index.get(idx)
            if dot is None:
                return [], []
            center = dot['center']
            try:
                cx = int(round(float(center[0]) * scale_x))
                cy = int(round(float(center[1]) * scale_y))
                radius = max(1, int(round(float(dot['radius']) * radius_scale)))
            except (TypeError, ValueError):
                return [], []
            circles.append({'center': (cx, cy), 'radius': radius})

        expected = [frozenset((i, i + 1)) for i in range(len(circles) - 1)]
        return circles, expected

    def _detected_straight_connections(
        self, frame: np.ndarray, circles: List[Dict], threshold: float = 0.72,
    ) -> set:
        """Detect straight red segments between metadata-defined dot pairs.
        """
        if len(circles) < 2:
            return set()
        red = self._red_mask(frame)
        if not red.any():
            return set()

        min_radius = min(max(1, int(c['radius'])) for c in circles)
        search_radius = max(4, int(round(min_radius * 0.14)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * search_radius + 1, 2 * search_radius + 1),
        )
        expanded_red = cv2.dilate(red, kernel, iterations=1)
        h, w = red.shape
        found: set = set()

        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                a = np.asarray(circles[i]['center'], dtype=np.float32)
                b = np.asarray(circles[j]['center'], dtype=np.float32)
                delta = b - a
                distance = float(np.linalg.norm(delta))
                if distance < 1.0:
                    continue
                start = min(0.45, 1.05 * float(circles[i]['radius']) / distance)
                end = max(0.55, 1.0 - 1.05 * float(circles[j]['radius']) / distance)
                ts = np.linspace(start, end, 120, dtype=np.float32)
                samples = np.rint(a[None, :] + ts[:, None] * delta[None, :]).astype(int)
                xs = np.clip(samples[:, 0], 0, w - 1)
                ys = np.clip(samples[:, 1], 0, h - 1)
                coverage = float(np.mean(expanded_red[ys, xs] > 0))
                if coverage >= threshold:
                    found.add(frozenset((i, j)))
        return found

    def _extract_connection_sequence(
        self,
        frames: List[np.ndarray],
        reference_circles: List[Dict],
        max_frames: int = 30,
        straight_geometry: bool = False,
    ) -> Tuple[List[frozenset], set]:
        if not frames or len(reference_circles) < 2:
            return [], set()
        if len(frames) > max_frames:
            step = (len(frames) - 1) / (max_frames - 1)
            sample_idxs = sorted({int(round(i * step)) for i in range(max_frames)})
        else:
            sample_idxs = list(range(len(frames)))
        seen: set = set()
        first_seen: Dict[frozenset, int] = {}
        last_set: set = set()
        for fi in sample_idxs:
            frame = frames[fi]
            if straight_geometry:
                curr = self._detected_straight_connections(frame, reference_circles)
                for pair in curr - seen:
                    first_seen[pair] = int(fi)
                seen |= curr
                last_set = curr
                continue
            local_circles = self._detect_circles(frame)
            if len(local_circles) < 2:
                continue
            local_to_ref = self._match_circles_to_gt(local_circles, reference_circles)
            lines = self._detected_connections(frame, local_circles)
            curr: set = set()
            for pair in lines:
                a, b = list(pair)
                ra = local_to_ref[a] if a < len(local_to_ref) else None
                rb = local_to_ref[b] if b < len(local_to_ref) else None
                if ra is None or rb is None or ra == rb:
                    continue
                curr.add(frozenset((ra, rb)))
            for pair in curr - seen:
                first_seen[pair] = int(fi)
            seen |= curr
            last_set = curr

        by_frame: Dict[int, List[frozenset]] = {}
        for pair, fi in first_seen.items():
            by_frame.setdefault(fi, []).append(pair)
        sequence: List[frozenset] = []
        for fi in sorted(by_frame):
            batch = by_frame[fi]
            if len(batch) == 1:
                sequence.append(batch[0])
        return sequence, last_set

    # ----------------------------------------------------------- main entry

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        if not video_frames or gt_final_frame is None:
            self._last_task_details = {'error': 'missing_frames'}
            return 0.0

        gen_final = video_frames[-1]
        if gen_final.shape[:2] != gt_final_frame.shape[:2]:
            video_frames = [normalize_frame_size(f, gt_final_frame) for f in video_frames]
            gen_final = video_frames[-1]
        gt_final = gt_final_frame
        gt_first = gt_first_frame

        gt_video_frames: List[np.ndarray] = []
        if gt_frames:
            for f in gt_frames:
                if f.shape[:2] != gt_final_frame.shape[:2]:
                    gt_video_frames.append(normalize_frame_size(f, gt_final_frame))
                else:
                    gt_video_frames.append(f)


        gt_circles, metadata_seq = self._metadata_connection_reference(
            eval_info, gt_final.shape,
        )
        reference_source = 'metadata' if len(gt_circles) >= 2 else 'visual_gt'
        if len(gt_circles) < 2:
            gt_circles = self._detect_circles(gt_first) if gt_first is not None else []
            if len(gt_circles) < 2:
                gt_circles = self._detect_circles(gt_final)

        N = len(gt_circles)
        if N < 2:
            self._last_task_details = {'error': 'no_gt_circles', 'gt_circles': N}
            return 0.0
        gt_lines = max(1, N - 1)

        # --- GT reference connection sequence (in GT-index space) ---
        if metadata_seq:
            expected_seq = metadata_seq
            expected_set = set(metadata_seq)
        else:
            # The GT video draws lines 1→2, 2→3, …, N-1→N in order, so
            # its sequence is the OCR-free fallback reference.
            if gt_video_frames:
                ref_seq, ref_last = self._extract_connection_sequence(
                    gt_video_frames, gt_circles,
                )
            else:
                ref_seq, ref_last = [], set()
            if not ref_last:
                ref_last = self._detected_connections(gt_final, gt_circles)
                if not ref_seq:
                    ref_seq = list(ref_last)
            expected_set = ref_last
            expected_seq = ref_seq

        # --- Gen connection sequence in GT-index space ---
        gen_seq, gen_last = self._extract_connection_sequence(
            video_frames, gt_circles,
            straight_geometry=bool(metadata_seq),
        )

        # 1) Connection completeness
        detected_lines = max(1, len(gen_last))
        correct = len(gen_last & expected_set)
        completeness = correct / max(gt_lines, detected_lines)
        completeness = float(max(0.0, min(1.0, completeness)))

        # 2) Connection order penalty
        lcs = self._lcs_length(gen_seq, expected_seq)
        order_denom = max(gt_lines, len(gen_seq), len(expected_seq))
        order_penalty = lcs / order_denom if order_denom else 0.0
        order_penalty = float(max(0.0, min(1.0, order_penalty)))

        # --- Canonical gen circle set (first ∪ final) for OCR / consistency ---
        canon_gen = self._merge_canonical_circles(
            self._detect_circles(video_frames[0]),
            self._detect_circles(gen_final),
        )

        # 3) Numerical consistency — OCR-based; defaults to 1.0 if no OCR.
        ocr_ready = self._get_ocr_reader() is not None
        changed = 0
        ocr_per_frame: Dict[int, List[Optional[int]]] = {}
        if ocr_ready and canon_gen:
            # 4 samples (≈ first / one-third / two-thirds / last) keeps OCR
            # cost bounded while still catching mid-video digit drift.
            cs_count = max(2, min(len(video_frames), 4))
            if len(video_frames) > 1:
                cs_idxs = sorted({
                    int(round(i * (len(video_frames) - 1) / (cs_count - 1)))
                    for i in range(cs_count)
                })
            else:
                cs_idxs = [0]
            per_circle_vals: List[set] = [set() for _ in canon_gen]
            for fi in cs_idxs:
                try:
                    nums = self._circle_numbers(video_frames[fi], canon_gen)
                except Exception:
                    nums = [None] * len(canon_gen)
                ocr_per_frame[int(fi)] = list(nums)
                for i, n in enumerate(nums):
                    if n is not None:
                        per_circle_vals[i].add(n)
            changed = sum(1 for vals in per_circle_vals if len(vals) > 1)
        numerical_penalty = float(max(0.0, 1.0 - 0.05 * changed))

        # 4) Consistency: mask circles + red lines, compare similarity
        gen_keep = self._mask_circles_and_red(gen_final, canon_gen or gt_circles)
        gt_keep = self._mask_circles_and_red(gt_final, gt_circles)
        combined = cv2.bitwise_and(gen_keep, gt_keep)
        similarity = self._masked_similarity(gt_final, gen_final, combined)
        consistency_subtract = 0.10 if similarity < 0.7 else 0.0

        final_score = (
            completeness * order_penalty * numerical_penalty - consistency_subtract
        )
        final_score = float(max(0.0, min(1.0, final_score)))

        self._last_task_details = {
            'connection_completeness': completeness,
            'connection_order_penalty': order_penalty,
            'numerical_consistency_penalty': numerical_penalty,
            'mask_region_similarity': float(similarity),
            'consistency_subtract': consistency_subtract,
            'gt_circles': int(N),
            'gt_lines_expected': int(gt_lines),
            'gt_reference_lines': int(len(expected_set)),
            'gt_reference_sequence': int(len(expected_seq)),
            'detected_lines_final': int(len(gen_last)),
            'detected_sequence_length': int(len(gen_seq)),
            'correct_connections': int(correct),
            'order_lcs': int(lcs),
            'order_denominator': int(order_denom),
            'numerical_changed_circles': int(changed),
            'ocr_available': ocr_ready,
            'ocr_per_frame': ocr_per_frame,
            'canon_gen_centers': [list(c['center']) for c in canon_gen],
            'reference_source': reference_source,
        }
        return final_score


# Export mapping for this batch
IN_DOMAIN_50_EVALUATORS_PART4 = {
    'O-21_construction_blueprint_data-generator': ConstructionBlueprintEvaluator,
    'O-23_domino_chain_branch_path_prediction_data-generator': DominoChainBranchEvaluator,
    'O-24_domino_chain_gap_analysis_data-generator': DominoChainGapEvaluator,
    'O-25_LEGO_construction_assembly_data-generator': LEGOConstructionEvaluator,
    'O-29_ballcolor_data-generator': BallColorEvaluator,
    'O-30_bookshelf_data-generator': BookshelfEvaluator,
    'O-31_ball_eating_data-generator': BallEatingEvaluator,
    'O-32_rolling_ball_data-generator': RollingBallEvaluator,
    'O-33_counting_object_data-generator': CountingObjectEvaluator,
    'O-34_dot_to_dot_task_data-generator': DotToDotEvaluator,
}
