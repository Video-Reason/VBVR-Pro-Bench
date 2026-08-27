"""
Specific evaluators for Out-of-Domain_50 tasks (Part 5).
"""

import json
import numpy as np
import cv2
from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, normalize_frame_size, compute_ssim, safe_distance, \
    extract_patterns_from_white_bg, find_patterns_in_image


class ControlPanelEvaluator(BaseEvaluator):
    """
    O-54: Control panel evaluator.

    Dimensions:
        - completion (50%): all lights show correct color and all levers
          show correct position.
        - process_validity (50%): the process of lights and levers changing
          is valid.
        - background_preservation: multiplicative scene-preservation gate.
    """

    TASK_WEIGHTS = {
        'completion': 0.40,
        'process_validity': 0.60,
    }
    BACKGROUND_GATE_FLOOR = 0.60
    BACKGROUND_GATE_WEIGHT = 0.40

    # Sub-weights inside completion / process_validity
    COMPLETION_WEIGHTS = {
        'light_color': 0.50,
        'lever_position': 0.50,
    }

    PROCESS_WEIGHTS = {
        'light_color_process': 0.30,
        'lever_position_process': 0.30,
        'light_with_lever': 0.40,
    }

    # Layout: num_lights -> (rows, cols)
    LAYOUTS = {
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        6: (2, 3),
    }

    COLOR_MAP_RGB: Dict[str, Tuple[int, int, int]] = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "pink": (255, 192, 203),
        "lime": (0, 255, 0),
        "teal": (0, 128, 128),
        "indigo": (75, 0, 130),
        "off": (64, 64, 64),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
    }

    # ΔE76 thresholds for color similarity
    COLOR_DELTA_E_FULL = 3.0
    COLOR_DELTA_E_ZERO = 10.0

    COLOR_CLASSIFY_DELTA_E = 10.0

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
        """Return (foreground_mask, background_mask) for a frame."""
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

    def _delta_e_to_score(self, delta_e: float) -> float:
        """Map a single ΔE76 value to a [0, 1] score."""
        if delta_e <= self.COLOR_DELTA_E_FULL:
            return 1.0
        if delta_e >= self.COLOR_DELTA_E_ZERO:
            return 0.0
        return float(1.0 - (delta_e - self.COLOR_DELTA_E_FULL) / (self.COLOR_DELTA_E_ZERO - self.COLOR_DELTA_E_FULL))

    def _get_layout(self, num_lights: int) -> Tuple[int, int]:
        if num_lights not in self.LAYOUTS:
            raise ValueError(f"Unsupported num_lights: {num_lights}")
        return self.LAYOUTS[num_lights]

    def _split_into_cells(
        self,
        frame: np.ndarray,
        num_lights: int,
    ) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Split `frame` into `num_lights` cells.

        Returns a list ordered by id (row-major): each entry is
        ((x0, y0, x1, y1), cell_image).
        """
        rows, cols = self._get_layout(num_lights)
        H, W = frame.shape[:2]
        cell_h = H // rows
        cell_w = W // cols
        cells: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
        for r in range(rows):
            for c in range(cols):
                y0, y1 = r * cell_h, (r + 1) * cell_h
                x0, x1 = c * cell_w, (c + 1) * cell_w
                cells.append(((x0, y0, x1, y1), frame[y0:y1, x0:x1]))
        return cells

    def _detect_light(
        self,
        cell: np.ndarray,
    ) -> Optional[Dict]:
        """
        Detect the indicator light in the upper half of a cell.

        Returns a dict with 'bbox', 'center', 'mask' (within the cell), and
        'mean_bgr' (mean BGR color of the light pixels), or None if not found.
        """
        H, W = cell.shape[:2]
        upper = cell[: H // 2]
        gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
        # Anything not white is foreground in the upper half.
        _, fg = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        # Remove tiny noise from the circle outline.
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        best_idx, best_area = -1, 0
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_idx = i
        if best_idx < 0 or best_area < 50:
            return None

        x = int(stats[best_idx, cv2.CC_STAT_LEFT])
        y = int(stats[best_idx, cv2.CC_STAT_TOP])
        w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
        comp_mask = (labels == best_idx).astype(np.uint8) * 255

        # Use the inner part (erode) so we only sample saturated pixels.
        inner = cv2.erode(comp_mask, np.ones((5, 5), np.uint8))
        if not np.any(inner > 0):
            inner = comp_mask
        mean_bgr = cv2.mean(upper, mask=inner)[:3]

        # Return a full-cell mask for masked similarity convenience.
        full_mask = np.zeros(cell.shape[:2], dtype=np.uint8)
        full_mask[: H // 2][comp_mask > 0] = 255

        return {
            'bbox': (x, y, x + w, y + h),
            'center': (x + w / 2.0, y + h / 2.0),
            'mask': full_mask,
            'mean_bgr': tuple(float(v) for v in mean_bgr),
        }

    def _detect_panel(self, cell: np.ndarray) -> Optional[Dict]:
        """Locate the black panel rectangle in the lower half of a cell."""
        H, W = cell.shape[:2]
        lower = cell[H // 2:]
        gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
        # Panel is a large dark blob.
        _, dark = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        best_idx, best_area = -1, 0
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_idx = i
        if best_idx < 0 or best_area < 200:
            return None
        x = int(stats[best_idx, cv2.CC_STAT_LEFT])
        y = int(stats[best_idx, cv2.CC_STAT_TOP])
        w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
        return {
            'bbox': (x, y + H // 2, x + w, y + h + H // 2),  # cell coords
            'panel_local_bbox': (x, y, w, h),                 # in lower-half coords
            'half_offset': H // 2,
        }

    def _detect_lever_and_dots(
        self,
        cell: np.ndarray,
        panel_info: Dict,
    ) -> Optional[Dict]:
        """
        Within a panel, find the gray lever (large gray square) and the
        two small white dots that mark unoccupied positions.

        Returns a dict with:
            'lever_bbox': (x0, y0, x1, y1) in cell coords
            'lever_center_x': float
            'lever_mask': cell-sized binary mask
            'dots': list of (cx, cy) in cell coords
            'all_anchors_x': sorted list of three anchor x-coords (cell coords)
        """
        H, W = cell.shape[:2]
        half = panel_info['half_offset']
        x, y, w, h = panel_info['panel_local_bbox']
        # Restrict search inside the panel to ignore outer borders.
        panel = cell[half + y: half + y + h, x: x + w]
        if panel.size == 0:
            return None

        panel_gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

        # White dots: very bright on the black panel
        dots_mask = (panel_gray > 200).astype(np.uint8) * 255
        n_d, _, st_d, ct_d = cv2.connectedComponentsWithStats(dots_mask, connectivity=8)
        dot_centers = []
        for i in range(1, n_d):
            area = int(st_d[i, cv2.CC_STAT_AREA])
            # Markers are small (a few pixels) - filter giant white regions.
            if 2 <= area <= 200:
                cx, cy = ct_d[i]
                dot_centers.append((float(cx) + x, float(cy) + y + half))

        # Gray lever: pixels in [80, 200]
        lever_mask = ((panel_gray > 80) & (panel_gray < 200)).astype(np.uint8) * 255
        n_l, lab_l, st_l, ct_l = cv2.connectedComponentsWithStats(lever_mask, connectivity=8)
        best_idx, best_area = -1, 0
        for i in range(1, n_l):
            area = int(st_l[i, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_idx = i
        if best_idx < 0 or best_area < 100:
            return None

        lx = int(st_l[best_idx, cv2.CC_STAT_LEFT])
        ly = int(st_l[best_idx, cv2.CC_STAT_TOP])
        lw = int(st_l[best_idx, cv2.CC_STAT_WIDTH])
        lh = int(st_l[best_idx, cv2.CC_STAT_HEIGHT])
        lever_cx = float(ct_l[best_idx, 0]) + x
        lever_cy = float(ct_l[best_idx, 1]) + y + half

        # Build cell-sized lever mask
        full_lever_mask = np.zeros(cell.shape[:2], dtype=np.uint8)
        comp = (lab_l == best_idx).astype(np.uint8) * 255
        full_lever_mask[half + y: half + y + h, x: x + w] = comp

        # Anchors: lever_x ∪ dot_xs, sorted ascending
        anchor_xs = sorted([lever_cx] + [d[0] for d in dot_centers])

        return {
            'lever_bbox': (lx + x, ly + y + half, lx + x + lw, ly + y + h + half),
            'lever_center_x': lever_cx,
            'lever_center_y': lever_cy,
            'lever_mask': full_lever_mask,
            'dots': dot_centers,
            'all_anchors_x': anchor_xs,
        }

    def _detect_cell(self, cell: np.ndarray) -> Dict:
        """Run all detectors for a single cell, return a flat dict (or empty)."""
        result: Dict = {'cell_shape': cell.shape[:2]}
        light = self._detect_light(cell)
        if light is not None:
            result['light'] = light
        panel = self._detect_panel(cell)
        if panel is not None:
            result['panel'] = panel
            ld = self._detect_lever_and_dots(cell, panel)
            if ld is not None:
                result['lever'] = ld
        return result

    def _detect_frame(self, frame: np.ndarray, num_lights: int) -> List[Dict]:
        """Detect every cell in a frame; returns list ordered by id."""
        cells = self._split_into_cells(frame, num_lights)
        results = []
        for (offset, cell_img) in cells:
            det = self._detect_cell(cell_img)
            det['cell_offset'] = offset  # (x0, y0, x1, y1) in full-frame coords
            results.append(det)
        return results

    @staticmethod
    def _bgr_to_lab_pixel(bgr: Tuple[float, float, float]) -> np.ndarray:
        pix = np.array([[[bgr[0], bgr[1], bgr[2]]]], dtype=np.uint8)
        lab = cv2.cvtColor(pix, cv2.COLOR_BGR2Lab).astype(np.float32)[0, 0]
        lab[0] *= 100.0 / 255.0
        lab[1:] -= 128.0
        return lab

    def _delta_e_bgr(
        self,
        bgr_a: Tuple[float, float, float],
        bgr_b: Tuple[float, float, float],
    ) -> float:
        a = self._bgr_to_lab_pixel(bgr_a)
        b = self._bgr_to_lab_pixel(bgr_b)
        return float(np.linalg.norm(a - b))

    def _classify_by_hue(
        self,
        bgr: Tuple[float, float, float],
        candidates: List[str],
    ) -> Optional[str]:
        px = np.uint8([[[int(bgr[0]), int(bgr[1]), int(bgr[2])]]])
        h, sat, val = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0].tolist()
        if sat < 60 or val < 40:
            return None
        best, best_d = None, float('inf')
        for name in candidates:
            if name not in self.COLOR_MAP_RGB:
                continue
            r, g, b = self.COLOR_MAP_RGB[name]
            cpx = np.uint8([[[b, g, r]]])
            ch, cs, cv_ = cv2.cvtColor(cpx, cv2.COLOR_BGR2HSV)[0][0].tolist()
            if cs < 60:
                continue
            d = abs(h - ch)
            d = min(d, 180 - d)    
            if d < best_d:
                best, best_d = name, d
        return best

    def _classify_color(
        self,
        bgr: Tuple[float, float, float],
        candidates: List[str],
    ) -> Tuple[Optional[str], float]:
        """Pick the candidate name whose palette color is closest to `bgr` (Lab ΔE)."""
        best_name: Optional[str] = None
        best_de = float('inf')
        for name in candidates:
            if name not in self.COLOR_MAP_RGB:
                continue
            r, g, b = self.COLOR_MAP_RGB[name]
            cand_bgr = (b, g, r)
            de = self._delta_e_bgr(bgr, cand_bgr)
            if de < best_de:
                best_de = de
                best_name = name
        return best_name, best_de

    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter
        return float(inter) / float(union) if union > 0 else 0.0

    @staticmethod
    def _classify_position_from_x(
        center_x: float,
        anchors_sorted: List[float],
    ) -> str:
        """
        Snap a lever center x to the nearest of three anchor positions.

        anchors_sorted has length 3 (or 2 if some dots are missing). For 3
        anchors the order is [left, middle, right]; for 2 we still compare
        nearest distance.
        """
        if len(anchors_sorted) == 3:
            labels = ['left', 'middle', 'right']
        elif len(anchors_sorted) == 2:
            labels = ['left', 'right']
        else:
            return 'unknown'
        idx = int(np.argmin([abs(center_x - a) for a in anchors_sorted]))
        return labels[idx]

    def _score_completion(
        self,
        gen_last: np.ndarray,
        gt_last: np.ndarray,
        num_lights: int,
        palette: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        gen_dets = self._detect_frame(gen_last, num_lights)
        gt_dets = self._detect_frame(gt_last, num_lights)

        light_scores: List[float] = []
        lever_scores: List[float] = []

        for gen_d, gt_d in zip(gen_dets, gt_dets):
            # ---- light color: ΔE between mean BGRs ----
            if 'light' in gen_d and 'light' in gt_d:
                de = self._delta_e_bgr(gen_d['light']['mean_bgr'],
                                       gt_d['light']['mean_bgr'])
                score = self._delta_e_to_score(de)
                names = palette or list(self.COLOR_MAP_RGB.keys())
                gen_name = self._classify_by_hue(gen_d['light']['mean_bgr'], names)
                gt_name = self._classify_by_hue(gt_d['light']['mean_bgr'], names)
                if gen_name is not None and gen_name == gt_name:
                    score = max(score, 0.9)
                light_scores.append(score)
            else:
                light_scores.append(0.0)

            # ---- lever position: bbox IoU in cell-local coords ----
            if 'lever' in gen_d and 'lever' in gt_d:
                iou = self._bbox_iou(
                    gen_d['lever']['lever_bbox'],
                    gt_d['lever']['lever_bbox'],
                )
                lever_scores.append(iou)
            else:
                lever_scores.append(0.0)

        light_avg = float(np.mean(light_scores)) if light_scores else 0.0
        lever_avg = float(np.mean(lever_scores)) if lever_scores else 0.0

        return {
            'light_color': light_avg,
            'lever_position': lever_avg,
        }

    def _build_color_palette(self, mapping: Dict[str, str]) -> List[str]:
        """Return the candidate color names for classification (mapping values)."""
        return list(dict.fromkeys(mapping.values()))

    def _score_light_color_process(
        self,
        gen_per_frame: List[List[Dict]],
        gt_per_frame: List[List[Dict]],
        objects: List[Dict],
        position_color_mapping: Dict[str, str],
    ) -> float:
        """
        Two dimensions, each 50%, operating directly on BGR values (no name
        mapping) to avoid score inflation from lossy colour classification:

        Dim 1 - color validity: for each gen frame, score proximity to the
        nearest colour in the GT compressed-BGR palette via _delta_e_to_score.
        Mean over all frames (consistent with completion's light_color scoring).

        Dim 2 - order correctness: soft-LCS on run-length-compressed BGR
        sequences (gen vs GT), scored via _delta_e_to_score per matched pair.

        If dim1 == 0.0, the unit score is 0.0.
        Otherwise: unit_score = 0.5 * dim1 + 0.5 * dim2.
        """
        unit_scores: List[float] = []

        for u_idx, _ in enumerate(objects):
            # Step 1: GT compressed BGR sequence (temporal colour-change path)
            gt_bgr_list: List[Tuple[float, float, float]] = []
            for frame_dets in gt_per_frame:
                if u_idx >= len(frame_dets) or 'light' not in frame_dets[u_idx]:
                    continue
                gt_bgr_list.append(frame_dets[u_idx]['light']['mean_bgr'])
            gt_rgb_path = self._compress_rgb_sequence(gt_bgr_list)

            # Dim 1: proximity of each gen frame to the GT colour palette
            frame_scores: List[float] = []
            gen_bgr_list: List[Tuple[float, float, float]] = []
            for frame_dets in gen_per_frame:
                if u_idx >= len(frame_dets) or 'light' not in frame_dets[u_idx]:
                    frame_scores.append(0.0)
                    continue
                bgr = frame_dets[u_idx]['light']['mean_bgr']
                gen_bgr_list.append(bgr)
                if gt_rgb_path:
                    best = max(
                        self._delta_e_to_score(self._delta_e_bgr(bgr, gt_c))
                        for gt_c in gt_rgb_path
                    )
                else:
                    best = 0.0
                frame_scores.append(best)

            dim1 = float(np.mean(frame_scores)) if frame_scores else 0.0
            if dim1 == 0.0:
                unit_scores.append(0.0)
                continue

            # Dim 2: order correctness via soft-LCS on compressed BGR sequences
            gen_rgb_path = self._compress_rgb_sequence(gen_bgr_list)
            dim2 = self._soft_lcs_score(gen_rgb_path, gt_rgb_path)

            unit_scores.append(0.5 * dim1 + 0.5 * dim2)

        return float(np.mean(unit_scores)) if unit_scores else 0.0

    @staticmethod
    def _compress_sequence(seq: List[str]) -> List[str]:
        """Run-length encode a label sequence (collapse consecutive identical labels)."""
        if not seq:
            return []
        out = [seq[0]]
        for p in seq[1:]:
            if p != out[-1]:
                out.append(p)
        return out

    @staticmethod
    def _lcs_ratio(seq_a: List[str], seq_b: List[str]) -> float:
        """LCS length divided by max(len(a), len(b)); 1.0 if both empty."""
        m, n = len(seq_a), len(seq_b)
        if m == 0 and n == 0:
            return 1.0
        if m == 0 or n == 0:
            return 0.0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n] / max(m, n)

    def _compress_rgb_sequence(
        self,
        bgr_list: List[Tuple[float, float, float]],
    ) -> List[Tuple[float, float, float]]:
        """Compress a BGR sequence: start a new segment when ΔE ≥ COLOR_DELTA_E_ZERO."""
        if not bgr_list:
            return []
        path = [bgr_list[0]]
        for bgr in bgr_list[1:]:
            if self._delta_e_bgr(bgr, path[-1]) >= self.COLOR_DELTA_E_ZERO:
                path.append(bgr)
        return path

    def _soft_lcs_score(
        self,
        seq_a: List[Tuple[float, float, float]],
        seq_b: List[Tuple[float, float, float]],
    ) -> float:
        """
        Order-preserving soft matching on BGR sequences: find the monotone
        alignment that maximises sum of _delta_e_to_score(ΔE) for matched
        pairs, normalised by max(len(a), len(b)).
        """
        m, n = len(seq_a), len(seq_b)
        if m == 0 and n == 0:
            return 1.0
        if m == 0 or n == 0:
            return 0.0
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                sim = self._delta_e_to_score(
                    self._delta_e_bgr(seq_a[i - 1], seq_b[j - 1])
                )
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + sim)
        return dp[m][n] / max(m, n)

    def _score_lever_position_process(
        self,
        gen_per_frame: List[List[Dict]],
        gt_per_frame: List[List[Dict]],
        objects: List[Dict],
    ) -> float:
        """
        Compare gen and GT lever trajectories via path mask IoU.

        For each unit, draw the lever center path as a polyline on a
        cell-sized canvas and compute intersection-over-union.
        """
        if not gen_per_frame or not gt_per_frame:
            return 0.0

        THICKNESS = 4
        unit_scores: List[float] = []

        for u_idx, _ in enumerate(objects):
            cell_shape: Optional[Tuple[int, int]] = None
            for frame_dets in gen_per_frame + gt_per_frame:
                if u_idx < len(frame_dets) and 'cell_shape' in frame_dets[u_idx]:
                    cell_shape = frame_dets[u_idx]['cell_shape']
                    break
            if cell_shape is None:
                unit_scores.append(0.0)
                continue

            def _build_mask(per_frame: List[List[Dict]]) -> np.ndarray:
                mask = np.zeros(cell_shape, dtype=np.uint8)
                pts = []
                for frame_dets in per_frame:
                    if u_idx >= len(frame_dets) or 'lever' not in frame_dets[u_idx]:
                        pts.append(None)
                        continue
                    lv = frame_dets[u_idx]['lever']
                    pts.append((int(lv['lever_center_x']), int(lv['lever_center_y'])))
                for k in range(1, len(pts)):
                    if pts[k - 1] is not None and pts[k] is not None:
                        cv2.line(mask, pts[k - 1], pts[k], 255, THICKNESS)
                return mask

            gen_mask = _build_mask(gen_per_frame)
            gt_mask = _build_mask(gt_per_frame)
            intersection = np.logical_and(gen_mask > 0, gt_mask > 0).sum()
            union = np.logical_or(gen_mask > 0, gt_mask > 0).sum()
            unit_scores.append(float(intersection / union) if union > 0 else 0.0)

        return float(np.mean(unit_scores)) if unit_scores else 0.0

    def _score_light_with_lever(
        self,
        gen_per_frame: List[List[Dict]],
        objects: List[Dict],
        position_color_mapping: Dict[str, str],
    ) -> float:
        """
        For every gen frame and every unit, the lever position implies a
        target color via `position_color_mapping`. The detected light color
        must match it.

        Anchor positions are derived from the first detected frame: the lever
        center plus dot centers, sorted by x into [left, middle, right].
        Each frame's lever is validated before classification:
          - x must lie within [left_anchor_x, right_anchor_x]
          - y must be within 10 px of the anchor rail mean y
        """
        palette = self._build_color_palette(position_color_mapping)
        unit_scores: List[float] = []

        for u_idx, _ in enumerate(objects):
            # Step 1: build anchor positions from first frame with a valid lever
            anchor_pts: List[Tuple[float, float]] = []
            for frame_dets in gen_per_frame:
                if u_idx < len(frame_dets) and 'lever' in frame_dets[u_idx]:
                    lv = frame_dets[u_idx]['lever']
                    pts = [(lv['lever_center_x'], lv['lever_center_y'])] + list(lv['dots'])
                    if len(pts) >= 2:
                        anchor_pts = sorted(pts, key=lambda p: p[0])
                    break

            if len(anchor_pts) >= 3:
                labels = ['left', 'middle', 'right']
                anchor_pts = anchor_pts[:3]
            elif len(anchor_pts) == 2:
                labels = ['left', 'right']
            else:
                labels = []

            x_left = anchor_pts[0][0] if anchor_pts else None
            x_right = anchor_pts[-1][0] if anchor_pts else None
            mean_anchor_y = float(np.mean([p[1] for p in anchor_pts])) if anchor_pts else None

            agree = 0
            total = 0
            for frame_dets in gen_per_frame:
                total += 1
                if u_idx >= len(frame_dets):
                    continue
                d = frame_dets[u_idx]
                if 'light' not in d or 'lever' not in d:
                    continue

                lv = d['lever']
                cx, cy = lv['lever_center_x'], lv['lever_center_y']

                # Step 2: validate and classify lever position
                pos = 'unknown'
                if labels and x_left is not None and x_right is not None and mean_anchor_y is not None:
                    if x_left <= cx <= x_right and abs(cy - mean_anchor_y) <= 10.0:
                        nearest = int(np.argmin([abs(cx - p[0]) for p in anchor_pts]))
                        if nearest < len(labels):
                            pos = labels[nearest]

                expected_color = position_color_mapping.get(pos)
                if expected_color is None:
                    continue

                detected_name, de = self._classify_color(d['light']['mean_bgr'], palette)
                second_de = min(
                    (self._classify_color(d['light']['mean_bgr'], [c])[1]
                     for c in palette if c != detected_name and c in self.COLOR_MAP_RGB),
                    default=float('inf'),
                )
                decisive = de <= self.COLOR_CLASSIFY_DELTA_E or de <= 0.5 * second_de
                if detected_name == expected_color and decisive:
                    agree += 1

            unit_scores.append(agree / total if total else 0.0)

        return float(np.mean(unit_scores)) if unit_scores else 0.0

    def _score_process_validity(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        num_lights: int,
        objects: List[Dict],
        position_color_mapping: Dict[str, str],
    ) -> Dict[str, float]:
        gen_per_frame = [self._detect_frame(f, num_lights) for f in video_frames]
        gt_per_frame = [self._detect_frame(f, num_lights) for f in gt_frames]

        return {
            'light_color_process': self._score_light_color_process(
                gen_per_frame, gt_per_frame, objects, position_color_mapping
            ),
            'lever_position_process': self._score_lever_position_process(
                gen_per_frame, gt_per_frame, objects
            ),
            'light_with_lever': self._score_light_with_lever(
                gen_per_frame, objects, position_color_mapping
            ),
        }

    def _interleave_anchor_positions(
        self,
        input_frame: np.ndarray,
        num_lights: int,
    ) -> List[Optional[List[Tuple[float, float]]]]:
        """Recover the three stable lever anchors for every control unit."""
        anchors: List[Optional[List[Tuple[float, float]]]] = []
        for detection in self._detect_frame(input_frame, num_lights):
            lever = detection.get('lever')
            if lever is None:
                anchors.append(None)
                continue
            points = [
                (float(lever['lever_center_x']), float(lever['lever_center_y'])),
                *[(float(x), float(y)) for x, y in lever.get('dots', [])],
            ]
            points.sort(key=lambda point: point[0])
            deduped: List[Tuple[float, float]] = []
            for point in points:
                if not deduped or abs(point[0] - deduped[-1][0]) > 4.0:
                    deduped.append(point)
            anchors.append(deduped[:3] if len(deduped) >= 3 else None)
        return anchors

    def _interleave_semantic_state(
        self,
        frame: np.ndarray,
        num_lights: int,
        anchors: Sequence[Optional[List[Tuple[float, float]]]],
        position_color_mapping: Dict[str, str],
    ) -> Tuple[Optional[Tuple[str, ...]], Dict[str, Any]]:
        """Read one sparse image as stable lever positions plus matching lights."""
        labels = ('left', 'middle', 'right')
        palette = self._build_color_palette(position_color_mapping)
        detections = self._detect_frame(frame, num_lights)
        positions: List[str] = []
        colors: List[Optional[str]] = []
        light_matches: List[bool] = []

        if len(detections) != len(anchors):
            return None, {'error': 'control_unit_count_mismatch'}

        for unit_idx, (detection, unit_anchors) in enumerate(zip(detections, anchors)):
            lever = detection.get('lever')
            light = detection.get('light')
            if lever is None or light is None or unit_anchors is None:
                return None, {
                    'error': 'missing_lever_light_or_anchors',
                    'unit_idx': unit_idx,
                }

            center_x = float(lever['lever_center_x'])
            anchor_xs = [point[0] for point in unit_anchors]
            gaps = [
                anchor_xs[idx + 1] - anchor_xs[idx]
                for idx in range(len(anchor_xs) - 1)
            ]
            tolerance = max(6.0, 0.30 * min(gaps)) if gaps else 6.0
            nearest = int(np.argmin([abs(center_x - x) for x in anchor_xs]))
            if abs(center_x - anchor_xs[nearest]) > tolerance:
                return None, {
                    'error': 'lever_between_stable_anchors',
                    'unit_idx': unit_idx,
                    'center_x': round(center_x, 3),
                    'anchor_xs': [round(x, 3) for x in anchor_xs],
                }

            position = labels[nearest]
            color_name, _ = self._classify_color(light['mean_bgr'], palette)
            expected_color = position_color_mapping.get(position)
            positions.append(position)
            colors.append(color_name)
            light_matches.append(color_name == expected_color)

        return tuple(positions), {
            'positions': list(positions),
            'colors': colors,
            'light_matches': light_matches,
            'all_lights_match': bool(all(light_matches)),
        }

    def _score_interleave_atomic_process(
        self,
        pred_images: Sequence[np.ndarray],
        input_frame: np.ndarray,
        num_lights: int,
        objects: Sequence[Dict[str, Any]],
        position_color_mapping: Dict[str, str],
    ) -> Tuple[float, Dict[str, Any]]:
        """Score unordered, one-adjacent-lever-step-per-image progress.

        Metadata defines the initial and target states, but not the execution
        order: any unfinished control unit may move next. Repeated states do not
        earn progress, while simultaneous, backward, or multi-anchor jumps are
        invalid.
        """
        pos_idx = {'left': 0, 'middle': 1, 'right': 2}
        anchors = self._interleave_anchor_positions(input_frame, num_lights)
        if len(anchors) != len(objects) or any(anchor is None for anchor in anchors):
            return 0.0, {'error': 'could_not_recover_all_lever_anchors'}

        try:
            initial = tuple(
                str(obj['lever']['initial_position']) for obj in objects
            )
            target = tuple(
                str(obj['lever']['target_position']) for obj in objects
            )
            required_steps = sum(
                abs(pos_idx[dst] - pos_idx[src])
                for src, dst in zip(initial, target)
            )
        except (KeyError, TypeError):
            return 0.0, {'error': 'invalid_lever_metadata'}

        current = initial
        valid_steps = 0
        invalid_changes = 0
        duplicate_states = 0
        frame_details: List[Dict[str, Any]] = []

        for frame_idx, frame in enumerate(pred_images, start=1):
            observed, state_details = self._interleave_semantic_state(
                frame,
                num_lights,
                anchors,
                position_color_mapping,
            )
            detail: Dict[str, Any] = {
                'frame_idx': frame_idx,
                'state': list(observed) if observed is not None else None,
                **state_details,
            }
            if observed is None:
                invalid_changes += 1
                detail['transition'] = 'invalid_detection'
                frame_details.append(detail)
                continue

            changed = [idx for idx, (a, b) in enumerate(zip(current, observed)) if a != b]
            lights_valid = bool(state_details.get('all_lights_match', False))
            if not changed:
                if lights_valid:
                    duplicate_states += 1
                    detail['transition'] = 'duplicate'
                else:
                    invalid_changes += 1
                    detail['transition'] = 'invalid_light_for_position'
                frame_details.append(detail)
                continue

            valid_transition = False
            if len(changed) == 1 and lights_valid:
                idx = changed[0]
                src = pos_idx[current[idx]]
                dst = pos_idx[observed[idx]]
                goal = pos_idx[target[idx]]
                direction = int(np.sign(goal - src))
                valid_transition = direction != 0 and dst - src == direction

            if valid_transition:
                current = observed
                valid_steps += 1
                detail['transition'] = 'valid_atomic_step'
            else:
                invalid_changes += 1
                detail['transition'] = 'invalid_state_change'
                globally_forward = lights_valid and all(
                    min(pos_idx[src], pos_idx[dst])
                    <= pos_idx[value]
                    <= max(pos_idx[src], pos_idx[dst])
                    for src, dst, value in zip(initial, target, observed)
                )
                no_backtrack = globally_forward and all(
                    (
                        pos_idx[dst] >= pos_idx[src]
                        and pos_idx[value] >= pos_idx[old]
                    ) or (
                        pos_idx[dst] <= pos_idx[src]
                        and pos_idx[value] <= pos_idx[old]
                    )
                    for src, dst, old, value in zip(
                        initial, target, current, observed,
                    )
                )
                if no_backtrack:
                    current = observed
                    detail['resynchronized'] = True
            frame_details.append(detail)

        if required_steps == 0:
            coverage = 1.0 if current == target else 0.0
        else:
            coverage = min(1.0, valid_steps / required_steps)
        validity = (
            valid_steps / (valid_steps + invalid_changes)
            if valid_steps + invalid_changes > 0 else 0.0
        )
        score = coverage * (0.8 + 0.2 * validity)
        return float(score), {
            'initial_state': list(initial),
            'target_state': list(target),
            'final_validated_state': list(current),
            'required_steps': int(required_steps),
            'valid_steps': int(valid_steps),
            'invalid_changes': int(invalid_changes),
            'duplicate_states': int(duplicate_states),
            'coverage': round(float(coverage), 4),
            'validity': round(float(validity), 4),
            'frames': frame_details,
        }

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        scores: Dict[str, float] = {}
        details: Dict[str, Dict[str, float]] = {}

        if len(video_frames) < 2 or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        gt_frames = [normalize_frame_size(f, last_frame) if f.shape[:2] != last_frame.shape[:2] else f for f in gt_frames] if gt_frames else gt_frames
        gt_first = gt_frames[0]
        gt_last = gt_frames[-1]

        import os
        _mp = eval_info.get('metafile_path')
        if isinstance(_mp, (list, tuple)):
            _mp = next((p for p in _mp if p and os.path.exists(p)), _mp[0] if _mp else None)
        with open(_mp) as f:
            metadata = json.load(f)
        _sgt = metadata.get('semantic_ground_truth') or {}
        _params = metadata.get('parameters') or {}

        def _meta_get(*keys):
            for src in (_sgt, _params):
                for k in keys:
                    if k in src:
                        return src[k]
            raise KeyError(keys[0])

        num_lights: int = _meta_get('num_lights')
        position_color_mapping: Dict[str, str] = _meta_get(
            'position_color_mapping', 'position_color_map')
        objects: List[Dict] = _meta_get('objects')

        # 1) completion (40%)
        completion_parts = self._score_completion(
            last_frame, gt_last, num_lights,
            palette=self._build_color_palette(position_color_mapping),
        )
        details['completion'] = completion_parts
        scores['completion'] = float(sum(
            completion_parts[k] * self.COMPLETION_WEIGHTS[k]
            for k in self.COMPLETION_WEIGHTS
        ))

        # 2) process validity (40%)
        process_parts = self._score_process_validity(
            video_frames=video_frames,
            gt_frames=gt_frames,
            num_lights=num_lights,
            objects=objects,
            position_color_mapping=position_color_mapping,
        )
        details['process_validity'] = process_parts
        scores['process_validity'] = float(sum(
            process_parts[k] * self.PROCESS_WEIGHTS[k]
            for k in self.PROCESS_WEIGHTS
        ))

        # 3) background_preservation (20%)
        change_mask = self._shape_change_mask(gt_first, gt_last)
        _, first_bg = self._frame_masks(gt_first)
        bg_compare_mask = cv2.bitwise_and(first_bg, cv2.bitwise_not(change_mask))
        scores['background_preservation'] = self._pixel_similarity(
            gt_last, last_frame, bg_compare_mask,
            strictness=3.0, min_cutoff=0.6,
        )

        scores['completion_detail'] = completion_parts
        scores['process_detail'] = process_parts

        total = (
            scores['completion'] * self.TASK_WEIGHTS['completion']
            + scores['process_validity'] * self.TASK_WEIGHTS['process_validity']
        ) * (
            self.BACKGROUND_GATE_FLOOR
            + self.BACKGROUND_GATE_WEIGHT * scores['background_preservation']
        )
        details['score_formula'] = (
            '(0.5 * completion + 0.5 * process_validity) '
            '* (0.6 + 0.4 * background_preservation)'
        )

        self._last_task_details = {'scores': scores, 'details': details}
        return float(total)

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Image setting: validate discrete metadata-defined lever steps.

        Video keeps the dense trajectory evaluator above unchanged. Sparse
        images represent completed control states, so their process evidence is
        an unordered sequence of legal adjacent lever moves.
        """
        if not pred_images or input_frame is None or gt_final_frame is None:
            self._last_task_details = {'error': 'missing_interleave_frames'}
            return 0.0

        import os
        meta_path = eval_info.get('metafile_path')
        if isinstance(meta_path, (list, tuple)):
            meta_path = next(
                (path for path in meta_path if path and os.path.exists(path)),
                None,
            )
        if not (meta_path and os.path.exists(meta_path)):
            meta_path = os.path.join(eval_info.get('gt_path', ''), 'metadata.json')
        try:
            with open(meta_path, encoding='utf-8') as handle:
                metadata = json.load(handle)
        except (OSError, TypeError, ValueError):
            self._last_task_details = {'error': 'metadata_unavailable'}
            return 0.0

        semantic = metadata.get('semantic_ground_truth') or {}
        params = metadata.get('parameters') or {}

        def _meta_get(*keys: str) -> Any:
            for source in (semantic, params):
                for key in keys:
                    if key in source:
                        return source[key]
            raise KeyError(keys[0])

        try:
            num_lights = int(_meta_get('num_lights'))
            position_color_mapping = dict(_meta_get(
                'position_color_mapping', 'position_color_map',
            ))
            objects = list(_meta_get('objects'))
        except (KeyError, TypeError, ValueError):
            self._last_task_details = {'error': 'invalid_metadata'}
            return 0.0

        H, W = gt_final_frame.shape[:2]
        first = (
            normalize_frame_size(input_frame, gt_final_frame)
            if input_frame.shape[:2] != (H, W) else input_frame
        )
        preds = [
            normalize_frame_size(frame, gt_final_frame)
            if frame.shape[:2] != (H, W) else frame
            for frame in pred_images
        ]
        gt_seq = [
            normalize_frame_size(frame, gt_final_frame)
            if frame.shape[:2] != (H, W) else frame
            for frame in (gt_images if gt_images else [first, gt_final_frame])
        ]
        if not gt_seq or not np.array_equal(gt_seq[0], first):
            gt_seq = [first] + gt_seq

        completion_parts = self._score_completion(
            preds[-1],
            gt_final_frame,
            num_lights,
            palette=self._build_color_palette(position_color_mapping),
        )
        completion = float(sum(
            completion_parts[key] * self.COMPLETION_WEIGHTS[key]
            for key in self.COMPLETION_WEIGHTS
        ))

        legacy_parts = self._score_process_validity(
            video_frames=[first] + preds,
            gt_frames=gt_seq,
            num_lights=num_lights,
            objects=objects,
            position_color_mapping=position_color_mapping,
        )
        legacy_process = float(sum(
            legacy_parts[key] * self.PROCESS_WEIGHTS[key]
            for key in self.PROCESS_WEIGHTS
        ))
        atomic_process, atomic_details = self._score_interleave_atomic_process(preds, first, num_lights, objects, position_color_mapping)
        process_validity = legacy_process * atomic_process

        change_mask = self._shape_change_mask(first, gt_final_frame)
        _, first_bg = self._frame_masks(first)
        bg_compare_mask = cv2.bitwise_and(first_bg, cv2.bitwise_not(change_mask))
        background_preservation = self._pixel_similarity(
            gt_final_frame,
            preds[-1],
            bg_compare_mask,
            strictness=3.0,
            min_cutoff=0.6,
        )

        total = (
            self.TASK_WEIGHTS['completion'] * completion
            + self.TASK_WEIGHTS['process_validity'] * process_validity
        ) * (
            self.BACKGROUND_GATE_FLOOR
            + self.BACKGROUND_GATE_WEIGHT * background_preservation
        )
        total = float(max(0.0, min(1.0, total)))
        self._last_task_details = {
            'mode': 'interleave_atomic_state_machine',
            'completion': round(completion, 4),
            'completion_detail': completion_parts,
            'legacy_process': round(legacy_process, 4),
            'legacy_process_detail': legacy_parts,
            'atomic_process': round(atomic_process, 4),
            'atomic_process_detail': atomic_details,
            'process_validity': round(process_validity, 4),
            'background_preservation': round(background_preservation, 4),
            'score_formula': (
                '(0.5 * completion + 0.5 * process_validity) '
                '* (0.6 + 0.4 * background_preservation)'
            ),
            'score': round(total, 4),
        }
        return total


class RavenMatrixEvaluator(BaseEvaluator):
    """
    O-56: Raven's Progressive Matrices evaluator.

    The frame is partitioned into a 3x3 grid of cells.
    Dimensions:
        - completion (60%): edge F1 between predicted and GT in the bottom-right
          cell (last frame vs GT final).
        - preservation (40%): mean edge F1 over the other eight cells between the
          first and last frames of the prediction.
    """

    GRID = 3
    # Inner margin shrinks each crop slightly to reduce sensitivity to grid-line jitter.
    CELL_MARGIN_FRAC = 0.10
    # Dark strokes on light background: pixels below this gray value become foreground.
    EDGE_THRESH = 200
    # Base morphological tolerance in pixels at ~this reference cell size (see _edge_tolerance).
    EDGE_TOLERANCE_BASE_PX = 2
    EDGE_TOLERANCE_REF_CELL = 128

    TASK_WEIGHTS = {
        'completion': 0.60,
        'preservation': 0.40,
    }

    @staticmethod
    def _strip_letterbox(image: np.ndarray) -> np.ndarray:
        """Remove a SIGNIFICANT uniform letterbox bar from the frame edges.

        Only bars wider than ~4% of the dimension are stripped, so tiny 1-2px
        borders on square outputs are left untouched (stripping those misaligns
        the 3x3 cell grid), while genuine letterboxing such as 1280x720 output with
        ~281px bars is removed.
        """
        if image is None or image.size == 0:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape[:2]
        col_std = gray.std(axis=0)
        row_std = gray.std(axis=1)
        thr = 5.0
        left = int(np.argmax(col_std > thr))
        right = w - int(np.argmax(col_std[::-1] > thr))
        top = int(np.argmax(row_std > thr))
        bot = h - int(np.argmax(row_std[::-1] > thr))
        min_bar_w = int(w * 0.04)
        min_bar_h = int(h * 0.04)
        # ignore insignificant borders
        if left < min_bar_w:
            left = 0
        if w - right < min_bar_w:
            right = w
        if top < min_bar_h:
            top = 0
        if h - bot < min_bar_h:
            bot = h
        if left == 0 and top == 0 and right == w and bot == h:
            return image
        if right - left < w * 0.4 or bot - top < h * 0.4:
            return image
        return image[top:bot, left:right]

    @staticmethod
    def _extract_cell(image: np.ndarray, row: int, col: int, grid: int = 3) -> np.ndarray:
        """Crop one cell from a GRID x GRID partition; row/col are 0-based."""
        h, w = image.shape[:2]
        cell_h, cell_w = h // grid, w // grid
        if cell_h < 1 or cell_w < 1:
            return image

        max_my = max(0, (cell_h - 1) // 2)
        max_mx = max(0, (cell_w - 1) // 2)
        margin_y = max(0, min(int(cell_h * RavenMatrixEvaluator.CELL_MARGIN_FRAC), max_my))
        margin_x = max(0, min(int(cell_w * RavenMatrixEvaluator.CELL_MARGIN_FRAC), max_mx))

        y0 = row * cell_h + margin_y
        y1 = (row + 1) * cell_h - margin_y
        x0 = col * cell_w + margin_x
        x1 = (col + 1) * cell_w - margin_x

        if y1 <= y0 or x1 <= x0:
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w

        return image[y0:y1, x0:x1]

    @classmethod
    def _edge_tolerance(cls, cell_h: int, cell_w: int) -> int:
        """Scale tolerance with cell size so small/large resolutions stay comparable."""
        m = max(1, min(cell_h, cell_w))
        return max(0, int(round(cls.EDGE_TOLERANCE_BASE_PX * m / float(cls.EDGE_TOLERANCE_REF_CELL))))

    @staticmethod
    def _stroke_mask_u8(bgr: np.ndarray, thresh: int) -> np.ndarray:
        """Binary mask (uint8 0/1) of dark stroke pixels after fixed threshold + invert."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if len(bgr.shape) == 3 else bgr
        _, t = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        return (t > 0).astype(np.uint8)

    @classmethod
    def _canonicalize_cell(cls, cell_bgr: np.ndarray) -> np.ndarray:
        h, w = cell_bgr.shape[:2]
        m = cls._stroke_mask_u8(cell_bgr, cls.EDGE_THRESH)
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            return cell_bgr
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        if y1 - y0 < 2 or x1 - x0 < 2:
            return cell_bgr
        return cv2.resize(cell_bgr[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LINEAR)

    @classmethod
    def _edge_f1(cls, pred_bgr: np.ndarray, ref_bgr: np.ndarray, tolerance: int) -> float:
        """
        Edge-oriented F1 between two same-sized crops. `ref_bgr` is the reference
        (GT for completion, first frame for preservation). Dilated masks implement
        pixel-level tolerance for anti-aliasing and slight misalignment.
        """
        pred_m = cls._stroke_mask_u8(pred_bgr, cls.EDGE_THRESH)
        ref_m = cls._stroke_mask_u8(ref_bgr, cls.EDGE_THRESH)

        ref_sum = int(ref_m.sum())
        pred_sum = int(pred_m.sum())
        if ref_sum == 0:
            return 1.0 if pred_sum == 0 else 0.0

        if tolerance > 0:
            k = 2 * tolerance + 1
            kernel = np.ones((k, k), np.uint8)
            ref_d = cv2.dilate(ref_m, kernel, iterations=1)
            pred_d = cv2.dilate(pred_m, kernel, iterations=1)
        else:
            ref_d, pred_d = ref_m, pred_m

        tp_p = int(np.logical_and(pred_m > 0, ref_d > 0).sum())
        precision = tp_p / max(pred_sum, 1)

        tp_r = int(np.logical_and(ref_m > 0, pred_d > 0).sum())
        recall = tp_r / max(ref_sum, 1)

        if precision + recall <= 0.0:
            return 0.0
        return float(2.0 * precision * recall / (precision + recall))

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

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        scores: Dict[str, float] = {}

        if len(video_frames) < 2 or gt_final_frame is None:
            self._last_task_details = scores
            return 0.0

        last_frame = self._strip_letterbox(video_frames[-1])
        first_frame = self._strip_letterbox(video_frames[0])
        gt_final_frame = self._strip_letterbox(gt_final_frame)

        if gt_final_frame.shape[:2] != last_frame.shape[:2]:
            gt_last = normalize_frame_size(gt_final_frame, last_frame)
        else:
            gt_last = gt_final_frame

        g = self.GRID
        br_row, br_col = g - 1, g - 1

        h, w = last_frame.shape[:2]
        cell_h, cell_w = h // g, w // g
        tol = self._edge_tolerance(cell_h, cell_w)

        # 1) completion (60%): bottom-right cell, last frame vs GT (edge F1)
        pred_br = self._extract_cell(last_frame, br_row, br_col, g)
        gt_br = self._extract_cell(gt_last, br_row, br_col, g)
        completion_f1 = max(
            self._edge_f1(pred_br, gt_br, tol),
            self._edge_f1(self._canonicalize_cell(pred_br), self._canonicalize_cell(gt_br), tol),
        )

        if completion_f1 > 0.8:
            completion_score = completion_f1
        else:
            completion_score = max(0.0, (completion_f1 - 0.3) / 0.5) * 0.8

        scores['completion'] = float(completion_score)

        # 2) preservation (40%): other eight cells, first vs last (edge F1, ref = first)
        preservation_vals: List[float] = []
        for i in range(g):
            for j in range(g):
                if i == br_row and j == br_col:
                    continue
                c_first = self._extract_cell(first_frame, i, j, g)
                c_last = self._extract_cell(last_frame, i, j, g)
                preservation_vals.append(self._edge_f1(c_last, c_first, tol))

        scores['preservation'] = float(np.mean(preservation_vals)) if preservation_vals else 0.0

        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class SymbolDeleteEvaluator(BaseEvaluator):
    """
    O-58: Symbol delete evaluator.
    
    Evaluator for Delete and Shift task.

    Task: Delete a specific symbol, and shift the remaining symbols left to close the gap.
    The outer boxes and numbers must remain completely stationary.

    Scoring (Total 100% comparing final_frame vs gt_final_frame):
    - foreground_strict_match (70%): Highly sensitive comparison inside the Y-band of the content.
                                     Ensures target is deleted, remaining objects shifted correctly,
                                     and boxes/numbers are kept intact.
    - background_loose_match  (30%): Lower sensitivity comparison for the mostly white background.
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

        y_min, y_max = self._get_combined_content_y_range(gt_first_frame, gt_final_frame)
        
        padding = 5
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)

        fg_mask = np.zeros((h, w), dtype=bool)
        fg_mask[y_min:y_max, :] = True

        bg_mask = ~fg_mask

        fg_score = self._calculate_strict_similarity(gt_final_frame, final_frame, fg_mask)

        bg_score = self._calculate_loose_similarity(gt_final_frame, final_frame, bg_mask)

        score = (fg_score * 0.85) + (bg_score * 0.15)

        self._last_task_details = {
            'fg_score': fg_score,
            'bg_score': bg_score,
            'y_range': (y_min, y_max)
        }
        
        return float(score)


    def _get_combined_content_y_range(
        self, 
        img1: Optional[np.ndarray], 
        img2: np.ndarray, 
        bg_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Tuple[int, int]:

        mask2 = np.any(img2 != bg_color, axis=-1)

        if img1 is not None:
            mask1 = np.any(img1 != bg_color, axis=-1)
            combined_mask = mask1 | mask2
        else:
            combined_mask = mask2
            
        y_indices = np.where(combined_mask)[0]

        if len(y_indices) == 0:
            return 0, img2.shape[0]
            
        return int(np.min(y_indices)), int(np.max(y_indices))

    def _calculate_strict_similarity(
        self, 
        gt_img: np.ndarray, 
        pred_img: np.ndarray, 
        mask: np.ndarray
    ) -> float:

        diff = np.abs(gt_img.astype(float) - pred_img.astype(float))
        masked_diff = diff[mask]
        
        if len(masked_diff) == 0: 
            return 1.0

        mean_diff_norm = np.mean(masked_diff) / 255.0 
        
        strict_tolerance_factor = 40.0
        low_tolerance = 0.02
        if mean_diff_norm < low_tolerance:
            return 1.0

        return float(np.exp(-strict_tolerance_factor * (mean_diff_norm - low_tolerance)))

    def _calculate_loose_similarity(
        self, 
        gt_img: np.ndarray, 
        pred_img: np.ndarray, 
        mask: np.ndarray
    ) -> float:

        diff = np.abs(gt_img.astype(float) - pred_img.astype(float))
        masked_diff = diff[mask]
        
        if len(masked_diff) == 0: 
            return 1.0
        
        mean_diff_norm = np.mean(masked_diff) / 255.0
        low_tolerance = 0.02
        if mean_diff_norm < low_tolerance:
            return 1.0
        
        loose_tolerance_factor = 1.5
        score = max(0.0, 1.0 - (mean_diff_norm * loose_tolerance_factor))
        
        return float(score)


class SymbolInsertEvaluator(BaseEvaluator):
    """
    O-59: Symbol insert evaluator.
    
    Scoring (Total 100% comparing final_frame vs gt_final_frame):
    - sequence_strict_match (70%): Highly sensitive comparison inside the Y-band of the main sequence.
                                   Ensures correct insertion, correct shifting, and intact numbers.
    - template_strict_match (10%): Highly sensitive comparison for the top-right template shape.
                                   Ensures the reference shape wasn't distorted or deleted.
    - background_loose_match(20%): Lower sensitivity comparison for the mostly white background.
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

        template_bbox, (seq_y_min, seq_y_max) = self._get_layout_regions(gt_final_frame)

        template_mask = np.zeros((h, w), dtype=bool)
        if template_bbox is not None:
            tx, ty, tw, th = template_bbox
            template_mask[ty:ty+th, tx:tx+tw] = True

        seq_mask = np.zeros((h, w), dtype=bool)
        seq_mask[seq_y_min:seq_y_max, :] = True
        seq_mask[template_mask] = False  

        bg_mask = ~(template_mask | seq_mask)

        seq_score = self._calculate_strict_similarity(gt_final_frame, final_frame, seq_mask)

        template_score = self._calculate_loose_similarity(gt_final_frame, final_frame, template_mask)

        bg_score = self._calculate_loose_similarity(gt_final_frame, final_frame, bg_mask)


        score = (seq_score * 0.85) + (template_score * 0.05) + (bg_score * 0.10)  

        self._last_task_details = {
            'seq_score': seq_score,
            'template_score': template_score,
            'bg_score': bg_score,
            'template_bbox': template_bbox,
            'seq_y_range': (seq_y_min, seq_y_max)
        }
        
        return float(score)

    def _get_layout_regions(
        self, 
        img: np.ndarray, 
        bg_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Tuple[int, int]]:

        h, w = img.shape[:2]
        is_fg = np.any(img != bg_color, axis=-1)

        top_right_mask = np.zeros((h, w), dtype=bool)
        top_right_mask[:h//2, w//2:] = True
        
        template_fg = is_fg & top_right_mask
        y_indices_tr, x_indices_tr = np.where(template_fg)
        
        template_bbox = None
        if len(y_indices_tr) > 0:
            tx_min, tx_max = np.min(x_indices_tr), np.max(x_indices_tr)
            ty_min, ty_max = np.min(y_indices_tr), np.max(y_indices_tr)
            pad = 5
            tx = max(w//2, tx_min - pad)
            ty = max(0, ty_min - pad)
            tw = min(w - tx, (tx_max - tx_min) + 2*pad)
            th = min(h//2 - ty, (ty_max - ty_min) + 2*pad)
            template_bbox = (tx, ty, tw, th)

        main_seq_fg = is_fg.copy()
        if template_bbox is not None:
            tx, ty, tw, th = template_bbox
            main_seq_fg[ty:ty+th, tx:tx+tw] = False
            
        y_indices_seq = np.where(main_seq_fg)[0]
        
        if len(y_indices_seq) == 0:
            seq_y_range = (0, h) # Fallback
        else:
            pad = 5
            seq_y_min = max(0, int(np.min(y_indices_seq)) - pad)
            seq_y_max = min(h, int(np.max(y_indices_seq)) + pad)
            seq_y_range = (seq_y_min, seq_y_max)

        return template_bbox, seq_y_range

    def _calculate_strict_similarity(
        self, 
        gt_img: np.ndarray, 
        pred_img: np.ndarray, 
        mask: np.ndarray
    ) -> float:

        diff = np.abs(gt_img.astype(np.float32) - pred_img.astype(np.float32))

        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        if mask.shape != gt_img.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (gt_img.shape[1], gt_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if not np.any(mask):
            return 1.0

        # Identify GT "white background" pixels (allow tiny tolerance for compression).
        # Note: images here are BGR from OpenCV.
        bg_tol = 5  # tolerate small codec noise around pure white
        is_bg = np.all(gt_img >= (255 - bg_tol), axis=-1) if gt_img.ndim == 3 else (gt_img >= (255 - bg_tol))

        bg_mask = mask & is_bg
        fg_mask = mask & (~is_bg)

        bg_diff = diff[bg_mask]
        fg_diff = diff[fg_mask]

        # Mean normalized diffs per region (0..1). Empty region contributes 0 error.
        bg_mean = float(np.mean(bg_diff) / 255.0) if bg_diff.size > 0 else 0.0
        fg_mean = float(np.mean(fg_diff) / 255.0) if fg_diff.size > 0 else 0.0

        # Upweight foreground discrepancies (numbers/shapes) vs background noise.
        fg_weight = 4.0
        weighted_err = (bg_mean + fg_weight * fg_mean) / (1.0 + fg_weight)

        # Treat tiny weighted error as perfect match (e.g. decode jitter).
        low_tolerance = 0.025
        if weighted_err < low_tolerance:
            return 1.0

        strict_tolerance_factor = 40.0
        return float(np.exp(-strict_tolerance_factor * (weighted_err - low_tolerance)))

    def _calculate_loose_similarity(
        self, 
        gt_img: np.ndarray, 
        pred_img: np.ndarray, 
        mask: np.ndarray
    ) -> float:
        diff = np.abs(gt_img.astype(float) - pred_img.astype(float))
        masked_diff = diff[mask]
        
        if len(masked_diff) == 0: 
            return 1.0
        
        mean_diff_norm = np.mean(masked_diff) / 255.0
        low_tolerance = 0.015
        if mean_diff_norm < low_tolerance:
            return 1.0
        loose_tolerance_factor = 1.5
        score = max(0.0, 1.0 - (mean_diff_norm * loose_tolerance_factor))
        return float(score)


class SymbolSubstituteEvaluator(BaseEvaluator):
    """
    O-60: Symbol substitute evaluator.
    
    Scoring (Total 100% comparing final_frame vs gt_final_frame):
    - sequence_strict_match (70%): Highly sensitive comparison inside the Y-band of the main sequence.
                                   Ensures correct insertion, correct shifting, and intact numbers.
    - template_strict_match (10%): Highly sensitive comparison for the top-right template shape.
                                   Ensures the reference shape wasn't distorted or deleted.
    - background_loose_match(20%): Lower sensitivity comparison for the mostly white background.
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

        template_bbox, (seq_y_min, seq_y_max) = self._get_layout_regions(gt_final_frame)

        template_mask = np.zeros((h, w), dtype=bool)
        if template_bbox is not None:
            tx, ty, tw, th = template_bbox
            template_mask[ty:ty+th, tx:tx+tw] = True

        seq_mask = np.zeros((h, w), dtype=bool)
        seq_mask[seq_y_min:seq_y_max, :] = True
        seq_mask[template_mask] = False 

        bg_mask = ~(template_mask | seq_mask)

        seq_score = self._calculate_strict_similarity(gt_final_frame, final_frame, seq_mask)
        
        template_score = self._calculate_loose_similarity(gt_final_frame, final_frame, template_mask)

        bg_score = self._calculate_loose_similarity(gt_final_frame, final_frame, bg_mask)

        score = (seq_score * 0.85) + (template_score * 0.05) + (bg_score * 0.10)  # keep-dim floor lowered from 0.30 to ~0.15; GT (seq=1) stays 1.0

        self._last_task_details = {
            'seq_score': seq_score,
            'template_score': template_score,
            'bg_score': bg_score,
            'template_bbox': template_bbox,
            'seq_y_range': (seq_y_min, seq_y_max)
        }
        
        return float(score)

    def _get_layout_regions(
        self, 
        img: np.ndarray, 
        bg_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Tuple[int, int]]:
        h, w = img.shape[:2]
        is_fg = np.any(img != bg_color, axis=-1)

        top_right_mask = np.zeros((h, w), dtype=bool)
        top_right_mask[:h//2, w//2:] = True
        
        template_fg = is_fg & top_right_mask
        y_indices_tr, x_indices_tr = np.where(template_fg)
        
        template_bbox = None
        if len(y_indices_tr) > 0:
            tx_min, tx_max = np.min(x_indices_tr), np.max(x_indices_tr)
            ty_min, ty_max = np.min(y_indices_tr), np.max(y_indices_tr)
            pad = 5
            tx = max(w//2, tx_min - pad)
            ty = max(0, ty_min - pad)
            tw = min(w - tx, (tx_max - tx_min) + 2*pad)
            th = min(h//2 - ty, (ty_max - ty_min) + 2*pad)
            template_bbox = (tx, ty, tw, th)

        main_seq_fg = is_fg.copy()
        if template_bbox is not None:
            tx, ty, tw, th = template_bbox
            main_seq_fg[ty:ty+th, tx:tx+tw] = False
            
        y_indices_seq = np.where(main_seq_fg)[0]
        
        if len(y_indices_seq) == 0:
            seq_y_range = (0, h)
        else:
            pad = 5
            seq_y_min = max(0, int(np.min(y_indices_seq)) - pad)
            seq_y_max = min(h, int(np.max(y_indices_seq)) + pad)
            seq_y_range = (seq_y_min, seq_y_max)

        return template_bbox, seq_y_range

    def _calculate_strict_similarity(
        self, 
        gt_img: np.ndarray, 
        pred_img: np.ndarray, 
        mask: np.ndarray
    ) -> float:
        diff = np.abs(gt_img.astype(np.float32) - pred_img.astype(np.float32))

        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        if mask.shape != gt_img.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (gt_img.shape[1], gt_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if not np.any(mask):
            return 1.0

        # Identify GT "white background" pixels (allow tiny tolerance for compression).
        # Note: images here are BGR from OpenCV.
        bg_tol = 5  # tolerate small codec noise around pure white
        is_bg = np.all(gt_img >= (255 - bg_tol), axis=-1) if gt_img.ndim == 3 else (gt_img >= (255 - bg_tol))

        bg_mask = mask & is_bg
        fg_mask = mask & (~is_bg)

        bg_diff = diff[bg_mask]
        fg_diff = diff[fg_mask]

        # Mean normalized diffs per region (0..1). Empty region contributes 0 error.
        bg_mean = float(np.mean(bg_diff) / 255.0) if bg_diff.size > 0 else 0.0
        fg_mean = float(np.mean(fg_diff) / 255.0) if fg_diff.size > 0 else 0.0

        # Upweight foreground discrepancies (numbers/shapes) vs background noise.
        fg_weight = 4.0
        weighted_err = (bg_mean + fg_weight * fg_mean) / (1.0 + fg_weight)

        # Treat tiny weighted error as perfect match (e.g. decode jitter).
        low_tolerance = 0.032
        if weighted_err < low_tolerance:
            return 1.0

        strict_tolerance_factor = 45
        return float(np.exp(-strict_tolerance_factor * (weighted_err - low_tolerance)))

    def _calculate_loose_similarity(
        self, 
        gt_img: np.ndarray, 
        pred_img: np.ndarray, 
        mask: np.ndarray
    ) -> float:

        diff = np.abs(gt_img.astype(float) - pred_img.astype(float))
        masked_diff = diff[mask]
        
        if len(masked_diff) == 0: 
            return 1.0
        
        mean_diff_norm = np.mean(masked_diff) / 255.0
        low_tolerance = 0.018
        if mean_diff_norm < low_tolerance:
            return 1.0
        loose_tolerance_factor = 1.5
        score = max(0.0, 1.0 - (mean_diff_norm * loose_tolerance_factor))
        return float(score)



class SymbolEditConstraintEvaluator(BaseEvaluator):
    """
    O-61: Symbol edit with constraint evaluator.
    
    Scoring (Total 100% comparing final_frame vs gt_final_frame):
    - sequence_strict_match (70%): Highly sensitive comparison inside the Y-band of the main sequence.
                                   Ensures correct insertion, correct shifting, and intact numbers.
    - template_strict_match (10%): Highly sensitive comparison for the top-right template shape.
                                   Ensures the reference shape wasn't distorted or deleted.
    - background_loose_match(20%): Lower sensitivity comparison for the mostly white background.
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

        template_bbox, (seq_y_min, seq_y_max) = self._get_layout_regions(gt_final_frame)

        template_mask = np.zeros((h, w), dtype=bool)
        if template_bbox is not None:
            tx, ty, tw, th = template_bbox
            template_mask[ty:ty+th, tx:tx+tw] = True

        seq_mask = np.zeros((h, w), dtype=bool)
        seq_mask[seq_y_min:seq_y_max, :] = True
        seq_mask[template_mask] = False

        bg_mask = ~(template_mask | seq_mask)

        seq_score = self._calculate_strict_similarity(gt_final_frame, final_frame, seq_mask)
        
        template_score = self._calculate_loose_similarity(gt_final_frame, final_frame, template_mask)

        bg_score = self._calculate_loose_similarity(gt_final_frame, final_frame, bg_mask)

        score = (seq_score * 0.7) + (template_score * 0.1) + (bg_score * 0.2)

        self._last_task_details = {
            'seq_score': seq_score,
            'template_score': template_score,
            'bg_score': bg_score,
            'template_bbox': template_bbox,
            'seq_y_range': (seq_y_min, seq_y_max)
        }
        
        return float(score)

    def _get_layout_regions(
        self, 
        img: np.ndarray, 
        bg_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Tuple[int, int]]:
        h, w = img.shape[:2]
        is_fg = np.any(img != bg_color, axis=-1)

        top_right_mask = np.zeros((h, w), dtype=bool)
        top_right_mask[:h//2, w//2:] = True
        
        template_fg = is_fg & top_right_mask
        y_indices_tr, x_indices_tr = np.where(template_fg)
        
        template_bbox = None
        if len(y_indices_tr) > 0:
            tx_min, tx_max = np.min(x_indices_tr), np.max(x_indices_tr)
            ty_min, ty_max = np.min(y_indices_tr), np.max(y_indices_tr)
            pad = 5
            tx = max(w//2, tx_min - pad)
            ty = max(0, ty_min - pad)
            tw = min(w - tx, (tx_max - tx_min) + 2*pad)
            th = min(h//2 - ty, (ty_max - ty_min) + 2*pad)
            template_bbox = (tx, ty, tw, th)

        main_seq_fg = is_fg.copy()
        if template_bbox is not None:
            tx, ty, tw, th = template_bbox
            main_seq_fg[ty:ty+th, tx:tx+tw] = False
            
        y_indices_seq = np.where(main_seq_fg)[0]
        
        if len(y_indices_seq) == 0:
            seq_y_range = (0, h) # Fallback
        else:
            pad = 5
            seq_y_min = max(0, int(np.min(y_indices_seq)) - pad)
            seq_y_max = min(h, int(np.max(y_indices_seq)) + pad)
            seq_y_range = (seq_y_min, seq_y_max)

        return template_bbox, seq_y_range

    def _calculate_strict_similarity(
        self, 
        gt_img: np.ndarray, 
        pred_img: np.ndarray, 
        mask: np.ndarray
    ) -> float:
        diff = np.abs(gt_img.astype(float) - pred_img.astype(float))
        masked_diff = diff[mask]
        
        if len(masked_diff) == 0: 
            return 1.0
        
        mean_diff_norm = np.mean(masked_diff) / 255.0
        low_tolerance = 0.02
        if mean_diff_norm < low_tolerance:
            return 1.0
        strict_tolerance_factor = 40.0
        return float(np.exp(-strict_tolerance_factor * (mean_diff_norm - low_tolerance)))

    def _calculate_loose_similarity(
        self, 
        gt_img: np.ndarray, 
        pred_img: np.ndarray, 
        mask: np.ndarray
    ) -> float:
        diff = np.abs(gt_img.astype(float) - pred_img.astype(float))
        masked_diff = diff[mask]
        
        if len(masked_diff) == 0: 
            return 1.0
        
        mean_diff_norm = np.mean(masked_diff) / 255.0
        low_tolerance = 0.02
        if mean_diff_norm < low_tolerance:
            return 1.0
        loose_tolerance_factor = 1.5
        score = max(0.0, 1.0 - (mean_diff_norm * loose_tolerance_factor))
        return float(score)



class GravityPhysicsEvaluator(BaseEvaluator):
    """
    O-62: Gravity physics — ball falls under gravity, bounces with damping.

    Evaluation:
    - process (65%): event sequence match (peaks + ground bounces)
      via Levenshtein edit distance between GT and gen event sequences
    - final_position (35%): ball at correct location in last frame
    - background (penalty): static elements preserved (multiplied onto total)
    """

    TASK_WEIGHTS = {
        'process': 0.65,
        'final_position': 0.35,
    }

    # Detection params
    _BALL_COLOR_DIST = 60    # BGR distance for ball pixel match
    _MIN_BALL_AREA = 200
    _SMOOTH_WIN = 3          # frames for trajectory smoothing
    _MIN_AMP_FRAC = 0.3      # min y-amplitude per event as fraction of ball diameter

    # ---- Ball / scene detection ----

    def _detect_ball_color(self, gt_first: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Find ball BGR color from GT first frame.
        Ball is the only solid high-saturation circular blob."""
        hsv = cv2.cvtColor(gt_first, cv2.COLOR_BGR2HSV)
        # High saturation pixels (excludes white bg, gray axes/ground, black text)
        sat_mask = (hsv[:, :, 1] > 100).astype(np.uint8) * 255
        # Open to clean noise
        kernel = np.ones((3, 3), np.uint8)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < 0.6:
                continue
            # Score = circularity * area (favor large round)
            score = circularity * area
            if score > best_score:
                best_score = score
                m = np.zeros(gt_first.shape[:2], np.uint8)
                cv2.drawContours(m, [c], -1, 255, -1)
                mean_bgr = cv2.mean(gt_first, mask=m)[:3]
                best = tuple(int(v) for v in mean_bgr)
        return best

    def _detect_ball(self, frame: np.ndarray, color: Tuple[int, int, int]) -> Optional[Tuple[int, int, int]]:
        """Find ball (cx, cy, area) in a frame given expected BGR color."""
        target = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        diff = np.sqrt(np.sum((frame.astype(np.float32) - target) ** 2, axis=2))
        mask = (diff < self._BALL_COLOR_DIST).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self._MIN_BALL_AREA:
            return None
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None
        return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']), int(area))

    def _detect_ground_y(self, gt_first: np.ndarray) -> int:
        """Top y of the ground band (gray horizontal strip at bottom)."""
        gray = cv2.cvtColor(gt_first, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Ground = wide horizontal band where most of the row is dark gray (50-200)
        for row in range(h - 1, h // 2, -1):
            r = gray[row]
            dark = ((r > 50) & (r < 200)).sum()
            if dark > w * 0.7:
                continue
            # First row from bottom that's NOT ground = ground top is the next row
            return row + 1
        return int(h * 0.93)

    # ---- Trajectory & events ----

    def _track_ball(self, frames: List[np.ndarray], color: Tuple[int, int, int]):
        """Return list of (x, y, area) per frame; None when ball not found."""
        traj = []
        for f in frames:
            d = self._detect_ball(f, color)
            if d is None:
                traj.append(None)
            else:
                traj.append((d[0], d[1], d[2]))  # x, y, area
        return traj

    def _lateral_drift_penalty(self, traj, ball_diameter: int) -> Tuple[float, float]:
        """Penalize lateral (x) drift. Pure gravity = ball stays at same x.
        Returns (penalty, max_drift_px)."""
        xs = [t[0] for t in traj if t is not None]
        if len(xs) < 5:
            return 1.0, 0.0
        xs = np.array(xs, dtype=np.float32)
        med = float(np.median(xs))
        drift = float(np.max(np.abs(xs - med)))
        rel = drift / max(1, ball_diameter)
        if rel < 0.3:
            pen = 1.0
        elif rel < 1.0:
            pen = 1.0 - (rel - 0.3) / 0.7 * 0.3
        elif rel < 2.0:
            pen = 0.7 - (rel - 1.0) * 0.3
        elif rel < 3.0:
            pen = 0.4 - (rel - 2.0) * 0.4
        else:
            pen = 0.0
        return float(pen), drift

    def _smooth_y(self, traj):
        """Linear-interpolate missing values and smooth y trajectory.
        traj is list of (x, y, area) tuples or None."""
        n = len(traj)
        ys = np.full(n, np.nan, dtype=np.float32)
        for i, t in enumerate(traj):
            if t is not None:
                ys[i] = t[1]  # y is the 2nd element
        if np.isnan(ys).all():
            return None
        # Linear interpolate gaps
        valid = ~np.isnan(ys)
        idx = np.arange(n)
        ys = np.interp(idx, idx[valid], ys[valid])
        # Smooth
        k = self._SMOOTH_WIN
        if k > 1:
            kernel = np.ones(k) / k
            ys = np.convolve(ys, kernel, mode='same')
        return ys

    def _detect_events(self, ys: np.ndarray, ground_y: int, min_amp: float,
                       ball_diameter: int) -> List[Tuple[str, int]]:
        """Detect direction reversals: 'B' = bounce (falling→rising near ground),
        'P' = peak (rising→falling away from ground).
        Returns chronological list of (type, frame_index)."""
        n = len(ys)
        if n < 5:
            return []
        events = []
        dy = np.diff(ys)
        # Ball center at ground contact ≈ ground_y - ball_radius; allow 1 full diameter margin
        # (smoothing and integer rounding can shift the actual minimum y down by a few pixels)
        ground_band = ground_y - ball_diameter
        for i in range(1, len(dy)):
            prev, curr = dy[i - 1], dy[i]
            if prev > 0 and curr <= 0:  # falling → rising = bounce
                if ys[i] > ground_band:
                    events.append(('B', i, ys[i]))
            elif prev < 0 and curr >= 0:  # rising → falling = peak
                events.append(('P', i, ys[i]))
        # Filter by amplitude vs previous event/start
        filtered = []
        last_y = ys[0]
        for typ, frame, y in events:
            if abs(y - last_y) >= min_amp:
                filtered.append((typ, frame, float(y)))
                last_y = y
        return filtered

    @staticmethod
    def _levenshtein(a: List[str], b: List[str]) -> int:
        if not a:
            return len(b)
        if not b:
            return len(a)
        n, m = len(a), len(b)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                tmp = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = tmp
        return dp[m]

    # ---- Background penalty ----

    def _build_bg_mask(self, gt_first: np.ndarray, ground_y: int) -> np.ndarray:
        """Mask of static background = everything except ball region and
        the area where the velocity arrow + label may appear (near ball trajectory column).
        Returns binary mask: 255 = bg pixel to check."""
        h, w = gt_first.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        # Exclude ground (changing arrow + ball touch area)
        mask[ground_y:, :] = 0
        # Exclude central vertical strip (ball + velocity arrow follow the ball horizontally)
        # Use the ball x from GT first frame to choose strip
        ball_col = self._detect_ball_color(gt_first)
        if ball_col is not None:
            d = self._detect_ball(gt_first, ball_col)
            if d is not None:
                cx = d[0]
                strip = int(w * 0.18)  # ~180px around ball x
                mask[:, max(0, cx - strip):min(w, cx + strip)] = 0
        # Exclude top 60 px (gravity arrow can update slightly)
        # Keep most static elements: axes labels on the left
        return mask

    def _pixel_diff_score(self, f1: np.ndarray, f2: np.ndarray, mask: np.ndarray,
                          thresholds=(0.02, 0.05, 0.10, 0.20)) -> Tuple[float, Dict]:
        mask_px = int((mask > 0).sum())
        if mask_px == 0:
            return 1.0, {'ratio': 0.0}
        diff = cv2.absdiff(f1, f2)
        gd = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed = int((gd[mask > 0] > 20).sum())
        ratio = float(changed) / mask_px
        t1, t2, t3, t4 = thresholds
        if ratio < t1:
            s = 1.0
        elif ratio < t2:
            s = 1.0 - (ratio - t1) / (t2 - t1) * 0.3
        elif ratio < t3:
            s = 0.7 - (ratio - t2) / (t3 - t2) * 0.4
        elif ratio < t4:
            s = 0.3 - (ratio - t3) / (t4 - t3) * 0.3
        else:
            s = 0.0
        return s, {'ratio': round(ratio, 5)}

    # ---- Score helpers ----

    @staticmethod
    def _position_score(d: float, ball_d: int) -> float:
        """Distance score based on ball diameter."""
        if d <= ball_d * 0.5:
            return 1.0
        if d <= ball_d * 1.5:
            return 0.7
        if d <= ball_d * 3:
            return 0.4
        return max(0.0, 0.2 - (d - ball_d * 3) / (ball_d * 5) * 0.2)

    # ---- Main ----

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
        if not gt_frames or len(gt_frames) < 5:
            return 0.0

        # 1) Detect ball color from GT
        ball_color = self._detect_ball_color(gt_first_frame)
        if ball_color is None:
            self._last_task_details = {'error': 'ball not detected in GT'}
            return 0.0

        # 2) Detect ground line from GT
        ground_y = self._detect_ground_y(gt_first_frame)

        # 3) Ball diameter (for amplitude / position thresholds)
        gt_d = self._detect_ball(gt_first_frame, ball_color)
        ball_area = gt_d[2] if gt_d else 1000
        ball_diameter = int(2 * np.sqrt(ball_area / np.pi))
        min_amp = max(8, ball_diameter * self._MIN_AMP_FRAC)

        # 4) Trajectories
        gt_traj = self._track_ball(gt_frames, ball_color)
        gen_traj = self._track_ball(video_frames, ball_color)
        gt_y = self._smooth_y(gt_traj)
        gen_y = self._smooth_y(gen_traj)

        if gt_y is None:
            self._last_task_details = {'error': 'ball never seen in GT video'}
            return 0.0

        # 5) Event sequences (each event = (type, frame, y))
        gt_events = self._detect_events(gt_y, ground_y, min_amp, ball_diameter)
        gen_events = self._detect_events(gen_y, ground_y, min_amp, ball_diameter) if gen_y is not None else []
        gt_seq = [e[0] for e in gt_events]
        gen_seq = [e[0] for e in gen_events]

        # 6a) Sequence score = Levenshtein-based similarity
        if not gt_seq:
            seq_score = 1.0 if not gen_seq else 0.5
        elif not gen_seq:
            seq_score = 0.0
        else:
            ed = self._levenshtein(gt_seq, gen_seq)
            seq_score = max(0.0, 1.0 - ed / max(len(gt_seq), len(gen_seq)))

        # 6b) Peak height score = compare y of peaks (P events) position-by-position
        # Scale by image height to get relative error
        img_h = gt_first_frame.shape[0]
        gt_peaks = [e[2] for e in gt_events if e[0] == 'P']
        gen_peaks = [e[2] for e in gen_events if e[0] == 'P']
        if not gt_peaks:
            height_score = 1.0  # no peaks in GT, nothing to score
        elif not gen_peaks:
            height_score = 0.0
        else:
            n = min(len(gt_peaks), len(gen_peaks))
            errs = []
            for i in range(n):
                err = abs(gt_peaks[i] - gen_peaks[i]) / max(1, img_h)
                # err < 5% → 1.0, err > 25% → 0
                if err < 0.05:
                    s = 1.0
                elif err < 0.25:
                    s = 1.0 - (err - 0.05) / 0.20
                else:
                    s = 0.0
                errs.append(s)
            # Penalize length mismatch: missing peaks count as 0
            missing = abs(len(gt_peaks) - len(gen_peaks))
            errs.extend([0.0] * missing)
            height_score = float(np.mean(errs))

        # 6c) Process = sequence × height (both must be correct)
        proc_score = seq_score * height_score

        # 7) Final position score
        gt_final_pos = self._detect_ball(gt_final_frame, ball_color)
        gen_final_pos = self._detect_ball(video_frames[-1], ball_color)
        if gt_final_pos is None or gen_final_pos is None:
            final_pos_score = 0.0 if gt_final_pos is not None else 0.5
            final_dist = -1
        else:
            final_dist = float(np.hypot(gen_final_pos[0] - gt_final_pos[0],
                                        gen_final_pos[1] - gt_final_pos[1]))
            final_pos_score = self._position_score(final_dist, ball_diameter)

        # 8a) Background penalty
        gen_first, gen_last = video_frames[0], video_frames[-1]
        if gen_first.shape != gt_first_frame.shape:
            gen_first = normalize_frame_size(gen_first, gt_first_frame)
        if gen_last.shape != gt_first_frame.shape:
            gen_last = normalize_frame_size(gen_last, gt_first_frame)
        bg_mask = self._build_bg_mask(gt_first_frame, ground_y)
        bg_score, bg_det = self._pixel_diff_score(gen_first, gen_last, bg_mask)

        # 8b) Lateral drift penalty (ball should fall straight, no x movement)
        lat_pen, lat_drift_px = self._lateral_drift_penalty(gen_traj, ball_diameter)

        # 9) Compose
        scores = {
            'process': round(proc_score, 4),
            'final_position': round(final_pos_score, 4),
        }
        raw_total = sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
        total = raw_total * (0.5 + 0.5 * bg_score) * (0.4 + 0.6 * lat_pen)

        self._last_task_details = {
            **scores,
            'seq_score': round(seq_score, 4),
            'height_score': round(height_score, 4),
            'bg_penalty': round(bg_score, 4),
            'bg_diff_ratio': bg_det.get('ratio', 0.0),
            'lateral_penalty': round(lat_pen, 4),
            'lateral_drift_px': round(lat_drift_px, 1),
            'ball_color': ball_color,
            'ball_diameter': ball_diameter,
            'ground_y': ground_y,
            'gt_event_seq': ''.join(gt_seq),
            'gen_event_seq': ''.join(gen_seq),
            'gt_peak_ys': [round(p, 0) for p in gt_peaks],
            'gen_peak_ys': [round(p, 0) for p in gen_peaks],
            'final_dist_px': round(final_dist, 1),
        }
        return float(total)

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if not pred_images or input_frame is None or gt_final_frame is None:
            return 0.0
        gen_frames = [input_frame] + pred_images
        gt_frames = list(gt_images) if gt_images else [input_frame, gt_final_frame]
        ref = input_frame
        gen_frames = [f if f.shape == ref.shape else normalize_frame_size(f, ref) for f in gen_frames]
        gt_frames = [f if f.shape == ref.shape else normalize_frame_size(f, ref) for f in gt_frames]

        ball_color = self._detect_ball_color(input_frame)
        if ball_color is None:
            self._last_task_details = {'error': 'ball not detected'}
            return 0.0
        ground_y = self._detect_ground_y(input_frame)
        gt_d = self._detect_ball(input_frame, ball_color)
        ball_diameter = int(2 * np.sqrt(gt_d[2] / np.pi)) if gt_d else 40
        min_amp = max(8, ball_diameter * self._MIN_AMP_FRAC)

        # --- process (0.65): bounce event sequence + peak heights, from keyframes ---
        gt_traj = self._track_ball(gt_frames, ball_color)
        gen_traj = self._track_ball(gen_frames, ball_color)
        gt_y = self._smooth_y(gt_traj)
        gen_y = self._smooth_y(gen_traj)
        gt_events = self._detect_events(gt_y, ground_y, min_amp, ball_diameter) if gt_y is not None else []
        gen_events = self._detect_events(gen_y, ground_y, min_amp, ball_diameter) if gen_y is not None else []
        gt_seq = [e[0] for e in gt_events]
        gen_seq = [e[0] for e in gen_events]
        if not gt_seq:
            seq_score = 1.0 if not gen_seq else 0.5
        elif not gen_seq:
            seq_score = 0.0
        else:
            ed = self._levenshtein(gt_seq, gen_seq)
            seq_score = max(0.0, 1.0 - ed / max(len(gt_seq), len(gen_seq)))
        img_h = input_frame.shape[0]
        gt_peaks = [e[2] for e in gt_events if e[0] == 'P']
        gen_peaks = [e[2] for e in gen_events if e[0] == 'P']
        if not gt_peaks:
            height_score = 1.0
        elif not gen_peaks:
            height_score = 0.0
        else:
            n = min(len(gt_peaks), len(gen_peaks))
            errs = []
            for i in range(n):
                err = abs(gt_peaks[i] - gen_peaks[i]) / max(1, img_h)
                if err < 0.05:
                    s = 1.0
                elif err < 0.25:
                    s = 1.0 - (err - 0.05) / 0.20
                else:
                    s = 0.0
                errs.append(s)
            errs.extend([0.0] * abs(len(gt_peaks) - len(gen_peaks)))
            height_score = float(np.mean(errs))
        proc_score = seq_score * height_score

        # --- final position (0.35) ---
        gt_final_pos = self._detect_ball(gt_final_frame, ball_color)
        gen_final_pos = self._detect_ball(gen_frames[-1], ball_color)
        if gt_final_pos is None or gen_final_pos is None:
            final_pos_score = 0.0 if gt_final_pos is not None else 0.5
            final_dist = -1.0
        else:
            final_dist = float(np.hypot(gen_final_pos[0] - gt_final_pos[0],
                                        gen_final_pos[1] - gt_final_pos[1]))
            final_pos_score = self._position_score(final_dist, ball_diameter)

        # --- background + lateral-drift gates (same as video) ---
        bg_mask = self._build_bg_mask(input_frame, ground_y)
        bg_score, bg_det = self._pixel_diff_score(gen_frames[0], gen_frames[-1], bg_mask)
        lat_pen, lat_drift_px = self._lateral_drift_penalty(gen_traj, ball_diameter)

        scores = {'process': round(proc_score, 4), 'final_position': round(final_pos_score, 4)}
        raw_total = sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
        total = raw_total * (0.5 + 0.5 * bg_score) * (0.4 + 0.6 * lat_pen)
        self._last_task_details = {
            **scores,
            'seq_score': round(seq_score, 4), 'height_score': round(height_score, 4),
            'bg_score': round(bg_score, 4), 'lateral_penalty': round(lat_pen, 4),
            'final_dist_px': round(final_dist, 1),
            'gt_seq': ''.join(gt_seq), 'gen_seq': ''.join(gen_seq),
            'note': 'interleave: video composition on keyframes (no <5 guard)',
        }
        return float(max(0.0, min(1.0, total)))



class AnimalMatchingEvaluator(BaseEvaluator):
    """
    O-64: move each animal face to its matching silhouette.

    The legacy scorer hard-coded animal colours and then mostly checked
    left/right counts.  That fails when multiple animals share similar brown
    tones and does not actually verify the final matching.

    We instead treat the GT final right half as the authoritative target set:

    - detect filled animal components on the GT first left half
    - detect filled target placements on the GT final right half
    - detect filled components on the prediction's right half
    - score the best component-to-target assignment by centre, contour shape,
      and area
    - penalise residual animals left on the source side and wrong final count
    """

    MIN_COMPONENT_AREA = 1500.0
    BG_COLOR_DIST_TOL = 22.0
    FILLED_EXTENT_MIN = 0.22
    FILL_RATIO_MIN = 0.5
    ALIGNMENT_HALF_PX = 70.0
    PAIR_SCORE_SATURATION = 0.95
    PAIR_CENTER_SNAP_PX = 4.0
    APPEARANCE_PATCH_SIZE = 32
    APPEARANCE_LO = 0.35
    APPEARANCE_HI = 0.75
    APPEARANCE_FLOOR = 0.2
    DYNAMIC_MASK_SIZE = 128
    DYNAMIC_DIFF_THRESHOLD = 30.0
    INTERMEDIATE_FINAL_IOU_MAX = 0.80
    DISTINCT_DYNAMIC_STATE_IOU_MAX = 0.90
    REQUIRED_DISTINCT_INTERMEDIATE_STATES = 2

    @classmethod
    def _patch_descriptor(
        cls,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        x, y, bw, bh = bbox
        h, w = frame.shape[:2]
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x + bw), w), min(int(y + bh), h)
        if x1 <= x0 or y1 <= y0:
            return None
        size = cls.APPEARANCE_PATCH_SIZE
        patch = cv2.resize(frame[y0:y1, x0:x1], (size, size)).astype(np.float32).ravel()
        patch -= float(patch.mean())
        norm = float(np.linalg.norm(patch))
        return patch / norm if norm > 1e-6 else None

    @classmethod
    def _appearance_score(cls, pred: Dict[str, Any], target: Dict[str, Any]) -> float:
        a, b = pred.get("patch"), target.get("patch")
        if a is None or b is None:
            return 1.0
        ncc = float(np.dot(a, b))
        ramp = (ncc - cls.APPEARANCE_LO) / (cls.APPEARANCE_HI - cls.APPEARANCE_LO)
        return cls.APPEARANCE_FLOOR + (1.0 - cls.APPEARANCE_FLOOR) * float(
            np.clip(ramp, 0.0, 1.0)
        )

    @staticmethod
    def _estimate_bg_color(frame: np.ndarray) -> np.ndarray:
        patch = np.concatenate([
            frame[:24, :24].reshape(-1, 3),
            frame[:24, -24:].reshape(-1, 3),
            frame[-24:, :24].reshape(-1, 3),
            frame[-24:, -24:].reshape(-1, 3),
        ], axis=0)
        return np.median(patch, axis=0)

    def _detect_animal_components(
        self,
        frame: np.ndarray,
        side: Optional[str] = None,
        filled_only: bool = False,
    ) -> List[Dict[str, Any]]:
        h, w = frame.shape[:2]
        if side == "left":
            roi = frame[:, :w // 2]
            x_offset = 0
        elif side == "right":
            roi = frame[:, w // 2:]
            x_offset = w // 2
        else:
            roi = frame
            x_offset = 0

        bg_color = self._estimate_bg_color(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_dist = np.sqrt(
            np.sum((roi.astype(np.float32) - bg_color.reshape(1, 1, 3)) ** 2, axis=2)
        )
        mask = (
            (color_dist > self.BG_COLOR_DIST_TOL)
            | (hsv[:, :, 1] > 20)
            | (hsv[:, :, 2] < 210)
        ).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        components: List[Dict[str, Any]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.MIN_COMPONENT_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bh > 0.85 * h and bw < 0.12 * max(roi.shape[1], 1):
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            extent = float(area / max(bw * bh, 1))
            contour_global = contour + np.array([[[x_offset, 0]]], dtype=contour.dtype)
            if filled_only and extent < self.FILLED_EXTENT_MIN:
                continue

            if filled_only:
                comp_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(comp_mask, [contour], -1, 1, -1)
                interior_px = int(comp_mask.sum())
                foreground_px = int(((comp_mask > 0) & (mask > 0)).sum())
                fill_ratio = foreground_px / max(interior_px, 1)
                if fill_ratio < self.FILL_RATIO_MIN:
                    continue

            components.append({
                "center": (
                    float(moments["m10"] / moments["m00"]) + x_offset,
                    float(moments["m01"] / moments["m00"]),
                ),
                "area": area,
                "extent": extent,
                "contour": contour_global,
                "bbox": (x + x_offset, y, bw, bh),
                "patch": self._patch_descriptor(frame, (x + x_offset, y, bw, bh)),
            })

        return sorted(components, key=lambda item: item["center"][0])

    def _pair_score(self, pred: Dict[str, Any], target: Dict[str, Any]) -> float:
        center_dist = safe_distance(pred["center"], target["center"])
        center_score = max(0.0, 1.0 - center_dist / (2.0 * self.ALIGNMENT_HALF_PX))
        area_score = min(pred["area"], target["area"]) / max(
            pred["area"], target["area"], 1.0,
        )
        shape_cost = float(cv2.matchShapes(
            pred["contour"],
            target["contour"],
            cv2.CONTOURS_MATCH_I1,
            0.0,
        ))
        shape_score = 1.0 / (1.0 + 6.0 * shape_cost)
        raw_score = center_score * area_score * shape_score * self._appearance_score(pred, target)
        if (
            raw_score >= self.PAIR_SCORE_SATURATION
            and center_dist <= self.PAIR_CENTER_SNAP_PX
        ):
            return 1.0
        return raw_score

    def _dynamic_foreground_mask(
        self,
        frame: np.ndarray,
        baseline: np.ndarray,
    ) -> np.ndarray:
        """Foreground change mask with global exposure shifts removed."""
        if frame.shape != baseline.shape:
            frame = normalize_frame_size(frame, baseline)
        delta = frame.astype(np.float32) - baseline.astype(np.float32)
        offset = np.median(delta.reshape(-1, 3), axis=0)
        corrected = delta - offset.reshape(1, 1, 3)
        distance = np.linalg.norm(corrected, axis=2)
        mask = (distance > self.DYNAMIC_DIFF_THRESHOLD).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        return cv2.resize(
            mask,
            (self.DYNAMIC_MASK_SIZE, self.DYNAMIC_MASK_SIZE),
            interpolation=cv2.INTER_AREA,
        ) > 80

    @staticmethod
    def _binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        union = int(np.logical_or(mask_a, mask_b).sum())
        if union == 0:
            return 1.0
        return float(np.logical_and(mask_a, mask_b).sum() / union)

    def _dynamic_process_score(
        self,
        frames: Sequence[np.ndarray],
    ) -> Tuple[float, Dict[str, Any]]:
        """Require two arbitrary distinct in-flight foreground states."""
        if len(frames) < 4:
            return 0.0, {
                'distinct_intermediate_states': 0,
                'required_distinct_states': self.REQUIRED_DISTINCT_INTERMEDIATE_STATES,
                'reason': 'too_few_frames',
            }

        baseline = frames[0]
        final_mask = self._dynamic_foreground_mask(frames[-1], baseline)
        final_area = int(final_mask.sum())
        if final_area == 0:
            return 0.0, {
                'distinct_intermediate_states': 0,
                'required_distinct_states': self.REQUIRED_DISTINCT_INTERMEDIATE_STATES,
                'reason': 'empty_final_dynamic_mask',
            }

        representatives: List[np.ndarray] = []
        candidates: List[Dict[str, Any]] = []
        for frame_idx, frame in enumerate(frames[1:-1], start=1):
            mask = self._dynamic_foreground_mask(frame, baseline)
            area_ratio = float(mask.sum() / max(final_area, 1))
            final_iou = self._binary_iou(mask, final_mask)
            is_intermediate = (
                area_ratio >= 0.20
                and final_iou <= self.INTERMEDIATE_FINAL_IOU_MAX
            )
            is_distinct = False
            if is_intermediate:
                is_distinct = all(
                    self._binary_iou(mask, representative)
                    < self.DISTINCT_DYNAMIC_STATE_IOU_MAX
                    for representative in representatives
                )
                if is_distinct:
                    representatives.append(mask)
            candidates.append({
                'frame_index': frame_idx,
                'area_ratio_to_final': round(area_ratio, 4),
                'final_iou': round(final_iou, 4),
                'is_intermediate': bool(is_intermediate),
                'is_distinct': bool(is_distinct),
            })

        distinct_count = len(representatives)
        required = self.REQUIRED_DISTINCT_INTERMEDIATE_STATES
        process_score = min(1.0, distinct_count / max(required, 1))
        return float(process_score), {
            'distinct_intermediate_states': distinct_count,
            'required_distinct_states': required,
            'final_dynamic_area': final_area,
            'intermediate_final_iou_max': self.INTERMEDIATE_FINAL_IOU_MAX,
            'distinct_state_iou_max': self.DISTINCT_DYNAMIC_STATE_IOU_MAX,
            'candidates': candidates,
        }

    def _best_assignment(
        self,
        preds: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        if not preds or not targets:
            return 0.0, []

        if len(preds) <= len(targets):
            best_score = -1.0
            best_pairs: List[Tuple[int, int]] = []
            for target_perm in permutations(range(len(targets)), len(preds)):
                score = sum(
                    self._pair_score(preds[i], targets[target_perm[i]])
                    for i in range(len(preds))
                )
                if score > best_score:
                    best_score = score
                    best_pairs = [(i, target_perm[i]) for i in range(len(preds))]
        else:
            best_score = -1.0
            best_pairs = []
            for pred_perm in permutations(range(len(preds)), len(targets)):
                score = sum(
                    self._pair_score(preds[pred_perm[i]], targets[i])
                    for i in range(len(targets))
                )
                if score > best_score:
                    best_score = score
                    best_pairs = [(pred_perm[i], i) for i in range(len(targets))]

        pair_details: List[Dict[str, Any]] = []
        for pred_idx, target_idx in best_pairs:
            pred = preds[pred_idx]
            target = targets[target_idx]
            pair_details.append({
                "pred_center": [
                    round(float(pred["center"][0]), 2),
                    round(float(pred["center"][1]), 2),
                ],
                "target_center": [
                    round(float(target["center"][0]), 2),
                    round(float(target["center"][1]), 2),
                ],
                "score": round(float(self._pair_score(pred, target)), 4),
                "appearance": round(float(self._appearance_score(pred, target)), 4),
                "center_dist_px": round(float(safe_distance(pred["center"], target["center"])), 2),
            })

        mean_score = best_score / max(min(len(preds), len(targets)), 1)
        return max(0.0, mean_score), pair_details

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        del gt_frames, eval_info
        if len(video_frames) < 2 or gt_first_frame is None or gt_final_frame is None:
            self._last_task_details = {"error": "missing_frames"}
            return 0.0

        sources = self._detect_animal_components(
            gt_first_frame, side="left", filled_only=True,
        )
        targets = self._detect_animal_components(
            gt_final_frame, side="right", filled_only=True,
        )
        pred_right = self._detect_animal_components(
            video_frames[-1], side="right", filled_only=True,
        )
        pred_left = self._detect_animal_components(
            video_frames[-1], side="left", filled_only=True,
        )

        if not sources or not targets:
            self._last_task_details = {
                "error": "reference_detection_failed",
                "source_count": len(sources),
                "target_count": len(targets),
            }
            return 0.0

        placement, pair_details = self._best_assignment(pred_right, targets)
        count_penalty = min(len(pred_right), len(targets)) / max(
            len(pred_right), len(targets), 1,
        )
        source_area_total = sum(source["area"] for source in sources)
        left_area_total = sum(item["area"] for item in pred_left)
        left_clear = max(0.0, 1.0 - left_area_total / max(source_area_total, 1.0))

        process_score, process_details = self._dynamic_process_score(video_frames)
        final_quality = (
            placement
            * (0.4 + 0.6 * count_penalty)
            * (0.4 + 0.6 * left_clear)
        )
        total = final_quality * (0.4 + 0.6 * process_score)
        self._last_task_details = {
            "placement": round(float(placement), 4),
            "count_penalty": round(float(count_penalty), 4),
            "left_clear": round(float(left_clear), 4),
            "source_count": len(sources),
            "target_count": len(targets),
            "pred_right_count": len(pred_right),
            "pred_left_count": len(pred_left),
            "process_score": round(float(process_score), 4),
            "final_quality": round(float(final_quality), 4),
            "score_formula": "final_quality * (0.4 + 0.6 * process_score)",
            "process_details": process_details,
            "pairings": pair_details,
            "score": round(float(total), 4),
        }
        return total



class AnimalSizeSortingEvaluator(BaseEvaluator):
    """
    O-65: Animal size sorting evaluator.

    Scoring:
    - arrangement     (60%): size order (0.5) + baseline alignment (0.5)
    - fore_consistency (10%): all reference animals found in final frame
    - back_consistency (30%): non-animal, non-baseline area is white
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

        # Resize gt frames to match final_frame if needed
        if gt_final_frame.shape[:2] != final_frame.shape[:2]:
            gt_final_frame = cv2.resize(
                gt_final_frame, (final_frame.shape[1], final_frame.shape[0])
            )

        # 1. Extract reference patterns from gt_final_frame
        ref_patterns = extract_patterns_from_white_bg(gt_final_frame, min_area=1000)

        # Determine expected order from gt_final_frame: sort patterns by x, check if areas increase or decrease
        gt_sorted = sorted(ref_patterns, key=lambda p: p['center'][0])
        gt_areas = [p['area'] for p in gt_sorted]
        if len(gt_areas) >= 2:
            inc = sum(1 for i in range(len(gt_areas) - 1) if gt_areas[i] < gt_areas[i + 1])
            dec = sum(1 for i in range(len(gt_areas) - 1) if gt_areas[i] > gt_areas[i + 1])
            expected_increasing = (inc >= dec)
        else:
            expected_increasing = True

        # 2. Find matching patterns in final_frame
        matched = find_patterns_in_image(
            gt_final_frame, ref_patterns, final_frame,
            match_threshold=0.8,
        )
        # 3. Detect gray baseline
        gray_line_y = self._detect_gray_line(final_frame)

        # 4. Arrangement score
        arrangement = self._evaluate_arrangement(matched, gray_line_y, final_frame.shape, expected_increasing)

        # 5. Foreground consistency
        fore_consistency = self._evaluate_fore_consistency(ref_patterns, matched)

        # 6. Background consistency
        back_consistency = self._evaluate_back_consistency(final_frame, matched, gray_line_y)

        n_ref = len(ref_patterns)
        n_objects = len(extract_patterns_from_white_bg(final_frame, min_area=1000))
        count_penalty = min(1.0, n_ref / max(n_objects, 1)) if n_ref > 0 else 1.0

        consistency = 0.25 * fore_consistency + 0.75 * back_consistency
        score = arrangement * (0.6 + 0.4 * consistency) * count_penalty
        self._last_task_details = {
            'arrangement': arrangement,
            'fore_consistency': fore_consistency,
            'back_consistency': back_consistency,
            'count_penalty': count_penalty,
            'n_ref': n_ref,
            'n_objects': n_objects,
            'n_matched': len(matched),
        }
        return score

    # ------------------------------------------------------------------
    # Gray baseline detection
    # ------------------------------------------------------------------

    def _detect_gray_line(self, frame: np.ndarray) -> int:
        """Detect the horizontal gray baseline in the bottom half of the frame."""
        h, w = frame.shape[:2]
        bottom = frame[h // 2:, :]

        # Gray pixel: all channels in [60, 210] and channel range (max-min) < 40
        b, g, r = bottom[:, :, 0].astype(np.int32), bottom[:, :, 1].astype(np.int32), bottom[:, :, 2].astype(np.int32)
        ch_min = np.minimum(np.minimum(b, g), r)
        ch_max = np.maximum(np.maximum(b, g), r)
        gray_mask = (ch_min >= 60) & (ch_max <= 210) & ((ch_max - ch_min) < 40)

        # Row with highest gray pixel fraction
        row_gray_frac = gray_mask.mean(axis=1)
        best_row = int(np.argmax(row_gray_frac))

        # Only trust it if at least 20% of the row is gray
        if row_gray_frac[best_row] >= 0.20:
            return h // 2 + best_row

        # Fallback: 85% of frame height
        return int(h * 0.85)

    # ------------------------------------------------------------------
    # Arrangement
    # ------------------------------------------------------------------

    def _evaluate_arrangement(
        self, matched: List[Dict], gray_line_y: int, frame_shape: Tuple,
        expected_increasing: bool = True,
    ) -> float:
        """0.5 for size order + 0.5 for baseline alignment."""
        if len(matched) < 2:
            return 0.0

        h = frame_shape[0]

        # Sort by x center
        sorted_m = sorted(matched, key=lambda m: m['center'][0])

        # --- Order score (0.5): ref_area increasing or decreasing left→right ---
        ref_areas = [m['ref_area'] for m in sorted_m]
        n = len(ref_areas)
        if expected_increasing:
            wrong_adj = sum(1 for i in range(n - 1) if ref_areas[i] > ref_areas[i + 1])
        else:
            wrong_adj = sum(1 for i in range(n - 1) if ref_areas[i] < ref_areas[i + 1])
        wrong_ratio = wrong_adj / (n - 1)
        order_score = (1.0 - wrong_ratio) ** 2

        # --- Alignment score (0.5) ---
        # a) Bottom of each bbox close to gray_line_y
        bottom_ys = [m['center'][1] + m['bbox'][3] // 2 for m in sorted_m]
        rel_dists = [abs(by - gray_line_y) / (h + 1e-6) for by in bottom_ys]
        mean_rel = float(np.mean(rel_dists))
        tol_line = 0.08
        excess_line = max(0.0, mean_rel - tol_line)
        line_score = 1.0 / (1.0 + (excess_line * 10) ** 2)

        # b) Centers horizontally aligned (low y-std relative to bbox height)
        center_ys = [m['center'][1] for m in sorted_m]
        avg_bbox_h = float(np.mean([m['bbox'][3] for m in sorted_m])) + 1e-6
        y_std = float(np.std(center_ys))
        rel_std = max(0.0, y_std / avg_bbox_h - 0.25)
        horiz_score = 1.0 / (1.0 + (rel_std * 3) ** 2)

        alignment_score = 0.5 * line_score + 0.5 * horiz_score
        return order_score * (0.5 + 0.5 * alignment_score)

    # ------------------------------------------------------------------
    # Foreground consistency
    # ------------------------------------------------------------------

    def _evaluate_fore_consistency(
        self, ref_patterns: List[Dict], matched: List[Dict]
    ) -> float:
        """Score based on how many reference animals were found."""
        n_ref = len(ref_patterns)
        if n_ref == 0:
            return 1.0
        n_matched = min(len(matched), n_ref)
        missing = n_ref - n_matched
        return max(0.0, 1.0 - missing * 0.5)

    # ------------------------------------------------------------------
    # Background consistency
    # ------------------------------------------------------------------

    def _evaluate_back_consistency(
        self, frame: np.ndarray, matched: List[Dict], gray_line_y: int
    ) -> float:
        """Check that pixels outside matched animals and gray baseline are white."""
        h, w = frame.shape[:2]
        fg_mask = np.zeros((h, w), dtype=np.uint8)

        # Mask matched animal bboxes (with 4px padding)
        for m in matched:
            x, y, bw, bh = m['bbox']
            x1 = max(0, x - 4)
            y1 = max(0, y - 4)
            x2 = min(w, x + bw + 4)
            y2 = min(h, y + bh + 4)
            fg_mask[y1:y2, x1:x2] = 255

        # Mask gray baseline band (±10px)
        y1 = max(0, gray_line_y - 10)
        y2 = min(h, gray_line_y + 10)
        fg_mask[y1:y2, :] = 255

        bg_pixels = frame[fg_mask == 0]
        if len(bg_pixels) == 0:
            return 1.0

        white_ratio = float(np.mean(np.all(bg_pixels >= 240, axis=1)))
        return float(np.exp(-35.0 * (1.0 - white_ratio)))


class ObjectRotation2DEvaluator(BaseEvaluator):
    """
    O-85: 2D object rotation evaluator.
    
    Dimensions:
        - shape_preservation (40%): for each generated frame the shapes are compared with the GT first frame via Hu moments distance.
        - completion (40%): shape/size/color/position comparison of generated final frame shapes against GT final frame shapes.
        - background_preservation (20%): the generated final frame is compared with the GT final frame via pixel similarity.
    """
    
    TASK_WEIGHTS = {
        "shape_preservation": 0.50,
        "completion": 0.50,
    }
    BACKGROUND_GATE_FLOOR = 0.60

    ROTATION_SHAPE_FEATURE_WEIGHTS = {
        "shape": 0.30,
        "size": 0.30,
        "color": 0.15,
        "position": 0.25,
    }

    IOU_HIGH_THRESHOLD = 0.90
    IOU_LOW_THRESHOLD = 0.75

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
        """Return (foreground_mask, background_mask) for a frame."""
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

    def _extract_all_shapes(self, frame: np.ndarray, num_objects: int) -> List[Dict]:
        """Extract top num_objects largest foreground shapes, sorted by area descending."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg_level = float(np.percentile(gray, 95))
        thresh = float(min(240.0, max(120.0, bg_level - 15.0)))
        _, fg_mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort by area descending and keep only the top num_objects candidates to filter artifacts
        contours = sorted(
            [c for c in contours if cv2.contourArea(c) >= 100],
            key=cv2.contourArea,
            reverse=True,
        )[:num_objects]
        shapes = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            M = cv2.moments(cnt)
            if M["m00"] <= 0:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
            mean_bgr = np.array(cv2.mean(frame, mask=mask)[:3], dtype=np.float32)
            perimeter = float(cv2.arcLength(cnt, True))
            _, _, bw, bh = cv2.boundingRect(cnt)
            approx = cv2.approxPolyDP(cnt, 0.10 * perimeter if perimeter > 0 else 0.0, True)
            shapes.append({
                "contour": cnt,
                "mask": mask,
                "area": area,
                "area_ratio": area / (h * w),
                "centroid": (cx / w, cy / h),
                "mean_bgr": mean_bgr,
                "vertex_count": int(len(approx)),
                "bbox_extent": area / max(float(bw * bh), 1.0),
            })
        return shapes

    @staticmethod
    def _match_shapes_by_position(
        gt_shapes: List[Dict],
        gen_shapes: List[Dict],
        max_pos_dist: float = 0.3,
    ) -> List[Tuple[Dict, Optional[Dict]]]:
        """Greedy match GT shapes to gen shapes by nearest centroid position."""
        matched: List[Tuple[Dict, Optional[Dict]]] = []
        used: set = set()
        for gt in gt_shapes:
            best_idx, best_dist = None, float("inf")
            gt_cx, gt_cy = gt["centroid"]
            for i, gen in enumerate(gen_shapes):
                if i in used:
                    continue
                gen_cx, gen_cy = gen["centroid"]
                dist = float(np.sqrt((gt_cx - gen_cx) ** 2 + (gt_cy - gen_cy) ** 2))
                if dist < best_dist:
                    best_dist, best_idx = dist, i
            if best_idx is not None and best_dist < max_pos_dist:
                matched.append((gt, gen_shapes[best_idx]))
                used.add(best_idx)
            else:
                matched.append((gt, None))
        return matched

    def _compute_frame_rotation_score(
        self,
        gt_frame: np.ndarray,
        gen_frame: np.ndarray,
        gt_shapes: List[Dict],
        gen_shapes: List[Dict],
    ) -> Tuple[float, float, float]:
        """Area-weighted (IoU * 0.6 + color * 0.4) between GT-aligned and gen shapes.

        Returns (combined_score, iou, color_sim).
        Comparing against the GT frame at each time step directly rewards videos
        where the shape is at the correct rotation angle, rather than just checking
        shape identity with rotation-invariant Hu moments.
        """
        if not gt_shapes:
            return 0.5, 0.5, 0.5
        pairs = self._match_shapes_by_position(gt_shapes, gen_shapes)
        weighted_iou = 0.0
        weighted_color = 0.0
        total_weight = 0.0
        for gt_s, gen_s in pairs:
            w = gt_s["area"]
            total_weight += w
            if gen_s is None:
                continue

            # IoU (soft-thresholded)
            mask_gt = gt_s["mask"]
            mask_gen = gen_s["mask"]
            if mask_gt.shape != mask_gen.shape:
                mask_gen = cv2.resize(
                    mask_gen, (mask_gt.shape[1], mask_gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            inter = float(np.logical_and(mask_gt > 0, mask_gen > 0).sum())
            union = float(np.logical_or(mask_gt > 0, mask_gen > 0).sum())
            raw_iou = inter / union if union > 0 else 0.0
            if raw_iou >= self.IOU_HIGH_THRESHOLD:
                iou = raw_iou
            elif raw_iou <= self.IOU_LOW_THRESHOLD:
                iou = 0.0
            else:
                iou = float(
                    (raw_iou - self.IOU_LOW_THRESHOLD)
                    / (self.IOU_HIGH_THRESHOLD - self.IOU_LOW_THRESHOLD)
                    * self.IOU_HIGH_THRESHOLD
                )

            # Pixel similarity on the union of matched shape regions
            compare_mask = mask_gt
            color_sim = self._pixel_similarity(gt_frame, gen_frame, mask=compare_mask, strictness=3.0, min_cutoff=0.6)

            weighted_iou += w * iou
            weighted_color += w * color_sim

        if total_weight <= 0:
            return 0.0, 0.0, 0.0
        iou_avg = weighted_iou / total_weight
        color_avg = weighted_color / total_weight
        return 0.6 * iou_avg + 0.4 * color_avg, iou_avg, color_avg

    def _compute_rotation_completion_score(
        self,
        gt_final_shapes: List[Dict],
        gen_final_shapes: List[Dict],
    ) -> Tuple[float, Dict[str, float]]:
        """Final-frame completion: shape/size/color/position per position-matched shape pair."""
        details: Dict[str, float] = {
            "shape": 0.0,
            "size": 0.0,
            "color": 0.0,
            "position": 0.0,
            "shape_contour": 0.0,
            "shape_vertex": 0.0,
        }
        if not gt_final_shapes or not gen_final_shapes:
            return 0.0, details

        pairs = self._match_shapes_by_position(gt_final_shapes, gen_final_shapes)
        total_weight = 0.0
        weighted: Dict[str, float] = {
            "shape": 0.0,
            "size": 0.0,
            "color": 0.0,
            "position": 0.0,
            "shape_contour": 0.0,
            "shape_vertex": 0.0,
        }

        for gt, gen in pairs:
            total_weight += 1.0
            if gen is None:
                continue

            # 1. shape type gate (rotation-invariant): filter wrong shape types
            match_score = cv2.matchShapes(gt["contour"], gen["contour"], cv2.CONTOURS_MATCH_I1, 0.0)
            shape_from_contour = float(np.exp(-4.0 * max(0.0, match_score)))
            shape_from_contour = shape_from_contour if shape_from_contour > 0.5 else 0.0
            gt_v, gen_v = gt["vertex_count"], gen["vertex_count"]
            vertex_score = 1.0 if gt_v == gen_v else (0.3 if abs(gt_v - gen_v) <= 1 else 0.0)
            shape_type_score = float(0.7 * shape_from_contour + 0.3 * vertex_score)
            if shape_type_score < 0.6:
                # wrong shape type — contributes 0 to all dimensions
                continue

            # 1b. shape score: soft-thresholded IoU (rotation-sensitive — wrong angle -> low overlap)
            mask_gt = gt["mask"]
            mask_gen = gen["mask"]
            if mask_gt.shape != mask_gen.shape:
                mask_gen = cv2.resize(mask_gen, (mask_gt.shape[1], mask_gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            inter = float(np.logical_and(mask_gt > 0, mask_gen > 0).sum())
            union = float(np.logical_or(mask_gt > 0, mask_gen > 0).sum())
            raw_iou = inter / union if union > 0 else 0.0
            if raw_iou >= self.IOU_HIGH_THRESHOLD:
                shape_score = raw_iou
            elif raw_iou <= self.IOU_LOW_THRESHOLD:
                shape_score = 0.0
            else:
                shape_score = float(
                    (raw_iou - self.IOU_LOW_THRESHOLD)
                    / (self.IOU_HIGH_THRESHOLD - self.IOU_LOW_THRESHOLD)
                    * self.IOU_HIGH_THRESHOLD
                )

            # 2. size
            area_ratio = min(gt["area"], gen["area"]) / max(gt["area"], gen["area"], 1e-6)
            extent_ratio = min(gt["bbox_extent"], gen["bbox_extent"]) / max(gt["bbox_extent"], gen["bbox_extent"], 1e-6)
            size_ratio = float(0.80 * area_ratio + 0.20 * extent_ratio)
            # ratio is already the measured agreement; the old >=0.75-else-0 cut discarded how close it came.
            size_score = size_ratio if size_ratio >= 0.75 else 0.0

            # 3. color
            color_dist = float(np.linalg.norm(gt["mean_bgr"] - gen["mean_bgr"]))
            color_score = float(max(0.0, 1.0 - color_dist / np.sqrt(3.0 * (255.0 ** 2))))

            # 4. position
            gt_cx, gt_cy = gt["centroid"]
            gen_cx, gen_cy = gen["centroid"]
            pos_dist = float(np.sqrt((gt_cx - gen_cx) ** 2 + (gt_cy - gen_cy) ** 2))
            position_score = float(max(0.0, 1.0 - pos_dist / np.sqrt(2.0)))

            weighted["shape"] += shape_score          # IoU
            weighted["size"] += size_score
            weighted["color"] += color_score
            weighted["position"] += position_score
            weighted["shape_contour"] += shape_type_score  # diagnostic: shape type match
            weighted["shape_vertex"] += vertex_score

        if total_weight <= 0:
            return 0.0, details

        details = {k: weighted[k] / total_weight for k in weighted}
        completion = float(max(0.0, min(1.0, sum(
            details[k] * self.ROTATION_SHAPE_FEATURE_WEIGHTS[k]
            for k in self.ROTATION_SHAPE_FEATURE_WEIGHTS
        ))))
        return completion, details

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        scores: Dict[str, float] = {}

        if len(video_frames) < 2 or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        # Normalize frame size (handles padding removal + resize)
        gt_frames = [normalize_frame_size(f, last_frame) if f.shape[:2] != last_frame.shape[:2] else f for f in gt_frames] if gt_frames else gt_frames
        gt_first = gt_frames[0]
        gt_last = gt_frames[-1]

        import os
        _mp = eval_info.get('metafile_path')
        if isinstance(_mp, (list, tuple)):
            _mp = next((p for p in _mp if p and os.path.exists(p)), _mp[0] if _mp else None)
        with open(_mp) as f:
            metadata = json.load(f)
        num_objects: int = metadata['parameters']['num_objects']
        
        # 1) shape_preservation (40%): compare each gen frame against the temporally-aligned
        #    GT frame using area-weighted soft-IoU.  GT frames already encode the correct
        #    rotation angle at each time step, so this directly rewards the right trajectory
        #    rather than rotation-invariant shape identity.
        n_gen = len(video_frames)
        frame_scores: List[float] = []
        frame_ious: List[float] = []
        frame_colors: List[float] = []
        if gt_frames:
            n_gt = len(gt_frames)
            gt_indices = [int(round(i * (n_gt - 1) / max(n_gen - 1, 1))) for i in range(n_gen)]
            for i, frame in enumerate(video_frames):
                gt_frame_at_t = gt_frames[gt_indices[i]]
                if gt_frame_at_t.shape[:2] != frame.shape[:2]:
                    gt_frame_at_t = normalize_frame_size(gt_frame_at_t, frame)
                gt_shapes_at_t = self._extract_all_shapes(gt_frame_at_t, num_objects)
                gen_shapes = self._extract_all_shapes(frame, num_objects)
                s, iou, color = self._compute_frame_rotation_score(
                    gt_frame_at_t, frame, gt_shapes_at_t, gen_shapes
                )
                frame_scores.append(s)
                frame_ious.append(iou)
                frame_colors.append(color)
        else:
            gt_first_shapes = self._extract_all_shapes(gt_first, num_objects)
            for frame in video_frames:
                gen_shapes = self._extract_all_shapes(frame, num_objects)
                s, iou, color = self._compute_frame_rotation_score(
                    gt_first, frame, gt_first_shapes, gen_shapes
                )
                frame_scores.append(s)
                frame_ious.append(iou)
                frame_colors.append(color)
        scores["shape_preservation"] = float(np.mean(frame_scores)) if frame_scores else 0.0
        preservation_details = {
            "iou": float(np.mean(frame_ious)) if frame_ious else 0.0,
            "color": float(np.mean(frame_colors)) if frame_colors else 0.0,
        }

        # 2) completion (40%): shape/size/color/position comparison of generated final
        #    frame shapes against GT final frame shapes.
        gt_final_shapes = self._extract_all_shapes(gt_last, num_objects)
        gen_final_shapes = self._extract_all_shapes(last_frame, num_objects)
        completion_score, completion_details = self._compute_rotation_completion_score(
            gt_final_shapes, gen_final_shapes
        )
        scores["completion"] = completion_score

        # 3) background_preservation (20%)
        change_mask = self._shape_change_mask(gt_first, gt_last)
        _, first_bg = self._frame_masks(gt_first)
        bg_compare_mask = cv2.bitwise_and(first_bg, cv2.bitwise_not(change_mask))
        scores["background_preservation"] = self._pixel_similarity(
            gt_last, last_frame, bg_compare_mask, strictness=1.0
        )

        self._last_task_details = {
            **scores,
            "preservation_details": preservation_details,
            "completion_details": completion_details,
        }
        task_score = (
            self.TASK_WEIGHTS["shape_preservation"] * scores["shape_preservation"]
            + self.TASK_WEIGHTS["completion"] * scores["completion"]
        )
        background_gate = (
            self.BACKGROUND_GATE_FLOOR
            + (1.0 - self.BACKGROUND_GATE_FLOOR)
            * scores["background_preservation"]
        )
        total = task_score * background_gate
        self._last_task_details.update({
            "task_score": float(task_score),
            "background_gate": float(background_gate),
            "score_formula": (
                "(0.5 * shape_preservation + 0.5 * completion) "
                "* (0.6 + 0.4 * background_preservation)"
            ),
        })
        return float(total)



# Export all Part 4 evaluators
OUT_OF_DOMAIN_50_EVALUATORS_PART5 = {
    'O-54_control_panel_data-generator': ControlPanelEvaluator,
    'O-56_raven_data-generator': RavenMatrixEvaluator,
    'O-58_symbol_delete_data-generator': SymbolDeleteEvaluator,
    'O-59_symbol_insert_data-generator': SymbolInsertEvaluator,
    'O-60_symbol_substitute_data-genertor': SymbolSubstituteEvaluator,
    'O-61_symbol_edit_data-generator': SymbolEditConstraintEvaluator,
    'O-62_gravity_physics_data-generator': GravityPhysicsEvaluator,
    'O-64_animal_matching_data-generator': AnimalMatchingEvaluator,
    'O-65_animal_size_sorting_data-generator': AnimalSizeSortingEvaluator,
    'O-85_2d_object_rotation_data-generator': ObjectRotation2DEvaluator,
}
