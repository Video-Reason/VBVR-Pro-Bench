"""
Specific evaluators for In-Domain_50 tasks (Part 2).
"""

import numpy as np
import cv2
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any
from .base_evaluator import BaseEvaluator
from .utils import maze
from ..utils import normalize_frame_size, detect_closed_contours_by_color, match_contours, COLOR_BOUNDS, \
    score_background_similarity, score_foreground_similarity, calculate_list_length_penalty
from ..utils import CircleSelectionProcessor, threshold_score
import os
import json
import shutil


# ---------------------------------------------------------------------------
# G-41 helper functions (4×4 grid)
# ---------------------------------------------------------------------------

def _g41_pixel_to_cell(
    px: int, py: int, frame_shape: Tuple[int, ...], grid_size: int = 4,
) -> Tuple[int, int]:
    """Pixel (x, y) -> (row, col).  Thin wrapper around :func:`maze.pixel_to_cell`."""
    return maze.pixel_to_cell(px, py, frame_shape, grid_size)


def _g41_extract_tens_digit(binary: np.ndarray) -> np.ndarray:
    """Extract the tens-digit portion from a binarised cell crop.

    Finds the inter-digit gap via column projection, returns everything
    to the left of that gap.
    """
    coords = np.argwhere(binary > 0)
    if len(coords) == 0:
        return binary
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    content = binary[y0 : y1 + 1, x0 : x1 + 1]
    col_sum = np.sum(content > 0, axis=0)
    w = content.shape[1]
    start, end = int(w * 0.3), int(w * 0.7)
    if end - start > 2:
        split = start + int(np.argmin(col_sum[start:end]))
    else:
        split = w // 2
    return content[:, :split]


def _g41_classify_tens_digit(tens: np.ndarray) -> int:
    """Classify a binarised single tens-digit image (1-9) using topology + geometry.

    Feature summary (measured on the actual grid font):
        digit | holes | aspect | L/R  | T/B
        ------+-------+--------+------+-----
          1   |   0   |  0.38  | 0.17 | 1.37
          2   |   0   |  0.67  | 0.98 | 0.95
          3   |   0   |  0.67  | 0.65 | 0.97
          4   |   1   |  0.72  | 0.55 | 0.73
          5   |   1   |  0.67  | 1.00 | 1.17
          6   |   1   |  ~0.6  |  —   | <0.9
          7   |   0   |  ~0.5  |  —   | >1.2
          8   |   2   |  ~0.6  |  —   |  —
          9   |   1   |  ~0.6  |  —   | >1.0
    """
    coords = np.argwhere(tens > 0)
    if len(coords) == 0:
        return 1

    # --- hole count ---
    contours, hierarchy = cv2.findContours(
        tens.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
    )
    holes = 0
    if hierarchy is not None:
        for h in hierarchy[0]:
            if h[3] != -1:
                holes += 1

    # --- geometry ---
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    aspect = (x1 - x0 + 1) / max(y1 - y0 + 1, 1)

    th, tw = tens.shape
    my, mx = th // 2, tw // 2
    tl = int(np.sum(tens[:my, :mx] > 0))
    tr = int(np.sum(tens[:my, mx:] > 0))
    bl = int(np.sum(tens[my:, :mx] > 0))
    br = int(np.sum(tens[my:, mx:] > 0))
    lr = (tl + bl) / max(tr + br, 1)   # left / right
    tb = (tl + tr) / max(bl + br, 1)   # top  / bottom

    # --- classification ---
    if holes >= 2:
        return 8

    if holes == 1:
        if aspect > 0.70:
            return 4
        if tb > 1.0:
            return 5 if lr > 0.85 else 9
        return 6

    # holes == 0
    if aspect < 0.45:
        return 1
    if aspect < 0.60:
        return 7
    if lr > 0.80:
        return 2
    return 3


def _g41_find_optimal_path(
    grid: List[List[int]], grid_size: int = 4,
) -> Tuple[List[Tuple[int, int]], int, Set[Tuple[int, int]]]:
    """DFS to find the highest-cost path from (0,0) to (grid_size-1, grid_size-1).

    Thin wrapper around :func:`maze.longest_path_dfs` preserved for callers
    that import ``_g41_find_optimal_path`` directly.
    """
    return maze.longest_path_dfs(grid, grid_size)

class ChartExtremeEvaluator(BaseEvaluator):
    """
    G-29: Chart extreme with data evaluator.

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
        if len(video_frames) < 1:
            return 0.0

        final_frame = video_frames[-1]
        canvas_size = (final_frame.shape[0], final_frame.shape[1])

        if gt_final_frame.shape[:2] != final_frame.shape[:2]:
            gt_final_frame = cv2.resize(
                gt_final_frame, (final_frame.shape[1], final_frame.shape[0])
            )

        gen_contours = detect_closed_contours_by_color(final_frame, COLOR_BOUNDS['red'],max_fill_ratio=0.7)
        gt_contours  = detect_closed_contours_by_color(gt_final_frame, COLOR_BOUNDS['red'],max_fill_ratio=0.7)

        match_results = match_contours(gt_contours, gen_contours, iou_threshold=0.1, canvas_size=canvas_size)

        valid_ious = [iou for iou in match_results if iou is not None]
        accuracy = float(np.mean(valid_ious)) if valid_ious else 0.0
        accuracy = accuracy * calculate_list_length_penalty(len(gt_contours), len(valid_ious), len(gen_contours))

        back_consistency = score_background_similarity(gt_final_frame, final_frame)

        fore_consistency = score_foreground_similarity(gt_final_frame, final_frame, COLOR_BOUNDS['red'])

        consistency = 0.5 * back_consistency + 0.5 * fore_consistency
        score = accuracy * (0.6 + 0.4 * consistency)

        self._last_task_details = {
            'accuracy': accuracy,
            'back_consistency': back_consistency,
            'fore_consistency': fore_consistency,
        }
        return score


class DirectedGraphNavigationEvaluator(BaseEvaluator):
    """
    G-31: Directed graph navigation evaluator.

    Dimensions:
        - completion (25%): Whether the blue triangle agent reaches the GT position
          in the final frame. Score is penalized by distance; zero beyond threshold.
        - path_validity (40%): IoU between the trajectory stroke drawn from the
          generated video and the GT trajectory stroke * smoothness score
        - foreground_preservation (20%): Pixel similarity of the graph structure
          between the first frame and the generated final frame, excluding agent
          pixels in both frames (the agent is the only expected change).
        - background_preservation (15%): Pixel similarity of the background region
          (pixels that are background in both the first and final frames).
    """

    TASK_WEIGHTS = {
        "completion": 0.25,
        "path_validity": 0.40,
        "foreground_preservation": 0.20,
        "background_preservation": 0.15,
    }

    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect the blue triangular agent and return its centroid (x, y).
        The method finds all blue triangular contours and selects the largest one.
        Returns None if no valid blue triangle is found.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blue color range in HSV.
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find external contours from the blue mask.
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return (None, 0)

        # Keep only triangular blue contours, then select the largest by area.
        triangular_contours = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.06 * perimeter, True)
            if len(approx) == 3:
                triangular_contours.append(contour)

        if not triangular_contours:
            return (None, 0)

        # Count valid triangular agents before choosing the largest one.
        agent_count = len(triangular_contours)

        largest_triangle = max(triangular_contours, key=cv2.contourArea)
        moments = cv2.moments(largest_triangle)
        if moments["m00"] <= 0:
            return (None, agent_count)

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        centroid = (cx, cy)
        return (centroid, agent_count)

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
        """Return (foreground_mask, background_mask) for a graph-navigation frame.

        Foreground = graph structure: nodes (coloured circles), edges (dark lines),
        and the agent.  Background = the white/light canvas.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Background is near-white; foreground is everything else
        _, bg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        fg_mask = cv2.bitwise_not(bg_mask)
        return fg_mask, bg_mask

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Evaluate directed graph navigation task."""
        scores: Dict[str, float] = {}

        if len(video_frames) < 2 or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        if last_frame.shape[:2] != gt_final_frame.shape[:2]:
            video_frames = [normalize_frame_size(f, gt_final_frame) for f in video_frames]
            first_frame, last_frame = video_frames[0], video_frames[-1]
        gt_last = gt_final_frame
        gt_frames = ([normalize_frame_size(f, gt_final_frame)
                      if f.shape[:2] != gt_final_frame.shape[:2] else f
                      for f in gt_frames] if gt_frames else gt_frames)

        # 1. completion (30%)
        # Detect agent in both GT final frame and generated final frame,
        # then score by their distance: closer = better.
        gt_agent_pos, gt_agent_count = self._detect_agent(gt_last)
        gen_agent_pos, gen_agent_count = self._detect_agent(last_frame)

        if gt_agent_pos is not None and gen_agent_pos is not None:
            dist = float(np.hypot(gen_agent_pos[0] - gt_agent_pos[0],
                                  gen_agent_pos[1] - gt_agent_pos[1]))
            # Use image diagonal as normalisation reference
            img_diag = float(np.hypot(last_frame.shape[0], last_frame.shape[1]))
            threshold_full = img_diag * 0.02   # within 2% diagonal -> full score
            threshold_zero = img_diag * 0.15   # beyond 15% diagonal -> zero score
            if dist <= threshold_full:
                scores["completion"] = 1.0
            elif dist >= threshold_zero:
                scores["completion"] = 0.0
            else:
                scores["completion"] = 1.0 - (dist - threshold_full) / (threshold_zero - threshold_full)
        else:
            scores["completion"] = 0.0
        
        # If multiple agents are detected, penalize the score by the number of agents
        if gen_agent_count > 1:
            scores["completion"] = scores["completion"] * (1 / gen_agent_count)

        # 2. path_validity (30%): smoothness * IoU with GT trajectory
        gen_agent_detections = [self._detect_agent(f) for f in video_frames]
        gen_positions = [detection[0] for detection in gen_agent_detections]
        gen_agent_counts = [detection[1] for detection in gen_agent_detections]
        gt_positions = [self._detect_agent(f)[0] for f in gt_frames]
        thickness = 45

        if len(gen_positions) >= 2:
            # Step A: smoothness — penalise large teleporting jumps
            img_diag = float(np.hypot(last_frame.shape[0], last_frame.shape[1]))
            jump_thresh = img_diag * 0.15
            large_jumps = 0
            for i in range(1, len(gen_positions)):
                p0, p1 = gen_positions[i - 1], gen_positions[i]
                if p0 is not None and p1 is not None:
                    dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                    if dist > jump_thresh:
                        large_jumps += 1
            smoothness_score = max(0.0, 1.0 - large_jumps * 0.3)

            # Step B: IoU with GT trajectory
            pred_mask = np.zeros(last_frame.shape[:2], dtype=np.uint8)
            for i in range(1, len(gen_positions)):
                p0, p1 = gen_positions[i - 1], gen_positions[i]
                if p0 is not None and p1 is not None:
                    cv2.line(pred_mask, p0, p1, 255, thickness)

            valid_gt = [p for p in gt_positions if p is not None]
            if len(valid_gt) >= 2:
                gt_mask = np.zeros(last_frame.shape[:2], dtype=np.uint8)
                for i in range(1, len(gt_positions)):
                    p0, p1 = gt_positions[i - 1], gt_positions[i]
                    if p0 is not None and p1 is not None:
                        cv2.line(gt_mask, p0, p1, 255, thickness)
                intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
                union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
                path_correctness = float(intersection / union) if union > 0 else 0.0
                scores["path_validity"] = smoothness_score * path_correctness
            else:
                scores["path_validity"] = smoothness_score
        else:
            scores["path_validity"] = 0.0

        # Penalize path quality when multiple agents appear across the video.
        avg_gen_agent_count = float(np.mean(gen_agent_counts)) if gen_agent_counts else 0.0
        if avg_gen_agent_count > 1.0:
            scores["path_validity"] = scores["path_validity"] * (1.0 / avg_gen_agent_count)

        # 3. foreground_preservation (25%)
        # Compare graph structure between the first frame and the generated final frame.
        first_fg, first_bg = self._frame_masks(first_frame)

        # Agent mask: dilate blue region to fully cover the triangle shape
        def _blue_mask(frame: np.ndarray) -> np.ndarray:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            return cv2.inRange(hsv, np.array([100, 80, 80]), np.array([130, 255, 255]))

        def _dilate(m: np.ndarray) -> np.ndarray:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            return cv2.dilate(m, k)

        _first_blue = _blue_mask(first_frame)
        _n0, _lb0, _st0, _ = cv2.connectedComponentsWithStats((_first_blue > 0).astype(np.uint8), 8)
        _areas0 = [int(_st0[i, cv2.CC_STAT_AREA]) for i in range(1, _n0)]
        _agent_area = float(max(_areas0)) if _areas0 else 0.0

        def _agent_mask(frame: np.ndarray) -> np.ndarray:
            m = (_blue_mask(frame) > 0).astype(np.uint8)
            if _agent_area <= 0:
                return _dilate(m * 255)
            n, lb, st, _c = cv2.connectedComponentsWithStats(m, 8)
            keep = np.zeros_like(m)
            for i in range(1, n):
                if int(st[i, cv2.CC_STAT_AREA]) <= 3.0 * _agent_area:
                    keep[lb == i] = 1
            return _dilate(keep * 255)

        agent_union = cv2.bitwise_or(_agent_mask(first_frame), _agent_mask(last_frame))

        # Foreground mask: first-frame foreground excluding agent pixels
        fg_compare_mask = cv2.bitwise_and(first_fg, cv2.bitwise_not(agent_union))
        scores["foreground_preservation"] = self._pixel_similarity(first_frame, last_frame, fg_compare_mask)

        # 4. background_preservation (15%)
        # Compare pixels that are background in both the first and final frames.
        bg_compare_mask = cv2.bitwise_and(first_bg, cv2.bitwise_not(agent_union))
        scores["background_preservation"] = self._pixel_similarity(first_frame, last_frame, bg_compare_mask, strictness=3.0, min_cutoff=0.6)
        self._last_task_details = scores

        ramp = min(1.0, scores["completion"] / 0.4)
        keep = (self.TASK_WEIGHTS["foreground_preservation"] * scores["foreground_preservation"]
                + self.TASK_WEIGHTS["background_preservation"] * scores["background_preservation"])
        total = (self.TASK_WEIGHTS["path_validity"] * scores["path_validity"]
                 + self.TASK_WEIGHTS["completion"] * scores["completion"]
                 + ramp * keep)
        return float(total)

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Interleave (image mode): the solution path is drawn by HIGHLIGHTING
        its graph edges (no moving agent is rendered). Score using the actual
        graph data from metadata: detect which edges are highlighted (changed vs
        the input frame, where every edge is the neutral colour) and compare that
        set to the GT solution-path edges.
          completion (25%):  recall of GT path edges (route fully marked → reaches goal)
          path_validity (40%): precision of highlighted edges (no wrong edges marked)
          foreground_preservation (20%): nodes + non-path edges match GT
          background_preservation (15%): background matches GT
        """
        import json as _json, os as _os
        if not pred_images or input_frame is None:
            self._last_task_details = {"error": "missing_frames"}
            return 0.0

        mp = eval_info.get("metafile_path")
        if isinstance(mp, (list, tuple)):
            mp = next((p for p in mp if p and _os.path.exists(p)), None)
        if not (mp and _os.path.exists(mp)):
            mp = _os.path.join(eval_info.get("gt_path", ""), "metadata.json")
        if not _os.path.exists(mp):
            self._last_task_details = {"error": "metadata not found"}
            return 0.0
        with open(mp) as f:
            meta = _json.load(f)
        params = meta.get("parameters") or {}
        graph = meta.get("semantic_ground_truth", {}).get("graph") or {}
        nodes = params.get("nodes") or graph.get("node_positions_px")
        edges = params.get("edges") or graph.get("edges")
        path = params.get("path") or meta.get("semantic_ground_truth", {}).get("shortest_path")
        if not nodes or not edges or not path or len(path) < 2:
            self._last_task_details = {"error": "graph data missing"}
            return 0.0
        node_r = int(params.get("node_radius", 39))

        ref = input_frame if input_frame is not None else gt_final_frame
        last_frame = pred_images[-1]
        if last_frame.shape[:2] != ref.shape[:2]:
            last_frame = normalize_frame_size(last_frame, ref)
        gt_last = gt_final_frame if gt_final_frame is not None else (
            gt_images[-1] if gt_images else last_frame)
        if gt_last.shape[:2] != ref.shape[:2]:
            gt_last = normalize_frame_size(gt_last, ref)
        h, w = ref.shape[:2]

        # scale node coords / radius if the canvas differs from the frame
        cv_size = params.get("canvas_size") or [w, h]
        sx, sy = w / float(cv_size[0]), h / float(cv_size[1])

        def _node_xy(i):
            n = nodes[i]
            xy = n if isinstance(n, (list, tuple)) else (n.get("position") or n.get("pos"))
            return (int(round(xy[0] * sx)), int(round(xy[1] * sy)))

        def _edge_change_frac(img, i, j):
            mask = np.zeros((h, w), np.uint8)
            cv2.line(mask, _node_xy(i), _node_xy(j), 255, 7)
            cv2.circle(mask, _node_xy(i), int(node_r * sx), 0, -1)
            cv2.circle(mask, _node_xy(j), int(node_r * sx), 0, -1)
            idx = mask > 0
            if int(idx.sum()) < 5:
                return 0.0
            d = np.linalg.norm(img[idx].astype(float) - ref[idx].astype(float), axis=1)
            return float((d > 40).mean())

        THR = 0.4
        pred_hl = {frozenset((i, j)) for (i, j) in edges
                   if _edge_change_frac(last_frame, i, j) > THR}
        gt_set = {frozenset((path[k], path[k + 1])) for k in range(len(path) - 1)}

        inter = len(pred_hl & gt_set)
        recall = inter / len(gt_set) if gt_set else 0.0
        precision = inter / len(pred_hl) if pred_hl else (1.0 if not gt_set else 0.0)

        scores: Dict[str, float] = {}
        scores["completion"] = recall
        scores["path_validity"] = precision

        # preservation: graph structure / background of pred vs GT final
        gt_fg, gt_bg = self._frame_masks(gt_last)
        scores["foreground_preservation"] = self._pixel_similarity(last_frame, gt_last, gt_fg)
        scores["background_preservation"] = self._pixel_similarity(
            last_frame, gt_last, gt_bg, strictness=3.0, min_cutoff=0.6)

        self._last_task_details = {
            **scores, "mode": "interleave",
            "pred_edges": sorted(tuple(sorted(e)) for e in pred_hl),
            "gt_edges": sorted(tuple(sorted(e)) for e in gt_set),
        }

        ramp = min(1.0, scores["completion"] / 0.4)
        keep = (self.TASK_WEIGHTS["foreground_preservation"] * scores["foreground_preservation"]
                + self.TASK_WEIGHTS["background_preservation"] * scores["background_preservation"])
        return float(self.TASK_WEIGHTS["path_validity"] * scores["path_validity"]
                     + self.TASK_WEIGHTS["completion"] * scores["completion"]
                     + ramp * keep)


class AttentionShiftEvaluator(BaseEvaluator):
    """
    G-39: Attention shift different evaluator.

    Evaluates:
    - box_position (60%): Green box covers target object; penalizes too small or too large
    - objects_preserved (20%): Scene objects remain unchanged
    - background_preserved (20%): Background remains unchanged
    """

    TASK_WEIGHTS = {
        'box_position': 0.80,
        'consistency': 0.20,
    }
    
    def _get_green_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return binary mask of all green regions."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

        return mask

    def _detect_green_boxes(self, frame: np.ndarray):
        green_mask = self._get_green_mask(frame)
        contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            return []

        boxes = []
        for i, contour in enumerate(contours):
            parent = hierarchy[0][i][3]
            child = hierarchy[0][i][2]
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            if parent != -1:
                continue
            if area < 200:
                continue
            rect_area = w * h
            if rect_area == 0:
                continue
            if child == -1:
                continue

            child_area = 0
            ci = child
            while ci != -1:
                child_area += cv2.contourArea(contours[ci])
                ci = hierarchy[0][ci][0]  # next sibling
            if child_area < area * 0.08:
                continue

            if area / float(rect_area) < 0.80:
                continue

            cx = x + w // 2
            cy = y + h // 2

            boxes.append({
                "center": (cx, cy),
                "bbox": (x, y, w, h),
                "area": rect_area
            })

        return boxes

    def _evaluate_box_position(self, gen_frame: np.ndarray, gt_frame: np.ndarray,
                               obj_bboxes: List[Tuple[int, int, int, int]]) -> Tuple[float, Dict]:
        gt_boxes = self._detect_green_boxes(gt_frame)
        gen_boxes = self._detect_green_boxes(gen_frame)
        details = {'pos_score': 0.0, 'size_score': 0.0, 'count_penalty': 0.0,
                   'gen_box_count': len(gen_boxes), 'gt_box_count': len(gt_boxes),
                   'norm_dist': None, 'size_ratio': None}

        if not gt_boxes or not gen_boxes:
            return 0.0, details

        gt_box = gt_boxes[0]
        gt_cx, gt_cy = gt_box['center']
        gt_area = gt_box['area']

        nearest = min(gen_boxes, key=lambda b: (b['center'][0] - gt_cx) ** 2 + (b['center'][1] - gt_cy) ** 2)
        frame_diag = np.sqrt(gen_frame.shape[0] ** 2 + gen_frame.shape[1] ** 2)
        norm_dist = np.sqrt((nearest['center'][0] - gt_cx) ** 2 + (nearest['center'][1] - gt_cy) ** 2) / frame_diag

        def _interp(x, xs, ys):
            if x <= xs[0]:
                return ys[0]
            for i in range(1, len(xs)):
                if x <= xs[i]:
                    t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
                    return ys[i - 1] + t * (ys[i] - ys[i - 1])
            return ys[-1]
        pos_score = _interp(
            norm_dist, [0.02, 0.08, 0.10, 0.14], [1.0, 0.6, 0.3, 0.0]
        )

        ratio = nearest['area'] / gt_area if gt_area > 0 else 0.0

        # Find the target object: the one from gt_first whose center is inside the GT green box
        gtx, gty, gtw, gth = gt_box['bbox']
        target_bbox = None
        best_dist = float('inf')
        for (sx, sy, sw, sh) in obj_bboxes:
            scx = sx + sw // 2
            scy = sy + sh // 2
            inside = (gtx <= scx <= gtx + gtw) and (gty <= scy <= gty + gth)
            if inside:
                dist = (scx - gt_cx) ** 2 + (scy - gt_cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    target_bbox = (sx, sy, sw, sh)

        # Check if generated box fully encloses the target object
        gx, gy, gw, gh = nearest['bbox']
        if target_bbox is not None:
            tx, ty, tw, th = target_bbox
            covers = (gx <= tx) and (gy <= ty) and (gx + gw >= tx + tw) and (gy + gh >= ty + th)
        else:
            covers = (gx <= gt_cx <= gx + gw) and (gy <= gt_cy <= gy + gh)
        if covers:
            # Covered: only penalize if too large
            if ratio <= 1.5:
                size_score = 1.0
            elif ratio <= 2.5:
                size_score = 0.4
            else:
                size_score = 0.1
        else:
            # Not covered: penalize by area ratio
            if 0.8 <= ratio <= 1.2:
                size_score = 0.6
            elif 0.5 <= ratio <= 2.0:
                size_score = 0.4
            else:
                size_score = 0.0

        n = len(gen_boxes)
        if n == 1:
            count_penalty = 1.0
        elif n == 2:
            count_penalty = 0.3
        else:
            count_penalty = 0.0

        details.update({'pos_score': pos_score, 'size_score': size_score, 'count_penalty': count_penalty,
                        'norm_dist': round(float(norm_dist), 4), 'size_ratio': round(float(ratio), 4),
                        'covers': covers, 'target_bbox': target_bbox})
        return pos_score * size_score * count_penalty, details

    def _pixel_diff_score(self, frame_a: np.ndarray, frame_b: np.ndarray, mask: np.ndarray,
                          thresholds: Tuple[float, float, float, float] = (0.05, 0.15, 0.30, 0.50)) -> Tuple[float, Dict]:
        """
        Tiered score based on ratio of changed pixels (|diff| > 30) within mask.
        thresholds: (t1, t2, t3, t4) → scores 1.0 / 0.8 / 0.5 / 0.2 / 0.0
        Returns (score, detail_dict) where detail includes ratio, changed_px, total_px.
        """
        total = int((mask > 0).sum())
        if total == 0:
            return 1.0, {'ratio': None, 'changed_px': 0, 'total_px': 0, 'note': 'empty_mask'}
        diff = cv2.absdiff(frame_a, frame_b)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed = int((gray_diff[mask > 0] > 20).sum())
        ratio = float(changed) / total
        t1, t2, t3, t4 = thresholds
        if ratio < t1:
            score = 1.0
        elif ratio < t2:
            score = 0.8
        elif ratio < t3:
            score = 0.5
        elif ratio < t4:
            score = 0.2
        else:
            score = 0.0
        return score, {'ratio': round(ratio, 4), 'changed_px': changed, 'total_px': total}

    def _detect_fg_mask(self, first_frame: np.ndarray, last_frame: np.ndarray) -> Tuple[np.ndarray, int, List[Tuple[int, int, int, int]]]:
        gray1 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, changed_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        kernel = np.ones((10, 10), np.uint8)
        changed_mask = cv2.dilate(changed_mask, kernel, iterations=1)

        frame_no_box = first_frame.copy()
        frame_no_box[changed_mask > 0] = (255, 255, 255)

        h, w = first_frame.shape[:2]
        corners = [frame_no_box[5, 5], frame_no_box[5, w-5],
                    frame_no_box[h-5, 5], frame_no_box[h-5, w-5]]
        bg_color = np.mean(corners, axis=0).astype(np.float32)
        dist = np.sqrt(np.sum((frame_no_box.astype(np.float32) - bg_color) ** 2, axis=2))
        binary = (dist > 30).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fg_mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
        obj_bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
            cv2.drawContours(fg_mask, [contour], -1, 255, cv2.FILLED)
            obj_bboxes.append(tuple(cv2.boundingRect(contour)))
        return fg_mask, len(obj_bboxes), obj_bboxes

    def _safe_green_mask(self, frame: np.ndarray, threshold: float = 0.05) -> np.ndarray:
        mask = self._get_green_mask(frame)
        frame_area = frame.shape[0] * frame.shape[1]
        if cv2.countNonZero(mask) / frame_area > threshold:
            return np.zeros_like(mask)
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def _evaluate_objects_preserved(self, first_frame: np.ndarray, last_frame: np.ndarray, fg_mask: np.ndarray) -> Tuple[float, Dict]:
        first_green = self._get_green_mask(first_frame)
        last_green = self._safe_green_mask(last_frame)
        combined_green = cv2.bitwise_or(first_green, last_green)
        mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(combined_green))
        score, detail = self._pixel_diff_score(first_frame, last_frame, mask, thresholds=(0.1, 0.2, 0.30, 0.50))
        return score, detail

    def _evaluate_background_preserved(self, first_frame: np.ndarray, last_frame: np.ndarray, fg_mask: np.ndarray) -> Tuple[float, Dict]:
        first_green = self._get_green_mask(first_frame)
        last_green = self._safe_green_mask(last_frame)
        combined_green = cv2.bitwise_or(first_green, last_green)
        bg_mask = cv2.bitwise_not(cv2.bitwise_or(fg_mask, combined_green))
        if int((bg_mask > 0).sum()) == 0:
            return 0.0, {'ratio': None, 'changed_px': 0, 'total_px': 0, 'note': 'no_bg_region'}
        score, detail = self._pixel_diff_score(first_frame, last_frame, bg_mask, thresholds=(0.015, 0.025, 0.035, 0.05))
        detail['fallback_used'] = (int((cv2.bitwise_not(cv2.bitwise_or(fg_mask, combined_green)) > 0).sum()) == 0)
        return score, detail

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


        gt_last = gt_final_frame
        gt_first = gt_first_frame

        if last_frame.shape != gt_last.shape:
            last_frame = normalize_frame_size(last_frame, gt_last)
        if first_frame.shape != gt_first.shape:
            first_frame = normalize_frame_size(first_frame, gt_first)

        fg_mask, obj_num, obj_bboxes = self._detect_fg_mask(gt_first, gt_last)

        # 1. Box position correct (60%)
        scores['box_position'], box_details = self._evaluate_box_position(last_frame, gt_last, obj_bboxes)

        # 2. Objects preserved (20%)
        scores['objects_preserved'], obj_details = self._evaluate_objects_preserved(gt_first, last_frame, fg_mask)

        # 3. Background preserved (20%)
        scores['background_preserved'], bg_details = self._evaluate_background_preserved(gt_first, last_frame, fg_mask)

        scores['consistency'] = (scores['objects_preserved'] + scores['background_preserved']) / 2

        self._last_task_details = {
            **scores,
            **box_details,
            "obj_num": obj_num,
            'objects_ratio': obj_details.get('ratio'),
            'objects_changed_px': obj_details.get('changed_px'),
            'objects_total_px': obj_details.get('total_px'),
            'objects_note': obj_details.get('note'),
            'bg_ratio': bg_details.get('ratio'),
            'bg_changed_px': bg_details.get('changed_px'),
            'bg_total_px': bg_details.get('total_px'),
            'bg_note': bg_details.get('note'),
            'bg_fallback_used': bg_details.get('fallback_used', False),
        }
        return float((scores['box_position']) * (0.6 + 0.4 * scores['consistency']))


class GridHighestCostEvaluator(BaseEvaluator):
    """
    G-41: Grid highest cost path evaluator.

    Scoring approach:
      1. OCR the 4×4 grid from GT first/final frames (structural classification).
      2. Compute the optimal (highest-cost) path via DFS.
      3. Track the model's yellow-dot movements with legal-move constraints;
         stop as soon as the endpoint (3, 3) is reached.
      4. score = model_cost / optimal_cost × on_path_discount
         where discount = (cells on optimal path) / (total cells visited).
    """

    GRID_SIZE = 4
    MAX_PENALTY = 0.70          # kept for interleave

    AGENT_HSV_LOWER = (20, 80, 80)
    AGENT_HSV_UPPER = (35, 255, 255)
    AGENT_MIN_AREA = 50
    AGENT_MAX_CELL_FRACTION = 0.8

    # Strict cell assignment: the dominant cell must contain at least this
    # fraction of the largest yellow blob's pixels before we treat the
    # agent as being in that cell. Frames where no cell clears the bar
    # are skipped (ambiguous — circle straddles cell boundaries).
    MIN_CELL_PIXEL_FRACTION = 0.6

    # Per-illegal-transition penalty, normalised by optimal-path length.
    VIOLATION_WEIGHT = 0.2
    # Per-frame penalty for "kept moving after reaching the endpoint".
    POST_END_FRAME_WEIGHT = 0.01
    # Bridging thresholds.
    MOTION_PX_THRESHOLD = 5
    AXIS_RATIO = 0.3
    BRIDGE_AXIS_FRACTION = 0.5

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cell_size(frame: np.ndarray) -> float:
        return maze.cell_size(frame, grid_size=4)

    @classmethod
    def _hidden_origin_digit_mask(cls, frame: np.ndarray) -> np.ndarray:
        """Mask the unseen inner crop of cell (0,0) for bg preservation.

        G-41's top-left cost digit is occluded in the input, so background
        fidelity should not reward or penalise whatever appears there later.
        Match the same inner crop that ``_ocr_grid`` uses for cell OCR rather
        than dropping the whole start cell.
        """
        h, w = frame.shape[:2]
        cell_h, cell_w = h // cls.GRID_SIZE, w // cls.GRID_SIZE
        pad = int(min(cell_h, cell_w) * 0.15)
        mask = np.zeros((h, w), dtype=np.uint8)
        y1, y2 = pad, cell_h - pad
        x1, x2 = pad, cell_w - pad
        if y2 > y1 and x2 > x1:
            mask[y1:y2, x1:x2] = 255
        return mask

    def _detect_all_agents(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect yellow Pac-Man blobs (hue 20-35), bounded by cell area."""
        cell_area = self._cell_size(frame) ** 2
        return maze.detect_color_centroids(
            frame, self.AGENT_HSV_LOWER, self.AGENT_HSV_UPPER,
            min_area=self.AGENT_MIN_AREA,
            max_area=cell_area * self.AGENT_MAX_CELL_FRACTION,
        )

    # HSV ranges for the fixed grid markers (green start, red end).
    START_HSV_LOWER = (40, 50, 50)
    START_HSV_UPPER = (85, 255, 255)
    END_HSV_LOWER_1 = (0, 80, 50)
    END_HSV_UPPER_1 = (10, 255, 255)
    END_HSV_LOWER_2 = (170, 80, 50)
    END_HSV_UPPER_2 = (180, 255, 255)
    MARKER_MIN_PIXELS = 100

    def _detect_grid_bbox(
        self, frame: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Locate the 4x4 grid within ``frame`` via green/red corner markers.

        Returns ``(x0, y0, x1, y1)`` covering the grid, or ``None`` if the
        markers can't be found reliably. The bbox may extend slightly past
        the frame edges when the grid is flush with the border.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(
            hsv, np.array(self.START_HSV_LOWER), np.array(self.START_HSV_UPPER),
        )
        red = cv2.bitwise_or(
            cv2.inRange(
                hsv, np.array(self.END_HSV_LOWER_1), np.array(self.END_HSV_UPPER_1),
            ),
            cv2.inRange(
                hsv, np.array(self.END_HSV_LOWER_2), np.array(self.END_HSV_UPPER_2),
            ),
        )
        gys, gxs = np.where(green > 0)
        rys, rxs = np.where(red > 0)
        if len(gxs) < self.MARKER_MIN_PIXELS or len(rxs) < self.MARKER_MIN_PIXELS:
            return None

        gcx, gcy = float(gxs.mean()), float(gys.mean())
        rcx, rcy = float(rxs.mean()), float(rys.mean())
        cw = (rcx - gcx) / (self.GRID_SIZE - 1)
        ch = (rcy - gcy) / (self.GRID_SIZE - 1)
        # Sanity: positive, roughly square
        if cw <= 1.0 or ch <= 1.0 or abs(cw / ch - 1.0) > 0.25:
            return None

        x0 = int(round(gcx - cw / 2))
        y0 = int(round(gcy - ch / 2))
        x1 = int(round(x0 + self.GRID_SIZE * cw))
        y1 = int(round(y0 + self.GRID_SIZE * ch))
        return (x0, y0, x1, y1)

    def _pixel_to_cell_bbox(
        self, px: int, py: int, bbox: Tuple[int, int, int, int],
    ) -> Tuple[int, int]:
        """Pixel ``(x, y)`` -> ``(row, col)`` within the detected grid bbox.

        Points outside the bbox clamp to the nearest edge cell.
        """
        GS = self.GRID_SIZE
        x0, y0, x1, y1 = bbox
        w = max(x1 - x0, 1)
        h = max(y1 - y0, 1)
        col = int((px - x0) * GS // w)
        row = int((py - y0) * GS // h)
        return (min(max(row, 0), GS - 1), min(max(col, 0), GS - 1))

    def _detect_agents_with_area(
        self, frame: np.ndarray,
    ) -> List[Tuple[int, int, float, np.ndarray]]:
        """Detect yellow blobs, returning ``(x, y, area, contour)`` for each.

        The raw contour is carried so :meth:`_primary_cell_strict` can
        check that a minimum fraction of the blob's pixels actually fall
        inside a single cell before accepting the assignment.
        """
        cell_area = self._cell_size(frame) ** 2
        max_area = cell_area * self.AGENT_MAX_CELL_FRACTION
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(self.AGENT_HSV_LOWER),
            np.array(self.AGENT_HSV_UPPER),
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        out: List[Tuple[int, int, float, np.ndarray]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.AGENT_MIN_AREA or area > max_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            out.append((
                int(M["m10"] / M["m00"]),
                int(M["m01"] / M["m00"]),
                float(area),
                cnt,
            ))
        return out

    def _primary_cell_strict(
        self,
        contour: np.ndarray,
        frame_shape: Tuple[int, ...],
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int]]:
        """Cell containing the majority of ``contour``'s filled pixels.

        Returns ``None`` when no single cell holds at least
        ``MIN_CELL_PIXEL_FRACTION`` of the blob's pixels — the circle is
        straddling a boundary and cell assignment would be guesswork.
        Centroid-based assignment (the previous behaviour) would silently
        commit to one side; skipping is strictly better for scoring.
        """
        GS = self.GRID_SIZE
        h, w = frame_shape[:2]

        filled = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
        total = int(np.count_nonzero(filled))
        if total == 0:
            return None

        if bbox is not None:
            x0, y0, x1, y1 = bbox
            step_x = max((x1 - x0) / GS, 1.0)
            step_y = max((y1 - y0) / GS, 1.0)
            origin_x, origin_y = x0, y0
        else:
            step_x = w / GS
            step_y = h / GS
            origin_x = origin_y = 0.0

        best_count = 0
        best_cell: Optional[Tuple[int, int]] = None
        for r in range(GS):
            for c in range(GS):
                cx1 = max(0, min(w, int(round(origin_x + c * step_x))))
                cx2 = max(0, min(w, int(round(origin_x + (c + 1) * step_x))))
                cy1 = max(0, min(h, int(round(origin_y + r * step_y))))
                cy2 = max(0, min(h, int(round(origin_y + (r + 1) * step_y))))
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                count = int(np.count_nonzero(filled[cy1:cy2, cx1:cx2]))
                if count > best_count:
                    best_count = count
                    best_cell = (r, c)

        if best_cell is None:
            return None
        if best_count / total < self.MIN_CELL_PIXEL_FRACTION:
            return None
        return best_cell

    # ------------------------------------------------------------------
    # OCR: read grid costs via structural digit classification
    # ------------------------------------------------------------------

    @staticmethod
    def _binarize_cell(gray: np.ndarray) -> np.ndarray:
        """Binarise a grayscale cell crop (white fg on black bg)."""
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def _ocr_grid(
        self,
        first_frame: np.ndarray,
        final_frame: Optional[np.ndarray] = None,
    ) -> Optional[List[List[int]]]:
        """Read 4×4 grid costs using structural digit classification.

        Cell (0, 0) is always left at 0: Pac-Man occludes it in the first
        frame so the model has no information about the top-left cost,
        and it is excluded from scoring (both ``optimal_cost`` and
        ``model_cost`` naturally drop it). ``final_frame`` is accepted
        for backwards compatibility but no longer read.
        """
        del final_frame  # unused since (0, 0) is excluded from scoring
        GS = self.GRID_SIZE
        h, w = first_frame.shape[:2]
        cell_h, cell_w = h // GS, w // GS
        pad = int(min(cell_h, cell_w) * 0.15)

        grid: List[List[int]] = [[0] * GS for _ in range(GS)]
        for r in range(GS):
            for c in range(GS):
                if r == 0 and c == 0:
                    continue

                y1 = r * cell_h + pad
                y2 = (r + 1) * cell_h - pad
                x1 = c * cell_w + pad
                x2 = (c + 1) * cell_w - pad
                gray = cv2.cvtColor(first_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                binary = self._binarize_cell(gray)
                tens = _g41_extract_tens_digit(binary)
                digit = _g41_classify_tens_digit(tens)
                grid[r][c] = digit * 10

        # Validate: every non-origin cell must be a multiple of 10 in [10, 90]
        valid = {10, 20, 30, 40, 50, 60, 70, 80, 90}
        for r in range(GS):
            for c in range(GS):
                if r == 0 and c == 0:
                    continue
                if grid[r][c] not in valid:
                    return None
        return grid

    # ------------------------------------------------------------------
    # Model path tracking
    # ------------------------------------------------------------------

    def _find_bridge_cell(
        self,
        frm: Tuple[int, int],
        to: Tuple[int, int],
        visited: Set[Tuple[int, int]],
        grid: Optional[List[List[int]]],
        optimal_cells: Optional[Set[Tuple[int, int]]],
        seen_loose: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """Pick an unvisited cell adjacent to both ``frm`` and ``to``.

        Used to recover a legal 2-step path when the strict cell-majority
        tracker drops the intermediate cell during smooth animation.
        Requires the bridge to appear in ``seen_loose`` — the set of
        cells the Pac-Man centroid actually visited since the last
        committed cell — so we only rescue genuine smooth animation,
        not pure diagonal teleports (a ``(0,0)→(1,1)→(2,2)`` jump never
        passes through any in-between cell).

        Prefers bridges that lie on some optimal route, then higher-cost
        bridges (most generous reading of the path). Returns ``None``
        when no candidate exists.
        """
        GS = self.GRID_SIZE

        def neigh(c: Tuple[int, int]) -> Set[Tuple[int, int]]:
            r, col = c
            out: Set[Tuple[int, int]] = set()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, col + dc
                if 0 <= nr < GS and 0 <= nc < GS:
                    out.add((nr, nc))
            return out

        candidates = (neigh(frm) & neigh(to)) - visited
        candidates = candidates & seen_loose
        if not candidates:
            return None

        def sort_key(c: Tuple[int, int]) -> Tuple[int, int]:
            on_opt = 1 if (optimal_cells is not None and c in optimal_cells) else 0
            cost = grid[c[0]][c[1]] if grid is not None else 0
            return (on_opt, cost)

        return max(candidates, key=sort_key)

    def _track_model_path(
        self,
        video_frames: List[np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        grid: Optional[List[List[int]]] = None,
        optimal_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]], int]:
        """Track the primary yellow blob across frames.

        Each frame's *primary* blob is the largest-area detection. Consecutive
        frames with the same primary cell collapse to a single event, so a
        model that lingers on a cell is not double-counted.

        Transitions are classified:
        - legal (adjacent + unvisited): advance the pointer, append to path
        - 2-step (``dr + dc == 2``, target unvisited, a common unvisited
          neighbour exists): insert the neighbour as a *bridge* and
          advance. Recovers the cell smoothly animated through but
          dropped by the strict cell-majority tracker.
        - ``diagonal`` (dr == 1 and dc == 1, no bridge): violation
        - ``teleport`` (dr + dc >= 2, not diagonal, no bridge): violation
        - ``revisit`` (target already in visited): violation
        - ``post_endpoint`` (pointer already reached end, primary drifts
          off of end): violation — task should have stopped
        """
        GS = self.GRID_SIZE
        end = (GS - 1, GS - 1)
        current = (0, 0)
        visited: Set[Tuple[int, int]] = {(0, 0)}
        path: List[Tuple[int, int]] = [(0, 0)]
        violations: List[Dict[str, Any]] = []
        last_primary_cell: Optional[Tuple[int, int]] = None
        post_endpoint_frames = 0
        seen_loose: Set[Tuple[int, int]] = {(0, 0)}
        prev_xy: Optional[Tuple[int, int]] = None
        axis_frames = 0
        motion_frames = 0

        for frame_idx, frame in enumerate(video_frames):
            agents = self._detect_agents_with_area(frame)
            if not agents:
                last_primary_cell = None
                continue

            x, y, _, contour = max(agents, key=lambda a: a[2])
            if bbox is not None:
                seen_loose.add(self._pixel_to_cell_bbox(x, y, bbox))
            if prev_xy is not None:
                dx = abs(x - prev_xy[0])
                dy = abs(y - prev_xy[1])
                if dx + dy > self.MOTION_PX_THRESHOLD:
                    motion_frames += 1
                    mn, mx = min(dx, dy), max(dx, dy)
                    if mx > 0 and mn <= self.AXIS_RATIO * mx:
                        axis_frames += 1
            prev_xy = (x, y)
            primary_cell = self._primary_cell_strict(
                contour, frame.shape, bbox,
            )
            if primary_cell is None:
                # Circle straddles cells — ambiguous, treat as no detection
                # rather than silently committing to a centroid-biased cell.
                last_primary_cell = None
                continue

            # Count every frame where the task is done but the agent is off
            # the endpoint. Before dedup so duration is counted accurately.
            if current == end and primary_cell != end:
                post_endpoint_frames += 1

            if primary_cell == last_primary_cell:
                continue
            last_primary_cell = primary_cell

            if current == end:
                # Task complete; any drift off the end cell is a violation
                if primary_cell != end:
                    violations.append({
                        "frame": frame_idx,
                        "type": "post_endpoint",
                        "from": current,
                        "to": primary_cell,
                    })
                continue

            if primary_cell == current:
                continue

            dr = abs(primary_cell[0] - current[0])
            dc = abs(primary_cell[1] - current[1])

            committed = False
            if primary_cell in visited:
                vtype = "revisit"
            elif dr + dc == 1:
                # Legal adjacent step
                current = primary_cell
                visited.add(primary_cell)
                path.append(primary_cell)
                committed = True
            elif dr + dc == 2:
                axis_ok = (
                    motion_frames > 0
                    and axis_frames
                    >= self.BRIDGE_AXIS_FRACTION * motion_frames
                )
                bridge = None
                if axis_ok:
                    bridge = self._find_bridge_cell(
                        current, primary_cell, visited,
                        grid, optimal_cells, seen_loose,
                    )
                if bridge is not None:
                    visited.add(bridge)
                    visited.add(primary_cell)
                    path.append(bridge)
                    path.append(primary_cell)
                    current = primary_cell
                    committed = True
                else:
                    vtype = "diagonal" if (dr == 1 and dc == 1) else "teleport"
            else:  # dr + dc >= 3
                vtype = "teleport"

            if committed:
                seen_loose = {primary_cell}
                axis_frames = 0
                motion_frames = 0
                continue

            violations.append({
                "frame": frame_idx,
                "type": vtype,
                "from": current,
                "to": primary_cell,
            })
        return path, violations, post_endpoint_frames

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
        """score = min(1, model_cost / optimal_cost * legality)

        Off-optimal cells collect lower cost than optimal ones, so the
        cost ratio already captures that. The legality factor separately
        penalises illegal transitions (diagonal, teleport, revisit) and
        movement after the endpoint has been reached.
        """
        if not video_frames or gt_first_frame is None or gt_final_frame is None:
            self._last_task_details = {"error": "missing frames"}
            return 0.0

        # 1. OCR grid costs from GT frames
        grid = self._ocr_grid(gt_first_frame, gt_final_frame)
        if grid is None:
            self._last_task_details = {"error": "ocr_failed"}
            return 0.0

        # 2. Optimal path via DFS
        optimal_path, optimal_cost, optimal_cells = _g41_find_optimal_path(
            grid, self.GRID_SIZE,
        )
        if optimal_cost <= 0:
            self._last_task_details = {
                "grid": grid, "optimal_cost": optimal_cost,
                "error": "zero_optimal_cost",
            }
            return 0.0

        # 3. Detect grid bbox in model-video coordinates (may differ from GT).
        model_bbox = self._detect_grid_bbox(video_frames[0])
        if model_bbox is None and len(video_frames) > 1:
            for idx in (len(video_frames) // 4, len(video_frames) // 2):
                model_bbox = self._detect_grid_bbox(video_frames[idx])
                if model_bbox is not None:
                    break

        # 4. Track model's path + violation log + post-endpoint frame count.
        model_path, violations, post_end_frames = self._track_model_path(
            video_frames, bbox=model_bbox,
            grid=grid, optimal_cells=optimal_cells,
        )

        # 5. Model cost
        model_cost = sum(grid[r][c] for r, c in model_path)

        # 6. Legality: two independent penalties combined multiplicatively.
        illegal_count = sum(
            1 for v in violations if v["type"] != "post_endpoint"
        )
        post_end_entries = len(violations) - illegal_count
        legality_illegal = max(
            0.0,
            1.0 - self.VIOLATION_WEIGHT
            * (illegal_count + post_end_entries) / len(optimal_path),
        )
        legality_post_end = max(
            0.0, 1.0 - self.POST_END_FRAME_WEIGHT * post_end_frames,
        )
        legality = legality_illegal * legality_post_end

        task_score = min(1.0, model_cost / optimal_cost * legality)

        bg_preservation = maze.background_preservation_frames(
            video_frames, gt_first_frame,
            detector=self._detect_all_agents,
            base_exclude_mask=self._hidden_origin_digit_mask(gt_first_frame),
            grid_size=self.GRID_SIZE,
        )
        # Background preservation is a penalty multiplier
        final_score = task_score * (0.6 + 0.4 * bg_preservation)

        on_path = sum(1 for cell in model_path if cell in optimal_cells)

        violations_by_type: Dict[str, int] = {}
        for v in violations:
            violations_by_type[v["type"]] = violations_by_type.get(v["type"], 0) + 1

        self._last_task_details = {
            "grid": grid,
            "optimal_path": [list(c) for c in optimal_path],
            "optimal_cost": optimal_cost,
            "num_optimal_cells": len(optimal_cells),
            "model_bbox": list(model_bbox) if model_bbox is not None else None,
            "model_path": [list(c) for c in model_path],
            "model_cost": model_cost,
            "on_path_steps": on_path,
            "off_path_steps": len(model_path) - on_path,
            "violations": len(violations),
            "violations_by_type": violations_by_type,
            "violation_details": violations[:20],
            "post_endpoint_frames": post_end_frames,
            "legality_illegal": round(float(legality_illegal), 4),
            "legality_post_end": round(float(legality_post_end), 4),
            "legality": round(float(legality), 4),
            "reached_endpoint": model_path[-1] == (self.GRID_SIZE - 1, self.GRID_SIZE - 1),
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

        - OCR the 4x4 grid from the input + GT final frame (fallbacks apply).
        - optimal_path via DFS; optimal_cells = union of all optimal-cost paths.
        - drawn = cells newly drawn in pred vs input_frame (grid_size=4).
        - score = proximity x coverage x cost_ratio
          where cost_ratio = min(1, model_cost / optimal_cost) prevents
          trivially-drawing-every-cell from scoring 1.0.
        """
        if not pred_images or input_frame is None:
            self._last_task_details = {"error": "no_input_or_pred"}
            return 0.0

        # OCR the grid.
        first_for_ocr = input_frame
        final_for_ocr = pred_images[-1] if pred_images else None
        grid = None
        if first_for_ocr is not None and final_for_ocr is not None:
            grid = self._ocr_grid(first_for_ocr, final_for_ocr)
        if grid is None:
            fallback_first = gt_images[0] if gt_images else input_frame
            fallback_final = gt_final_frame or input_frame
            if fallback_first is not None and fallback_final is not None:
                grid = self._ocr_grid(fallback_first, fallback_final)
        if grid is None:
            self._last_task_details = {"error": "ocr_failed"}
            return 0.0

        optimal_path, optimal_cost, optimal_cells = _g41_find_optimal_path(
            grid, self.GRID_SIZE,
        )
        if optimal_cost <= 0 or not optimal_path:
            self._last_task_details = {
                "error": "zero_optimal_cost",
                "grid": grid,
                "optimal_cost": optimal_cost,
            }
            return 0.0

        counts = maze.cell_draw_counts(
            pred_images, input_frame, grid_size=self.GRID_SIZE,
        )
        start_cell = (0, 0)
        end_cell = (self.GRID_SIZE - 1, self.GRID_SIZE - 1)
        drawn = set(counts) | {start_cell, end_cell}

        walk = maze.simulate_walk_through_drawn(
            drawn=drawn, start=start_cell, end=end_cell,
            grid_size=self.GRID_SIZE,
        )

        gt_drawn = maze.cells_from_pred_diff(
            gt_images, input_frame, grid_size=self.GRID_SIZE,
        )
        gt_drawn = (
            gt_drawn | {start_cell, end_cell}
            if gt_drawn else set(optimal_cells)
        )

        path_score, details = maze.score_interleave_walk(
            walk=walk, drawn=drawn, optimal_cells=gt_drawn,
            path_length=len(gt_drawn),
            draw_counts=counts,
        )

        # Cost bonus/penalty: reward the model for visiting high-cost cells.
        drawn_cost = sum(
            grid[r][c] for (r, c) in drawn
            if 0 <= r < self.GRID_SIZE and 0 <= c < self.GRID_SIZE
        )
        cost_ratio = min(1.0, drawn_cost / optimal_cost) if optimal_cost > 0 else 0.0
        task_score = path_score * cost_ratio

        hidden_origin_mask = self._hidden_origin_digit_mask(input_frame)
        pred_mask = maze.pred_diff_mask(pred_images, input_frame)
        if pred_mask is None:
            exclude_mask = hidden_origin_mask
        else:
            if pred_mask.shape[:2] != hidden_origin_mask.shape[:2]:
                pred_mask = cv2.resize(
                    pred_mask,
                    (hidden_origin_mask.shape[1], hidden_origin_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            exclude_mask = cv2.bitwise_or(
                pred_mask.astype(np.uint8), hidden_origin_mask,
            )
        bg_preservation = maze.background_preservation_image(
            pred_images[-1], input_frame, exclude_mask=exclude_mask,
        )
        # Background preservation is a penalty multiplier
        final_score = task_score * (0.6 + 0.4 * bg_preservation)

        details.update({
            "grid": grid,
            "optimal_path": [list(c) for c in optimal_path],
            "optimal_cost": optimal_cost,
            "drawn_cost": int(drawn_cost),
            "path_score": round(float(path_score), 4),
            "cost_ratio": round(float(cost_ratio), 4),
            "task_score": round(float(task_score), 4),
            "bg_preservation": round(float(bg_preservation), 4),
            "final_score": round(float(final_score), 4),
        })
        self._last_task_details = details
        return final_score


class UnderstandSceneStructureEvaluator(BaseEvaluator):
    """
    G-43: Understand scene structure evaluator.

    Scoring:
    - accuracy         (60%): IoU-based matching of green contours vs GT
    - back_consistency (20%): white background similarity between final and GT frames
    - fore_consistency (20%): non-white non-green foreground similarity
    """

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: Optional[List[np.ndarray]],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: np.ndarray,
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 1:
            return 0.0

        final_frame = video_frames[-1]
        canvas_size = (final_frame.shape[0], final_frame.shape[1])

        gt_ref = gt_frames[-1] if gt_frames else gt_final_frame
        if gt_ref.shape[:2] != final_frame.shape[:2]:
            gt_ref = cv2.resize(gt_ref, (final_frame.shape[1], final_frame.shape[0]))

        gt_contours = detect_closed_contours_by_color(gt_ref, COLOR_BOUNDS['green'], min_area=300)

        gen_contours = detect_closed_contours_by_color(
            final_frame, COLOR_BOUNDS['green'], min_area=300,
            hull_fallback=True,
            ref_area=max((cv2.contourArea(c) for c in gt_contours), default=None),
        )

        match_results = match_contours(gt_contours, gen_contours, iou_threshold=0.1, canvas_size=canvas_size)

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
            contained = any(
                cv2.pointPolygonTest(gt_cnt, (float(gx), float(gy)), False) >= 0
                for (gx, gy) in gen_centroids
            )
            if contained:
                n_contained += 1
                gt_mask = np.zeros(canvas_size, dtype=np.uint8)
                cv2.drawContours(gt_mask, [gt_cnt], -1, 255, -1)
                gen_mask = np.zeros(canvas_size, dtype=np.uint8)
                cv2.drawContours(gen_mask, gen_contours, -1, 255, -1)
                _gt_area = int((gt_mask > 0).sum())
                coverage = (int(((gt_mask > 0) & (gen_mask > 0)).sum()) / _gt_area) if _gt_area else 0.0
                per_gt_scores.append(max(base, 0.9 * coverage))
            else:
                per_gt_scores.append(base)
        accuracy = float(np.mean(per_gt_scores)) if per_gt_scores else 0.0
        n_effective_matches = max(len(valid_ious), n_contained)
        accuracy = accuracy * calculate_list_length_penalty(len(gt_contours), n_effective_matches, len(gen_contours))

        back_consistency = score_background_similarity(gt_ref, final_frame)

        fore_consistency = score_foreground_similarity(gt_ref, final_frame, COLOR_BOUNDS['green'])

        consistency = 0.5 * back_consistency + 0.5 * fore_consistency
        score = accuracy * (0.6 + 0.4 * consistency)

        self._last_task_details = {
            'accuracy': accuracy,
            'back_consistency': back_consistency,
            'fore_consistency': fore_consistency,
        }
        return score

class KeyDoorMatchingEvaluator(BaseEvaluator):
    """
    G-45: Key-door matching evaluator.

    Scoring follows the shared maze family (G-15 / G-16 / G-18 / G-47):

      score = proximity
            × (1 − 0.30 × continuity_penalty)
            × 0.5^key_missed          # did not visit the target-colour key
            × 0.5^num_wall_hit_cells  # stepped on a wall
            × coverage

    Target colour is whichever door vanished in the GT video between the
    first and final frame — the GT itself records the correct answer, so
    evaluation is colour-robust (samples cycle through blue/red/yellow/…).

    **Semantics.** In GT, the agent walks from start to the correct key
    and then stops; picking up the right key remotely unlocks (and
    visually removes) its same-colour door. So the terminal cell for
    path evaluation is the *matching key*, not the door. The door colour
    only serves as the authoritative label for "which key was right".
    """

    MAX_PENALTY = 0.70
    PENALTY_FLOOR = 0.05
    EXTRA_AGENT_PENALTY = 0.20
    COVERAGE_GAP_THRESHOLD = 2
    DISAPPEAR_RATE_CAP = 1.0

    AGENT_HSV_GREEN_LOWER = (35, 80, 80)
    AGENT_HSV_GREEN_UPPER = (85, 255, 255)
    AGENT_HSV_YELLOW_LOWER = (20, 100, 100)
    AGENT_HSV_YELLOW_UPPER = (35, 255, 255)

    # Cached grid size — set from the landmark frame at the start of each
    # video-level evaluation.  Used by _agent_area_bounds so per-frame
    # agent area bounds stay consistent.
    _grid_size: int = 13

    # ------------------------------------------------------------------
    # Grid-structure inference (same heuristics as G-47)
    # ------------------------------------------------------------------

    def _infer_grid_size(self, frame: np.ndarray) -> int:
        """Infer grid_size from outer black-border thickness.

        Scans several columns for the first non-black pixel from the top,
        takes the median, then divides the frame height by that. Clamped
        to [8, 32]; falls back to 13 on ambiguous input.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        border_tops: List[int] = []
        step = max(1, w // 8)
        for x in range(step, w, step):
            for y in range(h // 2):
                if gray[y, x] > 80:
                    if y >= 5:
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
                if float(np.mean(inner < 60)) > 0.7:
                    walls.add((r, c))
        return walls

    # ------------------------------------------------------------------
    # Agent detection (circular green/yellow blob)
    # ------------------------------------------------------------------

    def _agent_area_bounds(self, frame: np.ndarray) -> Tuple[float, float]:
        cell_px = max(frame.shape[:2]) / float(self._grid_size)
        cell_area = cell_px ** 2
        return (max(100.0, cell_area * 0.08), cell_area * 2.0)

    def _detect_all_agents(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Circular green/yellow blob centroids. Circularity filter keeps
        filled diamond keys (whose colour ranges overlap the agent's)
        from being detected as agents."""
        min_area, max_area = self._agent_area_bounds(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for lo, hi in (
            (self.AGENT_HSV_GREEN_LOWER, self.AGENT_HSV_GREEN_UPPER),
            (self.AGENT_HSV_YELLOW_LOWER, self.AGENT_HSV_YELLOW_UPPER),
        ):
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            agents: List[Tuple[int, int]] = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                perim = cv2.arcLength(cnt, True)
                if perim <= 0:
                    continue
                circularity = 4 * np.pi * area / (perim ** 2)
                if circularity < 0.4:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                agents.append(
                    (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])),
                )
            if agents:
                return agents
        return []

    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        agents = self._detect_all_agents(frame)
        return agents[0] if agents else None

    # ------------------------------------------------------------------
    # Target colour + cell-level proximity
    # ------------------------------------------------------------------

    def _find_disappeared_door_colors(
        self,
        first_doors: List[Dict],
        last_doors: List[Dict],
    ) -> Set[str]:
        """Colours whose door-count dropped from first → last frame."""
        first_counts: Dict[str, int] = {}
        for d in first_doors:
            first_counts[d["color"]] = first_counts.get(d["color"], 0) + 1
        last_counts: Dict[str, int] = {}
        for d in last_doors:
            last_counts[d["color"]] = last_counts.get(d["color"], 0) + 1
        return {
            color for color, cnt in first_counts.items()
            if last_counts.get(color, 0) < cnt
        }

    def _score_cell_proximity(
        self,
        video_frames: List[np.ndarray],
        optimal_cells: Set[Tuple[int, int]],
        grid_size: int,
    ) -> float:
        """Per-frame proximity: 1.0 if agent's cell is on the optimal tour,
        else linear decay by cell-Manhattan distance. Same as G-47."""
        if not optimal_cells or not video_frames:
            return 0.0
        opt_arr = np.array(list(optimal_cells))
        max_cells = 2.0
        frame_scores: List[float] = []
        for frame in video_frames:
            blobs = self._detect_all_agents(frame)
            if not blobs:
                frame_scores.append(0.0)
                continue
            cell_dists: List[int] = []
            for ax, ay in blobs:
                cell = maze.pixel_to_cell(ax, ay, frame.shape, grid_size)
                d = int(np.min(
                    np.abs(opt_arr[:, 0] - cell[0])
                    + np.abs(opt_arr[:, 1] - cell[1])
                ))
                cell_dists.append(d)
            cell_dists.sort()
            best = cell_dists[0]
            base = max(0.0, 1.0 - best / max_cells)
            n_hallucinated = sum(1 for d in cell_dists[1:] if d > 1)
            extra_pen = min(1.0, n_hallucinated * self.EXTRA_AGENT_PENALTY)
            frame_scores.append(base * (1.0 - extra_pen))
        return sum(frame_scores) / len(frame_scores) if frame_scores else 0.0

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
        if not video_frames or gt_first_frame is None or gt_final_frame is None:
            self._last_task_details = {"error": "missing_frames"}
            return 0.0

        # --- 1. Grid structure from GT first frame --------------------
        grid_size = self._infer_grid_size(gt_first_frame)
        self._grid_size = grid_size
        cell = maze.cell_size(gt_first_frame, grid_size)

        # --- 2. Target colour: door that vanished in GT ---------------
        doors_gt_first = self._detect_doors(gt_first_frame)
        doors_gt_last = self._detect_doors(gt_final_frame)
        disappeared = self._find_disappeared_door_colors(
            doors_gt_first, doors_gt_last,
        )
        if not disappeared:
            self._last_task_details = {
                "error": "no_door_disappeared_in_gt",
                "grid_size": grid_size,
                "num_first_doors": len(doors_gt_first),
                "num_last_doors": len(doors_gt_last),
            }
            return 0.0
        target_color = sorted(disappeared)[0]

        keys_first = self._detect_keys(gt_first_frame)
        target_key = next(
            (k for k in keys_first if k["color"] == target_color), None,
        )
        target_door = next(
            (d for d in doors_gt_first if d["color"] == target_color), None,
        )
        if target_key is None or target_door is None:
            self._last_task_details = {
                "error": "target_landmarks_missing",
                "target_color": target_color,
                "grid_size": grid_size,
            }
            return 0.0

        target_key_cell = maze.pixel_to_cell(
            target_key["center"][0], target_key["center"][1],
            gt_first_frame.shape, grid_size,
        )
        target_door_cell = maze.pixel_to_cell(
            target_door["center"][0], target_door["center"][1],
            gt_first_frame.shape, grid_size,
        )

        # --- 3. Agent start cell --------------------------------------
        agent_start = self._detect_agent(gt_first_frame)
        if agent_start is None:
            agent_start = self._detect_agent(video_frames[0])
        if agent_start is None:
            self._last_task_details = {
                "error": "agent_not_detected_in_start_frame",
                "target_color": target_color,
                "grid_size": grid_size,
            }
            return 0.0
        start_cell = maze.pixel_to_cell(
            agent_start[0], agent_start[1],
            gt_first_frame.shape, grid_size,
        )

        # --- 4. Walls -------------------------------------------------
        walls = self._detect_wall_cells(gt_first_frame, grid_size)
        landmark_cells: Set[Tuple[int, int]] = {start_cell, target_key_cell, target_door_cell}
        for item in (*keys_first, *doors_gt_first):
            landmark_cells.add(maze.pixel_to_cell(
                item["center"][0], item["center"][1],
                gt_first_frame.shape, grid_size,
            ))
        walls.difference_update(landmark_cells)

        # --- 5. Reference path from GT trajectory ---------------------
        gt_path_pixels = maze.extract_trajectory(
            gt_frames, self._detect_all_agents,
        )
        if not gt_path_pixels:
            self._last_task_details = {
                "error": "gt_agent_untrackable",
                "target_color": target_color,
                "grid_size": grid_size,
            }
            return 0.0
        ref_points = gt_path_pixels
        optimal_cells = {
            maze.pixel_to_cell(p[0], p[1], gt_first_frame.shape, grid_size)
            for p in gt_path_pixels
        }
        shortest = len(optimal_cells) - 1

        # --- 6. Score components --------------------------------------
        proximity = self._score_cell_proximity(
            video_frames, optimal_cells, grid_size,
        )
        _term_blobs = self._detect_all_agents(video_frames[-1]) if video_frames else []
        if _term_blobs:
            _term_d = min(
                abs(maze.pixel_to_cell(bx, by, video_frames[-1].shape, grid_size)[0] - target_key_cell[0])
                + abs(maze.pixel_to_cell(bx, by, video_frames[-1].shape, grid_size)[1] - target_key_cell[1])
                for bx, by in _term_blobs
            )
            _reach = max(0.0, 1.0 - _term_d / max(float(shortest), 1.0))
        else:
            _reach = 0.0
        proximity = proximity * (0.3 + 0.7 * _reach)
        coverage = maze.score_coverage_completion(
            video_frames, ref_points, start_cell, target_key_cell, walls, cell,
            self._detect_all_agents,
            gap_threshold=self.COVERAGE_GAP_THRESHOLD,
            grid_size=grid_size,
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
            trim_edge_gaps=True,
        )
        continuity_factor = 1.0 - self.MAX_PENALTY * cont_penalty

        visited_cells = {
            c for c in maze.best_blob_cells(
                video_frames, self._detect_all_agents, ref_points,
                grid_size=grid_size,
            ) if c is not None
        }
        key_visited = target_key_cell in visited_cells
        key_multiplier = 1.0 if key_visited else 0.25

        hit_report = maze.obstacle_hit_report(
            video_frames, walls, self._detect_all_agents,
            grid_size=grid_size, reference_points=ref_points,
        )
        num_wall_hit_cells = len(hit_report["hit_cells"])
        wall_multiplier = max(0.4, 1.0 - 0.15 * num_wall_hit_cells)

        score_without_coverage = (
            proximity * continuity_factor
            * key_multiplier * wall_multiplier
        )
        score = ((proximity + continuity_factor + coverage) / 3.0) * (0.4 + 0.6 * wall_multiplier) * (0.4 + 0.6 * key_multiplier)
        self._last_task_details = {
            "grid_size": grid_size,
            "target_color": target_color,
            "start_cell": list(start_cell),
            "target_key_cell": list(target_key_cell),
            "target_door_cell": list(target_door_cell),
            "num_walls": len(walls),
            "shortest_distance": shortest,
            "proximity": round(float(proximity), 4),
            "coverage": round(float(coverage), 4),
            "continuity_penalty": round(float(cont_penalty), 4),
            "continuity_factor": round(float(continuity_factor), 4),
            "key_visited": bool(key_visited),
            "key_multiplier": round(float(key_multiplier), 4),
            "wall_hit_cells": hit_report["hit_cells"],
            "num_wall_hit_cells": num_wall_hit_cells,
            "wall_multiplier": round(float(wall_multiplier), 6),
            "score_without_coverage": round(float(score_without_coverage), 4),
            "final_score": round(float(score), 4),
        }
        return score

    def _detect_keys(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect diamond-shaped keys of various colors.
        Keys are filled shapes (high fill ratio) with 4 vertices.
        """
        keys = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
            'cyan': ([85, 100, 100], [100, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
        }
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 10000:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                fill_ratio = area / rect_area if rect_area > 0 else 1
                
                # Keys are FILLED shapes (high fill ratio > 0.7)
                if fill_ratio < 0.7:
                    continue
                
                # Check if roughly diamond-shaped (4 vertices)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if 3 <= len(approx) <= 6:  # Diamond-like shapes
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        keys.append({
                            'color': color_name,
                            'center': (cx, cy),
                            'area': area,
                            'fill_ratio': fill_ratio
                        })
        
        return keys
    
    def _detect_doors(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect hollow rectangular doors.
        Doors are HOLLOW shapes (low fill ratio < 0.6) with 4 vertices.
        """
        doors = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
            'cyan': ([85, 100, 100], [100, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
        }
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 10000:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                fill_ratio = area / rect_area if rect_area > 0 else 1
                
                # Doors are HOLLOW shapes (low fill ratio < 0.6)
                if fill_ratio >= 0.6:
                    continue
                
                # Check if roughly rectangular (4 vertices)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if 3 <= len(approx) <= 6:  # Rectangular-like shapes
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        doors.append({
                            'color': color_name,
                            'center': (cx, cy),
                            'area': area,
                            'fill_ratio': fill_ratio
                        })
        
        return doors

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Interleave: cell-based scoring aligned with the video method.

        - target_color = whichever door colour vanished between
          ``gt_images[0]`` and ``gt_images[-1]``.
        - optimal_cells = cells on the GT agent trajectory (GT scores 1.0
          by construction; BFS shortest can be wrong due to noisy wall
          detection).
        - drawn = cells newly drawn in pred vs input_frame.
        - required_cells = [target_key_cell].
        - wall_cells = detected walls with landmarks cleared.
        - score aggregation mirrors video: proximity / coverage / path validity
          are averaged, then key and wall penalties are applied with floors.
        """
        if not pred_images or not gt_images or input_frame is None:
            self._last_task_details = {"error": "no_input_or_pred_or_gt"}
            return 0.0

        # 1. Grid size inference (same heuristic as the video method)
        landmark_first = input_frame
        gt_last = gt_final_frame if gt_final_frame is not None else gt_images[-1]
        grid_size = self._infer_grid_size(landmark_first)
        self._grid_size = grid_size

        # 2. Target colour: door that disappeared (input -> GT final)
        doors_first = self._detect_doors(landmark_first)
        doors_last = self._detect_doors(gt_last)
        disappeared = self._find_disappeared_door_colors(doors_first, doors_last)
        if not disappeared:
            self._last_task_details = {
                "error": "no_door_disappeared_in_gt",
                "grid_size": grid_size,
            }
            return 0.0
        target_color = sorted(disappeared)[0]

        keys_first = self._detect_keys(landmark_first)
        target_key = next(
            (k for k in keys_first if k["color"] == target_color), None,
        )
        target_door = next(
            (d for d in doors_first if d["color"] == target_color), None,
        )
        if target_key is None or target_door is None:
            self._last_task_details = {
                "error": "target_landmarks_missing",
                "target_color": target_color,
                "grid_size": grid_size,
            }
            return 0.0
        target_key_cell = maze.pixel_to_cell(
            target_key["center"][0], target_key["center"][1],
            landmark_first.shape, grid_size,
        )
        target_door_cell = maze.pixel_to_cell(
            target_door["center"][0], target_door["center"][1],
            landmark_first.shape, grid_size,
        )

        # 3. Start cell from input_frame (fallback to gt_images[0])
        agent_start = self._detect_agent(landmark_first)
        if agent_start is None and gt_images:
            agent_start = self._detect_agent(gt_images[0])
        if agent_start is None:
            self._last_task_details = {
                "error": "agent_not_detected",
                "target_color": target_color,
                "grid_size": grid_size,
            }
            return 0.0
        start_cell = maze.pixel_to_cell(
            agent_start[0], agent_start[1], landmark_first.shape, grid_size,
        )

        # 4. Walls + clear landmark cells
        walls = self._detect_wall_cells(landmark_first, grid_size)
        landmark_cells: Set[Tuple[int, int]] = {
            start_cell, target_key_cell, target_door_cell,
        }
        for item in (*keys_first, *doors_first):
            landmark_cells.add(maze.pixel_to_cell(
                item["center"][0], item["center"][1],
                landmark_first.shape, grid_size,
            ))
        walls.difference_update(landmark_cells)

        # 5. Reference "optimal" cells: use the drawn GT path itself.
        optimal_cells = maze.cells_from_pred_diff(
            gt_images, input_frame, grid_size=grid_size,
        )
        if not optimal_cells:
            self._last_task_details = {
                "error": "gt_path_untrackable",
                "target_color": target_color,
                "grid_size": grid_size,
            }
            return 0.0

        # 6. Extract the image equivalents of video's path components.  A valid
        # connected walk is image-mode continuity evidence; length_factor charges
        # for excessive detours/scribbles without turning partial paths into an
        # automatic zero.
        counts = maze.cell_draw_counts(
            pred_images, input_frame, grid_size=grid_size,
        )
        drawn = set(counts)
        walk = maze.simulate_walk_through_drawn(
            drawn=drawn, start=start_cell, end=target_key_cell,
            grid_size=grid_size,
        )
        on_path = drawn & set(optimal_cells)
        total_pixels = sum(counts.values())
        on_path_pixels = sum(n for c, n in counts.items() if c in optimal_cells)
        proximity = on_path_pixels / total_pixels if total_pixels else 0.0
        coverage = min(1.0, len(on_path) / max(len(optimal_cells), 1))
        length_factor = (
            min(1.0, len(optimal_cells) / len(drawn)) if drawn else 0.0
        )
        continuity_factor = length_factor if walk else 0.0

        key_visited = target_key_cell in drawn
        key_multiplier = 1.0 if key_visited else 0.25
        wall_hit_cells = sorted(drawn & walls)
        num_wall_hit_cells = len(wall_hit_cells)
        wall_multiplier = max(0.4, 1.0 - 0.15 * num_wall_hit_cells)

        path_core = (proximity + continuity_factor + coverage) / 3.0
        score = (
            path_core
            * (0.4 + 0.6 * wall_multiplier)
            * (0.4 + 0.6 * key_multiplier)
        )
        details = {
            "walk_reachable": bool(walk),
            "walk": [list(c) for c in (walk or [])[:60]],
            "num_drawn_cells": len(drawn),
            "num_optimal_cells": len(optimal_cells),
            "num_on_path": len(on_path),
            "proximity": round(float(proximity), 4),
            "coverage": round(float(coverage), 4),
            "length_factor": round(float(length_factor), 4),
            "continuity_factor": round(float(continuity_factor), 4),
            "key_visited": bool(key_visited),
            "key_multiplier": round(float(key_multiplier), 4),
            "wall_hit_cells": [list(c) for c in wall_hit_cells],
            "num_wall_hit_cells": num_wall_hit_cells,
            "wall_multiplier": round(float(wall_multiplier), 4),
            "path_core": round(float(path_core), 4),
            "final_score": round(float(score), 4),
            "score_breakdown": {
                "formula": (
                    "mean(proximity, continuity, coverage)"
                    " × (0.4 + 0.6 × wall_multiplier)"
                    " × (0.4 + 0.6 × key_multiplier)"
                ),
            },
        }
        details.update({
            "grid_size": grid_size,
            "target_color": target_color,
            "start_cell": list(start_cell),
            "target_key_cell": list(target_key_cell),
            "target_door_cell": list(target_door_cell),
            "num_walls": len(walls),
        })
        self._last_task_details = details
        return score


class PredictNextColorEvaluator(BaseEvaluator):
    """
    G-51: Predict next color evaluator.

    Dimensions:
        - completion (60%): use GT (first -> final) changed region as the generated
          shape region and compare generated final frame with GT final frame there.
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

    def _masked_color_similarity(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Color-based score in CIE L*a*b* space: exact color -> 1, large gap -> 0.
        """
        if mask is None or not np.any(mask > 0):
            return 0.0

        # Convert to Lab before sampling so distance reflects perceptual similarity.
        frame_a_lab = cv2.cvtColor(frame_a, cv2.COLOR_BGR2Lab).astype(np.float32)
        frame_b_lab = cv2.cvtColor(frame_b, cv2.COLOR_BGR2Lab).astype(np.float32)

        a = frame_a_lab[mask > 0]
        b = frame_b_lab[mask > 0]
        if a.size == 0 or b.size == 0:
            return 0.0

        # Rescale to real L*a*b* units for standard ΔE76 distance
        a_real = a.copy()
        b_real = b.copy()
        a_real[:, 0] *= 100.0 / 255.0
        b_real[:, 0] *= 100.0 / 255.0
        a_real[:, 1:] -= 128.0
        b_real[:, 1:] -= 128.0

        # mean_delta_e is now in standard ΔE76 units (perceptual difference).
        mean_delta_e = float(np.mean(np.linalg.norm(a_real - b_real, axis=1)))

        threshold_full = 5.0   # ΔE ≤ 10: same hue family -> full score
        threshold_zero = 50.0   # ΔE ≥ 50: clearly different color -> zero score
        if mean_delta_e <= threshold_full:
            return 1.0
        if mean_delta_e >= threshold_zero:
            return 0.0
        return float(1.0 - (mean_delta_e - threshold_full) / (threshold_zero - threshold_full))

    @staticmethod
    def _frame_masks(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (foreground_mask, background_mask) based on near-white background."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_not(bg_mask)
        return fg_mask, bg_mask

    @staticmethod
    def _shape_change_mask(gt_first: np.ndarray, gt_last: np.ndarray) -> np.ndarray:
        """Return GT first/final frame pixel-difference region."""
        diff = cv2.absdiff(gt_first, gt_last)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, change_mask = cv2.threshold(diff_gray, 18, 255, cv2.THRESH_BINARY)
        return change_mask

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate predict next color task."""
        scores: Dict[str, float] = {}

        if len(video_frames) < 2 or gt_first_frame is None or gt_final_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        if last_frame.shape[:2] != gt_final_frame.shape[:2]:
            first_frame = normalize_frame_size(first_frame, gt_final_frame)
            last_frame = normalize_frame_size(last_frame, gt_final_frame)
        gt_first, gt_last = gt_first_frame, gt_final_frame

        # 1) completion (60%): color similarity in GT-edited region with thresholds.
        change_mask = self._shape_change_mask(gt_first, gt_last)
        scores["completion"] = self._masked_color_similarity(gt_last, last_frame, change_mask)

        # 2) foreground_preservation (25%): first 4 shapes live in unchanged foreground.
        first_fg, first_bg = self._frame_masks(first_frame)
        fg_compare_mask = cv2.bitwise_and(first_fg, cv2.bitwise_not(change_mask))
        scores["foreground_preservation"] = self._masked_color_similarity(first_frame, last_frame, fg_compare_mask)

        # 3) background_preservation (15%): compare background region stability.
        bg_compare_mask = cv2.bitwise_and(first_bg, cv2.bitwise_not(change_mask))
        scores["background_preservation"] = self._pixel_similarity(first_frame, last_frame, bg_compare_mask, strictness=3.0, min_cutoff=0.6)

        self._last_task_details = scores
        return float(sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS))


class SelectFigureEvaluator(BaseEvaluator):
    """
    G-131, G-134, G-147, G-168, G-217, G-219, G-206, G-248: Circle selection evaluator.
    """

    TASK_WEIGHTS = {
        'consistency_score': 0.20,
        'match_score': 0.80
    }

    def _detect_shapes_with_size(self, frame: np.ndarray) -> List[Dict]:
        """Detect shapes and their sizes."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Find colored areas (non-white, non-black)
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            vertices = len(approx)

            if vertices == 3:
                shape_type = 'triangle'
            elif vertices == 4:
                shape_type = 'square'
            elif vertices == 5:
                shape_type = 'pentagon'
            else:
                shape_type = 'circle'

            shapes.append({
                'type': shape_type,
                'center': (cx, cy),
                'area': area,
                'vertices': vertices
            })

        return shapes

    def _detect_red_circle_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red circle marking and return its center."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity > 0.5:
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)

        return None

    def _detect_marking_by_diff(self, first_frame: np.ndarray, final_frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect marking by comparing first and final frames."""
        diff = cv2.absdiff(first_frame, final_frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)

        return None

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        # Use the meta source when a converter/annotation supplies labels for this
        # task; CircleSelectionProcessor falls back to image detection otherwise.
        use_metafile = True
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
            foreground_enlarge_pixels=20
        )
        circle_selection_info = circle_selection_processor.process(gt_first_frame, gt_final_frame, last_frame, debug_dir=debug_dir)
        
        scores = {}
        background_consistency_score = threshold_score(
            circle_selection_info['background_change_ratio'],
            [(0.02, 1.0), (0.2, 0.0)]
        )
        foreground_consistency_score = threshold_score(
            circle_selection_info['foreground_change_ratio'],
            [(0.03, 1.0), (0.3, 0.0)]
        )
        circle_area_penalty_score = threshold_score(
            circle_selection_info['circle_color_mask_ratio'],
            [(0.05, 1.0), (0.5, 0.0)]
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
        correct_match_score = 0.0
        wrong_match_score = 0.0
        for shape_id in range(len(per_shape_scores)):
            if circle_selection_info['is_target_shape'][shape_id] == 1:
                correct_match_score += per_shape_scores[shape_id] / num_target_shapes
            else:
                wrong_match_score = max(wrong_match_score, per_shape_scores[shape_id])
        scores['match_score'] = max(0, (correct_match_score - wrong_match_score) * (0.5 + 0.5 * foreground_consistency_score) * (0.4 + 0.6 * circle_area_penalty_score) - ambiguous_score)
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
            'consistency_score': scores['consistency_score'],
            'match_score': scores['match_score'],
            'background_consistency_score': background_consistency_score,
            'foreground_consistency_score': foreground_consistency_score,
            'circle_area_penalty_score': circle_area_penalty_score,
            'correct_match_score': correct_match_score,
            'wrong_match_score': wrong_match_score,
            'ambiguous_circles_count': ambiguous_circles_count,
            'num_target_shapes': num_target_shapes,
            'num_circles': num_circles,
            'background_change_ratio': circle_selection_info['background_change_ratio'],
            'foreground_change_ratio': circle_selection_info['foreground_change_ratio'],
            'circle_color_mask_ratio': circle_selection_info['circle_color_mask_ratio'],
            'total_score': total_score,
        }
        return total_score


class SpotUniqueColorEvaluator(BaseEvaluator):
    """
    G-138: Spot unique non-repeated color evaluator.

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

        gen_contours = detect_closed_contours_by_color(final_frame, COLOR_BOUNDS['black'])
        gt_contours  = detect_closed_contours_by_color(gt_final_frame, COLOR_BOUNDS['black'])

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

        def _enclosed_color(img, contour):
            m = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(m, [contour], -1, 255, -1)
            cv2.drawContours(m, [contour], -1, 0, 12)
            px = img[m > 0]
            if px.size == 0:
                return None
            gray = px.mean(axis=1)
            px = px[(gray > 60) & (gray < 235)] 
            return np.median(px, axis=0) if px.size else None

        gt_inner = [_enclosed_color(gt_final_frame, c) for c in gt_contours]
        gen_inner = [_enclosed_color(final_frame, c) for c in gen_contours]

        per_gt_scores = []
        n_contained = 0
        for gi, gt_cnt in enumerate(gt_contours):
            iou = match_results[gi] if gi < len(match_results) else None
            base = float(iou) if iou is not None else 0.0
            contained = any(
                cv2.pointPolygonTest(gt_cnt, (float(gx), float(gy)), False) >= 0
                for (gx, gy) in gen_centroids
            )
            by_position = contained
            if not contained and gt_inner[gi] is not None:
                contained = any(
                    gc is not None and float(np.linalg.norm(gt_inner[gi] - gc)) <= 60.0
                    for gc in gen_inner
                )
            if contained:
                n_contained += 1
                gt_mask_c = np.zeros(canvas_size, dtype=np.uint8)
                cv2.drawContours(gt_mask_c, [gt_cnt], -1, 255, -1)
                gen_mask_c = np.zeros(canvas_size, dtype=np.uint8)
                cv2.drawContours(gen_mask_c, gen_contours, -1, 255, -1)
                _ga_c = int((gt_mask_c > 0).sum())
                _cov_c = (int(((gt_mask_c > 0) & (gen_mask_c > 0)).sum()) / _ga_c) if _ga_c else 0.0
                per_gt_scores.append(max(base, 0.9 * _cov_c if by_position else 0.9))
            else:
                per_gt_scores.append(base)
        accuracy = float(np.mean(per_gt_scores)) if per_gt_scores else 0.0
        n_effective_matches = max(len(valid_ious), n_contained)
        accuracy = accuracy * calculate_list_length_penalty(len(gt_contours), n_effective_matches, len(gen_contours))

        back_consistency = score_background_similarity(gt_final_frame, final_frame, type='mse')

        fore_consistency = score_foreground_similarity(
            gt_final_frame, final_frame, COLOR_BOUNDS['red'], type='mse'
        )

        consistency = 0.5 * back_consistency + 0.5 * fore_consistency
        score = accuracy * (0.6 + 0.4 * consistency)
        self._last_task_details = {
            'accuracy': accuracy,
            'back_consistency': back_consistency,
            'fore_consistency': fore_consistency,
        }
        return score

# Mapping of task names to evaluators
IN_DOMAIN_50_EVALUATORS_PART2 = {
    'G-29_chart_extreme_with_data_data-generator': ChartExtremeEvaluator,
    'G-31_directed_graph_navigation_data-generator': DirectedGraphNavigationEvaluator,
    'G-39_attention_shift_different_data-generator': AttentionShiftEvaluator,
    'G-41_grid_highest_cost_data-generator': GridHighestCostEvaluator,
    'G-43_understand_scene_structure_data-generator': UnderstandSceneStructureEvaluator,
    'G-45_key_door_matching_data-generator': KeyDoorMatchingEvaluator,
    'G-51_predict_next_color_data-generator': PredictNextColorEvaluator,
    'G-131_select_next_figure_increasing_size_sequence_data-generator': SelectFigureEvaluator,
    'G-134_select_next_figure_large_small_alternating_sequence_data-generator': SelectFigureEvaluator,
    'G-147_identify_unique_figure_in_uniform_set_data-generator': SelectFigureEvaluator,
    'G-217_circle_central_dot_data-generator': SelectFigureEvaluator,
    'G-219_select_leftmost_shape_data-generator': SelectFigureEvaluator,
    'G-206_identify_pentagons_data-generator': SelectFigureEvaluator,
    'G-248_mark_asymmetrical_shape_data-generator': SelectFigureEvaluator,
    'G-168_identify_nearest_to_square_rectangle_data-generator': SelectFigureEvaluator,
    'G-138_spot_unique_non_repeated_color_data-generator': SpotUniqueColorEvaluator,
}
