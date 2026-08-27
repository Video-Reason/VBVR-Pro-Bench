"""
Specific evaluators for In-Domain_50 tasks (Part 5).
"""

import re
from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import safe_distance, normalize_frame_size, compute_ssim


class _LegacyGridShiftEvaluator(BaseEvaluator):
    """
    O-36: Grid Shift
    
    Task: Move all colored blocks in NxN grid simultaneously in specified 
    direction (up/down/left/right) by specified steps.
    
    Key evaluation criteria:
    1. Direction correctness (30%) - All blocks move correct direction
    2. Step accuracy (30%) - Exact number of steps moved
    3. Synchronization (20%) - All blocks move together
    4. Position precision (15%) - Final positions correct
    5. Completeness (5%) - All blocks moved, properties preserved
    """
    
    def __init__(self, device: str = 'cuda', task_name: str = ''):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {
            'direction_correctness': 0.30,
            'step_accuracy': 0.30,
            'synchronization': 0.20,
            'position_precision': 0.15,
            'completeness': 0.05
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate grid shift movement."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Align the prediction to GT: GT is the reference answer, so its resolution
        # is the baseline.
        if gen_final.shape != gt_final.shape:
            gen_final = normalize_frame_size(gen_final, gt_final)
            first_frame = normalize_frame_size(first_frame, gt_final)
        
        # Detect colored blocks in first and final frames
        first_blocks = self._detect_colored_blocks(first_frame)
        gen_final_blocks = self._detect_colored_blocks(gen_final)
        gt_final_blocks = self._detect_colored_blocks(gt_final)
        
        scores = {}
        
        # CRITICAL: First check if blocks are preserved (completeness)
        # If blocks change, the whole task fails
        completeness_score = self._evaluate_completeness(first_blocks, gen_final_blocks)
        
        # Also check pattern preservation
        pattern_score = self._evaluate_block_pattern_preservation(
            first_frame, gen_final, first_blocks, gen_final_blocks
        )
        
        # Combine: blocks must be preserved AND patterns must be unchanged
        block_preserved = min(completeness_score, pattern_score) > 0.5
        scores['completeness'] = min(completeness_score, pattern_score)
        
        # If blocks are NOT preserved, all other scores should be 0
        if not block_preserved:
            scores['direction_correctness'] = 0.0
            scores['step_accuracy'] = 0.0
            scores['synchronization'] = 0.0
            scores['position_precision'] = 0.0
        else:
            # 1. Direction correctness (30%): Check if blocks moved in correct direction
            direction_score = self._evaluate_direction(first_blocks, gen_final_blocks, gt_final_blocks)
            scores['direction_correctness'] = direction_score
            
            # 2. Step accuracy (30%): Check if blocks moved correct number of steps
            step_score = self._evaluate_step_accuracy(first_blocks, gen_final_blocks, gt_final_blocks, gen_final)
            scores['step_accuracy'] = step_score
            
            # 3. Synchronization (20%): Check if all blocks moved together
            sync_score = self._evaluate_synchronization(video_frames)
            scores['synchronization'] = sync_score
            
            # 4. Position precision (15%): Check final block positions
            position_score = self._evaluate_position_precision(gen_final_blocks, gt_final_blocks)
            scores['position_precision'] = position_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate grid shift for interleaved image generation.

        Interleave outputs a single frame showing the final grid state.
        Video version:
          - direction_correctness (30%): block displacement direction — first/last frame
          - step_accuracy (30%): block displacement distance — first/last frame
          - synchronization (20%): all blocks move together — needs multiple frames
          - position_precision (15%): final block positions vs GT — last frame
          - completeness (5%): block count/color preserved — first/last frame
        Interleave version:
          - direction_correctness (30%): reuse
          - step_accuracy (30%): reuse
          - diff_ssim (20%): replace synchronization — diff SSIM on change regions
          - position_precision (15%): reuse
          - completeness (5%): reuse
        """
        INTERLEAVE_WEIGHTS = {
            'direction_correctness': 0.30,
            'step_accuracy': 0.30,
            'diff_ssim': 0.20,
            'position_precision': 0.15,
            'completeness': 0.05,
        }

        if not pred_images or gt_final_frame is None or input_frame is None:
            return 0.0

        last_frame = pred_images[-1]

        # Normalize sizes
        # Align the prediction to GT, not GT to the prediction.
        if last_frame.shape != gt_final_frame.shape:
            last_frame = cv2.resize(last_frame, (gt_final_frame.shape[1], gt_final_frame.shape[0]))
        gt_final = gt_final_frame

        if input_frame.shape != last_frame.shape:
            input_resized = cv2.resize(input_frame, (last_frame.shape[1], last_frame.shape[0]))
        else:
            input_resized = input_frame

        first_blocks = self._detect_colored_blocks(input_resized)
        gen_final_blocks = self._detect_colored_blocks(last_frame)
        gt_final_blocks = self._detect_colored_blocks(gt_final)

        scores = {}

        # completeness
        completeness_score = self._evaluate_completeness(first_blocks, gen_final_blocks)
        pattern_score = self._evaluate_block_pattern_preservation(
            input_resized, last_frame, first_blocks, gen_final_blocks
        )
        block_preserved = min(completeness_score, pattern_score) > 0.5
        scores['completeness'] = min(completeness_score, pattern_score)

        if not block_preserved:
            scores['direction_correctness'] = 0.0
            scores['step_accuracy'] = 0.0
            scores['diff_ssim'] = 0.0
            scores['position_precision'] = 0.0
        else:
            scores['direction_correctness'] = self._evaluate_direction(
                first_blocks, gen_final_blocks, gt_final_blocks)
            scores['step_accuracy'] = self._evaluate_step_accuracy(
                first_blocks, gen_final_blocks, gt_final_blocks, last_frame)
            scores['position_precision'] = self._evaluate_position_precision(
                gen_final_blocks, gt_final_blocks)

            # diff_ssim: replace synchronization
            pred_diff = cv2.absdiff(last_frame, input_resized)
            gt_diff = cv2.absdiff(gt_final, input_resized)
            scores['diff_ssim'] = compute_ssim(pred_diff, gt_diff)

        self._last_task_details = scores
        return sum(scores[k] * INTERLEAVE_WEIGHTS[k] for k in INTERLEAVE_WEIGHTS)

    def _detect_colored_blocks(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored blocks in the frame."""
        blocks = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Define color ranges for common block colors (lower saturation threshold)
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255], [160, 50, 50], [180, 255, 255]),
            'green': ([35, 50, 50], [85, 255, 255], None, None),
            'blue': ([100, 50, 50], [130, 255, 255], None, None),
            'yellow': ([20, 50, 50], [35, 255, 255], None, None),
            'orange': ([10, 50, 50], [20, 255, 255], None, None),
            'purple': ([130, 50, 50], [160, 255, 255], None, None),
            'cyan': ([85, 50, 50], [100, 255, 255], None, None),
        }
        
        detected_centers = set()  # Avoid duplicates
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200:  # Filter noise
                    continue
                
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Avoid duplicates
                    center_key = (cx // 20, cy // 20)
                    if center_key in detected_centers:
                        continue
                    detected_centers.add(center_key)
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    blocks.append({
                        'color': color_name,
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        
        # Also detect gray/neutral blocks (low saturation, medium value)
        if not blocks:
            # Look for non-white, non-black regions
            non_white = ((gray > 50) & (gray < 220)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 50000:  # Filter noise and background
                    continue
                
                # Check if roughly square (block-like)
                x, y, w, h = cv2.boundingRect(contour)
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 10
                if aspect > 2:  # Not square enough
                    continue
                
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    blocks.append({
                        'color': 'gray',
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        
        return blocks
    
    def _evaluate_direction(self, first_blocks: List[Dict], gen_blocks: List[Dict],
                            gt_blocks: List[Dict]) -> float:
        """Evaluate if blocks moved in correct direction."""
        if not first_blocks or not gen_blocks or not gt_blocks:
            return 0.0
        
        # Calculate expected movement direction from GT
        gt_movements = []
        for fb in first_blocks:
            # Find matching GT block by color
            for gtb in gt_blocks:
                if fb['color'] == gtb['color']:
                    dx = float(gtb['center'][0]) - float(fb['center'][0])
                    dy = float(gtb['center'][1]) - float(fb['center'][1])
                    gt_movements.append((dx, dy))
                    break
        
        if not gt_movements:
            return 0.5
        
        # Determine expected direction
        avg_dx = np.mean([m[0] for m in gt_movements])
        avg_dy = np.mean([m[1] for m in gt_movements])
        
        # Calculate actual movement
        gen_movements = []
        for fb in first_blocks:
            for gb in gen_blocks:
                if fb['color'] == gb['color']:
                    dx = float(gb['center'][0]) - float(fb['center'][0])
                    dy = float(gb['center'][1]) - float(fb['center'][1])
                    gen_movements.append((dx, dy))
                    break
        
        if not gen_movements:
            return 0.0
        
        actual_dx = np.mean([m[0] for m in gen_movements])
        actual_dy = np.mean([m[1] for m in gen_movements])
        
        # Check direction match
        direction_match = 0.0
        
        # Check horizontal direction
        if avg_dx != 0:
            if np.sign(actual_dx) == np.sign(avg_dx):
                direction_match += 0.5
        else:
            if abs(actual_dx) < 10:  # No horizontal movement expected
                direction_match += 0.5
        
        # Check vertical direction
        if avg_dy != 0:
            if np.sign(actual_dy) == np.sign(avg_dy):
                direction_match += 0.5
        else:
            if abs(actual_dy) < 10:  # No vertical movement expected
                direction_match += 0.5
        
        return direction_match
    
    def _evaluate_step_accuracy(self, first_blocks: List[Dict], gen_blocks: List[Dict],
                                gt_blocks: List[Dict], frame: np.ndarray) -> float:
        """Evaluate if blocks moved correct number of steps."""
        if not first_blocks or not gen_blocks or not gt_blocks:
            return 0.0
        
        # Estimate grid cell size
        h, w = frame.shape[:2]
        # Assume 4-12 grid, estimate cell size
        estimated_cell_size = w / 8  # Average estimate
        
        # Calculate expected displacement from GT
        gt_displacements = []
        for fb in first_blocks:
            for gtb in gt_blocks:
                if fb['color'] == gtb['color']:
                    dx = abs(float(gtb['center'][0]) - float(fb['center'][0]))
                    dy = abs(float(gtb['center'][1]) - float(fb['center'][1]))
                    gt_displacements.append(max(dx, dy))
                    break
        
        # Calculate actual displacement
        gen_displacements = []
        for fb in first_blocks:
            for gb in gen_blocks:
                if fb['color'] == gb['color']:
                    dx = abs(float(gb['center'][0]) - float(fb['center'][0]))
                    dy = abs(float(gb['center'][1]) - float(fb['center'][1]))
                    gen_displacements.append(max(dx, dy))
                    break
        
        if not gt_displacements or not gen_displacements:
            return 0.0
        
        avg_gt_disp = np.mean(gt_displacements)
        avg_gen_disp = np.mean(gen_displacements)
        
        if avg_gt_disp < 1:
            return 1.0 if avg_gen_disp < estimated_cell_size * 0.5 else 0.5
        
        # Calculate step difference
        ratio = avg_gen_disp / avg_gt_disp
        
        if 0.8 <= ratio <= 1.2:
            return 1.0
        elif 0.5 <= ratio <= 1.5:
            return 0.7
        elif 0.3 <= ratio <= 2.0:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_synchronization(self, frames: List[np.ndarray]) -> float:
        """Check if all blocks move synchronously."""
        if len(frames) < 3:
            return 0.5
        
        # Track block positions through video
        n_samples = min(10, len(frames))
        sample_indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
        
        all_positions = []
        for idx in sample_indices:
            blocks = self._detect_colored_blocks(frames[idx])
            if blocks:
                positions = [b['center'] for b in blocks]
                all_positions.append(positions)
        
        if len(all_positions) < 3:
            return 0.5
        
        # Check if all blocks move together (similar displacement at each frame)
        sync_scores = []
        for i in range(1, len(all_positions)):
            if len(all_positions[i]) != len(all_positions[i-1]):
                continue
            
            displacements = []
            for j in range(len(all_positions[i])):
                dx = all_positions[i][j][0] - all_positions[i-1][j][0]
                dy = all_positions[i][j][1] - all_positions[i-1][j][1]
                displacements.append((dx, dy))
            
            if len(displacements) > 1:
                # Check variance in displacements
                dx_var = np.var([d[0] for d in displacements])
                dy_var = np.var([d[1] for d in displacements])
                
                # Low variance means synchronized movement
                max_var = max(dx_var, dy_var)
                if max_var < 100:
                    sync_scores.append(1.0)
                elif max_var < 500:
                    sync_scores.append(0.7)
                else:
                    sync_scores.append(0.3)
        
        return np.mean(sync_scores) if sync_scores else 0.5
    
    def _evaluate_position_precision(self, gen_blocks: List[Dict], gt_blocks: List[Dict]) -> float:
        """Evaluate final position accuracy."""
        if not gen_blocks or not gt_blocks:
            return 0.0
        
        matched_scores = []
        
        for gtb in gt_blocks:
            best_dist = float('inf')
            for gb in gen_blocks:
                if gb['color'] == gtb['color']:
                    dist = safe_distance(gb['center'], gtb['center'])
                    best_dist = min(best_dist, dist)
            
            if best_dist < float('inf'):
                # Score based on distance
                if best_dist < 10:
                    matched_scores.append(1.0)
                elif best_dist < 30:
                    matched_scores.append(0.8)
                elif best_dist < 50:
                    matched_scores.append(0.5)
                else:
                    matched_scores.append(max(0.1, 1.0 - best_dist / 100))
        
        return np.mean(matched_scores) if matched_scores else 0.0
    
    def _evaluate_completeness(self, first_blocks: List[Dict], gen_blocks: List[Dict]) -> float:
        """Evaluate if all blocks are preserved with same colors."""
        if not first_blocks:
            return 0.0
        
        if not gen_blocks:
            return 0.0
        
        # Check if same number of blocks
        if len(gen_blocks) != len(first_blocks):
            return 0.0  # Block count changed - STRICT failure
        
        # Check if all block colors are preserved
        first_colors = sorted([b['color'] for b in first_blocks])
        gen_colors = sorted([b['color'] for b in gen_blocks])
        
        if first_colors != gen_colors:
            return 0.0  # Block colors changed - STRICT failure
        
        return 1.0  # All blocks preserved with same colors
    
    def _evaluate_block_pattern_preservation(
        self, 
        first_frame: np.ndarray, 
        gen_final: np.ndarray,
        first_blocks: List[Dict],
        gen_blocks: List[Dict]
    ) -> float:
        """Check if block patterns/content remain unchanged during shift."""
        if not first_blocks or not gen_blocks:
            return 0.0
        
        # For each block in first frame, extract its appearance
        # and compare with corresponding block in final frame
        preservation_scores = []
        
        for fb in first_blocks:
            # Find matching block by color in gen_blocks
            matching_gb = None
            for gb in gen_blocks:
                if gb['color'] == fb['color']:
                    matching_gb = gb
                    break
            
            if matching_gb is None:
                preservation_scores.append(0.0)
                continue
            
            # Extract block regions
            fx, fy, fw, fh = fb['bbox']
            gx, gy, gw, gh = matching_gb['bbox']
            
            # Get block regions
            first_region = first_frame[fy:fy+fh, fx:fx+fw]
            gen_region = gen_final[gy:gy+gh, gx:gx+gw]
            
            # Resize to same size for comparison
            if first_region.size > 0 and gen_region.size > 0:
                target_size = (max(fw, gw), max(fh, gh))
                first_resized = cv2.resize(first_region, target_size)
                gen_resized = cv2.resize(gen_region, target_size)
                
                # Compare patterns
                diff = np.abs(first_resized.astype(float) - gen_resized.astype(float)).mean()
                
                if diff < 30:  # Very similar
                    preservation_scores.append(1.0)
                elif diff < 60:
                    preservation_scores.append(0.5)
                else:
                    preservation_scores.append(0.0)  # Pattern changed
            else:
                preservation_scores.append(0.0)
        
        return np.mean(preservation_scores) if preservation_scores else 0.0


class GridShiftEvaluator(BaseEvaluator):
    """
    O-36: Grid Shift

    GT-driven grid-occupancy scorer.

    O-36 is not one fixed prompt: grid size, block colour, direction, and
    number of steps all vary per sample. The robust invariant is the GT
    transformation itself: every occupied grid cell is translated by the same
    integer vector.

        process_score = temporal_similarity × occupancy_count_consistency
        score = 0.35 × final_cell_score + 0.65 × process_score × final_cell_gate

    - ``final_cell_score`` gives each GT target cell 1/n credit and uses
      distance-aware matching plus an extra-block penalty.
    - ``temporal_similarity`` compares the whole predicted motion curve to the
      GT motion curve via centroid progress along the GT shift direction.
    - ``occupancy_count_consistency`` compares predicted and GT occupied-cell
      count curves, so dense hallucinated grids lose process credit.
    - ``final_cell_gate`` lets good process help only when the final cells are
      also plausible.
    """

    GRID_SIZE_RANGE = range(4, 13)
    INNER_MARGIN = 0.20
    OCCUPANCY_THRESHOLD = 0.15
    GRAY_THRESHOLD = 220
    SAT_THRESHOLD = 60
    FINAL_CELL_WEIGHT = 0.35
    PROCESS_WEIGHT = 0.65
    FINAL_CELL_GATE_FLOOR = 0.15

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        if not video_frames:
            return 0.0

        gt_seq: List[np.ndarray] = [f for f in gt_frames if f is not None]
        if not gt_seq:
            if gt_first_frame is not None:
                gt_seq.append(gt_first_frame)
            if gt_final_frame is not None and (
                not gt_seq or gt_seq[-1] is not gt_final_frame
            ):
                gt_seq.append(gt_final_frame)
        if not gt_seq:
            self._last_task_details = {'error': 'no_gt_sequence'}
            return 0.0

        return self._score_sequences(video_frames, gt_seq, eval_info)

    def _evaluate_task_specific_interleave(
        self,
        pred_images: List[np.ndarray],
        gt_images: List[np.ndarray],
        input_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        if not pred_images or input_frame is None:
            return 0.0

        pred_seq = self._normalize_sequence([input_frame] + pred_images, input_frame)
        gt_seq: List[np.ndarray] = [f for f in gt_images if f is not None]
        if not gt_seq:
            gt_seq = [input_frame]
            if gt_final_frame is not None:
                gt_seq.append(gt_final_frame)
        if len(gt_seq) < 2:
            self._last_task_details = {'error': 'no_gt_sequence'}
            return 0.0

        gt_seq = self._normalize_sequence(gt_seq, pred_seq[0])
        return self._score_sequences(pred_seq, gt_seq, eval_info)

    def _normalize_sequence(
        self,
        frames: Sequence[np.ndarray],
        reference_frame: np.ndarray,
    ) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for frame in frames:
            if frame.shape != reference_frame.shape:
                out.append(normalize_frame_size(frame, reference_frame))
            else:
                out.append(frame)
        return out

    def _parse_prompt_grid_size(self, prompt: str) -> Optional[int]:
        if not prompt:
            return None
        match = re.search(r'(\d+)x\d+\s+grid', prompt)
        if match is None:
            return None
        return int(match.group(1))

    def _grid_spec_from_metadata(
        self,
        eval_info: Dict[str, Any],
    ) -> Optional[Tuple[int, Tuple[int, int], Dict[str, Any]]]:
        """Read O-36's authoritative grid and shift from sample metadata."""
        import os

        meta_path = eval_info.get('metafile_path')
        if isinstance(meta_path, (list, tuple)):
            meta_path = next(
                (path for path in meta_path if path and os.path.exists(path)),
                None,
            )
        if not (meta_path and os.path.exists(meta_path)):
            meta_path = os.path.join(eval_info.get('gt_path', ''), 'metadata.json')
        if not os.path.exists(meta_path):
            return None

        try:
            with open(meta_path, encoding='utf-8') as handle:
                params = (json.load(handle).get('parameters') or {})
            grid_size = int(params['grid_size'])
            steps = int(params['steps'])
            direction = str(params['direction']).strip().lower()
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

        direction_vectors = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
        }
        if grid_size <= 0 or steps <= 0 or direction not in direction_vectors:
            return None

        unit_dr, unit_dc = direction_vectors[direction]
        shift = (unit_dr * steps, unit_dc * steps)
        return grid_size, shift, {
            'source': 'metadata',
            'grid_size': grid_size,
            'direction': direction,
            'steps': steps,
            'shift': shift,
            'metafile_path': meta_path,
        }

    def _detect_occupied_cells(
        self,
        frame: np.ndarray,
        grid_size: int,
    ) -> Tuple[set, List[Tuple[int, int, int, int]]]:
        if grid_size <= 0:
            return set(), []

        h, w = frame.shape[:2]
        cell_h = h / grid_size
        cell_w = w / grid_size
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sat_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        occupied = set()
        boxes: List[Tuple[int, int, int, int]] = []

        for r in range(grid_size):
            y0 = int(round(r * cell_h + self.INNER_MARGIN * cell_h))
            y1 = int(round((r + 1) * cell_h - self.INNER_MARGIN * cell_h))
            for c in range(grid_size):
                x0 = int(round(c * cell_w + self.INNER_MARGIN * cell_w))
                x1 = int(round((c + 1) * cell_w - self.INNER_MARGIN * cell_w))
                if x1 <= x0 or y1 <= y0:
                    continue
                gray = gray_full[y0:y1, x0:x1]
                sat = sat_full[y0:y1, x0:x1]
                if gray.size == 0 or sat.size == 0:
                    continue
                mask = (gray < self.GRAY_THRESHOLD) | (sat > self.SAT_THRESHOLD)
                if float(mask.mean()) > self.OCCUPANCY_THRESHOLD:
                    occupied.add((r, c))
                    boxes.append((x0, y0, x1, y1))

        return occupied, boxes

    def _infer_shift(
        self,
        start_cells: set,
        end_cells: set,
        grid_size: int,
    ) -> Optional[Tuple[int, int]]:
        if not start_cells or len(start_cells) != len(end_cells):
            return None

        best_shift: Optional[Tuple[int, int]] = None
        best_overlap = -1
        for dr in range(-grid_size + 1, grid_size):
            for dc in range(-grid_size + 1, grid_size):
                shifted = {(r + dr, c + dc) for r, c in start_cells}
                overlap = len(shifted & end_cells)
                if shifted == end_cells:
                    return (dr, dc)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_shift = (dr, dc)
        return best_shift

    def _infer_grid_size(
        self,
        gt_seq: Sequence[np.ndarray],
        prompt: str,
    ) -> Tuple[Optional[int], Optional[Tuple[int, int]], Dict[str, Any]]:
        prompt_grid = self._parse_prompt_grid_size(prompt)
        candidates = [prompt_grid] if prompt_grid is not None else list(self.GRID_SIZE_RANGE)
        candidates = [c for c in candidates if c is not None]

        first_gt = gt_seq[0]
        last_gt = gt_seq[-1]
        best = None
        best_info: Dict[str, Any] = {}

        for grid_size in candidates:
            start_cells, _ = self._detect_occupied_cells(first_gt, grid_size)
            end_cells, _ = self._detect_occupied_cells(last_gt, grid_size)
            shift = self._infer_shift(start_cells, end_cells, grid_size)
            overlap = -1
            if shift is not None:
                overlap = len({(r + shift[0], c + shift[1]) for r, c in start_cells} & end_cells)
            info = {
                'grid_size': grid_size,
                'start_count': len(start_cells),
                'end_count': len(end_cells),
                'shift': shift,
                'overlap': overlap,
            }
            if shift is not None and len(start_cells) == len(end_cells) and len(start_cells) > 0:
                shifted = {(r + shift[0], c + shift[1]) for r, c in start_cells}
                if shifted == end_cells:
                    return grid_size, shift, info
            if best is None or overlap > best[0]:
                best = (overlap, grid_size, shift)
                best_info = info

        if best is None:
            return None, None, {}
        return best[1], best[2], best_info

    def _best_assignment(self, score_matrix: np.ndarray) -> List[Tuple[int, int]]:
        if score_matrix.size == 0:
            return []

        try:
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(1.0 - score_matrix)
            return list(zip(rows.tolist(), cols.tolist()))
        except ImportError:
            n_pred, n_gt = score_matrix.shape
            if n_pred <= n_gt:
                best_score = -1.0
                best_pairs: List[Tuple[int, int]] = []
                for cols in permutations(range(n_gt), n_pred):
                    score = float(sum(score_matrix[i, cols[i]] for i in range(n_pred)))
                    if score > best_score:
                        best_score = score
                        best_pairs = [(i, cols[i]) for i in range(n_pred)]
                return best_pairs

            best_score = -1.0
            best_pairs = []
            for rows in permutations(range(n_pred), n_gt):
                score = float(sum(score_matrix[rows[j], j] for j in range(n_gt)))
                if score > best_score:
                    best_score = score
                    best_pairs = [(rows[j], j) for j in range(n_gt)]
            return best_pairs

    def _score_final_cells(
        self,
        pred_cells: Sequence[Tuple[int, int]],
        gt_cells: Sequence[Tuple[int, int]],
        *,
        shift_len: int,
    ) -> Tuple[float, Dict[str, Any]]:
        gt_count = len(gt_cells)
        pred_count = len(pred_cells)
        if gt_count == 0:
            return 0.0, {'error': 'no_gt_cells'}
        if pred_count == 0:
            return 0.0, {'error': 'no_pred_cells', 'gt_count': gt_count}

        score_matrix = np.zeros((pred_count, gt_count), dtype=float)
        pair_details: Dict[Tuple[int, int], Dict[str, float]] = {}
        norm = float(max(shift_len, 1))

        for pred_idx, pred_cell in enumerate(pred_cells):
            for gt_idx, gt_cell in enumerate(gt_cells):
                dist = abs(pred_cell[0] - gt_cell[0]) + abs(pred_cell[1] - gt_cell[1])
                score = max(0.0, 1.0 - dist / norm)
                score_matrix[pred_idx, gt_idx] = score
                pair_details[(pred_idx, gt_idx)] = {
                    'pred_row': float(pred_cell[0]),
                    'pred_col': float(pred_cell[1]),
                    'gt_row': float(gt_cell[0]),
                    'gt_col': float(gt_cell[1]),
                    'manhattan_dist': float(dist),
                    'score': float(score),
                }

        matches = self._best_assignment(score_matrix)
        per_gt_scores = [0.0] * gt_count
        match_details: List[Dict[str, float]] = []
        for pred_idx, gt_idx in matches:
            detail = pair_details[(pred_idx, gt_idx)]
            per_gt_scores[gt_idx] = float(detail['score'])
            match_details.append(detail)

        mean_block_score = float(sum(per_gt_scores) / gt_count)
        extra_block_penalty = gt_count / max(pred_count, gt_count)
        total = mean_block_score * extra_block_penalty

        return total, {
            'formula': 'mean(per_gt_block_score) × extra_block_penalty',
            'gt_cell_count': gt_count,
            'pred_cell_count': pred_count,
            'per_gt_scores': [round(float(s), 4) for s in per_gt_scores],
            'mean_block_score': round(float(mean_block_score), 4),
            'extra_block_penalty': round(float(extra_block_penalty), 4),
            'matches': match_details,
        }

    def _centroid_progress_curve(
        self,
        frames: Sequence[np.ndarray],
        grid_size: int,
        shift: Tuple[int, int],
    ) -> np.ndarray:
        axis = np.array([float(shift[0]), float(shift[1])], dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return np.zeros(len(frames), dtype=float)
        axis /= norm

        values: List[float] = []
        for frame in frames:
            cells, _ = self._detect_occupied_cells(frame, grid_size)
            if not cells:
                if values:
                    values.append(values[-1])
                else:
                    values.append(0.0)
                continue
            coords = np.array([[float(r), float(c)] for r, c in sorted(cells)], dtype=float)
            values.append(float(coords.mean(axis=0).dot(axis)))
        return np.array(values, dtype=float)

    def _resample_curve(self, curve: np.ndarray, sample_count: int) -> np.ndarray:
        if len(curve) == 0:
            return np.zeros(sample_count, dtype=float)
        if len(curve) == 1:
            return np.repeat(curve[:1], sample_count)
        src_x = np.linspace(0.0, 1.0, len(curve))
        dst_x = np.linspace(0.0, 1.0, sample_count)
        return np.interp(dst_x, src_x, curve)

    @staticmethod
    def _dtw_path(a: Sequence[float], b: Sequence[float]) -> List[Tuple[int, int]]:
        """Dynamic-time-warping alignment between two 1-D curves.

        Returns the list of (i, j) index pairs on the optimal warping path. DTW lets
        the time axis stretch, so two curves with the same shape but different pacing
        align at near-zero cost; a different shape (teleport, reversal, partial run)
        still costs. Both endpoints are pinned (start↔start, end↔end), so a shift that
        ends in the wrong place is not rewarded.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return []
        acc = np.full((n + 1, m + 1), np.inf)
        acc[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(a[i - 1] - b[j - 1])
                acc[i, j] = cost + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
        i, j = n, m
        path: List[Tuple[int, int]] = []
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
            if step == 0:
                i -= 1; j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
        path.reverse()
        return path

    def _temporal_similarity(
        self,
        pred_seq: Sequence[np.ndarray],
        gt_seq: Sequence[np.ndarray],
        grid_size: int,
        shift: Tuple[int, int],
    ) -> Tuple[float, Dict[str, Any]]:
        step_len = abs(shift[0]) + abs(shift[1])
        if step_len == 0:
            return 1.0, {'curve_error': 0.0}

        pred_curve = self._centroid_progress_curve(pred_seq, grid_size, shift)
        gt_curve = self._centroid_progress_curve(gt_seq, grid_size, shift)

        path = self._dtw_path(pred_curve, gt_curve)
        if not path:
            return 1.0, {'curve_error': 0.0}
        errors = np.array([abs(float(pred_curve[i]) - float(gt_curve[j])) for i, j in path])
        similarity = float(np.clip(1.0 - errors / float(step_len), 0.0, 1.0).mean())
        return similarity, {
            'curve_error_mean': float(errors.mean()) if len(errors) else 0.0,
            'curve_error_max': float(errors.max()) if len(errors) else 0.0,
            'pred_curve_start_end': [float(pred_curve[0]), float(pred_curve[-1])],
            'gt_curve_start_end': [float(gt_curve[0]), float(gt_curve[-1])],
        }

    def _occupancy_count_consistency(
        self,
        pred_seq: Sequence[np.ndarray],
        gt_seq: Sequence[np.ndarray],
        grid_size: int,
    ) -> Tuple[float, Dict[str, Any]]:
        if not pred_seq or not gt_seq:
            return 0.0, {
                'count_score_mean': 0.0,
                'pred_count_min': 0,
                'pred_count_max': 0,
                'gt_count_min': 0,
                'gt_count_max': 0,
            }

        pred_counts: List[int] = []
        for frame in pred_seq:
            cells, _ = self._detect_occupied_cells(frame, grid_size)
            pred_counts.append(len(cells))

        gt_counts: List[int] = []
        for frame in gt_seq:
            cells, _ = self._detect_occupied_cells(frame, grid_size)
            gt_counts.append(len(cells))

        pred_arr = np.array(pred_counts, dtype=float)
        gt_arr = np.array(gt_counts, dtype=float)
        path = self._dtw_path(pred_arr, gt_arr)
        if not path:
            path = list(zip(range(len(pred_arr)), range(len(gt_arr))))
        scores: List[float] = []
        for i, j in path:
            pred_count, gt_count = pred_arr[i], gt_arr[j]
            if pred_count <= 0.0 or gt_count <= 0.0:
                scores.append(0.0)
            else:
                scores.append(float(min(pred_count, gt_count) / max(pred_count, gt_count)))

        return float(np.mean(scores)) if scores else 0.0, {
            'count_score_mean': float(np.mean(scores)) if scores else 0.0,
            'pred_count_min': int(min(pred_counts)) if pred_counts else 0,
            'pred_count_max': int(max(pred_counts)) if pred_counts else 0,
            'gt_count_min': int(min(gt_counts)) if gt_counts else 0,
            'gt_count_max': int(max(gt_counts)) if gt_counts else 0,
            'pred_count_start_end': [
                int(pred_counts[0]) if pred_counts else 0,
                int(pred_counts[-1]) if pred_counts else 0,
            ],
            'gt_count_start_end': [
                int(gt_counts[0]) if gt_counts else 0,
                int(gt_counts[-1]) if gt_counts else 0,
            ],
        }

    def _intermediate_state_match_ratio(
        self,
        pred_seq: Sequence[np.ndarray],
        gt_seq: Sequence[np.ndarray],
        grid_size: int,
        shift: Tuple[int, int],
    ) -> Tuple[float, Dict[str, Any]]:
        step_count = abs(shift[0]) + abs(shift[1])
        required_count = max(0, step_count - 1)
        if required_count == 0:
            return 1.0, {
                'required_count': 0,
                'matched_count': 0,
                'match_ratio': 1.0,
                'targets': [],
                'matched': [],
            }

        pred_curve = self._centroid_progress_curve(
            pred_seq, grid_size, shift,
        )
        gt_curve = self._centroid_progress_curve(
            gt_seq, grid_size, shift,
        )
        if len(gt_curve) == 0:
            return 0.0, {
                'required_count': required_count,
                'matched_count': 0,
                'match_ratio': 0.0,
                'error': 'no_gt_progress_curve',
            }

        start_progress = float(gt_curve[0])
        targets = [
            start_progress + float(step)
            for step in range(1, step_count)
        ]
        tolerance = 0.25
        matched = [
            bool(np.any(np.abs(pred_curve - target) <= tolerance))
            for target in targets
        ]
        matched_count = int(sum(matched))
        ratio = matched_count / required_count
        return float(ratio), {
            'required_count': required_count,
            'matched_count': matched_count,
            'match_ratio': round(float(ratio), 4),
            'targets': [round(target, 4) for target in targets],
            'matched': matched,
            'tolerance': tolerance,
        }

    def _final_cell_gate(self, final_cell_score: float) -> float:
        final_cell_score = max(0.0, min(1.0, float(final_cell_score)))
        return self.FINAL_CELL_GATE_FLOOR + (1.0 - self.FINAL_CELL_GATE_FLOOR) * final_cell_score

    def _combine_scene_score(self, final_cell_score: float, process_score: float) -> float:
        final_cell_score = max(0.0, min(1.0, float(final_cell_score)))
        process_score = max(0.0, min(1.0, float(process_score)))
        gated_process = process_score * self._final_cell_gate(final_cell_score)
        return float(max(
            0.0, min(1.0, self.FINAL_CELL_WEIGHT * final_cell_score + self.PROCESS_WEIGHT * gated_process)
        ))

    def _score_sequences(
        self,
        pred_seq: Sequence[np.ndarray],
        gt_seq: Sequence[np.ndarray],
        eval_info: Dict[str, Any],
    ) -> float:
        if len(pred_seq) < 2 or len(gt_seq) < 2:
            self._last_task_details = {
                'error': 'sequence_too_short',
                'pred_frames': len(pred_seq),
                'gt_frames': len(gt_seq),
            }
            return 0.0

        reference = gt_seq[0]
        pred_norm = self._normalize_sequence(pred_seq, reference)
        gt_norm = self._normalize_sequence(gt_seq, reference)

        metadata_spec = self._grid_spec_from_metadata(eval_info)
        if metadata_spec is not None:
            grid_size, gt_shift, grid_debug = metadata_spec
        else:
            grid_size, gt_shift, grid_debug = self._infer_grid_size(
                gt_norm, eval_info.get('prompt', ''),
            )
            grid_debug = {'source': 'visual_fallback', **grid_debug}
        if grid_size is None or gt_shift is None:
            self._last_task_details = {
                'error': 'grid_or_shift_inference_failed',
                'grid_debug': grid_debug,
            }
            return 0.0

        gt_start_cells, _ = self._detect_occupied_cells(gt_norm[0], grid_size)
        gt_final_cells, _ = self._detect_occupied_cells(gt_norm[-1], grid_size)
        pred_final_cells, _ = self._detect_occupied_cells(pred_norm[-1], grid_size)

        shift_len = abs(gt_shift[0]) + abs(gt_shift[1])
        final_cell_score, final_detail = self._score_final_cells(sorted(pred_final_cells), sorted(gt_final_cells), shift_len=shift_len)
        temporal_similarity, temporal_detail = self._temporal_similarity(pred_norm, gt_norm, grid_size, gt_shift)

        count_consistency, count_detail = self._occupancy_count_consistency(pred_norm, gt_norm, grid_size)

        intermediate_match_ratio, intermediate_detail = self._intermediate_state_match_ratio(pred_norm, gt_norm, grid_size, gt_shift)

        process_score = temporal_similarity * count_consistency * intermediate_match_ratio

        final_cell_gate = self._final_cell_gate(final_cell_score)

        total = self._combine_scene_score(final_cell_score, process_score)
        self._last_task_details = {
            'formula': 'final_cell_score_weighted_plus_final_gated_process',
            'grid_size': grid_size,
            'gt_shift': [int(gt_shift[0]), int(gt_shift[1])],
            'gt_start_count': len(gt_start_cells),
            'gt_final_count': len(gt_final_cells),
            'pred_final_count': len(pred_final_cells),
            'final_cell_score': round(float(final_cell_score), 4),
            'temporal_similarity': round(float(temporal_similarity), 4),
            'count_consistency': round(float(count_consistency), 4),
            'intermediate_match_ratio': round(
                float(intermediate_match_ratio), 4,
            ),
            'process_score': round(float(process_score), 4),
            'final_cell_gate': round(float(final_cell_gate), 4),
            'final_cell_weight': self.FINAL_CELL_WEIGHT,
            'process_weight': self.PROCESS_WEIGHT,
            'final_cell_gate_floor': self.FINAL_CELL_GATE_FLOOR,
            'grid_debug': grid_debug,
            'final_detail': final_detail,
            'temporal_detail': temporal_detail,
            'count_detail': count_detail,
            'intermediate_detail': intermediate_detail,
        }
        return total


class LightSequenceEvaluator(BaseEvaluator):
    """
    O-37: Light Sequence State Control

    Task: Row of balls (gray=off, gold=on). Some should light up.
    Evaluation:
    1. Colored ball correctness (50%): balls that should be on have correct color
    2. Gray ball preservation (30%): gray balls stay gray
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

        fg_last = self._detect_fg_mask(gt_last)
        fg_first = self._detect_fg_mask(gt_first)
        gt_hsv = cv2.cvtColor(gt_last, cv2.COLOR_BGR2HSV)

        # Per-object: find each ball, classify by mean saturation
        contours, _ = cv2.findContours(fg_last, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colored_mask = np.zeros(gt_last.shape[:2], dtype=np.uint8)
        gray_mask = np.zeros(gt_last.shape[:2], dtype=np.uint8)
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            ball_mask = np.zeros(gt_last.shape[:2], dtype=np.uint8)
            cv2.drawContours(ball_mask, [cnt], -1, 255, -1)
            mean_sat = cv2.mean(gt_hsv[:, :, 1], mask=ball_mask)[0]
            if mean_sat > 40:
                cv2.drawContours(colored_mask, [cnt], -1, 255, -1)
            else:
                cv2.drawContours(gray_mask, [cnt], -1, 255, -1)

        kernel = np.ones((5, 5), np.uint8)
        colored_dilated = colored_mask
        gray_dilated = gray_mask
        all_fg = cv2.bitwise_or(fg_first, fg_last)
        all_fg_dilated = cv2.dilate(all_fg, kernel, iterations=1)


        # 1. Colored ball correctness (50%): gt_last vs gen_last in colored region
        colored_score, colored_details = self._pixel_diff_score(
            gt_last, gen_last, colored_dilated, thresholds=(0.15, 0.25, 0.35, 0.50))

        # 2. Gray ball preservation (30%): gt_last vs gen_last in gray region
        gray_score, gray_details = self._pixel_diff_score(
            gt_last, gen_last, gray_dilated, thresholds=(0.15, 0.25, 0.35, 0.50))

        # 3. Background preservation (20%): gen_first vs gen_last outside all fg
        bg_mask = cv2.bitwise_not(all_fg_dilated)
        bg_score, bg_details = self._pixel_diff_score(
            gen_first, gen_last, bg_mask, thresholds=(0.004, 0.01, 0.04, 0.8))

        scores = {
            'completion': round(colored_score * gray_score, 4),
            'background_preservation': round(bg_score, 4),
        }
        self._last_task_details = {
            **scores,
            'colored_correctness': round(colored_score, 4),
            'gray_preservation': round(gray_score, 4),
            'colored_px': int((colored_mask > 0).sum()),
            'gray_px': int((gray_mask > 0).sum()),
            **{f'colored_{k}': v for k, v in colored_details.items()},
            **{f'gray_{k}': v for k, v in gray_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class MajorityColorEvaluator(BaseEvaluator):
    """
    O-38: Majority Color - keep only the majority color shapes, remove others.

    Evaluation:
    1. Kept color correctness (40%): remaining shapes have correct (majority) color
    2. Removed shapes deletion (40%): non-majority shapes are removed
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

        kept_mask = self._detect_fg_mask(gt_last)
        changed_mask = self._detect_changed_region(gt_first, gt_last)
        fg_first = self._detect_fg_mask(gt_first)
        all_fg = cv2.bitwise_or(fg_first, kept_mask)

        kernel = np.ones((5, 5), np.uint8)
        kept_dilated = cv2.dilate(kept_mask, kernel, iterations=1)
        changed_dilated = cv2.dilate(changed_mask, kernel, iterations=1)
        all_fg_dilated = cv2.dilate(all_fg, kernel, iterations=1)

        # 1. Kept color correctness (40%): gt_last vs gen_last in kept region
        kept_score, kept_details = self._pixel_diff_score(
            gt_last, gen_last, cv2.erode(kept_mask, kernel, iterations=1), thresholds=(0.15, 0.25, 0.35, 0.60))

        # 2. Removed shapes deletion (40%): gt_last vs gen_last in changed region
        removed_score, removed_details = self._pixel_diff_score(
            gt_last, gen_last, cv2.erode(changed_mask, kernel, iterations=1), thresholds=(0.15, 0.25, 0.35, 0.60))

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
            'kept_color': round(kept_score, 4),
            'removed_shapes': round(removed_score, 4),
            **{f'kept_{k}': v for k, v in kept_details.items()},
            **{f'removed_{k}': v for k, v in removed_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class RotationPuzzleEvaluator(BaseEvaluator):
    """
    O-44: Rotation Puzzle evaluator.

    Dimensions:
        - template_preservation (40%): for each generated frame the 4 cells are
          compared with the corresponding L-shape templates extracted from the GT
          first frame via Hu moments distance.
        - completion (40%): cell-by-cell IoU + color comparison of the generated
          final frame against the GT final frame.
        - background_preservation (20%): pixel similarity on the stable background
          region (area unchanged between GT first and GT final frames).
    """

    TASK_WEIGHTS = {
        "template_preservation": 0.40,
        "completion": 0.40,
        "background_preservation": 0.20,
    }

    CELL_COMPLETION_WEIGHTS = {
        "iou": 0.80,
        "color": 0.20,
    }

    IOU_HIGH_THRESHOLD = 0.90
    IOU_LOW_THRESHOLD = 0.50
    INTERMEDIATE_ENDPOINT_IOU_MAX = 0.80
    DISTINCT_STATE_IOU_MAX = 0.90
    REQUIRED_DISTINCT_INTERMEDIATE_STATES = 2

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

        offset = np.median(pixels_a, axis=0) - np.median(pixels_b, axis=0)
        pixel_distances = np.linalg.norm(pixels_a - (pixels_b + offset), axis=1)
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

    @staticmethod
    def _split_into_four_cells(frame: np.ndarray) -> List[np.ndarray]:
        """Split frame into 4 equal cells: [TL, TR, BL, BR]."""
        h, w = frame.shape[:2]
        hh, hw = h // 2, w // 2
        return [frame[:hh, :hw], frame[:hh, hw:], frame[hh:, :hw], frame[hh:, hw:]]

    def _extract_cell_features(self, cell: np.ndarray) -> Optional[Dict]:
        """Extract L-shape foreground features (mask, centroid, color) from a single cell."""
        if cell.size == 0:
            return None
        ch, cw = cell.shape[:2]
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 50:
            return None
        M = cv2.moments(largest)
        if M["m00"] <= 0:
            return None
        cx_m = float(M["m10"] / M["m00"])
        cy_m = float(M["m01"] / M["m00"])
        contour_mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.drawContours(contour_mask, [largest], -1, 255, thickness=-1)
        mean_bgr = np.array(cv2.mean(cell, mask=contour_mask)[:3], dtype=np.float32)

        return {
            "contour": largest,
            "mask": fg_mask,
            "centroid": (cx_m, cy_m),
            "mean_bgr": mean_bgr,
        }

    def _rotation_sim(self, feat_tmpl: Optional[Dict], feat_gen: Optional[Dict]) -> float:
        """
        Shape similarity based on Hu moments distance.
        """
        if feat_tmpl is None or feat_gen is None:
            return 0.0
        cnt_temp = feat_tmpl.get("contour")
        cnt_tgt = feat_gen.get("contour")
        if cnt_temp is None or cnt_tgt is None:
            return 0.0
        distance = cv2.matchShapes(cnt_temp, cnt_tgt, cv2.CONTOURS_MATCH_I2, 0)
        similarity = max(0.0, 1.0 - (distance / 0.5))
        return float(similarity)

    def _iou_score(self, feat_a: Optional[Dict], feat_b: Optional[Dict]) -> float:
        """Pixel-level IoU between two cell shape masks."""
        if feat_a is None or feat_b is None:
            return 0.0
        mask_a = feat_a["mask"]
        mask_b = feat_b["mask"]
        if mask_a.shape != mask_b.shape:
            mask_b = cv2.resize(mask_b, (mask_a.shape[1], mask_a.shape[0]), interpolation=cv2.INTER_NEAREST)
        intersection = float(np.logical_and(mask_a > 0, mask_b > 0).sum())
        union = float(np.logical_or(mask_a > 0, mask_b > 0).sum())
        raw_iou = intersection / union if union > 0 else 0.0
        if raw_iou >= self.IOU_HIGH_THRESHOLD:
            return float(raw_iou)
        if raw_iou <= self.IOU_LOW_THRESHOLD:
            return 0.0
        score = float(((raw_iou - self.IOU_LOW_THRESHOLD) / (self.IOU_HIGH_THRESHOLD - self.IOU_LOW_THRESHOLD)) * self.IOU_HIGH_THRESHOLD)
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _raw_feature_iou(
        feat_a: Optional[Dict],
        feat_b: Optional[Dict],
    ) -> float:
        if feat_a is None or feat_b is None:
            return 0.0
        mask_a = feat_a['mask'] > 0
        mask_b = feat_b['mask'] > 0
        if mask_a.shape != mask_b.shape:
            mask_b = cv2.resize(
                mask_b.astype(np.uint8),
                (mask_a.shape[1], mask_a.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        union = int(np.logical_or(mask_a, mask_b).sum())
        if union == 0:
            return 1.0
        return float(np.logical_and(mask_a, mask_b).sum() / union)

    def _rotation_activity_score(
        self,
        frames: Sequence[np.ndarray],
        gt_first: np.ndarray,
        gt_last: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Require two arbitrary distinct intermediate rotations, not fixed angles."""
        if len(frames) < 4:
            return 0.0, {
                'distinct_intermediate_states': 0,
                'required_distinct_states': self.REQUIRED_DISTINCT_INTERMEDIATE_STATES,
                'reason': 'too_few_frames',
            }

        start_features = [
            self._extract_cell_features(cell)
            for cell in self._split_into_four_cells(gt_first)
        ]
        final_features = [
            self._extract_cell_features(cell)
            for cell in self._split_into_four_cells(gt_last)
        ]
        moving_cells = [
            idx for idx in range(4)
            if self._raw_feature_iou(start_features[idx], final_features[idx]) < 0.98
        ]
        if not moving_cells:
            return 1.0, {
                'distinct_intermediate_states': 0,
                'required_distinct_states': 0,
                'moving_cells': [],
                'reason': 'no_observable_rotating_cells',
            }

        representatives: List[List[Optional[Dict]]] = []
        candidate_details: List[Dict[str, Any]] = []
        for frame_idx, frame in enumerate(frames[1:-1], start=1):
            features = [
                self._extract_cell_features(cell)
                for cell in self._split_into_four_cells(frame)
            ]
            start_iou = float(np.mean([
                self._raw_feature_iou(features[idx], start_features[idx])
                for idx in moving_cells
            ]))
            final_iou = float(np.mean([
                self._raw_feature_iou(features[idx], final_features[idx])
                for idx in moving_cells
            ]))
            is_intermediate = (
                start_iou <= self.INTERMEDIATE_ENDPOINT_IOU_MAX
                and final_iou <= self.INTERMEDIATE_ENDPOINT_IOU_MAX
            )
            is_distinct = False
            if is_intermediate:
                is_distinct = all(
                    float(np.mean([
                        self._raw_feature_iou(features[idx], representative[idx])
                        for idx in moving_cells
                    ])) < self.DISTINCT_STATE_IOU_MAX
                    for representative in representatives
                )
                if is_distinct:
                    representatives.append(features)
            candidate_details.append({
                'frame_index': frame_idx,
                'start_iou': round(start_iou, 4),
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
            'moving_cells': moving_cells,
            'endpoint_iou_max': self.INTERMEDIATE_ENDPOINT_IOU_MAX,
            'distinct_state_iou_max': self.DISTINCT_STATE_IOU_MAX,
            'candidates': candidate_details,
        }

    def _color_sim(self, feat_a: Optional[Dict], feat_b: Optional[Dict]) -> float:
        """Mean-BGR color similarity between two cell shapes."""
        if feat_a is None or feat_b is None:
            return 0.0
        dist = float(np.linalg.norm(feat_a["mean_bgr"] - feat_b["mean_bgr"]))
        return float(max(0.0, 1.0 - dist / np.sqrt(3.0 * (255.0 ** 2))))

    def _compute_completion_score(
        self,
        gt_last_feats: List[Optional[Dict]],
        gen_last_feats: List[Optional[Dict]],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute per-cell completion as weighted IoU + color similarity,
        averaged over the 4 cells.
        """
        cell_scores: List[float] = []
        details: Dict[str, float] = {}
        for i in range(4):
            iou = self._iou_score(gt_last_feats[i], gen_last_feats[i])
            color = self._color_sim(gt_last_feats[i], gen_last_feats[i])
            cell_score = (
                self.CELL_COMPLETION_WEIGHTS["iou"] * iou
                + self.CELL_COMPLETION_WEIGHTS["color"] * color
            )
            cell_scores.append(cell_score)
            details[f"cell_{i}_iou"] = iou
            details[f"cell_{i}_color"] = color
            details[f"cell_{i}_score"] = cell_score

        completion = float(np.mean(cell_scores)) if cell_scores else 0.0
        return completion, details

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Evaluate rotation puzzle task."""
        scores: Dict[str, float] = {}

        if len(video_frames) < 2 or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        if last_frame.shape[:2] != gt_final_frame.shape[:2]:
            video_frames = [normalize_frame_size(f, gt_final_frame) for f in video_frames]
            first_frame, last_frame = video_frames[0], video_frames[-1]
        gt_first, gt_last = gt_first_frame, gt_final_frame

        # 1) template_preservation (40%): extract L-shape templates from the 4 cells
        gt_first_cells = self._split_into_four_cells(gt_first)
        templates = [self._extract_cell_features(c) for c in gt_first_cells]

        frame_scores: List[float] = []
        for frame in video_frames:
            gen_cells = self._split_into_four_cells(frame)
            gen_feats = [self._extract_cell_features(c) for c in gen_cells]
            cell_sims = [self._rotation_sim(templates[i], gen_feats[i]) for i in range(4)]
            frame_scores.append(float(np.mean(cell_sims)))
        scores["template_preservation"] = float(np.mean(frame_scores)) if frame_scores else 0.0

        # 2) completion (40%): cell-by-cell IoU + color between generated final and GT final.
        gt_last_cells = self._split_into_four_cells(gt_last)
        gt_last_feats = [self._extract_cell_features(c) for c in gt_last_cells]
        gen_last_cells = self._split_into_four_cells(last_frame)
        gen_last_feats = [self._extract_cell_features(c) for c in gen_last_cells]
        completion_score, completion_details = self._compute_completion_score(
            gt_last_feats, gen_last_feats
        )

        def _stroke(img):
            return (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1] > 60).astype(np.uint8)

        def _canon_frame(img):
            m = _stroke(img)
            ys, xs = np.nonzero(m)
            if ys.size < 50:
                return None
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            if y1 - y0 < 8 or x1 - x0 < 8:
                return None
            return cv2.resize(img[y0:y1, x0:x1], (512, 512), interpolation=cv2.INTER_LINEAR)

        _cg, _cp = _canon_frame(gt_last), _canon_frame(last_frame)
        if _cg is not None and _cp is not None:
            _gf = [self._extract_cell_features(c) for c in self._split_into_four_cells(_cg)]
            _pf = [self._extract_cell_features(c) for c in self._split_into_four_cells(_cp)]
            _cs, _ = self._compute_completion_score(_gf, _pf)
            if _cs > completion_score:
                completion_score = _cs
                completion_details["canonical_cellwise"] = round(_cs, 4)

        gt_fg = _stroke(gt_last)
        gen_fg = _stroke(last_frame)
        _union = float(np.logical_or(gt_fg > 0, gen_fg > 0).sum())
        whole_frame_iou = (
            float(np.logical_and(gt_fg > 0, gen_fg > 0).sum()) / _union if _union > 0 else 0.0
        )
        completion_details["whole_frame_iou"] = whole_frame_iou
        completion_score = max(completion_score, whole_frame_iou)
        scores["completion"] = completion_score

        # 3) background_preservation (20%): pixel similarity on stable background region.
        change_mask = self._shape_change_mask(gt_first, gt_last)
        _, first_bg = self._frame_masks(first_frame)
        bg_compare_mask = cv2.bitwise_and(first_bg, cv2.bitwise_not(change_mask))
        scores["background_preservation"] = self._pixel_similarity(
            first_frame, last_frame, bg_compare_mask, strictness=3.0, min_cutoff=0.6
        )

        rotation_process, rotation_process_details = self._rotation_activity_score(
            video_frames, gt_first, gt_last,
        )
        scores["rotation_process"] = rotation_process

        self._last_task_details = {
            **scores,
            "completion_details": completion_details,
            "rotation_process_details": rotation_process_details,
        }

        TAU = 0.35
        ramp = min(1.0, scores["completion"] / TAU)
        keep = (self.TASK_WEIGHTS["template_preservation"] * scores["template_preservation"]
                + self.TASK_WEIGHTS["background_preservation"] * scores["background_preservation"])
        completion_score = (
            self.TASK_WEIGHTS["completion"] * scores["completion"] + ramp * keep
        )
        total = completion_score * (0.4 + 0.6 * rotation_process)
        self._last_task_details["final_state_score"] = round(
            float(completion_score), 4,
        )
        self._last_task_details["score_formula"] = (
            "final_state_score * (0.4 + 0.6 * rotation_process)"
        )
        return float(total)



class SequenceCompletionEvaluator(BaseEvaluator):
    """
    O-45: Sequence Completion - replace ? with correct next element.

    Evaluation:
    1. Generated object correctness (60%): new element color matches GT
    2. Existing object preservation (20%): original elements preserved
    3. Background preservation (20%): background stays clean
    """

    TASK_WEIGHTS = {
        'generated_object': 0.80,
        'consistency': 0.20,
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

        # New element region: where GT changed (? replaced with answer)
        new_obj_mask = self._detect_changed_region(gt_first, gt_last)
        # Existing objects: foreground in GT first minus the ? mark area
        fg_first = self._detect_fg_mask(gt_first)
        # The ? mark is part of fg_first but inside changed region
        existing_mask = cv2.bitwise_and(fg_first, cv2.bitwise_not(new_obj_mask))
        # All foreground
        fg_last = self._detect_fg_mask(gt_last)
        all_fg = cv2.bitwise_or(fg_first, fg_last)

        kernel = np.ones((5, 5), np.uint8)
        new_obj_dilated = cv2.dilate(new_obj_mask, kernel, iterations=1)
        existing_dilated = cv2.dilate(existing_mask, kernel, iterations=1)
        all_fg_dilated = cv2.dilate(all_fg, kernel, iterations=1)

        # 1. Generated object correctness: gt_last vs gen_last in new object region (erode edges)
        new_score, new_details = self._pixel_diff_score(
            gt_last, gen_last, cv2.erode(new_obj_mask, kernel, iterations=1),
            thresholds=(0.15, 0.25, 0.35, 0.60))

        _eroded_existing = cv2.erode(existing_mask, kernel, iterations=1)
        _n_obj, _labels = cv2.connectedComponents((_eroded_existing > 0).astype(np.uint8))
        _per_obj = []
        for _lb in range(1, _n_obj):
            _m = np.where(_labels == _lb, 255, 0).astype(np.uint8)
            if int((_m > 0).sum()) < 100:
                continue
            _s, _ = self._pixel_diff_score(gt_last, gen_last, _m,
                                           thresholds=(0.15, 0.25, 0.35, 0.60))
            _per_obj.append(_s)
        if _per_obj:
            obj_score = float(np.mean(_per_obj))
            obj_details = {'per_object': [round(x, 3) for x in _per_obj]}
        else:
            obj_score, obj_details = self._pixel_diff_score(
                gt_last, gen_last, _eroded_existing,
                thresholds=(0.15, 0.25, 0.35, 0.60))

        # 3. Background preservation: gen_first vs gen_last outside all fg
        bg_mask = cv2.bitwise_not(all_fg_dilated)
        bg_score, bg_details = self._pixel_diff_score(
            gen_first, gen_last, bg_mask, thresholds=(0.01, 0.025, 0.05, 0.10))

        def _solidity_seq(img):
            m = (self._detect_fg_mask(img) > 0).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out = []
            for c in cs:
                a = cv2.contourArea(c)
                if a < 300:
                    continue
                ha = cv2.contourArea(cv2.convexHull(c))
                x, _y, _w2, _h2 = cv2.boundingRect(c)
                out.append((x, a / max(ha, 1.0)))
            out.sort()
            return [v for _x, v in out]

        _gt_seq, _gen_seq = _solidity_seq(gt_last), _solidity_seq(gen_last)
        if len(_gt_seq) >= 2 and len(_gen_seq) == len(_gt_seq):
            _agrees = []
            for _a, _b in zip(_gt_seq[:-1], _gen_seq[:-1]):
                _d = abs(_a - _b)
                _agrees.append(1.0 if _d <= 0.10 else max(0.0, 1.0 - (_d - 0.10) / 0.25))
            _seq_gate = min(_agrees) if _agrees else 1.0
        elif len(_gen_seq) != len(_gt_seq):
            _seq_gate = 0.0  
        else:
            _worst_obj = min(_per_obj) if _per_obj else obj_score
            _seq_gate = min(1.0, _worst_obj / 0.4)
        scores = {
            'generated_object': round(new_score * _seq_gate, 4),
            'consistency': round((obj_score + bg_score) / 2, 4),
        }
        self._last_task_details = {
            **scores,
            'object_preservation': round(obj_score, 4),
            'background_preservation': round(bg_score, 4),
            **{f'new_{k}': v for k, v in new_details.items()},
            **{f'obj_{k}': v for k, v in obj_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }
        
        return float((scores['generated_object']) * (0.6 + 0.4 * scores['consistency']))


class SlidingPuzzleEvaluator(BaseEvaluator):
    """
    O-47: Sliding Puzzle

    Task: Solve 3x3 sliding puzzle in exactly N moves.
    Goal: arrange tiles 1-8 in row-major order; empty space at bottom-right.

    Approach:
    1. Detect grid bounds from GT first frame (dark gridlines).
    2. Extract key frames: frames where exactly 8 cells occupied + 1 empty,
       deduplicated by empty-cell position (consecutive same position → keep first).
    3. OCR each key frame to read tile numbers.
    4. Final state (60%): last key frame numbers match expected 1-8 order.
    5. Process (40%): key frame sequence shows valid single-tile moves.
    """

    TASK_WEIGHTS = {'final_state': 0.60, 'process': 0.40}
    GRID_SIZE = 3
    EMPTY_BRIGHTNESS = 220

    def __init__(self, device: str = 'cpu', task_name: str = ''):
        super().__init__(device, task_name)
        self._easyocr_reader = None

    def _get_easyocr_reader(self):
        if self._easyocr_reader is None:
            import os, easyocr, torch
            self._easyocr_reader = easyocr.Reader(
                ['en'], gpu=torch.cuda.is_available(),
                model_storage_directory=(os.environ.get('VBVR_EASYOCR_MODELS') or None))
        return self._easyocr_reader

    def _detect_grid_lines(self, frame: np.ndarray) -> Tuple[List[int], List[int]]:
        """Detect grid lines and return exactly GRID_SIZE+1 boundary positions.
        Returns (h_bounds, v_bounds) where len = 4 for a 3x3 grid.
        Tries multiple thresholds to find all 4 lines per axis.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        dark = (gray < 80).astype(np.uint8)
        row_dark = dark.sum(axis=1)
        col_dark = dark.sum(axis=0)

        def collapse(lines, gap=15):
            if not lines:
                return []
            groups, cur = [], [lines[0]]
            for l in lines[1:]:
                if l - cur[-1] <= gap:
                    cur.append(l)
                else:
                    groups.append(int(np.mean(cur)))
                    cur = [l]
            groups.append(int(np.mean(cur)))
            return groups

        def detect_bounds(density, total_size):
            even = [total_size * i // self.GRID_SIZE for i in range(self.GRID_SIZE + 1)]
            lines = collapse(sorted(
                np.where(density > total_size * 0.5)[0].tolist()))
            if len(lines) >= 4:
                d1, d2 = lines[1], lines[2]
                cell_size = d2 - d1
                border_left = max(0, d1 - cell_size)
                border_right = min(total_size, d2 + cell_size)
                bounds = [border_left, d1, d2, border_right]
            elif len(lines) == 1:
                cs = lines[0]
                bounds = [0, cs, cs * 2, min(total_size, cs * 3)]
            else:
                return even
            # Reject degenerate bounds that don't span the frame; use even split.
            if bounds[-1] - bounds[0] < 0.6 * total_size:
                return even
            return bounds

        return detect_bounds(row_dark, w), detect_bounds(col_dark, h)

    def _get_cell_bounds(self, row: int, col: int,
                         h_dividers: List[int], v_dividers: List[int],
                         fh: int, fw: int) -> Tuple[int, int, int, int]:
        """Get (x1, y1, x2, y2) for cell (row, col) from actual divider positions."""
        pad = 6
        # y bounds
        if row < len(h_dividers):
            y1 = h_dividers[row]
        else:
            y1 = h_dividers[-1] if h_dividers else 0
        if row + 1 < len(h_dividers):
            y2 = h_dividers[row + 1]
        else:
            y2 = fh
        # x bounds
        if col < len(v_dividers):
            x1 = v_dividers[col]
        else:
            x1 = v_dividers[-1] if v_dividers else 0
        if col + 1 < len(v_dividers):
            x2 = v_dividers[col + 1]
        else:
            x2 = fw
        return max(0, x1 + pad), max(0, y1 + pad), min(fw, x2 - pad), min(fh, y2 - pad)

    def _get_cell(self, frame: np.ndarray, row: int, col: int,
                  h_dividers: List[int], v_dividers: List[int]) -> np.ndarray:
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = self._get_cell_bounds(row, col, h_dividers, v_dividers, fh, fw)
        return frame[y1:y2, x1:x2]

    def _is_empty_cell(self, cell: np.ndarray) -> bool:
        if cell.size == 0:
            return False
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell
        bright_ratio = float((gray > self.EMPTY_BRIGHTNESS).sum()) / gray.size
        return bright_ratio > 0.7

    def _get_occupancy(self, frame: np.ndarray,
                       h_dividers: List[int],
                       v_dividers: List[int]) -> Optional[Tuple[int, int]]:
        """Return (row, col) of the empty cell if exactly 1 empty + 8 occupied.
        Returns None if not a valid stable state (mid-animation)."""
        empty_cells = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell = self._get_cell(frame, r, c, h_dividers, v_dividers)
                if self._is_empty_cell(cell):
                    empty_cells.append((r, c))
        if len(empty_cells) == 1:
            return empty_cells[0]
        return None

    def _extract_key_frames(self, video_frames: List[np.ndarray],
                            h_dividers: List[int],
                            v_dividers: List[int]) -> List[Tuple[int, Tuple[int, int]]]:
        """Extract key frames: valid states (8 occupied + 1 empty), deduplicated
        by empty-cell position. Use the LAST frame of each run (most settled).
        Returns list of (frame_index, empty_cell_pos).
        """
        # Collect all valid (frame_idx, empty_pos) pairs
        runs: List[Tuple[int, Tuple[int, int]]] = []
        for i, frame in enumerate(video_frames):
            empty_pos = self._get_occupancy(frame, h_dividers, v_dividers)
            if empty_pos is not None:
                runs.append((i, empty_pos))

        # Group consecutive frames with same empty position, keep MIDDLE of each run
        key_frames = []
        run_start = 0
        for idx in range(len(runs)):
            is_last = (idx + 1 >= len(runs)) or (runs[idx][1] != runs[idx + 1][1])
            if is_last:
                mid = (run_start + idx) // 2
                key_frames.append(runs[mid])
                run_start = idx + 1
        return key_frames

    def _ocr_cell_number(self, cell: np.ndarray) -> Optional[int]:
        import re
        from collections import Counter
        if cell.size == 0:
            return None
        ch, cw = cell.shape[:2]
        margin_y, margin_x = ch // 4, cw // 4
        cropped = cell[margin_y:ch - margin_y, margin_x:cw - margin_x]
        if cropped.size == 0:
            return None
        # Resize FIRST (smooth), then threshold (clean edges)
        big = cv2.resize(cropped, (100, 100), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        reader = self._get_easyocr_reader()

        mean_b = float(gray.mean())
        # Tile colour varies between renders, so the white digit's contrast does
        # too; read at several cut points and take the majority vote.
        votes = []
        for thresh_val in (min(245, int(mean_b + 30)),
                           min(245, int(mean_b + 15)),
                           int((mean_b + 255) / 2)):
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            ocr_img = cv2.dilate(cv2.bitwise_not(binary), np.ones((3, 3), np.uint8), iterations=1)
            rgb = cv2.cvtColor(ocr_img, cv2.COLOR_GRAY2RGB)
            for _, text, conf in reader.readtext(rgb, allowlist='12345678', text_threshold=0.3, low_text=0.2):
                if conf < 0.5:
                    continue
                digits = re.findall(r'[1-8]', text.strip())
                if digits:
                    votes.append(int(digits[0]))
                    break
        if not votes:
            return None
        return Counter(votes).most_common(1)[0][0]

    def _ocr_grid(self, frame: np.ndarray,
                  h_dividers: List[int], v_dividers: List[int],
                  debug_dir: str = '', debug_prefix: str = '') -> List[Optional[int]]:
        """OCR all 9 cells, return list of 9 values (int 1-8 or None for empty)."""
        grid = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell = self._get_cell(frame, r, c, h_dividers, v_dividers)
                is_emp = self._is_empty_cell(cell)
                if debug_dir and cell.size > 0:
                    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                    mean_b = float(gray.mean())
                    cv2.imwrite(f'{debug_dir}/{debug_prefix}cell_{r}{c}_b{mean_b:.0f}{"_empty" if is_emp else ""}.png', cell)
                    # Save OCR preprocessed image (same as _ocr_cell_number pipeline)
                    if not is_emp:
                        _ch, _cw = cell.shape[:2]
                        _my, _mx = _ch // 4, _cw // 4
                        _crop = cell[_my:_ch - _my, _mx:_cw - _mx]
                        _big = cv2.resize(_crop, (100, 100), interpolation=cv2.INTER_CUBIC)
                        _gray_c = cv2.cvtColor(_big, cv2.COLOR_BGR2GRAY)
                        _thresh = min(245, int(float(_gray_c.mean()) + 30))
                        _, _bin = cv2.threshold(_gray_c, _thresh, 255, cv2.THRESH_BINARY)
                        _ocr = cv2.bitwise_not(_bin)
                        _ocr = cv2.dilate(_ocr, np.ones((3, 3), np.uint8), iterations=1)
                        cv2.imwrite(f'{debug_dir}/{debug_prefix}cell_{r}{c}_ocr.png', _ocr)
                if is_emp:
                    grid.append(None)
                else:
                    grid.append(self._ocr_cell_number(cell))
        return grid

    def _score_final_state(self, grid: List[Optional[int]]) -> Tuple[float, Dict]:
        """Check if grid matches target: [1,2,3,4,5,6,7,8,None]."""
        target = [1, 2, 3, 4, 5, 6, 7, 8, None]
        correct = 0
        details = []
        for i, (g, t) in enumerate(zip(grid, target)):
            r, c = i // 3, i % 3
            if g == t:
                correct += 1
                details.append(f'({r},{c}):OK')
            else:
                details.append(f'({r},{c}):got={g} exp={t}')
        sc = correct / 9
        return round(sc, 4), {
            'correct': correct,
            'grid': str(grid),
            'cells': str(details),
        }

    def _score_process(self, key_frames: List[Tuple[int, Tuple[int, int]]],
                       grids: List[List[Optional[int]]],
                       expected_moves: int) -> Tuple[float, Dict]:
        """Evaluate process from key frame sequence.
        Check: valid single-tile moves, correct move count.
        """
        n_states = len(key_frames)
        if n_states < 2:
            return 0.0, {'n_moves': 0, 'expected_moves': expected_moves,
                         'valid_moves': 0, 'count_sc': 0.0, 'validity': 0.0,
                         'moves': '[]', 'reason': 'too_few_key_frames'}

        n_moves = n_states - 1  # each key frame transition = 1 move
        valid_moves = 0
        move_details = []

        for i in range(1, n_states):
            fi_prev, empty_prev = key_frames[i - 1]
            fi_curr, empty_curr = key_frames[i]
            grid_prev = grids[i - 1]
            grid_curr = grids[i]

            dr = abs(empty_curr[0] - empty_prev[0])
            dc = abs(empty_curr[1] - empty_prev[1])
            adjacent = (dr + dc) == 1

            diffs = sum(1 for a, b in zip(grid_prev, grid_curr) if a != b)

            tile_prev = grid_prev[empty_curr[0] * self.GRID_SIZE + empty_curr[1]]
            tile_curr = grid_curr[empty_prev[0] * self.GRID_SIZE + empty_prev[1]]
            if tile_prev is not None and tile_curr is not None:
                tile_consistent = (tile_prev == tile_curr)
            else:
                # OCR missed the sliding tile on a key frame; accept when the
                # geometry is still a clean single-tile slide.
                tile_consistent = (diffs <= 2)

            if adjacent and diffs <= 3 and tile_consistent:
                valid_moves += 1
                move_details.append(
                    f'f{fi_prev}→f{fi_curr}:valid empty={empty_prev}→{empty_curr}')
            else:
                move_details.append(
                    f'f{fi_prev}→f{fi_curr}:invalid adj={adjacent} diffs={diffs} tile={tile_prev}→{tile_curr}')


        validity = valid_moves / max(1, n_moves)
        count_diff = abs(n_moves - expected_moves)
        if count_diff == 0:
            count_sc = 1.0
        elif count_diff <= 1:
            count_sc = 0.7
        elif count_diff <= 2:
            count_sc = 0.3
        else:
            count_sc = 0.0

        proc_sc = validity * count_sc

        return round(proc_sc, 4), {
            'n_moves': n_moves,
            'expected_moves': expected_moves,
            'valid_moves': valid_moves,
            'count_sc': round(count_sc, 2),
            'validity': round(validity, 2),
            'moves': str(move_details),
        }

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

        if video_frames[0].shape != gt_first_frame.shape:
            video_frames = [normalize_frame_size(f, gt_first_frame) for f in video_frames]

        # Grid lines from GT first frame
        h_divs, v_divs = self._detect_grid_lines(gt_first_frame)

        # # Debug directory
        debug_dir = ''


        # Extract key frames (stable puzzle states)
        key_frames = self._extract_key_frames(video_frames, h_divs, v_divs)

        if not key_frames:
            self._last_task_details = {'error': 'no_key_frames_found'}
            return 0.0


        # OCR each key frame (save cell images for first and last key frame)
        grids = []
        for ki, (fi, _) in enumerate(key_frames):
            dbg = debug_dir if ki in (0, len(key_frames) - 1) else ''
            grids.append(self._ocr_grid(video_frames[fi], h_divs, v_divs,
                                        debug_dir=dbg, debug_prefix=f'kf{ki}_'))


        # Expected moves from GT video's key frames
        gt_key_frames = self._extract_key_frames(gt_frames, h_divs, v_divs)
        expected_moves = max(1, len(gt_key_frames) - 1)

        # Final state: OCR the actual last video frame
        last_frame_grid = self._ocr_grid(video_frames[-1], h_divs, v_divs,
                                         debug_dir=debug_dir, debug_prefix='final_')
        final_sc, f_det = self._score_final_state(last_frame_grid)

        # Process: validate key frame sequence
        proc_sc, p_det = self._score_process(key_frames, grids, expected_moves)

        # Format final grid as 3x3 for readability
        fg = last_frame_grid
        final_grid_str = ' | '.join(
            '/'.join(str(fg[r*3+c]) if fg[r*3+c] is not None else '_'
                     for c in range(3))
            for r in range(3))

        scores = {'final_state': final_sc, 'process': proc_sc}
        self._last_task_details = {
            **scores,
            'grid': f'h={h_divs} v={v_divs}',
            'n_key_frames': len(key_frames),
            'gt_key_frames': len(gt_key_frames),
            'key_frame_indices': str([fi for fi, _ in key_frames]),
            'final_grid': final_grid_str,
            'final_correct': f'{f_det["correct"]}/9',
            'final_cells': str(f_det['cells']),
            'n_moves': f'{p_det["n_moves"]}/{expected_moves}',
            'valid_moves': f'{p_det["valid_moves"]}/{p_det["n_moves"]}',
            'moves': str(p_det['moves']),
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
        """Interleave: same final_state + process scoring as video, but use an
        even GRID_SIZE split instead of `_detect_grid_lines`.

        On the image render the tile digits and thick borders create spurious
        dark lines, so `_detect_grid_lines` returns a tiny bogus cluster and
        every frame fails the 8-occupied-1-empty check -> no_key_frames_found.
        The puzzle fills the frame, so an even split lands the cells correctly.
        Video path (_evaluate_task_specific) is untouched.
        """
        if not pred_images or input_frame is None:
            self._last_task_details = {"error": "no_input_or_pred"}
            return 0.0
        video_frames = [input_frame] + pred_images
        gt_frames = gt_images if gt_images else video_frames
        h, w = input_frame.shape[:2]
        h_divs = [h * i // self.GRID_SIZE for i in range(self.GRID_SIZE + 1)]
        v_divs = [w * i // self.GRID_SIZE for i in range(self.GRID_SIZE + 1)]

        key_frames = self._extract_key_frames(video_frames, h_divs, v_divs)
        if not key_frames:
            self._last_task_details = {"error": "no_key_frames_found"}
            return 0.0

        grids = [self._ocr_grid(video_frames[fi], h_divs, v_divs)
                 for fi, _ in key_frames]
        gt_key_frames = self._extract_key_frames(gt_frames, h_divs, v_divs)
        expected_moves = max(1, len(gt_key_frames) - 1)

        last_frame_grid = self._ocr_grid(video_frames[-1], h_divs, v_divs)
        final_sc, f_det = self._score_final_state(last_frame_grid)
        proc_sc, p_det = self._score_process(key_frames, grids, expected_moves)

        scores = {'final_state': final_sc, 'process': proc_sc}
        self._last_task_details = {
            **scores,
            'grid': f'even h={h_divs} v={v_divs}',
            'n_key_frames': len(key_frames),
            'gt_key_frames': len(gt_key_frames),
            'final_correct': f'{f_det["correct"]}/9',
            'n_moves': f'{p_det["n_moves"]}/{expected_moves}',
            'valid_moves': f'{p_det["valid_moves"]}/{p_det["n_moves"]}',
            'note': 'interleave: even grid split (no _detect_grid_lines)',
        }
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class TrafficLightEvaluator(BaseEvaluator):
    """
    O-52: Traffic Light Reasoning

    Task: Crossroad with N active traffic lights. Each cycles Red→Yellow→Green→Yellow→Red
    with countdown numbers. Simulate T seconds and show the final state.

    Evaluation:
    Final state (60%):
      - Light colors match GT final frame
      - Countdown numbers match GT final frame (OCR)
      - Background preserved
    Process (40%):
      - Countdown speed consistent (decreases at steady rate)
      - Color + countdown synchronized (color changes when countdown ≈ 0)
      - Background preserved across frames
    """

    TASK_WEIGHTS = {'final_state': 0.40, 'process': 0.60}

    def __init__(self, device: str = 'cpu', task_name: str = ''):
        super().__init__(device, task_name)
        self._easyocr_reader = None

    def _get_easyocr_reader(self):
        if self._easyocr_reader is None:
            import os, easyocr, torch
            self._easyocr_reader = easyocr.Reader(
                ['en'], gpu=torch.cuda.is_available(),
                model_storage_directory=(os.environ.get('VBVR_EASYOCR_MODELS') or None))
        return self._easyocr_reader

    def _detect_lights(self, frame: np.ndarray) -> List[Dict]:
        """Detect traffic light circles in frame via HSV color + contour.
        Returns list of {'cx', 'cy', 'r', 'color', 'bbox': (x,y,w,h)}.
        """
        fh, fw = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        min_area = (fh * fw) * 0.002  # minimum circle area

        lights = []
        # Red
        red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)
        # Yellow
        yellow_mask = cv2.inRange(hsv, np.array([10, 80, 80]), np.array([40, 255, 255]))
        # Green
        green_mask = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([90, 255, 255]))

        kernel = np.ones((5, 5), np.uint8)
        for color_name, mask in [('red', red_mask), ('yellow', yellow_mask), ('green', green_mask)]:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                # Circularity check
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circ = 4 * np.pi * area / (perimeter * perimeter)
                if circ < 0.5:
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                lights.append({
                    'cx': int(cx), 'cy': int(cy), 'r': int(r),
                    'color': color_name,
                    'bbox': (bx, by, bw, bh),
                    'area': int(area),
                })
        # Deduplicate overlapping detections (keep largest)
        lights.sort(key=lambda l: l['area'], reverse=True)
        kept = []
        for l in lights:
            overlap = False
            for k in kept:
                dist = np.sqrt((l['cx'] - k['cx'])**2 + (l['cy'] - k['cy'])**2)
                if dist < max(l['r'], k['r']):
                    overlap = True
                    break
            if not overlap:
                kept.append(l)
        return kept

    def _detect_color_in_region(self, frame: np.ndarray,
                                bx: int, by: int, bw: int, bh: int) -> str:
        """Detect dominant traffic light color in a bbox region."""
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(fw, bx + bw), min(fh, by + bh)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 'unknown'
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        red_px = int((cv2.bitwise_or(red1, red2) > 0).sum())
        yellow_px = int((cv2.inRange(hsv, np.array([10, 80, 80]),
                                     np.array([40, 255, 255])) > 0).sum())
        green_px = int((cv2.inRange(hsv, np.array([40, 80, 80]),
                                    np.array([90, 255, 255])) > 0).sum())
        dominant = max(red_px, yellow_px, green_px)
        if dominant < 50:
            return 'unknown'
        if dominant == red_px:
            return 'red'
        if dominant == yellow_px:
            return 'yellow'
        return 'green'

    def _ocr_countdown(self, frame: np.ndarray,
                       light: Dict, debug_dir: str = '',
                       debug_prefix: str = '') -> Optional[int]:
        """OCR the countdown number below a light circle."""
        import re
        fh, fw = frame.shape[:2]
        bx, by, bw, bh = light['bbox']
        # Search area: tightly below the light circle
        cy1 = min(fh, by + bh + 5)
        cy2 = min(fh, cy1 + int(bh * 0.8))
        cx1 = max(0, bx - 5)
        cx2 = min(fw, bx + bw + 5)
        search = frame[cy1:cy2, cx1:cx2]
        if search.size == 0:
            return None
        # Detect white rectangle within search area
        gray_s = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY) if len(search.shape) == 3 else search
        _, white_mask = cv2.threshold(gray_s, 200, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            rx, ry, rrw, rrh = cv2.boundingRect(largest)
            # Crop to white box with small inward margin
            m = 3
            roi = search[ry + m:ry + rrh - m, rx + m:rx + rrw - m]
        else:
            roi = search
        if roi.size == 0:
            return None
        # Resize to 100x100 for stable OCR
        roi = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_CUBIC)
        if debug_dir:
            cv2.imwrite(f'{debug_dir}/{debug_prefix}countdown_roi.png', roi)
        try:
            reader = self._get_easyocr_reader()
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = reader.readtext(rgb, allowlist='0123456789',
                                      text_threshold=0.3, low_text=0.2)
            for _, text, conf in results:
                if conf < 0.3:
                    continue
                digits = re.findall(r'\d+', text)
                if digits:
                    return int(digits[0])
            return None
        except Exception:
            return None

    def _read_frame_state(self, frame: np.ndarray,
                          light_positions: List[Dict],
                          debug_dir: str = '',
                          debug_prefix: str = '') -> List[Dict]:
        """Read color + countdown for each light in a frame."""
        states = []
        for i, lp in enumerate(light_positions):
            bx, by, bw, bh = lp['bbox']
            color = self._detect_color_in_region(frame, bx, by, bw, bh)
            countdown = self._ocr_countdown(frame, lp,
                                            debug_dir=debug_dir,
                                            debug_prefix=f'{debug_prefix}L{i}_')
            states.append({'color': color, 'countdown': countdown})
        return states

    def _build_bg_mask(self, frame: np.ndarray,
                       light_positions: List[Dict]) -> np.ndarray:
        """Background mask: everything outside light circles and countdown boxes."""
        fh, fw = frame.shape[:2]
        mask = np.ones((fh, fw), dtype=np.uint8) * 255
        for lp in light_positions:
            bx, by, bw, bh = lp['bbox']
            # Light circle region
            cv2.rectangle(mask, (max(0, bx - 10), max(0, by - 10)),
                          (min(fw, bx + bw + 10), min(fh, by + bh + 10)), 0, -1)
            # Countdown box below
            cy1 = min(fh, by + bh)
            cy2 = min(fh, cy1 + int(bh * 1.0))
            cv2.rectangle(mask, (max(0, bx - 15), cy1),
                          (min(fw, bx + bw + 15), cy2), 0, -1)
        return mask

    def _score_bg_preservation(self, gen_first: np.ndarray, gen_target: np.ndarray,
                               bg_mask: np.ndarray) -> float:
        """Background change ratio → score."""
        mask_px = int((bg_mask > 0).sum())
        if mask_px == 0:
            return 1.0
        diff = cv2.absdiff(gen_first, gen_target)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed = int((gray[bg_mask > 0] > 40).sum())
        ratio = changed / mask_px
        if ratio < 0.02:
            return 1.0
        elif ratio < 0.05:
            return 0.7
        elif ratio < 0.1:
            return 0.4
        elif ratio < 0.20:
            return 0.2
        return 0.0

    def _score_final_state(self, gen_last: np.ndarray, gt_last: np.ndarray,
                           gt_first: np.ndarray,
                           light_positions: List[Dict],
                           bg_mask: np.ndarray,
                           debug_dir: str = '') -> Tuple[float, Dict]:
        """Final state: color + countdown (OCR) + background."""
        # GT final state
        gt_states = self._read_frame_state(gt_last, light_positions,
                                           debug_dir=debug_dir, debug_prefix='gt_final_')
        gen_states = self._read_frame_state(gen_last, light_positions,
                                            debug_dir=debug_dir, debug_prefix='gen_final_')

        color_scores = []
        cd_scores = []
        details = []
        for i, (gt_s, gen_s) in enumerate(zip(gt_states, gen_states)):
            # Color
            if gen_s['color'] == gt_s['color']:
                c_sc = 1.0
            elif gen_s['color'] == 'unknown':
                c_sc = 0.2
            else:
                c_sc = 0.0
            color_scores.append(c_sc)

            # Countdown
            if gt_s['countdown'] is not None and gen_s['countdown'] is not None:
                if gen_s['countdown'] == gt_s['countdown']:
                    cd_sc = 1.0
                elif abs(gen_s['countdown'] - gt_s['countdown']) <= 1:
                    cd_sc = 0.5
                else:
                    cd_sc = 0.0
            elif gt_s['countdown'] is None and gen_s['countdown'] is None:
                cd_sc = 1.0
            else:
                cd_sc = 0.0
            cd_scores.append(cd_sc)

            lp = light_positions[i]
            details.append(
                f'light{i}({lp["cx"]},{lp["cy"]}): '
                f'gt={gt_s["color"]}/{gt_s["countdown"]} '
                f'gen={gen_s["color"]}/{gen_s["countdown"]} '
                f'c_sc={c_sc:.1f} cd_sc={cd_sc:.1f}')

        color_sc = float(np.mean(color_scores)) if color_scores else 0.0
        countdown_sc = float(np.mean(cd_scores)) if cd_scores else 0.0
        bg_sc = self._score_bg_preservation(gt_first, gen_last, bg_mask)

        sc = (color_sc * 0.5 + countdown_sc * 0.5) * (0.4 + 0.6 * bg_sc)

        return round(sc, 4), {
            'color_sc': round(color_sc, 4),
            'countdown_sc': round(countdown_sc, 4),
            'bg_sc': round(bg_sc, 4),
            'details': str(details),
        }

    def _extract_gt_cycle(self, gt_frames: List[np.ndarray],
                          light_positions: List[Dict]) -> List[List[Tuple[str, int]]]:
        """Extract color cycle rules from GT video.
        Returns per-light list of (color, countdown_start) phases.
        e.g. light0: [('red', 4), ('yellow', 3), ('green', 4), ...]
        """
        n_lights = len(light_positions)
        # Read all GT frames
        per_light_readings = [[] for _ in range(n_lights)]
        state_cache = {}
        for frame in gt_frames:
            frame_key = id(frame)
            states = state_cache.get(frame_key)
            if states is None:
                states = self._read_frame_state(frame, light_positions)
                state_cache[frame_key] = states
            for li in range(n_lights):
                if li < len(states):
                    s = states[li]
                    per_light_readings[li].append((s['color'], s['countdown']))

        # Extract phase sequence per light
        cycles = []
        for li in range(n_lights):
            readings = per_light_readings[li]
            phases = []
            cur_color = None
            cur_max_cd = None
            for color, cd in readings:
                if color == 'unknown':
                    continue
                if color != cur_color:
                    # New phase started
                    if cur_color is not None and cur_max_cd is not None:
                        phases.append((cur_color, cur_max_cd))
                    cur_color = color
                    cur_max_cd = cd
                else:
                    # Same phase, track max countdown (start value)
                    if cd is not None:
                        if cur_max_cd is None or cd > cur_max_cd:
                            cur_max_cd = cd
            # Append last phase
            if cur_color is not None and cur_max_cd is not None:
                phases.append((cur_color, cur_max_cd))
            cycles.append(phases)
        return cycles

    def _score_process(self, video_frames: List[np.ndarray],
                       gt_frames: List[np.ndarray],
                       light_positions: List[Dict],
                       bg_mask: np.ndarray) -> Tuple[float, Dict]:
        """Process evaluation using GT-extracted color cycle rules.

        Metrics:
          countdown_sc (40%): per-light, countdown decreases by 1 correctly
          color_change_sc (30%): on countdown finish, next color & countdown match GT cycle
          sync_sc (30%): all lights tick at same frame
        """
        n = len(video_frames)
        if n < 5:
            return 0.0, {
                'countdown_sc': 0.0,
                'color_change_sc': 0.0,
                'sync_sc': 0.0,
                'bg_sc': 1.0,
                'details': [],
                'reason': 'too_few_frames_for_process_evidence',
                'frame_count': n,
            }

        # Extract GT cycle rules
        gt_cycles = self._extract_gt_cycle(gt_frames, light_positions)

        # Read all model frames
        n_lights = len(light_positions)
        timeline = []  # list of [states_per_light]
        state_cache = {}
        for frame in video_frames:
            frame_key = id(frame)
            states = state_cache.get(frame_key)
            if states is None:
                states = self._read_frame_state(frame, light_positions)
                state_cache[frame_key] = states
            timeline.append(states)

        countdown_scores = []
        color_change_scores = []
        proc_details = []

        # Per-light tick frames for sync calculation
        per_light_tick_frames = []

        for li in range(n_lights):
            gt_cycle = gt_cycles[li] if li < len(gt_cycles) else []
            # Build color→next mapping from GT cycle
            # e.g. [('red',4),('yellow',3),('green',4)] → red→('yellow',3), yellow→('green',4)
            next_phase = {}
            for pi in range(len(gt_cycle) - 1):
                cur_col = gt_cycle[pi][0]
                nxt_col, nxt_cd = gt_cycle[pi + 1]
                next_phase[cur_col] = (nxt_col, nxt_cd)

            # Collect per-frame readings
            readings = []
            for fi, states in enumerate(timeline):
                if li < len(states):
                    s = states[li]
                    readings.append((fi, s['color'], s['countdown']))

            # Filter valid countdown readings
            valid = [(fi, c, cd) for fi, c, cd in readings if cd is not None and c != 'unknown']

            # --- Countdown score: within same color, each tick should be -1 ---
            cd_checks = 0
            cd_correct = 0
            tick_frames = set()
            for j in range(1, len(valid)):
                fi_prev, c_prev, cd_prev = valid[j - 1]
                fi_curr, c_curr, cd_curr = valid[j]
                if c_prev == c_curr:
                    if cd_curr != cd_prev:
                        # A tick happened
                        cd_checks += 1
                        if cd_curr == cd_prev - 1:
                            cd_correct += 1
                            tick_frames.add(fi_curr)
                        # else: skip or wrong value

            if cd_checks > 0:
                countdown_scores.append(cd_correct / cd_checks)
            else:
                countdown_scores.append(0.0)  # no ticks detected

            per_light_tick_frames.append(tick_frames)

            # --- Color change score: when countdown goes from 1 to new phase ---
            cc_checks = 0
            cc_correct = 0
            for j in range(1, len(valid)):
                fi_prev, c_prev, cd_prev = valid[j - 1]
                fi_curr, c_curr, cd_curr = valid[j]
                if c_prev != c_curr:
                    # Color changed
                    cc_checks += 1
                    # A transition that does not exist in the GT-derived
                    # cycle is an unexpected color change, not an implicitly
                    # valid one.
                    if c_prev not in next_phase:
                        continue
                    correct_color = True
                    correct_cd = True
                    # Was previous countdown 1? (ideal) or 0? (acceptable)
                    if cd_prev == 0:
                        cd_prev_penalty = 0.3
                    elif cd_prev == 1:
                        cd_prev_penalty = 1.0
                    else:
                        continue
                    exp_col, exp_cd = next_phase[c_prev]
                    if c_curr != exp_col:
                        correct_color = False
                    if cd_curr != exp_cd:
                        correct_cd = False
                    if correct_color and correct_cd:
                        cc_correct += cd_prev_penalty
                    elif correct_color:
                        cc_correct += cd_prev_penalty * 0.5  # right color, wrong countdown start

            if cc_checks > 0:
                color_change_scores.append(cc_correct / cc_checks)
            elif len(gt_cycle) <= 1:
                color_change_scores.append(1.0)
            else:
                color_change_scores.append(0.0)  # expected transition missing

            # Extract gen's observed color phases (same logic as _extract_gt_cycle)
            gen_phases = []
            gen_cur_color = None
            gen_cur_max_cd = None
            for _, c, cd in valid:
                if c != gen_cur_color:
                    if gen_cur_color is not None and gen_cur_max_cd is not None:
                        gen_phases.append((gen_cur_color, gen_cur_max_cd))
                    gen_cur_color = c
                    gen_cur_max_cd = cd
                else:
                    if cd is not None and (gen_cur_max_cd is None or cd > gen_cur_max_cd):
                        gen_cur_max_cd = cd
            if gen_cur_color is not None and gen_cur_max_cd is not None:
                gen_phases.append((gen_cur_color, gen_cur_max_cd))

            gt_cycle_str = '→'.join(f'{c}/{d}' for c, d in gt_cycle) if gt_cycle else '?'
            gen_cycle_str = '→'.join(f'{c}/{d}' for c, d in gen_phases) if gen_phases else '?'
            proc_details.append({
                'light': li,
                'gt_cycle': gt_cycle_str,
                'gen_cycle': gen_cycle_str,
                'ticks': cd_checks,
                'ticks_ok': cd_correct,
                'cd_sc': round(countdown_scores[-1], 2),
                'color_changes': cc_checks,
                'cc_ok': round(cc_correct, 1),
                'cc_sc': round(color_change_scores[-1], 2),
            })

        # --- Sync score: all lights change on the same 1-second grid ---
        per_light_change_frames = []
        for li in range(n_lights):
            prev = None
            changes = set()
            for fi, states in enumerate(timeline):
                if li >= len(states):
                    continue
                s = states[li]
                if s['countdown'] is None or s['color'] == 'unknown':
                    continue
                cur = (s['color'], s['countdown'])
                if prev is not None and cur != prev:
                    changes.add(fi)
                prev = cur
            per_light_change_frames.append(changes)

        all_changes = sorted(f for cf in per_light_change_frames for f in cf)
        if n_lights >= 2 and all_changes:
            # Cluster change-frames that fall within ±2 frames into one "moment".
            clusters = [[all_changes[0]]]
            for f in all_changes[1:]:
                if f - clusters[-1][-1] <= 2:
                    clusters[-1].append(f)
                else:
                    clusters.append([f])
            synced = 0
            for cl in clusters:
                lo, hi = cl[0] - 2, cl[-1] + 2
                # How many lights have a change somewhere in this moment's window
                n_here = sum(1 for cf in per_light_change_frames
                             if any(lo <= f <= hi for f in cf))
                if n_here >= 2:
                    synced += 1
            sync_sc = synced / len(clusters) if clusters else 0.0
        else:
            sync_sc = 0.0

        countdown_sc = float(np.mean(countdown_scores)) if countdown_scores else 0.0
        color_change_sc = float(np.mean(color_change_scores)) if color_change_scores else 0.0

        # Background preservation as penalty (compare against GT first frame)
        bg_scores = []
        gt_first = gt_frames[0]
        bg_cache = {}
        for frame in video_frames:
            frame_key = id(frame)
            if frame_key not in bg_cache:
                bg_cache[frame_key] = self._score_bg_preservation(
                    gt_first, frame, bg_mask,
                )
            bg_scores.append(bg_cache[frame_key])
        bg_sc = float(np.mean(bg_scores)) if bg_scores else 1.0

        # Combined: countdown 40%, color_change 30%, sync 30%, bg as penalty multiplier
        raw_sc = countdown_sc * 0.4 + color_change_sc * 0.3 + sync_sc * 0.3
        proc_sc = raw_sc * (0.4 + 0.6 * bg_sc)

        return round(proc_sc, 4), {
            'countdown_sc': round(countdown_sc, 4),
            'color_change_sc': round(color_change_sc, 4),
            'sync_sc': round(sync_sc, 4),
            'bg_sc': round(bg_sc, 4),
            'details': proc_details,
        }


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

        if video_frames[0].shape != gt_first_frame.shape:
            video_frames = [normalize_frame_size(f, gt_first_frame) for f in video_frames]

        # Detect light positions from GT first frame
        light_positions = self._detect_lights(gt_first_frame)
        if not light_positions:
            self._last_task_details = {'error': 'no_lights_detected'}
            return 0.0

        # Build background mask
        bg_mask = self._build_bg_mask(gt_first_frame, light_positions)

        # # Save debug images
        debug_dir=''

        gen_last = video_frames[-1]
        gt_last = gt_final_frame
        if gen_last.shape != gt_last.shape:
            gen_last = normalize_frame_size(gen_last, gt_last)

        # Final state
        final_sc, f_det = self._score_final_state(
            gen_last, gt_last, gt_first_frame, light_positions, bg_mask,
            debug_dir=debug_dir)

        # Process (needs GT frames for cycle extraction)
        proc_sc, p_det = self._score_process(video_frames, gt_frames, light_positions, bg_mask)

        scores = {'final_state': final_sc, 'process': proc_sc}
        # Per-light final summary
        gt_states = self._read_frame_state(gt_last, light_positions)
        gen_states = self._read_frame_state(gen_last, light_positions)
        final_light_strs = []
        for i, (gt_s, gen_s) in enumerate(zip(gt_states, gen_states)):
            gt_cd = gt_s['countdown'] if gt_s['countdown'] is not None else '_'
            gen_cd = gen_s['countdown'] if gen_s['countdown'] is not None else '_'
            match = '✓' if gt_s['color'] == gen_s['color'] and gt_s['countdown'] == gen_s['countdown'] else '✗'
            final_light_strs.append(f'L{i}: gt={gt_s["color"]}/{gt_cd} gen={gen_s["color"]}/{gen_cd} {match}')
        # Per-light process summary
        proc_light_strs = []
        for d in p_det['details']:
            proc_light_strs.append(
                f'L{d["light"]}: gt={d["gt_cycle"]} gen={d["gen_cycle"]} '
                f'ticks={d["ticks_ok"]}/{d["ticks"]}({d["cd_sc"]}) '
                f'cc={d["cc_ok"]}/{d["color_changes"]}({d["cc_sc"]})')
        self._last_task_details = {
            **scores,
            'n_lights': len(light_positions),
            'final_color': f_det['color_sc'],
            'final_countdown': f_det['countdown_sc'],
            'final_bg': f_det['bg_sc'],
            'final_lights': ' | '.join(final_light_strs),
            'proc_countdown': p_det['countdown_sc'],
            'proc_color_change': p_det['color_change_sc'],
            'proc_sync': p_det['sync_sc'],
            'proc_bg': p_det['bg_sc'],
            'proc_lights': ' | '.join(proc_light_strs),
        }
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

class ClockTimeEvaluator(BaseEvaluator):
    """
    O-53: Clock time evaluator.

    The video starts with a clock showing some time; the model must rotate the
    hour and minute hands clockwise by a fixed number of hours (hours_to_add).

    Dimensions:
        - completion (40%): hour/minute hand final angle vs GT final frame,
          scaled by a soft uniqueness penalty for duplicate hands.
        - process_validity (40%): per-frame detection rate * uniqueness *
          clockwise direction score; hour hand additionally requires the
          correct total rotation (hours_to_add * 30°).
        - element_preservation (20%): pixel stability of the clock face
          mid-ring (70%) and the outer background (30%).
    """

    TASK_WEIGHTS = {
        "completion": 0.40,
        "process_validity": 0.40,
        "element_preservation": 0.20,
    }

    # Maximum number of frames sampled from the video for per-frame checks.
    _MAX_PROC_FRAMES = 60

    def _hex_to_bgr(self, hex_color) -> Tuple[int, int, int]:
        """Convert a '#RRGGBB' hex string OR an [R,G,B] list/tuple to BGR.

        VBVR-Pro stores hand colours as `color_rgb` lists; the legacy schema
        used hex strings. Accept both.
        """
        if isinstance(hex_color, (list, tuple)):
            r, g, b = (int(c) for c in hex_color[:3])
            return (b, g, r)
        hex_color = hex_color.lstrip("#")
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])

    def _get_color_mask(
        self,
        img: np.ndarray,
        hex_color: str,
        tolerance: int = 60,
    ) -> np.ndarray:
        """Return an HSV-based binary mask for the given hex color.

        Uses plain int arithmetic before constructing the numpy arrays to
        avoid uint8 overflow (e.g. 255 + 50 → 49).
        """
        color_bgr = self._hex_to_bgr(hex_color)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_t = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv_t[0]), int(hsv_t[1]), int(hsv_t[2])
        lower = np.array([max(0, h - 10), max(0, s - tolerance), max(0, v - tolerance)], dtype=np.uint8)
        upper = np.array([min(179, h + 10), min(255, s + tolerance), min(255, v + tolerance)], dtype=np.uint8)
        return cv2.inRange(hsv_img, lower, upper)

    def _detect_hand_angle_deg(
        self,
        img: np.ndarray,
        hand_color: str,
        hand_length: float,
        center_position: Tuple[int, int],
        min_area: int = 100,
        max_length_error: int = 60,
    ) -> Optional[float]:
        """Detect a clock hand and return its arctan2 angle in degrees, or None.

        Finds the largest color-matching contour, locates its tip as the point
        farthest from the clock center, then returns arctan2(dy, dx) in degrees
        (image coordinates: y-down, so clockwise rotation → increasing angle).
        Returns None when no valid contour is found or the tip distance deviates
        from hand_length by more than max_length_error pixels.
        """
        hand_mask = self._get_color_mask(img, hand_color)
        contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
            return None
        ctr = np.array(center_position, dtype=np.float64)
        tip = max(largest, key=lambda p: np.linalg.norm(np.array(p[0], dtype=np.float64) - ctr))
        tip = np.array(tip[0], dtype=np.float64)
        if abs(np.linalg.norm(tip - ctr) - hand_length) > max_length_error:
            return None
        return float(np.degrees(np.arctan2(tip[1] - ctr[1], tip[0] - ctr[0])))

    def _count_valid_hand_contours(
        self,
        img: np.ndarray,
        hand_color: str,
        hand_length: float,
        center_position: Tuple[int, int],
        min_area: int = 100,
        max_length_error: int = 60,
    ) -> int:
        """Count distinct clock-hand contours of the given color in one frame.

        Used to detect hallucinated duplicate hands: a well-formed video should
        have exactly one contour per hand per frame.
        """
        hand_mask = self._get_color_mask(img, hand_color)
        contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ctr = np.array(center_position, dtype=np.float64)
        count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            tip = max(cnt, key=lambda p: np.linalg.norm(np.array(p[0], dtype=np.float64) - ctr))
            if abs(np.linalg.norm(np.array(tip[0], dtype=np.float64) - ctr) - hand_length) <= max_length_error:
                count += 1
        return count

    def _angle_diff_deg(self, a1: float, a2: float) -> float:
        """Signed angular difference (a1 − a2) in degrees, wrapped to (−180, 180]."""
        return float(((a1 - a2) + 180.0) % 360.0 - 180.0)

    def _angle_score_deg(
        self,
        ang1: Optional[float],
        ang2: Optional[float],
        tolerance: float = 15.0,
    ) -> float:
        """Linear score in [0, 1]: 1.0 when angles match, 0.0 at ≥ tolerance degrees apart."""
        if ang1 is None or ang2 is None:
            return 0.0
        diff = abs(self._angle_diff_deg(ang1, ang2))
        return float(max(0.0, 1.0 - diff / tolerance))

    def _unwrap_angles(self, angles_deg: List[Optional[float]]) -> List[Optional[float]]:
        """Unwrap a sequence of arctan2 angles (degrees) to remove ±180° discontinuities."""
        result: List[Optional[float]] = []
        prev: Optional[float] = None
        for a in angles_deg:
            if a is None:
                result.append(None)
                continue
            if prev is None:
                result.append(a)
                prev = a
                continue
            diff = ((a - prev) + 180.0) % 360.0 - 180.0
            new_a = prev + diff
            result.append(new_a)
            prev = new_a
        return result

    def _clockwise_direction_score(self, unwrapped: List[Optional[float]]) -> float:
        """Fraction of consecutive frame pairs where the hand moves clockwise (or stays still).

        A tolerance of 1° is applied to ignore single-pixel tip jitter from
        video compression or anti-aliasing.
        """
        count = total = 0
        for i in range(1, len(unwrapped)):
            if unwrapped[i] is None or unwrapped[i - 1] is None:
                continue
            total += 1
            if unwrapped[i] >= unwrapped[i - 1] - 1.0:
                count += 1
        return count / total if total else 0.0

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
        mean_dist = float(np.mean(np.linalg.norm(pixels_a - pixels_b, axis=1)))
        max_dist = float(np.sqrt(3.0 * (255.0 ** 2)))
        base_sim = max(0.0, 1.0 - mean_dist / max_dist)
        final_sim = float(max(0.0, min(1.0, base_sim ** strictness)))
        return final_sim if final_sim >= min_cutoff else 0.0

    def _ring_mask(
        self,
        shape: Tuple[int, int],
        center: Tuple[int, int],
        r_inner: float,
        r_outer: float,
    ) -> np.ndarray:
        """Binary mask for an annular region between r_inner and r_outer."""
        h, w = shape[:2]
        Y, X = np.ogrid[:h, :w]
        d2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2
        return ((d2 >= r_inner ** 2) & (d2 <= r_outer ** 2)).astype(np.uint8) * 255

    def _outer_mask(
        self,
        shape: Tuple[int, int],
        center: Tuple[int, int],
        r_outer: float,
    ) -> np.ndarray:
        """Binary mask for the region strictly outside r_outer."""
        h, w = shape[:2]
        Y, X = np.ogrid[:h, :w]
        d2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2
        return (d2 > r_outer ** 2).astype(np.uint8) * 255

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict,
    ) -> float:
        """Evaluate O-53: clock hand rotation correctness and visual preservation."""
        scores: Dict[str, float] = {}

        if len(video_frames) < 2 or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        first_frame = video_frames[0]
        last_frame = video_frames[-1]

        if gt_frames:
            gt_frames = [
                normalize_frame_size(f, last_frame) if f.shape[:2] != last_frame.shape[:2] else f
                for f in gt_frames
            ]
        gt_last = gt_frames[-1] if gt_frames else gt_final_frame

        import os
        _mp = eval_info.get("metafile_path")
        if isinstance(_mp, (list, tuple)):
            _mp = next((p for p in _mp if p and os.path.exists(p)), _mp[0] if _mp else None)
        with open(_mp) as f:
            metadata = json.load(f)

        params = metadata.get("parameters") or {}
        sgt = metadata.get("semantic_ground_truth") or {}
        objects = sgt.get("objects") or params.get("objects") or []

        def _obj(*syms):
            for o in objects:
                if o.get("symbol") in syms:
                    return o
            raise KeyError(syms[0])

        def _field(o, *keys):
            for k in keys:
                if k in o:
                    return o[k]
            raise KeyError(keys[0])

        clock_face  = _obj("clock_face")
        hour_hand   = _obj("hour_hand")
        minute_hand = _obj("minute_hand")
        center_obj  = _obj("center", "center_dot")
        center_pos: Tuple[int, int] = tuple(
            int(c) for c in _field(center_obj, "position", "position_px", "center_px"))
        clock_radius: float = float(_field(clock_face, "radius", "radius_px"))
        hours_to_add: int = int(params.get("hours_to_add", sgt.get("hours_to_add", 0)))

        h_color = _field(hour_hand, "color", "color_rgb")
        h_len   = float(_field(hour_hand, "length", "length_px"))
        m_color = _field(minute_hand, "color", "color_rgb")
        m_len   = float(_field(minute_hand, "length", "length_px"))

        # 1) completion (40%): final hand angles vs GT, penalized by per-hand
        #    uniqueness in the last frame (1/count; 1.0 when exactly one hand).
        gt_last_h  = self._detect_hand_angle_deg(gt_last, h_color, h_len, center_pos)
        gt_last_m  = self._detect_hand_angle_deg(gt_last, m_color, m_len, center_pos)
        gen_last_h = self._detect_hand_angle_deg(last_frame, h_color, h_len, center_pos)
        gen_last_m = self._detect_hand_angle_deg(last_frame, m_color, m_len, center_pos)

        if gt_last_h is None:
            gt_last_h = self._detect_hand_angle_deg(gt_final_frame, h_color, h_len, center_pos)
        if gt_last_m is None:
            gt_last_m = self._detect_hand_angle_deg(gt_final_frame, m_color, m_len, center_pos)

        hour_angle_score   = self._angle_score_deg(gt_last_h, gen_last_h, tolerance=30.0)
        minute_angle_score = self._angle_score_deg(gt_last_m, gen_last_m, tolerance=60.0)

        h_final_count  = self._count_valid_hand_contours(last_frame, h_color, h_len, center_pos)
        m_final_count  = self._count_valid_hand_contours(last_frame, m_color, m_len, center_pos)
        h_final_unique = 1.0 / h_final_count if h_final_count >= 1 else 1.0
        m_final_unique = 1.0 / m_final_count if m_final_count >= 1 else 1.0

        scores["completion"] = (0.7 * hour_angle_score * h_final_unique
                                + 0.3 * minute_angle_score * m_final_unique)

        # 2) process_validity (40%): per-frame analysis on up to _MAX_PROC_FRAMES.
        #    Hour hand (70%): detection_rate × uniqueness × direction × rotation_accuracy.
        #    Minute hand (30%): detection_rate × uniqueness × direction.
        step = max(1, len(video_frames) // self._MAX_PROC_FRAMES)
        sampled = video_frames[::step]

        h_angles = [self._detect_hand_angle_deg(f, h_color, h_len, center_pos) for f in sampled]
        m_angles = [self._detect_hand_angle_deg(f, m_color, m_len, center_pos) for f in sampled]

        h_det_rate = sum(1 for a in h_angles if a is not None) / len(h_angles)
        m_det_rate = sum(1 for a in m_angles if a is not None) / len(m_angles)

        h_unwrapped = self._unwrap_angles(h_angles)
        m_unwrapped = self._unwrap_angles(m_angles)

        h_dir_score = self._clockwise_direction_score(h_unwrapped)
        m_dir_score = self._clockwise_direction_score(m_unwrapped)

        # Hour total rotation: full score within 30° error, zero at 90° error.
        valid_h = [a for a in h_unwrapped if a is not None]
        if len(valid_h) >= 2:
            rot_err     = abs((valid_h[-1] - valid_h[0]) - float(hours_to_add * 30))
            h_rot_score = float(max(0.0, 1.0 - rot_err / 90.0))
        else:
            h_rot_score = 0.0

        # Uniqueness: fraction of detected frames with exactly one contour.
        h_counts     = [self._count_valid_hand_contours(f, h_color, h_len, center_pos) for f in sampled]
        n_detected_h = sum(1 for c in h_counts if c >= 1)
        n_unique_h   = sum(1 for c in h_counts if c == 1)
        h_uniqueness = n_unique_h / n_detected_h if n_detected_h > 0 else 1.0

        m_counts     = [self._count_valid_hand_contours(f, m_color, m_len, center_pos) for f in sampled]
        n_detected_m = sum(1 for c in m_counts if c >= 1)
        n_unique_m   = sum(1 for c in m_counts if c == 1)
        m_uniqueness = n_unique_m / n_detected_m if n_detected_m > 0 else 1.0

        hour_component   = h_det_rate * h_uniqueness * h_dir_score * h_rot_score
        minute_component = m_det_rate * m_uniqueness * m_dir_score
        is_image_setting = "pred_images" in eval_info
        if is_image_setting:
            scores["process_validity"] = hour_component
        else:
            scores["process_validity"] = 0.7 * hour_component + 0.3 * minute_component

        # 3) element_preservation (20%): GT first frame vs generated last frame.
        mid_ring = self._ring_mask(
            first_frame.shape, center_pos,
            r_inner=clock_radius * 0.25, r_outer=clock_radius * 0.72,
        )
        clock_face_score = self._pixel_similarity(gt_first_frame, last_frame, mid_ring)

        outer    = self._outer_mask(gt_first_frame.shape, center_pos, clock_radius * 1.05)
        bg_score = self._pixel_similarity(gt_first_frame, last_frame, outer)

        scores["element_preservation"] = 0.7 * clock_face_score + 0.3 * bg_score

        self._last_task_details = {
            **scores,
            "completion_details": {
                "hour_angle_score":   hour_angle_score,
                "h_final_unique":     h_final_unique,
                "minute_angle_score": minute_angle_score,
                "m_final_unique":     m_final_unique,
            },
            "process_validity_details": {
                "mode": "image_hour_only" if is_image_setting else "video_hour_and_minute",
                "h_detection_rate": h_det_rate,
                "h_uniqueness":     h_uniqueness,
                "h_direction_score": h_dir_score,
                "h_rotation_score": h_rot_score,
                "hour_component":   hour_component,
                "m_detection_rate": m_det_rate,
                "m_uniqueness":     m_uniqueness,
                "m_direction_score": m_dir_score,
                "minute_component": minute_component,
            },
            "element_preservation_details": {
                "clock_face_score": clock_face_score,
                "background_score": bg_score,
            },
        }
        return float(sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS))


class RotationEvaluator(BaseEvaluator):
    """
    O-55: 3D Mental Rotation

    Task: A voxel sculpture on a table; camera rotates 180° horizontally.
    GT video shows the smooth rotation; first/final frames show start/end views.

    Evaluation:
    1. Final state (60%): gen final frame matches GT final frame.
       Compared via foreground SSIM (background masked out).
    2. Process (40%): pick 3 evenly-spaced intermediate GT frames,
       pick 3 gen frames at the same relative positions,
       compare each pair via SSIM. Measures smooth rotation fidelity.
    """

    TASK_WEIGHTS = {'final_state': 0.60, 'process': 0.40}
    ENDPOINT_DIFF_MIN = 0.08
    DISTINCT_INTERMEDIATE_DIFF_MIN = 0.06
    REQUIRED_DISTINCT_INTERMEDIATES = 2

    def _fg_mask(self, frame: np.ndarray) -> np.ndarray:
        """Foreground mask: non-background pixels (background = corner color)."""
        h, w = frame.shape[:2]
        corners = [frame[2, 2], frame[2, w - 3], frame[h - 3, 2], frame[h - 3, w - 3]]
        bg = np.mean(corners, axis=0).astype(np.float32)
        diff = np.sqrt(np.sum((frame.astype(np.float32) - bg) ** 2, axis=2))
        mask = (diff > 30).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _foreground_difference(
        self, frame_a: np.ndarray, frame_b: np.ndarray,
    ) -> float:
        """Normalized color difference on the union foreground region."""
        if frame_b.shape != frame_a.shape:
            frame_b = normalize_frame_size(frame_b, frame_a)
        union = (self._fg_mask(frame_a) > 0) | (self._fg_mask(frame_b) > 0)
        if not np.any(union):
            return 0.0
        color_dist = np.linalg.norm(
            frame_a[union].astype(np.float32) - frame_b[union].astype(np.float32),
            axis=1,
        )
        return float(np.mean(color_dist) / np.sqrt(3.0 * 255.0 ** 2))

    def _score_intermediate_view_evidence(
        self,
        video_frames: List[np.ndarray],
        candidate_indices: List[int],
    ) -> Tuple[float, Dict]:
        """Credit arbitrary distinct views, but reject repeated endpoints."""
        if len(video_frames) < 3:
            return 0.0, {
                'distinct_intermediates': 0,
                'required': self.REQUIRED_DISTINCT_INTERMEDIATES,
                'candidates': [],
            }

        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        representatives: List[int] = []
        details = []
        for frame_idx in candidate_indices:
            frame = video_frames[frame_idx]
            from_start = self._foreground_difference(frame, first_frame)
            from_final = self._foreground_difference(frame, final_frame)
            away_from_endpoints = (
                from_start >= self.ENDPOINT_DIFF_MIN
                and from_final >= self.ENDPOINT_DIFF_MIN
            )
            distinct = False
            if away_from_endpoints:
                distinct = all(
                    self._foreground_difference(
                        frame, video_frames[representative_idx],
                    ) >= self.DISTINCT_INTERMEDIATE_DIFF_MIN
                    for representative_idx in representatives
                )
                if distinct:
                    representatives.append(frame_idx)
            details.append({
                'frame': frame_idx,
                'from_start': round(from_start, 4),
                'from_final': round(from_final, 4),
                'away_from_endpoints': bool(away_from_endpoints),
                'is_distinct': bool(distinct),
            })

        required = self.REQUIRED_DISTINCT_INTERMEDIATES
        score = min(1.0, len(representatives) / max(required, 1))
        return float(score), {
            'distinct_intermediates': len(representatives),
            'required': required,
            'representative_frames': representatives,
            'endpoint_diff_min': self.ENDPOINT_DIFF_MIN,
            'distinct_diff_min': self.DISTINCT_INTERMEDIATE_DIFF_MIN,
            'candidates': details,
        }

    def _compute_psnr(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Full-image PSNR between two frames."""
        if frame_a.shape != frame_b.shape:
            frame_b = normalize_frame_size(frame_b, frame_a)
        mse = float(np.mean((frame_a.astype(float) - frame_b.astype(float)) ** 2))
        if mse == 0:
            return 100.0
        return 10.0 * np.log10(255.0 ** 2 / mse)

    def _score_final_state(self, gen_last: np.ndarray,
                           gt_final: np.ndarray) -> Tuple[float, Dict]:
        """Compare gen final vs GT final via full-image PSNR."""
        psnr = self._compute_psnr(gen_last, gt_final)
        sc = self._psnr_to_score(psnr)
        return round(sc, 4), {'psnr': round(psnr, 2)}

    def _psnr_to_score(self, psnr: float) -> float:
        if psnr >= 40:
            return 1.0
        elif psnr >= 25:
            return 0.7 + (psnr - 25) / 15.0 * 0.3
        elif psnr >= 21:
            return 0.6 + (psnr - 21) / 4.0 * 0.1
        elif psnr >= 18:
            return 0.4 + (psnr - 18) / 3.0 * 0.2
        elif psnr >= 15:
            return 0.1 + (psnr - 15) / 3.0 * 0.2
        elif psnr >= 10:
            return (psnr - 10) / 5.0 * 0.1
        return 0.0

    def _score_process(self, video_frames: List[np.ndarray],
                       gt_frames: List[np.ndarray]) -> Tuple[float, Dict]:
        """Pick 3 evenly-spaced GT intermediate frames.
        For each GT frame, find the best matching gen frame (highest PSNR),
        with the constraint that matched indices must be in ascending order.
        """
        n_gt = len(gt_frames)
        n_gen = len(video_frames)
        if n_gt < 5 or n_gen < 5:
            return 0.0, {'reason': 'too_few_frames'}

        gt_indices = [n_gt // 4, n_gt // 2, 3 * n_gt // 4]

        intermediate_indices = list(range(1, n_gen - 1))
        gen_sample_step = max(1, len(intermediate_indices) // 20)
        gen_candidates = intermediate_indices[::gen_sample_step]
        if n_gen - 2 not in gen_candidates:
            gen_candidates.append(n_gen - 2)

        # For each GT frame, compute PSNR with all gen candidates
        psnr_matrix = []
        for gi in gt_indices:
            gt_f = gt_frames[gi]
            row = []
            for gci in gen_candidates:
                gen_f = video_frames[gci]
                if gen_f.shape != gt_f.shape:
                    gen_f = normalize_frame_size(gen_f, gt_f)
                row.append(self._compute_psnr(gen_f, gt_f))
            psnr_matrix.append(row)

        # DP ordered matching: maximize total PSNR with ascending indices
        n_k = len(gt_indices)
        n_c = len(gen_candidates)
        dp = [[-1.0] * n_c for _ in range(n_k)]
        parent = [[-1] * n_c for _ in range(n_k)]

        for c in range(n_c):
            dp[0][c] = psnr_matrix[0][c]

        for k in range(1, n_k):
            best_prev = -1.0
            best_prev_c = -1
            for c in range(n_c):
                if c > 0 and dp[k - 1][c - 1] > best_prev:
                    best_prev = dp[k - 1][c - 1]
                    best_prev_c = c - 1
                if best_prev >= 0:
                    val = best_prev + psnr_matrix[k][c]
                    if val > dp[k][c]:
                        dp[k][c] = val
                        parent[k][c] = best_prev_c

        best_total = -1.0
        best_last_c = -1
        for c in range(n_c):
            if dp[n_k - 1][c] > best_total:
                best_total = dp[n_k - 1][c]
                best_last_c = c

        matched = [0] * n_k
        matched[n_k - 1] = best_last_c
        for k in range(n_k - 2, -1, -1):
            matched[k] = parent[k + 1][matched[k + 1]]

        pair_scores = []
        details = []
        for k, gi in enumerate(gt_indices):
            ci = matched[k]
            psnr = psnr_matrix[k][ci]
            sc = self._psnr_to_score(psnr)
            pair_scores.append(sc)
            details.append(f'gt[{gi}]↔gen[{gen_candidates[ci]}]:psnr={psnr:.1f},sc={sc:.2f}')

        matching_score = float(np.mean(pair_scores)) if pair_scores else 0.0
        view_evidence, view_details = self._score_intermediate_view_evidence(
            video_frames, gen_candidates,
        )
        proc_sc = matching_score * view_evidence
        return round(proc_sc, 4), {
            'pairs': details,
            'matching_score': round(matching_score, 4),
            'intermediate_view_evidence': round(view_evidence, 4),
            'intermediate_view_details': view_details,
        }

    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if not video_frames or gt_final_frame is None:
            return 0.0

        gen_last = video_frames[-1]
        if gen_last.shape != gt_final_frame.shape:
            gen_last = normalize_frame_size(gen_last, gt_final_frame)

        final_sc, f_det = self._score_final_state(gen_last, gt_final_frame)
        proc_sc, p_det = self._score_process(video_frames, gt_frames)

        scores = {'final_state': final_sc, 'process': proc_sc}
        self._last_task_details = {
            **scores,
            'final_psnr': f_det.get('psnr'),
            'process_pairs': str(p_det.get('pairs', p_det.get('reason', ''))),
            'process_matching': p_det.get('matching_score', 0.0),
            'process_view_evidence': p_det.get('intermediate_view_evidence', 0.0),
            'process_view_details': p_det.get('intermediate_view_details', {}),
        }
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class CommunicatingVesselsEvaluator(BaseEvaluator):
    """
    O-75: Communicating Vessels - liquid equalizes across connected tubes.

    Evaluation:
    1. Final liquid correct (40%): liquid levels match GT equilibrium
    2. Volume conservation (30%): total liquid amount stays constant during process
    3. Vessel structure preserved (20%): tubes/labels not destroyed
    4. Background clean (10%): no extra objects
    """

    TASK_WEIGHTS = {
        'final_liquid': 0.60,
        'volume_conservation': 0.40,
    }
    EQUILIBRATION_GATE_FLOOR = 0.60
    
    def _detect_fg_mask(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        corners = [frame[2, 2], frame[2, w-3], frame[h-3, 2], frame[h-3, w-3]]
        bg_color = np.mean(corners, axis=0)
        diff = np.sqrt(np.sum((frame.astype(float) - bg_color.astype(float)) ** 2, axis=2))
        binary = (diff > 30).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _get_vessel_interior_mask(self, gt_frame: np.ndarray, vessel_regions: List[Tuple[int, int]]) -> np.ndarray:
        """Get interior mask of vessels using vessel_regions x-ranges and fg_mask y-ranges."""
        h, w = gt_frame.shape[:2]
        fg = self._detect_fg_mask(gt_frame)
        interior = np.zeros((h, w), dtype=np.uint8)
        for center, rw in vessel_regions:
            half = rw // 2
            x1 = max(0, center - half)
            x2 = min(w, center + half)
            # Check vessel walls outside liquid region
            edge_w = max(3, rw // 10)
            left_edge = fg[:, max(0, x1-edge_w):x1]
            right_edge = fg[:, x2:min(w, x2+edge_w)]
            wall_sums = np.sum(left_edge > 0, axis=1) + np.sum(right_edge > 0, axis=1)
            fg_rows = np.where(wall_sums > edge_w * 0.5)[0]
            if len(fg_rows) > 0:
                y2 = fg_rows[-1]
                y1 = fg_rows[-1]
                for j in range(len(fg_rows) - 2, -1, -1):
                    if fg_rows[j+1] - fg_rows[j] <= 5:
                        y1 = fg_rows[j]
                    else:
                        break
                interior[y1:y2, x1:x2] = 255
        return interior

    def _count_liquid_pixels(self, frame: np.ndarray, liquid_hue_range: Tuple[int, int] = None,
                             vessel_mask: np.ndarray = None) -> int:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if liquid_hue_range is not None:
            hue = hsv[:, :, 0]
            sat = hsv[:, :, 1]
            h_lo, h_hi = liquid_hue_range
            if h_lo <= h_hi:
                mask = (hue >= h_lo) & (hue <= h_hi) & (sat > 50)
            else:
                mask = ((hue >= h_lo) | (hue <= h_hi)) & (sat > 50)
        else:
            mask = hsv[:, :, 1] > 80
        if vessel_mask is not None:
            mask = mask & (vessel_mask > 0)
        return int(mask.sum())

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

    def _evaluate_final_equilibrium_vs_gt(self, gen_levels: List[int], gt_levels: List[int]) -> float:
        """Compare generated levels against GT target levels.
        """
        if len(gen_levels) < 2 or len(gt_levels) < 2:
            return 0.2

        gt_mean = np.mean(gt_levels)
        
        gen_std = np.std(gen_levels)
        gen_range = max(gen_levels) - min(gen_levels)
        gen_mean = np.mean(gen_levels)
        
        if gen_range <= 15:
            level_diff = abs(gen_mean - gt_mean)
            if level_diff <= 2.0:
                return 1.0
            return float(np.clip(1.0 - (level_diff - 2.0) / 80.0, 0.4, 1.0))
        elif gen_range <= 30:
            return 0.3
        else:
            return 0.0

    def _score_equilibration_process(
        self,
        video_frames: List[np.ndarray],
        initial_spread: float,
        vessel_regions: List[Tuple[int, int]],
        liquid_hue_range: Tuple[int, int],
    ) -> Tuple[float, Dict[str, Any]]:
        """Check that a liquid-level change is actually shown before the end.

        Final equilibrium judges the endpoint and volume conservation judges
        the physical constraint.  This process term deliberately does not
        require monotonic motion: it only requires at least one detected liquid
        configuration that is meaningfully different from both the starting
        and final configurations.  A direct input-to-final jump has no such
        intermediate state.
        """
        if not video_frames or initial_spread <= 15:
            return 1.0, {
                'has_intermediate_change': True,
                'reason': 'initially_near_equilibrium',
            }

        level_vectors: List[Optional[List[float]]] = []
        for frame in video_frames:
            levels, _, _, _ = self._detect_liquid_levels(
                frame,
                vessel_regions=vessel_regions,
                liquid_hue_range=liquid_hue_range,
            )
            level_vectors.append(
                [float(level) for level in levels]
                if len(levels) == len(vessel_regions) else None
            )

        valid = [(i, levels) for i, levels in enumerate(level_vectors)
                 if levels is not None]
        if len(valid) < 3:
            return 0.0, {
                'has_intermediate_change': False,
                'valid_level_frames': len(valid),
                'reason': 'fewer_than_three_detected_states',
            }

        start_levels = np.asarray(valid[0][1], dtype=float)
        final_levels = np.asarray(valid[-1][1], dtype=float)
        change_threshold = max(5.0, initial_spread * 0.05)
        intermediate_frames = []
        distinct_states = set()
        for frame_idx, levels in valid[1:-1]:
            cur = np.asarray(levels, dtype=float)
            from_start = float(np.mean(np.abs(cur - start_levels)))
            from_final = float(np.mean(np.abs(cur - final_levels)))
            if from_start >= change_threshold and from_final >= change_threshold:
                intermediate_frames.append(frame_idx)
                distinct_states.add(tuple(
                    int(round(level / change_threshold)) for level in levels
                ))

        has_intermediate_change = bool(distinct_states)
        score = 1.0 if has_intermediate_change else 0.0
        sample_count = min(10, len(level_vectors))
        sample_indices = (
            np.linspace(0, len(level_vectors) - 1, sample_count, dtype=int).tolist()
            if sample_count else []
        )
        details = {
            'has_intermediate_change': has_intermediate_change,
            'valid_level_frames': len(valid),
            'change_threshold': round(change_threshold, 3),
            'intermediate_frames': intermediate_frames,
            'distinct_intermediate_states': len(distinct_states),
            'level_samples': [
                None if level_vectors[i] is None else [
                    round(float(level), 1) for level in level_vectors[i]
                ]
                for i in sample_indices
            ],
        }
        return score, details
    
    def _detect_liquid_levels(self, frame: np.ndarray, n_vessels: int = None,
                              vessel_regions: List[Tuple[int, int]] = None,
                              liquid_hue_range: Tuple[int, int] = None) -> Tuple[List[int], List[Tuple[int, int]], Tuple[int, int]]:
        """Detect liquid levels in vessels using pixel color detection.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        if liquid_hue_range is not None:
            hue = hsv[:, :, 0]
            sat = hsv[:, :, 1]
            h_lo, h_hi = liquid_hue_range
            if h_lo <= h_hi:
                liquid_mask = (hue >= h_lo) & (hue <= h_hi) & (sat > 50)
            else:
                liquid_mask = ((hue >= h_lo) | (hue <= h_hi)) & (sat > 50)
        else:
            saturation = hsv[:, :, 1]
            liquid_mask = saturation > 80
            
        kernel_open = np.ones((7, 7), np.uint8)
        liquid_mask = cv2.morphologyEx(liquid_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open).astype(bool)

        if vessel_regions is None:
            col_sums = np.sum(liquid_mask, axis=0)
            threshold = np.max(col_sums) * 0.3 if np.max(col_sums) > 0 else 0

            vessel_regions = []
            in_vessel = False
            start_col = 0

            for x in range(w):
                if col_sums[x] > threshold and not in_vessel:
                    in_vessel = True
                    start_col = x
                elif col_sums[x] <= threshold and in_vessel:
                    in_vessel = False
                    region_width = x - start_col
                    if region_width > w // 30:
                        center = (start_col + x) // 2
                        vessel_regions.append((center, region_width))

            if in_vessel:
                region_width = w - start_col
                if region_width > w // 30:
                    vessel_regions.append(((start_col + w) // 2, region_width))

            vessel_regions.sort(key=lambda x: x[0])

            if len(vessel_regions) < 2:
                n_vessels = n_vessels or 3
                default_w = w // (n_vessels * 2)
                vessel_regions = [((i * 2 + 1) * w // (n_vessels * 2), default_w) for i in range(n_vessels)]

        # Extract liquid hue range from GT (saturation mode only)
        if liquid_hue_range is None:
            liquid_pixels_hue = hsv[:, :, 0][liquid_mask]
            if len(liquid_pixels_hue) > 0:
                median_hue = int(np.median(liquid_pixels_hue))
                liquid_hue_range = (max(0, median_hue - 15), min(180, median_hue + 15))
            else:
                liquid_hue_range = (0, 180)

        # Detect liquid levels at each vessel column using its actual width
        levels = []
        for center, rw in vessel_regions:
            half = rw // 2
            x1 = max(0, center - half)
            x2 = min(w, center + half)
            col_mask = liquid_mask[:, x1:x2]
            row_sums = np.sum(col_mask, axis=1)
            liquid_rows = np.where(row_sums > rw * 0.5)[0]
            if len(liquid_rows) > 0:
                # Find all continuous segments from bottom up, pick the longest
                segments = []
                seg_end = liquid_rows[-1]
                seg_start = liquid_rows[-1]
                for j in range(len(liquid_rows) - 2, -1, -1):
                    if liquid_rows[j+1] - liquid_rows[j] == 1:
                        seg_start = liquid_rows[j]
                    else:
                        segments.append((seg_start, seg_end))
                        seg_end = liquid_rows[j]
                        seg_start = liquid_rows[j]
                segments.append((seg_start, seg_end))
                # Pick the longest segment 
                best = max(segments, key=lambda s: s[1] - s[0])
                levels.append(best[0])

        return levels, vessel_regions, liquid_hue_range, liquid_mask.astype(np.uint8) * 255


    def _evaluate_task_specific(self, video_frames, gt_frames, gt_first_frame, gt_final_frame, eval_info):
        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0

        gen_first, gen_last = video_frames[0], video_frames[-1]
        gt_first, gt_last = gt_first_frame, gt_final_frame

        if gen_first.shape != gt_first.shape:
            gen_first = normalize_frame_size(gen_first, gt_first)
        if gen_last.shape != gt_last.shape:
            gen_last = normalize_frame_size(gen_last, gt_last)


        # 1. Final liquid correct (40%): levels match GT
        # Detect vessel positions from GT, reuse for gen
        gt_levels, gt_vessel_regions, gt_hue_range, gt_liquid_mask = self._detect_liquid_levels(gt_last)
        gen_levels, _, _, gen_liquid_mask = self._detect_liquid_levels(gen_last, vessel_regions=gt_vessel_regions, liquid_hue_range=gt_hue_range)
        eq_score = self._evaluate_final_equilibrium_vs_gt(gen_levels, gt_levels)

        gt_initial_levels, _, _, _ = self._detect_liquid_levels(
            gt_first,
            vessel_regions=gt_vessel_regions,
            liquid_hue_range=gt_hue_range,
        )
        initial_spread = (
            float(max(gt_initial_levels) - min(gt_initial_levels))
            if len(gt_initial_levels) >= 2 else 0.0
        )
        flow_score, flow_details = self._score_equilibration_process(
            video_frames,
            initial_spread,
            gt_vessel_regions,
            gt_hue_range,
        )
        

        # # 2. Volume conservation (30%): liquid amount constant across all frames
        vessel_interior = self._get_vessel_interior_mask(gt_first, gt_vessel_regions)
        vessel_interior = cv2.bitwise_or(vessel_interior, gt_liquid_mask)
        volumes = []
        for f in video_frames:
            if f.shape != gt_first.shape:
                f = normalize_frame_size(f, gt_first)
            volumes.append(self._count_liquid_pixels(f, gt_hue_range, vessel_interior))

        reference_volumes = []
        reference_frames = gt_frames if gt_frames else [gt_first, gt_last]
        for f in reference_frames:
            if f.shape != gt_first.shape:
                f = normalize_frame_size(f, gt_first)
            reference_volumes.append(
                self._count_liquid_pixels(f, gt_hue_range, vessel_interior)
            )

        def max_relative_deviation(values):
            if len(values) < 2 or values[0] <= 0:
                return -1.0
            return max(abs(value / values[0] - 1.0) for value in values)

        max_dev = max_relative_deviation(volumes)
        reference_max_dev = max_relative_deviation(reference_volumes)
        effective_max_dev = (
            max(0.0, max_dev - max(reference_max_dev, 0.0))
            if max_dev >= 0 else -1.0
        )

        if len(volumes) >= 2 and volumes[0] > 0:
            if effective_max_dev < 0.05:
                vol_score = 1.0
            elif effective_max_dev < 0.1:
                vol_score = 1.0 - (effective_max_dev - 0.05) / 0.05 * 0.3
            elif effective_max_dev < 0.2:
                vol_score = 0.7 - (effective_max_dev - 0.1) / 0.1 * 0.4
            else:
                vol_score = max(
                    0.0,
                    0.3 - (effective_max_dev - 0.2) / 0.2 * 0.3,
                )
        else:
            vol_score = 0.0

        # 3. Consistency: vessel structure * background
        hsv_first = cv2.cvtColor(gt_first, cv2.COLOR_BGR2HSV)
        hue_first = hsv_first[:, :, 0]
        sat_first = hsv_first[:, :, 1]
        h_lo, h_hi = gt_hue_range
        if h_lo <= h_hi:
            liquid_mask = ((hue_first >= h_lo) & (hue_first <= h_hi) & (sat_first > 50)).astype(np.uint8) * 255
        else:
            liquid_mask = (((hue_first >= h_lo) | (hue_first <= h_hi)) & (sat_first > 50)).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_OPEN, kernel)
        fg_first = self._detect_fg_mask(gt_first)
        liquid_dilated = cv2.dilate(liquid_mask, kernel, iterations=1)
        vessel_mask = cv2.bitwise_and(fg_first, cv2.bitwise_not(liquid_dilated))
        vessel_score, vessel_details = self._pixel_diff_score(
            gen_first, gen_last, vessel_mask, thresholds=(0.2, 0.35, 0.50, 0.60))

        all_fg = cv2.bitwise_or(self._detect_fg_mask(gt_first), self._detect_fg_mask(gt_last))
        all_fg_dilated = cv2.dilate(all_fg, kernel, iterations=1)
        bg_mask = cv2.bitwise_not(all_fg_dilated)
        bg_score, bg_details = self._pixel_diff_score(
            gen_first, gen_last, bg_mask, thresholds=(0.01, 0.025, 0.05, 0.10))


        scores = {
            'final_liquid': round(eq_score, 4),
            'volume_conservation': round(vol_score * vessel_score, 4),
            'equilibration_process': round(flow_score, 4),
            'consistency': round((vessel_score + bg_score) / 2, 4),
        }
        self._last_task_details = {
            **scores,
            'gt_levels': gt_levels, 'gen_levels': gen_levels,
            'vol_max_deviation': round(float(max_dev), 4),
            'vol_reference_max_deviation': round(float(reference_max_dev), 4),
            'vol_effective_max_deviation': round(float(effective_max_dev), 4),
            'initial_level_spread': round(initial_spread, 4),
            'vessel_score': round(vessel_score, 4),
            'bg_score': round(bg_score, 4),
            **{f'flow_{k}': v for k, v in flow_details.items()},
            **{f'vessel_{k}': v for k, v in vessel_details.items()},
            **{f'bg_{k}': v for k, v in bg_details.items()},
        }

        final_and_volume = sum(
            scores[name] * weight for name, weight in self.TASK_WEIGHTS.items()
        )
        process_gate = self.EQUILIBRATION_GATE_FLOOR + (
            (1.0 - self.EQUILIBRATION_GATE_FLOOR)
            * scores['equilibration_process']
        )
        task_score = final_and_volume * process_gate
        self._last_task_details.update({
            'final_and_volume_score': round(final_and_volume, 4),
            'equilibration_gate': round(process_gate, 4),
        })
        return float(task_score * (0.6 + 0.4 * scores['consistency']))

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

        # BaseEvaluator.evaluate_interleave prepends input_frame to gt_images but
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


# Export mapping for this batch
IN_DOMAIN_50_EVALUATORS_PART5 = {
    'O-36_grid_shift_data-generator': GridShiftEvaluator,
    'O-37_light_sequence_data-generator': LightSequenceEvaluator,
    'O-38_majority_color_data-generator': MajorityColorEvaluator,
    'O-44_rotation_puzzle_data-generator': RotationPuzzleEvaluator,
    'O-45_sequence_completion_data-generator': SequenceCompletionEvaluator,
    'O-47_sliding_puzzle_data-generator': SlidingPuzzleEvaluator,
    'O-52_traffic_light_data-generator': TrafficLightEvaluator,
    'O-53_clock_data-generator': ClockTimeEvaluator,
    'O-55_rotation_data-generator': RotationEvaluator,
    'O-75_communicating_vessels_data-generator': CommunicatingVesselsEvaluator,
}
