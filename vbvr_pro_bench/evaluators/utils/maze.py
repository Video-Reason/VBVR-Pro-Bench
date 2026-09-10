"""
Shared grid-maze helpers used by G-13, G-15, G-16, G-18, G-41.

Before this module the five grid evaluators each kept private copies of the
same pixel-to-cell math, BFS, colour-blob detection, path-proximity scoring
and discontinuity-penalty logic.  This module is the single source of truth
for those helpers so that behaviour changes only need to happen once.

Conventions:
- Cells are ``(row, col)``.  Pixels are ``(x, y)``.
- Frames are BGR uint8 images.
- A *detector* is a callable ``frame -> List[(x, y)]`` returning every
  agent-like blob.  Evaluators build task-specific detectors from
  :func:`make_color_detector` or pass their own bound method.
- Grid size defaults to 10x10; G-41 passes ``grid_size=4`` explicitly.
"""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from ...utils import compute_ssim

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Cell = Tuple[int, int]           # (row, col)
Pixel = Tuple[int, int]          # (x, y)
Detector = Callable[[np.ndarray], List[Pixel]]

DEFAULT_GRID_SIZE = 10

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def cell_size(frame: np.ndarray, grid_size: int = DEFAULT_GRID_SIZE) -> float:
    """Pixels per cell side (``max(H, W) / grid_size``, resolution-independent)."""
    return max(frame.shape[:2]) / float(grid_size)


def pixel_to_cell(
    px: int, py: int, frame_shape: Tuple[int, ...],
    grid_size: int = DEFAULT_GRID_SIZE,
) -> Cell:
    """Pixel ``(x, y)`` -> ``(row, col)``, clamped to grid bounds."""
    h, w = frame_shape[:2]
    return (
        min(max(py * grid_size // h, 0), grid_size - 1),
        min(max(px * grid_size // w, 0), grid_size - 1),
    )


def cell_center_px(
    cell: Cell, frame_shape: Tuple[int, ...],
    grid_size: int = DEFAULT_GRID_SIZE,
) -> Pixel:
    """``(row, col)`` -> pixel ``(x, y)`` at the cell centre."""
    h, w = frame_shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    return (cell[1] * cell_w + cell_w // 2, cell[0] * cell_h + cell_h // 2)


def manhattan(a: Sequence[int], b: Sequence[int]) -> float:
    """L1 distance between two pixel or cell coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Colour-blob detection
# ---------------------------------------------------------------------------

def detect_color_centroids(
    frame: np.ndarray,
    hsv_lower: Sequence[int],
    hsv_upper: Sequence[int],
    *,
    min_area: float = 50.0,
    max_area: Optional[float] = None,
) -> List[Pixel]:
    """Return ``(x, y)`` centroids of every HSV-filtered blob within area bounds.

    ``hsv_lower``/``hsv_upper`` may be a single range; callers that need hue
    wrap-around (e.g. red) should build a compound detector themselves.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Pixel] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        out.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
    return out


def make_color_detector(
    hsv_lower: Sequence[int],
    hsv_upper: Sequence[int],
    *,
    min_area: float = 50.0,
    max_area_cell_fraction: Optional[float] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> Detector:
    """Build a detector that returns ``(x, y)`` centroids of matching blobs.

    ``max_area_cell_fraction`` is expressed in cell-areas so the upper bound
    adapts to the frame resolution (avoids tripping the limit on large frames).
    """
    def _detect(frame: np.ndarray) -> List[Pixel]:
        max_area: Optional[float] = None
        if max_area_cell_fraction is not None:
            max_area = max_area_cell_fraction * (cell_size(frame, grid_size) ** 2)
        return detect_color_centroids(
            frame, hsv_lower, hsv_upper,
            min_area=min_area, max_area=max_area,
        )
    return _detect


def detect_grid_colors(
    frame: np.ndarray, grid_size: int = DEFAULT_GRID_SIZE,
) -> Dict[str, List[Cell]]:
    """Classify each cell of a uniform grid into coloured buckets.

    Returns ``{"blue": [...], "green": [...], "red": [...], "yellow": [...],
    "obstacle": [...]}`` with ``(row, col)`` values.

    - Coloured cells use the inner region (1/5 border skipped) for robustness
      against grid lines.
    - Yellow hue range deliberately avoids the orange agent (10-25).
    - Obstacles are "mostly black centre, not otherwise coloured" cells, so
      a coloured cell that happens to contain a black X is not re-labelled.
    """
    h, w = frame.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result: Dict[str, List[Cell]] = {
        "blue": [], "green": [], "red": [], "yellow": [], "obstacle": [],
    }

    for r in range(grid_size):
        for c in range(grid_size):
            y1 = r * cell_h + cell_h // 5
            y2 = (r + 1) * cell_h - cell_h // 5
            x1 = c * cell_w + cell_w // 5
            x2 = (c + 1) * cell_w - cell_w // 5
            if y2 <= y1 or x2 <= x1:
                continue

            region_hsv = hsv[y1:y2, x1:x2]
            hue = region_hsv[:, :, 0]
            sat = region_hsv[:, :, 1]
            n_pixels = (y2 - y1) * (x2 - x1)
            sat_mask = sat > 80
            threshold = n_pixels * 0.15

            blue_n = int(np.sum((hue > 100) & (hue < 130) & sat_mask))
            green_n = int(np.sum((hue > 35) & (hue < 85) & sat_mask))
            red_n = int(np.sum(((hue < 10) | (hue > 160)) & sat_mask))
            yellow_n = int(np.sum((hue > 25) & (hue < 45) & sat_mask))

            if blue_n > threshold:
                result["blue"].append((r, c))
            elif green_n > threshold:
                result["green"].append((r, c))
            elif red_n > threshold:
                result["red"].append((r, c))
            elif yellow_n > threshold:
                result["yellow"].append((r, c))

            cy1 = r * cell_h + cell_h // 3
            cy2 = (r + 1) * cell_h - cell_h // 3
            cx1 = c * cell_w + cell_w // 3
            cx2 = (c + 1) * cell_w - cell_w // 3
            centre_gray = gray[cy1:cy2, cx1:cx2]
            black_n = int(np.sum(centre_gray < 60))
            centre_total = max((cy2 - cy1) * (cx2 - cx1), 1)
            if (
                black_n > centre_total * 0.10
                and blue_n < threshold and red_n < threshold
                and green_n < threshold and yellow_n < threshold
            ):
                result["obstacle"].append((r, c))

    return result


# ---------------------------------------------------------------------------
# Graph search
# ---------------------------------------------------------------------------

def grid_bfs(
    start: Cell,
    obstacles: Set[Cell],
    grid_size: int = DEFAULT_GRID_SIZE,
) -> Dict[Cell, int]:
    """BFS distances from ``start`` to every reachable non-obstacle cell."""
    dist: Dict[Cell, int] = {start: 0}
    q: deque = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = (r + dr, c + dc)
            if (
                0 <= nb[0] < grid_size
                and 0 <= nb[1] < grid_size
                and nb not in dist
                and nb not in obstacles
            ):
                dist[nb] = dist[(r, c)] + 1
                q.append(nb)
    return dist


def optimal_cell_set(
    start: Cell,
    end: Cell,
    obstacles: Set[Cell],
    grid_size: int = DEFAULT_GRID_SIZE,
) -> Optional[Tuple[FrozenSet[Cell], Dict[Cell, int], int]]:
    """All cells on *some* shortest path from ``start`` to ``end``.

    Returns ``(optimal_cells, dist_from_start, shortest_distance)`` or
    ``None`` when ``end`` is unreachable.
    """
    dist_s = grid_bfs(start, obstacles, grid_size)
    if end not in dist_s:
        return None
    dist_e = grid_bfs(end, obstacles, grid_size)
    shortest = dist_s[end]
    optimal = frozenset(
        c for c in dist_s if c in dist_e and dist_s[c] + dist_e[c] == shortest
    )
    return optimal, dist_s, shortest


def grid_state_bfs(
    start: Cell,
    end: Cell,
    required_cells: Sequence[Cell],
    obstacles: Set[Cell],
    grid_size: int = DEFAULT_GRID_SIZE,
) -> Tuple[
    Dict[Tuple[Cell, int], int],
    Dict[Cell, int],
    int,
    Tuple[Cell, int],
    Tuple[Cell, int],
]:
    """BFS on ``(cell, visited_required_mask)`` states.

    ``end`` is only considered reachable once every ``required_cells`` entry
    has been visited, matching the "visit all blocks before reaching red"
    semantic used by G-16.
    """
    required_sorted = tuple(sorted(set(required_cells)))
    required_bits = {cell: 1 << i for i, cell in enumerate(required_sorted)}
    all_mask = (1 << len(required_sorted)) - 1

    start_mask = required_bits.get(start, 0)
    start_state = (start, start_mask)
    goal_state = (end, all_mask)

    dist: Dict[Tuple[Cell, int], int] = {start_state: 0}
    q: deque = deque([start_state])

    while q:
        cell, mask = q.popleft()
        cur = dist[(cell, mask)]
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = (cell[0] + dr, cell[1] + dc)
            if (
                not (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size)
                or nb in obstacles
            ):
                continue
            next_mask = mask | required_bits.get(nb, 0)
            if nb == end and next_mask != all_mask:
                continue
            state = (nb, next_mask)
            if state not in dist:
                dist[state] = cur + 1
                q.append(state)

    return dist, required_bits, all_mask, start_state, goal_state


def grid_state_reverse_bfs(
    end: Cell,
    required_bits: Dict[Cell, int],
    all_mask: int,
    obstacles: Set[Cell],
    grid_size: int = DEFAULT_GRID_SIZE,
) -> Dict[Tuple[Cell, int], int]:
    """Reverse BFS companion to :func:`grid_state_bfs` on the same state graph."""
    goal_state = (end, all_mask)
    dist: Dict[Tuple[Cell, int], int] = {goal_state: 0}
    q: deque = deque([goal_state])

    while q:
        cell, mask = q.popleft()
        cur = dist[(cell, mask)]
        cell_bit = required_bits.get(cell, 0)

        if cell_bit and not (mask & cell_bit):
            continue

        candidate_prev_masks = [mask]
        if cell_bit:
            candidate_prev_masks.append(mask & ~cell_bit)

        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            prev_cell = (cell[0] + dr, cell[1] + dc)
            if (
                not (0 <= prev_cell[0] < grid_size and 0 <= prev_cell[1] < grid_size)
                or prev_cell in obstacles
            ):
                continue
            prev_bit = required_bits.get(prev_cell, 0)
            for prev_mask in candidate_prev_masks:
                if prev_bit and not (prev_mask & prev_bit):
                    continue
                if prev_cell == end and prev_mask != all_mask:
                    continue
                state = (prev_cell, prev_mask)
                if state not in dist:
                    dist[state] = cur + 1
                    q.append(state)

    return dist


def longest_path_dfs(
    grid: Sequence[Sequence[int]],
    grid_size: int,
    start: Cell = (0, 0),
    end: Optional[Cell] = None,
) -> Tuple[List[Cell], int, Set[Cell]]:
    """Find the highest-cost simple path from ``start`` to ``end`` via DFS.

    Returns ``(one_optimal_path, best_cost, union_of_all_optimal_cells)``.
    The third element is the union over every path that ties for the best
    cost, so on-path discounts don't punish alternative-but-equally-good
    routes.
    """
    if end is None:
        end = (grid_size - 1, grid_size - 1)
    best_cost = [0]
    best_path: List[List[Cell]] = [[]]
    all_optimal_cells: List[Set[Cell]] = [set()]

    def dfs(r: int, c: int, visited: Set[Cell], path: List[Cell], cost: int) -> None:
        if (r, c) == end:
            if cost > best_cost[0]:
                best_cost[0] = cost
                best_path[0] = list(path)
                all_optimal_cells[0] = set(path)
            elif cost == best_cost[0]:
                all_optimal_cells[0].update(path)
            return
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in visited:
                visited.add((nr, nc))
                path.append((nr, nc))
                dfs(nr, nc, visited, path, cost + grid[nr][nc])
                path.pop()
                visited.remove((nr, nc))

    visited = {start}
    dfs(start[0], start[1], visited, [start], grid[start[0]][start[1]])
    return best_path[0], best_cost[0], all_optimal_cells[0]


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def extract_trajectory(
    frames: Sequence[np.ndarray],
    detector: Detector,
    *,
    dedup_threshold: float = 3.0,
) -> List[Pixel]:
    """Collect the first detected blob per frame, dedup'd by Manhattan distance.

    Consecutive positions closer than ``dedup_threshold`` pixels collapse to a
    single entry -- useful when the same cell is held for many frames.
    """
    path: List[Pixel] = []
    for frame in frames:
        blobs = detector(frame)
        if not blobs:
            continue
        pos = blobs[0]
        if not path or manhattan(pos, path[-1]) > dedup_threshold:
            path.append(pos)
    return path


# ---------------------------------------------------------------------------
# Scoring: proximity
# ---------------------------------------------------------------------------

def score_proximity(
    video_frames: Sequence[np.ndarray],
    reference_points: Sequence[Pixel],
    cell_px: float,
    detector: Detector,
    *,
    jitter_tol: float = 0.05,
    extra_agent_penalty: float = 0.20,
    skip_undetected: bool = False,
    max_distance_cells: float = 2.0,
    hallucination_cells: float = 1.0,
    optimal_cells: Optional[Set[Cell]] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> float:
    """Mean per-frame proximity of the best blob to any reference point.

    - The per-frame *base* score uses the blob closest to any reference
      point — that's "the agent" for scoring purposes.
    - Every *other* blob that sits more than ``hallucination_cells`` cell
      widths from the nearest reference is counted as a hallucinated agent
      and deducts ``extra_agent_penalty`` from the base score (capped at
      1.0).  Close-to-path extras (common when the detector splits a
      single agent blob at a coloured cell boundary) don't hurt.
    - ``skip_undetected=True`` skips frames where no blob is visible (used
      by G-18 where same-hue coloured cells can hide the agent); the default
      scores them as zero (G-13 / G-15 behaviour).
    - ``jitter_tol`` (in cell fractions) clamps sub-threshold distances to 0
      so pixel-level jitter doesn't cost anything.
    - When ``optimal_cells`` is provided, per-blob distance is *cell*-based:
      a blob whose cell is in the set has distance 0, otherwise the
      Manhattan cell-hop to the nearest optimal cell (× ``cell_px``).  This
      stops smooth mid-transition animation from being penalised against
      a reference set of discrete cell centres.
    """
    if not reference_points:
        return 0.0

    ref = np.asarray(reference_points)  # (N, 2)
    frame_scores: List[float] = []
    halluc_thresh = hallucination_cells * cell_px
    frame_shape = video_frames[0].shape if video_frames else None
    opt_list = list(optimal_cells) if optimal_cells else None

    def blob_dist(bx: int, by: int) -> float:
        if opt_list is not None and frame_shape is not None:
            bc = pixel_to_cell(bx, by, frame_shape, grid_size)
            if bc in optimal_cells:
                return 0.0
            return min(
                abs(bc[0] - oc[0]) + abs(bc[1] - oc[1]) for oc in opt_list
            ) * cell_px
        return float((np.abs(ref[:, 0] - bx) + np.abs(ref[:, 1] - by)).min())

    for frame in video_frames:
        blobs = detector(frame)
        if not blobs:
            if skip_undetected:
                continue
            frame_scores.append(0.0)
            continue

        dists = sorted(blob_dist(ax, ay) for ax, ay in blobs)
        best = dists[0]
        if best < jitter_tol * cell_px:
            best = 0.0
        base = max(0.0, 1.0 - best / (max_distance_cells * cell_px))

        # Count only off-path extras as hallucinations; extras near any ref
        # point are usually the detector fragmenting a single agent.
        n_hallucinated = sum(1 for d in dists[1:] if d > halluc_thresh)
        extra_pen = min(1.0, n_hallucinated * extra_agent_penalty)
        frame_scores.append(base * (1.0 - extra_pen))

    return sum(frame_scores) / len(frame_scores) if frame_scores else 0.0


# ---------------------------------------------------------------------------
# Scoring: ordered path-completion coverage
# ---------------------------------------------------------------------------

def decode_cell_hops(
    video_frames: Sequence[np.ndarray],
    reference_points: Sequence[Pixel],
    start_cell: Cell,
    end_cell: Cell,
    obstacles: Set[Cell],
    detector: Detector,
    *,
    gap_threshold: int = 2,
    grid_size: int = DEFAULT_GRID_SIZE,
    boundary_tolerance: int = 1,
) -> Tuple[int, int]:
    """Decode the per-frame best-blob trajectory into ``(actual, fill)`` hops.

    Iterates frames, picks the blob closest to any reference point per frame,
    rounds to a cell, and deduplicates consecutive repeats.  Then:

    - Contiguous hops of Manhattan ≤ ``gap_threshold`` are trusted as real
      motion and added to ``actual``.
    - Larger jumps are patched in with ``grid_bfs`` shortest-path distance
      (respecting ``obstacles``) and added to ``fill``.
    - The boundary gap before ``cell_seq[0]`` (vs ``start_cell``) and after
      ``cell_seq[-1]`` (vs ``end_cell``) is treated as ``actual`` when it
      fits within ``boundary_tolerance`` cells, otherwise as ``fill``.

    Both ``score_coverage_completion`` and ``path_length`` build on this
    decoder — one pass over the video yields both the ``actual/(actual+fill)``
    coverage ratio and the ``actual+fill`` traversal total.
    """
    if not video_frames or not reference_points:
        return 0, 0

    frame_shape = video_frames[0].shape
    ref = np.asarray(reference_points)

    cell_seq: List[Cell] = []
    for frame in video_frames:
        blobs = detector(frame)
        if not blobs:
            continue
        best_blob = min(
            blobs,
            key=lambda b: float((np.abs(ref[:, 0] - b[0]) + np.abs(ref[:, 1] - b[1])).min()),
        )
        c = pixel_to_cell(best_blob[0], best_blob[1], frame_shape, grid_size)
        if not cell_seq or c != cell_seq[-1]:
            cell_seq.append(c)

    if not cell_seq:
        return 0, 0

    actual = 0
    fill = 0

    if cell_seq[0] != start_cell:
        dist_map = grid_bfs(start_cell, obstacles, grid_size)
        d = dist_map.get(cell_seq[0], 2 * grid_size)
        if d <= boundary_tolerance:
            actual += d
        else:
            fill += d

    for i in range(1, len(cell_seq)):
        md = manhattan(cell_seq[i], cell_seq[i - 1])
        if md <= gap_threshold:
            actual += int(md)
        else:
            dist_map = grid_bfs(cell_seq[i - 1], obstacles, grid_size)
            fill += dist_map.get(cell_seq[i], int(md))

    if cell_seq[-1] != end_cell:
        dist_map = grid_bfs(cell_seq[-1], obstacles, grid_size)
        d = dist_map.get(end_cell, 2 * grid_size)
        if d <= boundary_tolerance:
            actual += d
        else:
            fill += d

    return actual, fill


def score_coverage_completion(
    video_frames: Sequence[np.ndarray],
    reference_points: Sequence[Pixel],
    start_cell: Cell,
    end_cell: Cell,
    obstacles: Set[Cell],
    cell_px: float,
    detector: Detector,
    *,
    gap_threshold: int = 2,
    grid_size: int = DEFAULT_GRID_SIZE,
    boundary_tolerance: int = 1,
) -> float:
    """Coverage via path completion: ``actual / (actual + fill)``.

    Thin wrapper over :func:`decode_cell_hops` — see its docstring for the
    decoding rules.  ``cell_px`` is accepted for signature stability but is
    unused.

    Pairs with :func:`path_length` when a caller needs both coverage and the
    raw traversal length; run :func:`decode_cell_hops` once and derive both
    to avoid redecoding the video twice.
    """
    actual, fill = decode_cell_hops(
        video_frames, reference_points, start_cell, end_cell, obstacles,
        detector,
        gap_threshold=gap_threshold,
        grid_size=grid_size,
        boundary_tolerance=boundary_tolerance,
    )
    total = actual + fill
    return actual / total if total > 0 else 1.0


def path_length(
    video_frames: Sequence[np.ndarray],
    reference_points: Sequence[Pixel],
    start_cell: Cell,
    end_cell: Cell,
    obstacles: Set[Cell],
    detector: Detector,
    *,
    gap_threshold: int = 2,
    grid_size: int = DEFAULT_GRID_SIZE,
    boundary_tolerance: int = 1,
) -> int:
    """Total agent-traversal length in cell hops, BFS-filling detection gaps.

    Same decoder as :func:`score_coverage_completion`; returns ``actual+fill``
    instead of the ratio.  Useful for a path-efficiency factor of the form
    ``min(1, optimal_len / path_length)`` which catches back-and-forth motion
    that coverage and proximity both miss (same ``actual`` bucket grows for
    forward and backward hops alike).
    """
    actual, fill = decode_cell_hops(
        video_frames, reference_points, start_cell, end_cell, obstacles,
        detector,
        gap_threshold=gap_threshold,
        grid_size=grid_size,
        boundary_tolerance=boundary_tolerance,
    )
    return actual + fill


# ---------------------------------------------------------------------------
# Scoring: discontinuity / animation quality
# ---------------------------------------------------------------------------

def discontinuity_penalty(
    video_frames: Sequence[np.ndarray],
    cell_px: float,
    single_detector: Callable[[np.ndarray], Optional[Pixel]],
    *,
    disappear_cap: float = 1.0,
    penalty_floor: float = 0.05,
    jump_cells: float = 2.0,
    cell_based: bool = False,
    grid_size: int = DEFAULT_GRID_SIZE,
    bridge_gap: int = 2,
    trim_edge_gaps: bool = False,
) -> float:
    """Animation-quality penalty in [0, 1].  0 = smooth, 1 = worst.

    **Pixel mode** (default, ``cell_based=False``) combines:
    - 50% disappear rate (capped at ``disappear_cap``);
    - 50% jump rate (transitions whose pixel displacement exceeds
      ``jump_cells`` cell widths).

    Speed variance was removed: task semantics only care about *where*
    the agent moves, not *how fast* — uneven frame pacing is an
    animation-quality question, not a correctness question.

    **Cell mode** (``cell_based=True``) combines:
    - 50% disappear rate (bridged — see below);
    - 50% cell-jump rate — pairs of consecutive detected frames whose
      cell-Manhattan exceeds what the gap between them could physically
      cover (``cell_dist > gap + 1``).

    Cell mode is the right fit for grids that render one discrete cell per
    frame: pixel displacement from cell *i* to an adjacent cell *j* is one
    full cell width, which pixel mode wrongly reads as a "fast frame" and
    inflates speed variance.

    *Bridging*: a run of up to ``bridge_gap`` consecutive undetected frames
    between two detections is forgiven (does not count towards
    ``disappear_rate``) as long as the two end-cells are physically
    reachable (``cell_dist <= gap + 1``) — a brief detector hiccup with
    the agent re-appearing at an adjacent cell is a detection artefact,
    not an animation defect.  Longer gaps still count as disappearance
    even if geometrically plausible.

    ``trim_edge_gaps`` (cell mode only): ignore leading/trailing None
    runs before first and after last detection. Useful when the agent
    shares a hue with the start/end cells, so it's invisible while
    inside them — that's a colour-aliasing artefact, not a real
    disappearance mid-motion.

    Raw penalty below ``penalty_floor`` is treated as detection jitter and
    clamped to 0.
    """
    n = len(video_frames)
    if n < 2:
        return 1.0

    positions = [single_detector(frame) for frame in video_frames]

    detected = sum(1 for p in positions if p is not None)

    if cell_based:
        frame_shape = video_frames[0].shape
        cells: List[Optional[Cell]] = [
            pixel_to_cell(pos[0], pos[1], frame_shape, grid_size) if pos is not None else None
            for pos in positions
        ]
        detected_idx = [i for i, c in enumerate(cells) if c is not None]
        if not detected_idx:
            return 1.0

        bridged_frames = 0
        jump_count = 0
        transitions = 0
        for i, j in zip(detected_idx, detected_idx[1:]):
            gap = j - i - 1  # None frames strictly between i and j
            ci, cj = cells[i], cells[j]
            assert ci is not None and cj is not None  # narrows for the type checker
            cell_dist = manhattan(ci, cj)
            transitions += 1
            if cell_dist > gap + 1:
                jump_count += 1  # physically impossible motion → true jump
            elif 0 < gap <= bridge_gap:
                bridged_frames += gap  # short plausible gap → detector hiccup

        if trim_edge_gaps:
            # Count only None runs between first and last detection;
            # leading/trailing None are usually same-hue occlusion while
            # the agent sits inside the start/end cell.
            first, last = detected_idx[0], detected_idx[-1]
            effective_n = last - first + 1
            none_frames = effective_n - len(detected_idx)
        else:
            effective_n = n
            none_frames = n - len(detected_idx)
        true_disappear = max(0, none_frames - bridged_frames)
        disappear_rate = min(disappear_cap, true_disappear / max(effective_n, 1))
        cell_jump_rate = jump_count / max(transitions, 1)

        penalty = 0.5 * disappear_rate + 0.5 * cell_jump_rate
        if penalty < penalty_floor:
            penalty = 0.0
        return min(1.0, penalty)

    disappear_rate = min(disappear_cap, 1.0 - detected / n)

    jump_count = 0
    transitions = 0
    prev: Optional[Pixel] = None
    for pos in positions:
        if pos is None:
            prev = None
            continue
        if prev is not None:
            d = manhattan(pos, prev)
            transitions += 1
            if d > jump_cells * cell_px:
                jump_count += 1
        prev = pos

    jump_rate = jump_count / max(transitions, 1)

    penalty = 0.5 * disappear_rate + 0.5 * jump_rate
    if penalty < penalty_floor:
        penalty = 0.0
    return min(1.0, penalty)


# ---------------------------------------------------------------------------
# Interleave scoring
# ---------------------------------------------------------------------------

def score_interleave_path_ssim(
    last_frame: np.ndarray,
    input_frame: Optional[np.ndarray],
    gt_final: np.ndarray,
    *,
    max_penalty: float = 0.70,
    diff_threshold: int = 15,
) -> float:
    """Interleave scoring for drawn-path images.

    - Base: SSIM between ``(pred - input)`` and ``(gt_final - input)`` when
      ``input_frame`` is available, else SSIM(pred, gt_final) directly.
    - Penalty: fragmentation of the drawn path.  A single connected path
      scores 0 penalty; many pieces approach 1.
    """
    if last_frame.shape != gt_final.shape:
        gt_final = cv2.resize(gt_final, (last_frame.shape[1], last_frame.shape[0]))

    input_resized: Optional[np.ndarray] = input_frame
    if input_resized is not None and input_resized.shape != last_frame.shape:
        input_resized = cv2.resize(
            input_resized, (last_frame.shape[1], last_frame.shape[0]),
        )

    if input_resized is not None:
        pred_path = cv2.absdiff(last_frame, input_resized)
        gt_path = cv2.absdiff(gt_final, input_resized)
        base = compute_ssim(pred_path, gt_path)
    else:
        base = compute_ssim(last_frame, gt_final)

    penalty = 0.0
    if input_resized is not None:
        diff_gray = cv2.cvtColor(
            cv2.absdiff(last_frame, input_resized), cv2.COLOR_BGR2GRAY,
        )
        _, mask = cv2.threshold(diff_gray, diff_threshold, 255, cv2.THRESH_BINARY)
        n_labels, _ = cv2.connectedComponents(mask)
        n_components = max(n_labels - 1, 0)
        penalty = 1.0 - 1.0 / max(n_components, 1)

    return base * (1.0 - max_penalty * penalty)


def cells_from_pred_diff(
    pred_frames: Sequence[np.ndarray],
    input_frame: np.ndarray,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    sat_threshold: int = 60,
    diff_threshold: int = 25,
    inner_margin: float = 0.2,
    inner_frac: float = 0.05,
) -> Set[Cell]:
    """Extract grid cells whose interior has newly-drawn saturated content.

    For each predicted frame, compute a saturation-gated diff vs ``input_frame``
    and mark a cell as "drawn" if >= ``inner_frac`` of its inner core
    (the centre ``(1-2*inner_margin)^2`` fraction) has positive diff mask.

    The saturation gate filters grayscale lighting noise; the inner-core gate
    filters edge-bleed from outline rendering. Both are the same patterns
    used successfully in the O-39 redesign.
    """
    counts = cell_draw_counts(
        pred_frames, input_frame,
        grid_size=grid_size,
        sat_threshold=sat_threshold,
        diff_threshold=diff_threshold,
        inner_margin=inner_margin,
        inner_frac=inner_frac,
    )
    return set(counts.keys())


def cell_draw_counts(
    pred_frames: Sequence[np.ndarray],
    input_frame: np.ndarray,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    sat_threshold: int = 60,
    diff_threshold: int = 25,
    inner_margin: float = 0.2,
    inner_frac: float = 0.05,
) -> Dict[Cell, int]:
    """Per-cell pixel count of newly-drawn saturated content, inner-core only.

    Same detection as :func:`cells_from_pred_diff` but returns the pixel count
    instead of discarding it. Callers that care about "how much" was drawn
    (e.g. pixel-weighted proximity) can sum these.
    """
    if input_frame is None or not pred_frames:
        return {}
    H, W = input_frame.shape[:2]
    cell_h, cell_w = H // grid_size, W // grid_size
    if cell_h == 0 or cell_w == 0:
        return {}
    mh, mw = int(cell_h * inner_margin), int(cell_w * inner_margin)
    inner_h = max(1, cell_h - 2 * mh)
    inner_w = max(1, cell_w - 2 * mw)
    inner_area = inner_h * inner_w
    min_pixels = max(1, int(inner_frac * inner_area))

    counts: Dict[Cell, int] = {}
    for frame in pred_frames:
        if frame is None:
            continue
        f = frame
        if f.shape[:2] != (H, W):
            f = cv2.resize(f, (W, H))
        diff = cv2.absdiff(f, input_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(gray, diff_threshold, 255, cv2.THRESH_BINARY)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        _, sat_mask = cv2.threshold(sat, sat_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(diff_mask, sat_mask)
        for r in range(grid_size):
            y0 = r * cell_h + mh
            y1 = y0 + inner_h
            for c in range(grid_size):
                x0 = c * cell_w + mw
                x1 = x0 + inner_w
                n = int(cv2.countNonZero(mask[y0:y1, x0:x1]))
                if n >= min_pixels:
                    prev = counts.get((r, c), 0)
                    if n > prev:
                        counts[(r, c)] = n
    return counts


def pred_diff_mask(
    pred_frames: Sequence[np.ndarray],
    input_frame: np.ndarray,
    *,
    sat_threshold: int = 60,
    diff_threshold: int = 25,
) -> Optional[np.ndarray]:
    """Union saturated-diff mask across ``pred_frames`` vs ``input_frame``.

    Full-frame (not inner-core-gated) so the connected-component structure
    can straddle cell boundaries.  Same ``sat_threshold`` / ``diff_threshold``
    defaults as :func:`cell_draw_counts` for consistency.  Returns ``None``
    when inputs are missing so callers can skip the component check.
    """
    if input_frame is None or not pred_frames:
        return None
    H, W = input_frame.shape[:2]
    acc: Optional[np.ndarray] = None
    for frame in pred_frames:
        if frame is None:
            continue
        f = frame
        if f.shape[:2] != (H, W):
            f = cv2.resize(f, (W, H))
        diff = cv2.absdiff(f, input_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(gray, diff_threshold, 255, cv2.THRESH_BINARY)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        _, sat_mask = cv2.threshold(sat, sat_threshold, 255, cv2.THRESH_BINARY)
        m = cv2.bitwise_and(diff_mask, sat_mask)
        acc = m if acc is None else cv2.bitwise_or(acc, m)
    return acc


def skeletonize_mask(mask: Optional[np.ndarray]) -> np.ndarray:
    """Return a one-pixel-wide skeleton for a binary drawn-line mask."""
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask

    img = ((gray > 0).astype(np.uint8)) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(img) > 0:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        residue = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, residue)
        img = eroded

    return skel


def line_length_from_mask(mask: Optional[np.ndarray]) -> float:
    """Approximate drawn line length in pixels from a binary mask.

    The mask is skeletonized first, so a thick stroke and a thin stroke with
    the same centreline have roughly the same measured length.  The returned
    value is a graph length over neighbouring skeleton pixels, with diagonal
    edges weighted by sqrt(2).
    """
    skel = skeletonize_mask(mask)
    if skel.size == 0:
        return 0.0

    b = skel > 0
    n_pixels = int(np.sum(b))
    if n_pixels == 0:
        return 0.0

    horizontal = int(np.sum(b[:, :-1] & b[:, 1:]))
    vertical = int(np.sum(b[:-1, :] & b[1:, :]))
    diag_down = int(np.sum(b[:-1, :-1] & b[1:, 1:]))
    diag_up = int(np.sum(b[1:, :-1] & b[:-1, 1:]))
    edge_length = (
        float(horizontal + vertical)
        + float(diag_down + diag_up) * float(np.sqrt(2.0))
    )
    if edge_length == 0.0:
        return float(n_pixels)

    n_labels, _ = cv2.connectedComponents(b.astype(np.uint8))
    n_components = max(n_labels - 1, 0)
    return edge_length + float(n_components)


def walk_line_length(
    walk: Optional[Sequence[Cell]],
    frame_shape: Tuple[int, ...],
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> float:
    """Length in pixels of the simulated walk through grid-cell centres."""
    if walk is None or len(walk) < 2:
        return 0.0

    total = 0.0
    prev = cell_center_px(walk[0], frame_shape, grid_size)
    for cell in walk[1:]:
        cur = cell_center_px(cell, frame_shape, grid_size)
        total += float(np.hypot(cur[0] - prev[0], cur[1] - prev[1]))
        prev = cur
    return total


def _cell_bounds(
    cell: Cell,
    frame_shape: Tuple[int, ...],
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    expand: int = 0,
) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    r, c = cell
    r0 = max(0, r - expand)
    r1 = min(grid_size, r + expand + 1)
    c0 = max(0, c - expand)
    c1 = min(grid_size, c + expand + 1)
    return (
        int(r0 * h // grid_size),
        int(r1 * h // grid_size),
        int(c0 * w // grid_size),
        int(c1 * w // grid_size),
    )


def _skeleton_candidates_for_cell(
    skel: np.ndarray,
    cell: Cell,
    frame_shape: Tuple[int, ...],
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    max_nearest: int = 12,
) -> List[Tuple[int, int]]:
    """Skeleton pixels that can represent a route landmark cell."""
    if skel.size == 0:
        return []

    cx, cy = cell_center_px(cell, frame_shape, grid_size)

    def _nearest_from_points(
        ys: np.ndarray,
        xs: np.ndarray,
        *,
        y_offset: int = 0,
        x_offset: int = 0,
    ) -> List[Tuple[int, int]]:
        if ys.size == 0:
            return []
        abs_ys = ys + y_offset
        abs_xs = xs + x_offset
        dist2 = (abs_xs - cx) ** 2 + (abs_ys - cy) ** 2
        n = min(max_nearest, int(dist2.size))
        idx = np.argpartition(dist2, n - 1)[:n]
        idx = idx[np.argsort(dist2[idx])]
        return [(int(abs_ys[i]), int(abs_xs[i])) for i in idx]

    for expand in (0, 1):
        y0, y1, x0, x1 = _cell_bounds(
            cell, frame_shape, grid_size=grid_size, expand=expand,
        )
        ys, xs = np.where(skel[y0:y1, x0:x1] > 0)
        if ys.size:
            return _nearest_from_points(ys, xs, y_offset=y0, x_offset=x0)

    ys_all, xs_all = np.where(skel > 0)
    if ys_all.size == 0:
        return []

    return _nearest_from_points(ys_all, xs_all)


def _shortest_skeleton_path_length(
    skel: np.ndarray,
    src: Sequence[Tuple[int, int]],
    dst: Sequence[Tuple[int, int]],
    *,
    src_weights: Optional[Dict[Tuple[int, int], float]] = None,
    dst_weights: Optional[Dict[Tuple[int, int], float]] = None,
) -> Optional[float]:
    """Shortest 8-connected path length on a skeleton image."""
    if skel.size == 0 or not src or not dst:
        return None

    h, w = skel.shape[:2]
    src_set = {(y, x) for y, x in src if 0 <= y < h and 0 <= x < w and skel[y, x] > 0}
    dst_set = {(y, x) for y, x in dst if 0 <= y < h and 0 <= x < w and skel[y, x] > 0}
    if not src_set or not dst_set:
        return None
    if src_set & dst_set:
        return min(
            float((src_weights or {}).get(p, 0.0))
            + float((dst_weights or {}).get(p, 0.0))
            for p in src_set & dst_set
        )

    dist = np.full((h, w), np.inf, dtype=np.float32)
    heap: List[Tuple[float, int, int]] = []
    for y, x in src_set:
        start_cost = float((src_weights or {}).get((y, x), 0.0))
        dist[y, x] = start_cost
        heappush(heap, (start_cost, y, x))

    neighbours = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, float(np.sqrt(2.0))), (-1, 1, float(np.sqrt(2.0))),
        (1, -1, float(np.sqrt(2.0))), (1, 1, float(np.sqrt(2.0))),
    )

    while heap:
        cur, y, x = heappop(heap)
        if cur > float(dist[y, x]):
            continue
        if (y, x) in dst_set:
            return float(cur + float((dst_weights or {}).get((y, x), 0.0)))
        for dy, dx, step in neighbours:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w) or skel[ny, nx] == 0:
                continue
            nd = cur + step
            if nd < float(dist[ny, nx]):
                dist[ny, nx] = nd
                heappush(heap, (nd, ny, nx))

    return None


def _candidate_center_weights(
    candidates: Sequence[Tuple[int, int]],
    cell: Cell,
    frame_shape: Tuple[int, ...],
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> Dict[Tuple[int, int], float]:
    cx, cy = cell_center_px(cell, frame_shape, grid_size)
    return {
        (y, x): float(np.hypot(float(x - cx), float(y - cy)))
        for y, x in candidates
    }


def used_line_length_from_mask(
    mask: Optional[np.ndarray],
    route_cells: Sequence[Cell],
    frame_shape: Tuple[int, ...],
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> Optional[float]:
    """Shortest drawn-line length needed to follow ``route_cells`` in order.

    Both this and :func:`line_length_from_mask` operate on the same skeleton,
    so their ratio is a real line-length ratio rather than a mix of pixel
    geometry and grid-cell counts.
    """
    if mask is None or len(route_cells) < 2:
        return None

    skel = skeletonize_mask(mask)
    if skel.size == 0 or cv2.countNonZero(skel) == 0:
        return None

    total = 0.0
    for a, b in zip(route_cells, route_cells[1:]):
        src = _skeleton_candidates_for_cell(
            skel, a, frame_shape, grid_size=grid_size,
        )
        dst = _skeleton_candidates_for_cell(
            skel, b, frame_shape, grid_size=grid_size,
        )
        seg = _shortest_skeleton_path_length(
            skel, src, dst,
            src_weights=_candidate_center_weights(
                src, a, frame_shape, grid_size=grid_size,
            ),
            dst_weights=_candidate_center_weights(
                dst, b, frame_shape, grid_size=grid_size,
            ),
        )
        if seg is None:
            return None
        total += seg
    return total


def cell_pixel_components(
    mask: Optional[np.ndarray],
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> Dict[Cell, FrozenSet[int]]:
    """Per-cell set of connected-component labels from a binary ``mask``.

    Returns ``{cell: {label, ...}}`` — every component that has at least one
    pixel inside each cell.  Used to enforce pixel-level line connectivity
    in :func:`simulate_walk_through_drawn`: cells A and B may be joined by
    a walk edge only if some component exists in both (and, more strictly,
    the BFS keeps a single active component throughout the walk).

    Empty/absent mask → empty dict; callers can skip the component check.
    """
    if mask is None or mask.size == 0:
        return {}
    H, W = mask.shape[:2]
    n_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    if n_labels <= 1:
        return {}
    ys, xs = np.where(labels > 0)
    if ys.size == 0:
        return {}
    cell_rows = np.clip(ys * grid_size // H, 0, grid_size - 1)
    cell_cols = np.clip(xs * grid_size // W, 0, grid_size - 1)
    labs = labels[ys, xs]
    stacked = np.stack([cell_rows, cell_cols, labs], axis=1)
    unique_pairs = np.unique(stacked, axis=0)

    collected: Dict[Cell, Set[int]] = {}
    for r, c, lab in unique_pairs:
        collected.setdefault((int(r), int(c)), set()).add(int(lab))
    return {cell: frozenset(labs) for cell, labs in collected.items()}


def _connected_component_containing(
    cells: Set[Cell], seed: Cell,
) -> Set[Cell]:
    """4-connected component of ``seed`` inside ``cells`` (empty if seed ∉ cells)."""
    if seed not in cells:
        return set()
    visited: Set[Cell] = {seed}
    q: deque = deque([seed])
    while q:
        r, c = q.popleft()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = (r + dr, c + dc)
            if nb in cells and nb not in visited:
                visited.add(nb)
                q.append(nb)
    return visited


def all_cells_connected(
    cells: Set[Cell], required: Sequence[Cell],
) -> bool:
    """True iff every cell in ``required`` is in the same 4-connected component.

    Used by interleave scoring to reject drawings that hit the required cells
    but leave them disconnected (e.g. painting only start + waypoints + end
    without any connecting path).
    """
    if not required:
        return True
    if any(r not in cells for r in required):
        return False
    comp = _connected_component_containing(cells, required[0])
    return all(r in comp for r in required)


def score_interleave_cells(
    drawn_cells: Set[Cell],
    optimal_cells: Set[Cell],
    *,
    required_cells: Optional[Sequence[Cell]] = None,
    wall_cells: Optional[Set[Cell]] = None,
    path_length: Optional[int] = None,
    draw_counts: Optional[Dict[Cell, int]] = None,
    connectivity_cells: Optional[Sequence[Cell]] = None,
    missing_base: float = 0.5,
    wall_base: float = 0.5,
) -> Tuple[float, Dict[str, object]]:
    """Interleave cell-based scoring aligned with the video maze family.

    Two formulas are supported:

    - **Legacy (coverage-based)** when ``connectivity_cells`` is None:
      ``score = proximity × coverage × length_factor × missing_base^missed × wall_base^walls``
    - **Connectivity-based** when ``connectivity_cells`` is provided:
      ``score = proximity × connected × length_factor × missing_base^missed × wall_base^walls``
      where ``connected`` = 1.0 iff every cell in ``connectivity_cells`` is
      in the same 4-connected component of ``drawn_cells``.

    Prefer connectivity for multi-segment tasks (G-13) where overlapping legs
    make coverage under-count GT drawings. GT tours are always one connected
    orange line; "paint only the endpoints" cheating gets filtered out.

    - proximity  — pixel-weighted on_path / total when ``draw_counts`` is
      given, else cell-count ``|drawn ∩ optimal| / |drawn|``.
    - coverage   — ``|drawn ∩ optimal| / path_length`` capped at 1.0.
    - length_factor — ``min(1, path_length / num_drawn)`` when
      ``path_length`` is known; else 1.0.  Penalises drawing more cells
      than one shortest path needs (painting the whole shortest-path DAG,
      or scribbling everywhere), symmetric with the video-side
      ``length_factor`` on total hop count.  Floor prevents double-counting
      with proximity for extreme over-drawing.
    - num_missed = required cells not in drawn (only if ``required_cells``)
    - num_wall_hits = |drawn ∩ walls|                 (only if ``wall_cells``)

    Returns ``(score, details)``; details keys are always present so callers
    can populate ``_last_task_details`` directly.
    """
    drawn_set = set(drawn_cells or ())
    opt_set = set(optimal_cells or ())
    on_path = drawn_set & opt_set
    off_path_cells_set = drawn_set - opt_set
    total_drawn = len(drawn_set)
    total_opt = len(opt_set)
    if draw_counts:
        total_pixels = sum(draw_counts.values())
        on_path_pixels = sum(
            n for c, n in draw_counts.items() if c in opt_set
        )
        off_path_pixels = total_pixels - on_path_pixels
        proximity = on_path_pixels / total_pixels if total_pixels else 0.0
    else:
        total_pixels = None
        on_path_pixels = None
        off_path_pixels = None
        proximity = len(on_path) / total_drawn if total_drawn else 0.0
    denom = path_length if path_length is not None and path_length > 0 else total_opt
    coverage = min(1.0, len(on_path) / denom) if denom else 0.0

    if path_length is not None and path_length > 0 and total_drawn > 0:
        length_factor = min(1.0, path_length / total_drawn)
    else:
        length_factor = 1.0

    missed: List[Cell] = []
    if required_cells:
        for c in required_cells:
            if c not in drawn_set:
                missed.append(c)
    missed_factor = missing_base ** len(missed)

    wall_hits: List[Cell] = []
    if wall_cells:
        wall_hits = sorted(drawn_set & set(wall_cells))
    wall_factor = wall_base ** len(wall_hits)

    if connectivity_cells is not None:
        connected = 1.0 if all_cells_connected(drawn_set, connectivity_cells) else 0.0
        score = proximity * connected * length_factor * missed_factor * wall_factor
        formula = "proximity × connected × length_factor × 0.5^missed × 0.5^wall_hits"
    else:
        connected = None
        score = proximity * coverage * length_factor * missed_factor * wall_factor
        formula = "proximity × coverage × length_factor × 0.5^missed × 0.5^wall_hits"
    score = float(max(0.0, min(1.0, score)))

    details: Dict[str, object] = {
        "num_drawn_cells": total_drawn,
        "num_optimal_cells": total_opt,
        "num_on_path": len(on_path),
        "num_off_path": len(off_path_cells_set),
        "off_path_cells": sorted([list(c) for c in off_path_cells_set])[:60],
        "total_pixels": total_pixels,
        "on_path_pixels": on_path_pixels,
        "off_path_pixels": off_path_pixels,
        "path_length_denom": denom,
        "proximity": round(proximity, 4),
        "coverage": round(coverage, 4),
        "length_factor": round(length_factor, 4),
        "num_missed_required": len(missed),
        "num_wall_hit_cells": len(wall_hits),
        "missed_required": [list(c) for c in missed],
        "wall_hit_cells": [list(c) for c in wall_hits],
        "missed_factor": round(missed_factor, 6),
        "wall_factor": round(wall_factor, 6),
        "final_score": round(score, 4),
        "score_breakdown": {
            "formula": formula,
            "proximity": round(proximity, 4),
            **({"connected": connected} if connected is not None else {"coverage": round(coverage, 4)}),
            "length_factor": round(length_factor, 4),
            "missed_factor": round(missed_factor, 6),
            "wall_factor": round(wall_factor, 6),
            "final": round(score, 4),
        },
    }
    if connected is not None:
        details["connected"] = connected
    return score, details


# ---------------------------------------------------------------------------
# Interleave scoring — walk simulation through drawn cells
# ---------------------------------------------------------------------------


def simulate_walk_through_drawn(
    drawn: Set[Cell],
    start: Cell,
    end: Cell,
    *,
    waypoints: Optional[Sequence[Cell]] = None,
    required: Optional[Set[Cell]] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    allow_cells: Optional[Set[Cell]] = None,
    cell_components: Optional[Dict[Cell, FrozenSet[int]]] = None,
) -> Optional[List[Cell]]:
    """Simulate an agent walking start → end using only ``drawn`` cells.

    The agent can only step on cells the model actually drew (plus start, end,
    any waypoints/required cells, and ``allow_cells`` — typically landmarks
    visible in the input that the model doesn't need to re-draw).

    Modes:
      * ``waypoints`` given → ordered sequential BFS: start → wp1 → ... → end.
        Each segment must exist in the drawn-cell subgraph.
      * ``required`` given (unordered) → state-space BFS over
        ``(cell, visited_mask)``, returns a shortest walk that reaches ``end``
        with every required cell visited.
      * neither → plain shortest BFS start → end through drawn cells.

    When ``cell_components`` is given (from :func:`cell_pixel_components`), the
    BFS is component-aware: the walk carries an *active component* label and
    can only transition through cells that contain that component's pixels.
    Landmarks (start/end/waypoints/``allow_cells``) that have no drawn pixels
    inherit a sentinel "any" component so the walk can still enter them, but
    cannot bridge across disconnected pixel components in a drawn cell —
    exactly the G-13 00003 failure mode where two visually-disjoint arms of
    a broken path share a cell but not a pixel component.

    Returns the cell sequence (consecutive cells are 4-adjacent) or ``None``
    if no valid walk exists. This strict-connectivity test is the core of
    the interleave scoring: a scribble that covers the right cells but
    doesn't form a walk from start to end gets None → score 0.
    """
    walkable: Set[Cell] = set(drawn or ())
    walkable.add(start)
    walkable.add(end)
    if waypoints:
        walkable.update(waypoints)
    if required:
        walkable.update(required)
    if allow_cells:
        walkable.update(allow_cells)

    use_comp = cell_components is not None
    _ANY = -1

    def _comps_at(cell: Cell) -> FrozenSet[int]:
        if not use_comp:
            return frozenset({_ANY})
        labs = cell_components.get(cell)
        if labs:
            return labs
        return frozenset({_ANY})

    def _next_comp(
        current: int, nx_comps: FrozenSet[int],
    ) -> Optional[int]:
        """Active component after stepping into a cell with ``nx_comps``.

        - If the current walk is still on the ``_ANY`` sentinel (no
          concrete pixel component chosen yet), pick any label from
          ``nx_comps``; an all-``_ANY`` walk that never touches drawn
          pixels stays on ``_ANY``.
        - Else the step is valid iff the new cell contains the current
          component (or the sentinel).  Returns the updated component or
          ``None`` if the step is invalid.
        """
        if current == _ANY:
            concrete = [c for c in nx_comps if c != _ANY]
            if concrete:
                return min(concrete)
            return _ANY
        if current in nx_comps or _ANY in nx_comps:
            return current
        return None

    def _bfs_segment(src: Cell, dst: Cell) -> Optional[List[Cell]]:
        if src == dst:
            return [src]
        src_comps = _comps_at(src)
        # Seed one BFS per possible starting component.
        parents: Dict[Tuple[Cell, int], Optional[Tuple[Cell, int]]] = {}
        q: deque = deque()
        for c0 in src_comps:
            state = (src, c0)
            parents[state] = None
            q.append(state)
        while q:
            cur_state = q.popleft()
            cur, active = cur_state
            if cur == dst:
                out: List[Cell] = []
                s: Optional[Tuple[Cell, int]] = cur_state
                while s is not None:
                    out.append(s[0])
                    s = parents[s]
                return list(reversed(out))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cur[0] + dr, cur[1] + dc
                if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                    continue
                nx = (nr, nc)
                if nx not in walkable:
                    continue
                nx_active = _next_comp(active, _comps_at(nx))
                if nx_active is None:
                    continue
                ns = (nx, nx_active)
                if ns in parents:
                    continue
                parents[ns] = cur_state
                q.append(ns)
        return None

    if waypoints:
        walk: List[Cell] = [start]
        for wp in list(waypoints) + [end]:
            seg = _bfs_segment(walk[-1], wp)
            if seg is None:
                return None
            walk.extend(seg[1:])
        return walk

    if required:
        req_list = list(required)
        req_idx = {c: i for i, c in enumerate(req_list)}
        all_mask = (1 << len(req_list)) - 1

        def _mask_at(cell: Cell, prev: int) -> int:
            return prev | (1 << req_idx[cell]) if cell in req_idx else prev

        start_mask = _mask_at(start, 0)
        goal_mask_end = _mask_at(end, all_mask)
        parents_s: Dict[
            Tuple[Cell, int, int], Optional[Tuple[Cell, int, int]]
        ] = {}
        qs: deque = deque()
        for c0 in _comps_at(start):
            s0 = (start, start_mask, c0)
            parents_s[s0] = None
            qs.append(s0)
        goal_state: Optional[Tuple[Cell, int, int]] = None
        while qs:
            state = qs.popleft()
            cell, mask, active = state
            if cell == end and mask == goal_mask_end:
                goal_state = state
                break
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cell[0] + dr, cell[1] + dc
                if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                    continue
                nx = (nr, nc)
                if nx not in walkable:
                    continue
                nx_active = _next_comp(active, _comps_at(nx))
                if nx_active is None:
                    continue
                nm = _mask_at(nx, mask)
                ns = (nx, nm, nx_active)
                if ns in parents_s:
                    continue
                parents_s[ns] = state
                qs.append(ns)
        if goal_state is None:
            return None
        out: List[Cell] = []
        s2: Optional[Tuple[Cell, int, int]] = goal_state
        while s2 is not None:
            out.append(s2[0])
            s2 = parents_s[s2]
        return list(reversed(out))

    return _bfs_segment(start, end)


def score_interleave_walk(
    walk: Optional[Sequence[Cell]],
    drawn: Set[Cell],
    optimal_cells: Set[Cell],
    *,
    required_cells: Optional[Sequence[Cell]] = None,
    wall_cells: Optional[Set[Cell]] = None,
    path_length: Optional[int] = None,
    draw_counts: Optional[Dict[Cell, int]] = None,
    used_line_length: Optional[float] = None,
    drawn_line_length: Optional[float] = None,
    missing_base: float = 0.5,
    wall_base: float = 0.5,
) -> Tuple[float, Dict[str, object]]:
    """Score drawn cells when gated on a valid walk from start to end.

    Walk existence is a binary gate (the "no route, no credit" rule):

    - ``walk is None`` (drawn cells don't contain a path from start through
      required/waypoint nodes to end) → score = 0.
    - Otherwise: ``score = proximity × coverage × length_factor × 0.5^missed × 0.5^wall_hits``,
      computed over the **drawn cells themselves** (pixel-weighted when
      ``draw_counts`` is given). A model that paints the path correctly
      plus lots of scribble gets proximity<1 from the off-path pixel mass;
      a model that paints only the endpoints without connecting them fails
      the gate and scores 0; and a model that paints the entire shortest-path
      DAG (e.g. G-18 sample 00000) is caught by ``length_factor``.

    - proximity  = pixel-weighted on_path / total if draw_counts else
      |drawn ∩ optimal| / |drawn|
    - coverage   = min(1, |drawn ∩ optimal| / path_length)
    - length_factor = ``min(1, used_line_length / drawn_line_length)`` when
      both line lengths are provided. This is preferred for drawn-line tasks
      because it penalises extra stroke length even inside already-counted
      cells. If line lengths are unavailable, falls back to the legacy
      ``min(1, path_length / num_drawn)`` cell-count ratio.
    - missed     = required cells not in drawn               (0.5 per miss)
    - wall_hits  = drawn cells that are walls                (0.5 per hit)
    """
    drawn_set = set(drawn or ())
    opt_set = set(optimal_cells or ())
    line_lengths_available = (
        drawn_line_length is not None
        and drawn_line_length > 0
        and used_line_length is not None
    )
    length_factor_unit = "line_length" if line_lengths_available else "cell_count"

    def _round_optional(value: Optional[float]) -> Optional[float]:
        return None if value is None else round(float(value), 4)

    details: Dict[str, object] = {
        "walk_length": 0,
        "walk": [],
        "num_drawn_cells": len(drawn_set),
        "num_optimal_cells": len(opt_set),
        "num_on_path": 0,
        "num_off_path": len(drawn_set),
        "off_path_cells": sorted(list(c) for c in (drawn_set - opt_set))[:60],
        "total_pixels": (sum(draw_counts.values()) if draw_counts else None),
        "on_path_pixels": 0 if draw_counts else None,
        "off_path_pixels": (sum(draw_counts.values()) if draw_counts else None),
        "proximity": 0.0,
        "coverage": 0.0,
        "length_factor": 0.0,
        "length_factor_unit": length_factor_unit,
        "used_line_length": _round_optional(used_line_length),
        "drawn_line_length": _round_optional(drawn_line_length),
        "num_missed_required": 0,
        "num_wall_hit_cells": 0,
        "missed_required": [],
        "wall_hit_cells": [],
        "missed_factor": 0.0,
        "wall_factor": 0.0,
        "walk_reachable": bool(walk),
        "final_score": 0.0,
        "score_breakdown": {
            "formula": "proximity × coverage × length_factor × 0.5^missed × 0.5^wall_hits",
            "length_factor_unit": length_factor_unit,
        },
    }
    # An empty walk ([]) is as unreachable as None — guard both so a caller
    # that coerces a failed walk to [] can't slip through to a perfect score.
    if not walk:
        return 0.0, details

    walk_seq = list(walk)
    on_path = drawn_set & opt_set
    off_path_cells = drawn_set - opt_set

    if draw_counts:
        total_pixels = sum(draw_counts.values())
        on_path_pixels = sum(n for c, n in draw_counts.items() if c in opt_set)
        off_path_pixels = total_pixels - on_path_pixels
        proximity = on_path_pixels / total_pixels if total_pixels else 0.0
    else:
        total_pixels = None
        on_path_pixels = None
        off_path_pixels = None
        proximity = len(on_path) / len(drawn_set) if drawn_set else 0.0

    denom = path_length if path_length is not None and path_length > 0 else len(opt_set)
    coverage = min(1.0, len(on_path) / denom) if denom else 0.0

    total_drawn = len(drawn_set)
    if line_lengths_available:
        length_factor = min(
            1.0,
            max(0.0, float(used_line_length)) / float(drawn_line_length),
        )
    elif path_length is not None and path_length > 0 and total_drawn > 0:
        length_factor = min(1.0, path_length / total_drawn)
    else:
        length_factor = 1.0

    missed: List[Cell] = []
    if required_cells:
        for c in required_cells:
            if c not in drawn_set:
                missed.append(c)
    missed_factor = missing_base ** len(missed)

    wall_hits: List[Cell] = []
    if wall_cells:
        wall_hits = sorted(drawn_set & set(wall_cells))
    wall_factor = wall_base ** len(wall_hits)

    score = proximity * coverage * length_factor * missed_factor * wall_factor
    score = float(max(0.0, min(1.0, score)))

    details.update({
        "walk_length": len(walk_seq),
        "walk": [list(c) for c in walk_seq[:60]],
        "num_on_path": len(on_path),
        "num_off_path": len(off_path_cells),
        "off_path_cells": sorted([list(c) for c in off_path_cells])[:60],
        "total_pixels": total_pixels,
        "on_path_pixels": on_path_pixels,
        "off_path_pixels": off_path_pixels,
        "path_length_denom": denom,
        "proximity": round(proximity, 4),
        "coverage": round(coverage, 4),
        "length_factor": round(length_factor, 4),
        "length_factor_unit": length_factor_unit,
        "used_line_length": _round_optional(used_line_length),
        "drawn_line_length": _round_optional(drawn_line_length),
        "num_missed_required": len(missed),
        "num_wall_hit_cells": len(wall_hits),
        "missed_required": [list(c) for c in missed],
        "wall_hit_cells": [list(c) for c in wall_hits],
        "missed_factor": round(missed_factor, 6),
        "wall_factor": round(wall_factor, 6),
        "final_score": round(score, 4),
        "score_breakdown": {
            "formula": "proximity × coverage × length_factor × 0.5^missed × 0.5^wall_hits",
            "proximity": round(proximity, 4),
            "coverage": round(coverage, 4),
            "length_factor": round(length_factor, 4),
            "length_factor_unit": length_factor_unit,
            "missed_factor": round(missed_factor, 6),
            "wall_factor": round(wall_factor, 6),
            "final": round(score, 4),
        },
    })
    return score, details


# ---------------------------------------------------------------------------
# Background preservation
# ---------------------------------------------------------------------------

def background_preservation_image(
    pred: Optional[np.ndarray],
    ref: Optional[np.ndarray],
    *,
    exclude_mask: Optional[np.ndarray] = None,
    floor_fraction: float = 0.05,
    noise_floor: float = 5.0,
    saturation_threshold: float = 0.999,
    bad_pixel_threshold: Optional[float] = None,
    bad_pixel_tolerance: float = 0.10,
) -> float:
    """Mean pixel similarity of ``pred`` vs ``ref`` outside ``exclude_mask``.

    Similarity = ``1 - |pred - ref| / 255`` averaged over the valid region.
    ``exclude_mask`` is the region the model legitimately changed (drawn path,
    piece region, etc.) and should not be counted against background fidelity.

    ``noise_floor`` (default 5) treats per-pixel absolute differences below
    this as zero — lets GT video survive codec artifacts at 1.0 while still
    catching real background drift (tens of pixel values per channel).

    ``saturation_threshold`` (default 0.999) snaps scores at or above this
    value to exactly 1.0. The sub-0.001 residual on GT-as-pred after
    `noise_floor` clipping is codec / anti-alias / motion-blur drift beyond
    the agent mask — effectively zero visual damage. Clamping makes
    "essentially perfect" actually perfect so GT-as-pred is strict 1.0.

    ``bad_pixel_threshold`` optionally adds a stricter area-damage term:
    pixels whose raw per-channel mean diff exceeds the threshold count as
    damaged, and ``bad_pixel_tolerance`` is the damaged-pixel fraction at
    which this term reaches 0. The final score is capped by that term. This
    prevents small but obvious repainted regions from being diluted by a
    large unchanged background.

    Returns ``0.0`` when the valid region is smaller than ``floor_fraction``
    of the image — i.e. the model repainted so much that there's no real
    "background" left to preserve.
    """
    if pred is None or ref is None:
        return 0.0
    p = pred
    if p.shape[:2] != ref.shape[:2]:
        p = cv2.resize(p, (ref.shape[1], ref.shape[0]))
    H, W = ref.shape[:2]
    if exclude_mask is not None:
        if exclude_mask.shape[:2] != (H, W):
            exclude_mask = cv2.resize(exclude_mask, (W, H))
        valid = exclude_mask == 0
    else:
        valid = np.ones((H, W), dtype=bool)
    if valid.sum() < floor_fraction * valid.size:
        return 0.0
    p_f = p.astype(np.float32)
    r_f = ref.astype(np.float32)
    diff = np.abs(p_f - r_f)
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    raw_valid_diff = diff[valid]
    if noise_floor > 0:
        diff = np.where(diff < noise_floor, 0.0, diff)
    sim = 1.0 - diff[valid] / 255.0
    bg = float(sim.mean())
    if bad_pixel_threshold is not None:
        bad_fraction = float(np.mean(raw_valid_diff > bad_pixel_threshold))
        bad_score = max(
            0.0,
            1.0 - bad_fraction / max(float(bad_pixel_tolerance), 1e-6),
        )
        bg = min(bg, bad_score)
    if bg >= saturation_threshold:
        bg = 1.0
    return bg


def background_preservation_frames(
    pred_frames: Sequence[np.ndarray],
    ref_frame: Optional[np.ndarray],
    *,
    detector: Optional[Detector] = None,
    base_exclude_mask: Optional[np.ndarray] = None,
    mask_radius_cells: float = 0.9,
    grid_size: int = DEFAULT_GRID_SIZE,
    floor_fraction: float = 0.05,
    noise_floor: float = 5.0,
    saturation_threshold: float = 0.999,
    bad_pixel_threshold: Optional[float] = None,
    bad_pixel_tolerance: float = 0.10,
) -> float:
    """Mean per-frame background preservation across a video.

    For each pred frame, mask out a disk of radius
    ``mask_radius_cells * cell_size(ref_frame)`` around every agent centroid
    returned by ``detector``. Compare the remaining pixels to ``ref_frame``
    via :func:`background_preservation_image`, then average across frames.

    When ``detector`` is ``None`` or finds no blobs, the full frame is
    compared. Empty inputs return ``0.0``.

    ``base_exclude_mask`` is unioned into every frame's exclude mask before
    any detector disks are added. Use it for task-specific regions that the
    model never had access to in the reference image.

    Agents visible in ``ref_frame`` are pre-masked once and unioned into
    every per-frame exclude mask — otherwise the start-cell pixels (agent
    in ref, background in pred once the agent has moved) contribute MAE
    on every frame and silently discount any moved output.
    """
    if not pred_frames or ref_frame is None:
        return 0.0
    H, W = ref_frame.shape[:2]
    cell = cell_size(ref_frame, grid_size)
    radius = max(1, int(round(mask_radius_cells * cell)))

    ref_exclude: Optional[np.ndarray] = None
    if base_exclude_mask is not None:
        ref_exclude = base_exclude_mask
        if ref_exclude.shape[:2] != (H, W):
            ref_exclude = cv2.resize(
                ref_exclude, (W, H), interpolation=cv2.INTER_NEAREST,
            )
        ref_exclude = np.where(ref_exclude > 0, 255, 0).astype(np.uint8)
    if detector is not None:
        ref_blobs = detector(ref_frame)
        if ref_blobs:
            if ref_exclude is None:
                ref_exclude = np.zeros((H, W), dtype=np.uint8)
            for (x, y) in ref_blobs:
                cv2.circle(ref_exclude, (int(x), int(y)), radius, 255, -1)

    scores: List[float] = []
    for frame in pred_frames:
        if frame is None:
            continue
        exclude: Optional[np.ndarray] = (
            ref_exclude.copy() if ref_exclude is not None else None
        )
        if detector is not None:
            blobs = detector(frame)
            if blobs:
                if exclude is None:
                    exclude = np.zeros((H, W), dtype=np.uint8)
                for (x, y) in blobs:
                    cv2.circle(exclude, (int(x), int(y)), radius, 255, -1)
        scores.append(background_preservation_image(
            frame, ref_frame,
            exclude_mask=exclude,
            floor_fraction=floor_fraction,
            noise_floor=noise_floor,
            saturation_threshold=saturation_threshold,
            bad_pixel_threshold=bad_pixel_threshold,
            bad_pixel_tolerance=bad_pixel_tolerance,
        ))
    if not scores:
        return 0.0
    mean = float(np.mean(scores))
    if mean >= saturation_threshold:
        mean = 1.0
    return mean


# ---------------------------------------------------------------------------
# Convenience: obstacle-hit check
# ---------------------------------------------------------------------------

def _pick_best_blob(
    blobs: Sequence[Pixel], ref: np.ndarray,
) -> Optional[Pixel]:
    """Single blob closest (L1) to any reference point; ``None`` if no blobs."""
    if not blobs:
        return None
    return min(
        blobs,
        key=lambda b: float(
            (np.abs(ref[:, 0] - b[0]) + np.abs(ref[:, 1] - b[1])).min(),
        ),
    )


def best_blob_cells(
    video_frames: Sequence[np.ndarray],
    detector: Detector,
    reference_points: Sequence[Pixel],
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> List[Optional[Cell]]:
    """Per-frame cell of the blob closest to any reference point (strict tracking).

    Frames with no detection contribute ``None``.  Fall back to ``agents[0]``
    when ``reference_points`` is empty so callers stay safe.
    """
    ref = np.asarray(reference_points) if reference_points else None
    out: List[Optional[Cell]] = []
    for frame in video_frames:
        blobs = detector(frame)
        if not blobs:
            out.append(None)
            continue
        best = _pick_best_blob(blobs, ref) if ref is not None else blobs[0]
        if best is None:
            out.append(None)
        else:
            out.append(pixel_to_cell(best[0], best[1], frame.shape, grid_size))
    return out


def make_strict_single_detector(
    detector: Detector,
    reference_points: Sequence[Pixel],
) -> Callable[[np.ndarray], Optional[Pixel]]:
    """Adapter: wraps a multi-blob detector into a ``frame -> Optional[Pixel]``.

    Use for ``discontinuity_penalty``'s ``single_detector`` parameter so
    continuity tracks the ref-closest blob instead of the arbitrary first one.
    """
    ref = np.asarray(reference_points) if reference_points else None

    def _single(frame: np.ndarray) -> Optional[Pixel]:
        blobs = detector(frame)
        if not blobs:
            return None
        if ref is None:
            return blobs[0]
        return _pick_best_blob(blobs, ref)

    return _single


def any_agent_on_obstacle(
    video_frames: Sequence[np.ndarray],
    obstacles: Set[Cell],
    detector: Detector,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    reference_points: Optional[Sequence[Pixel]] = None,
) -> bool:
    """Return True if a detected blob ever sits on an obstacle cell.

    ``reference_points`` enables *strict mode*: only the blob closest to the
    reference set is considered per frame, so a hallucinated extra blob
    sitting on an obstacle while the real agent is on the path does not
    trigger a hit.
    """
    if not obstacles:
        return False
    ref = np.asarray(reference_points) if reference_points else None
    for frame in video_frames:
        blobs = detector(frame)
        if not blobs:
            continue
        if ref is not None:
            best = _pick_best_blob(blobs, ref)
            if best is not None and pixel_to_cell(best[0], best[1], frame.shape, grid_size) in obstacles:
                return True
        else:
            for ax, ay in blobs:
                if pixel_to_cell(ax, ay, frame.shape, grid_size) in obstacles:
                    return True
    return False


def obstacle_hit_report(
    video_frames: Sequence[np.ndarray],
    obstacles: Set[Cell],
    detector: Detector,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    reference_points: Optional[Sequence[Pixel]] = None,
) -> Dict[str, object]:
    """Full obstacle-collision report, for debug visualisation.

    ``reference_points`` enables *strict mode*: only the blob closest to the
    reference set contributes per frame, avoiding inflated hit counts from
    phantom detections.

    Returns a dict with:
    - ``hit``: bool — any blob ever occupied an obstacle cell;
    - ``hit_cells``: sorted list of distinct obstacle cells touched;
    - ``hit_frames``: count of frames where at least one blob was on an obstacle;
    - ``per_cell_frames``: ``{cell: frame_count}`` for every touched cell.
    """
    hit_cells: Set[Cell] = set()
    hit_frames = 0
    per_cell_frames: Dict[Cell, int] = {}
    ref = np.asarray(reference_points) if reference_points else None
    for frame in video_frames:
        blobs = detector(frame)
        if not blobs:
            continue
        frame_hit = False
        seen_this_frame: Set[Cell] = set()
        iter_blobs: Sequence[Pixel]
        if ref is not None:
            best = _pick_best_blob(blobs, ref)
            iter_blobs = [best] if best is not None else []
        else:
            iter_blobs = blobs
        for ax, ay in iter_blobs:
            c = pixel_to_cell(ax, ay, frame.shape, grid_size)
            if c in obstacles and c not in seen_this_frame:
                hit_cells.add(c)
                per_cell_frames[c] = per_cell_frames.get(c, 0) + 1
                seen_this_frame.add(c)
                frame_hit = True
        if frame_hit:
            hit_frames += 1
    # Cells are serialised as ``[row, col]`` lists (JSON-friendly; tuples
    # aren't valid JSON and can't be dict keys in JSON either).
    return {
        "hit": bool(hit_cells),
        "hit_cells": [list(c) for c in sorted(hit_cells)],
        "hit_frames": hit_frames,
        "per_cell_frames": [
            {"cell": list(c), "frames": per_cell_frames[c]}
            for c in sorted(per_cell_frames)
        ],
    }
