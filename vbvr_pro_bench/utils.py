"""
Utility functions for VBVR-Bench evaluation.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from PIL import Image
import colorsys

COLOR_BOUNDS: Dict[str, np.ndarray] = {
    'green': np.array([
        [0,  100,  0],   # R_min, G_min, B_min
        [100, 255, 100], # R_max, G_max, B_max
    ], dtype=np.uint8),

    'black': np.array([
        [0,   0,   0],
        [120,  120,  120],
    ], dtype=np.uint8),

    'red': np.array([
        [150,  0,   0],
        [255,  80,  80],
    ], dtype=np.uint8),

    'blue': np.array([
        [0,   0,  150],
        [80,  80, 255],
    ], dtype=np.uint8),
}


def safe_distance(p1: Tuple, p2: Tuple) -> float:
    """
    Calculate Euclidean distance between two points, avoiding integer overflow.
    
    Args:
        p1: First point as (x, y) tuple
        p2: Second point as (x, y) tuple
        
    Returns:
        Euclidean distance as float
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def normalize_frame_size(frame: np.ndarray, target_frame: np.ndarray, 
                         background_color: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Normalize frame size to match target_frame dimensions.
    
    Strategy:
    - If dimensions already match, return as-is
    - If aspect ratios are similar (within 5%), just resize
    - If aspect ratios differ, try to detect and crop padding (gray, white, or black)
      then resize to target dimensions
    - If the target frame itself has dark semantic borders, do not treat black
      as removable padding; fall back to a centered aspect-ratio crop instead
    
    Args:
        frame: Frame to normalize (source)
        target_frame: Frame with target dimensions
        background_color: Background color to detect for cropping (auto-detect if None)
        
    Returns:
        Frame normalized to target dimensions
    """
    if frame.shape == target_frame.shape:
        return frame
    
    h_src, w_src = frame.shape[:2]
    h_tgt, w_tgt = target_frame.shape[:2]
    
    # Calculate aspect ratios
    ar_src = w_src / h_src if h_src > 0 else 1
    ar_tgt = w_tgt / h_tgt if h_tgt > 0 else 1
    
    # If aspect ratios are similar (within 5%), just resize
    if abs(ar_src - ar_tgt) / max(ar_src, ar_tgt) < 0.05:
        return cv2.resize(frame, (w_tgt, h_tgt))
    
    # Try multiple background colors if not specified
    if background_color is None:
        background_colors_to_try = [
            (128, 128, 128),  # Gray
        ]
        if not _target_edges_match_color(target_frame, (255, 255, 255)):
            background_colors_to_try.append((255, 255, 255))  # White
        if not _target_edges_match_color(target_frame, (0, 0, 0)):
            background_colors_to_try.append((0, 0, 0))  # Black

        candidates = [_center_crop_to_aspect(frame, ar_tgt)]

        for bg_color in background_colors_to_try:
            cropped = _crop_padded_content(frame, bg_color)
            candidates.append(cropped)

        # Prefer the smallest plausible crop. This keeps old behaviour for
        # real gray/white padding, but avoids stretching aspect-mismatched
        # frames when no padding colour is safely identifiable.
        result_frame = min(
            candidates,
            key=lambda crop: (
                not (0.2 < (crop.shape[0] * crop.shape[1]) / (h_src * w_src) <= 1.0),
                (crop.shape[0] * crop.shape[1]) / (h_src * w_src),
            ),
        )
    else:
        result_frame = _crop_padded_content(frame, background_color)
    
    # Resize the cropped content to target dimensions
    return cv2.resize(result_frame, (w_tgt, h_tgt))


def _center_crop_to_aspect(frame: np.ndarray, target_aspect: float) -> np.ndarray:
    """Center-crop ``frame`` to ``target_aspect`` without changing scale."""
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0 or target_aspect <= 0:
        return frame

    src_aspect = w / h
    if abs(src_aspect - target_aspect) / max(src_aspect, target_aspect) < 0.01:
        return frame

    if src_aspect > target_aspect:
        new_w = max(1, min(w, int(round(h * target_aspect))))
        left = max(0, (w - new_w) // 2)
        return frame[:, left:left + new_w]

    new_h = max(1, min(h, int(round(w / target_aspect))))
    top = max(0, (h - new_h) // 2)
    return frame[top:top + new_h, :]


def _target_edges_match_color(
    target_frame: np.ndarray,
    color: Tuple[int, int, int],
    *,
    tolerance: int = 20,
    min_fraction: float = 0.25,
) -> bool:
    """Whether target-frame edges contain enough of ``color`` to be semantic."""
    h, w = target_frame.shape[:2]
    if h <= 0 or w <= 0:
        return False
    edge = max(2, int(round(min(h, w) * 0.02)))
    if target_frame.ndim < 3:
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_GRAY2BGR)
    edge_pixels = np.concatenate([
        target_frame[:edge, :, :].reshape(-1, target_frame.shape[2]),
        target_frame[-edge:, :, :].reshape(-1, target_frame.shape[2]),
        target_frame[:, :edge, :].reshape(-1, target_frame.shape[2]),
        target_frame[:, -edge:, :].reshape(-1, target_frame.shape[2]),
    ], axis=0)
    color_arr = np.array(color, dtype=np.int16)
    diff = np.abs(edge_pixels.astype(np.int16) - color_arr)
    return float(np.mean(np.all(diff <= tolerance, axis=1))) >= min_fraction


def _crop_padded_content(frame: np.ndarray, background_color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """
    Crop out padding from a frame to extract the original content.
    Detects rows/columns that are mostly the background color and removes them.
    
    Args:
        frame: Frame that may have padding
        background_color: Color of the padding (BGR format)
        
    Returns:
        Cropped frame with padding removed
    """
    h, w = frame.shape[:2]
    bg_color = np.array(background_color, dtype=np.uint8)
    
    # Tolerance for background detection
    tolerance = 20
    
    # Create a mask of pixels that are NOT background
    diff = np.abs(frame.astype(np.int16) - bg_color.astype(np.int16))
    not_bg_mask = np.any(diff > tolerance, axis=2)
    
    # Find rows and columns with content
    row_has_content = np.any(not_bg_mask, axis=1)
    col_has_content = np.any(not_bg_mask, axis=0)
    
    # Find content boundaries
    rows_with_content = np.where(row_has_content)[0]
    cols_with_content = np.where(col_has_content)[0]
    
    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        # No content detected, return original
        return frame
    
    top = rows_with_content[0]
    bottom = rows_with_content[-1] + 1
    left = cols_with_content[0]
    right = cols_with_content[-1] + 1
    
    # Add small margin to avoid cutting off content edges
    margin = 2
    top = max(0, top - margin)
    bottom = min(h, bottom + margin)
    left = max(0, left - margin)
    right = min(w, right + margin)
    
    # Crop
    cropped = frame[top:bottom, left:right]
    
    # Only return cropped if it's significantly smaller (padding was actually removed)
    if cropped.shape[0] < h * 0.95 or cropped.shape[1] < w * 0.95:
        return cropped
    
    # No significant padding detected, return original
    return frame


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_json(data: Any, path: str):
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def get_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    frame_indices: Optional[List[int]] = None
) -> List[np.ndarray]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (evenly sampled)
        frame_indices: Specific frame indices to extract
        
    Returns:
        List of frames as numpy arrays (BGR format)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_indices is not None:
        indices_set = set(frame_indices)
        max_idx = max(frame_indices)
    elif max_frames is not None and max_frames < total_frames:
        indices_set = set(np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist())
        max_idx = total_frames - 1
    else:
        indices_set = None  # Read all frames
        max_idx = total_frames - 1

    # Read frames sequentially to avoid unreliable seeking with cap.set()
    frames = []
    for idx in range(max_idx + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if indices_set is None or idx in indices_set:
            frames.append(frame)

    cap.release()
    return frames



def get_frame_count(video_path: str) -> int:
    """Get total frame count of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def get_video_info(video_path: str) -> Dict:
    """Get video information (fps, width, height, frame_count)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return info


def load_image(path: str) -> np.ndarray:
    """Load an image file."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img


def load_gt_metadata(gt_path: str) -> Dict:
    """
    Load ground truth metadata for a task instance.
    
    Args:
        gt_path: Path to GT folder containing ground_truth.mp4, first_frame.png, etc.
        
    Returns:
        Dictionary with GT metadata
    """
    metadata = {
        'path': gt_path,
        'has_video': os.path.exists(os.path.join(gt_path, 'ground_truth.mp4')),
        'has_first_frame': os.path.exists(os.path.join(gt_path, 'first_frame.png')),
        'has_final_frame': os.path.exists(os.path.join(gt_path, 'final_frame.png')),
        'has_prompt': os.path.exists(os.path.join(gt_path, 'prompt.txt')),
    }
    
    if metadata['has_video']:
        metadata['video_info'] = get_video_info(os.path.join(gt_path, 'ground_truth.mp4'))
    
    if metadata['has_prompt']:
        with open(os.path.join(gt_path, 'prompt.txt'), 'r') as f:
            metadata['prompt'] = f.read().strip()
    
    return metadata


def extract_task_info_from_path(path: str) -> Dict:
    """Extract task information from a video path."""
    parts = Path(path).parts
    
    info = {
        'full_path': path,
        'filename': parts[-1] if parts else '',
    }
    
    # Try to extract split and task name
    for i, part in enumerate(parts):
        if part in ['In-Domain_50', 'Out-of-Domain_50']:
            info['split'] = part
            if i + 1 < len(parts):
                info['task_name'] = parts[i + 1]
            break
    
    return info


# ============================================================================
# Image/Frame Comparison Utilities
# ============================================================================

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    Returns value between 0 and 1, where 1 means identical.
    """
    if img1.shape != img2.shape:
        img2 = normalize_frame_size(img2, img1)
    
    # Convert to grayscale if color
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim_map.mean())


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two images."""
    if img1.shape != img2.shape:
        img2 = normalize_frame_size(img2, img1)
    
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(255 ** 2 / mse))


def compute_histogram_similarity(img1: np.ndarray, img2: np.ndarray, method: str = 'correlation') -> float:
    """
    Compute histogram similarity between two images.
    
    Args:
        img1, img2: Input images (BGR)
        method: One of 'correlation', 'chi-square', 'intersection', 'bhattacharyya'
        
    Returns:
        Similarity score (higher is more similar for correlation/intersection)
    """
    # Convert to HSV
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    # Compute histograms
    h_bins, s_bins = 50, 60
    hist_size = [h_bins, s_bins]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    channels = [0, 1]
    
    hist1 = cv2.calcHist([hsv1], channels, None, hist_size, ranges)
    hist2 = cv2.calcHist([hsv2], channels, None, hist_size, ranges)
    
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    methods = {
        'correlation': cv2.HISTCMP_CORREL,
        'chi-square': cv2.HISTCMP_CHISQR,
        'intersection': cv2.HISTCMP_INTERSECT,
        'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA
    }
    
    return float(cv2.compareHist(hist1, hist2, methods.get(method, cv2.HISTCMP_CORREL)))


# ============================================================================
# Color Analysis Utilities
# ============================================================================

def get_dominant_colors(img: np.ndarray, n_colors: int = 5) -> List[Tuple[int, int, int]]:
    """
    Extract dominant colors from an image using k-means clustering.
    
    Returns:
        List of (B, G, R) color tuples
    """
    # Reshape image
    pixels = img.reshape(-1, 3).astype(np.float32)
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Get colors sorted by frequency
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    
    colors = []
    for idx in sorted_indices:
        color = tuple(int(c) for c in centers[idx])
        colors.append(color)
    
    return colors


def color_distance(c1: Tuple, c2: Tuple, method: str = 'euclidean') -> float:
    """
    Compute distance between two colors.
    
    Args:
        c1, c2: Color tuples (B, G, R) or (H, S, V)
        method: 'euclidean' or 'deltaE' (perceptual)
        
    Returns:
        Distance value (lower = more similar)
    """
    if method == 'euclidean':
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    else:
        # Simple euclidean as fallback
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to HSV (H: 0-360, S: 0-100, V: 0-100)."""
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360, s * 100, v * 100


def color_name_match(color_bgr: Tuple[int, int, int], expected_name: str) -> float:
    """
    Check if a BGR color matches an expected color name.
    
    Args:
        color_bgr: (B, G, R) tuple
        expected_name: Color name like 'red', 'green', 'blue', etc.
        
    Returns:
        Match score between 0 and 1
    """
    b, g, r = color_bgr
    h, s, v = rgb_to_hsv(r, g, b)
    
    # Define color ranges (H: 0-360, S: 0-100, V: 0-100)
    color_ranges = {
        'red': [(0, 15), (345, 360)],  # Red wraps around
        'orange': [(15, 45)],
        'yellow': [(45, 70)],
        'green': [(70, 170)],
        'cyan': [(170, 200)],
        'blue': [(200, 260)],
        'purple': [(260, 290)],
        'magenta': [(290, 345)],
        'white': None,  # High V, low S
        'black': None,  # Low V
        'gray': None,   # Low S
    }
    
    expected_name = expected_name.lower()
    
    if expected_name == 'white':
        return 1.0 if v > 80 and s < 20 else max(0, (v - 50) / 50 * (1 - s / 100))
    elif expected_name == 'black':
        return 1.0 if v < 20 else max(0, (50 - v) / 50)
    elif expected_name == 'gray':
        return 1.0 if s < 20 and 20 < v < 80 else max(0, 1 - s / 50)
    elif expected_name in color_ranges:
        ranges = color_ranges[expected_name]
        for h_min, h_max in ranges:
            if h_min <= h <= h_max and s > 30 and v > 30:
                return 1.0
            elif h_min <= h <= h_max:
                return max(0, min(s / 50, v / 50))
    
    return 0.0


def color_name_to_hsv_tolerance(
    hsv_img: np.ndarray,
    color_name: str = 'red',
    tolerance: int = 10,
    min_saturation: int = 80,
    min_value: int = 80
) -> np.ndarray:
    """
    Build a color mask in HSV space from a color name and hue tolerance.

    Args:
        hsv_img: HSV image in OpenCV format (H: 0-179, S/V: 0-255)
        color_name: Target color name
        tolerance: Hue tolerance (0-90), for white black and gray, set tolerance to 10. 
        min_saturation: Minimum saturation for chromatic colors
        min_value: Minimum value for chromatic colors

    Returns:
        Binary mask (0/255)
    """
    color_name = color_name.lower()
    tolerance = int(max(0, min(90, tolerance)))

    if color_name == 'white':
        return cv2.inRange(hsv_img, np.array([0, 0, max(0, 255 - tolerance * 5)]), np.array([179, 40, 255]))
    if color_name == 'black':
        return cv2.inRange(hsv_img, np.array([0, 0, 0]), np.array([179, 255, min(255, tolerance * 5)]))
    if color_name == 'gray':
        return cv2.inRange(
            hsv_img,
            np.array([0, 0, max(0, 80 - tolerance * 2)]),
            np.array([179, 50, min(255, 180 + tolerance * 2)])
        )

    hue_centers = {
        'red': 0,
        'orange': 15,
        'yellow': 30,
        'green': 60,
        'cyan': 90,
        'blue': 120,
        'purple': 140,
        'magenta': 160,
    }
    center = hue_centers.get(color_name, hue_centers['red'])

    def _hue_range_mask(center_h: int) -> np.ndarray:
        lo = center_h - tolerance
        hi = center_h + tolerance
        if lo >= 0 and hi <= 179:
            lower = np.array([lo, min_saturation, min_value], dtype=np.uint8)
            upper = np.array([hi, 255, 255], dtype=np.uint8)
            return cv2.inRange(hsv_img, lower, upper)

        # Hue wraps around 0/179.
        if lo < 0:
            lower1 = np.array([0, min_saturation, min_value], dtype=np.uint8)
            upper1 = np.array([hi, 255, 255], dtype=np.uint8)
            lower2 = np.array([180 + lo, min_saturation, min_value], dtype=np.uint8)
            upper2 = np.array([179, 255, 255], dtype=np.uint8)
            return cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)

        lower1 = np.array([lo, min_saturation, min_value], dtype=np.uint8)
        upper1 = np.array([179, 255, 255], dtype=np.uint8)
        lower2 = np.array([0, min_saturation, min_value], dtype=np.uint8)
        upper2 = np.array([hi - 180, 255, 255], dtype=np.uint8)
        return cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)

    return _hue_range_mask(center)


# ============================================================================
# Shape Detection Utilities
# ============================================================================

def denoise_contour(contour: np.ndarray, epsilon_ratio: float = 0.02) -> np.ndarray:
    if contour is None or len(contour) < 4:
        return contour
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return contour
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
    return approx if len(approx) >= 3 else contour


def detect_closed_contours_by_color(
    img: np.ndarray,
    color_bounds: np.ndarray,
    min_area: int = 100,
    closure_gap: int = 5,
    closure_ratio: float = 0.05,
    hollow_only: bool = True,
    max_fill_ratio: float = 0.6,
    hull_fallback: bool = False,
    ref_area: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Detect closed (or near-closed) contours within a given color range.

    Args:
        img: Input image, BGR numpy array of shape (H, W, 3).
        color_bounds: numpy array of shape (2, 3).
            - color_bounds[0]: lower color bound in RGB, e.g. [R_min, G_min, B_min].
            - color_bounds[1]: upper color bound in RGB, e.g. [R_max, G_max, B_max].
            Note: bounds are given in RGB order; the function converts to
            OpenCV's BGR order internally.
        min_area: Minimum contour area in pixels, used to reject noise.
            Default 100.
        closure_gap: Kernel size in pixels for the morphological closing that
            bridges small gaps in a stroke. Larger values close bigger gaps but
            may also merge neighbouring contours. Default 5.
        closure_ratio: Threshold for treating a contour as "near-closed",
            computed as contour_area / convex_hull_area -- the closer to 1, the
            more closed the shape. Contours below this are dropped. Default 0.05,
            which is very permissive and essentially only rejects line segments.
        hollow_only: If True, keep only hollow (outlined) contours and drop
            solid-filled shapes. Solidity is judged by the fraction of the
            contour's interior covered by matching color pixels (fill_ratio);
            anything above max_fill_ratio counts as filled. Default True.
        max_fill_ratio: Only used when hollow_only=True. Upper bound on the
            fraction of colored pixels inside a contour before it is treated as
            solid. Default 0.6.
        hull_fallback: If True and no contour survives filtering, retry by
            taking the convex hull of the remaining strokes (see below).
        ref_area: Optional reference area used to reject an implausibly large
            hull when hull_fallback is active.

    Returns:
        List[np.ndarray]: Detected closed contours, each a numpy array of shape
            (N, 1, 2) in OpenCV's standard contour format, sorted by area from
            largest to smallest. Empty list if nothing matches.

    Example:
        >>> import numpy as np
        >>> import cv2
        >>> img = cv2.imread("frame.png")
        >>> # Detect red contours (RGB lower [150, 0, 0], upper [255, 80, 80])
    """
    # --- 1. Color space conversion: RGB -> BGR (OpenCV uses BGR) ---
    lower_bgr = np.array([color_bounds[0, 2], color_bounds[0, 1], color_bounds[0, 0]], dtype=np.uint8)
    upper_bgr = np.array([color_bounds[1, 2], color_bounds[1, 1], color_bounds[1, 0]], dtype=np.uint8)

    # --- 2. Build the color mask ---
    mask = cv2.inRange(img, lower_bgr, upper_bgr)

    # --- 3. Morphological closing to bridge small gaps in the strokes ---
    if closure_gap > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closure_gap, closure_gap)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- 4. Extract contours (outer boundaries only) ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 5. Filter: area + closedness + optional hollow check ---
    closed_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 5a. Area filter
        if area < min_area:
            continue
        
        # 5b. Closedness check
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1:
            continue
        solidity = area / hull_area
        if solidity < closure_ratio:
            continue

        # 5c. Hollow check: fraction of the contour interior covered by color pixels
        if hollow_only:
            filled_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(filled_mask, [cnt], -1, 255, -1)
            interior_total = int(filled_mask.sum() // 255)
            if interior_total > 0:
                interior_colored = int(cv2.bitwise_and(mask, filled_mask).sum() // 255)
                fill_ratio = interior_colored / interior_total
                if fill_ratio > max_fill_ratio:
                    continue

        closed_contours.append(cnt)

    # --- 6. Sort by area, largest first ---
    closed_contours.sort(key=cv2.contourArea, reverse=True)

    if hull_fallback and not closed_contours:
        stroke, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stroke = [c for c in stroke if cv2.contourArea(c) >= 50]
        if stroke and len(stroke) <= 2:                                    # ①
            hull = cv2.convexHull(np.vstack(stroke))
            hull_area = float(cv2.contourArea(hull))
            perim = np.zeros(mask.shape[:2], dtype=np.uint8)
            cv2.drawContours(perim, [hull], -1, 255, 1)
            grown = cv2.dilate(mask, np.ones((11, 11), np.uint8))
            n_perim = int((perim > 0).sum())
            cov = (int(((perim > 0) & (grown > 0)).sum()) / n_perim) if n_perim else 0.0
            fill = int((mask > 0).sum()) / max(hull_area, 1.0)
            if (hull_area >= min_area                              
                    and cov >= 0.85                                 
                    and fill <= 0.75                                       # ④
                    and (ref_area is None or hull_area <= 3.0 * ref_area)): 
                closed_contours = [hull]

    return closed_contours


def contour_iou(
    contour1: np.ndarray,
    contour2: np.ndarray,
    canvas_size: Optional[Tuple[int, int]] = None,
) -> float:
    """
    Compute the IoU (Intersection over Union) between two contours.

    Both contours are rasterized into filled masks and the IoU is taken over
    pixels, so the result is valid for any shape -- convex, concave, or
    irregular.

    Args:
        contour1: First contour, numpy array of shape (N, 1, 2) -- i.e. a single
            element of the list returned by detect_closed_contours_by_color.
        contour2: Second contour, numpy array of shape (M, 1, 2).
        canvas_size: (height, width) of the canvas used to draw the masks.
            - If None, the smallest size holding both contours is used (the
              union of their bounding boxes) plus a 1-pixel margin, so a contour
              touching the edge is still drawn completely.
            - If both contours come from the same frame, pass that frame's
              (H, W) so the coordinates stay aligned.

    Returns:
        float: IoU in [0.0, 1.0].
            - 1.0 means the two contours coincide exactly.
            - 0.0 means they do not overlap at all, or one of them has zero area.

    Example:
        >>> contours_a = detect_closed_contours_by_color(img, bounds_a)
        >>> contours_b = detect_closed_contours_by_color(img, bounds_b)
        >>> if contours_a and contours_b:
        ...     iou = contour_iou(contours_a[0], contours_b[0])
        ...     print(f"IoU = {iou:.4f}")
    """
    # --- 1. Determine the canvas size ---
    if canvas_size is not None:
        h, w = canvas_size
    else:
        # Take each contour's bounding box and merge them into a minimal canvas
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        x2, y2, w2, h2 = cv2.boundingRect(contour2)
        # Use the max right/bottom edge of both boxes, plus a 1-pixel margin
        w = max(x1 + w1, x2 + w2) + 1
        h = max(y1 + h1, y2 + h2) + 1

    # --- 2. Rasterize each contour into a filled mask ---
    # Single-channel uint8: 255 inside the contour, 0 outside
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask1, [contour1], contourIdx=0, color=255, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [contour2], contourIdx=0, color=255, thickness=cv2.FILLED)

    # --- 3. Pixel-wise intersection and union ---
    # Intersection: pixels set to 255 in both masks
    intersection = np.count_nonzero(cv2.bitwise_and(mask1, mask2))
    # Union: pixels set to 255 in at least one mask
    union = np.count_nonzero(cv2.bitwise_or(mask1, mask2))

    # --- 4. Compute IoU ---
    # union == 0 means both contours have zero area (degenerate); return 0.0
    if union == 0:
        return 0.0

    return intersection / union


def match_contours(
    gt_contours: List[np.ndarray],
    pred_contours: List[np.ndarray],
    iou_threshold: float = 0.1,
    canvas_size: Optional[Tuple[int, int]] = None,
) -> List[Optional[float]]:
    """
    Match pred_contours one-to-one against gt_contours.

    Matching rules:
    - For each contour in gt_contours, the highest-IoU contour in pred_contours
      is taken as its candidate.
    - The match counts only if that best IoU >= iou_threshold.
    - Each pred contour can match at most one gt contour (one-to-one).
    - IoU is computed over the filled area each contour encloses, so stroke
      thickness does not matter.

    Args:
        gt_contours: Ground-truth contours, i.e. the output of
            detect_closed_contours_by_color.
        pred_contours: Predicted contours to match, same format.
        iou_threshold: Minimum IoU for a candidate to count as a match.
            Default 0.1.
        canvas_size: (height, width) forwarded to contour_iou. If both sets come
            from the same frame, pass that frame's size so coordinates stay
            aligned; if None, each pair gets its own minimal canvas.

    Returns:
        List[Optional[float]] of the same length as gt_contours. Element i is
        the result for gt_contours[i]:
        - matched: float, the IoU of that pair.
        - unmatched: None.

    Example:
        >>> gt   = detect_closed_contours_by_color(gt_frame, bounds)
        >>> pred = detect_closed_contours_by_color(pred_frame, bounds)
        >>> results = match_contours(gt, pred, iou_threshold=0.3,
        ...                          canvas_size=(gt_frame.shape[0], gt_frame.shape[1]))
        >>> for i, iou in enumerate(results):
        ...     if iou is None:
        ...         print(f"gt[{i}]: no match")
        ...     else:
        ...         print(f"gt[{i}]: IoU={iou:.4f}")
    """
    # Result list, initially all None
    results: List[Optional[float]] = [None] * len(gt_contours)

    if not gt_contours or not pred_contours:
        return results

    # --- 1. Precompute the full gt x pred IoU matrix ---
    # iou_matrix[i][j] = IoU of gt_contours[i] against pred_contours[j]
    n_gt = len(gt_contours)
    n_pred = len(pred_contours)
    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)

    for i, gt_cnt in enumerate(gt_contours):
        for j, pred_cnt in enumerate(pred_contours):
            iou_matrix[i, j] = contour_iou(gt_cnt, pred_cnt, canvas_size=canvas_size)

    # --- 2. Greedy matching: repeatedly take the globally best (i, j) pair ---
    # Index sets of already-matched gt / pred, to prevent reuse
    matched_gt = set()
    matched_pred = set()

    # At most min(n_gt, n_pred) matches are possible
    for _ in range(min(n_gt, n_pred)):
        # Find the maximum among unmatched rows/columns by masking
        # already-matched rows and columns to -1 so they cannot be picked
        remaining = iou_matrix.copy()
        for i in matched_gt:
            remaining[i, :] = -1.0
        for j in matched_pred:
            remaining[:, j] = -1.0

        best_iou = remaining.max()

        # Best IoU is below threshold, so every remaining candidate is too -- stop
        if best_iou < iou_threshold:
            break

        # Locate the maximum
        best_i, best_j = np.unravel_index(remaining.argmax(), remaining.shape)

        results[best_i] = float(best_iou)
        matched_gt.add(best_i)
        matched_pred.add(best_j)

    return results


def _masked_mse_01(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, sensitivity: float = 0.0005) -> float:
    """
    Compute the MSE between two images over the non-zero region of a mask and
    map it into [0, 1]. Higher means more similar (1.0 means identical).
    The exponential-decay mapping makes this extremely sensitive to small
    pixel-level changes.

    :param img1: BGR image 1, shape (H, W, 3), normally uint8
    :param img2: BGR image 2, shape (H, W, 3), normally uint8
    :param mask: Mask of shape (H, W) or (H, W, 1)
    :param sensitivity: Sensitivity coefficient. Larger values react more
                        strongly to differences and drop the score faster.
                        Default 0.0005. The score is exp(-sensitivity * MSE);
                        e.g. at sensitivity=0.05 an average error of 4 grey
                        levels gives MSE=16 and a score of exp(-0.8) ~= 0.45.
    :return: Similarity score in [0.0, 1.0]
    """
    # 1. Cast to float32 so uint8 subtraction cannot wrap around (e.g. 5 - 10 = 251)
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    
    # 2. Normalize the mask to a 2D boolean array
    if mask.ndim == 3:
        mask = mask.squeeze()  # ensure the mask is 2D (H, W)
    bool_mask = mask > 0       # take the non-zero region
    
    # 3. Gather the pixels inside the valid region.
    # pixels1 / pixels2 come out as (N, 3), N = number of non-zero mask pixels
    pixels1 = img1_f[bool_mask]
    pixels2 = img2_f[bool_mask]
    
    # An empty mask means there is nothing to compare; treat it as identical (1.0)
    if len(pixels1) == 0:
        return 1.0
        
    # 4. Mean squared error
    mse = np.mean((pixels1 - pixels2) ** 2)
    
    # 5. Exponential decay maps MSE into [0, 1] while staying highly sensitive
    score = np.exp(-sensitivity * mse)
    
    return float(score)


def _masked_ssim_01(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute SSIM between two BGR images over the non-zero region of a mask,
    mapped into [0, 1].

    SSIM is theoretically in [-1, 1] and is mapped linearly via (x + 1) / 2.
    An empty mask (no valid pixels) returns 1.0, i.e. treated as identical.
    """
    if np.count_nonzero(mask) == 0:
        return 1.0

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    m = (mask > 0)
    g1_masked = np.where(m, g1, 0.0)
    g2_masked = np.where(m, g2, 0.0)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ksize, sigma = (11, 11), 1.5

    mu1 = cv2.GaussianBlur(g1_masked, ksize, sigma)
    mu2 = cv2.GaussianBlur(g2_masked, ksize, sigma)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1_masked ** 2,      ksize, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2_masked ** 2,      ksize, sigma) - mu2_sq
    sigma12   = cv2.GaussianBlur(g1_masked * g2_masked, ksize, sigma) - mu1_mu2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    ssim_val = float(ssim_map[m].mean())
    return (ssim_val + 1.0) / 2.0  # [-1, 1] → [0, 1]


def score_background_similarity(
    ref_img: np.ndarray,
    test_img: np.ndarray,
    white_threshold: int = 240,
    type: str = 'ssim',
) -> float:
    """
    Similarity between the white background regions of two images.

    The white region of ref_img (all three channels >= white_threshold) is used
    as a mask, and SSIM is computed inside it and mapped into [0, 1].

    Args:
        ref_img: Reference image, BGR numpy array of shape (H, W, 3).
        test_img: Test image, BGR, shape must match ref_img exactly.
        white_threshold: Grey level above which a pixel counts as white.
            Default 240.
        type: 'ssim' (default) or 'mse' to select the comparison metric.

    Returns:
        float: White-region similarity in [0.0, 1.0].
            Returns 1.0 when ref_img contains no white pixels.

    Raises:
        ValueError: If the two images have different shapes.
    """
    if ref_img.shape != test_img.shape:
        raise ValueError(
            f"image sizes differ: ref={ref_img.shape}, test={test_img.shape}"
        )

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    white_mask = (ref_gray >= white_threshold).astype(np.uint8) * 255
    
    if type == 'mse':
        return _masked_mse_01(ref_img, test_img, white_mask)
    else:
        return _masked_ssim_01(ref_img, test_img, white_mask)


def score_foreground_similarity(
    ref_img: np.ndarray,
    test_img: np.ndarray,
    color_bounds: np.ndarray,
    white_threshold: int = 240,
    type: str = 'ssim',
) -> float:
    """
    Similarity between the foreground regions of two images, where foreground
    is everything left after removing white and one specified color range.

    Using ref_img as the reference, the white region and the region matching
    color_bounds are excluded, then SSIM is computed over what remains and
    mapped into [0, 1].

    Args:
        ref_img: Reference image, BGR numpy array of shape (H, W, 3).
        test_img: Test image, BGR, shape must match ref_img exactly.
        color_bounds: numpy array of shape (2, 3), in RGB order.
            - color_bounds[0]: lower bound [R_min, G_min, B_min].
            - color_bounds[1]: upper bound [R_max, G_max, B_max].
            Pixels in this range are excluded from the foreground along with
            white pixels.
        white_threshold: Grey level above which a pixel counts as white.
            Default 240.
        type: 'ssim' (default) or 'mse' to select the comparison metric.

    Returns:
        float: Foreground similarity in [0.0, 1.0].
            Returns 1.0 when ref_img has no foreground pixels, i.e. it is
            entirely white or entirely the target color.

    Raises:
        ValueError: If the two images have different shapes.
    """
    if ref_img.shape != test_img.shape:
        raise ValueError(
            f"image sizes differ: ref={ref_img.shape}, test={test_img.shape}"
        )

    # White mask
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    white_mask = (ref_gray >= white_threshold).astype(np.uint8) * 255

    # Mask for the specified color (RGB -> BGR)
    lower_bgr = np.array([color_bounds[0, 2], color_bounds[0, 1], color_bounds[0, 0]], dtype=np.uint8)
    upper_bgr = np.array([color_bounds[1, 2], color_bounds[1, 1], color_bounds[1, 0]], dtype=np.uint8)
    color_mask = cv2.inRange(ref_img, lower_bgr, upper_bgr)

    # Foreground = everything except white and the target color
    fg_mask = cv2.bitwise_not(cv2.bitwise_or(white_mask, color_mask))

    if type == 'ssim':
        ssim_score = _masked_ssim_01(ref_img, test_img, fg_mask)
        if ssim_score >= 0.95:
            return ssim_score
        elif ssim_score >= 0.9:
            return 0.5
        else:
            return 0.0
    else:
        return _masked_mse_01(ref_img, test_img, fg_mask)


def detect_shapes(
    img: np.ndarray,
    min_area: int = 100
) -> List[Dict]:
    """
    Detect shapes in an image.
    
    Returns:
        List of shape dictionaries with 'type', 'contour', 'center', 'area', 'bbox'
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Approximate contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Classify shape
        vertices = len(approx)
        shape_type = classify_shape(vertices, contour)
        
        # Get center and bounding box
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
        
        x, y, w, h = cv2.boundingRect(contour)
        
        shapes.append({
            'type': shape_type,
            'contour': contour,
            'vertices': vertices,
            'center': (cx, cy),
            'area': area,
            'bbox': (x, y, w, h),
            'approx': approx
        })
    
    return shapes

def detect_shapes_white_background(
    img: np.ndarray,
    min_area: int = 100,
    preload_masks: Optional[List[np.ndarray]] = None,
    preload_bg_remove: bool = False,
    white_tolerance: int = 10
) -> List[Dict]:
    """
    Detect shapes in an image with white background. 
    Can directly use the preloaded masks (and remove white background if preload_bg_remove is True).
    
    Returns:
        List of shape dictionaries with 'type', 'fill_ratio', 'contour', 'center', 'area', 'bbox'
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_mask = color_name_to_hsv_tolerance(hsv, color_name='white', tolerance=white_tolerance)
    unwhite_mask = cv2.bitwise_not(white_mask)
    if preload_masks is None:
        contours, _ = cv2.findContours(unwhite_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        h, w = img.shape[:2]
        contours = []
        for mask in preload_masks:
            if mask is None:
                continue
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if preload_bg_remove:
                mask = np.where(unwhite_mask > 0, mask, 0)
            mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
            mask_contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not mask_contours:
                contours.append(None)
                continue
            contours.append(max(mask_contours, key=cv2.contourArea))
    
    shapes = []
    for contour in contours:
        if contour is None:
            shapes.append(None)
            continue

        area = cv2.contourArea(contour)
        if area < min_area:
            if preload_masks is not None:
                shapes.append(None)
            continue

        # Filter out solid red objects using fill ratio
        tmp_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(tmp_mask, [contour], -1, 255, -1)
        interior_pixels = np.sum(tmp_mask > 0)
        filled_pixels = np.sum((tmp_mask > 0) & (unwhite_mask > 0))
        fill_ratio = filled_pixels / interior_pixels if interior_pixels > 0 else 0
        
        # Approximate contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Classify shape
        vertices = len(approx)
        shape_type = classify_shape(vertices, approx)
        
        # Get center and bounding box
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
        
        x, y, w, h = cv2.boundingRect(contour)
        
        shapes.append({
            'type': shape_type,
            'fill_ratio': fill_ratio,
            'contour': contour,
            'vertices': vertices,
            'center': (cx, cy),
            'area': area,
            'bbox': (x, y, w, h),
            'approx': approx
        })
    
    return shapes

def detect_circles(
    img: np.ndarray,
    min_area: int = 100,
    color_name: str = 'red',
    color_tolerance: int = 10,
    color_min_saturation: int = 80,
    color_min_value: int = 80,
    fill_max_ratio: float = 0.5
) -> List[Dict]:
    """Detect hollow circle markings in the frame.

    Returns list of dicts with 'center' and 'contour' keys.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color_mask = color_name_to_hsv_tolerance(
        hsv,
        color_name=color_name,
        tolerance=color_tolerance,
        min_saturation=color_min_saturation,
        min_value=color_min_value,
    )
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # #save color_mask to tmp.png

    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Filter out solid red objects using fill ratio
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        interior_pixels = np.sum(mask > 0)
        filled_pixels = np.sum((mask > 0) & (color_mask > 0))
        fill_ratio = filled_pixels / interior_pixels if interior_pixels > 0 else 0
        if fill_ratio >= fill_max_ratio:
            continue

        # Approximate contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Classify shape
        vertices = len(approx)
        shape_type = classify_shape(vertices, approx)
        
        # Get center and bounding box
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        circles.append({
            'type': shape_type,
            'fill_ratio': fill_ratio,
            'contour': contour,
            'vertices': vertices,
            'center': (cx, cy),
            'area': area,
            'bbox': (x, y, w, h),
            'approx': approx
        })
    
    return circles


def _hsv_change_ratio(
    reference_image: np.ndarray,
    target_image: np.ndarray,
    eval_mask: np.ndarray,
    hsv_delta_tolerance: Tuple[float, float, float],
    return_changed_mask: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:
    """Compute HSV change ratio with per-channel delta thresholds."""
    if reference_image.shape != target_image.shape:
        target_image = normalize_frame_size(target_image, reference_image)

    hsv_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_tgt = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV).astype(np.float32)

    dh = np.abs(hsv_ref[..., 0] - hsv_tgt[..., 0])
    dh = np.minimum(dh, 180.0 - dh)
    ds = np.abs(hsv_ref[..., 1] - hsv_tgt[..., 1])
    dv = np.abs(hsv_ref[..., 2] - hsv_tgt[..., 2])

    if len(hsv_delta_tolerance) != 3:
        raise ValueError("hsv_delta_tolerance must be a 3-tuple: (delta_h, delta_s, delta_v)")
    delta_h, delta_s, delta_v = [float(v) for v in hsv_delta_tolerance]
    delta_h = max(0.0, delta_h)
    delta_s = max(0.0, delta_s)
    delta_v = max(0.0, delta_v)

    valid = eval_mask > 0
    valid_count = int(np.sum(valid))

    black_v_thr = 50.0
    white_v_thr = 205.0
    achromatic_s_thr = 40.0

    ref_black = hsv_ref[..., 2] <= black_v_thr
    ref_white = (hsv_ref[..., 2] >= white_v_thr) & (hsv_ref[..., 1] <= achromatic_s_thr)
    ref_gray = (
        (~ref_black)
        & (~ref_white)
        & (hsv_ref[..., 1] <= achromatic_s_thr)
    )
    ref_color = ~(ref_black | ref_white | ref_gray)

    tgt_black = hsv_tgt[..., 2] <= black_v_thr
    tgt_white = (hsv_tgt[..., 2] >= white_v_thr) & (hsv_tgt[..., 1] <= achromatic_s_thr)
    tgt_gray = (
        (~tgt_black)
        & (~tgt_white)
        & (hsv_tgt[..., 1] <= achromatic_s_thr)
    )
    tgt_color = ~(tgt_black | tgt_white | tgt_gray)

    both_color = ref_color & tgt_color
    both_white = ref_white & tgt_white
    both_black = ref_black & tgt_black
    both_gray = ref_gray & tgt_gray

    # Any class transition is treated as changed:
    # white/black/gray/color -> any different class.
    class_mismatch = ~(both_color | both_white | both_black | both_gray)

    color_changed = (dh > delta_h) | (ds > delta_s) | (dv > delta_v)
    white_changed = dv > delta_v
    black_changed = dv > delta_v
    gray_changed = dv > delta_v

    changed = (
        class_mismatch |
        (both_color & color_changed) |
        (both_white & white_changed) |
        (both_black & black_changed) |
        (both_gray & gray_changed)
    ) & valid
    changed_mask = np.where(changed, 255, 0).astype(np.uint8)

    if valid_count == 0:
        if return_changed_mask:
            return 0.0, changed_mask
        return 0.0

    ratio = float(np.sum(changed) / valid_count)
    if return_changed_mask:
        return ratio, changed_mask
    return ratio


def classify_shape(vertices: int, contour: np.ndarray) -> str:
    """Classify a shape based on number of vertices and contour properties."""
    if vertices == 3:
        return 'triangle'
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.9 <= aspect_ratio <= 1.1:
            return 'square'
        else:
            return 'rectangle'
    elif vertices == 5:
        return 'pentagon'
    elif vertices == 6:
        return 'hexagon'
    elif vertices > 6:
        # Check circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.7:
                return 'circle'
    
    return 'polygon'


def count_objects_by_color(img: np.ndarray, target_color_bgr: Tuple[int, int, int], tolerance: int = 30) -> int:
    """
    Count distinct objects of a specific color in an image.
    
    Args:
        img: Input image (BGR)
        target_color_bgr: Target color as (B, G, R)
        tolerance: Color matching tolerance
        
    Returns:
        Number of detected objects
    """
    # Create mask for target color
    lower = np.array([max(0, c - tolerance) for c in target_color_bgr])
    upper = np.array([min(255, c + tolerance) for c in target_color_bgr])
    mask = cv2.inRange(img, lower, upper)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by minimum area
    min_area = 50
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    return len(valid_contours)


# ============================================================================
# Motion and Flow Utilities
# ============================================================================

def compute_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute optical flow between two frames.
    
    Returns:
        Tuple of (flow array, average magnitude)
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    
    return flow, float(np.mean(magnitude))


def compute_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute normalized difference between two frames."""
    if frame1.shape != frame2.shape:
        frame2 = normalize_frame_size(frame2, frame1)
    
    diff = cv2.absdiff(frame1, frame2)
    return float(np.mean(diff) / 255.0)


def detect_motion_regions(frame1: np.ndarray, frame2: np.ndarray, threshold: int = 30) -> List[Tuple[int, int, int, int]]:
    """
    Detect regions with motion between two frames.
    
    Returns:
        List of bounding boxes (x, y, w, h) for motion regions
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Dilate to fill gaps
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))
    
    return bboxes


# ============================================================================
# Score Calculation Utilities
# ============================================================================

def linear_score(value: float, min_val: float, max_val: float, invert: bool = False) -> float:
    """
    Calculate a linear score between 0 and 1.
    
    Args:
        value: The value to score
        min_val: Value that maps to 0 (or 1 if inverted)
        max_val: Value that maps to 1 (or 0 if inverted)
        invert: If True, higher values give lower scores
        
    Returns:
        Score between 0 and 1
    """
    if max_val == min_val:
        return 0.5
    
    score = (value - min_val) / (max_val - min_val)
    score = max(0, min(1, score))
    
    if invert:
        score = 1 - score
    
    return score


def threshold_score(value: float, thresholds: List[Tuple[float, float]]) -> float:
    """
    Calculate score based on threshold ranges.
    
    Args:
        value: The value to score
        thresholds: List of (threshold, score) pairs, sorted by threshold ascending
                   Score is interpolated between thresholds
                   
    Returns:
        Score between 0 and 1
    """
    if not thresholds:
        return 0.5
    
    # Sort by threshold
    thresholds = sorted(thresholds, key=lambda x: x[0])
    
    if value <= thresholds[0][0]:
        return thresholds[0][1]
    if value >= thresholds[-1][0]:
        return thresholds[-1][1]
    
    # Interpolate
    for i in range(len(thresholds) - 1):
        t1, s1 = thresholds[i]
        t2, s2 = thresholds[i + 1]
        if t1 <= value <= t2:
            ratio = (value - t1) / (t2 - t1)
            return s1 + ratio * (s2 - s1)
    
    return 0.5


def weighted_average(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculate weighted average of scores.
    
    Args:
        scores: Dictionary of dimension -> score
        weights: Dictionary of dimension -> weight (should sum to 1 or will be normalized)
        
    Returns:
        Weighted average score
    """
    total_weight = sum(weights.get(dim, 0) for dim in scores)
    if total_weight == 0:
        return sum(scores.values()) / len(scores) if scores else 0
    
    weighted_sum = sum(scores[dim] * weights.get(dim, 0) for dim in scores)
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# VBVR-Pro (v2) metadata.json -> legacy annotation items adapter.
#
# The CircleSelection / bbox evaluators expect the legacy external annotation
# shape  {"correct_items": [{"bbox"|"line": [...]}], "wrong_items": [...]}  with
# NORMALIZED ([0,1]) coordinates. VBVR-Pro stores the same information under
# `semantic_ground_truth` (per-task schema). The converters below translate the
# v2 schema into that legacy shape, keyed by `semantic_ground_truth.task_type`.
# ---------------------------------------------------------------------------

def _pro_canvas_wh(meta, sgt):
    """Best-effort canvas pixel size (defaults to 1024x1024)."""
    for src in (sgt.get('canvas'), sgt.get('canvas_px')):
        if isinstance(src, dict) and 'width' in src:
            return float(src['width']), float(src['height'])
        if isinstance(src, (list, tuple)) and len(src) == 2:
            return float(src[0]), float(src[1])
    gdr = (meta.get('generic_declarative_render') or {}).get('canvas') or {}
    if 'width' in gdr:
        return float(gdr['width']), float(gdr['height'])
    return 1024.0, 1024.0


def _pro_bbox_center(cx, cy, half_w, half_h, W, H):
    return {'bbox': [(cx - half_w) / W, (cy - half_h) / H,
                     (cx + half_w) / W, (cy + half_h) / H]}


def _pro_bbox_xywh(x, y, w, h, W, H, pad=0.0):
    return {'bbox': [(x - pad) / W, (y - pad) / H, (x + w + pad) / W, (y + h + pad) / H]}


def _pro_line(p1, p2, W, H):
    return {'line': [p1[0] / W, p1[1] / H, p2[0] / W, p2[1] / H]}


def _pro_largest_number(meta, sgt):  # G-160
    W, H = _pro_canvas_wh(meta, sgt)
    half = (sgt.get('red_circle_around_largest') or {}).get('max_radius_pixels', 60) * 0.55
    tgt = sgt['largest_value_index']
    cor, wro = [], []
    for i, n in enumerate(sgt['numbers']):
        x, y = n['position']
        (cor if i == tgt else wro).append(_pro_bbox_center(x, y, half, half, W, H))
    return cor, wro


def _pro_largest_angle(meta, sgt):  # G-218
    W, H = _pro_canvas_wh(meta, sgt)
    half = (sgt.get('red_circle') or {}).get('radius_px', 40)
    tgt = sgt['target_vertex_index']
    cor, wro = [], []
    for v in sgt['vertices']:
        x, y = v['xy']
        (cor if v['index'] == tgt else wro).append(_pro_bbox_center(x, y, half, half, W, H))
    return cor, wro


def _pro_chinese_char(meta, sgt):  # G-247
    W, H = _pro_canvas_wh(meta, sgt)
    half = 70.0
    cor, wro = [], []
    for c in sgt['all_characters']:
        x, y = c['position']
        (cor if c.get('is_chinese') else wro).append(_pro_bbox_center(x, y, half, half, W, H))
    return cor, wro


def _pro_nearest_square_rect(meta, sgt):  # G-168
    W, H = _pro_canvas_wh(meta, sgt)
    tgt = sgt['answer_index']
    cor, wro = [], []
    for r in sgt['rectangles']:
        x, y, w, h = r['axis_aligned_bbox_xywh']
        (cor if r['index'] == tgt else wro).append(_pro_bbox_xywh(x, y, w, h, W, H, pad=2))
    return cor, wro


def _pro_longest_polygon_side(meta, sgt):  # G-167
    # The mark is a small circle on the longest edge's midpoint. A full-edge bbox
    # is huge for a diagonal edge (tiny overlap), so target the midpoint directly;
    # the other edges stay as line distractors.
    W, H = _pro_canvas_wh(meta, sgt)
    tgt = sgt['longest_side_index']
    mid = sgt.get('marking_target_midpoint')
    cor = [_pro_bbox_center(mid[0], mid[1], 30, 30, W, H)] if mid else []
    wro = [_pro_line(e[0], e[1], W, H) for i, e in enumerate(sgt['edges']) if i != tgt]
    return cor, wro


def _pro_seg_bbox(p1, p2, half_thick, W, H):
    """Axis-aligned bbox around a line segment (matches v1 segment annotations)."""
    x1, y1 = p1
    x2, y2 = p2
    lo_x, hi_x = min(x1, x2) - half_thick, max(x1, x2) + half_thick
    lo_y, hi_y = min(y1, y2) - half_thick, max(y1, y2) + half_thick
    return {'bbox': [lo_x / W, lo_y / H, hi_x / W, hi_y / H]}


def _pro_wave_peaks(meta, sgt):  # G-202 (every peak is a target; no distractors)
    W, H = _pro_canvas_wh(meta, sgt)
    # Small box that sits INSIDE the marker ring so overlap(box, ring) -> 1.0.
    half = max(sgt.get('marker_ring_radius_px', 15) * 0.5, 5.0)
    return [_pro_bbox_center(p[0], p[1], half, half, W, H) for p in sgt['peaks']], []


def _pro_segment_intersection(meta, sgt):  # G-169 (single target, no distractors)
    W, H = _pro_canvas_wh(meta, sgt)
    half = max((sgt.get('rendering') or {}).get('circle_radius_px', 40) * 0.45, 8.0)
    ix, iy = sgt['intersection']
    return [_pro_bbox_center(ix, iy, half, half, W, H)], []


def _pro_tangent_point(meta, sgt):  # G-222 (single target, no distractors)
    W, H = _pro_canvas_wh(meta, sgt)
    half = max((sgt.get('expected_mark') or {}).get('radius_px', 12) * 0.5, 8.0)
    tx, ty = sgt['tangent_point_px']
    return [_pro_bbox_center(tx, ty, half, half, W, H)], []


def _pro_horizontal_lines(meta, sgt):  # G-223 (segment bboxes, not line masks)
    W, H = _pro_canvas_wh(meta, sgt)
    cor = [_pro_seg_bbox(s['start'], s['end'], s.get('thickness', 6), W, H)
           for s in sgt.get('horizontal_segments', [])]
    wro = [_pro_seg_bbox(s['start'], s['end'], s.get('thickness', 6), W, H)
           for s in sgt.get('non_horizontal_segments', [])]
    return cor, wro


def _pro_points_in_overlap(meta, sgt):  # G-136 (target dots only; no distractors)
    W, H = _pro_canvas_wh(meta, sgt)
    half = 9.0
    targets = sgt.get('target_points') or []
    cor = []
    for p in targets:
        xy = p if isinstance(p, (list, tuple)) else (p.get('xy') or p.get('position'))
        cor.append(_pro_bbox_center(xy[0], xy[1], half, half, W, H))
    return cor, []


def _pro_incorrect_arrow(meta, sgt):  # G-212
    W, H = _pro_canvas_wh(meta, sgt)
    objs = sgt.get('objects') or []
    tgt = sgt.get('incorrect_index')
    cor, wro = [], []
    for i, o in enumerate(objs):
        role = str(o.get('role', ''))
        if 'arrow' not in role:  # skip non-arrow entries (e.g. answer_annotation)
            continue
        xy = o.get('position_px') or o.get('position')
        if xy is None:
            continue
        half = o.get('size_px', 80) * 0.45
        is_target = (i == tgt) or ('incorrect' in role)
        (cor if is_target else wro).append(_pro_bbox_center(xy[0], xy[1], half, half, W, H))
    return cor, wro


def _pro_layer_center_half(L):
    """(center_xy, half_extent) for a declarative render layer, or (None, None)."""
    if 'cx_px' in L:  # circle
        return (L['cx_px'], L['cy_px']), L.get('radius_px', 20)
    if 'points_px' in L and L.get('points_px'):  # polygon
        xs = [p[0] for p in L['points_px']]
        ys = [p[1] for p in L['points_px']]
        return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2), \
               max(max(xs) - min(xs), max(ys) - min(ys)) / 2
    if 'x_px' in L and 'w_px' in L:  # rect
        return (L['x_px'] + L['w_px'] / 2, L['y_px'] + L['h_px'] / 2), \
               max(L['w_px'], L['h_px']) / 2
    return None, None


def _pro_select_next_figure(meta, sgt):  # G-131
    W, H = _pro_canvas_wh(meta, sgt)
    layers = ((meta.get('generic_declarative_render') or {}).get('first_frame') or {}).get('layers', [])
    opts = []
    for L in layers:
        if L.get('type') not in ('filled_shape', 'stroked_shape'):
            continue
        if 'dash_pattern' in L:  # skip the dashed option-slot frames
            continue
        fill = L.get('fill_rgb')
        if fill and min(fill) >= 200:  # skip light-gray option-slot backgrounds
            continue
        c, half = _pro_layer_center_half(L)
        if c is None or c[1] < 0.74 * H:  # bottom option row only
            continue
        opts.append((c, half))
    opts.sort(key=lambda o: o[0][0])  # left-to-right
    tgt = sgt.get('correct_option_index', 0)
    cor, wro = [], []
    for i, (c, half) in enumerate(opts):
        (cor if i == tgt else wro).append(_pro_bbox_center(c[0], c[1], half, half, W, H))
    return cor, wro


_PRO_CONVERTERS = {
    'find_incorrect_arrow_direction': _pro_incorrect_arrow,
    'select_next_figure_increasing_size_sequence': _pro_select_next_figure,
    'select_next_figure_large_small_alternating_sequence': _pro_select_next_figure,
    'select_next_figure_small_large_alternating_sequence': _pro_select_next_figure,
    'G-160_circle_largest_numerical_value': _pro_largest_number,
    'identify_largest_angle_in_triangle': _pro_largest_angle,
    'identify_chinese_character': _pro_chinese_char,
    'G-168_identify_nearest_to_square_rectangle': _pro_nearest_square_rect,
    'G-167_select_longest_polygon_side': _pro_longest_polygon_side,
    'mark_wave_peaks': _pro_wave_peaks,
    'locate_intersection_of_segments': _pro_segment_intersection,
    'mark_tangent_point_of_circles': _pro_tangent_point,
    'highlight_horizontal_lines': _pro_horizontal_lines,
    'locate_point_in_overlapping_area': _pro_points_in_overlap,
}


def coerce_meta_paths(meta_file_path):
    """Normalize a metafile spec (str | list | None) into a list of candidate paths."""
    if meta_file_path is None:
        return []
    if isinstance(meta_file_path, (list, tuple)):
        return [p for p in meta_file_path if p]
    return [meta_file_path]


def pro_metadata_to_items(meta):
    """VBVR-Pro metadata.json -> {'correct_items','wrong_items'} (normalized) or None."""
    sgt = meta.get('semantic_ground_truth') if isinstance(meta, dict) else None
    if not isinstance(sgt, dict):
        return None
    conv = _PRO_CONVERTERS.get(sgt.get('task_type'))
    if conv is None:
        return None
    try:
        correct, wrong = conv(meta, sgt)
    except Exception:
        return None
    return {'correct_items': correct, 'wrong_items': wrong}


class CircleSelectionProcessor:
    """
    Processor for circle-selection tasks where target shapes are selected by circles.

    Inputs:
        - Config params in __init__:
            - meta_file_path: optional meta json path. If provided, use its bbox/line labels
              (`correct_items`/`wrong_items`) as target annotations.
            - circle_color / circle_fill_max_ratio: controls circle detection by color 
              and fill ratio. 
            - circle_hsv_tolerance: tolerance for circle color (h_tolerance, min_s, min_v)
            - foreground_hsv_delta_tolerance / background_hsv_delta_tolerance: 
              HSV change thresholds for colored regions of foreground / background. 
              (delta_h, delta_s, delta_v). black / white regions calculated automatically.
            - foreground_enlarge_pixels: optional dilation size for foreground eval mask.
            - consistency_foreground_seperate: if True, compute foreground consistency
              per-shape instead of union mask.
            - consistency_forground_remove_bg: optional color name. If set, pixels that
              are this color in both reference and target will be removed from each
              foreground eval mask before calling _hsv_change_ratio.
            - consistency_foreground_min_area: minimum valid pixels required for one
              foreground eval mask. Masks with valid area below this threshold are ignored.
        - Runtime params in process(...):
            - gt_first_image: first GT frame (BGR ndarray), used for shape extraction.
            - gt_last_image: last GT frame (BGR ndarray), used for GT circle reference.
            - pred_last_image: last predicted frame (BGR ndarray), used for scoring.
            - debug_dir: optional output folder for debug visualizations.

    Outputs:
        process(...) returns Dict[str, Any] with:
            - foreground_change_ratio / background_change_ratio: ratio of pixels exceeding
              HSV change thresholds.
            - circle_vs_shape_overlap: overlap matrix of predicted circles vs foreground shapes. 
              ratio = IOU / area(shape)
            - circle_vs_background_overlap: overlap ratio of each predicted circle with
              background. ratio = IOU / area(circle)
            - pred_circle_shape_types / foreground_shape_types: estimated shape type of 
              generated circles and foreground shapes.
            - is_target_shape: 0/1 labels for each foreground shape.
    """

    def __init__(
        self,
        meta_file_path: Optional[str] = None,
        circle_color: str = 'red',
        circle_fill_max_ratio: float = 0.5,
        circle_hsv_tolerance: Tuple[int, int, int] = (20, 80, 80),
        foreground_hsv_delta_tolerance: Tuple[float, float, float] = (20.0, 20.0, 20.0),
        background_hsv_delta_tolerance: Tuple[float, float, float] = (20.0, 20.0, 20.0),
        foreground_enlarge_pixels: int = 0,
        consistency_foreground_seperate: bool = False,
        consistency_forground_remove_bg: Optional[str] = None,
        consistency_foreground_min_area: int = 0,
    ):
        self.meta_file_path = meta_file_path
        self.circle_color = circle_color
        self.circle_fill_max_ratio = circle_fill_max_ratio
        if len(circle_hsv_tolerance) != 3:
            raise ValueError("circle_hsv_tolerance must be a 3-tuple: (h_tolerance, min_s, min_v)")
        self.circle_h_tolerance = int(max(0, min(90, circle_hsv_tolerance[0])))
        self.circle_min_s = int(max(0, min(255, circle_hsv_tolerance[1])))
        self.circle_min_v = int(max(0, min(255, circle_hsv_tolerance[2])))

        if len(foreground_hsv_delta_tolerance) != 3:
            raise ValueError(
                "foreground_hsv_delta_tolerance must be a 3-tuple: (delta_h, delta_s, delta_v)"
            )
        if len(background_hsv_delta_tolerance) != 3:
            raise ValueError(
                "background_hsv_delta_tolerance must be a 3-tuple: (delta_h, delta_s, delta_v)"
            )
        self.foreground_hsv_delta_tolerance = tuple(float(max(0.0, v)) for v in foreground_hsv_delta_tolerance)
        self.background_hsv_delta_tolerance = tuple(float(max(0.0, v)) for v in background_hsv_delta_tolerance)
        self.foreground_enlarge_pixels = int(max(0, foreground_enlarge_pixels))
        self.consistency_foreground_seperate = bool(consistency_foreground_seperate)
        self.consistency_forground_remove_bg = (
            str(consistency_forground_remove_bg).lower().strip()
            if consistency_forground_remove_bg is not None
            else None
        )
        if self.consistency_forground_remove_bg not in (None, "", "white", "black", "gray", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"):
            raise ValueError(
                "consistency_forground_remove_bg must be one of: "
                "white/black/gray/red/orange/yellow/green/cyan/blue/purple/magenta"
            )
        self.consistency_foreground_min_area = int(max(0, consistency_foreground_min_area))

    def _extract_shape_mask_from_bbox(self, gt_first_image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Extract one foreground shape mask from a normalized bbox."""
        h, w = gt_first_image.shape[:2]
        if len(bbox) != 4:
            raise ValueError("bbox must have 4 values: [w1, h1, w2, h2]")

        x1 = int(round(max(0.0, min(1.0, bbox[0])) * w))
        y1 = int(round(max(0.0, min(1.0, bbox[1])) * h))
        x2 = int(round(max(0.0, min(1.0, bbox[2])) * w))
        y2 = int(round(max(0.0, min(1.0, bbox[3])) * h))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    def _extract_shape_mask_from_line(
        self,
        gt_first_image: np.ndarray,
        line: List[float],
        line_width_pixels: int = 10,
        direction: str = 'left'
    ) -> np.ndarray:
        """Extract one foreground shape mask from a normalized line segment."""
        h, w = gt_first_image.shape[:2]
        if len(line) != 4:
            raise ValueError("line must have 4 values: [w1, h1, w2, h2]")

        x1 = int(round(max(0.0, min(1.0, line[0])) * max(0, w - 1)))
        y1 = int(round(max(0.0, min(1.0, line[1])) * max(0, h - 1)))
        x2 = int(round(max(0.0, min(1.0, line[2])) * max(0, w - 1)))
        y2 = int(round(max(0.0, min(1.0, line[3])) * max(0, h - 1)))
        thickness = max(1, int(round(line_width_pixels)))

        mask = np.zeros((h, w), dtype=np.uint8)
        if direction == 'left':
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=thickness)
        elif direction == 'right':
            cv2.line(mask, (x2, y1), (x1, y2), 255, thickness=thickness)
        else:
            raise ValueError("direction must be one of: left, right")
        return mask

    def read_meta(self, meta_file_path: str, gt_first_image: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Read meta file and return foreground masks + target labels.

        Meta format:
        {
          "correct_items": [{"bbox": [w1, h1, w2, h2]} or {"line": [w1, h1, w2, h2]}, ...],
          "wrong_items":   [{"bbox": [w1, h1, w2, h2]} or {"line": [w1, h1, w2, h2]}, ...]
        }
        """
        # `meta_file_path` may be an ordered list of candidate sources. Scan it
        # and use the first that holds usable labels: legacy `correct_items`
        # (v1 annotations), or a convertible VBVR-Pro `semantic_ground_truth`
        # (v2 in-bench metadata.json). A source with neither (e.g. a v1
        # metadata.json that only has `parameters`) is skipped.
        meta = {}
        for _p in coerce_meta_paths(meta_file_path):
            if not os.path.exists(_p):
                continue
            try:
                _m = load_json(_p)
            except Exception:
                continue
            if 'correct_items' in _m:
                meta = _m
                break
            if 'semantic_ground_truth' in _m:
                converted = pro_metadata_to_items(_m)
                if converted is not None:
                    meta = converted
                    break
        foreground_shape_masks: List[np.ndarray] = []
        is_target_shape: List[int] = []

        for item in meta.get('correct_items', []):
            if 'line' in item:
                foreground_shape_masks.append(
                    self._extract_shape_mask_from_line(gt_first_image, item.get('line', []), direction=item.get('direction', 'left'))
                )
            else:
                bbox = item.get('bbox', [])
                foreground_shape_masks.append(self._extract_shape_mask_from_bbox(gt_first_image, bbox))
            is_target_shape.append(1)
        
        for item in meta.get('wrong_items', []):
            if 'line' in item:
                foreground_shape_masks.append(
                    self._extract_shape_mask_from_line(gt_first_image, item.get('line', []), direction=item.get('direction', 'left'))
                )
            else:
                bbox = item.get('bbox', [])
                foreground_shape_masks.append(self._extract_shape_mask_from_bbox(gt_first_image, bbox))
            is_target_shape.append(0)

        return foreground_shape_masks, is_target_shape

    def _contour_to_mask(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        return mask

    def _compute_shared_color_mask(
        self,
        reference_image: np.ndarray,
        target_image: np.ndarray,
        color_name: str,
    ) -> np.ndarray:
        """Mask of pixels that are the given color in both reference and target."""
        hsv_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV)
        hsv_tgt = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)
        ref_mask = color_name_to_hsv_tolerance(hsv_ref, color_name=color_name, tolerance=10)
        tgt_mask = color_name_to_hsv_tolerance(hsv_tgt, color_name=color_name, tolerance=10)
        return np.where((ref_mask > 0) & (tgt_mask > 0), 255, 0).astype(np.uint8)

    def process(
        self,
        gt_first_image: np.ndarray,
        gt_last_image: np.ndarray,
        pred_last_image: np.ndarray,
        debug_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        if gt_last_image.shape != gt_first_image.shape:
            gt_last_image = normalize_frame_size(gt_last_image, gt_first_image)
        if pred_last_image.shape != gt_first_image.shape:
            pred_last_image = normalize_frame_size(pred_last_image, gt_first_image)

        # 1) Detect circles in GT last frame and prediction last frame.
        gt_circles = detect_circles(
            gt_last_image,
            color_name=self.circle_color,
            color_tolerance=self.circle_h_tolerance,
            color_min_saturation=self.circle_min_s,
            color_min_value=self.circle_min_v,
            fill_max_ratio=self.circle_fill_max_ratio
        )
        pred_circles = detect_circles(
            pred_last_image,
            color_name=self.circle_color,
            color_tolerance=self.circle_h_tolerance,
            color_min_saturation=self.circle_min_s,
            color_min_value=self.circle_min_v,
            fill_max_ratio=self.circle_fill_max_ratio
        )

        # 2) Build candidate foreground shapes and target labels.
        foreground_shape_masks: Optional[List[np.ndarray]] = None
        is_target_shape: List[int] = []
        if self.meta_file_path is not None:
            foreground_shape_masks, is_target_shape = self.read_meta(self.meta_file_path, gt_first_image)
            if not foreground_shape_masks:
                # No usable labels in any meta source (e.g. no converter for this
                # task_type) -> fall back to image-based shape detection.
                foreground_shape_masks = None
                is_target_shape = []

        foreground_shapes = detect_shapes_white_background(gt_first_image, preload_masks=foreground_shape_masks)

        if not is_target_shape:
            is_target_shape = [0] * len(foreground_shapes)
            for gt_circle in gt_circles:
                if not foreground_shapes:
                    break
                nearest_idx = int(np.argmin([
                    safe_distance(gt_circle['center'], fg_shape['center'])
                    for fg_shape in foreground_shapes
                ]))
                is_target_shape[nearest_idx] = 1

        if len(is_target_shape) == 0:
            raise ValueError(
                "is_target_shape is empty. Please check meta file or shape detection results."
            )
        if int(np.sum(is_target_shape)) <= 0:
            raise ValueError(
                "is_target_shape has no positive label. Expected sum(is_target_shape) > 0."
            )

        # Convert each foreground contour to one mask.
        h, w = gt_first_image.shape[:2]
        fg_shape_masks = [self._contour_to_mask(s['contour'], (h, w)) for s in foreground_shapes]

        # 3) Foreground union (enlarged) and background (non-shape area before enlarge).
        fg_union = np.zeros((h, w), dtype=np.uint8)
        for m in fg_shape_masks:
            fg_union = cv2.bitwise_or(fg_union, m)

        if self.foreground_enlarge_pixels > 0:
            k = 2 * self.foreground_enlarge_pixels + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            foreground_mask = cv2.dilate(fg_union, kernel, iterations=1)
        else:
            foreground_mask = fg_union.copy()

        background_mask = np.where(fg_union == 0, 255, 0).astype(np.uint8)

        # 4) Remove detected circle-color regions from eval masks.
        # For each detected circle, keep only pixels inside contour that match circle color.
        hsv_pred_last = cv2.cvtColor(pred_last_image, cv2.COLOR_BGR2HSV)
        circle_color_candidate_mask = color_name_to_hsv_tolerance(
            hsv_pred_last,
            self.circle_color,
            self.circle_h_tolerance,
            min_saturation=self.circle_min_s,
            min_value=self.circle_min_v,
        )
        circle_color_mask = np.zeros((h, w), dtype=np.uint8)
        for pred_circle in pred_circles:
            contour = pred_circle.get('contour', None)
            if contour is None:
                continue
            contour_mask = self._contour_to_mask(contour, (h, w))
            circle_in_color_mask = np.where(
                (contour_mask > 0) & (circle_color_candidate_mask > 0),
                255,
                0
            ).astype(np.uint8)
            circle_color_mask = cv2.bitwise_or(circle_color_mask, circle_in_color_mask)
        circle_color_mask_ratio = float(np.sum(circle_color_mask > 0) / float(h * w)) if (h * w) > 0 else 0.0

        foreground_remove_bg_mask = None
        if self.consistency_forground_remove_bg is not None and self.consistency_forground_remove_bg != "":
            foreground_remove_bg_mask = self._compute_shared_color_mask(
                gt_first_image,
                pred_last_image,
                self.consistency_forground_remove_bg,
            )

        # Build a unified foreground mask list, then process with one loop.
        foreground_input_masks: List[np.ndarray] = []
        if self.consistency_foreground_seperate:
            if self.foreground_enlarge_pixels > 0:
                k = 2 * self.foreground_enlarge_pixels + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                for shape_mask in fg_shape_masks:
                    foreground_input_masks.append(cv2.dilate(shape_mask, kernel, iterations=1))
            else:
                foreground_input_masks = [m.copy() for m in fg_shape_masks]
        else:
            foreground_input_masks = [foreground_mask]

        foreground_eval_masks: List[np.ndarray] = []
        for fg_mask in foreground_input_masks:
            eval_mask = np.where((fg_mask > 0) & (circle_color_mask == 0), 255, 0).astype(np.uint8)
            if foreground_remove_bg_mask is not None:
                eval_mask = np.where(
                    (eval_mask > 0) & (foreground_remove_bg_mask == 0),
                    255,
                    0
                ).astype(np.uint8)
            eval_area = int(np.sum(eval_mask > 0))
            if eval_area < self.consistency_foreground_min_area:
                continue
            foreground_eval_masks.append(eval_mask)

        foreground_change_ratios: List[float] = []
        foreground_changed_masks: List[np.ndarray] = []
        for eval_mask in foreground_eval_masks:
            ratio, changed_mask = _hsv_change_ratio(
                gt_first_image,
                pred_last_image,
                eval_mask,
                self.foreground_hsv_delta_tolerance,
                return_changed_mask=True,
            )
            foreground_change_ratios.append(ratio)
            foreground_changed_masks.append(changed_mask)

        if self.consistency_foreground_seperate:
            foreground_change_ratio = foreground_change_ratios if len(foreground_change_ratios) > 0 else 0.0
        else:
            foreground_change_ratio = float(foreground_change_ratios[0]) if len(foreground_change_ratios) > 0 else 0.0

        background_eval_mask = np.where((background_mask > 0) & (circle_color_mask == 0), 255, 0).astype(np.uint8)

        background_change_ratio, background_changed_mask = _hsv_change_ratio(
            gt_first_image,
            pred_last_image,
            background_eval_mask,
            self.background_hsv_delta_tolerance,
            return_changed_mask=True,
        )

        # 5) Overlap ratio: each pred circle vs each foreground shape.
        circle_vs_shape_overlap: List[List[float]] = []
        for pc in pred_circles:
            circle_mask = self._contour_to_mask(pc['contour'], (h, w)) > 0
            row = []
            for shape_mask in fg_shape_masks:
                shape_bool = shape_mask > 0
                shape_area = int(np.sum(shape_bool))
                if shape_area == 0:
                    row.append(0.0)
                    continue
                inter = int(np.sum(circle_mask & shape_bool))
                row.append(float(inter / shape_area))
            circle_vs_shape_overlap.append(row)

        # 6) Overlap ratio: each pred circle vs background, normalized by circle area.
        circle_vs_background_overlap: List[float] = []
        background_bool = background_mask > 0
        for pc in pred_circles:
            circle_mask = self._contour_to_mask(pc['contour'], (h, w)) > 0
            circle_area = int(np.sum(circle_mask))
            if circle_area == 0:
                circle_vs_background_overlap.append(0.0)
                continue
            inter = int(np.sum(circle_mask & background_bool))
            circle_vs_background_overlap.append(float(inter / circle_area))

        # Build per-shape masks for prediction typing:
        # 1) start from original fg_shape_masks
        # 2) enlarge each mask by foreground_enlarge_pixels
        # 3) remove current circle-color pixels from each mask
        pred_shape_input_masks: List[np.ndarray] = []
        if self.foreground_enlarge_pixels > 0:
            k = 2 * self.foreground_enlarge_pixels + 1
            pred_shape_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            enlarged_fg_shape_masks = [
                cv2.dilate(shape_mask, pred_shape_kernel, iterations=1) for shape_mask in fg_shape_masks
            ]
        else:
            enlarged_fg_shape_masks = [shape_mask.copy() for shape_mask in fg_shape_masks]

        for shape_mask in enlarged_fg_shape_masks:
            pred_shape_mask = np.where(
                (shape_mask > 0) & (circle_color_mask == 0),
                255,
                0
            ).astype(np.uint8)
            pred_shape_input_masks.append(pred_shape_mask)

        pred_foreground_shapes = detect_shapes_white_background(
            pred_last_image,
            preload_masks=pred_shape_input_masks,
            preload_bg_remove=True
        )
        
        if debug_dir is not None:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)

            # Save key input frames for debugging.
            cv2.imwrite(str(debug_path / "gt_first.png"), gt_first_image)
            cv2.imwrite(str(debug_path / "gt_last.png"), gt_last_image)
            cv2.imwrite(str(debug_path / "pred_last.png"), pred_last_image)

            # 1) Save eval/changed overlay maps:
            #    black: not evaluated, white: evaluated, red: changed(>bar)
            if self.consistency_foreground_seperate:
                for shape_idx, (fg_eval_mask, fg_changed_mask) in enumerate(
                    zip(foreground_eval_masks, foreground_changed_masks)
                ):
                    fg_eval_vis = np.zeros((h, w, 3), dtype=np.uint8)
                    fg_eval_vis[fg_eval_mask > 0] = (255, 255, 255)
                    fg_eval_vis[fg_changed_mask > 0] = (0, 0, 255)
                    cv2.imwrite(str(debug_path / f"foreground_eval_changed_overlay_shape_{shape_idx}.png"), fg_eval_vis)
            else:
                if len(foreground_eval_masks) > 0:
                    fg_eval_vis = np.zeros((h, w, 3), dtype=np.uint8)
                    fg_eval_vis[foreground_eval_masks[0] > 0] = (255, 255, 255)
                    fg_eval_vis[foreground_changed_masks[0] > 0] = (0, 0, 255)
                    cv2.imwrite(str(debug_path / "foreground_eval_changed_overlay.png"), fg_eval_vis)

            bg_eval_vis = np.zeros((h, w, 3), dtype=np.uint8)
            bg_eval_vis[background_eval_mask > 0] = (255, 255, 255)
            bg_eval_vis[background_changed_mask > 0] = (0, 0, 255)
            cv2.imwrite(str(debug_path / "background_eval_changed_overlay.png"), bg_eval_vis)

            # 2) For each circle, draw one debug image:
            #    - circle contour: thin black line
            #    - each foreground shape contour: thin black line + overlap label + target flag
            circle_line_color = (0, 0, 0)
            for circle_idx, pc in enumerate(pred_circles):
                vis = pred_last_image.copy()
                cv2.drawContours(vis, [pc['contour']], -1, circle_line_color, 1)
                circle_cx, circle_cy = pc.get('center', (0, 0))
                circle_shape_type = str(pc.get('type', 'unknown'))
                cv2.putText(
                    vis,
                    f"circle: {circle_shape_type}",
                    (int(circle_cx) + 2, int(circle_cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    circle_line_color,
                    1,
                    cv2.LINE_AA
                )

                overlaps = circle_vs_shape_overlap[circle_idx] if circle_idx < len(circle_vs_shape_overlap) else []
                for shape_idx, shape in enumerate(foreground_shapes):
                    cv2.drawContours(vis, [shape['contour']], -1, (0, 0, 0), 1)
                    ratio = overlaps[shape_idx] if shape_idx < len(overlaps) else 0.0
                    target_flag = int(is_target_shape[shape_idx]) if shape_idx < len(is_target_shape) else 0
                    cx, cy = shape.get('center', (0, 0))
                    shape_type = str(shape.get('type', 'unknown'))
                    text_lines = [
                        f"{shape_type}",
                        f"t={target_flag}",
                        f"{ratio:.3f}",
                    ]
                    for line_idx, line_text in enumerate(text_lines):
                        cv2.putText(
                            vis,
                            line_text,
                            (int(cx) + 2, int(cy) - 2 + line_idx * 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA
                        )

                cv2.imwrite(str(debug_path / f"circle_{circle_idx}_overlap.png"), vis)

        # 8) is_target_shape already prepared.
        return {
            'foreground_change_ratio': foreground_change_ratio,
            'background_change_ratio': background_change_ratio,
            'circle_vs_shape_overlap': circle_vs_shape_overlap,
            'circle_vs_background_overlap': circle_vs_background_overlap,
            'pred_circles': pred_circles,
            'foreground_shapes': foreground_shapes,
            'pred_foreground_shapes': pred_foreground_shapes,
            'is_target_shape': is_target_shape,
            'circle_color_mask_ratio': circle_color_mask_ratio
        }


# ============================================================================
# Pattern Extraction and Matching Utilities
# ============================================================================

def extract_patterns_from_white_bg(
    img: np.ndarray,
    white_threshold: int = 240,
    min_area: int = 1000,
    padding: int = 4,
) -> List[Dict]:
    """
    Extract every separate colored pattern from an image with a pure white
    background, automatically discarding grey horizontal rules.

    Steps:
    1. Build a foreground mask (non-white pixels).
    2. Detect and remove grey horizontal rules (width > 60% of image width,
       height < 5% of image height, low saturation).
    3. Run connected-component analysis on the remaining foreground and extract
       each separate pattern.

    Args:
        img: Input image, BGR, with a pure white background.
        white_threshold: Grey level above which a pixel counts as white
            background. Default 240.
        min_area: Minimum pattern area in pixels, used to reject noise.
            Default 50.
        padding: Extra margin in pixels kept around the bbox when cropping.
            Default 4.

    Returns:
        List[Dict], one entry per pattern, each containing:
        - 'contour':   np.ndarray of shape (N, 1, 2), the pattern contour
                       in OpenCV format
        - 'bbox':      Tuple[int, int, int, int], (x, y, w, h) bounding box
        - 'center':    Tuple[int, int], (cx, cy) centroid
        - 'area':      float, contour area
        - 'crop':      np.ndarray, the pattern cropped from the source image
                       (with padding, BGR)
        - 'crop_mask': np.ndarray, foreground mask the same size as 'crop'
                       (uint8, 255 = foreground)
        Sorted by area, largest first.
    """
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 1. Foreground mask: non-white pixels ---
    fg_mask = (gray < white_threshold).astype(np.uint8) * 255

    # --- 2. Detect and remove grey horizontal rules ---
    # Grey pixel: all channels in [50, 220] and channel range < 40 (low saturation)
    b, g, r = img[:, :, 0].astype(np.int32), img[:, :, 1].astype(np.int32), img[:, :, 2].astype(np.int32)
    ch_min = np.minimum(np.minimum(b, g), r)
    ch_max = np.maximum(np.maximum(b, g), r)
    gray_pixel_mask = ((ch_min >= 50) & (ch_max <= 220) & ((ch_max - ch_min) < 40)).astype(np.uint8) * 255

    # Find components in the grey mask and keep the "rule" shaped ones (width > 60% of image width, height < 5% of image height)
    n_gray, labels_gray, stats_gray, _ = cv2.connectedComponentsWithStats(gray_pixel_mask, connectivity=8)
    for lbl in range(1, n_gray):
        lw = int(stats_gray[lbl, cv2.CC_STAT_WIDTH])
        lh = int(stats_gray[lbl, cv2.CC_STAT_HEIGHT])
        if lw > w_img * 0.6 and lh < h_img * 0.05:
            # This is a grey horizontal rule; remove it from the foreground mask
            fg_mask[labels_gray == lbl] = 0

    # --- 3. Morphological closing to bridge small gaps inside a pattern ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # --- 4. Connected-component analysis ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)

    patterns = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = int(centroids[label][0]), int(centroids[label][1])

        # Extra guard: a component wider than 60% of the image and shorter than
        # 5% of it may still be a leftover rule, so skip it
        if w > w_img * 0.6 and h < h_img * 0.05:
            continue

        component_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)

        crop = img[y1:y2, x1:x2].copy()
        crop_mask = component_mask[y1:y2, x1:x2].copy()

        patterns.append({
            'contour': contour,
            'bbox': (x, y, w, h),
            'center': (cx, cy),
            'area': float(area),
            'crop': crop,
            'crop_mask': crop_mask,
        })

    patterns.sort(key=lambda p: p['area'], reverse=True)
    return patterns


def find_patterns_in_image(
    ref_img: np.ndarray,
    ref_patterns: List[Dict],
    search_img: np.ndarray,
    match_threshold: float = 0.5,
    nms_overlap: float = 0.3,
    white_threshold: int = 240,
    size_tolerance: float = 0.0,
) -> List[Dict]:
    """
    Search a target image for each pattern in a reference list, at fixed scale,
    comparing masked pixels directly.

    For each reference pattern, crop_mask restricts the comparison to foreground
    pixels. A sliding window over search_img first shortlists the top-5
    candidate positions with TM_SQDIFF_NORMED, then the normalized L1 similarity
    of the foreground pixels is computed exactly. After a pattern matches, its
    region is filled white so later patterns cannot match the same place twice.

    Args:
        ref_img: Retained for API compatibility; not used directly.
        ref_patterns: Pattern list returned by extract_patterns_from_white_bg.
        search_img: Target image to search, BGR.
        match_threshold: Similarity threshold; below this a candidate is not a
            valid match. Default 0.5. Similarity is 1 - mean_L1_diff/255, so
            values closer to 1 are more similar.
        nms_overlap: Retained for API compatibility; superseded by the
            "mask out matched regions" mechanism.
        white_threshold: Retained for API compatibility; no longer used.
        size_tolerance: Maximum fraction (0..1) of non-white pixels tolerated in
            the template's background region within a search patch. Default 0.0
            disables the extra size check, preserving the original behavior.
            With a positive value such as 0.15, a candidate whose patch has more
            than that fraction of non-white pixels outside the template's
            fg_mask -- meaning the pattern in the search image is larger than
            the template -- is treated as a size mismatch and skipped.

    Returns:
        List[Dict], one entry per successful match, each containing:
        - 'pattern_idx': int, index into ref_patterns
        - 'bbox':        Tuple[int, int, int, int], (x, y, w, h) within search_img
        - 'center':      Tuple[int, int], (cx, cy) of the match
        - 'score':       float, similarity in [0, 1]
        - 'scale':       float, always 1.0
        - 'ref_area':    float, area of the reference pattern
    """
    sh, sw = search_img.shape[:2]
    matches = []

    # Working copy (float32): matched regions are filled white so later patterns cannot match the same place
    search_work = search_img.astype(np.float32).copy()

    for pat_idx, pattern in enumerate(ref_patterns):
        crop = pattern['crop'].astype(np.float32)   # (ph, pw, 3)
        crop_mask = pattern['crop_mask']             # (ph, pw), uint8, 255=foreground
        ph, pw = crop.shape[:2]

        if pw > sw or ph > sh:
            continue

        fg_mask = (crop_mask > 0)       # bool (ph, pw)
        fg_count = int(fg_mask.sum())
        if fg_count == 0:
            continue

        # Template values at foreground pixels, shape (fg_count, 3)
        tmpl_fg = crop[fg_mask]

        # --- Shortlist: greyscale TM_SQDIFF_NORMED to find the top-K candidates ---
        crop_gray = cv2.cvtColor(pattern['crop'], cv2.COLOR_BGR2GRAY).astype(np.float32)
        search_gray = cv2.cvtColor(search_work.clip(0, 255).astype(np.uint8),
                                   cv2.COLOR_BGR2GRAY).astype(np.float32)
        mask_f = fg_mask.astype(np.float32)

        result = cv2.matchTemplate(search_gray, crop_gray, cv2.TM_SQDIFF_NORMED, mask=mask_f)

        # Top-10 candidates (lower SQDIFF is better)
        result_flat = result.ravel()
        n_cands = min(10, result_flat.size)
        top_indices = np.argpartition(result_flat, n_cands - 1)[:n_cands]
        top_indices = top_indices[np.argsort(result_flat[top_indices])]  # ascending

        best_score = -1.0
        best_loc = None

        for flat_idx in top_indices:
            ry = int(flat_idx // result.shape[1])
            rx = int(flat_idx % result.shape[1])

            patch = search_work[ry:ry + ph, rx:rx + pw]
            if patch.shape[:2] != (ph, pw):
                continue

            # Exact L1 comparison (color, foreground pixels only)
            patch_fg = patch[fg_mask]
            mean_l1 = float(np.mean(np.abs(patch_fg - tmpl_fg)))
            score = 1.0 - mean_l1 / 255.0

            # Size tolerance check: when size_tolerance > 0, test whether the
            # template's background region in the patch holds too many non-white
            # pixels, which means the search pattern is larger than the template
            if size_tolerance > 0.0:
                bg_mask = ~fg_mask  # template background region
                bg_count = int(bg_mask.sum())
                if bg_count > 0:
                    patch_bg = patch[bg_mask]  # shape (bg_count, 3)
                    non_white = float(np.mean(np.any(patch_bg < 240, axis=1)))
                    if non_white > size_tolerance:
                        continue  # pattern is larger than the template; skip

            if score > best_score:
                best_score = score
                best_loc = (rx, ry)

        if best_score < match_threshold or best_loc is None:
            continue

        x, y = best_loc

        # Fill the matched region white so later patterns cannot match here
        x2 = min(sw, x + pw)
        y2 = min(sh, y + ph)
        search_work[y:y2, x:x2] = 255.0

        matches.append({
            'pattern_idx': pat_idx,
            'bbox': (x, y, pw, ph),
            'center': (x + pw // 2, y + ph // 2),
            'score': float(best_score),
            'scale': 1.0,
            'ref_area': pattern['area'],
        })

    return matches


def extract_patterns_from_gray_bg(
    img: np.ndarray,
    min_area: int = 1000,
    padding: int = 4,
) -> List[Dict]:
    """
    Extract every separate colored pattern from an image with a grey background
    (the patterns themselves are never grey).

    A grey pixel is defined as: all three channels in [50, 220] and channel
    range < 40 (low saturation). Foreground is everything non-grey, i.e. the
    saturated or out-of-range colored patterns.

    Steps:
    1. Build the grey pixel mask.
    2. Foreground mask = NOT grey.
    3. Morphological closing to bridge small gaps inside a pattern.
    4. Connected-component analysis to extract each separate pattern.

    Args:
        img: Input image, BGR, with a grey background.
        min_area: Minimum pattern area in pixels, used to reject noise.
            Default 1000.
        padding: Extra margin in pixels kept around the bbox when cropping.
            Default 4.

    Returns:
        List[Dict], one entry per pattern, each containing:
        - 'contour':   np.ndarray of shape (N, 1, 2), the pattern contour
                       in OpenCV format
        - 'bbox':      Tuple[int, int, int, int], (x, y, w, h) bounding box
        - 'center':    Tuple[int, int], (cx, cy) centroid
        - 'area':      float, contour area
        - 'crop':      np.ndarray, the pattern cropped from the source image
                       (with padding, BGR)
        - 'crop_mask': np.ndarray, foreground mask the same size as 'crop'
                       (uint8, 255 = foreground)
        Sorted by area, largest first.
    """
    h_img, w_img = img.shape[:2]

    # --- 1. Grey pixel mask ---
    b = img[:, :, 0].astype(np.int32)
    g = img[:, :, 1].astype(np.int32)
    r = img[:, :, 2].astype(np.int32)
    ch_min = np.minimum(np.minimum(b, g), r)
    ch_max = np.maximum(np.maximum(b, g), r)
    gray_pixel_mask = (ch_min >= 20) & (ch_max <= 245) & ((ch_max - ch_min) < 60)

    # --- 2. Foreground mask: non-grey pixels ---
    fg_mask = (~gray_pixel_mask).astype(np.uint8) * 255

    # --- 3. Morphological closing to bridge small gaps inside a pattern ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # --- 4. Connected-component analysis ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)

    patterns = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = int(centroids[label][0]), int(centroids[label][1])

        component_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)

        crop = img[y1:y2, x1:x2].copy()
        crop_mask = component_mask[y1:y2, x1:x2].copy()

        patterns.append({
            'contour': contour,
            'bbox': (x, y, w, h),
            'center': (cx, cy),
            'area': float(area),
            'crop': crop,
            'crop_mask': crop_mask,
        })

    patterns.sort(key=lambda p: p['area'], reverse=True)
    return patterns


def cluster_patterns_by_shape(
    patterns: List[Dict],
    n_clusters: int = 2,
    max_iter: int = 50,
) -> List[List[Dict]]:
    """
    Split a pattern list into n_clusters groups of equal size by shape
    similarity, independent of scale.

    Intended for the case where the list holds n_clusters distinct shapes, each
    shape appearing at several sizes, and patterns of the same shape need to be
    grouped together.

    Algorithm (optimized for n_clusters=2):
    1. Compute the shape distance between every pair of patterns (mean of the
       Hu-moment I1 and I2 metrics).
    2. Take the two most distant patterns as the "representatives" of the two
       groups.
    3. Greedily assign each pattern to the group of its nearest representative.
    4. Adjust the boundary points so both groups end up the same size.

    Args:
        patterns: Pattern list from extract_patterns_from_white_bg or
                  extract_patterns_from_gray_bg. Each item must have a
                  'contour' key.
        n_clusters: Number of groups. Default 2; only n_clusters=2 is supported.
        max_iter: Retained for API compatibility; no longer used.

    Returns:
        List[List[Dict]] of length n_clusters. Each sublist holds the patterns
        assigned to that group, with their original fields preserved. Groups
        that cannot be filled -- because there are fewer patterns than
        n_clusters -- come back empty.
    """
    n = len(patterns)
    if n == 0:
        return [[] for _ in range(n_clusters)]
    if n <= n_clusters:
        result = [[p] for p in patterns]
        while len(result) < n_clusters:
            result.append([])
        return result

    if n_clusters != 2:
        # For anything other than 2, fall back to the original logic (kept for backward compatibility)
        return _cluster_patterns_kmeans(patterns, n_clusters, max_iter)

    contours = [p['contour'] for p in patterns]

    # --- 1. Compute the N x N shape distance matrix (mean of I1 and I2) ---
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d1 = cv2.matchShapes(contours[i], contours[j], cv2.CONTOURS_MATCH_I1, 0.0)
            d2 = cv2.matchShapes(contours[i], contours[j], cv2.CONTOURS_MATCH_I2, 0.0)
            d = (d1 + d2) / 2.0
            dist[i, j] = d
            dist[j, i] = d

    # --- 2. Hierarchical clustering: find the best split point ---
    # Following the single-linkage idea: sort all pairwise distances and find
    # the largest "gap" between consecutive distances, then split there so the
    # between-group distance is maximized

    # Collect all off-diagonal distances
    all_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            all_dists.append((dist[i, j], i, j))

    all_dists.sort(key=lambda x: x[0])

    # Find the largest gap between consecutive distances
    max_gap = 0.0
    best_split_idx = 0
    for k in range(len(all_dists) - 1):
        gap = all_dists[k + 1][0] - all_dists[k][0]
        if gap > max_gap:
            max_gap = gap
            best_split_idx = k

    # Split at the largest gap: pairs with distance <= all_dists[best_split_idx][0] stay together
    threshold = all_dists[best_split_idx][0]

    # --- 3. Connected-component analysis via union-find ---
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union every pair with distance <= threshold
    for d, i, j in all_dists:
        if d <= threshold:
            union(i, j)
        else:
            break

    # Collect the connected components
    components = {}
    for i in range(n):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)


    # --- 4. Use the components directly when there are exactly 2; otherwise adjust ---
    if len(components) == 2:
        comp_list = list(components.values())
        assignments = np.zeros(n, dtype=int)
        for idx in comp_list[0]:
            assignments[idx] = 0
        for idx in comp_list[1]:
            assignments[idx] = 1
    else:
        # Not exactly 2 components: assign using the most distant pair as representatives
        max_dist_idx = np.unravel_index(np.argmax(dist), dist.shape)
        rep0, rep1 = int(max_dist_idx[0]), int(max_dist_idx[1])
        assignments = np.zeros(n, dtype=int)
        for i in range(n):
            if dist[i, rep0] <= dist[i, rep1]:
                assignments[i] = 0
            else:
                assignments[i] = 1

    # --- 5. Rebalance so both groups are the same size ---
    target_size = n // 2
    group0_indices = np.where(assignments == 0)[0]
    group1_indices = np.where(assignments == 1)[0]


    # Sizes differ, so move boundary points across
    if len(group0_indices) > target_size:
        excess = len(group0_indices) - target_size
        # Take the point in Group 0 closest to Group 1 and move it to Group 1
        min_dists_to_g1 = np.min(dist[group0_indices][:, group1_indices], axis=1)
        worst_indices = np.argsort(-min_dists_to_g1)[:excess]
        for idx in worst_indices:
            assignments[group0_indices[idx]] = 1
    elif len(group1_indices) > target_size:
        excess = len(group1_indices) - target_size
        min_dists_to_g0 = np.min(dist[group1_indices][:, group0_indices], axis=1)
        worst_indices = np.argsort(-min_dists_to_g0)[:excess]
        for idx in worst_indices:
            assignments[group1_indices[idx]] = 0

    # --- 6. Assemble the result ---
    groups: List[List[Dict]] = [[], []]
    for idx, k in enumerate(assignments):
        groups[k].append(patterns[idx])

    final_sizes = [len(groups[0]), len(groups[1])]

    return groups


def _cluster_patterns_kmeans(
    patterns: List[Dict],
    n_clusters: int = 2,
    max_iter: int = 50,
) -> List[List[Dict]]:
    """
    k-medoids clustering, used when n_clusters != 2.
    """
    n = len(patterns)
    contours = [p['contour'] for p in patterns]

    # Distance matrix
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d1 = cv2.matchShapes(contours[i], contours[j], cv2.CONTOURS_MATCH_I1, 0.0)
            d2 = cv2.matchShapes(contours[i], contours[j], cv2.CONTOURS_MATCH_I2, 0.0)
            d = (d1 + d2) / 2.0
            dist[i, j] = d
            dist[j, i] = d

    # Several random restarts of k-medoids
    best_assignments = None
    best_cost = float('inf')

    import itertools
    if n <= 12:
        seed_candidates = list(itertools.combinations(range(n), n_clusters))
    else:
        rng = np.random.default_rng(42)
        seed_candidates = [
            tuple(rng.choice(n, n_clusters, replace=False))
            for _ in range(30)
        ]
        i0, j0 = np.unravel_index(dist.argmax(), dist.shape)
        seed_candidates.append((int(i0), int(j0)))

    for init_seeds in seed_candidates:
        seeds = list(init_seeds)
        assignments = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            seed_dists = dist[:, seeds]
            new_assignments = np.argmin(seed_dists, axis=1)
            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments

            new_seeds = []
            for k in range(n_clusters):
                members = np.where(assignments == k)[0]
                if len(members) == 0:
                    new_seeds.append(seeds[k])
                    continue
                intra = dist[np.ix_(members, members)]
                best_local = int(np.argmin(intra.sum(axis=1)))
                new_seeds.append(int(members[best_local]))
            seeds = new_seeds

        cost = 0.0
        for k in range(n_clusters):
            members = np.where(assignments == k)[0]
            if len(members) > 1:
                cost += dist[np.ix_(members, members)].sum() / 2.0

        if cost < best_cost:
            best_cost = cost
            best_assignments = assignments.copy()

    assignments = best_assignments

    # Assemble the result
    groups: List[List[Dict]] = [[] for _ in range(n_clusters)]
    for idx, k in enumerate(assignments):
        groups[k].append(patterns[idx])

    return groups


def calculate_list_length_penalty(gt_len: int, out_len: int, gen_len: int=None) -> float:
    """
    Penalty for the length mismatch between two lists.

    Args:
        gt_len: Ground truth list length
        out_len: Matched list length
        gen_len: Generated list length

    Returns:
        float: Penalty in [0.0, 1.0]
    """
    less_penalty, more_penalty = 1.0, 1.0

    if out_len < gt_len:
        # How many are missing
        missing = gt_len - out_len
        penalty = 1.0 - 0.3 * missing
        less_penalty = max(0.0, penalty)
    if gen_len is not None and out_len < gen_len:
        # How many are extra
        excess = gen_len - out_len
        penalty = 1.0 - 0.5 * excess
        more_penalty = max(0.0, penalty)

    return less_penalty * more_penalty
