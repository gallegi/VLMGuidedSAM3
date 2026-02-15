#!/usr/bin/env python3
"""
Generate training data for the tracking verification RL task.

Takes the output of `sam3/sam3/vlm/collect_training_data.py` (all_sequences_data.json)
and produces a training JSON file compatible with EasyR1's RLHFDataset.

Supports both:
  - SA-V validation format: JPEG frames in folders (--frames_root)
  - SA-V training format: MP4 videos (video_path stored in collected data)

Each training sample consists of:
  - N context images (frames with GREEN bbox around the correctly-tracked target)
  - 1 query image (current frame with RED bbox around SAM3's prediction)
  - Ground truth: {"assessment": "CORRECT"} or {"assessment": "INCORRECT", "boxes": [gt_bbox]}

The images are saved to disk with bbox overlays drawn on them.

Usage (JPEG format):
    python gen_tracking_verification_data.py \
        --collected_data /path/to/all_sequences_data.json \
        --frames_root /path/to/dataset/JPEGImages \
        --output_dir /path/to/output

Usage (MP4 format - video_path is read from collected data):
    python gen_tracking_verification_data.py \
        --collected_data /path/to/all_sequences_data.json \
        --output_dir /path/to/output
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image as PILImage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ===================== Video frame loading =====================

class VideoFrameLoader:
    """Load specific frames from MP4 videos with caching."""

    def __init__(self):
        self._cache = {}  # {video_path: {frame_idx: np.ndarray (RGB)}}
        self._caps = {}   # {video_path: cv2.VideoCapture}

    def _get_cap(self, video_path: str) -> Optional[cv2.VideoCapture]:
        if video_path not in self._caps:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return None
            self._caps[video_path] = cap
        return self._caps[video_path]

    def get_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """Load a specific frame from a video file (returns RGB numpy array)."""
        # Check cache first
        if video_path in self._cache and frame_idx in self._cache[video_path]:
            return self._cache[video_path][frame_idx].copy()

        cap = self._get_cap(video_path)
        if cap is None:
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning(f"Could not read frame {frame_idx} from {video_path}")
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Cache the frame
        if video_path not in self._cache:
            self._cache[video_path] = {}
        self._cache[video_path][frame_idx] = frame_rgb.copy()

        return frame_rgb

    def preload_frames(self, video_path: str, frame_indices: List[int]) -> None:
        """Preload multiple frames from a video for efficiency (sequential read)."""
        if not frame_indices:
            return

        cap = self._get_cap(video_path)
        if cap is None:
            return

        if video_path not in self._cache:
            self._cache[video_path] = {}

        # Sort for sequential access
        for idx in sorted(set(frame_indices)):
            if idx in self._cache[video_path]:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                self._cache[video_path][idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def clear_cache(self, video_path: Optional[str] = None) -> None:
        """Clear cached frames (for a specific video or all)."""
        if video_path:
            self._cache.pop(video_path, None)
        else:
            self._cache.clear()

    def close(self) -> None:
        """Release all video captures."""
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._cache.clear()


# Global video frame loader instance
_video_loader = VideoFrameLoader()


# ===================== IoU utilities =====================

def compute_bbox_iou(box1: List[int], box2: List[int]) -> float:
    """Compute 2D bounding box IoU."""
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = map(float, box1)
        X1, Y1, X2, Y2 = map(float, box2)
    except Exception:
        return 0.0
    inter_x1, inter_y1 = max(x1, X1), max(y1, Y1)
    inter_x2, inter_y2 = min(x2, X2), min(y2, Y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 1e-12 else 0.0


# ===================== Drawing utilities =====================

def draw_bbox_outline(
    frame: np.ndarray,
    bbox: List[int],
    color: Tuple[int, int, int],
    thickness: int = 3,
) -> np.ndarray:
    """
    Draw a bounding box outline on a frame (no fill).
    
    Only draws the border so visual features inside the box are preserved,
    similar to SecVOS's contour-based annotation approach.
    
    Args:
        frame: RGB image (H, W, 3)
        bbox: [x1, y1, x2, y2] in pixels
        color: RGB tuple
        thickness: Border thickness
    
    Returns:
        Frame with outline drawn
    """
    frame_out = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame_out.shape[:2]
    
    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return frame_out
    
    # Outline only — no fill, preserves visual features
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, thickness)
    
    return frame_out


def save_overlay_image(
    frame_source: Union[str, np.ndarray],
    bbox: List[int],
    color: Tuple[int, int, int],
    output_path: str,
    resize: Optional[int] = None,
) -> bool:
    """Load a frame (from path or numpy array), draw bbox overlay, and save.

    Args:
        frame_source: Either a file path (str) or an RGB numpy array.
        bbox: [x1, y1, x2, y2] in pixels.
        color: RGB color tuple.
        output_path: Where to save the overlay image.
        resize: Optional max dimension for resizing.
    """
    if isinstance(frame_source, str):
        frame = cv2.imread(frame_source)
        if frame is None:
            logger.warning(f"Could not read frame: {frame_source}")
            return False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif isinstance(frame_source, np.ndarray):
        frame_rgb = frame_source.copy()
    else:
        logger.warning(f"Unsupported frame_source type: {type(frame_source)}")
        return False

    if resize is not None:
        h, w = frame_rgb.shape[:2]
        scale = resize / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Scale bbox
        bbox = [int(v * scale) for v in bbox]

    frame_with_overlay = draw_bbox_outline(frame_rgb, bbox, color)

    # Save as JPEG
    pil_img = PILImage.fromarray(frame_with_overlay)
    pil_img.save(output_path, quality=90)
    return True


# ===================== Prompt template =====================

ASSESSMENT_PROMPT_TEMPLATE = (
    "{image_tokens}\n"
    "You are analyzing a video object tracking task.\n\n"
    "FRAME SEQUENCE:\n"
    "- Frames 1-{num_context} (context frames): Show the TARGET OBJECT with a GREEN bounding box. "
    "These are ground truth reference frames where the object was correctly tracked.\n"
    "- Frame {total} (final frame): Shows a RED bounding box which is the current tracker's prediction.\n\n"
    "TASK:\n"
    "Compare the RED box in the final frame with the GREEN boxes in the context frames.\n"
    "- If the RED box is on the SAME object as the GREEN boxes (correct tracking) → assessment: CORRECT\n"
    "- If the RED box is on a DIFFERENT object, wrong location, or covers no object → assessment: INCORRECT, "
    "and provide the bounding box of the correct target object in the final frame.\n\n"
    "Please think step by step, putting your thinking process "
    "within <think>...</think> tags, then give your final answer "
    "within <answer>...</answer> tags.\n"
    "If CORRECT, output: {{\"assessment\": \"CORRECT\"}}\n"
    "If INCORRECT, also provide the bounding box [x1, y1, x2, y2] of the correct target object "
    "in the final frame:\n"
    "{{\"assessment\": \"INCORRECT\", \"boxes\": [x1, y1, x2, y2]}}\n"
    "Coordinates are in pixels relative to the image dimensions.\n"
    "Example (correct):\n<answer>{{\"assessment\": \"CORRECT\"}}</answer>\n"
    "Example (incorrect):\n<answer>{{\"assessment\": \"INCORRECT\", \"boxes\": [120, 45, 350, 280]}}</answer>"
)


# ===================== Frame loading =====================

def get_frame_path(frames_root: str, sequence_id: str, frame_idx: int) -> str:
    """Get the path to a video frame (JPEG folder format)."""
    # Common patterns: JPEGImages/{seq_id}/{frame:05d}.jpg or {frame:06d}.jpg
    seq_dir = os.path.join(frames_root, sequence_id)
    if not os.path.isdir(seq_dir):
        return ""
    
    # Try common naming patterns
    for pattern in [f"{frame_idx:05d}.jpg", f"{frame_idx:06d}.jpg",
                    f"{frame_idx:05d}.png", f"{frame_idx:06d}.png"]:
        path = os.path.join(seq_dir, pattern)
        if os.path.exists(path):
            return path
    
    # Fallback: list directory and find by index
    frame_files = sorted([
        f for f in os.listdir(seq_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    if frame_idx < len(frame_files):
        return os.path.join(seq_dir, frame_files[frame_idx])
    
    return ""


def load_frame(
    frame_idx: int,
    frames_root: Optional[str],
    sequence_id: str,
    video_path: Optional[str],
) -> Optional[Union[str, np.ndarray]]:
    """Load a frame from either JPEG folder or MP4 video.

    Returns:
        - file path (str) if loading from JPEG folder
        - RGB numpy array if loading from video
        - None if loading failed
    """
    if video_path and os.path.exists(video_path):
        # Load from MP4 video
        frame = _video_loader.get_frame(video_path, frame_idx)
        return frame  # np.ndarray (RGB) or None
    elif frames_root:
        # Load from JPEG folder
        path = get_frame_path(frames_root, sequence_id, frame_idx)
        return path if path else None
    else:
        logger.warning(f"No frame source available for {sequence_id} frame {frame_idx}")
        return None


def find_correct_context_frames(
    obj_frames: List[Dict],
    target_frame_idx: int,
    num_context: int,
    min_iou: float = 0.75,
) -> List[Dict]:
    """
    Find N frames where tracking was correct to use as context.
    Uses the last N frames before the target frame (to capture scene changes).
    Never uses frames after the target frame.

    Args:
        min_iou: Minimum IoU to consider a frame as "correct" for context.
                 Should be higher than the failure detection threshold to ensure
                 context frames show tight, accurate bounding boxes.
    """
    # Find all correct frames BEFORE the target frame
    candidates = []
    for f in obj_frames:
        if f["frame_idx"] >= target_frame_idx:  # Only frames before (never after)
            continue
        if not f.get("is_correct", False):
            continue
        if not f.get("has_gt", False):
            continue
        if f.get("iou", 0.0) < min_iou:
            continue
        if f.get("pred_bbox") is None or f.get("gt_bbox") is None:
            continue
        candidates.append(f)

    if not candidates:
        return []

    # Sort by frame_idx to get temporal order
    candidates = sorted(candidates, key=lambda x: x["frame_idx"])
    
    # Take the last N frames before the failure
    if len(candidates) <= num_context:
        return candidates
    
    return candidates[-num_context:]


def find_reappearance_context_frames(
    obj_frames: List[Dict],
    obj_data: Dict,
    failure_frame_idx: int,
    occlusion_id: Optional[int],
    num_before_occlusion: int = 4,
    num_during_occlusion: int = 1,
    min_iou: float = 0.75,
) -> List[Dict]:
    """
    Find context frames for reappearance failures:
    - First frame (initial appearance)
    - Last N frames before occlusion (to see scene changes)
    - Optional: 1-2 frames during occlusion
    
    Args:
        obj_frames: All frames for this object
        obj_data: Object data containing occlusions list
        failure_frame_idx: Frame where reappearance failure occurred
        occlusion_id: ID of the occlusion that just ended (if available)
        num_before_occlusion: Number of frames to take before occlusion (default 4)
        num_during_occlusion: Number of frames during occlusion (default 1, set to 0 to disable)
        min_iou: Minimum IoU for correct frames
    """
    context = []
    
    # 1. First frame (initial appearance)
    first_frame = None
    for f in obj_frames:
        if f.get("is_correct", False) and f.get("has_gt", False) and f.get("iou", 0.0) >= min_iou:
            if f.get("pred_bbox") and f.get("gt_bbox"):
                first_frame = f
                break
    
    if first_frame:
        context.append(first_frame)
    
    # 2. Find the occlusion that just ended (if occlusion_id provided)
    occlusion_start = None
    occlusion_end = None
    if occlusion_id is not None:
        occlusions = obj_data.get("occlusions", [])
        for occ in occlusions:
            if occ.get("occlusion_id") == occlusion_id:
                occlusion_start = occ.get("start_frame")
                occlusion_end = occ.get("end_frame")
                break
    
    # If no occlusion_id, try to find occlusion ending at or before failure frame
    if occlusion_start is None:
        occlusions = obj_data.get("occlusions", [])
        for occ in occlusions:
            end_frame = occ.get("end_frame")
            if end_frame is not None and end_frame <= failure_frame_idx:
                occlusion_start = occ.get("start_frame")
                occlusion_end = end_frame
                occlusion_id = occ.get("occlusion_id")
                break
    
    # 3. Last N frames before occlusion
    if occlusion_start is not None:
        frames_before = [
            f for f in obj_frames
            if f["frame_idx"] < occlusion_start
            and f.get("is_correct", False)
            and f.get("has_gt", False)
            and f.get("iou", 0.0) >= min_iou
            and f.get("pred_bbox") and f.get("gt_bbox")
        ]
        if frames_before:
            # Take last num_before_occlusion frames
            frames_before = sorted(frames_before, key=lambda x: x["frame_idx"])
            last_before = frames_before[-num_before_occlusion:]
            for f in last_before:
                if f not in context:
                    context.append(f)
    
    # 4. Optional: frames during occlusion (if GT available)
    if num_during_occlusion > 0 and occlusion_start is not None and occlusion_end is not None:
        frames_during = [
            f for f in obj_frames
            if occlusion_start <= f["frame_idx"] <= occlusion_end
            and f.get("has_gt", False)  # During occlusion, object is hidden but GT might exist
            and f.get("gt_bbox")  # We can show GT even if tracking failed
        ]
        if frames_during:
            # Take evenly spaced frames during occlusion
            if len(frames_during) <= num_during_occlusion:
                selected_during = frames_during
            else:
                indices = np.linspace(0, len(frames_during) - 1, num_during_occlusion, dtype=int)
                selected_during = [frames_during[i] for i in indices]
            
            for f in selected_during:
                if f not in context:
                    context.append(f)
    
    # Sort by frame_idx to maintain temporal order
    context = sorted(context, key=lambda x: x["frame_idx"])
    
    return context


def generate_samples_for_object(
    sequence_id: str,
    obj_id: int,
    obj_data: Dict,
    frames_root: Optional[str],
    video_path: Optional[str],
    output_images_dir: str,
    num_context_frames: int,
    max_incorrect_samples: int,
    correct_ratio: float,
    correct_iou_threshold: float,
    resize: Optional[int],
) -> List[Dict]:
    """Generate training samples for one object in one sequence.

    Supports both JPEG folder format (frames_root) and MP4 video format (video_path).

    Args:
        correct_iou_threshold: Minimum IoU for a frame to be used as context or
            labeled CORRECT. Higher than the failure detection threshold (0.5) to
            ensure clean training signal — context frames show tight bboxes and
            CORRECT labels are only given to genuinely well-tracked frames.
    """

    obj_frames = obj_data.get("frames", [])
    if len(obj_frames) < num_context_frames + 1:
        return []

    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    samples = []

    # Preload needed frames from video for efficiency
    if video_path and os.path.exists(video_path):
        all_frame_indices = [f["frame_idx"] for f in obj_frames]
        failure_indices = [f["frame_idx"] for f in obj_data.get("failure_frames", [])]
        _video_loader.preload_frames(video_path, list(set(all_frame_indices + failure_indices)))

    # ---- INCORRECT samples: frames where SAM3 failed ----
    failure_frames = obj_data.get("failure_frames", [])
    
    # ALWAYS prioritize reappearance failures (even if fewer than max_incorrect_samples)
    reapp_failures = [f for f in failure_frames if f.get("is_at_sam3_reappearance", False)]
    other_failures = [f for f in failure_frames if not f.get("is_at_sam3_reappearance", False)]
    random.shuffle(other_failures)
    
    # Take all reappearance failures first, then fill remaining slots with others
    selected_failures = reapp_failures.copy()
    remaining = max_incorrect_samples - len(selected_failures)
    if remaining > 0:
        selected_failures.extend(other_failures[:remaining])

    for failure in selected_failures:
        frame_idx = failure["frame_idx"]
        pred_bbox = failure.get("pred_bbox")
        gt_bbox = failure.get("gt_bbox")
        is_reappearance = failure.get("is_at_sam3_reappearance", False)

        if pred_bbox is None or gt_bbox is None:
            continue

        # Find context frames
        # For reappearance failures, use special context selection (first frame + last before occlusion)
        if is_reappearance:
            occlusion_id = failure.get("occlusion_id")
            context = find_reappearance_context_frames(
                obj_frames=obj_frames,
                obj_data=obj_data,
                failure_frame_idx=frame_idx,
                occlusion_id=occlusion_id,
                num_before_occlusion=4,
                num_during_occlusion=1,  # Include 1 frame during occlusion
                min_iou=correct_iou_threshold,
            )
        else:
            # Regular failures: use evenly-spread context
            context = find_correct_context_frames(
                obj_frames, frame_idx, num_context_frames, min_iou=correct_iou_threshold,
            )
        
        if len(context) < max(1, num_context_frames // 2):
            continue  # Not enough context

        # Create overlay images
        sample_id = f"{sequence_id}_obj{obj_id}_f{frame_idx}_incorrect"
        image_paths = []
        success = True

        # Context frames (green bbox using GT bbox)
        for ci, ctx_frame in enumerate(context):
            ctx_frame_idx = ctx_frame["frame_idx"]
            ctx_bbox = ctx_frame.get("gt_bbox") or ctx_frame.get("pred_bbox")
            if ctx_bbox is None:
                success = False
                break

            frame_source = load_frame(ctx_frame_idx, frames_root, sequence_id, video_path)
            if frame_source is None:
                success = False
                break

            out_name = f"{sample_id}_ctx{ci}.jpg"
            out_path = os.path.join(output_images_dir, out_name)
            if not save_overlay_image(frame_source, ctx_bbox, GREEN, out_path, resize):
                success = False
                break

            image_paths.append(out_name)

        if not success:
            continue

        # Query frame (red bbox using pred_bbox)
        frame_source = load_frame(frame_idx, frames_root, sequence_id, video_path)
        if frame_source is None:
            continue

        out_name = f"{sample_id}_query.jpg"
        out_path = os.path.join(output_images_dir, out_name)
        if not save_overlay_image(frame_source, pred_bbox, RED, out_path, resize):
            continue

        image_paths.append(out_name)

        # Build training sample
        num_images = len(image_paths)
        image_tokens = "<image>" * num_images
        prompt = ASSESSMENT_PROMPT_TEMPLATE.format(
            image_tokens=image_tokens,
            num_context=num_images - 1,
            total=num_images,
        )

        gt = json.dumps({"assessment": "INCORRECT", "boxes": gt_bbox})
        
        # Store IoU for visualization/debugging
        mask_iou = failure.get("iou", 0.0)
        bbox_iou = compute_bbox_iou(pred_bbox, gt_bbox) if pred_bbox and gt_bbox else 0.0

        samples.append({
            "prompt": prompt,
            "images": image_paths,
            "answer": gt,
            "data_type": "image",
            "problem_type": "tracking_verification",
            "problem_reserved_text": prompt,
            "problem_id": f"{sequence_id}_obj{obj_id}",
            "_mask_iou": mask_iou,  # Metadata for viz
            "_bbox_iou": bbox_iou,   # Metadata for viz
            "_is_reappearance": is_reappearance,  # Track if this is a reappearance failure
        })

    # ---- CORRECT samples ----
    num_correct = max(1, int(len(samples) * correct_ratio / max(0.01, 1.0 - correct_ratio)))

    correct_frames = [
        f for f in obj_frames
        if f.get("is_correct", False) and f.get("has_gt", False)
        and f.get("pred_bbox") is not None
        and f.get("iou", 0.0) >= correct_iou_threshold  # Only truly well-tracked frames
    ]

    if correct_frames:
        random.shuffle(correct_frames)
        for cf in correct_frames[:num_correct]:
            frame_idx = cf["frame_idx"]
            pred_bbox = cf["pred_bbox"]
            gt_bbox = cf.get("gt_bbox")

            context = find_correct_context_frames(
                obj_frames, frame_idx, num_context_frames, min_iou=correct_iou_threshold,
            )
            if len(context) < max(1, num_context_frames // 2):
                continue

            sample_id = f"{sequence_id}_obj{obj_id}_f{frame_idx}_correct"
            image_paths = []
            success = True

            # Context frames (green bbox)
            for ci, ctx_frame in enumerate(context):
                ctx_frame_idx = ctx_frame["frame_idx"]
                ctx_bbox = ctx_frame.get("gt_bbox") or ctx_frame.get("pred_bbox")
                if ctx_bbox is None:
                    success = False
                    break

                frame_source = load_frame(ctx_frame_idx, frames_root, sequence_id, video_path)
                if frame_source is None:
                    success = False
                    break

                out_name = f"{sample_id}_ctx{ci}.jpg"
                out_path = os.path.join(output_images_dir, out_name)
                if not save_overlay_image(frame_source, ctx_bbox, GREEN, out_path, resize):
                    success = False
                    break

                image_paths.append(out_name)

            if not success:
                continue

            # Query frame (red bbox - same as green since correct)
            frame_source = load_frame(frame_idx, frames_root, sequence_id, video_path)
            if frame_source is None:
                continue

            out_name = f"{sample_id}_query.jpg"
            out_path = os.path.join(output_images_dir, out_name)
            if not save_overlay_image(frame_source, pred_bbox, RED, out_path, resize):
                continue

            image_paths.append(out_name)

            num_images = len(image_paths)
            image_tokens = "<image>" * num_images
            prompt = ASSESSMENT_PROMPT_TEMPLATE.format(
                image_tokens=image_tokens,
                num_context=num_images - 1,
                total=num_images,
            )

            gt = json.dumps({"assessment": "CORRECT"})
            
            # Store IoU for visualization/debugging
            mask_iou = cf.get("iou", 0.0)
            bbox_iou = compute_bbox_iou(pred_bbox, gt_bbox) if pred_bbox and gt_bbox else 0.0

            samples.append({
                "prompt": prompt,
                "images": image_paths,
                "answer": gt,
                "data_type": "image",
                "problem_type": "tracking_verification",
                "problem_reserved_text": prompt,
                "problem_id": f"{sequence_id}_obj{obj_id}",
                "_mask_iou": mask_iou,  # Metadata for viz
                "_bbox_iou": bbox_iou,   # Metadata for viz
            })

    return samples


# ===================== Main =====================

def _save_viz_grid(sample: Dict, images_dir: Path, viz_dir: Path, viz_idx: int) -> None:
    """Save a visualization grid for one training sample.
    
    Shows all context frames + query frame side by side with labels.
    For INCORRECT samples, also draws the GT bbox (GREEN) on the query frame
    so you can see the difference between pred (RED) and GT (GREEN).
    """
    image_names = sample["images"]
    gt = json.loads(sample["answer"])
    assessment = gt["assessment"]

    # Load all images
    frames = []
    for img_name in image_names:
        img_path = images_dir / img_name
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                frames.append(img)

    if not frames:
        return

    # For INCORRECT samples, overlay GT bbox (GREEN) on query frame too
    # so the viz shows both RED (pred) and GREEN (GT) on the last frame
    if assessment == "INCORRECT" and "boxes" in gt:
        gt_bbox = gt["boxes"]
        query_frame = frames[-1].copy()
        x1, y1, x2, y2 = [int(v) for v in gt_bbox]
        h, w = query_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            # Draw GREEN dashed-style bbox (thicker to distinguish from RED)
            cv2.rectangle(query_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frames[-1] = query_frame

    # Resize all to same height
    target_h = 300
    resized = []
    for fr in frames:
        h, w = fr.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized.append(cv2.resize(fr, (new_w, target_h)))

    # Add label bar under each frame
    labeled = []
    for fi, fr in enumerate(resized):
        bar_h = 30
        bar = np.zeros((bar_h, fr.shape[1], 3), dtype=np.uint8)
        if fi < len(resized) - 1:
            label = f"Context {fi+1} (GREEN=GT)"
            color = (0, 255, 0)
        else:
            if assessment == "INCORRECT":
                label = "Query (RED=pred, GREEN=GT)"
            else:
                label = "Query (RED=pred, CORRECT)"
            color = (0, 0, 255) if assessment == "INCORRECT" else (0, 255, 0)
        cv2.putText(bar, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        labeled.append(np.vstack([fr, bar]))

    # Concatenate horizontally
    grid = np.hstack(labeled)

    # Title bar with IoU info
    title_h = 50  # Taller to fit two lines
    title_bar = np.zeros((title_h, grid.shape[1], 3), dtype=np.uint8)
    gt_text = sample["answer"]
    if len(gt_text) > 70:
        gt_text = gt_text[:70] + "..."
    
    # Get IoU values
    mask_iou = sample.get("_mask_iou")
    bbox_iou = sample.get("_bbox_iou")
    
    # Line 1: Assessment and mask IoU (used for threshold)
    if mask_iou is not None:
        mask_iou_str = f"Mask IoU={mask_iou:.3f}"
        if assessment == "INCORRECT":
            mask_iou_str += " (<0.5)"
        line1 = f"Sample {viz_idx}: {assessment} | {mask_iou_str}"
    else:
        line1 = f"Sample {viz_idx}: {assessment}"
    
    # Line 2: Bbox IoU (for comparison)
    if bbox_iou is not None:
        line2 = f"Bbox IoU={bbox_iou:.3f} | {gt_text}"
    else:
        line2 = gt_text
    
    cv2.putText(title_bar, line1, (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(title_bar, line2, (5, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    grid = np.vstack([title_bar, grid])

    out_path = viz_dir / f"viz_{viz_idx:04d}_{assessment.lower()}.jpg"
    cv2.imwrite(str(out_path), grid)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tracking verification training data for EasyR1",
    )
    parser.add_argument(
        "--collected_dir",
        type=str,
        required=True,
        help="Path to the output folder from collect_training_data.py",
    )
    parser.add_argument(
        "--sub_dir",
        type=str,
        required=True,
        help="The subfolder folder in SAV dataset (e.g. sav_000, sav_001, etc.)",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default=None,
        help="Root directory for video frames (e.g., JPEGImages/). "
             "Frames are expected at {frames_root}/{sequence_id}/{frame:05d}.jpg. "
             "Not required when using SA-V training format (video_path in collected data).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for overlay images and training JSON",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="tracking_verification_train.json",
        help="Output JSON filename",
    )
    parser.add_argument(
        "--num_context_frames",
        type=int,
        default=4,
        help="Number of context frames (green bbox) per sample",
    )
    parser.add_argument(
        "--max_samples_per_object",
        type=int,
        default=20,
        help="Maximum INCORRECT samples per object per sequence",
    )
    parser.add_argument(
        "--correct_ratio",
        type=float,
        default=0.4,
        help="Ratio of CORRECT samples (0.4 = 40%% CORRECT, 60%% INCORRECT)",
    )
    parser.add_argument(
        "--correct_iou_threshold",
        type=float,
        default=0.75,
        help="Minimum IoU for a frame to be used as context or labeled CORRECT. "
             "Higher than failure threshold (0.5) to ensure clean training signal. "
             "Context frames show tight GREEN bboxes, CORRECT labels are only given "
             "to genuinely well-tracked frames.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Resize images (max dimension). None = no resize.",
    )
    parser.add_argument(
        "--save_viz",
        type=int,
        default=0,
        help="Number of visualization grids to save (0 = disabled). "
             "Saves side-by-side context+query frames with labels.",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to process (for testing/debugging). "
             "If None, processes all sequences.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load collected data (supports both JSON array and JSONL formats)
    logger.info(f"Loading collected data from folder: {args.collected_dir}")
    collected_dir = Path(args.collected_dir)
    sub_dir = args.sub_dir

    # Load all the json files. 
    # Folder structure: <collected_dir>/<sub_dir>/tracking_result/*.json
    all_sequences = []
    for current_sub_dir in collected_dir.iterdir():
        if not current_sub_dir.is_dir():
            continue
        if sub_dir is not None and current_sub_dir.name != sub_dir:
            continue
        json_dir = current_sub_dir / "tracking_result"
        if not json_dir.exists():
            logger.warning(f"No tracking_result directory in {current_sub_dir}, skipping")
            continue
        json_files = list(json_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {json_dir}, skipping")
            continue

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    all_sequences.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON file {json_file}: {e}")

    print(f"Loaded {len(all_sequences)} sequences from {collected_dir} with sub_dir={sub_dir}")
    print(f"Example sequence keys: {list(all_sequences[0].keys()) if all_sequences else 'N/A'}")
    
    # Limit sequences if requested
    if args.max_sequences and args.max_sequences < len(all_sequences):
        all_sequences = all_sequences[:args.max_sequences]
        logger.info(f"Limited to {len(all_sequences)} sequences (--max_sequences={args.max_sequences})")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup viz directory early if needed
    viz_dir = None
    viz_saved = 0
    if args.save_viz > 0:
        viz_dir = output_dir / "viz"
        viz_dir.mkdir(exist_ok=True)
    
    # Generate samples
    all_samples = []
    stats = {
        "total_sequences": 0,
        "total_correct": 0,
        "total_incorrect": 0,
        "total_incorrect_reappearance": 0,  # Track reappearance samples
        "skipped_sequences": 0,
    }
    
    for i, seq_data in tqdm(enumerate(all_sequences)):
        seq_id = seq_data.get("sequence_id", "")
        logger.info(f"Processing sequence {i}/{len(all_sequences)}: {seq_id}")
        objects = seq_data.get("objects", {})
        video_path = seq_data.get("video_path")  # MP4 path (SA-V train format)

        if not objects:
            stats["skipped_sequences"] += 1
            continue

        # Verify we have a frame source
        if not video_path and not args.frames_root:
            logger.warning(f"No video_path in data and no --frames_root specified, skipping {seq_id}")
            stats["skipped_sequences"] += 1
            continue

        stats["total_sequences"] += 1

        for obj_id_str, obj_data in objects.items():
            obj_id = int(obj_id_str)

            samples = generate_samples_for_object(
                sequence_id=seq_id,
                obj_id=obj_id,
                obj_data=obj_data,
                frames_root=args.frames_root,
                video_path=video_path,
                output_images_dir=str(images_dir),
                num_context_frames=args.num_context_frames,
                max_incorrect_samples=args.max_samples_per_object,
                correct_ratio=args.correct_ratio,
                correct_iou_threshold=args.correct_iou_threshold,
                resize=args.resize,
            )

            for s in samples:
                gt = json.loads(s["answer"])
                if gt["assessment"] == "CORRECT":
                    stats["total_correct"] += 1
                else:
                    stats["total_incorrect"] += 1
                    if s.get("_is_reappearance", False):
                        stats["total_incorrect_reappearance"] += 1

            # Save viz grids immediately as samples come in
            if viz_dir is not None and viz_saved < args.save_viz:
                for sample in samples:
                    if viz_saved >= args.save_viz:
                        break
                    _save_viz_grid(sample, images_dir, viz_dir, viz_saved)
                    viz_saved += 1

            all_samples.extend(samples)

        # Clear video frame cache after processing each sequence to save memory
        if video_path:
            _video_loader.clear_cache(video_path)
    
    if viz_dir is not None:
        logger.info(f"Saved {viz_saved} visualizations to {viz_dir}")
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Save training JSON
    output_json_path = output_dir / args.output_json
    with open(output_json_path, "w") as f:
        json.dump(all_samples, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Data generation complete!")
    logger.info(f"  Sequences processed: {stats['total_sequences']}")
    logger.info(f"  Sequences skipped: {stats['skipped_sequences']}")
    logger.info(f"  Total samples: {len(all_samples)}")
    logger.info(f"  CORRECT samples: {stats['total_correct']}")
    logger.info(f"  INCORRECT samples: {stats['total_incorrect']}")
    if stats['total_incorrect'] > 0:
        logger.info(f"    - Reappearance failures: {stats['total_incorrect_reappearance']} ({stats['total_incorrect_reappearance']/max(1,stats['total_incorrect'])*100:.1f}% of incorrect)")
    logger.info(f"  Correct ratio: {stats['total_correct'] / max(1, len(all_samples)):.2%}")
    logger.info(f"  Output JSON: {output_json_path}")
    logger.info(f"  Output images: {images_dir}")
    logger.info(f"{'='*60}")

    # Cleanup video loader
    _video_loader.close()


if __name__ == "__main__":
    main()
