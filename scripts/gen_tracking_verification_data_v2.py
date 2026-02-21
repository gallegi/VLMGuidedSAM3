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

Usage (reappearance-only mode - for fine-tuning on reappearance detection):
    python gen_tracking_verification_data.py \
        --collected_data /path/to/all_sequences_data.json \
        --output_dir /path/to/output \
        --reappearance_only \
        --reappearance_window 30
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


# ===================== Bbox utilities =====================

def scale_bbox(bbox: List[int], scale: float) -> List[int]:
    """Scale a bbox [x1, y1, x2, y2] by a uniform factor."""
    return [int(v * scale) for v in bbox]


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
    bbox: Optional[List[int]],
    color: Tuple[int, int, int],
    output_path: str,
    resize: Optional[int] = None,
) -> bool:
    """Load a frame (from path or numpy array), optionally draw bbox overlay, and save.

    Args:
        frame_source: Either a file path (str) or an RGB numpy array.
        bbox: [x1, y1, x2, y2] in pixels, or None to save without bbox.
        color: RGB color tuple (ignored if bbox is None).
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
        if bbox is not None:
            bbox = [int(v * scale) for v in bbox]

    if bbox is not None:
        frame_out = draw_bbox_outline(frame_rgb, bbox, color)
    else:
        frame_out = frame_rgb

    # Save as JPEG
    pil_img = PILImage.fromarray(frame_out)
    pil_img.save(output_path, quality=90)
    return True


# ===================== Prompt template =====================

ASSESSMENT_PROMPT_TEMPLATE = (
    "{image_tokens}\n"
    "You are analyzing a video object tracking task.\n\n"
    "FRAME SEQUENCE:\n"
    "- Frames 1-{num_context} (context frames): Most show the TARGET OBJECT with a GREEN bounding box "
    "from ground truth reference frames where the object was correctly tracked.\n"
    "{occlusion_note}"
    "- Frame {total} (final frame): Shows a RED bounding box which is the current tracker's prediction, "
    "or NO bounding box if the tracker failed to predict anything.\n\n"
    "TASK:\n"
    "Compare the tracker's prediction (RED box or absence of box) in the final frame with the GREEN boxes "
    "in the context frames.\n"
    "- If the RED box is on the SAME object as the GREEN boxes (correct tracking) → assessment: CORRECT\n"
    "- If the RED box is on a DIFFERENT object or wrong location → assessment: INCORRECT, "
    "and provide the bounding box of the correct target object in the final frame.\n"
    "- If there is NO RED box but the target object IS VISIBLE in the final frame → assessment: INCORRECT, "
    "and provide the bounding box of the correct target object in the final frame.\n"
    "- If the target object is NOT VISIBLE in the final frame (e.g. still occluded or out of view) "
    "but the tracker is predicting something → assessment: INCORRECT, with no bounding box.\n"
    "- If there is NO RED box and the target object is also NOT VISIBLE in the final frame "
    "(tracker correctly did not predict) → assessment: CORRECT\n\n"
    "Please think step by step, putting your thinking process "
    "within <think>...</think> tags, then give your final answer "
    "within <answer>...</answer> tags.\n"
    "If CORRECT, output: {{\"assessment\": \"CORRECT\"}}\n"
    "If INCORRECT and target is visible, provide the bounding box [x1, y1, x2, y2] of the correct "
    "target object in the final frame:\n"
    "{{\"assessment\": \"INCORRECT\", \"boxes\": [x1, y1, x2, y2]}}\n"
    "If INCORRECT and target is NOT visible in the final frame:\n"
    "{{\"assessment\": \"INCORRECT\"}}\n"
    "Coordinates are in pixels relative to the image dimensions.\n"
    "Example (correct):\n<answer>{{\"assessment\": \"CORRECT\"}}</answer>\n"
    "Example (incorrect, target visible):\n<answer>{{\"assessment\": \"INCORRECT\", \"boxes\": [120, 45, 350, 280]}}</answer>\n"
    "Example (incorrect, target not visible):\n<answer>{{\"assessment\": \"INCORRECT\"}}</answer>"
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
    
    # 4. Optional: frames during occlusion (no bbox — object is hidden)
    #    These show the scene while the object is occluded, giving temporal context.
    #    Use exclusive end: end_frame is the reappearance frame (object visible again).
    if num_during_occlusion > 0 and occlusion_start is not None and occlusion_end is not None:
        frames_during = [
            f for f in obj_frames
            if occlusion_start <= f["frame_idx"] < occlusion_end
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
                    # Shallow copy so we can mark it without mutating the original
                    f_copy = dict(f)
                    f_copy["_during_occlusion"] = True
                    context.append(f_copy)
    
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
    reappearance_only: bool = False,
    reappearance_window: int = 30,
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

    # Compute resize scale factor (if --resize is used) so GT bboxes in the answer
    # match the saved image coordinate space.  Without this, the VLM would learn in
    # the resized coordinate space but the reward function would compare against
    # original-resolution GT bboxes, causing near-zero IoU even for perfect predictions.
    _resize_scale: Optional[float] = None
    if resize is not None and obj_frames:
        sample_frame = load_frame(obj_frames[0]["frame_idx"], frames_root, sequence_id, video_path)
        if sample_frame is not None:
            if isinstance(sample_frame, np.ndarray):
                h, w = sample_frame.shape[:2]
            else:  # str path
                _tmp = cv2.imread(sample_frame)
                if _tmp is not None:
                    h, w = _tmp.shape[:2]
                else:
                    h, w = None, None
            if h is not None and w is not None:
                _resize_scale = resize / max(h, w)

    # ---- INCORRECT samples: frames where SAM3 failed ----
    failure_frames = obj_data.get("failure_frames", [])
    
    # Separate reappearance vs other failures
    reapp_failures = [f for f in failure_frames if f.get("is_at_sam3_reappearance", False)]
    other_failures = [f for f in failure_frames if not f.get("is_at_sam3_reappearance", False)]
    random.shuffle(other_failures)
    
    if reappearance_only:
        # Only reappearance failures — skip drift/scale/other errors entirely
        selected_failures = reapp_failures[:max_incorrect_samples]
    else:
        # Take all reappearance failures first, then fill remaining slots with others
        selected_failures = reapp_failures.copy()
        remaining = max_incorrect_samples - len(selected_failures)
        if remaining > 0:
            selected_failures.extend(other_failures[:remaining])

    for failure in selected_failures:
        frame_idx = failure["frame_idx"]
        pred_bbox = failure.get("pred_bbox")  # None when SAM3 didn't predict (missed detection)
        gt_bbox = failure.get("gt_bbox")  # None when object is not visible (false positive)
        is_reappearance = failure.get("is_at_sam3_reappearance", False)
        object_not_present = (gt_bbox is None)
        sam3_no_prediction = (pred_bbox is None)
        both_none = (pred_bbox is None and gt_bbox is None)

        # Include all cases - even when both are None, this teaches the VLM that
        # "no prediction when object is not visible" is acceptable behavior

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
                num_during_occlusion=1,  # 1 frame during occlusion (no bbox, shows scene)
                min_iou=correct_iou_threshold,
            )
        else:
            # Regular failures: use evenly-spread context
            context = find_correct_context_frames(
                obj_frames, frame_idx, num_context_frames, min_iou=correct_iou_threshold,
            )
        
        # Remove any context frame at the same frame_idx as the query
        # (can happen when query is during occlusion and picked as context too)
        context = [c for c in context if c["frame_idx"] != frame_idx]

        if len(context) < max(1, num_context_frames // 2):
            continue  # Not enough context

        # Create overlay images
        sample_id = f"{sequence_id}_obj{obj_id}_f{frame_idx}_incorrect"
        image_paths = []
        success = True

        # Context frames (green bbox using GT bbox, or no bbox during occlusion)
        has_occlusion_frame = False
        for ci, ctx_frame in enumerate(context):
            ctx_frame_idx = ctx_frame["frame_idx"]
            is_occlusion_frame = ctx_frame.get("_during_occlusion", False)

            if is_occlusion_frame:
                ctx_bbox = None  # No bbox — object is hidden
                has_occlusion_frame = True
            else:
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

        # Query frame (red bbox using pred_bbox, or no bbox if SAM3 didn't predict)
        frame_source = load_frame(frame_idx, frames_root, sequence_id, video_path)
        if frame_source is None:
            continue

        out_name = f"{sample_id}_query.jpg"
        out_path = os.path.join(output_images_dir, out_name)
        # If pred_bbox is None, save frame without any bbox (SAM3 missed the object)
        query_bbox = pred_bbox if pred_bbox is not None else None
        if not save_overlay_image(frame_source, query_bbox, RED, out_path, resize):
            continue

        image_paths.append(out_name)

        # Build training sample
        num_images = len(image_paths)
        image_tokens = "<image>" * num_images
        occlusion_note = (
            "Some context frames may show the scene during occlusion "
            "(when the object was temporarily hidden) and have no bounding box.\n"
            if has_occlusion_frame else ""
        )
        prompt = ASSESSMENT_PROMPT_TEMPLATE.format(
            image_tokens=image_tokens,
            num_context=num_images - 1,
            total=num_images,
            occlusion_note=occlusion_note,
        )

        # Build GT answer:
        # - Both None (no pred, no object): {"assessment": "CORRECT"} (acceptable behavior)
        # - Object visible but SAM3 missed: {"assessment": "INCORRECT", "boxes": [x1,y1,x2,y2]}
        # - Object not present but SAM3 predicted: {"assessment": "INCORRECT"} (false positive)
        if both_none:
            # No prediction when object is not visible = correct behavior
            gt = json.dumps({"assessment": "CORRECT"})
        elif object_not_present:
            # Object not present but SAM3 predicted something = false positive
            gt = json.dumps({"assessment": "INCORRECT"})
        else:
            # Object visible but SAM3 prediction is wrong = incorrect, provide correction
            # Scale GT bbox to match the saved image coordinate space (when --resize is used).
            answer_gt_bbox = scale_bbox(gt_bbox, _resize_scale) if _resize_scale is not None else gt_bbox
            gt = json.dumps({"assessment": "INCORRECT", "boxes": answer_gt_bbox})
        
        # Store IoU for visualization/debugging (computed in original coords)
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
            "_object_not_present": object_not_present,  # Track false-positive during occlusion
            "_both_none": both_none,  # Track "no prediction + no object" (labeled CORRECT)
        })

    # ---- CORRECT samples ----
    num_correct = max(1, int(len(samples) * correct_ratio / max(0.01, 1.0 - correct_ratio)))

    if reappearance_only:
        # Only pick correct frames shortly after reappearance (post-occlusion).
        # These are frames where SAM3 correctly re-identified the object after it
        # reappeared, paired with the same pre-occlusion context so the model sees
        # both successful and failed reappearances.
        occlusions = obj_data.get("occlusions", [])
        reapp_correct_candidates = []  # list of (frame_dict, occlusion_id)

        for occ in occlusions:
            end_frame = occ.get("end_frame")
            occ_id = occ.get("occlusion_id")
            if end_frame is None:
                continue

            for f in obj_frames:
                if f["frame_idx"] < end_frame:
                    continue  # Before reappearance — skip
                if f["frame_idx"] > end_frame + reappearance_window:
                    continue
                if not f.get("is_correct", False):
                    continue
                if not f.get("has_gt", False):
                    continue
                if f.get("iou", 0.0) < correct_iou_threshold:
                    continue
                if f.get("pred_bbox") is None:
                    continue
                reapp_correct_candidates.append((f, occ_id))

        # Deduplicate by frame_idx (a frame could fall in window of multiple occlusions)
        seen_frame_idxs = set()
        unique_candidates = []
        for f, occ_id in reapp_correct_candidates:
            if f["frame_idx"] not in seen_frame_idxs:
                seen_frame_idxs.add(f["frame_idx"])
                unique_candidates.append((f, occ_id))

        random.shuffle(unique_candidates)
        correct_to_process = unique_candidates[:num_correct]
    else:
        # Generic: any correctly-tracked frame
        correct_frames = [
            f for f in obj_frames
            if f.get("is_correct", False) and f.get("has_gt", False)
            and f.get("pred_bbox") is not None
            and f.get("iou", 0.0) >= correct_iou_threshold
        ]
        random.shuffle(correct_frames)
        correct_to_process = [(f, None) for f in correct_frames[:num_correct]]

    for cf, occ_id in correct_to_process:
        frame_idx = cf["frame_idx"]
        pred_bbox = cf["pred_bbox"]
        gt_bbox = cf.get("gt_bbox")

        # Use reappearance context (pre-occlusion + during-occlusion frames)
        # so the model sees the same visual pattern for both CORRECT and INCORRECT
        if reappearance_only and occ_id is not None:
            context = find_reappearance_context_frames(
                obj_frames=obj_frames,
                obj_data=obj_data,
                failure_frame_idx=frame_idx,
                occlusion_id=occ_id,
                num_before_occlusion=4,
                num_during_occlusion=1,  # 1 frame during occlusion (no bbox, shows scene)
                min_iou=correct_iou_threshold,
            )
        else:
            context = find_correct_context_frames(
                obj_frames, frame_idx, num_context_frames, min_iou=correct_iou_threshold,
            )

        if len(context) < max(1, num_context_frames // 2):
            continue

        sample_id = f"{sequence_id}_obj{obj_id}_f{frame_idx}_correct"
        image_paths = []
        success = True

        # Context frames (green bbox, or no bbox during occlusion)
        has_occlusion_frame = False
        for ci, ctx_frame in enumerate(context):
            ctx_frame_idx = ctx_frame["frame_idx"]
            is_occlusion_frame = ctx_frame.get("_during_occlusion", False)

            if is_occlusion_frame:
                ctx_bbox = None  # No bbox — object is hidden
                has_occlusion_frame = True
            else:
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
        occlusion_note = (
            "Some context frames may show the scene during occlusion "
            "(when the object was temporarily hidden) and have no bounding box.\n"
            if has_occlusion_frame else ""
        )
        prompt = ASSESSMENT_PROMPT_TEMPLATE.format(
            image_tokens=image_tokens,
            num_context=num_images - 1,
            total=num_images,
            occlusion_note=occlusion_note,
        )

        gt = json.dumps({"assessment": "CORRECT"})
        
        # Store IoU for visualization/debugging
        mask_iou = cf.get("iou", 0.0)
        bbox_iou = compute_bbox_iou(pred_bbox, gt_bbox) if pred_bbox and gt_bbox else 0.0

        is_reapp_sample = reappearance_only and occ_id is not None
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
            "_is_reappearance": is_reapp_sample,
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
        "--collected_data",
        type=str,
        required=True,
        help="Path to all_sequences_data.json from collect_training_data.py",
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
    parser.add_argument(
        "--reappearance_only",
        action="store_true",
        default=False,
        help="Only generate reappearance samples (both CORRECT and INCORRECT). "
             "INCORRECT: frames where SAM3 failed after object reappeared post-occlusion. "
             "CORRECT: frames where SAM3 correctly re-identified the object post-occlusion. "
             "Both use pre-occlusion context frames so the model learns the reappearance pattern.",
    )
    parser.add_argument(
        "--reappearance_window",
        type=int,
        default=30,
        help="Number of frames after occlusion end to consider as 'reappearance' for "
             "CORRECT samples (only used with --reappearance_only). Default: 30.",
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load collected data (supports both JSON array and JSONL formats)
    logger.info(f"Loading collected data from {args.collected_data}")
    collected_path = args.collected_data
    if collected_path.endswith(".jsonl"):
        # JSONL format: one JSON object per line
        all_sequences = []
        with open(collected_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_sequences.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSONL line: {e}")
    else:
        # Standard JSON array format
        with open(collected_path) as f:
            all_sequences = json.load(f)
    
    logger.info(f"Loaded {len(all_sequences)} sequences")
    
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
    
    # Resume: Load existing samples and track which sequences have been processed
    output_json_path = output_dir / args.output_json
    all_samples = []
    processed_sequences = set()
    
    if output_json_path.exists():
        logger.info(f"Found existing output file: {output_json_path}")
        logger.info("Loading existing samples to resume processing...")
        with open(output_json_path) as f:
            existing_samples = json.load(f)
        all_samples = existing_samples
        # Extract sequence IDs from problem_id (format: "{sequence_id}_obj{obj_id}")
        for sample in existing_samples:
            problem_id = sample.get("problem_id", "")
            if problem_id:
                # Extract sequence_id (everything before "_obj")
                seq_id = problem_id.split("_obj")[0]
                processed_sequences.add(seq_id)
        logger.info(f"Loaded {len(existing_samples)} existing samples from {len(processed_sequences)} sequences")
        logger.info(f"Will skip already-processed sequences: {sorted(list(processed_sequences))[:10]}{'...' if len(processed_sequences) > 10 else ''}")
    else:
        logger.info("No existing output file found. Starting fresh.")
    
    # Track initial count for logging
    initial_sample_count = len(all_samples)
    
    # Generate samples
    stats = {
        "total_sequences": 0,
        "total_correct": 0,
        "total_correct_reappearance": 0,
        "total_incorrect": 0,
        "total_incorrect_reappearance": 0,
        "total_incorrect_not_present": 0,
        "total_correct_both_none": 0,
        "skipped_sequences": 0,
    }
    
    for seq_data in all_sequences:
        seq_id = seq_data.get("sequence_id", "")
        objects = seq_data.get("objects", {})
        video_path = seq_data.get("video_path")  # MP4 path (SA-V train format)

        # Skip if already processed (resume functionality)
        if seq_id in processed_sequences:
            logger.info(f"Skipping already-processed sequence: {seq_id}")
            stats["skipped_sequences"] += 1
            continue

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
                reappearance_only=args.reappearance_only,
                reappearance_window=args.reappearance_window,
            )

            for s in samples:
                gt = json.loads(s["answer"])
                is_reapp = s.get("_is_reappearance", False)
                is_not_present = s.get("_object_not_present", False)
                is_both_none = s.get("_both_none", False)
                if gt["assessment"] == "CORRECT":
                    stats["total_correct"] += 1
                    if is_reapp:
                        stats["total_correct_reappearance"] += 1
                    if is_both_none:
                        stats["total_correct_both_none"] += 1
                else:
                    stats["total_incorrect"] += 1
                    if is_reapp:
                        stats["total_incorrect_reappearance"] += 1
                    if is_not_present:
                        stats["total_incorrect_not_present"] += 1

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
    
    # Save training JSON (overwrite with combined existing + new samples)
    with open(output_json_path, "w") as f:
        json.dump(all_samples, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Data generation complete!")
    if args.reappearance_only:
        logger.info(f"  Mode: REAPPEARANCE ONLY (window={args.reappearance_window} frames)")
    if initial_sample_count > 0:
        logger.info(f"  Existing samples: {initial_sample_count}")
        logger.info(f"  New samples added: {len(all_samples) - initial_sample_count}")
    logger.info(f"  Sequences processed (this run): {stats['total_sequences']}")
    logger.info(f"  Sequences skipped: {stats['skipped_sequences']}")
    logger.info(f"  Total samples (cumulative): {len(all_samples)}")
    logger.info(f"  CORRECT samples: {stats['total_correct']}")
    if stats['total_correct_reappearance'] > 0:
        logger.info(f"    - Reappearance correct: {stats['total_correct_reappearance']} ({stats['total_correct_reappearance']/max(1,stats['total_correct'])*100:.1f}% of correct)")
    if stats['total_correct_both_none'] > 0:
        logger.info(f"    - No pred + no object (both none): {stats['total_correct_both_none']} ({stats['total_correct_both_none']/max(1,stats['total_correct'])*100:.1f}% of correct)")
    logger.info(f"  INCORRECT samples: {stats['total_incorrect']}")
    if stats['total_incorrect'] > 0:
        logger.info(f"    - Reappearance failures: {stats['total_incorrect_reappearance']} ({stats['total_incorrect_reappearance']/max(1,stats['total_incorrect'])*100:.1f}% of incorrect)")
        logger.info(f"    - Object not present: {stats['total_incorrect_not_present']} ({stats['total_incorrect_not_present']/max(1,stats['total_incorrect'])*100:.1f}% of incorrect)")
    logger.info(f"  Correct ratio: {stats['total_correct'] / max(1, len(all_samples)):.2%}")
    logger.info(f"  Output JSON: {output_json_path}")
    logger.info(f"  Output images: {images_dir}")
    logger.info(f"{'='*60}")

    # Cleanup video loader
    _video_loader.close()


if __name__ == "__main__":
    main()
