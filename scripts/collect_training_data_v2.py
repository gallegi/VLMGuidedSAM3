#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Collect SAM3 tracking data from SA-V dataset for OneThinker training.

This script:
1. Runs SAM3 baseline tracking on SA-V sequences
2. Collects per-frame tracking results, occlusion events, failures
3. Computes assessment labels (CORRECT/INCORRECT) based on IoU
4. Extracts relocalization ground truth (correct bbox when SAM3 fails)
5. Stores all data in structured format for training
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from sam3.model_builder import build_sam3_video_model

try:
    import pycocotools.mask as mask_utils
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("pycocotools not found, RLE mask decoding will not work")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

FRAME_INTERVAL = 4


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between predicted and ground truth masks."""
    if pred_mask is None or gt_mask is None:
        return 0.0
    
    # Ensure masks are 2D and have compatible shapes
    pred_mask = np.asarray(pred_mask).squeeze()
    gt_mask = np.asarray(gt_mask).squeeze()
    
    if pred_mask.ndim != 2 or gt_mask.ndim != 2:
        return 0.0
    
    # Resize if shapes don't match
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(
            gt_mask.astype(np.float32),
            (pred_mask.shape[1], pred_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
    
    # Convert to binary masks
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    gt_binary = (gt_mask > 0.5).astype(np.uint8)
    
    # Ensure C-contiguous for logical operations
    pred_binary = np.ascontiguousarray(pred_binary)
    gt_binary = np.ascontiguousarray(gt_binary)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    return float(intersection) / float(union) if union > 0 else 0.0


def decode_rle_mask(rle_dict: Dict) -> Optional[np.ndarray]:
    """Decode RLE mask from SA-V training format.
    
    Args:
        rle_dict: Dictionary with 'size' [H, W] and 'counts' (RLE string)
    
    Returns:
        Binary mask as numpy array (H, W) or None if failed
    """
    if not HAS_PYCOCOTOOLS:
        logger.error("pycocotools required for RLE decoding")
        return None
    
    if rle_dict is None or not rle_dict:
        return None
    
    try:
        size = rle_dict.get('size', [])
        counts = rle_dict.get('counts', '')
        
        if not size or not counts:
            return None
        
        # pycocotools expects {'size': [H, W], 'counts': bytes or string}
        rle = {'size': [int(size[0]), int(size[1])], 'counts': counts}
        
        # If counts is a string, encode it to bytes for pycocotools
        if isinstance(counts, str):
            rle['counts'] = counts.encode('utf-8')
        
        mask = mask_utils.decode(rle)
        # Ensure mask is C-contiguous for proper indexing with frame arrays
        if not mask.flags['C_CONTIGUOUS']:
            mask = np.ascontiguousarray(mask)
        return mask.astype(np.uint8)
    except Exception as e:
        logger.warning(f"Failed to decode RLE mask: {e}")
        return None


def load_frames_from_video(video_path: str) -> List[np.ndarray]:
    """Load frames from MP4 video into memory (for visualization).
    
    SAM3 can load video files directly via init_state(video_path=...), 
    so we only need this for visualization purposes.
    
    Args:
        video_path: Path to MP4 video file
    
    Returns:
        List of frames as RGB numpy arrays
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    logger.info(f"Loaded {len(frames)} frames from {video_path}")
    
    return frames


def load_sav_train_annotations(json_path: str) -> Dict:
    """Load SA-V training annotations from JSON file.
    
    Args:
        json_path: Path to annotation JSON (e.g., sav_000001_manual.json)
    
    Returns:
        Dictionary with:
        - 'num_objects': number of objects
        - 'first_appeared_frame': dict of {obj_id: first_frame_idx}
        - 'masks': dict of {frame_idx: {obj_id: mask}}
        - 'annotated_frames': list of frame indices with annotations
    """
    if not os.path.exists(json_path):
        logger.error(f"Annotation file not found: {json_path}")
        return {}
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    num_objects = int(data.get('masklet_num', 0))
    first_appeared = data.get('masklet_first_appeared_frame', [])
    masklets = data.get('masklet', [])
    video_height = int(data.get('video_height', 0))
    video_width = int(data.get('video_width', 0))
    
    # Build first appearance frame dict
    first_appeared_frame = {}
    # SAV annotate at 6fps from 24fps video, 
    # there are annotations at every 4th frame, so we multiply annotation index by 4 to get frame index
    for obj_idx, ann_idx in enumerate(first_appeared):
        frame_idx = int(ann_idx * FRAME_INTERVAL)  # Convert annotation index to frame index
        first_appeared_frame[obj_idx + 1] = int(ann_idx * FRAME_INTERVAL)  # obj_id is 1-indexed
    
    # Build masks dict
    # masklets is a list where masklets[ann_idx] is a list of per-object masks
    # SA-V training annotations are at 6fps from 24fps video (every 4th frame)
    # So ann_idx 0 -> frame 0, ann_idx 1 -> frame 4, ann_idx 2 -> frame 8, etc.
    
    masks = {}
    annotated_frames = []
    
    video_frame_count = int(data.get('video_frame_count', 0))
    num_annotated = len(masklets)
    
    if num_annotated > 0 and video_frame_count > 0:
        # SA-V training: annotations at 6fps from 24fps = every 4 frames
        
        for ann_idx, frame_masks in enumerate(masklets):
            frame_idx = ann_idx * FRAME_INTERVAL
            if frame_idx >= video_frame_count:
                frame_idx = video_frame_count - 1
            
            annotated_frames.append(frame_idx)
            masks[frame_idx] = {}
            
            for obj_idx, rle_mask in enumerate(frame_masks):
                obj_id = obj_idx + 1  # 1-indexed
                mask = decode_rle_mask(rle_mask)
                if mask is not None:
                    masks[frame_idx][obj_id] = mask
        
        logger.info(f"Loaded {len(masks)} annotated frames (interval={FRAME_INTERVAL}, total_video_frames={video_frame_count})")
    
    return {
        'num_objects': num_objects,
        'first_appeared_frame': first_appeared_frame,
        'masks': masks,
        'annotated_frames': sorted(annotated_frames),
        'video_height': video_height,
        'video_width': video_width,
        'video_frame_count': video_frame_count,
    }


def load_first_frame_masks_from_sav_train(annotations: Dict) -> Dict[int, Dict]:
    """Extract first-frame masks for each object from SA-V training annotations.
    
    Args:
        annotations: Output from load_sav_train_annotations()
    
    Returns:
        Dict of {obj_id: {'mask': mask, 'first_frame': frame_idx}}
    """
    first_frame_masks = {}
    
    first_appeared = annotations.get('first_appeared_frame', {})
    masks_dict = annotations.get('masks', {})
    annotated_frames = annotations.get('annotated_frames', [])
    
    for obj_id, first_frame in first_appeared.items():
        # Find the first annotated frame at or after the first appearance
        mask = None
        mask_frame = None
        
        for ann_frame in annotated_frames:
            if ann_frame >= first_frame:
                frame_masks = masks_dict.get(ann_frame, {})
                if obj_id in frame_masks and frame_masks[obj_id] is not None:
                    mask = frame_masks[obj_id]
                    mask_frame = ann_frame
                    break
        
        if mask is not None:
            first_frame_masks[obj_id] = {
                'mask': mask,
                'first_frame': mask_frame,
            }
    
    return first_frame_masks


def draw_mask_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
    """Draw semi-transparent mask overlay on image."""
    overlay = image.copy()
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # Draw filled mask
    overlay[mask_binary > 0] = (
        overlay[mask_binary > 0] * (1 - alpha) + np.array(color) * alpha
    ).astype(np.uint8)
    
    # Draw boundary
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    
    return overlay


def get_distinct_colors():
    """Get distinct colors for visualization."""
    return [
        np.array([255, 0, 0]),      # Red
        np.array([0, 255, 0]),      # Green
        np.array([0, 0, 255]),      # Blue
        np.array([255, 255, 0]),    # Yellow
        np.array([255, 0, 255]),    # Magenta
        np.array([0, 255, 255]),    # Cyan
        np.array([255, 128, 0]),    # Orange
        np.array([128, 0, 255]),    # Purple
        np.array([0, 255, 128]),    # Spring Green
        np.array([255, 0, 128]),    # Rose
        np.array([128, 255, 0]),    # Lime
        np.array([0, 128, 255]),    # Sky Blue
    ]


def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    """Convert mask to bounding box [x1, y1, x2, y2]."""
    if mask is None or mask.sum() == 0:
        return None
    
    binary_mask = (mask > 0.5).astype(np.uint8)
    coords = np.column_stack(np.where(binary_mask > 0))
    
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]




def load_first_frame_masks(annot_dir: str) -> Dict[int, Dict]:
    """
    Load first-frame masks for each object in the video.
    
    SAV structure: annot_dir/{obj_id}/{frame}.png
    LVOS/MOSE structure: annot_dir/{frame}.png (with object IDs in mask)
    
    Returns: dict of {obj_id: {'mask': mask, 'first_frame': frame_idx}}
    """
    if not annot_dir or not os.path.exists(annot_dir):
        return {}
    
    first_frame_masks = {}
    
    # Check if SAV format (subdirectories per object)
    obj_dirs = sorted([d for d in os.listdir(annot_dir) if os.path.isdir(os.path.join(annot_dir, d))])
    
    if obj_dirs:
        # SAV format: annot_dir/{obj_id}/{frame}.png
        for obj_id_str in obj_dirs:
            try:
                obj_id = int(obj_id_str)
            except ValueError:
                continue
            
            obj_path = os.path.join(annot_dir, obj_id_str)
            mask_files = sorted([f for f in os.listdir(obj_path) if f.endswith('.png')])
            
            if mask_files:
                # Get first annotated frame for this object
                first_mask_file = mask_files[0]
                mask_path = os.path.join(obj_path, first_mask_file)
                mask = np.array(PILImage.open(mask_path))
                
                # Convert to binary
                if mask.dtype == bool:
                    mask = mask.astype(np.uint8) * 255
                mask = (mask > 0).astype(np.uint8)
                
                first_frame = int(first_mask_file.replace('.png', ''))
                
                first_frame_masks[obj_id] = {
                    'mask': mask,
                    'first_frame': first_frame
                }
    else:
        # LVOS/MOSE format: annot_dir/{frame}.png (with object IDs in mask)
        mask_files = sorted([f for f in os.listdir(annot_dir) if f.endswith(('.png', '.jpg'))])
        
        # Find all unique object IDs across all frames
        obj_ids_found = set()
        for mask_file in mask_files:
            mask_path = os.path.join(annot_dir, mask_file)
            mask_img = PILImage.open(mask_path)
            
            if mask_img.mode == "P":
                mask_array = np.array(mask_img)
                unique_values = np.unique(mask_array)
                obj_ids_found.update(unique_values[unique_values > 0])
        
        # For each object, find its first appearance frame
        for obj_id in sorted(obj_ids_found):
            for frame_idx, mask_file in enumerate(mask_files):
                mask_path = os.path.join(annot_dir, mask_file)
                mask_img = PILImage.open(mask_path)
                
                if mask_img.mode == "P":
                    mask_array = np.array(mask_img)
                    mask = (mask_array == obj_id).astype(np.uint8)
                else:
                    mask_array = np.array(mask_img.convert("L"))
                    mask = (mask_array > 127).astype(np.uint8)
                
                if mask.sum() > 0:
                    first_frame_masks[obj_id] = {
                        'mask': mask,
                        'first_frame': frame_idx
                    }
                    break
    
    return first_frame_masks


def load_ground_truth_mask(annotation_path: str, frame_idx: int, obj_id: int = 1) -> Optional[np.ndarray]:
    """Load ground truth mask for a specific frame and object."""
    if not annotation_path or not os.path.exists(annotation_path):
        return None
    
    # Check if SAV format (subdirectories per object)
    obj_dirs = sorted([d for d in os.listdir(annotation_path) if os.path.isdir(os.path.join(annotation_path, d))])
    
    if obj_dirs:
        # SAV format: annotation_path/{obj_id}/{frame}.png
        obj_path = os.path.join(annotation_path, str(obj_id))
        if not os.path.exists(obj_path):
            return None
        
        mask_files = sorted([f for f in os.listdir(obj_path) if f.endswith('.png')])
        frame_name = f"{frame_idx:05d}.png"
        
        if frame_name not in mask_files:
            return None
        
        mask_path = os.path.join(obj_path, frame_name)
        mask = np.array(PILImage.open(mask_path))
        
        # Convert to binary
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        mask = (mask > 0).astype(np.uint8)
        
        return mask
    else:
        # LVOS/MOSE format: annotation_path/{frame}.png (with object IDs in mask)
        mask_files = sorted([
            f for f in os.listdir(annotation_path)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if frame_idx >= len(mask_files):
            return None
        
        mask_file = os.path.join(annotation_path, mask_files[frame_idx])
        mask_img = PILImage.open(mask_file)
        
        # Handle different mask formats
        if mask_img.mode == "P":
            mask_array = np.array(mask_img)
            # Extract object ID
            unique_values = np.unique(mask_array)
            non_zero_values = unique_values[unique_values > 0]
            if len(non_zero_values) > 0:
                obj_id_value = non_zero_values[0] if obj_id == 1 else non_zero_values[min(obj_id - 1, len(non_zero_values) - 1)]
                mask = (mask_array == obj_id_value).astype(np.uint8)
            else:
                return None
        else:
            mask_array = np.array(mask_img.convert("L"))
            mask = (mask_array > 127).astype(np.uint8)
        
        return mask


def get_sequence_info(sequence_id: str, dataset_root: str, is_sav: bool = False) -> Dict:
    """Get information about a sequence."""
    import glob
    
    if is_sav:
        # Try multiple possible structures
        base_name = os.path.basename(dataset_root.rstrip('/'))
        possible_image_paths = [
            os.path.join(dataset_root, "JPEGImages_24fps", sequence_id),
            os.path.join(dataset_root, "JPEGImages", sequence_id),
            os.path.join(dataset_root, "sav_train", "JPEGImages_24fps", sequence_id),
            os.path.join(dataset_root, base_name, "JPEGImages_24fps", sequence_id),
        ]
        possible_ann_paths = [
            os.path.join(dataset_root, "Annotations_6fps", sequence_id),
            os.path.join(dataset_root, "Annotations", sequence_id),
            os.path.join(dataset_root, "sav_train", "Annotations_6fps", sequence_id),
            os.path.join(dataset_root, base_name, "Annotations_6fps", sequence_id),
        ]
    else:
        possible_image_paths = [
            os.path.join(dataset_root, "JPEGImages", sequence_id),
            os.path.join(dataset_root, "images", sequence_id),
        ]
        possible_ann_paths = [
            os.path.join(dataset_root, "Annotations", sequence_id),
            os.path.join(dataset_root, "annotations", sequence_id),
        ]
    
    # Find image directory
    image_path = None
    for path in possible_image_paths:
        if os.path.exists(path) and os.path.isdir(path):
            image_files = glob.glob(os.path.join(path, "*.jpg")) + \
                         glob.glob(os.path.join(path, "*.png"))
            if image_files:
                image_path = path
                break
    
    # Find annotation directory
    ann_path = None
    for path in possible_ann_paths:
        if os.path.exists(path) and os.path.isdir(path):
            ann_path = path
            break
    
    if image_path:
        return {
            "sequence_id": sequence_id,
            "image_path": image_path,
            "annotation_path": ann_path,
            "found": True,
        }
    
    return {"sequence_id": sequence_id, "found": False}


def collect_tracking_data_sav_train(
    sequence_id: str,
    video_path: str,
    annotations: Dict,
    first_frame_masks: Dict[int, Dict],
    tracker,
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Collect tracking data for SA-V training format (MP4 + JSON).
    
    SAM3 loads the video file directly - no need to extract frames to disk.
    
    Args:
        sequence_id: Video ID
        video_path: Path to MP4 video file (SAM3 loads this directly)
        annotations: Parsed annotations from load_sav_train_annotations()
        first_frame_masks: First-frame masks from load_first_frame_masks_from_sav_train()
        tracker: SAM3 tracker
        iou_threshold: IoU threshold for CORRECT/INCORRECT
    
    Returns:
        Same structure as collect_tracking_data()
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return {"error": "video_not_found"}
    
    if not first_frame_masks:
        logger.error(f"No objects found for {sequence_id}")
        return {"error": "no_objects"}
    
    # Initialize tracker state - SAM3 loads video directly from MP4
    inference_state = tracker.init_state(video_path=video_path)
    
    video_height = inference_state["video_height"]
    video_width = inference_state["video_width"]
    num_frames = inference_state["num_frames"]
    
    logger.info(f"SAM3 loaded {num_frames} frames from {video_path}")
    
    # Add objects - SAM3 only supports adding objects before tracking starts
    # So we only track objects that appear at frame 0 or have masks available at frame 0
    tracked_obj_ids = []
    
    for obj_id, info in first_frame_masks.items():
        mask = info['mask']
        first_frame = info['first_frame']
        
        # Add objects at their first appearance frame
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        tracker.add_new_mask(
            inference_state=inference_state,
            frame_idx=first_frame,
            obj_id=int(obj_id),
            mask=mask_tensor,
        )
        tracked_obj_ids.append(obj_id)
        logger.info(f"  Added object {obj_id} at frame {first_frame} ({mask.sum()} pixels)")
    
    if not tracked_obj_ids:
        logger.error(f"No objects to track for {sequence_id}")
        return {"error": "no_trackable_objects"}
    
    # Prepare tracking
    tracker.propagate_in_video_preflight(inference_state)
    
    output_dict = inference_state["output_dict"]
    consolidated_frame_inds = inference_state["consolidated_frame_inds"]
    obj_ids = inference_state["obj_ids"]
    batch_size = len(obj_ids)
    
    # Per-object tracking: failures, occlusions, and reappearance failures
    objects_summary = {}
    for obj_id in tracked_obj_ids:
        objects_summary[obj_id] = {
            "ious": [],  # IoU values when GT available (for computing avg)
            "frames": [],  # Per-frame data on annotated frames (for training data generation)
            "occlusions": [],  # List of {start_frame, end_frame, occlusion_id}
            "failure_frames": [],  # List of {frame_idx, iou, is_at_sam3_reappearance, occlusion_id, pred_bbox, gt_bbox}
            "num_failures": 0,  # Count of frames with IoU < threshold
        }
    
    # Occlusion tracking state per object
    occlusion_state = {}
    occlusion_counter = defaultdict(int)  # Per-object occlusion counter
    for obj_id in tracked_obj_ids:
        occlusion_state[obj_id] = {
            "is_occluded": False,
            "was_ever_visible": False,
            "occlusion_start_frame": None,
            "current_occlusion_id": None,
            "just_reappeared": False,  # Track if object just reappeared (for next few frames)
            "reappearance_frame": None,  # Frame where reappearance happened
        }
    
    # Build GT masks lookup from annotations
    gt_masks_lookup = annotations.get('masks', {})
    
    # For visualization only (not saved to JSON)
    pred_masks_cache = {}  # {frame_idx: {obj_id: mask}}
    
    logger.info(f"  Tracking {num_frames} frames with {len(tracked_obj_ids)} objects...")
    
    with torch.inference_mode():
        frame_idx = 0
        
        while frame_idx < num_frames:
            # Get features for tracking
            (
                _,
                _,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
            ) = tracker._get_image_feature(inference_state, frame_idx, batch_size)
            
            image = inference_state["images"][frame_idx].cuda(non_blocking=True).float().unsqueeze(0)
            
            # Track
            current_out = tracker.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=False,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                image=image,
                point_inputs=None,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=inference_state["num_frames"],
                track_in_reverse=False,
                run_mem_encoder=True,
            )

            # to cpu to save memory
            for k, v in current_out.items():
                if isinstance(v, torch.Tensor) and k not in ["obj_ptr"]:
                    current_out[k] = v.cpu()
            
            output_dict["non_cond_frame_outputs"][frame_idx] = current_out
            consolidated_frame_inds["non_cond_frame_outputs"].add(frame_idx)
            
            # Process each object
            for i, obj_id in enumerate(obj_ids):
                obj_id = int(obj_id)

                # if first_frame_masks[obj_id]['first_frame'] > frame_idx:
                #     # Object hasn't appeared yet - skip
                #     continue
                
                if obj_id not in objects_summary:
                    objects_summary[obj_id] = {"ious": [], "frames": [], "occlusions": [], "failure_frames": [], "num_failures": 0}
                    occlusion_state[obj_id] = {
                        "is_occluded": False,
                        "was_ever_visible": False,
                        "occlusion_start_frame": None,
                        "current_occlusion_id": None,
                        "just_reappeared": False,
                        "reappearance_frame": None,
                    }
                
                # Extract prediction mask
                pred_masks = current_out.get("pred_masks_high_res", current_out.get("pred_masks"))
                if pred_masks is not None:
                    if pred_masks.ndim == 4:
                        pred_mask = pred_masks[i, 0].cpu().numpy()
                    elif pred_masks.ndim == 3:
                        pred_mask = pred_masks[i].cpu().numpy()
                    else:
                        pred_mask = pred_masks.cpu().numpy()
                    
                    if pred_mask.shape != (video_height, video_width):
                        pred_mask = cv2.resize(pred_mask.astype(np.float32), (video_width, video_height), interpolation=cv2.INTER_LINEAR)
                    pred_mask = (pred_mask > 0.0).astype(np.uint8)
                else:
                    pred_mask = np.zeros((video_height, video_width), dtype=np.uint8)
                
                
                # Detect occlusion/reappearance based ONLY on GT
                # Only check occlusion on frames that have GT annotations (every 4th frame)
                obj_state = occlusion_state[obj_id]
                was_occluded = obj_state["is_occluded"]
                
                # Check if this frame has GT annotations
                frame_has_gt = frame_idx in gt_masks_lookup
                has_gt_mask = False
                
                if frame_has_gt:
                    # This frame has GT annotations - check if object is present
                    if obj_id in gt_masks_lookup[frame_idx]:
                        gt_mask_for_occlusion = gt_masks_lookup[frame_idx][obj_id]
                        if gt_mask_for_occlusion is not None and gt_mask_for_occlusion.sum() > 0:
                            has_gt_mask = True
                    
                    # Only check occlusion on annotated frames
                    # Object is occluded if GT mask doesn't exist or is empty (on this annotated frame)
                    is_now_occluded = not has_gt_mask
                    
                    # Track if object was ever visible (based on GT)
                    if has_gt_mask:
                        obj_state["was_ever_visible"] = True
                    
                    # Occlusion start - only if object was previously visible
                    if not was_occluded and is_now_occluded and obj_state["was_ever_visible"]:
                        obj_state["is_occluded"] = True
                        obj_state["occlusion_start_frame"] = frame_idx
                        occlusion_counter[obj_id] += 1
                        obj_state["current_occlusion_id"] = occlusion_counter[obj_id]
                    
                    # Reappearance (occlusion end)
                    is_at_reappearance = False
                    if was_occluded and not is_now_occluded:
                        obj_state["is_occluded"] = False
                        is_at_reappearance = True
                        obj_state["just_reappeared"] = True
                        obj_state["reappearance_frame"] = frame_idx
                        
                        objects_summary[obj_id]["occlusions"].append({
                            "start_frame": obj_state["occlusion_start_frame"],
                            "end_frame": frame_idx,
                            "occlusion_id": obj_state["current_occlusion_id"],
                        })
                        obj_state["occlusion_start_frame"] = None
                        obj_state["current_occlusion_id"] = None
                else:
                    # Frame doesn't have GT annotations - don't check occlusion
                    # Keep current occlusion state unchanged
                    is_now_occluded = was_occluded
                    is_at_reappearance = False
                
                # Check if we're still in "reappearance window" (within 5 frames after reappearance)
                if obj_state["just_reappeared"] and obj_state["reappearance_frame"] is not None:
                    frames_since_reappearance = frame_idx - obj_state["reappearance_frame"]
                    if frames_since_reappearance > 5:
                        obj_state["just_reappeared"] = False
                
                # Track failures ONLY on frames with GT annotations
                # Only check failures on annotated frames (every 4th frame)
                frame_has_gt_annotations = frame_idx in gt_masks_lookup
                
                if not frame_has_gt_annotations:
                    # Skip failure tracking on non-annotated frames
                    pass
                else:
                    # Frame has GT annotations - check for failures
                    has_gt = False
                    iou = 0.0  # Default to 0.0 (no overlap)
                    gt_mask = None  # Initialize for bbox computation
                    
                    if obj_id in gt_masks_lookup[frame_idx]:
                        gt_mask = gt_masks_lookup[frame_idx][obj_id]
                        
                        # Only compute IoU if object is actually present in GT (non-empty mask)
                        if gt_mask is not None and gt_mask.sum() > 0:
                            has_gt = True
                            iou = compute_mask_iou(pred_mask, gt_mask)
                            objects_summary[obj_id]["ious"].append(iou)
                    
                    # Store per-frame data for training data generation
                    is_correct = has_gt and iou >= iou_threshold
                    objects_summary[obj_id]["frames"].append({
                        "frame_idx": frame_idx,
                        "has_prediction": pred_mask is not None and pred_mask.sum() > 0,
                        "has_gt": has_gt,
                        "iou": round(float(iou), 2),
                        "is_correct": is_correct,
                        "pred_bbox": mask_to_bbox(pred_mask),
                        "gt_bbox": mask_to_bbox(gt_mask) if has_gt else None,
                    })
                    
                    # Track failures: low IoU OR prediction exists but object not in GT (on annotated frame)
                    is_failure = False
                    failure_type = None
                    if has_gt:
                        # Failure if IoU is below threshold
                        if iou < iou_threshold:
                            is_failure = True
                            failure_type = "low_iou"
                    elif pred_mask.sum() > 0:
                        # Failure if prediction exists but object not in GT (false positive on annotated frame)
                        is_failure = True
                        failure_type = "false_positive"
                    
                    if is_failure:
                        objects_summary[obj_id]["num_failures"] += 1
                        
                        # Determine if failure is at SAM3 reappearance
                        # Failures during occlusion (after occlusion starts, before reappearance) are also reappearance failures
                        # Failures at reappearance frame or within 5 frames after are also reappearance failures
                        failure_is_at_sam3_reappearance = (
                            obj_state["is_occluded"] or  # During occlusion
                            is_at_reappearance or  # At reappearance frame
                            obj_state["just_reappeared"]  # Within 5 frames after reappearance
                        )
                        
                        # Get occlusion_id
                        failure_occlusion_id = None
                        if obj_state["is_occluded"]:
                            failure_occlusion_id = obj_state["current_occlusion_id"]
                        elif failure_is_at_sam3_reappearance and obj_state["reappearance_frame"] is not None:
                            # Find the occlusion that just ended
                            for occ in objects_summary[obj_id]["occlusions"]:
                                if occ["end_frame"] == obj_state["reappearance_frame"]:
                                    failure_occlusion_id = occ["occlusion_id"]
                                    break
                        
                        objects_summary[obj_id]["failure_frames"].append({
                            "frame_idx": frame_idx,
                            "iou": round(float(iou), 2),
                            "failure_type": failure_type,
                            "has_gt": has_gt,
                            "is_at_sam3_reappearance": failure_is_at_sam3_reappearance,
                            "occlusion_id": failure_occlusion_id,
                            "pred_bbox": mask_to_bbox(pred_mask),
                            "gt_bbox": mask_to_bbox(gt_mask) if has_gt else None,
                        })
                
                # Store mask for visualization only
                if frame_idx not in pred_masks_cache:
                    pred_masks_cache[frame_idx] = {}
                if pred_mask.sum() > 0:
                    pred_masks_cache[frame_idx][obj_id] = pred_mask.copy()
            
            frame_idx += 1
    
    # Build final summary - just what's needed for sampling
    objects_final = {}
    total_failures = 0
    total_gt_frames = 0
    all_ious = []
    
    for obj_id, data in objects_summary.items():
        avg_iou = np.mean(data["ious"]) if data["ious"] else 0.0
        all_ious.extend(data["ious"])
        total_failures += data["num_failures"]
        total_gt_frames += len(data["ious"])
        
        # Count failures at SAM3 reappearance
        failures_at_reappearance = sum(1 for f in data["failure_frames"] if f.get("is_at_sam3_reappearance", False))
        
        objects_final[str(obj_id)] = {
            "avg_iou": round(float(avg_iou), 2),
            "num_failures": data["num_failures"],
            "num_gt_frames": len(data["ious"]),
            "failed": avg_iou < iou_threshold,
            "frames": data["frames"],  # Per-frame data for training data generation
            "occlusions": data["occlusions"],
            "has_occlusion": len(data["occlusions"]) > 0,
            "failure_frames": data["failure_frames"],  # Frame indices with low IoU + bboxes
            "failures_at_reappearance": failures_at_reappearance,
        }
    
    overall_avg_iou = np.mean(all_ious) if all_ious else 0.0
    failure_rate = total_failures / max(1, total_gt_frames)
    
    # Count total failures at reappearance across all objects
    total_failures_at_reappearance = sum(o["failures_at_reappearance"] for o in objects_final.values())
    
    collected_data = {
        "sequence_id": sequence_id,
        "video_path": video_path,  # MP4 path for frame extraction during training data generation
        "num_frames": num_frames,
        "num_objects": len(objects_final),
        "objects": objects_final,
        "sequence_failed": overall_avg_iou < iou_threshold or failure_rate > 0.5,
        "has_occlusion": any(o["has_occlusion"] for o in objects_final.values()),
        "avg_iou": round(float(overall_avg_iou), 2),
        "failure_rate": round(float(failure_rate), 2),
        "total_failures": total_failures,
        "total_failures_at_reappearance": total_failures_at_reappearance,
        "_pred_masks_cache": pred_masks_cache,  # For visualization only, not saved
    }
    
    logger.info(
        f"  {len(objects_final)} objects, avg_iou={overall_avg_iou:.3f}, "
        f"failed={collected_data['sequence_failed']}, has_occlusion={collected_data['has_occlusion']}, "
        f"failures={total_failures} (at_reappearance={total_failures_at_reappearance})"
    )
    
    return collected_data


def visualize_tracking_video_sav_train(
    collected_data: Dict,
    frames: List[np.ndarray],
    annotations: Dict,
    output_video_path: str,
    fps: int = 6,
) -> None:
    """Create visualization video for SA-V training format.
    
    Two-row layout:
    - Top row: Predictions
    - Bottom row: Ground truth (only on annotated frames)
    """
    if not frames:
        logger.warning("No frames provided for visualization")
        return
    
    H, W = frames[0].shape[:2]
    colors = get_distinct_colors()
    
    # Create video writer (two rows)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H * 2))
    
    if not video_writer.isOpened():
        logger.error(f"Could not open video writer for {output_video_path}")
        return
    
    gt_masks_lookup = annotations.get('masks', {})
    pred_masks_cache = collected_data.get('_pred_masks_cache', {})
    
    # Only visualize frames that have GT annotations (every 4th frame)
    annotated_frame_indices = sorted(gt_masks_lookup.keys())
    
    logger.info(f"Creating visualization video with {len(annotated_frame_indices)} annotated frames (out of {len(frames)} total)...")
    
    for frame_idx in annotated_frame_indices:
        if frame_idx >= len(frames):
            continue
        frame = frames[frame_idx]
        # Top row: Predictions
        pred_frame = frame.copy()
        
        # Draw predictions for each object
        if frame_idx in pred_masks_cache:
            for obj_id, pred_mask in pred_masks_cache[frame_idx].items():
                if pred_mask is None or pred_mask.sum() == 0:
                    continue
                    
                color_idx = (obj_id - 1) % len(colors)
                color = colors[color_idx]
                
                # Resize if needed
                if pred_mask.shape != (H, W):
                    pred_mask = cv2.resize(pred_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                pred_mask = (pred_mask > 0).astype(bool)
                
                # Overlay mask
                alpha = 0.5
                pred_frame[pred_mask] = (
                    pred_frame[pred_mask].astype(np.float32) * (1 - alpha) + 
                    color.astype(np.float32) * alpha
                ).astype(np.uint8)
                
                # Draw contour
                contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(pred_frame, contours, -1, color.tolist(), 2)
                
                # Add object ID label
                if contours:
                    M = cv2.moments(contours[0])
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(pred_frame, str(obj_id), (cx-10, cy+10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color.tolist(), 2)
        
        # Add frame label
        cv2.putText(pred_frame, f"Prediction (Frame {frame_idx})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Bottom row: Ground truth
        gt_frame = frame.copy()
        
        # Only show GT masks on frames that have annotations (every 4th frame)
        if frame_idx in gt_masks_lookup:
            frame_gt_masks = gt_masks_lookup[frame_idx]
        else:
            frame_gt_masks = {}  # No GT masks for this frame
        
        for obj_id, gt_mask in frame_gt_masks.items():
            if gt_mask is None or gt_mask.sum() == 0:
                continue
            
            # Resize if needed
            if gt_mask.shape != (H, W):
                gt_mask = cv2.resize(
                    gt_mask.astype(np.uint8), (W, H),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            
            color_idx = (obj_id - 1) % len(colors)  # obj_id is 1-indexed, colors is 0-indexed
            color = colors[color_idx]
            
            # Overlay mask
            alpha = 0.5
            gt_frame[gt_mask > 0] = (
                gt_frame[gt_mask > 0].astype(np.float32) * (1 - alpha) + 
                color.astype(np.float32) * alpha
            ).astype(np.uint8)
            
            # Draw contour
            contours, _ = cv2.findContours(
                (gt_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(gt_frame, contours, -1, color.tolist(), 2)
            
            # Add object ID label
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(gt_frame, str(obj_id), (cx-10, cy+10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color.tolist(), 2)
        
        cv2.putText(gt_frame, f"Ground Truth (Frame {frame_idx})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Stack vertically
        combined = np.vstack([pred_frame, gt_frame])
        
        # Convert RGB to BGR for video writer
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        video_writer.write(combined_bgr)
    
    video_writer.release()
    logger.info(f"Saved visualization video to {output_video_path}")

def optimize_video(src: Path, dst: Path, max_width: int, overwrite: bool):
    if dst.exists() and not overwrite:
        return

    logger.info(f"Optimizing: {src.name}")

    # FFmpeg Command Breakdown:
    # -an: Removes all audio tracks
    # -vf scale: Resizes width, -2 maintains aspect ratio (must be even for many codecs)
    # -vcodec libx265: The most efficient modern compressor
    # -crf 28: Quality level (23 is default, 28 is smaller, 30+ starts looking blurry)
    # -preset faster: Balancing compression time vs. file size
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(src),
        '-vf', f'scale={max_width}:-2',
        '-vcodec', 'libx265',
        '-crf', '28',
        '-preset', 'faster',
        '-an', 
        str(dst)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to process {src.name}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Collect SAM3 tracking data from SA-V (occlusion/reappearance events and failures)"
    )
    parser.add_argument(
        "--sav_dataset",
        type=str,
        required=True,
        help="Path to SAV dataset (e.g., sav_val)",
    )
    parser.add_argument(
        "--sequence_id",
        type=str,
        default=None,
        help="Specific sequence ID to process (if not provided, processes all)",
    )
    parser.add_argument(
        "--sub_dir",
        type=str,
        default=None,
        help="Specific sub folder of SAV dataset to process (e.g., sav_000, if not provided, processes all sub folders)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./training_data",
        help="Output directory for collected data",
    )
    parser.add_argument(
        "--sam3_checkpoint",
        type=str,
        default=None,
        help="Path to SAM3 checkpoint",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for CORRECT/INCORRECT assessment",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to process",
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save visualization videos showing tracking results",
    )
    parser.add_argument(
        "--max_viz_videos",
        type=int,
        default=None,
        help="Maximum number of visualization videos to save (default: all if --save_videos is set)",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=6,
        help="FPS for output videos",
    )
    parser.add_argument(
        "--sav_train_format",
        action="store_true",
        help="Use SA-V training format (MP4 + JSON instead of JPEG folders)",
    )

    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Whether to overwrite existing output files (JSON and videos)",
    )
    
    args = parser.parse_args()
    
    # Load SAM3 model
    logger.info("Loading SAM3 model...")
    sam3_model = build_sam3_video_model()
    tracker = sam3_model.tracker
    tracker.backbone = sam3_model.detector.backbone
    tracker.eval()
    
    if torch.cuda.is_available():
        tracker = tracker.cuda()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find sequences
    dataset_root = args.sav_dataset
    
    # Detect dataset format
    # SA-V training format: sav_000/sav_000001.mp4 + sav_000001_manual.json
    # SA-V validation format: JPEGImages_24fps/sequence_id/*.jpg + Annotations_6fps/sequence_id/*.png
    
    is_sav_train_format = args.sav_train_format
    
    # Auto-detect format if not specified
    if not is_sav_train_format:
        # Check for sav_XXX subdirs with MP4 files
        subdirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
        for subdir in subdirs[:5]:  # Check first few subdirs
            subdir_path = os.path.join(dataset_root, subdir)
            mp4_files = [f for f in os.listdir(subdir_path) if f.endswith('.mp4')]
            if mp4_files:
                is_sav_train_format = True
                logger.info("Auto-detected SA-V training format (MP4 + JSON)")
                break
    
    if is_sav_train_format:
        # SA-V training format: sav_000/sav_000001.mp4 + sav_000001_manual.json
        logger.info("Using SA-V training format (MP4 + JSON)")
        
        # Find all sav_XXX subdirectories
        sav_subdirs = sorted([
            d for d in os.listdir(dataset_root) 
            if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith('sav_')
        ])
        
        if not sav_subdirs:
            logger.error(f"No sav_XXX subdirectories found in {dataset_root}")
            sys.exit(1)
        
        # Build list of all sequences (video_id from MP4 files)
        all_sequences = []
        for sav_subdir in sav_subdirs:
            subdir_path = os.path.join(dataset_root, sav_subdir)
            mp4_files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.mp4')])
            for mp4_file in mp4_files:
                video_id = mp4_file.replace('.mp4', '')
                # Check if annotation exists
                manual_json = os.path.join(subdir_path, f"{video_id}_manual.json")
                auto_json = os.path.join(subdir_path, f"{video_id}_auto.json")
                if os.path.exists(manual_json) or os.path.exists(auto_json):
                    all_sequences.append({
                        'video_id': video_id,
                        'video_path': os.path.join(subdir_path, mp4_file),
                        'annotation_path': manual_json if os.path.exists(manual_json) else auto_json,
                        'subdir': sav_subdir,
                    })
        
        logger.info(f"Found {len(all_sequences)} sequences with annotations")

        if args.sub_dir:
            # Filter to specific subdirectory
            all_sequences = [s for s in all_sequences if s['subdir'] == args.sub_dir]
            if not all_sequences:
                logger.error(f"Subdirectory {args.sub_dir} not found or contains no annotated sequences")
                sys.exit(1)
        
        if args.sequence_id:
            # Filter to specific sequence
            all_sequences = [s for s in all_sequences if s['video_id'] == args.sequence_id]
            if not all_sequences:
                if args.sub_dir:
                    logger.error(f"Sequence {args.sequence_id} not found in subdirectory {args.sub_dir}")
                else:
                    logger.error(f"Sequence {args.sequence_id} not found")
                sys.exit(1)
        
        if args.max_sequences:
            all_sequences = all_sequences[:args.max_sequences]
        
        logger.info(f"Processing {len(all_sequences)} sequences")
        
        # Process each sequence (SA-V training format)
        skipped_sequences = []
        sequence_summaries = []
        viz_videos_saved = 0

        print("All sequences:", all_sequences)
        
        for seq_idx, seq_info in enumerate(all_sequences):
            video_id = seq_info['video_id']
            video_path = seq_info['video_path']
            annotation_json = seq_info['annotation_path']
            subdir = seq_info['subdir']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing sequence {seq_idx+1}/{len(all_sequences)}: {video_id}")
            logger.info(f"{'='*60}")

            output_subdir = output_dir / subdir
            (output_subdir / "tracking_result").mkdir(parents=True, exist_ok=True)
            (output_subdir / "visualization").mkdir(parents=True, exist_ok=True)
            json_output_path = output_dir / subdir / "tracking_result" / f"{video_id}.json"
            vis_output_path = output_dir / subdir / "visualization" / f"{video_id}.mp4"

            if json_output_path.exists():
                if args.overwrite_existing:
                    logger.info(f"Overwriting existing output for {video_id}")
                else:
                    logger.info(f"SAM3's prediction for {video_id} already exists at {json_output_path}, skipping...")
                    skipped_sequences.append(video_id)
                    continue
            
            try:
                # Load annotations first to check if valid
                annotations = load_sav_train_annotations(annotation_json)
                if not annotations or annotations.get('num_objects', 0) == 0:
                    logger.error(f"No objects in annotations for {video_id}")
                    continue
                
                logger.info(f"  {annotations['video_frame_count']} frames, {annotations['num_objects']} objects, "
                           f"{len(annotations['annotated_frames'])} annotated frames")
                
                # Get first frame masks
                first_frame_masks_sav = load_first_frame_masks_from_sav_train(annotations)
                for k, v in first_frame_masks_sav.items():
                    if v is not None:
                        print(f"  Object {k}: first frame {v['first_frame']}")
                    else:
                        print(f"  Object {k}: no valid first frame mask found")
                # print(f"DEBUG: First frame masks keys: {first_frame_masks_sav}")
                if not first_frame_masks_sav:
                    logger.warning(f"No first-frame masks found for {video_id}, skipping")
                    continue

                # logger.info(f"  First frame masks: {first_frame_masks_sav}")

                # Run tracking - SAM3 loads video directly from MP4
                collected_data = collect_tracking_data_sav_train(
                    sequence_id=video_id,
                    video_path=video_path,
                    annotations=annotations,
                    first_frame_masks=first_frame_masks_sav,
                    tracker=tracker,
                    iou_threshold=args.iou_threshold,
                )
                
                if "error" in collected_data:
                    logger.error(f"Failed to collect data for {video_id}: {collected_data['error']}")
                    continue
                
                # Collect sequence data (excluding mask cache to save space)
                data_to_save = {k: v for k, v in collected_data.items() if not k.startswith('_')}
                
                # Save visualization video if requested (respecting max_viz_videos limit)
                should_save_viz = args.save_videos and (args.max_viz_videos is None or viz_videos_saved < args.max_viz_videos)
                if should_save_viz:
                    # Load frames into memory only for visualization
                    frames = load_frames_from_video(video_path)
                    if frames:
                        visualize_tracking_video_sav_train(
                            collected_data=collected_data,
                            frames=frames,
                            annotations=annotations,
                            output_video_path=str(vis_output_path),
                            fps=args.video_fps,
                        )
                        viz_videos_saved += 1
                        # reduce video size
                        light_vis_output_path = vis_output_path.with_name(f"{vis_output_path.stem}_light.mp4")
                        optimize_video(vis_output_path,
                                    light_vis_output_path, 
                                    max_width=640, 
                                    overwrite=True)
                    else:
                        logger.warning(f"Could not load frames for visualization of {video_id}")
                
                # Save sequence summary
                sequence_summaries.append({
                    "sequence_id": video_id,
                    "subdir": subdir,
                    "num_frames": collected_data["num_frames"],
                    "num_objects": collected_data["num_objects"],
                    "sequence_failed": collected_data["sequence_failed"],
                    "has_occlusion": collected_data["has_occlusion"],
                    "avg_iou": collected_data["avg_iou"],
                    "failure_rate": collected_data["failure_rate"],
                    "total_failures": collected_data["total_failures"],
                    "total_failures_at_reappearance": collected_data["total_failures_at_reappearance"],
                })
                
                logger.info(f"✓ {video_id}: failed={collected_data['sequence_failed']}, "
                           f"has_occlusion={collected_data['has_occlusion']}, "
                           f"avg_iou={collected_data['avg_iou']:.3f}, "
                           f"failures={collected_data['total_failures']} (at_reappearance={collected_data['total_failures_at_reappearance']})")
                
                with open(json_output_path, 'w') as f:
                    json.dump(data_to_save, f, indent=2, default=str)
                logger.info(f"Saved data for {video_id} to {json_output_path}")

            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save summary
    logger.info(f"\n{'='*60}")
    logger.info("Saving summary...")
    
    sequences_with_occlusion = [s for s in sequence_summaries if s.get("has_occlusion", False)]
    failed_sequences = [s for s in sequence_summaries if s.get("sequence_failed", False)]
    
    summary = {
        "total_sequences": len(sequence_summaries),
        "num_skipped_sequences": len(skipped_sequences),
        "sequences_with_occlusion": len(sequences_with_occlusion),
        "failed_sequences_count": len(failed_sequences),
        "failed_sequence_ids": [s["sequence_id"] for s in failed_sequences],
        "sequences_with_occlusion_ids": [s["sequence_id"] for s in sequences_with_occlusion],
        "sequences": sequence_summaries,
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\n{'='*60}")
    logger.info("Data collection complete!")
    logger.info(f"Total sequences: {len(sequence_summaries)}")
    logger.info(f"Sequences with occlusion: {len(sequences_with_occlusion)}")
    logger.info(f"Failed sequences: {len(failed_sequences)}/{len(sequence_summaries)}")
    logger.info(f"Skipped sequences: {len(skipped_sequences)}")
    if failed_sequences:
        logger.info(f"Failed sequence IDs: {[s['sequence_id'] for s in failed_sequences[:10]]}")
        if len(failed_sequences) > 10:
            logger.info(f"  ... and {len(failed_sequences) - 10} more")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}")

    
if __name__ == "__main__":
    main()
