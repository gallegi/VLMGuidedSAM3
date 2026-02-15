# -*- coding: utf-8 -*-
"""
Reward function for video tracking verification task.

This task asks the VLM to:
1. Assess whether SAM3's tracking prediction is CORRECT or INCORRECT
2. If INCORRECT, provide a correction bounding box for the target object

Expected model output format:
    <think>...</think>
    <answer>{"assessment": "CORRECT"}</answer>
  or:
    <think>...</think>
    <answer>{"assessment": "INCORRECT", "boxes": [x1, y1, x2, y2]}</answer>

Ground truth format (in the "answer" / "ground_truth" field):
    {"assessment": "CORRECT"} or {"assessment": "INCORRECT", "boxes": [x1, y1, x2, y2]}

Reward components:
    - format:    1.0 if <think>...</think><answer>...</answer> tags are present
    - structure: 0.5 if answer JSON is valid with correct keys
    - accuracy:  assessment correctness + bbox IoU for INCORRECT cases
"""

import json
import math
import re
import random
from typing import Any, Dict, List, Optional

# Module-level constants expected by EasyR1's AutoRewardManager
REWARD_NAME = "tracking_verification"
REWARD_TYPE = "batch"

# ---------------------
# Patterns
# ---------------------
THINK_ANSWER_PATTERN = re.compile(
    r"\A\s*<think>.*?</think>\s*<answer>.*?</answer>\s*\Z",
    re.DOTALL,
)

ANSWER_CAPTURE_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL,
)


# ---------------------
# Utilities
# ---------------------
def extract_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = ANSWER_CAPTURE_PATTERN.search(text)
    return m.group(1).strip() if m else None


def _is_list_of_numbers(x, n=None):
    if not isinstance(x, list):
        return False
    if n is not None and len(x) != n:
        return False
    try:
        for v in x:
            float(v)
        return True
    except Exception:
        return False


def _json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def iou_2d(box1: List[float], box2: List[float]) -> float:
    """Compute 2D bounding box IoU."""
    if not _is_list_of_numbers(box1, 4) or not _is_list_of_numbers(box2, 4):
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


# ---------------------
# Format reward
# ---------------------
def tag_format_reward(response: str) -> float:
    """1.0 if response has strict <think>...</think><answer>...</answer> tags."""
    return 1.0 if THINK_ANSWER_PATTERN.fullmatch(response or "") else 0.0


# ---------------------
# Structure reward
# ---------------------
def structure_reward(answer: str) -> float:
    """
    +0.5 if the answer JSON is well-formed with correct keys:
      - Must have "assessment" key with value "CORRECT" or "INCORRECT"
      - If "INCORRECT", must also have "boxes" key with 4 numbers
    """
    obj = _json(answer)
    if not isinstance(obj, dict):
        return 0.0

    assessment = obj.get("assessment")
    if assessment not in ("CORRECT", "INCORRECT"):
        return 0.0

    if assessment == "CORRECT":
        # Valid CORRECT: no boxes needed
        return 0.5

    if assessment == "INCORRECT":
        # Must have valid bbox
        if _is_list_of_numbers(obj.get("boxes"), 4):
            return 0.5
        return 0.0

    return 0.0


# ---------------------
# Accuracy reward
# ---------------------
def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Accuracy reward ∈ [0, 1]:
      - Assessment correctness: 0.5 if assessment matches GT
      - For INCORRECT with correct assessment: 0.5 * IoU of correction bbox
      - For CORRECT with correct assessment: full 0.5 bonus (no bbox needed)
      - Penalty: 0.0 for wrong assessment (especially false negatives)

    This means:
      - Perfect CORRECT prediction:   0.5 (assessment) + 0.5 (no correction needed) = 1.0
      - Perfect INCORRECT prediction:  0.5 (assessment) + 0.5 * IoU                  = up to 1.0
      - Wrong assessment:              0.0
    """
    ans = extract_answer(response) or ""
    pred = _json(ans)
    gt = _json(ground_truth)

    if not isinstance(pred, dict) or not isinstance(gt, dict):
        return 0.0

    pred_assessment = pred.get("assessment")
    gt_assessment = gt.get("assessment")

    if pred_assessment not in ("CORRECT", "INCORRECT") or gt_assessment not in ("CORRECT", "INCORRECT"):
        return 0.0

    # Assessment matches
    if pred_assessment == gt_assessment:
        base = 0.5  # Correct assessment

        if gt_assessment == "CORRECT":
            # No correction needed, full bonus
            return base + 0.5

        # gt_assessment == "INCORRECT": evaluate correction bbox
        pred_boxes = pred.get("boxes")
        gt_boxes = gt.get("boxes")
        if pred_boxes is not None and gt_boxes is not None:
            iou = iou_2d(pred_boxes, gt_boxes)
            return base + 0.5 * iou
        else:
            # Correct assessment but no/bad correction box
            return base

    # Assessment mismatch
    return 0.0


# ---------------------
# Public API (batch interface)
# ---------------------
def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
) -> List[Dict[str, float]]:
    """
    Batch reward interface compatible with EasyR1's BatchFunctionRewardManager.

    Each item in reward_inputs:
        {
            "response": str,        # Model's full output
            "response_length": int,
            "ground_truth": str,    # JSON string: {"assessment": "CORRECT"} or {"assessment": "INCORRECT", "boxes": [...]}
            "data_type": str,       # "image"
            "problem_type": str,    # "tracking_verification"
        }

    Returns: list of dict with keys {overall, format, accuracy, structure_reward}
        overall = (1 - format_weight) * accuracy + format_weight * format + structure_reward
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    results: List[Dict[str, float]] = []

    for idx, item in enumerate(reward_inputs):
        try:
            # Normalize tag whitespaces
            raw_response = item.get("response", "") or ""
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", raw_response)

            gt_raw = item.get("ground_truth", "") or ""
            # ground_truth may be wrapped in <answer>...</answer>; extract if so
            gt_extracted = extract_answer(gt_raw) or gt_raw

            # 1) Format reward
            f_score = tag_format_reward(response)

            # 2) Structure reward
            ans = extract_answer(response) or ""
            s_reward = structure_reward(ans) if f_score > 0 else 0.0

            # 3) Accuracy reward
            a_score = accuracy_reward(response, gt_extracted)

            overall = (1.0 - format_weight) * a_score + format_weight * f_score + s_reward

            results.append({
                "overall": float(overall),
                "format": float(f_score),
                "accuracy": float(a_score),
                "structure_reward": float(s_reward),
            })
        except Exception:
            results.append({
                "overall": 0.0,
                "format": 0.0,
                "accuracy": 0.0,
                "structure_reward": 0.0,
            })

    # Periodic logging (1% of batches)
    if random.random() < 0.01:
        for idx, item in enumerate(reward_inputs):
            print("type", item.get("problem_type", ""))
            print("gt", extract_answer(item.get("ground_truth", "")))
            print("ans", extract_answer(item.get("response", "")))
            print({
                "overall": results[idx]["overall"],
                "format": results[idx]["format"],
                "accuracy": results[idx]["accuracy"],
                "structure_reward": results[idx]["structure_reward"],
            })

    return results
