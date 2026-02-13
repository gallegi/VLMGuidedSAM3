import lmdb
import json
import numpy as np
import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy
import cv2
import tempfile
import argparse
import os
import shutil
from PIL import Image
import io
from dotenv import load_dotenv
from huggingface_hub import login
from random import shuffle

from sam3.model_builder import build_sam3_video_model

load_dotenv()
msgpack_numpy.patch()

login(token=os.environ["HF_TOKEN"])

num_videos = 280
pred_color = (0,0,255)
gt_color = (0,255,0)

sam3_model = build_sam3_video_model()
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone

def infer_fn(video_path, frames, anno, W, H):
    x,y,w,h = anno[0].astype(int)
    prompt_bbox = np.array([x,y,x+w,y+h]).astype('float32')
    prompt_bbox[0] = prompt_bbox[0] / W
    prompt_bbox[1] = prompt_bbox[1] / H
    prompt_bbox[2] = prompt_bbox[2] / W
    prompt_bbox[3] = prompt_bbox[3] / H

    all_pred_boxes = []
    all_drawn_images = []
    state = predictor.init_state(video_path=video_path, offload_video_to_cpu=True)
    _, _, _, video_res_masks = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=0,
                    box=prompt_bbox,
                    clear_old_points=True,
                    rel_coordinates=True,
                )

    for (frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores) \
        in predictor.propagate_in_video(state, 
                                        start_frame_idx=0, 
                                        max_frame_num_to_track=len(anno), 
                                        reverse=False, 
                                        propagate_preflight=True):

        mask = video_res_masks[0][0].cpu().numpy()
        mask = mask > 0.0
        non_zero_indices = np.argwhere(mask)
        y_min, x_min, y_max, x_max = 0, 0, 0, 0
        if len(non_zero_indices) > 0:
            y_min, x_min = non_zero_indices.min(axis=0).tolist()
            y_max, x_max = non_zero_indices.max(axis=0).tolist()

        all_pred_boxes.append([x_min, y_min, x_max-x_min, y_max-y_min]) # xywh

        # draw box and mask to image
        draw = frames[frame_idx].copy()

        mask_img = np.zeros((H, W, 3), np.uint8)
        mask_img[mask] = pred_color
        draw = cv2.addWeighted(draw, 1, mask_img, 0.2, 0)

        cv2.rectangle(draw, (x_min, y_min), (x_max, y_max), pred_color, 2)

        # draw gt box
        gt_x, gt_y, gt_w, gt_h = anno[frame_idx].astype(int)
        cv2.rectangle(draw, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), gt_color, 2)

        all_drawn_images.append(draw)

        # if frame_idx == 10: # for debugging purpose
        #     break

    all_pred_boxes = np.array(all_pred_boxes)
    return all_pred_boxes, all_drawn_images


def run_with_temp_video(frames, anno, fps, infer_fn):
    h, w, _ = frames[0].shape

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "input.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for frame in frames:
            writer.write(frame[..., ::-1])  # RGB → BGR

        writer.release()

        output = infer_fn(video_path, frames, anno, w, h)

    # directory + video removed here
    return output

def main(args):
    env = lmdb.open(
        args.lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    all_video_names = [f"{i:08d}" for i in range(num_videos)]
    # shuffle videos to maximize multiprocess efficiency
    shuffle(all_video_names)

    for i, video_name in enumerate(all_video_names):

        if os.path.exists(os.path.join(args.output_dir, "pred_boxes", video_name+"_pred_bbox.txt")):
                print(f"Video {video_name} is already processed. Skipping...")
                continue

        print(f"Processing video {i+1}/ {num_videos}: {video_name}")
        
        try:
            video_data = None
            with env.begin(write=False) as txn:
                video_data = txn.get(video_name.encode("utf-8"))

            img_bytes, annotations = msgpack.unpackb(video_data, raw=False)

            frames = [np.array(Image.open(io.BytesIO(img_bytes[img_idx])).convert("RGB")) 
                        for img_idx in range(len(img_bytes))]

            all_pred_boxes, all_drawn_images = run_with_temp_video(frames, annotations, 30, infer_fn)

            # save visualization result to video
            if args.save_to_video:
                video_path = os.path.join(args.output_dir, f"{video_name}.mp4")

                h, w, _ = all_drawn_images[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))

                for frame in all_drawn_images:
                    writer.write(frame[..., ::-1])  # RGB → BGR

                writer.release()

            # save prediction bbox to file
            all_pred_boxes_str = "\n".join([f"{x},{y},{w},{h}" for x, y, w, h in all_pred_boxes])
            save_box_dir = os.path.join(args.output_dir, "pred_boxes")
            os.makedirs(save_box_dir, exist_ok=True)
            with open(os.path.join(save_box_dir, f"{video_name}_pred_bbox.txt"), "w") as f:
                f.write(all_pred_boxes_str)

        except Exception as e:
            print(f"Error processing video {video_name}: {e}")
            continue
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", required=True, help="Input to video data saved as lmdb")
    parser.add_argument("--save_to_video", action="store_true", help="Save results to a video.")
    parser.add_argument("--output_dir", default="demo_results", help="Output directory.")
    args = parser.parse_args()
    main(args)


