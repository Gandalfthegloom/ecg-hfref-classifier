"""
dataset.py

This script is part of the pipeline with three functions:
"""
# ML Packages
import torch
import torchvision

# Video decoding
import cv2

# data & math
import pandas as pd
import numpy as np
from pathlib import Path

# utils
import sys
ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(ROOT))
from src.utils import load_manifest, get_data_paths

def load_clip(
    path,
    clip_len=16,
    stride=2,
    policy="center" # We want random for training and center for val/test
):
    """
    loads clip using given path through opencv.
    Arguments:
    - clip_len : Amount of frames to take
    - stride : Space between frames (so you take a frame once in every [stride] frames.)
    - policy : random (random start) vs. center (take center window)
    """
    cap = cv2.VideoCapture(path)
    # Check if it actually opens
    if not cap.isOpened():
        print(f"Unable to open {path}")
        return 0

    # Frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Make sure the video reports a valid number of frames
    if frame_count <= 0:
        print(f"Invalid frame count for {path}")
        cap.release()
        return 0

    # Total span (in frames) covered by the sampled clip
    clip_span = (clip_len - 1) * stride + 1

    # Handle short videos by falling back to start at 0
    if frame_count < clip_span:
        start_frame = 0
    else:
        max_start = frame_count - clip_span

        if policy == "center":
            center_index = frame_count // 2
            center_offset = (clip_len // 2) * stride
            start_frame = center_index - center_offset

        elif policy == "random":
            start_frame = np.random.randint(0, max_start + 1)

        else:
            print("Please choose either random or center policy.")
            cap.release()
            raise ValueError

        # Clamp to valid range
        start_frame = int(np.clip(start_frame, 0, max_start))

    frames_indices = [start_frame + stride * i for i in range(clip_len)]

    frames = []
    last_good = None
    for index in frames_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()

        if ok == False:
            # If decoding fails mid-clip, repeat the last good frame (keeps tensor shapes consistent)
            if last_good is None:
                print(f"Unable to read frame in {path}")
                cap.release()
                return 1
            frame = last_good.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        last_good = frame
        frames.append(frame)

    cap.release()
    return frames
        


def main() -> None:
    # Read manifest to get filepath of everything
    manifest_path = ROOT / "data" / "manifest.json"
    manifest = load_manifest(manifest_path)
    path_dict = get_data_paths(manifest, ROOT)
    
    # Get the file list
    filelist = pd.read_csv(path_dict["filelist_csv"])
    fl_input = filelist[["FileName", "EF", "Split"]] # We only care about these columns
    fl_input["label_hfref"] = (fl_input["EF"] <= 40).astype("int8")
    
    # Get path to video
    video_path = path_dict["videos_root"]
    
    # iterate over every video in the filelist
    for row in filelist:
        name = row["filename"] + ".avi"
        path = ROOT / name
        load_clip(path)
    
    

if __name__ == "__main__":
    main()