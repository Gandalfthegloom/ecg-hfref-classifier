"""
dataset.py

This script is part of the pipeline with three functions:
"""
# ML Packages
import torch
import torchvision
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

# Video decoding
import cv2

# data & math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
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
    - stride : Space between frames (so we take a frame once in every [stride] frames.)
    - policy : random (random start) vs. center (take center window)
    """
    path = str(path)
    cap = cv2.VideoCapture(path)
    # Check if it actually opens
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Unable to open video: {path}")


    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            raise RuntimeError(f"Invalid frame count reported for video: {path}")

        # Total span (in frames) covered by the sampled clip
        clip_span = (clip_len - 1) * stride + 1

        # If the video is shorter than the span, we'll still sample by clamping indices
        if frame_count <= clip_span:
            max_start = 0
        else:
            max_start = frame_count - clip_span

        if policy == "center":
            start_frame = max_start // 2  # truly centered window
        elif policy == "random":
            start_frame = np.random.randint(0, max_start + 1)
        else:
            raise ValueError(f"policy must be 'center' or 'random', got: {policy}")

        # Clamp indices so we never seek past the last frame (prevents first-frame failure)
        frames_indices = [
            min(start_frame + stride * i, frame_count - 1) for i in range(clip_len)
        ]

        frames = []
        last_good_bgr = None

        for index in frames_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()

            if (not ok) or (frame is None):
                if last_good_bgr is None:
                    raise RuntimeError(f"Unable to decode first requested frame in: {path}")
                frame = last_good_bgr.copy()
            else:
                # Handle grayscale outputs
                if frame.ndim == 2:
                    frame = frame[:, :, None]  # (H, W) -> (H, W, 1)

                if frame.shape[2] == 1:
                    frame = np.repeat(frame, 3, axis=2)  # (H, W, 1) -> (H, W, 3)
                elif frame.shape[2] != 3:
                    raise ValueError(f"Unexpected frame shape from decoder: {frame.shape}")

                last_good_bgr = frame.copy()

            # OpenCV gives BGR; convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        return frames

    finally:
        cap.release()
        


def run_clip_sanity_check(
    fl_input: pd.DataFrame,
    videos_root: Path,
    clip_len: int = 16,
    stride: int = 2,
    policy: str = "center",
    max_clips: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Lightweight decoder sanity check we can reuse anytime.

    Prints:
      - progress per clip
      - number of frames returned
      - unique frame shapes within each clip
      - DONE!/FAILED per clip
      - total timing summary

    Returns:
      A summary dict with counts and any anomalies.
    """
    import time

    # How many clips are we actually going to process?
    total_available = len(fl_input)
    total = total_available if max_clips is None else min(total_available, max_clips)

    # Overall timer (use perf_counter for decent timing resolution)
    overall_t0 = time.perf_counter()

    # We'll use the first "good" clip's frame shape as the baseline
    expected_frame_shape = None

    # Simple counters
    n_ok, n_fail = 0, 0

    # Tracking lists (kept lightweight and easy to inspect after the run)
    intra_clip_inconsistent: List[str] = []          # clips where frames vary in shape within the clip
    shape_mismatches: List[Tuple[str, tuple]] = []   # clips whose shape differs from the baseline
    failures: List[Tuple[str, str]] = []             # (video_name, error_string)

    # Per-clip timing so we can report avg/max at the end
    clip_times: List[float] = []

    # Iterate rows efficiently (avoids pandas overhead of .iloc in a loop)
    rows = fl_input.itertuples(index=False)
    for i, row in enumerate(rows, start=1):
        if i > total:
            break  # stop early if max_clips was set

        # Manifest/filelist must have FileName (same expectation as elsewhere in our pipeline)
        name = getattr(row, "FileName", None)
        if name is None:
            raise ValueError("fl_input must contain a 'FileName' column.")

        # Build full path to the video file
        video_file = Path(videos_root) / str(name)

        # If our manifest stores bare names, add extension; otherwise keep it
        if video_file.suffix == "":
            video_file = video_file.with_suffix(".avi")

        print(f"[{i}/{total}] Starting clip: {video_file.name}")

        # Per-clip timer (helps spot slow decodes / bad files)
        clip_t0 = time.perf_counter()
        try:
            # Decode the clip with our existing loader
            frames = load_clip(video_file, clip_len=clip_len, stride=stride, policy=policy)

            # Gather shapes so we can detect weird decoder outputs (e.g. grayscale, mismatched sizes)
            shapes = [f.shape for f in frames]
            unique_shapes = sorted(set(shapes))

            print(f"    frames: {len(frames)}")
            print(f"    unique frame shapes in clip: {unique_shapes}")

            # If there is more than 1 unique shape, something is inconsistent inside this clip
            if len(unique_shapes) != 1:
                intra_clip_inconsistent.append(video_file.name)
                print("    WARNING: frame shapes are not consistent within this clip!")
            else:
                # One unique shape -> compare to baseline
                this_shape = unique_shapes[0]
                if expected_frame_shape is None:
                    # First good clip becomes baseline (simple + pragmatic)
                    expected_frame_shape = this_shape
                    print(f"    baseline frame shape set to: {expected_frame_shape}")
                elif this_shape != expected_frame_shape:
                    # Shape changed vs baseline -> log it (doesn't fail the run, just flags it)
                    shape_mismatches.append((video_file.name, this_shape))
                    print(
                        f"    WARNING: frame shape changed vs baseline! "
                        f"baseline={expected_frame_shape}, this_clip={this_shape}"
                    )

            # Record timing and mark as OK
            clip_dt = time.perf_counter() - clip_t0
            clip_times.append(clip_dt)
            print(f"    DONE! clip_time={clip_dt:.3f}s\n")
            n_ok += 1

        except Exception as e:
            # Catch decode errors per-clip so one bad file doesn't kill the whole scan
            clip_dt = time.perf_counter() - clip_t0
            clip_times.append(clip_dt)

            # Store a compact error string for quick debugging
            err = f"{type(e).__name__}: {e}"
            failures.append((video_file.name, err))

            print(f"    FAILED after {clip_dt:.3f}s: {err}\n")
            n_fail += 1

    # Final summary timings
    overall_dt = time.perf_counter() - overall_t0
    avg_time = (sum(clip_times) / len(clip_times)) if clip_times else 0.0
    max_time = max(clip_times) if clip_times else 0.0

    print(
        f"ALL DONE! processed={n_ok + n_fail}, ok={n_ok}, failed={n_fail}, "
        f"total_time={overall_dt:.3f}s, avg_clip_time={avg_time:.3f}s, max_clip_time={max_time:.3f}s"
    )

    # Return structured summary so we can log it, dump it, or assert on it in tests
    return {
        "processed": n_ok + n_fail,
        "ok": n_ok,
        "failed": n_fail,
        "baseline_shape": expected_frame_shape,
        "intra_clip_inconsistent": intra_clip_inconsistent,
        "shape_mismatches": shape_mismatches,
        "failures": failures,
        "total_time_s": overall_dt,
        "avg_clip_time_s": avg_time,
        "max_clip_time_s": max_time,
    }



def frames_to_tc_hw(frames_rgb_uint8: list[np.ndarray]) -> torch.Tensor:
    """
    frames_rgb_uint8: list of (H, W, 3) uint8 arrays in RGB order
    returns: (T, C, H, W) uint8 torch tensor
    """
    clip_thwc = np.stack(frames_rgb_uint8, axis=0)          # (T, H, W, C)
    clip_tchw = torch.from_numpy(clip_thwc).permute(0, 3, 1, 2)  # (T, C, H, W)
    return clip_tchw  # keep uint8; preprocess will scale + normalize



class EchoDataset(Dataset):
    def __init__(self, df, videos_root, split: str, clip_len=16, stride=2):
        self.df = df[df["Split"] == split].reset_index(drop=True)
        self.videos_root = Path(videos_root)
        self.clip_len = clip_len
        self.stride = stride
        
        # Train uses random clips; val/test use center clips
        self.policy = "random" if split.lower() == "train" else "center"

        weights = R2Plus1D_18_Weights.KINETICS400_V1
        self.preprocess = weights.transforms()
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = row["FileName"]
        label = row["label_hfref"]

        video_path = self.videos_root / filename
        # handle extension if needed
        if video_path.suffix == "":
            video_path = video_path.with_suffix(".avi")

        frames = load_clip(video_path, clip_len=self.clip_len, stride=self.stride, policy=self.policy)
        clip_tchw = frames_to_tc_hw(frames)
        x = self.preprocess(clip_tchw)  # -> (C, T, H, W), float32

        y = torch.tensor(label, dtype=torch.float32)

        return x, y
        

# Main is mostly for checking that the processing works.
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
    
    # Sanity Check: try a few center clips
    # Set max_clips to a small number (e.g., 10) for quick checks, or None to scan everything.
    _summary = run_clip_sanity_check(
        fl_input=fl_input,
        videos_root=video_path,
        clip_len=16,
        stride=2,
        policy="center",
        max_clips=10,
    )
    

    
    
# if __name__ == "__main__":
#     main()