# ML Packages
import torch
import torchvision
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torch.utils.data import DataLoader

# Video decoding
import cv2

# data & math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple

# script imports & utilities
from dataset import EchoDataset
from models import build_mvit_extractor, build_r2plus1d_extractor, LinearProbe, get_model_transform

import sys
ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(ROOT))
from src.utils import seed_everything, load_manifest, get_data_paths
from tqdm import tqdm # Epic progress bar makes everything 25% faster, trust me


def extract_features(
    df: pd.DataFrame,
    videos_root: Path,
    output_path: Path,
    split: str, 
    model_type: str,
    batch_size: int = 8,
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    Function to instantiate a backbone (CNN or MViT), process the dataset, 
    and save embeddings to disk.

    Args:
        df (pd.DataFrame): Master filelist.
        videos_root (Path): Path to .avi files.
        output_path (Path): Destination .pt file.
        split (str): 'train', 'val', or 'test'.
        model_type (str): 'r2plus1d' or 'mvit'. Controls model loading and preprocessing.
        batch_size (int): Batch size (lower this for MViT if OOM occurs).
    """

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_type.upper()} | {split.upper()}] using device: {device}")
    
    # 1. Switch logic for Model and Transform
    # Based on 'model_type', initialize the correct model and get the correct transform.
    # We also need to record the output dimension (dim) for the metadata.
    if model_type == "r2plus1d":
        model = build_r2plus1d_extractor(pretrained=True)
        transform = get_model_transform("r2plus1d")
        output_dim = 512
        desc_label = "R(2+1)D"
    elif model_type == "mvit":
        model = build_mvit_extractor(pretrained=True)
        transform = get_model_transform("mvit")
        output_dim = 768
        desc_label = "MViT"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Move to device and freeze with eval
    model = model.to(device)
    model.eval()
    
    # 2. Dataset Setup
    ds = EchoDataset(
        df=df,
        videos_root=videos_root,
        split=split,
        clip_len=16,
        stride=2,
        transform=transform
    )
    
    if len(ds) == 0:
        print(f"WARNING: Dataset for split '{split}' is empty!")
        return {}

    loader = DataLoader(
        dataset=ds,
        batch_size=batch_size, # Adjust based on device specs; If too big, batches will slow down processing.
        shuffle=False, # STRICTLY FALSE. WE NEED LABELS FOR THE EMBEDDINGS
        num_workers=num_workers,      
        pin_memory=True if device.type == 'cuda' else False 
    )
    
    # 3. Extraction Loop
    all_embeddings = []
    all_names = []
    all_labels = []
    
    with torch.no_grad():
        for names, x, y in tqdm(loader, desc=f"Extracting {desc_label} ({split})"):
            x = x.to(device)
            
            # Forward pass
            embeddings = model(x)
            
            # Move to CPU immediately, freeing up GPU space
            all_embeddings.append(embeddings.cpu())
            all_labels.append(y.cpu())
            all_names.extend(names)
            
    # 4. Save to Disk
    if len(all_embeddings) == 0:
        return {}

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    feature_data = {
        "embeddings": embeddings_tensor,
        "filenames": all_names,
        "labels": labels_tensor,
        "model_name": model_type,
        "split": split,
        "dim": output_dim
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feature_data, output_path)
    print(f"Saved to: {output_path}")
    print(f"Shape: {embeddings_tensor.shape}")

    return feature_data


def main():
    # 1. Setup Data Paths
    manifest_path = ROOT / "data" / "manifest.json"
    manifest = load_manifest(manifest_path)
    path_dict = get_data_paths(manifest, ROOT)
    
    # 2. Load FileList and Normalize Split Column
    filelist = pd.read_csv(path_dict["filelist_csv"])
    
    if "Split" in filelist.columns:
        filelist["Split"] = filelist["Split"].astype(str).str.lower().str.strip()
    else:
        raise ValueError("CRITICAL: 'Split' column not found in FileList.csv!")

    fl_input = filelist[["FileName", "EF", "Split"]].copy()
    fl_input["label_hfref"] = (fl_input["EF"] <= 40).astype("int8")
    
    video_path = path_dict["videos_root"]
    output_dir = ROOT / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Configuration for the Run
    # We define a list of configurations for each model
    # These are picked based on MY laptop's specs, if you have different ones, switch accordingly.
    tasks = [
        {"model": "r2plus1d", "batch_size": 16},
        {"model": "mvit",     "batch_size": 4} # Smaller batch for transformer
    ]
    splits = ["train", "val", "test"]
    
    # 4. Master Loop
    print(f"Starting extraction for {len(tasks)} models over {len(splits)} splits...")

    for task in tasks:
        model_name = task["model"]
        bs = task["batch_size"]
        
        print(f"\n" + "="*40)
        print(f"      MODEL: {model_name.upper()}      ")
        print("="*40)

        for split in splits:
            save_filename = f"{model_name}_{split}_features.pt"
            save_path = output_dir / save_filename
            
            if save_path.exists():
                print(f"Skipping {split}, file already exists: {save_filename}")
                continue
            
            extract_features(
                df=fl_input,
                videos_root=video_path,
                output_path=save_path,
                split=split,
                model_type=model_name,
                batch_size=bs
            )

    print("\nAll extractions complete. Ready for Phase 2 (train_probe.py).")

if __name__ == "__main__":
    seed_everything(42) # To ensure reproducibility. Also 42 is just a great number d;
    main()