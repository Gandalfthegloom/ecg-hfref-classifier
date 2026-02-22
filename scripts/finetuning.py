import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc

import sys
ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(ROOT))
from src.utils import seed_everything, load_manifest, get_data_paths
from dataset import EchoDataset, load_clip, frames_to_tc_hw
from models import build_r2plus1d_finetuner, build_mvit_finetuner, get_model_transform

def evaluate_clip_averaging(model, df_eval, videos_root, transform, device, num_clips=5):
    """
    EchoNet-Dynamic Inference Style:
    Sample multiple clips across the video, run inference on each, and average the probabilities.
    """
    model.eval()
    all_filenames = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Clip Averaging Eval"):
            filename = row["FileName"]
            label = row["label_hfref"]
            video_path = videos_root / (filename if filename.endswith('.avi') else f"{filename}.avi")
            
            # Predict on multiple random clips (to simulate covering the whole beat cycle)
            clip_probs = []
            for _ in range(num_clips):
                try:
                    frames = load_clip(video_path, policy="random")
                    x = frames_to_tc_hw(frames)
                    x = transform(x).unsqueeze(0).to(device) # Add batch dim
                    
                    logit = model(x)
                    prob = torch.sigmoid(logit).item()
                    clip_probs.append(prob)
                except Exception as e:
                    continue # Skip broken clips
            
            if clip_probs:
                all_probs.append(np.mean(clip_probs))
                all_labels.append(label)
                all_filenames.append(filename)

    auc = roc_auc_score(all_labels, all_probs)
    preds = (np.array(all_probs) >= 0.5).astype(int)
    f1 = f1_score(all_labels, preds)
    
    results_df = pd.DataFrame({
        "FileName": all_filenames,
        "Label": all_labels,
        "Probability": all_probs
    })
    
    return auc, f1, results_df

def run_finetuning_pipeline(
    model_type: str, 
    batch_size: int, 
    tune_blocks: int,
    epochs: int, 
    lr: float, 
    filelist: pd.DataFrame, 
    videos_root: Path, 
    device: torch.device
):
    print(f"\n" + "="*50)
    print(f"STARTING PIPELINE FOR: {model_type.upper()}")
    print("="*50)
    
    transform = get_model_transform(model_type)
    
    # Datasets 
    train_ds = EchoDataset(filelist, videos_root, "train", transform=transform, policy="random")
    val_ds   = EchoDataset(filelist, videos_root, "val", transform=transform, policy="center")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model Setup
    print(f"Building {model_type.upper()} Fine-Tuner...")
    if model_type == "r2plus1d":
        model = build_r2plus1d_finetuner(pretrained=True, num_classes=1, tune_blocks=tune_blocks)
    else:
        model = build_mvit_finetuner(pretrained=True, num_classes=1, tune_blocks=tune_blocks)
        
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,}")

    # Training Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # To prevent CPU crashes with non-cuda devices. (Gemini recommended this, somehow it works??)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Training Loop
    best_val_auc = 0.0
    weights_dir = ROOT / "outputs" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_path = weights_dir / f"{model_type}_finetuned_best.pth"

    # How many epochs? 
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Actual fine-tuning part.
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for _, x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits.view(-1), y.view(-1))
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_probs, val_labels = [], []
        
        with torch.no_grad():
            for _, x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                x = x.to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(y.numpy())
                
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_auc) # Change the learning rate of next epoch based on the AUC score.
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f" -> Saved new best model (AUC: {best_val_auc:.4f})")

    # Final Evaluation & Saving Raw Predictions
    print(f"\nEvaluating Best {model_type.upper()} Model on Test Set...")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    df_test = filelist[filelist["Split"] == "test"]
    
    test_auc, test_f1, df_preds = evaluate_clip_averaging(
        model=model, 
        df_eval=df_test, 
        videos_root=videos_root, 
        transform=transform, 
        device=device,
        num_clips=5  
    )
    
    preds_dir = ROOT / "outputs" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / f"{model_type}_test_predictions.csv"
    
    df_preds.to_csv(preds_path, index=False)
    print(f"Saved raw test predictions to: {preds_path}")
    
    #  Memory cleanup before the next model runs
    print(f"Cleaning up VRAM...")
    del model
    del optimizer
    del train_loader
    del val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return test_auc, test_f1
