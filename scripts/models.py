import torch
import torch.nn as nn
from torchvision.models.video import (
    r2plus1d_18, 
    R2Plus1D_18_Weights,
    mvit_v2_s,
    MViT_V2_S_Weights
)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

def build_r2plus1d_extractor(pretrained: bool = True) -> nn.Module:
    """
    Returns R(2+1)D-18 backbone.
    Output shape: (Batch, 512)
    """
    weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
    model = r2plus1d_18(weights=weights)
    
    # Replace the classification head with Identity to get the 512-dim vector
    model.fc = Identity()
    
    # Freeze strictly for extraction
    for p in model.parameters():
        p.requires_grad = False
        
    model.eval()
    return model

def build_mvit_extractor(pretrained: bool = True) -> nn.Module:
    """
    Returns MVIT v2 Small backbone.
    Output shape: (Batch, 768)
    """
    weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
    model = mvit_v2_s(weights=weights)
    
    # The head in MVIT is usually a Sequential(Dropout, Linear)
    # We replace the whole head structure with Identity
    model.head = Identity()
    
    # Freeze strictly for extraction
    for p in model.parameters():
        p.requires_grad = False
        
    model.eval()
    return model

class LinearProbe(nn.Module):
    """
    Simple Linear Classifier (Logistic Regression) for Phase 2.
    Takes pre-computed embeddings as input.
    """
    def __init__(self, input_dim: int, num_classes: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        
        # Optional: BatchNorm can help stabilize linear probes on disparate features
        self.bn = nn.BatchNorm1d(input_dim)
        
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Input_Dim)
        x = self.bn(x)
        x = self.drop(x)
        return self.fc(x)

def get_model_transform(model_name: str):
    """
    Factory to get the correct preprocessing transform for the specific backbone.
    """
    if model_name == 'r2plus1d':
        return R2Plus1D_18_Weights.KINETICS400_V1.transforms()
    elif model_name == 'mvit':
        return MViT_V2_S_Weights.KINETICS400_V1.transforms()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    

def main():
    """
    Integration test:
    1. Generates a dummy video and dataframe.
    2. Instantiates EchoDataset.
    3. Runs a sample through R(2+1)D and MVIT to verify output shapes.
    """
    import sys
    import os
    import tempfile
    import cv2
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    print("--- Starting models.py Test ---")
    
    # Try to import EchoDataset. 
    # Assumes dataset.py is in the same directory as models.py
    try:
        from dataset import EchoDataset
    except ImportError:
        # Fallback: add current dir to path
        sys.path.append(str(Path(__file__).parent))
        try:
            from dataset import EchoDataset
        except ImportError:
            print("[Error] Could not import EchoDataset. Ensure dataset.py is in the same folder.")
            return

    # Create a temporary directory for our dummy video
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"[Setup] Created temporary workspace: {temp_path}")
        
        # 1. Create a dummy video (30 frames, 112x112, random noise)
        video_name = "test_dummy.avi"
        video_file = temp_path / video_name
        
        # Video params
        width, height = 112, 112
        fps = 30
        frames_to_write = 30
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))
        
        for _ in range(frames_to_write):
            # Create a random RGB frame
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        print(f"[Setup] Generated dummy video: {video_name}")
        
        # 2. Create a dummy DataFrame
        df = pd.DataFrame({
            "FileName": [video_name],
            "Split": ["train"],
            "label_hfref": [1] # Example label
        })

        # --- TEST 1: R(2+1)D Pipeline ---
        print("\n[Test 1] Verifying R(2+1)D Pipeline...")
        
        try:
            transform_cnn = get_model_transform("r2plus1d")
            ds_cnn = EchoDataset(
                df=df,
                videos_root=temp_path,
                split="train",
                clip_len=16,
                stride=1,
                transform=transform_cnn
            )
            
            # Load one sample
            name, x, y = ds_cnn[0]
            print(f"    - Sample loaded: {name}")
            print(f"    - Tensor shape (C, T, H, W): {x.shape}")
            print(f"    - Label: {y.item()}")
            
            # Initialize Model
            print("    - Building R(2+1)D extractor (pretrained=False)...")
            model_cnn = build_r2plus1d_extractor(pretrained=False)
            
            # Forward Pass
            # Add batch dimension: (C, T, H, W) -> (1, C, T, H, W)
            input_tensor = x.unsqueeze(0)
            
            with torch.no_grad():
                emb = model_cnn(input_tensor)
                
            print(f"    - Output Embedding Shape: {emb.shape}")
            
            if emb.shape == (1, 512):
                print("    [SUCCESS] R(2+1)D output matches expected (1, 512).")
            else:
                print(f"    [FAILURE] R(2+1)D output mismatch. Expected (1, 512), got {emb.shape}")

        except Exception as e:
            print(f"    [FAILURE] R(2+1)D Pipeline crashed: {e}")
            import traceback
            traceback.print_exc()

        # --- TEST 2: MVIT Pipeline ---
        print("\n[Test 2] Verifying MVIT Pipeline...")
        
        try:
            transform_mvit = get_model_transform("mvit")
            ds_mvit = EchoDataset(
                df=df,
                videos_root=temp_path,
                split="train",
                clip_len=16,
                stride=1,
                transform=transform_mvit
            )
            
            # Load one sample
            name, x, y = ds_mvit[0]
            print(f"    - Sample loaded: {name}")
            print(f"    - Tensor shape (C, T, H, W): {x.shape}")
            
            # Initialize Model
            print("    - Building MVIT extractor (pretrained=False)...")
            model_mvit = build_mvit_extractor(pretrained=False)
            
            # Forward Pass
            input_tensor = x.unsqueeze(0)
            
            with torch.no_grad():
                emb = model_mvit(input_tensor)
                
            print(f"    - Output Embedding Shape: {emb.shape}")
            
            if emb.shape == (1, 768):
                print("    [SUCCESS] MVIT output matches expected (1, 768).")
            else:
                print(f"    [FAILURE] MVIT output mismatch. Expected (1, 768), got {emb.shape}")
                
        except Exception as e:
            print(f"    [FAILURE] MVIT Pipeline crashed: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    main()