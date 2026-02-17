import json
import os
import random
import numpy as np
import torch
from pathlib import Path


# -------------------- Manifest Utilities -----------------------------------------=
def load_manifest(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))

def get_data_paths(manifest: dict, repo_root: str | Path = ".") -> dict:
    rr = Path(repo_root)
    return {
        "filelist_csv": rr / manifest["filelist"]["path"],
        "volumetracings_csv": rr / manifest["volumetracings"]["path"],
        "videos_root": rr / manifest["videos"]["root"],
    }


# -------------- seed setter for reproducibility --------------------------------
def seed_everything(seed: int):
    """Seed all major RNG sources for reproducibility."""
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed] Global seed set to {seed}")

