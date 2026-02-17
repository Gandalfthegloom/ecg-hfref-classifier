import json
from pathlib import Path


###################################################################################
# Manifest Utilities
###################################################################################
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
