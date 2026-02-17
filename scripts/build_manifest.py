#!/usr/bin/env python3
"""
build_manifest.py

Creates a manifest.json file in data/ containing metadata for reproducibility.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# --- Dataset signature (schema-based) ---------------------

FILELIST_REQUIRED_COLS = {
    "FileName", "EF", "ESV", "EDV", "FrameHeight", "FrameWidth",
    "FPS", "NumberOfFrames", "Split"
}
VOLUMETRACINGS_REQUIRED_COLS = {"FileName", "X1", "Y1", "X2", "Y2", "Frame"}

# expected hashes of "official" CSV contents normalized for line endings.
EXPECTED_NORMALIZED_SHA256 = {
    "filelist": "31942534B7BB3DEBB9F0F3DE42AF2FB4342F7EA918B32A4095A0B66E91FBACE1",
    "volumetracings": "A51A49A8294B031CAC7DFB2DC01E02E6EFEF93A8A1B9D58D607BE12AB5A716E7",
}

FILELIST_DTYPES = {
    "FileName": "string",
    "EF": "float32",
    "ESV": "float32",
    "EDV": "float32",
    "FrameHeight": "int32",
    "FrameWidth": "int32",
    "FPS": "int32",
    "NumberOfFrames": "int32",
    "Split": "string",
}
VOLUMETRACINGS_DTYPES = {
    "FileName": "string",
    "X1": "float32",
    "Y1": "float32",
    "X2": "float32",
    "Y2": "float32",
    "Frame": "int32",
}


# --- Utilities ----------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def try_relpath_posix(path: Path, base: Path) -> str:
    path = path.resolve()
    base = base.resolve()
    try:
        rel = path.relative_to(base)
        return rel.as_posix()
    except Exception:
        return str(path)

def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def sha256_normalized_newlines(path: Path) -> str:
    """
    Normalizes CRLF/CR to LF before hashing, and strips trailing newlines.
    Makes hashes stable across OS line endings.
    """
    b = path.read_bytes()
    b = b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    b = b.rstrip(b"\n")
    return hashlib.sha256(b).hexdigest()

def read_csv_header(path: Path) -> List[str]:
    # utf-8-sig handles BOM if present
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return [h.strip() for h in header if h is not None]

def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)

def find_files(root: Path, max_depth: int) -> Iterable[Path]:
    """
    Depth-limited recursive walk.
    """
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        rel_parts = d.relative_to(root).parts
        if len(rel_parts) > max_depth:
            dirnames[:] = []
            continue
        for name in filenames:
            yield d / name

def is_csv(p: Path) -> bool:
    return p.suffix.lower() == ".csv"

def is_avi(p: Path) -> bool:
    return p.suffix.lower() == ".avi"

def normalize_video_id(s: str) -> str:
    return (s or "").strip().lower()

def stem_from_filename_field(s: str) -> str:
    """
    VolumeTracings 'FileName' includes ".avi" and might include paths.
    We normalize by taking the last path component and then removing the suffix.
    """
    s = (s or "").strip().replace("\\", "/")
    last = s.split("/")[-1]
    return Path(last).stem.strip().lower()


# --- Matching logic -----------------------------------------------------------

@dataclass
class CsvMatch:
    path: Path
    header: List[str]
    nrows: int
    normalized_sha256: str
    raw_sha256: str
    missing_required: List[str]
    extra_cols: List[str]
    expected_hash_ok: Optional[bool]  # None if no expected hash supplied

@dataclass
class VideoMatch:
    root: Path
    avi_count: int
    total_bytes: int

@dataclass
class DatasetManifest:
    schema_version: str
    created_at_utc: str
    repo_root: str
    data_root: str
    detected_dataset_root: str
    filelist: Dict
    volumetracings: Dict
    videos: Dict
    file_index: Dict
    validation: Dict


def best_csv_by_schema(
    data_root: Path,
    required_cols: Set[str],
    max_depth: int,
    expected_hash: Optional[str] = None,
) -> Optional[CsvMatch]:
    candidates: List[Tuple[int, Path, List[str]]] = []

    for p in find_files(data_root, max_depth=max_depth):
        if not is_csv(p):
            continue
        try:
            header = read_csv_header(p)
        except Exception:
            continue

        header_set = set(header)
        missing = sorted(required_cols - header_set)
        if missing:
            continue

        extras = len(header_set - required_cols)
        size = p.stat().st_size
        score = 10_000 - extras * 100 + min(size // 1024, 10_000)
        candidates.append((score, p, header))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    _, best_path, best_header = candidates[0]

    nrows = count_csv_rows(best_path)
    norm_hash = sha256_normalized_newlines(best_path)
    raw_hash = sha256_file(best_path)
    missing = sorted(required_cols - set(best_header))
    extras = sorted(set(best_header) - required_cols)
    expected_ok = (norm_hash == expected_hash) if expected_hash else None

    return CsvMatch(
        path=best_path,
        header=best_header,
        nrows=nrows,
        normalized_sha256=norm_hash,
        raw_sha256=raw_hash,
        missing_required=missing,
        extra_cols=extras,
        expected_hash_ok=expected_ok,
    )

def choose_video_subtree(data_root: Path, max_depth: int, min_avi: int) -> Optional[VideoMatch]:
    avies: List[Path] = []
    for p in find_files(data_root, max_depth=max_depth):
        if is_avi(p):
            avies.append(p)

    if len(avies) < min_avi:
        return None

    counts: Dict[Path, int] = {}
    bytes_by_dir: Dict[Path, int] = {}
    data_root = data_root.resolve()

    for avi in avies:
        size = avi.stat().st_size
        cur = avi.parent.resolve()
        while True:
            if cur == data_root or data_root in cur.parents:
                counts[cur] = counts.get(cur, 0) + 1
                bytes_by_dir[cur] = bytes_by_dir.get(cur, 0) + size
            if cur == data_root:
                break
            cur = cur.parent

    best = sorted(counts.items(), key=lambda kv: (kv[1], len(kv[0].parts)), reverse=True)[0][0]
    return VideoMatch(root=best, avi_count=counts[best], total_bytes=bytes_by_dir[best])

def load_filelist_ids(filelist_csv: Path) -> Set[str]:
    ids: Set[str] = set()
    with filelist_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None or "FileName" not in rdr.fieldnames:
            return ids
        for row in rdr:
            v = normalize_video_id(row.get("FileName") or "")
            if v:
                ids.add(v)
    return ids

def load_volumetracings_ids(vol_csv: Path) -> Set[str]:
    ids: Set[str] = set()
    with vol_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None or "FileName" not in rdr.fieldnames:
            return ids
        for row in rdr:
            v = stem_from_filename_field(row.get("FileName") or "")
            if v:
                ids.add(v)
    return ids

def index_avi_stems(video_root: Path) -> Set[str]:
    stems: Set[str] = set()
    for dirpath, _, filenames in os.walk(video_root):
        for fn in filenames:
            if fn.lower().endswith(".avi"):
                stems.add(Path(fn).stem.lower())
    return stems


# --- File index builder (stream join -> parquet) ------------------------------

def build_file_index_parquet(
    filelist_csv: Path,
    volumetracings_csv: Path,
    out_parquet: Path,
    chunksize: int = 200_000,
) -> Dict:
    """
    Produces file_index.parquet by joining VolumeTracings (many rows per video)
    with FileList (one row per video) on normalized video_id.
    """
    import pandas as pd

    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError as e:
        raise SystemExit(
            "pyarrow is required to write parquet.\n"
            "Install it with: pip install pyarrow\n"
        ) from e

    # Read FileList fully (small)
    fl = pd.read_csv(filelist_csv, dtype=FILELIST_DTYPES, encoding="utf-8-sig")
    if "FileName" not in fl.columns:
        raise SystemExit(f"FileList missing FileName column: {filelist_csv}")

    fl = fl.rename(columns={"FileName": "VideoId"})
    fl["video_id"] = fl["VideoId"].astype("string").map(normalize_video_id)

    # Ensure 1 row per video_id
    fl = fl.drop_duplicates(subset=["video_id"], keep="first")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    if out_parquet.exists():
        out_parquet.unlink()

    writer = None
    total_rows = 0
    unmatched_rows = 0

    # Stream VolumeTracings in chunks
    vt_iter = pd.read_csv(
        volumetracings_csv,
        dtype=VOLUMETRACINGS_DTYPES,
        encoding="utf-8-sig",
        chunksize=chunksize,
    )

    for chunk in vt_iter:
        if "FileName" not in chunk.columns:
            raise SystemExit(f"VolumeTracings missing FileName column: {volumetracings_csv}")

        chunk = chunk.rename(columns={"FileName": "AviFileName"})
        chunk["video_id"] = chunk["AviFileName"].astype("string").map(stem_from_filename_field)

        merged = chunk.merge(fl, on="video_id", how="left")

        # Count unmatched rows (no FileList info found)
        unmatched = merged["VideoId"].isna().sum()
        unmatched_rows += int(unmatched)

        # Output columns: keep a clean, unified view
        # (video_id is internal; keep it too for debugging)
        cols_order = (
            ["video_id", "VideoId", "AviFileName"]
            + [c for c in ["EF", "ESV", "EDV", "FrameHeight", "FrameWidth", "FPS", "NumberOfFrames", "Split"] if c in merged.columns]
            + [c for c in ["X1", "Y1", "X2", "Y2", "Frame"] if c in merged.columns]
        )
        merged = merged[cols_order]

        table = pa.Table.from_pandas(merged, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table.schema, compression="snappy")
        writer.write_table(table)

        total_rows += len(merged)

    if writer is not None:
        writer.close()

    info = {
        "path": str(out_parquet),
        "nrows": int(total_rows),
        "unmatched_rows": int(unmatched_rows),
        "sha256_raw": sha256_file(out_parquet),
        "compression": "snappy",
        "join_key": "video_id (FileList.VideoId == stem(VolumeTracings.AviFileName))",
    }
    return info


# --- Main build ----------------------------------------------------------------

def build_manifest(
    data_root: Path,
    repo_root: Path,
    out_manifest_path: Path,
    out_file_index_path: Path,
    max_depth: int,
    min_avi: int,
    index_chunksize: int,
) -> DatasetManifest:
    data_root = data_root.resolve()
    repo_root = repo_root.resolve()

    filelist = best_csv_by_schema(
        data_root, FILELIST_REQUIRED_COLS, max_depth=max_depth,
        expected_hash=EXPECTED_NORMALIZED_SHA256.get("filelist"),
    )
    if not filelist:
        raise SystemExit(f"Could not find a CSV matching FileList schema under {data_root}")

    volumetracings = best_csv_by_schema(
        data_root, VOLUMETRACINGS_REQUIRED_COLS, max_depth=max_depth,
        expected_hash=EXPECTED_NORMALIZED_SHA256.get("volumetracings"),
    )
    if not volumetracings:
        raise SystemExit(f"Could not find a CSV matching VolumeTracings schema under {data_root}")

    videos = choose_video_subtree(data_root, max_depth=max_depth, min_avi=min_avi)
    if not videos:
        raise SystemExit(f"Could not find a video subtree with >= {min_avi} .avi files under {data_root}")

    # Dataset root = common ancestor of detected pieces
    dataset_root = Path(os.path.commonpath([
        filelist.path.resolve(),
        volumetracings.path.resolve(),
        videos.root.resolve(),
    ])).resolve()

    # Validation: ID overlap
    filelist_ids = load_filelist_ids(filelist.path)
    vol_ids = load_volumetracings_ids(volumetracings.path)
    avi_stems = index_avi_stems(videos.root)

    fl_hit = len(filelist_ids & avi_stems)
    vt_hit = len(vol_ids & avi_stems)

    fl_ratio = fl_hit / max(1, len(filelist_ids))
    vt_ratio = vt_hit / max(1, len(vol_ids))

    validation = {
        "filelist_ids": int(len(filelist_ids)),
        "volumetracings_ids": int(len(vol_ids)),
        "avi_files_indexed": int(len(avi_stems)),
        "filelist_to_avi_match_ratio": float(fl_ratio),
        "volumetracings_to_avi_match_ratio": float(vt_ratio),
        "warnings": [],
    }

    if filelist.expected_hash_ok is False:
        validation["warnings"].append("FileList CSV hash does not match expected (different version?).")
    if volumetracings.expected_hash_ok is False:
        validation["warnings"].append("VolumeTracings CSV hash does not match expected (different version?).")
    if fl_ratio < 0.95:
        validation["warnings"].append("Low FileList↔AVI match ratio; videos may be missing or renamed.")
    if vt_ratio < 0.95:
        validation["warnings"].append("Low VolumeTracings↔AVI match ratio; videos may be missing or renamed.")

    # Build parquet index (streaming join)
    file_index_info = build_file_index_parquet(
        filelist_csv=filelist.path,
        volumetracings_csv=volumetracings.path,
        out_parquet=out_file_index_path,
        chunksize=index_chunksize,
    )
    if file_index_info["unmatched_rows"] > 0:
        validation["warnings"].append(
            f"file_index.parquet has {file_index_info['unmatched_rows']} rows with no FileList match."
        )

    manifest = DatasetManifest(
        schema_version="1.1",
        created_at_utc=utc_now_iso(),
        repo_root=str(repo_root),
        data_root=try_relpath_posix(data_root, repo_root),
        detected_dataset_root=try_relpath_posix(dataset_root, repo_root),
        filelist={
            "path": try_relpath_posix(filelist.path, repo_root),
            "header": filelist.header,
            "nrows": int(filelist.nrows),
            "sha256_raw": filelist.raw_sha256,
            "sha256_normalized_newlines": filelist.normalized_sha256,
            "expected_hash_ok": filelist.expected_hash_ok,
        },
        volumetracings={
            "path": try_relpath_posix(volumetracings.path, repo_root),
            "header": volumetracings.header,
            "nrows": int(volumetracings.nrows),
            "sha256_raw": volumetracings.raw_sha256,
            "sha256_normalized_newlines": volumetracings.normalized_sha256,
            "expected_hash_ok": volumetracings.expected_hash_ok,
        },
        videos={
            "root": try_relpath_posix(videos.root, repo_root),
            "avi_count": int(videos.avi_count),
            "total_bytes": int(videos.total_bytes),
        },
        file_index={
            **file_index_info,
            "path": try_relpath_posix(Path(file_index_info["path"]), repo_root),
        },
        validation=validation,
    )

    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a manifest by structure and also write file_index.parquet.")
    ap.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repo root (default: cwd)")
    ap.add_argument("--data-root", type=Path, default=Path("data"), help="Where the extracted dataset lives (default: data/)")
    ap.add_argument("--manifest-out", type=Path, default=Path("data/manifest.json"), help="Output manifest path")
    ap.add_argument("--file-index-out", type=Path, default=Path("data/file_index.parquet"), help="Output parquet path")
    ap.add_argument("--max-depth", type=int, default=8, help="Max depth to search under data-root")
    ap.add_argument("--min-avi", type=int, default=500, help="Minimum .avi files required to consider a subtree a videos dir")
    ap.add_argument("--index-chunksize", type=int, default=200_000, help="Rows per chunk when building parquet index")
    args = ap.parse_args()

    manifest = build_manifest(
        data_root=args.repo_root / args.data_root,
        repo_root=args.repo_root,
        out_manifest_path=args.repo_root / args.manifest_out,
        out_file_index_path=args.repo_root / args.file_index_out,
        max_depth=args.max_depth,
        min_avi=args.min_avi,
        index_chunksize=args.index_chunksize,
    )

    print(f"Wrote manifest: {(args.repo_root / args.manifest_out).resolve()}")
    print(f"Wrote file index: {(args.repo_root / args.file_index_out).resolve()}")
    print(f"Detected dataset root: {manifest.detected_dataset_root}")
    print(f"FileList CSV: {manifest.filelist['path']} ({manifest.filelist['nrows']} rows)")
    print(f"VolumeTracings CSV: {manifest.volumetracings['path']} ({manifest.volumetracings['nrows']} rows)")
    print(f"Videos root: {manifest.videos['root']} ({manifest.videos['avi_count']} avi)")
    print(f"file_index.parquet: {manifest.file_index['path']} ({manifest.file_index['nrows']} rows)")
    if manifest.validation["warnings"]:
        print("Warnings:")
        for w in manifest.validation["warnings"]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()