#!/usr/bin/env python3
"""
validate_filelist.py

Expected structure (relative to this script file, not terminal CWD):

  <script_dir>/
    validate_filelist.py
    EchoNet-Dynamic/
      FileList.csv
      Video/
        <all video files>

All relative CLI args are resolved relative to this script directory.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXPECTED_COLUMNS = [
    "FileName",
    "EF",
    "ESV",
    "EDV",
    "FrameHeight",
    "FrameWidth",
    "FPS",
    "NumberOfFrames",
    "Split",
]
ALLOWED_SPLITS = {"TRAIN", "VAL", "TEST"}


@dataclass
class Issue:
    kind: str
    message: str
    row_index: Optional[int] = None
    filename: Optional[str] = None
    video_path: Optional[str] = None

    def to_dict(self):
        return {
            "kind": self.kind,
            "message": self.message,
            "row_index": self.row_index,
            "filename": self.filename,
            "video_path": self.video_path,
        }


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def _safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def load_csv(csv_path: Path):
    """Return (rows, columns). Uses pandas if available, else csv module."""
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records"), list(df.columns)
    except ImportError:
        import csv

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            cols = reader.fieldnames or []
        return rows, cols


def index_videos(video_dir: Path, recursive: bool = False) -> Dict[str, List[Path]]:
    stems: Dict[str, List[Path]] = {}
    if not video_dir.exists():
        return stems

    paths = [p for p in (video_dir.rglob("*") if recursive else video_dir.iterdir()) if p.is_file()]
    for p in paths:
        stems.setdefault(p.stem.casefold(), []).append(p)
    return stems


def validate_schema(columns: List[str], allow_extra: bool) -> List[Issue]:
    issues: List[Issue] = []
    missing = [c for c in EXPECTED_COLUMNS if c not in columns]
    if missing:
        issues.append(Issue("schema_missing_columns",
                            f"Missing required columns: {missing}. Expected: {EXPECTED_COLUMNS}"))

    extra = [c for c in columns if c not in EXPECTED_COLUMNS]
    if extra and not allow_extra:
        issues.append(Issue("schema_extra_columns",
                            f"Unexpected extra columns present: {extra}. Use --allow-extra to only warn."))
    elif extra:
        issues.append(Issue("schema_extra_columns_warning", f"Extra columns present (allowed): {extra}"))
    return issues


def validate_rows_basic(rows: List[dict]) -> List[Issue]:
    issues: List[Issue] = []
    seen = set()

    for idx, r in enumerate(rows):
        fn = str(r.get("FileName", "")).strip()
        if not fn:
            issues.append(Issue("value_missing", "FileName is empty.", row_index=idx))
            continue

        if "/" in fn or "\\" in fn:
            issues.append(Issue("value_invalid",
                                "FileName contains a path separator. It should be just a base name.",
                                row_index=idx, filename=fn))

        if Path(fn).suffix:
            issues.append(Issue("value_warning",
                                "FileName includes an extension. Validator matches by stem; "
                                "consider storing extension-less FileName.",
                                row_index=idx, filename=fn))

        key = Path(fn).stem.casefold()
        if key in seen:
            issues.append(Issue("value_duplicate", "Duplicate FileName (by stem) in CSV.",
                                row_index=idx, filename=fn))
        else:
            seen.add(key)

        split = str(r.get("Split", "")).strip().upper()
        if split not in ALLOWED_SPLITS:
            issues.append(Issue("value_invalid",
                                f"Split must be one of {sorted(ALLOWED_SPLITS)}; got '{r.get('Split')}'.",
                                row_index=idx, filename=fn))

        for col in ("EF", "ESV", "EDV"):
            v = _safe_float(r.get(col))
            if v is None:
                issues.append(Issue("value_invalid", f"{col} must be a finite number; got '{r.get(col)}'.",
                                    row_index=idx, filename=fn))
            elif v < 0:
                issues.append(Issue("value_invalid", f"{col} must be >= 0; got {v}.",
                                    row_index=idx, filename=fn))

        for col in ("FrameHeight", "FrameWidth", "FPS", "NumberOfFrames"):
            v = _safe_int(r.get(col))
            if v is None:
                issues.append(Issue("value_invalid", f"{col} must be an integer; got '{r.get(col)}'.",
                                    row_index=idx, filename=fn))
            elif v <= 0:
                issues.append(Issue("value_invalid", f"{col} must be > 0; got {v}.",
                                    row_index=idx, filename=fn))

    return issues


def validate_video_existence(
    rows: List[dict], video_index: Dict[str, List[Path]]
) -> Tuple[List[Issue], Dict[int, Path]]:
    issues: List[Issue] = []
    resolved: Dict[int, Path] = {}

    for idx, r in enumerate(rows):
        fn = str(r.get("FileName", "")).strip()
        if not fn:
            continue

        key = Path(fn).stem.casefold()
        matches = video_index.get(key, [])

        if not matches:
            issues.append(Issue("video_missing",
                                "No matching video found in Video/ for this FileName stem.",
                                row_index=idx, filename=fn))
        elif len(matches) > 1:
            issues.append(Issue("video_ambiguous",
                                f"Multiple matching videos for stem: {[str(p) for p in matches]}",
                                row_index=idx, filename=fn))
        else:
            resolved[idx] = matches[0]

    return issues, resolved


def check_video_metadata_with_opencv(
    rows: List[dict],
    resolved_paths: Dict[int, Path],
    fps_tolerance: float = 1.0,
    frames_tolerance: int = 3,
) -> List[Issue]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return [Issue("metadata_skipped",
                      "OpenCV (cv2) not installed; skipping metadata check. "
                      "Install: pip install opencv-python")]

    issues: List[Issue] = []
    for idx, vp in resolved_paths.items():
        r = rows[idx]
        fn = str(r.get("FileName", "")).strip()

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            issues.append(Issue("metadata_unreadable", "OpenCV could not open video file.",
                                row_index=idx, filename=fn, video_path=str(vp)))
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        exp_w = _safe_int(r.get("FrameWidth")) or 0
        exp_h = _safe_int(r.get("FrameHeight")) or 0
        exp_fps = float(_safe_int(r.get("FPS")) or 0)
        exp_frames = _safe_int(r.get("NumberOfFrames")) or 0

        if (w, h) != (exp_w, exp_h):
            issues.append(Issue("metadata_mismatch",
                                f"Width/Height mismatch: CSV ({exp_w}x{exp_h}) vs video ({w}x{h}).",
                                row_index=idx, filename=fn, video_path=str(vp)))

        if exp_fps > 0 and fps > 0 and abs(fps - exp_fps) > fps_tolerance:
            issues.append(Issue("metadata_mismatch",
                                f"FPS mismatch: CSV {exp_fps} vs video {fps:.3f} (±{fps_tolerance}).",
                                row_index=idx, filename=fn, video_path=str(vp)))

        if exp_frames > 0 and frames > 0 and abs(frames - exp_frames) > frames_tolerance:
            issues.append(Issue("metadata_mismatch",
                                f"Frame count mismatch: CSV {exp_frames} vs video {frames} (±{frames_tolerance}).",
                                row_index=idx, filename=fn, video_path=str(vp)))

    return issues


def resolve_against(base: Path, maybe_path: str) -> Path:
    """
    Resolve maybe_path against base if it's relative.
    This avoids depending on the terminal's current working directory.
    """
    p = Path(maybe_path)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def main():
    script_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=str, default="EchoNet-Dynamic",
                    help="Dataset folder (relative to script unless absolute).")
    ap.add_argument("--csv", type=str, default="",
                    help="Override CSV path (relative to script unless absolute).")
    ap.add_argument("--video-dir", type=str, default="",
                    help="Override video dir (relative to script unless absolute).")
    ap.add_argument("--recursive", action="store_true", help="Search Video/ recursively")
    ap.add_argument("--allow-extra", action="store_true", help="Allow extra CSV columns (warn only)")
    ap.add_argument("--check-metadata", action="store_true", help="Validate video metadata (OpenCV)")
    ap.add_argument("--fps-tol", type=float, default=1.0, help="FPS tolerance for metadata checks")
    ap.add_argument("--frames-tol", type=int, default=3, help="Frame-count tolerance for metadata checks")
    ap.add_argument("--report", type=str, default="",
                    help="Write JSON report (relative to script unless absolute).")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any errors found")
    args = ap.parse_args()

    dataset_dir = resolve_against(script_dir, args.dataset_dir)
    csv_path = resolve_against(script_dir, args.csv) if args.csv else (dataset_dir / "FileList.csv").resolve()
    video_dir = resolve_against(script_dir, args.video_dir) if args.video_dir else (dataset_dir / "Videos").resolve()
    report_path = resolve_against(script_dir, args.report) if args.report else None

    issues: List[Issue] = []

    if not csv_path.exists():
        print(f"FATAL: CSV not found: {csv_path}")
        raise SystemExit(2)

    rows, columns = load_csv(csv_path)
    issues.extend(validate_schema(columns, allow_extra=args.allow_extra))
    issues.extend(validate_rows_basic(rows))

    if not video_dir.exists():
        issues.append(Issue("fatal", f"Video directory not found: {video_dir}"))
    else:
        vid_index = index_videos(video_dir, recursive=args.recursive)
        vid_issues, resolved = validate_video_existence(rows, vid_index)
        issues.extend(vid_issues)

        if args.check_metadata and resolved:
            issues.extend(check_video_metadata_with_opencv(
                rows, resolved, fps_tolerance=float(args.fps_tol), frames_tolerance=int(args.frames_tol)
            ))

    def is_error(kind: str) -> bool:
        return not (kind.endswith("_warning") or kind in {"value_warning", "metadata_skipped"})

    errors = [i for i in issues if is_error(i.kind)]
    warnings = [i for i in issues if not is_error(i.kind)]

    print("=== Validation Summary ===")
    print(f"Script dir:  {script_dir}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"CSV:         {csv_path}")
    print(f"Video dir:   {video_dir}")
    print(f"Rows:        {len(rows)}")
    print(f"Columns:     {len(columns)}")
    print(f"Errors:      {len(errors)}")
    print(f"Warnings:    {len(warnings)}")

    if issues:
        print("\n=== Issues (first 20) ===")
        for n, it in enumerate(issues[:20], 1):
            loc = []
            if it.row_index is not None:
                loc.append(f"row={it.row_index}")
            if it.filename:
                loc.append(f"FileName={it.filename}")
            if it.video_path:
                loc.append(f"video={it.video_path}")
            loc_s = (" " + " ".join(loc)) if loc else ""
            print(f"{n:02d}. [{it.kind}]{loc_s} :: {it.message}")
        if len(issues) > 20:
            print(f"... and {len(issues) - 20} more.")

    if report_path is not None:
        payload = {
            "script_dir": str(script_dir),
            "dataset_dir": str(dataset_dir),
            "csv": str(csv_path),
            "video_dir": str(video_dir),
            "row_count": len(rows),
            "columns": columns,
            "expected_columns": EXPECTED_COLUMNS,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "issues": [i.to_dict() for i in issues],
        }
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote report: {report_path}")

    if args.strict and errors:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
