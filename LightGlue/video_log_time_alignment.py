import csv
import json
import os
import re
from datetime import datetime, timedelta


TIME_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")


def parse_dt(text):
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_float(text):
    if text is None:
        return None
    try:
        return float(str(text).strip())
    except ValueError:
        return None


def parse_video_filename_start_utc(video_path: str, utc_offset_hours: float = 3.0):
    base_name = os.path.basename(video_path)
    match = re.search(r"(\d{4})_(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", base_name)
    if not match:
        return None
    year, month, day, hour, minute, second = [int(item) for item in match.groups()]
    local_dt = datetime(year, month, day, hour, minute, second)
    return local_dt - timedelta(hours=float(utc_offset_hours))


def _candidate_epoch_sync_paths(video_path: str) -> list[str]:
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    current_dir = os.path.dirname(os.path.abspath(video_path))
    lightglue_root = os.path.dirname(current_dir)
    return [
        os.path.join(lightglue_root, "diagnostics", f"epoch_sync_{video_stem}.json"),
        os.path.join(os.path.dirname(lightglue_root), "SuperGlue", "diagnostics", f"epoch_sync_{video_stem}.json"),
    ]


def load_cached_epoch_sync_start(video_path: str):
    for report_path in _candidate_epoch_sync_paths(video_path):
        if not os.path.exists(report_path):
            continue
        try:
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        video_start_utc = parse_dt(report.get("video_start_utc"))
        if video_start_utc is None:
            continue
        return {
            "video_start_utc": video_start_utc,
            "report_path": report_path,
            "offset_video_to_log_start_sec": parse_float(report.get("offset_video_to_log_start_sec")),
            "filename_delta_sec": parse_float(report.get("filename_delta_sec")),
        }
    return None


def resolve_video_start_utc(video_path: str, utc_offset_hours: float = 3.0):
    cached = load_cached_epoch_sync_start(video_path)
    if cached is not None:
        return cached["video_start_utc"], "cached_epoch_sync", cached
    parsed = parse_video_filename_start_utc(video_path, utc_offset_hours=utc_offset_hours)
    if parsed is None:
        raise RuntimeError(f"Could not infer video start UTC from filename: {video_path}")
    return parsed, "filename_timestamp", None


def load_log_index(log_csv_root: str):
    rows = []
    index_path = os.path.join(log_csv_root, "export_index.csv")
    if not os.path.exists(index_path):
        return rows
    with open(index_path, "r", encoding="utf-8-sig", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            start_utc = parse_dt(row.get("FirstGpsUtc"))
            end_utc = parse_dt(row.get("LastGpsUtc"))
            folder = (row.get("Folder") or "").strip()
            if not folder or start_utc is None or end_utc is None:
                continue
            rows.append(
                {
                    "log_file": (row.get("LogFile") or "").strip(),
                    "folder": folder,
                    "start": start_utc,
                    "end": end_utc,
                }
            )
    return rows


def choose_log_for_video(video_start_utc: datetime, video_duration_sec: float | None, log_rows: list[dict]):
    if video_start_utc is None or not log_rows:
        return None
    video_end_utc = None if video_duration_sec is None else video_start_utc + timedelta(seconds=float(video_duration_sec))

    best_row = None
    best_key = None
    for row in log_rows:
        overlap_sec = 0.0
        contains_start = row["start"] <= video_start_utc <= row["end"]
        if video_end_utc is not None:
            overlap_start = max(video_start_utc, row["start"])
            overlap_end = min(video_end_utc, row["end"])
            overlap_sec = max(0.0, (overlap_end - overlap_start).total_seconds())

        distance_start = abs((row["start"] - video_start_utc).total_seconds())
        distance_edge = min(
            abs((row["start"] - video_start_utc).total_seconds()),
            abs((row["end"] - video_start_utc).total_seconds()),
        )

        key = (
            1 if overlap_sec > 0 else 0,
            overlap_sec,
            1 if contains_start else 0,
            -distance_start,
            -distance_edge,
        )
        if best_row is None or key > best_key:
            best_row = dict(row)
            best_row["overlap_sec"] = overlap_sec
            best_key = key
    return best_row
