import argparse
import csv
import os
import re
import unicodedata
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


GPS_EPOCH = datetime(1980, 1, 6)
LEAP_SECONDS = 18
KEY_CHANNELS = {
    "GPS": "GNSS position and velocity",
    "GPA": "GNSS accuracy metrics",
    "ATT": "attitude controller state",
    "AHR2": "fused attitude and position",
    "POS": "position estimate",
    "IMU": "raw fused IMU sample",
    "ACC": "accelerometer stream",
    "GYR": "gyroscope stream",
    "BARO": "barometer",
    "MAG": "magnetometer",
    "BAT": "battery",
    "MODE": "flight mode changes",
    "CAM": "camera trigger pose",
    "TRIG": "camera trigger events",
    "XKF1": "EKF state estimate",
    "XKF2": "EKF sensor state",
    "XKF3": "EKF innovations",
    "XKF4": "EKF consistency",
    "XKF5": "EKF terrain/range",
    "VIBE": "vibration diagnostics",
    "RCIN": "input PWM",
    "RCOU": "output PWM",
}


def sanitize_name(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_only).strip("_.")
    return cleaned or "log"


def gps_to_utc(week_value: str, ms_value: str) -> str:
    try:
        week = float(week_value)
        milliseconds = float(ms_value)
    except ValueError:
        return ""
    utc = GPS_EPOCH + timedelta(weeks=week, milliseconds=milliseconds) - timedelta(seconds=LEAP_SECONDS)
    return utc.strftime("%Y-%m-%d %H:%M:%S.%f")


def parse_time_us(value: str):
    try:
        return int(float(value))
    except ValueError:
        return None


def is_info_line(line: str) -> bool:
    return line.isascii() and "\x00" not in line


def build_time_interpolator(anchors):
    anchors = sorted(anchors, key=lambda item: item[0])
    if not anchors:
        return lambda _time_us: ""

    xs = [item[0] for item in anchors]
    epoch = datetime(1970, 1, 1)
    ys = [(item[1] - epoch).total_seconds() for item in anchors]

    def interpolate(time_us):
        if time_us is None:
            return ""
        index = bisect_left(xs, time_us)
        if len(xs) == 1:
            ts = ys[0]
        elif index <= 0:
            x0, x1 = xs[0], xs[1]
            y0, y1 = ys[0], ys[1]
            ratio = 0.0 if x1 == x0 else (time_us - x0) / float(x1 - x0)
            ts = y0 + ratio * (y1 - y0)
        elif index >= len(xs):
            x0, x1 = xs[-2], xs[-1]
            y0, y1 = ys[-2], ys[-1]
            ratio = 0.0 if x1 == x0 else (time_us - x0) / float(x1 - x0)
            ts = y0 + ratio * (y1 - y0)
        else:
            x0, x1 = xs[index - 1], xs[index]
            y0, y1 = ys[index - 1], ys[index]
            ratio = 0.0 if x1 == x0 else (time_us - x0) / float(x1 - x0)
            ts = y0 + ratio * (y1 - y0)
        return (epoch + timedelta(seconds=ts)).strftime("%Y-%m-%d %H:%M:%S.%f")

    return interpolate


def first_pass(log_path: Path):
    formats = {}
    counts = Counter()
    info_count = 0
    unparsed_count = 0
    gps_anchors = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = [item.strip() for item in line.split(",")]
            if not parts:
                continue
            message_type = parts[0]

            if message_type == "FMT" and len(parts) >= 6:
                counts[message_type] += 1
                formats[parts[3]] = parts[5:]
                continue

            if message_type in formats:
                counts[message_type] += 1
            elif is_info_line(line):
                info_count += 1
            else:
                unparsed_count += 1

            if message_type == "GPS" and len(parts) >= 6:
                try:
                    status = int(float(parts[3]))
                except ValueError:
                    continue
                if status < 3:
                    continue
                time_us = parse_time_us(parts[1])
                utc_str = gps_to_utc(parts[5], parts[4])
                if time_us is None or not utc_str:
                    continue
                gps_anchors.append((time_us, datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S.%f")))

    return formats, counts, gps_anchors, info_count, unparsed_count


def exact_gps_utc(columns, values):
    if "GWk" in columns and "GMS" in columns:
        return gps_to_utc(values[columns.index("GWk")], values[columns.index("GMS")])
    if "GPSWeek" in columns and "GPSTime" in columns:
        return gps_to_utc(values[columns.index("GPSWeek")], values[columns.index("GPSTime")])
    return ""


def export_log(log_path: Path, output_root: Path):
    formats, counts, gps_anchors, info_count, unparsed_count = first_pass(log_path)
    interpolate_utc = build_time_interpolator(gps_anchors)

    log_output_dir = output_root / sanitize_name(log_path.stem)
    log_output_dir.mkdir(parents=True, exist_ok=True)

    writers = {}
    handles = {}
    summary_rows = []
    info_writer = None
    info_handle = None
    unparsed_writer = None
    unparsed_handle = None

    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                raw_parts = line.split(",", 1)
                message_type = raw_parts[0].strip()
                if message_type == "FMT":
                    continue

                columns = formats.get(message_type)
                if not columns:
                    if is_info_line(line):
                        if info_writer is None:
                            info_path = log_output_dir / "INFO.csv"
                            info_handle = info_path.open("w", newline="", encoding="utf-8")
                            info_writer = csv.writer(info_handle)
                            info_writer.writerow(["LineNo", "RawLine"])
                        info_writer.writerow([line_no, line])
                    else:
                        if unparsed_writer is None:
                            unparsed_path = log_output_dir / "UNPARSED.csv"
                            unparsed_handle = unparsed_path.open("w", newline="", encoding="utf-8")
                            unparsed_writer = csv.writer(unparsed_handle)
                            unparsed_writer.writerow(["LineNo", "RawLine"])
                        unparsed_writer.writerow([line_no, repr(line)])
                    continue

                parts = [item.strip() for item in line.split(",", len(columns))]
                values = parts[1:]
                if len(values) < len(columns):
                    values += [""] * (len(columns) - len(values))
                elif len(values) > len(columns):
                    values = values[: len(columns)]

                if message_type not in writers:
                    extra_columns = ["LineNo"]
                    if "TimeUS" in columns:
                        extra_columns.append("UTC_Approx")
                    if exact_gps_utc(columns, values):
                        extra_columns.append("UTC_GPS")

                    output_path = log_output_dir / f"{sanitize_name(message_type)}.csv"
                    output_handle = output_path.open("w", newline="", encoding="utf-8")
                    writer = csv.writer(output_handle)
                    writer.writerow(extra_columns + columns)
                    writers[message_type] = (writer, columns, extra_columns)
                    handles[message_type] = output_handle

                writer, columns, extra_columns = writers[message_type]
                output_row = [line_no]
                if "UTC_Approx" in extra_columns:
                    time_us = parse_time_us(values[columns.index("TimeUS")])
                    output_row.append(interpolate_utc(time_us))
                if "UTC_GPS" in extra_columns:
                    output_row.append(exact_gps_utc(columns, values))
                output_row.extend(values)
                writer.writerow(output_row)

        formats_path = log_output_dir / "formats.csv"
        with formats_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Message", "Columns"])
            for name in sorted(formats):
                writer.writerow([name, ", ".join(formats[name])])

        for message_type, count in sorted(counts.items()):
            if message_type == "FMT":
                continue
            file_name = f"{sanitize_name(message_type)}.csv"
            columns = formats.get(message_type, [])
            summary_rows.append([
                message_type,
                count,
                1 if "TimeUS" in columns else 0,
                1 if (("GWk" in columns and "GMS" in columns) or ("GPSWeek" in columns and "GPSTime" in columns)) else 0,
                file_name,
                KEY_CHANNELS.get(message_type, ""),
                ", ".join(columns),
            ])

        if info_count:
            summary_rows.append(["INFO", info_count, 0, 0, "INFO.csv", "plain-text information lines", "RawLine"])
        if unparsed_count:
            summary_rows.append(["UNPARSED", unparsed_count, 0, 0, "UNPARSED.csv", "binary or broken lines", "RawLine"])

        summary_path = log_output_dir / "message_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Message", "Count", "HasTimeUS", "HasExactGpsUtc", "File", "Purpose", "Columns"])
            writer.writerows(summary_rows)

        overview_path = log_output_dir / "log_overview.csv"
        first_utc = gps_anchors[0][1].strftime("%Y-%m-%d %H:%M:%S.%f") if gps_anchors else ""
        last_utc = gps_anchors[-1][1].strftime("%Y-%m-%d %H:%M:%S.%f") if gps_anchors else ""
        with overview_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["LogFile", "AnchorCount", "FirstGpsUtc", "LastGpsUtc", "MessageTypeCount", "InfoLineCount", "UnparsedLineCount"])
            writer.writerow([log_path.name, len(gps_anchors), first_utc, last_utc, len(counts) - (1 if "FMT" in counts else 0), info_count, unparsed_count])

    finally:
        for handle in handles.values():
            handle.close()
        if info_handle is not None:
            info_handle.close()
        if unparsed_handle is not None:
            unparsed_handle.close()

    return {
        "log": log_path.name,
        "folder": str(log_output_dir),
        "message_types": len([key for key in counts if key != "FMT"]) + (1 if info_count else 0) + (1 if unparsed_count else 0),
        "gps_anchors": len(gps_anchors),
        "first_gps_utc": gps_anchors[0][1].strftime("%Y-%m-%d %H:%M:%S.%f") if gps_anchors else "",
        "last_gps_utc": gps_anchors[-1][1].strftime("%Y-%m-%d %H:%M:%S.%f") if gps_anchors else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Export ArduPilot flight logs into per-message CSV files for the LightGlue workspace")
    parser.add_argument("--logs-dir", default=os.path.join(os.path.dirname(__file__), "logs"))
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "log_csv"))
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_files = sorted(logs_dir.glob("*.log"))
    if not log_files:
        raise FileNotFoundError(f"No .log files found under {logs_dir}")

    index_rows = []
    for log_path in log_files:
        info = export_log(log_path, output_dir)
        index_rows.append(info)
        print(f"exported {sanitize_name(log_path.name)} -> {info['folder']}")

    index_path = output_dir / "export_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["LogFile", "Folder", "MessageTypes", "GpsAnchors", "FirstGpsUtc", "LastGpsUtc"])
        for row in index_rows:
            writer.writerow([
                row["log"],
                row["folder"],
                row["message_types"],
                row["gps_anchors"],
                row["first_gps_utc"],
                row["last_gps_utc"],
            ])


if __name__ == "__main__":
    main()
