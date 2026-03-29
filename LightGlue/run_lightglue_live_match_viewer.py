import argparse
import csv
import os
import threading
import time
from dataclasses import dataclass
from datetime import timedelta

import cv2
import numpy as np

from flight_log_navigation import FlightLogNavigation, FlightNavPoint
from georeferenced_orthophoto_map import GeoreferencedOrthophotoMap
from gps_denied_video_orthophoto_localizer import (
    CandidateLocalization,
    FrameLocalizationResult,
    GpsDeniedVideoOrthophotoLocalizer,
)
from lightglue_feature_matcher import LightGlueFeatureMatcher
from orthophoto_tile_index import OrthophotoTileIndex
from video_log_time_alignment import choose_log_for_video, load_log_index, resolve_video_start_utc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class MatchTask:
    frame_index: int
    sampled_index: int
    video_time_sec: float
    frame_utc: object
    frame_bgr: np.ndarray
    prior_map_center_xy: tuple[float, float] | None
    nav_point: FlightNavPoint


@dataclass
class MatchOutput:
    task: MatchTask
    result: FrameLocalizationResult
    candidate: CandidateLocalization | None


def discover_default_video(movie_dir: str) -> str:
    if not os.path.isdir(movie_dir):
        return os.path.join(movie_dir, "2023_0120_134558_001.MP4")
    videos = sorted(
        os.path.join(movie_dir, item_name)
        for item_name in os.listdir(movie_dir)
        if item_name.lower().endswith(".mp4")
    )
    if videos:
        return videos[0]
    return os.path.join(movie_dir, "2023_0120_134558_001.MP4")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LightGlue live video-orthophoto match viewer")
    parser.add_argument("--video-path", default=discover_default_video(os.path.join(BASE_DIR, "movie")))
    parser.add_argument("--orthophoto-path", default=os.path.join(BASE_DIR, "ortho_rtk.tif"))
    parser.add_argument("--log-csv-root", default=os.path.join(BASE_DIR, "log_csv"))
    parser.add_argument("--output-csv-path", default=os.path.join(BASE_DIR, "lightglue_live_viewer_results.csv"))
    parser.add_argument("--feature-backend", choices=["sift", "superpoint", "disk", "aliked"], default="sift")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--extractor-resize", type=int, default=768)
    parser.add_argument("--frame-long-edge-px", type=int, default=960)
    parser.add_argument("--video-start-seconds", type=float, default=0.0)
    parser.add_argument("--frame-step-seconds", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--map-resize-width", type=int, default=1600)
    parser.add_argument("--tile-sizes-px", type=int, nargs="+", default=[512])
    parser.add_argument("--tile-stride-ratio", type=float, default=0.5)
    parser.add_argument("--min-match-count", type=int, default=8)
    parser.add_argument("--min-inlier-count", type=int, default=4)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.10)
    parser.add_argument("--max-reprojection-error-px", type=float, default=6.0)
    parser.add_argument("--local-search-radius-px", type=float, default=420.0)
    parser.add_argument("--relocalize-every-n-frames", type=int, default=12)
    parser.add_argument("--telemetry-prior-tile-limit", type=int, default=3)
    parser.add_argument("--track-local-tile-limit", type=int, default=3)
    parser.add_argument("--depth-confidence", type=float, default=0.95)
    parser.add_argument("--width-confidence", type=float, default=0.99)
    parser.add_argument("--filter-threshold", type=float, default=0.10)
    parser.add_argument("--compile-matcher", action="store_true")
    parser.add_argument("--utc-offset-hours", type=float, default=3.0)
    parser.add_argument("--playback-delay-ms", type=int, default=0)
    parser.add_argument("--map-crop-size-px", type=int, default=720)
    parser.add_argument("--match-line-limit", type=int, default=80)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--preview-output-path", default=os.path.join(BASE_DIR, "lightglue_live_viewer_preview.png"))
    return parser


def resize_to_box(image_bgr: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    src_h, src_w = image_bgr.shape[:2]
    scale = min(float(target_width) / float(src_w), float(target_height) / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    offset_x = (target_width - new_w) // 2
    offset_y = (target_height - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return canvas


def draw_label(image_bgr: np.ndarray, text: str, x: int, y: int, color=(255, 255, 255), scale: float = 0.6):
    cv2.putText(image_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def show_loading_window(title: str, lines: list[str], base_canvas: np.ndarray | None = None):
    if base_canvas is None:
        canvas = np.zeros((900, 1600, 3), dtype=np.uint8)
    else:
        canvas = base_canvas.copy()
        overlay = canvas.copy()
        panel_height = min(canvas.shape[0] - 30, 120 + 42 * len(lines))
        cv2.rectangle(overlay, (24, 24), (980, panel_height), (0, 0, 0), -1, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0.0)
    draw_label(canvas, title, 40, 80, color=(255, 255, 255), scale=1.2)
    y = 150
    for line in lines:
        draw_label(canvas, line, 40, y, color=(120, 220, 255), scale=0.8)
        y += 48
    cv2.imshow("LightGlue Live Match Viewer", canvas)
    cv2.waitKey(1)


def overlay_status_panel(
    base_canvas: np.ndarray,
    title: str,
    lines: list[str],
    x0: int = 24,
    y0: int = 24,
    width: int = 760,
) -> np.ndarray:
    canvas = base_canvas.copy()
    overlay = canvas.copy()
    panel_height = min(canvas.shape[0] - y0 - 16, 96 + 34 * max(1, len(lines)))
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + panel_height), (0, 0, 0), -1, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0.0)
    draw_label(canvas, title, x0 + 18, y0 + 36, color=(255, 255, 255), scale=0.8)
    y = y0 + 76
    for line in lines:
        draw_label(canvas, line, x0 + 18, y, color=(120, 220, 255), scale=0.58)
        y += 34
    return canvas


def compose_viewer_canvas(video_panel: np.ndarray, map_panel: np.ndarray, match_panel: np.ndarray) -> np.ndarray:
    top_row = np.zeros((540, 1600, 3), dtype=np.uint8)
    top_row[:, :640] = video_panel
    top_row[:, 640:] = map_panel
    canvas = np.zeros((900, 1600, 3), dtype=np.uint8)
    canvas[:540] = top_row
    canvas[540:900] = resize_to_box(match_panel, 1600, 360)
    return canvas


def make_video_panel(
    frame_bgr: np.ndarray,
    frame_index: int,
    video_time_sec: float,
    frame_utc,
    nav_point,
    result=None,
    status_text: str | None = None,
    panel_width: int = 640,
    panel_height: int = 540,
) -> np.ndarray:
    panel = resize_to_box(frame_bgr, panel_width, panel_height)
    draw_label(panel, "Current Video Frame", 20, 28, color=(255, 255, 255))
    draw_label(panel, f"Frame: {frame_index}  Time: {video_time_sec:.1f}s", 20, 58, color=(180, 255, 180), scale=0.55)
    draw_label(panel, f"UTC: {frame_utc.strftime('%H:%M:%S.%f')[:-3]}", 20, 86, color=(180, 255, 180), scale=0.55)

    if status_text is None:
        if result is None:
            status_text = "Status: processing current frame"
        else:
            status_text = f"Status: {result.status}  Mode: {result.search_mode}"
    draw_label(panel, status_text, 20, 114, color=(255, 220, 120), scale=0.55)

    if nav_point.latitude is not None and nav_point.longitude is not None:
        draw_label(panel, f"Telem Lat/Lon: {nav_point.latitude:.7f}, {nav_point.longitude:.7f}", 20, 142, color=(255, 255, 0), scale=0.5)
    if nav_point.gps_speed_mps is not None or nav_point.imu_gyro_magnitude is not None:
        speed_text = "n/a" if nav_point.gps_speed_mps is None else f"{nav_point.gps_speed_mps:.2f} m/s"
        imu_text = "n/a" if nav_point.imu_gyro_magnitude is None else f"{nav_point.imu_gyro_magnitude:.4f}"
        draw_label(panel, f"GPS Speed: {speed_text}  IMU GyroMag: {imu_text}", 20, 170, color=(180, 220, 255), scale=0.5)

    if result is not None and result.latitude is not None and result.longitude is not None:
        draw_label(panel, f"Pred Lat/Lon: {result.latitude:.7f}, {result.longitude:.7f}", 20, 198, color=(0, 255, 255), scale=0.5)
    if result is not None and result.score is not None:
        draw_label(panel, f"Score: {result.score:.2f}  Matches/Inliers: {result.match_count}/{result.inlier_count}", 20, 226, color=(0, 255, 180), scale=0.5)
    return panel


def make_match_panel(
    query_bgr: np.ndarray,
    candidate: CandidateLocalization | None,
    match_line_limit: int,
    panel_height: int = 360,
) -> np.ndarray:
    if candidate is None or candidate.tile_image_bgr is None:
        panel = np.zeros((panel_height, 1280, 3), dtype=np.uint8)
        draw_label(panel, "No accepted LightGlue match on this frame", 30, 60, color=(80, 180, 255), scale=0.9)
        return panel

    left = resize_to_box(query_bgr, 560, panel_height)
    right = resize_to_box(candidate.tile_image_bgr, 720, panel_height)
    panel = np.zeros((panel_height, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    panel[:, :left.shape[1]] = left
    panel[:, left.shape[1]:] = right

    query_h, query_w = query_bgr.shape[:2]
    tile_h, tile_w = candidate.tile_image_bgr.shape[:2]
    left_scale = min(float(left.shape[1]) / float(query_w), float(left.shape[0]) / float(query_h))
    right_scale = min(float(right.shape[1]) / float(tile_w), float(right.shape[0]) / float(tile_h))
    left_offset = ((left.shape[1] - int(round(query_w * left_scale))) // 2, (left.shape[0] - int(round(query_h * left_scale))) // 2)
    right_offset = ((right.shape[1] - int(round(tile_w * right_scale))) // 2, (right.shape[0] - int(round(tile_h * right_scale))) // 2)

    query_points = candidate.matched_query_keypoints_xy
    tile_points = candidate.matched_reference_keypoints_xy
    inlier_mask = candidate.inlier_mask
    if query_points is not None and tile_points is not None and inlier_mask is not None:
        draw_indices = np.where(inlier_mask)[0]
        if len(draw_indices) == 0:
            draw_indices = np.arange(len(query_points))
        if len(draw_indices) > match_line_limit:
            step = max(1, len(draw_indices) // match_line_limit)
            draw_indices = draw_indices[::step][:match_line_limit]

        for idx in draw_indices:
            qx = int(round(query_points[idx][0] * left_scale + left_offset[0]))
            qy = int(round(query_points[idx][1] * left_scale + left_offset[1]))
            tx = int(round(tile_points[idx][0] * right_scale + right_offset[0])) + left.shape[1]
            ty = int(round(tile_points[idx][1] * right_scale + right_offset[1]))
            color = (90, 255, 90) if bool(inlier_mask[idx]) else (0, 200, 255)
            cv2.circle(panel, (qx, qy), 2, color, -1, cv2.LINE_AA)
            cv2.circle(panel, (tx, ty), 2, color, -1, cv2.LINE_AA)
            cv2.line(panel, (qx, qy), (tx, ty), color, 1, cv2.LINE_AA)

    draw_label(panel, "Video Frame", 20, 28, color=(255, 255, 255))
    draw_label(panel, f"Matched Tile: {candidate.tile_id}", left.shape[1] + 20, 28, color=(255, 255, 255))
    return panel


def make_map_panel(
    orthophoto_map: GeoreferencedOrthophotoMap,
    candidate: CandidateLocalization | None,
    prior_map_center_xy: tuple[float, float] | None,
    track_history_xy: list[tuple[float, float]],
    crop_size_px: int,
    panel_width: int = 960,
    panel_height: int = 540,
) -> np.ndarray:
    focus_x = orthophoto_map.width_px * 0.5
    focus_y = orthophoto_map.height_px * 0.5
    if candidate is not None:
        focus_x, focus_y = candidate.map_center_x_px, candidate.map_center_y_px
    elif prior_map_center_xy is not None:
        focus_x, focus_y = prior_map_center_xy

    crop_bgr, crop_x0, crop_y0 = orthophoto_map.crop_centered_tile(focus_x, focus_y, crop_size_px)
    overlay = crop_bgr.copy()

    local_track = []
    for x_map, y_map in track_history_xy[-40:]:
        local_track.append((int(round(x_map - crop_x0)), int(round(y_map - crop_y0))))
    if len(local_track) >= 2:
        cv2.polylines(overlay, [np.asarray(local_track, dtype=np.int32)], False, (255, 160, 80), 2, cv2.LINE_AA)

    if prior_map_center_xy is not None:
        px = int(round(prior_map_center_xy[0] - crop_x0))
        py = int(round(prior_map_center_xy[1] - crop_y0))
        cv2.circle(overlay, (px, py), 8, (255, 255, 0), 2, cv2.LINE_AA)
        draw_label(overlay, "Telemetry Prior", px + 10, py - 10, color=(255, 255, 0), scale=0.5)

    if candidate is not None:
        px = int(round(candidate.map_center_x_px - crop_x0))
        py = int(round(candidate.map_center_y_px - crop_y0))
        cv2.circle(overlay, (px, py), 8, (0, 255, 255), -1, cv2.LINE_AA)
        draw_label(overlay, "LightGlue Pred", px + 10, py + 18, color=(0, 255, 255), scale=0.5)

        polygon_local = candidate.projected_polygon_xy.copy()
        polygon_local[:, 0] += float(candidate.tile_map_x0_px - crop_x0)
        polygon_local[:, 1] += float(candidate.tile_map_y0_px - crop_y0)
        cv2.polylines(overlay, [polygon_local.astype(np.int32)], True, (80, 255, 120), 2, cv2.LINE_AA)

    panel = resize_to_box(overlay, panel_width, panel_height)
    draw_label(panel, "Orthophoto Focus View", 20, 28, color=(255, 255, 255))
    return panel


def append_csv_row(csv_path: str, row: dict[str, object], fieldnames: list[str]):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = build_argument_parser().parse_args()

    if not args.no_display:
        cv2.namedWindow("LightGlue Live Match Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LightGlue Live Match Viewer", 1600, 900)
        show_loading_window(
            "LightGlue Live Match Viewer",
            [
                "Viewer aciliyor...",
                "CPU hizli profil varsayilan olarak aciliyor.",
                "Model ve ortofoto hazirlaniyor.",
                "Ilk kare eslestirmesi yine de biraz surebilir.",
            ],
        )

    orthophoto_map = GeoreferencedOrthophotoMap(args.orthophoto_path, resize_width=args.map_resize_width)
    tile_index = OrthophotoTileIndex(orthophoto_map)
    tile_index.build(tile_sizes_px=args.tile_sizes_px, stride_ratio=args.tile_stride_ratio)

    feature_matcher = LightGlueFeatureMatcher(
        feature_backend=args.feature_backend,
        device=args.device,
        max_keypoints=args.max_keypoints,
        extractor_resize=args.extractor_resize,
        depth_confidence=args.depth_confidence,
        width_confidence=args.width_confidence,
        filter_threshold=args.filter_threshold,
        compile_matcher=args.compile_matcher,
    )

    if not args.no_display:
        show_loading_window(
            "LightGlue Live Match Viewer",
            [
                "Viewer acildi.",
                "LightGlue modeli yuklendi.",
                "Ilk frame telemetry prior ile ortofotoda aranacak.",
            ],
        )

    localizer = GpsDeniedVideoOrthophotoLocalizer(
        orthophoto_map=orthophoto_map,
        tile_index=tile_index,
        feature_matcher=feature_matcher,
        frame_long_edge_px=args.frame_long_edge_px,
        min_match_count=args.min_match_count,
        min_inlier_count=args.min_inlier_count,
        min_inlier_ratio=args.min_inlier_ratio,
        max_reprojection_error_px=args.max_reprojection_error_px,
        local_search_radius_px=args.local_search_radius_px,
        relocalize_every_n_frames=args.relocalize_every_n_frames,
        telemetry_prior_tile_limit=args.telemetry_prior_tile_limit,
        track_local_tile_limit=args.track_local_tile_limit,
        precompute_reference_features=False,
    )

    capture = cv2.VideoCapture(args.video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        fps = 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration_sec = (frame_count / fps) if frame_count > 0 else None

    video_start_utc, video_start_source, sync_info = resolve_video_start_utc(args.video_path, utc_offset_hours=args.utc_offset_hours)
    log_rows = load_log_index(args.log_csv_root)
    selected_log = choose_log_for_video(video_start_utc, video_duration_sec, log_rows)
    if selected_log is None:
        raise RuntimeError(f"No matching log folder found under {args.log_csv_root}")
    nav = FlightLogNavigation.from_log_folder(selected_log["folder"])

    print("=" * 72)
    print("LightGlue Live Match Viewer")
    print("=" * 72)
    print(f"video path              : {args.video_path}")
    print(f"log folder              : {selected_log['folder']}")
    print(f"video start utc         : {video_start_utc.strftime('%Y-%m-%d %H:%M:%S.%f')}")
    print(f"video start source      : {video_start_source}")
    if sync_info is not None:
        print(f"epoch sync report       : {sync_info['report_path']}")
    print(f"feature backend         : {args.feature_backend}")
    print(f"max keypoints           : {args.max_keypoints}")
    print(f"extractor resize        : {args.extractor_resize}")
    print(f"tile count              : {len(tile_index.tiles)}")
    print("reference cache mode    : lazy_on_demand")
    print(f"telemetry prior tiles   : {args.telemetry_prior_tile_limit}")
    print(f"track local tiles       : {args.track_local_tile_limit}")
    if args.frame_step_seconds <= 0.0:
        print("match mode              : continuous_latest_frame")
    else:
        print(f"match interval seconds  : {args.frame_step_seconds}")
    print(f"video start seconds     : {args.video_start_seconds}")

    continuous_match_mode = float(args.frame_step_seconds) <= 0.0
    match_interval_frames = None if continuous_match_mode else max(1, int(round(float(args.frame_step_seconds) * fps)))
    start_frame_index = max(0, int(round(float(args.video_start_seconds) * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    frame_index = start_frame_index
    sampled_index = 0
    displayed_frame_count = 0
    track_history_xy: list[tuple[float, float]] = []
    csv_path = args.output_csv_path
    if os.path.exists(csv_path):
        os.remove(csv_path)

    last_canvas = None
    last_match_output: MatchOutput | None = None
    csv_fields = [
        "FrameIndex",
        "VideoTimeSec",
        "Status",
        "SearchMode",
        "Latitude",
        "Longitude",
        "MapCenterXPx",
        "MapCenterYPx",
        "Score",
        "TileId",
        "MatchCount",
        "InlierCount",
        "InlierRatio",
        "MeanMatchConfidence",
        "ReprojectionErrorPx",
    ]

    worker_state = {
        "pending_task": None,
        "latest_output": None,
        "busy": False,
        "stop": False,
        "progress_stage": "idle",
        "progress_current": 0,
        "progress_total": 0,
        "progress_tile": "",
    }
    worker_condition = threading.Condition()
    progress_stage_labels = {
        "idle": "Bekliyor",
        "query_features_start": "Video frame ozellikleri cikartiliyor",
        "query_features_ready": "Video frame ozellikleri hazir",
        "reference_features_start": "Ortofoto tile ozellikleri cikartiliyor",
        "reference_features_ready": "Ortofoto tile ozellikleri hazir",
        "match_features": "LightGlue eslestiriyor",
        "estimate_homography": "Homography hesaplaniyor",
    }

    def matcher_worker():
        while True:
            with worker_condition:
                while worker_state["pending_task"] is None and not worker_state["stop"]:
                    worker_condition.wait(timeout=0.1)
                if worker_state["stop"]:
                    break
                task = worker_state["pending_task"]
                worker_state["pending_task"] = None
                worker_state["busy"] = True
                worker_state["progress_stage"] = "query_features_start"
                worker_state["progress_current"] = 0
                worker_state["progress_total"] = 0
                worker_state["progress_tile"] = ""

            def progress_callback(stage, current_idx, total_idx, tile_id):
                with worker_condition:
                    worker_state["progress_stage"] = stage
                    worker_state["progress_current"] = current_idx
                    worker_state["progress_total"] = total_idx
                    worker_state["progress_tile"] = tile_id or ""

            result, candidate = localizer.localize_frame_with_debug(
                frame_bgr=task.frame_bgr,
                frame_index=task.frame_index,
                sampled_frame_index=task.sampled_index,
                video_time_sec=task.video_time_sec,
                prior_map_center_xy=task.prior_map_center_xy,
                progress_callback=progress_callback,
            )

            with worker_condition:
                worker_state["latest_output"] = MatchOutput(task=task, result=result, candidate=candidate)
                worker_state["busy"] = False
                worker_state["progress_stage"] = "idle"
                worker_state["progress_current"] = 0
                worker_state["progress_total"] = 0
                worker_state["progress_tile"] = ""
                worker_condition.notify_all()

    worker_thread = threading.Thread(target=matcher_worker, name="lightglue_matcher_worker", daemon=True)
    worker_thread.start()

    target_delay_ms = int(round(1000.0 / fps)) if args.playback_delay_ms <= 0 else int(args.playback_delay_ms)

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            video_time_sec = frame_index / fps
            frame_utc = video_start_utc + timedelta(seconds=float(video_time_sec))
            nav_point = nav.at(frame_utc)

            prior_map_center_xy = None
            if nav_point.latitude is not None and nav_point.longitude is not None:
                try:
                    prior_map_center_xy = orthophoto_map.latlon_to_pixel(nav_point.latitude, nav_point.longitude)
                except Exception:
                    prior_map_center_xy = None

            frame_for_localization = localizer._resize_query_frame(frame_bgr)

            if continuous_match_mode:
                should_submit_match = True
            else:
                should_submit_match = (
                    displayed_frame_count == 0
                    or (frame_index - start_frame_index) % match_interval_frames == 0
                )
            if should_submit_match:
                task = MatchTask(
                    frame_index=frame_index,
                    sampled_index=sampled_index,
                    video_time_sec=video_time_sec,
                    frame_utc=frame_utc,
                    frame_bgr=frame_for_localization.copy(),
                    prior_map_center_xy=prior_map_center_xy,
                    nav_point=nav_point,
                )
                with worker_condition:
                    worker_state["pending_task"] = task
                    worker_condition.notify_all()
                sampled_index += 1

            if args.no_display and should_submit_match:
                if continuous_match_mode and displayed_frame_count > 0:
                    pass
                else:
                    deadline = time.perf_counter() + 60.0
                    while time.perf_counter() < deadline:
                        with worker_condition:
                            latest_output = worker_state["latest_output"]
                            worker_busy = worker_state["busy"]
                            pending_task = worker_state["pending_task"]
                        if latest_output is not None and latest_output.task.frame_index == frame_index:
                            break
                        if (not worker_busy) and pending_task is None and latest_output is not None:
                            break
                        time.sleep(0.02)

            with worker_condition:
                latest_output = worker_state["latest_output"]
                worker_busy = bool(worker_state["busy"] or worker_state["pending_task"] is not None)
                progress_stage = worker_state["progress_stage"]
                progress_current = worker_state["progress_current"]
                progress_total = worker_state["progress_total"]
                progress_tile = worker_state["progress_tile"]

            if latest_output is not None and (
                last_match_output is None
                or latest_output.task.frame_index != last_match_output.task.frame_index
            ):
                last_match_output = latest_output
                append_csv_row(csv_path, latest_output.result.as_csv_row(), csv_fields)
                if latest_output.result.map_center_x_px is not None and latest_output.result.map_center_y_px is not None:
                    track_history_xy.append((latest_output.result.map_center_x_px, latest_output.result.map_center_y_px))

            video_panel = make_video_panel(
                frame_bgr=frame_for_localization,
                frame_index=frame_index,
                video_time_sec=video_time_sec,
                frame_utc=frame_utc,
                nav_point=nav_point,
                result=None,
                status_text="Status: live video stream",
            )

            latest_match_age_sec = None
            latest_match_text = "Last Match: waiting for first localization"
            if last_match_output is not None:
                latest_match_age_sec = max(0.0, video_time_sec - last_match_output.task.video_time_sec)
                latest_match_text = (
                    f"Last Match Frame: {last_match_output.task.frame_index}  "
                    f"Age: {latest_match_age_sec:.2f}s  "
                    f"Status: {last_match_output.result.status}"
                )
            draw_label(video_panel, latest_match_text, 20, 198, color=(0, 255, 255), scale=0.5)

            if last_match_output is not None and last_match_output.result.score is not None:
                draw_label(
                    video_panel,
                    f"Last Score: {last_match_output.result.score:.2f}  "
                    f"Matches/Inliers: {last_match_output.result.match_count}/{last_match_output.result.inlier_count}",
                    20,
                    226,
                    color=(0, 255, 180),
                    scale=0.5,
                )

            worker_status_text = f"Matcher: {'calisiyor' if worker_busy else 'hazir'}"
            draw_label(video_panel, worker_status_text, 20, 254, color=(255, 220, 120), scale=0.5)

            map_panel = make_map_panel(
                orthophoto_map=orthophoto_map,
                candidate=(last_match_output.candidate if last_match_output is not None else None),
                prior_map_center_xy=prior_map_center_xy,
                track_history_xy=track_history_xy,
                crop_size_px=args.map_crop_size_px,
                panel_width=960,
                panel_height=540,
            )

            if last_match_output is None:
                match_panel = np.zeros((360, 1280, 3), dtype=np.uint8)
                draw_label(match_panel, "Waiting for first LightGlue match...", 30, 56, color=(80, 180, 255), scale=0.9)
            else:
                match_panel = make_match_panel(
                    last_match_output.task.frame_bgr,
                    last_match_output.candidate,
                    args.match_line_limit,
                    panel_height=360,
                )
                match_age_label = (
                    f"Latest Match Time: {last_match_output.task.video_time_sec:.2f}s"
                    if latest_match_age_sec is None
                    else f"Latest Match Time: {last_match_output.task.video_time_sec:.2f}s  Age: {latest_match_age_sec:.2f}s"
                )
                draw_label(match_panel, match_age_label, 20, 58, color=(255, 220, 120), scale=0.55)

            canvas = compose_viewer_canvas(video_panel, map_panel, match_panel)

            status_lines = [
                (
                    "Video akisi canli. Match modu: her an en guncel frame"
                    if continuous_match_mode
                    else f"Video akisi canli. Match araligi: {args.frame_step_seconds:.2f}s"
                ),
                f"Asama: {progress_stage_labels.get(progress_stage, progress_stage)}",
                f"Backend: {args.feature_backend}  Keypoint: {args.max_keypoints}  Resize: {args.extractor_resize}",
            ]
            if progress_total > 0:
                status_lines.append(f"Tile taraniyor: {progress_current}/{progress_total}")
            if progress_tile:
                status_lines.append(f"Tile: {progress_tile}")
            if worker_busy:
                canvas = overlay_status_panel(canvas, "Background Matcher", status_lines)

            last_canvas = canvas
            if not args.no_display:
                cv2.imshow("LightGlue Live Match Viewer", canvas)
                key = cv2.waitKey(max(1, target_delay_ms)) & 0xFF
                if key in (27, ord("q")):
                    break

            displayed_frame_count += 1
            if args.max_frames is not None and displayed_frame_count >= int(args.max_frames):
                break

            frame_index += 1
            if frame_count > 0 and frame_index >= frame_count:
                break
    finally:
        with worker_condition:
            worker_state["stop"] = True
            worker_condition.notify_all()
        worker_thread.join(timeout=2.0)

    capture.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    elif last_canvas is not None:
        cv2.imwrite(args.preview_output_path, last_canvas)

    print("=" * 72)
    print(f"viewer results csv      : {csv_path}")
    if args.no_display and last_canvas is not None:
        print(f"preview image           : {args.preview_output_path}")


if __name__ == "__main__":
    main()
