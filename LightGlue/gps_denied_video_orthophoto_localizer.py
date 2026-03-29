import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from georeferenced_orthophoto_map import GeoreferencedOrthophotoMap
from lightglue_feature_matcher import ExtractedImageFeatures, LightGlueFeatureMatcher, MatchedKeypoints
from orthophoto_tile_index import OrthophotoTile, OrthophotoTileIndex


@dataclass
class CandidateLocalization:
    tile_id: str
    search_mode: str
    map_center_x_px: float
    map_center_y_px: float
    latitude: float
    longitude: float
    score: float
    match_count: int
    inlier_count: int
    inlier_ratio: float
    mean_match_confidence: float
    reprojection_error_px: float
    projected_polygon_xy: np.ndarray
    tile_map_x0_px: int = 0
    tile_map_y0_px: int = 0
    tile_image_bgr: Optional[np.ndarray] = None
    matched_query_keypoints_xy: Optional[np.ndarray] = None
    matched_reference_keypoints_xy: Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None
    homography_matrix: Optional[np.ndarray] = None


@dataclass
class FrameLocalizationResult:
    frame_index: int
    video_time_sec: float
    status: str
    search_mode: str
    latitude: Optional[float]
    longitude: Optional[float]
    map_center_x_px: Optional[float]
    map_center_y_px: Optional[float]
    score: Optional[float]
    tile_id: str
    match_count: int
    inlier_count: int
    inlier_ratio: float
    mean_match_confidence: float
    reprojection_error_px: Optional[float]

    def as_csv_row(self) -> dict[str, object]:
        return {
            "FrameIndex": self.frame_index,
            "VideoTimeSec": round(float(self.video_time_sec), 3),
            "Status": self.status,
            "SearchMode": self.search_mode,
            "Latitude": "" if self.latitude is None else f"{self.latitude:.8f}",
            "Longitude": "" if self.longitude is None else f"{self.longitude:.8f}",
            "MapCenterXPx": "" if self.map_center_x_px is None else f"{self.map_center_x_px:.2f}",
            "MapCenterYPx": "" if self.map_center_y_px is None else f"{self.map_center_y_px:.2f}",
            "Score": "" if self.score is None else f"{self.score:.4f}",
            "TileId": self.tile_id,
            "MatchCount": self.match_count,
            "InlierCount": self.inlier_count,
            "InlierRatio": f"{self.inlier_ratio:.4f}",
            "MeanMatchConfidence": f"{self.mean_match_confidence:.4f}",
            "ReprojectionErrorPx": "" if self.reprojection_error_px is None else f"{self.reprojection_error_px:.4f}",
        }


class GpsDeniedVideoOrthophotoLocalizer:
    def __init__(
        self,
        orthophoto_map: GeoreferencedOrthophotoMap,
        tile_index: OrthophotoTileIndex,
        feature_matcher: LightGlueFeatureMatcher,
        frame_long_edge_px: int = 960,
        min_match_count: int = 20,
        min_inlier_count: int = 8,
        min_inlier_ratio: float = 0.15,
        max_reprojection_error_px: float = 6.0,
        local_search_radius_px: float = 420.0,
        relocalize_every_n_frames: int = 12,
        telemetry_prior_tile_limit: int = 4,
        track_local_tile_limit: int = 4,
        precompute_reference_features: bool = True,
    ) -> None:
        self.orthophoto_map = orthophoto_map
        self.tile_index = tile_index
        self.feature_matcher = feature_matcher
        self.frame_long_edge_px = int(frame_long_edge_px)
        self.min_match_count = int(min_match_count)
        self.min_inlier_count = int(min_inlier_count)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.local_search_radius_px = float(local_search_radius_px)
        self.relocalize_every_n_frames = max(1, int(relocalize_every_n_frames))
        self.telemetry_prior_tile_limit = max(1, int(telemetry_prior_tile_limit))
        self.track_local_tile_limit = max(1, int(track_local_tile_limit))

        self.last_map_center_xy: Optional[tuple[float, float]] = None
        self.reference_features_by_tile_id: dict[str, ExtractedImageFeatures] = {}

        if precompute_reference_features:
            self._precompute_reference_features()

    def _precompute_reference_features(self) -> None:
        total_tiles = len(self.tile_index.tiles)
        print(f"[localizer] Precomputing LightGlue reference features for {total_tiles} orthophoto tiles...")
        for tile in self.tile_index.tiles:
            self.reference_features_by_tile_id[tile.tile_id] = self.feature_matcher.extract_features(tile.image_bgr)
        print("[localizer] Reference feature cache is ready.")

    def localize_video(
        self,
        video_path: str,
        frame_step_seconds: float,
        output_csv_path: str,
        max_frames: int | None = None,
        video_start_seconds: float = 0.0,
    ) -> list[FrameLocalizationResult]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_step = max(1, int(round(float(frame_step_seconds) * fps)))
        start_frame_index = max(0, int(round(float(video_start_seconds) * fps)))

        results: list[FrameLocalizationResult] = []
        sampled_frame_index = 0
        frame_index = start_frame_index
        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok:
                break

            video_time_sec = frame_index / fps
            frame_bgr = self._resize_query_frame(frame_bgr)
            result = self.localize_frame(
                frame_bgr=frame_bgr,
                frame_index=frame_index,
                sampled_frame_index=sampled_frame_index,
                video_time_sec=video_time_sec,
            )
            results.append(result)

            sampled_frame_index += 1
            if max_frames is not None and sampled_frame_index >= int(max_frames):
                break
            frame_index += frame_step
            if total_frames > 0 and frame_index >= total_frames:
                break

        capture.release()
        self._write_results_csv(output_csv_path, results)
        return results

    def localize_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        sampled_frame_index: int,
        video_time_sec: float,
        prior_map_center_xy: Optional[tuple[float, float]] = None,
        progress_callback=None,
    ) -> FrameLocalizationResult:
        result, _ = self._localize_frame_impl(
            frame_bgr=frame_bgr,
            frame_index=frame_index,
            sampled_frame_index=sampled_frame_index,
            video_time_sec=video_time_sec,
            prior_map_center_xy=prior_map_center_xy,
            collect_debug=False,
            progress_callback=progress_callback,
        )
        return result

    def localize_frame_with_debug(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        sampled_frame_index: int,
        video_time_sec: float,
        prior_map_center_xy: Optional[tuple[float, float]] = None,
        progress_callback=None,
    ) -> tuple[FrameLocalizationResult, Optional[CandidateLocalization]]:
        return self._localize_frame_impl(
            frame_bgr=frame_bgr,
            frame_index=frame_index,
            sampled_frame_index=sampled_frame_index,
            video_time_sec=video_time_sec,
            prior_map_center_xy=prior_map_center_xy,
            collect_debug=True,
            progress_callback=progress_callback,
        )

    def _localize_frame_impl(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        sampled_frame_index: int,
        video_time_sec: float,
        prior_map_center_xy: Optional[tuple[float, float]],
        collect_debug: bool,
        progress_callback,
    ) -> tuple[FrameLocalizationResult, Optional[CandidateLocalization]]:
        if progress_callback is not None:
            progress_callback("query_features_start", 0, 0, None)
        frame_features = self.feature_matcher.extract_features(frame_bgr)
        if progress_callback is not None:
            progress_callback("query_features_ready", 0, 0, None)

        use_global_search = (
            self.last_map_center_xy is None
            or sampled_frame_index == 0
            or sampled_frame_index % self.relocalize_every_n_frames == 0
        )

        if (not use_global_search) and self.last_map_center_xy is not None:
            search_mode = "track_local"
            candidate_tiles = self.tile_index.nearest_tiles(
                center_x_px=self.last_map_center_xy[0],
                center_y_px=self.last_map_center_xy[1],
                max_tiles=self.track_local_tile_limit,
            )
            if not candidate_tiles:
                candidate_tiles = self.tile_index.all_tiles()
                search_mode = "global_fallback"
        elif prior_map_center_xy is not None:
            search_mode = "telemetry_prior"
            candidate_tiles = self.tile_index.nearest_tiles(
                center_x_px=prior_map_center_xy[0],
                center_y_px=prior_map_center_xy[1],
                max_tiles=self.telemetry_prior_tile_limit,
            )
            if not candidate_tiles:
                candidate_tiles = self.tile_index.all_tiles()
                search_mode = "global_fallback"
        elif use_global_search:
            search_mode = "global"
            candidate_tiles = self.tile_index.all_tiles()
        else:
            search_mode = "global"
            candidate_tiles = self.tile_index.all_tiles()

        best_candidate = self._evaluate_tiles(
            frame_bgr=frame_bgr,
            frame_features=frame_features,
            candidate_tiles=candidate_tiles,
            search_mode=search_mode,
            collect_debug=collect_debug,
            progress_callback=progress_callback,
        )
        if best_candidate is None and search_mode != "global":
            best_candidate = self._evaluate_tiles(
                frame_bgr=frame_bgr,
                frame_features=frame_features,
                candidate_tiles=self.tile_index.all_tiles(),
                search_mode="global_fallback",
                collect_debug=collect_debug,
                progress_callback=progress_callback,
            )

        if best_candidate is None:
            return FrameLocalizationResult(
                frame_index=frame_index,
                video_time_sec=video_time_sec,
                status="lost",
                search_mode=search_mode,
                latitude=None,
                longitude=None,
                map_center_x_px=self.last_map_center_xy[0] if self.last_map_center_xy else None,
                map_center_y_px=self.last_map_center_xy[1] if self.last_map_center_xy else None,
                score=None,
                tile_id="",
                match_count=0,
                inlier_count=0,
                inlier_ratio=0.0,
                mean_match_confidence=0.0,
                reprojection_error_px=None,
            ), None

        self.last_map_center_xy = (best_candidate.map_center_x_px, best_candidate.map_center_y_px)
        return FrameLocalizationResult(
            frame_index=frame_index,
            video_time_sec=video_time_sec,
            status="localized",
            search_mode=best_candidate.search_mode,
            latitude=best_candidate.latitude,
            longitude=best_candidate.longitude,
            map_center_x_px=best_candidate.map_center_x_px,
            map_center_y_px=best_candidate.map_center_y_px,
            score=best_candidate.score,
            tile_id=best_candidate.tile_id,
            match_count=best_candidate.match_count,
            inlier_count=best_candidate.inlier_count,
            inlier_ratio=best_candidate.inlier_ratio,
            mean_match_confidence=best_candidate.mean_match_confidence,
            reprojection_error_px=best_candidate.reprojection_error_px,
        ), best_candidate

    def _evaluate_tiles(
        self,
        frame_bgr: np.ndarray,
        frame_features: ExtractedImageFeatures,
        candidate_tiles: list[OrthophotoTile],
        search_mode: str,
        collect_debug: bool,
        progress_callback,
    ) -> Optional[CandidateLocalization]:
        best_candidate: Optional[CandidateLocalization] = None
        total_tiles = len(candidate_tiles)
        for tile_index, tile in enumerate(candidate_tiles, start=1):
            if progress_callback is not None:
                progress_callback(search_mode, tile_index, total_tiles, tile.tile_id)
            candidate = self._evaluate_single_tile(
                frame_bgr=frame_bgr,
                frame_features=frame_features,
                tile=tile,
                search_mode=search_mode,
                collect_debug=collect_debug,
                progress_callback=progress_callback,
            )
            if candidate is None:
                continue
            if best_candidate is None or candidate.score > best_candidate.score:
                best_candidate = candidate
        return best_candidate

    def _evaluate_single_tile(
        self,
        frame_bgr: np.ndarray,
        frame_features: ExtractedImageFeatures,
        tile: OrthophotoTile,
        search_mode: str,
        collect_debug: bool,
        progress_callback,
    ) -> Optional[CandidateLocalization]:
        tile_features = self.reference_features_by_tile_id.get(tile.tile_id)
        if tile_features is None:
            if progress_callback is not None:
                progress_callback("reference_features_start", 0, 0, tile.tile_id)
            tile_features = self.feature_matcher.extract_features(tile.image_bgr)
            self.reference_features_by_tile_id[tile.tile_id] = tile_features
            if progress_callback is not None:
                progress_callback("reference_features_ready", 0, 0, tile.tile_id)

        if progress_callback is not None:
            progress_callback("match_features", 0, 0, tile.tile_id)
        matched = self.feature_matcher.match_feature_sets(frame_features, tile_features)
        if matched.match_count < self.min_match_count:
            return None

        if progress_callback is not None:
            progress_callback("estimate_homography", 0, 0, tile.tile_id)
        homography, inlier_mask = self._estimate_homography(matched)
        if homography is None or inlier_mask is None:
            return None

        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = float(inlier_count) / float(max(1, matched.match_count))
        if inlier_count < self.min_inlier_count or inlier_ratio < self.min_inlier_ratio:
            return None

        reprojection_error_px = self._compute_reprojection_error(
            homography=homography,
            query_keypoints_xy=matched.query_keypoints_xy,
            reference_keypoints_xy=matched.reference_keypoints_xy,
            inlier_mask=inlier_mask,
        )
        if reprojection_error_px > self.max_reprojection_error_px:
            return None

        projected_polygon_xy, projected_center_xy = self._project_query_geometry(
            homography=homography,
            query_height=frame_bgr.shape[0],
            query_width=frame_bgr.shape[1],
        )
        if not self._is_projected_center_reasonable(projected_center_xy, tile.tile_size_px):
            return None

        map_center_x_px = float(tile.map_x0_px) + float(projected_center_xy[0])
        map_center_y_px = float(tile.map_y0_px) + float(projected_center_xy[1])
        latitude, longitude = self.orthophoto_map.pixel_to_latlon(map_center_x_px, map_center_y_px)

        mean_confidence = float(matched.match_confidence[inlier_mask].mean()) if inlier_count > 0 else 0.0
        score = (
            1.5 * float(inlier_count)
            + 30.0 * float(inlier_ratio)
            + 10.0 * float(mean_confidence)
            - 0.75 * float(reprojection_error_px)
        )

        return CandidateLocalization(
            tile_id=tile.tile_id,
            search_mode=search_mode,
            map_center_x_px=map_center_x_px,
            map_center_y_px=map_center_y_px,
            latitude=latitude,
            longitude=longitude,
            score=float(score),
            match_count=matched.match_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            mean_match_confidence=mean_confidence,
            reprojection_error_px=float(reprojection_error_px),
            projected_polygon_xy=projected_polygon_xy,
            tile_map_x0_px=int(tile.map_x0_px),
            tile_map_y0_px=int(tile.map_y0_px),
            tile_image_bgr=tile.image_bgr if collect_debug else None,
            matched_query_keypoints_xy=matched.query_keypoints_xy if collect_debug else None,
            matched_reference_keypoints_xy=matched.reference_keypoints_xy if collect_debug else None,
            inlier_mask=inlier_mask if collect_debug else None,
            homography_matrix=homography if collect_debug else None,
        )

    def _estimate_homography(self, matched: MatchedKeypoints) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        query_points = matched.query_keypoints_xy.astype(np.float32)
        reference_points = matched.reference_keypoints_xy.astype(np.float32)
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        homography, inlier_mask = cv2.findHomography(
            query_points,
            reference_points,
            method=method,
            ransacReprojThreshold=self.max_reprojection_error_px,
            confidence=0.999,
            maxIters=10000,
        )
        if homography is None or inlier_mask is None:
            return None, None
        return homography, inlier_mask.ravel().astype(bool)

    @staticmethod
    def _compute_reprojection_error(
        homography: np.ndarray,
        query_keypoints_xy: np.ndarray,
        reference_keypoints_xy: np.ndarray,
        inlier_mask: np.ndarray,
    ) -> float:
        projected = cv2.perspectiveTransform(query_keypoints_xy.reshape(-1, 1, 2), homography).reshape(-1, 2)
        error = np.linalg.norm(projected - reference_keypoints_xy, axis=1)
        if not np.any(inlier_mask):
            return float("inf")
        return float(error[inlier_mask].mean())

    @staticmethod
    def _project_query_geometry(homography: np.ndarray, query_height: int, query_width: int) -> tuple[np.ndarray, np.ndarray]:
        query_corners_and_center = np.array(
            [
                [0.0, 0.0],
                [float(query_width - 1), 0.0],
                [float(query_width - 1), float(query_height - 1)],
                [0.0, float(query_height - 1)],
                [float(query_width - 1) * 0.5, float(query_height - 1) * 0.5],
            ],
            dtype=np.float32,
        )
        projected = cv2.perspectiveTransform(query_corners_and_center.reshape(-1, 1, 2), homography).reshape(-1, 2)
        return projected[:4], projected[4]

    @staticmethod
    def _is_projected_center_reasonable(projected_center_xy: np.ndarray, tile_size_px: int) -> bool:
        margin = float(tile_size_px) * 0.15
        x, y = float(projected_center_xy[0]), float(projected_center_xy[1])
        return (-margin <= x <= float(tile_size_px) + margin) and (-margin <= y <= float(tile_size_px) + margin)

    @staticmethod
    def _write_results_csv(output_csv_path: str, results: list[FrameLocalizationResult]) -> None:
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
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
                ],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(result.as_csv_row())

    def _resize_query_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_height, frame_width = frame_bgr.shape[:2]
        current_long_edge = max(frame_height, frame_width)
        if current_long_edge <= self.frame_long_edge_px:
            return frame_bgr

        scale = float(self.frame_long_edge_px) / float(current_long_edge)
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        return cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
