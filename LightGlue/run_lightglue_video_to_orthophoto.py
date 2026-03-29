import argparse
import os

from georeferenced_orthophoto_map import GeoreferencedOrthophotoMap
from gps_denied_video_orthophoto_localizer import GpsDeniedVideoOrthophotoLocalizer
from lightglue_feature_matcher import LightGlueFeatureMatcher
from orthophoto_tile_index import OrthophotoTileIndex


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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
    parser = argparse.ArgumentParser(description="LightGlue tabanli GPS-denied video -> orthophoto lokalizasyonu")
    parser.add_argument("--video-path", default=discover_default_video(os.path.join(BASE_DIR, "movie")))
    parser.add_argument("--orthophoto-path", default=os.path.join(BASE_DIR, "ortho_rtk.tif"))
    parser.add_argument("--output-csv-path", default=os.path.join(BASE_DIR, "lightglue_video_to_orthophoto_results.csv"))
    parser.add_argument("--feature-backend", choices=["sift", "superpoint", "disk", "aliked"], default="aliked")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--extractor-resize", type=int, default=1024)
    parser.add_argument("--frame-long-edge-px", type=int, default=960)
    parser.add_argument("--video-start-seconds", type=float, default=0.0)
    parser.add_argument("--frame-step-seconds", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--map-resize-width", type=int, default=1600)
    parser.add_argument("--tile-sizes-px", type=int, nargs="+", default=[512, 768])
    parser.add_argument("--tile-stride-ratio", type=float, default=0.5)
    parser.add_argument("--min-match-count", type=int, default=20)
    parser.add_argument("--min-inlier-count", type=int, default=8)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.15)
    parser.add_argument("--max-reprojection-error-px", type=float, default=6.0)
    parser.add_argument("--local-search-radius-px", type=float, default=420.0)
    parser.add_argument("--relocalize-every-n-frames", type=int, default=12)
    parser.add_argument("--depth-confidence", type=float, default=0.95)
    parser.add_argument("--width-confidence", type=float, default=0.99)
    parser.add_argument("--filter-threshold", type=float, default=0.10)
    parser.add_argument("--compile-matcher", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    orthophoto_map = GeoreferencedOrthophotoMap(
        tif_path=args.orthophoto_path,
        resize_width=args.map_resize_width,
    )
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
        precompute_reference_features=True,
    )

    print("=" * 72)
    print("LightGlue GPS-Denied Video -> Orthophoto Localizer")
    print("=" * 72)
    print(f"video path              : {args.video_path}")
    print(f"orthophoto path         : {args.orthophoto_path}")
    print(f"output csv path         : {args.output_csv_path}")
    print(f"feature backend         : {args.feature_backend}")
    print(f"device                  : {feature_matcher.device}")
    print(f"tile count              : {len(tile_index.tiles)}")
    print(f"tile sizes (px)         : {args.tile_sizes_px}")
    print(f"video start (sec)       : {args.video_start_seconds}")
    print(f"frame step (sec)        : {args.frame_step_seconds}")
    print(f"min matches / inliers   : {args.min_match_count} / {args.min_inlier_count}")
    print(f"min inlier ratio        : {args.min_inlier_ratio}")
    print(f"max reprojection px     : {args.max_reprojection_error_px}")
    print(f"local search radius px  : {args.local_search_radius_px}")
    print(f"relocalize every frames : {args.relocalize_every_n_frames}")

    results = localizer.localize_video(
        video_path=args.video_path,
        video_start_seconds=args.video_start_seconds,
        frame_step_seconds=args.frame_step_seconds,
        output_csv_path=args.output_csv_path,
        max_frames=args.max_frames,
    )

    localized_count = sum(1 for item in results if item.status == "localized")
    print("=" * 72)
    print(f"processed frames        : {len(results)}")
    print(f"localized frames        : {localized_count}")
    if localized_count:
        localization_scores = [item.score for item in results if item.score is not None]
        mean_score = sum(localization_scores) / float(len(localization_scores))
        print(f"mean localization score : {mean_score:.3f}")
    print(f"results csv             : {args.output_csv_path}")


if __name__ == "__main__":
    main()
