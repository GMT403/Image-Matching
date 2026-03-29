from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from lightglue import ALIKED, DISK, LightGlue, SIFT, SuperPoint
from lightglue.utils import rbd


@dataclass
class ExtractedImageFeatures:
    feature_inputs: dict[str, Any]
    keypoints_xy: np.ndarray
    image_height: int
    image_width: int


@dataclass
class MatchedKeypoints:
    query_keypoints_xy: np.ndarray
    reference_keypoints_xy: np.ndarray
    match_confidence: np.ndarray

    @property
    def match_count(self) -> int:
        return int(len(self.match_confidence))


class LightGlueFeatureMatcher:
    def __init__(
        self,
        feature_backend: str = "sift",
        device: str = "auto",
        max_keypoints: int = 2048,
        extractor_resize: int | None = 1024,
        depth_confidence: float = 0.95,
        width_confidence: float = 0.99,
        filter_threshold: float = 0.1,
        compile_matcher: bool = False,
    ) -> None:
        self.device = self._resolve_device(device)
        self.feature_backend = str(feature_backend).lower()
        self.extractor_resize = extractor_resize

        self.extractor = self._build_extractor(self.feature_backend, max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(
            features=self.feature_backend,
            depth_confidence=float(depth_confidence),
            width_confidence=float(width_confidence),
            filter_threshold=float(filter_threshold),
        ).eval().to(self.device)

        if compile_matcher and self.device.type == "cuda" and hasattr(self.matcher, "compile"):
            self.matcher.compile(mode="reduce-overhead")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if str(device).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _build_extractor(feature_backend: str, max_keypoints: int):
        if feature_backend == "superpoint":
            return SuperPoint(max_num_keypoints=max_keypoints)
        if feature_backend == "disk":
            return DISK(max_num_keypoints=max_keypoints)
        if feature_backend == "aliked":
            return ALIKED(max_num_keypoints=max_keypoints)
        if feature_backend == "sift":
            return SIFT(max_num_keypoints=max_keypoints, backend="opencv")
        raise ValueError("feature_backend must be one of: superpoint, disk, aliked, sift")

    def extract_features(self, image_bgr: np.ndarray) -> ExtractedImageFeatures:
        image_tensor = self._image_bgr_to_tensor(image_bgr)
        with torch.inference_mode():
            feature_inputs = self.extractor.extract(image_tensor, resize=self.extractor_resize)
        feature_preview = rbd(feature_inputs)
        keypoints_xy = feature_preview["keypoints"].detach().cpu().numpy().astype(np.float32)
        return ExtractedImageFeatures(
            feature_inputs=feature_inputs,
            keypoints_xy=keypoints_xy,
            image_height=int(image_bgr.shape[0]),
            image_width=int(image_bgr.shape[1]),
        )

    def match_images(self, query_bgr: np.ndarray, reference_bgr: np.ndarray) -> MatchedKeypoints:
        query_features = self.extract_features(query_bgr)
        reference_features = self.extract_features(reference_bgr)
        return self.match_feature_sets(query_features, reference_features)

    def match_feature_sets(
        self,
        query_features: ExtractedImageFeatures,
        reference_features: ExtractedImageFeatures,
    ) -> MatchedKeypoints:
        with torch.inference_mode():
            matches01 = self.matcher({"image0": query_features.feature_inputs, "image1": reference_features.feature_inputs})

        matches_preview = rbd(matches01)
        raw_matches = matches_preview["matches"]
        raw_scores = matches_preview["scores"]

        if isinstance(raw_matches, torch.Tensor):
            matches = raw_matches.detach().cpu().numpy().astype(np.int32)
        else:
            matches = np.asarray(raw_matches, dtype=np.int32)

        if matches.size == 0:
            empty_points = np.zeros((0, 2), dtype=np.float32)
            empty_scores = np.zeros((0,), dtype=np.float32)
            return MatchedKeypoints(empty_points, empty_points, empty_scores)

        if isinstance(raw_scores, torch.Tensor):
            scores = raw_scores.detach().cpu().numpy().astype(np.float32)
        else:
            scores = np.asarray(raw_scores, dtype=np.float32)

        matched_query = query_features.keypoints_xy[matches[:, 0]]
        matched_reference = reference_features.keypoints_xy[matches[:, 1]]
        return MatchedKeypoints(matched_query, matched_reference, scores)

    def _image_bgr_to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb.transpose(2, 0, 1))
        return torch.from_numpy(image_rgb).float().div(255.0).to(self.device)
