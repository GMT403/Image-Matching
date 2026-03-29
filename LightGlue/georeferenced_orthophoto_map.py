import os
from typing import Iterator, Tuple

import cv2
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS


os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)


class GeoreferencedOrthophotoMap:
    def __init__(self, tif_path: str, resize_width: int | None = None) -> None:
        self.tif_path = tif_path

        proj_dir = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
        with rasterio.Env(PROJ_DATA=proj_dir, PROJ_LIB=proj_dir):
            with rasterio.open(tif_path) as dataset:
                self.transform = dataset.transform
                self.crs = dataset.crs
                self.raster_width = dataset.width
                self.raster_height = dataset.height

        self.to_wgs84 = Transformer.from_crs(self.crs, CRS.from_epsg(4326), always_xy=True)
        self.from_wgs84 = Transformer.from_crs(CRS.from_epsg(4326), self.crs, always_xy=True)

        image_bgr = cv2.imread(tif_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"OpenCV could not open GeoTIFF preview image: {tif_path}")

        if resize_width is None:
            self.scale = 1.0
        else:
            original_height, original_width = image_bgr.shape[:2]
            self.scale = float(resize_width) / float(original_width)
            resized_height = max(1, int(round(original_height * self.scale)))
            image_bgr = cv2.resize(image_bgr, (int(resize_width), resized_height), interpolation=cv2.INTER_AREA)

        self.image_bgr = image_bgr
        self.image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        self.height_px, self.width_px = self.image_gray.shape[:2]

    def pixel_to_projected(self, x_px: float, y_px: float) -> Tuple[float, float]:
        x_original = float(x_px) / self.scale
        y_original = float(y_px) / self.scale
        x_proj, y_proj = self.transform * (x_original, y_original)
        return float(x_proj), float(y_proj)

    def projected_to_pixel(self, x_proj: float, y_proj: float) -> Tuple[float, float]:
        x_original, y_original = ~self.transform * (float(x_proj), float(y_proj))
        return float(x_original) * self.scale, float(y_original) * self.scale

    def projected_to_latlon(self, x_proj: float, y_proj: float) -> Tuple[float, float]:
        lon, lat = self.to_wgs84.transform(float(x_proj), float(y_proj))
        return float(lat), float(lon)

    def latlon_to_projected(self, lat: float, lon: float) -> Tuple[float, float]:
        x_proj, y_proj = self.from_wgs84.transform(float(lon), float(lat))
        return float(x_proj), float(y_proj)

    def pixel_to_latlon(self, x_px: float, y_px: float) -> Tuple[float, float]:
        x_proj, y_proj = self.pixel_to_projected(x_px, y_px)
        return self.projected_to_latlon(x_proj, y_proj)

    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[float, float]:
        x_proj, y_proj = self.latlon_to_projected(lat, lon)
        return self.projected_to_pixel(x_proj, y_proj)

    def crop_centered_tile(self, center_x_px: float, center_y_px: float, tile_size_px: int):
        half = int(tile_size_px // 2)
        x0 = int(round(center_x_px)) - half
        y0 = int(round(center_y_px)) - half
        x1 = x0 + int(tile_size_px)
        y1 = y0 + int(tile_size_px)

        tile = cv2.copyMakeBorder(
            self.image_bgr[max(0, y0):min(self.height_px, y1), max(0, x0):min(self.width_px, x1)],
            top=max(0, -y0),
            bottom=max(0, y1 - self.height_px),
            left=max(0, -x0),
            right=max(0, x1 - self.width_px),
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        return tile, x0, y0

    def grid_centers(self, tile_size_px: int, stride_px: int) -> Iterator[Tuple[int, int]]:
        x_positions = self._grid_positions(self.width_px, tile_size_px, stride_px)
        y_positions = self._grid_positions(self.height_px, tile_size_px, stride_px)
        for center_y in y_positions:
            for center_x in x_positions:
                yield center_x, center_y

    @staticmethod
    def _grid_positions(length_px: int, tile_size_px: int, stride_px: int) -> list[int]:
        if length_px <= tile_size_px:
            return [length_px // 2]

        half = tile_size_px // 2
        last_center = max(half, length_px - half)
        positions = list(range(half, last_center + 1, max(1, stride_px)))
        if positions[-1] != last_center:
            positions.append(last_center)
        return positions
