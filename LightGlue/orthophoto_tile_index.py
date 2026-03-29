from dataclasses import dataclass
from typing import Iterable

import numpy as np

from georeferenced_orthophoto_map import GeoreferencedOrthophotoMap


@dataclass
class OrthophotoTile:
    tile_id: str
    tile_size_px: int
    stride_px: int
    center_x_px: int
    center_y_px: int
    map_x0_px: int
    map_y0_px: int
    image_bgr: np.ndarray


class OrthophotoTileIndex:
    def __init__(self, orthophoto_map: GeoreferencedOrthophotoMap) -> None:
        self.orthophoto_map = orthophoto_map
        self.tiles: list[OrthophotoTile] = []

    def build(self, tile_sizes_px: Iterable[int], stride_ratio: float = 0.5) -> None:
        self.tiles = []
        for tile_size_px in tile_sizes_px:
            tile_size_px = int(tile_size_px)
            stride_px = max(32, int(round(tile_size_px * float(stride_ratio))))
            tile_counter = 0
            for center_x_px, center_y_px in self.orthophoto_map.grid_centers(tile_size_px, stride_px):
                tile_image_bgr, map_x0_px, map_y0_px = self.orthophoto_map.crop_centered_tile(
                    center_x_px=center_x_px,
                    center_y_px=center_y_px,
                    tile_size_px=tile_size_px,
                )
                self.tiles.append(
                    OrthophotoTile(
                        tile_id=f"tile_{tile_size_px}px_{tile_counter:04d}",
                        tile_size_px=tile_size_px,
                        stride_px=stride_px,
                        center_x_px=int(center_x_px),
                        center_y_px=int(center_y_px),
                        map_x0_px=int(map_x0_px),
                        map_y0_px=int(map_y0_px),
                        image_bgr=tile_image_bgr,
                    )
                )
                tile_counter += 1

    def all_tiles(self) -> list[OrthophotoTile]:
        return list(self.tiles)

    def tiles_near(self, center_x_px: float, center_y_px: float, search_radius_px: float) -> list[OrthophotoTile]:
        center_x_px = float(center_x_px)
        center_y_px = float(center_y_px)
        radius_sq = float(search_radius_px) * float(search_radius_px)
        nearby_tiles = []
        for tile in self.tiles:
            dx = float(tile.center_x_px) - center_x_px
            dy = float(tile.center_y_px) - center_y_px
            if dx * dx + dy * dy <= radius_sq:
                nearby_tiles.append(tile)
        return nearby_tiles

    def nearest_tiles(self, center_x_px: float, center_y_px: float, max_tiles: int) -> list[OrthophotoTile]:
        center_x_px = float(center_x_px)
        center_y_px = float(center_y_px)
        ranked = []
        for tile in self.tiles:
            dx = float(tile.center_x_px) - center_x_px
            dy = float(tile.center_y_px) - center_y_px
            ranked.append((dx * dx + dy * dy, tile))
        ranked.sort(key=lambda item: item[0])
        return [tile for _, tile in ranked[: max(1, int(max_tiles))]]
