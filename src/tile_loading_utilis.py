import os
import math
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List


class TileCache:
    """
    Simple LRU cache for tiles to limit memory usage.
    Stores up to max_items tiles keyed by (x, y), evicting least-recently-used entries.
    """
    def __init__(self, max_items: int = 256):
        self.max_items = max_items
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._lru: List[Tuple[int, int]] = []

    def get(self, key: Tuple[int, int]) -> Optional[np.ndarray]:
        if key in self._cache:
            # update LRU ordering
            try:
                self._lru.remove(key)
            except ValueError:
                pass
            self._lru.append(key)
            return self._cache[key]
        return None

    def set(self, key: Tuple[int, int], value: np.ndarray) -> None:
        if key in self._cache:
            self._cache[key] = value
            try:
                self._lru.remove(key)
            except ValueError:
                pass
            self._lru.append(key)
            return
        if len(self._cache) >= self.max_items:
            # evict least recently used
            evict_key = self._lru.pop(0)
            self._cache.pop(evict_key, None)
        self._cache[key] = value
        self._lru.append(key)


def _imread_color(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def load_tile(tiles_dir: str, x: int, y: int, z: int, ext: str = ".png", cache: Optional[TileCache] = None) -> Optional[np.ndarray]:
    """
    Load a single tile image by x, y coordinates.
    Files are expected as <x>_<y><ext> within tiles_dir (e.g., 12345_67890.png).
    """
    key = (x, y)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    filename = f"tile_z{z}_x{x}_y{y}{ext}"
    path = os.path.join(tiles_dir, filename)
    img = _imread_color(path)
    if img is not None and cache is not None:
        cache.set(key, img)
    return img


def stitch_grid(tiles: Dict[Tuple[int, int], np.ndarray], xs: List[int], ys: List[int]) -> Optional[np.ndarray]:
    """
    Stitch a grid of tiles provided in a dict keyed by (x, y).
    - xs and ys must be sorted lists representing columns and rows to stitch.
    Returns a single concatenated image (H, W, 3) or None if any required tile missing.
    Uses NumPy for efficient stacking.
    """
    if not xs or not ys:
        return None

    row_images: List[np.ndarray] = []
    for y in ys:
        row_tiles: List[np.ndarray] = []
        for x in xs:
            tile = tiles.get((x, y))
            if tile is None:
                return None
            row_tiles.append(tile)
        # horizontally concatenate row using NumPy (assumes same height)
        row_img = np.concatenate(row_tiles, axis=1)
        row_images.append(row_img)

    # vertically concatenate rows (assumes same width per row)
    stitched = np.concatenate(row_images, axis=0)
    return stitched


def load_and_stitch_grid(
    tiles_dir: str,
    center_x: int,
    center_y: int,
    z: int = 18,
    grid_size: int = 3,
    ext: str = ".png",
    cache: Optional[TileCache] = None,
) -> Optional[np.ndarray]:
    """
    Load an N x N grid of tiles centered at (center_x, center_y) and stitch into one image.
    - grid_size: must be odd (3, 5, 7, ...). If even, it will be incremented by 1 to make it odd.
    - Returns stitched image or None if any tile is missing.
    """
    if grid_size < 1:
        return None
    if grid_size % 2 == 0:
        grid_size += 1

    half = grid_size // 2
    xs = list(range(center_x - half, center_x + half + 1))
    ys = list(range(center_y - half, center_y + half + 1))

    tiles: Dict[Tuple[int, int], np.ndarray] = {}
    for y in ys:
        for x in xs:
            img = load_tile(tiles_dir, x, y, z, ext=ext, cache=cache)
            if img is None:
                return None
            tiles[(x, y)] = img

    return stitch_grid(tiles, xs, ys)


def load_and_stitch_rect(
    tiles_dir: str,
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
    z: int = 19,
    ext: str = ".png",
    cache: Optional[TileCache] = None,
) -> Optional[np.ndarray]:
    """
    Load and stitch a rectangular range of tiles [min_x..max_x] x [min_y..max_y].
    Returns stitched image or None if any required tile is missing.
    """
    if min_x > max_x or min_y > max_y:
        return None
    xs = list(range(min_x, max_x + 1))
    ys = list(range(min_y, max_y + 1))
    tiles: Dict[Tuple[int, int], np.ndarray] = {}
    for y in ys:
        for x in xs:
            img = load_tile(tiles_dir, x, y, z, ext=ext, cache=cache)
            if img is None:
                return None
            tiles[(x, y)] = img
    return stitch_grid(tiles, xs, ys)


def update_center_from_match(
    last_center: Tuple[int, int],
    match_offset_pixels: Tuple[float, float],
    tile_px: int,
) -> Tuple[int, int]:
    """
    Estimate new tile center given an estimated pixel offset of best match within the stitched image.
    - last_center: previous tile center (x, y)
    - match_offset_pixels: estimated movement in pixels relative to stitched image center (dx, dy)
    - tile_px: tile size in pixels (typically 256 or 512)
    Rounds to nearest tile index step.
    """
    dx_px, dy_px = match_offset_pixels
    # convert pixel movement to tile steps; stitched image center is at tile center
    step_x = int(round(dx_px / tile_px))
    step_y = int(round(dy_px / tile_px))
    return (last_center[0] + step_x, last_center[1] + step_y)


def detect_tile_size(sample_tile: np.ndarray) -> int:
    """
    Return tile size in pixels (assumes square tiles).
    """
    return int(sample_tile.shape[0])


