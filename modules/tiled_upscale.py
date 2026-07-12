import math

import numpy as np


TILED_DETAIL_LATENT_KEY = 'tiled_detail_image'
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_OVERLAP = 192


def is_tiled_detail_upscale(uov_method: str) -> bool:
    method = str(uov_method).casefold()
    return 'upscale' in method and 'tiled' in method and 'detail' in method


def get_upscale_factor(uov_method: str) -> float:
    method = str(uov_method).casefold()
    if '1.5x' in method:
        return 1.5
    if '2x' in method:
        return 2.0
    return 1.0


def get_tile_positions(length: int, tile_size: int = DEFAULT_TILE_SIZE,
                       overlap: int = DEFAULT_TILE_OVERLAP) -> list[int]:
    if length <= 0:
        return [0]
    if tile_size <= 0:
        raise ValueError('tile_size must be greater than zero')
    if overlap < 0:
        raise ValueError('overlap must be zero or greater')
    if length <= tile_size:
        return [0]

    step = max(1, tile_size - overlap)
    max_start = length - tile_size
    positions = list(range(0, max_start + 1, step))
    if positions[-1] != max_start:
        positions.append(max_start)
    return positions


def get_tiled_detail_tile_count(height: int, width: int, tile_size: int = DEFAULT_TILE_SIZE,
                                overlap: int = DEFAULT_TILE_OVERLAP) -> int:
    return len(get_tile_positions(width, tile_size, overlap)) * len(get_tile_positions(height, tile_size, overlap))


def estimate_tiled_detail_tile_count(height: int, width: int, scale: float,
                                     tile_size: int = DEFAULT_TILE_SIZE,
                                     overlap: int = DEFAULT_TILE_OVERLAP) -> int:
    height = max(1, int(math.ceil(height * scale)))
    width = max(1, int(math.ceil(width * scale)))
    return get_tiled_detail_tile_count(height, width, tile_size, overlap)


def make_blend_mask(tile_height: int, tile_width: int, y: int, x: int,
                    image_height: int, image_width: int,
                    overlap: int = DEFAULT_TILE_OVERLAP) -> np.ndarray:
    y_weights = np.ones(tile_height, dtype=np.float32)
    x_weights = np.ones(tile_width, dtype=np.float32)

    def apply_feather(weights, has_leading_neighbor, has_trailing_neighbor):
        feather = min(overlap, len(weights))
        if feather <= 1:
            return weights
        if has_leading_neighbor:
            weights[:feather] = np.minimum(weights[:feather], np.linspace(0.0, 1.0, feather, dtype=np.float32))
        if has_trailing_neighbor:
            weights[-feather:] = np.minimum(weights[-feather:], np.linspace(1.0, 0.0, feather, dtype=np.float32))
        return weights

    y_weights = apply_feather(y_weights, y > 0, y + tile_height < image_height)
    x_weights = apply_feather(x_weights, x > 0, x + tile_width < image_width)

    mask = y_weights[:, None] * x_weights[None, :]
    return np.maximum(mask, 1e-6)
