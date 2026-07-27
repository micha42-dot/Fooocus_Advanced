import math

import cv2
import numpy as np


TILED_DETAIL_LATENT_KEY = 'tiled_detail_image'
TILED_DETAIL_VERSION = '2.0'
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_OVERLAP = 256
MAX_TILE_BATCH_SIZE = 4
BLEND_CONFIDENCE_POWER = 2.0
LOW_FREQUENCY_MATCH_STRENGTH = 0.65
LOW_FREQUENCY_MATCH_SIGMA = 24.0


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
    if overlap >= tile_size:
        raise ValueError('overlap must be smaller than tile_size')
    if length <= tile_size:
        return [0]

    desired_step = tile_size - overlap
    max_start = length - tile_size
    minimum_intervals = max(1, int(math.ceil(max_start / tile_size)))
    ideal_intervals = max_start / desired_step
    lower = max(minimum_intervals, int(math.floor(ideal_intervals)))
    upper = max(minimum_intervals, int(math.ceil(ideal_intervals)))
    candidates = range(max(minimum_intervals, lower - 1), upper + 2)
    valid_candidates = [
        count for count in candidates
        if max_start / count <= tile_size
    ]
    interval_count = min(
        valid_candidates,
        key=lambda count: (abs(max_start / count - desired_step), -count)
    )

    positions = np.rint(np.linspace(0, max_start, interval_count + 1)).astype(np.int64).tolist()
    return sorted(set(int(position) for position in positions))


def get_tiled_detail_tile_count(height: int, width: int, tile_size: int = DEFAULT_TILE_SIZE,
                                overlap: int = DEFAULT_TILE_OVERLAP) -> int:
    return len(get_tile_positions(width, tile_size, overlap)) * len(get_tile_positions(height, tile_size, overlap))


def estimate_tiled_detail_tile_count(height: int, width: int, scale: float,
                                     tile_size: int = DEFAULT_TILE_SIZE,
                                     overlap: int = DEFAULT_TILE_OVERLAP) -> int:
    height = max(1, int(math.ceil(height * scale)))
    width = max(1, int(math.ceil(width * scale)))
    return get_tiled_detail_tile_count(height, width, tile_size, overlap)


def get_tiled_detail_batch_size(total_vram_mb: float, requested: int = 0,
                                total_tiles: int | None = None) -> int:
    if requested < 0:
        raise ValueError('requested batch size must be zero or greater')

    if requested > 0:
        batch_size = min(requested, MAX_TILE_BATCH_SIZE)
    elif total_vram_mb >= 24576:
        batch_size = 4
    elif total_vram_mb >= 16384:
        batch_size = 2
    else:
        batch_size = 1

    if total_tiles is not None:
        batch_size = min(batch_size, max(1, total_tiles))
    return batch_size


def get_padded_tile(image: np.ndarray, y: int, x: int,
                    tile_size: int = DEFAULT_TILE_SIZE) -> tuple[np.ndarray, int, int]:
    if image.ndim < 2:
        raise ValueError('image must have at least two dimensions')
    if tile_size <= 0:
        raise ValueError('tile_size must be greater than zero')

    image_height, image_width = image.shape[:2]
    if not (0 <= y < image_height and 0 <= x < image_width):
        raise ValueError('tile origin must be inside the image')

    tile_height = min(tile_size, image_height - y)
    tile_width = min(tile_size, image_width - x)
    tile = image[y:y + tile_height, x:x + tile_width]

    if tile_height == tile_size and tile_width == tile_size:
        return np.ascontiguousarray(tile.copy()), tile_height, tile_width

    pad_width = [(0, tile_size - tile_height), (0, tile_size - tile_width)]
    pad_width.extend([(0, 0)] * (image.ndim - 2))
    pad_mode = 'reflect' if tile_height > 1 and tile_width > 1 else 'edge'
    padded = np.pad(tile, pad_width, mode=pad_mode)
    return np.ascontiguousarray(padded.copy()), tile_height, tile_width


def get_tile_seed(base_seed: int, y: int, x: int, max_seed: int) -> int:
    if max_seed < 0:
        raise ValueError('max_seed must be zero or greater')

    mask = (1 << 64) - 1
    value = (int(base_seed) ^ (int(y) << 32) ^ int(x) ^ 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return int(value % (max_seed + 1))


def anchor_tile_low_frequency(generated: np.ndarray, reference: np.ndarray,
                              strength: float = LOW_FREQUENCY_MATCH_STRENGTH,
                              sigma: float = LOW_FREQUENCY_MATCH_SIGMA) -> np.ndarray:
    if generated.shape != reference.shape:
        raise ValueError('generated and reference tiles must have the same shape')
    if not 0.0 <= strength <= 1.0:
        raise ValueError('strength must be between zero and one')
    if sigma < 0.0:
        raise ValueError('sigma must be zero or greater')

    generated_float = generated.astype(np.float32, copy=False)
    if strength == 0.0 or sigma == 0.0:
        return generated_float.copy()

    reference_float = reference.astype(np.float32, copy=False)
    generated_low = cv2.GaussianBlur(
        generated_float, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    reference_low = cv2.GaussianBlur(
        reference_float, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    corrected = generated_float + float(strength) * (reference_low - generated_low)
    return np.clip(corrected, 0.0, 255.0).astype(np.float32, copy=False)


def _axis_blend_weights(length: int, position: int, image_length: int, overlap: int,
                        positions: list[int] | None) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    leading_overlap = min(overlap, length) if position > 0 else 0
    trailing_overlap = min(overlap, length) if position + length < image_length else 0

    if positions is not None and position in positions:
        index = positions.index(position)
        leading_overlap = (
            max(0, positions[index - 1] + length - position)
            if index > 0 else 0
        )
        trailing_overlap = (
            max(0, position + length - positions[index + 1])
            if index + 1 < len(positions) else 0
        )

    leading_overlap = min(leading_overlap, length)
    trailing_overlap = min(trailing_overlap, length)
    if leading_overlap > 1:
        phase = np.linspace(0.0, math.pi / 2.0, leading_overlap, dtype=np.float32)
        weights[:leading_overlap] = np.minimum(weights[:leading_overlap], np.sin(phase) ** 2)
    if trailing_overlap > 1:
        phase = np.linspace(0.0, math.pi / 2.0, trailing_overlap, dtype=np.float32)
        weights[-trailing_overlap:] = np.minimum(weights[-trailing_overlap:], np.cos(phase) ** 2)
    return weights


def make_blend_mask(tile_height: int, tile_width: int, y: int, x: int,
                    image_height: int, image_width: int,
                    overlap: int = DEFAULT_TILE_OVERLAP,
                    y_positions: list[int] | None = None,
                    x_positions: list[int] | None = None,
                    confidence_power: float = 1.0) -> np.ndarray:
    if confidence_power <= 0.0:
        raise ValueError('confidence_power must be greater than zero')

    y_weights = _axis_blend_weights(
        tile_height, y, image_height, overlap, y_positions)
    x_weights = _axis_blend_weights(
        tile_width, x, image_width, overlap, x_positions)

    mask = y_weights[:, None] * x_weights[None, :]
    if confidence_power != 1.0:
        mask = np.power(mask, confidence_power)
    return np.maximum(mask, 1e-6)
