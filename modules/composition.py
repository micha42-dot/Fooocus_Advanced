"""Small, model-independent box layouts and spatial conditioning for composition."""

import json
import math

MAX_BOXES = 6
MAX_PROMPT_LENGTH = 2000
COMPOSITION_METHOD = 'regional-v2'
EMPTY_COMPOSITION = '{"version":1,"enabled":false,"boxes":[]}'
SPATIAL_KEYS = ('area', 'strength', 'mask', 'mask_strength', 'set_area_to_bounds',
                'start_percent', 'end_percent', 'timestep_start', 'timestep_end')


class CompositionError(ValueError):
    pass


def parse_composition(value):
    """Validate persisted/client data; coordinates are fractions of the whole image."""
    if value is None or value == '':
        value = EMPTY_COMPOSITION
    if isinstance(value, str):
        if len(value) > 100000:
            raise CompositionError('Box composition is too large.')
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise CompositionError('Could not read the box composition.') from exc
    if not isinstance(value, dict) or value.get('version', 1) != 1:
        raise CompositionError('Unsupported box composition format.')
    boxes = value.get('boxes', [])
    if not isinstance(boxes, list) or len(boxes) > MAX_BOXES:
        raise CompositionError(f'Use at most {MAX_BOXES} composition boxes.')
    if not isinstance(value.get('enabled', False), bool):
        raise CompositionError('Invalid box composition toggle.')
    normalized = []
    for index, box in enumerate(boxes):
        if not isinstance(box, dict):
            raise CompositionError(f'Invalid box {index + 1}.')
        prompt = box.get('prompt', '')
        if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT_LENGTH:
            raise CompositionError(f'Box {index + 1}: prompt must be at most {MAX_PROMPT_LENGTH} characters.')
        numbers = {}
        for key in ('x', 'y', 'w', 'h', 'strength'):
            number = box.get(key, 1.0 if key == 'strength' else None)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise CompositionError(f'Box {index + 1}: invalid {key}.')
            numbers[key] = float(number)
        x, y, w, h = (numbers[k] for k in ('x', 'y', 'w', 'h'))
        if not (0 <= x < 1 and 0 <= y < 1 and .02 <= w <= 1 and .02 <= h <= 1
                and x + w <= 1.000001 and y + h <= 1.000001
                and .1 <= numbers['strength'] <= 3):
            raise CompositionError(f'Box {index + 1}: keep the box inside the canvas and strength between 0.1 and 3.')
        normalized.append(dict(numbers, prompt=prompt.strip(), w=min(w, 1 - x), h=min(h, 1 - y)))
    return {'version': 1, 'enabled': value.get('enabled', False), 'boxes': normalized}


def serialize_composition(value):
    # The private history logger embeds metadata in HTML. Keep prompt text inert there.
    return (json.dumps(parse_composition(value), ensure_ascii=False, separators=(',', ':'))
            .replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e'))


def active_boxes(value):
    composition = parse_composition(value)
    return [b for b in composition['boxes'] if b['prompt']] if composition['enabled'] else []


def validate_composition_mode(value, api_mode=False, goals=()):
    composition = parse_composition(value)
    if not composition['enabled']:
        return
    if not active_boxes(composition):
        raise CompositionError('Compose with boxes: add a box and enter its prompt, or disable composition.')
    if api_mode or set(goals).intersection(('inpaint', 'upscale', 'vary', 'enhance')):
        raise CompositionError('Compose with boxes supports local text-to-image and Image Prompt. '
                               'Disable it for API generation, Vary, Upscale, Inpaint or Enhance input. '
                               'Enhancement after text-to-image is supported.')


def build_box_masks(boxes, width, height):
    """Latent masks blend the cropped regional passes into the global scene."""
    import numpy as np

    h, w = max(1, height // 8), max(1, width // 8)
    masks = []
    for box in boxes:
        x0 = min(w - 1, int(box['x'] * w))
        y0 = min(h - 1, int(box['y'] * h))
        x1 = min(w, max(x0 + 1, math.ceil((box['x'] + box['w']) * w)))
        y1 = min(h, max(y0 + 1, math.ceil((box['y'] + box['h']) * h)))

        def ramp(size, start, end, total):
            feather = max(1, min(8, size // 4))
            weights = np.ones(size, dtype=np.float32)
            edge = np.arange(1, feather + 1, dtype=np.float32) / (feather + 1)
            if start > 0:
                weights[:feather] *= edge
            if end < total:
                weights[-feather:] *= edge[::-1]
            return weights

        mask = np.zeros((h, w), dtype=np.float32)
        mask[y0:y1, x0:x1] = (ramp(y1 - y0, y0, y1, h)[:, None]
                              * ramp(x1 - x0, x0, x1, w)[None, :])
        masks.append(mask)
    # Keep a little global guidance inside objects and full guidance elsewhere.
    background = np.ones((h, w), dtype=np.float32)
    if masks:
        background -= .85 * np.maximum.reduce(masks)
    return background, masks


def box_sampling_area(box, width, height):
    """A bounded local latent window, aligned to 64 image pixels for the UNet.

    The soft mask still limits influence to the requested box. Percentages allow
    the sampler to resolve the area again if a separate refiner changes resolution.
    """
    h, w = max(1, height // 8), max(1, width // 8)
    x0 = max(0, int(box['x'] * w) // 8 * 8)
    y0 = max(0, int(box['y'] * h) // 8 * 8)
    x1 = min(w, max(x0 + 8, math.ceil((box['x'] + box['w']) * w / 8) * 8))
    y1 = min(h, max(y0 + 8, math.ceil((box['y'] + box['h']) * h / 8) * 8))
    # At a non-aligned image edge, keep at least one complete UNet window.
    x0 = max(0, min(x0, x1 - min(8, w)))
    y0 = max(0, min(y0, y1 - min(8, h)))
    return ('percentage', (y1 - y0) / h, (x1 - x0) / w, y0 / h, x0 / w)


def compose_conditioning(positive, composition, width, height, encode, global_texts, pool_top_k,
                         tensor_from_numpy=None):
    boxes = active_boxes(composition)
    if not boxes:
        return positive
    if tensor_from_numpy is None:
        import torch
        tensor_from_numpy = torch.from_numpy
    background, masks = build_box_masks(boxes, width, height)

    def masked(conditions, mask, strength, area=None):
        tensor = tensor_from_numpy(mask)
        result = []
        for embedding, metadata in conditions:
            regional = dict(metadata, mask=tensor, mask_strength=strength, set_area_to_bounds=False)
            if area is not None:
                regional['area'] = area
            result.append([embedding, regional])
        return result

    result = masked(positive, background, 1.0)
    for box, mask in zip(boxes, masks):
        # A single CLIP workload gives the object description priority. Do not add
        # Fooocus V2's global expansion to every box (it may invent other subjects).
        text = ', '.join([box['prompt']] + [s for s in global_texts[:pool_top_k] if s.strip()])
        conditions = encode(texts=[text], pool_top_k=1)
        result.extend(masked(conditions, mask, box['strength'], box_sampling_area(box, width, height)))
    return result


def spatial_metadata(metadata):
    """Retain regions across refiner swaps without carrying base-only model/control data."""
    return {key: metadata[key] for key in SPATIAL_KEYS if key in metadata}
