"""Small, model-independent box layouts and spatial conditioning for composition."""

import json
import math

MAX_BOXES = 6
MAX_PROMPT_LENGTH = 2000
COMPOSITION_METHOD = 'attention-v1'
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
    """Full-frame masks for weighting text attention, with inward feathering."""
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


def compose_conditioning(positive, composition, width, height, encode, global_texts, pool_top_k,
                         tensor_from_numpy=None):
    boxes = active_boxes(composition)
    if not boxes:
        return positive
    if tensor_from_numpy is None:
        import torch
        tensor_from_numpy = torch.from_numpy
    from modules.composition_attention import CompositionAttention
    background, masks = build_box_masks(boxes, width, height)
    grouped = {}
    shared_text = [s for s in global_texts[:pool_top_k] if s.strip()]
    for box, mask in zip(boxes, masks):
        # A single CLIP workload gives the object description priority. Do not add
        # Fooocus V2's global expansion to every box (it may invent other subjects).
        text = ', '.join([box['prompt']] + shared_text)
        if text in grouped:
            previous_mask, previous_strength = grouped[text]
            grouped[text] = (previous_mask * previous_strength + mask * box['strength'], 1.0)
        else:
            grouped[text] = (mask, box['strength'])
    # Identical text produces identical attention. Add its weighted masks first
    # so repeated subjects need only one encoding and one attention pass.
    # Background still comes from the original boxes, preserving overlap weights.
    regions = []
    for text, (mask, strength) in grouped.items():
        conditions = encode(texts=[text], pool_top_k=1)
        if len(conditions) != 1:
            raise CompositionError('A box prompt must encode to one conditioning entry.')
        regions.append((conditions[0][0], tensor_from_numpy(mask), strength))
    attention = CompositionAttention(tensor_from_numpy(background), regions)
    return [[embedding, dict(metadata, composition_attention=attention)] for embedding, metadata in positive]


def spatial_metadata(metadata):
    """Retain regions across refiner swaps without carrying base-only model/control data."""
    return {key: metadata[key] for key in SPATIAL_KEYS if key in metadata}


def crop_control_hint(hint, area):
    """Crop an NCHW ControlNet reference to a regional latent sampling window."""
    if area is None:
        return hint
    if area[0] == 'percentage':
        height, width = hint.shape[-2] // 8, hint.shape[-1] // 8
        h, w, y, x = (max(1, round(area[1] * height)), max(1, round(area[2] * width)),
                      round(area[3] * height), round(area[4] * width))
    else:
        h, w, y, x = area
    return hint[:, :, y * 8:(y + h) * 8, x * 8:(x + w) * 8]
