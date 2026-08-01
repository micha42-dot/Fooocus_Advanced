import os
import pickle
from functools import lru_cache


FAMILY_SD1 = 'sd1'
FAMILY_SD2 = 'sd2'
FAMILY_SDXL = 'sdxl'
FAMILY_UNKNOWN = 'unknown'

_CONTEXT_DIM_TO_FAMILY = {
    768: FAMILY_SD1,
    1024: FAMILY_SD2,
    1280: FAMILY_SDXL,
    2048: FAMILY_SDXL,
}


def family_from_context_dimensions(context_dimensions):
    families = {
        _CONTEXT_DIM_TO_FAMILY[dimension]
        for dimension in context_dimensions
        if dimension in _CONTEXT_DIM_TO_FAMILY
    }
    if len(families) == 1:
        return families.pop()
    return FAMILY_UNKNOWN


def _is_unet_cross_attention_key(key):
    return (
        key.endswith('attn2.to_k.weight')
        and any(section in key for section in (
            'input_blocks.',
            'middle_block.',
            'output_blocks.',
            'down_blocks.',
            'mid_block.',
            'up_blocks.',
        ))
    )


def _read_safetensors_context_dimensions(checkpoint_path):
    from safetensors import safe_open

    dimensions = set()
    with safe_open(checkpoint_path, framework='pt', device='cpu') as checkpoint:
        for key in checkpoint.keys():
            if not _is_unet_cross_attention_key(key):
                continue
            shape = checkpoint.get_slice(key).get_shape()
            if len(shape) == 2:
                dimensions.add(int(shape[1]))
    return dimensions


def _read_torch_context_dimensions(checkpoint_path):
    import warnings

    import torch

    load_options = {
        'map_location': 'meta',
        'weights_only': True,
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            checkpoint = torch.load(checkpoint_path, mmap=True, **load_options)
    except RuntimeError:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            checkpoint = torch.load(checkpoint_path, **load_options)

    state_dict = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state_dict, dict):
        return set()

    dimensions = set()
    for key, value in state_dict.items():
        if not _is_unet_cross_attention_key(key):
            continue
        shape = getattr(value, 'shape', ())
        if len(shape) == 2:
            dimensions.add(int(shape[1]))
    return dimensions


@lru_cache(maxsize=256)
def _detect_checkpoint_family_cached(checkpoint_path, file_size, modified_ns):
    del file_size, modified_ns
    try:
        if checkpoint_path.lower().endswith('.safetensors'):
            dimensions = _read_safetensors_context_dimensions(checkpoint_path)
        elif checkpoint_path.lower().endswith(('.ckpt', '.pt', '.pth', '.bin')):
            dimensions = _read_torch_context_dimensions(checkpoint_path)
        else:
            return FAMILY_UNKNOWN
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError):
        return FAMILY_UNKNOWN
    return family_from_context_dimensions(dimensions)


def detect_checkpoint_family(checkpoint_path):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return FAMILY_UNKNOWN

    stat = os.stat(checkpoint_path)
    return _detect_checkpoint_family_cached(
        os.path.abspath(checkpoint_path),
        stat.st_size,
        stat.st_mtime_ns,
    )


def detect_model_family(model_name, checkpoint_paths):
    from modules.util import get_file_from_folder_list

    checkpoint_path = get_file_from_folder_list(model_name, checkpoint_paths)
    return detect_checkpoint_family(checkpoint_path)


def choose_resolution(current_value, choices, preferred_value):
    if current_value in choices:
        return current_value
    if preferred_value in choices:
        return preferred_value
    return choices[0]
