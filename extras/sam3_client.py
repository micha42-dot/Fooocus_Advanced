import base64
import io

import httpx
import numpy as np
from PIL import Image

import args_manager


class SAM3UnavailableError(RuntimeError):
    pass


def generate_sam3_mask(image: np.ndarray, prompt: str, confidence_threshold: float = 0.3,
                       max_detections: int = 0):
    if not prompt or not prompt.strip():
        raise ValueError('SAM 3 requires a detection prompt.')

    image_buffer = io.BytesIO()
    Image.fromarray(image.astype(np.uint8)).convert('RGB').save(image_buffer, format='PNG')
    payload = {
        'image': base64.b64encode(image_buffer.getvalue()).decode('ascii'),
        'prompt': prompt.strip(),
        'confidence_threshold': float(confidence_threshold),
        'max_detections': int(max_detections),
    }

    sam3_url = getattr(args_manager.args, 'sam3_url', 'http://127.0.0.1:7866')
    sam3_timeout = getattr(args_manager.args, 'sam3_timeout', 300.0)
    base_url = sam3_url.rstrip('/')
    try:
        health_response = httpx.get(f'{base_url}/health', timeout=5.0)
        health_response.raise_for_status()
    except Exception as error:
        raise SAM3UnavailableError(
            f'Could not reach the SAM 3 worker at {sam3_url}: {error}') from error

    from ldm_patched.modules import model_management

    model_management.unload_all_models()
    model_management.soft_empty_cache(force=True)

    try:
        response = httpx.post(
            f'{base_url}/segment', json=payload, timeout=sam3_timeout)
        response.raise_for_status()
        result = response.json()
    except Exception as error:
        raise SAM3UnavailableError(f'SAM 3 mask generation failed: {error}') from error

    if 'error' in result:
        raise SAM3UnavailableError(result['error'])

    try:
        mask_bytes = base64.b64decode(result['mask'])
        mask = np.array(Image.open(io.BytesIO(mask_bytes)).convert('L'), dtype=np.uint8)
    except Exception as error:
        raise SAM3UnavailableError(f'SAM 3 returned an invalid mask: {error}') from error

    mask_rgb = np.repeat(mask[:, :, None], 3, axis=2)
    detection_count = int(result.get('detection_count', 0))
    return mask_rgb, detection_count, detection_count, detection_count
