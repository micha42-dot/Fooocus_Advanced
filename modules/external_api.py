import base64
import io
import json
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


API_BASE_MODEL = 'Other via API'
PROVIDER_FAL = 'fal.ai'
PROVIDER_REPLICATE = 'Replicate'
PROVIDERS = [PROVIDER_FAL, PROVIDER_REPLICATE]

FAL_KEY_ENV = 'FAL_KEY'
REPLICATE_KEY_ENV = 'REPLICATE_API_TOKEN'
ENV_FILE_PATH = Path(__file__).resolve().parent.parent / '.env'
_ENV_FILE_LOCK = threading.Lock()


class ExternalAPIError(RuntimeError):
    pass


class ExternalAPICancelled(ExternalAPIError):
    pass


@dataclass(frozen=True)
class APIModel:
    key: str
    label: str
    provider: str
    endpoint: str
    payload_kind: str
    max_images: int = 1


MODELS: Tuple[APIModel, ...] = (
    APIModel('fal:krea-v2-medium', 'Krea 2 Medium', PROVIDER_FAL,
             'krea/v2/medium/text-to-image', 'fal_krea'),
    APIModel('fal:krea-v2-large', 'Krea 2 Large', PROVIDER_FAL,
             'krea/v2/large/text-to-image', 'fal_krea'),
    APIModel('fal:flux-1-krea', 'FLUX.1 Krea', PROVIDER_FAL,
             'fal-ai/flux-1/krea', 'fal_flux', 4),
    APIModel('fal:flux-2-klein-4b', 'FLUX.2 Klein 4B', PROVIDER_FAL,
             'fal-ai/flux-2/klein/4b', 'fal_flux', 4),
    APIModel('fal:nano-banana-pro', 'Nano Banana Pro', PROVIDER_FAL,
             'fal-ai/nano-banana-pro', 'fal_nano_banana', 4),
    APIModel('fal:nano-banana-2', 'Nano Banana 2', PROVIDER_FAL,
             'fal-ai/nano-banana-2', 'fal_nano_banana', 4),
    APIModel('fal:minimax-image-01', 'MiniMax Image-01', PROVIDER_FAL,
             'fal-ai/minimax/image-01', 'fal_minimax', 9),
    APIModel('fal:dreamina-v3.1', 'Dreamina 3.1', PROVIDER_FAL,
             'fal-ai/bytedance/dreamina/v3.1/text-to-image', 'fal_dreamina', 4),
    APIModel('replicate:nano-banana-pro', 'Nano Banana Pro', PROVIDER_REPLICATE,
             'google/nano-banana-pro', 'replicate_nano_banana'),
    APIModel('replicate:nano-banana-2', 'Nano Banana 2', PROVIDER_REPLICATE,
             'google/nano-banana-2', 'replicate_nano_banana'),
    APIModel('replicate:minimax-image-01', 'MiniMax Image-01', PROVIDER_REPLICATE,
             'minimax/image-01', 'replicate_minimax', 9),
    APIModel('replicate:dreamina-v3.1', 'Dreamina 3.1', PROVIDER_REPLICATE,
             'bytedance/dreamina-3.1', 'replicate_dreamina'),
)
MODEL_BY_KEY: Dict[str, APIModel] = {model.key: model for model in MODELS}

NANO_BANANA_RATIOS = (
    '21:9', '16:9', '3:2', '4:3', '5:4', '1:1', '4:5', '3:4', '2:3', '9:16',
)
KREA_RATIOS = ('1:1', '4:3', '3:2', '16:9', '2.35:1', '4:5', '2:3', '9:16')
MINIMAX_RATIOS = ('1:1', '16:9', '4:3', '3:2', '2:3', '3:4', '9:16', '21:9')
DREAMINA_RATIOS = ('1:1', '16:9', '4:3', '3:2', '2:3', '3:4', '9:16', '21:9')


def get_model(model_key: str) -> APIModel:
    try:
        return MODEL_BY_KEY[model_key]
    except KeyError as exc:
        raise ExternalAPIError(f'Unknown external API model: {model_key}') from exc


def get_model_choices(provider: str) -> List[str]:
    return [model.label for model in MODELS if model.provider == provider]


def get_default_model_key(provider: str) -> str:
    for model in MODELS:
        if model.provider == provider:
            return model.key
    raise ExternalAPIError(f'No external API models are configured for {provider}.')


def get_default_model_label(provider: str) -> str:
    return get_model(get_default_model_key(provider)).label


def resolve_model(provider: str, model_value: str) -> APIModel:
    model = MODEL_BY_KEY.get(model_value)
    if model is not None and model.provider == provider:
        return model
    for candidate in MODELS:
        if candidate.provider == provider and candidate.label == model_value:
            return candidate
    raise ExternalAPIError(
        f'Unknown external API model for {provider}: {model_value}')


def select_refiner_for_base(base_model_name: str, refiner_model_name: str,
                            reset_api_refiner: bool = False) -> str:
    if base_model_name == API_BASE_MODEL and reset_api_refiner:
        return 'None'
    return refiner_model_name or 'None'


def get_generation_metadata(provider: str, model_key: str,
                            refiner_model_name: str = 'None',
                            refine_strength: Optional[float] = None) -> List[tuple]:
    model = get_model(model_key)
    if model.provider != provider:
        raise ExternalAPIError(
            f'External API model {model.label} does not belong to {provider}.')

    has_local_refiner = refiner_model_name not in ('', 'None', None)
    metadata = [
        ('Generation Source', 'generation_source', 'External API'),
        ('API Provider', 'api_provider', provider),
        ('API Model', 'api_model', model.label),
        ('Local Refinement', 'api_local_refinement',
         refiner_model_name if has_local_refiner else 'Disabled'),
    ]
    if has_local_refiner and refine_strength is not None:
        metadata.append(
            ('Local API Refine Strength', 'api_refine_strength', refine_strength))
    return metadata


def get_provider_env_name(provider: str) -> str:
    if provider == PROVIDER_FAL:
        return FAL_KEY_ENV
    if provider == PROVIDER_REPLICATE:
        return REPLICATE_KEY_ENV
    raise ExternalAPIError(f'Unknown external API provider: {provider}')


def _decode_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, str) else value
        except json.JSONDecodeError:
            return value
    if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def _parse_env_lines(lines: Sequence[str]) -> Dict[str, str]:
    values = {}
    for line in lines:
        candidate = line.strip()
        if not candidate or candidate.startswith('#'):
            continue
        if candidate.startswith('export '):
            candidate = candidate[7:].lstrip()
        key, separator, raw_value = candidate.partition('=')
        key = key.strip()
        if not separator or not key:
            continue
        values[key] = _decode_env_value(raw_value)
    return values


def _read_env_lines() -> List[str]:
    if not ENV_FILE_PATH.exists():
        return []
    if ENV_FILE_PATH.is_symlink():
        raise ExternalAPIError('Refusing to read API keys from a symbolic-link .env file.')
    try:
        return ENV_FILE_PATH.read_text(encoding='utf-8').splitlines(keepends=True)
    except OSError as exc:
        raise ExternalAPIError('Could not read the local .env file.') from exc


def _line_env_name(line: str) -> Optional[str]:
    candidate = line.strip()
    if not candidate or candidate.startswith('#'):
        return None
    if candidate.startswith('export '):
        candidate = candidate[7:].lstrip()
    key, separator, _ = candidate.partition('=')
    return key.strip() if separator else None


def _write_env_lines(lines: Sequence[str]):
    if ENV_FILE_PATH.is_symlink():
        raise ExternalAPIError('Refusing to replace a symbolic-link .env file.')

    ENV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix='.env.', suffix='.tmp', dir=str(ENV_FILE_PATH.parent))
        temp_path = Path(temp_name)
        try:
            os.chmod(temp_path, 0o600)
        except OSError:
            pass
        with os.fdopen(file_descriptor, 'w', encoding='utf-8', newline='\n') as env_file:
            env_file.writelines(lines)
            env_file.flush()
            os.fsync(env_file.fileno())
        os.replace(temp_path, ENV_FILE_PATH)
        temp_path = None
        try:
            os.chmod(ENV_FILE_PATH, 0o600)
        except OSError:
            pass
    except OSError as exc:
        raise ExternalAPIError('Could not write the local .env file.') from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def save_api_key(provider: str, api_key: str):
    env_name = get_provider_env_name(provider)
    key = (api_key or '').strip()
    if not key:
        raise ExternalAPIError(f'Enter a {provider} API key before saving.')
    if len(key) > 4096 or any(character.isspace() for character in key):
        raise ExternalAPIError('API keys must not contain whitespace and may be at most 4096 characters.')

    with _ENV_FILE_LOCK:
        lines = _read_env_lines()
        replacement = f'{env_name}={json.dumps(key)}\n'
        updated_lines = []
        was_replaced = False
        for line in lines:
            if _line_env_name(line) != env_name:
                updated_lines.append(line)
            elif not was_replaced:
                updated_lines.append(replacement)
                was_replaced = True
        if not was_replaced:
            if updated_lines and not updated_lines[-1].endswith(('\n', '\r')):
                updated_lines[-1] += '\n'
            updated_lines.append(replacement)
        _write_env_lines(updated_lines)


def delete_saved_api_key(provider: str) -> bool:
    env_name = get_provider_env_name(provider)
    with _ENV_FILE_LOCK:
        lines = _read_env_lines()
        updated_lines = [line for line in lines if _line_env_name(line) != env_name]
        if len(updated_lines) == len(lines):
            return False
        if updated_lines:
            _write_env_lines(updated_lines)
        else:
            try:
                ENV_FILE_PATH.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise ExternalAPIError('Could not delete the local .env file.') from exc
        return True


def get_saved_api_key(provider: str) -> str:
    env_name = get_provider_env_name(provider)
    with _ENV_FILE_LOCK:
        return _parse_env_lines(_read_env_lines()).get(env_name, '').strip()


def get_api_key_status(provider: str) -> str:
    env_name = get_provider_env_name(provider)
    try:
        saved = bool(get_saved_api_key(provider))
    except ExternalAPIError as exc:
        return f'Local key storage unavailable: {exc}'
    process_environment = bool(os.getenv(env_name, '').strip())
    if saved and process_environment:
        return f'Saved in local .env; {env_name} from the process environment takes precedence.'
    if saved:
        return 'Saved locally in .env.'
    if process_environment:
        return f'Using {env_name} from the process environment; nothing is stored by Fooocus.'
    return 'No key is saved locally.'


def resolve_api_key(provider: str, entered_key: str) -> str:
    key = (entered_key or '').strip()
    env_name = get_provider_env_name(provider)
    if not key:
        key = os.getenv(env_name, '').strip()
    if not key:
        key = get_saved_api_key(provider)
    if not key:
        raise ExternalAPIError(
            f'No {provider} API key configured. Enter one in the API tab, '
            f'save one locally, or set {env_name}.')
    return key


def nearest_aspect_ratio(width: int, height: int, supported: Sequence[str]) -> str:
    target = float(width) / max(float(height), 1.0)

    def distance(value: str) -> float:
        left, right = value.split(':', 1)
        ratio = float(left) / float(right)
        return abs(math.log(target / ratio))

    return min(supported, key=distance)


def _round_dimension(value: int, minimum: int = 512, maximum: int = 2048) -> int:
    value = max(minimum, min(maximum, int(value)))
    return max(minimum, min(maximum, int(round(value / 8.0)) * 8))


def _resolution_for_size(width: int, height: int, allow_half: bool = False) -> str:
    longest = max(width, height)
    if allow_half and longest <= 768:
        return '0.5K'
    if longest <= 1536:
        return '1K'
    if longest <= 3072:
        return '2K'
    return '4K'


def _split_image_counts(total: int, maximum: int) -> List[int]:
    counts = []
    remaining = max(1, int(total))
    while remaining:
        count = min(remaining, maximum)
        counts.append(count)
        remaining -= count
    return counts


def build_requests(model_key: str, prompt: str, width: int, height: int, num_images: int,
                   seed: int, output_format: str, enhance_prompt: bool = True,
                   creativity: str = 'medium') -> List[dict]:
    model = get_model(model_key)
    counts = _split_image_counts(num_images, model.max_images)
    requests = []
    image_offset = 0

    for count in counts:
        request_seed = int(seed) + image_offset
        payload = {'prompt': prompt}

        if model.payload_kind == 'fal_krea':
            payload.update({
                'aspect_ratio': nearest_aspect_ratio(width, height, KREA_RATIOS),
                'creativity': creativity,
                'seed': request_seed,
            })
        elif model.payload_kind == 'fal_flux':
            payload.update({
                'image_size': {
                    'width': _round_dimension(width),
                    'height': _round_dimension(height),
                },
                'num_images': count,
                'seed': request_seed,
                'output_format': 'jpeg' if output_format == 'jpeg' else output_format,
            })
        elif model.payload_kind == 'fal_nano_banana':
            payload.update({
                'num_images': count,
                'seed': request_seed,
                'aspect_ratio': nearest_aspect_ratio(width, height, NANO_BANANA_RATIOS),
                'output_format': 'jpeg' if output_format == 'jpeg' else output_format,
                'resolution': _resolution_for_size(
                    width, height, allow_half=model.key == 'fal:nano-banana-2'),
                'limit_generations': True,
            })
        elif model.payload_kind == 'fal_minimax':
            payload.update({
                'aspect_ratio': nearest_aspect_ratio(width, height, MINIMAX_RATIOS),
                'num_images': count,
                'prompt_optimizer': bool(enhance_prompt),
            })
        elif model.payload_kind == 'fal_dreamina':
            payload.update({
                'image_size': {
                    'width': _round_dimension(width),
                    'height': _round_dimension(height),
                },
                'enhance_prompt': bool(enhance_prompt),
                'num_images': count,
                'seed': request_seed,
            })
        elif model.payload_kind == 'replicate_nano_banana':
            payload.update({
                'resolution': _resolution_for_size(
                    width, height, allow_half=model.key == 'replicate:nano-banana-2'),
                'aspect_ratio': nearest_aspect_ratio(width, height, NANO_BANANA_RATIOS),
                'output_format': 'jpg' if output_format == 'jpeg'
                else 'png' if output_format == 'webp' else output_format,
            })
        elif model.payload_kind == 'replicate_minimax':
            payload.update({
                'aspect_ratio': nearest_aspect_ratio(width, height, MINIMAX_RATIOS),
                'number_of_images': count,
                'prompt_optimizer': bool(enhance_prompt),
            })
        elif model.payload_kind == 'replicate_dreamina':
            payload.update({
                'aspect_ratio': nearest_aspect_ratio(width, height, DREAMINA_RATIOS),
                'resolution': _resolution_for_size(width, height),
                'enhance_prompt': bool(enhance_prompt),
                'seed': request_seed,
            })
        else:
            raise ExternalAPIError(f'Unsupported payload type: {model.payload_kind}')

        requests.append(payload)
        image_offset += count

    return requests


def _response_json(response, api_key: str) -> dict:
    import httpx

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text[:500].replace(api_key, '[redacted]')
        raise ExternalAPIError(
            f'External API returned HTTP {response.status_code}: {detail}') from exc
    try:
        return response.json()
    except ValueError as exc:
        raise ExternalAPIError('External API returned an invalid JSON response.') from exc


def _cancel_fal(client, cancel_url: Optional[str], headers: dict):
    import httpx

    if cancel_url:
        try:
            client.put(cancel_url, headers=headers)
        except httpx.HTTPError:
            pass


def _run_fal_request(client, model: APIModel, payload: dict, api_key: str,
                     store_io: bool, cancel_requested: Callable[[], bool],
                     status_callback: Callable[[str], None]) -> dict:
    headers = {
        'Authorization': f'Key {api_key}',
        'Content-Type': 'application/json',
    }
    if not store_io:
        headers['X-Fal-Store-IO'] = '0'

    result = _response_json(
        client.post(f'https://queue.fal.run/{model.endpoint}', headers=headers, json=payload),
        api_key)
    status_url = result.get('status_url')
    response_url = result.get('response_url')
    cancel_url = result.get('cancel_url')
    if not status_url or not response_url:
        raise ExternalAPIError('fal.ai did not return queue status URLs.')

    deadline = time.monotonic() + 900.0
    last_status = None
    while time.monotonic() < deadline:
        if cancel_requested():
            _cancel_fal(client, cancel_url, headers)
            raise ExternalAPICancelled('fal.ai generation was cancelled.')

        status = _response_json(client.get(status_url, headers=headers), api_key)
        state = str(status.get('status', '')).upper()
        if state != last_status:
            status_callback('Queued at fal.ai ...' if state == 'IN_QUEUE'
                            else 'Generating with fal.ai ...')
            last_status = state
        if state == 'COMPLETED':
            return _response_json(client.get(response_url, headers=headers), api_key)
        if state in {'FAILED', 'CANCELLED', 'ERROR'}:
            raise ExternalAPIError(
                f'fal.ai generation ended with status {state}: {status.get("error", "")}')
        time.sleep(0.75)

    _cancel_fal(client, cancel_url, headers)
    raise ExternalAPIError('fal.ai generation timed out after 15 minutes.')


def _cancel_replicate(client, cancel_url: Optional[str], headers: dict):
    import httpx

    if cancel_url:
        try:
            client.post(cancel_url, headers=headers)
        except httpx.HTTPError:
            pass


def _run_replicate_request(client, model: APIModel, payload: dict, api_key: str,
                           cancel_requested: Callable[[], bool],
                           status_callback: Callable[[str], None]) -> dict:
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Cancel-After': '15m',
    }
    prediction = _response_json(
        client.post(f'https://api.replicate.com/v1/models/{model.endpoint}/predictions',
                    headers=headers, json={'input': payload}),
        api_key)

    deadline = time.monotonic() + 900.0
    last_status = None
    while time.monotonic() < deadline:
        state = str(prediction.get('status', '')).lower()
        if state != last_status:
            status_callback('Queued at Replicate ...' if state == 'starting'
                            else 'Generating with Replicate ...')
            last_status = state
        if state == 'succeeded':
            return prediction
        if state in {'failed', 'canceled'}:
            raise ExternalAPIError(
                f'Replicate generation ended with status {state}: {prediction.get("error", "")}')

        urls = prediction.get('urls') or {}
        if cancel_requested():
            _cancel_replicate(client, urls.get('cancel'), headers)
            raise ExternalAPICancelled('Replicate generation was cancelled.')
        get_url = urls.get('get')
        if not get_url:
            raise ExternalAPIError('Replicate did not return a prediction status URL.')
        time.sleep(0.75)
        prediction = _response_json(client.get(get_url, headers=headers), api_key)

    urls = prediction.get('urls') or {}
    _cancel_replicate(client, urls.get('cancel'), headers)
    raise ExternalAPIError('Replicate generation timed out after 15 minutes.')


def extract_image_urls(value) -> List[str]:
    urls = []

    def visit(item):
        if isinstance(item, str):
            if item.startswith(('https://', 'http://', 'data:image/')):
                urls.append(item)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            if isinstance(item.get('url'), str):
                visit(item['url'])
            else:
                for key in ('images', 'image', 'output', 'data'):
                    if key in item:
                        visit(item[key])

    visit(value)
    return urls


def _download_image(client, url: str) -> np.ndarray:
    import httpx

    if url.startswith('data:image/'):
        try:
            encoded = url.split(',', 1)[1]
            data = base64.b64decode(encoded, validate=True)
        except (IndexError, ValueError) as exc:
            raise ExternalAPIError('External API returned an invalid image data URI.') from exc
    else:
        try:
            response = client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExternalAPIError('Could not download an image returned by the external API.') from exc
        if len(response.content) > 64 * 1024 * 1024:
            raise ExternalAPIError('External API image exceeds the 64 MB download limit.')
        data = response.content

    try:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert('RGB'), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ExternalAPIError('External API returned an invalid image file.') from exc


def generate_images(model_key: str, api_key: str, prompt: str, width: int, height: int,
                    num_images: int, seed: int, output_format: str,
                    enhance_prompt: bool = True, creativity: str = 'medium',
                    store_io: bool = False,
                    cancel_requested: Optional[Callable[[], bool]] = None,
                    status_callback: Optional[Callable[[str], None]] = None) -> List[np.ndarray]:
    import httpx

    model = get_model(model_key)
    cancel_requested = cancel_requested or (lambda: False)
    status_callback = status_callback or (lambda _message: None)
    requests = build_requests(
        model_key, prompt, width, height, num_images, seed, output_format,
        enhance_prompt=enhance_prompt, creativity=creativity)

    images = []
    timeout = httpx.Timeout(30.0, connect=15.0)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            for index, payload in enumerate(requests):
                if cancel_requested():
                    raise ExternalAPICancelled('External API generation was cancelled.')
                status_callback(
                    f'Submitting API request {index + 1}/{len(requests)} to {model.provider} ...')
                if model.provider == PROVIDER_FAL:
                    response = _run_fal_request(
                        client, model, payload, api_key, store_io,
                        cancel_requested, status_callback)
                else:
                    response = _run_replicate_request(
                        client, model, payload, api_key, cancel_requested, status_callback)

                urls = extract_image_urls(response)
                if not urls:
                    raise ExternalAPIError(
                        f'{model.provider} completed the request but returned no image URL.')
                for url in urls:
                    if len(images) >= num_images:
                        break
                    status_callback(f'Downloading image {len(images) + 1}/{num_images} ...')
                    images.append(_download_image(client, url))
    except httpx.HTTPError as exc:
        raise ExternalAPIError(
            f'Network error while contacting {model.provider}.') from exc

    if len(images) < num_images:
        raise ExternalAPIError(
            f'{model.provider} returned {len(images)} of {num_images} requested images.')
    return images[:num_images]
