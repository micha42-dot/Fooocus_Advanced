import hashlib
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime, timezone

import torch


_state = threading.local()


def _round(value):
    return round(float(value), 6)


def _prompt_hash(prompt: str, negative_prompt: str) -> str:
    text = f'{prompt}\0{negative_prompt}'.encode('utf-8', errors='replace')
    return hashlib.sha256(text).hexdigest()[:16]


def _gpu_details():
    if not torch.cuda.is_available():
        return {'device': 'cpu', 'cuda': None}
    try:
        device = torch.cuda.current_device()
        return {'device': torch.cuda.get_device_name(device), 'cuda': torch.version.cuda}
    except Exception:
        return {'device': 'cuda', 'cuda': torch.version.cuda}


class GenerationMetrics:
    def __init__(self, task, label: str):
        self.started = time.perf_counter()
        self.record = {
            'schema': 1,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'label': label,
            'status': 'running',
            'workload': {
                'prompt_hash': _prompt_hash(task.prompt, task.negative_prompt),
                'seed': task.seed,
                'model': task.base_model_name,
                'refiner': task.refiner_model_name,
                'performance': task.performance_selection.value,
                'resolution': task.aspect_ratios_selection,
                'images': task.image_number,
                'steps': task.steps,
                'sampler': task.sampler_name,
                'scheduler': task.scheduler_name,
                'cfg': task.cfg_scale,
                'guidance_mode': getattr(task, 'guidance_mode', 'Standard'),
                'deep_cache_profile': getattr(task, 'deep_cache_profile', 'off'),
                'clip_skip': task.clip_skip,
                'vae': task.vae_name,
                'loras': [[str(name), float(weight)] for name, weight in task.loras],
                'input_image': bool(task.input_image_checkbox),
                'uov_method': task.uov_method,
                'enhance': bool(task.enhance_checkbox),
            },
            'environment': {
                'python': platform.python_version(),
                'torch': torch.__version__,
                **_gpu_details(),
            },
            'configuration': {},
            'stages': {},
            'counters': {},
            'images': [],
        }

        import args_manager

        args = args_manager.args
        self.record['configuration'] = {
            'attention_backend': getattr(args, 'attention_backend', 'legacy'),
            'torch_compile': getattr(args, 'torch_compile', False),
            'torch_compile_mode': getattr(args, 'torch_compile_mode', 'default'),
            'torch_compile_profile': getattr(args, 'torch_compile_profile', 'dynamic'),
            'unet_cache': getattr(args, 'unet_cache', 'off'),
            'deep_cache_default': getattr(args, 'unet_cache', 'off'),
            'vram_policy': getattr(args, 'vram_policy', 'auto'),
        }
        model_management = sys.modules.get('ldm_patched.modules.model_management')
        if model_management is not None and hasattr(model_management, 'active_attention_backend'):
            self.record['configuration']['active_attention_backend'] = model_management.active_attention_backend()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        try:
            from modules.clip_cache import get_clip_cache
            self._clip_stats_start = get_clip_cache().stats()
        except Exception:
            self._clip_stats_start = None

    def stage(self, name: str, seconds: float):
        stages = self.record['stages']
        stages[name] = _round(stages.get(name, 0.0) + seconds)

    def counter(self, name: str, value: int = 1):
        counters = self.record['counters']
        counters[name] = int(counters.get(name, 0) + value)

    def image(self, path: str, array):
        digest = hashlib.sha256(memoryview(array).tobytes()).hexdigest()
        self.record['images'].append({
            'path': os.path.abspath(path),
            'sha256': digest,
            'shape': list(array.shape),
        })

    def finish(self, status: str):
        self.record['status'] = status
        self.record['total_seconds'] = _round(time.perf_counter() - self.started)
        if self._clip_stats_start is not None:
            try:
                from modules.clip_cache import get_clip_cache
                clip_stats = get_clip_cache().stats()
                self.record['counters']['clip_cache_hits'] = clip_stats['hits'] - self._clip_stats_start['hits']
                self.record['counters']['clip_cache_misses'] = clip_stats['misses'] - self._clip_stats_start['misses']
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                self.record['peak_vram_mb'] = _round(torch.cuda.max_memory_allocated() / (1024 * 1024))
            except Exception:
                pass
        return self.record


def _configured_path():
    import args_manager

    configured = getattr(args_manager.args, 'performance_log', None)
    if configured is None:
        return None
    if os.path.isabs(configured):
        return configured

    import modules.config
    return os.path.join(modules.config.path_outputs, configured)


def begin_generation(task):
    try:
        path = _configured_path()
        if path is None:
            _state.metrics = None
            return None

        import args_manager
        label = getattr(args_manager.args, 'performance_run_label', 'default')
        _state.metrics = GenerationMetrics(task, label)
        return _state.metrics
    except Exception as error:
        _state.metrics = None
        print(f'[Performance] Benchmark logging is unavailable: {error}')
        return None


def current_metrics():
    return getattr(_state, 'metrics', None)


def record_stage(name: str, seconds: float):
    metrics = current_metrics()
    if metrics is not None:
        metrics.stage(name, seconds)


def update_workload(**values):
    metrics = current_metrics()
    if metrics is not None:
        metrics.record['workload'].update(values)


def record_counter(name: str, value: int = 1):
    metrics = current_metrics()
    if metrics is not None:
        metrics.counter(name, value)


def record_image(path: str, array):
    metrics = current_metrics()
    if metrics is not None:
        metrics.image(path, array)


def finish_generation(status: str = 'complete'):
    metrics = current_metrics()
    _state.metrics = None
    if metrics is None:
        return None

    try:
        record = metrics.finish(status)
        path = _configured_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as output:
            output.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + '\n')
        print(f'[Performance] Benchmark record appended to {path}')
        return record
    except Exception as error:
        print(f'[Performance] Could not write benchmark record: {error}')
        return None
