from collections import OrderedDict
from itertools import count
from threading import RLock

import torch


_namespace_counter = count(1)
_namespace_lock = RLock()


def _clip_namespace(clip):
    namespace = getattr(clip, '_fooocus_clip_cache_namespace', None)
    if namespace is not None:
        return namespace
    with _namespace_lock:
        namespace = getattr(clip, '_fooocus_clip_cache_namespace', None)
        if namespace is None:
            namespace = next(_namespace_counter)
            setattr(clip, '_fooocus_clip_cache_namespace', namespace)
    return namespace


def _clone_condition(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_condition(item) for item in value)
    if isinstance(value, list):
        return [_clone_condition(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_condition(item) for key, item in value.items()}
    return value


def _condition_size(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.nelement() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(_condition_size(item) for item in value)
    if isinstance(value, dict):
        return sum(_condition_size(item) for item in value.values())
    return 0


class ClipConditionCache:
    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self._entries = OrderedDict()
        self._size_bytes = 0
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    @staticmethod
    def key_for(clip, text: str):
        layer_idx = getattr(clip, 'layer_idx', None)
        return _clip_namespace(clip), layer_idx, text

    def get_or_encode(self, clip, text: str, encode):
        if self.max_bytes <= 0:
            return encode()

        key = self.key_for(clip, text)
        with self._lock:
            cached = self._entries.pop(key, None)
            if cached is not None:
                self._entries[key] = cached
                self._hits += 1
                return _clone_condition(cached)
            self._misses += 1

        result = encode()
        stored = _clone_condition(result)
        stored_size = _condition_size(stored)
        if stored_size > self.max_bytes:
            return result

        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._size_bytes -= _condition_size(previous)
            self._entries[key] = stored
            self._size_bytes += stored_size
            while self._size_bytes > self.max_bytes and self._entries:
                _, evicted = self._entries.popitem(last=False)
                self._size_bytes -= _condition_size(evicted)
        return result

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._size_bytes = 0

    def stats(self):
        with self._lock:
            return {
                'hits': self._hits,
                'misses': self._misses,
                'entries': len(self._entries),
                'size_bytes': self._size_bytes,
                'max_bytes': self.max_bytes,
            }


_clip_cache = None


def get_clip_cache() -> ClipConditionCache:
    global _clip_cache
    if _clip_cache is None:
        import args_manager

        disabled = getattr(args_manager.args, 'disable_clip_cache', False)
        size_mb = max(0, getattr(args_manager.args, 'clip_cache_size', 256))
        _clip_cache = ClipConditionCache(0 if disabled else size_mb * 1024 * 1024)
    return _clip_cache
