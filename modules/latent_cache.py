import hashlib
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class CacheEntry:
    vae_ref: weakref.ReferenceType
    samples: torch.Tensor
    size: int


def tensor_fingerprint(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().to('cpu').contiguous().numpy()
    digest = hashlib.blake2b(digest_size=20)
    digest.update(str(array.shape).encode('ascii'))
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(memoryview(array).cast('B'))
    return digest.digest()


class LatentCache:
    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self.entries = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()

    def _key(self, vae, pixels: torch.Tensor, variant: str):
        vae_dtype = str(getattr(vae, 'vae_dtype', 'unknown'))
        return id(vae), vae_dtype, variant, tensor_fingerprint(pixels)

    def get(self, vae, pixels: torch.Tensor, variant: str, key=None):
        if self.max_bytes == 0:
            return None

        if key is None:
            key = self._key(vae, pixels, variant)
        with self.lock:
            entry = self.entries.get(key)
            if entry is None or entry.vae_ref() is not vae:
                if entry is not None:
                    self.current_bytes -= entry.size
                    del self.entries[key]
                self.misses += 1
                return None

            self.entries.move_to_end(key)
            self.hits += 1
            print(f'[Latent Cache] Hit ({self.hits} hits, {self.misses} misses).')
            return entry.samples.clone()

    def put(self, vae, pixels: torch.Tensor, variant: str, samples: torch.Tensor, key=None):
        if self.max_bytes == 0:
            return

        cached_samples = samples.detach().to('cpu').contiguous().clone()
        size = cached_samples.nelement() * cached_samples.element_size()
        if size > self.max_bytes:
            return

        if key is None:
            key = self._key(vae, pixels, variant)
        with self.lock:
            previous = self.entries.pop(key, None)
            if previous is not None:
                self.current_bytes -= previous.size

            self.entries[key] = CacheEntry(weakref.ref(vae), cached_samples, size)
            self.current_bytes += size

            while self.current_bytes > self.max_bytes and self.entries:
                _, entry = self.entries.popitem(last=False)
                self.current_bytes -= entry.size

    def get_or_encode(self, vae, pixels: torch.Tensor, variant: str, encode):
        try:
            key = self._key(vae, pixels, variant) if self.max_bytes > 0 else None
            cached = self.get(vae, pixels, variant, key=key)
        except Exception as error:
            print(f'[Latent Cache] Could not fingerprint input; bypassing cache: {error}')
            return encode()

        if cached is not None:
            return cached

        samples = encode()
        try:
            self.put(vae, pixels, variant, samples, key=key)
        except Exception as error:
            print(f'[Latent Cache] Could not store latent: {error}')
        return samples

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.current_bytes = 0


_latent_cache = None


def get_latent_cache():
    global _latent_cache
    if _latent_cache is None:
        import args_manager

        cache_disabled = getattr(args_manager.args, 'disable_latent_cache', False)
        configured_size_mb = getattr(args_manager.args, 'latent_cache_size', 256)
        cache_size_mb = 0 if cache_disabled else max(0, configured_size_mb)
        _latent_cache = LatentCache(cache_size_mb * 1024 * 1024)
        if cache_size_mb > 0:
            print(f'[Latent Cache] Enabled with a {cache_size_mb} MB limit.')
    return _latent_cache
