from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeepCacheProfile:
    cache_interval: int
    branch_id: int
    start: float
    end: float
    min_steps: int = 12


PROFILES = {
    'conservative': DeepCacheProfile(cache_interval=2, branch_id=0, start=0.20, end=0.80),
    'balanced': DeepCacheProfile(cache_interval=3, branch_id=0, start=0.15, end=0.90),
    'aggressive': DeepCacheProfile(cache_interval=4, branch_id=1, start=0.10, end=0.95),
}


@dataclass
class CacheState:
    feature: torch.Tensor
    timestep: float
    transformer_index: int
    reuses: int = 0


@dataclass
class CachePlan:
    key: tuple
    cache_after: int
    output_start: int
    input_count: int
    reuse: bool = False
    feature: torch.Tensor | None = None
    transformer_index: int = 0


class DeepCache:
    def __init__(self):
        self.profile_name = 'off'
        self.profile = None
        self.steps = 0
        self.enabled = False
        self._compile_warning_shown = False
        self._short_run_warnings = set()
        self.reset_stats()

    def reset_stats(self):
        self.states = {}
        self.hits = 0
        self.misses = 0
        self.bypasses = 0

    def configure(self, profile_name='off', steps=None, compiled=False):
        self.reset_stats()
        self.profile_name = profile_name if profile_name in PROFILES else 'off'
        self.profile = PROFILES.get(self.profile_name)
        self.steps = max(0, int(steps or 0))
        self.enabled = self.profile is not None and self.steps >= self.profile.min_steps and not compiled

        if (self.profile is not None and self.steps < self.profile.min_steps
                and self.profile_name not in self._short_run_warnings):
            print(f'[Performance] DeepCache {self.profile_name} needs at least {self.profile.min_steps} steps; disabled.')
            self._short_run_warnings.add(self.profile_name)
        if compiled and self.profile is not None and not self._compile_warning_shown:
            print('[Performance] DeepCache is disabled while torch.compile is active.')
            self._compile_warning_shown = True

    def stats(self):
        return {
            'hits': self.hits,
            'misses': self.misses,
            'bypasses': self.bypasses,
            'profile': self.profile_name,
            'active': self.enabled,
        }

    @staticmethod
    def _progress(timesteps):
        if not isinstance(timesteps, torch.Tensor) or timesteps.numel() == 0:
            return None, None
        timestep = float(timesteps.detach().flatten()[0].float().cpu())
        progress = 1.0 - max(0.0, min(999.0, timestep)) / 999.0
        return timestep, progress

    @staticmethod
    def _conditioning_key(x, context, y, transformer_options):
        cond_or_uncond = tuple(transformer_options.get('cond_or_uncond', ()))
        conditioning_slots = tuple(transformer_options.get('fooocus_conditioning_slots', ()))
        return (
            cond_or_uncond,
            conditioning_slots,
            tuple(x.shape),
            str(x.dtype),
            x.device.type,
            tuple(context.shape) if isinstance(context, torch.Tensor) else None,
            tuple(y.shape) if isinstance(y, torch.Tensor) else None,
        )

    @staticmethod
    def _is_full_image(x, transformer_options):
        full_shape = transformer_options.get('fooocus_full_shape')
        if not isinstance(full_shape, (list, tuple)) or len(full_shape) != x.ndim:
            return True
        return tuple(x.shape[2:]) == tuple(full_shape[2:])

    def begin(self, x, timesteps, context, y, control, transformer_options, input_blocks, output_blocks):
        if not self.enabled:
            return None

        if (not isinstance(x, torch.Tensor) or control is not None
                or transformer_options.get('patches')
                or not self._is_full_image(x, transformer_options)
                or input_blocks != output_blocks):
            self.bypasses += 1
            return None

        timestep, progress = self._progress(timesteps)
        if progress is None or not self.profile.start <= progress <= self.profile.end:
            self.bypasses += 1
            return None

        cache_after = min((self.profile.branch_id + 1) * 3 - 1, output_blocks - 2)
        output_start = cache_after + 1
        required_inputs = input_blocks - output_start
        if cache_after < 0 or required_inputs <= 0:
            self.bypasses += 1
            return None

        key = self._conditioning_key(x, context, y, transformer_options)
        state = self.states.get(key)
        if (state is not None and timestep != state.timestep
                and state.reuses < self.profile.cache_interval - 1):
            state.reuses += 1
            state.timestep = timestep
            self.hits += 1
            return CachePlan(
                key=key,
                cache_after=cache_after,
                output_start=output_start,
                input_count=required_inputs,
                reuse=True,
                feature=state.feature.clone(),
                transformer_index=state.transformer_index,
            )

        self.misses += 1
        return CachePlan(
            key=key,
            cache_after=cache_after,
            output_start=output_start,
            input_count=input_blocks,
        )

    def store(self, plan, timestep, feature, transformer_index):
        if plan is None or plan.reuse or not isinstance(feature, torch.Tensor):
            return
        timestep_value, _ = self._progress(timestep)
        self.states[plan.key] = CacheState(
            feature=feature.detach().clone(),
            timestep=timestep_value,
            transformer_index=int(transformer_index),
        )


def install_unet_cache(diffusion_model, profile_name=None) -> bool:
    if diffusion_model is None:
        return False
    cache = getattr(diffusion_model, '_fooocus_deep_cache', None)
    if cache is None:
        cache = DeepCache()
        diffusion_model._fooocus_deep_cache = cache
    if profile_name is not None:
        cache.configure(profile_name)
    return True


def get_unet_cache(model_patcher):
    model = getattr(model_patcher, 'model', None)
    diffusion_model = getattr(model, 'diffusion_model', None)
    return getattr(diffusion_model, '_fooocus_deep_cache', None)


def reset_unet_cache(model_patcher, profile_name='off', steps=None):
    cache = get_unet_cache(model_patcher)
    if cache is not None:
        diffusion_model = getattr(getattr(model_patcher, 'model', None), 'diffusion_model', None)
        compiled = hasattr(diffusion_model, '_fooocus_compiled_forward')
        cache.configure(profile_name=profile_name, steps=steps, compiled=compiled)
    return cache


UNetCacheProfile = DeepCacheProfile
