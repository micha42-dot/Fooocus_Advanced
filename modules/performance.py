from collections import OrderedDict

import torch

import args_manager
from modules.unet_cache import install_unet_cache


class CompiledForward:
    def __init__(self, original_forward, compiled_forward, profile_name='dynamic'):
        self.original_forward = original_forward
        self.compiled_forward = compiled_forward
        self.profile_name = profile_name
        self.has_succeeded = False
        self.has_failed = False

    def __call__(self, *args, **kwargs):
        if self.has_failed:
            return self.original_forward(*args, **kwargs)

        try:
            result = self.compiled_forward(*args, **kwargs)
            self.has_succeeded = True
            return result
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as error:
            if self.has_succeeded:
                raise

            self.has_failed = True
            print(f'[Performance] torch.compile profile {self.profile_name} failed, using eager UNet: {error}')
            return self.original_forward(*args, **kwargs)


def compile_profile_key(args, kwargs):
    x = args[0] if args else kwargs.get('x')
    context = kwargs.get('context')
    if not isinstance(x, torch.Tensor):
        return ('unknown',)
    context_shape = tuple(context.shape) if isinstance(context, torch.Tensor) else None
    return tuple(x.shape), str(x.dtype), x.device.type, context_shape


class ResolutionCompiledForward:
    def __init__(self, original_forward, mode, max_profiles):
        self.original_forward = original_forward
        self.mode = mode
        self.max_profiles = max(1, int(max_profiles))
        self.profiles = OrderedDict()
        self.skipped_profiles = set()

    def _create_profile(self, key):
        compiled = torch.compile(
            self.original_forward,
            mode=self.mode,
            dynamic=False,
            fullgraph=False,
        )
        return CompiledForward(self.original_forward, compiled, profile_name=str(key[0]))

    def __call__(self, *args, **kwargs):
        key = compile_profile_key(args, kwargs)
        profile = self.profiles.get(key)
        if profile is not None:
            self.profiles.move_to_end(key)
            return profile(*args, **kwargs)

        if len(self.profiles) >= self.max_profiles:
            if key not in self.skipped_profiles:
                self.skipped_profiles.add(key)
                print(f'[Performance] Compile profile limit reached; using eager UNet for {key[0]}.')
            return self.original_forward(*args, **kwargs)

        try:
            profile = self._create_profile(key)
        except Exception as error:
            self.skipped_profiles.add(key)
            print(f'[Performance] Could not create compile profile {key[0]}: {error}')
            return self.original_forward(*args, **kwargs)

        self.profiles[key] = profile
        print(f'[Performance] Created UNet compile profile for {key[0]}.')
        return profile(*args, **kwargs)


def compile_unet(model_patcher) -> bool:
    compile_enabled = getattr(args_manager.args, 'torch_compile', False)
    if not compile_enabled or model_patcher is None:
        return False
    if not hasattr(torch, 'compile'):
        print('[Performance] torch.compile is not supported by this PyTorch version.')
        return False
    if not torch.cuda.is_available():
        print('[Performance] torch.compile is currently enabled only for CUDA devices.')
        return False

    diffusion_model = getattr(model_patcher.model, 'diffusion_model', None)
    if diffusion_model is None or hasattr(diffusion_model, '_fooocus_compiled_forward'):
        return False

    compile_mode = getattr(args_manager.args, 'torch_compile_mode', 'reduce-overhead')
    compile_profile = getattr(args_manager.args, 'torch_compile_profile', 'dynamic')
    mode = None if compile_mode == 'default' else compile_mode

    try:
        original_forward = diffusion_model.forward
        if compile_profile == 'resolution':
            guarded_forward = ResolutionCompiledForward(
                original_forward,
                mode=mode,
                max_profiles=getattr(args_manager.args, 'torch_compile_max_profiles', 3),
            )
        else:
            compiled_forward = torch.compile(
                original_forward,
                mode=mode,
                dynamic=True,
                fullgraph=False,
            )
            guarded_forward = CompiledForward(original_forward, compiled_forward)
        diffusion_model._fooocus_compiled_forward = guarded_forward
        diffusion_model.forward = guarded_forward
        print(f'[Performance] SDXL UNet compilation enabled ({compile_mode}, {compile_profile}).')
        return True
    except Exception as error:
        print(f'[Performance] Could not enable torch.compile: {error}')
        return False


def configure_unet(model_patcher):
    if model_patcher is None:
        return
    compile_unet(model_patcher)
    diffusion_model = getattr(model_patcher.model, 'diffusion_model', None)
    if hasattr(diffusion_model, '_fooocus_compiled_forward'):
        print('[Performance] DeepCache support skipped for compiled UNet.')
        return
    install_unet_cache(diffusion_model)
