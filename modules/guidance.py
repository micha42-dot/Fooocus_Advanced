import torch


GUIDANCE_STANDARD = 'Standard'
GUIDANCE_APG = 'APG'
GUIDANCE_CFGPP = 'CFG++'
GUIDANCE_MODES = [GUIDANCE_STANDARD, GUIDANCE_APG, GUIDANCE_CFGPP]

CFGPP_SAMPLERS = {
    'euler',
    'euler_ancestral',
    'dpmpp_2m',
    'dpmpp_2m_sde',
    'dpmpp_2m_sde_gpu',
}


def normalize_guidance_mode(mode):
    return mode if mode in GUIDANCE_MODES else GUIDANCE_STANDARD


def supports_cfgpp_sampler(sampler_name):
    return sampler_name in CFGPP_SAMPLERS


def cfgpp_scale(cfg_scale):
    """Map Fooocus' familiar CFG range to CFG++'s lambda range [0, 1]."""
    return max(0.0, min(1.0, float(cfg_scale) / 12.5))


def adaptive_projected_guidance(
        uncond,
        cond,
        cfg_scale,
        running_average=None,
        momentum=-0.5,
        norm_threshold=15.0,
        eta=0.0):
    update = cond - uncond
    if running_average is not None:
        update = update + float(momentum) * running_average
    new_running_average = update.detach()

    dims = tuple(range(1, update.ndim))
    if len(dims) == 0:
        dims = (0,)

    if norm_threshold > 0:
        update_norm = torch.linalg.vector_norm(update.float(), dim=dims, keepdim=True)
        scale = (float(norm_threshold) / update_norm.clamp_min(1e-6)).clamp(max=1.0)
        update = update * scale.to(update)

    update_float = update.float()
    cond_float = cond.float()
    cond_norm = torch.linalg.vector_norm(cond_float, dim=dims, keepdim=True).clamp_min(1e-6)
    cond_direction = cond_float / cond_norm
    parallel = (update_float * cond_direction).sum(dim=dims, keepdim=True) * cond_direction
    orthogonal = update_float - parallel
    projected = orthogonal + float(eta) * parallel
    result = uncond + float(cfg_scale) * projected.to(uncond)
    return result, new_running_average
