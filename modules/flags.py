from enum import IntEnum, Enum

disabled = 'Disabled'
enabled = 'Enabled'
subtle_variation = 'Vary (Subtle)'
strong_variation = 'Vary (Strong)'
upscale_15 = 'Upscale (1.5x)'
upscale_2 = 'Upscale (2x)'
upscale_tiled_detail_legacy = 'Upscale (Tiled SDXL Detail 2x)'
upscale_tiled_detail = 'Upscale (Tiled SDXL Detail 2.0, 2x)'
upscale_fast = 'Upscale (Fast 2x)'

uov_list = [disabled, subtle_variation, strong_variation, upscale_15, upscale_2, upscale_tiled_detail, upscale_fast]

enhancement_uov_before = "Before First Enhancement"
enhancement_uov_after = "After Last Enhancement"
enhancement_uov_processing_order = [enhancement_uov_before, enhancement_uov_after]

enhancement_uov_prompt_type_original = 'Original Prompts'
enhancement_uov_prompt_type_last_filled = 'Last Filled Enhancement Prompts'
enhancement_uov_prompt_types = [enhancement_uov_prompt_type_original, enhancement_uov_prompt_type_last_filled]

CIVITAI_NO_KARRAS = ["euler", "euler_ancestral", "heun", "dpm_fast", "dpm_adaptive", "ddim", "uni_pc"]

# fooocus: a1111 (Civitai)
KSAMPLER = {
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "heun": "Heun",
    "heunpp2": "",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "lms": "LMS",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "",
    "dpmpp_3m_sde_gpu": "",
    "ddpm": "",
    "lcm": "LCM",
    "tcd": "TCD",
    "restart": "Restart"
}

SAMPLER_EXTRA = {
    "ddim": "DDIM",
    "uni_pc": "UniPC",
    "uni_pc_bh2": ""
}

SAMPLERS = KSAMPLER | SAMPLER_EXTRA

KSAMPLER_NAMES = list(KSAMPLER.keys())

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "lcm", "turbo", "align_your_steps", "tcd", "edm_playground_v2.5"]
SAMPLER_NAMES = KSAMPLER_NAMES + list(SAMPLER_EXTRA.keys())

factory_default_sampler = "dpmpp_2m_sde_gpu"
factory_default_scheduler = "karras"

SAMPLER_DESCRIPTIONS = {
    "euler": "Fast deterministic first-order sampler; a reliable general-purpose baseline.",
    "euler_ancestral": "Ancestral Euler variant that adds noise between steps for more variation and texture.",
    "heun": "Second-order deterministic solver with smoother results, but roughly twice the model evaluations per step.",
    "heunpp2": "Refined Heun variant designed for smoother high-noise transitions and detailed results.",
    "dpm_2": "Deterministic second-order DPM solver; smooth and accurate, with extra model evaluations.",
    "dpm_2_ancestral": "Ancestral DPM2 variant that trades strict determinism for variation and texture.",
    "lms": "Linear multistep sampler that becomes more effective with moderate or higher step counts.",
    "dpm_fast": "Fixed-budget DPM solver focused on reaching a result quickly; less predictable than curated defaults.",
    "dpm_adaptive": "Adaptive DPM solver that chooses its own step sizes; runtime and effective step count may vary.",
    "dpmpp_2s_ancestral": "Two-stage ancestral DPM++ sampler with crisp detail and creative variation.",
    "dpmpp_sde": "Stochastic DPM++ SDE sampler with rich texture and strong detail, at higher compute cost.",
    "dpmpp_sde_gpu": "GPU-noise variant of DPM++ SDE; usually faster, but not pixel-identical to the CPU-noise version.",
    "dpmpp_2m": "Efficient deterministic multistep DPM++ sampler; an excellent general-purpose quality/speed choice.",
    "dpmpp_2m_sde": "Stochastic multistep DPM++ sampler balancing coherent structure with natural texture.",
    "dpmpp_2m_sde_gpu": "GPU-noise variant of DPM++ 2M SDE; fast and well suited to common SDXL workflows.",
    "dpmpp_3m_sde": "Higher-order stochastic multistep solver that can reward longer runs with refined detail.",
    "dpmpp_3m_sde_gpu": "GPU-noise variant of DPM++ 3M SDE; higher-order detail with faster noise generation.",
    "ddpm": "Original stochastic diffusion-style sampler; mainly useful for experiments and compatible workflows.",
    "lcm": "Few-step Latent Consistency sampler; use with an LCM model or LoRA and the matching LCM scheduler.",
    "tcd": "Few-step Trajectory Consistency sampler; intended for compatible TCD models or LoRAs.",
    "restart": "Uses controlled noise restarts during sampling to recover detail, with additional processing cost.",
    "ddim": "Classic implicit sampler with stable, repeatable behavior and a comparatively restrained look.",
    "uni_pc": "High-order predictor-corrector sampler offering good quality at low to moderate step counts.",
    "uni_pc_bh2": "Alternate UniPC BH2 solver variant with different stability and sharpness characteristics.",
}

SCHEDULER_DESCRIPTIONS = {
    "normal": "Uses the model's standard discrete noise schedule; broad compatibility and a dependable baseline.",
    "karras": "Karras sigma schedule concentrates more work near low noise, often improving final detail.",
    "exponential": "Spaces sigma values exponentially for even coverage across the logarithmic noise range.",
    "sgm_uniform": "Uniform SGM timestep schedule; commonly paired with distilled and consistency workflows.",
    "simple": "Selects a simple sequence from the model's native sigma table; predictable but less finely tuned.",
    "ddim_uniform": "Uniform DDIM timestep schedule for classic DDIM-style sampling behavior.",
    "lcm": "LCM-compatible model schedule for very low step counts; use with the LCM sampler and model or LoRA.",
    "turbo": "Schedule intended for SDXL Turbo and similar few-step distilled models.",
    "align_your_steps": "Optimized Align Your Steps schedule with dedicated SD1 and SDXL noise levels; used by Quality 10 and especially effective around 10 steps.",
    "tcd": "TCD-compatible schedule for few-step Trajectory Consistency sampling.",
    "edm_playground_v2.5": "Continuous EDM schedule configured for Playground v2.5; not a general SDXL default.",
}

sampler_list = SAMPLER_NAMES
scheduler_list = SCHEDULER_NAMES
sampler_descriptions = SAMPLER_DESCRIPTIONS
scheduler_descriptions = SCHEDULER_DESCRIPTIONS
guidance_modes = ['Standard', 'APG', 'CFG++', 'PAG']
deep_cache_profiles = ['off', 'conservative', 'balanced', 'aggressive']
tomesd_profiles = ['off', 'conservative', 'balanced', 'aggressive']

clip_skip_max = 12

default_vae = 'Default (model)'

refiner_swap_method = 'joint'

default_input_image_tab = 'uov_tab'
input_image_tab_ids = ['uov_tab', 'ip_tab', 'inpaint_tab', 'describe_tab', 'enhance_tab', 'metadata_tab']

cn_ip = "ImagePrompt"
cn_ip_face = "FaceSwap"
cn_canny = "PyraCanny"
cn_cpds = "CPDS"

ip_list = [cn_ip, cn_canny, cn_cpds, cn_ip_face]
default_ip = cn_ip

default_parameters = {
    cn_ip: (0.5, 0.6), cn_ip_face: (0.9, 0.75), cn_canny: (0.5, 1.0), cn_cpds: (0.5, 1.0)
}  # stop, weight

output_formats = ['png', 'jpeg', 'webp']

inpaint_mask_sam3 = 'sam3'
inpaint_mask_sam3_label = 'sam3 (Recommended / latest enhancement model)'
inpaint_mask_models = ['u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta', 'isnet-general-use', 'isnet-anime', 'sam', inpaint_mask_sam3]
inpaint_mask_model_choices = [
    inpaint_mask_sam3_label if model == inpaint_mask_sam3 else model
    for model in inpaint_mask_models
]
inpaint_mask_cloth_category = ['full', 'upper', 'lower']
inpaint_mask_sam_model = ['vit_b', 'vit_l', 'vit_h']


def normalize_inpaint_mask_model(model):
    return inpaint_mask_sam3 if model == inpaint_mask_sam3_label else model


def display_inpaint_mask_model(model):
    return inpaint_mask_sam3_label if model == inpaint_mask_sam3 else model

inpaint_engine_versions = ['None', 'v1', 'v2.5', 'v2.6']
inpaint_option_default = 'Inpaint or Outpaint (default)'
inpaint_option_detail = 'Improve Detail (face, hand, eyes, etc.)'
inpaint_option_modify = 'Modify Content (add objects, change background, etc.)'
inpaint_options = [inpaint_option_default, inpaint_option_detail, inpaint_option_modify]

describe_type_photo = 'Photograph'
describe_type_anime = 'Art/Anime'
describe_types = [describe_type_photo, describe_type_anime]

sdxl_aspect_ratios = [
    '704*1408', '704*1344', '768*1344', '768*1280', '832*1216', '832*1152',
    '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024', '1024*960',
    '1088*960', '1088*896', '1152*896', '1152*832', '1216*832', '1280*768',
    '1344*768', '1344*704', '1408*704', '1472*704', '1536*640', '1600*640',
    '1664*576', '1728*576'
]

sd15_aspect_ratios = [
    '384*640', '448*576', '512*768', '512*704', '512*640', '512*576',
    '512*512', '576*512', '640*512', '704*512', '768*512', '576*704',
    '640*640', '704*576', '768*768'
]


class MetadataScheme(Enum):
    FOOOCUS = 'fooocus'
    A1111 = 'a1111'


metadata_scheme = [
    (f'{MetadataScheme.FOOOCUS.value} (json)', MetadataScheme.FOOOCUS.value),
    (f'{MetadataScheme.A1111.value} (plain text)', MetadataScheme.A1111.value),
]


class OutputFormat(Enum):
    PNG = 'png'
    JPEG = 'jpeg'
    WEBP = 'webp'

    @classmethod
    def list(cls) -> list:
        return list(map(lambda c: c.value, cls))


class PerformanceLoRA(Enum):
    QUALITY = None
    QUALITY_10 = None
    SPEED = None
    EXTREME_SPEED = 'sdxl_lcm_lora.safetensors'
    LIGHTNING = 'sdxl_lightning_4step_lora.safetensors'
    HYPER_SD = 'sdxl_hyper_sd_4step_lora.safetensors'


class Steps(IntEnum):
    QUALITY = 60
    QUALITY_10 = 10
    SPEED = 30
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 4

    @classmethod
    def keys(cls) -> list:
        return list(map(lambda c: c, Steps.__members__))


class StepsUOV(IntEnum):
    QUALITY = 36
    QUALITY_10 = 10
    SPEED = 18
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 4


class Performance(Enum):
    QUALITY = 'Quality'
    QUALITY_10 = 'Quality 10 (AYS)'
    SPEED = 'Speed'
    EXTREME_SPEED = 'Extreme Speed'
    LIGHTNING = 'Lightning'
    HYPER_SD = 'Hyper-SD'

    @classmethod
    def list(cls) -> list:
        return list(map(lambda c: (c.name, c.value), cls))

    @classmethod
    def values(cls) -> list:
        return list(map(lambda c: c.value, cls))

    @classmethod
    def by_steps(cls, steps: int | str):
        return cls[Steps(int(steps)).name]

    @classmethod
    def has_restricted_features(cls, x) -> bool:
        if isinstance(x, Performance):
            x = x.value
        return x in [cls.EXTREME_SPEED.value, cls.LIGHTNING.value, cls.HYPER_SD.value]

    def steps(self) -> int | None:
        return Steps[self.name].value if self.name in Steps.__members__ else None

    def steps_uov(self) -> int | None:
        return StepsUOV[self.name].value if self.name in StepsUOV.__members__ else None

    def lora_filename(self) -> str | None:
        return PerformanceLoRA[self.name].value if self.name in PerformanceLoRA.__members__ else None

    def sampler_name(self) -> str | None:
        if self == Performance.QUALITY_10:
            return 'dpmpp_2m_sde_gpu'
        return None

    def scheduler_name(self) -> str | None:
        if self == Performance.QUALITY_10:
            return 'align_your_steps'
        return None


performance_selections = []

for name, value in Performance.list():
    restricted_text = ''
    if Performance.has_restricted_features(value):
        restricted_text = '*'
    performance_selections.append((f'{value} <span style="color: grey;"> \U00002223  {Steps[name].value} steps {restricted_text}</span>', Performance[name].value))
