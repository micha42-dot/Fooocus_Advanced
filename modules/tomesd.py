from dataclasses import dataclass

from ldm_patched.contrib.external_tomesd import TomePatchModel


@dataclass(frozen=True)
class ToMeSDProfile:
    ratio: float


PROFILES = {
    'conservative': ToMeSDProfile(ratio=0.15),
    'balanced': ToMeSDProfile(ratio=0.30),
    'aggressive': ToMeSDProfile(ratio=0.50),
}

PROFILE_NAMES = ['off', *PROFILES.keys()]


def normalize_tomesd_profile(profile_name):
    return profile_name if profile_name in PROFILE_NAMES else 'off'


def tomesd_ratio(profile_name):
    profile = PROFILES.get(normalize_tomesd_profile(profile_name))
    return 0.0 if profile is None else profile.ratio


def apply_tomesd_profile(model, profile_name):
    ratio = tomesd_ratio(profile_name)
    if model is None or ratio <= 0.0:
        return model
    return TomePatchModel().patch(model, ratio, deterministic=True)[0]
