VRAM_POLICY_CONSERVATIVE = 'conservative'
VRAM_POLICY_BALANCED = 'balanced'
VRAM_POLICY_RESIDENT = 'resident'


def choose_vram_policy(total_vram_mb: float, requested: str = 'auto', force_resident: bool = False) -> str:
    if force_resident:
        return VRAM_POLICY_RESIDENT
    if requested != 'auto':
        return requested
    if total_vram_mb <= 8192:
        return VRAM_POLICY_CONSERVATIVE
    if total_vram_mb < 20480:
        return VRAM_POLICY_BALANCED
    return VRAM_POLICY_RESIDENT
