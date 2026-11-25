"""Object-mask preparation shared by the Enhance workflow and its tests."""

import numpy as np


def object_context_field(mask, context):
    """Translate the context slider to an object-sized crop for InpaintWorker."""
    active = mask > 0
    rows = np.flatnonzero(active.any(axis=1))
    if not rows.size:
        return 0.0
    columns = np.flatnonzero(active.any(axis=0))
    object_size = max(int(rows[-1] - rows[0] + 1), int(columns[-1] - columns[0] + 1))
    crop_size = object_size * (1.0 + 2.0 * min(1.0, max(0.0, float(context))))
    return min(1.0, crop_size / max(mask.shape))


def prepare_enhance_masks(mask, instance_masks, separate_objects, invert=False, transform=None):
    """Keep detector instances intact; inverted masks always describe one background."""
    union = np.asarray(mask)
    if union.ndim == 3:
        union = union[:, :, 0]
    if union.ndim != 2:
        raise ValueError('Enhance masks must be two-dimensional.')
    candidates = instance_masks if separate_objects and not invert and instance_masks else [union]
    prepared = []
    seen = set()
    for candidate in candidates:
        candidate = np.asarray(candidate)
        if candidate.shape != union.shape:
            raise ValueError('Object mask size does not match the combined mask.')
        candidate = candidate.astype(np.uint8, copy=True)
        if transform is not None:
            candidate = transform(candidate)
        if invert:
            candidate = 255 - candidate
        if not np.any(candidate > 0):
            continue
        # Duplicate detections should not repaint the same object twice.
        key = np.packbits(candidate > 0).tobytes()
        if key not in seen:
            seen.add(key)
            prepared.append(candidate)
    return prepared
