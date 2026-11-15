"""Regional text attention over one shared, full-frame UNet state.

No model weights or third-party attention implementation are needed. Only the
cross-attention result is blended; convolutions, self-attention and SDXL ADM
continue to describe the whole image.
"""


class CompositionAttention:
    def __init__(self, background, regions):
        # Each region is (CLIP embedding, full-frame latent mask, strength).
        self.background = background
        self.regions = tuple(regions)
        self._weights = {}
        self._contexts = {}

    def convert(self, convert_embedding):
        """Rebuild text for a refiner without retaining base-model device caches."""
        return CompositionAttention(self.background,
                                    [(convert_embedding(text), mask, strength)
                                     for text, mask, strength in self.regions])

    def weights(self, height, width, reference):
        import torch
        import torch.nn.functional as F

        key = (height, width, reference.device, reference.dtype)
        if key not in self._weights:
            masks = [self.background] + [mask * strength for _, mask, strength in self.regions]
            masks = torch.stack(masks).to(device=reference.device, dtype=torch.float32).unsqueeze(1)
            if masks.shape[-2:] != (height, width):
                if height <= masks.shape[-2] and width <= masks.shape[-1]:
                    masks = F.interpolate(masks, size=(height, width), mode='area')
                else:
                    masks = F.interpolate(masks, size=(height, width), mode='bilinear', align_corners=False)
            masks = masks / masks.sum(dim=0, keepdim=True).clamp_min(1e-6)
            self._weights[key] = tuple(mask.reshape(1, height * width, 1).to(dtype=reference.dtype)
                                       for mask in masks)
        return self._weights[key]

    def __call__(self, attend, context, options):
        # Only positive conditioning carries this patch. The sampler keeps it in
        # a separate batch from the unpatched negative conditioning.
        base = attend(context)
        if any(kind != 0 for kind in options.get('cond_or_uncond', (0,))):
            raise ValueError('Composition attention must only receive positive conditioning.')
        height, width = options['spatial_shape']
        if base.shape[1] != height * width:
            raise ValueError('Composition attention requires the full spatial token grid.')
        weights = self.weights(height, width, base)
        result = base * weights[0]
        for index, (text, _, _) in enumerate(self.regions):
            key = (index, context.device, context.dtype, context.shape[0])
            if key not in self._contexts:
                regional = text.to(device=context.device, dtype=context.dtype)
                batch = context.shape[0]
                if regional.shape[0] != batch:
                    regional = regional.repeat((batch + regional.shape[0] - 1) // regional.shape[0], 1, 1)[:batch]
                self._contexts[key] = regional
            # The accumulator is private to this call; reuse it rather than
            # allocating another full spatial feature tensor for every box.
            result += attend(self._contexts[key]) * weights[index + 1]
        return result
