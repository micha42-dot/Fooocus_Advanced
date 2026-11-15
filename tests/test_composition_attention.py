"""Exercise the shipped attention and sampler code with a small NumPy tensor shim.

These are CPU algorithm/integration tests, not an SDXL image-quality evaluation.
The shim implements only tensor operations used by CompositionAttention.
"""
import ast
import collections
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from modules.composition import compose_conditioning, spatial_metadata
from modules.composition_attention import CompositionAttention
from tests.test_composition import layout, load_function, ScalarCondition

ROOT = Path(__file__).resolve().parents[1]


class Array(np.ndarray):
    @property
    def device(self):
        return 'cpu'

    def to(self, device=None, dtype=None):
        return array(np.asarray(self, dtype=dtype))

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis)

    def repeat(self, *sizes):
        return array(np.tile(np.asarray(self), sizes))

    def sum(self, dim=None, keepdim=False):
        return array(np.asarray(self).sum(axis=dim, keepdims=keepdim))

    def clamp_min(self, value):
        return array(np.maximum(self, value))

    def chunk(self, count):
        return np.split(self, count)

    def clone(self):
        return self.copy()

    def contiguous(self):
        return self


def array(value):
    return np.asarray(value, dtype=np.float32).view(Array)


def interpolate(value, size, mode, **kwargs):
    """Adaptive average pooling used by torch's area downsampling."""
    if mode != 'area':
        raise AssertionError('This test only exercises native-size and downsampled attention masks.')
    h, w = size
    ih, iw = value.shape[-2:]
    out = np.empty((*value.shape[:-2], h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            crop = np.asarray(value)[..., y * ih // h:( (y + 1) * ih + h - 1) // h,
                                     x * iw // w:((x + 1) * iw + w - 1) // w]
            out[..., y, x] = crop.mean(axis=(-2, -1))
    return array(out)


def load_method(class_name, method):
    tree = ast.parse((ROOT / 'ldm_patched/ldm/modules/attention.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    # This shim's Attention is deliberately not the shipped CrossAttention;
    # the real PyTorch suite covers that class and its shared-query path.
    namespace = dict(CrossAttention=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<real attention method>', 'exec'), namespace)
    return namespace[method]


class TestCompositionAttention(unittest.TestCase):
    def setUp(self):
        functional = types.ModuleType('torch.nn.functional')
        functional.interpolate = interpolate
        nn = types.ModuleType('torch.nn')
        nn.functional = functional
        torch = types.ModuleType('torch')
        torch.nn = nn
        torch.float32 = np.float32
        torch.stack = lambda xs: array(np.stack(xs))
        self.tensor_modules = patch.dict('sys.modules', {'torch': torch, 'torch.nn': nn,
                                                        'torch.nn.functional': functional})
        self.tensor_modules.start()
        self.addCleanup(self.tensor_modules.stop)

    def composed(self, width=1152, height=896, overlapping=False):
        value = layout(x=0, y=0, w=.5, h=1, prompt='fox')
        value['boxes'].append(dict(x=0 if overlapping else .5, y=0, w=.5, h=1,
                                   strength=2, prompt='dog'))
        return compose_conditioning([[array([[[1]]]), {'pooled_output': array([7])}]], value,
                                    width, height,
                                    lambda texts, **kw: [[array([[[10 if texts[0].startswith('fox') else 20]]]), {}]],
                                    ['shared scene'], 1, tensor_from_numpy=array)

    def block(self):
        calls = []

        class Attention:
            def __call__(self, query, context, value=None):
                calls.append((query.shape, context.shape))
                # Uniform token attention: enough to distinguish each prompt.
                mean = np.asarray(context if value is None else value).mean(axis=1, keepdims=True)
                return array(np.broadcast_to(mean, query.shape))

            to_q = to_k = to_v = to_out = staticmethod(lambda x: x)

        block = types.SimpleNamespace(ff_in=False, disable_self_attn=False, switch_temporal_ca_to_sa=False,
                                      n_heads=1, d_head=1, norm1=lambda x: x, norm2=lambda x: x,
                                      norm3=lambda x: x, attn1=Mock(side_effect=lambda x, **kw: np.zeros_like(x)),
                                      attn2=Attention(), ff=lambda x: np.zeros_like(x), is_res=True)
        block._cross_attention = types.MethodType(load_method('BasicTransformerBlock', '_cross_attention'), block)
        block.forward = types.MethodType(load_method('BasicTransformerBlock', '_forward'), block)
        return block, calls

    def test_one_full_frame_condition_and_global_pooled_embedding(self):
        conditions = self.composed()
        self.assertEqual(len(conditions), 1)
        _, meta = conditions[0]
        self.assertNotIn('area', meta)
        self.assertNotIn('mask', meta)
        self.assertEqual(meta['pooled_output'][0], 7)
        self.assertEqual(len(meta['composition_attention'].regions), 2)

    def test_actual_transformer_shares_self_attention_and_routes_text_in_landscape_and_portrait(self):
        for width, height in ((1152, 896), (896, 1152), (1536, 768), (1024, 1024)):
            h, w = height // 32, width // 32
            context, meta = self.composed(width, height)[0]
            block, calls = self.block()
            output_patch = Mock(side_effect=lambda n, options: n)
            result = block.forward(array(np.zeros((1, h * w, 1))), context,
                                   {'spatial_shape': (h, w), 'cond_or_uncond': [0],
                                    'patches': {'fooocus_composition': [meta['composition_attention']],
                                                'attn2_output_patch': [output_patch]}})
            image = result.reshape(h, w)
            self.assertGreater(image[h // 2, w * 3 // 4], image[h // 2, w // 4])
            self.assertGreater(image[h // 2, w // 4], 1)
            self.assertTrue(np.isfinite(image).all())
            self.assertEqual(block.attn1.call_count, 1)
            self.assertEqual(output_patch.call_count, 1)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(query == (1, h * w, 1) for query, _ in calls))

    def test_spatial_transformer_passes_exact_rectangular_grid(self):
        fn = load_method('SpatialTransformer', 'forward')

        def rearrange(x, pattern, **kwargs):
            if pattern == 'b c h w -> b (h w) c':
                return array(x.transpose(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1]))
            return array(x.reshape(x.shape[0], kwargs['h'], kwargs['w'], x.shape[-1]).transpose(0, 3, 1, 2))

        fn.__globals__['rearrange'] = rearrange
        observed = []

        def block(x, context, transformer_options):
            observed.append(transformer_options['spatial_shape'])
            return x

        layer = types.SimpleNamespace(norm=lambda x: x, use_linear=False, proj_in=lambda x: x,
                                      proj_out=lambda x: x, transformer_blocks=[block])
        original_options = {'block': ('input', 1), 'patches': {'fooocus_composition': [object()]}}
        fn(layer, array(np.zeros((1, 1, 28, 36))), transformer_options=original_options)
        self.assertEqual(observed, [(28, 36)])
        self.assertNotIn('spatial_shape', original_options)

    def test_overlaps_are_normalized_and_uncovered_background_stays_global(self):
        context, meta = self.composed(overlapping=True)[0]
        composition = meta['composition_attention']
        h, w = 28, 36
        reference = array(np.zeros((1, h * w, 1)))
        weights = composition.weights(h, w, reference)
        np.testing.assert_allclose(sum(weights), 1, atol=1e-6)
        self.assertIs(weights, composition.weights(h, w, reference))
        result = composition(lambda c: reference + c[0, 0, 0], context,
                             {'spatial_shape': (h, w), 'cond_or_uncond': [0]}).reshape(h, w)
        self.assertEqual(result[h // 2, w * 3 // 4], 1)
        self.assertGreater(result[h // 2, w // 4], 10)
        self.assertLess(result[h // 2, w // 4], 20)

    def test_image_prompt_replacement_is_used_for_every_text_context(self):
        context, meta = self.composed()[0]
        block, calls = self.block()
        replace = Mock(side_effect=lambda q, k, v, options: np.zeros_like(q) + k[0, 0, 0] + 3)
        options = {'spatial_shape': (28, 36), 'cond_or_uncond': [0], 'block': ('input', 4), 'block_index': 0,
                   'patches': {'fooocus_composition': [meta['composition_attention']]},
                   'patches_replace': {'attn2': {('input', 4, 0): replace}}}
        output = block.forward(array(np.zeros((1, 28 * 36, 1))), context, options)
        self.assertEqual(replace.call_count, 3)
        self.assertEqual(calls, [])
        self.assertTrue(np.isfinite(output).all())

    def test_unpatched_transformer_uses_original_attention_path(self):
        block, calls = self.block()
        result = block.forward(array(np.zeros((1, 28 * 36, 1))), array([[[5]]]),
                               {'spatial_shape': (28, 36), 'cond_or_uncond': [1]})
        self.assertTrue(np.all(result == 5))
        self.assertEqual(len(calls), 1)
        self.assertEqual(block.attn1.call_count, 1)

    def test_sampler_runs_one_full_frame_positive_and_unmodified_negative(self):
        context, meta = self.composed()[0]
        positive = [dict(meta, model_conds={'c_crossattn': ScalarCondition(1)})]
        negative = [dict(model_conds={'c_crossattn': ScalarCondition(-1)})]
        namespace = dict(collections=collections,
                         torch=types.SimpleNamespace(zeros_like=np.zeros_like, ones_like=np.ones_like,
                                                     cat=lambda xs: array(np.concatenate(xs))),
                         model_management=types.SimpleNamespace(get_free_memory=lambda device: 10**12))
        for name in ('get_area_and_mult', 'cond_equal_size', 'can_concat_cond', 'cond_cat', 'calc_cond_uncond_batch'):
            load_function('ldm_patched/modules/samplers.py', name, namespace)
        seen = []

        def denoise(x, timestep, c_crossattn, transformer_options):
            seen.append((x.shape, transformer_options))
            return np.zeros_like(x) + c_crossattn[0]

        model = types.SimpleNamespace(memory_required=lambda shape: 0, apply_model=denoise)
        pos, neg = namespace['calc_cond_uncond_batch'](
            model, positive, negative, array(np.zeros((1, 4, 112, 144))), array([1]), {})
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(shape == (1, 4, 112, 144) for shape, _ in seen))
        for _, options in seen:
            has_composition = bool(options.get('patches', {}).get('fooocus_composition'))
            self.assertEqual(has_composition, options['cond_or_uncond'] == [0])
        self.assertTrue(np.all(pos == 1))
        self.assertTrue(np.all(neg == -1))
        seen.clear()
        namespace['calc_cond_uncond_batch'](
            model, positive, None, array(np.zeros((1, 4, 112, 144))), array([1]), {})
        self.assertEqual(len(seen), 1)  # CFG 1 omits the negative pass.

        # Model-wide hooks (e.g. ToMe) must coexist with conditioning hooks.
        # Exercise repeated calls to detect mutation/leakage into the negative pass.
        hook = lambda *args: args
        existing_hooks = [hook]
        replacement = {'attn2': {('input', 1): hook}}
        for existing in ({}, {'attn1_patch': existing_hooks}):
            options = {'transformer_options': {'patches': existing, 'patches_replace': replacement}}
            for _ in range(2):
                seen.clear()
                namespace['calc_cond_uncond_batch'](
                    model, positive, negative, array(np.zeros((1, 4, 112, 144))), array([1]), options)
                for _, passed in seen:
                    hooks = passed['patches']
                    self.assertEqual('fooocus_composition' in hooks, passed['cond_or_uncond'] == [0])
                    self.assertIs(passed['patches_replace'], replacement)
                    if 'attn1_patch' in existing:
                        self.assertIs(hooks['attn1_patch'], existing_hooks)
                self.assertNotIn('fooocus_composition', existing)
                self.assertEqual(existing_hooks, [hook])

    def test_different_prompt_lengths_and_multi_image_batch_keep_full_query_grid(self):
        context, meta = self.composed()[0]
        original = meta['composition_attention']
        regions = [(array(np.full((1, length, 1), text[0, 0, 0])), mask, strength)
                   for (text, mask, strength), length in zip(original.regions, (3, 5))]
        attention = CompositionAttention(original.background, regions)
        block, calls = self.block()
        output = block.forward(array(np.zeros((2, 28 * 36, 1))), context.repeat(2, 1, 1),
                               {'spatial_shape': (28, 36), 'cond_or_uncond': [0],
                                'patches': {'fooocus_composition': [attention]}})
        self.assertEqual([shape[1] for _, shape in calls], [1, 3, 5])
        self.assertTrue(all(shape[0] == 2 for _, shape in calls))
        np.testing.assert_array_equal(output[0], output[1])

    def test_joint_and_separate_refiners_convert_all_region_embeddings(self):
        context, meta = self.composed()[0]
        namespace = dict(spatial_metadata=spatial_metadata,
                         clip_separate_inner=lambda c, p, **kw: (c + 100, p),
                         CONDRegular=lambda c: types.SimpleNamespace(cond=c))
        separate = load_function('modules/sample_hijack.py', 'clip_separate', namespace)
        joint = load_function('modules/sample_hijack.py', 'clip_separate_after_preparation', namespace)
        conversions = [separate([[context, meta]])[0][1],
                       joint([dict(meta, model_conds={'c_crossattn': types.SimpleNamespace(cond=context)})])[0]]
        for converted in conversions:
            attention = converted['composition_attention']
            self.assertIsNot(attention, meta['composition_attention'])
            self.assertEqual(attention.regions[0][0][0, 0, 0], 110)
            self.assertEqual(attention.regions[1][0][0, 0, 0], 120)
            self.assertIs(attention.background, meta['composition_attention'].background)
            self.assertEqual(converted['pooled_output'][0], 7)
        self.assertEqual(meta['composition_attention'].regions[0][0][0, 0, 0], 10)


if __name__ == '__main__':
    unittest.main()
