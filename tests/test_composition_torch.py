"""Real CPU PyTorch regression tests; no checkpoint or Fooocus server required.

Load the shipped transformer classes without importing the GPU model loader.
Their projections, normalization, feed-forward layers and SDPA all use PyTorch.
"""
import ast
import collections
from functools import partial
import math
from pathlib import Path
import types
from typing import List, NamedTuple, Optional, Protocol
import unittest
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:
    torch = None

try:
    from einops import rearrange
except ImportError:
    rearrange = None

from modules.composition import build_box_masks, compose_conditioning
from modules.composition_attention import CompositionAttention

ROOT = Path(__file__).resolve().parents[1]


def load_nodes(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)


@unittest.skipIf(torch is None, 'Requires the project PyTorch runtime.')
class TestCompositionTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, ops=torch.nn,
                         rearrange=rearrange)
        load_nodes('ldm_patched/ldm/modules/diffusionmodules/util.py', {'checkpoint'}, namespace)
        load_nodes('ldm_patched/ldm/modules/attention.py',
                   {'exists', 'default', 'GEGLU', 'FeedForward', 'CrossAttention',
                    'BasicTransformerBlock', 'SpatialTransformer', 'attention_pytorch'}, namespace)
        namespace['optimized_attention'] = namespace['attention_pytorch']
        namespace['optimized_attention_masked'] = namespace['attention_pytorch']
        cls.Block = namespace['BasicTransformerBlock']
        cls.Spatial = namespace['SpatialTransformer']
        cls.attend = staticmethod(namespace['attention_pytorch'])
        cls.attention_globals = namespace
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def scene(self, dtype=torch.float32 if torch is not None else None):
        torch.manual_seed(41)
        block = self.Block(32, 4, 8, context_dim=32, checkpoint=False).eval().to(dtype=dtype)
        x = torch.randn(1, 28 * 36, 32, dtype=dtype)
        context = torch.randn(1, 77, 32, dtype=dtype)
        value = dict(version=1, enabled=True,
                     boxes=[dict(x=0, y=0, w=.5, h=1, strength=1, prompt='fox'),
                            dict(x=.5, y=0, w=.5, h=1, strength=1, prompt='dog')])
        return block, x, context, value

    @staticmethod
    def options(attention=None):
        options = dict(spatial_shape=(28, 36), original_shape=[1, 4, 112, 144], cond_or_uncond=[0])
        if attention is not None:
            options['patches'] = {'fooocus_composition': [attention]}
        return options

    def test_identical_prompts_preserve_full_frame_result_in_three_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                block, x, context, value = self.scene(dtype)
                composed = compose_conditioning([[context, {}]], value, 1152, 896,
                                                lambda **kw: [[context, {}]], ['scene'], 1)
                coupled = block(x.clone(), context, self.options(composed[0][1]['composition_attention']))
                original = block(x.clone(), context, self.options())
                torch.testing.assert_close(coupled, original, atol=.02 if dtype == torch.bfloat16 else .003,
                                           rtol=.01 if dtype == torch.bfloat16 else .002)
                self.assertEqual(coupled.dtype, dtype)
                self.assertTrue(torch.isfinite(coupled).all())

    def test_region_changes_cross_attention_only_inside_its_mask_in_one_block(self):
        with torch.inference_mode():
            block, x, context, value = self.scene()
            value['boxes'] = value['boxes'][:1]
            regional = torch.randn(1, 154, 32)  # Different prompt length is supported.
            composition = compose_conditioning([[context, {}]], value, 1152, 896,
                                               lambda **kw: [[regional, {}]], ['scene'], 1)
            attention = composition[0][1]['composition_attention']
            coupled = block(x.clone(), context, self.options(attention)).reshape(1, 28, 36, 32)
            original = block(x.clone(), context, self.options()).reshape(1, 28, 36, 32)
            torch.testing.assert_close(coupled[:, :, 18:], original[:, :, 18:])
            self.assertGreater((coupled[:, :, :16] - original[:, :, :16]).abs().max().item(), .001)

    def test_regional_attention_projects_shared_query_once_per_block_call(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for batch in (1, 2):
                with self.subTest(dtype=dtype, batch=batch), torch.inference_mode():
                    block, x, context, layout = self.scene(dtype)
                    x, context = x.repeat(batch, 1, 1), context.repeat(batch, 1, 1)
                    layout['boxes'] = [dict(x=i / 6, y=0, w=1 / 6, h=1, strength=1,
                                            prompt=f'subject {i}') for i in range(6)]
                    regional = {f'subject {i}, scene': torch.randn(1, 77 * (1 + i % 2), 32, dtype=dtype)
                                for i in range(6)}
                    composed = compose_conditioning([[context, {}]], layout, 1152, 896,
                        lambda texts, **kw: [[regional[texts[0]], {}]], ['scene'], 1)
                    attention = composed[0][1]['composition_attention']
                    options = self.options(attention)
                    # An identity query hook forces the independent projection path.
                    reference_options = dict(options, patches=dict(options['patches'],
                        attn2_patch=[lambda q, k, v, opt: (q, k, v)]))
                    for query in (x, x.flip(-1)):
                        with patch.object(block.attn2.to_q, 'forward', wraps=block.attn2.to_q.forward) as project:
                            expected = block(query.clone(), context, reference_options)
                            self.assertEqual(project.call_count, 7)
                            project.reset_mock()
                            actual = block(query.clone(), context, options)
                            self.assertEqual(project.call_count, 1)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_query_sharing_falls_back_for_attention_replacements_and_module_hooks(self):
        with torch.inference_mode():
            for kind in ('query_patch', 'replacement', 'attention_pre_hook', 'attention_hook', 'query_pre_hook',
                         'global_pre_hook', 'global_hook'):
                with self.subTest(kind=kind):
                    block, x, context, layout = self.scene()
                    composed = compose_conditioning([[context, {}]], layout, 1152, 896,
                        lambda **kw: [[torch.randn(1, 77, 32), {}]], ['scene'], 1)
                    options = self.options(composed[0][1]['composition_attention'])
                    handle = None
                    if kind == 'query_patch':
                        options['patches']['attn2_patch'] = [lambda q, k, v, opt: (q + k.mean(), k, v)]
                    elif kind == 'replacement':
                        options.update(block=('input', 1), block_index=0)
                        def replacement(q, k, v, opt):
                            q.add_(k.mean())
                            return self.attend(q, k, v, opt['n_heads'])
                        options['patches_replace'] = {'attn2': {('input', 1, 0): replacement}}
                    elif kind == 'attention_pre_hook':
                        def pre_hook(module, args, kwargs):
                            return (args[0] + kwargs['context'].mean(),), kwargs
                        handle = block.attn2.register_forward_pre_hook(pre_hook, with_kwargs=True)
                    elif kind == 'attention_hook':
                        handle = block.attn2.register_forward_hook(lambda module, args, output: output * 1.05)
                    elif kind == 'query_pre_hook':
                        handle = block.attn2.to_q.register_forward_pre_hook(lambda module, args: (args[0] + .3,))
                    elif kind == 'global_pre_hook':
                        def global_pre_hook(module, args):
                            if module is block.attn2:
                                return (args[0] + .3,)
                        handle = torch.nn.modules.module.register_module_forward_pre_hook(global_pre_hook)
                    else:
                        handle = torch.nn.modules.module.register_module_forward_hook(
                            lambda module, args, output: output * 1.05 if module is block.attn2 else None)
                    try:
                        with patch.object(block.attn2.to_q, 'forward', wraps=block.attn2.to_q.forward) as project:
                            actual = block(x.clone(), context, options)
                            self.assertEqual(project.call_count, 3)
                        self.assertTrue(torch.isfinite(actual).all())
                        reference_options = dict(options, patches=dict(options['patches']))
                        if 'attn2_patch' not in reference_options['patches']:
                            reference_options['patches']['attn2_patch'] = [lambda q, k, v, opt: (q, k, v)]
                        expected = block(x.clone(), context, reference_options)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    finally:
                        if handle is not None:
                            handle.remove()

    def test_shared_query_supports_manual_weight_casting(self):
        namespace = dict(torch=torch, ldm_patched=types.SimpleNamespace(modules=types.SimpleNamespace(
            model_management=types.SimpleNamespace(device_supports_non_blocking=lambda device: False))))
        load_nodes('ldm_patched/modules/ops.py', {'cast_bias_weight', 'disable_weight_init', 'manual_cast'}, namespace)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                original, x, context, layout = self.scene(dtype)
                block = self.Block(32, 4, 8, context_dim=32, checkpoint=False,
                                   operations=namespace['manual_cast']).eval()
                block.load_state_dict(original.state_dict())
                self.assertEqual(block.attn2.to_q.weight.dtype, torch.float32)
                attention = compose_conditioning([[context, {}]], layout, 1152, 896,
                    lambda **kw: [[torch.randn(1, 77, 32, dtype=dtype), {}]], ['scene'], 1)[0][1]['composition_attention']
                options = self.options(attention)
                reference_options = dict(options, patches=dict(options['patches'],
                    attn2_patch=[lambda q, k, v, opt: (q, k, v)]))
                expected = block(x.clone(), context, reference_options)
                with patch.object(block.attn2.to_q, 'forward', wraps=block.attn2.to_q.forward) as project:
                    actual = block(x.clone(), context, options)
                    self.assertEqual(project.call_count, 1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shared_query_survives_dynamo_tracing_and_changing_inputs(self):
        if not hasattr(torch, 'compile'):
            self.skipTest('Requires torch.compile.')
        with torch.inference_mode():
            block, x, context, layout = self.scene()
            attention = compose_conditioning([[context, {}]], layout, 1152, 896,
                lambda **kw: [[torch.randn(1, 77, 32), {}]], ['scene'], 1)[0][1]['composition_attention']
            # The eager backend checks Dynamo capture; no CUDA/Inductor claim.
            graphs = []
            def eager_backend(graph, example_inputs):
                graphs.append(graph)
                return graph.forward
            compiled = torch.compile(block, backend=eager_backend, fullgraph=False)
            for query in (x, x.flip(-1)):
                for options in (self.options(attention), self.options()):
                    actual = compiled(query.clone(), context, options)
                    expected = block(query.clone(), context, options)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(graphs, 'Dynamo must capture graphs rather than only run the eager fallback.')

    def test_real_interpolation_normalizes_overlap_after_resize(self):
        with torch.inference_mode():
            _, _, context, value = self.scene()
            value['boxes'][1]['x'] = .25
            value['boxes'][1]['strength'] = 3
            attention = compose_conditioning([[context, {}]], value, 1152, 896,
                                             lambda **kw: [[context, {}]], ['scene'], 1)[0][1]['composition_attention']
            for height, width in ((28, 36), (14, 18), (7, 9), (113, 145)):
                reference = torch.zeros(1, height * width, 32)
                weights = attention.weights(height, width, reference)
                torch.testing.assert_close(sum(weights), torch.ones(1, height * width, 1))
                self.assertTrue(all(torch.isfinite(w).all() and (w >= 0).all() for w in weights))
                self.assertIs(weights, attention.weights(height, width, reference))

    def test_duplicate_prompts_share_attention_without_changing_mask_blend(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                block, x, context, layout = self.scene(dtype)
                prompts = ('fox', 'fox', 'dog', 'fox', 'dog', 'fox')
                strengths = (.1, .3, 1, 3, 1.7, 2)
                layout['boxes'] = [dict(x=.05 + i * .12, y=.1, w=.25, h=.8,
                                        prompt=prompt, strength=strength)
                                   for i, (prompt, strength) in enumerate(zip(prompts, strengths))]
                embeddings = {name: torch.randn(1, 77, 32, dtype=dtype) for name in ('fox', 'dog')}
                encode = Mock(side_effect=lambda texts, **kw: [[embeddings[texts[0].split(',')[0]], {}]])
                coupled = compose_conditioning([[context, {}]], layout, 1152, 896, encode, ['scene'], 1)
                optimized = coupled[0][1]['composition_attention']
                self.assertEqual(encode.call_count, 2)
                self.assertEqual(len(optimized.regions), 2)
                background, masks = build_box_masks(layout['boxes'], 1152, 896)
                reference = CompositionAttention(torch.from_numpy(background),
                    [(embeddings[box['prompt']], torch.from_numpy(mask), box['strength'])
                     for box, mask in zip(layout['boxes'], masks)])
                original = block(x.clone(), context, self.options(reference))
                calls = []
                hook = block.attn2.register_forward_hook(lambda *args: calls.append(1))
                try:
                    result = block(x.clone(), context, self.options(optimized))
                finally:
                    hook.remove()
                self.assertEqual(len(calls), 3)  # Global + two distinct prompts, rather than seven.
                torch.testing.assert_close(result, original,
                    atol=.02 if dtype == torch.bfloat16 else .003 if dtype == torch.float16 else 1e-6,
                    rtol=.01 if dtype == torch.bfloat16 else .002 if dtype == torch.float16 else 1e-5)

    def test_accumulation_does_not_mutate_attention_outputs_or_reuse_old_results(self):
        with torch.inference_mode():
            _, _, context, layout = self.scene()
            attention = compose_conditioning([[context, {}]], layout, 1152, 896,
                                             lambda **kw: [[context, {}]], ['scene'], 1)[0][1]['composition_attention']
            base = torch.randn(1, 28 * 36, 32)
            region = torch.randn_like(base)
            saved_base, saved_region = base.clone(), region.clone()
            weights = attention.weights(28, 36, base)
            expected = base * weights[0] + region * weights[1] + region * weights[2]
            for _ in range(2):
                attend = Mock(side_effect=[base, region, region])
                result = attention(attend, context, self.options())
                torch.testing.assert_close(result, expected)
                torch.testing.assert_close(base, saved_base, rtol=0, atol=0)
                torch.testing.assert_close(region, saved_region, rtol=0, atol=0)

    def test_attention_backends_match_with_long_regional_prompts_and_small_memory_budget(self):
        from torch.utils.checkpoint import checkpoint

        def free_memory(device, torch_free=False):
            return (8_000_000, 8_000_000) if torch_free else 8_000_000

        namespace = dict(torch=torch, Tensor=torch.Tensor, math=math, partial=partial,
                         List=List, NamedTuple=NamedTuple, Optional=Optional, Protocol=Protocol,
                         checkpoint=checkpoint, einsum=torch.einsum, _ATTN_PRECISION='fp32',
                         model_management=types.SimpleNamespace(get_free_memory=free_memory,
                                                               OOM_EXCEPTION=torch.OutOfMemoryError))
        path = 'ldm_patched/ldm/modules/sub_quadratic_attention.py'
        tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
        names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        load_nodes(path, names, namespace)
        load_nodes('ldm_patched/ldm/modules/attention.py',
                   {'exists', 'attention_basic', 'attention_split', 'attention_sub_quad'}, namespace)
        with torch.inference_mode():
            block, x, context, layout = self.scene()
            regional = torch.randn(1, 154, 32)
            attention = compose_conditioning([[context, {}]], layout, 1152, 896,
                                             lambda **kw: [[regional, {}]], ['scene'], 1)[0][1]['composition_attention']
            expected = block(x.clone(), context, self.options(attention))
            for name in ('attention_basic', 'attention_split', 'attention_sub_quad'):
                with self.subTest(backend=name), patch.dict(self.attention_globals,
                        optimized_attention=namespace[name], optimized_attention_masked=namespace[name]):
                    actual = block(x.clone(), context, self.options(attention))
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_pag_replacement_preserves_regional_cross_attention(self):
        from modules.guidance import make_pag_model_options, PAG_APPLIED_LAYER
        with torch.inference_mode():
            block, x, context, layout = self.scene()
            regional = torch.randn_like(context)
            attention = compose_conditioning([[context, {}]], layout, 1152, 896,
                                             lambda **kw: [[regional, {}]], ['scene'], 1)[0][1]['composition_attention']
            options = dict(self.options(attention), block=PAG_APPLIED_LAYER[:2], block_index=PAG_APPLIED_LAYER[2])
            pag = make_pag_model_options({'transformer_options': options})['transformer_options']
            calls = []
            hook = block.attn2.register_forward_hook(lambda *args: calls.append(1))
            try:
                actual = block(x.clone(), context, pag)
            finally:
                hook.remove()
            self.assertEqual(len(calls), 3)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertGreater((actual - block(x.clone(), context, options)).abs().max().item(), .001)
            self.assertNotIn('patches_replace', options)

    def test_tomesd_and_composition_preserve_spatial_grid(self):
        from ldm_patched.contrib.external_tomesd import TomePatchModel

        class Patcher:
            def __init__(self):
                self.patches = {}

            def clone(self):
                return self

            def set_model_attn1_patch(self, fn):
                self.patches['attn1_patch'] = [fn]

            def set_model_attn1_output_patch(self, fn):
                self.patches['attn1_output_patch'] = [fn]

        with torch.inference_mode():
            block, x, context, value = self.scene()
            attention = compose_conditioning([[context, {}]], value, 1152, 896,
                                             lambda **kw: [[context, {}]], ['scene'], 1)[0][1]['composition_attention']
            patched = TomePatchModel().patch(Patcher(), .3)[0]
            original_options = self.options()
            # Use the current grid as original_shape to actually exercise merging.
            original_options['original_shape'] = [1, 32, 28, 36]
            original_options['patches'] = dict(patched.patches)
            coupled_options = dict(original_options,
                                   patches=dict(patched.patches, fooocus_composition=[attention]))
            original = block(x.clone(), context, original_options)
            coupled = block(x.clone(), context, coupled_options)
            torch.testing.assert_close(coupled, original)

    def test_deepcache_bypasses_composition_and_refiner_has_fresh_caches(self):
        from modules.unet_cache import DeepCache
        with torch.inference_mode():
            _, x, context, value = self.scene()
            attention = compose_conditioning([[context, {}]], value, 1152, 896,
                                             lambda **kw: [[context, {}]], ['scene'], 1)[0][1]['composition_attention']
            attention.weights(28, 36, x)
            converted = attention.convert(lambda text: text[..., :16].clone())
            self.assertEqual(converted._weights, {})
            self.assertEqual(converted._contexts, {})
            self.assertEqual(converted.regions[0][0].shape[-1], 16)
            cache = DeepCache()
            cache.configure('balanced', steps=30)
            plan = cache.begin(torch.zeros(1, 4, 112, 144), torch.tensor([500.]), context, None, None,
                               self.options(attention), 9, 9)
            self.assertIsNone(plan)
            self.assertEqual(cache.bypasses, 1)

    @unittest.skipIf(rearrange is None, 'Requires the project einops dependency.')
    def test_sampler_to_spatial_transformer_with_existing_hooks_and_cfg(self):
        """Run CFG -> condition processing -> hook merge -> spatial attention together."""
        namespace = dict(torch=torch, math=math, collections=collections,
                         model_management=types.SimpleNamespace(get_free_memory=lambda device: 10**12))
        load_nodes('ldm_patched/modules/utils.py', {'repeat_to_batch_size'}, namespace)
        namespace['ldm_patched'] = types.SimpleNamespace(
            modules=types.SimpleNamespace(utils=types.SimpleNamespace(
                repeat_to_batch_size=namespace['repeat_to_batch_size'])))
        load_nodes('ldm_patched/modules/conds.py', {'CONDRegular'}, namespace)
        load_nodes('ldm_patched/modules/samplers.py',
                   {'get_area_and_mult', 'cond_equal_size', 'can_concat_cond', 'cond_cat',
                    'calc_cond_uncond_batch', 'sampling_function'}, namespace)
        Condition = namespace['CONDRegular']

        for width, height, grid in ((1152, 896, (28, 36)), (896, 1152, (36, 28)),
                                    (1536, 768, (12, 24))):
            for use_linear in (False, True):
                with self.subTest(size=(width, height), linear=use_linear), torch.inference_mode():
                    _, _, context, layout = self.scene()
                    layout['boxes'] = layout['boxes'][:1]
                    regional = torch.randn(1, 154, 32)
                    composition = compose_conditioning([[context, {}]], layout, width, height,
                                                       lambda **kw: [[regional, {}]], ['scene'], 1)
                    positive = [dict(composition[0][1], model_conds={'c_crossattn': Condition(context)})]
                    negative = [dict(model_conds={'c_crossattn': Condition(-context)})]
                    spatial = self.Spatial(32, 4, 8, depth=2, context_dim=32,
                                           use_linear=use_linear, use_checkpoint=False).eval()
                    x = torch.randn(2, 32, *grid)
                    calls, text_lengths, grids = [], [], []

                    def text_hook(q, k, v, options):
                        text_lengths.append((tuple(options['cond_or_uncond']), k.shape[1]))
                        return q, k, v

                    def replacement(q, k, v, options):
                        return self.attend(q, k, v, options['n_heads'])

                    def denoise(data, timestep, c_crossattn, transformer_options):
                        calls.append(tuple(transformer_options['cond_or_uncond']))
                        options = dict(transformer_options, original_shape=list(data.shape), block=('input', 1))
                        return spatial(data, c_crossattn, options)

                    original_call = CompositionAttention.__call__

                    def record_grid(attention, attend, text, options):
                        grids.append(options['spatial_shape'])
                        return original_call(attention, attend, text, options)

                    hooks = {'attn2_patch': [text_hook]}
                    replacements = {'attn2': {('input', 1): replacement}}
                    model_options = {'transformer_options': {'patches': hooks,
                                                            'patches_replace': replacements}}
                    model = types.SimpleNamespace(memory_required=lambda shape: 0, apply_model=denoise)
                    expected_positive = denoise(x.clone(), None, context.repeat(2, 1, 1),
                                               dict(cond_or_uncond=[0], patches=dict(hooks,
                                                    fooocus_composition=[composition[0][1]['composition_attention']]),
                                                    patches_replace=replacements))
                    expected_negative = denoise(x.clone(), None, -context.repeat(2, 1, 1),
                                               dict(cond_or_uncond=[1], patches=hooks,
                                                    patches_replace=replacements))
                    for cfg in (4., 1., 4.):
                        calls.clear()
                        text_lengths.clear()
                        grids.clear()
                        with patch.object(CompositionAttention, '__call__', record_grid):
                            result = namespace['sampling_function'](
                                model, x.clone(), torch.ones(2), negative, positive, cfg, model_options)
                        torch.testing.assert_close(result, expected_negative +
                                                   (expected_positive - expected_negative) * cfg)
                        self.assertEqual(calls, [(0,)] if cfg == 1 else [(0,), (1,)])
                        self.assertEqual(grids, [grid, grid])
                        self.assertEqual(text_lengths.count(((0,), 154)), 2)
                        self.assertEqual(text_lengths.count(((1,), 154)), 0)
                        self.assertNotIn('fooocus_composition', hooks)


if __name__ == '__main__':
    unittest.main()
