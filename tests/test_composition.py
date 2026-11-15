"""Composition data, mask blending, worker wiring and real refiner conversion paths."""

import ast
import collections
import json
from pathlib import Path
import random
import types
import unittest
from unittest.mock import Mock

import numpy as np

from modules.composition import (
    EMPTY_COMPOSITION, COMPOSITION_METHOD, CompositionError, active_boxes, build_box_masks, compose_conditioning,
    parse_composition, serialize_composition, spatial_metadata, validate_composition_mode,
    crop_control_hint,
)

ROOT = Path(__file__).resolve().parents[1]


def layout(**changes):
    box = dict(x=.1, y=.2, w=.4, h=.5, strength=1.0, prompt='red fox')
    box.update(changes)
    return dict(version=1, enabled=True, boxes=[box])


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]


class Tensor:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return Tensor(self.value)


class SamplerArray(np.ndarray):
    """CPU tensor operations needed by the real sampler's crop/blend path."""
    @property
    def device(self):
        return 'cpu'

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis)

    def repeat(self, *sizes):
        return np.tile(np.asarray(self), sizes).view(SamplerArray)

    def chunk(self, count):
        return np.split(self, count)

    def movedim(self, source, destination):
        return np.moveaxis(self, source, destination)

    def to(self, device=None):
        return self


class ScalarCondition:
    def __init__(self, value):
        self.value = value

    def process_cond(self, **kwargs):
        return self

    def can_concat(self, other):
        return True

    def concat(self, others):
        return [self.value] + [other.value for other in others]


class TestComposition(unittest.TestCase):


    def test_non_composition_area_keeps_existing_size_conditioning(self):
        encode = load_function('ldm_patched/modules/samplers.py', 'encode_model_conds', {})
        model = Mock(return_value={})
        encode(model, [dict(area=(96, 96, 0, 0), model_conds={})],
               np.zeros((1, 4, 96, 192)), 'cpu', 'positive')
        self.assertEqual(model.call_args.kwargs['width'], 1536)
        self.assertEqual(model.call_args.kwargs['height'], 768)

    def test_worker_clears_processing_and_finishes_after_success_or_error(self):
        tree = ast.parse((ROOT / 'modules/async_worker.py').read_text(encoding='utf-8'))
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
        loop = next(n for n in worker.body if isinstance(n, ast.While))
        code = compile(ast.Module(body=[loop], type_ignores=[]), '<worker lifecycle>', 'exec')

        class APIError(Exception):
            pass

        class Cancelled(Exception):
            pass

        class EndLoop(Exception):
            pass

        for error in (None, CompositionError('empty boxes'), RuntimeError('sampling failed'),
                      APIError('API failed'), Cancelled('cancelled')):
            with self.subTest(error=error):
                task = types.SimpleNamespace(processing=True, generate_image_grid=False, yields=[], results=[])
                pending = [task]

                def sleep(_):
                    if not pending:
                        raise EndLoop()

                def handler(_):
                    if error:
                        raise error

                settings = {123: object()}
                metrics = types.SimpleNamespace(begin_generation=Mock(), finish_generation=Mock())
                namespace = dict(time=types.SimpleNamespace(sleep=sleep), async_tasks=pending,
                                 handler=handler, performance_metrics=metrics, CompositionError=CompositionError,
                                 external_api=types.SimpleNamespace(ExternalAPICancelled=Cancelled, ExternalAPIError=APIError),
                                 pipeline=types.SimpleNamespace(prepare_text_encoder=Mock()),
                                 traceback=types.SimpleNamespace(print_exc=Mock()), pid=123,
                                 modules=types.SimpleNamespace(patch=types.SimpleNamespace(patch_settings=settings)))
                with self.assertRaises(EndLoop):
                    exec(code, namespace)
                self.assertFalse(task.processing)
                self.assertEqual(sum(flag == 'finish' for flag, _ in task.yields), 1)
                self.assertEqual(settings, {})
                metrics.finish_generation.assert_called_once_with(
                    'complete' if error is None else 'cancelled' if isinstance(error, Cancelled) else 'error')


    def test_controlnet_uses_distinct_spatial_crops_and_preserves_chains(self):
        tree = ast.parse((ROOT / 'ldm_patched/contrib/external.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ControlNetApplyAdvanced')
        method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'apply_controlnet')
        namespace = dict(crop_control_hint=crop_control_hint)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<ControlNet apply>', 'exec'), namespace)
        apply = namespace['apply_controlnet']

        class Control:
            def copy(self):
                return Control()

            def set_cond_hint(self, hint, *args):
                self.hint = hint
                return self

            def set_previous_controlnet(self, previous):
                self.previous = previous

        image = np.zeros((1, 768, 1024, 3), dtype=np.float32).view(SamplerArray)
        image[:, :, :512] = 1
        image[:, :, 512:] = 2
        positive = [[1, {}], [2, {'area': ('percentage', 1, .5, 0, 0)}],
                    [3, {'area': ('percentage', 1, .5, 0, .5)}]]
        negative = [[0, dict(meta)] for _, meta in positive]
        pos, neg = apply(None, positive, negative, Control(), image, 1, 0, 1)
        self.assertEqual(pos[0][1]['control'].hint.shape, (1, 3, 768, 1024))
        for i in (1, 2):
            hint = pos[i][1]['control'].hint
            self.assertEqual(hint.shape, (1, 3, 768, 512))
            self.assertTrue(np.all(hint == i))
            self.assertIs(pos[i][1]['control'], neg[i][1]['control'])
            self.assertIsNot(pos[i][1]['control'], pos[0][1]['control'])
        chained, _ = apply(None, pos, neg, Control(), image, .5, 0, .8)
        for first, second in zip(pos, chained):
            self.assertIs(second[1]['control'].previous, first[1]['control'])
        np.testing.assert_array_equal(crop_control_hint(image.movedim(-1, 1), (96, 64, 0, 64)),
                                      pos[2][1]['control'].hint)
        self.assertEqual(apply(None, positive, negative, Control(), image, 0, 0, 1), (positive, negative))


    def test_roundtrip_keeps_unicode_quotes_and_multiline_prompt(self):
        value = layout(prompt='Fuchs, "rot"\n背景: Wald')
        self.assertEqual(parse_composition(serialize_composition(value)), value)

    def test_metadata_cannot_insert_markup_in_private_history(self):
        value = layout(prompt='<img src=x onerror=alert(1)> & fox')
        encoded = serialize_composition(value)
        self.assertNotIn('<', encoded)
        self.assertNotIn('&', encoded)
        self.assertEqual(parse_composition(encoded), value)

    def test_invalid_inputs_are_rejected(self):
        for value in ('{bad', '[]', dict(version=2, boxes=[]), layout(x=float('nan')),
                      layout(w=-1), layout(x=.9), layout(strength=float('inf')),
                      layout(prompt='x' * 2001), layout(y=True), dict(enabled='false', boxes=[]),
                      dict(boxes=[layout()['boxes'][0]] * 7)):
            with self.subTest(value=str(value)[:50]), self.assertRaises(CompositionError):
                parse_composition(value)

    def test_disabled_or_empty_boxes_do_not_encode_or_allocate_masks(self):
        positive = [[object(), {}]]
        encode = Mock()
        for value in (EMPTY_COMPOSITION, layout(prompt='  '), dict(layout(), enabled=False)):
            result = compose_conditioning(positive, value, 1024, 1024, encode, [], 1)
            self.assertIs(result, positive)
        encode.assert_not_called()

    def test_masks_follow_aspect_ratio_and_stay_inside_box(self):
        box = layout(x=.25, y=.25, w=.5, h=.5)['boxes'][0]
        background, masks = build_box_masks([box], 1536, 768)
        mask = masks[0]
        self.assertEqual(mask.shape, (96, 192))
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(mask[48, 96], 1)
        self.assertEqual(mask[:24].sum(), 0)
        self.assertEqual(mask[:, :48].sum(), 0)
        self.assertTrue(0 < mask[24, 48] < 1)
        self.assertAlmostEqual(float(background[48, 96]), .15, places=6)
        self.assertEqual(background[0, 0], 1)
        self.assertTrue(np.all(background > 0))

    def test_tiny_and_edge_boxes_have_nonzero_finite_masks(self):
        for box in (layout(x=.98, y=.98, w=.02, h=.02)['boxes'][0],
                    layout(x=0, y=0, w=1, h=1)['boxes'][0]):
            background, masks = build_box_masks([box], 64, 64)
            self.assertGreater(masks[0].sum(), 0)
            self.assertTrue(np.isfinite(masks[0]).all())
            self.assertGreater(background.min(), 0)


    def test_overlapping_boxes_have_finite_blend_and_full_background_coverage(self):
        box = layout()['boxes'][0]
        background, masks = build_box_masks([box, box], 1024, 1024)
        total = background + sum(masks)
        self.assertTrue((total > 0).all())
        self.assertTrue(np.isfinite(total).all())


    def test_global_expansion_does_not_dilute_each_box_prompt(self):
        encode = Mock(return_value=[[1, {}]])
        compose_conditioning([[0, {}]], layout(), 1024, 1024, encode,
                             ['forest', 'expansion describing other subjects'], 1, tensor_from_numpy=lambda x: x)
        encode.assert_called_once_with(texts=['red fox, forest'], pool_top_k=1)

    def test_mode_validation_allows_post_enhancement_but_rejects_image_edits(self):
        validate_composition_mode(layout(), goals=['cn'])
        validate_composition_mode(EMPTY_COMPOSITION, api_mode=True, goals=['inpaint'])
        for goals in (['inpaint'], ['upscale'], ['vary'], ['enhance']):
            with self.assertRaises(CompositionError):
                validate_composition_mode(layout(), goals=goals)
        with self.assertRaises(CompositionError):
            validate_composition_mode(layout(), api_mode=True)
        with self.assertRaises(CompositionError):
            validate_composition_mode(layout(prompt=''))

    def test_refiner_conversions_keep_regions_but_rebuild_model_conditioning(self):
        embedding, pooled, mask = Tensor('text'), Tensor('pooled'), object()
        region = dict(area=(64, 64, 0, 0), mask=mask, mask_strength=2, strength=1,
                      set_area_to_bounds=False, timestep_start=10, timestep_end=1)
        metadata = dict(region, pooled_output=pooled, control='base control')
        namespace = dict(spatial_metadata=spatial_metadata,
                         clip_separate_inner=lambda c, p, **kwargs: (c.clone(), p),
                         CONDRegular=lambda c: types.SimpleNamespace(cond=c))
        convert = load_function('modules/sample_hijack.py', 'clip_separate', namespace)
        result = convert([[embedding, metadata]])
        self.assertEqual(spatial_metadata(result[0][1]), region)
        self.assertNotIn('control', result[0][1])
        self.assertIsNot(result[0][1]['pooled_output'], pooled)
        joint = load_function('modules/sample_hijack.py', 'clip_separate_after_preparation', namespace)
        prepared = dict(metadata, model_conds={'c_crossattn': types.SimpleNamespace(cond=embedding), 'c_adm': 'base ADM'})
        refined = joint([prepared])[0]
        self.assertEqual(spatial_metadata(refined), region)
        self.assertEqual(list(refined['model_conds']), ['c_crossattn'])
        self.assertEqual(prepared['model_conds']['c_adm'], 'base ADM')
        # SD1.5 refiner conversion removes pooled_output, while keeping the layout.
        namespace['clip_separate_inner'] = lambda c, p, **kwargs: (c, None)
        self.assertNotIn('pooled_output', convert([[embedding, metadata]])[0][1])

    def test_cfg_one_clone_retains_region_metadata(self):
        namespace = {'torch': types.SimpleNamespace(Tensor=Tensor)}
        clone = load_function('modules/default_pipeline.py', 'clone_cond', namespace)
        original = [[Tensor('text'), dict(pooled_output=Tensor('pooled'), mask=object(), strength=2)]]
        copied = clone(original)
        self.assertIsNot(copied[0][0], original[0][0])
        self.assertIsNot(copied[0][1], original[0][1])
        self.assertIs(copied[0][1]['mask'], original[0][1]['mask'])
        self.assertEqual(copied[0][1]['strength'], 2)

    def test_real_worker_adds_attention_only_to_primary_positive_conditioning(self):
        tree = ast.parse((ROOT / 'modules/async_worker.py').read_text(encoding='utf-8'))
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and any(isinstance(a, ast.Constant) and a.value == 'Encoding composition boxes ...'
                             for a in ast.walk(n)) and isinstance(n.test, ast.Call)
                     and isinstance(n.test.func, ast.Name) and n.test.func.id == 'active_boxes')
        task = dict(task_seed=123, c=[[1, {}]], uc=[[2, {}]], positive=['forest'], positive_top_k=1)
        original_negative = task['uc']
        async_task = types.SimpleNamespace(composition=layout(), read_wildcards_in_order=False)
        namespace = dict(active_boxes=active_boxes, COMPOSITION_METHOD=COMPOSITION_METHOD, parse_composition=parse_composition, random=random,
                         apply_arrays=lambda text, i: text, apply_wildcards=lambda text, *args: text,
                         async_task=async_task, task=task, current_task_id=0, progressbar=Mock(),
                         current_progress=1, task_width=1024, task_height=768,
                         pipeline=types.SimpleNamespace(clip_encode=lambda **kwargs: [[3, {}]]),
                         serialize_composition=serialize_composition,
                         compose_conditioning=lambda *args: compose_conditioning(*args, tensor_from_numpy=lambda x: x))
        exec(compile(ast.Module(body=[block], type_ignores=[]), '<worker composition>', 'exec'), namespace)
        self.assertEqual(len(task['c']), 1)
        self.assertEqual(task['c'][0][1]['composition_attention'].regions[0][1].shape, (96, 128))
        self.assertIs(task['uc'], original_negative)
        self.assertEqual(original_negative, [[2, {}]])
        self.assertEqual(parse_composition(async_task.composition_metadata), layout())

    def test_worker_releases_conditioning_after_skip_stop_error_and_success(self):
        tree = ast.parse((ROOT / 'modules/async_worker.py').read_text(encoding='utf-8'))
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Tuple)
                    and [getattr(x, 'id', None) for x in n.target.elts] == ['current_task_id', 'task'])
        code = compile(ast.Module(body=[loop], type_ignores=[]), '<worker task loop>', 'exec')

        class Interrupted(Exception):
            pass

        for outcome in ('complete', 'skip', 'stop', 'error'):
            with self.subTest(outcome=outcome):
                tasks = [dict(c=object(), uc=object(), task_seed=i, positive=['scene']) for i in range(2)]
                async_task = types.SimpleNamespace(composition=EMPTY_COMPOSITION, image_number=2,
                                                   steps=1, last_stop=outcome, composition_metadata=None)
                processed = []

                def process(*args):
                    if processed:
                        # Previous skipped images must not retain device caches while the next runs.
                        self.assertNotIn('c', tasks[processed[-1]])
                        self.assertNotIn('uc', tasks[processed[-1]])
                    processed.append(args[5])
                    if len(processed) == 1:
                        if outcome in ('skip', 'stop'):
                            raise Interrupted()
                        if outcome == 'error':
                            raise RuntimeError('sampling failed')
                    return [], [], 1

                namespace = dict(tasks=tasks, async_task=async_task, progressbar=Mock(), current_progress=1,
                    time=types.SimpleNamespace(perf_counter=lambda: 0), initial_latent=None, width=1152, height=896,
                    api_mode=False, active_boxes=active_boxes, process_task=process, all_steps=2, callback=None,
                    controlnet_canny_path=None, controlnet_cpds_path=None, denoising_strength=1,
                    final_scheduler_name='test', goals=[], switch=1, loras=[], tiled=False, use_expansion=False,
                    preparation_steps=1, show_intermediate_results=False, persist_image=False,
                    get_sampling_step_count=lambda *args: 1, images_to_enhance=[], image_compositions=[],
                    performance_metrics=types.SimpleNamespace(record_stage=Mock()),
                    ldm_patched=types.SimpleNamespace(modules=types.SimpleNamespace(
                        model_management=types.SimpleNamespace(InterruptProcessingException=Interrupted))))
                if outcome == 'error':
                    with self.assertRaisesRegex(RuntimeError, 'sampling failed'):
                        exec(code, namespace)
                else:
                    exec(code, namespace)
                self.assertEqual(processed, [0] if outcome in ('stop', 'error') else [0, 1])
                for index in processed:
                    self.assertNotIn('c', tasks[index])
                    self.assertNotIn('uc', tasks[index])
                    self.assertEqual(tasks[index]['positive'], ['scene'])

    def test_metadata_import_clears_old_layout_and_loads_new_layout(self):
        namespace = dict(gr=types.SimpleNamespace(update=lambda **kwargs: kwargs), json=json,
                         modules=types.SimpleNamespace(config=types.SimpleNamespace(default_max_lora_number=0)),
                         Performance=types.SimpleNamespace(values=lambda: []),
                         EMPTY_COMPOSITION=EMPTY_COMPOSITION, CompositionError=CompositionError,
                         serialize_composition=serialize_composition)
        tree = ast.parse((ROOT / 'modules/meta_parser.py').read_text(encoding='utf-8'))
        load = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_parameter_button_click')
        for call in ast.walk(load):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id.startswith('get_'):
                namespace[call.func.id] = lambda *args, **kwargs: None
        load = load_function('modules/meta_parser.py', 'load_parameter_button_click', namespace)
        self.assertEqual(load({}, False, '')[-1], EMPTY_COMPOSITION)
        self.assertEqual(parse_composition(load({'composition': serialize_composition(layout())}, False, '')[-1]), layout())
        self.assertEqual(load({'composition': '{bad'}, False, '')[-1], EMPTY_COMPOSITION)

    def test_editor_is_between_aspect_ratio_and_image_count_and_control_is_appended(self):
        source = (ROOT / 'webui.py').read_text(encoding='utf-8')
        self.assertLess(source.index("label='Aspect Ratios'"), source.index("label='Compose with boxes'"))
        self.assertLess(source.index("label='Compose with boxes'"), source.index("label='Image Number'"))
        self.assertIn('ctrls += enhance_ctrls\n        ctrls += [composition_data]', source)
        self.assertIn('+ lora_ctrls + [composition_data]', source)

    def test_generation_completion_has_no_frontend_only_barrier(self):
        """Gradio 3.41 only dispatches trigger_after from backend status events."""
        tree = ast.parse((ROOT / 'webui.py').read_text(encoding='utf-8'))
        expressions = [n.value for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
        chain = next(n for n in expressions if any(isinstance(a, ast.Attribute)
                     and isinstance(a.value, ast.Name) and a.value.id == 'generate_button'
                     and a.attr == 'click' for a in ast.walk(n)))
        calls = []
        while isinstance(chain, ast.Call):
            calls.append(chain)
            chain = chain.func.value if isinstance(chain.func, ast.Attribute) else None
        calls.reverse()
        completed = next(i for i, c in enumerate(calls) if any(k.arg == 'fn'
                         and isinstance(k.value, ast.Name) and k.value.id == 'generate_clicked' for k in c.keywords))
        cleanup = calls[completed + 1]
        self.assertTrue(cleanup.args and isinstance(cleanup.args[0], ast.Lambda))
        fn = eval(compile(ast.Expression(cleanup.args[0]), '<cleanup>', 'eval'),
                  {'gr': types.SimpleNamespace(update=lambda **kwargs: kwargs)})
        updates = fn()
        self.assertTrue(updates[0]['visible'])  # Generate returns.
        self.assertFalse(updates[1]['visible'])  # Stop disappears.
        self.assertIs(updates[-1], False)  # Backend state_is_generating is reset.
        js = next(k.value.value for k in cleanup.keywords if k.arg == '_js')
        self.assertIn('generating(false)', js)
        for call in calls[:completed]:
            self.assertFalse(any(k.arg == 'fn' and isinstance(k.value, ast.Constant)
                                 and k.value.value is None for k in call.keywords))


if __name__ == '__main__':
    unittest.main()
