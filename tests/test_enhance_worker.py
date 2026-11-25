"""Exercise the real Enhance loop with a deterministic CPU renderer, without starting its worker thread."""

import ast
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from modules.enhance_masks import object_context_field, prepare_enhance_masks


class Interrupted(Exception):
    pass


class TestEnhanceWorker(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).resolve().parents[1] / 'modules' / 'async_worker.py'
        tree = ast.parse(self.path.read_text(encoding='utf-8'))
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
        handler = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == 'handler')
        loop = next(n for n in ast.walk(handler) if isinstance(n, ast.For)
                    and isinstance(n.iter, ast.Attribute) and n.iter.attr == 'enhance_ctrls')
        self.code = compile(ast.Module(body=[loop], type_ignores=[]), str(self.path), 'exec')
        self.process_task_node = next(n for n in worker.body if isinstance(n, ast.FunctionDef)
                                      and n.name == 'process_task')
        self.log_functions = [n for n in worker.body if isinstance(n, ast.FunctionDef)
                              and n.name in ('log_image', 'persist_last_enhancement')]
        self.masks = [np.zeros((40, 80), dtype=np.uint8) for _ in range(2)]
        self.masks[0][10:20, 5:15] = 255
        self.masks[1][10:20, 65:75] = 255
        self.rendered = []
        self.interrupt_at = None
        self.stop_kind = 'stop'
        # Same per-tab field order as AsyncTask; one active Enhance tab.
        fields = ['face', 'portrait', '', 'sam', '', 'vit_b', .25, .3, 0,
                  False, 'None', .3, .618, 0, False, True]
        self.task = types.SimpleNamespace(
            enhance_ctrls=[fields], save_final_enhanced_image_only=True,
            dino_erode_or_dilate=0, debugging_dino=False, debugging_enhance_masks_checkbox=False,
            black_out_nsfw=False, disable_intermediate_results=False, last_stop=False,
            yields=[], enhance_stats={0: 0})
        self.saved = []

        def render(*args, **kwargs):
            if self.interrupt_at == len(self.rendered):
                self.task.last_stop = self.stop_kind
                raise Interrupted()
            image, mask = args[17], args[18]
            output = image.copy()
            output[mask > 0] = len(self.rendered) + 1
            self.rendered.append(mask.copy())
            args[2](0, None, None, 1, output)
            kwargs['deferred_output'].update(height=40, imgs=[output], task={},
                                             use_expansion=False, width=80, loras=[])
            progress = args[5] + (100 - args[19]) * args[20] / args[0]
            return progress, output, 'portrait', ''

        def save(async_task, **kwargs):
            self.saved.append(kwargs['imgs'][0].copy())
            return ['result.png']

        self.env = dict(
            async_task=self.task, current_task_id=-1, current_progress=0, base_progress=0,
            preparation_steps=0, all_steps=10, enhance_steps=10, done_steps_upscaling=0,
            done_steps_inpainting=0, active_enhance_tabs=1, enhance_uov_after=False, total_count=1,
            index=0, img=np.zeros((40, 80, 3), dtype=np.uint8), height=40, width=80,
            time=types.SimpleNamespace(perf_counter=lambda: 0), progressbar=Mock(),
            generate_mask_from_image=Mock(side_effect=lambda *a, **k:
                                         (self.masks[0] | self.masks[1], 2, 2, 2, self.masks)),
            SAMOptions=lambda **k: k, object_context_field=object_context_field,
            prepare_enhance_masks=prepare_enhance_masks, erode_or_dilate=lambda mask, amount: mask,
            yield_result=Mock(), controlnet_canny_path=None, controlnet_cpds_path=None,
            denoising_strength=1, switch=5, tiled=False, use_expansion=False, use_style=False,
            use_synthetic_refiner=False, final_scheduler_name='karras', should_process_enhance_uov=False,
            process_enhance=render, save_and_log=save, exception_result=None,
            performance_metrics=types.SimpleNamespace(record_stage=Mock()),
            ldm_patched=types.SimpleNamespace(modules=types.SimpleNamespace(
                model_management=types.SimpleNamespace(InterruptProcessingException=Interrupted))))

    def run_loop(self):
        exec(self.code, self.env)

    def test_gallery_keeps_grid_after_grouping_finished_enhancements(self):
        path = self.path.parents[1] / 'webui.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        sort = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sort_enhance_images')
        namespace = {}
        exec(compile(ast.Module(body=[sort], type_ignores=[]), str(path), 'exec'), namespace)
        task = types.SimpleNamespace(should_enhance=True, images_to_enhance_count=3, enhance_stats={0: 1, 1: 0, 2: 1})
        images = ['one', 'two', 'three', 'one enhanced', 'three enhanced', 'grid']
        self.assertEqual(namespace['sort_enhance_images'](images, task),
                         ['one', 'one enhanced', 'two', 'three', 'three enhanced', 'grid'])

    def test_final_only_keeps_last_completed_tab_in_outputs(self):
        from modules import private_logger

        self.task.enhance_ctrls *= 2
        self.task.output_format = 'png'
        self.task.images_to_enhance_count = 1
        self.env.update(active_enhance_tabs=2, total_count=2, all_steps=20)
        namespace = dict(log=private_logger.log, persist_image_copy=private_logger.persist_image_copy)
        exec(compile(ast.Module(body=self.log_functions, type_ignores=[]), str(self.path), 'exec'), namespace)

        def save(async_task, imgs, persist_image, **kwargs):
            return [namespace['log_image'](async_task, imgs[0], [('Prompt', 'prompt', 'fox')],
                                           task={'positive': ['fox'], 'negative': ['fog']},
                                           persist_image=persist_image)]

        self.env.update(save_and_log=save, yield_result=lambda task, paths, *a, **kw: task.results.extend(paths))
        for last_detected in (False, True):
            with self.subTest(last_detected=last_detected), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'outputs'
                temp = Path(directory) / 'temp'
                self.task.results = []
                self.task.pending_enhanced_image = None
                self.env['current_task_id'] = -1
                first, last = self.masks
                self.env['generate_mask_from_image'] = Mock(side_effect=[
                    (first, 1, 1, 1, [first]),
                    (last, 1, 1, 1, [last]) if last_detected else (last * 0, 0, 0, 0, [])])
                with patch.object(private_logger.modules.config, 'path_outputs', str(output)), \
                     patch.object(private_logger.modules.config, 'temp_path', str(temp)), \
                     patch.object(private_logger.args_manager.args, 'disable_image_log', False):
                    self.run_loop()
                    latest = Path(self.task.results[-1])
                    self.assertTrue(latest.is_relative_to(output if last_detected else temp))
                    encoded = latest.read_bytes()
                    namespace['persist_last_enhancement'](self.task)
                    namespace['persist_last_enhancement'](self.task)
                final = Path(self.task.results[-1])
                self.assertEqual(list(output.rglob('*.png')), [final])
                self.assertEqual(final.read_bytes(), encoded)
                self.assertIn('<summary>Negative</summary>fog', final.with_name('log.html').read_text(encoding='utf-8'))

    def test_objects_accumulate_and_only_one_composite_is_saved(self):
        self.run_loop()
        self.assertEqual(2, len(self.rendered))
        self.assertEqual(1, len(self.saved))
        self.assertTrue(np.all(self.saved[0][self.masks[0] > 0] == 1))
        self.assertTrue(np.all(self.saved[0][self.masks[1] > 0] == 2))
        self.assertEqual({0: 1}, self.task.enhance_stats)
        progress = [entry[1][0] for entry in self.task.yields]
        self.assertEqual([50, 100], progress)
        self.assertEqual(10, self.env['done_steps_inpainting'])

    def test_stop_after_first_object_retains_completed_composite(self):
        self.interrupt_at = 1
        self.run_loop()
        self.assertEqual(1, len(self.saved))
        self.assertTrue(np.all(self.saved[0][self.masks[0] > 0] == 1))
        self.assertTrue(np.all(self.saved[0][self.masks[1] > 0] == 0))
        self.assertEqual('break', self.env['exception_result'])

    def test_skip_before_first_object_saves_nothing_and_clears_skip(self):
        self.interrupt_at = 0
        self.stop_kind = 'skip'
        self.run_loop()
        self.assertEqual([], self.saved)
        self.assertFalse(self.task.last_stop)
        self.assertEqual({0: 0}, self.task.enhance_stats)

    def test_combined_mode_and_inverted_mask_run_once(self):
        for separate, invert in [(False, False), (True, True)]:
            with self.subTest(separate=separate, invert=invert):
                self.rendered.clear()
                self.saved.clear()
                self.task.enhance_ctrls[0][-2:] = [invert, separate]
                self.run_loop()
                self.assertEqual(1, len(self.rendered))
                self.assertEqual(1, len(self.saved))

    def test_empty_mask_skips_rendering_and_accounts_for_progress(self):
        for mask in self.masks:
            mask[:] = 0
        self.run_loop()
        self.assertEqual([], self.rendered)
        self.assertEqual([], self.saved)
        self.assertEqual(10, self.env['done_steps_inpainting'])

    def test_intermediate_diffusion_pass_is_composited_without_saving(self):
        from modules.composition import CompositionError, compose_conditioning, verify_composition_applied

        image = np.ones((4, 4, 3), dtype=np.uint8)
        pipeline = types.SimpleNamespace(process_diffusion=Mock(return_value=[image]))
        save, publish = Mock(), Mock()
        namespace = dict(
            pipeline=pipeline, is_tiled_detail_latent=lambda latent: False,
            inpaint_worker=types.SimpleNamespace(current_task=types.SimpleNamespace(post_process=lambda img: img * 3)),
            get_sampling_step_count=lambda steps, latent: steps,
            modules=types.SimpleNamespace(config=types.SimpleNamespace(default_black_out_nsfw=False)),
            progressbar=Mock(), save_and_log=save, yield_result=publish,
            verify_composition_applied=verify_composition_applied)
        exec(compile(ast.Module(body=[self.process_task_node], type_ignores=[]), str(self.path), 'exec'), namespace)
        task = types.SimpleNamespace(last_stop=False, sampler_name='euler', cfg_scale=5,
                                     refiner_swap_method='joint', disable_preview=False, deep_cache_profile='off',
                                     black_out_nsfw=False, disable_intermediate_results=False)
        deferred = {}
        kwargs = dict(all_steps=20, async_task=task, callback=Mock(), controlnet_canny_path=None,
                      controlnet_cpds_path=None, current_task_id=0, denoising_strength=.3,
                      final_scheduler_name='karras', goals=['inpaint'], initial_latent={}, steps=10,
                      switch=5, positive_cond=[], negative_cond=[], task={'task_seed': 42}, loras=[],
                      tiled=False, use_expansion=False, width=4, height=4, base_progress=0,
                      preparation_steps=0, total_count=1, show_intermediate_results=True)
        imgs, paths, progress = namespace['process_task'](**kwargs, deferred_output=deferred)
        np.testing.assert_array_equal(image * 3, imgs[0])
        np.testing.assert_array_equal(imgs[0], deferred['imgs'][0])
        self.assertEqual([], paths)
        self.assertEqual(50, progress)
        save.assert_not_called()
        publish.assert_not_called()
        # The normal generation path must still save and publish immediately.
        save.return_value = ['normal.png']
        namespace['process_task'](**kwargs)
        save.assert_called_once()
        publish.assert_called_once()
        save.reset_mock()
        publish.reset_mock()
        layout = dict(version=1, enabled=True, boxes=[dict(x=0, y=0, w=.5, h=1, strength=1, prompt='fox')])
        kwargs['positive_cond'] = compose_conditioning([[1, {}]], layout, 64, 64,
            lambda **kw: [[2, {}]], ['scene'], 1, tensor_from_numpy=lambda x: x)
        with self.assertRaisesRegex(CompositionError, 'did not reach cross-attention'):
            namespace['process_task'](**kwargs)
        save.assert_not_called()
        publish.assert_not_called()

    def test_ui_and_task_parser_agree_on_enhance_field_order(self):
        ui = ast.parse((self.path.parents[1] / 'webui.py').read_text(encoding='utf-8'))
        controls = next(n.value for n in ast.walk(ui) if isinstance(n, ast.AugAssign)
                        and isinstance(n.target, ast.Name) and n.target.id == 'enhance_ctrls')
        tree = ast.parse(self.path.read_text(encoding='utf-8'))
        task_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AsyncTask')
        init = next(n for n in task_class.body if isinstance(n, ast.FunctionDef))
        loop = next(n for n in init.body if isinstance(n, ast.For)
                    and any(isinstance(a, ast.Attribute) and a.attr == 'default_enhance_tabs'
                            for a in ast.walk(n.iter)))
        parser_names = [n.targets[0].id for n in loop.body if isinstance(n, ast.Assign)]
        self.assertEqual([n.id for n in controls.elts], parser_names)


if __name__ == '__main__':
    unittest.main()
