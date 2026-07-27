import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules import external_api


class TestExternalAPIModels(unittest.TestCase):
    def test_requested_models_are_available_on_both_providers_where_supported(self):
        fal_labels = set(external_api.get_model_choices(external_api.PROVIDER_FAL))
        replicate_labels = set(
            external_api.get_model_choices(external_api.PROVIDER_REPLICATE))

        for label in ('Nano Banana Pro', 'Nano Banana 2', 'MiniMax Image-01', 'Dreamina 3.1'):
            self.assertIn(label, fal_labels)
            self.assertIn(label, replicate_labels)
        self.assertIn('Krea 2 Medium', fal_labels)
        self.assertIn('Krea 2 Large', fal_labels)
        self.assertTrue(all(isinstance(choice, str) for choice in fal_labels))

    def test_gradio_label_resolves_to_internal_model_key(self):
        model = external_api.resolve_model(
            external_api.PROVIDER_REPLICATE, 'Nano Banana 2')

        self.assertEqual(model.key, 'replicate:nano-banana-2')
        self.assertEqual(
            external_api.get_default_model_label(external_api.PROVIDER_FAL),
            'Krea 2 Medium')

    def test_api_selection_starts_without_a_local_refiner(self):
        self.assertEqual(
            external_api.select_refiner_for_base(
                external_api.API_BASE_MODEL,
                'juggernautXL.safetensors',
                reset_api_refiner=True),
            'None')
        self.assertEqual(
            external_api.select_refiner_for_base(
                'juggernautXL.safetensors',
                'sd_xl_refiner.safetensors',
                reset_api_refiner=True),
            'sd_xl_refiner.safetensors')

    def test_api_generation_metadata_records_direct_generation(self):
        metadata = external_api.get_generation_metadata(
            external_api.PROVIDER_FAL,
            'fal:nano-banana-2')
        values = {key: value for _, key, value in metadata}

        self.assertEqual(values['generation_source'], 'External API')
        self.assertEqual(values['api_provider'], external_api.PROVIDER_FAL)
        self.assertEqual(values['api_model'], 'Nano Banana 2')
        self.assertEqual(values['api_local_refinement'], 'Disabled')
        self.assertNotIn('api_refine_strength', values)

    def test_api_generation_metadata_records_optional_refinement(self):
        metadata = external_api.get_generation_metadata(
            external_api.PROVIDER_REPLICATE,
            'replicate:dreamina-v3.1',
            'juggernautXL.safetensors',
            0.25)
        values = {key: value for _, key, value in metadata}

        self.assertEqual(values['api_model'], 'Dreamina 3.1')
        self.assertEqual(
            values['api_local_refinement'], 'juggernautXL.safetensors')
        self.assertEqual(values['api_refine_strength'], 0.25)

    def test_nano_banana_payload_maps_fooocus_settings(self):
        requests = external_api.build_requests(
            'fal:nano-banana-2',
            'portrait',
            1152,
            896,
            2,
            123,
            'jpeg',
        )

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['num_images'], 2)
        self.assertEqual(requests[0]['seed'], 123)
        self.assertEqual(requests[0]['aspect_ratio'], '5:4')
        self.assertEqual(requests[0]['output_format'], 'jpeg')
        self.assertEqual(requests[0]['resolution'], '1K')

    def test_dreamina_uses_custom_fal_dimensions(self):
        requests = external_api.build_requests(
            'fal:dreamina-v3.1',
            'cinematic street',
            1344,
            768,
            1,
            77,
            'png',
            enhance_prompt=False,
        )

        self.assertEqual(requests[0]['image_size'], {'width': 1344, 'height': 768})
        self.assertFalse(requests[0]['enhance_prompt'])
        self.assertEqual(requests[0]['seed'], 77)

    def test_minimax_batches_to_provider_limit(self):
        requests = external_api.build_requests(
            'replicate:minimax-image-01',
            'editorial portrait',
            1024,
            1024,
            12,
            5,
            'png',
        )

        self.assertEqual([request['number_of_images'] for request in requests], [9, 3])
        self.assertTrue(all(request['prompt_optimizer'] for request in requests))

    def test_single_image_replicate_models_create_multiple_requests(self):
        requests = external_api.build_requests(
            'replicate:nano-banana-pro',
            'product photo',
            1024,
            1024,
            3,
            8,
            'webp',
        )

        self.assertEqual(len(requests), 3)
        self.assertTrue(all(request['aspect_ratio'] == '1:1' for request in requests))

    def test_extract_image_urls_ignores_prediction_control_urls(self):
        response = {
            'output': [
                'https://replicate.delivery/one.png',
                {'url': 'https://replicate.delivery/two.png'},
            ],
            'urls': {
                'get': 'https://api.replicate.com/v1/predictions/123',
                'cancel': 'https://api.replicate.com/v1/predictions/123/cancel',
            },
        }

        self.assertEqual(
            external_api.extract_image_urls(response),
            [
                'https://replicate.delivery/one.png',
                'https://replicate.delivery/two.png',
            ],
        )


class TestExternalAPIKeyStorage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_path = Path(self.temp_dir.name) / '.env'
        self.env_path_patcher = patch.object(
            external_api, 'ENV_FILE_PATH', self.env_path)
        self.env_path_patcher.start()

    def tearDown(self):
        self.env_path_patcher.stop()
        self.temp_dir.cleanup()

    def test_save_and_resolve_key_from_local_env_file(self):
        with patch.dict(os.environ, {}, clear=True):
            external_api.save_api_key(
                external_api.PROVIDER_FAL, 'fal-secret-token')

            self.assertEqual(
                external_api.resolve_api_key(external_api.PROVIDER_FAL, ''),
                'fal-secret-token')
            self.assertIn(
                'FAL_KEY="fal-secret-token"',
                self.env_path.read_text(encoding='utf-8'))

    def test_entered_and_process_keys_take_precedence_over_saved_key(self):
        external_api.save_api_key(
            external_api.PROVIDER_FAL, 'saved-secret-token')

        with patch.dict(
                os.environ, {external_api.FAL_KEY_ENV: 'process-secret-token'},
                clear=True):
            self.assertEqual(
                external_api.resolve_api_key(
                    external_api.PROVIDER_FAL, 'entered-secret-token'),
                'entered-secret-token')
            self.assertEqual(
                external_api.resolve_api_key(external_api.PROVIDER_FAL, ''),
                'process-secret-token')

    def test_delete_removes_only_selected_provider_key(self):
        self.env_path.write_text(
            '# Keep this comment\n'
            'UNRELATED_SETTING="untouched"\n'
            'FAL_KEY="fal-secret-token"\n'
            'REPLICATE_API_TOKEN="replicate-secret-token"\n',
            encoding='utf-8')

        self.assertTrue(
            external_api.delete_saved_api_key(external_api.PROVIDER_FAL))
        remaining = self.env_path.read_text(encoding='utf-8')

        self.assertNotIn('fal-secret-token', remaining)
        self.assertIn('UNRELATED_SETTING="untouched"', remaining)
        self.assertIn('REPLICATE_API_TOKEN="replicate-secret-token"', remaining)
        self.assertEqual(
            external_api.get_saved_api_key(external_api.PROVIDER_REPLICATE),
            'replicate-secret-token')

    def test_saved_key_status_never_contains_the_key(self):
        external_api.save_api_key(
            external_api.PROVIDER_REPLICATE, 'private-replicate-token')

        with patch.dict(os.environ, {}, clear=True):
            status = external_api.get_api_key_status(
                external_api.PROVIDER_REPLICATE)

        self.assertIn('Saved locally', status)
        self.assertNotIn('private-replicate-token', status)

    def test_save_rejects_whitespace_and_line_injection(self):
        for invalid_key in ('', 'contains spaces', 'token\nINJECTED=value'):
            with self.subTest(invalid_key=invalid_key):
                with self.assertRaises(external_api.ExternalAPIError):
                    external_api.save_api_key(
                        external_api.PROVIDER_FAL, invalid_key)
        self.assertFalse(self.env_path.exists())


if __name__ == '__main__':
    unittest.main()
