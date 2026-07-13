import os
import tempfile
import unittest

from modules.model_paths import find_file_in_folder_list, get_file_name_from_folder_list


class TestModelPaths(unittest.TestCase):
    def test_finds_file_in_later_configured_folder(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            expected = os.path.join(second, 'model.safetensors')
            with open(expected, 'wb') as model_file:
                model_file.write(b'model')

            actual = find_file_in_folder_list('model.safetensors', [first, second])

            self.assertEqual(actual, os.path.realpath(expected))

    def test_finds_model_in_nested_folder_by_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            nested = os.path.join(folder, 'SDXL')
            os.makedirs(nested)
            expected = os.path.join(nested, 'Juggernaut.safetensors')
            with open(expected, 'wb') as model_file:
                model_file.write(b'model')

            actual = find_file_in_folder_list('juggernaut.safetensors', folder, recursive=True)

            self.assertEqual(actual, os.path.realpath(expected))
            self.assertEqual(get_file_name_from_folder_list(actual, folder),
                             os.path.join('SDXL', 'Juggernaut.safetensors'))

    def test_recursive_search_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as folder:
            nested = os.path.join(folder, 'SDXL')
            os.makedirs(nested)
            with open(os.path.join(nested, 'model.safetensors'), 'wb') as model_file:
                model_file.write(b'model')

            self.assertIsNone(find_file_in_folder_list('model.safetensors', folder))


if __name__ == '__main__':
    unittest.main()
