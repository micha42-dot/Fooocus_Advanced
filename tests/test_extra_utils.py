import numbers
import os
import tempfile
import unittest

import modules.flags
from modules import extra_utils


class TestUtils(unittest.TestCase):
    def test_try_eval_env_var(self):
        test_cases = [
            {
                "input": ("foo", str),
                "output": "foo"
            },
            {
                "input": ("1", int),
                "output": 1
            },
            {
                "input": ("1.0", float),
                "output": 1.0
            },
            {
                "input": ("1", numbers.Number),
                "output": 1
            },
            {
                "input": ("1.0", numbers.Number),
                "output": 1.0
            },
            {
                "input": ("true", bool),
                "output": True
            },
            {
                "input": ("True", bool),
                "output": True
            },
            {
                "input": ("false", bool),
                "output": False
            },
            {
                "input": ("False", bool),
                "output": False
            },
            {
                "input": ("True", str),
                "output": "True"
            },
            {
                "input": ("False", str),
                "output": "False"
            },
            {
                "input": ("['a', 'b', 'c']", list),
                "output": ['a', 'b', 'c']
            },
            {
                "input": ("{'a':1}", dict),
                "output": {'a': 1}
            },
            {
                "input": ("('foo', 1)", tuple),
                "output": ('foo', 1)
            }
        ]

        for test in test_cases:
            value, expected_type = test["input"]
            expected = test["output"]
            actual = extra_utils.try_eval_env_var(value, expected_type)
            self.assertEqual(expected, actual)

    def test_get_files_from_folder_filters_by_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            filenames = [
                'base-model.safetensors',
                'refiner-model.safetensors',
                'base-model.txt',
            ]
            for filename in filenames:
                with open(os.path.join(temp_dir, filename), 'w', encoding='utf-8'):
                    pass

            actual = extra_utils.get_files_from_folder(temp_dir, ['.safetensors'], name_filter='base')

            self.assertEqual(['base-model.safetensors'], actual)
