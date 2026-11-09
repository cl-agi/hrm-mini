import json
from pathlib import Path
import tempfile
import unittest
from hrm.config import read_config


class ConfigLoadingTests(unittest.TestCase):
    def test_json_scientific_numbers_remain_numeric(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('config.json', 'config.yaml'):
                path=Path(directory)/name
                value={'lr':1e-5,'arch':{'norm_eps':1e-6}}
                path.write_text(json.dumps(value))
                self.assertEqual(read_config(path),value)

    def test_legacy_yaml_in_json_filename_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model_config.json'
            path.write_text('arch:\n  name: hrm@HRM\n  norm_eps: 1.0e-06\n')
            self.assertEqual(read_config(path),{'arch':{'name':'hrm@HRM','norm_eps':1e-6}})

    def test_non_mapping_config_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            for text in ('', 'null', '[]', 'invalid configuration'):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaisesRegex(ValueError, 'JSON object or YAML mapping'):
                        read_config(path)


if __name__=='__main__':unittest.main()
