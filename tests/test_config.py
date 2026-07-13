"""YAML config 경계에서 silent fallback을 허용하지 않는지 검증한다."""

from pathlib import Path
import tempfile
import unittest

from Utils.config import load_config


class ConfigLoadingTest(unittest.TestCase):
    def test_empty_yaml_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.yaml"
            path.write_text("", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                load_config(str(path))

    def test_non_mapping_paths_section_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text("paths: []\n", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                load_config(str(path))


if __name__ == "__main__":
    unittest.main()
