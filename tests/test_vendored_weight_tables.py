import importlib.util
import os
import unittest


def _import_from_path(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestVendoredWeightTables(unittest.TestCase):
    def test_vendored_files_exist(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_dir = os.path.join(repo_root, "reporting_data")
        self.assertTrue(os.path.isfile(os.path.join(data_dir, "iestable.py")))
        self.assertTrue(os.path.isfile(os.path.join(data_dir, "iectable.py")))

    def test_iestable_has_expected_density(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(repo_root, "reporting_data", "iestable.py")
        mod = _import_from_path(path, "_test_iestable")
        rows = list(getattr(mod, "iestab", []) or [])
        self.assertGreaterEqual(len(rows), 180)
        self.assertAlmostEqual(float(rows[0][0]), 200.0, places=6)
        self.assertAlmostEqual(float(rows[-1][0]), 400.0, places=6)

    def test_iectable_has_expected_density(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(repo_root, "reporting_data", "iectable.py")
        mod = _import_from_path(path, "_test_iectable")
        pairs = getattr(mod, "iec_slambda", None)
        self.assertIsNotNone(pairs)
        self.assertGreaterEqual(len(list(pairs[0])), 180)


if __name__ == "__main__":
    unittest.main()
