import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


class GenerationControlTest(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.generate_filtered'),
            'Explicit candidate-budget/accepted-target entry point is required')
        return importlib.import_module('scripts.generate_filtered')

    def test_batch_sizes_never_exceed_cap_and_preserve_exact_budget(self):
        api = self.api()
        self.assertEqual(list(api.batch_sizes(400, 200)), [200, 200])
        self.assertEqual(list(api.batch_sizes(401, 200)), [200, 200, 1])
        self.assertEqual(sum(api.batch_sizes(3200, 333)), 3200)
        self.assertEqual(sum(api.batch_sizes(6400, 777)), 6400)

    def test_invalid_batch_sizes_rejected(self):
        api = self.api()
        for total, size in [(0, 2), (-1, 2), (4, 0), (4, -1)]:
            with self.assertRaises(ValueError):
                list(api.batch_sizes(total, size))

    def test_target_shortfall_is_not_success(self):
        api = self.api()
        self.assertFalse(api.target_met(99, 100))
        self.assertTrue(api.target_met(100, 100))
        self.assertTrue(api.target_met(120, 100))
        self.assertTrue(api.target_met(0, 0))  # Explicit candidate-shard mode.

    def test_nonempty_output_directory_is_not_overwritten(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / 'output'
            api.prepare_output(out)
            (out / 'molecules.sdf').write_text('preserve me')
            with self.assertRaises(FileExistsError):
                api.prepare_output(out)
            self.assertEqual((out / 'molecules.sdf').read_text(), 'preserve me')


if __name__ == '__main__':
    unittest.main()
