"""Exercise the actual allocator without importing firmware-dependent patch modules."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def allocator(filename):
    tree = ast.parse((ROOT / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Space')
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), filename, 'exec'), namespace)
    return namespace['Space']


class SpaceBoundsTests(unittest.TestCase):
    def test_padding_cannot_extend_past_region(self):
        for filename in ('make_slots.py', 'make_slots14.py', 'make_slots15.py'):
            with self.subTest(filename=filename):
                space = allocator(filename)([(0x1000, 0x1003)])
                with self.assertRaises(SystemExit):
                    space.take(3, 'three bytes require four')
                self.assertEqual(space.free, [[0x1000, 0x1003]])
                self.assertEqual(space.used, [])

    def test_small_region_is_skipped_and_aligned_region_is_consumed(self):
        for filename in ('make_slots.py', 'make_slots14.py', 'make_slots15.py'):
            with self.subTest(filename=filename):
                space = allocator(filename)([(0x1000, 0x1003), (0x2000, 0x2004)])
                self.assertEqual(space.take(3, 'payload'), 0x2000)
                self.assertEqual(space.free, [[0x1000, 0x1003], [0x2004, 0x2004]])
                self.assertEqual(space.used, [('payload', 0x2000, 3)])


if __name__ == '__main__':
    unittest.main()
