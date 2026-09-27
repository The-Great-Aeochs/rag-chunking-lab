import unittest

import numpy as np

from vectordb.ann_indexes import INDEX_NAMES, ann_recall, build_index, search_index


class AnnIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.vectors = rng.normal(size=(256, 48)).astype("float32")
        cls.queries = cls.vectors[:5]

    def test_all_indexes_build_and_search(self):
        for name in INDEX_NAMES:
            with self.subTest(index=name):
                index, settings, build_ms = build_index(
                    name,
                    self.vectors,
                    nlist=8,
                    nprobe=2,
                    pq_bits=4,
                )
                _, indices, latency = search_index(index, self.queries, k=5, repeats=1)
                self.assertEqual(indices.shape, (5, 5))
                self.assertTrue(settings)
                self.assertGreaterEqual(build_ms, 0)
                self.assertGreaterEqual(latency, 0)

    def test_ann_recall_uses_exact_neighbors_as_reference(self):
        exact = np.array([[1, 2, 3], [4, 5, 6]])
        candidate = np.array([[1, 8, 3], [4, 5, 6]])
        self.assertAlmostEqual(ann_recall(exact, candidate), (2 / 3 + 1) / 2)


if __name__ == "__main__":
    unittest.main()
