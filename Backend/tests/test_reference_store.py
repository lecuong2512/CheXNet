import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from Backend.cbir import SimilarCaseFinder
from Backend.ai_service.reference_store import ReferenceStore


class ReferenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.finder = SimilarCaseFinder(str(self.root / "legacy.faiss"), str(self.root / "legacy.json"))
        self.store = ReferenceStore(self.finder, self.root / "published")
        self.vector = np.ones((1, 1536), dtype=np.float32)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_replay_and_label_conflict(self):
        metadata = {"filename": "ref_one", "diseases": ["Pneumonia"]}
        self.assertTrue(self.store.add_case(self.vector, metadata))
        generation = self.store.generation
        self.assertTrue(self.store.add_case(self.vector, metadata))
        self.assertEqual(self.store.generation, generation)
        self.assertEqual(self.store.index.ntotal, 1)
        self.assertFalse(self.store.add_case(self.vector, {"filename": "ref_one", "diseases": ["Effusion"]}))
        self.assertEqual(self.store.index.ntotal, 1)

    def test_unpublished_write_is_invisible_and_restart_uses_committed_pair(self):
        self.store.add_case(self.vector, {"filename": "ref_one", "diseases": ["Pneumonia"]})
        with patch.object(self.store, "_publish", side_effect=OSError("controlled disk failure")):
            with self.assertRaises(OSError):
                self.store.add_case(self.vector, {"filename": "ref_two", "diseases": []})
        self.assertEqual(self.store.index.ntotal, 1)
        self.assertEqual(self.store.search(self.vector)[0]["filename"], "ref_one")
        self.store.close()
        self.store = ReferenceStore(self.finder, self.root / "published")
        self.assertEqual(self.store.index.ntotal, 1)
        self.assertEqual(len(self.store.metadata), 1)

    def test_corrupt_latest_generation_falls_back_to_previous(self):
        self.store.add_case(self.vector, {"filename": "ref_one", "diseases": []})
        self.store.add_case(self.vector, {"filename": "ref_two", "diseases": []})
        manifest = json.loads((self.root / "published/current.json").read_text())
        (self.root / "published" / manifest["generation"] / "metadata.json").write_text("broken")
        self.store.close()
        self.store = ReferenceStore(self.finder, self.root / "published")
        self.assertEqual(self.store.index.ntotal, 1)
        self.assertEqual(self.store.metadata[0]["filename"], "ref_one")

    def test_second_writer_cannot_open_same_store(self):
        with self.assertRaises(RuntimeError):
            ReferenceStore(self.finder, self.root / "published")

    def test_concurrent_replays_publish_one_vector_and_keep_two_generations(self):
        from concurrent.futures import ThreadPoolExecutor
        metadata = {"filename": "ref_one", "diseases": []}
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(lambda _: self.store.add_case(self.vector, metadata), range(2)))
        self.assertEqual(results, [True, True])
        self.assertEqual(self.store.index.ntotal, 1)
        self.store.add_case(self.vector, {"filename": "ref_two", "diseases": []})
        self.store.add_case(self.vector, {"filename": "ref_three", "diseases": []})
        folders = [p for p in (self.root / "published").iterdir() if p.is_dir()]
        self.assertEqual(len(folders), 2)
        self.assertEqual(self.store.index.ntotal, 3)


if __name__ == "__main__":
    unittest.main()
