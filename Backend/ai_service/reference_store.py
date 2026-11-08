"""Publish matched FAISS/index metadata generations around the original add_case algorithm."""
import copy
import json
import os
from pathlib import Path
import re
import shutil
import threading
import uuid

import faiss


class ReferenceStore:
    def __init__(self, finder, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self._writer = (self.root / "writer.lock").open("a+b")
        if (self.root / "writer.lock").stat().st_size == 0:
            self._writer.write(b"0")
            self._writer.flush()
        self._writer.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._writer.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._writer.close()
            raise RuntimeError("A writer already owns this CBIR store")
        self._current = finder
        self.generation = None
        self.previous = None
        try:
            self._restore(finder)
            self._validate(self._current)
        except Exception:
            self.close()
            raise

    def close(self):
        if self._writer.closed:
            return
        self._writer.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self._writer.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self._writer.fileno(), fcntl.LOCK_UN)
        self._writer.close()

    @staticmethod
    def _validate(finder):
        if not finder.available or finder.index is None or finder.metadata is None:
            raise RuntimeError("CBIR finder is unavailable")
        if finder.index.ntotal != len(finder.metadata):
            raise RuntimeError("FAISS count does not match metadata")
        keys = [m.get("filename", "") for m in finder.metadata if m.get("filename", "").startswith("ref_")]
        if len(keys) != len(set(keys)):
            raise RuntimeError("Duplicate reference identifiers")

    def _load(self, finder, generation):
        if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise RuntimeError("Invalid CBIR generation")
        folder = self.root / generation
        restored = copy.copy(finder)
        restored.index_path = str(folder / "index.faiss")
        restored.metadata_path = str(folder / "metadata.json")
        restored.index = faiss.read_index(restored.index_path)
        restored.metadata = json.loads(Path(restored.metadata_path).read_text(encoding="utf-8"))
        restored.available = True
        self._validate(restored)
        return restored

    def _restore(self, finder):
        manifest = self.root / "current.json"
        if not manifest.exists():
            return
        for candidate in (manifest, self.root / "previous.json"):
            if not candidate.exists():
                continue
            try:
                info = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for generation in (info.get("generation"), info.get("previous")):
                if generation is None:
                    continue
                try:
                    self._current = self._load(finder, generation)
                    self.generation = generation
                    return
                except (OSError, ValueError, RuntimeError, TypeError):
                    continue
        raise RuntimeError("No valid published CBIR generation remains")

    @property
    def available(self):
        return self._current.available

    @property
    def index(self):
        return self._current.index

    @property
    def metadata(self):
        return self._current.metadata

    def search(self, vector, k=5):
        snapshot = self._current
        return snapshot.search(vector, k=k)

    def reference(self, reference_id):
        return next((m for m in self._current.metadata if m.get("filename") == reference_id), None)

    def _publish(self, generation):
        current = self.root / "current.json"
        if current.exists():
            backup = self.root / "previous.tmp"
            backup.write_bytes(current.read_bytes())
            os.replace(backup, self.root / "previous.json")
        pending = self.root / "current.tmp"
        with pending.open("w", encoding="utf-8") as stream:
            json.dump({"generation": generation, "previous": self.generation}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, current)

    def add_case(self, embedding_vector, metadata):
        with self._mutex:
            if self._writer.closed:
                return False
            key = metadata.get("filename", "")
            if key.startswith("ref_"):
                existing = self.reference(key)
                if existing:
                    return sorted(existing.get("diseases", [])) == sorted(metadata.get("diseases", []))
            generation = uuid.uuid4().hex
            folder = self.root / generation
            folder.mkdir()
            staged = copy.copy(self._current)
            staged.index = faiss.clone_index(self._current.index)
            staged.metadata = copy.deepcopy(self._current.metadata)
            staged.index_path = str(folder / "index.faiss")
            staged.metadata_path = str(folder / "metadata.json")
            # Reuse normalization, dimension handling and persistence in the original algorithm.
            if not staged.add_case(embedding_vector, metadata):
                return False
            self._validate(staged)
            for name in ("index.faiss", "metadata.json"):
                with (folder / name).open("r+b") as stream:
                    os.fsync(stream.fileno())
            self._publish(generation)
            self.previous = self.generation
            self.generation = generation
            self._current = staged
            # Keep a recoverable previous pair while bounding disk use.
            for child in self.root.iterdir():
                if child.name in {self.generation, self.previous}:
                    continue
                if re.fullmatch(r"[0-9a-f]{32}", child.name) and child.is_dir() and child.resolve().parent == self.root:
                    try:
                        shutil.rmtree(child)
                    except OSError:
                        pass  # An old snapshot can be cleaned after the next successful publication.
            return True
