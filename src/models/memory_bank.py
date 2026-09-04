"""
Speaker memory-bank subsystem used for online personalization.

Corresponds to notebook Cell 21 ("Speaker Memory Bank subsystem"):
``MemoryBank``, ``MemoryQualityController``, ``ConfidenceGate``,
``MemoryAgingManager``, ``PrototypeMemoryManager``, ``AdaptiveMemoryRetriever``,
``MemoryUpdateController``.
"""

import sqlite3
import time

import faiss
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from src import config


class MemoryBank:
    """SQLite-backed store of per-speaker embedding rows, with an in-memory
    FAISS index per speaker for fast similarity search."""

    def __init__(self, db_path=":memory:", embedding_dim=config.GENERAL_EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
        self._speaker_cache = {}

    def _init_schema(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker TEXT NOT NULL,
                embedding BLOB NOT NULL,
                confidence REAL NOT NULL,
                age_weight REAL NOT NULL DEFAULT 1.0,
                physics BLOB NOT NULL,
                cluster_size INTEGER NOT NULL DEFAULT 1,
                inserted_at REAL NOT NULL,
                last_touched_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def reset(self):
        self.conn.execute("DELETE FROM memory")
        self.conn.commit()
        self._speaker_cache = {}

    def insert(self, speaker, embedding, confidence, physics, cluster_size=1, now=None):
        now = now if now is not None else time.time()
        emb_blob = np.asarray(embedding, dtype=np.float32).tobytes()
        phys_blob = np.asarray(physics, dtype=np.float32).tobytes()
        cur = self.conn.execute(
            "INSERT INTO memory (speaker, embedding, confidence, age_weight, physics, "
            "cluster_size, inserted_at, last_touched_at) VALUES (?,?,?,?,?,?,?,?)",
            (speaker, emb_blob, float(confidence), 1.0, phys_blob, int(cluster_size), now, now),
        )
        self.conn.commit()
        self._mark_dirty(speaker)
        return cur.lastrowid

    def replace_speaker_rows(self, speaker, rows):
        self.conn.execute("DELETE FROM memory WHERE speaker = ?", (speaker,))
        for r in rows:
            self.conn.execute(
                "INSERT INTO memory (speaker, embedding, confidence, age_weight, physics, "
                "cluster_size, inserted_at, last_touched_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    speaker,
                    np.asarray(r["embedding"], dtype=np.float32).tobytes(),
                    float(r["confidence"]),
                    float(r["age_weight"]),
                    np.asarray(r["physics"], dtype=np.float32).tobytes(),
                    int(r["cluster_size"]),
                    float(r["inserted_at"]),
                    float(r["last_touched_at"]),
                ),
            )
        self.conn.commit()
        self._mark_dirty(speaker)

    def update_age_weights(self, speaker, id_to_weight):
        for row_id, w in id_to_weight.items():
            self.conn.execute("UPDATE memory SET age_weight = ? WHERE id = ?", (float(w), int(row_id)))
        self.conn.commit()
        self._mark_dirty(speaker)

    def get_speaker_rows(self, speaker):
        cur = self.conn.execute(
            "SELECT id, embedding, confidence, age_weight, physics, cluster_size, inserted_at, last_touched_at "
            "FROM memory WHERE speaker = ?",
            (speaker,),
        )
        rows = []
        for (rid, emb_blob, conf, age_w, phys_blob, cs, ins_at, touch_at) in cur.fetchall():
            rows.append(
                {
                    "id": rid,
                    "embedding": np.frombuffer(emb_blob, dtype=np.float32).copy(),
                    "confidence": conf,
                    "age_weight": age_w,
                    "physics": np.frombuffer(phys_blob, dtype=np.float32).copy(),
                    "cluster_size": cs,
                    "inserted_at": ins_at,
                    "last_touched_at": touch_at,
                }
            )
        return rows

    def count(self, speaker):
        cur = self.conn.execute("SELECT COUNT(*) FROM memory WHERE speaker = ?", (speaker,))
        return cur.fetchone()[0]

    def _mark_dirty(self, speaker):
        entry = self._speaker_cache.setdefault(speaker, {"index": None, "row_ids": [], "dirty": True})
        entry["dirty"] = True

    def rebuild_index(self, speaker):
        rows = self.get_speaker_rows(speaker)
        if len(rows) == 0:
            self._speaker_cache[speaker] = {"index": None, "row_ids": [], "dirty": False}
            return
        embs = np.stack([r["embedding"] for r in rows]).astype(np.float32)
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embs)
        self._speaker_cache[speaker] = {
            "index": index,
            "row_ids": [r["id"] for r in rows],
            "rows": rows,
            "dirty": False,
        }

    def rebuild_all_dirty(self):
        for speaker, entry in list(self._speaker_cache.items()):
            if entry.get("dirty", True):
                self.rebuild_index(speaker)

    def search(self, speaker, query_embedding, k):
        entry = self._speaker_cache.get(speaker)
        if entry is None or entry.get("dirty", True):
            self.rebuild_index(speaker)
            entry = self._speaker_cache.get(speaker)
        if entry is None or entry["index"] is None or entry["index"].ntotal == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(q)
        k = min(k, entry["index"].ntotal)
        sims, idxs = entry["index"].search(q, k)
        results = []
        for sim, i in zip(sims[0], idxs[0]):
            row = entry["rows"][i]
            results.append({**row, "similarity": float(sim)})
        return results


class MemoryQualityController:
    """Scores a candidate embedding against a speaker's existing memory
    (physics-vector cosine similarity + embedding cosine similarity)."""

    def __init__(self, memory_bank: MemoryBank):
        self.bank = memory_bank

    def score(self, speaker, candidate_embedding, candidate_physics, general_confidence):
        rows = self.bank.get_speaker_rows(speaker)
        if len(rows) == 0:
            return {"confidence": general_confidence, "physics_score": 1.0, "similarity": 1.0}

        physics_hist = np.stack([r["physics"] for r in rows])
        physics_mean = physics_hist.mean(axis=0)
        physics_score = self._cosine(candidate_physics, physics_mean)

        emb_hist = np.stack([r["embedding"] for r in rows])
        emb_mean = emb_hist.mean(axis=0)
        similarity = self._cosine(candidate_embedding, emb_mean)

        return {"confidence": general_confidence, "physics_score": float(physics_score), "similarity": float(similarity)}

    @staticmethod
    def _cosine(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)


class ConfidenceGate:
    """Decides whether a candidate embedding is admitted into memory."""

    def __init__(
        self,
        bootstrap_confidence=config.MEMORY_BOOTSTRAP_CONFIDENCE,
        confidence=config.MEMORY_CONFIDENCE_THRESH,
        physics_thresh=config.MEMORY_PHYSICS_THRESH,
        similarity_thresh=config.MEMORY_SIMILARITY_THRESH,
    ):
        self.bootstrap_confidence = bootstrap_confidence
        self.confidence = confidence
        self.physics_thresh = physics_thresh
        self.similarity_thresh = similarity_thresh

    def admit(self, is_bootstrap, quality_scores):
        if is_bootstrap:
            return quality_scores["confidence"] >= self.bootstrap_confidence
        return (
            quality_scores["confidence"] >= self.confidence
            and quality_scores["physics_score"] >= self.physics_thresh
            and quality_scores["similarity"] >= self.similarity_thresh
        )


class MemoryAgingManager:
    """Applies exponential decay to memory row weights based on age."""

    def __init__(
        self,
        decay_rate=config.MEMORY_DECAY_RATE,
        floor_weight=config.MEMORY_FLOOR_WEIGHT,
        seconds_per_year=None,
    ):
        self.decay_rate = decay_rate
        self.floor_weight = floor_weight
        self.seconds_per_year = (
            seconds_per_year if seconds_per_year is not None else config.MEMORY_SECONDS_PER_YEAR
        )

    def refresh(self, memory_bank: MemoryBank, speaker, now):
        rows = memory_bank.get_speaker_rows(speaker)
        if not rows:
            return
        updates = {}
        for r in rows:
            age_years = (now - r["inserted_at"]) / self.seconds_per_year
            weight = max(self.floor_weight, (1 - self.decay_rate) ** age_years)
            updates[r["id"]] = weight
        memory_bank.update_age_weights(speaker, updates)


class PrototypeMemoryManager:
    """Compresses a speaker's memory rows into weighted K-means prototypes
    once the row count exceeds ``trigger_size``."""

    def __init__(
        self,
        trigger_size=config.MEMORY_PROTOTYPE_TRIGGER_SIZE,
        n_prototypes=config.MEMORY_PROTOTYPE_N,
        minibatch_threshold=config.MEMORY_PROTOTYPE_MINIBATCH_THRESHOLD,
    ):
        self.trigger_size = trigger_size
        self.n_prototypes = n_prototypes
        self.minibatch_threshold = minibatch_threshold

    def maybe_compress(self, memory_bank: MemoryBank, speaker, now):
        rows = memory_bank.get_speaker_rows(speaker)
        if len(rows) <= self.trigger_size:
            return False

        embs = np.stack([r["embedding"] for r in rows]).astype(np.float32)
        weights = np.array([r["age_weight"] for r in rows], dtype=np.float32)
        physics = np.stack([r["physics"] for r in rows]).astype(np.float32)
        confidences = np.array([r["confidence"] for r in rows], dtype=np.float32)
        cluster_sizes = np.array([r["cluster_size"] for r in rows], dtype=np.int64)

        n_clusters = min(self.n_prototypes, len(rows))
        KMeansCls = MiniBatchKMeans if len(rows) > self.minibatch_threshold else KMeans
        km = KMeansCls(n_clusters=n_clusters, n_init=10 if KMeansCls is KMeans else 3, random_state=0)
        labels = km.fit_predict(embs, sample_weight=weights)

        new_rows = []
        for c in range(n_clusters):
            mask = labels == c
            if mask.sum() == 0:
                continue
            w = weights[mask]
            w_sum = w.sum() + 1e-8
            centroid = (embs[mask] * w[:, None]).sum(axis=0) / w_sum
            physics_centroid = (physics[mask] * w[:, None]).sum(axis=0) / w_sum
            conf_centroid = float((confidences[mask] * w).sum() / w_sum)
            total_members = int(cluster_sizes[mask].sum())
            new_rows.append(
                {
                    "embedding": centroid,
                    "confidence": conf_centroid,
                    "age_weight": float(w.mean()),
                    "physics": physics_centroid,
                    "cluster_size": total_members,
                    "inserted_at": now,
                    "last_touched_at": now,
                }
            )

        memory_bank.replace_speaker_rows(speaker, new_rows)
        return True


class AdaptiveMemoryRetriever:
    """Chooses how many memory rows (``top_k``) to retrieve based on how
    scattered/disagreeing the speaker's existing memory is with the query."""

    def __init__(
        self,
        memory_bank: MemoryBank,
        k_min=config.MEMORY_RETRIEVER_K_MIN,
        k_max=config.MEMORY_RETRIEVER_K_MAX,
    ):
        self.bank = memory_bank
        self.k_min = k_min
        self.k_max = k_max

    def complexity(self, speaker, query_embedding):
        rows = self.bank.get_speaker_rows(speaker)
        if len(rows) < 2:
            return 0.0
        embs = np.stack([r["embedding"] for r in rows]).astype(np.float32)
        embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        centroid = embs_n.mean(axis=0)
        spread = float(np.mean(1 - embs_n @ centroid))
        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        disagreement = float(np.mean(1 - embs_n @ q))
        raw = 0.5 * spread + 0.5 * disagreement
        return float(np.clip(raw, 0.0, 1.0))

    def retrieve(self, speaker, query_embedding):
        complexity = self.complexity(speaker, query_embedding)
        top_k = int(round(self.k_min + complexity * (self.k_max - self.k_min)))
        top_k = int(np.clip(top_k, self.k_min, self.k_max))
        results = self.bank.search(speaker, query_embedding, top_k)
        return results, complexity, top_k


class MemoryUpdateController:
    """Orchestrates quality scoring, gating, aging, prototype compression
    and periodic re-indexing for a single write."""

    def __init__(
        self,
        memory_bank,
        quality_controller,
        confidence_gate,
        aging_manager,
        prototype_manager,
        reindex_every=config.PERSONALIZER_REINDEX_EVERY,
    ):
        self.bank = memory_bank
        self.quality = quality_controller
        self.gate = confidence_gate
        self.aging = aging_manager
        self.prototype = prototype_manager
        self.reindex_every = reindex_every
        self._since_reindex = 0

    def maybe_write(self, speaker, embedding, physics, general_confidence, now=None):
        now = now if now is not None else time.time()
        is_bootstrap = self.bank.count(speaker) == 0
        scores = self.quality.score(speaker, embedding, physics, general_confidence)
        accepted = self.gate.admit(is_bootstrap, scores)

        if accepted:
            self.bank.insert(speaker, embedding, general_confidence, physics, now=now)
            self.aging.refresh(self.bank, speaker, now)
            self.prototype.maybe_compress(self.bank, speaker, now)
            self._since_reindex += 1
            if self._since_reindex >= self.reindex_every:
                self.bank.rebuild_all_dirty()
                self._since_reindex = 0

        return accepted, scores

    def flush(self):
        self.bank.rebuild_all_dirty()
        self._since_reindex = 0
