"""
Searches the local peptide dataset for sequences similar to task properties.
Uses FAISS for fast vector similarity search.
SAFE: fails silently if the dataset/faiss/numpy/pandas aren't available.
Never blocks generation.
"""

from __future__ import annotations
import os
import pickle

try:
    import numpy as np
    import pandas as pd
    _DEPS_OK = True
except ImportError:
    np = None
    pd = None
    _DEPS_OK = False

# Dataset path - can be overridden by env var. Real on-disk file in this repo
# is data/peptides.csv (sequence + 13 binary activity columns), not the
# "peptides_dataset.csv" name assumed elsewhere - corrected here.
DATASET_PATH = os.environ.get(
    "PEPTIDE_DATASET_PATH",
    "data/peptides.csv"
)
INDEX_PATH = os.environ.get(
    "PEPTIDE_INDEX_PATH",
    "data/peptide_index.faiss"
)
META_PATH = os.environ.get(
    "PEPTIDE_META_PATH",
    "data/peptide_index_meta.pkl"
)

# Activity columns in the dataset
ACTIVITY_COLS = [
    "anti-bacterial", "anti-cancer", "anti-fungal", "anti-parasitic",
    "anti-viral", "cell-cell-communication", "drug-delivery",
    "immunological", "inhibitor", "metabolic", "other-functional",
    "signal-peptide", "toxic",
]

CHARGE_AA  = {'K': 1, 'R': 1, 'D': -1, 'E': -1}
HYDRO_AA   = set('LIVFWMAYC')


def _compute_properties(sequence: str) -> tuple:
    """Compute charge and hydrophobicity from sequence using same logic as rulebook."""
    seq    = sequence.upper()
    charge = sum(CHARGE_AA.get(aa, 0) for aa in seq)
    hydro  = 100 * sum(1 for aa in seq if aa in HYDRO_AA) / max(len(seq), 1)
    return charge, hydro


def _row_to_vector(row) -> "np.ndarray":
    """
    Convert a dataset row to a searchable feature vector.
    Properties computed from sequence since dataset has no property columns.
    """
    seq    = str(row.get('sequence', ''))
    length = len(seq)
    charge, hydro = _compute_properties(seq)

    # Normalise continuous features
    features = [
        charge / 10.0,        # charge: typical range -5 to +10
        hydro  / 100.0,       # hydrophobicity: 0-100%
        length / 50.0,        # length: typical range 5-50
    ]

    # Binary activity flags
    for col in ACTIVITY_COLS:
        features.append(float(row.get(col, 0)))

    return np.array(features, dtype=np.float32)


def _task_to_vector(task: dict) -> "np.ndarray":
    """Convert a generation task to the same feature vector format."""
    charge     = task.get('charge', 0)
    hydro_min  = task.get('hydro_min', 35)
    hydro_max  = task.get('hydro_max', 55)
    hydro      = (hydro_min + hydro_max) / 2.0
    length     = task.get('length', 15)
    activities = task.get('activities', [])

    features = [
        charge / 10.0,
        hydro  / 100.0,
        length / 50.0,
    ]

    for col in ACTIVITY_COLS:
        features.append(1.0 if col in activities else 0.0)

    return np.array([features], dtype=np.float32)


class PeptideRetriever:
    """
    Searches peptide dataset for sequences matching task properties.
    Loads pre-built FAISS index if available, builds it otherwise.
    Fails gracefully if dataset/dependencies aren't available.
    """

    def __init__(self):
        self._index    = None
        self._metadata = None   # list of dicts (sequence + activity info)
        self._ready    = False
        self._load()

    def _load(self):
        """Load pre-built index or build from scratch."""
        if not _DEPS_OK:
            print("[RAG] numpy/pandas not installed. RAG disabled.")
            return

        # Try loading pre-built index first (fast)
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            try:
                import faiss
                self._index    = faiss.read_index(INDEX_PATH)
                with open(META_PATH, 'rb') as f:
                    self._metadata = pickle.load(f)
                self._ready = True
                print(f"[RAG] Loaded index: {self._index.ntotal} peptides")
                return
            except Exception as e:
                print(f"[RAG] Could not load pre-built index: {e}")

        # Build from scratch
        self._build()

    def _build(self):
        """Build FAISS index from CSV dataset."""
        if not os.path.exists(DATASET_PATH):
            print(f"[RAG] Dataset not found at {DATASET_PATH}. RAG disabled.")
            return

        try:
            import faiss
            print(f"[RAG] Building index from {DATASET_PATH}...")
            df = pd.read_csv(DATASET_PATH)

            vectors  = []
            metadata = []

            for _, row in df.iterrows():
                seq = str(row.get('sequence', ''))
                if not seq or len(seq) < 4:
                    continue
                try:
                    vec = _row_to_vector(row)
                    vectors.append(vec)
                    charge, hydro = _compute_properties(seq)
                    metadata.append({
                        'sequence':   seq,
                        'length':     len(seq),
                        'charge':     charge,
                        'hydro_pct':  round(hydro, 1),
                        'activities': [
                            col for col in ACTIVITY_COLS
                            if row.get(col, 0) == 1
                        ],
                    })
                except Exception:
                    continue

            if not vectors:
                print("[RAG] No valid vectors built. RAG disabled.")
                return

            mat = np.array(vectors, dtype=np.float32)
            dim = mat.shape[1]

            # Use L2 index - simple and effective for this feature space
            self._index    = faiss.IndexFlatL2(dim)
            self._index.add(mat)
            self._metadata = metadata
            self._ready    = True

            # Save for next time
            try:
                os.makedirs(os.path.dirname(INDEX_PATH) or '.', exist_ok=True)
                faiss.write_index(self._index, INDEX_PATH)
                with open(META_PATH, 'wb') as f:
                    pickle.dump(metadata, f)
                print(f"[RAG] Index saved: {len(metadata)} peptides, dim={dim}")
            except Exception as e:
                print(f"[RAG] Could not save index: {e}")

        except ImportError:
            print("[RAG] faiss not installed. Run: pip install faiss-cpu")
        except Exception as e:
            print(f"[RAG] Index build failed: {e}. RAG disabled.")

    def search(self, task: dict, top_k: int = 5) -> list:
        """
        Find top-k most similar peptides to the task.
        Returns list of dicts. Returns [] if RAG not ready.
        NEVER raises exceptions.
        """
        if not self._ready or self._index is None:
            return []

        try:
            query = _task_to_vector(task)
            k     = min(top_k, self._index.ntotal)
            _, indices = self._index.search(query, k)

            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self._metadata):
                    results.append(self._metadata[idx])
            return results

        except Exception as e:
            print(f"[RAG] Search failed: {e}")
            return []

    @property
    def ready(self) -> bool:
        return self._ready


# Singleton instance - loaded once at server start
_retriever = None

def get_retriever() -> PeptideRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PeptideRetriever()
    return _retriever


def search_similar_peptides(task: dict, top_k: int = 5) -> list:
    """Convenience function for agent.py to call."""
    return get_retriever().search(task, top_k=top_k)
