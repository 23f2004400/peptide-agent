"""
Run this ONCE (e.g. on the college server) to pre-build the FAISS index.
After this, the agent loads the index instantly at startup.

Usage (from project root):
    python -m backend.rag.build_index
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.rag.peptide_retriever import PeptideRetriever, INDEX_PATH

if __name__ == "__main__":
    print("Building peptide retrieval index...")
    r = PeptideRetriever()
    if r.ready:
        print(f"Done. Index ready with {r._index.ntotal} peptides.")
        print(f"Index saved to {INDEX_PATH}")
    else:
        print("Failed. Check dataset path and try again.")
