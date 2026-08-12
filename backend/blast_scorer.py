# backend/blast_scorer.py
# ─────────────────────────────────────────────────────────────────────────────
# BLAST similarity scoring for novelty assessment.
#
# Compares generated peptide against the peptide database using BLASTP.
# Similarity Score = (Identity% / 100) * (Coverage% / 100)
#
# LOW score  = novel peptide (good for research)
# HIGH score = already exists in database (not novel)
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import shutil
import subprocess
import tempfile
import pandas as pd

# Paths — DATASET_PATH shares the same env var/default as backend/rag/peptide_retriever.py
# (data/peptides.csv is the real dataset file in this repo, not peptides_dataset.csv).
DATASET_PATH = os.environ.get(
    "PEPTIDE_DATASET_PATH",
    "data/peptides.csv"
)
BLAST_DB_PATH = os.environ.get(
    "BLAST_DB_PATH",
    "data/blast_db/known_peptides_db"
)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _check_blast_available() -> bool:
    """Check if BLAST binary is installed."""
    return shutil.which('blastp') is not None


def _check_makeblastdb_available() -> bool:
    """Check if makeblastdb binary is installed."""
    return shutil.which('makeblastdb') is not None


def is_natural(peptide: str) -> bool:
    """Check if peptide contains only standard amino acids."""
    return all(aa in VALID_AA for aa in peptide.upper())


def peptides_to_fasta(peptide_list: list, keys: list = None,
                       output_file: str = None,
                       prefix: str = "seq") -> str:
    """Convert list of peptide sequences to FASTA format."""
    fasta_data = []
    for i, seq in enumerate(peptide_list):
        header = f">{keys[i]}" if keys is not None else f">{prefix}_{i+1}"
        sequence = str(seq).strip().upper()
        fasta_data.append(f"{header}\n{sequence}\n")
    fasta_string = "".join(fasta_data)
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(fasta_string)
        return output_file
    return fasta_string


def build_blast_db(
    db_path: str = BLAST_DB_PATH,
    min_len: int = 6,
    max_len: int = 50,
) -> bool:
    """
    Build BLAST database from peptide dataset CSV.
    Only needs to run once — database saved to disk.
    Returns True if successful, False otherwise.
    """
    if not _check_makeblastdb_available():
        print("[BLAST] makeblastdb not found. "
              "Install: sudo apt-get install ncbi-blast+")
        return False

    if not os.path.exists(DATASET_PATH):
        print(f"[BLAST] Dataset not found at {DATASET_PATH}")
        return False

    try:
        print(f"[BLAST] Building database from {DATASET_PATH}...")
        df = pd.read_csv(DATASET_PATH)

        # Get sequence column
        seq_col = None
        for col in df.columns:
            if 'seq' in col.lower():
                seq_col = col
                break
        if seq_col is None:
            seq_col = df.columns[0]

        peptides = df[seq_col].dropna().tolist()

        # Filter by length and valid AA
        filtered = [
            p.strip().upper() for p in peptides
            if min_len <= len(p.strip()) <= max_len
            and is_natural(p.strip())
        ]
        print(f"[BLAST] Retained {len(filtered)}/{len(peptides)} peptides")

        # Write FASTA
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        fasta_path = f"{db_path}.fasta"
        peptides_to_fasta(filtered, output_file=fasta_path, prefix="known")

        # Build BLAST DB
        cmd = (f"makeblastdb -in {fasta_path} "
               f"-dbtype prot -out {db_path}")
        subprocess.run(
            cmd, shell=True, check=True,
            capture_output=True, text=True
        )
        print(f"[BLAST] Database built at {db_path}")
        return True

    except Exception as e:
        print(f"[BLAST] Database build failed: {e}")
        return False


def _db_exists() -> bool:
    """Check if BLAST database files exist."""
    return (
        os.path.exists(f"{BLAST_DB_PATH}.phr") or
        os.path.exists(f"{BLAST_DB_PATH}.pin") or
        os.path.exists(f"{BLAST_DB_PATH}.psq")
    )


def assess_novelty(
    sequence: str,
    top_k: int = 1,
    evalue: float = 10.0,
    matrix: str = "PAM30",
    word_size: int = 2,
    task: str = "blastp-short",
) -> dict:
    """
    Assess novelty of a generated peptide via BLAST similarity.

    LOW similarity score = novel peptide (good)
    HIGH similarity score = already known peptide (not novel)

    SAFE: Always returns dict. Never raises. Never blocks generation.

    Returns:
        {
            similarity_score:   float 0-1 (top hit) or None
            all_scores:         list of top_k scores
            novelty_label:      "novel" | "low_similarity" | "similar" | "known" | "unknown"
            interpretation:     human-readable string
            blast_available:    bool
            db_exists:          bool
            error:              None or error string
        }
    """
    if not sequence or not is_natural(sequence):
        return _empty("Invalid sequence")

    if not _check_blast_available():
        return _empty(
            "BLAST not installed. "
            "Run: sudo apt-get install ncbi-blast+"
        )

    if not _db_exists():
        print("[BLAST] DB not found — building now...")
        success = build_blast_db()
        if not success:
            return _empty("Could not build BLAST database")

    query_path = None
    out_path = None
    try:
        # Write query to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.fasta',
            delete=False, prefix='pepforge_blast_query_'
        ) as qf:
            qf.write(f">query\n{sequence.upper()}\n")
            query_path = qf.name

        # Run BLASTP
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.tsv',
            delete=False, prefix='pepforge_blast_out_'
        ) as of:
            out_path = of.name

        outfmt = (
            "6 qseqid sseqid pident length "
            "qcovs evalue bitscore"
        )
        cmd = (
            f"blastp -query {query_path} "
            f"-db {BLAST_DB_PATH} "
            f"-task {task} "
            f"-matrix {matrix} "
            f"-word_size {word_size} "
            f"-evalue {evalue} "
            f"-outfmt \"{outfmt}\" "
            f"-out {out_path} "
            f"-num_threads 1"
        )

        subprocess.run(
            cmd, shell=True, check=True,
            capture_output=True, text=True,
            timeout=30
        )

        # Parse results
        scores = []
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            cols = [
                "qseqid", "sseqid", "pident",
                "length", "qcovs", "evalue", "bitscore"
            ]
            df = pd.read_csv(out_path, sep="\t", names=cols)
            df["similarity_score"] = (
                (df["pident"] / 100.0) * (df["qcovs"] / 100.0)
            )
            df_sorted = df.sort_values(
                "similarity_score", ascending=False
            )
            all_scores = df_sorted["similarity_score"].round(4).tolist()
            scores = all_scores[:top_k] if top_k > 0 else all_scores

        top_score = scores[0] if scores else 0.0

        # Novelty label
        if not scores:
            label = "novel"
            interp = "No match found in database — completely novel peptide"
        elif top_score >= 0.95:
            label = "known"
            interp = (
                f"Very high similarity ({top_score:.2f}) — "
                f"nearly identical to a known peptide in database"
            )
        elif top_score >= 0.70:
            label = "similar"
            interp = (
                f"Moderate similarity ({top_score:.2f}) — "
                f"structurally related to known peptide"
            )
        elif top_score >= 0.40:
            label = "low_similarity"
            interp = (
                f"Low similarity ({top_score:.2f}) — "
                f"distant relative of known peptide"
            )
        else:
            label = "novel"
            interp = (
                f"Very low similarity ({top_score:.2f}) — "
                f"novel peptide with minimal database overlap"
            )

        return {
            "similarity_score": round(top_score, 4),
            "all_scores": scores,
            "novelty_label": label,
            "interpretation": interp,
            "blast_available": True,
            "db_exists": True,
            "error": None,
        }

    except subprocess.TimeoutExpired:
        return _empty("BLAST timed out after 30 seconds")
    except Exception as e:
        return _empty(str(e))
    finally:
        for path in (query_path, out_path):
            if path:
                try:
                    os.unlink(path)
                except Exception:
                    pass


def _empty(error: str) -> dict:
    return {
        "similarity_score": None,
        "all_scores": [],
        "novelty_label": "unknown",
        "interpretation": error,
        "blast_available": _check_blast_available(),
        "db_exists": _db_exists(),
        "error": error,
    }


if __name__ == "__main__":
    print(f"BLAST available:    {_check_blast_available()}")
    print(f"makeblastdb:        {_check_makeblastdb_available()}")
    print(f"DB exists:          {_db_exists()}")

    if not _db_exists():
        print("Building DB...")
        build_blast_db()

    result = assess_novelty("GIGKFLHSAKKFGKAFVGEIMNS")
    print(f"Test sequence similarity: {result['similarity_score']}")
    print(f"Novelty: {result['novelty_label']}")
    print(f"Interpretation: {result['interpretation']}")
