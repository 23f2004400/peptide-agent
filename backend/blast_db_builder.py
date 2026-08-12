# backend/blast_db_builder.py
# Run once: python backend/blast_db_builder.py
# Builds BLAST database from data/peptides.csv

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blast_scorer import build_blast_db

if __name__ == "__main__":
    print("Building BLAST database from peptide dataset...")
    success = build_blast_db()
    if success:
        print("Done. Database ready at data/blast_db/")
    else:
        print("Failed. Check dataset path and BLAST installation.")
