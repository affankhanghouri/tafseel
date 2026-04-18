"""
ingest_all.py
-------------
Run this script to ingest all .txt files in the data/ folder.
Safe to re-run — already ingested chunks are skipped automatically.

Usage:
    python ingest_all.py
    python ingest_all.py --data-dir data
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from src.ingestion import ingest_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Folder containing .txt files")
    args = parser.parse_args()

    ingest_all(data_dir=args.data_dir)