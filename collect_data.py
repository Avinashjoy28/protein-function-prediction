"""
Protein Function Prediction — Data Collection Script.

Downloads reviewed protein sequences from UniProt/Swiss-Prot along with
their Gene Ontology (GO) functional annotations via the UniProt REST API.

Usage:
    python collect_data.py
    python collect_data.py --max_proteins 5000
    python collect_data.py --organism 9606  (Human only)

Output CSV Datasets (saved to data/raw/):
    1. proteins.csv           — Master protein table (ID, name, organism, sequence, length)
    2. go_annotations.csv     — Protein → GO term mapping (long format)
    3. protein_go_summary.csv — Per-protein summary (GO count, GO terms list)
    4. go_term_statistics.csv — Per-GO-term frequency statistics
    5. sequence_statistics.csv — Sequence composition & length statistics
"""

import os
import sys
import time
import argparse
import requests
import numpy as np
import pandas as pd
from io import StringIO
from collections import Counter
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "data", "raw")
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"

DEFAULT_MAX_PROTEINS = 2000
DEFAULT_MIN_LENGTH = 50
DEFAULT_MAX_LENGTH = 1000


# ─────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────
def build_query(min_len, max_len, organism=""):
    """Build a UniProt query string."""
    parts = [
        "(reviewed:true)",
        f"(length:[{min_len} TO {max_len}])",
        "(go:*)",
    ]
    if organism:
        parts.append(f"(organism_id:{organism})")
    return " AND ".join(parts)


def fetch_page(query, fmt, fields, size, cursor=None, retries=3):
    """Fetch one page of results from UniProt REST API."""
    params = {"query": query, "format": fmt, "size": size}
    if fields:
        params["fields"] = fields
    if cursor:
        params["cursor"] = cursor

    for attempt in range(retries):
        try:
            resp = requests.get(
                f"{UNIPROT_API}/search", params=params, timeout=120
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"  ⚠ Attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch data after {retries} retries")


def get_next_cursor(response):
    """Extract pagination cursor from UniProt Link header."""
    link = response.headers.get("Link", "")
    if 'rel="next"' in link:
        for part in link.split("&"):
            if "cursor=" in part:
                return part.split("cursor=")[1].split("&")[0].rstrip(">")
    return None


# ─────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────
def download_annotations(max_proteins, min_len, max_len, organism):
    """Download protein GO annotations from UniProt."""
    query = build_query(min_len, max_len, organism)
    fields = "accession,protein_name,go_id,organism_name,sequence,length"

    print(f"📡 Query : {query}")
    print(f"📥 Target: {max_proteins} proteins\n")

    all_rows = []
    page_size = min(500, max_proteins)
    cursor = None
    total_fetched = 0

    pbar = tqdm(total=max_proteins, desc="Downloading", unit="proteins")

    while total_fetched < max_proteins:
        remaining = max_proteins - total_fetched
        current_size = min(page_size, remaining)

        resp = fetch_page(query, "tsv", fields, current_size, cursor)
        text = resp.text.strip()
        if not text:
            break

        lines = text.split("\n")
        if total_fetched == 0:
            all_rows.append(lines[0])   # header
            all_rows.extend(lines[1:])
        else:
            all_rows.extend(lines[1:])  # skip header on subsequent pages

        new_rows = len(lines) - 1
        total_fetched += new_rows
        pbar.update(new_rows)

        cursor = get_next_cursor(resp)
        if cursor is None or new_rows == 0:
            break
        time.sleep(0.5)

    pbar.close()

    tsv_text = "\n".join(all_rows)
    df = pd.read_csv(StringIO(tsv_text), sep="\t")
    print(f"\n✅ Downloaded {len(df)} rows for {df['Entry'].nunique()} unique proteins")
    return df


# ─────────────────────────────────────────────────────────
# Process & Save 5 CSV datasets
# ─────────────────────────────────────────────────────────
def compute_amino_acid_composition(seq):
    """Compute % of common amino acid groups."""
    seq = seq.upper()
    n = max(len(seq), 1)
    hydrophobic = sum(seq.count(a) for a in "AILMFWVP")
    charged = sum(seq.count(a) for a in "DEKRH")
    polar = sum(seq.count(a) for a in "STNQYC")
    return {
        "pct_hydrophobic": round(hydrophobic / n * 100, 2),
        "pct_charged": round(charged / n * 100, 2),
        "pct_polar": round(polar / n * 100, 2),
    }


def process_and_save(df):
    """Parse downloaded data → save 5 CSV datasets."""
    os.makedirs(RAW_DIR, exist_ok=True)

    # ─────────────────────────────────────────────────────
    # DATASET 1: proteins.csv — Master protein table
    # ─────────────────────────────────────────────────────
    protein_rows = []
    for entry_id, group in df.groupby("Entry"):
        row = group.iloc[0]
        seq = str(row.get("Sequence", ""))
        if pd.isna(seq) or len(seq.strip()) == 0:
            continue
        protein_rows.append({
            "protein_id": entry_id,
            "protein_name": str(row.get("Protein names", "Unknown")),
            "organism": str(row.get("Organism", "Unknown")),
            "sequence": seq,
            "length": int(row.get("Length", len(seq))),
        })

    proteins_df = pd.DataFrame(protein_rows)
    proteins_path = os.path.join(RAW_DIR, "proteins.csv")
    proteins_df.to_csv(proteins_path, index=False)
    print(f"💾 Dataset 1 → {proteins_path}  ({len(proteins_df):,} proteins)")

    # ─────────────────────────────────────────────────────
    # DATASET 2: go_annotations.csv — Protein→GO mappings
    # ─────────────────────────────────────────────────────
    go_records = []
    for _, row in df.iterrows():
        entry_id = row["Entry"]
        go_ids = str(row.get("Gene Ontology IDs", ""))
        seq = str(row.get("Sequence", ""))
        length = row.get("Length", len(seq) if not pd.isna(seq) else 0)

        if pd.isna(go_ids) or go_ids.strip() == "":
            continue

        for go_id in go_ids.split("; "):
            go_id = go_id.strip()
            if go_id.startswith("GO:"):
                go_records.append({
                    "protein_id": entry_id,
                    "go_id": go_id,
                    "sequence": seq,
                    "length": int(length),
                })

    go_df = pd.DataFrame(go_records)
    go_path = os.path.join(RAW_DIR, "go_annotations.csv")
    go_df.to_csv(go_path, index=False)
    print(f"💾 Dataset 2 → {go_path}  ({len(go_df):,} annotations)")

    # ─────────────────────────────────────────────────────
    # DATASET 3: protein_go_summary.csv — Per-protein GO summary
    # ─────────────────────────────────────────────────────
    summary_rows = []
    for pid, grp in go_df.groupby("protein_id"):
        go_list = sorted(grp["go_id"].unique().tolist())
        prot_row = proteins_df[proteins_df["protein_id"] == pid]
        organism = prot_row["organism"].values[0] if len(prot_row) > 0 else "Unknown"
        name = prot_row["protein_name"].values[0] if len(prot_row) > 0 else "Unknown"
        length = prot_row["length"].values[0] if len(prot_row) > 0 else 0
        summary_rows.append({
            "protein_id": pid,
            "protein_name": name,
            "organism": organism,
            "length": int(length),
            "num_go_terms": len(go_list),
            "go_terms": "; ".join(go_list),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("num_go_terms", ascending=False)
    summary_path = os.path.join(RAW_DIR, "protein_go_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Dataset 3 → {summary_path}  ({len(summary_df):,} proteins)")

    # ─────────────────────────────────────────────────────
    # DATASET 4: go_term_statistics.csv — GO term frequency
    # ─────────────────────────────────────────────────────
    go_counts = go_df.groupby("go_id").agg(
        protein_count=("protein_id", "nunique"),
        annotation_count=("protein_id", "count"),
        avg_protein_length=("length", "mean"),
        min_protein_length=("length", "min"),
        max_protein_length=("length", "max"),
    ).reset_index().sort_values("protein_count", ascending=False)

    go_counts["pct_of_proteins"] = (
        go_counts["protein_count"] / proteins_df["protein_id"].nunique() * 100
    ).round(2)
    go_counts["avg_protein_length"] = go_counts["avg_protein_length"].round(1)

    go_stats_path = os.path.join(RAW_DIR, "go_term_statistics.csv")
    go_counts.to_csv(go_stats_path, index=False)
    print(f"💾 Dataset 4 → {go_stats_path}  ({len(go_counts):,} GO terms)")

    # ─────────────────────────────────────────────────────
    # DATASET 5: sequence_statistics.csv — Sequence analysis
    # ─────────────────────────────────────────────────────
    seq_stats_rows = []
    for _, row in proteins_df.iterrows():
        seq = row["sequence"]
        comp = compute_amino_acid_composition(seq)
        aa_counts = Counter(seq.upper())
        most_common_aa = aa_counts.most_common(1)[0][0] if aa_counts else "X"
        unique_aas = len(set(seq.upper()))

        seq_stats_rows.append({
            "protein_id": row["protein_id"],
            "length": row["length"],
            "molecular_weight_approx": round(row["length"] * 110, 1),  # avg aa ~110 Da
            "unique_amino_acids": unique_aas,
            "most_common_aa": most_common_aa,
            "pct_hydrophobic": comp["pct_hydrophobic"],
            "pct_charged": comp["pct_charged"],
            "pct_polar": comp["pct_polar"],
        })

    seq_stats_df = pd.DataFrame(seq_stats_rows)
    seq_stats_path = os.path.join(RAW_DIR, "sequence_statistics.csv")
    seq_stats_df.to_csv(seq_stats_path, index=False)
    print(f"💾 Dataset 5 → {seq_stats_path}  ({len(seq_stats_df):,} proteins)")

    return go_df


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download Swiss-Prot protein sequences and GO annotations"
    )
    parser.add_argument(
        "--max_proteins", type=int, default=DEFAULT_MAX_PROTEINS,
        help=f"Max proteins to download (default: {DEFAULT_MAX_PROTEINS})",
    )
    parser.add_argument(
        "--min_length", type=int, default=DEFAULT_MIN_LENGTH,
        help=f"Min sequence length (default: {DEFAULT_MIN_LENGTH})",
    )
    parser.add_argument(
        "--max_length", type=int, default=DEFAULT_MAX_LENGTH,
        help=f"Max sequence length (default: {DEFAULT_MAX_LENGTH})",
    )
    parser.add_argument(
        "--organism", type=str, default="",
        help="UniProt organism ID filter (e.g. 9606 for Human)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PROTEIN FUNCTION PREDICTION — DATA COLLECTION")
    print("=" * 60)
    print(f"  Max proteins : {args.max_proteins}")
    print(f"  Length range  : {args.min_length}–{args.max_length} aa")
    print(f"  Organism      : {args.organism or 'All'}")
    print(f"  Output dir    : {RAW_DIR}")
    print("=" * 60 + "\n")

    raw_df = download_annotations(
        args.max_proteins, args.min_length, args.max_length, args.organism
    )
    go_df = process_and_save(raw_df)

    # ── Final summary ──
    print(f"\n{'=' * 60}")
    print("  DATA COLLECTION COMPLETE — 5 CSV DATASETS SAVED")
    print(f"{'=' * 60}")
    print(f"  📂 Output directory : {RAW_DIR}")
    print(f"  🧬 Unique proteins  : {go_df['protein_id'].nunique():,}")
    print(f"  🏷️  Unique GO terms  : {go_df['go_id'].nunique():,}")
    print(f"  📊 Total annotations: {len(go_df):,}")
    go_per_protein = go_df.groupby("protein_id")["go_id"].count()
    print(f"  📊 GO terms/protein : mean={go_per_protein.mean():.1f}, "
          f"median={go_per_protein.median():.0f}")
    print(f"\n  📋 Datasets:")
    print(f"     1. proteins.csv           — Master protein table")
    print(f"     2. go_annotations.csv     — Protein → GO mappings")
    print(f"     3. protein_go_summary.csv — Per-protein GO summary")
    print(f"     4. go_term_statistics.csv — GO term frequency stats")
    print(f"     5. sequence_statistics.csv — Sequence composition")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
