"""
PharmaBot — Data Cleaning & Preprocessing Pipeline

Handles:
  1. Loading all 4 datasets (2 workspace CSVs + 2 Kaggle CSVs)
  2. Column mapping to unified schema
  3. Text cleaning (HTML, special chars, whitespace)
  4. Parsing rich drug_content text into structured fields
  5. Normalization (names, categories, booleans)
  6. Deduplication & field-level merging by medicine name
  7. Disease ↔ Medicine linking
  8. Export to clean Parquet / CSV for ingestion

Usage:
    cd backend
    python data_cleaning.py --kaggle-az ../data/kaggle_az_india.csv \
                            --kaggle-med ../data/kaggle_medicines.csv \
                            --output-dir ../data/cleaned
"""

import re
import sys
import html
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_schema import (
    MEDICINE_SCHEMA_FIELDS,
    DISEASE_SCHEMA_FIELDS,
    WORKSPACE_MED_MAP,
    WORKSPACE_DISEASE_MAP,
    KAGGLE_AZ_MAP,
    KAGGLE_MED_MAP,
    SOURCE_PRIORITY,
    normalize_category,
    normalize_name,
    normalize_bool,
)
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TEXT CLEANING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def clean_text(text: object) -> str:
    """Strip HTML, collapse whitespace, decode entities."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"\[.*?\]", "", text)            # remove [citation] brackets
    text = re.sub(r"\s+", " ", text)               # collapse whitespace
    text = text.strip()
    return text


def clean_price(val: object) -> Optional[float]:
    """Extract numeric price from various formats: '₹125.50', 'MRP: ₹200', etc."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val)
    # Find first decimal number
    match = re.search(r"[\d,]+\.?\d*", s.replace(",", ""))
    if match:
        try:
            return round(float(match.group()), 2)
        except ValueError:
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# DRUG_CONTENT PARSER  (Kaggle Medicines / Netmeds)
# ═══════════════════════════════════════════════════════════════════════════

# Netmeds drug_content is a single long text block with sections like:
#   "Introduction: Amoxicillin is ... Side Effects: nausea, vomiting ...
#    How It Works: ... Safety Advice: ... Interactions: ..."

_SECTION_PATTERNS = [
    ("description",         r"(?:introduction|about|overview)[:\s]*(.+?)(?=\b(?:side effects?|how (?:it )?works?|safety|warning|interaction|precaution|contra|faq|storage)|$)"),
    ("side_effects",        r"(?:side effects?|adverse (?:effects?|reactions?))[:\s]*(.+?)(?=\b(?:how (?:it )?works?|safety|warning|interaction|precaution|contra|faq|storage)|$)"),
    ("mechanism_of_action", r"(?:how (?:it )?works?|mechanism of action)[:\s]*(.+?)(?=\b(?:side effects?|safety|warning|interaction|precaution|contra|faq|storage)|$)"),
    ("warnings",            r"(?:safety advice|warnings?|precautions?|contra[\s-]*indications?)[:\s]*(.+?)(?=\b(?:side effects?|how (?:it )?works?|interaction|faq|storage)|$)"),
    ("drug_interactions",   r"(?:interactions?|drug interactions?)[:\s]*(.+?)(?=\b(?:side effects?|how (?:it )?works?|safety|warning|precaution|faq|storage)|$)"),
]


def parse_drug_content(raw: str) -> dict:
    """
    Parse the Netmeds drug_content blob into structured fields.
    Returns dict with keys: description, side_effects, mechanism_of_action,
                            warnings, drug_interactions
    """
    result = {k: "" for k, _ in _SECTION_PATTERNS}
    if not raw or not isinstance(raw, str):
        return result

    text = clean_text(raw)

    for field, pattern in _SECTION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result[field] = clean_text(match.group(1))

    # If no section matched, treat the whole thing as description
    if not any(result.values()):
        result["description"] = text

    return result


# ═══════════════════════════════════════════════════════════════════════════
# DATASET LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def load_workspace_medicines(path: Optional[str] = None) -> pd.DataFrame:
    """Load workspace medicine_dataset.csv → unified schema."""
    path = path or settings.MEDICINE_CSV
    log.info("Loading workspace medicines: %s", path)
    df = pd.read_csv(path, dtype=str)
    log.info("  Raw rows: %d", len(df))

    # Rename columns
    df = df.rename(columns=WORKSPACE_MED_MAP)

    # Clean
    df["medicine_name"] = df["medicine_name"].apply(normalize_name)
    df["medicine_category"] = df["medicine_category"].apply(normalize_category)
    df["classification"] = df["classification"].apply(
        lambda x: x.strip().title() if isinstance(x, str) else ""
    )
    df["source"] = "workspace_medicine"
    df["source_priority"] = SOURCE_PRIORITY["workspace_medicine"]

    # Ensure all schema columns exist
    for col in MEDICINE_SCHEMA_FIELDS:
        if col not in df.columns:
            df[col] = ""

    return df[MEDICINE_SCHEMA_FIELDS].copy()


def load_workspace_diseases(path: Optional[str] = None) -> pd.DataFrame:
    """Load workspace Diseases_Symptoms.csv → unified schema."""
    path = path or settings.DISEASE_CSV
    log.info("Loading workspace diseases: %s", path)
    df = pd.read_csv(path, dtype=str)
    log.info("  Raw rows: %d", len(df))

    df = df.rename(columns=WORKSPACE_DISEASE_MAP)
    df["disease_name"] = df["disease_name"].apply(
        lambda x: clean_text(x).strip().title() if isinstance(x, str) else ""
    )
    df["symptoms"] = df["symptoms"].apply(clean_text)
    df["treatments"] = df["treatments"].apply(clean_text)
    df["contagious"] = df["contagious"].apply(normalize_bool)
    df["chronic"] = df["chronic"].apply(normalize_bool)
    df["source"] = "workspace_disease"

    for col in DISEASE_SCHEMA_FIELDS:
        if col not in df.columns:
            df[col] = ""

    return df[DISEASE_SCHEMA_FIELDS].copy()


def load_kaggle_az(path: str) -> pd.DataFrame:
    """
    Load Kaggle A-Z Medicine Dataset of India → unified schema.
    Expected columns: name, price, Is_discontinued, manufacturer_name,
                      type, pack_size_label, short_composition1, short_composition2
    """
    log.info("Loading Kaggle A-Z India: %s", path)
    df = pd.read_csv(path, dtype=str, on_bad_lines="skip")
    log.info("  Raw rows: %d, Columns: %s", len(df), list(df.columns))

    # Rename mapped columns
    rename_map = {k: v for k, v in KAGGLE_AZ_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Build composition from short_composition1 + short_composition2
    comp1 = df.get("short_composition1", pd.Series(dtype=str)).fillna("").astype(str)
    comp2 = df.get("short_composition2", pd.Series(dtype=str)).fillna("").astype(str)
    df["composition"] = (comp1 + " + " + comp2).str.strip(" +")

    # Normalize
    df["medicine_name"] = df["medicine_name"].apply(normalize_name)
    if "medicine_category" in df.columns:
        df["medicine_category"] = df["medicine_category"].apply(normalize_category)
    df["price"] = df.get("price", pd.Series(dtype=str)).apply(lambda x: str(clean_price(x)) if clean_price(x) else "")
    df["is_discontinued"] = df.get("is_discontinued", pd.Series(dtype=str)).apply(
        lambda x: str(normalize_bool(x)) if normalize_bool(x) is not None else ""
    )

    df["source"] = "kaggle_az_india"
    df["source_priority"] = SOURCE_PRIORITY["kaggle_az_india"]

    for col in MEDICINE_SCHEMA_FIELDS:
        if col not in df.columns:
            df[col] = ""

    return df[MEDICINE_SCHEMA_FIELDS].copy()


def load_kaggle_medicines(path: str) -> pd.DataFrame:
    """
    Load Kaggle Medicines Dataset (Netmeds) → unified schema.
    Expected columns: disease_name, med_name, drug_content, generic_name,
                      drug_manufacturer, prescription_required, final_price,
                      drug_variant
    """
    log.info("Loading Kaggle Medicines (Netmeds): %s", path)
    df = pd.read_csv(path, dtype=str, on_bad_lines="skip")
    log.info("  Raw rows: %d, Columns: %s", len(df), list(df.columns))

    # Rename mapped columns
    rename_map = {k: v for k, v in KAGGLE_MED_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Parse drug_content into structured fields
    log.info("  Parsing drug_content into structured fields…")
    parsed = df["description"].apply(
        lambda x: parse_drug_content(x) if isinstance(x, str) else {}
    )
    parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)

    # drug_content → description; parsed fields fill side_effects, etc.
    for field in ["side_effects", "mechanism_of_action", "warnings", "drug_interactions"]:
        if field in parsed_df.columns:
            df[field] = parsed_df[field]
    # Override description with parsed intro if available
    parsed_desc = parsed_df.get("description", pd.Series(dtype=str))
    df["description"] = parsed_desc.where(parsed_desc.str.len() > 0, df.get("description", ""))

    # Normalize
    df["medicine_name"] = df["medicine_name"].apply(normalize_name)
    df["disease_category"] = df.get("disease_category", pd.Series(dtype=str)).apply(
        lambda x: clean_text(x).strip().title() if isinstance(x, str) else ""
    )
    df["generic_name"] = df.get("generic_name", pd.Series(dtype=str)).apply(clean_text)
    df["price"] = df.get("price", pd.Series(dtype=str)).apply(lambda x: str(clean_price(x)) if clean_price(x) else "")
    df["prescription_required"] = df.get("prescription_required", pd.Series(dtype=str)).apply(
        lambda x: "Yes" if isinstance(x, str) and "required" in x.lower() else ("No" if isinstance(x, str) else "")
    )

    df["source"] = "kaggle_medicines_netmeds"
    df["source_priority"] = SOURCE_PRIORITY["kaggle_medicines_netmeds"]

    for col in MEDICINE_SCHEMA_FIELDS:
        if col not in df.columns:
            df[col] = ""

    return df[MEDICINE_SCHEMA_FIELDS].copy()


# ═══════════════════════════════════════════════════════════════════════════
# DEDUPLICATION & FIELD-LEVEL MERGE
# ═══════════════════════════════════════════════════════════════════════════

def merge_medicines(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple medicine DataFrames with field-level priority.
    When the same medicine_name appears in multiple sources:
      - For each field, pick the value from the highest-priority source
        (lowest source_priority number) that has a non-empty value.
    """
    log.info("Merging %d datasets…", len(dfs))

    combined = pd.concat(dfs, ignore_index=True)
    log.info("  Combined rows before dedup: %d", len(combined))

    # Drop rows with empty medicine_name
    combined = combined[combined["medicine_name"].str.len() > 0].copy()

    # Sort by source_priority (ascending = best first)
    combined["source_priority"] = pd.to_numeric(combined["source_priority"], errors="coerce").fillna(99)
    combined = combined.sort_values("source_priority")

    # Group by normalized medicine_name and merge fields
    def _merge_group(group: pd.DataFrame) -> pd.Series:
        result = {}
        for col in MEDICINE_SCHEMA_FIELDS:
            # Take the first non-empty value (already sorted by priority)
            vals = group[col].dropna().astype(str)
            vals = vals[vals.str.strip().str.len() > 0]
            result[col] = vals.iloc[0] if len(vals) > 0 else ""

        # Special: merge sources into a list
        sources = group["source"].dropna().unique().tolist()
        result["source"] = "|".join(sources)

        return pd.Series(result)

    log.info("  Performing field-level merge (this may take a moment)…")
    merged = combined.groupby("medicine_name", sort=False).apply(
        _merge_group
    ).reset_index(drop=True)

    log.info("  Unique medicines after merge: %d", len(merged))
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# DISEASE ↔ MEDICINE LINKING
# ═══════════════════════════════════════════════════════════════════════════

def build_disease_medicine_map(
    diseases_df: pd.DataFrame,
    medicines_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a mapping table: disease_name → list of related medicine names.
    Uses:
      1. medicine.disease_category directly matches disease_name
      2. medicine.indication keywords overlap with disease symptoms
    """
    log.info("Building Disease ↔ Medicine mapping…")

    links = []

    # Direct match: medicine.disease_category == disease.disease_name
    for _, med in medicines_df.iterrows():
        cat = str(med.get("disease_category", "")).strip()
        if cat and cat != "nan":
            links.append({
                "disease_name": cat,
                "medicine_name": med["medicine_name"],
                "link_type": "direct_category",
            })

    # Keyword match: indication ↔ disease name
    disease_names = set(diseases_df["disease_name"].str.lower().dropna())
    for _, med in medicines_df.iterrows():
        indication = str(med.get("indication", "")).lower()
        for dn in disease_names:
            if dn and len(dn) > 3 and dn in indication:
                links.append({
                    "disease_name": dn.title(),
                    "medicine_name": med["medicine_name"],
                    "link_type": "indication_keyword",
                })

    link_df = pd.DataFrame(links).drop_duplicates()
    log.info("  Disease-Medicine links: %d", len(link_df))
    return link_df


# ═══════════════════════════════════════════════════════════════════════════
# QUALITY REPORTS
# ═══════════════════════════════════════════════════════════════════════════

def generate_quality_report(
    medicines_df: pd.DataFrame,
    diseases_df: pd.DataFrame,
    output_path: str,
):
    """Generate a Markdown quality report of the cleaned data."""
    log.info("Generating quality report…")

    lines = [
        "# PharmaBot — Data Quality Report\n",
        f"## Summary",
        f"- **Total unique medicines**: {len(medicines_df):,}",
        f"- **Total diseases**: {len(diseases_df):,}",
        "",
        "## Field Completeness (Medicines)\n",
        "| Field | Filled | % |",
        "|-------|--------|---|",
    ]

    for col in MEDICINE_SCHEMA_FIELDS:
        if col in medicines_df.columns:
            filled = medicines_df[col].astype(str).str.strip().str.len().gt(0).sum()
            pct = round(100 * filled / len(medicines_df), 1) if len(medicines_df) > 0 else 0
            lines.append(f"| {col} | {filled:,} | {pct}% |")

    lines.extend([
        "",
        "## Source Distribution\n",
        "| Source | Count |",
        "|--------|-------|",
    ])

    # Sources can be pipe-separated
    source_counts = {}
    for src_str in medicines_df["source"].dropna():
        for s in str(src_str).split("|"):
            s = s.strip()
            if s:
                source_counts[s] = source_counts.get(s, 0) + 1
    for src, cnt in sorted(source_counts.items()):
        lines.append(f"| {src} | {cnt:,} |")

    lines.extend([
        "",
        "## Top 20 Medicine Categories\n",
        "| Category | Count |",
        "|----------|-------|",
    ])
    cat_counts = medicines_df["medicine_category"].value_counts().head(20)
    for cat, cnt in cat_counts.items():
        lines.append(f"| {cat} | {cnt:,} |")

    lines.extend([
        "",
        "## Disease Field Completeness\n",
        "| Field | Filled | % |",
        "|-------|--------|---|",
    ])
    for col in DISEASE_SCHEMA_FIELDS:
        if col in diseases_df.columns:
            filled = diseases_df[col].astype(str).str.strip().str.len().gt(0).sum()
            pct = round(100 * filled / len(diseases_df), 1) if len(diseases_df) > 0 else 0
            lines.append(f"| {col} | {filled:,} | {pct}% |")

    report_text = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    log.info("  Report written to %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    kaggle_az_path: Optional[str] = None,
    kaggle_med_path: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """
    Full pipeline:
      1. Load all datasets
      2. Clean & normalize
      3. Merge medicines
      4. Build disease-medicine map
      5. Export cleaned data
      6. Generate quality report
    """
    output_dir = output_dir or str(Path(settings.DATA_DIR) / "data" / "cleaned")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ─── Load ───
    med_frames = []

    # Workspace medicines (always available)
    ws_med = load_workspace_medicines()
    med_frames.append(ws_med)

    # Workspace diseases (always available)
    diseases = load_workspace_diseases()

    # Kaggle A-Z India (optional)
    if kaggle_az_path and Path(kaggle_az_path).exists():
        az_df = load_kaggle_az(kaggle_az_path)
        med_frames.append(az_df)
    else:
        log.warning("Kaggle A-Z India CSV not provided or not found — skipping.")

    # Kaggle Medicines / Netmeds (optional)
    if kaggle_med_path and Path(kaggle_med_path).exists():
        km_df = load_kaggle_medicines(kaggle_med_path)
        med_frames.append(km_df)

        # Also extract disease info from this dataset
        if "disease_category" in km_df.columns:
            extra_diseases = km_df[["disease_category"]].rename(
                columns={"disease_category": "disease_name"}
            ).drop_duplicates()
            extra_diseases["source"] = "kaggle_medicines_netmeds"
            for col in DISEASE_SCHEMA_FIELDS:
                if col not in extra_diseases.columns:
                    extra_diseases[col] = ""
            diseases = pd.concat(
                [diseases, extra_diseases[DISEASE_SCHEMA_FIELDS]],
                ignore_index=True,
            ).drop_duplicates(subset=["disease_name"])
    else:
        log.warning("Kaggle Medicines CSV not provided or not found — skipping.")

    # ─── Merge ───
    medicines = merge_medicines(med_frames)

    # ─── Disease ↔ Medicine Map ───
    dm_map = build_disease_medicine_map(diseases, medicines)

    # ─── Export ───
    medicines_csv = str(out / "medicines_clean.csv")
    diseases_csv = str(out / "diseases_clean.csv")
    dm_csv = str(out / "disease_medicine_map.csv")
    report_path = str(out / "quality_report.md")

    medicines.to_csv(medicines_csv, index=False)
    log.info("✓ Medicines exported: %s  (%d rows)", medicines_csv, len(medicines))

    diseases.to_csv(diseases_csv, index=False)
    log.info("✓ Diseases exported: %s  (%d rows)", diseases_csv, len(diseases))

    dm_map.to_csv(dm_csv, index=False)
    log.info("✓ Disease-Medicine map: %s  (%d links)", dm_csv, len(dm_map))

    generate_quality_report(medicines, diseases, report_path)

    log.info("═══ Pipeline complete ═══")
    return {
        "medicines": medicines,
        "diseases": diseases,
        "disease_medicine_map": dm_map,
        "output_dir": output_dir,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PharmaBot Data Cleaning Pipeline")
    parser.add_argument(
        "--kaggle-az",
        type=str,
        default=None,
        help="Path to Kaggle A-Z India CSV (A_Z_medicines_dataset_of_India.csv)",
    )
    parser.add_argument(
        "--kaggle-med",
        type=str,
        default=None,
        help="Path to Kaggle Medicines/Netmeds CSV (medicines.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for cleaned data (default: ../data/cleaned)",
    )
    args = parser.parse_args()

    run_pipeline(
        kaggle_az_path=args.kaggle_az,
        kaggle_med_path=args.kaggle_med,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
