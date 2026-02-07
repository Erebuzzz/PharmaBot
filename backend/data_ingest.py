"""
PharmaBot Data Ingestion — Loads cleaned datasets into ChromaDB collections.

Two modes:
  1. Quick mode (no Kaggle CSVs): uses workspace CSVs directly
  2. Full mode (after data_cleaning.py): uses cleaned/merged data from data/cleaned/

Run:
    cd backend

    # Quick mode — just workspace CSVs
    python data_ingest.py

    # Full mode — after running data_cleaning.py first
    python data_ingest.py --cleaned-dir ../data/cleaned
"""

import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BATCH_SIZE = 500  # ChromaDB add batch size


def _get_client_and_ef():
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL,
    )
    return client, ef


# --------------------------------------------------------------------------
# Medicine ingestion — unified schema
# --------------------------------------------------------------------------

def _build_medicine_document(row: dict) -> str:
    """
    Build a rich text document from a medicine row for embedding.
    The more informative the text, the better the retrieval quality.
    """
    parts = []

    name = row.get("medicine_name", "") or row.get("name", "")
    if name:
        parts.append(f"Medicine: {name}")

    generic = row.get("generic_name", "")
    if generic:
        parts.append(f"Generic Name: {generic}")

    cat = row.get("medicine_category", "") or row.get("category", "")
    if cat:
        parts.append(f"Category: {cat}")

    composition = row.get("composition", "")
    if composition:
        parts.append(f"Composition: {composition}")

    indication = row.get("indication", "")
    if indication:
        parts.append(f"Indication: {indication}")

    disease_cat = row.get("disease_category", "")
    if disease_cat:
        parts.append(f"Used for: {disease_cat}")

    form = row.get("dosage_form", "")
    if form:
        parts.append(f"Dosage Form: {form}")

    strength = row.get("strength", "")
    if strength:
        parts.append(f"Strength: {strength}")

    manufacturer = row.get("manufacturer", "")
    if manufacturer:
        parts.append(f"Manufacturer: {manufacturer}")

    classification = row.get("classification", "")
    if classification:
        parts.append(f"Classification: {classification}")

    # Long text fields — include a snippet for embedding quality
    for field, label in [
        ("description", "Description"),
        ("side_effects", "Side Effects"),
        ("mechanism_of_action", "Mechanism"),
        ("warnings", "Warnings"),
    ]:
        val = row.get(field, "")
        if val and len(str(val)) > 5:
            snippet = str(val)[:500]
            parts.append(f"{label}: {snippet}")

    return " | ".join(parts)


def _build_metadata(row: dict, is_unified: bool) -> dict:
    """Build ChromaDB metadata dict from a row."""
    if is_unified:
        return {
            "name": str(row.get("medicine_name", "")),
            "generic_name": str(row.get("generic_name", "")),
            "category": str(row.get("medicine_category", "")),
            "disease_category": str(row.get("disease_category", "")),
            "composition": str(row.get("composition", "")),
            "dosage_form": str(row.get("dosage_form", "")),
            "strength": str(row.get("strength", "")),
            "manufacturer": str(row.get("manufacturer", "")),
            "indication": str(row.get("indication", "")),
            "classification": str(row.get("classification", "")),
            "prescription_required": str(row.get("prescription_required", "")),
            "price": str(row.get("price", "")),
            "source": str(row.get("source", "")),
        }
    else:
        return {
            "name": str(row.get("Name", "")),
            "category": str(row.get("Category", "")),
            "dosage_form": str(row.get("Dosage Form", "")),
            "strength": str(row.get("Strength", "")),
            "manufacturer": str(row.get("Manufacturer", "")),
            "indication": str(row.get("Indication", "")),
            "classification": str(row.get("Classification", "")),
            "source": "workspace_medicine",
        }


def ingest_medicines(client, ef, csv_path: str, is_unified: bool = True):
    """Ingest medicines CSV into ChromaDB."""
    log.info("Loading medicines: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    log.info("  %d rows loaded.", len(df))

    col = client.get_or_create_collection("medicines", embedding_function=ef)

    name_col = "medicine_name" if is_unified else "Name"

    docs, ids, metas = [], [], []
    for idx, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        if not name:
            continue

        if is_unified:
            text = _build_medicine_document(row.to_dict())
        else:
            text = (
                f"Medicine: {name} | "
                f"Category: {row.get('Category', '')} | "
                f"Dosage Form: {row.get('Dosage Form', '')} | "
                f"Strength: {row.get('Strength', '')} | "
                f"Manufacturer: {row.get('Manufacturer', '')} | "
                f"Indication: {row.get('Indication', '')} | "
                f"Classification: {row.get('Classification', '')}"
            )

        docs.append(text)
        ids.append(f"med_{idx}")
        metas.append(_build_metadata(row.to_dict(), is_unified))

    total = len(docs)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        col.upsert(
            documents=docs[start:end],
            ids=ids[start:end],
            metadatas=metas[start:end],
        )
        log.info("  Medicines: %d / %d ingested", end, total)

    log.info("✓ Medicines collection: %d documents", col.count())


# --------------------------------------------------------------------------
# Disease / Symptom ingestion
# --------------------------------------------------------------------------

def ingest_diseases(client, ef, csv_path: str, is_unified: bool = True):
    """Ingest diseases CSV into ChromaDB."""
    log.info("Loading diseases: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    log.info("  %d rows loaded.", len(df))

    col = client.get_or_create_collection("diseases", embedding_function=ef)

    name_col = "disease_name" if is_unified else "Name"
    symp_col = "symptoms" if is_unified else "Symptoms"
    treat_col = "treatments" if is_unified else "Treatments"
    cont_col = "contagious" if is_unified else "Contagious"
    chro_col = "chronic" if is_unified else "Chronic"
    code_col = "disease_code" if is_unified else "Disease_Code"

    docs, ids, metas = [], [], []
    for idx, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        if not name:
            continue

        symptoms = str(row.get(symp_col, ""))
        treatments = str(row.get(treat_col, ""))

        text = (
            f"Disease: {name} | "
            f"Symptoms: {symptoms} | "
            f"Treatments: {treatments} | "
            f"Contagious: {row.get(cont_col, '')} | "
            f"Chronic: {row.get(chro_col, '')}"
        )

        docs.append(text)
        ids.append(f"dis_{idx}")
        metas.append({
            "name": name,
            "disease_code": str(row.get(code_col, "")),
            "symptoms": symptoms,
            "treatments": treatments,
            "contagious": str(row.get(cont_col, "")),
            "chronic": str(row.get(chro_col, "")),
            "source": str(row.get("source", "workspace_disease")),
        })

    total = len(docs)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        col.upsert(
            documents=docs[start:end],
            ids=ids[start:end],
            metadatas=metas[start:end],
        )
        log.info("  Diseases: %d / %d ingested", end, total)

    log.info("✓ Diseases collection: %d documents", col.count())


# --------------------------------------------------------------------------
# Disease ↔ Medicine map
# --------------------------------------------------------------------------

def ingest_disease_medicine_map(client, ef, csv_path: str):
    """Ingest disease-medicine mapping as a searchable collection."""
    log.info("Loading disease-medicine map: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    log.info("  %d links loaded.", len(df))

    col = client.get_or_create_collection("disease_medicine_map", embedding_function=ef)

    docs, ids, metas = [], [], []
    for idx, row in df.iterrows():
        disease = str(row.get("disease_name", "")).strip()
        medicine = str(row.get("medicine_name", "")).strip()
        if not disease or not medicine:
            continue

        text = f"Disease: {disease} → Medicine: {medicine}"
        docs.append(text)
        ids.append(f"dm_{idx}")
        metas.append({
            "disease_name": disease,
            "medicine_name": medicine,
            "link_type": str(row.get("link_type", "")),
        })

    total = len(docs)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        col.upsert(
            documents=docs[start:end],
            ids=ids[start:end],
            metadatas=metas[start:end],
        )
        log.info("  Disease-Medicine map: %d / %d ingested", end, total)

    log.info("✓ Disease-Medicine map: %d documents", col.count())


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PharmaBot Data Ingestion into ChromaDB")
    parser.add_argument(
        "--cleaned-dir",
        type=str,
        default=None,
        help="Path to cleaned data directory (output of data_cleaning.py). "
             "If not provided, ingests raw workspace CSVs directly.",
    )
    args = parser.parse_args()

    log.info("=== PharmaBot Data Ingestion ===")
    log.info("ChromaDB path: %s", settings.CHROMA_DB_PATH)
    log.info("Embedding model: %s", settings.EMBEDDING_MODEL)

    client, ef = _get_client_and_ef()

    if args.cleaned_dir:
        cleaned = Path(args.cleaned_dir)
        med_csv = str(cleaned / "medicines_clean.csv")
        dis_csv = str(cleaned / "diseases_clean.csv")
        dm_csv = str(cleaned / "disease_medicine_map.csv")

        if not Path(med_csv).exists():
            log.error("Cleaned medicines CSV not found: %s", med_csv)
            log.error("Run data_cleaning.py first, then pass --cleaned-dir")
            sys.exit(1)

        ingest_medicines(client, ef, med_csv, is_unified=True)

        if Path(dis_csv).exists():
            ingest_diseases(client, ef, dis_csv, is_unified=True)
        else:
            log.warning("Cleaned diseases not found, using workspace CSV")
            ingest_diseases(client, ef, settings.DISEASE_CSV, is_unified=False)

        if Path(dm_csv).exists():
            ingest_disease_medicine_map(client, ef, dm_csv)
    else:
        log.info("No --cleaned-dir. Ingesting raw workspace CSVs only.")
        ingest_medicines(client, ef, settings.MEDICINE_CSV, is_unified=False)
        ingest_diseases(client, ef, settings.DISEASE_CSV, is_unified=False)

    log.info("=== Ingestion complete ===")


if __name__ == "__main__":
    main()
