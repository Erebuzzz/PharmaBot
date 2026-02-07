"""
PharmaBot Data Pipeline — Unified Schema & Constants

Defines:
1. The unified data model (Disease Category → Medicines → Info)
2. Column mappings for each source dataset
3. Standard category/disease normalization maps
"""

# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED SCHEMA
# ═══════════════════════════════════════════════════════════════════════════
#
# We store TWO core entity types:
#
#   ┌──────────────────────┐       ┌──────────────────────────────────┐
#   │  Disease / Category  │  1:N  │         Medicine                 │
#   ├──────────────────────┤ ────→ ├──────────────────────────────────┤
#   │ disease_name         │       │ medicine_name (normalized)       │
#   │ symptoms             │       │ generic_name                     │
#   │ treatments           │       │ disease_category                 │
#   │ disease_code         │       │ medicine_category (e.g. Antibi.) │
#   │ contagious           │       │ composition                      │
#   │ chronic              │       │ dosage_form                      │
#   │ source               │       │ strength                         │
#   └──────────────────────┘       │ manufacturer                     │
#                                  │ indication                       │
#                                  │ classification (OTC/Rx)          │
#                                  │ description          (long text) │
#                                  │ side_effects         (long text) │
#                                  │ drug_interactions    (long text) │
#                                  │ warnings             (long text) │
#                                  │ mechanism_of_action  (long text) │
#                                  │ price                            │
#                                  │ pack_info                        │
#                                  │ is_discontinued                  │
#                                  │ prescription_required            │
#                                  │ source          (dataset origin) │
#                                  │ source_priority (1=highest)      │
#                                  └──────────────────────────────────┘
#
# ═══════════════════════════════════════════════════════════════════════════

MEDICINE_SCHEMA_FIELDS = [
    "medicine_name",
    "generic_name",
    "disease_category",
    "medicine_category",
    "composition",
    "dosage_form",
    "strength",
    "manufacturer",
    "indication",
    "classification",
    "description",
    "side_effects",
    "drug_interactions",
    "warnings",
    "mechanism_of_action",
    "price",
    "pack_info",
    "is_discontinued",
    "prescription_required",
    "source",
    "source_priority",
]

DISEASE_SCHEMA_FIELDS = [
    "disease_name",
    "symptoms",
    "treatments",
    "disease_code",
    "contagious",
    "chronic",
    "source",
]


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE COLUMN MAPPINGS → Unified Schema
# ═══════════════════════════════════════════════════════════════════════════

# Dataset 1: Workspace medicine_dataset.csv  (50K, synthetic-ish)
WORKSPACE_MED_MAP = {
    "Name": "medicine_name",
    "Category": "medicine_category",
    "Dosage Form": "dosage_form",
    "Strength": "strength",
    "Manufacturer": "manufacturer",
    "Indication": "indication",
    "Classification": "classification",
}

# Dataset 2: Workspace Diseases_Symptoms.csv  (407 diseases)
WORKSPACE_DISEASE_MAP = {
    "Name": "disease_name",
    "Symptoms": "symptoms",
    "Treatments": "treatments",
    "Disease_Code": "disease_code",
    "Contagious": "contagious",
    "Chronic": "chronic",
}

# Dataset 3: Kaggle A-Z India  (~250K real medicines)
# Columns: name, price, Is_discontinued, manufacturer_name, type,
#           pack_size_label, short_composition1, short_composition2
KAGGLE_AZ_MAP = {
    "name": "medicine_name",
    "manufacturer_name": "manufacturer",
    "type": "medicine_category",
    "pack_size_label": "pack_info",
    "price": "price",
    "Is_discontinued": "is_discontinued",
    # composition is built from short_composition1 + short_composition2
}

# Dataset 4: Kaggle Medicines (Netmeds)  (rich text data)
# Columns: disease_name, med_name, drug_content, generic_name,
#           drug_manufacturer, prescription_required, final_price,
#           drug_variant, drug_manufacturer_origin
KAGGLE_MED_MAP = {
    "med_name": "medicine_name",
    "disease_name": "disease_category",
    "generic_name": "generic_name",
    "drug_manufacturer": "manufacturer",
    "final_price": "price",
    "drug_variant": "pack_info",
    "prescription_required": "prescription_required",
    "drug_content": "description",  # rich text — parsed further
}

# ═══════════════════════════════════════════════════════════════════════════
# SOURCE PRIORITIES  (lower = higher priority for field-level merging)
# ═══════════════════════════════════════════════════════════════════════════
SOURCE_PRIORITY = {
    "kaggle_medicines_netmeds": 1,   # richest text data (descriptions, side effects)
    "kaggle_az_india": 2,            # real medicines, composition, prices
    "workspace_medicine": 3,          # basic structured data
    "workspace_disease": 1,          # disease/symptom authority
}


# ═══════════════════════════════════════════════════════════════════════════
# NORMALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

# Common category normalization (handles different naming conventions)
CATEGORY_ALIASES = {
    # Antibiotics
    "antibiotic": "Antibiotic",
    "antibiotics": "Antibiotic",
    "antibacterial": "Antibiotic",
    # Antifungals
    "antifungal": "Antifungal",
    "antifungals": "Antifungal",
    # Antivirals
    "antiviral": "Antiviral",
    "antivirals": "Antiviral",
    # Pain / Analgesics
    "analgesic": "Analgesic",
    "pain relief": "Analgesic",
    "pain reliever": "Analgesic",
    "nsaid": "NSAID",
    # Antipyretics
    "antipyretic": "Antipyretic",
    "fever": "Antipyretic",
    # Antidepressants
    "antidepressant": "Antidepressant",
    "antidepressants": "Antidepressant",
    # Antidiabetic
    "antidiabetic": "Antidiabetic",
    "diabetes": "Antidiabetic",
    "anti-diabetic": "Antidiabetic",
    # Cardiovascular
    "cardiovascular": "Cardiovascular",
    "cardiac": "Cardiovascular",
    "heart": "Cardiovascular",
    "antihypertensive": "Antihypertensive",
    # Respiratory
    "respiratory": "Respiratory",
    "asthma": "Respiratory",
    "bronchodilator": "Respiratory",
    # Gastrointestinal
    "gastrointestinal": "Gastrointestinal",
    "gastric": "Gastrointestinal",
    "antacid": "Gastrointestinal",
    # Allergy
    "antihistamine": "Antihistamine",
    "allergy": "Antihistamine",
    "anti-allergy": "Antihistamine",
    # Vitamins / Supplements
    "vitamin": "Supplement",
    "supplement": "Supplement",
    "nutritional": "Supplement",
    # Other
    "antiseptic": "Antiseptic",
    "vaccine": "Vaccine",
    "hormonal": "Hormonal",
    "steroid": "Steroid",
    "corticosteroid": "Steroid",
}


def normalize_category(raw: str) -> str:
    """Normalize medicine category to a standard label."""
    if not raw or not isinstance(raw, str):
        return "Unknown"
    clean = raw.strip().lower()
    return CATEGORY_ALIASES.get(clean, raw.strip().title())


def normalize_name(raw: str) -> str:
    """Normalize a medicine / disease name for dedup and matching."""
    if not raw or not isinstance(raw, str):
        return ""
    # Strip, title-case, collapse whitespace
    import re
    name = raw.strip()
    name = re.sub(r"\s+", " ", name)
    # Remove trailing dosage info for matching  ("Amoxicillin 500mg" → "Amoxicillin")
    name = re.sub(r"\s+\d+\s*(mg|ml|mcg|g|iu|%)\b.*$", "", name, flags=re.IGNORECASE)
    return name.strip().title()


def normalize_bool(val) -> bool | None:
    """Normalize various boolean representations."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "yes", "1", "y"):
            return True
        if low in ("false", "no", "0", "n"):
            return False
    return None
