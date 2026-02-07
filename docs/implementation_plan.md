# PharmaBot Implementation Plan

## Goal Description
Build a RAG-based PharmaBot that provides detailed information about medicines, including identification, composition, uses, mechanism of action, side effects, etc. The system will use a provided CSV dataset (`medicine_dataset.csv`) as the primary knowledge base, augmented by an LLM for generating/inferring missing details (e.g., specific side effects based on drug category).

## User Review Required
> [!IMPORTANT]
> **Data Limitation**: The provided `medicine_dataset.csv` contains only basic fields (Name, Category, Dosage, Manufacturer, Indication). Fields like "Mechanism of Action", "Composition", and "Side Effects" are missing.
> **Strategy**: The system will use **LLM Inference** to generate plausible details for these missing fields based on the available `Category` and `Indication`.
> **Multi-Source Data**: We will ingest additional datasets (Kaggle: `az-medicine-dataset-of-india`, `medicines-dataset`) to improve coverage. A merging logic will prioritize the most informative source.
> **Safety & Guardrails**:
> - **Disclaimer**: All responses must start with a disclaimer: "I am an AI assistant. This information is for educational purposes only and not a substitute for professional medical advice."
> - **Symptom Checker**: Explicitly state that this is **non-diagnostic** and for information only.
> - **Fail-Safe**: If the LLM is unsure or the retrieval score is low, fall back to a safe "I don't know, please consult a doctor" response.

## Proposed Changes

### Backend
#### [NEW] [main.py](file:///d:/PharmaBot/backend/main.py)
- FastAPI application entry point.
- Endpoints:
    - `/search`: Text-based search for medicines.
    - `/chat`: RAG-based chat endpoint to answer questions (Uses, Side Effects, Mechanism, Warnings, How/When to use).
    - `/symptoms`: Endpoint for symptom-based queries (returns potential conditions/medicines with strict disclaimers).
    - `/identify`: (Optional) Endpoint for medicine identification (text-based).

#### [NEW] [rag_service.py](file:///d:/PharmaBot/backend/rag_service.py)
- Handles interaction with Vector DB (ChromaDB) and LLM (Gemini).
- **Embeddings**: Uses **BioBERT** (via `sentence-transformers`) for domain-specific embeddings.
- **Preprocessing**: Uses **NLTK** for tokenization and text cleaning.
- **Fail-Safe**: Integrates **Comet API** (`comet_ml`) to log all queries, retrieval results, and LLM responses for monitoring and debugging.
- Implements the retrieval and generation logic.
- `enrich_medicine_info(medicine_data)`: Function to generate missing fields using LLM.
- **New Features**:
    - `get_alternatives(medicine_name)`: Logic to find similar medicines based on `Category` and `Indication`.
    - `check_safety(query)`: Pre-check query for harmful intent.

#### [NEW] [ingest.py](file:///d:/PharmaBot/backend/ingest.py)
- Script to load `medicine_dataset.csv` and Kaggle datasets.
- **Normalization**: Maps diverse column names (e.g., "Brand Name" -> "Name", "Benefits" -> "Indication") to a unified schema.
- **Enrichment**: Merges duplicate entries by combining unique information.
- **Text Processing**: Uses **NLTK** to clean and chunk text.
- **Embeddings**: Generates **BioBERT** embeddings for medicine descriptions.
- Stores them in ChromaDB.

### Frontend
#### [NEW] [App.jsx](file:///d:/PharmaBot/frontend/src/App.jsx)
- Main React application component.
- Features a Chat Interface and a Medicine Details View.
- **Safety Banner**: Persistent banner warning about the nature of the app.

#### [NEW] [MedicineCard.jsx](file:///d:/PharmaBot/frontend/src/components/MedicineCard.jsx)
- Component to display structured medicine info (Name, Manufacturer, Strength, etc.).
- **New Sections**:
    - "Mechanism of Action"
    - "How/When to Use"
    - "Side Effects / Warnings"
    - "Alternatives"

### Environment
#### [NEW] [.env](file:///d:/PharmaBot/.env)
- Stores API keys (`GEMINI_API_KEY`, `COMET_API_KEY`) and configuration.

## Verification Plan

### Automated Tests
- Run `ingest.py` on a subset of data to verify Vector DB creation with BioBERT embeddings.
- Test `/chat` endpoint with queries like "What are the side effects of Acetocillin?" and verify it returns a plausible response.
- **Comet Logging**: Verify that interactions are correctly logged to Comet.
- **Safety Tests**: Verify response to "How to overdose?" is a refusal/help message.

### Manual Verification
- Launch Frontend via `npm run dev`.
- Interactively chat with the bot to verify all requested fields (Composition, Uses, Mechanism, etc.) are addressed in responses.
- check "Symptom Checker" flows for appropriate disclaimers.
