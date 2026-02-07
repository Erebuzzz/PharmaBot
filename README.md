# PharmaBot

A RAG-based Pharmaceutical Assistant that provides high-confidence medicine information by combining **OpenFDA API** data, **multi-dataset knowledge**, and **LLM reasoning** with strict safety guardrails.

## Key Features
- **Three-Tier Confidence**: FDA-verified (high) → Dataset-matched (medium) → Insufficient data (low, "consult doctor")
- **Threaded Conversations**: Follow-up queries within persistent chat threads
- **OpenFDA Integration**: Live lookup of FDA-approved drug labels (indications, warnings, interactions, dosage)
- **Multi-Dataset RAG**: 4 datasets unified into a single knowledge base with disease→medicine mapping
- **Symptom Checker**: Educational symptom-based information (NOT diagnosis)
- **Safety Guardrails**: Never fabricates medical info; always cites sources; includes disclaimers

## Architecture
```
Frontend (React + Vite)  →  FastAPI Backend  →  RAG Pipeline
                                                  ├── OpenFDA API (real-time)
                                                  ├── ChromaDB (vector search)
                                                  │    ├── medicines collection
                                                  │    ├── diseases collection
                                                  │    └── disease_medicine_map
                                                  └── Gemini / OpenRouter LLM
```

## Datasets
| Dataset | Source | Records | Key Fields |
|---------|--------|---------|------------|
| `medicine_dataset.csv` | Workspace | ~50K | Name, Category, Dosage Form, Strength, Manufacturer, Indication |
| `Diseases_Symptoms.csv` | Workspace | ~400 | Name, Symptoms, Treatments, Contagious, Chronic |
| `A_Z_medicines_dataset_of_India.csv` | [Kaggle](https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india/data) | ~250K | Name, Price, Manufacturer, Composition, Type, Discontinued |
| `medicines.csv` | [Kaggle](https://www.kaggle.com/datasets/drowsyng/medicines-dataset) | Varies | Disease, Medicine, Drug Content (side effects, mechanism, warnings), Generic Name, Price |

## Getting Started

### 1. Environment Setup
```bash
# Clone and create .env
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (get from https://aistudio.google.com/apikey)
```

### 2. Download Kaggle Datasets (Optional but Recommended)
Download from Kaggle and place in a `data/` folder at the project root:
```
PharmaBot/
  data/
    kaggle_az_india.csv    ← A_Z_medicines_dataset_of_India.csv
    kaggle_medicines.csv   ← medicines.csv (from drowsyng/medicines-dataset)
```

### 3. Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### 4. Data Pipeline
```bash
cd backend

# Quick mode (workspace CSVs only — no Kaggle needed):
python data_ingest.py

# Full mode (all 4 datasets with cleaning & deduplication):
python data_cleaning.py --kaggle-az ../data/kaggle_az_india.csv \
                        --kaggle-med ../data/kaggle_medicines.csv \
                        --output-dir ../data/cleaned
python data_ingest.py --cleaned-dir ../data/cleaned
```

### 5. Start Backend
```bash
cd backend
uvicorn main:app --reload
# API at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 6. Start Frontend
```bash
cd frontend
npm install
npm run dev
# UI at http://localhost:5173
```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Main RAG chat (threaded conversations) |
| POST | `/api/search` | Semantic medicine search |
| GET | `/api/medicine/{name}` | Detailed medicine info (FDA + dataset) |
| GET | `/api/disease/{name}/medicines` | Medicines linked to a disease |
| POST | `/api/symptoms` | Symptom-based information lookup |
| GET | `/api/health` | Health check |

## Project Structure
```
backend/
  config.py          — Settings from .env
  prompts.py         — LLM prompt templates with safety rules
  openfda_service.py — FDA API client with TTL cache
  data_schema.py     — Unified schema & column mappings
  data_cleaning.py   — Full preprocessing pipeline (4 datasets)
  data_ingest.py     — CSV → ChromaDB ingestion
  rag_service.py     — Core RAG pipeline (FDA + ChromaDB + LLM)
  main.py            — FastAPI application
frontend/
  src/
    api.js                   — API client
    App.jsx                  — Root component
    components/
      ChatInterface.jsx      — Threaded chat UI
      MessageBubble.jsx      — Message with confidence badges
      MedicineCard.jsx       — Structured medicine info
      SafetyBanner.jsx       — Medical disclaimer
      SourceTag.jsx          — FDA/Dataset/AI source tags
docs/                        — Planning documentation
```

## Configuration (.env)
```env
GEMINI_API_KEY=your_key_here       # Required for Gemini LLM
LLM_PROVIDER=gemini                # gemini or openrouter
OPENROUTER_API_KEY=                # Only if using openrouter
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Sentence transformer model
CHROMA_DB_PATH=./chroma_db        # Vector database storage
```

## Disclaimer
**This tool is for educational & informational purposes only.** It does not substitute professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.