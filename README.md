# PharmaBot

A RAG-based Pharmaceutical Assistant designed to provide comprehensive information about medicines, including:
- **Identification**: Recognize medicines from text (and potentially images).
- **Composition & Uses**: Detailed breakdown of active ingredients and their purposes.
- **Mechanism of Action**: Explanation of how the drug works.
- **Side Effects & Warnings**: Critical safety information and potential adverse reactions.
- **Alternatives**: Suggested substitutes based on composition and indication.
- **Symptom Checker**: Educational info on symptoms (Non-diagnostic).

## Features
- **RAG Pipeline**: Utilizes **BioBERT** embeddings and **ChromaDB** for accurate retrieval of medical data.
- **LLM Integration**: Powered by **Gemini** for natural language generation and data enrichment.
- **Safety First**: strict guardrails, disclaimers, and fail-safe mechnisms via **Comet ML**.
- **Modern UI**: React + Vite + Tailwind CSS frontend.

## Project Structure
- `backend/`: FastAPI application and RAG logic.
- `frontend/`: React application.
- `docs/`: Project documentation (Implementation Plan, Team Roles, Task List).
- `data/`: Datasets (e.g., `medicine_dataset.csv`).

## Getting Started
1. **Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```
2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Disclaimer
This tool is for educational purposes only and does not substitute professional medical advice. Always consult a doctor.