# Team Roles & Responsibilities

## 1. Data Engineer / MLOps Specialist
**Responsibilities:**
- **Data Acquisition & Ingestion:**
    - Download and manage datasets from multiple sources (Kaggle: `az-medicine-dataset-of-india`, `medicines-dataset`, and local `medicine_dataset.csv`).
    - Handle data cleaning, normalization, and schema alignment (unifying columns like "Name", "Composition", "Uses").
- **Embedding Pipeline:**
    - Implement the **BioBERT** embedding generation using `sentence-transformers`.
    - Manage the **ChromaDB** vector database setup and indexing.
- **Enrichment:**
    - Develop logic to enrich missing data points using LLM inference before indexing.

## 2. Backend Developer (FastAPI & RAG Logic)
**Responsibilities:**
- **API Development:**
    - Build and maintain the **FastAPI** application.
    - Implement endpoints: `/search`, `/chat`, `/identify`, `/symptoms`.
- **RAG Implementation:**
    - Develop the retrieval logic (querying ChromaDB).
    - Integrate **Gemini LLM** for response generation.
    - Implement context construction prompt engineering.
- **Fail-Safe & Monitoring:**
    - Integrate **Comet ML** to log all requests, responses, and retrieval metrics.
    - Implement fallback mechanisms for low-confidence queries.

## 3. Frontend Developer (React & UI/UX)
**Responsibilities:**
- **UI Implementation:**
    - Develop the **React** application using **Vite**.
    - Implement responsive design with **Tailwind CSS**.
    - Build components: Chat Interface, Medicine Cards, Search Bar.
- **Safety Integration:**
    - Implement persistent safety banners and disclaimers.
    - Ensure "Symptom Checker" and "Identification" features have clear non-diagnostic warnings.
- **User Experience:**
    - Create intuitive flows for "Alternative" suggestions and detailed medicine views.

## 4. QA & Safety Officer
**Responsibilities:**
- **Verification:**
    - Test all API endpoints for correctness and performance.
    - Verify RAG responses against medical common knowledge (finding hallucinations).
- **Safety Compliance:**
    - adversarial testing (e.g., asking for dangerous combinations) to ensure guardrails work.
    - Review "Symptom Checker" outputs for appropriate disclaimers.
- **Documentation:**
    - Maintain `task.md` and `implementation_plan.md`.
    - Ensure code is well-documented and modular.
