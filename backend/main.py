"""
PharmaBot — FastAPI application.

Endpoints:
  POST /api/chat         — main RAG chat (supports threaded conversations)
  POST /api/search       — semantic medicine search
  GET  /api/medicine/{name} — detailed medicine lookup (FDA + dataset)
  POST /api/symptoms     — symptom-based information
  GET  /api/health       — health check
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_service import get_rag_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

app = FastAPI(
    title="PharmaBot API",
    description="RAG-powered pharmaceutical information assistant",
    version="1.0.0",
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    sources: list[dict]
    confidence: str
    medicines_referenced: list[dict]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(10, ge=1, le=50)


class SymptomRequest(BaseModel):
    symptoms: list[str] = Field(..., min_length=1)
    thread_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "PharmaBot"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main conversational endpoint.
    Supports threaded follow-up queries via thread_id.
    """
    rag = get_rag_service()
    result = rag.chat(message=req.message, thread_id=req.thread_id)
    return result


@app.post("/api/search")
async def search(req: SearchRequest):
    """Semantic search across the medicine dataset."""
    rag = get_rag_service()
    results = rag.search_medicines(query=req.query, n=req.limit)
    return {"results": results, "count": len(results)}


@app.get("/api/medicine/{name}")
async def medicine_detail(name: str):
    """
    Get comprehensive information about a specific medicine.
    Combines FDA-approved labeling with dataset info.
    """
    rag = get_rag_service()
    detail = rag.get_medicine_detail(name)
    if not detail["fda"] and not detail["dataset"]:
        raise HTTPException(status_code=404, detail=f"Medicine '{name}' not found in any source.")
    return detail


@app.get("/api/disease/{name}/medicines")
async def medicines_for_disease(name: str, limit: int = 10):
    """
    Get medicines linked to a disease.
    Uses the disease-medicine mapping from the unified data pipeline.
    """
    rag = get_rag_service()
    results = rag.get_medicines_for_disease(disease_name=name, n=limit)
    return {"disease": name, "medicines": results, "count": len(results)}


@app.post("/api/symptoms")
async def symptoms(req: SymptomRequest):
    """
    Symptom-based informational lookup.
    NOT a diagnosis tool — always returns with disclaimers.
    """
    rag = get_rag_service()
    result = rag.check_symptoms(symptoms=req.symptoms, thread_id=req.thread_id)
    return result
