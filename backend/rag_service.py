"""
Core RAG service — ties together ChromaDB retrieval, OpenFDA lookup,
LLM synthesis, and conversation-thread management.
"""

import re
import uuid
import logging
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from config import settings
from prompts import (
    SYSTEM_PROMPT,
    CHAT_PROMPT_TEMPLATE,
    SYMPTOM_PROMPT_TEMPLATE,
    MEDICINE_ENRICHMENT_PROMPT,
    SAFETY_CHECK_PROMPT,
)
import openfda_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM abstraction
# ---------------------------------------------------------------------------

class LLMService:
    """Thin wrapper over Gemini / OpenRouter."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER

        if self.provider == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._model = genai.GenerativeModel(settings.gemini_model)
        elif self.provider == "openrouter":
            from openai import OpenAI

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.OPENROUTER_API_KEY,
            )
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def generate(self, prompt: str) -> str:
        try:
            if self.provider == "gemini":
                resp = self._model.generate_content(prompt)
                return resp.text
            else:
                completion = self._client.chat.completions.create(
                    model=settings.openrouter_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = completion.choices[0].message.content or ""
                # Strip <think>...</think> tags from reasoning models
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return text
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return "I'm sorry, I encountered an error generating a response. Please try again."


# ---------------------------------------------------------------------------
# Conversation threads (in-memory)
# ---------------------------------------------------------------------------

_threads: dict[str, list[dict]] = {}


def _get_thread(thread_id: Optional[str]) -> tuple[str, list[dict]]:
    if thread_id and thread_id in _threads:
        return thread_id, _threads[thread_id]
    tid = thread_id or str(uuid.uuid4())
    _threads[tid] = []
    return tid, _threads[tid]


def _format_history(history: list[dict], max_turns: int = 6) -> str:
    """Format recent conversation history for LLM context."""
    recent = history[-max_turns * 2 :]  # last N user+assistant pairs
    if not recent:
        return "(No prior conversation)"
    lines = []
    for msg in recent:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------------------

class RAGService:
    def __init__(self):
        self.llm = LLMService()

        # ChromaDB
        self._chroma = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL,
        )
        self._med_col = self._chroma.get_or_create_collection(
            "medicines", embedding_function=self._ef
        )
        self._dis_col = self._chroma.get_or_create_collection(
            "diseases", embedding_function=self._ef
        )
        self._dm_col = self._chroma.get_or_create_collection(
            "disease_medicine_map", embedding_function=self._ef
        )

    # ----- public API -----

    def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
    ) -> dict:
        """
        Main chat endpoint logic.
        Returns dict with: response, thread_id, sources, confidence, medicines_referenced
        """
        tid, history = _get_thread(thread_id)

        # 1. Safety check (lightweight — skip LLM call, use keyword heuristic)
        if self._is_unsafe(message):
            answer = (
                "I'm unable to provide information that could be harmful. "
                "If you or someone you know is in distress, please contact:\n"
                "• **National Crisis Helpline (US):** 988\n"
                "• **Poison Control:** 1-800-222-1222\n"
                "• **Emergency:** 911"
            )
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            return {
                "response": answer,
                "thread_id": tid,
                "sources": [],
                "confidence": "N/A",
                "medicines_referenced": [],
            }

        # 2. Extract medicine names from the query (+ conversation context)
        drug_names = self._extract_drug_names(message, history)

        # 3. Gather context from all sources
        sources = []
        context_parts = []

        # 3a. OpenFDA (highest confidence)
        fda_hits = {}
        for name in drug_names:
            fda_info = openfda_service.search_drug(name)
            if fda_info:
                fda_hits[name] = fda_info
                context_parts.append(openfda_service.format_fda_context(fda_info))
                sources.append({"type": "fda", "label": f"FDA: {fda_info.get('brand_name') or name}"})

        # 3b. ChromaDB — medicines
        med_results = self._query_collection(self._med_col, message, n=5)
        if med_results:
            context_parts.append("[SOURCE: Medicine Dataset]\n" + "\n".join(med_results))
            sources.append({"type": "dataset", "label": "Medicine Dataset"})

        # 3c. ChromaDB — diseases/symptoms
        dis_results = self._query_collection(self._dis_col, message, n=3)
        if dis_results:
            context_parts.append("[SOURCE: Disease & Symptoms Dataset]\n" + "\n".join(dis_results))
            sources.append({"type": "dataset", "label": "Disease & Symptom Dataset"})

        # 3d. Disease ↔ Medicine mapping
        dm_results = self._query_collection(self._dm_col, message, n=5)
        if dm_results:
            context_parts.append("[SOURCE: Disease-Medicine Mapping]\n" + "\n".join(dm_results))
            sources.append({"type": "dataset", "label": "Disease-Medicine Map"})

        # 3e. Enrich with metadata from ChromaDB (composition, side effects)
        med_meta = self._query_collection_with_meta(self._med_col, message, n=3)
        for meta in med_meta:
            extra_parts = []
            for field in ["composition", "generic_name", "disease_category"]:
                val = meta.get(field, "")
                if val and len(val) > 2:
                    extra_parts.append(f"{field.replace('_', ' ').title()}: {val}")
            if extra_parts:
                context_parts.append("[SOURCE: Enriched Metadata]\n" + " | ".join(extra_parts))

        # 4. Determine confidence
        if fda_hits:
            confidence = "high"
        elif med_results or dis_results:
            confidence = "medium"
        else:
            confidence = "low"

        context = "\n\n".join(context_parts) if context_parts else "(No relevant context found in databases.)"

        # 5. Build LLM prompt
        prompt = CHAT_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            chat_history=_format_history(history),
            context=context,
            query=message,
        )

        # 6. Generate response
        answer = self.llm.generate(prompt)

        # 7. Update thread
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})

        # 8. Referenced medicines metadata
        medicines_ref = []
        for name, info in fda_hits.items():
            medicines_ref.append({
                "name": info.get("brand_name") or name,
                "generic_name": info.get("generic_name", ""),
                "manufacturer": info.get("manufacturer", ""),
                "route": info.get("route", ""),
                "source": "fda",
            })

        return {
            "response": answer,
            "thread_id": tid,
            "sources": sources,
            "confidence": confidence,
            "medicines_referenced": medicines_ref,
        }

    def search_medicines(self, query: str, n: int = 10) -> list[dict]:
        """Semantic search across medicine collection."""
        results = self._med_col.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            meta["relevance"] = round(1 - dist, 4) if dist else 0
            out.append(meta)
        return out

    def get_medicine_detail(self, name: str) -> dict:
        """Get comprehensive info for a single medicine from all sources."""
        # OpenFDA first
        fda_info = openfda_service.search_drug(name)

        # Dataset search — use metadata for structured fields
        dataset_results = self._med_col.query(
            query_texts=[name],
            n_results=5,
            include=["metadatas"],
        )
        dataset_matches = dataset_results["metadatas"][0] if dataset_results["metadatas"] else []

        # Build enriched detail from best dataset match
        enriched = {}
        if dataset_matches:
            best = dataset_matches[0]
            for key in [
                "name", "generic_name", "manufacturer", "composition",
                "disease_category", "dosage_form", "strength", "price",
                "prescription_required", "classification", "indication",
                "source",
            ]:
                val = best.get(key, "")
                if val and str(val).strip():
                    enriched[key] = val

        # Related diseases (via disease-medicine map)
        related_diseases = []
        if name:
            dm_results = self._dm_col.query(
                query_texts=[name],
                n_results=5,
                include=["metadatas"],
            ) if self._dm_col.count() > 0 else {"metadatas": [[]]}
            for meta in (dm_results["metadatas"][0] if dm_results["metadatas"] else []):
                disease = meta.get("disease_name", "")
                if disease and disease not in related_diseases:
                    related_diseases.append(disease)

        return {
            "fda": fda_info,
            "dataset": dataset_matches,
            "enriched": enriched,
            "related_diseases": related_diseases,
            "confidence": "high" if fda_info else ("medium" if dataset_matches else "low"),
        }

    def check_symptoms(self, symptoms: list[str], thread_id: Optional[str] = None) -> dict:
        """Symptom-based information lookup (NOT diagnosis)."""
        tid, history = _get_thread(thread_id)

        symptom_text = ", ".join(symptoms)

        # Query diseases collection
        disease_results = self._query_collection(self._dis_col, symptom_text, n=5)
        context_parts = []
        if disease_results:
            context_parts.append("[Disease/Symptom matches]\n" + "\n".join(disease_results))

        # Query disease-medicine mapping for treatment info
        dm_results = self._query_collection(self._dm_col, symptom_text, n=5)
        if dm_results:
            context_parts.append("[Related Medicines]\n" + "\n".join(dm_results))

        context = "\n\n".join(context_parts) if context_parts else "(No matching conditions found in database.)"

        prompt = SYMPTOM_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            symptoms=symptom_text,
        )

        answer = self.llm.generate(prompt)

        history.append({"role": "user", "content": f"Symptoms: {symptom_text}"})
        history.append({"role": "assistant", "content": answer})

        sources = []
        if disease_results:
            sources.append({"type": "dataset", "label": "Disease & Symptom Dataset"})
        if dm_results:
            sources.append({"type": "dataset", "label": "Disease-Medicine Map"})

        return {
            "response": answer,
            "thread_id": tid,
            "sources": sources,
            "confidence": "medium" if (disease_results or dm_results) else "low",
        }

    # ----- private helpers -----

    def _query_collection(self, col, query: str, n: int = 5) -> list[str]:
        """Query a ChromaDB collection and return document texts."""
        try:
            if col.count() == 0:
                return []
            results = col.query(query_texts=[query], n_results=min(n, col.count()))
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            logger.warning("ChromaDB query failed: %s", e)
            return []

    def _query_collection_with_meta(self, col, query: str, n: int = 5) -> list[dict]:
        """Query a ChromaDB collection and return metadata dicts."""
        try:
            if col.count() == 0:
                return []
            results = col.query(
                query_texts=[query],
                n_results=min(n, col.count()),
                include=["metadatas"],
            )
            return results["metadatas"][0] if results["metadatas"] else []
        except Exception as e:
            logger.warning("ChromaDB metadata query failed: %s", e)
            return []

    def get_medicines_for_disease(self, disease_name: str, n: int = 10) -> list[dict]:
        """Get medicines linked to a disease via the mapping collection."""
        try:
            if self._dm_col.count() == 0:
                return []
            results = self._dm_col.query(
                query_texts=[disease_name],
                n_results=min(n, self._dm_col.count()),
                include=["metadatas"],
            )
            return results["metadatas"][0] if results["metadatas"] else []
        except Exception as e:
            logger.warning("Disease-medicine map query failed: %s", e)
            return []

    def _extract_drug_names(self, message: str, history: list[dict]) -> list[str]:
        """
        Simple heuristic to extract potential drug names from the message.
        Looks for capitalised words and common drug-name patterns.
        Also checks recent history for drug names in context.
        """
        names = set()

        # Patterns: capitalised words that might be drug names
        # Exclude very common English words
        _common = {
            "I", "The", "What", "How", "Why", "When", "Where", "Can", "Is",
            "Are", "Do", "Does", "Please", "Tell", "About", "This", "That",
            "Which", "Could", "Would", "Should", "Will", "Has", "Have", "Had",
            "Been", "Not", "And", "But", "For", "With", "From", "Also", "Any",
            "Its", "Use", "Side", "Effects", "Drug", "Medicine", "Tablet",
            "Dose", "Dosage", "Interactions", "Warnings", "Symptoms",
        }

        # Look for quoted names first: "amoxicillin"
        quoted = re.findall(r'"([^"]+)"', message) + re.findall(r"'([^']+)'", message)
        names.update(q.strip() for q in quoted if len(q.strip()) > 2)

        # Look for capitalised words (potential drug names)
        words = re.findall(r'\b[A-Z][a-z]{2,}\b', message)
        for w in words:
            if w not in _common:
                names.add(w)

        # Also check for known patterns: words ending in common drug suffixes
        _suffixes = (
            "cillin", "mycin", "azole", "pril", "olol", "statin", "prazole",
            "sartan", "vir", "mab", "nib", "tide", "gliptin", "phen",
            "done", "pine", "lam", "pam", "zepam", "oxin", "formin",
        )
        all_words = re.findall(r'\b\w{4,}\b', message.lower())
        for w in all_words:
            for suf in _suffixes:
                if w.endswith(suf):
                    names.add(w.capitalize())

        # Also add any explicitly lowercase drug names
        lower_words = re.findall(r'\b[a-z]{4,}\b', message)
        for w in lower_words:
            for suf in _suffixes:
                if w.endswith(suf):
                    names.add(w)

        # Check recent history for drug names mentioned
        for msg in history[-4:]:
            content = msg.get("content", "")
            for suf in _suffixes:
                matches = re.findall(rf'\b\w*{suf}\b', content, re.IGNORECASE)
                names.update(m for m in matches if len(m) > 3)

        return list(names)

    def _is_unsafe(self, message: str) -> bool:
        """Keyword-based safety check (fast, no LLM needed)."""
        _unsafe_patterns = [
            r"\b(overdose|OD)\b",
            r"\b(suicide|self[- ]?harm|kill\s*(my)?self)\b",
            r"\b(abuse|get\s*high|recreational)\b",
            r"\bhow\s+to\s+(die|hurt|poison)\b",
            r"\b(lethal\s+dose|LD50)\b",
            r"\b(dangerous\s+combination|mix.*drug)\b",
        ]
        text = message.lower()
        return any(re.search(p, text, re.IGNORECASE) for p in _unsafe_patterns)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_rag: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag
    if _rag is None:
        _rag = RAGService()
    return _rag
