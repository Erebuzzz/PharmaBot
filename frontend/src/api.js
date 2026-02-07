/**
 * API client for PharmaBot backend.
 */

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

/** Send a chat message (supports threaded conversations). */
export async function sendChat(message, threadId = null) {
  return request("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message, thread_id: threadId }),
  });
}

/** Semantic search for medicines. */
export async function searchMedicines(query, limit = 10) {
  return request("/api/search", {
    method: "POST",
    body: JSON.stringify({ query, limit }),
  });
}

/** Get detailed info for a specific medicine. */
export async function getMedicineDetail(name) {
  return request(`/api/medicine/${encodeURIComponent(name)}`);
}

/** Get medicines linked to a disease. */
export async function getMedicinesForDisease(name, limit = 10) {
  return request(`/api/disease/${encodeURIComponent(name)}/medicines?limit=${limit}`);
}

/** Symptom-based lookup. */
export async function checkSymptoms(symptoms, threadId = null) {
  return request("/api/symptoms", {
    method: "POST",
    body: JSON.stringify({ symptoms, thread_id: threadId }),
  });
}

/** Health check. */
export async function healthCheck() {
  return request("/api/health");
}
