"""
rag.py — RAG-style explanation for InsightOps risk predictions

Retrieves similar historical tickets from the FAISS index and generates
an informative explanation based on:
  - risk score from the model
  - number of similar tickets retrieved
  - average resolution time of similar tickets
  - percentage of similar tickets classified as high risk
  - dominant category/subcategory from similar tickets
"""

import faiss
import json
import numpy as np
from collections import Counter

# load metadata
with open("data/tickets_metadata.json") as f:
    metadata = json.load(f)

# load FAISS index
index = faiss.read_index("data/tickets.faiss")
assert index.ntotal == len(metadata), "Index and metadata mismatch"


def retrieve_similar(query_embedding, k=5):
    """Retrieve k most similar tickets from the FAISS index."""
    q = np.array(query_embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)

    scores, idx = index.search(q, k)
    results = [metadata[i] for i in idx[0]]

    return results, scores[0]


def explain_risk(query_embedding, risk_score, k=5):
    """
    Generate a RAG-style explanation for the risk prediction.

    Parameters
    ----------
    query_embedding : array-like
        The embedding of the incoming ticket.
    risk_score : float
        The model's predicted risk probability.
    k : int
        Number of similar tickets to retrieve.

    Returns
    -------
    dict with keys:
        explanation : str
        avg_resolution_time : float
        high_risk_ratio : float
        similar_tickets : list[dict]
        dominant_category : str
        dominant_subcategory : str
    """
    results, scores = retrieve_similar(query_embedding, k)

    # Extract metrics from similar tickets (with safe defaults)
    resolution_times = []
    risk_labels = []
    categories = []
    subcategories = []

    for r in results:
        rt = r.get("resolution_time")
        if rt is not None:
            resolution_times.append(float(rt))
        rl = r.get("risk_label")
        if rl is not None:
            risk_labels.append(int(rl))
        cat = r.get("category", "Unknown")
        if cat:
            categories.append(cat)
        subcat = r.get("subcategory", "Unknown")
        if subcat:
            subcategories.append(subcat)

    avg_resolution = float(np.mean(resolution_times)) if resolution_times else 0.0
    high_risk_ratio = float(np.mean(risk_labels)) if risk_labels else 0.0

    # Find dominant category/subcategory
    dominant_category = Counter(categories).most_common(1)[0][0] if categories else "Unknown"
    dominant_subcategory = Counter(subcategories).most_common(1)[0][0] if subcategories else "Unknown"

    # Build explanation based on risk level
    risk_pct = int(high_risk_ratio * 100)

    # Load threshold dynamically
    import joblib
    try:
        threshold = joblib.load("models/risk_threshold.joblib")
    except Exception:
        threshold = 0.61  # fallback default

    if risk_score >= threshold:
        severity = "High operational risk detected"
    elif risk_score >= threshold * 0.5:
        severity = "Moderate operational risk detected"
    else:
        severity = "Low operational risk assessed"

    explanation = (
        f"{severity}. The model assigned a risk score of {risk_score:.2f}. "
        f"Among the {k} most similar historical tickets, "
        f"{risk_pct}% were classified as high-risk "
        f"and the average resolution time was {avg_resolution:,.0f} minutes. "
        f"Similar incidents were primarily associated with "
        f"{dominant_category}/{dominant_subcategory} issues."
    )

    return {
        "explanation": explanation,
        "avg_resolution_time": avg_resolution,
        "high_risk_ratio": high_risk_ratio,
        "similar_tickets": results,
        "dominant_category": dominant_category,
        "dominant_subcategory": dominant_subcategory,
    }
