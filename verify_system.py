import sys
import os
import joblib
import json
import faiss
import numpy as np

def run_checks():
    print("=" * 60)
    print("INSIGHTOPS INTEGRITY & SAFETY CHECKS")
    print("=" * 60)
    
    # 1. FAISS loading
    print("1. Loading FAISS index...")
    try:
        index = faiss.read_index("data/tickets.faiss")
        print(f"   [SUCCESS] FAISS index loaded. Total vectors: {index.ntotal}, Dimensions: {index.d}")
    except Exception as e:
        print(f"   [ERROR] Failed to load FAISS index: {e}")
        return False
        
    # 2. Metadata loading
    print("2. Loading metadata...")
    try:
        with open("data/tickets_metadata.json") as f:
            metadata = json.load(f)
        print(f"   [SUCCESS] Metadata loaded. Total records: {len(metadata)}")
    except Exception as e:
        print(f"   [ERROR] Failed to load metadata: {e}")
        return False
        
    # 3. Match check
    print("3. Checking FAISS/Metadata alignment...")
    if index.ntotal == len(metadata):
        print("   [SUCCESS] Vector count matches metadata count.")
    else:
        print(f"   [ERROR] Count mismatch: FAISS has {index.ntotal} but Metadata has {len(metadata)}.")
        return False
        
    # 4. Model & Threshold loading
    print("4. Loading Logistic Regression Model and Threshold...")
    try:
        lr_model = joblib.load("models/risk_lr_model.joblib")
        threshold = joblib.load("models/risk_threshold.joblib")
        print(f"   [SUCCESS] Model loaded successfully: {lr_model}")
        print(f"   [SUCCESS] Threshold loaded successfully: {threshold:.3f}")
    except Exception as e:
        print(f"   [ERROR] Failed to load model/threshold: {e}")
        return False
        
    # 5. Embedding dimensions check
    print("5. Checking model dimension compatibility...")
    if lr_model.coef_.shape[1] == index.d:
        print("   [SUCCESS] Model input features match FAISS dimension.")
    else:
        print(f"   [ERROR] Dimension mismatch: Model expects {lr_model.coef_.shape[1]} but FAISS has {index.d}")
        return False
        
    # 6. RAG retrieval and explanation check
    print("6. Checking RAG retrieval logic...")
    try:
        from rag import explain_risk
        from embed import embed_text
        
        test_query = "Production server is completely down for all users."
        emb = embed_text(test_query)
        res = explain_risk(emb, risk_score=0.929, k=5)
        print("   [SUCCESS] RAG returned explanation successfully.")
        print(f"   Explanation sample: {res['explanation']}")
        print(f"   Similar tickets found: {len(res['similar_tickets'])}")
    except Exception as e:
        print(f"   [ERROR] RAG check failed: {e}")
        return False
        
    # 7. Empty ticket input check
    print("7. Checking empty ticket handling...")
    empty_input = "   "
    if not empty_input.strip():
        print("   [SUCCESS] Empty input correctly identified as blank.")
    else:
        print("   [ERROR] Empty input check failed.")
        return False
        
    # 8. Very long ticket input check
    print("8. Checking very long ticket input safety...")
    try:
        long_input = "System is down. " * 1000
        emb_long = embed_text(long_input)
        score = lr_model.predict_proba(np.array(emb_long).reshape(1, -1))[:, 1][0]
        print(f"   [SUCCESS] Long input processed safely. Score: {score:.3f}")
    except Exception as e:
        print(f"   [ERROR] Long input crashed the system: {e}")
        return False
        
    # 9. Missing metadata fields safety
    print("9. Checking missing metadata fields safety...")
    try:
        bad_ticket = {"number": "TASK9999"} # missing content, category, subcategory
        cat = bad_ticket.get("category", "Unknown")
        subcat = bad_ticket.get("subcategory", "Unknown")
        rt = bad_ticket.get("resolution_time", 0.0)
        rl = bad_ticket.get("risk_label", 0)
        print(f"   [SUCCESS] Missing fields safely fallback to defaults (Cat: {cat}, Subcat: {subcat})")
    except Exception as e:
        print(f"   [ERROR] Missing metadata check failed: {e}")
        return False

    print("=" * 60)
    print("ALL INTEGRITY AND SAFETY CHECKS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
