"""
retrain_model.py — Retrain the InsightOps risk classifier

This script:
  1. Loads metadata from data/tickets_metadata.json
  2. Reconstructs embeddings from the FAISS index (no CSV needed)
  3. Applies the new operational-risk labeler (risk_labeler.py)
  4. Performs a 60/20/20 stratified train/val/test split
  5. Trains LogisticRegression with class_weight='balanced'
  6. Selects threshold on validation set (optimizing F1)
  7. Reports full metrics on held-out test set
  8. Saves model + threshold + updated metadata
  9. Runs 5 verification test cases
"""

import json
import warnings
import numpy as np
import pandas as pd
import faiss
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from risk_labeler import label_dataframe

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: Loading data")
print("=" * 60)

with open("data/tickets_metadata.json") as f:
    metadata = json.load(f)
df = pd.DataFrame(metadata)
print(f"Loaded {len(df)} records from metadata")

index = faiss.read_index("data/tickets.faiss")
print(f"FAISS index: {index.ntotal} vectors, {index.d} dimensions")
assert index.ntotal == len(df), f"FAISS/metadata mismatch: {index.ntotal} vs {len(df)}"

# Reconstruct all embeddings from FAISS index
print("Reconstructing embeddings from FAISS index...")
X_all = np.zeros((index.ntotal, index.d), dtype="float32")
for i in range(index.ntotal):
    X_all[i] = index.reconstruct(i)
print(f"Embeddings shape: {X_all.shape}")

# ──────────────────────────────────────────────────────────
# 2. Apply new risk labels
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2: Applying new operational risk labels")
print("=" * 60)

# Save old labels for comparison
old_labels = df["risk_label"].copy()

df = label_dataframe(df)
y_all = df["risk_label"].values

print(f"\nOLD label distribution:")
print(f"  HIGH (1): {(old_labels == 1).sum()} ({(old_labels == 1).mean():.1%})")
print(f"  LOW  (0): {(old_labels == 0).sum()} ({(old_labels == 0).mean():.1%})")

print(f"\nNEW label distribution:")
print(f"  HIGH (1): {(y_all == 1).sum()} ({(y_all == 1).mean():.1%})")
print(f"  LOW  (0): {(y_all == 0).sum()} ({(y_all == 0).mean():.1%})")

print(f"\nLabel changes: {(old_labels.values != y_all).sum()} tickets relabeled")

# ──────────────────────────────────────────────────────────
# 3. Train/Validation/Test split (60/20/20, stratified)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: Train/Validation/Test split")
print("=" * 60)

# First split: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.4, random_state=42, stratify=y_all
)

# Second split: 50/50 on temp → 20% val, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(y_train)} samples (HIGH: {y_train.sum()}, LOW: {(y_train==0).sum()})")
print(f"Val:   {len(y_val)} samples (HIGH: {y_val.sum()}, LOW: {(y_val==0).sum()})")
print(f"Test:  {len(y_test)} samples (HIGH: {y_test.sum()}, LOW: {(y_test==0).sum()})")

# ──────────────────────────────────────────────────────────
# 4. Train LogisticRegression (class_weight='balanced')
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 4: Training LogisticRegression")
print("=" * 60)

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
    solver="lbfgs",
)
lr_model.fit(X_train, y_train)
print("Model trained successfully")
print(f"  Coef shape: {lr_model.coef_.shape}")
print(f"  Classes: {lr_model.classes_}")

# ──────────────────────────────────────────────────────────
# 5. Threshold selection on VALIDATION set
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 5: Threshold selection (validation set)")
print("=" * 60)

y_val_prob = lr_model.predict_proba(X_val)[:, 1]

# For operational risk: missing a real outage (FN) is much worse than
# a false alarm (FP). We use the F2-score which weights recall 2x higher
# than precision, then verify the recall is acceptable.

from sklearn.metrics import fbeta_score

thresholds = np.linspace(0.1, 0.9, 81)
f2s = []
for t in thresholds:
    y_pred_t = (y_val_prob >= t).astype(int)
    if y_pred_t.sum() == 0:
        f2s.append(0.0)
    else:
        f2s.append(fbeta_score(y_val, y_pred_t, beta=2))

best_idx = np.argmax(f2s)
best_threshold = thresholds[best_idx]
best_f2_val = f2s[best_idx]

y_val_pred_best = (y_val_prob >= best_threshold).astype(int)
val_recall = recall_score(y_val, y_val_pred_best)
val_precision = precision_score(y_val, y_val_pred_best)
val_f1 = f1_score(y_val, y_val_pred_best)

print(f"Best threshold (F2-optimized): {best_threshold:.2f}")
print(f"  Validation F2:        {best_f2_val:.4f}")
print(f"  Validation F1:        {val_f1:.4f}")
print(f"  Validation Precision: {val_precision:.4f}")
print(f"  Validation Recall:    {val_recall:.4f}")
print(f"\nRationale: F2-score weights recall 2x over precision,")
print(f"appropriate for operational risk where missing a real")
print(f"incident is worse than a false alarm.")

print(f"\nFinal selected threshold: {best_threshold:.3f}")

# ──────────────────────────────────────────────────────────
# 6. Evaluate on TEST set (untouched until now)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6: Test set evaluation")
print("=" * 60)

y_test_prob = lr_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)

roc_auc = roc_auc_score(y_test, y_test_prob)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["LOW", "HIGH"]))

# ──────────────────────────────────────────────────────────
# 7. Save model, threshold, updated metadata
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 7: Saving artifacts")
print("=" * 60)

joblib.dump(lr_model, "models/risk_lr_model.joblib")
print("Saved: models/risk_lr_model.joblib")

joblib.dump(best_threshold, "models/risk_threshold.joblib")
print(f"Saved: models/risk_threshold.joblib (threshold={best_threshold:.3f})")

# Update metadata with new labels
updated_metadata = []
for i, row in df.iterrows():
    rec = {
        "number": row["number"],
        "category": row["category"],
        "subcategory": row["subcategory"],
        "content": row["content"],
        "resolution_time": row["resolution_time"],
        "risk_label": int(row["risk_label"]),
    }
    updated_metadata.append(rec)

with open("data/tickets_metadata.json", "w") as f:
    json.dump(updated_metadata, f)
print(f"Saved: data/tickets_metadata.json ({len(updated_metadata)} records)")

# ──────────────────────────────────────────────────────────
# 8. Verify 5 test cases
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 8: Five test cases")
print("=" * 60)

from embed import embed_text

test_cases = [
    "Production server is completely down for all users.",
    "Users are unable to login to the application.",
    "My money was deducted by 800000 from my account.",
    "The application is slightly slow today.",
    "I need to reset my password.",
]

expected_direction = [
    "HIGH",
    "HIGH/MEDIUM",
    "HIGH",
    "LOW",
    "LOW",
]

print(f"\n{'Ticket':<60} {'Score':>7} {'Level':>6}  Expected")
print("-" * 100)

for ticket, expected in zip(test_cases, expected_direction):
    emb = np.array(embed_text(ticket), dtype="float32").reshape(1, -1)
    score = lr_model.predict_proba(emb)[:, 1][0]
    label = "HIGH" if score >= best_threshold else "LOW"
    match = "PASS" if label in expected else "MISS"
    print(f"{ticket:<60} {score:>7.3f} {label:>6}  {expected} {match}")

print("\n" + "=" * 60)
print("DONE — All phases complete")
print("=" * 60)
