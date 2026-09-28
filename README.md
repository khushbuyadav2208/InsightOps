# InsightOps — AI-Powered Operational Risk Detection for Support Tickets

## 1. Project Title
**InsightOps — AI-Powered Operational Risk Detection for Support Tickets**

---

## 2. Project Overview
InsightOps is an intelligent IT support ticket analysis platform designed to identify, assess, and explain operational and business risks. Rather than relying on simple metadata or static keyword rules, InsightOps uses semantic embeddings and machine learning to scan incoming support tickets, estimate their operational threat level, and fetch matching historical precedents to explain the diagnosis in real-time. By catching critical outages, security breaches, and financial issues early, IT operations teams can proactively triage incidents before they scale.

---

## 3. Problem Statement
The original iteration of the risk classification system generated target labels based purely on support ticket **resolution time** (setting the 75th percentile as the high-risk threshold). This led to severe operational discrepancies:
*   **False Positives**: Routine administrative tasks (such as requesting group mailboxes or Teams updates) that sat in backlog queues for days were incorrectly classified as **HIGH RISK**.
*   **False Negatives**: Critical business-impacting issues (such as a database server crash or a security authentication failure) that were resolved quickly in under an hour were incorrectly classified as **LOW RISK**.

As a result, the model was predicting "Will this ticket take a long time to close?" rather than "Is this ticket operationally critical?"

---

## 4. Solution
InsightOps introduces a refined, multi-layered operational risk detection approach:
*   **Structured & Text Rules**: Re-mapped risk labeling using a robust taxonomy based on the ticket's `category`, `subcategory`, and underlying `content`.
*   **Severity Recognition**: Detects explicit operational indicators (e.g., outages, server crashes, unauthorized deductions, data corruption) to mark high-impact events as **HIGH RISK** immediately.
*   **Routine Access Overrides**: Suppresses standard individual requests (e.g., routine password resets, single PIV card exemptions, minor software updates) to **LOW RISK**.
*   **Machine-Learning Classification**: Learns the complex boundary between low-impact administration and severe operational failures using high-dimensional sentence embeddings.
*   **Precedent Retrieval (RAG)**: Employs FAISS similarity search to locate historical equivalents and construct contextual explanations regarding why a ticket is high risk.

---

## 5. Architecture
The runtime pipeline flows as follows:

```
          Incoming Support Ticket Text
                       ↓
         [embed.py] Sentence Transformer Embedding (all-MiniLM-L6-v2)
                       ↓
         [models/risk_lr_model.joblib] Logistic Regression Classifier
                       ↓
         Risk Score (Probability) + Risk Level (LOW / MEDIUM / HIGH)
                       ↓
         [data/tickets.faiss] FAISS Similarity Search (k=5 Nearest Neighbors)
                       ↓
         Historical Metadata Lookup (data/tickets_metadata.json)
                       ↓
         [rag.py] RAG Explanation Generation (Avg. Resolution Time, Dominant Category, High-Risk Ratio)
                       ↓
         [app.py] Streamlit Enterprise Dashboard Interface
```

---

## 6. Technology Stack
*   **Language**: Python 3.12+
*   **Dashboard**: Streamlit
*   **Semantic Representation**: Sentence Transformers (`all-MiniLM-L6-v2`)
*   **Vector Search Engine**: FAISS (Facebook AI Similarity Search - CPU edition)
*   **Machine Learning**: Scikit-Learn (Logistic Regression, evaluation helpers)
*   **Data Structures**: Pandas, NumPy, Joblib, JSON

---

## 7. Model Training
*   **Model**: Logistic Regression (`class_weight='balanced'` to compensate for the imbalanced minority class representing ~9.9% of the dataset).
*   **Features**: 384-dimensional dense semantic vectors extracted using the `all-MiniLM-L6-v2` model.
*   **Data Split**: Stratified train/validation/test split of **60% / 20% / 20%** to avoid data leakage.
*   **Threshold Selection**: Evaluated validation predictions across 81 candidate thresholds using the **F2-Score** metric (weighting Recall 2x higher than Precision) to prioritize catching critical failures over minimizing minor false positives.
*   **Final Decision Threshold**: **`0.610`**

---

## 8. Evaluation Results
The final retrained model achieved high performance metrics on the untouched test set:
*   **ROC-AUC**: `0.9484`
*   **High Risk Recall**: `79.72%`
*   **High Risk Precision**: `57.37%`
*   **F1-Score**: `0.6672`

---

## 9. RAG Explanation Layer
When a ticket is submitted, the system queries the FAISS index for the **5 most similar historical tickets** and aggregates their metadata to compile a descriptive diagnosis:
1.  **Risk Score**: Displays the model's precise probability estimation.
2.  **Average Resolution Time**: Details how long similar tickets took to resolve on average (in minutes).
3.  **High-Risk Ratio**: The percentage of nearest neighbor tickets that were classified as high-risk under the operational taxonomy.
4.  **Dominant Category**: Identifies the primary category/subcategory (e.g. `SOFTWARE/ERROR`) where similar incidents occurred.
5.  **Precedent Inspection**: Displays the category, subcategory, historical risk status, resolution time, and content preview for all 5 similar tickets.

---

## 10. Test Case Verification
The system was verified against 5 key test inputs and achieved **100% correct alignment** with business expectations under the F2-optimized threshold (`0.610`):

| Test Case | Input Text | Risk Score | Risk Level | Expected | Status |
|---|---|:---:|:---:|:---:|:---:|
| **Test 1** | "Production server is completely down for all users." | `0.929` | **HIGH** | HIGH | **PASS** |
| **Test 2** | "Users are unable to login to the application." | `0.860` | **HIGH** | HIGH / MEDIUM | **PASS** |
| **Test 3** | "My money was deducted by 800000 from my account." | `0.647` | **HIGH** | HIGH | **PASS** |
| **Test 4** | "The application is slightly slow today." | `0.021` | **LOW** | LOW | **PASS** |
| **Test 5** | "I need to reset my password." | `0.099` | **LOW** | LOW | **PASS** |

---

## 11. Dataset & FAISS Verification
*   **Total Index Size**: **25,649** dense vectors stored in `data/tickets.faiss`.
*   **Metadata Alignments**: **25,649** JSON records in `data/tickets_metadata.json`.
*   **Alignment Validation**: Verified that every FAISS vector ID aligns exactly with its respective metadata index.

---

## 12. How to Run
Follow these instructions to run the application locally on Windows:

1.  **Open PowerShell/Command Prompt** and navigate to the project directory:
    ```powershell
    cd c:\Users\ujjwa\Downloads\InsightOps-master\InsightOps-master
    ```
2.  **Ensure dependencies are installed**:
    ```powershell
    pip install -r requirements.txt
    ```
3.  **Launch the Streamlit dashboard**:
    ```powershell
    python -m streamlit run app.py
    ```
    *The web application will open automatically in your browser at `http://localhost:8501`.*

---

## 13. Project Structure
```
InsightOps-master/
├── data/
│   ├── tickets.faiss                # FAISS vector index (25,649 tickets)
│   └── tickets_metadata.json        # Aligned ticket metadata records
├── models/
│   ├── risk_lr_model.joblib         # Trained Logistic Regression classifier
│   └── risk_threshold.joblib        # F2-optimized decision threshold (0.610)
├── app.py                           # Streamlit dashboard interface
├── rag.py                           # FAISS search & RAG explanation pipeline
├── embed.py                         # Sentence Transformer embedding wrapper
├── risk_labeler.py                  # Operational-risk labeling logic
├── retrain_model.py                 # Pipeline retraining and validation script
├── verify_system.py                 # Automated integrity safety checks script
└── requirements.txt                 # Project dependencies
```

---

## 14. Limitations
*   **Support Tool Only**: This platform is designed to assist IT helpdesks and operations engineers by flagging tickets. It should support, not replace, human operational judgment.
*   **Static Threshold**: The threshold (`0.610`) is optimized on the static historical snapshot and does not automatically shift without retraining.

---

## 15. Future Improvements
*   **Richer Severity Signals**: Incorporate explicit system urgency metadata, user role levels, or system tags.
*   **Monitoring Drift**: Implement alerts to track changes in vocabulary (embedding drift) and resolution times over time.
*   **Feedback Loops**: Provide a "Correct Label" mechanism in the Streamlit UI to log correction data for automated incremental retraining.
*   **Granular Classes**: Move from binary prediction to multi-class estimation (Low vs. Medium vs. High) once sufficient labeled data representing distinct medium-severity categories is acquired.
