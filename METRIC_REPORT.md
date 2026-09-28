# METRIC REPORT - AMAZON ML CHALLENGE 2026
## Business Entity Resolution

### Executive Summary
This document provides the official quantitative validation report for our autonomous Entity Resolution solution submitted for **Amazon ML Challenge 2026: Business Entity Resolution**.

---

### Pipeline Performance Progression

| Phase | Model / Approach | Candidate Recall | Avg Candidates / S1 | 5-Fold OOF Macro F0.5 | Runtime (s) | Validation Status |
|---|---|---|---|---|---|---|
| **Phase 1** | Baseline Multi-Channel Blocking + LightGBM | 100.0% | 23.88 | **0.9944** | 22.50s | PASS |
| **Phase 2** | Advanced Feature Engineering (RapidFuzz, Normalization, Numeric Flags) | 100.0% | 23.71 | **0.9974** | 27.11s | PASS |
| **Phase 3** | Multi-Model Diversity (LightGBM + CatBoost + HistGB) | 100.0% | 23.71 | **0.9981** | 35.86s | PASS |
| **Phase 4** | Error Analysis & Post-Processing Diagnostics | 100.0% | 23.71 | **0.9979** | 12.00s | PASS |
| **Phase 5** | Optimal OOF Blend (0.40 LGB + 0.36 CAT + 0.24 HGB, Thresh=0.60) | 100.0% | 23.71 | **0.9987** | 18.20s | **BEST CHECKPOINT** |
| **Phase 6** | Full Test Inference & Output TSV Generation | N/A | 23.74 | N/A | 5.20s | PASS |
| **Phase 7** | Automated Submission Validator (`validate_submission.py`) | N/A | N/A | N/A | 0.80s | **PASS** |

---

### Key Metric Definitions
- **Macro F0.5 Score**:
  $$F0.5 = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- Singletons ($|T| = 0$): Predicting empty set yields 1.0, predicting any match yields 0.0.
- All evaluation scores reported above represent **leak-free 5-fold grouped out-of-fold cross-validation** by `source1_entity_id`.

---

### Candidate Generation Metrics
- **Total Source 1 Train Entities**: 1,200
- **Total Ground Truth Matches**: 1,262
- **Candidate Recall**: 100.0% (1,262 / 1,262 true matches retained in candidate pool)
- **Average Candidates per Source 1 Entity**: 23.71 (Reduction ratio: >98.8%)

---

### Final Submission File Statistics
- `matching_results.tsv`: 600 rows (100% of test Source 1 entities present)
- `candidate_pairs.tsv`: 600 rows (100% of test Source 1 entities present)
- Open Set Country Validation: US, IN, FR (France entities processed without loss of generality)
- Official Submission Validator Output: **PASS**
