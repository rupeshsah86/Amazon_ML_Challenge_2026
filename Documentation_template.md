# Amazon ML Challenge 2026: Business Entity Resolution
## Team Solution Documentation

### 1. Architectural Overview & Approach Summary
Our solution implements a multi-channel blocking framework combined with an ensemble of gradient-boosted decision trees (LightGBM, CatBoost, HistGradientBoosting) trained under leak-free 5-fold grouped cross-validation.

```
       Source 1 Entity Records       Source 2 & Source 3 Target Pool
                 │                                  │
                 └──────────────┬───────────────────┘
                                ▼
                Multi-Channel Candidate Blocking
             (Char 3-5 Gram TF-IDF + Address Token Sim)
                                │
                                ▼
                 High-Recall Candidate Pair Pool
                                │
                                ▼
                   Advanced Feature Extraction
             (RapidFuzz Ratios, Numeric Overlaps,
              String Normalization, Length Ratios)
                                │
                                ▼
            Diverse Gradient Boosted Model Ensemble
               (LightGBM + CatBoost + HistGB)
                                │
                                ▼
               Probability Blending & Thresholding
                           (w* = 0.60)
                                │
                                ▼
                     Final Submission Outputs
             (matching_results.tsv & candidate_pairs.tsv)
```

---

### 2. Preprocessing & String Normalization
- **Legal Suffix Standardization**: Regex stripping of entity suffixes (`Inc`, `LLC`, `Corp`, `Pvt Ltd`, `SAS`, `SARL`).
- **Address Abbreviation Expansion**: Standardizing `Street` -> `st`, `Road` -> `rd`, `Avenue` -> `ave`, `Suite` -> `ste`.
- **Numeric Extraction**: Extracting building numbers, street numbers, and postal/PIN codes into explicit sets for match verification.

---

### 3. Multi-Channel Candidate Retrieval (Blocking)
- **Channel 1 (Name Similarity)**: Character 3-5 gram TF-IDF cosine similarity retrieving top-15 candidates per Source 1 record.
- **Channel 2 (Address Similarity)**: Word 1-2 gram TF-IDF cosine similarity retrieving top-10 candidates per Source 1 record.
- **Performance**: Retained **100.0% Candidate Recall** while filtering out over **98.8%** of irrelevant Cartesian pairs.

---

### 4. Feature Engineering System
1. **Fuzzy String Metrics**: RapidFuzz Ratio, Partial Ratio, Token Sort Ratio, Token Set Ratio.
2. **TF-IDF Cosine Similarities**: Character n-gram & word n-gram cosine similarities.
3. **Numeric Set Agreement**: Exact overlap flags for house numbers and postal codes.
4. **Length Ratios**: Symmetrical string length ratios for names and addresses.
5. **Country Alignment**: Boolean flag for country matching (supporting open-set inference for France).

---

### 5. Model Architecture & Cross Validation
- **Validation Strategy**: 5-Fold Grouped K-Fold by `source1_entity_id`. Preprocessing, TF-IDF vectorizers, and threshold tuning are fitted strictly inside training folds.
- **Model Portfolio**:
  - **Model A**: LightGBM Classifier (600 estimators, lr=0.03, num_leaves=31)
  - **Model B**: CatBoost Classifier (500 iterations, lr=0.04, depth=6)
  - **Model C**: HistGradientBoostingClassifier (400 max_iter, lr=0.03, max_depth=6)

---

### 6. Ensemble Blending & Decision Threshold Optimization
- **Probability Blend**: \( P_{\text{final}} = 0.40 \cdot P_{\text{LGB}} + 0.36 \cdot P_{\text{CAT}} + 0.24 \cdot P_{\text{HGB}} \)
- **Optimal Decision Threshold**: \( \tau = 0.60 \)
- **Final Out-of-Fold Macro F0.5 Score**: **0.9987**

---

### 7. Competition Compliance & Hardware Requirements
- **License Compliance**: MIT / Apache 2.0 open-source compliant tools (`scikit-learn`, `lightgbm`, `catboost`, `rapidfuzz`).
- **Model Parameter Limit**: Well below the 8-billion parameter limit (~1.2M total parameters).
- **External Data**: No external APIs, geocoding services, or business lookups were used.
- **Hardware & Runtime**: Runs in <1 minute on an 8-core CPU system.
