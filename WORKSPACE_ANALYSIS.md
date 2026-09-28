# WORKSPACE ANALYSIS - AMAZON ML CHALLENGE 2026

## 1. System & Environment Inspection
- **Operating System**: macOS (Apple Silicon arm64)
- **CPU Cores**: 8 CPU cores available
- **Python Runtime**: Python 3.14 (Virtual Environment located at `./venv`)
- **Key Installed Libraries**: `pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `catboost`, `rapidfuzz`

## 2. Directory Structure Analysis
- Workspace root: `/Users/rupeshmacbook/Desktop/amazon_entity_resolution`
- Subdirectories:
  - `dataset/`: Contains `train/` and `test/` splits.
  - `code/business_entity_resolution/src/`: Contains source modular code.
  - `output/`: Stores `matching_results.tsv` and `candidate_pairs.tsv`.
  - `utils/`: Submission validator and metric evaluation modules.
  - `checkpoints/`: Model weights, candidate caches, and fold predictions.

## 3. Competition Dataset & Challenge Specification
- **Task**: Multi-source Business Entity Resolution (Record Linkage & Deduplication).
- **Sources**: Source 1 (deduplicated reference source), Source 2, and Source 3.
- **Fields**: `entity_id`, `business_name`, `business_address`, `country`.
- **Countries**: Open set. Training covers US and India; Test additionally includes France.
- **Goal**: For every Source 1 entity, identify matching records in Source 2 and Source 3.
- **Cardinality**: 0, 1, or multiple matches per Source 1 entity.
- **Noise Types**: Abbreviations, legal suffixes, typos, transliterations, address component reordering, missing postal codes/landmarks.

## 4. Evaluation Metric
- **Metric**: Macro F0.5 score calculated per Source 1 entity:
  $$\text{Macro } F0.5 = \frac{1}{|S_1|} \sum_{e \in S_1} \frac{1.25 \times P_e \times R_e}{0.25 \times P_e + R_e}$$
  - For singletons ($|R_{\text{true}}| = 0$), predicting empty set yields $F0.5 = 1.0$, while predicting any match yields $0.0$.
- **Validation Strategy**: 5-Fold Grouped Stratified Cross-Validation by `source1_entity_id` to strictly avoid data leakage.

## 5. Execution & Experimentation Roadmap
1. **Dataset Initializer**: Synthetic dataset generator creating realistic multi-country noisy entity pairs and ground truth TSVs.
2. **Phase 1 Baseline (`01_baseline_test.py`)**: Multi-channel candidate retrieval (blocking) + initial TF-IDF similarity features + LightGBM baseline.
3. **Phase 2 Advanced Features (`02_build_features.py`)**: String normalization, rapidfuzz ratio metrics, numeric agreement flags, address component token overlap, rare token counts.
4. **Phase 3 Diverse Ensemble (`03_train_ensemble.py`)**: 5-fold grouped model training with LightGBM and CatBoost classifiers.
5. **Phase 4 Error Analysis (`04_error_analysis.py`)**: Diagnose false positives, false negatives, candidate recall bottlenecks.
6. **Phase 5 Blending & Threshold Optimization (`05_optimize_blend.py`)**: OOF probability blending and leak-free decision threshold tuning.
7. **Phase 6 & 7 Final Inference & Validation (`06_generate_submission.py`)**: Test candidate generation, ensemble inference, TSV formatting, automated validator execution.
8. **Phase 8-10 Documentation & Packaging (`run_pipeline.py`)**: Generate comprehensive reports and package `N3Rflix_submission.zip`.
