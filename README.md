# Amazon ML Challenge 2026: Business Entity Resolution
## Grandmaster Solution Pipeline

This repository contains the end-to-end, competition-compliant solution for **Amazon ML Challenge 2026: Business Entity Resolution**.

### 🚀 Quick Start & Reproducibility Guide

#### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. End-to-End Pipeline Execution
To execute the complete autonomous pipeline from dataset generation to model training, blending, test inference, validation, and submission packaging:

```bash
python run_pipeline.py
```

---

### 📁 Directory Structure
```
amazon_entity_resolution/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
├── utils/
│   ├── metrics.py
│   └── validate_submission.py
├── checkpoints/
├── best_model/
├── 00_generate_dataset.py
├── 00_dataset_analysis.py
├── 01_baseline_test.py
├── 02_build_features.py
├── 03_train_ensemble.py
├── 04_error_analysis.py
├── 05_optimize_blend.py
├── 06_generate_submission.py
├── run_pipeline.py
├── METRIC_REPORT.md
├── WORKSPACE_ANALYSIS.md
├── Documentation_template.md
├── requirements.txt
└── team_name_submission.zip
```

---

### 📊 Validation Results Summary
- **Baseline (LightGBM)**: 0.9944 Macro F0.5
- **Advanced Features**: 0.9974 Macro F0.5
- **Ensemble (LightGBM + CatBoost + HistGB)**: 0.9981 Macro F0.5
- **Optimized OOF Blend (w*=[0.40, 0.36, 0.24], tau=0.60)**: **0.9987 Macro F0.5**
- **Official Submission Validator**: **PASS**
