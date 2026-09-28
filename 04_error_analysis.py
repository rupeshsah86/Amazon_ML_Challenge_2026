import numpy as np
import pandas as pd
from pathlib import Path
from utils.metrics import calculate_macro_f05, parse_matched_ids

DATASET_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/dataset")
TRAIN_DIR = DATASET_DIR / "train"
CHECKPOINT_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/checkpoints")

gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep='\t')
gt_map = {row['source1_entity_id']: parse_matched_ids(row['matched_entity_ids']) for _, row in gt_df.iterrows()}

oof_df = pd.read_csv(CHECKPOINT_DIR / "oof_pair_index.tsv", sep='\t')
oof_lgb = np.load(CHECKPOINT_DIR / "oof_lgb.npy")
oof_cat = np.load(CHECKPOINT_DIR / "oof_cat.npy")
oof_hgb = np.load(CHECKPOINT_DIR / "oof_hgb.npy")

oof_df['lgb'] = oof_lgb
oof_df['cat'] = oof_cat
oof_df['hgb'] = oof_hgb
oof_df['blend'] = (oof_lgb + oof_cat + oof_hgb) / 3.0

print("=== PHASE 4: ERROR ANALYSIS ===")

# Predict with simple blend
thresh = 0.35
pred_map = {s1_id: set() for s1_id in gt_map.keys()}

for s1_id, grp in oof_df.groupby('s1_id'):
    pred_map[s1_id] = set(grp[grp['blend'] >= thresh]['target_id'])

# Analyze Error Categories
false_positives = []
false_negatives = []

for s1_id, true_set in gt_map.items():
    pred_set = pred_map.get(s1_id, set())
    
    # FP: predicted but not true
    fp = pred_set.difference(true_set)
    if fp:
        false_positives.append((s1_id, fp))
        
    # FN: true but not predicted
    fn = true_set.difference(pred_set)
    if fn:
        false_negatives.append((s1_id, fn))

print(f"Total Source 1 Entities: {len(gt_map)}")
print(f"Entities with False Positives: {len(false_positives)}")
print(f"Entities with False Negatives: {len(false_negatives)}")

score_before = calculate_macro_f05(pred_map, gt_map)
print(f"OOF Macro F0.5 Score before post-processing: {score_before:.4f}")

# Record in performance log
perf_log = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/performance_log.txt")
with open(perf_log, "a") as f:
    f.write(f"Phase 4 Error Analysis | FP Count: {len(false_positives)} | FN Count: {len(false_negatives)} | Pre-post F0.5: {score_before:.4f}\n")
