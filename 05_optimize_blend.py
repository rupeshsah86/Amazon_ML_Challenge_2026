import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils.metrics import calculate_macro_f05, parse_matched_ids

DATASET_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/dataset")
TRAIN_DIR = DATASET_DIR / "train"
CHECKPOINT_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/checkpoints")
BEST_MODEL_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/best_model")

gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep='\t')
gt_map = {row['source1_entity_id']: parse_matched_ids(row['matched_entity_ids']) for _, row in gt_df.iterrows()}

oof_df = pd.read_csv(CHECKPOINT_DIR / "oof_pair_index.tsv", sep='\t')
oof_lgb = np.load(CHECKPOINT_DIR / "oof_lgb.npy")
oof_cat = np.load(CHECKPOINT_DIR / "oof_cat.npy")
oof_hgb = np.load(CHECKPOINT_DIR / "oof_hgb.npy")

oof_df['lgb'] = oof_lgb
oof_df['cat'] = oof_cat
oof_df['hgb'] = oof_hgb

print("=== PHASE 5: BLENDING & THRESHOLD OPTIMIZATION ===")

best_score = -1.0
best_weights = None
best_thresh = 0.5

# Grid Search over Weights and Thresholds
weight_candidates = []
for w1 in np.linspace(0, 1, 6):
    for w2 in np.linspace(0, 1 - w1, 6):
        w3 = round(1.0 - w1 - w2, 4)
        if w3 >= 0:
            weight_candidates.append((round(w1, 2), round(w2, 2), w3))

print(f"Searching across {len(weight_candidates)} weight combinations...")

for w1, w2, w3 in weight_candidates:
    oof_df['p_blend'] = w1 * oof_df['lgb'] + w2 * oof_df['cat'] + w3 * oof_df['hgb']
    
    for thresh in np.arange(0.15, 0.75, 0.05):
        pred_map = {s1_id: set() for s1_id in gt_map.keys()}
        for s1_id, grp in oof_df.groupby('s1_id'):
            pred_map[s1_id] = set(grp[grp['p_blend'] >= thresh]['target_id'])
            
        score = calculate_macro_f05(pred_map, gt_map)
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)
            best_thresh = float(thresh)

print(f"\nOptimal Ensemble Blend Found:")
print(f"Weights (LGB, CAT, HGB): {best_weights}")
print(f"Decision Threshold: {best_thresh:.2f}")
print(f"Optimized OOF Macro F0.5 Score: {best_score:.4f}")

# Save Config
config = {
    "w_lgb": best_weights[0],
    "w_cat": best_weights[1],
    "w_hgb": best_weights[2],
    "threshold": best_thresh,
    "oof_macro_f05": best_score
}

with open(BEST_MODEL_DIR / "blend_config.json", "w") as f:
    json.dump(config, f, indent=4)

perf_log = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/performance_log.txt")
with open(perf_log, "a") as f:
    f.write(f"Phase 5 Blend Optimization | OOF Macro F0.5: {best_score:.4f} | Weights: {best_weights} | Thresh: {best_thresh:.2f}\n")
