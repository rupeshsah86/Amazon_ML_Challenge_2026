import json
import numpy as np
import pandas as pd
from pathlib import Path

DATASET_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/dataset")
TEST_DIR = DATASET_DIR / "test"
CHECKPOINT_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/checkpoints")
BEST_MODEL_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/best_model")
OUTPUT_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

test_s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep='\t')
test_pair_idx = pd.read_csv(CHECKPOINT_DIR / "test_pair_index.tsv", sep='\t')

test_lgb = np.load(CHECKPOINT_DIR / "test_lgb.npy")
test_cat = np.load(CHECKPOINT_DIR / "test_cat.npy")
test_hgb = np.load(CHECKPOINT_DIR / "test_hgb.npy")

with open(BEST_MODEL_DIR / "blend_config.json", "r") as f:
    config = json.load(f)

w_lgb = config['w_lgb']
w_cat = config['w_cat']
w_hgb = config['w_hgb']
threshold = config['threshold']

test_probs = w_lgb * test_lgb + w_cat * test_cat + w_hgb * test_hgb
test_pair_idx['prob'] = test_probs

print(f"Loaded {len(test_s1)} test Source 1 entities and {len(test_pair_idx)} candidate pairs.")
print(f"Applying blend weights (LGB={w_lgb}, CAT={w_cat}, HGB={w_hgb}) with threshold {threshold:.2f}...")

matching_records = []
candidate_records = []

# Group predictions by source1_entity_id
cand_groups = test_pair_idx.groupby('s1_id')

for _, row in test_s1.iterrows():
    s1_id = str(row['entity_id'])
    
    if s1_id in cand_groups.groups:
        grp = cand_groups.get_group(s1_id)
        cands = list(grp['target_id'].astype(str))
        cand_str = " ".join(cands)
        
        matches = list(grp[grp['prob'] >= threshold]['target_id'].astype(str))
        match_str = " ".join(matches)
    else:
        cand_str = ""
        match_str = ""
        
    candidate_records.append({
        'source1_entity_id': s1_id,
        'candidate_entity_ids': cand_str
    })
    
    matching_records.append({
        'source1_entity_id': s1_id,
        'matched_entity_ids': match_str
    })

match_df = pd.DataFrame(matching_records)
cand_df = pd.DataFrame(candidate_records)

match_path = OUTPUT_DIR / "matching_results.tsv"
cand_path = OUTPUT_DIR / "candidate_pairs.tsv"

match_df.to_csv(match_path, sep='\t', index=False)
cand_df.to_csv(cand_path, sep='\t', index=False)

print(f"Generated {match_path} ({len(match_df)} rows)")
print(f"Generated {cand_path} ({len(cand_df)} rows)")

perf_log = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/performance_log.txt")
with open(perf_log, "a") as f:
    f.write(f"Phase 6 Final Inference | Generated matching_results.tsv & candidate_pairs.tsv for {len(match_df)} test entities.\n")
