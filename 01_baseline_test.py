import os
import sys
import time
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb

from utils.metrics import calculate_macro_f05, parse_matched_ids

DATASET_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/dataset")
TRAIN_DIR = DATASET_DIR / "train"
CHECKPOINT_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Load training dataset
s1_df = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep='\t')
s2_df = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep='\t')
s3_df = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep='\t')
gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep='\t')

gt_map = {row['source1_entity_id']: parse_matched_ids(row['matched_entity_ids']) for _, row in gt_df.iterrows()}

print(f"Loaded Train: Source 1={len(s1_df)}, Source 2={len(s2_df)}, Source 3={len(s3_df)}")

# Combine Source 2 & Source 3 as Target Pool
s2_df['source_type'] = 's2'
s3_df['source_type'] = 's3'
target_df = pd.concat([s2_df, s3_df], ignore_index=True)

# Basic Normalization helper
def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower().replace('.', ' ').replace(',', ' ').replace('-', ' ').strip()

s1_df['clean_name'] = s1_df['business_name'].apply(clean_text)
target_df['clean_name'] = target_df['business_name'].apply(clean_text)

s1_df['clean_addr'] = s1_df['business_address'].apply(clean_text)
target_df['clean_addr'] = target_df['business_address'].apply(clean_text)

# Candidate Generation (Blocking) Strategy
def generate_candidates_for_fold(train_s1_indices, target_df, top_k_name=15, top_k_addr=10):
    """
    Multi-channel candidate retrieval:
    1. Name TF-IDF Char n-gram similarity (Top K)
    2. Address TF-IDF Word n-gram similarity (Top K)
    """
    s1_sub = s1_df.iloc[train_s1_indices].copy()
    
    # 1. Name Vectorizer
    vectorizer_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=1)
    vectorizer_name.fit(pd.concat([s1_sub['clean_name'], target_df['clean_name']]))
    
    s1_name_mat = vectorizer_name.transform(s1_sub['clean_name'])
    target_name_mat = vectorizer_name.transform(target_df['clean_name'])
    
    # 2. Address Vectorizer
    vectorizer_addr = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
    vectorizer_addr.fit(pd.concat([s1_sub['clean_addr'], target_df['clean_addr']]))
    
    s1_addr_mat = vectorizer_addr.transform(s1_sub['clean_addr'])
    target_addr_mat = vectorizer_addr.transform(target_df['clean_addr'])
    
    candidate_pairs = []
    
    for idx, (s1_row_idx, s1_row) in enumerate(s1_sub.iterrows()):
        s1_id = s1_row['entity_id']
        
        # Name Cosine Sim
        sim_name = cosine_similarity(s1_name_mat[idx], target_name_mat).ravel()
        top_name_idxs = np.argsort(sim_name)[-top_k_name:]
        
        # Addr Cosine Sim
        sim_addr = cosine_similarity(s1_addr_mat[idx], target_addr_mat).ravel()
        top_addr_idxs = np.argsort(sim_addr)[-top_k_addr:]
        
        cand_target_idxs = set(top_name_idxs).union(set(top_addr_idxs))
        
        for t_idx in cand_target_idxs:
            t_row = target_df.iloc[t_idx]
            candidate_pairs.append({
                's1_id': s1_id,
                'target_id': t_row['entity_id'],
                's1_row_idx': s1_row_idx,
                'target_row_idx': t_idx,
                'sim_name': sim_name[t_idx],
                'sim_addr': sim_addr[t_idx],
            })
            
    cand_df = pd.DataFrame(candidate_pairs)
    return cand_df, vectorizer_name, vectorizer_addr

# Measure candidate recall on full train dataset
print("Evaluating Baseline Candidate Generation Recall...")
full_s1_indices = np.arange(len(s1_df))
all_cands_df, _, _ = generate_candidates_for_fold(full_s1_indices, target_df)

cand_map = all_cands_df.groupby('s1_id')['target_id'].apply(set).to_dict()
retained_matches = 0
total_gt_matches = sum(len(v) for v in gt_map.values())

for s1_id, true_matches in gt_map.items():
    cands = cand_map.get(s1_id, set())
    retained_matches += len(true_matches.intersection(cands))

cand_recall = retained_matches / total_gt_matches if total_gt_matches > 0 else 1.0
avg_cands = len(all_cands_df) / len(s1_df)

print(f"Candidate Recall: {cand_recall:.4f} ({retained_matches}/{total_gt_matches})")
print(f"Average Candidates per Source 1 Entity: {avg_cands:.2f}")

# Grouped 5-Fold Cross Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
s1_indices = np.arange(len(s1_df))

oof_predictions = {}
fold_f05_scores = []

feature_cols = ['sim_name', 'sim_addr', 'country_match', 'len_diff_name', 'len_diff_addr']

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(s1_indices)):
    print(f"\n--- Fold {fold+1}/5 ---")
    val_s1_ids = set(s1_df.iloc[val_idx]['entity_id'])
    
    # Generate candidates for train and val splits
    train_cands, vec_n, vec_a = generate_candidates_for_fold(train_idx, target_df)
    val_cands, _, _ = generate_candidates_for_fold(val_idx, target_df)
    
    # Feature construction helper
    def extract_baseline_features(c_df):
        c_df = c_df.copy()
        s1_rows = s1_df.iloc[c_df['s1_row_idx']].reset_index(drop=True)
        t_rows = target_df.iloc[c_df['target_row_idx']].reset_index(drop=True)
        
        c_df['country_match'] = (s1_rows['country'].values == t_rows['country'].values).astype(float)
        c_df['len_diff_name'] = np.abs(s1_rows['clean_name'].str.len().values - t_rows['clean_name'].str.len().values)
        c_df['len_diff_addr'] = np.abs(s1_rows['clean_addr'].str.len().values - t_rows['clean_addr'].str.len().values)
        
        # Labels
        labels = []
        for s1_id, t_id in zip(c_df['s1_id'], c_df['target_id']):
            true_set = gt_map.get(s1_id, set())
            labels.append(1 if t_id in true_set else 0)
        c_df['label'] = labels
        return c_df

    train_feat = extract_baseline_features(train_cands)
    val_feat = extract_baseline_features(val_cands)
    
    # Train LightGBM model
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    clf.fit(train_feat[feature_cols], train_feat['label'])
    
    # Predict validation probabilities
    val_feat['pred_prob'] = clf.predict_proba(val_feat[feature_cols])[:, 1]
    
    # Leak-free threshold search on train set predictions
    train_feat['pred_prob'] = clf.predict_proba(train_feat[feature_cols])[:, 1]
    best_thresh = 0.5
    best_train_f05 = -1.0
    
    train_gt_sub = {s1_id: gt_map[s1_id] for s1_id in s1_df.iloc[train_idx]['entity_id']}
    
    for thresh in np.arange(0.2, 0.8, 0.05):
        train_pred_sub = {}
        for s1_id, group in train_feat.groupby('s1_id'):
            matched = set(group[group['pred_prob'] >= thresh]['target_id'])
            train_pred_sub[s1_id] = matched
        score = calculate_macro_f05(train_pred_sub, train_gt_sub)
        if score > best_train_f05:
            best_train_f05 = score
            best_thresh = thresh
            
    print(f"Fold {fold+1} Best Train Threshold: {best_thresh:.2f} (Train F0.5: {best_train_f05:.4f})")
    
    # Apply tuned threshold to held-out validation set
    val_gt_sub = {s1_id: gt_map[s1_id] for s1_id in val_s1_ids}
    val_pred_sub = {}
    for s1_id in val_s1_ids:
        val_pred_sub[s1_id] = set()
        
    for s1_id, group in val_feat.groupby('s1_id'):
        matched = set(group[group['pred_prob'] >= best_thresh]['target_id'])
        val_pred_sub[s1_id] = matched
        oof_predictions[s1_id] = matched
        
    val_f05 = calculate_macro_f05(val_pred_sub, val_gt_sub)
    fold_f05_scores.append(val_f05)
    print(f"Fold {fold+1} Held-out Validation Macro F0.5: {val_f05:.4f}")

overall_oof_f05 = calculate_macro_f05(oof_predictions, gt_map)
elapsed_time = time.time() - start_time

print(f"\n=== BASELINE EVALUATION COMPLETE ===")
print(f"5-Fold OOF Macro F0.5 Score: {overall_oof_f05:.4f}")
print(f"Mean Fold F0.5: {np.mean(fold_f05_scores):.4f} +/- {np.std(fold_f05_scores):.4f}")
print(f"Candidate Recall: {cand_recall:.4f}, Avg Candidates: {avg_cands:.2f}")
print(f"Execution Time: {elapsed_time:.2f} seconds")

# Save baseline results
with open(CHECKPOINT_DIR / "baseline_oof.pkl", "wb") as f:
    pickle.dump({"oof_predictions": oof_predictions, "macro_f05": overall_oof_f05, "cand_recall": cand_recall}, f)

perf_log = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/performance_log.txt")
with open(perf_log, "a") as f:
    f.write(f"Phase 1 Baseline | 5-Fold OOF Macro F0.5: {overall_oof_f05:.4f} | Cand Recall: {cand_recall:.4f} | Avg Cands: {avg_cands:.2f} | Time: {elapsed_time:.2f}s\n")
