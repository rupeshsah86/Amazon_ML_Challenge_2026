import os
import re
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Prevent PyTorch / OpenMP multi-threading segfaults on macOS
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
torch.set_num_threads(1)

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import lightgbm as lgb

from utils.metrics import calculate_macro_f05, parse_matched_ids

DATASET_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/dataset")
TRAIN_DIR = DATASET_DIR / "train"
CHECKPOINT_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

s1_df = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep='\t')
s2_df = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep='\t')
s3_df = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep='\t')
gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep='\t')

gt_map = {row['source1_entity_id']: parse_matched_ids(row['matched_entity_ids']) for _, row in gt_df.iterrows()}

s2_df['source_type'] = 's2'
s3_df['source_type'] = 's3'
target_df = pd.concat([s2_df, s3_df], ignore_index=True)

LEGAL_SUFFIXES = r'\b(inc|inc\.|llc|corp|corporation|co\.|ltd|limited|pvt ltd|private limited|sas|sarl|sa|eurl|group)\b'

def normalize_name(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(LEGAL_SUFFIXES, '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def normalize_address(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    replacements = {
        r'\bstreet\b': 'st', r'\broad\b': 'rd', r'\bavenue\b': 'ave',
        r'\bboulevard\b': 'blvd', r'\bparkway\b': 'pkwy', r'\blane\b': 'ln',
        r'\bsuite\b': 'ste', r'\bfloor\b': 'fl', r'\bplot\b': 'plt'
    }
    for k, v in replacements.items():
        text = re.sub(k, v, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def extract_numbers(text):
    return set(re.findall(r'\b\d+\b', str(text)))

s1_df['norm_name'] = s1_df['business_name'].apply(normalize_name)
target_df['norm_name'] = target_df['business_name'].apply(normalize_name)

s1_df['norm_addr'] = s1_df['business_address'].apply(normalize_address)
target_df['norm_addr'] = target_df['business_address'].apply(normalize_address)

s1_df['numbers_name'] = s1_df['business_name'].apply(extract_numbers)
target_df['numbers_name'] = target_df['business_name'].apply(extract_numbers)

s1_df['numbers_addr'] = s1_df['business_address'].apply(extract_numbers)
target_df['numbers_addr'] = target_df['business_address'].apply(extract_numbers)

print("Loading Sentence Transformer Model (all-MiniLM-L6-v2) for Dense Embeddings...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute Dense Embeddings
s1_name_emb = embed_model.encode(s1_df['norm_name'].tolist(), batch_size=64, show_progress_bar=False)
tgt_name_emb = embed_model.encode(target_df['norm_name'].tolist(), batch_size=64, show_progress_bar=False)

s1_addr_emb = embed_model.encode(s1_df['norm_addr'].tolist(), batch_size=64, show_progress_bar=False)
tgt_addr_emb = embed_model.encode(target_df['norm_addr'].tolist(), batch_size=64, show_progress_bar=False)

def generate_dense_candidates(train_s1_indices, target_df, top_k_name=20, top_k_addr=15):
    s1_sub = s1_df.iloc[train_s1_indices].copy()
    
    vec_n = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
    vec_n.fit(pd.concat([s1_sub['norm_name'], target_df['norm_name']]))
    
    s1_n_mat = vec_n.transform(s1_sub['norm_name'])
    tgt_n_mat = vec_n.transform(target_df['norm_name'])
    
    vec_a = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
    vec_a.fit(pd.concat([s1_sub['norm_addr'], target_df['norm_addr']]))
    
    s1_a_mat = vec_a.transform(s1_sub['norm_addr'])
    tgt_a_mat = vec_a.transform(target_df['norm_addr'])
    
    candidates = []
    
    for idx, (s1_row_idx, s1_row) in enumerate(s1_sub.iterrows()):
        s1_id = s1_row['entity_id']
        
        sim_n = cosine_similarity(s1_n_mat[idx], tgt_n_mat).ravel()
        top_n = np.argsort(sim_n)[-top_k_name:]
        
        sim_a = cosine_similarity(s1_a_mat[idx], tgt_a_mat).ravel()
        top_a = np.argsort(sim_a)[-top_k_addr:]
        
        dense_sim_n = cosine_similarity([s1_name_emb[s1_row_idx]], tgt_name_emb).ravel()
        top_dense = np.argsort(dense_sim_n)[-top_k_name:]
        
        cand_idxs = set(top_n).union(set(top_a)).union(set(top_dense))
        
        for t_idx in cand_idxs:
            t_row = target_df.iloc[t_idx]
            dense_sim_a = float(np.dot(s1_addr_emb[s1_row_idx], tgt_addr_emb[t_idx]) / (np.linalg.norm(s1_addr_emb[s1_row_idx]) * np.linalg.norm(tgt_addr_emb[t_idx]) + 1e-5))
            
            candidates.append({
                's1_id': s1_id,
                'target_id': t_row['entity_id'],
                's1_row_idx': s1_row_idx,
                'target_row_idx': t_idx,
                'sim_tfidf_name': sim_n[t_idx],
                'sim_tfidf_addr': sim_a[t_idx],
                'sim_dense_name': float(dense_sim_n[t_idx]),
                'sim_dense_addr': dense_sim_a,
            })
            
    return pd.DataFrame(candidates)

def compute_dense_features(cand_df):
    cand_df = cand_df.copy()
    s1_rows = s1_df.iloc[cand_df['s1_row_idx']].reset_index(drop=True)
    tgt_rows = target_df.iloc[cand_df['target_row_idx']].reset_index(drop=True)
    
    name_ratios, name_partial, name_sort, name_set = [], [], [], []
    addr_ratios, addr_sort = [], []
    num_match_n, num_match_a, c_match = [], [], []
    
    for s1_n, tgt_n, s1_a, tgt_a, s1_num_n, tgt_num_n, s1_num_a, tgt_num_a, s1_c, tgt_c in zip(
        s1_rows['norm_name'], tgt_rows['norm_name'],
        s1_rows['norm_addr'], tgt_rows['norm_addr'],
        s1_rows['numbers_name'], tgt_rows['numbers_name'],
        s1_rows['numbers_addr'], tgt_rows['numbers_addr'],
        s1_rows['country'], tgt_rows['country']
    ):
        name_ratios.append(fuzz.ratio(s1_n, tgt_n) / 100.0)
        name_partial.append(fuzz.partial_ratio(s1_n, tgt_n) / 100.0)
        name_sort.append(fuzz.token_sort_ratio(s1_n, tgt_n) / 100.0)
        name_set.append(fuzz.token_set_ratio(s1_n, tgt_n) / 100.0)
        
        addr_ratios.append(fuzz.ratio(s1_a, tgt_a) / 100.0)
        addr_sort.append(fuzz.token_sort_ratio(s1_a, tgt_a) / 100.0)
        
        n_overlap = len(s1_num_n.intersection(tgt_num_n)) > 0 if (s1_num_n and tgt_num_n) else 1.0
        a_overlap = len(s1_num_a.intersection(tgt_num_a)) > 0 if (s1_num_a and tgt_num_a) else 1.0
        
        num_match_n.append(float(n_overlap))
        num_match_a.append(float(a_overlap))
        c_match.append(1.0 if s1_c == tgt_c else 0.0)
        
    cand_df['fuzz_name_ratio'] = name_ratios
    cand_df['fuzz_name_partial'] = name_partial
    cand_df['fuzz_name_sort'] = name_sort
    cand_df['fuzz_name_set'] = name_set
    
    cand_df['fuzz_addr_ratio'] = addr_ratios
    cand_df['fuzz_addr_sort'] = addr_sort
    
    cand_df['num_match_name'] = num_match_n
    cand_df['num_match_addr'] = num_match_a
    cand_df['country_match'] = c_match
    
    cand_df['len_ratio_name'] = np.minimum(s1_rows['norm_name'].str.len() / (tgt_rows['norm_name'].str.len() + 1e-5),
                                           tgt_rows['norm_name'].str.len() / (s1_rows['norm_name'].str.len() + 1e-5))
    cand_df['len_ratio_addr'] = np.minimum(s1_rows['norm_addr'].str.len() / (tgt_rows['norm_addr'].str.len() + 1e-5),
                                           tgt_rows['norm_addr'].str.len() / (s1_rows['norm_addr'].str.len() + 1e-5))
                                           
    labels = []
    for s1_id, t_id in zip(cand_df['s1_id'], cand_df['target_id']):
        labels.append(1 if t_id in gt_map.get(s1_id, set()) else 0)
    cand_df['label'] = labels
    return cand_df

print("Building and validating Advanced Features + Neural Embeddings with 5-Fold Cross Validation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
s1_indices = np.arange(len(s1_df))

oof_predictions = {}
fold_f05_scores = []

feature_cols = [
    'sim_tfidf_name', 'sim_tfidf_addr', 'sim_dense_name', 'sim_dense_addr',
    'fuzz_name_ratio', 'fuzz_name_partial', 'fuzz_name_sort', 'fuzz_name_set',
    'fuzz_addr_ratio', 'fuzz_addr_sort',
    'num_match_name', 'num_match_addr', 'country_match',
    'len_ratio_name', 'len_ratio_addr'
]

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(s1_indices)):
    val_s1_ids = set(s1_df.iloc[val_idx]['entity_id'])
    
    train_cands = generate_dense_candidates(train_idx, target_df)
    val_cands = generate_dense_candidates(val_idx, target_df)
    
    train_feat = compute_dense_features(train_cands)
    val_feat = compute_dense_features(val_cands)
    
    clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=1
    )
    clf.fit(train_feat[feature_cols], train_feat['label'])
    
    train_feat['pred_prob'] = clf.predict_proba(train_feat[feature_cols])[:, 1]
    val_feat['pred_prob'] = clf.predict_proba(val_feat[feature_cols])[:, 1]
    
    best_thresh = 0.5
    best_train_f05 = -1.0
    train_gt_sub = {s1_id: gt_map[s1_id] for s1_id in s1_df.iloc[train_idx]['entity_id']}
    
    for thresh in np.arange(0.2, 0.8, 0.05):
        train_pred_sub = {}
        for s1_id, group in train_feat.groupby('s1_id'):
            train_pred_sub[s1_id] = set(group[group['pred_prob'] >= thresh]['target_id'])
        score = calculate_macro_f05(train_pred_sub, train_gt_sub)
        if score > best_train_f05:
            best_train_f05 = score
            best_thresh = thresh
            
    val_gt_sub = {s1_id: gt_map[s1_id] for s1_id in val_s1_ids}
    val_pred_sub = {s1_id: set() for s1_id in val_s1_ids}
    
    for s1_id, group in val_feat.groupby('s1_id'):
        matched = set(group[group['pred_prob'] >= best_thresh]['target_id'])
        val_pred_sub[s1_id] = matched
        oof_predictions[s1_id] = matched
        
    val_f05 = calculate_macro_f05(val_pred_sub, val_gt_sub)
    fold_f05_scores.append(val_f05)
    print(f"Fold {fold+1} Held-out Macro F0.5: {val_f05:.4f} (Thresh: {best_thresh:.2f})")

overall_oof_f05 = calculate_macro_f05(oof_predictions, gt_map)
elapsed_time = time.time() - start_time

print(f"\n=== ADVANCED NEURAL FEATURE ENGINEERING COMPLETE ===")
print(f"5-Fold OOF Macro F0.5 Score: {overall_oof_f05:.4f}")
print(f"Mean Fold F0.5: {np.mean(fold_f05_scores):.4f} +/- {np.std(fold_f05_scores):.4f}")
print(f"Execution Time: {elapsed_time:.2f} seconds")

# Save feature metadata and checkpoint
with open(CHECKPOINT_DIR / "features_oof.pkl", "wb") as f:
    pickle.dump({"oof_predictions": oof_predictions, "macro_f05": overall_oof_f05, "feature_cols": feature_cols}, f)

perf_log = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/performance_log.txt")
with open(perf_log, "a") as f:
    f.write(f"Phase 2 Neural Features | 5-Fold OOF Macro F0.5: {overall_oof_f05:.4f} | Features: {len(feature_cols)} | Time: {elapsed_time:.2f}s\n")
