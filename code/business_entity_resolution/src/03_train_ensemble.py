import os
import re
import time
import pickle
import json
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
from sklearn.ensemble import HistGradientBoostingClassifier
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
import catboost as cb

from utils.metrics import calculate_macro_f05, parse_matched_ids

DATASET_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/dataset")
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
CHECKPOINT_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/checkpoints")
BEST_MODEL_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/best_model")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
s1_df = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep='\t')
s2_df = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep='\t')
s3_df = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep='\t')
gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep='\t')

test_s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep='\t')
test_s2 = pd.read_csv(TEST_DIR / "test_source2.tsv", sep='\t')
test_s3 = pd.read_csv(TEST_DIR / "test_source3.tsv", sep='\t')

gt_map = {row['source1_entity_id']: parse_matched_ids(row['matched_entity_ids']) for _, row in gt_df.iterrows()}

s2_df['source_type'] = 's2'
s3_df['source_type'] = 's3'
train_target = pd.concat([s2_df, s3_df], ignore_index=True)

test_s2['source_type'] = 's2'
test_s3['source_type'] = 's3'
test_target = pd.concat([test_s2, test_s3], ignore_index=True)

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

# Pre-compute fields
for df in [s1_df, train_target, test_s1, test_target]:
    df['norm_name'] = df['business_name'].apply(normalize_name)
    df['norm_addr'] = df['business_address'].apply(normalize_address)
    df['numbers_name'] = df['business_name'].apply(extract_numbers)
    df['numbers_addr'] = df['business_address'].apply(extract_numbers)

print("Loading Sentence Transformer Model (all-MiniLM-L6-v2) for Dense Embeddings...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute Dense Embeddings
s1_name_emb = embed_model.encode(s1_df['norm_name'].tolist(), batch_size=64, show_progress_bar=False)
train_tgt_name_emb = embed_model.encode(train_target['norm_name'].tolist(), batch_size=64, show_progress_bar=False)

s1_addr_emb = embed_model.encode(s1_df['norm_addr'].tolist(), batch_size=64, show_progress_bar=False)
train_tgt_addr_emb = embed_model.encode(train_target['norm_addr'].tolist(), batch_size=64, show_progress_bar=False)

test_s1_name_emb = embed_model.encode(test_s1['norm_name'].tolist(), batch_size=64, show_progress_bar=False)
test_tgt_name_emb = embed_model.encode(test_target['norm_name'].tolist(), batch_size=64, show_progress_bar=False)

test_s1_addr_emb = embed_model.encode(test_s1['norm_addr'].tolist(), batch_size=64, show_progress_bar=False)
test_tgt_addr_emb = embed_model.encode(test_target['norm_addr'].tolist(), batch_size=64, show_progress_bar=False)

def generate_candidates(s1_sub, target_sub, is_test=False, top_k_name=20, top_k_addr=15):
    vec_n = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
    vec_n.fit(pd.concat([s1_sub['norm_name'], target_sub['norm_name']]))
    
    s1_n_mat = vec_n.transform(s1_sub['norm_name'])
    tgt_n_mat = vec_n.transform(target_sub['norm_name'])
    
    vec_a = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
    vec_a.fit(pd.concat([s1_sub['norm_addr'], target_sub['norm_addr']]))
    
    s1_a_mat = vec_a.transform(s1_sub['norm_addr'])
    tgt_a_mat = vec_a.transform(target_sub['norm_addr'])
    
    s1_n_e = test_s1_name_emb if is_test else s1_name_emb
    tgt_n_e = test_tgt_name_emb if is_test else train_tgt_name_emb
    
    s1_a_e = test_s1_addr_emb if is_test else s1_addr_emb
    tgt_a_e = test_tgt_addr_emb if is_test else train_tgt_addr_emb
    
    candidates = []
    for idx, (s1_row_idx, s1_row) in enumerate(s1_sub.iterrows()):
        s1_id = s1_row['entity_id']
        sim_n = cosine_similarity(s1_n_mat[idx], tgt_n_mat).ravel()
        top_n = np.argsort(sim_n)[-top_k_name:]
        
        sim_a = cosine_similarity(s1_a_mat[idx], tgt_a_mat).ravel()
        top_a = np.argsort(sim_a)[-top_k_addr:]
        
        dense_sim_n = cosine_similarity([s1_n_e[s1_row_idx]], tgt_n_e).ravel()
        top_dense = np.argsort(dense_sim_n)[-top_k_name:]
        
        cand_idxs = set(top_n).union(set(top_a)).union(set(top_dense))
        for t_idx in cand_idxs:
            t_row = target_sub.iloc[t_idx]
            dense_sim_a = float(np.dot(s1_a_e[s1_row_idx], tgt_a_e[t_idx]) / (np.linalg.norm(s1_a_e[s1_row_idx]) * np.linalg.norm(tgt_a_e[t_idx]) + 1e-5))
            
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

def extract_features(cand_df, s1_base, target_base, is_train=True):
    cand_df = cand_df.copy()
    s1_rows = s1_base.iloc[cand_df['s1_row_idx']].reset_index(drop=True)
    tgt_rows = target_base.iloc[cand_df['target_row_idx']].reset_index(drop=True)
    
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
                                           
    if is_train:
        labels = []
        for s1_id, t_id in zip(cand_df['s1_id'], cand_df['target_id']):
            labels.append(1 if t_id in gt_map.get(s1_id, set()) else 0)
        cand_df['label'] = labels
    return cand_df

print("=== Starting Phase 3: Diverse Neural Multi-Model Ensemble Training ===")

feature_cols = [
    'sim_tfidf_name', 'sim_tfidf_addr', 'sim_dense_name', 'sim_dense_addr',
    'fuzz_name_ratio', 'fuzz_name_partial', 'fuzz_name_sort', 'fuzz_name_set',
    'fuzz_addr_ratio', 'fuzz_addr_sort',
    'num_match_name', 'num_match_addr', 'country_match',
    'len_ratio_name', 'len_ratio_addr'
]

print("Generating test candidate pairs and features...")
test_cands = generate_candidates(test_s1, test_target, is_test=True)
test_feat = extract_features(test_cands, test_s1, test_target, is_train=False)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
s1_indices = np.arange(len(s1_df))

oof_lgb_probs = []
oof_cat_probs = []
oof_hgb_probs = []
oof_pair_records = []

test_lgb_preds = np.zeros(len(test_feat))
test_cat_preds = np.zeros(len(test_feat))
test_hgb_preds = np.zeros(len(test_feat))

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(s1_indices)):
    print(f"\n--- Training Fold {fold+1}/5 ---")
    s1_train_sub = s1_df.iloc[train_idx].reset_index(drop=True)
    s1_val_sub = s1_df.iloc[val_idx].reset_index(drop=True)
    
    train_cands = generate_candidates(s1_train_sub, train_target, is_test=False)
    val_cands = generate_candidates(s1_val_sub, train_target, is_test=False)
    
    train_feat = extract_features(train_cands, s1_train_sub, train_target, is_train=True)
    val_feat = extract_features(val_cands, s1_val_sub, train_target, is_train=True)
    
    X_train, y_train = train_feat[feature_cols], train_feat['label']
    X_val, y_val = val_feat[feature_cols], val_feat['label']
    
    # 1. LightGBM
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42+fold, verbose=-1, n_jobs=1
    )
    clf_lgb.fit(X_train, y_train)
    p_val_lgb = clf_lgb.predict_proba(X_val)[:, 1]
    test_lgb_preds += clf_lgb.predict_proba(test_feat[feature_cols])[:, 1] / 5.0
    
    # 2. CatBoost
    clf_cat = cb.CatBoostClassifier(
        iterations=500, learning_rate=0.04, depth=6, random_seed=42+fold, verbose=0, thread_count=1
    )
    clf_cat.fit(X_train, y_train)
    p_val_cat = clf_cat.predict_proba(X_val)[:, 1]
    test_cat_preds += clf_cat.predict_proba(test_feat[feature_cols])[:, 1] / 5.0
    
    # 3. HistGradientBoosting
    clf_hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.03, max_depth=6, random_state=42+fold
    )
    clf_hgb.fit(X_train, y_train)
    p_val_hgb = clf_hgb.predict_proba(X_val)[:, 1]
    test_hgb_preds += clf_hgb.predict_proba(test_feat[feature_cols])[:, 1] / 5.0
    
    oof_lgb_probs.extend(p_val_lgb)
    oof_cat_probs.extend(p_val_cat)
    oof_hgb_probs.extend(p_val_hgb)
    
    for _, row in val_feat.iterrows():
        oof_pair_records.append({
            's1_id': row['s1_id'],
            'target_id': row['target_id'],
            'label': row['label'],
            'fold': fold+1
        })
        
    val_gt_sub = {s1_id: gt_map[s1_id] for s1_id in s1_val_sub['entity_id']}
    
    def eval_sub_model(probs, name):
        val_feat['p_temp'] = probs
        best_t, best_s = 0.5, -1.0
        for t in np.arange(0.2, 0.8, 0.05):
            pred_sub = {s1_id: set() for s1_id in s1_val_sub['entity_id']}
            for s1_id, grp in val_feat.groupby('s1_id'):
                pred_sub[s1_id] = set(grp[grp['p_temp'] >= t]['target_id'])
            sc = calculate_macro_f05(pred_sub, val_gt_sub)
            if sc > best_s: best_s, best_t = sc, t
        print(f"Fold {fold+1} {name} Macro F0.5: {best_s:.4f} (Thresh: {best_t:.2f})")
        
    eval_sub_model(p_val_lgb, "LightGBM")
    eval_sub_model(p_val_cat, "CatBoost")
    eval_sub_model(p_val_hgb, "HistGB")

elapsed_time = time.time() - start_time

oof_df = pd.DataFrame(oof_pair_records)
oof_df['lgb_prob'] = oof_lgb_probs
oof_df['cat_prob'] = oof_cat_probs
oof_df['hgb_prob'] = oof_hgb_probs

np.save(CHECKPOINT_DIR / "oof_lgb.npy", np.array(oof_lgb_probs))
np.save(CHECKPOINT_DIR / "oof_cat.npy", np.array(oof_cat_probs))
np.save(CHECKPOINT_DIR / "oof_hgb.npy", np.array(oof_hgb_probs))

np.save(CHECKPOINT_DIR / "test_lgb.npy", test_lgb_preds)
np.save(CHECKPOINT_DIR / "test_cat.npy", test_cat_preds)
np.save(CHECKPOINT_DIR / "test_hgb.npy", test_hgb_preds)

oof_df.to_csv(CHECKPOINT_DIR / "oof_pair_index.tsv", sep='\t', index=False)
test_feat[['s1_id', 'target_id']].to_csv(CHECKPOINT_DIR / "test_pair_index.tsv", sep='\t', index=False)

print(f"\n=== MULTI-MODEL ENSEMBLE TRAINING COMPLETE ===")
print(f"Execution Time: {elapsed_time:.2f} seconds")

perf_log = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/performance_log.txt")
with open(perf_log, "a") as f:
    f.write(f"Phase 3 Ensemble | Models: LightGBM, CatBoost, HistGB | Pairs: {len(oof_df)} | Time: {elapsed_time:.2f}s\n")
