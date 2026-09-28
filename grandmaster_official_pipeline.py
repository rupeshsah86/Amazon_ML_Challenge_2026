import sys
import os
import re
import time
import zipfile
import subprocess
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution")
TEST_DIR = BASE_DIR / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== STARTING GRANDMASTER HIGH-PRECISION OFFICIAL PIPELINE ===")
start_time = time.time()

# 1. Load Test TSVs
print("Loading Test Source 1, Source 2, and Source 3...")
t0 = time.time()
test_s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep='\t', keep_default_na=False, dtype=str)
test_s2 = pd.read_csv(TEST_DIR / "test_source2.tsv", sep='\t', keep_default_na=False, dtype=str)
test_s3 = pd.read_csv(TEST_DIR / "test_source3.tsv", sep='\t', keep_default_na=False, dtype=str)

print(f"Loaded Test S1: {len(test_s1):,}, Test S2: {len(test_s2):,}, Test S3: {len(test_s3):,} in {time.time()-t0:.2f}s")

# Combine Target Pool
target_df = pd.concat([test_s2, test_s3], ignore_index=True)

LEGAL_REGEX = re.compile(r'\b(inc|llc|corp|corporation|co|ltd|limited|pvt|private|sas|sarl|sa|eurl|group)\b', re.I)

def get_tokens(s):
    if not s: return []
    clean_s = re.sub(r'[^\w\s]', ' ', LEGAL_REGEX.sub('', str(s).lower()))
    return [t for t in clean_s.split() if len(t) >= 3 and t not in {'the', 'and', 'for', 'near', 'opp', 'road', 'street', 'main'}]

# 2. Build Fast Multi-Token Inverted Index on Target Pool
print("Building Target Pool Token Index...")
t0 = time.time()

target_ids = target_df['entity_id'].values
target_names = target_df['business_name'].values
target_addrs = target_df['business_address'].values
target_countries = target_df['country'].values

# Index token -> list of target row indices
token_index = defaultdict(list)
for idx, (b_name, b_addr) in enumerate(zip(target_names, target_addrs)):
    n_toks = get_tokens(b_name)
    a_toks = get_tokens(b_addr)
    seen = set()
    for t in n_toks[:4] + a_toks[:2]:
        if t not in seen:
            token_index[t].append(idx)
            seen.add(t)

print(f"Built Inverted Token Index ({len(token_index):,} tokens) in {time.time()-t0:.2f}s")

# 3. High-Recall Candidate Retrieval & High-Precision Matcher
print("Running High-Precision Entity Resolution for 1.73M Test Records...")
t0 = time.time()

s1_ids = test_s1['entity_id'].values
s1_names = test_s1['business_name'].values
s1_addrs = test_s1['business_address'].values
s1_countries = test_s1['country'].values

matching_lines = ["source1_entity_id\tmatched_entity_ids\n"]
candidate_lines = ["source1_entity_id\tcandidate_entity_ids\n"]

total_records = len(test_s1)

for idx in range(total_records):
    s1_id = s1_ids[idx]
    s1_n = s1_names[idx]
    s1_a = s1_addrs[idx]
    s1_c = s1_countries[idx]
    
    n_toks = set(get_tokens(s1_n))
    a_toks = set(get_tokens(s1_a))
    
    cand_counts = defaultdict(int)
    for t in n_toks:
        for tgt_idx in token_index.get(t, [])[:500]:
            cand_counts[tgt_idx] += 2  # Higher weight for name token match
            
    for t in a_toks:
        for tgt_idx in token_index.get(t, [])[:300]:
            cand_counts[tgt_idx] += 1  # Weight for address token match
            
    if not cand_counts:
        candidate_lines.append(f"{s1_id}\t\n")
        matching_lines.append(f"{s1_id}\t\n")
        continue
        
    # Sort candidates by shared token score
    top_cands = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:25]
    
    cand_id_list = [target_ids[tgt_idx] for tgt_idx, score in top_cands]
    cands_str = ",".join(cand_id_list)
    candidate_lines.append(f"{s1_id}\t{cands_str}\n")
    
    # High Precision Matching Filter
    matched_id_list = []
    s1_n_set = set(get_tokens(s1_n))
    
    for tgt_idx, score in top_cands:
        # Country agreement check
        if s1_c and target_countries[tgt_idx] and s1_c != target_countries[tgt_idx]:
            continue
            
        t_n_set = set(get_tokens(target_names[tgt_idx]))
        
        # Jaccard / Overlap Precision Metric
        if s1_n_set and t_n_set:
            overlap = len(s1_n_set.intersection(t_n_set))
            min_len = min(len(s1_n_set), len(t_n_set))
            
            # High-precision threshold: at least 70% shared significant tokens or 2+ shared unique tokens
            if overlap >= 2 or (min_len > 0 and overlap / min_len >= 0.70):
                matched_id_list.append(target_ids[tgt_idx])
                
    matches_str = ",".join(matched_id_list)
    matching_lines.append(f"{s1_id}\t{matches_str}\n")
    
    if (idx + 1) % 500000 == 0:
        print(f"Processed {idx+1:,} / {total_records:,} records ({time.time()-t0:.2f}s)...")

print(f"Processed all {total_records:,} records in {time.time()-t0:.2f}s")

# 4. Save Output Files
match_path = OUTPUT_DIR / "matching_results.tsv"
cand_path = OUTPUT_DIR / "candidate_pairs.tsv"

with open(match_path, "w", encoding="utf-8") as f:
    f.writelines(matching_lines)

with open(cand_path, "w", encoding="utf-8") as f:
    f.writelines(candidate_lines)

print(f"Saved {match_path}")
print(f"Saved {cand_path}")

# 5. Run Official Validator
print("Running Official Submission Validator...")
val_script = BASE_DIR / "utils" / "validate_submission.py"
val_res = subprocess.run([
    sys.executable, str(val_script),
    "--matching", str(match_path),
    "--candidate", str(cand_path),
    "--test-dir", str(TEST_DIR)
], cwd=BASE_DIR)

if val_res.returncode == 0:
    print("Validator PASSED!")
else:
    print("Validator Failed!")
    sys.exit(1)

# 6. Create Submission ZIP
zip_path = BASE_DIR / "N3Rflix_submission.zip"
print(f"Creating Submission ZIP Package: {zip_path}")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(match_path, "output/matching_results.tsv")
    zipf.write(cand_path, "output/candidate_pairs.tsv")
    if (BASE_DIR / "Documentation_template.md").exists():
        zipf.write(BASE_DIR / "Documentation_template.md", "Documentation_template.md")
    
    code_dir = BASE_DIR / "code" / "business_entity_resolution"
    for root, dirs, files in os.walk(code_dir):
        for file in files:
            full_p = Path(root) / file
            rel_p = full_p.relative_to(BASE_DIR)
            zipf.write(full_p, rel_p)

print(f"=== GRANDMASTER PIPELINE COMPLETE IN {time.time()-start_time:.2f}s ===")
