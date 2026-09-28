import sys
import os
import re
import time
import zipfile
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from rapidfuzz import fuzz

BASE_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution")
TEST_DIR = BASE_DIR / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== STARTING FAST HIGH-SCALE OFFICIAL PIPELINE ===")
start_time = time.time()

# 1. Load Test Data
print("Loading Test Source 1, Source 2, and Source 3...")
t0 = time.time()
test_s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep='\t', keep_default_na=False)
test_s2 = pd.read_csv(TEST_DIR / "test_source2.tsv", sep='\t', keep_default_na=False)
test_s3 = pd.read_csv(TEST_DIR / "test_source3.tsv", sep='\t', keep_default_na=False)

print(f"Loaded Test S1: {len(test_s1):,}, Test S2: {len(test_s2):,}, Test S3: {len(test_s3):,} in {time.time()-t0:.2f}s")

# 2. Text Normalization
LEGAL_REGEX = re.compile(r'\b(inc|llc|corp|corporation|co|ltd|limited|pvt|private|sas|sarl|sa|eurl|group)\b', re.I)

def clean_name(s):
    if not s: return ""
    s = LEGAL_REGEX.sub('', s.lower())
    return re.sub(r'[^\w\s]', '', s).strip()

def get_block_keys(name):
    tokens = [t for t in clean_name(name).split() if len(t) > 2]
    keys = set()
    if not tokens:
        return keys
    # First 2 tokens key
    if len(tokens) >= 2:
        keys.add(tokens[0] + "_" + tokens[1])
    keys.add(tokens[0])
    return keys

# 3. Build Blocking Index on Target Pool (Source 2 + Source 3)
print("Building Target Pool Inverted Blocking Index...")
t0 = time.time()
target_df = pd.concat([test_s2, test_s3], ignore_index=True)

# Inverted index: block_key -> list of target entity_ids
block_index = defaultdict(list)
target_names = {}
target_addrs = {}
target_countries = {}

for _, row in target_df.iterrows():
    t_id = str(row['entity_id'])
    b_name = str(row['business_name'])
    b_addr = str(row['business_address'])
    cntry = str(row['country'])
    
    target_names[t_id] = b_name
    target_addrs[t_id] = b_addr
    target_countries[t_id] = cntry
    
    for key in get_block_keys(b_name):
        block_index[key].append(t_id)

print(f"Built Index across {len(block_index):,} block keys in {time.time()-t0:.2f}s")

# 4. Process Source 1 Test Entities & Match Scoring
print("Processing Test Source 1 entities and generating matches...")
t0 = time.time()

matching_rows = []
candidate_rows = []

batch_size = 100000
total_s1 = len(test_s1)

for start_idx in range(0, total_s1, batch_size):
    chunk_s1 = test_s1.iloc[start_idx:start_idx+batch_size]
    print(f"Processing chunk {start_idx:,} to {min(start_idx+batch_size, total_s1):,} / {total_s1:,}...")
    
    for _, row in chunk_s1.iterrows():
        s1_id = str(row['entity_id'])
        s1_name = str(row['business_name'])
        s1_addr = str(row['business_address'])
        s1_cntry = str(row['country'])
        
        # Candidate Retrieval
        keys = get_block_keys(s1_name)
        cand_ids = set()
        for k in keys:
            cand_ids.update(block_index.get(k, []))
            if len(cand_ids) > 100:
                break
                
        cand_list = list(cand_ids)[:50]
        cands_str = ",".join(cand_list)
        candidate_rows.append(f"{s1_id}\t{cands_str}\n")
        
        # Match Scoring
        s1_clean = clean_name(s1_name)
        matched_list = []
        
        for c_id in cand_list:
            t_name = target_names.get(c_id, "")
            t_clean = clean_name(t_name)
            
            # Country agreement check
            if s1_cntry and target_countries.get(c_id, "") and s1_cntry != target_countries.get(c_id, ""):
                continue
                
            # Token Sort Ratio
            sim_score = fuzz.token_sort_ratio(s1_clean, t_clean) / 100.0
            
            # High precision threshold
            if sim_score >= 0.82:
                matched_list.append(c_id)
                
        matches_str = ",".join(matched_list)
        matching_rows.append(f"{s1_id}\t{matches_str}\n")

# 5. Write Output Files
print("Writing matching_results.tsv and candidate_pairs.tsv...")
match_path = OUTPUT_DIR / "matching_results.tsv"
cand_path = OUTPUT_DIR / "candidate_pairs.tsv"

with open(match_path, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    f.writelines(matching_rows)
    
with open(cand_path, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    f.writelines(candidate_rows)

print(f"Generated {match_path} ({len(matching_rows):,} rows)")
print(f"Generated {cand_path} ({len(candidate_rows):,} rows)")

# 6. Validate Submission
print("Running Official Submission Validator...")
val_script = BASE_DIR / "utils" / "validate_submission.py"
val_res = subprocess.run([
    sys.executable, str(val_script),
    "--matching", str(match_path),
    "--candidate", str(cand_path),
    "--test-dir", str(TEST_DIR)
], cwd=BASE_DIR)

if val_res.returncode == 0:
    print("Validator PASS!")
else:
    print("Validator Failed!")
    sys.exit(1)

# 7. Package Zip Submission
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

print(f"=== PIPELINE COMPLETE IN {time.time()-start_time:.2f}s ===")
