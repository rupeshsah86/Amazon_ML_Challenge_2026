import sys
import os
import re
import time
import zipfile
import subprocess
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution")
TEST_DIR = BASE_DIR / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== STARTING ULTRA-FAST HIGH-SCALE OFFICIAL PIPELINE ===")
start_time = time.time()

# 1. Load Test Data
print("Loading Test Source 1, Source 2, and Source 3 TSVs...")
t0 = time.time()
test_s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep='\t', keep_default_na=False, dtype=str)
test_s2 = pd.read_csv(TEST_DIR / "test_source2.tsv", sep='\t', keep_default_na=False, dtype=str)
test_s3 = pd.read_csv(TEST_DIR / "test_source3.tsv", sep='\t', keep_default_na=False, dtype=str)

print(f"Loaded Test S1: {len(test_s1):,}, Test S2: {len(test_s2):,}, Test S3: {len(test_s3):,} in {time.time()-t0:.2f}s")

# 2. Combine Target Pool
print("Preparing Target Pool (Source 2 + Source 3)...")
target_df = pd.concat([test_s2, test_s3], ignore_index=True)

# Clean names for fast indexing
def clean_str(col):
    return col.str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()

test_s1['clean_n'] = clean_str(test_s1['business_name'])
target_df['clean_n'] = clean_str(target_df['business_name'])

# Extract primary blocking key (first 4 chars of name)
test_s1['block_key'] = test_s1['clean_n'].str[:4]
target_df['block_key'] = target_df['clean_n'].str[:4]

print("Building fast groupby index for target pool...")
t0 = time.time()
target_groups = dict(tuple(target_df.groupby('block_key')))
print(f"Built groupby index across {len(target_groups):,} keys in {time.time()-t0:.2f}s")

# 3. Fast Candidate Retrieval & Matcher
print("Generating Candidates & Matches for 1.73M Source 1 Test Records...")
t0 = time.time()

s1_ids = test_s1['entity_id'].values
s1_names = test_s1['clean_n'].values
s1_keys = test_s1['block_key'].values

matching_lines = ["source1_entity_id\tmatched_entity_ids\n"]
candidate_lines = ["source1_entity_id\tcandidate_entity_ids\n"]

total_records = len(test_s1)

for idx in range(total_records):
    s1_id = s1_ids[idx]
    s1_n = s1_names[idx]
    key = s1_keys[idx]
    
    # Candidate lookup
    if key in target_groups:
        tgt_sub = target_groups[key]
        tgt_ids = tgt_sub['entity_id'].values
        tgt_names = tgt_sub['clean_n'].values
        
        # Limit candidate pool
        cand_list = list(tgt_ids[:30])
        cands_str = ",".join(cand_list)
        
        # Exact/prefix name matching
        matched_list = []
        for t_id, t_n in zip(tgt_ids[:30], tgt_names[:30]):
            if s1_n == t_n or (len(s1_n) > 5 and len(t_n) > 5 and s1_n[:8] == t_n[:8]):
                matched_list.append(t_id)
                
        matches_str = ",".join(matched_list)
    else:
        cands_str = ""
        matches_str = ""
        
    candidate_lines.append(f"{s1_id}\t{cands_str}\n")
    matching_lines.append(f"{s1_id}\t{matches_str}\n")
    
    if (idx + 1) % 500000 == 0:
        print(f"Processed {idx+1:,} / {total_records:,} records ({time.time()-t0:.2f}s)...")

print(f"Processed all {total_records:,} records in {time.time()-t0:.2f}s")

# 4. Write Output TSV Files
print("Writing matching_results.tsv and candidate_pairs.tsv...")
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
zip_path = BASE_DIR / "team_name_submission.zip"
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
