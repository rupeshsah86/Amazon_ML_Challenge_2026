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

print("=== STARTING HIGH-PRECISION VECTORIZED MATCHING PIPELINE ===")
start_time = time.time()

# 1. Load Test Data
print("Loading Test Source 1, Source 2, and Source 3 TSVs...")
t0 = time.time()
test_s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep='\t', keep_default_na=False, dtype=str)
test_s2 = pd.read_csv(TEST_DIR / "test_source2.tsv", sep='\t', keep_default_na=False, dtype=str)
test_s3 = pd.read_csv(TEST_DIR / "test_source3.tsv", sep='\t', keep_default_na=False, dtype=str)

print(f"Loaded Test S1: {len(test_s1):,}, Test S2: {len(test_s2):,}, Test S3: {len(test_s3):,} in {time.time()-t0:.2f}s")

# Combine Target Pool
target_df = pd.concat([test_s2, test_s3], ignore_index=True)

# Normalization regex
LEGAL_REGEX = re.compile(r'\b(inc|llc|corp|corporation|co|ltd|limited|pvt|private|sas|sarl|sa|eurl|group)\b', re.I)

def clean_name(series):
    return series.str.lower().str.replace(LEGAL_REGEX, '', regex=True).str.replace(r'[^\w\s]', '', regex=True).str.strip()

print("Normalizing business names and building exact name hash index...")
t0 = time.time()

test_s1['clean_n'] = clean_name(test_s1['business_name'])
target_df['clean_n'] = clean_name(target_df['business_name'])

# Filter out empty clean names
target_valid = target_df[target_df['clean_n'].str.len() > 2].copy()

# Group target entities by clean business name
print("Building exact normalized name dictionary...")
from collections import defaultdict
name_to_target_ids = defaultdict(list)
for cn, eid in zip(target_df['clean_n'].values, target_df['entity_id'].values):
    if len(cn) > 2:
        name_to_target_ids[cn].append(eid)

print(f"Built hash map with {len(name_to_target_ids):,} distinct business names in {time.time()-t0:.2f}s")

# 2. Vectorized Matching & Candidate Generation
print("Generating Candidates & High-Precision Matches for 1.73M Test Records...")
t0 = time.time()

s1_ids = test_s1['entity_id'].values
s1_clean_names = test_s1['clean_n'].values

matching_lines = ["source1_entity_id\tmatched_entity_ids\n"]
candidate_lines = ["source1_entity_id\tcandidate_entity_ids\n"]

total_records = len(test_s1)

for idx in range(total_records):
    s1_id = s1_ids[idx]
    cn = s1_clean_names[idx]
    
    if cn in name_to_target_ids:
        matched_ids = name_to_target_ids[cn]
        
        # High-precision matching logic:
        # If generic business name (too many hits), limit predictions to avoid precision collapse
        if len(matched_ids) > 6:
            matches_str = ",".join(matched_ids[:2])
            cands_str = ",".join(matched_ids[:15])
        else:
            matches_str = ",".join(matched_ids)
            cands_str = ",".join(matched_ids)
    else:
        cands_str = ""
        matches_str = ""
        
    candidate_lines.append(f"{s1_id}\t{cands_str}\n")
    matching_lines.append(f"{s1_id}\t{matches_str}\n")
    
    if (idx + 1) % 500000 == 0:
        print(f"Processed {idx+1:,} / {total_records:,} records ({time.time()-t0:.2f}s)...")

print(f"Processed all {total_records:,} records in {time.time()-t0:.2f}s")

# 3. Write Output Files
print("Writing matching_results.tsv and candidate_pairs.tsv...")
match_path = OUTPUT_DIR / "matching_results.tsv"
cand_path = OUTPUT_DIR / "candidate_pairs.tsv"

with open(match_path, "w", encoding="utf-8") as f:
    f.writelines(matching_lines)

with open(cand_path, "w", encoding="utf-8") as f:
    f.writelines(candidate_lines)

print(f"Saved {match_path}")
print(f"Saved {cand_path}")

# 4. Run Official Validator
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

# 5. Create Submission ZIP
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
