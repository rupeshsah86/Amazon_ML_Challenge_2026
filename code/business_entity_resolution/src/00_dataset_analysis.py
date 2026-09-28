import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils.metrics import parse_matched_ids

DATASET_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/dataset")
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

s1 = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep='\t')
s2 = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep='\t')
s3 = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep='\t')
gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep='\t')

test_s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep='\t')
test_s2 = pd.read_csv(TEST_DIR / "test_source2.tsv", sep='\t')
test_s3 = pd.read_csv(TEST_DIR / "test_source3.tsv", sep='\t')

gt_map = {row['source1_entity_id']: parse_matched_ids(row['matched_entity_ids']) for _, row in gt.iterrows()}
match_counts = [len(m) for m in gt_map.values()]

singletons = sum(1 for c in match_counts if c == 0)
singleton_pct = (singletons / len(match_counts)) * 100
total_matches = sum(match_counts)

print("=== DATASET ANALYSIS REPORT ===")
print(f"Train Source 1 Records: {len(s1)}")
print(f"Train Source 2 Records: {len(s2)}")
print(f"Train Source 3 Records: {len(s3)}")
print(f"Train Total True Matches: {total_matches}")
print(f"Train Singletons: {singletons} ({singleton_pct:.2f}%)")
print(f"Match Count Distribution: 0: {singletons}, 1: {sum(1 for c in match_counts if c == 1)}, 2+: {sum(1 for c in match_counts if c >= 2)}")
print(f"Train Source 1 Country Distribution:\n{s1['country'].value_counts().to_dict()}")
print(f"Test Source 1 Country Distribution:\n{test_s1['country'].value_counts().to_dict()}")
print("================================")

# Initialize performance_log.txt
perf_log = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/performance_log.txt")
with open(perf_log, "a") as f:
    f.write("=== AMAZON ML CHALLENGE 2026 EXPERIMENT PERFORMANCE LOG ===\n")
    f.write(f"Dataset Loaded: Train S1={len(s1)}, S2={len(s2)}, S3={len(s3)}, GT Matches={total_matches}, Singletons={singleton_pct:.2f}%\n")
