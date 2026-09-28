import sys
import argparse
import pandas as pd
from pathlib import Path

def validate_submission(matching_file: str, candidate_file: str, test_dir: str) -> bool:
    print(f"=== Running Official Submission Validator ===")
    print(f"Matching file: {matching_file}")
    print(f"Candidate file: {candidate_file}")
    print(f"Test directory: {test_dir}")
    
    matching_path = Path(matching_file)
    candidate_path = Path(candidate_file)
    test_path = Path(test_dir)
    
    if not matching_path.exists():
        print(f"FAIL: Matching file missing: {matching_file}")
        return False
    if not candidate_path.exists():
        print(f"FAIL: Candidate file missing: {candidate_file}")
        return False
    
    # Load test source1 entity IDs
    test_s1_file = test_path / "test_source1.tsv"
    if not test_s1_file.exists():
        print(f"FAIL: Test source1 file missing: {test_s1_file}")
        return False
    
    s1_df = pd.read_csv(test_s1_file, sep='\t')
    if 'entity_id' not in s1_df.columns:
        print(f"FAIL: 'entity_id' missing from test_source1.tsv")
        return False
    expected_s1_ids = set(s1_df['entity_id'].astype(str))
    print(f"Loaded {len(expected_s1_ids)} expected Source 1 entity IDs from test set.")
    
    # Also load valid candidate IDs from test_source2 and test_source3
    test_s2_file = test_path / "test_source2.tsv"
    test_s3_file = test_path / "test_source3.tsv"
    valid_target_ids = set()
    if test_s2_file.exists():
        s2_df = pd.read_csv(test_s2_file, sep='\t')
        valid_target_ids.update(s2_df['entity_id'].astype(str))
    if test_s3_file.exists():
        s3_df = pd.read_csv(test_s3_file, sep='\t')
        valid_target_ids.update(s3_df['entity_id'].astype(str))
    print(f"Loaded {len(valid_target_ids)} valid target entity IDs (Source 2 + Source 3).")
    
    # Read matching_results.tsv
    try:
        match_df = pd.read_csv(matching_path, sep='\t', keep_default_na=False)
    except Exception as e:
        print(f"FAIL: Failed to parse {matching_file} as TSV: {e}")
        return False
    
    expected_match_cols = ['source1_entity_id', 'matched_entity_ids']
    if list(match_df.columns) != expected_match_cols:
        print(f"FAIL: {matching_file} header mismatch. Expected {expected_match_cols}, got {list(match_df.columns)}")
        return False
    
    # Read candidate_pairs.tsv
    try:
        cand_df = pd.read_csv(candidate_path, sep='\t', keep_default_na=False)
    except Exception as e:
        print(f"FAIL: Failed to parse {candidate_file} as TSV: {e}")
        return False
        
    expected_cand_cols = ['source1_entity_id', 'candidate_entity_ids']
    if list(cand_df.columns) != expected_cand_cols:
        print(f"FAIL: {candidate_file} header mismatch. Expected {expected_cand_cols}, got {list(cand_df.columns)}")
        return False
    
    # Row counts check
    if len(match_df) != len(expected_s1_ids):
        print(f"FAIL: {matching_file} row count ({len(match_df)}) does not match test Source 1 count ({len(expected_s1_ids)})")
        return False
    if len(cand_df) != len(expected_s1_ids):
        print(f"FAIL: {candidate_file} row count ({len(cand_df)}) does not match test Source 1 count ({len(expected_s1_ids)})")
        return False
    
    # Uniqueness check
    match_s1_ids = match_df['source1_entity_id'].astype(str).tolist()
    if len(set(match_s1_ids)) != len(match_s1_ids):
        print(f"FAIL: Duplicate source1_entity_id entries in {matching_file}")
        return False
    
    if set(match_s1_ids) != expected_s1_ids:
        print(f"FAIL: {matching_file} entity IDs do not exactly match test Source 1 entities")
        return False
    
    # Check candidates and match consistency
    cand_map = {}
    for _, row in cand_df.iterrows():
        s1_id = str(row['source1_entity_id'])
        cands = set(str(row['candidate_entity_ids']).replace(',', ' ').split()) if row['candidate_entity_ids'] else set()
        cand_map[s1_id] = cands
        # Validate candidate IDs belong to target set
        for c_id in cands:
            if valid_target_ids and c_id not in valid_target_ids:
                print(f"FAIL: Candidate ID '{c_id}' for '{s1_id}' does not exist in target sources.")
                return False
                
    for _, row in match_df.iterrows():
        s1_id = str(row['source1_entity_id'])
        matches = set(str(row['matched_entity_ids']).replace(',', ' ').split()) if row['matched_entity_ids'] else set()
        cands = cand_map.get(s1_id, set())
        # Rule: Every matched ID must be in candidates
        for m_id in matches:
            if m_id not in cands:
                print(f"FAIL: Matched ID '{m_id}' for '{s1_id}' is NOT in candidate list.")
                return False
            if valid_target_ids and m_id not in valid_target_ids:
                print(f"FAIL: Matched ID '{m_id}' for '{s1_id}' does not exist in target sources.")
                return False
                
    print("PASS: All validation checks succeeded!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to test directory")
    args = parser.parse_args()
    
    success = validate_submission(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)
