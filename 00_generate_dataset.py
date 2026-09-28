import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

DATASET_DIR = Path("/Users/rupeshmacbook/Desktop/amazon_entity_resolution/dataset")
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# Helper lists for realistic business name and address generation
BUSINESS_TYPES = ["Technologies", "Solutions", "Logistics", "Retail", "Services", "Global", "Industries", "Group", "Enterprises", "Labs", "Systems", "Pharma", "Capital", "Media"]
LEGAL_SUFFIXES_US = ["Inc", "Inc.", "LLC", "Corp", "Corporation", "Co.", "Ltd", "Limited", "Group Inc"]
LEGAL_SUFFIXES_IN = ["Pvt Ltd", "Private Limited", "Ltd", "Enterprises", "Limited", "LTD"]
LEGAL_SUFFIXES_FR = ["SAS", "SARL", "SA", "EURL", "Societe Anonyme"]

STREET_TYPES = [("Street", "St."), ("Road", "Rd."), ("Avenue", "Ave."), ("Boulevard", "Blvd."), ("Parkway", "Pkwy"), ("Lane", "Ln.")]
CITIES_US = ["New York, NY", "Seattle, WA", "San Francisco, CA", "Austin, TX", "Chicago, IL", "Boston, MA", "Los Angeles, CA"]
CITIES_IN = ["Bangalore, KA", "Mumbai, MH", "Delhi, DL", "Hyderabad, TS", "Chennai, TN", "Pune, MH"]
CITIES_FR = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Bordeaux", "Lille"]

US_NAMES = ["Apex", "Titan", "Summit", "Beacon", "Nexus", "Velocity", "Horizon", "Pinnacle", "Vanguard", "Omni", "Quantum", "Cascade", "Crest", "Silverline", "BlueSky"]
IN_NAMES = ["Reliance", "Tata", "Infosys", "Wipro", "Adani", "Mahindra", "HCL", "Bharti", "Zenith", "Surya", "Vanguard", "Swastik", "Navbharat", "Standard"]
FR_NAMES = ["Lumiere", "Etoile", "Massif", "Riviera", "Soleil", "Capgemini", "Danone", "Total", "Sanofi", "Orange", "Nexans", "Michelin"]

def generate_address(country):
    num = random.randint(10, 9999)
    st_full, st_abbr = random.choice(STREET_TYPES)
    st_name = random.choice(["Main", "Oak", "Commercial", "Tech", "Industrial", "Central", "Grand", "Station"])
    use_abbr = random.random() < 0.5
    st_str = st_abbr if use_abbr else st_full
    
    if country == "US":
        city = random.choice(CITIES_US)
        zip_code = f"{random.randint(10000, 99999)}"
        suite = f"Suite {random.randint(100, 900)}" if random.random() < 0.3 else ""
        return f"{num} {st_name} {st_str} {suite}, {city} {zip_code}".strip().replace("  ", " ")
    elif country == "IN":
        city = random.choice(CITIES_IN)
        pin_code = f"{random.randint(110001, 700099)}"
        floor = f"Plot No {random.randint(1, 200)}, Sector {random.randint(1, 50)}" if random.random() < 0.4 else ""
        return f"{floor} {num} {st_name} {st_str}, {city} - {pin_code}".strip().replace("  ", " ")
    else: # FR
        city = random.choice(CITIES_FR)
        postal = f"{random.randint(75001, 93000)}"
        return f"{num} {st_str} {st_name}, {postal} {city}".strip().replace("  ", " ")

def apply_noise(name, address):
    # Apply minor typos or word order permutations
    name_tokens = name.split()
    if len(name_tokens) > 2 and random.random() < 0.25:
        # Swap adjacent tokens
        idx = random.randint(0, len(name_tokens) - 2)
        name_tokens[idx], name_tokens[idx+1] = name_tokens[idx+1], name_tokens[idx]
        name = " ".join(name_tokens)
    
    if random.random() < 0.2:
        # Drop legal suffix
        for s in LEGAL_SUFFIXES_US + LEGAL_SUFFIXES_IN + LEGAL_SUFFIXES_FR:
            if name.endswith(s):
                name = name[:-len(s)].strip()
                break
                
    if random.random() < 0.15 and len(name) > 5:
        # Character typo
        pos = random.randint(1, len(name) - 2)
        name = name[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + name[pos+1:]
        
    # Address variations
    if random.random() < 0.3:
        address = address.replace("Street", "St").replace("Road", "Rd").replace("Avenue", "Ave").replace("Suite", "Ste")
    if random.random() < 0.15:
        # Drop postal/zip code
        tokens = address.split()
        address = " ".join([t for t in tokens if not t.isdigit()])
        
    return name, address

def generate_split(num_s1, split_name, countries):
    s1_rows = []
    s2_rows = []
    s3_rows = []
    gt_rows = []
    
    s2_id_counter = 100000
    s3_id_counter = 200000
    
    for i in range(1, num_s1 + 1):
        s1_id = f"s1_{split_name}_{i:05d}"
        country = random.choice(countries)
        
        if country == "US":
            prefix = random.choice(US_NAMES)
            b_type = random.choice(BUSINESS_TYPES)
            suffix = random.choice(LEGAL_SUFFIXES_US)
        elif country == "IN":
            prefix = random.choice(IN_NAMES)
            b_type = random.choice(BUSINESS_TYPES)
            suffix = random.choice(LEGAL_SUFFIXES_IN)
        else:
            prefix = random.choice(FR_NAMES)
            b_type = random.choice(BUSINESS_TYPES)
            suffix = random.choice(LEGAL_SUFFIXES_FR)
            
        base_name = f"{prefix} {b_type} {suffix}"
        base_address = generate_address(country)
        
        s1_rows.append({
            "entity_id": s1_id,
            "business_name": base_name,
            "business_address": base_address,
            "country": country
        })
        
        # Ground truth matching logic
        # 30% singleton (0 matches), 50% single match, 20% multi-match
        prob = random.random()
        matched_ids = []
        
        if prob >= 0.30:
            # Create match in Source 2
            if random.random() < 0.8:
                s2_id_counter += 1
                s2_id = f"s2_{split_name}_{s2_id_counter:06d}"
                n_s2, a_s2 = apply_noise(base_name, base_address)
                s2_rows.append({
                    "entity_id": s2_id,
                    "business_name": n_s2,
                    "business_address": a_s2,
                    "country": country
                })
                matched_ids.append(s2_id)
                
            # Create match in Source 3
            if prob >= 0.60 or len(matched_ids) == 0:
                s3_id_counter += 1
                s3_id = f"s3_{split_name}_{s3_id_counter:06d}"
                n_s3, a_s3 = apply_noise(base_name, base_address)
                s3_rows.append({
                    "entity_id": s3_id,
                    "business_name": n_s3,
                    "business_address": a_s3,
                    "country": country
                })
                matched_ids.append(s3_id)
                
        gt_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": " ".join(matched_ids)
        })
        
    # Generate additional non-matching distractor records in Source 2 & Source 3
    for _ in range(int(num_s1 * 0.3)):
        country = random.choice(countries)
        prefix = random.choice(US_NAMES + IN_NAMES + FR_NAMES)
        b_type = random.choice(BUSINESS_TYPES)
        suffix = random.choice(LEGAL_SUFFIXES_US + LEGAL_SUFFIXES_IN + LEGAL_SUFFIXES_FR)
        
        s2_id_counter += 1
        s2_rows.append({
            "entity_id": f"s2_{split_name}_{s2_id_counter:06d}",
            "business_name": f"{prefix} {b_type} {suffix}",
            "business_address": generate_address(country),
            "country": country
        })
        
        s3_id_counter += 1
        s3_rows.append({
            "entity_id": f"s3_{split_name}_{s3_id_counter:06d}",
            "business_name": f"{prefix} {b_type} {suffix}",
            "business_address": generate_address(country),
            "country": country
        })
        
    # Shuffle s2 and s3 rows
    random.shuffle(s2_rows)
    random.shuffle(s3_rows)
    
    return pd.DataFrame(s1_rows), pd.DataFrame(s2_rows), pd.DataFrame(s3_rows), pd.DataFrame(gt_rows)

print("Generating realistic synthetic dataset for Amazon ML Challenge 2026...")
train_s1, train_s2, train_s3, train_gt = generate_split(1200, "train", ["US", "IN"])
test_s1, test_s2, test_s3, _ = generate_split(600, "test", ["US", "IN", "FR"])

train_s1.to_csv(TRAIN_DIR / "train_source1.tsv", sep='\t', index=False)
train_s2.to_csv(TRAIN_DIR / "train_source2.tsv", sep='\t', index=False)
train_s3.to_csv(TRAIN_DIR / "train_source3.tsv", sep='\t', index=False)
train_gt.to_csv(TRAIN_DIR / "train_ground_truth.tsv", sep='\t', index=False)

test_s1.to_csv(TEST_DIR / "test_source1.tsv", sep='\t', index=False)
test_s2.to_csv(TEST_DIR / "test_source2.tsv", sep='\t', index=False)
test_s3.to_csv(TEST_DIR / "test_source3.tsv", sep='\t', index=False)

print(f"Train Source 1 records: {len(train_s1)}")
print(f"Train Source 2 records: {len(train_s2)}")
print(f"Train Source 3 records: {len(train_s3)}")
print(f"Train Ground Truth rows: {len(train_gt)}")
print(f"Test Source 1 records: {len(test_s1)}")
print(f"Test Source 2 records: {len(test_s2)}")
print(f"Test Source 3 records: {len(test_s3)}")
print("Dataset generation complete!")
