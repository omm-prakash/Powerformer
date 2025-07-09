import re

import itertools
import hashlib
import pickle
import os
import json
import shutil

def get_max_case_number(data):
    expt_numbers = []
    for entry in data:
        match = re.search(r'case-(\d+)', entry)
        if match:
            expt_numbers.append(int(match.group(1)))
    return max(expt_numbers) if expt_numbers else 0

def get_config_hash(config):
    """Create a hash of the config to use as cache key"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def get_all_combinations(var_dict, cache_dir="cache"):
    """Generate or load cached combinations"""
    os.makedirs(cache_dir, exist_ok=True)
    hash_key = get_config_hash(var_dict)
    cache_file = os.path.join(cache_dir, f"{hash_key}.pkl")

    # Load from cache if exists
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            print("Loaded combinations from cache.")
            return pickle.load(f)

    # Compute combinations
    keys = list(var_dict.keys())
    values = list(var_dict.values())
    all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # Save to cache
    with open(cache_file, "wb") as f:
        pickle.dump(all_combinations, f)
        print("Computed and cached combinations.")

    return all_combinations

def copy_file(src_file, dest_path):
    try:
        if not os.path.isfile(src_file):
            raise FileNotFoundError(f"Source file '{src_file}' not found.")
        
        if os.path.isdir(dest_path):
            dest_file = os.path.join(dest_path, os.path.basename(src_file))
        else:
            dest_file = dest_path
        
        dest_dir = os.path.dirname(dest_file)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        shutil.copy2(src_file, dest_file)
    except Exception as e:
        print(f"An error occurred while copying: {e}")