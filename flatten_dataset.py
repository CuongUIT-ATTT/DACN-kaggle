#!/usr/bin/env python3
"""
flatten_dataset.py

Transforms the preprocessed dataset into a format compatible with train.py.

Input:
  - datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_test_input.pkl
  - Each row contains both original (orig_func, orig_cpg) and adversarial (func, cpg)

Output:
  - datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl
  - Each pair is split into two rows:
    1. Original row: adv=False, target flipped
    2. Adversarial row: adv=True, target unchanged
  - Both rows share the same ID for pairwise evaluation

This ensures a perfectly balanced dataset (50% benign, 50% vulnerable)
compatible with pairwise evaluation in train.py.
"""

import pandas as pd
import numpy as np
import os
import sys

def validate_row(row, debug=False):
    """
    Validate that a row has all required fields and they are not null.
    Handles CPG data stored as either Dict or List[Dict].
    
    Args:
        row: A pandas Series representing one row
        debug: If True, print why validation fails
    
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ['func', 'cpg', 'orig_func', 'orig_cpg', 'target']
    
    # Check all required fields exist
    for field in required_fields:
        if field not in row.index:
            if debug:
                print(f"  ❌ Row failed: field '{field}' not in row")
            return False
        if pd.isna(row[field]) or row[field] is None:
            if debug:
                print(f"  ❌ Row failed: field '{field}' is null/None")
            return False
    
    # Special check for target: must be numeric
    try:
        target_val = int(row['target'])
        if target_val not in [0, 1]:
            if debug:
                print(f"  ❌ Row failed: target value {target_val} not in [0, 1]")
            return False
    except (ValueError, TypeError):
        if debug:
            print(f"  ❌ Row failed: target '{row['target']}' is not convertible to int")
        return False
    
    # Validate CPG data (can be Dict or List[Dict])
    for cpg_field in ['cpg', 'orig_cpg']:
        cpg_data = row[cpg_field]
        
        # Case 1: It's a list
        if isinstance(cpg_data, list):
            if len(cpg_data) == 0:
                if debug:
                    print(f"  ❌ Row failed: {cpg_field} is a list but empty")
                return False
            # Check first element is a dict
            if not isinstance(cpg_data[0], dict) or not cpg_data[0]:
                if debug:
                    print(f"  ❌ Row failed: {cpg_field}[0] is not a dict or is empty")
                return False
        # Case 2: It's a dict
        elif isinstance(cpg_data, dict):
            if not cpg_data:
                if debug:
                    print(f"  ❌ Row failed: {cpg_field} is a dict but empty")
                return False
        # Case 3: Invalid type
        else:
            if debug:
                print(f"  ❌ Row failed: {cpg_field} is neither dict nor list (type: {type(cpg_data)})")
            return False
    
    return True

def flip_target(target):
    """
    Flip the target label: 0 -> 1, 1 -> 0
    
    Args:
        target: Original target value
    
    Returns:
        int: Flipped target value
    """
    return 1 if target == 0 else 0

def flatten_dataset(input_path, output_path):
    """
    Transform the paired dataset into a flat format for training.
    
    Args:
        input_path (str): Path to input pickle file
        output_path (str): Path to save output pickle file
    """
    print("=" * 80)
    print("FLATTEN DATASET - PREPROCESSING")
    print("=" * 80)
    
    # Step 1: Load data
    print(f"\n[1/6] Loading data from: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ ERROR: File not found: {input_path}")
        sys.exit(1)
    
    try:
        raw_df = pd.read_pickle(input_path)
        print(f"✓ Loaded {len(raw_df)} rows")
        print(f"✓ Columns: {list(raw_df.columns)}")
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        sys.exit(1)
    
    # Step 2: Filter invalid rows
    print(f"\n[2/6] Filtering rows with null/invalid CPG data...")
    initial_count = len(raw_df)
    
    # First, show sample of first row's CPG structure
    if len(raw_df) > 0:
        print(f"\n  📊 Sample CPG structure from first row:")
        print(f"    - cpg type: {type(raw_df.iloc[0]['cpg'])}")
        if isinstance(raw_df.iloc[0]['cpg'], list):
            print(f"    - cpg is a list with {len(raw_df.iloc[0]['cpg'])} element(s)")
            if len(raw_df.iloc[0]['cpg']) > 0:
                print(f"    - cpg[0] type: {type(raw_df.iloc[0]['cpg'][0])}")
        print(f"    - orig_cpg type: {type(raw_df.iloc[0]['orig_cpg'])}")
        if isinstance(raw_df.iloc[0]['orig_cpg'], list):
            print(f"    - orig_cpg is a list with {len(raw_df.iloc[0]['orig_cpg'])} element(s)")
            if len(raw_df.iloc[0]['orig_cpg']) > 0:
                print(f"    - orig_cpg[0] type: {type(raw_df.iloc[0]['orig_cpg'][0])}")
    
    valid_rows = raw_df.apply(validate_row, axis=1)
    invalid_count = (~valid_rows).sum()
    
    # Debug first few invalid rows
    if invalid_count > 0:
        print(f"\n  ⚠ Found {invalid_count} invalid rows. Showing first few reasons:")
        invalid_indices = raw_df[~valid_rows].index[:3]
        for idx in invalid_indices:
            print(f"\n  Row index {idx}:")
            validate_row(raw_df.iloc[idx], debug=True)
    
    raw_df = raw_df[valid_rows].reset_index(drop=True)
    filtered_count = initial_count - len(raw_df)
    
    if filtered_count > 0:
        print(f"\n✓ Filtered out {filtered_count} invalid rows")
    print(f"✓ Remaining: {len(raw_df)} valid rows")
    
    if len(raw_df) == 0:
        print("\n❌ ERROR: No valid rows remaining after filtering")
        print("   Please check the CPG data structure in your input file.")
        sys.exit(1)
    
    # Step 3: Create original rows (adv=False)
    print(f"\n[3/6] Creating original rows (adv=False)...")
    original_rows = []
    
    for idx, row in raw_df.iterrows():
        # Extract CPG dict from list if needed
        orig_cpg_data = row['orig_cpg']
        if isinstance(orig_cpg_data, list):
            orig_cpg_data = orig_cpg_data[0]  # Extract first element
        
        # Ensure target is int before flipping
        target_val = int(row['target'])
        
        original_row = {
            'id': str(idx),
            'adv': False,
            'func': row['orig_func'],
            'target': flip_target(target_val),  # Flip target
            'cpg': orig_cpg_data  # Clean dict object
        }
        
        # Include 'input' if it exists
        if 'orig_input' in row and pd.notna(row['orig_input']):
            original_row['input'] = row['orig_input']
        
        # Include 'cwe' if it exists
        if 'cwe' in row:
            original_row['cwe'] = row['cwe']
        
        original_rows.append(original_row)
    
    original_df = pd.DataFrame(original_rows)
    print(f"✓ Created {len(original_df)} original rows")
    print(f"  - Target distribution: {original_df['target'].value_counts().to_dict()}")
    
    # Step 4: Create adversarial rows (adv=True)
    print(f"\n[4/6] Creating adversarial rows (adv=True)...")
    adversarial_rows = []
    
    for idx, row in raw_df.iterrows():
        # Extract CPG dict from list if needed
        cpg_data = row['cpg']
        if isinstance(cpg_data, list):
            cpg_data = cpg_data[0]  # Extract first element
        
        # Ensure target is int
        target_val = int(row['target'])
        
        adversarial_row = {
            'id': str(idx),
            'adv': True,
            'func': row['func'],
            'target': target_val,  # Keep original target
            'cpg': cpg_data  # Clean dict object
        }
        
        # Include 'input' if it exists
        if 'input' in row and pd.notna(row['input']):
            adversarial_row['input'] = row['input']
        
        # Include 'cwe' if it exists
        if 'cwe' in row:
            adversarial_row['cwe'] = row['cwe']
        
        adversarial_rows.append(adversarial_row)
    
    adversarial_df = pd.DataFrame(adversarial_rows)
    print(f"✓ Created {len(adversarial_df)} adversarial rows")
    print(f"  - Target distribution: {adversarial_df['target'].value_counts().to_dict()}")
    
    # Step 5: Concatenate and unify
    print(f"\n[5/6] Combining original and adversarial rows...")
    dataset_df = pd.concat([original_df, adversarial_df], ignore_index=True)
    
    # Ensure correct data types
    dataset_df['target'] = dataset_df['target'].astype(int)
    dataset_df['id'] = dataset_df['id'].astype(str)
    dataset_df['adv'] = dataset_df['adv'].astype(bool)
    
    print(f"✓ Combined dataset: {len(dataset_df)} rows")
    print(f"  - Original rows: {len(original_df)}")
    print(f"  - Adversarial rows: {len(adversarial_df)}")
    
    # Step 6: Validate and display statistics
    print(f"\n[6/6] Final dataset statistics:")
    print("-" * 80)
    
    # Wrap in try-except to handle complex objects (e.g., torch_geometric.Data)
    try:
        print(f"Total rows: {len(dataset_df)}")
        print(f"Unique IDs: {dataset_df['id'].nunique()}")
        print(f"Columns: {list(dataset_df.columns)}")
        
        # Target distribution - use to_dict() instead of dict()
        print("\nTarget distribution:")
        target_counts = dataset_df['target'].value_counts().sort_index()
        for target_val, count in target_counts.to_dict().items():
            print(f"  {target_val}: {count}")
        
        # Adv distribution
        print("\nAdv distribution:")
        adv_counts = dataset_df['adv'].value_counts()
        for adv_val, count in adv_counts.to_dict().items():
            print(f"  {adv_val}: {count}")
        
        # Cross-tabulation (Target vs Adv)
        print("\nCross-tabulation (Target vs Adv):")
        crosstab = pd.crosstab(dataset_df['target'], dataset_df['adv'], margins=True)
        print(crosstab)
        
        # Check balance
        total_benign = (dataset_df['target'] == 0).sum()
        total_vulnerable = (dataset_df['target'] == 1).sum()
        balance_ratio = min(total_benign, total_vulnerable) / max(total_benign, total_vulnerable)
        
        print(f"\nClass balance:")
        print(f"  - Benign (0): {total_benign} ({total_benign/len(dataset_df)*100:.1f}%)")
        print(f"  - Vulnerable (1): {total_vulnerable} ({total_vulnerable/len(dataset_df)*100:.1f}%)")
        print(f"  - Balance ratio: {balance_ratio:.2%}")
        
        if balance_ratio > 0.95:
            print("  ✓ Dataset is well-balanced!")
        else:
            print("  ⚠ Warning: Dataset is imbalanced")
        
    except Exception as e:
        print(f"\n⚠ Warning: Could not print all statistics due to complex objects in dataset")
        print(f"  Error: {e}")
        print(f"  Dataset has {len(dataset_df)} rows with {dataset_df['id'].nunique()} unique IDs")
        print(f"  Proceeding to save the file...")
    
    # Step 7: Save output
    print(f"\n[7/7] Saving to: {output_path}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created directory: {output_dir}")
    
    try:
        dataset_df.to_pickle(output_path)
        print(f"✓ Saved successfully!")
        
        # Verify saved file
        file_size = os.path.getsize(output_path)
        print(f"✓ File size: {file_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ ERROR saving file: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nOutput file: {output_path}")
    print(f"Ready for training with train.py")
    
    # Display sample
    print("\n" + "-" * 80)
    print("Sample rows (first 5):")
    print("-" * 80)
    print(dataset_df[['id', 'adv', 'target']].head(10))
    
    return dataset_df

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_test_input.pkl"
    OUTPUT_FILE = "datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FILE = sys.argv[2]
    
    print("Configuration:")
    print(f"  Input:  {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print()
    
    # Run the transformation
    dataset_df = flatten_dataset(INPUT_FILE, OUTPUT_FILE)
    
    print("\n✅ Script completed successfully!")
    print(f"\nYou can now run: python train.py")
