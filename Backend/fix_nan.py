#!/usr/bin/env python3
"""
Fix NaN handling in epe_patch_window_predictor.py
This script adds robust NaN handling to prevent model training errors
"""

import re
import os
import shutil
from datetime import datetime

def fix_nan_handling():
    """Add better NaN handling to the predictor"""
    
    filename = 'epe_patch_window_predictor.py'
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"❌ Error: {filename} not found in current directory")
        print(f"📁 Current directory: {os.getcwd()}")
        print("\n💡 Please either:")
        print("   1. Navigate to the directory containing the file")
        print("   2. Provide the full path to the file")
        return False
    
    # Create backup
    backup_file = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filename, backup_file)
    print(f"💾 Created backup: {backup_file}")
    
    with open(filename, 'r') as f:
        content = f.read()
    
    original_content = content
    fixes_applied = []
    
    # Fix 1: Add fillna to rolling features
    old_rolling = """    df = df.dropna()
    
    return df"""
    
    new_rolling = """    # Fill NaN values in rolling features with forward fill, then backward fill, then 0
    for col in df.columns:
        if 'rolling' in col:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    df = df.dropna()
    
    return df"""
    
    if old_rolling in content and new_rolling not in content:
        content = content.replace(old_rolling, new_rolling)
        fixes_applied.append("✅ Added fillna to rolling features")
    
    # Fix 2: Add fillna to lag features
    old_lag = """    df = df.dropna()
    
    return df


def create_rolling_features"""
    
    new_lag = """    # Fill NaN values in lag features with forward fill, then backward fill, then 0
    for col in df.columns:
        if 'lag' in col:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    df = df.dropna()
    
    return df


def create_rolling_features"""
    
    if old_lag in content and new_lag not in content:
        content = content.replace(old_lag, new_lag)
        fixes_applied.append("✅ Added fillna to lag features")
    
    # Fix 3: Add final NaN check before model training
    # Try multiple patterns to match the function definition
    patterns = [
        r'def train_models\(X_train: pd\.DataFrame, y_train: pd\.Series,\s*\n\s*X_test: pd\.DataFrame, y_test: pd\.Series\) -> Dict:\s*\n\s*"""[^"]*"""',
        r'def train_models\(X_train: pd\.DataFrame, y_train: pd\.Series,\s*\n\s*X_test: pd\.DataFrame, y_test: pd\.Series\) -> Dict:',
        r'def train_models\([^)]+\):[^\n]*\n\s*"""',
    ]
    
    nan_check_code = '''    # Critical: Remove any remaining NaN values before training
    print("🔍 Checking for NaN values...")
    
    # Check X_train
    nan_cols_train = X_train.columns[X_train.isna().any()].tolist()
    if nan_cols_train:
        print(f"  ⚠️  Found NaN in training features: {nan_cols_train}")
        print(f"  🔧 Filling NaN values with column means...")
        X_train = X_train.fillna(X_train.mean())
        # If still NaN (empty columns), fill with 0
        X_train = X_train.fillna(0)
    
    # Check X_test
    nan_cols_test = X_test.columns[X_test.isna().any()].tolist()
    if nan_cols_test:
        print(f"  ⚠️  Found NaN in test features: {nan_cols_test}")
        print(f"  🔧 Filling NaN values with column means...")
        X_test = X_test.fillna(X_train.mean())  # Use training means
        X_test = X_test.fillna(0)
    
    # Final verification
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("❌ Failed to remove all NaN values from features")
    
    print("  ✅ No NaN values detected")
    
'''
    
    # Check if NaN handling is already present
    if "Checking for NaN values" not in content:
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                # Find the position after the docstring or function definition
                insert_pos = match.end()
                # Find the next line after docstring
                next_newline = content.find('\n', insert_pos)
                if next_newline != -1:
                    content = content[:next_newline+1] + nan_check_code + content[next_newline+1:]
                    fixes_applied.append("✅ Added pre-training NaN check and cleanup")
                    break
    
    # Check if any fixes were applied
    if content == original_content:
        print("ℹ️  No changes needed - file may already be fixed or patterns not found")
        print("\n💡 If you're still experiencing NaN errors, try manual fix:")
        print_manual_fix()
        return False
    
    # Write back
    with open(filename, 'w') as f:
        f.write(content)
    
    print("\n" + "="*60)
    print("🎉 NaN Handling Fix Applied Successfully!")
    print("="*60)
    for fix in fixes_applied:
        print(fix)
    print(f"\n💾 Backup saved as: {backup_file}")
    print("🔄 Please restart api_server.py")
    print("="*60)
    return True

def print_manual_fix():
    """Print manual fix instructions"""
    print("\n📝 Manual fix instructions:")
    print("="*60)
    print("1. Open epe_patch_window_predictor.py")
    print("2. Find the train_models() function (search for 'def train_models')")
    print("3. Add this code at the very beginning of the function:")
    print("""
    # Remove NaN values before training
    print("🔍 Checking for NaN values...")
    X_train = X_train.fillna(X_train.mean()).fillna(0)
    X_test = X_test.fillna(X_train.mean()).fillna(0)
    
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Failed to remove all NaN values")
    print("✅ No NaN values detected")
""")
    print("="*60)

if __name__ == '__main__':
    try:
        success = fix_nan_handling()
        if not success:
            print("\n⚠️  Fix script completed with warnings")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        print_manual_fix()