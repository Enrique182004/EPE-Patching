#!/usr/bin/env python3
"""
Quick fix script to make LightGBM optional in epe_patch_window_predictor.py
Run this in your Backend folder: python fix_lightgbm.py
"""

import re

# Read the file
with open('epe_patch_window_predictor.py', 'r') as f:
    content = f.read()

# Fix 1: Make import optional (line 40)
old_import = "from lightgbm import LGBMClassifier"
new_import = """try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM not available, using Random Forest, XGBoost, and Gradient Boosting instead")"""

content = content.replace(old_import, new_import)

# Fix 2: Make LightGBM model optional (around line 523)
# Find the models dictionary and wrap LightGBM in a condition
old_pattern = r"'LightGBM': LGBMClassifier\(n_estimators=100, random_state=42, verbose=-1\)"
new_pattern = "'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1) if HAS_LIGHTGBM else None"

content = re.sub(old_pattern, new_pattern, content)

# Also need to filter out None models later
# Find where models are trained and add a filter
old_train_pattern = r"for name, model in models\.items\(\):"
new_train_pattern = """for name, model in models.items():
        if model is None:
            continue  # Skip models that aren't available"""

content = re.sub(old_train_pattern, new_train_pattern, content)

# Write back
with open('epe_patch_window_predictor.py', 'w') as f:
    f.write(content)

print("✅ Fixed epe_patch_window_predictor.py")
print("✅ LightGBM is now optional")
print("✅ Try running: python api_server.py")