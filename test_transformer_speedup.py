#!/usr/bin/env python3
"""
Test script to verify transformer speedup changes work correctly.

This script checks the function signature and defaults without requiring
full transformer dependencies.
"""

import sys
import inspect
import ast
from pathlib import Path

project_root = Path(__file__).parent

print("="*60)
print("TESTING TRANSFORMER SPEEDUP CHANGES")
print("="*60)

# Read and parse the source file
print("\n1. Reading transformer_model.py source...")
transformer_file = project_root / "src" / "06_transformer_model.py"
with open(transformer_file, 'r') as f:
    source = f.read()
print("   ✓ File read successfully")

# Parse AST to check function defaults
print("\n2. Checking train_transformer() function signature...")
tree = ast.parse(source)

# Find the train_transformer function
train_transformer_node = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'train_transformer':
        train_transformer_node = node
        break

if not train_transformer_node:
    print("   ✗ train_transformer function not found")
    sys.exit(1)

print("   ✓ Function found")

# Extract defaults
param_defaults = {}
for i, arg in enumerate(train_transformer_node.args.args):
    param_name = arg.arg
    # Skip self/model/tokenizer which are required
    if param_name in ['model', 'tokenizer', 'train_texts', 'train_labels']:
        continue
    
    # Find corresponding default value
    defaults_start = len(train_transformer_node.args.args) - len(train_transformer_node.args.defaults)
    if i >= defaults_start:
        default_idx = i - defaults_start
        default_node = train_transformer_node.args.defaults[default_idx]
        
        # Extract the default value
        if isinstance(default_node, ast.Constant):
            param_defaults[param_name] = default_node.value
        elif isinstance(default_node, ast.NameConstant):  # Python < 3.8
            param_defaults[param_name] = default_node.value
        elif isinstance(default_node, ast.Num):  # Python < 3.8
            param_defaults[param_name] = default_node.n
        elif isinstance(default_node, ast.Name):  # Variables like None, True, False
            param_defaults[param_name] = default_node.id

# Expected defaults for speedup
expected_defaults = {
    'num_train_epochs': 0.5,
    'per_device_train_batch_size': 32,
    'max_length': 128,
}

print("\n   Checking defaults:")
all_correct = True
for param_name, expected_value in expected_defaults.items():
    if param_name in param_defaults:
        actual_value = param_defaults[param_name]
        if actual_value == expected_value:
            print(f"   ✓ {param_name} = {actual_value} (correct)")
        else:
            print(f"   ✗ {param_name} = {actual_value} (expected {expected_value})")
            all_correct = False
    else:
        print(f"   ✗ {param_name} parameter default not found")
        all_correct = False

# Verify max_length is used in FakeNewsDataset creation
print("\n3. Checking max_length is used in train_transformer...")
if 'max_length=max_length' in source or f'max_length={param_defaults.get("max_length", 128)}' in source:
    print("   ✓ max_length parameter is used in FakeNewsDataset creation")
else:
    print("   ⚠ Warning: Could not verify max_length is used")
    all_correct = False

if not all_correct:
    print("\n   ✗ FAILED: Default values don't match expected speedup settings")
    sys.exit(1)

print("\n" + "="*60)
print("✅ CODE STRUCTURE CHECKS PASSED!")
print("="*60)

print("\n" + "="*60)
print("EXPECTED OUTPUT WHEN RUNNING NOTEBOOK")
print("="*60)
print("""
When you run the transformer training cell in the notebook, you should see:

1. Initial setup:
   ============================================================
   Fine-tuning DistilBERT on ISOT data...
   ============================================================
   Preparing dataset for transformer fine-tuning...
   Configuring TrainingArguments...
   Initializing Trainer and starting training...

2. Training progress:
   - Progress bar showing approximately 1,400 steps (half of 2,807)
   - This is because num_train_epochs=0.5 (half epoch)
   - Example: [1400/2807 XX:XX < XX:XX, X.XX it/s, Epoch 0.50/0.5]

3. Speed indicators:
   - Training should complete in roughly 15-30 minutes (vs 45-60 min before)
   - max_length=128 means shorter sequences (2x faster tokenization)
   - batch_size=32 means fewer steps per epoch
   - 0.5 epochs means half the training time

4. What to verify:
   ✓ Training completes successfully
   ✓ Progress shows ~0.5 epochs
   ✓ Training time is significantly faster than before
   ✓ No memory errors (batch_size=32 should work on MPS)
   ✓ Model can be evaluated after training

If you see these, the speedup is working correctly!
""")
