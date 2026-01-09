
from datasets import load_dataset
import os

try:
    print("Attempting to load 'sreeramajay/pollution'...")
    ds = load_dataset("sreeramajay/pollution")
    print("Success!")
    print(ds)
    # Check split keys
    print(ds.keys())
    # Check first item to see if it has 'image' column
    print(ds['train'][0])
except Exception as e:
    print(f"Failed to load dataset: {e}")
