
from datasets import load_dataset
import pandas as pd
import sys

def check_keyword():
    print("Loading dataset...", flush=True)
    try:
        ds = load_dataset("teyler/epstein-files-20k", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr, flush=True)
        return

    keyword = "Modi"
    print(f"Searching for '{keyword}' in streaming dataset...", flush=True)
    
    matches = []
    count = 0
    max_rows = 22000 # Dataset is ~20k
    
    for i, row in enumerate(ds):
        count += 1
        if count > max_rows:
            break
            
        text = row.get("text", "")
        if keyword.lower() in text.lower():
            matches.append((i, text[:200])) # Preview first 200 chars
            
    print(f"Found {len(matches)} matches.", flush=True)
    for i, preview in matches[:5]:
        print(f"Match at index {i}: {preview}", flush=True)

    # Also check what a typical row looks like for filename parsing
    print("\n--- Sample Row ---", flush=True)
    if len(ds) > 0:
        print(ds[0]["text"][:500], flush=True)

if __name__ == "__main__":
    check_keyword()
