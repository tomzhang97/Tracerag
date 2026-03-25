#!/usr/bin/env python3
"""
maap_merge.py - Merge Reviewed Annotations Back to Main

Takes REVIEWED annotations and merges them back into the main pairs file,
then regenerates the unified eng_bench.jsonl.

Usage:
    python tools/maap_merge.py
"""

import json
from pathlib import Path


def load_jsonl(path):
    """Load JSONL file."""
    items = []
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    return items


def save_jsonl(items, path):
    """Save items to JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')


def main():
    base = Path(".")
    
    # Paths
    gold_path = base / "visualdiff/annotations/visualdiff_pairs_GOLD.jsonl"
    reviewed_path = base / "visualdiff/annotations/visualdiff_pairs_REVIEWED.jsonl"
    output_path = base / "visualdiff/annotations/visualdiff_pairs.jsonl"
    
    print("[*] MAAP Merge Tool")
    print(f"    GOLD: {gold_path}")
    print(f"    REVIEWED: {reviewed_path}")
    print(f"    Output: {output_path}")
    
    # Load files
    gold = load_jsonl(gold_path)
    reviewed = load_jsonl(reviewed_path)
    
    print(f"\n    GOLD pairs: {len(gold)}")
    print(f"    REVIEWED pairs: {len(reviewed)}")
    
    # Filter reviewed to only accepted/edited (not flagged)
    valid_reviewed = [r for r in reviewed if r.get("annotation_status") in ["accepted", "edited"]]
    flagged = [r for r in reviewed if r.get("annotation_status") == "flagged"]
    
    print(f"    Valid (accepted/edited): {len(valid_reviewed)}")
    print(f"    Flagged (excluded): {len(flagged)}")
    
    # Merge
    merged = gold + valid_reviewed
    
    # Remove MAAP metadata fields
    for item in merged:
        item.pop("annotation_source", None)
        item.pop("annotation_status", None)
        item.pop("reviewed_at", None)
        item.pop("flag_reason", None)
    
    print(f"\n[*] Merged total: {len(merged)}")
    
    # Backup existing
    if output_path.exists():
        backup_path = output_path.with_suffix(".jsonl.bak")
        print(f"    Backing up existing to {backup_path}")
        output_path.rename(backup_path)
    
    # Save merged
    save_jsonl(merged, output_path)
    print(f"    Saved to {output_path}")
    
    # Stats
    print("\n" + "="*50)
    print("MERGE COMPLETE")
    print("="*50)
    print(f"Previous GOLD:  {len(gold)}")
    print(f"New reviewed:   {len(valid_reviewed)}")
    print(f"New total:      {len(merged)}")
    print(f"Completion:     {len(merged)/1488*100:.1f}%")
    print("="*50)
    print("\nNext step: Run 'python tools/unify_dataset.py' to rebuild eng_bench.jsonl")


if __name__ == "__main__":
    main()
