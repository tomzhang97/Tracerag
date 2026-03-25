#!/usr/bin/env python3
"""
maap_review.py - Review and Accept MAAP Draft Annotations

Interactive CLI to review VLM-generated draft annotations.
Human reviewer can: Accept, Edit, Skip, or Flag each draft.

Usage:
    python tools/maap_review.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime


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


def print_pair(pair, idx, total):
    """Display a pair for review."""
    print("\n" + "="*60)
    print(f"[{idx+1}/{total}] {pair['pair_id']}")
    print("="*60)
    print(f"Document: {pair.get('doc_id', 'N/A')}")
    print(f"Versions: {pair.get('version_id_old', '?')} → {pair.get('version_id_new', '?')}")
    print(f"Page: {pair.get('page_index_old', 0)}")
    print(f"Change Type: {pair.get('change_type', [])}")
    print(f"BBox Old: {pair.get('bbox_old', [])}")
    print(f"BBox New: {pair.get('bbox_new', [])}")
    print("-"*60)
    print("DRAFT ANNOTATION:")
    print(f"  {pair.get('change_desc_gt', 'N/A')}")
    print("-"*60)


def main():
    base = Path(".")
    draft_path = base / "visualdiff/annotations/visualdiff_pairs_DRAFT.jsonl"
    reviewed_path = base / "visualdiff/annotations/visualdiff_pairs_REVIEWED.jsonl"
    
    print("[*] MAAP Review Interface")
    print(f"    Draft file: {draft_path}")
    print(f"    Output: {reviewed_path}")
    
    # Load drafts
    if not draft_path.exists():
        print(f"\nERROR: Draft file not found: {draft_path}")
        print("Run maap_annotate.py first to generate drafts.")
        return
    
    drafts = load_jsonl(draft_path)
    print(f"    Loaded {len(drafts)} drafts")
    
    # Load already reviewed
    reviewed = load_jsonl(reviewed_path)
    reviewed_ids = {r["pair_id"] for r in reviewed}
    print(f"    Already reviewed: {len(reviewed_ids)}")
    
    # Filter to unreviewed
    to_review = [d for d in drafts if d["pair_id"] not in reviewed_ids]
    print(f"    Pending review: {len(to_review)}")
    
    if not to_review:
        print("\n✅ All drafts have been reviewed!")
        return
    
    print("\nCommands: [a]ccept | [e]dit | [s]kip | [f]lag | [q]uit")
    print("="*60)
    
    accepted = 0
    edited = 0
    skipped = 0
    flagged = 0
    
    for i, pair in enumerate(to_review):
        print_pair(pair, i, len(to_review))
        
        while True:
            choice = input("\nAction [a/e/s/f/q]: ").strip().lower()
            
            if choice == 'a':
                # Accept as-is
                pair["annotation_status"] = "accepted"
                pair["reviewed_at"] = datetime.now().isoformat()
                reviewed.append(pair)
                save_jsonl(reviewed, reviewed_path)
                accepted += 1
                print("  ✅ Accepted")
                break
                
            elif choice == 'e':
                # Edit the description
                print("\nEnter corrected description (or press Enter to cancel):")
                new_desc = input("> ").strip()
                if new_desc:
                    pair["change_desc_gt"] = new_desc
                    pair["annotation_status"] = "edited"
                    pair["reviewed_at"] = datetime.now().isoformat()
                    reviewed.append(pair)
                    save_jsonl(reviewed, reviewed_path)
                    edited += 1
                    print("  ✏️ Edited and saved")
                    break
                else:
                    print("  Cancelled edit")
                    
            elif choice == 's':
                # Skip for later
                skipped += 1
                print("  ⏭️ Skipped")
                break
                
            elif choice == 'f':
                # Flag as problematic
                pair["annotation_status"] = "flagged"
                pair["reviewed_at"] = datetime.now().isoformat()
                reason = input("Reason for flagging: ").strip()
                pair["flag_reason"] = reason
                reviewed.append(pair)
                save_jsonl(reviewed, reviewed_path)
                flagged += 1
                print("  🚩 Flagged")
                break
                
            elif choice == 'q':
                print("\n" + "="*60)
                print("SESSION SUMMARY")
                print("="*60)
                print(f"Accepted: {accepted}")
                print(f"Edited:   {edited}")
                print(f"Skipped:  {skipped}")
                print(f"Flagged:  {flagged}")
                print(f"Remaining: {len(to_review) - i}")
                print("="*60)
                return
                
            else:
                print("  Invalid. Use: a/e/s/f/q")
    
    # Final summary
    print("\n" + "="*60)
    print("REVIEW COMPLETE")
    print("="*60)
    print(f"Accepted: {accepted}")
    print(f"Edited:   {edited}")
    print(f"Skipped:  {skipped}")
    print(f"Flagged:  {flagged}")
    print(f"Total reviewed: {len(reviewed)}")
    print("="*60)


if __name__ == "__main__":
    main()
