#!/usr/bin/env python3
import json
import os
import re
import sys
import argparse
from pathlib import Path
from collections import Counter

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_text_file(path: Path):
    return normalize_text(path.read_text(encoding='utf-8'))

def reference_in_source(reference: str, source_text: str) -> bool:
    # if reference is "UNKNOWN", it's allowed but counts as unknown
    if reference.strip().upper() == "UNKNOWN":
        return False
    ref_norm = normalize_text(reference)
    # direct substring
    if ref_norm in source_text:
        return True
    # fallback: check token overlap (simple)
    ref_tokens = set(ref_norm.split())
    source_tokens = set(source_text.split())
    if not ref_tokens:
        return False
    overlap = len(ref_tokens & source_tokens) / len(ref_tokens)
    return overlap >= 0.75

def validate_file(jsonl_path: Path, source_text_path: Path, language: str, max_fabrication_rate=0.05):
    source_text = load_text_file(source_text_path)
    entries = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entries.append(obj)
            except Exception as e:
                print(f"[ERROR] line {i}: invalid JSON => {e}")
                return False, {"error": "invalid_json", "line": i}
    if not entries:
        print("[WARN] No entries found in", jsonl_path)
        return False, {"error": "no_entries"}
    stats = {"total": len(entries), "verdicts": Counter(), "suspected_fabrication": 0, "unknown_refs": 0, "bad_schema": 0}
    failures = []
    for idx, e in enumerate(entries, 1):
        # basic schema expectations
        inst = e.get("instruction") or e.get("instruction_text") or ""
        inp = e.get("input") or ""
        out = e.get("output") or ""
        meta = e.get("meta") or {}
        verdict_found = False
        ref_value = ""
        # output parsing
        if isinstance(out, str):
            m = re.search(r'VERDICT\s*:\s*(True|False|True\.|False\.)', out, re.IGNORECASE)
            if m:
                verdict = m.group(1).lower().startswith('t')
                stats["verdicts"][ "true" if verdict else "false" ] += 1
                verdict_found = True
            ref_m = re.search(r'Reference\s*:\s*(.+)', out)
            if ref_m:
                ref_value = ref_m.group(1).strip()
            else:
                # try meta reference
                ref_value = meta.get("reference", "").strip()
        else:
            # Check if it's the new schema format
            verdict = e.get("verdict")
            if verdict in ["True", "False"]:
                stats["verdicts"]["true" if verdict == "True" else "false"] += 1
                verdict_found = True
            ref_value = e.get("reference", "").strip()
            
        if not verdict_found:
            stats["bad_schema"] += 1
            failures.append((idx, "no_verdict"))
        # check claim/context in input or new schema
        claim = e.get("claim", "")
        context = e.get("context_excerpt", "")
        if not claim and not context:
            if "Claim:" not in inp or "Context:" not in inp:
                stats["bad_schema"] += 1
                failures.append((idx, "missing_claim_or_context"))
        # check reference existence
        if not ref_value:
            stats["unknown_refs"] += 1
        else:
            if not reference_in_source(ref_value, source_text):
                # If ref given but not found -> suspected fabrication
                stats["suspected_fabrication"] += 1
                failures.append((idx, "ref_not_found", ref_value))
    # compute rates
    fab_rate = stats["suspected_fabrication"] / stats["total"]
    unknown_rate = stats["unknown_refs"] / stats["total"]
    true_count = stats["verdicts"].get("true", 0)
    false_count = stats["verdicts"].get("false", 0)
    balance = abs(true_count - false_count) / max(1, stats["total"])
    ok = True
    issues = []
    if fab_rate > max_fabrication_rate:
        ok = False
        issues.append(f"fabrication_rate_too_high: {fab_rate:.3f} > {max_fabrication_rate}")
    # require approximately balanced (tolerance 0.1 in smoke-test)
    if stats["total"] < 50:
        tol = 0.25
    else:
        tol = 0.03
    if balance > tol:
        ok = False
        issues.append(f"unbalanced_verdicts: true={true_count}, false={false_count}, balance={balance:.3f} (tol={tol})")
    result = {
        "total": stats["total"],
        "true": true_count,
        "false": false_count,
        "fabrication_count": stats["suspected_fabrication"],
        "fabrication_rate": fab_rate,
        "unknown_refs": stats["unknown_refs"],
        "bad_schema": stats["bad_schema"],
        "failures_sample": failures[:20],
        "ok": ok,
        "issues": issues
    }
    return ok, result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to generated preview JSONL (e.g. data/.../preview_ar_20.jsonl)")
    parser.add_argument("--source", required=True, help="Path to aaofi_cleaned text (arabic or english)")
    parser.add_argument("--lang", choices=["ar","en"], required=True)
    args = parser.parse_args()
    jsonl = Path(args.jsonl)
    source = Path(args.source)
    
    if not jsonl.exists():
        print(f"[ERROR] JSONL file not found: {jsonl}")
        sys.exit(1)
    if not source.exists():
        print(f"[ERROR] Source text file not found: {source}")
        sys.exit(1)
    
    print(f"Validating {jsonl} against {source} (language: {args.lang})")
    ok, result = validate_file(jsonl, source, args.lang)
    
    print("\n=== VALIDATION RESULTS ===")
    print(f"Total examples: {result['total']}")
    print(f"True examples: {result['true']}")
    print(f"False examples: {result['false']}")
    print(f"Fabrication rate: {result['fabrication_rate']:.3f}")
    print(f"Unknown references: {result['unknown_refs']}")
    print(f"Schema errors: {result['bad_schema']}")
    
    if result['issues']:
        print("\nISSUES:")
        for issue in result['issues']:
            print(f"  - {issue}")
    
    if result['failures_sample']:
        print("\nSAMPLE FAILURES:")
        for failure in result['failures_sample'][:5]:
            print(f"  - Example {failure[0]}: {failure[1]}")
            if len(failure) > 2:
                print(f"    Reference: {failure[2]}")
    
    if ok:
        print("\n✅ VALIDATION PASSED")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
