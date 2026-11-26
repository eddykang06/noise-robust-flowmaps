"""Prints every field that differs between two consolidate_check.py outputs (OLD, NEW).

    python diff_cases.py <cases-OLD.json> <cases-NEW.json>
"""
import json
import sys

old, new = (json.load(open(path)) for path in sys.argv[1:])
diffs = [(case, key) for case in old.keys() | new.keys()
         for key in old.get(case, {}).keys() | new.get(case, {}).keys()
         if old.get(case, {}).get(key) != new.get(case, {}).get(key)]
for case, key in sorted(diffs):
    print(f"DIFF {case}.{key}: {old.get(case, {}).get(key)} -> {new.get(case, {}).get(key)}")
print(f"{len(diffs)} differing fields")
