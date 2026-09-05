import re

with open("paper/hri_full_paper.tex", encoding="utf-8") as f:
    tex = f.read()
labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
print("Labels defined:", len(labels))
print("Refs used:", len(refs))
missing = refs - labels
print("REFS WITH NO MATCHING LABEL:", missing if missing else "none -- all good")
unused = labels - refs
print("LABELS NEVER REFERENCED (fine, just FYI):", unused if unused else "none")
