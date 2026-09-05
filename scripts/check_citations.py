import re

with open("paper/hri_full_paper.tex", encoding="utf-8") as f:
    tex = f.read()
used = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", tex):
    for k in m.group(1).split(","):
        used.add(k.strip())

with open("paper/references.bib", encoding="utf-8") as f:
    bib = f.read()
defined = set(re.findall(r"^@\w+\{([^,]+),", bib, re.MULTILINE))

print("Used keys:", len(used))
print("Defined keys:", len(defined))
print()
missing = used - defined
print("USED BUT NOT DEFINED:", missing if missing else "none -- all good")
print()
unused = defined - used
print("DEFINED BUT NEVER CITED:", unused if unused else "none")
