import re

with open("paper/hri_full_paper.tex", encoding="utf-8") as f:
    content = f.read()

for env in ["table", r"table\*", r"figure\*", "algorithm", "itemize", "enumerate", "equation", "aligned", "CCSXML", "abstract", "document"]:
    begins = len(re.findall(r"\\begin\{" + env + r"\}", content))
    ends = len(re.findall(r"\\end\{" + env + r"\}", content))
    status = "OK" if begins == ends else "MISMATCH!"
    print(f"{env}: begin={begins} end={ends} {status}")
