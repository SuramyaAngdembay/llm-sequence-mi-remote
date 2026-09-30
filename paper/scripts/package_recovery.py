"""Package only the current manuscript inputs; never include historical tables."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root=Path(__file__).resolve().parents[1]
files=[root/n for n in ['main.tex','tmlr.sty','tmlr.bst','fancyhdr.sty','references.bib']]
files+=list((root/'recovery_content').glob('*.tex'))
files+=list((root/'tables').glob('recovery_*.tex'))
files+=[root/'figures/recovery_score_mitigation.pdf',root/'data/recovery_registry.json']
with ZipFile(root/'build/overleaf.zip','w',ZIP_DEFLATED) as z:
    for p in files:
        if p.exists():z.write(p,p.relative_to(root))
print(root/'build/overleaf.zip')
