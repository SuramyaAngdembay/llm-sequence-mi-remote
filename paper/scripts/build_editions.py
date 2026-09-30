#!/usr/bin/env python3
"""Build the three current editions, each with a separate log and provenance."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    for script in ('generate_tables.py','generate_figures.py'):
        subprocess.run([sys.executable,str(ROOT/'scripts'/script)],check=True)
    subprocess.run([sys.executable, str(ROOT.parent/'scripts/review/map_recovery_claims.py')], check=True)
    tectonic=os.environ.get('TECTONIC') or shutil.which('tectonic')
    if not tectonic and not shutil.which('pdflatex'):
        raise SystemExit('Install Tectonic or pdfLaTeX, or set TECTONIC to its executable.')
    builds=[]
    for path in (ROOT/'main.tex',ROOT/'companion/readable_full.tex',ROOT/'companion/plain_summary.tex'):
        out=path.parent/'build';out.mkdir(exist_ok=True)
        log=out/(path.stem+'.build.log')
        with log.open('w') as fh:
            if tectonic:
                cmd=[tectonic,'--keep-logs','--keep-intermediates','--outdir','build',path.name]
                subprocess.run(cmd,cwd=path.parent,stdout=fh,stderr=subprocess.STDOUT,check=True)
            else:
                cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build',path.name]
                subprocess.run(cmd,cwd=path.parent,stdout=fh,stderr=subprocess.STDOUT,check=True)
                if '\\bibdata' in (out/(path.stem+'.aux')).read_text():
                    subprocess.run(['bibtex','build/'+path.stem],cwd=path.parent,stdout=fh,stderr=subprocess.STDOUT,check=True)
                for _ in range(2):subprocess.run(cmd,cwd=path.parent,stdout=fh,stderr=subprocess.STDOUT,check=True)
        pdf=out/(path.stem+'.pdf')
        builds.append({'source':str(path.relative_to(ROOT)),'pdf':str(pdf.relative_to(ROOT)),
                       'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()})
        print('Built',pdf,flush=True)
    manifest={'registry_sha256':hashlib.sha256((ROOT/'data/recovery_registry.json').read_bytes()).hexdigest(),
              'engine':tectonic or shutil.which('pdflatex'),'editions':builds,
              'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(list((ROOT/'recovery_content').glob('*.tex')) +
                                          list((ROOT/'tables').glob('recovery_*.tex')) +
                                          [ROOT/'main.tex', ROOT/'companion/readable_full.tex',
                                           ROOT/'companion/plain_summary.tex', ROOT/'companion/readable_appendix.tex'])}}
    (ROOT/'build/recovery_build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
