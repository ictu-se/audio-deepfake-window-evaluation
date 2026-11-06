"""Recompute the locked statistics and article tables/figures; fail on numerical drift."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES=['expansion_summary.json','all_cells.csv','policies.csv','input_placement.csv','offset_profiles.csv']+[f'bootstrap_{g}_{m}.npy' for g in ('vi','itw') for m in ('aasist','ssl','rawnet2','antideepfake')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 hashes=json.loads((ROOT/'checksums.json').read_text())
 for name,digest in hashes.items():
  assert sha(ROOT/name)==digest,name
 expected={n:sha(ROOT/'study/results'/n) for n in NAMES}
 subprocess.run([sys.executable,'study/scripts/analyze_expansion.py'],cwd=ROOT,check=True)
 actual={n:sha(ROOT/'study/results'/n) for n in NAMES}
 assert expected==actual,[n for n in NAMES if expected[n]!=actual[n]]
 tables=list((ROOT/'article_assets/data').glob('tab_*.tex'));before={q.name:sha(q) for q in tables}
 subprocess.run([sys.executable,'article_assets/scripts/make_tables.py'],cwd=ROOT,check=True)
 assert before=={q.name:sha(q) for q in tables},'Table values differ'
 subprocess.run([sys.executable,'article_assets/figures/plot_figures.py'],cwd=ROOT,check=True)
 print('PASS: checksums; 13 statistics outputs bitwise identical; table sources identical; three Matplotlib figures regenerated.')
if __name__=='__main__':main()
