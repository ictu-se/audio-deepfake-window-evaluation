"""Reproduce the historical inconclusive padding contrast quoted in Methods."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('historical_analysis',ROOT/'provenance/analyze_shortform_original.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
_,y,s=m.matrix(ROOT/'historical/main_vi_ssl.csv',ROOT/'historical/main_vi_manifest.csv')
result,draws=m.primary(y,s,draws=2000)
expected=json.loads((ROOT/'historical/padding_contrast_ssl.json').read_text())
assert result['estimate_percentage_points']==expected['estimate_percentage_points']
assert result['interval_97_5_percentage_points']==expected['interval_97_5_percentage_points']
assert np.array_equal(draws,np.load(ROOT/'historical/bootstrap_primary_ssl.npy',allow_pickle=False))
print('PASS: historical SSL padding contrast and 2,000 bootstrap draws reproduced exactly.')
