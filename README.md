# Where We Listen Matters in Audio Deepfake Detection

Paired evaluation of four frozen pipelines, two new cohorts (1,200 Vietnamese and 2,000 In-the-Wild files), 43 input conditions and a separate 400-file calibration cohort. The repository contains 552,000 expanded-study score rows plus the historical SSL padding data quoted in Methods, coverage predictions, manifests, uncertainty analysis, processing-time observations, and editable Matplotlib/table sources. It contains no manuscript, audio or weights.

## Reproduce the reported analysis

Use Python 3.9 (the recorded environment was Python 3.9 on macOS). From the repository root:

```sh
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_reproduction.py
python scripts/verify_historical_contrast.py
```

This verifies every packaged checksum, recomputes all eight 10,000-draw paired primary bootstraps and five statistical summaries, requires 13 outputs to be bitwise identical, regenerates the four article table bodies, and renders the three quantitative figures. No GPU, recordings, model downloads or API keys are needed. Typical execution takes a few minutes. See COLUMN_DEFINITIONS.txt for proportions, percentages, policy inputs, and interval interpretation.

The analysis runs from paths relative to the clone. `study/results` holds scored inputs and analysis outputs; `article_assets` holds source data and editable chart/table scripts. Original execution code and locking records are retained separately as provenance. Historical SSL data retain the earlier inconclusive padding result; they are not pooled into the new primary family. The union of earlier identifiers (including unused candidates) is supplied for exclusion verification, without speaker names. Hashes are SHA256 in checksums.json; Git commits identify a release.

## Fresh inference (optional, substantially larger downloads)

Read LICENSES.txt and DATA_SOURCES.md and comply with the original audio/model terms. Install `requirements-inference.txt` with pip 24.0 (older Hydra/OmegaConf metadata requires that installer). Use Python 3.9; modern Python dataclass changes are incompatible with the pinned Fairseq source. Obtain RADAR2026-eval.tar.gz from the cited provider (8.75 GB), then run:

```sh
python scripts/fetch_inference_assets.py
python scripts/acquire_audio.py --radar-archive /path/to/RADAR2026-eval.tar.gz
python study/scripts/measure_vad.py vi
python study/scripts/measure_vad.py itw
python study/scripts/run_inference.py --model aasist --device mps --output-dir rerun_scores
python study/scripts/run_inference.py --model ssl --device mps --output-dir rerun_scores
python study/scripts/run_inference.py --model rawnet2 --device mps --output-dir rerun_scores
python study/scripts/run_inference.py --model antideepfake --device mps --output-dir rerun_scores
```

The source downloader obtains upstream Fairseq at its pinned commit; classifier weights are checked against the frozen hashes. The XLS-R change removes unsupported configuration keys, without changing tensors. Fresh scores go to a separate directory, preserving published scores. CPU/CUDA are optional `--device` choices; they do not reproduce the original MPS execution exactly. New-cohort decoded arrays must match frozen cache hashes. The original experiment used float32, batch eight, four CPU threads and PyTorch 2.5.1 MPS. SSL-AASIST had known CPU–MPS score sensitivity; do not promise bitwise backend equivalence.

For repeat processing-time measurements, use `python study/scripts/benchmark_policies.py` after acquisition. It writes new timings to study/results; timings are descriptive, machine-dependent and end before final score reduction/host transfer. Mean-three is three inputs in one batch/call. File-cache state and shared-host workload were uncontrolled in the reported run. Repeating timing is not necessary for numerical score/statistical replication.

The saved-score workflow has been executed and validated in an isolated environment installed from requirements.txt. Fresh acquisition/inference entry points are refactored from the completed experiment but have not been rerun with all downloads in a clean environment. Network availability, provider permission and the pinned Python/Fairseq stack are prerequisites. Historical selection/probing scripts in provenance are records, not current entry points.

## Scientific scope

There was no training. The eight primary contrasts use nominal Bonferroni-allocated 99.375% percentile intervals; secondary contrasts use nominal 95% intervals. Vietnamese resampling is class-stratified by file; In-the-Wild resampling uses 53 provided groups. Restricted VAD analysis failed the predefined minimum-class rule and was not estimated. Native preprocessing, training mixtures and architecture differ together; the study does not identify a causal architectural mechanism, a universally best policy or an upstream-unseen benchmark for every model.
