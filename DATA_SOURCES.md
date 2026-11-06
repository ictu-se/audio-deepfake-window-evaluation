# Data and model provenance

- RADAR evaluation release, version 1: https://doi.org/10.5281/zenodo.22110815 . Provider license: CC BY 4.0. Required archive RADAR2026-eval.tar.gz: 8,752,649,011 bytes, MD5 008b8c10bae684057d7f9d43bff2b55d. Metadata and source fingerprints are preserved in the study records.
- In-the-Wild: https://doi.org/10.21437/Interspeech.2022-108 . Author-hosted archive and pinned revision are in study/itw/external_plan.json, with member CRC/size metadata in candidate_manifest.csv. Only selected ZIP members were verified, not the whole archive. Provider metadata: CC BY-SA 4.0. Speaker names are replaced by stable hashes.
- AASIST / SSL-AASIST / RawNet2 / AntiDeepfake: frozen checkpoint hashes and score conventions are in study/frozen_run.json. Source revisions are in core/vendor/source_revisions.json and study/sources. The NII weights are CC BY-NC-SA 4.0 and must be obtained from the cited provider. No weights are redistributed.
- Silero VAD: pinned GitHub source revision and upstream MIT license are in study/sources. Its predictions are not human annotations.

Cohort identifiers and hashes are original locked records. Refactored code has new hashes in checksums.json; historical code hashes in frozen_run.json refer to the original implementation, retained in provenance/original_scripts (the historical launcher path is normalized to the current interpreter). Refactoring does not constitute a new preregistration. Selection manifests, not a new selection, define replication. Historical unused-candidate exclusion hashes are provenance; rerunning selection is not required.

The study reused released recordings and collected no new human data. This does not establish consent for unrestricted redistribution or an institutional ethics exemption. Aggregate results and hashed group identifiers are released; access to audio remains with source providers. Institutional ethics/consent determinations are not fabricated in this package.
