# Q1 feature bundle

This directory is a question-1 result, not a frozen I01/I02 exchange bundle.

- `q1_samples.csv`: labels, transcripts, IDs and sequence lengths.
- `q1_features.npz`: float32 `text/audio/vision` tensors and explicit boolean masks.
- `q1_alignment.csv`: zero-based text anchors mapped to half-open time/sample/frame ranges.
- `q1_feature_manifest.csv`: per-sample and per-modality quality rows.
- `q1_normalization.json`: confirms that no train-fitted normalization was applied.
- `q1_extraction_reproduction.json`: extraction command, versions, hashes and invariant checks.
- `../复现清单.json`: the single full-chain reproduction record.

Load with `numpy.load('q1_features.npz')` and `pandas.read_csv(...)`. Missingness must be read from masks; never infer it from zero-valued features.
