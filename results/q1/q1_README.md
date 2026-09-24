# Q1 feature bundle

This directory is a question-1 result, not a frozen I01/I02 exchange bundle.

- `q1_samples.csv`: labels, transcripts, IDs and sequence lengths.
- `q1_features.npz`: aligned-50 float32 `text/audio/vision` tensors, valid/injected/padding masks, and derived `observed_*` masks.
- `q1_features_unaligned.npz`: independent audio/vision sequences padded or deterministically resampled to 500 positions, with seconds and source-frame maps.
- `q1_alignment.csv`: zero-based anchors mapped to half-open time ranges, 16 kHz sample indexes, 10 fps sampled-frame indexes, source-frame indexes and milliseconds.
- `q1_feature_manifest.csv`: per-sample and per-modality quality rows.
- `q1_data_quality.csv`: declared versus decodable durations, truncation flags and face-detection rates.
- `q1_feature_statistics.csv`: per-dimension valid-position statistics and zero-variance flags.
- `q1_normalization.json`: explains why train-fitted normalization is deferred to I01.
- `q1_extraction_reproduction.json`: extraction command, versions, hashes and invariant checks.
- `../复现清单.json`: the single full-chain reproduction record.

Load with `numpy.load('q1_features.npz')` and `pandas.read_csv(...)`. All index intervals are left-closed/right-open. `audio_*_idx` uses 16 kHz samples; `video_*_idx` uses the 10 fps sampled list; `video_*_frame_src` uses decoded source-frame numbers. Missingness must be read from masks; never infer it from zero-valued features. Empty transcripts use 50 uniform time bins with `valid_text=False`, `padding_mask=False`, and `alignment_source=uniform_missing_text`.
