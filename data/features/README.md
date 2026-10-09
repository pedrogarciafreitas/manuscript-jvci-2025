# Experimental feature inputs

This directory contains 300 read-only source CSVs: the 292 original JVCI inputs and eight historical predictor supplements retained on 2026-10-09. Existing JVCI predictors, benchmarks, metadata, populations, and folds remain authoritative. Keep generated results separate from this source collection.

## Directory map

```text
data/features/
├── pointpca/
│   ├── matlab/
│   │   ├── pointpca1/<DATASET>/PointPCA1.csv
│   │   └── pointpca2/
│   │       ├── benchmark/<DATASET>/PointPCA2.csv
│   │       └── features/<DATASET>/PointPCA2_matlab.csv
│   ├── pointpca2-rs/<DATASET>/PointPCA2_rs.csv
│   └── pointpca3/<DATASET>/<WORKERS>/PointPCA3-Rust.csv
├── cubemap/
│   ├── crop_padding/<PADDING_METHOD>/<DATASET>/<IQA>/Cubemap.csv
│   ├── only_padding/<PADDING_METHOD>/<DATASET>/<IQA>/Cubemap.csv
│   ├── only_crop/<DATASET>/<IQA>/Cubemap.csv
│   └── nocrop_nopadding/<DATASET>/<IQA>/Cubemap.csv
└── pointpcapp/
    └── complete/<DATASET>/PointPCAPP-Complete.csv
```

All paths in this README are relative to `data/features/`.

## Feature families and coverage

| Family | Contents | Dataset coverage | Primary use |
|---|---|---|---|
| `pointpca/matlab/pointpca1/` | MATLAB PointPCA1; 46 finite predictors on APSIPA; all 40 feature fields empty on UnB_PC | APSIPA usable; UnB_PC provenance only | Original PointPCA baseline; APSIPA feature semantics still need audit |
| `pointpca/matlab/pointpca2/benchmark/` | Original JVCI MATLAB PointPCA2 records: one feature column and authoritative timing | All eight inventoried datasets | PointPCA$^3$ and PointPCA++ runtime comparisons |
| `pointpca/matlab/pointpca2/features/` | Historical MATLAB PointPCA2, 40 predictors | APSIPA, ICIP2023, SJTU-PCQA, UnB_PC | Standalone prediction and per-feature comparisons |
| `pointpca/pointpca2-rs/` | Historical Rust PointPCA2-RS, 40 predictors | APSIPA, ICIP2023, SJTU-PCQA, UnB_PC | Matched RS versus MATLAB/JVCI comparisons |
| `pointpca/pointpca3/` | 40 PointPCA$^3$ predictors at 1, 2, 4, 8, 16, and 32 workers | All eight inventoried datasets | Worker scaling, runtime comparison, and point-cloud modality |
| `cubemap/` | Six-view IQA outputs for crop/padding variants | APSIPA controlled study; one isolated ICIP2023 auxiliary file | Projection-pipeline and IQA study |
| `pointpcapp/complete/` | Selected 46-feature PointPCA++ input and complete-pipeline timing | APSIPA, ICIP2023, LS-PCQA, SJTU-PCQA, UnB_PC, WPC, WPC2 | Final within-dataset and transfer evaluation |

QoMEX2019 remains in the original JVCI benchmark and PointPCA$^3$ inventory, but is setup-only and excluded from revised analyses. It has no complete PointPCA++ input file.

## Method names and historical supplements

PointPCA = PointPCA1; PointPCA+ = PointPCA2; PointPCA+RS = PointPCA2-RS, the Rust reimplementation described in the SBrT paper; PointPCA³ = PointPCA3, the VoSS-based version described in BMSB. Preserve the recorded filenames and column identifiers.

The four supplemented datasets contain 1,568 samples from 68 reference contents. Each MATLAB/RS file has unique keys, 40 finite predictors, and populations, MOS, class, and point counts matching JVCI; persisted JVCI fold coverage is complete. Future comparisons must reuse those folds and the fixed JVCI regressor. Feature-index semantics still require verification before per-feature agreement claims. Original PointPCA may join the APSIPA comparison after its 46-column schema is audited; its missing coverage elsewhere does not block the four-dataset comparison.

The supplements were retained byte-for-byte from the author-supplied BMSB collection. Their SBrT counterparts were byte-identical. Historical timing and memory columns remain in these raw CSVs for provenance but are **excluded from JVCI benchmarks**. Benchmark loaders must read `pointpca/matlab/pointpca2/benchmark/` explicitly, never recursively scan the parent `pointpca2/` directory. Do not substitute historical PointPCA³ data for JVCI predictors or benchmarks.

## CSV contract

The composite key is `(SIGNAL, REF)`. Every analysis joins sources on that key and requires it to be unique within a file. The usual metadata are `SIGNAL`, `REF`, `ATTACK`, `CLASS`, `SCORE`, `REF_NUM_POINTS`, `TEST_NUM_POINTS`, `TIME_TAKEN_SECONDS`, and `PEAK_RAM_USAGE_GiB`; some complete PointPCA++ files may omit metadata not required by their final-evaluation contract.

Predictor blocks are fixed:

- MATLAB PointPCA2 benchmark files: one scalar `FV_PointPCA2_0`; this is not the complete predictor vector.
- Historical MATLAB PointPCA2 and Rust PointPCA2-RS: 40 columns `QUALITY_INDEX_PointPCA2_0` through `QUALITY_INDEX_PointPCA2_39`, in their original order. The shared column prefix does not identify the implementation; use the method-specific path.
- PointPCA$^3$: 40 columns `FV_PointPCA3-Rust_0` through `FV_PointPCA3-Rust_39`.
- Cubemap IQA: six columns `FV_Cubemap_0` through `FV_Cubemap_5`.
- Complete PointPCA++: 46 columns `FV_PointPCAPP-Complete_0` through `FV_PointPCAPP-Complete_45`; columns 0--39 are PointPCA$^3$ and columns 40--45 are DISTS views.

`TIME_TAKEN_SECONDS` records an artifact-reported per-sample interval. Only the original JVCI benchmark and pipeline files supply current runtime results; the two historical predictor families are feature-only inputs for analysis. The complete PointPCA++ files measure the implemented sequential PointPCA$^3$-then-cubemap pipeline after warm-up, including point counting, loading, extraction, projection, cropping, Navier--Stokes inpainting, and DISTS evaluation. They exclude CSV serialization and Extra Trees inference.

## Frozen complete configuration

The selected complete PointPCA++ input uses PointPCA$^3$, cropping, Navier--Stokes inpainting, and six DISTS view scores. Complete extraction used **32 workers**, confirmed by the author. This supersedes the provisional 16-worker attribution in historical run manifests, without changing recorded times. The separate PointPCA³ worker-scaling study retains all six worker settings and its 16-worker baseline comparison.

## Integrity and updates

Do not edit or overwrite source CSV contents. Author-approved relocations must preserve bytes. Consumers should record source paths and SHA-256 hashes with their analysis runs. New datasets or regenerated feature families must:

1. follow the appropriate path and filename pattern above;
2. preserve the required metadata, key uniqueness, and fixed predictor-block order;
3. document the generator, commit, hardware, worker count, and timing scope; and
4. pass the input audit before they are used in manuscript results.

## Completed organization and deletion — 2026-10-09

Eight original JVCI PointPCA2 files moved into `benchmark/`, and eight historical predictor files were retained under `features/` and `pointpca2-rs/`. All 292 original JVCI source contents were preserved.

The remaining 63 historical files were deleted: 43 BMSB PointPCA³ files, five BMSB baseline files outside the selected comparison, and all 15 SBrT files (including the byte-identical duplicates and incomplete WPC/WPC2 MATLAB extracts). Both original import directories were removed. MATLAB failures and memory exhaustion motivate using complete datasets; individual missing rows are not attributed to a particular failure without logs.

The pre-migration `pointpcap_features/`, `pointpcapp_complete/`, and `pointpca/matlab/pointpca2/<DATASET>/` paths are historical. Consumers should use the current directory layout above. No models, statistical results, figures, or manuscript results were regenerated during this cleanup.
