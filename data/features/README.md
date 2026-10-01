# Experimental feature inputs

This directory is the canonical, read-only source collection for the JVCI PointPCA++ experimental analysis. It contains 292 CSV files used to audit inputs, construct normalized tables, run grouped validation, and reproduce runtime analyses. Derived artifacts belong under [`data/analysis/`](../analysis/), never here.

## Directory map

```text
data/features/
├── pointpca/
│   ├── matlab/
│   │   ├── pointpca1/<DATASET>/PointPCA1.csv
│   │   └── pointpca2/<DATASET>/PointPCA2.csv
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
| `pointpca/matlab/pointpca1/` | Legacy MATLAB PointPCA1 scalar output | APSIPA, UnB_PC | Retained baseline provenance |
| `pointpca/matlab/pointpca2/` | MATLAB PointPCA2 scalar output and timing | All eight inventoried datasets | PointPCA$^3$ and PointPCA++ runtime comparisons |
| `pointpca/pointpca3/` | 40 PointPCA$^3$ predictors at 1, 2, 4, 8, 16, and 32 workers | All eight inventoried datasets | Worker scaling, runtime comparison, and point-cloud modality |
| `cubemap/` | Six-view IQA outputs for crop/padding variants | APSIPA controlled study; one isolated ICIP2023 auxiliary file | Projection-pipeline and IQA study |
| `pointpcapp/complete/` | Selected 46-feature PointPCA++ input and complete-pipeline timing | APSIPA, ICIP2023, LS-PCQA, SJTU-PCQA, UnB_PC, WPC, WPC2 | Final within-dataset and transfer evaluation |

QoMEX2019 is available in the PointPCA2 and PointPCA$^3$ families, but has no complete PointPCA++ input file. It remains pending for final PointPCA++ evaluation.

## CSV contract

The composite key is `(SIGNAL, REF)`. Every analysis joins sources on that key and requires it to be unique within a file. The usual metadata are `SIGNAL`, `REF`, `ATTACK`, `CLASS`, `SCORE`, `REF_NUM_POINTS`, `TEST_NUM_POINTS`, `TIME_TAKEN_SECONDS`, and `PEAK_RAM_USAGE_GiB`; some complete PointPCA++ files may omit metadata not required by their final-evaluation contract.

Predictor blocks are fixed:

- MATLAB PointPCA2: one scalar `FV_PointPCA2_0`.
- PointPCA$^3$: 40 columns `FV_PointPCA3-Rust_0` through `FV_PointPCA3-Rust_39`.
- Cubemap IQA: six columns `FV_Cubemap_0` through `FV_Cubemap_5`.
- Complete PointPCA++: 46 columns `FV_PointPCAPP-Complete_0` through `FV_PointPCAPP-Complete_45`; columns 0--39 are PointPCA$^3$ and columns 40--45 are DISTS views.

`TIME_TAKEN_SECONDS` records an artifact-reported per-sample interval. The complete PointPCA++ files measure the implemented sequential PointPCA$^3$-then-cubemap pipeline after warm-up, including point counting, loading, extraction, projection, cropping, Navier--Stokes inpainting, and DISTS evaluation. They exclude CSV serialization and Extra Trees inference.

## Frozen complete configuration

The selected complete PointPCA++ input uses PointPCA$^3$, cropping, Navier--Stokes inpainting, and six DISTS view scores. Complete extraction used **32 workers**, confirmed by the author. This supersedes the provisional 16-worker attribution in historical run manifests, without changing recorded times. The separate PointPCA³ worker-scaling study retains all six worker settings and its 16-worker baseline comparison. Current timing provenance is recorded in `../analysis/manifests/runtime_provenance_confirmation.json`.

## Integrity and updates

Do not edit, rename, or overwrite an existing CSV. Before consuming a file, record its SHA-256 hash; the analysis manifests provide this provenance for each run. New datasets or regenerated feature families must:

1. follow the appropriate path and filename pattern above;
2. preserve the required metadata, key uniqueness, and fixed predictor-block order;
3. document the generator, commit, hardware, worker count, and timing scope; and
4. pass the input audit before they are used in manuscript results.

The pre-migration `pointpcap_features/` and `pointpcapp_complete/` paths are not active input locations. Older manifests may mention them only as the historical paths from which earlier runs were recorded.
