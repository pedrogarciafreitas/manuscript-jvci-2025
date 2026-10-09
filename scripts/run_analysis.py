#!/usr/bin/env python3
"""Reproducible experimental analysis for the JVCI PointPCA++ manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pointpcapp-matplotlib")

import lazypredict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from lazypredict.Supervised import LazyRegressor, REGRESSORS
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


SEED = 42
WORKERS = [1, 2, 4, 8, 16, 32]
DATASETS = [
    "APSIPA",
    "ICIP2023",
    "LS-PCQA",
    "QoMEX2019",
    "SJTU-PCQA",
    "UnB_PC",
    "WPC",
    "WPC2",
]
FINAL_DATASETS = [
    "APSIPA",
    "ICIP2023",
    "LS-PCQA",
    "SJTU-PCQA",
    "UnB_PC",
    "WPC",
    "WPC2",
]
MODEL_SELECTION_DATASETS = ["APSIPA", "WPC", "LS-PCQA"]
EVALUATION_ONLY_DATASETS = [
    dataset for dataset in FINAL_DATASETS if dataset not in MODEL_SELECTION_DATASETS
]
CORE_FEATURE_SETS = {
    "pointpca3_only": [f"pc3__f{index:02d}" for index in range(40)],
    "dists_only": [
        f"proj__crop_pad_navier__distsmetric__view{index}" for index in range(6)
    ],
}
CORE_FEATURE_SETS["fusion_46"] = (
    CORE_FEATURE_SETS["pointpca3_only"] + CORE_FEATURE_SETS["dists_only"]
)
FINAL_EXPECTED_POPULATIONS = {
    "APSIPA": (232, 8),
    "ICIP2023": (898, 45),
    "LS-PCQA": (930, 85),
    "SJTU-PCQA": (378, 9),
    "UnB_PC": (60, 6),
    "WPC": (740, 20),
    "WPC2": (400, 16),
}
META_COLUMNS = [
    "SIGNAL",
    "REF",
    "ATTACK",
    "CLASS",
    "SCORE",
    "REF_NUM_POINTS",
    "TEST_NUM_POINTS",
    "TIME_TAKEN_SECONDS",
    "PEAK_RAM_USAGE_GiB",
]
KEY_COLUMNS = ["SIGNAL", "REF"]
COMPARE_META = ["SCORE", "CLASS", "REF_NUM_POINTS", "TEST_NUM_POINTS"]
IQAS = [
    "AHIQMetric",
    "CKDNMetric",
    "CW_SSIMMetric",
    "DISTSMetric",
    "DSSMetric",
    "FSIMMetric",
    "GMSDMetric",
    "HaarPSIMetric",
    "MADMetric",
    "MDSIMetric",
    "MSGMSDMetric",
    "MSSSIMMetric",
    "NLPDMetric",
    "PIEAPPMetric",
    "PSNRMetric",
    "SRSSIMMetric",
    "SSIMCMetric",
    "SSIMMetric",
    "TOPIQ_FRMetric",
    "VIFMetric",
    "VSIMetric",
    "WADIQAM_FRMetric",
]
PIPELINES = {
    "crop_pad_fsr": ("crop_padding", "FSRFastPadding"),
    "crop_pad_navier": ("crop_padding", "NavierStokesPadding"),
    "crop_pad_telea": ("crop_padding", "TeleaPadding"),
    "crop_pad_shiftmap": ("crop_padding", "ShiftmapPadding"),
    "nocrop_pad_fsr": ("only_padding", "FSRFastPadding"),
    "nocrop_pad_navier": ("only_padding", "NavierStokesPadding"),
    "nocrop_pad_telea": ("only_padding", "TeleaPadding"),
    "nocrop_pad_shiftmap": ("only_padding", "ShiftmapPadding"),
    "crop_nopad": ("only_crop", None),
    "nocrop_nopad": ("nocrop_nopadding", None),
}
PIPELINE_FAMILY = {
    "crop_pad_fsr": "crop + padding",
    "crop_pad_navier": "crop + padding",
    "crop_pad_telea": "crop + padding",
    "crop_pad_shiftmap": "crop + padding",
    "nocrop_pad_fsr": "no crop + padding",
    "nocrop_pad_navier": "no crop + padding",
    "nocrop_pad_telea": "no crop + padding",
    "nocrop_pad_shiftmap": "no crop + padding",
    "crop_nopad": "crop + no padding",
    "nocrop_nopad": "no crop + no padding",
}

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[1]
WORKSPACE = REPO.parent
FEATURES = REPO / "data" / "features"
MATLAB_FEATURES = FEATURES / "pointpca" / "matlab"
POINTPCA2_BENCHMARKS = MATLAB_FEATURES / "pointpca2" / "benchmark"
POINTPCA3_FEATURES = FEATURES / "pointpca" / "pointpca3"
COMPLETE_FEATURES = FEATURES / "pointpcapp" / "complete"
OUT = REPO / "data" / "analysis"
DIRS = {
    name: OUT / name
    for name in [
        "manifests",
        "audits",
        "normalized",
        "folds",
        "predictions",
        "summaries",
        "figures",
        "tables",
        "logs",
        "checkpoints",
    ]
}
for directory in DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
np.random.seed(SEED)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_saved_source_path(relative_path: str) -> Path:
    """Resolve relocated inputs without rewriting historical hash manifests."""
    path = REPO / relative_path
    if path.exists():
        return path
    recorded = Path(relative_path)
    relocations = {
        Path("data/features/pointpca/matlab/pointpca2"): POINTPCA2_BENCHMARKS,
        Path("pointpcap_features/pointpca2_matlab"): POINTPCA2_BENCHMARKS,
        Path("pointpcap_features/pointpca3"): POINTPCA3_FEATURES,
        Path("pointpcap_features/cubemap"): FEATURES / "cubemap",
        Path("pointpcapp_complete"): COMPLETE_FEATURES,
    }
    for original, current in relocations.items():
        if recorded.is_relative_to(original):
            candidate = current / recorded.relative_to(original)
            if candidate.exists():
                return candidate
    return path


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def iqr(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    s = float(spearmanr(y_true, y_pred).statistic)
    p = float(pearsonr(y_true, y_pred).statistic)
    r = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return {"srocc": s, "plcc": p, "rmse": r}


def validate_frame(df: pd.DataFrame, required: list[str], source: Path) -> list[str]:
    errors = []
    missing = sorted(set(required) - set(df.columns))
    if missing:
        errors.append(f"missing columns: {missing}")
        return errors
    if df.duplicated(KEY_COLUMNS).any():
        errors.append("duplicate (SIGNAL, REF) keys")
    numeric = [
        c for c in required if c not in ["SIGNAL", "REF", "ATTACK", "CLASS"]
    ]
    for column in numeric:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            errors.append(f"nonnumeric/missing values in {column}")
        if np.isinf(values.to_numpy(dtype=float)).any():
            errors.append(f"infinite values in {column}")
    if "TIME_TAKEN_SECONDS" in df and (df["TIME_TAKEN_SECONDS"] < 0).any():
        errors.append("negative TIME_TAKEN_SECONDS")
    return errors


def assert_same_population(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    l = left.sort_values(KEY_COLUMNS).reset_index(drop=True)
    r = right.sort_values(KEY_COLUMNS).reset_index(drop=True)
    if list(map(tuple, l[KEY_COLUMNS].to_numpy())) != list(
        map(tuple, r[KEY_COLUMNS].to_numpy())
    ):
        raise ValueError(f"population mismatch: {label}")
    for column in COMPARE_META:
        if column == "CLASS":
            equal = l[column].fillna("<NA>").astype(str).equals(
                r[column].fillna("<NA>").astype(str)
            )
        else:
            equal = np.allclose(
                l[column].to_numpy(), r[column].to_numpy(), equal_nan=True
            )
        if not equal:
            raise ValueError(f"metadata mismatch for {column}: {label}")


def projection_path(pipeline: str, iqa: str) -> Path:
    variant, padder = PIPELINES[pipeline]
    root = FEATURES / "cubemap" / variant
    if padder:
        root = root / padder
    return root / "APSIPA" / iqa / "Cubemap.csv"


def load_projection(pipeline: str, iqa: str) -> pd.DataFrame:
    path = projection_path(pipeline, iqa)
    df = pd.read_csv(path)
    rename = {
        f"FV_Cubemap_{view}": f"proj__{pipeline}__{iqa.lower()}__view{view}"
        for view in range(6)
    }
    return df.rename(columns=rename)


def projection_features(pipeline: str, iqa: str) -> list[str]:
    return [f"proj__{pipeline}__{iqa.lower()}__view{view}" for view in range(6)]


def load_pc3(dataset: str, workers: int) -> pd.DataFrame:
    path = POINTPCA3_FEATURES / dataset / str(workers) / "PointPCA3-Rust.csv"
    df = pd.read_csv(path)
    rename = {f"FV_PointPCA3-Rust_{i}": f"pc3__f{i:02d}" for i in range(40)}
    return df.rename(columns=rename)


def pc3_features() -> list[str]:
    return [f"pc3__f{i:02d}" for i in range(40)]


def audit() -> None:
    rows = []
    hashes = []
    errors = []
    source_files = sorted(POINTPCA2_BENCHMARKS.rglob("*.csv"))
    source_files += sorted(POINTPCA3_FEATURES.rglob("*.csv"))
    source_files += sorted(
        {projection_path(pipeline, iqa) for pipeline in PIPELINES for iqa in IQAS}
    )
    for path in source_files:
        df = pd.read_csv(path)
        fv = [c for c in df if c.startswith("FV_")]
        required = META_COLUMNS + fv
        found = validate_frame(df, required, path)
        rows.append(
            {
                "path": str(path.relative_to(REPO)),
                "rows": len(df),
                "columns": len(df.columns),
                "feature_columns": len(fv),
                "reference_groups": int(df["REF"].nunique()) if "REF" in df else None,
                "errors": " | ".join(found),
            }
        )
        hashes.append(
            {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
        errors.extend([f"{path}: {error}" for error in found])

    for dataset in DATASETS:
        matlab = pd.read_csv(
            POINTPCA2_BENCHMARKS / dataset / "PointPCA2.csv"
        )
        base_raw = pd.read_csv(
            POINTPCA3_FEATURES / dataset / "1" / "PointPCA3-Rust.csv"
        )
        assert_same_population(matlab, base_raw, f"{dataset}: MATLAB versus PC3")
        feature_columns = [c for c in base_raw if c.startswith("FV_PointPCA3-Rust_")]
        if len(feature_columns) != 40:
            errors.append(f"{dataset}: expected 40 PointPCA3 features")
        base = base_raw.sort_values(KEY_COLUMNS).reset_index(drop=True)
        for worker in WORKERS[1:]:
            other = pd.read_csv(
                POINTPCA3_FEATURES
                / dataset
                / str(worker)
                / "PointPCA3-Rust.csv"
            ).sort_values(KEY_COLUMNS).reset_index(drop=True)
            assert_same_population(base, other, f"{dataset}: PC3 workers {worker}")
            if not np.array_equal(
                base[feature_columns].to_numpy(), other[feature_columns].to_numpy()
            ):
                errors.append(f"{dataset}: PointPCA3 features differ at {worker} workers")

    projection_frames = []
    for pipeline in PIPELINES:
        for iqa in IQAS:
            path = projection_path(pipeline, iqa)
            if not path.exists():
                errors.append(f"missing balanced projection file: {path}")
                continue
            df = pd.read_csv(path)
            if len([c for c in df if c.startswith("FV_Cubemap_")]) != 6:
                errors.append(f"expected six cubemap features: {path}")
            projection_frames.append((pipeline, iqa, df))
    if projection_frames:
        base = projection_frames[0][2]
        for pipeline, iqa, frame in projection_frames[1:]:
            assert_same_population(base, frame, f"projection {pipeline}/{iqa}")

    pd.DataFrame(rows).to_csv(DIRS["audits"] / "input_audit.csv", index=False)
    pd.DataFrame(hashes).to_csv(DIRS["manifests"] / "source_hashes.csv", index=False)
    report = {
        "source_csvs": len(source_files),
        "balanced_projection_cases": len(projection_frames),
        "expected_projection_cases": len(PIPELINES) * len(IQAS),
        "errors": errors,
    }
    write_json(DIRS["audits"] / "input_audit.json", report)
    if errors:
        raise RuntimeError("Input audit failed:\n" + "\n".join(errors))


def build_normalized() -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    pc3 = load_pc3("APSIPA", 16).sort_values(KEY_COLUMNS).reset_index(drop=True)
    keep = [c for c in META_COLUMNS if c != "PEAK_RAM_USAGE_GiB"] + pc3_features()
    pc3 = pc3[keep].copy()
    pc3 = pc3.rename(columns={"TIME_TAKEN_SECONDS": "time__pc3__w16"})
    pc3.to_csv(DIRS["normalized"] / "apsipa_pc3_w16.csv", index=False)

    projections = {}
    dictionary = [
        {
            "normalized": name,
            "source": f"FV_PointPCA3-Rust_{i}",
            "modality": "pointpca3",
            "pipeline": "",
            "iqa": "",
            "view": "",
        }
        for i, name in enumerate(pc3_features())
    ]
    for pipeline in PIPELINES:
        for iqa in IQAS:
            frame = load_projection(pipeline, iqa).sort_values(KEY_COLUMNS).reset_index(drop=True)
            features = projection_features(pipeline, iqa)
            frame = frame[
                KEY_COLUMNS + COMPARE_META + ["TIME_TAKEN_SECONDS"] + features
            ].copy()
            frame = frame.rename(
                columns={"TIME_TAKEN_SECONDS": f"time__proj__{pipeline}__{iqa.lower()}"}
            )
            projections[(pipeline, iqa)] = frame
            for view, name in enumerate(features):
                dictionary.append(
                    {
                        "normalized": name,
                        "source": f"FV_Cubemap_{view}",
                        "modality": "projection",
                        "pipeline": pipeline,
                        "iqa": iqa,
                        "view": view,
                    }
                )
    pd.DataFrame(dictionary).to_csv(
        DIRS["normalized"] / "feature_dictionary.csv", index=False
    )
    return pc3, projections


def create_folds(base: pd.DataFrame) -> tuple[list, dict[int, list]]:
    outer = list(GroupKFold(n_splits=5).split(base, base["SCORE"], base["REF"]))
    inner_by_outer = {}
    rows = []
    for outer_fold, (train_idx, test_idx) in enumerate(outer):
        train_set, test_set = set(train_idx), set(test_idx)
        inner = list(
            GroupKFold(n_splits=5).split(
                base.iloc[train_idx],
                base.iloc[train_idx]["SCORE"],
                base.iloc[train_idx]["REF"],
            )
        )
        inner_by_outer[outer_fold] = inner
        for idx in range(len(base)):
            rows.append(
                {
                    "SIGNAL": base.iloc[idx]["SIGNAL"],
                    "REF": base.iloc[idx]["REF"],
                    "outer_fold": outer_fold,
                    "inner_fold": -1,
                    "role": "train" if idx in train_set else "test",
                }
            )
        for inner_fold, (itr_rel, iva_rel) in enumerate(inner):
            itr = set(train_idx[itr_rel])
            iva = set(train_idx[iva_rel])
            for idx in train_idx:
                rows.append(
                    {
                        "SIGNAL": base.iloc[idx]["SIGNAL"],
                        "REF": base.iloc[idx]["REF"],
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "role": "train" if idx in itr else "validation",
                    }
                )
            assert not set(base.iloc[list(itr)]["REF"]) & set(base.iloc[list(iva)]["REF"])
        assert not set(base.iloc[list(train_set)]["REF"]) & set(
            base.iloc[list(test_set)]["REF"]
        )
    pd.DataFrame(rows).to_csv(
        DIRS["folds"] / "apsipa_nested_groupkfold.csv", index=False
    )
    return outer, inner_by_outer


def fit_extra_trees_predict(X_train, y_train, X_test) -> np.ndarray:
    model = ExtraTreesRegressor(random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def rank_projection(summary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    ranked = []
    for iqa, frame in summary.groupby("iqa"):
        frame = frame.sort_values(
            ["srocc", "plcc", "rmse", "median_time"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        frame["rank"] = np.arange(1, len(frame) + 1)
        ranked.append(frame)
    ranked = pd.concat(ranked, ignore_index=True)
    aggregate = (
        ranked.groupby("pipeline")
        .agg(
            median_rank=("rank", "median"),
            mean_rank=("rank", "mean"),
            rank_iqr=("rank", iqr),
            median_srocc=("srocc", "median"),
            median_plcc=("plcc", "median"),
            median_rmse=("rmse", "median"),
            median_time=("median_time", "median"),
        )
        .reset_index()
        .sort_values(
            ["median_rank", "mean_rank", "rank_iqr", "median_time"],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )
    aggregate["aggregate_rank"] = np.arange(1, len(aggregate) + 1)
    aggregate["family"] = aggregate["pipeline"].map(PIPELINE_FAMILY)
    return aggregate, str(aggregate.iloc[0]["pipeline"])


def projection_analysis(base, projections, outer, inner_by_outer):
    prediction_rows = []
    fold_rows = []
    for pipeline in PIPELINES:
        for iqa in IQAS:
            frame = projections[(pipeline, iqa)]
            features = projection_features(pipeline, iqa)
            for fold, (train_idx, test_idx) in enumerate(outer):
                pred = fit_extra_trees_predict(
                    frame.iloc[train_idx][features],
                    frame.iloc[train_idx]["SCORE"],
                    frame.iloc[test_idx][features],
                )
                score = metrics(frame.iloc[test_idx]["SCORE"], pred)
                fold_rows.append(
                    {"pipeline": pipeline, "iqa": iqa, "fold": fold, **score}
                )
                for idx, value in zip(test_idx, pred):
                    prediction_rows.append(
                        {
                            "pipeline": pipeline,
                            "iqa": iqa,
                            "fold": fold,
                            "SIGNAL": frame.iloc[idx]["SIGNAL"],
                            "REF": frame.iloc[idx]["REF"],
                            "y_true": frame.iloc[idx]["SCORE"],
                            "y_pred": value,
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    predictions.to_csv(
        DIRS["predictions"] / "projection_case_predictions.csv", index=False
    )
    folds.to_csv(DIRS["summaries"] / "projection_case_fold_metrics.csv", index=False)
    summary_rows = []
    for (pipeline, iqa), frame in folds.groupby(["pipeline", "iqa"]):
        source = projections[(pipeline, iqa)]
        summary_rows.append(
            {
                "pipeline": pipeline,
                "family": PIPELINE_FAMILY[pipeline],
                "iqa": iqa,
                "srocc": frame["srocc"].median(),
                "plcc": frame["plcc"].median(),
                "rmse": frame["rmse"].median(),
                "srocc_iqr": iqr(frame["srocc"]),
                "plcc_iqr": iqr(frame["plcc"]),
                "rmse_iqr": iqr(frame["rmse"]),
                "median_time": source.filter(like="time__proj__").iloc[:, 0].median(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DIRS["summaries"] / "projection_case_summary.csv", index=False)
    ranks, selected = rank_projection(summary)
    ranks.to_csv(DIRS["summaries"] / "projection_pipeline_ranks.csv", index=False)

    selection_rows = []
    for outer_fold, (outer_train, _) in enumerate(outer):
        inner_rows = []
        for pipeline in PIPELINES:
            for iqa in IQAS:
                frame = projections[(pipeline, iqa)].iloc[outer_train].reset_index(drop=True)
                features = projection_features(pipeline, iqa)
                scores = []
                for inner_fold, (train_rel, valid_rel) in enumerate(
                    inner_by_outer[outer_fold]
                ):
                    pred = fit_extra_trees_predict(
                        frame.iloc[train_rel][features],
                        frame.iloc[train_rel]["SCORE"],
                        frame.iloc[valid_rel][features],
                    )
                    scores.append(metrics(frame.iloc[valid_rel]["SCORE"], pred))
                inner_rows.append(
                    {
                        "pipeline": pipeline,
                        "iqa": iqa,
                        "srocc": np.median([x["srocc"] for x in scores]),
                        "plcc": np.median([x["plcc"] for x in scores]),
                        "rmse": np.median([x["rmse"] for x in scores]),
                        "median_time": frame.filter(like="time__proj__").iloc[:, 0].median(),
                    }
                )
        inner_summary = pd.DataFrame(inner_rows)
        inner_ranks, winner = rank_projection(inner_summary)
        for _, row in inner_ranks.iterrows():
            selection_rows.append(
                {"outer_fold": outer_fold, "selected": row["pipeline"] == winner, **row.to_dict()}
            )
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(DIRS["summaries"] / "projection_outer_selection.csv", index=False)
    frequency = (
        selection[selection["selected"]]
        .groupby("pipeline")
        .size()
        .rename("outer_selection_count")
    )
    ranks["outer_selection_count"] = ranks["pipeline"].map(frequency).fillna(0).astype(int)
    ranks.to_csv(DIRS["summaries"] / "projection_pipeline_ranks.csv", index=False)
    write_json(
        DIRS["manifests"] / "projection_selected_pipeline.json",
        {
            "selected_pipeline": selected,
            "selection_rule": "median rank, mean rank, rank IQR, median time",
            "outer_selection_frequency": frequency.to_dict(),
        },
    )
    return selected, selection


def build_fusion(base, projections, pipeline: str, save: bool = False) -> pd.DataFrame:
    fusion = base[KEY_COLUMNS + ["ATTACK", "CLASS", "SCORE"] + pc3_features()].copy()
    for iqa in IQAS:
        frame = projections[(pipeline, iqa)]
        fusion = fusion.merge(
            frame[KEY_COLUMNS + projection_features(pipeline, iqa)],
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
    expected = 40 + 22 * 6
    features = [c for c in fusion if c.startswith("pc3__") or c.startswith("proj__")]
    if len(features) != expected or len(fusion) != len(base):
        raise RuntimeError("invalid full fusion table")
    if save:
        fusion.to_csv(DIRS["normalized"] / "fusion_full_172.csv", index=False)
    return fusion


def run_lazy_split(X_train, X_test, y_train, y_test, regressor_classes=None):
    lazy = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        predictions=True,
        random_state=SEED,
        regressors=regressor_classes if regressor_classes is not None else "all",
        n_jobs=-1,
    )
    scores, predictions = lazy.fit(X_train, X_test, y_train, y_test)
    records = []
    for model in predictions.columns:
        score = metrics(y_test, predictions[model])
        records.append({"model": model, **score})
    failures = [
        {"model": name, "error": f"{type(error).__name__}: {error}"}
        for name, error in lazy.errors.items()
    ]
    return pd.DataFrame(records), predictions, failures, lazy


def model_class_map():
    return {name: cls for name, cls in REGRESSORS}


def fit_named_model(name, X_train, y_train, X_test):
    cls = model_class_map()[name]
    _, predictions, failures, lazy = run_lazy_split(
        X_train,
        X_test,
        y_train,
        np.zeros(len(X_test)),
        regressor_classes=[cls],
    )
    if failures or name not in lazy.models:
        raise RuntimeError(f"failed to fit selected model {name}: {failures}")
    return lazy.models[name], predictions[name].to_numpy()


def regressor_analysis(base, projections, selection, outer, inner_by_outer):
    inner_records = []
    outer_records = []
    prediction_records = []
    failure_records = []
    outer_winners = []
    top10_by_outer = {}
    pipeline_by_outer = {
        int(row["outer_fold"]): str(row["pipeline"])
        for _, row in selection[selection["selected"]].iterrows()
    }
    for outer_fold, (outer_train, outer_test) in enumerate(outer):
        pipeline = pipeline_by_outer[outer_fold]
        fusion = build_fusion(base, projections, pipeline)
        features = [c for c in fusion if c.startswith("pc3__") or c.startswith("proj__")]
        train_frame = fusion.iloc[outer_train].reset_index(drop=True)
        per_model = []
        eligible_counts = Counter()
        for inner_fold, (train_rel, valid_rel) in enumerate(inner_by_outer[outer_fold]):
            result, _, failures, _ = run_lazy_split(
                train_frame.iloc[train_rel][features],
                train_frame.iloc[valid_rel][features],
                train_frame.iloc[train_rel]["SCORE"],
                train_frame.iloc[valid_rel]["SCORE"],
            )
            result["outer_fold"] = outer_fold
            result["inner_fold"] = inner_fold
            result["pipeline"] = pipeline
            inner_records.extend(result.to_dict("records"))
            for model in result["model"]:
                eligible_counts[model] += 1
            for failure in failures:
                failure_records.append(
                    {"outer_fold": outer_fold, "inner_fold": inner_fold, **failure}
                )
        inner_frame = pd.DataFrame(
            [r for r in inner_records if r["outer_fold"] == outer_fold]
        )
        eligible = [model for model, count in eligible_counts.items() if count == 5 and model != "DummyRegressor"]
        aggregate = (
            inner_frame[inner_frame["model"].isin(eligible)]
            .groupby("model")
            .agg(
                median_srocc=("srocc", "median"),
                median_plcc=("plcc", "median"),
                median_rmse=("rmse", "median"),
            )
            .reset_index()
            .sort_values(
                ["median_srocc", "median_plcc", "median_rmse"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )
        aggregate["inner_rank"] = np.arange(1, len(aggregate) + 1)
        if aggregate.empty:
            raise RuntimeError(f"no eligible regressors in outer fold {outer_fold}")
        winner = str(aggregate.iloc[0]["model"])
        outer_winners.append(winner)
        top10 = aggregate.head(10)["model"].tolist()
        top10_by_outer[outer_fold] = top10
        classes = [model_class_map()[name] for name in top10]
        result, predictions, failures, _ = run_lazy_split(
            fusion.iloc[outer_train][features],
            fusion.iloc[outer_test][features],
            fusion.iloc[outer_train]["SCORE"],
            fusion.iloc[outer_test]["SCORE"],
            regressor_classes=classes,
        )
        result["outer_fold"] = outer_fold
        result["pipeline"] = pipeline
        result["inner_winner"] = result["model"] == winner
        outer_records.extend(result.to_dict("records"))
        for model in predictions:
            for idx, value in zip(outer_test, predictions[model]):
                prediction_records.append(
                    {
                        "outer_fold": outer_fold,
                        "pipeline": pipeline,
                        "model": model,
                        "SIGNAL": fusion.iloc[idx]["SIGNAL"],
                        "REF": fusion.iloc[idx]["REF"],
                        "y_true": fusion.iloc[idx]["SCORE"],
                        "y_pred": value,
                    }
                )
        for failure in failures:
            failure_records.append({"outer_fold": outer_fold, "inner_fold": -1, **failure})

    inner_df = pd.DataFrame(inner_records)
    outer_df = pd.DataFrame(outer_records)
    failures_df = pd.DataFrame(failure_records)
    pred_df = pd.DataFrame(prediction_records)
    inner_df.to_csv(DIRS["summaries"] / "regressor_screen_inner.csv", index=False)
    outer_df.to_csv(DIRS["summaries"] / "regressor_shortlist_outer.csv", index=False)
    failures_df.to_csv(DIRS["logs"] / "regressor_failures.csv", index=False)
    pred_df.to_csv(DIRS["predictions"] / "regressor_outer_predictions.csv", index=False)

    top_counts = Counter(name for names in top10_by_outer.values() for name in names)
    winner_counts = Counter(outer_winners)
    candidate_rows = []
    for model, count in winner_counts.items():
        sub = inner_df[inner_df["model"] == model]
        ranks = []
        for outer_fold in range(5):
            frame = inner_df[
                (inner_df["outer_fold"] == outer_fold)
                & (inner_df["model"] != "DummyRegressor")
            ]
            agg = frame.groupby("model")["srocc"].median().rank(ascending=False, method="min")
            if model in agg:
                ranks.append(agg[model])
        out = outer_df[outer_df["model"] == model]
        candidate_rows.append(
            {
                "model": model,
                "winner_count": count,
                "median_inner_rank": np.median(ranks) if ranks else np.inf,
                "median_outer_srocc": out["srocc"].median() if len(out) else -np.inf,
                "median_outer_plcc": out["plcc"].median() if len(out) else -np.inf,
                "median_outer_rmse": out["rmse"].median() if len(out) else np.inf,
            }
        )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["winner_count", "median_inner_rank", "median_outer_srocc", "median_outer_plcc", "median_outer_rmse"],
        ascending=[False, True, False, False, True],
    )
    winner = str(candidates.iloc[0]["model"])
    stable = sorted([name for name, count in top_counts.items() if count >= 4])
    write_json(
        DIRS["manifests"] / "regressor_selected.json",
        {
            "selected_regressor": winner,
            "outer_inner_winners": outer_winners,
            "winner_frequency": winner_counts,
            "stable_shortlist": stable,
            "top10_frequency": top_counts,
        },
    )
    return winner, pipeline_by_outer, outer_df, stable


def permute_within_groups(frame, columns, groups, rng):
    result = frame.copy()
    groups = np.asarray(groups)
    for group in pd.unique(groups):
        positions = np.flatnonzero(groups == group)
        if len(positions) > 1:
            shuffled = rng.permutation(positions)
            result.iloc[positions, result.columns.get_indexer(columns)] = frame.iloc[
                shuffled
            ][columns].to_numpy()
    return result


def importance_analysis(base, projections, outer, pipeline_by_outer, winner):
    raw = []
    for outer_fold, (train_idx, test_idx) in enumerate(outer):
        pipeline = pipeline_by_outer[outer_fold]
        fusion = build_fusion(base, projections, pipeline)
        features = [c for c in fusion if c.startswith("pc3__") or c.startswith("proj__")]
        model, pred = fit_named_model(
            winner,
            fusion.iloc[train_idx][features],
            fusion.iloc[train_idx]["SCORE"],
            fusion.iloc[test_idx][features],
        )
        y = fusion.iloc[test_idx]["SCORE"].to_numpy()
        groups = fusion.iloc[test_idx]["REF"].to_numpy()
        X_test = fusion.iloc[test_idx][features].reset_index(drop=True)
        base_score = metrics(y, pred)
        feature_groups = []
        for feature in features:
            if feature.startswith("proj__"):
                _, _, iqa_name, view_name = feature.split("__", 3)
                display_name = f"proj__{iqa_name}__{view_name}"
            else:
                display_name = feature
            feature_groups.append(("individual", display_name, [feature]))
        feature_groups += [
            ("iqa_block", iqa, projection_features(pipeline, iqa)) for iqa in IQAS
        ]
        feature_groups += [
            ("modality", "PointPCA3", pc3_features()),
            (
                "modality",
                "Projections",
                [f for f in features if f.startswith("proj__")],
            ),
        ]
        for level, name, columns in feature_groups:
            for repeat in range(50):
                rng = np.random.default_rng(SEED + outer_fold * 100000 + repeat * 1000 + len(raw) % 997)
                permuted = permute_within_groups(X_test, columns, groups, rng)
                perm_pred = model.predict(permuted)
                score = metrics(y, perm_pred)
                raw.append(
                    {
                        "outer_fold": outer_fold,
                        "pipeline": pipeline,
                        "level": level,
                        "feature": name,
                        "repeat": repeat,
                        "srocc_decrease": base_score["srocc"] - score["srocc"],
                        "plcc_decrease": base_score["plcc"] - score["plcc"],
                        "rmse_increase": score["rmse"] - base_score["rmse"],
                    }
                )
    raw_df = pd.DataFrame(raw)
    raw_df.to_csv(DIRS["predictions"] / "importance_repetitions.csv", index=False)
    summary = (
        raw_df.groupby(["level", "feature"])
        .agg(
            median_srocc_decrease=("srocc_decrease", "median"),
            iqr_srocc_decrease=("srocc_decrease", iqr),
            median_plcc_decrease=("plcc_decrease", "median"),
            median_rmse_increase=("rmse_increase", "median"),
        )
        .reset_index()
    )
    for level, filename in [
        ("individual", "importance_individual.csv"),
        ("iqa_block", "importance_iqa_blocks.csv"),
        ("modality", "importance_modalities.csv"),
    ]:
        summary[summary["level"] == level].sort_values(
            "median_srocc_decrease", ascending=False
        ).to_csv(DIRS["summaries"] / filename, index=False)
    return summary


def choose_iqa(scores: pd.DataFrame) -> str:
    ordered = scores.sort_values(
        ["srocc", "plcc", "rmse", "median_time"],
        ascending=[False, False, True, True],
    )
    return str(ordered.iloc[0]["iqa"])


def single_iqa_and_ablation(base, projections, outer, inner_by_outer, pipeline_by_outer, global_pipeline, winner):
    selection_rows = []
    ablation_predictions = []
    selected_by_outer = {}
    for outer_fold, (outer_train, outer_test) in enumerate(outer):
        pipeline = pipeline_by_outer[outer_fold]
        pc = base[KEY_COLUMNS + ["SCORE"] + pc3_features()].copy()
        candidate_rows = []
        train_base = pc.iloc[outer_train].reset_index(drop=True)
        for iqa in IQAS:
            proj = projections[(pipeline, iqa)]
            compact = pc.merge(
                proj[KEY_COLUMNS + projection_features(pipeline, iqa)],
                on=KEY_COLUMNS,
                validate="one_to_one",
            )
            compact_train = compact.iloc[outer_train].reset_index(drop=True)
            features = pc3_features() + projection_features(pipeline, iqa)
            fold_scores = []
            for train_rel, valid_rel in inner_by_outer[outer_fold]:
                _, pred = fit_named_model(
                    winner,
                    compact_train.iloc[train_rel][features],
                    compact_train.iloc[train_rel]["SCORE"],
                    compact_train.iloc[valid_rel][features],
                )
                fold_scores.append(
                    metrics(compact_train.iloc[valid_rel]["SCORE"], pred)
                )
            candidate_rows.append(
                {
                    "outer_fold": outer_fold,
                    "pipeline": pipeline,
                    "iqa": iqa,
                    "srocc": np.median([x["srocc"] for x in fold_scores]),
                    "plcc": np.median([x["plcc"] for x in fold_scores]),
                    "rmse": np.median([x["rmse"] for x in fold_scores]),
                    "median_time": proj.filter(like="time__proj__").iloc[:, 0].median(),
                }
            )
        candidates = pd.DataFrame(candidate_rows)
        chosen = choose_iqa(candidates)
        selected_by_outer[outer_fold] = chosen
        candidates["selected"] = candidates["iqa"] == chosen
        selection_rows.extend(candidates.to_dict("records"))

        proj = projections[(pipeline, chosen)]
        compact = pc.merge(
            proj[KEY_COLUMNS + projection_features(pipeline, chosen)],
            on=KEY_COLUMNS,
            validate="one_to_one",
        )
        full = build_fusion(base, projections, pipeline)
        sets = {
            "pointpca3_only": (pc, pc3_features()),
            "selected_iqa_only": (compact, projection_features(pipeline, chosen)),
            "full_fusion_172": (
                full,
                [c for c in full if c.startswith("pc3__") or c.startswith("proj__")],
            ),
            "compact_fusion_46": (
                compact,
                pc3_features() + projection_features(pipeline, chosen),
            ),
        }
        for set_name, (frame, features) in sets.items():
            _, pred = fit_named_model(
                winner,
                frame.iloc[outer_train][features],
                frame.iloc[outer_train]["SCORE"],
                frame.iloc[outer_test][features],
            )
            for idx, value in zip(outer_test, pred):
                ablation_predictions.append(
                    {
                        "outer_fold": outer_fold,
                        "pipeline": pipeline,
                        "selected_iqa": chosen,
                        "feature_set": set_name,
                        "SIGNAL": frame.iloc[idx]["SIGNAL"],
                        "REF": frame.iloc[idx]["REF"],
                        "y_true": frame.iloc[idx]["SCORE"],
                        "y_pred": value,
                    }
                )

    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(DIRS["summaries"] / "single_iqa_outer_selection.csv", index=False)
    pred_df = pd.DataFrame(ablation_predictions)
    pred_df.to_csv(DIRS["predictions"] / "feature_set_ablation_predictions.csv", index=False)
    fold_rows = []
    for (feature_set, fold), frame in pred_df.groupby(["feature_set", "outer_fold"]):
        fold_rows.append(
            {"feature_set": feature_set, "outer_fold": fold, **metrics(frame["y_true"], frame["y_pred"])}
        )
    fold_df = pd.DataFrame(fold_rows)
    summaries = []
    for feature_set, frame in fold_df.groupby("feature_set"):
        summaries.append(
            {
                "feature_set": feature_set,
                "srocc": frame["srocc"].median(),
                "srocc_iqr": iqr(frame["srocc"]),
                "plcc": frame["plcc"].median(),
                "plcc_iqr": iqr(frame["plcc"]),
                "rmse": frame["rmse"].median(),
                "rmse_iqr": iqr(frame["rmse"]),
            }
        )
    ablation_summary = pd.DataFrame(summaries)
    ablation_summary.to_csv(DIRS["summaries"] / "feature_set_ablation_summary.csv", index=False)

    global_rows = []
    pc = base[KEY_COLUMNS + ["SCORE"] + pc3_features()].copy()
    for iqa in IQAS:
        proj = projections[(global_pipeline, iqa)]
        compact = pc.merge(
            proj[KEY_COLUMNS + projection_features(global_pipeline, iqa)],
            on=KEY_COLUMNS,
            validate="one_to_one",
        )
        features = pc3_features() + projection_features(global_pipeline, iqa)
        scores = []
        for train_idx, test_idx in outer:
            _, pred = fit_named_model(
                winner,
                compact.iloc[train_idx][features],
                compact.iloc[train_idx]["SCORE"],
                compact.iloc[test_idx][features],
            )
            scores.append(metrics(compact.iloc[test_idx]["SCORE"], pred))
        global_rows.append(
            {
                "iqa": iqa,
                "srocc": np.median([x["srocc"] for x in scores]),
                "plcc": np.median([x["plcc"] for x in scores]),
                "rmse": np.median([x["rmse"] for x in scores]),
                "median_time": proj.filter(like="time__proj__").iloc[:, 0].median(),
            }
        )
    global_candidates = pd.DataFrame(global_rows)
    selected = choose_iqa(global_candidates)
    global_candidates["selected"] = global_candidates["iqa"] == selected
    global_candidates.to_csv(DIRS["summaries"] / "single_iqa_candidates.csv", index=False)
    write_json(
        DIRS["manifests"] / "single_iqa_selected.json",
        {
            "selected_iqa": selected,
            "selected_pipeline": global_pipeline,
            "selected_regressor": winner,
            "outer_selection_frequency": Counter(selected_by_outer.values()),
        },
    )
    return selected, ablation_summary


def validate_saved_source_hashes() -> None:
    manifest_path = DIRS["manifests"] / "source_hashes.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing source-hash manifest: {manifest_path}")
    mismatches = []
    for row in pd.read_csv(manifest_path).itertuples(index=False):
        path = resolve_saved_source_path(row.path)
        if not path.exists():
            mismatches.append(f"missing: {row.path}")
        elif sha256(path) != row.sha256:
            mismatches.append(f"hash changed: {row.path}")
        elif path.stat().st_size != int(row.bytes):
            mismatches.append(f"size changed: {row.path}")
    if mismatches:
        raise RuntimeError(
            "saved source hashes do not match the current inputs:\n"
            + "\n".join(mismatches)
        )


def load_contribution_inputs(pipelines: set[str]):
    base = load_pc3("APSIPA", 16).sort_values(KEY_COLUMNS).reset_index(drop=True)
    keep = [c for c in META_COLUMNS if c != "PEAK_RAM_USAGE_GiB"] + pc3_features()
    base = base[keep].copy().rename(
        columns={"TIME_TAKEN_SECONDS": "time__pc3__w16"}
    )
    projections = {}
    for pipeline in sorted(pipelines):
        for iqa in IQAS:
            frame = load_projection(pipeline, iqa).sort_values(KEY_COLUMNS).reset_index(
                drop=True
            )
            assert_same_population(base, frame, f"contribution {pipeline}/{iqa}")
            features = projection_features(pipeline, iqa)
            frame = frame[
                KEY_COLUMNS + COMPARE_META + ["TIME_TAKEN_SECONDS"] + features
            ].copy()
            frame = frame.rename(
                columns={
                    "TIME_TAKEN_SECONDS": f"time__proj__{pipeline}__{iqa.lower()}"
                }
            )
            projections[(pipeline, iqa)] = frame
    return base, projections


def load_persisted_outer_folds(base: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    manifest = pd.read_csv(DIRS["folds"] / "apsipa_nested_groupkfold.csv")
    manifest = manifest[manifest["inner_fold"] == -1].copy()
    if manifest.duplicated(["outer_fold", *KEY_COLUMNS]).any():
        raise ValueError("persisted outer-fold manifest has duplicate sample rows")
    outer = []
    for outer_fold in sorted(manifest["outer_fold"].unique()):
        fold = manifest[manifest["outer_fold"] == outer_fold]
        aligned = base[KEY_COLUMNS].merge(
            fold[KEY_COLUMNS + ["role"]],
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        if aligned["role"].isna().any() or len(aligned) != len(base):
            raise ValueError(f"incomplete persisted outer fold {outer_fold}")
        train_idx = np.flatnonzero(aligned["role"].eq("train").to_numpy())
        test_idx = np.flatnonzero(aligned["role"].eq("test").to_numpy())
        if len(train_idx) + len(test_idx) != len(base):
            raise ValueError(f"invalid roles in persisted outer fold {outer_fold}")
        if set(base.iloc[train_idx]["REF"]) & set(base.iloc[test_idx]["REF"]):
            raise ValueError(f"reference leakage in persisted outer fold {outer_fold}")
        outer.append((train_idx, test_idx))
    if len(outer) != 5:
        raise ValueError("the IQA contribution analysis requires five outer folds")
    coverage = np.zeros(len(base), dtype=int)
    for _, test_idx in outer:
        coverage[test_idx] += 1
    if not np.array_equal(coverage, np.ones(len(base), dtype=int)):
        raise ValueError("persisted outer folds do not give every sample one test role")
    return outer


def contribution_effect(reference: dict[str, float], variant: dict[str, float], analysis: str):
    if analysis == "add_one":
        return {
            "srocc": variant["srocc"] - reference["srocc"],
            "plcc": variant["plcc"] - reference["plcc"],
            "rmse": reference["rmse"] - variant["rmse"],
        }
    if analysis == "remove_one":
        return {
            "srocc": reference["srocc"] - variant["srocc"],
            "plcc": reference["plcc"] - variant["plcc"],
            "rmse": variant["rmse"] - reference["rmse"],
        }
    raise ValueError(f"unknown contribution analysis: {analysis}")


def contribution_bootstrap_indices(groups, repeats=10000) -> list[np.ndarray]:
    groups = np.asarray(groups)
    unique = pd.unique(groups)
    rng = np.random.default_rng(SEED)
    indices = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices.append(
            np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        )
    return indices


def metric_arrays(y_true, prediction_matrix) -> dict[str, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    if prediction_matrix.ndim == 1:
        prediction_matrix = prediction_matrix[:, None]

    centered_y = y_true - y_true.mean()
    centered_predictions = prediction_matrix - prediction_matrix.mean(axis=0)
    plcc = (centered_predictions * centered_y[:, None]).sum(axis=0) / np.sqrt(
        (centered_predictions**2).sum(axis=0) * (centered_y**2).sum()
    )
    ranked_y = rankdata(y_true)
    ranked_predictions = rankdata(prediction_matrix, axis=0)
    centered_ranked_y = ranked_y - ranked_y.mean()
    centered_ranked_predictions = ranked_predictions - ranked_predictions.mean(axis=0)
    srocc = (
        centered_ranked_predictions * centered_ranked_y[:, None]
    ).sum(axis=0) / np.sqrt(
        (centered_ranked_predictions**2).sum(axis=0)
        * (centered_ranked_y**2).sum()
    )
    rmse = np.sqrt(((prediction_matrix - y_true[:, None]) ** 2).mean(axis=0))
    return {"srocc": srocc, "plcc": plcc, "rmse": rmse}


def iqa_contribution_analysis(
    base, projections, outer, pipeline_by_outer, winner, reuse_predictions=False
):
    prediction_path = DIRS["predictions"] / "iqa_contribution_oof_predictions.csv"
    if reuse_predictions and prediction_path.exists():
        prediction_rows = pd.read_csv(prediction_path).to_dict("records")
        fold_iterator = []
    else:
        prediction_rows = []
        fold_iterator = enumerate(outer)
    fit_count = 0
    for outer_fold, (train_idx, test_idx) in fold_iterator:
        pipeline = pipeline_by_outer[outer_fold]
        pc = base[KEY_COLUMNS + ["REF", "SCORE"] + pc3_features()].copy()
        pc = pc.loc[:, ~pc.columns.duplicated()].copy()
        full = build_fusion(base, projections, pipeline)
        full_features = [
            column
            for column in full
            if column.startswith("pc3__") or column.startswith("proj__")
        ]
        if len(full_features) != 172:
            raise RuntimeError("full contribution reference does not have 172 features")

        _, pc_reference = fit_named_model(
            winner,
            pc.iloc[train_idx][pc3_features()],
            pc.iloc[train_idx]["SCORE"],
            pc.iloc[test_idx][pc3_features()],
        )
        _, full_reference = fit_named_model(
            winner,
            full.iloc[train_idx][full_features],
            full.iloc[train_idx]["SCORE"],
            full.iloc[test_idx][full_features],
        )
        fit_count += 2

        for iqa in IQAS:
            block = projection_features(pipeline, iqa)
            projection = projections[(pipeline, iqa)]
            compact = pc.merge(
                projection[KEY_COLUMNS + block],
                on=KEY_COLUMNS,
                validate="one_to_one",
            )
            compact_features = pc3_features() + block
            without_features = [column for column in full_features if column not in block]
            if len(compact_features) != 46 or len(without_features) != 166:
                raise RuntimeError(f"invalid contribution feature count for {iqa}")
            if not compact[KEY_COLUMNS].equals(pc[KEY_COLUMNS]):
                raise RuntimeError(f"compact merge changed sample order for {pipeline}/{iqa}")

            _, add_variant = fit_named_model(
                winner,
                compact.iloc[train_idx][compact_features],
                compact.iloc[train_idx]["SCORE"],
                compact.iloc[test_idx][compact_features],
            )
            _, remove_variant = fit_named_model(
                winner,
                full.iloc[train_idx][without_features],
                full.iloc[train_idx]["SCORE"],
                full.iloc[test_idx][without_features],
            )
            fit_count += 2

            for analysis, reference_pred, variant_pred, reference_count, variant_count in [
                ("add_one", pc_reference, add_variant, 40, 46),
                ("remove_one", full_reference, remove_variant, 172, 166),
            ]:
                for idx, reference_value, variant_value in zip(
                    test_idx, reference_pred, variant_pred
                ):
                    prediction_rows.append(
                        {
                            "analysis": analysis,
                            "iqa": iqa,
                            "outer_fold": outer_fold,
                            "pipeline": pipeline,
                            "SIGNAL": base.iloc[idx]["SIGNAL"],
                            "REF": base.iloc[idx]["REF"],
                            "y_true": base.iloc[idx]["SCORE"],
                            "y_reference": reference_value,
                            "y_variant": variant_value,
                            "reference_feature_count": reference_count,
                            "variant_feature_count": variant_count,
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    expected_rows = 2 * len(IQAS) * len(base)
    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} contribution predictions, found {len(predictions)}"
        )
    if predictions.duplicated(["analysis", "iqa", *KEY_COLUMNS]).any():
        raise RuntimeError("duplicate OOF prediction in IQA contribution analysis")
    if not np.isfinite(
        predictions[["y_true", "y_reference", "y_variant"]].to_numpy()
    ).all():
        raise RuntimeError("non-finite IQA contribution prediction")
    counts = predictions.groupby(["analysis", "iqa"]).size()
    if not (counts == len(base)).all():
        raise RuntimeError("incomplete IQA contribution OOF coverage")
    predictions.to_csv(prediction_path, index=False)

    fold_rows = []
    for (analysis, iqa, outer_fold), frame in predictions.groupby(
        ["analysis", "iqa", "outer_fold"], sort=False
    ):
        reference = metrics(frame["y_true"], frame["y_reference"])
        variant = metrics(frame["y_true"], frame["y_variant"])
        effect = contribution_effect(reference, variant, analysis)
        row = {
            "analysis": analysis,
            "iqa": iqa,
            "outer_fold": outer_fold,
            "pipeline": frame["pipeline"].iloc[0],
        }
        for metric_name in ["srocc", "plcc", "rmse"]:
            row[f"reference_{metric_name}"] = reference[metric_name]
            row[f"variant_{metric_name}"] = variant[metric_name]
            row[f"effect_{metric_name}"] = effect[metric_name]
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(
        DIRS["summaries"] / "iqa_contribution_fold_metrics.csv", index=False
    )

    bootstrap_indices = contribution_bootstrap_indices(base["REF"], repeats=10000)
    summary_rows = []
    for analysis in ["add_one", "remove_one"]:
        y_true = None
        y_reference = None
        variant_columns = []
        for iqa in IQAS:
            case = predictions[
                (predictions["analysis"] == analysis) & (predictions["iqa"] == iqa)
            ].copy()
            case = base[KEY_COLUMNS].merge(
                case, on=KEY_COLUMNS, how="left", validate="one_to_one"
            )
            case_y = case["y_true"].to_numpy()
            case_reference = case["y_reference"].to_numpy()
            if y_true is None:
                y_true = case_y
                y_reference = case_reference
            elif not np.array_equal(y_true, case_y) or not np.allclose(
                y_reference, case_reference, rtol=0, atol=1e-12
            ):
                raise RuntimeError(
                    f"unpaired reference predictions in {analysis}/{iqa}"
                )
            variant_columns.append(case["y_variant"].to_numpy())
        variants = np.column_stack(variant_columns)
        all_predictions = np.column_stack([y_reference, variants])
        pooled = metric_arrays(y_true, all_predictions)
        bootstrap_effects = {
            metric_name: np.empty((len(bootstrap_indices), len(IQAS)), dtype=float)
            for metric_name in ["srocc", "plcc", "rmse"]
        }
        for repeat, selected in enumerate(bootstrap_indices):
            sampled = metric_arrays(y_true[selected], all_predictions[selected])
            for metric_name in ["srocc", "plcc"]:
                if analysis == "add_one":
                    bootstrap_effects[metric_name][repeat] = (
                        sampled[metric_name][1:] - sampled[metric_name][0]
                    )
                else:
                    bootstrap_effects[metric_name][repeat] = (
                        sampled[metric_name][0] - sampled[metric_name][1:]
                    )
            if analysis == "add_one":
                bootstrap_effects["rmse"][repeat] = (
                    sampled["rmse"][0] - sampled["rmse"][1:]
                )
            else:
                bootstrap_effects["rmse"][repeat] = (
                    sampled["rmse"][1:] - sampled["rmse"][0]
                )

        for iqa_index, iqa in enumerate(IQAS):
            reference = {
                metric_name: float(pooled[metric_name][0])
                for metric_name in ["srocc", "plcc", "rmse"]
            }
            variant = {
                metric_name: float(pooled[metric_name][iqa_index + 1])
                for metric_name in ["srocc", "plcc", "rmse"]
            }
            effect = contribution_effect(reference, variant, analysis)
            fold_frame = fold_metrics[
                (fold_metrics["analysis"] == analysis)
                & (fold_metrics["iqa"] == iqa)
            ]
            row = {"analysis": analysis, "iqa": iqa}
            for metric_name in ["srocc", "plcc", "rmse"]:
                samples = bootstrap_effects[metric_name][:, iqa_index]
                fold_values = fold_frame[f"effect_{metric_name}"].to_numpy()
                row[f"reference_{metric_name}"] = reference[metric_name]
                row[f"variant_{metric_name}"] = variant[metric_name]
                row[f"effect_{metric_name}"] = effect[metric_name]
                row[f"effect_{metric_name}_ci_low"] = float(
                    np.nanpercentile(samples, 2.5)
                )
                row[f"effect_{metric_name}_ci_high"] = float(
                    np.nanpercentile(samples, 97.5)
                )
                row[f"fold_median_effect_{metric_name}"] = float(
                    np.nanmedian(fold_values)
                )
                row[f"fold_iqr_effect_{metric_name}"] = iqr(fold_values)
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DIRS["summaries"] / "iqa_contribution_summary.csv", index=False)
    dists = summary[summary["iqa"] == "DISTSMetric"].set_index("analysis")
    macro_values = {
        "DISTSAddSROCCEffect": dists.loc["add_one", "effect_srocc"],
        "DISTSAddSROCCLow": dists.loc["add_one", "effect_srocc_ci_low"],
        "DISTSAddSROCCHigh": dists.loc["add_one", "effect_srocc_ci_high"],
        "DISTSRemoveSROCCEffect": dists.loc["remove_one", "effect_srocc"],
        "DISTSRemoveSROCCLow": dists.loc["remove_one", "effect_srocc_ci_low"],
        "DISTSRemoveSROCCHigh": dists.loc["remove_one", "effect_srocc_ci_high"],
    }
    macro_text = "\n".join(
        f"\\newcommand{{\\{name}}}{{{value:.3f}}}"
        for name, value in macro_values.items()
    )
    (DIRS["tables"] / "iqa_contribution_macros.tex").write_text(
        macro_text + "\n"
    )

    contribution_manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "bootstrap_repeats": 10000,
        "bootstrap_cluster": "REF",
        "samples": len(base),
        "reference_groups": int(base["REF"].nunique()),
        "outer_folds": len(outer),
        "iqa_blocks": len(IQAS),
        "model_fits_for_predictions": 230,
        "model_fits_this_invocation": fit_count,
        "predictions_reused": bool(reuse_predictions and prediction_path.exists()),
        "selected_regressor": winner,
        "selected_iqa_frozen": "DISTSMetric",
        "selection_reopened": False,
        "pipeline_by_outer": pipeline_by_outer,
        "effect_orientation": "positive values favor retaining the IQA block",
        "script_sha256": sha256(SCRIPT),
        "input_artifact_hashes": {
            "source_hashes": sha256(DIRS["manifests"] / "source_hashes.csv"),
            "folds": sha256(DIRS["folds"] / "apsipa_nested_groupkfold.csv"),
            "projection_selection": sha256(
                DIRS["summaries"] / "projection_outer_selection.csv"
            ),
            "regressor_selection": sha256(
                DIRS["manifests"] / "regressor_selected.json"
            ),
            "iqa_selection": sha256(
                DIRS["manifests"] / "single_iqa_selected.json"
            ),
        },
        "prediction_sha256": sha256(prediction_path),
        "artifacts": {
            "predictions": "iqa_contribution_oof_predictions.csv",
            "fold_metrics": "iqa_contribution_fold_metrics.csv",
            "summary": "iqa_contribution_summary.csv",
            "latex_macros": "iqa_contribution_macros.tex",
        },
    }
    write_json(
        DIRS["manifests"] / "iqa_contribution_manifest.json",
        contribution_manifest,
    )
    return summary


def cluster_bootstrap_median(values, groups, repeats=10000):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    unique = pd.unique(groups)
    rng = np.random.default_rng(SEED)
    stats = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        stats.append(np.median(values[selected]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def final_pc3_features() -> list[str]:
    return [f"pc3__f{i:02d}" for i in range(40)]


def final_dists_features() -> list[str]:
    return [f"proj__crop_pad_navier__distsmetric__view{i}" for i in range(6)]


def final_feature_columns() -> list[str]:
    return final_pc3_features() + final_dists_features()


def final_complete_path(dataset: str) -> Path:
    return COMPLETE_FEATURES / dataset / "PointPCAPP-Complete.csv"


def load_final_complete_dataset(dataset: str) -> pd.DataFrame:
    source = final_complete_path(dataset)
    frame = pd.read_csv(source)
    source_features = [f"FV_PointPCAPP-Complete_{i}" for i in range(46)]
    actual_features = [column for column in frame if column.startswith("FV_")]
    if actual_features != source_features:
        raise ValueError(
            f"{dataset}: complete feature columns are not the ordered 0--45 contract"
        )
    rename = {
        **{
            f"FV_PointPCAPP-Complete_{i}": final_pc3_features()[i]
            for i in range(40)
        },
        **{
            f"FV_PointPCAPP-Complete_{40 + i}": final_dists_features()[i]
            for i in range(6)
        },
    }
    return frame.rename(columns=rename)


def audit_and_normalize_final_pointpcapp() -> pd.DataFrame:
    audit_rows = []
    normalized = []
    source_hashes = []
    apsipa_dists_difference = None
    for dataset in FINAL_DATASETS:
        source = final_complete_path(dataset)
        if not source.exists():
            raise FileNotFoundError(f"missing complete PointPCA++ file: {source}")
        raw = pd.read_csv(source)
        source_columns = [f"FV_PointPCAPP-Complete_{i}" for i in range(46)]
        errors = validate_frame(raw, META_COLUMNS + source_columns, source)
        if (pd.to_numeric(raw["TIME_TAKEN_SECONDS"], errors="coerce") <= 0).any():
            errors.append("non-positive TIME_TAKEN_SECONDS")
        expected_samples, expected_references = FINAL_EXPECTED_POPULATIONS[dataset]
        if len(raw) != expected_samples:
            errors.append(
                f"expected {expected_samples} samples, found {len(raw)}"
            )
        if raw["REF"].nunique() != expected_references:
            errors.append(
                f"expected {expected_references} references, found {raw['REF'].nunique()}"
            )

        pc3_path = POINTPCA3_FEATURES / dataset / "16" / "PointPCA3-Rust.csv"
        matlab_path = POINTPCA2_BENCHMARKS / dataset / "PointPCA2.csv"
        pc3_raw = pd.read_csv(pc3_path)
        matlab_raw = pd.read_csv(matlab_path)
        try:
            assert_same_population(raw, pc3_raw, f"{dataset}: complete versus PC3")
            assert_same_population(raw, matlab_raw, f"{dataset}: complete versus MATLAB")
        except ValueError as error:
            errors.append(str(error))

        ordered_complete = raw.sort_values(KEY_COLUMNS).reset_index(drop=True)
        ordered_pc3 = pc3_raw.sort_values(KEY_COLUMNS).reset_index(drop=True)
        pc3_source_columns = [f"FV_PointPCA3-Rust_{i}" for i in range(40)]
        if not np.array_equal(
            ordered_complete[source_columns[:40]].to_numpy(),
            ordered_pc3[pc3_source_columns].to_numpy(),
        ):
            errors.append("complete columns 0--39 do not exactly match PointPCA3")

        if dataset == "APSIPA":
            dists = pd.read_csv(
                projection_path("crop_pad_navier", "DISTSMetric")
            ).sort_values(KEY_COLUMNS).reset_index(drop=True)
            dists_columns = [f"FV_Cubemap_{i}" for i in range(6)]
            assert_same_population(
                ordered_complete, dists, "APSIPA: complete versus standalone DISTS"
            )
            apsipa_dists_difference = float(
                np.max(
                    np.abs(
                        ordered_complete[source_columns[40:]].to_numpy(dtype=float)
                        - dists[dists_columns].to_numpy(dtype=float)
                    )
                )
            )

        audit_rows.append(
            {
                "dataset": dataset,
                "path": str(source.relative_to(REPO)),
                "samples": len(raw),
                "reference_groups": int(raw["REF"].nunique()),
                "feature_columns": len(source_columns),
                "duplicate_keys": int(raw.duplicated(KEY_COLUMNS).sum()),
                "missing_class": int(raw["CLASS"].isna().sum()),
                "errors": " | ".join(errors),
            }
        )
        source_hashes.append(
            {
                "dataset": dataset,
                "path": str(source.relative_to(REPO)),
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
            }
        )
        if errors:
            continue
        frame = load_final_complete_dataset(dataset).copy()
        frame.insert(0, "dataset", dataset)
        normalized.append(
            frame[["dataset"] + META_COLUMNS + final_feature_columns()]
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(DIRS["audits"] / "final_pointpcapp_input_audit.csv", index=False)
    write_json(
        DIRS["audits"] / "final_pointpcapp_input_audit.json",
        {
            "datasets_expected": FINAL_DATASETS,
            "datasets_found": audit["dataset"].tolist(),
            "total_samples": int(audit["samples"].sum()),
            "total_reference_groups": int(audit["reference_groups"].sum()),
            "apsipa_max_absolute_dists_difference_from_standalone": apsipa_dists_difference,
            "complete_files_are_authoritative": True,
            "errors": audit.loc[audit["errors"] != "", ["dataset", "errors"]].to_dict("records"),
        },
    )
    pd.DataFrame(source_hashes).to_csv(
        DIRS["manifests"] / "final_pointpcapp_source_hashes.csv", index=False
    )
    failed = audit[audit["errors"] != ""]
    if len(failed):
        raise RuntimeError(
            "Final PointPCA++ input audit failed:\n"
            + "\n".join(
                f"{row.dataset}: {row.errors}" for row in failed.itertuples()
            )
        )

    combined = pd.concat(normalized, ignore_index=True)
    if len(combined) != sum(value[0] for value in FINAL_EXPECTED_POPULATIONS.values()):
        raise RuntimeError("normalized complete PointPCA++ population is incomplete")
    combined.to_csv(
        DIRS["normalized"] / "final_pointpcapp_46.csv", index=False
    )
    dictionary_rows = []
    for index, feature in enumerate(final_feature_columns()):
        dictionary_rows.append(
            {
                "normalized": feature,
                "source": f"FV_PointPCAPP-Complete_{index}",
                "modality": "pointpca3" if index < 40 else "projection",
                "pipeline": "" if index < 40 else "crop_pad_navier",
                "iqa": "" if index < 40 else "DISTSMetric",
                "view": "" if index < 40 else index - 40,
            }
        )
    pd.DataFrame(dictionary_rows).to_csv(
        DIRS["normalized"] / "final_pointpcapp_feature_dictionary.csv",
        index=False,
    )
    return combined


def create_final_pointpcapp_folds(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in FINAL_DATASETS:
        dataset_frame = frame[frame["dataset"] == dataset].reset_index(drop=True)
        splits = GroupKFold(n_splits=5).split(
            dataset_frame[final_feature_columns()],
            dataset_frame["SCORE"],
            dataset_frame["REF"],
        )
        for fold, (train_idx, test_idx) in enumerate(splits):
            train_refs = set(dataset_frame.iloc[train_idx]["REF"])
            test_refs = set(dataset_frame.iloc[test_idx]["REF"])
            if train_refs & test_refs:
                raise RuntimeError(f"{dataset} fold {fold}: reference leakage")
            for idx in test_idx:
                rows.append(
                    {
                        "dataset": dataset,
                        "SIGNAL": dataset_frame.iloc[idx]["SIGNAL"],
                        "REF": dataset_frame.iloc[idx]["REF"],
                        "fold": fold,
                    }
                )
    folds = pd.DataFrame(rows)
    if folds.duplicated(["dataset"] + KEY_COLUMNS).any() or len(folds) != len(frame):
        raise RuntimeError("invalid final PointPCA++ fold assignment")
    folds.to_csv(
        DIRS["folds"] / "final_pointpcapp_groupkfold.csv", index=False
    )
    return folds


def create_core3_nested_folds(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, object]]]]:
    manifest_rows = []
    splits: dict[str, list[dict[str, object]]] = {}
    for dataset in MODEL_SELECTION_DATASETS:
        dataset_frame = frame[frame["dataset"] == dataset].reset_index(drop=True)
        outer_splits = list(
            GroupKFold(n_splits=5).split(
                dataset_frame[final_feature_columns()],
                dataset_frame["SCORE"],
                dataset_frame["REF"],
            )
        )
        dataset_splits = []
        outer_coverage = np.zeros(len(dataset_frame), dtype=int)
        for outer_fold, (outer_train, outer_test) in enumerate(outer_splits):
            outer_coverage[outer_test] += 1
            train_refs = set(dataset_frame.iloc[outer_train]["REF"])
            test_refs = set(dataset_frame.iloc[outer_test]["REF"])
            if train_refs & test_refs:
                raise RuntimeError(
                    f"{dataset} outer fold {outer_fold}: reference leakage"
                )
            for role, indices in [("train", outer_train), ("test", outer_test)]:
                for index in indices:
                    row = dataset_frame.iloc[index]
                    manifest_rows.append(
                        {
                            "dataset": dataset,
                            "outer_fold": outer_fold,
                            "inner_fold": -1,
                            "role": role,
                            "SIGNAL": row["SIGNAL"],
                            "REF": row["REF"],
                        }
                    )

            outer_train_frame = dataset_frame.iloc[outer_train].reset_index(drop=True)
            inner_splits = []
            inner_coverage = np.zeros(len(outer_train_frame), dtype=int)
            for inner_fold, (inner_train_rel, inner_valid_rel) in enumerate(
                GroupKFold(n_splits=5).split(
                    outer_train_frame[final_feature_columns()],
                    outer_train_frame["SCORE"],
                    outer_train_frame["REF"],
                )
            ):
                inner_coverage[inner_valid_rel] += 1
                inner_train = outer_train[inner_train_rel]
                inner_valid = outer_train[inner_valid_rel]
                if set(dataset_frame.iloc[inner_train]["REF"]) & set(
                    dataset_frame.iloc[inner_valid]["REF"]
                ):
                    raise RuntimeError(
                        f"{dataset} outer {outer_fold} inner {inner_fold}: reference leakage"
                    )
                for role, indices in [
                    ("train", inner_train),
                    ("validation", inner_valid),
                ]:
                    for index in indices:
                        row = dataset_frame.iloc[index]
                        manifest_rows.append(
                            {
                                "dataset": dataset,
                                "outer_fold": outer_fold,
                                "inner_fold": inner_fold,
                                "role": role,
                                "SIGNAL": row["SIGNAL"],
                                "REF": row["REF"],
                            }
                        )
                inner_splits.append((inner_train, inner_valid))
            if not np.array_equal(inner_coverage, np.ones(len(outer_train_frame), dtype=int)):
                raise RuntimeError(
                    f"{dataset} outer fold {outer_fold}: incomplete inner validation coverage"
                )
            dataset_splits.append(
                {
                    "outer_train": outer_train,
                    "outer_test": outer_test,
                    "inner": inner_splits,
                }
            )
        if not np.array_equal(outer_coverage, np.ones(len(dataset_frame), dtype=int)):
            raise RuntimeError(f"{dataset}: incomplete outer test coverage")
        splits[dataset] = dataset_splits
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(DIRS["folds"] / "core3_nested_groupkfold.csv", index=False)
    return manifest, splits


def core_split_hash(
    dataset: str,
    outer_fold: int,
    inner_fold: int,
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> str:
    digest = hashlib.sha256()
    for value in [dataset, outer_fold, inner_fold, SEED, *final_feature_columns()]:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    for split_name, split in [("train", train), ("validation", valid)]:
        digest.update(split_name.encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(
                split[KEY_COLUMNS + ["SCORE"] + final_feature_columns()], index=False
            ).values.tobytes()
        )
    return digest.hexdigest()


def run_core_regressor_checkpoint(
    dataset: str,
    outer_fold: int,
    inner_fold: int,
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> pd.DataFrame:
    checkpoint_dir = DIRS["checkpoints"] / "core3_regressor"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{dataset}_outer{outer_fold}_inner{inner_fold}.csv"
    split_hash = core_split_hash(
        dataset, outer_fold, inner_fold, train, valid
    )
    if path.exists():
        saved = pd.read_csv(path)
        if (
            len(saved)
            and saved["split_hash"].nunique() == 1
            and saved["split_hash"].iloc[0] == split_hash
            and set(saved["model"]) == set(model_class_map())
        ):
            return saved

    try:
        result, _, failures, _ = run_lazy_split(
            train[final_feature_columns()],
            valid[final_feature_columns()],
            train["SCORE"],
            valid["SCORE"],
        )
        success = {
            row.model: {
                "status": "complete",
                "srocc": row.srocc,
                "plcc": row.plcc,
                "rmse": row.rmse,
                "error": "",
            }
            for row in result.itertuples()
        }
        failed = {row["model"]: row["error"] for row in failures}
    except Exception as error:
        success = {}
        failed = {
            model: f"split failure: {type(error).__name__}: {error}"
            for model in model_class_map()
        }
    rows = []
    for model in model_class_map():
        values = success.get(model)
        if values is None:
            values = {
                "status": "failed",
                "srocc": np.nan,
                "plcc": np.nan,
                "rmse": np.nan,
                "error": failed.get(model, "model returned no result"),
            }
        rows.append(
            {
                "dataset": dataset,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "model": model,
                "split_hash": split_hash,
                **values,
            }
        )
    saved = pd.DataFrame(rows)
    saved.to_csv(path, index=False)
    return saved


def rank_core_candidates(
    inner_success: pd.DataFrame,
    candidate_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_rows = []
    for (outer_fold, dataset), group in inner_success.groupby(
        ["outer_fold", "dataset"], sort=False
    ):
        aggregate = (
            group.groupby(candidate_column)
            .agg(
                median_srocc=("srocc", "median"),
                median_plcc=("plcc", "median"),
                median_rmse=("rmse", "median"),
            )
            .reset_index()
            .sort_values(
                ["median_srocc", "median_plcc", "median_rmse", candidate_column],
                ascending=[False, False, True, True],
            )
            .reset_index(drop=True)
        )
        aggregate["dataset_rank"] = np.arange(1, len(aggregate) + 1)
        aggregate["outer_fold"] = int(outer_fold)
        aggregate["dataset"] = dataset
        dataset_rows.extend(aggregate.to_dict("records"))
    dataset_ranks = pd.DataFrame(dataset_rows)

    aggregate_rows = []
    for (outer_fold, candidate), group in dataset_ranks.groupby(
        ["outer_fold", candidate_column], sort=False
    ):
        if set(group["dataset"]) != set(MODEL_SELECTION_DATASETS):
            continue
        ls = group[group["dataset"] == "LS-PCQA"].iloc[0]
        aggregate_rows.append(
            {
                "outer_fold": int(outer_fold),
                candidate_column: candidate,
                "aggregate_rank": float(group["dataset_rank"].mean()),
                "ls_srocc": float(ls["median_srocc"]),
                "mean_plcc": float(group["median_plcc"].mean()),
                "mean_rmse": float(group["median_rmse"].mean()),
                **{
                    f"{dataset}_rank": int(
                        group[group["dataset"] == dataset]["dataset_rank"].iloc[0]
                    )
                    for dataset in MODEL_SELECTION_DATASETS
                },
            }
        )
    aggregate_ranks = pd.DataFrame(aggregate_rows)
    if aggregate_ranks.empty:
        raise RuntimeError(f"no complete {candidate_column} ranks across core datasets")
    aggregate_ranks = aggregate_ranks.sort_values(
        [
            "outer_fold",
            "aggregate_rank",
            "ls_srocc",
            "mean_plcc",
            "mean_rmse",
            candidate_column,
        ],
        ascending=[True, True, False, False, True, True],
    ).reset_index(drop=True)
    aggregate_ranks["selected"] = False
    winner_indices = aggregate_ranks.groupby("outer_fold", sort=False).head(1).index
    aggregate_ranks.loc[winner_indices, "selected"] = True
    return dataset_ranks, aggregate_ranks


def select_core_winner(
    ranks: pd.DataFrame,
    candidate_column: str,
) -> tuple[str, pd.DataFrame]:
    winners = ranks[ranks["selected"]][candidate_column].value_counts()
    rows = []
    for candidate, group in ranks.groupby(candidate_column, sort=False):
        rows.append(
            {
                candidate_column: candidate,
                "selection_count": int(winners.get(candidate, 0)),
                "median_aggregate_rank": float(group["aggregate_rank"].median()),
                "median_ls_srocc": float(group["ls_srocc"].median()),
                "median_mean_plcc": float(group["mean_plcc"].median()),
                "median_mean_rmse": float(group["mean_rmse"].median()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        [
            "selection_count",
            "median_aggregate_rank",
            "median_ls_srocc",
            "median_mean_plcc",
            "median_mean_rmse",
            candidate_column,
        ],
        ascending=[False, True, False, False, True, True],
    ).reset_index(drop=True)
    return str(summary.iloc[0][candidate_column]), summary


def core3_regressor_selection(
    frame: pd.DataFrame,
    splits: dict[str, list[dict[str, object]]],
) -> tuple[str, list[str], dict[str, object]]:
    checkpoint_frames = []
    total_splits = len(MODEL_SELECTION_DATASETS) * 5 * 5
    completed = 0
    for dataset in MODEL_SELECTION_DATASETS:
        dataset_frame = frame[frame["dataset"] == dataset].reset_index(drop=True)
        for outer_fold, fold in enumerate(splits[dataset]):
            for inner_fold, (train_idx, valid_idx) in enumerate(fold["inner"]):
                completed += 1
                print(
                    f"      regressor screen {completed}/{total_splits}: "
                    f"{dataset} outer {outer_fold + 1} inner {inner_fold + 1}",
                    flush=True,
                )
                checkpoint_frames.append(
                    run_core_regressor_checkpoint(
                        dataset,
                        outer_fold,
                        inner_fold,
                        dataset_frame.iloc[train_idx],
                        dataset_frame.iloc[valid_idx],
                    )
                )
    checkpoint = pd.concat(checkpoint_frames, ignore_index=True)
    success = checkpoint[
        checkpoint["status"].eq("complete")
        & np.isfinite(checkpoint[["srocc", "plcc", "rmse"]]).all(axis=1)
    ].copy()
    expected_splits = total_splits
    counts = success.groupby("model").size()
    eligible = sorted(
        model
        for model, count in counts.items()
        if count == expected_splits and model != "DummyRegressor"
    )
    if not eligible:
        raise RuntimeError("no regressor completed all 75 core-selection inner folds")
    inner_success = success[success["model"].isin(eligible)].copy()
    inner_success.to_csv(
        DIRS["summaries"] / "core3_regressor_inner.csv", index=False
    )
    checkpoint[~checkpoint["status"].eq("complete")].to_csv(
        DIRS["logs"] / "core3_regressor_failures.csv", index=False
    )
    dataset_ranks, aggregate_ranks = rank_core_candidates(
        inner_success, "model"
    )
    dataset_ranks.to_csv(
        DIRS["summaries"] / "core3_regressor_dataset_ranks.csv", index=False
    )
    aggregate_ranks.to_csv(
        DIRS["summaries"] / "core3_regressor_ranks.csv", index=False
    )
    winner, selection_summary = select_core_winner(aggregate_ranks, "model")
    selection_summary.to_csv(
        DIRS["summaries"] / "core3_regressor_selection_summary.csv", index=False
    )
    stable = selection_summary.head(min(6, len(selection_summary)))["model"].tolist()
    if winner not in stable:
        stable[-1] = winner

    prediction_rows = []
    fold_rows = []
    classes = [model_class_map()[name] for name in stable]
    for dataset in MODEL_SELECTION_DATASETS:
        dataset_frame = frame[frame["dataset"] == dataset].reset_index(drop=True)
        for outer_fold, fold in enumerate(splits[dataset]):
            train_idx = fold["outer_train"]
            test_idx = fold["outer_test"]
            result, predictions, failures, _ = run_lazy_split(
                dataset_frame.iloc[train_idx][final_feature_columns()],
                dataset_frame.iloc[test_idx][final_feature_columns()],
                dataset_frame.iloc[train_idx]["SCORE"],
                dataset_frame.iloc[test_idx]["SCORE"],
                regressor_classes=classes,
            )
            if failures or set(predictions.columns) != set(stable):
                raise RuntimeError(
                    f"core regressor outer evaluation failed for {dataset} fold "
                    f"{outer_fold}: {failures}"
                )
            for row in result.itertuples():
                fold_rows.append(
                    {
                        "dataset": dataset,
                        "outer_fold": outer_fold,
                        "model": row.model,
                        "srocc": row.srocc,
                        "plcc": row.plcc,
                        "rmse": row.rmse,
                        "selected": row.model == winner,
                    }
                )
            test = dataset_frame.iloc[test_idx]
            for model in stable:
                for row, prediction in zip(test.itertuples(), predictions[model]):
                    prediction_rows.append(
                        {
                            "dataset": dataset,
                            "outer_fold": outer_fold,
                            "model": model,
                            "SIGNAL": row.SIGNAL,
                            "REF": row.REF,
                            "y_true": row.SCORE,
                            "y_pred": float(prediction),
                        }
                    )
    outer_predictions = pd.DataFrame(prediction_rows)
    outer_fold_metrics = pd.DataFrame(fold_rows)
    outer_predictions.to_csv(
        DIRS["predictions"] / "core3_regressor_outer_predictions.csv", index=False
    )
    outer_fold_metrics.to_csv(
        DIRS["summaries"] / "core3_regressor_outer_metrics.csv", index=False
    )
    if len(outer_predictions) != sum(
        len(frame[frame["dataset"] == dataset]) for dataset in MODEL_SELECTION_DATASETS
    ) * len(stable):
        raise RuntimeError("incomplete core regressor outer predictions")

    details = {
        "eligible_models": eligible,
        "ineligible_models": sorted(set(model_class_map()) - set(eligible)),
        "stable_shortlist": stable,
        "outer_inner_winners": aggregate_ranks[aggregate_ranks["selected"]][
            ["outer_fold", "model"]
        ].to_dict("records"),
        "selection_summary": selection_summary.to_dict("records"),
    }
    return winner, stable, details


def run_core_feature_checkpoint(
    selected_regressor: str,
    dataset: str,
    outer_fold: int,
    inner_fold: int,
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> pd.DataFrame:
    checkpoint_dir = DIRS["checkpoints"] / "core3_feature_sets"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{dataset}_outer{outer_fold}_inner{inner_fold}.csv"
    split_hash = core_split_hash(
        dataset, outer_fold, inner_fold, train, valid
    )
    expected_hash = hashlib.sha256(
        f"{split_hash}\0{selected_regressor}".encode("utf-8")
    ).hexdigest()
    if path.exists():
        saved = pd.read_csv(path)
        if (
            len(saved) == len(CORE_FEATURE_SETS)
            and saved["selection_hash"].nunique() == 1
            and saved["selection_hash"].iloc[0] == expected_hash
            and set(saved["feature_set"]) == set(CORE_FEATURE_SETS)
        ):
            return saved
    rows = []
    for feature_set, features in CORE_FEATURE_SETS.items():
        try:
            _, predictions = fit_named_model(
                selected_regressor,
                train[features],
                train["SCORE"],
                valid[features],
            )
            score = metrics(valid["SCORE"], predictions)
            rows.append(
                {
                    "dataset": dataset,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "feature_set": feature_set,
                    "selection_hash": expected_hash,
                    "status": "complete",
                    "error": "",
                    **score,
                }
            )
        except Exception as error:
            rows.append(
                {
                    "dataset": dataset,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "feature_set": feature_set,
                    "selection_hash": expected_hash,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "srocc": np.nan,
                    "plcc": np.nan,
                    "rmse": np.nan,
                }
            )
    saved = pd.DataFrame(rows)
    saved.to_csv(path, index=False)
    return saved


def core3_feature_set_selection(
    frame: pd.DataFrame,
    splits: dict[str, list[dict[str, object]]],
    selected_regressor: str,
) -> tuple[str, dict[str, object]]:
    checkpoint_frames = []
    for dataset in MODEL_SELECTION_DATASETS:
        dataset_frame = frame[frame["dataset"] == dataset].reset_index(drop=True)
        for outer_fold, fold in enumerate(splits[dataset]):
            for inner_fold, (train_idx, valid_idx) in enumerate(fold["inner"]):
                checkpoint_frames.append(
                    run_core_feature_checkpoint(
                        selected_regressor,
                        dataset,
                        outer_fold,
                        inner_fold,
                        dataset_frame.iloc[train_idx],
                        dataset_frame.iloc[valid_idx],
                    )
                )
    checkpoint = pd.concat(checkpoint_frames, ignore_index=True)
    failures = checkpoint[~checkpoint["status"].eq("complete")]
    failures.to_csv(DIRS["logs"] / "core3_feature_set_failures.csv", index=False)
    if len(failures):
        raise RuntimeError("one or more core feature-set inner fits failed")
    inner = checkpoint.drop(columns=["selection_hash", "status", "error"])
    inner.to_csv(DIRS["summaries"] / "core3_feature_set_inner.csv", index=False)
    dataset_ranks, aggregate_ranks = rank_core_candidates(inner, "feature_set")
    dataset_ranks.to_csv(
        DIRS["summaries"] / "core3_feature_set_dataset_ranks.csv", index=False
    )
    aggregate_ranks.to_csv(
        DIRS["summaries"] / "core3_feature_set_ranks.csv", index=False
    )
    winner, selection_summary = select_core_winner(aggregate_ranks, "feature_set")
    selection_summary.to_csv(
        DIRS["summaries"] / "core3_feature_set_selection_summary.csv", index=False
    )

    prediction_rows = []
    fold_rows = []
    for dataset in MODEL_SELECTION_DATASETS:
        dataset_frame = frame[frame["dataset"] == dataset].reset_index(drop=True)
        for outer_fold, fold in enumerate(splits[dataset]):
            train_idx = fold["outer_train"]
            test_idx = fold["outer_test"]
            test = dataset_frame.iloc[test_idx]
            for feature_set, features in CORE_FEATURE_SETS.items():
                _, predictions = fit_named_model(
                    selected_regressor,
                    dataset_frame.iloc[train_idx][features],
                    dataset_frame.iloc[train_idx]["SCORE"],
                    test[features],
                )
                score = metrics(test["SCORE"], predictions)
                fold_rows.append(
                    {
                        "dataset": dataset,
                        "outer_fold": outer_fold,
                        "feature_set": feature_set,
                        "selected": feature_set == winner,
                        **score,
                    }
                )
                for row, prediction in zip(test.itertuples(), predictions):
                    prediction_rows.append(
                        {
                            "dataset": dataset,
                            "outer_fold": outer_fold,
                            "feature_set": feature_set,
                            "SIGNAL": row.SIGNAL,
                            "REF": row.REF,
                            "y_true": row.SCORE,
                            "y_pred": float(prediction),
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    predictions.to_csv(
        DIRS["predictions"] / "core3_feature_set_oof_predictions.csv", index=False
    )
    fold_metrics.to_csv(
        DIRS["summaries"] / "core3_feature_set_fold_metrics.csv", index=False
    )
    summary = (
        fold_metrics.groupby(["dataset", "feature_set"])
        .agg(
            srocc=("srocc", "median"),
            srocc_iqr=("srocc", iqr),
            plcc=("plcc", "median"),
            plcc_iqr=("plcc", iqr),
            rmse=("rmse", "median"),
            rmse_iqr=("rmse", iqr),
        )
        .reset_index()
    )
    summary["selected"] = summary["feature_set"].eq(winner)
    summary.to_csv(
        DIRS["summaries"] / "core3_feature_set_summary.csv", index=False
    )
    expected_predictions = sum(
        len(frame[frame["dataset"] == dataset]) for dataset in MODEL_SELECTION_DATASETS
    ) * len(CORE_FEATURE_SETS)
    if len(predictions) != expected_predictions or predictions.duplicated(
        ["dataset", "feature_set", *KEY_COLUMNS]
    ).any():
        raise RuntimeError("incomplete or duplicate core feature-set OOF predictions")
    return winner, {
        "outer_inner_winners": aggregate_ranks[aggregate_ranks["selected"]][
            ["outer_fold", "feature_set"]
        ].to_dict("records"),
        "selection_summary": selection_summary.to_dict("records"),
    }


def core3_selection_manifest(require_fusion: bool = True) -> dict[str, object]:
    path = DIRS["manifests"] / "core3_selection.json"
    if not path.exists():
        raise FileNotFoundError(
            "missing three-dataset selection manifest; run --core-reselection-only"
        )
    values = json.loads(path.read_text())
    if require_fusion and not values.get("fusion_gate_passed", False):
        raise RuntimeError("three-dataset selection did not pass the fusion gate")
    return values


def write_core3_tables() -> None:
    regressor = pd.read_csv(
        DIRS["summaries"] / "core3_regressor_selection_summary.csv"
    ).head(6)
    selected = core3_selection_manifest()["selected_regressor"]
    regressor_rows = []
    for row in regressor.itertuples():
        label = str(row.model).replace("RandomForestRegressor", "Random Forest")
        if row.model == selected:
            label += "$^{*}$"
        regressor_rows.append(
            f"{label} & {int(row.selection_count)} & "
            f"{row.median_aggregate_rank:.2f} & {row.median_ls_srocc:.3f} \\\\"
        )
    (DIRS["tables"] / "core3_regressor_rows.tex").write_text(
        "\n".join(regressor_rows) + "\n\\bottomrule\n"
    )

    feature_summary = pd.read_csv(
        DIRS["summaries"] / "core3_feature_set_summary.csv"
    )
    feature_labels = {
        "pointpca3_only": "PointPCA$^3$ only",
        "dists_only": "DISTS only",
        "fusion_46": "Fusion (46)$^{*}$",
    }
    dataset_labels = {"APSIPA": "APSIPA", "WPC": "WPC", "LS-PCQA": "LS-PCQA"}
    rows = []
    for dataset in MODEL_SELECTION_DATASETS:
        for feature_set in ["pointpca3_only", "dists_only", "fusion_46"]:
            row = feature_summary[
                feature_summary["dataset"].eq(dataset)
                & feature_summary["feature_set"].eq(feature_set)
            ].iloc[0]
            rows.append(
                f"{dataset_labels[dataset]} & {feature_labels[feature_set]} & "
                f"{row.srocc:.3f} [{row.srocc_iqr:.3f}] & "
                f"{row.plcc:.3f} [{row.plcc_iqr:.3f}] & "
                f"{row.rmse:.3f} [{row.rmse_iqr:.3f}] \\\\"
            )
    (DIRS["tables"] / "core3_feature_set_rows.tex").write_text(
        "\n".join(rows) + "\n\\bottomrule\n"
    )
    provenance_path = DIRS["manifests"] / "artifact_provenance.csv"
    provenance = (
        pd.read_csv(provenance_path) if provenance_path.exists() else pd.DataFrame()
    )
    run_id = str(core3_selection_manifest()["run_id"])
    generated = [
        (
            DIRS["tables"] / "core3_regressor_rows.tex",
            "summaries/core3_regressor_selection_summary.csv",
            "tab_regressor_comparison",
        ),
        (
            DIRS["tables"] / "core3_feature_set_rows.tex",
            "summaries/core3_feature_set_summary.csv",
            "tab_feature_set_ablation",
        ),
    ]
    rows = [
        {
            "artifact": str(path.relative_to(REPO)),
            "run_id": run_id,
            "source_artifacts": sources,
            "manuscript_label": label,
            "script": str(SCRIPT.relative_to(REPO)),
            "script_sha256": sha256(SCRIPT),
            "render_mode": "core-reselection-only",
            "seaborn_version": sns.__version__,
            "palette": "deep",
        }
        for path, sources, label in generated
    ]
    provenance = pd.concat([provenance, pd.DataFrame(rows)], ignore_index=True)
    provenance.drop_duplicates("artifact", keep="last").to_csv(
        provenance_path, index=False
    )


def render_core3_selection_figures() -> None:
    style = configure_figure_style()
    manifest = core3_selection_manifest()
    selected = str(manifest["selected_regressor"])
    order = list(manifest["stable_shortlist"])
    names = {
        "ExtraTreesRegressor": "Extra Trees", "RandomForestRegressor": "Random Forest",
        "GradientBoostingRegressor": "Gradient Boosting",
        "HistGradientBoostingRegressor": "Hist. Gradient Boosting",
        "PoissonRegressor": "Poisson", "BaggingRegressor": "Bagging",
    }
    inner = pd.read_csv(DIRS["summaries"] / "core3_regressor_inner.csv")
    scores = inner[inner["status"].eq("complete")].groupby(
        ["dataset", "outer_fold", "model"]
    )["srocc"].median()
    ranks = pd.read_csv(DIRS["summaries"] / "core3_regressor_dataset_ranks.csv")
    selection = pd.read_csv(
        DIRS["summaries"] / "core3_regressor_selection_summary.csv"
    ).set_index("model")
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    for ax, dataset in zip(axes[:3], MODEL_SELECTION_DATASETS):
        for y, model in enumerate(order):
            values = scores.loc[dataset].xs(model, level="model").sort_index().to_numpy()
            assert len(values) == 5 and np.isfinite(values).all()
            saved = ranks[ranks["dataset"].eq(dataset) & ranks["model"].eq(model)]
            assert np.allclose(values, saved.sort_values("outer_fold")["median_srocc"])
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            color = style["selected"] if model == selected else style["primary"]
            ax.scatter(values, np.full(5, y), s=19, color=color, alpha=0.35)
            ax.plot([q1, q3], [y, y], color=color, linewidth=2.5)
            ax.scatter([median], [y], color=color, marker="D", s=32, zorder=4)
        ax.set_title(dataset)
        ax.set_xlabel("Inner-validation SROCC\nHigher is better")
        ax.margins(x=0.12)
    ax = axes[3]
    for y, model in enumerate(order):
        row = selection.loc[model]
        value = float(row["median_aggregate_rank"])
        color = style["selected"] if model == selected else style["primary"]
        ax.scatter([value], [y], color=color, marker="D", s=32)
        ax.annotate(f'{value:.2f}  ({int(row["selection_count"])}/5)',
                    (value, y), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8)
    ax.set_xlim(0, selection.loc[order, "median_aggregate_rank"].max() + 6)
    ax.set_title("Selection evidence")
    ax.set_xlabel("Median aggregate rank\nLower is better; wins in parentheses")
    axes[0].set_yticks(range(len(order)), [names[m] for m in order])
    axes[0].invert_yaxis()
    for label, model in zip(axes[0].get_yticklabels(), order):
        if model == selected:
            label.set_color(style["selected"])
            label.set_fontweight("bold")
    for ax in axes:
        sns.despine(ax=ax)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "regressor_comparison.pdf")

    feature = pd.read_csv(DIRS["summaries"] / "core3_feature_set_fold_metrics.csv")
    designs = ["pointpca3_only", "dists_only", "fusion_46"]
    labels = ["PointPCA$^3$", "DISTS", "Fusion (46)"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8))
    for col, dataset in enumerate(MODEL_SELECTION_DATASETS):
        data = feature[feature["dataset"].eq(dataset)]
        for metric, color, marker, offset in [
            ("srocc", style["primary"], "o", -0.10),
            ("plcc", style["secondary"], "D", 0.10),
            ("rmse", style["primary"], "o", 0),
        ]:
            ax = axes[1 if metric == "rmse" else 0, col]
            for x, design in enumerate(designs):
                values = data[data["feature_set"].eq(design)][metric].to_numpy()
                assert len(values) == 5 and np.isfinite(values).all()
                q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
                ax.errorbar(x + offset, median,
                            yerr=[[median - q1], [q3 - median]],
                            fmt=marker, color=color, capsize=3, markersize=5,
                            label=metric.upper() if x == 0 else None)
        axes[0, col].set_title(dataset)
        axes[0, col].set_ylabel("Correlation (higher is better)" if col == 0 else "")
        axes[1, col].set_ylabel("RMSE (lower is better)" if col == 0 else "")
        axes[0, col].legend(frameon=False, fontsize=8)
        for ax in axes[:, col]:
            ax.set_xticks(range(3), labels)
            ax.set_xlim(-0.4, 2.4)
            ax.get_xticklabels()[-1].set_fontweight("bold")
            ax.grid(axis="x", visible=False)
            sns.despine(ax=ax)
    # Correlations share a scale; RMSE remains dataset-specific.
    limits = [ax.get_ylim() for ax in axes[0]]
    for ax in axes[0]:
        ax.set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
    fig.tight_layout()
    save_figure(fig, "feature_set_ablation.pdf")

def run_core_reselection_only() -> None:
    start = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime(start))
    protected_paths = [
        DIRS["manifests"] / "projection_selected_pipeline.json",
        DIRS["manifests"] / "single_iqa_selected.json",
        DIRS["summaries"] / "projection_pipeline_ranks.csv",
        DIRS["summaries"] / "single_iqa_outer_selection.csv",
        DIRS["summaries"] / "single_iqa_candidates.csv",
    ]
    protected_hashes = {str(path): sha256(path) for path in protected_paths}
    print("[1/6] Auditing complete inputs and constructing core nested folds", flush=True)
    normalized = audit_and_normalize_final_pointpcapp()
    _, splits = create_core3_nested_folds(normalized)
    print("[2/6] Screening regressors across APSIPA, WPC, and LS-PCQA", flush=True)
    selected_regressor, stable, regressor_details = core3_regressor_selection(
        normalized, splits
    )
    print(f"      selected regressor: {selected_regressor}", flush=True)
    print("[3/6] Comparing the 40-, 6-, and 46-feature representations", flush=True)
    selected_feature_set, feature_details = core3_feature_set_selection(
        normalized, splits, selected_regressor
    )
    fusion_gate = selected_feature_set == "fusion_46"
    write_json(
        DIRS["manifests"] / "core3_selection.json",
        {
            "run_id": run_id,
            "started_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)
            ),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_selection_datasets": MODEL_SELECTION_DATASETS,
            "evaluation_only_datasets": EVALUATION_ONLY_DATASETS,
            "selected_projection": "crop_pad_navier",
            "selected_iqa": "DISTSMetric",
            "selected_regressor": selected_regressor,
            "stable_shortlist": stable,
            "selected_feature_set": selected_feature_set,
            "fusion_gate_passed": fusion_gate,
            "dataset_weighting": "equal dataset ranks",
            "tie_break": "LS-PCQA SROCC, aggregate PLCC, aggregate RMSE",
            "seed": SEED,
            "regressor_details": regressor_details,
            "feature_set_details": feature_details,
            "packages": {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "lazypredict": lazypredict.__version__,
            },
            "script_sha256": sha256(SCRIPT),
        },
    )
    print(f"      selected feature set: {selected_feature_set}", flush=True)
    print("[4/6] Verifying APSIPA-only selection provenance", flush=True)
    for path in protected_paths:
        if sha256(path) != protected_hashes[str(path)]:
            raise RuntimeError(f"core reselection changed protected artifact: {path}")
    if not fusion_gate:
        print(
            "[5/6] Fusion gate failed; diagnostics saved and downstream replacement stopped",
            flush=True,
        )
        raise RuntimeError(
            f"fusion gate failed: selected feature set is {selected_feature_set}"
        )
    print("[5/6] Writing gated tables and selection figures", flush=True)
    write_core3_tables()
    render_core3_selection_figures()
    print("[6/6] Core reselection complete; final evaluation is now authorized", flush=True)
    print(
        f"Three-dataset reselection completed in {(time.time() - start) / 60:.1f} minutes",
        flush=True,
    )


def clustered_metric_intervals(
    frame: pd.DataFrame, repeats: int = 10_000
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    groups = frame["REF"].to_numpy()
    refs = pd.unique(groups)
    group_positions = [np.flatnonzero(groups == ref) for ref in refs]
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    samples = {name: [] for name in ["srocc", "plcc", "rmse"]}
    invalid = {name: 0 for name in samples}
    for _ in range(repeats):
        sampled_groups = rng.integers(0, len(refs), size=len(refs))
        selected = np.concatenate(
            [group_positions[index] for index in sampled_groups]
        )
        result = metrics(y_true[selected], y_pred[selected])
        for name, value in result.items():
            if np.isfinite(value):
                samples[name].append(value)
            else:
                invalid[name] += 1
    intervals = {}
    for name, values in samples.items():
        if not values:
            raise RuntimeError(f"all clustered bootstrap replicates invalid for {name}")
        intervals[name] = (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )
    return intervals, invalid


def evaluate_final_pointpcapp(
    frame: pd.DataFrame, folds: pd.DataFrame, selected_regressor: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    prediction_rows = []
    fold_rows = []
    resolved_pipeline = None
    for dataset in FINAL_DATASETS:
        print(f"    fitting frozen PointPCA++ model on {dataset}", flush=True)
        dataset_frame = frame[frame["dataset"] == dataset].reset_index(drop=True)
        assignment = folds[folds["dataset"] == dataset]
        fold_by_key = {
            (row.SIGNAL, row.REF): int(row.fold)
            for row in assignment.itertuples()
        }
        for fold in range(5):
            test_mask = np.array(
                [fold_by_key[(row.SIGNAL, row.REF)] == fold for row in dataset_frame.itertuples()]
            )
            train_mask = ~test_mask
            train_refs = set(dataset_frame.loc[train_mask, "REF"])
            test_refs = set(dataset_frame.loc[test_mask, "REF"])
            if train_refs & test_refs:
                raise RuntimeError(f"{dataset} fold {fold}: reference leakage")
            model, predictions = fit_named_model(
                selected_regressor,
                dataset_frame.loc[train_mask, final_feature_columns()],
                dataset_frame.loc[train_mask, "SCORE"],
                dataset_frame.loc[test_mask, final_feature_columns()],
            )
            if resolved_pipeline is None:
                regressor = model.named_steps["regressor"]
                imputer = model.named_steps["preprocessor"].named_transformers_[
                    "numeric"
                ].named_steps["imputer"]
                scaler = model.named_steps["preprocessor"].named_transformers_[
                    "numeric"
                ].named_steps["scaler"]
                resolved_pipeline = {
                    "pipeline": repr(model),
                    "imputer": repr(imputer),
                    "imputer_strategy": imputer.strategy,
                    "scaler": repr(scaler),
                    "regressor_parameters": regressor.get_params(deep=True),
                }
            test = dataset_frame.loc[test_mask]
            score = metrics(test["SCORE"], predictions)
            fold_rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "test_samples": len(test),
                    "test_references": test["REF"].nunique(),
                    **score,
                }
            )
            for row, prediction in zip(test.itertuples(), predictions):
                prediction_rows.append(
                    {
                        "dataset": dataset,
                        "SIGNAL": row.SIGNAL,
                        "REF": row.REF,
                        "fold": fold,
                        "y_true": row.SCORE,
                        "y_pred": float(prediction),
                        "evaluation_role": (
                            "model_selection"
                            if dataset in MODEL_SELECTION_DATASETS
                            else "evaluation_only"
                        ),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    if predictions.duplicated(["dataset"] + KEY_COLUMNS).any():
        raise RuntimeError("duplicate final PointPCA++ OOF prediction")
    if len(predictions) != len(frame):
        raise RuntimeError("not every final PointPCA++ sample has an OOF prediction")
    prediction_path = (
        DIRS["predictions"] / "final_pointpcapp_oof_predictions.csv"
    )
    predictions.to_csv(prediction_path, index=False)

    # Treat the persisted prediction table as the reporting interface.  In
    # particular, rank correlations can change at the last decimal place when
    # nearly equal floating-point predictions become tied after CSV
    # serialization.  Re-reading here guarantees that every reported metric is
    # exactly reproducible from the distributed OOF artifact.
    predictions = pd.read_csv(prediction_path)
    fold_rows = []
    for (dataset, fold), test in predictions.groupby(
        ["dataset", "fold"], sort=False
    ):
        fold_rows.append(
            {
                "dataset": dataset,
                "fold": int(fold),
                "test_samples": len(test),
                "test_references": test["REF"].nunique(),
                **metrics(test["y_true"], test["y_pred"]),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(
        DIRS["summaries"] / "final_pointpcapp_fold_metrics.csv", index=False
    )

    summary_rows = []
    invalid_bootstrap = {}
    for dataset in FINAL_DATASETS:
        pred = predictions[predictions["dataset"] == dataset]
        folds_for_dataset = fold_metrics[fold_metrics["dataset"] == dataset]
        pooled = metrics(pred["y_true"], pred["y_pred"])
        intervals, invalid = clustered_metric_intervals(pred)
        invalid_bootstrap[dataset] = invalid
        row = {
            "dataset": dataset,
            "evaluation_role": pred["evaluation_role"].iloc[0],
            "samples": len(pred),
            "reference_groups": pred["REF"].nunique(),
            "folds": pred["fold"].nunique(),
        }
        for name in ["srocc", "plcc", "rmse"]:
            row[name] = pooled[name]
            row[f"{name}_ci_low"] = intervals[name][0]
            row[f"{name}_ci_high"] = intervals[name][1]
            row[f"{name}_fold_median"] = folds_for_dataset[name].median()
            row[f"{name}_fold_iqr"] = iqr(folds_for_dataset[name])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        DIRS["summaries"] / "final_pointpcapp_dataset_summary.csv", index=False
    )
    return predictions, fold_metrics, summary, {
        "resolved_pipeline": resolved_pipeline,
        "invalid_bootstrap_replicates": invalid_bootstrap,
    }


def clustered_transfer_intervals(
    frame: pd.DataFrame, pair_seed: int, repeats: int = 10_000
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """Bootstrap transfer correlations by resampling target reference contents."""
    groups = frame["REF"].to_numpy()
    refs = pd.unique(groups)
    group_positions = [np.flatnonzero(groups == ref) for ref in refs]
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    rng = np.random.default_rng(pair_seed)
    samples = {"srocc": [], "plcc": []}
    invalid = {"srocc": 0, "plcc": 0}
    for _ in range(repeats):
        sampled_groups = rng.integers(0, len(refs), size=len(refs))
        selected = np.concatenate(
            [group_positions[index] for index in sampled_groups]
        )
        values = {
            "srocc": float(spearmanr(y_true[selected], y_pred[selected]).statistic),
            "plcc": float(pearsonr(y_true[selected], y_pred[selected]).statistic),
        }
        for metric_name, value in values.items():
            if np.isfinite(value):
                samples[metric_name].append(value)
            else:
                invalid[metric_name] += 1
    intervals = {}
    for metric_name, values in samples.items():
        if not values:
            raise RuntimeError(
                f"all transfer bootstrap replicates invalid for {metric_name}"
            )
        intervals[metric_name] = (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )
    return intervals, invalid


def cross_dataset_transfer(
    frame: pd.DataFrame, selected_regressor: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one model per selection dataset and infer on every other dataset."""
    expected_features = final_feature_columns()
    if len(frame) != sum(value[0] for value in FINAL_EXPECTED_POPULATIONS.values()):
        raise RuntimeError("transfer input does not contain the expected 3,638 samples")
    if frame[expected_features].shape[1] != 46:
        raise RuntimeError("transfer input does not contain exactly 46 features")
    if not np.isfinite(frame[expected_features].to_numpy(dtype=float)).all():
        raise RuntimeError("transfer input contains non-finite feature values")

    prediction_rows = []
    resolved_models = {}
    for source_index, source_dataset in enumerate(MODEL_SELECTION_DATASETS):
        print(f"    fitting transfer source model on {source_dataset}", flush=True)
        source = frame[frame["dataset"].eq(source_dataset)].reset_index(drop=True)
        target = frame[~frame["dataset"].eq(source_dataset)].reset_index(drop=True)
        model, predictions = fit_named_model(
            selected_regressor,
            source[expected_features],
            source["SCORE"],
            target[expected_features],
        )
        resolved_models[source_dataset] = {
            "pipeline": repr(model),
            "regressor_parameters": model.named_steps["regressor"].get_params(
                deep=True
            ),
            "training_samples": len(source),
            "training_references": int(source["REF"].nunique()),
        }
        for row, prediction in zip(target.itertuples(), predictions):
            prediction_rows.append(
                {
                    "source_dataset": source_dataset,
                    "target_dataset": row.dataset,
                    "SIGNAL": row.SIGNAL,
                    "REF": row.REF,
                    "y_true": row.SCORE,
                    "y_pred": float(prediction),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    prediction_path = (
        DIRS["predictions"] / "cross_dataset_transfer_predictions.csv"
    )
    if len(predictions) != 9012:
        raise RuntimeError(
            f"expected 9,012 cross-dataset predictions, found {len(predictions)}"
        )
    if predictions.duplicated(
        ["source_dataset", "target_dataset"] + KEY_COLUMNS
    ).any():
        raise RuntimeError("duplicate cross-dataset transfer prediction")
    if predictions["source_dataset"].eq(predictions["target_dataset"]).any():
        raise RuntimeError("a transfer source was evaluated on itself")
    predictions.to_csv(prediction_path, index=False)
    predictions = pd.read_csv(prediction_path)

    summary_rows = []
    invalid_bootstrap = {}
    pair_seeds = {}
    for source_index, source_dataset in enumerate(MODEL_SELECTION_DATASETS):
        for target_index, target_dataset in enumerate(FINAL_DATASETS):
            if target_dataset == source_dataset:
                continue
            pair = predictions[
                predictions["source_dataset"].eq(source_dataset)
                & predictions["target_dataset"].eq(target_dataset)
            ]
            expected_samples, expected_refs = FINAL_EXPECTED_POPULATIONS[target_dataset]
            if len(pair) != expected_samples or pair["REF"].nunique() != expected_refs:
                raise RuntimeError(
                    f"{source_dataset}->{target_dataset}: incomplete target population"
                )
            score = metrics(pair["y_true"], pair["y_pred"])
            pair_seed = int(
                np.random.SeedSequence(
                    [SEED, source_index, target_index]
                ).generate_state(1)[0]
            )
            intervals, invalid = clustered_transfer_intervals(pair, pair_seed)
            pair_name = f"{source_dataset}->{target_dataset}"
            pair_seeds[pair_name] = pair_seed
            invalid_bootstrap[pair_name] = invalid
            summary_rows.append(
                {
                    "source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "target_role": (
                        "model_selection"
                        if target_dataset in MODEL_SELECTION_DATASETS
                        else "evaluation_only"
                    ),
                    "target_samples": len(pair),
                    "target_references": pair["REF"].nunique(),
                    "srocc": score["srocc"],
                    "srocc_ci_low": intervals["srocc"][0],
                    "srocc_ci_high": intervals["srocc"][1],
                    "plcc": score["plcc"],
                    "plcc_ci_low": intervals["plcc"][0],
                    "plcc_ci_high": intervals["plcc"][1],
                }
            )
    summary = pd.DataFrame(summary_rows)
    if len(summary) != 18 or summary.duplicated(
        ["source_dataset", "target_dataset"]
    ).any():
        raise RuntimeError("expected exactly 18 unique transfer summaries")
    summary_path = DIRS["summaries"] / "cross_dataset_transfer_summary.csv"
    summary.to_csv(summary_path, index=False)

    write_json(
        DIRS["manifests"] / "cross_dataset_transfer.json",
        {
            "source_datasets": MODEL_SELECTION_DATASETS,
            "target_datasets": FINAL_DATASETS,
            "self_evaluation": False,
            "source_models": 3,
            "directed_pairs": 18,
            "prediction_rows": len(predictions),
            "features": 46,
            "feature_order": expected_features,
            "selected_regressor": selected_regressor,
            "resolved_source_models": resolved_models,
            "target_preprocessing_or_calibration": False,
            "target_labels_used_for_fitting": False,
            "reported_metrics": ["SROCC", "PLCC"],
            "rmse_omitted_reason": "source and target subjective-score scales differ",
            "bootstrap": {
                "cluster": "target REF",
                "repeats": 10000,
                "root_seed": SEED,
                "pair_seed_derivation": "numpy SeedSequence([42, source_index, target_index])",
                "pair_seeds": pair_seeds,
                "invalid_replicates": invalid_bootstrap,
                "interpretation": "target-content variation; excludes source-training uncertainty",
            },
            "source_features_sha256": sha256(
                DIRS["normalized"] / "final_pointpcapp_46.csv"
            ),
            "predictions_sha256": sha256(prediction_path),
            "summary_sha256": sha256(summary_path),
        },
    )
    return predictions, summary


def cross_dataset_modality_importance(
    frame: pd.DataFrame, folds: pd.DataFrame, selected_regressor: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure held-out reliance on each modality for the frozen fused model."""
    repetition_rows = []
    fold_rows = []
    modalities = {
        "PointPCA3": CORE_FEATURE_SETS["pointpca3_only"],
        "DISTS": CORE_FEATURE_SETS["dists_only"],
    }
    for dataset_index, dataset in enumerate(FINAL_DATASETS):
        print(f"    grouped modality importance on {dataset}", flush=True)
        dataset_frame = frame[frame["dataset"].eq(dataset)].reset_index(drop=True)
        assignments = folds[folds["dataset"].eq(dataset)]
        fold_by_key = {
            (row.SIGNAL, row.REF): int(row.fold)
            for row in assignments.itertuples()
        }
        for fold in range(5):
            test_mask = np.array(
                [
                    fold_by_key[(row.SIGNAL, row.REF)] == fold
                    for row in dataset_frame.itertuples()
                ]
            )
            train_mask = ~test_mask
            model, baseline_predictions = fit_named_model(
                selected_regressor,
                dataset_frame.loc[train_mask, final_feature_columns()],
                dataset_frame.loc[train_mask, "SCORE"],
                dataset_frame.loc[test_mask, final_feature_columns()],
            )
            test = dataset_frame.loc[test_mask].reset_index(drop=True)
            X_test = test[final_feature_columns()].copy()
            y_true = test["SCORE"].to_numpy(dtype=float)
            groups = test["REF"].to_numpy()
            baseline = metrics(y_true, baseline_predictions)
            for modality_index, (modality, columns) in enumerate(modalities.items()):
                effects = []
                for repeat in range(50):
                    rng = np.random.default_rng(
                        SEED
                        + dataset_index * 100_000
                        + fold * 10_000
                        + modality_index * 1_000
                        + repeat
                    )
                    permuted = permute_within_groups(
                        X_test, columns, groups, rng
                    )
                    permuted_score = metrics(y_true, model.predict(permuted))
                    effect = baseline["srocc"] - permuted_score["srocc"]
                    effects.append(effect)
                    repetition_rows.append(
                        {
                            "dataset": dataset,
                            "evaluation_role": (
                                "model_selection"
                                if dataset in MODEL_SELECTION_DATASETS
                                else "evaluation_only"
                            ),
                            "fold": fold,
                            "modality": modality,
                            "repeat": repeat,
                            "baseline_srocc": baseline["srocc"],
                            "permuted_srocc": permuted_score["srocc"],
                            "srocc_decrease": effect,
                        }
                    )
                fold_rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "modality": modality,
                        "baseline_srocc": baseline["srocc"],
                        "median_srocc_decrease": float(np.median(effects)),
                        "iqr_srocc_decrease": iqr(pd.Series(effects)),
                    }
                )
    repetitions = pd.DataFrame(repetition_rows)
    if len(repetitions) != 3500 or repetitions.duplicated(
        ["dataset", "fold", "modality", "repeat"]
    ).any():
        raise RuntimeError("expected exactly 3,500 unique modality permutations")
    repetitions.to_csv(
        DIRS["predictions"]
        / "cross_dataset_modality_importance_repetitions.csv",
        index=False,
    )
    fold_summary = pd.DataFrame(fold_rows)
    if len(fold_summary) != 70:
        raise RuntimeError("expected 70 fold-level modality-importance records")
    fold_summary.to_csv(
        DIRS["summaries"] / "cross_dataset_modality_importance_fold.csv",
        index=False,
    )
    summary = (
        fold_summary.groupby(["dataset", "modality"], sort=False)
        .agg(
            median_srocc_decrease=("median_srocc_decrease", "median"),
            q1_srocc_decrease=(
                "median_srocc_decrease",
                lambda values: float(np.percentile(values, 25)),
            ),
            q3_srocc_decrease=(
                "median_srocc_decrease",
                lambda values: float(np.percentile(values, 75)),
            ),
            fold_count=("fold", "nunique"),
        )
        .reset_index()
    )
    summary["evaluation_role"] = summary["dataset"].map(
        lambda dataset: (
            "model_selection"
            if dataset in MODEL_SELECTION_DATASETS
            else "evaluation_only"
        )
    )
    summary.to_csv(
        DIRS["summaries"] / "cross_dataset_modality_importance_summary.csv",
        index=False,
    )
    write_json(
        DIRS["manifests"] / "cross_dataset_modality_importance.json",
        {
            "selected_regressor": selected_regressor,
            "features": 46,
            "modalities": {name: columns for name, columns in modalities.items()},
            "folds": 5,
            "permutations_per_modality_fold": 50,
            "permutation_scope": "within held-out REF",
            "importance_definition": "baseline SROCC minus permuted SROCC",
            "seed": SEED,
            "records": len(repetitions),
            "source_folds_sha256": sha256(
                DIRS["folds"] / "final_pointpcapp_groupkfold.csv"
            ),
            "source_features_sha256": sha256(
                DIRS["normalized"] / "final_pointpcapp_46.csv"
            ),
        },
    )
    return fold_summary, summary


def integrated_pointpcapp_runtime(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows = []
    summary_rows = []
    for dataset in FINAL_DATASETS:
        complete = frame[frame["dataset"] == dataset][
            ["dataset"] + KEY_COLUMNS + [
                "REF_NUM_POINTS",
                "TEST_NUM_POINTS",
                "TIME_TAKEN_SECONDS",
            ]
        ].rename(columns={"TIME_TAKEN_SECONDS": "pointpcapp_time"})
        matlab = pd.read_csv(
            POINTPCA2_BENCHMARKS / dataset / "PointPCA2.csv"
        )[[*KEY_COLUMNS, "TIME_TAKEN_SECONDS"]].rename(
            columns={"TIME_TAKEN_SECONDS": "matlab_time"}
        )
        merged = complete.merge(matlab, on=KEY_COLUMNS, validate="one_to_one")
        if len(merged) != len(complete):
            raise RuntimeError(f"{dataset}: incomplete integrated runtime join")
        merged["speedup"] = merged["matlab_time"] / merged["pointpcapp_time"]
        merged["total_points"] = (
            merged["REF_NUM_POINTS"] + merged["TEST_NUM_POINTS"]
        )
        sample_rows.extend(merged.to_dict("records"))
        ci_low, ci_high = cluster_bootstrap_median(
            merged["speedup"], merged["REF"]
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "samples": len(merged),
                "reference_groups": merged["REF"].nunique(),
                "matlab_median_time": merged["matlab_time"].median(),
                "matlab_time_iqr": iqr(merged["matlab_time"]),
                "pointpcapp_median_time": merged["pointpcapp_time"].median(),
                "pointpcapp_time_iqr": iqr(merged["pointpcapp_time"]),
                "median_speedup": merged["speedup"].median(),
                "speedup_iqr": iqr(merged["speedup"]),
                "speedup_ci_low": ci_low,
                "speedup_ci_high": ci_high,
            }
        )
    samples = pd.DataFrame(sample_rows)
    summary = pd.DataFrame(summary_rows)
    samples.to_csv(
        DIRS["predictions"] / "runtime_pointpcapp_integrated_by_sample.csv",
        index=False,
    )
    summary.to_csv(
        DIRS["summaries"] / "runtime_pointpcapp_integrated_summary.csv",
        index=False,
    )
    return samples, summary


def runtime_analysis(selected_pipeline: str, selected_iqa: str):
    timing_rows = []
    summary_rows = []
    scaling_rows = []
    for dataset in DATASETS:
        matlab = pd.read_csv(
            POINTPCA2_BENCHMARKS / dataset / "PointPCA2.csv"
        ).sort_values(KEY_COLUMNS).reset_index(drop=True)
        worker_frames = {
            worker: pd.read_csv(
                POINTPCA3_FEATURES / dataset / str(worker) / "PointPCA3-Rust.csv"
            )
            .sort_values(KEY_COLUMNS)
            .reset_index(drop=True)
            for worker in WORKERS
        }
        for worker, frame in worker_frames.items():
            speedup = worker_frames[1]["TIME_TAKEN_SECONDS"] / frame["TIME_TAKEN_SECONDS"]
            for idx in range(len(frame)):
                scaling_rows.append(
                    {
                        "dataset": dataset,
                        "worker": worker,
                        "SIGNAL": frame.iloc[idx]["SIGNAL"],
                        "REF": frame.iloc[idx]["REF"],
                        "time": frame.iloc[idx]["TIME_TAKEN_SECONDS"],
                        "speedup_from_1": speedup.iloc[idx],
                        "parallel_efficiency": speedup.iloc[idx] / worker,
                    }
                )
        pc16 = worker_frames[16]
        speedup = matlab["TIME_TAKEN_SECONDS"] / pc16["TIME_TAKEN_SECONDS"]
        ci_low, ci_high = cluster_bootstrap_median(speedup, matlab["REF"])
        summary_rows.append(
            {
                "dataset": dataset,
                "samples": len(matlab),
                "reference_groups": matlab["REF"].nunique(),
                "matlab_median_time": matlab["TIME_TAKEN_SECONDS"].median(),
                "matlab_time_iqr": iqr(matlab["TIME_TAKEN_SECONDS"]),
                "pc3_w16_median_time": pc16["TIME_TAKEN_SECONDS"].median(),
                "pc3_w16_time_iqr": iqr(pc16["TIME_TAKEN_SECONDS"]),
                "median_speedup": speedup.median(),
                "speedup_iqr": iqr(speedup),
                "speedup_ci_low": ci_low,
                "speedup_ci_high": ci_high,
            }
        )
        for idx in range(len(matlab)):
            timing_rows.append(
                {
                    "dataset": dataset,
                    "SIGNAL": matlab.iloc[idx]["SIGNAL"],
                    "REF": matlab.iloc[idx]["REF"],
                    "matlab_time": matlab.iloc[idx]["TIME_TAKEN_SECONDS"],
                    "pc3_w16_time": pc16.iloc[idx]["TIME_TAKEN_SECONDS"],
                    "speedup": speedup.iloc[idx],
                }
            )
    pd.DataFrame(timing_rows).to_csv(DIRS["predictions"] / "runtime_pointpca_by_sample.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DIRS["summaries"] / "runtime_pointpca_summary.csv", index=False)
    scaling = pd.DataFrame(scaling_rows)
    scaling.to_csv(DIRS["predictions"] / "runtime_scaling_by_sample.csv", index=False)
    scaling_summary = (
        scaling.groupby(["dataset", "worker"])
        .agg(
            median_time=("time", "median"),
            time_iqr=("time", iqr),
            median_speedup=("speedup_from_1", "median"),
            speedup_iqr=("speedup_from_1", iqr),
            median_efficiency=("parallel_efficiency", "median"),
        )
        .reset_index()
    )
    scaling_summary.to_csv(DIRS["summaries"] / "runtime_scaling_summary.csv", index=False)

    matlab = pd.read_csv(POINTPCA2_BENCHMARKS / "APSIPA" / "PointPCA2.csv")
    pc3 = pd.read_csv(POINTPCA3_FEATURES / "APSIPA" / "16" / "PointPCA3-Rust.csv")
    proj = pd.read_csv(projection_path(selected_pipeline, selected_iqa))
    merged = matlab[KEY_COLUMNS + ["TIME_TAKEN_SECONDS"]].rename(
        columns={"TIME_TAKEN_SECONDS": "matlab_time"}
    )
    merged = merged.merge(
        pc3[KEY_COLUMNS + ["TIME_TAKEN_SECONDS"]].rename(
            columns={"TIME_TAKEN_SECONDS": "pc3_w16_time"}
        ),
        on=KEY_COLUMNS,
        validate="one_to_one",
    ).merge(
        proj[KEY_COLUMNS + ["TIME_TAKEN_SECONDS"]].rename(
            columns={"TIME_TAKEN_SECONDS": "cubemap_time"}
        ),
        on=KEY_COLUMNS,
        validate="one_to_one",
    )
    merged["pointpcapp_time"] = merged[["pc3_w16_time", "cubemap_time"]].max(axis=1)
    merged["speedup"] = merged["matlab_time"] / merged["pointpcapp_time"]
    merged["bottleneck"] = np.where(
        merged["pc3_w16_time"] >= merged["cubemap_time"], "PointPCA3", "Cubemap"
    )
    merged.to_csv(DIRS["predictions"] / "runtime_pointpcapp_by_sample.csv", index=False)
    ci_low, ci_high = cluster_bootstrap_median(merged["speedup"], merged["REF"])
    complete = {
        "dataset": "APSIPA",
        "selected_pipeline": selected_pipeline,
        "selected_iqa": selected_iqa,
        "samples": len(merged),
        "reference_groups": merged["REF"].nunique(),
        "matlab_median_time": merged["matlab_time"].median(),
        "matlab_time_iqr": iqr(merged["matlab_time"]),
        "pointpcapp_median_time": merged["pointpcapp_time"].median(),
        "pointpcapp_time_iqr": iqr(merged["pointpcapp_time"]),
        "median_speedup": merged["speedup"].median(),
        "speedup_iqr": iqr(merged["speedup"]),
        "speedup_ci_low": ci_low,
        "speedup_ci_high": ci_high,
        "pointpca3_bottleneck_samples": int((merged["bottleneck"] == "PointPCA3").sum()),
        "cubemap_bottleneck_samples": int((merged["bottleneck"] == "Cubemap").sum()),
    }
    pd.DataFrame([complete]).to_csv(DIRS["summaries"] / "runtime_pointpcapp_summary.csv", index=False)
    return summary, scaling_summary, pd.DataFrame([complete])


def configure_figure_style() -> dict[str, object]:
    palette = sns.color_palette("deep")
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="deep",
        font="DejaVu Sans",
        font_scale=1.0,
        rc={
            "axes.edgecolor": "0.25",
            "axes.labelcolor": "0.15",
            "axes.linewidth": 0.7,
            "grid.alpha": 0.45,
            "grid.color": "0.82",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "text.color": "0.15",
            "xtick.color": "0.2",
            "ytick.color": "0.2",
        },
    )
    return {
        "palette": palette,
        "primary": palette[0],
        "secondary": palette[1],
        "positive": palette[2],
        "selected": palette[3],
        "neutral": palette[7],
    }


def save_figure(fig, filename: str) -> None:
    fig.savefig(
        DIRS["figures"] / filename,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def load_pointpcapp_runtime_samples() -> pd.DataFrame:
    runtime_path = (
        DIRS["predictions"] / "runtime_pointpcapp_integrated_by_sample.csv"
    )
    runtime = pd.read_csv(runtime_path)

    runtime_required = ["dataset", *KEY_COLUMNS] + [
        "matlab_time",
        "pointpcapp_time",
        "speedup",
    ]
    missing_runtime = sorted(set(runtime_required) - set(runtime.columns))
    if missing_runtime:
        raise ValueError(
            f"PointPCA++ runtime samples are missing columns: {missing_runtime}"
        )
    if runtime.duplicated(["dataset", *KEY_COLUMNS]).any():
        raise ValueError(
            "Figure 16 sources contain duplicate dataset/(SIGNAL, REF) keys"
        )
    samples = runtime
    observed = {
        dataset: (
            len(samples[samples["dataset"].eq(dataset)]),
            samples[samples["dataset"].eq(dataset)]["REF"].nunique(),
        )
        for dataset in FINAL_DATASETS
    }
    if len(samples) != 3638 or observed != FINAL_EXPECTED_POPULATIONS:
        raise ValueError(
            "Figure 16 requires all 3,638 samples and 189 references from the "
            "seven complete datasets"
        )

    numeric_columns = [
        "matlab_time",
        "pointpcapp_time",
        "speedup",
        "REF_NUM_POINTS",
        "TEST_NUM_POINTS",
    ]
    numeric = samples[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric <= 0).any().any():
        raise ValueError("Figure 16 sources contain non-finite or non-positive values")

    expected_speedup = samples["matlab_time"] / samples["pointpcapp_time"]
    if not np.allclose(samples["speedup"], expected_speedup, rtol=1e-12, atol=1e-12):
        raise ValueError("Saved Figure 16 speedups do not match the paired sample times")

    expected_total = samples["REF_NUM_POINTS"] + samples["TEST_NUM_POINTS"]
    if "total_points" in samples and not np.array_equal(
        samples["total_points"].to_numpy(), expected_total.to_numpy()
    ):
        raise ValueError("Saved Figure 16 total-point values are inconsistent")
    samples["total_points"] = expected_total
    samples["total_points_millions"] = samples["total_points"] / 1_000_000
    return samples


def prepare_iqa_selection_pairs(selection: pd.DataFrame) -> pd.DataFrame:
    required = {
        "outer_fold",
        "iqa",
        "srocc",
        "plcc",
        "rmse",
        "median_time",
        "selected",
    }
    missing = sorted(required - set(selection.columns))
    if missing:
        raise ValueError(f"IQA selection summary is missing columns: {missing}")
    if len(selection) != 110 or selection.duplicated(["outer_fold", "iqa"]).any():
        raise ValueError("Figure 15 requires 110 unique outer-partition/IQA rows")
    folds = sorted(selection["outer_fold"].unique())
    if folds != list(range(5)):
        raise ValueError("Figure 15 requires outer partitions 0 through 4")

    expected_runner_up = {
        0: "TOPIQ_FRMetric",
        1: "HaarPSIMetric",
        2: "GMSDMetric",
        3: "DSSMetric",
        4: "WADIQAM_FRMetric",
    }
    pair_rows = []
    for outer_fold in folds:
        candidates = selection[selection["outer_fold"] == outer_fold].copy()
        if len(candidates) != len(IQAS) or set(candidates["iqa"]) != set(IQAS):
            raise ValueError(
                f"Outer partition {outer_fold} does not contain all 22 IQAs"
            )
        ordered = candidates.sort_values(
            ["srocc", "plcc", "rmse", "median_time"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        winner = ordered.iloc[0]
        runner_up = ordered.iloc[1]
        selected_rows = candidates[candidates["selected"].astype(bool)]
        if (
            winner["iqa"] != "DISTSMetric"
            or len(selected_rows) != 1
            or selected_rows.iloc[0]["iqa"] != "DISTSMetric"
        ):
            raise ValueError(
                f"DISTS is not the unique selected winner in partition {outer_fold}"
            )
        if runner_up["iqa"] != expected_runner_up[outer_fold]:
            raise ValueError(
                f"Unexpected Figure 15 runner-up in partition {outer_fold}: "
                f"{runner_up['iqa']}"
            )
        margin = float(winner["srocc"] - runner_up["srocc"])
        if margin <= 0:
            raise ValueError(
                f"DISTS does not have a positive SROCC margin in partition {outer_fold}"
            )
        pair_rows.append(
            {
                "outer_fold": outer_fold,
                "partition": f"Partition {outer_fold + 1}",
                "dists_srocc": float(winner["srocc"]),
                "runner_up": str(runner_up["iqa"]),
                "runner_up_srocc": float(runner_up["srocc"]),
                "margin": margin,
            }
        )
    return pd.DataFrame(pair_rows)


def generate_figures(
    projection_summary,
    projection_ranks,
    regressor_outer,
    stable,
    selected_regressor,
    importance,
    iqa_outer_selection,
    ablation,
    runtime_summary,
    scaling_summary,
    complete_runtime,
    runtime_samples,
):
    del runtime_summary, complete_runtime
    # The worker-scaling figure consumes scaling_summary directly; Figure 16 uses
    # paired sample-level times rather than the aggregate runtime summary.
    style = configure_figure_style()
    palette = style["palette"]

    dataset_colors = dict(
        zip(DATASETS, sns.color_palette("deep", n_colors=len(DATASETS)))
    )
    markers = ["o", "s", "D", "^", "v", "P", "X", "h"]
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (5, 2, 1, 2))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for index, dataset in enumerate(DATASETS):
        frame = scaling_summary[scaling_summary["dataset"] == dataset]
        common = {
            "color": dataset_colors[dataset],
            "label": dataset,
            "marker": markers[index],
            "linestyle": line_styles[index],
        }
        axes[0].plot(frame["worker"], frame["median_time"], **common)
        axes[1].plot(frame["worker"], frame["median_speedup"], **common)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Workers")
    axes[0].set_ylabel("Median time (s, log scale)")
    axes[1].set_xlabel("Workers")
    axes[1].set_ylabel("Median speedup over one worker")
    axes[0].set_xticks(WORKERS)
    axes[1].set_xticks(WORKERS)
    axes[1].axvline(16, color="0.2", linestyle=":", linewidth=1)
    for ax in axes:
        sns.despine(ax=ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    save_figure(fig, "pointpca3_worker_scaling.pdf")

    pivot = projection_summary.pivot(
        index="pipeline", columns="iqa", values="srocc"
    ).loc[list(PIPELINES)]
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=sns.color_palette("crest", as_cmap=True),
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Median SROCC"},
    )
    ax.set_xlabel("IQA metric")
    ax.set_ylabel("Projection pipeline")
    ax.tick_params(axis="x", rotation=60)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    save_figure(fig, "projection_srocc_heatmap.pdf")

    padder_labels = {
        "crop_pad_navier": "Navier–Stokes",
        "nocrop_pad_navier": "Navier–Stokes",
        "crop_pad_fsr": "FSRFast",
        "nocrop_pad_fsr": "FSRFast",
        "crop_pad_telea": "Telea",
        "nocrop_pad_telea": "Telea",
        "crop_pad_shiftmap": "ShiftMap",
        "nocrop_pad_shiftmap": "ShiftMap",
        "crop_nopad": "No padding",
        "nocrop_nopad": "No padding",
    }
    padder_markers = {
        "Navier–Stokes": "o",
        "FSRFast": "s",
        "Telea": "^",
        "ShiftMap": "D",
        "No padding": "X",
    }
    family_order = [
        "crop + padding",
        "no crop + padding",
        "crop + no padding",
        "no crop + no padding",
    ]
    family_palette = dict(zip(family_order, palette[: len(family_order)]))
    family_display = {
        "crop + padding": "Crop + pad",
        "no crop + padding": "No crop + pad",
        "crop + no padding": "Crop only",
        "no crop + no padding": "No crop/no pad",
    }
    pipeline_labels = {
        "crop_pad_navier": "Crop + Navier–Stokes",
        "crop_pad_fsr": "Crop + FSRFast",
        "crop_pad_telea": "Crop + Telea",
        "crop_pad_shiftmap": "Crop + ShiftMap",
        "nocrop_pad_navier": "No crop + Navier–Stokes",
        "nocrop_pad_fsr": "No crop + FSRFast",
        "nocrop_pad_telea": "No crop + Telea",
        "nocrop_pad_shiftmap": "No crop + ShiftMap",
        "crop_nopad": "Crop + no padding",
        "nocrop_nopad": "No crop + no padding",
    }
    plot_frame = projection_ranks.copy()
    plot_frame["padder"] = plot_frame["pipeline"].map(padder_labels)
    competitive = plot_frame[plot_frame["aggregate_rank"] <= 6].copy()
    if set(plot_frame["aggregate_rank"].astype(int)) != set(range(1, 11)):
        raise RuntimeError("Projection overview must contain aggregate ranks 1 through 10")
    if set(competitive["aggregate_rank"].astype(int)) != set(range(1, 7)):
        raise RuntimeError("Projection zoom panel must contain aggregate ranks 1 through 6")

    def plot_projection_points(target, frame, size=60):
        for _, point in frame.iterrows():
            target.scatter(
                point["median_time"],
                point["median_srocc"],
                s=size,
                marker=padder_markers[point["padder"]],
                facecolor=family_palette[point["family"]],
                edgecolor="white",
                linewidth=0.65,
                zorder=3,
            )

    fig, (overview_ax, zoom_ax) = plt.subplots(
        2,
        1,
        figsize=(5.8, 6.8),
        gridspec_kw={"height_ratios": [1, 2]},
    )
    plot_projection_points(overview_ax, plot_frame, size=54)
    selected = plot_frame.loc[plot_frame["aggregate_rank"].idxmin()]

    def highlight_selected(target, size):
        target.scatter(
            selected["median_time"],
            selected["median_srocc"],
            s=size,
            facecolors="none",
            edgecolors=[style["selected"]],
            linewidths=1.4,
            zorder=5,
        )

    highlight_selected(overview_ax, 100)

    overview_label_offsets = {
        "crop_nopad": (5, -6),
        "nocrop_nopad": (5, 6),
        "crop_pad_shiftmap": (-5, 6),
        "nocrop_pad_shiftmap": (-5, 6),
    }
    for _, row in plot_frame[plot_frame["aggregate_rank"] >= 7].iterrows():
        offset = overview_label_offsets[row["pipeline"]]
        overview_ax.annotate(
            pipeline_labels[row["pipeline"]],
            (row["median_time"], row["median_srocc"]),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            va="center",
            fontsize=7.5,
            zorder=6,
        )
    overview_ax.set_xscale("log")
    overview_ax.set_title("(a) Overview — all pipelines", loc="left", fontsize=9.5, pad=3)
    overview_ax.tick_params(axis="both", which="both", labelsize=8)

    plot_projection_points(zoom_ax, competitive, size=64)
    highlight_selected(zoom_ax, 116)
    zoom_ax.set_xscale("log")
    zoom_ax.set_xlim(4.7, 16.0)
    zoom_ax.set_ylim(0.815, 0.890)
    zoom_ax.set_title(
        "(b) Competitive region — six leading pipelines",
        loc="left",
        fontsize=9.5,
        pad=3,
    )
    zoom_ax.tick_params(axis="both", which="both", labelsize=8)
    zoom_label_offsets = {
        "crop_pad_navier": (5, 6),
        "crop_pad_fsr": (-5, 6),
        "nocrop_pad_telea": (5, 6),
        "nocrop_pad_navier": (5, -6),
        "crop_pad_telea": (5, -6),
        "nocrop_pad_fsr": (-5, -6),
    }
    for _, row in competitive.iterrows():
        offset = zoom_label_offsets[row["pipeline"]]
        zoom_ax.annotate(
            pipeline_labels[row["pipeline"]],
            (row["median_time"], row["median_srocc"]),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            va="center",
            fontsize=7.5,
            zorder=6,
        )
    source_box = Rectangle(
        (4.7, 0.815),
        16.0 - 4.7,
        0.890 - 0.815,
        fill=False,
        edgecolor="0.4",
        linewidth=0.65,
        linestyle="--",
        zorder=2,
    )
    overview_ax.add_patch(source_box)

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=family_palette[family],
            markeredgecolor="white",
            markersize=5.5,
            label=family_display[family],
        )
        for family in family_order
    ]
    padder_handles = [
        Line2D(
            [0],
            [0],
            marker=padder_markers[padder],
            linestyle="none",
            markerfacecolor="0.45",
            markeredgecolor="white",
            markersize=5.5,
            label=padder,
        )
        for padder in padder_markers
    ]
    family_legend = fig.legend(
        handles=family_handles,
        title="Crop/padding family",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=4,
        fontsize=8,
        title_fontsize=8.5,
        handletextpad=0.25,
        columnspacing=0.75,
    )
    fig.add_artist(family_legend)
    fig.legend(
        handles=padder_handles,
        title="Padding algorithm",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        fontsize=8,
        title_fontsize=8.5,
        handletextpad=0.25,
        columnspacing=0.75,
    )
    for target in [overview_ax, zoom_ax]:
        sns.despine(ax=target)
    fig.supxlabel("Median cubemap time across IQAs (s, log scale)", y=0.185)
    fig.supylabel("Median SROCC across IQAs", x=0.025, y=0.61)
    fig.subplots_adjust(
        left=0.15, right=0.985, top=0.965, bottom=0.245, hspace=0.28
    )
    save_figure(fig, "projection_accuracy_runtime.pdf")

    stable_frame = regressor_outer[regressor_outer["model"].isin(stable)]
    models = (
        stable_frame.groupby("model")["srocc"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    model_palette = {
        model: style["selected"] if model == selected_regressor else style["primary"]
        for model in models
    }
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 0.65), 4.5))
    sns.boxplot(
        data=stable_frame,
        x="model",
        y="srocc",
        order=models,
        hue="model",
        hue_order=models,
        palette=model_palette,
        dodge=False,
        width=0.62,
        linewidth=0.8,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": style["secondary"],
            "markeredgecolor": "white",
            "markersize": 4.5,
        },
        legend=False,
        ax=ax,
    )
    for model, box in zip(models, ax.patches):
        if model == selected_regressor:
            box.set_hatch("///")
            box.set_edgecolor("0.15")
            box.set_linewidth(1.0)
    ax.tick_params(axis="x", rotation=55)
    ax.set_ylabel("Outer-fold SROCC")
    ax.set_xlabel("Regressor")
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, "regressor_comparison.pdf")

    individual = importance[importance["level"] == "individual"].nlargest(
        15, "median_srocc_decrease"
    ).copy()
    individual["feature"] = individual["feature"].str.replace(
        "metric", "", regex=False
    )
    blocks = importance[importance["level"] == "iqa_block"].nlargest(
        15, "median_srocc_decrease"
    ).copy()
    blocks["feature"] = blocks["feature"].str.replace("Metric", "", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, frame, title in [
        (axes[0], individual, "Individual features"),
        (axes[1], blocks, "IQA blocks"),
    ]:
        frame = frame.sort_values("median_srocc_decrease")
        bar_palette = {
            feature: style["selected"] if feature == "DISTS" else style["primary"]
            for feature in frame["feature"]
        }
        sns.barplot(
            data=frame,
            x="median_srocc_decrease",
            y="feature",
            order=frame["feature"].tolist(),
            hue="feature",
            palette=bar_palette,
            dodge=False,
            legend=False,
            ax=ax,
        )
        for feature, bar in zip(frame["feature"], ax.patches):
            if feature == "DISTS":
                bar.set_hatch("///")
                bar.set_edgecolor("0.15")
                bar.set_linewidth(0.9)
        ax.set_xlabel("Median held-out SROCC decrease")
        ax.set_ylabel("")
        ax.set_title(title)
        sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, "feature_importance.pdf")

    pairs = prepare_iqa_selection_pairs(iqa_outer_selection)
    y_positions = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    for y, row in zip(y_positions, pairs.to_dict("records")):
        ax.hlines(
            y,
            row["runner_up_srocc"],
            row["dists_srocc"],
            color=style["neutral"],
            linewidth=1.4,
            zorder=1,
        )
        ax.scatter(
            row["runner_up_srocc"],
            y,
            color=style["primary"],
            marker="o",
            s=42,
            zorder=2,
        )
        ax.scatter(
            row["dists_srocc"],
            y,
            color=style["selected"],
            marker="D",
            s=48,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        runner_label = (
            row["runner_up"].replace("Metric", "").replace("_", "-")
        )
        ax.annotate(
            runner_label,
            (row["runner_up_srocc"], y),
            xytext=(-7, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.5,
            color=style["primary"],
        )
        ax.annotate(
            f"+{row['margin']:.4f}",
            (row["dists_srocc"], y),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color=style["selected"],
        )
    ax.set_yticks(y_positions, pairs["partition"])
    ax.invert_yaxis()
    ax.set_xlim(0.956, 0.979)
    ax.set_xlabel("Median inner-validation SROCC")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=style["primary"],
                marker="o",
                linestyle="none",
                markersize=5.5,
                label="Strongest alternative",
            ),
            Line2D(
                [0],
                [0],
                color=style["selected"],
                marker="D",
                linestyle="none",
                markeredgecolor="white",
                markersize=6,
                label="DISTS",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        frameon=False,
        ncol=2,
    )
    ax.text(
        1,
        1.06,
        "DISTS selected in 5/5 partitions",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=style["selected"],
        fontweight="bold",
    )
    sns.despine(ax=ax)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "single_iqa_selection.pdf")

    order = [
        "pointpca3_only",
        "selected_iqa_only",
        "full_fusion_172",
        "compact_fusion_46",
    ]
    frame = ablation.set_index("feature_set").loc[order].reset_index()
    display_sets = {
        "pointpca3_only": "PointPCA$^3$",
        "selected_iqa_only": "DISTS",
        "full_fusion_172": "Full (172)",
        "compact_fusion_46": "Compact (46)",
    }
    frame["display_set"] = frame["feature_set"].map(display_sets)
    long_frame = frame.melt(
        id_vars=["feature_set", "display_set"],
        value_vars=["srocc", "plcc"],
        var_name="metric",
        value_name="correlation",
    )
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.barplot(
        data=long_frame,
        x="display_set",
        y="correlation",
        order=[display_sets[item] for item in order],
        hue="metric",
        hue_order=["srocc", "plcc"],
        palette={"srocc": style["primary"], "plcc": style["secondary"]},
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Median outer-fold correlation")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(title=None, labels=["SROCC", "PLCC"])
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, "feature_set_ablation.pdf")

    generate_pointpcapp_runtime_figure(runtime_samples)


def generate_pointpcapp_runtime_figure(runtime_samples: pd.DataFrame) -> None:
    style = configure_figure_style()
    layout = [
        ("APSIPA", 0),
        ("ICIP2023", 1),
        ("LS-PCQA", 2),
        ("SJTU-PCQA", 3),
        ("UnB_PC", 0),
        ("WPC", 1),
        ("WPC2", 2),
    ]
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(12.2, 9.0),
        gridspec_kw={"hspace": 0.42, "wspace": 0.34},
    )
    panel_letters = "abcdefg"
    for index, (dataset, column) in enumerate(layout):
        row_offset = 0 if index < 4 else 2
        time_ax = axes[row_offset, column]
        speedup_ax = axes[row_offset + 1, column]
        data = runtime_samples[runtime_samples["dataset"].eq(dataset)].copy()
        common = {
            "data": data,
            "x": "total_points_millions",
            "units": "REF",
            "order": 2,
            "ci": 95,
            "n_boot": 10_000,
            "seed": SEED,
            "truncate": True,
        }
        sns.regplot(
            **common,
            y="matlab_time",
            color=style["neutral"],
            marker="o",
            scatter_kws={
                "s": 10,
                "alpha": 0.48,
                "edgecolor": "none",
                "zorder": 2,
            },
            line_kws={"linewidth": 1.5, "zorder": 3},
            ax=time_ax,
        )
        sns.regplot(
            **common,
            y="pointpcapp_time",
            color=style["primary"],
            marker="^",
            scatter_kws={
                "s": 12,
                "alpha": 0.52,
                "edgecolor": "none",
                "zorder": 2,
            },
            line_kws={"linewidth": 1.5, "zorder": 3},
            ax=time_ax,
        )
        sns.regplot(
            **common,
            y="speedup",
            color=style["secondary"],
            marker="o",
            scatter_kws={
                "s": 10,
                "alpha": 0.48,
                "edgecolor": "none",
                "zorder": 2,
            },
            line_kws={"linewidth": 1.5, "zorder": 3},
            ax=speedup_ax,
        )
        maximum_x = data["total_points_millions"].max() * 1.03
        for ax in (time_ax, speedup_ax):
            ax.set_xlim(0, maximum_x)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("")
            sns.despine(ax=ax)
        time_ax.set_title(
            f"({panel_letters[index]}) {dataset}", fontsize=9, pad=3
        )
        time_ax.set_ylabel("Time (s)" if column == 0 else "")
        speedup_ax.set_ylabel("Speedup ($\\times$)" if column == 0 else "")
        speedup_ax.set_xlabel("Total points (millions)")

    axes[2, 3].axis("off")
    axes[3, 3].axis("off")
    fig.legend(
        handles=[
            Line2D(
                [0], [0], color=style["neutral"], marker="o", linewidth=1.5,
                markersize=4.5, label="MATLAB PointPCA2",
            ),
            Line2D(
                [0], [0], color=style["primary"], marker="^", linewidth=1.5,
                markersize=4.5, label="PointPCA++ (measured)",
            ),
            Line2D(
                [0], [0], color=style["secondary"], marker="o", linewidth=1.5,
                markersize=4.5, label="Paired speedup",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.09)
    save_figure(fig, "pointpcapp_runtime.pdf")


def generate_cross_dataset_modality_importance_figure(
    summary: pd.DataFrame,
) -> None:
    style = configure_figure_style()
    expected = {(dataset, modality) for dataset in FINAL_DATASETS for modality in ["DISTS", "PointPCA3"]}
    observed = set(zip(summary["dataset"], summary["modality"]))
    if observed != expected or not summary["fold_count"].eq(5).all():
        raise ValueError("incomplete seven-dataset modality-importance summary")
    display_order = MODEL_SELECTION_DATASETS + EVALUATION_ONLY_DATASETS
    y = np.arange(len(display_order), dtype=float)
    offsets = {"DISTS": -0.13, "PointPCA3": 0.13}
    colors = {"DISTS": style["selected"], "PointPCA3": style["primary"]}
    markers = {"DISTS": "D", "PointPCA3": "o"}
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    ax.axhspan(-0.45, 2.45, color=style["selected"], alpha=0.055, zorder=0)
    ax.axhspan(2.55, 6.45, color=style["primary"], alpha=0.04, zorder=0)
    ax.axvline(0, color="0.35", linewidth=0.9, linestyle="--", zorder=1)
    for modality in ["DISTS", "PointPCA3"]:
        modality_rows = summary[summary["modality"].eq(modality)].set_index("dataset")
        for position, dataset in zip(y, display_order):
            row = modality_rows.loc[dataset]
            value = float(row["median_srocc_decrease"])
            q1 = float(row["q1_srocc_decrease"])
            q3 = float(row["q3_srocc_decrease"])
            ax.errorbar(
                value,
                position + offsets[modality],
                xerr=np.array([[value - q1], [q3 - value]]),
                fmt=markers[modality],
                color=colors[modality],
                ecolor=colors[modality],
                elinewidth=1.4,
                capsize=2.5,
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=3,
            )
    labels = [
        f"{dataset}  [modelling]" if dataset in MODEL_SELECTION_DATASETS
        else f"{dataset}  [evaluation]"
        for dataset in display_order
    ]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Grouped permutation importance: baseline SROCC $-$ permuted SROCC"
    )
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="D", linestyle="none", color=colors["DISTS"], label="DISTS (6 views)"),
            Line2D([0], [0], marker="o", linestyle="none", color=colors["PointPCA3"], label="PointPCA$^3$ (40 features)"),
        ],
        loc="lower right",
        ncol=2,
        frameon=False,
    )
    ax.text(
        0.01,
        0.985,
        "positive values indicate model reliance",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.35",
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, "pointpcapp_modality_importance.pdf")


def generate_cross_dataset_transfer_figure(summary: pd.DataFrame) -> None:
    expected_pairs = {
        (source, target)
        for source in MODEL_SELECTION_DATASETS
        for target in FINAL_DATASETS
        if source != target
    }
    observed_pairs = set(zip(summary["source_dataset"], summary["target_dataset"]))
    if observed_pairs != expected_pairs or len(summary) != 18:
        raise RuntimeError("incomplete cross-dataset transfer summary")
    interval_columns = [
        "srocc_ci_low",
        "srocc_ci_high",
        "plcc_ci_low",
        "plcc_ci_high",
    ]
    if summary[interval_columns].min().min() < 0.45 or summary[
        interval_columns
    ].max().max() > 1.0:
        raise RuntimeError("a transfer interval would be clipped by Figure 17")

    style = configure_figure_style()
    display_order = MODEL_SELECTION_DATASETS + EVALUATION_ONLY_DATASETS
    display = {dataset: dataset for dataset in display_order}
    y_by_dataset = {
        dataset: position for position, dataset in enumerate(display_order)
    }
    metric_style = {
        "srocc": {"label": "SROCC", "color": style["primary"], "marker": "o", "offset": -0.11},
        "plcc": {"label": "PLCC", "color": style["secondary"], "marker": "D", "offset": 0.11},
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.3), sharex=True, sharey=True)
    for ax, source_dataset in zip(axes, MODEL_SELECTION_DATASETS):
        ax.axhspan(-0.5, 2.5, color=style["selected"], alpha=0.05, zorder=0)
        ax.axhspan(2.5, 6.5, color=style["primary"], alpha=0.04, zorder=0)
        source_y = y_by_dataset[source_dataset]
        ax.axhspan(
            source_y - 0.42,
            source_y + 0.42,
            color=style["neutral"],
            alpha=0.15,
            zorder=1,
        )
        source_rows = summary[summary["source_dataset"].eq(source_dataset)]
        for metric_name, encoding in metric_style.items():
            for row in source_rows.to_dict("records"):
                position = y_by_dataset[row["target_dataset"]] + encoding["offset"]
                value = float(row[metric_name])
                ax.errorbar(
                    value,
                    position,
                    xerr=np.array(
                        [
                            [value - float(row[f"{metric_name}_ci_low"])],
                            [float(row[f"{metric_name}_ci_high"]) - value],
                        ]
                    ),
                    fmt=encoding["marker"],
                    color=encoding["color"],
                    ecolor=encoding["color"],
                    elinewidth=1.2,
                    capsize=2.3,
                    markersize=5.2,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    zorder=3,
                )
        ax.text(
            0.725,
            source_y,
            "TRAINING SOURCE",
            ha="center",
            va="center",
            fontsize=7.2,
            color="0.35",
            fontweight="semibold",
            zorder=4,
        )
        ax.axhline(2.5, color="0.72", linewidth=0.7, zorder=2)
        ax.set_xlim(0.45, 1.00)
        ax.set_xlabel("Target-set correlation")
        ax.set_title(f"Trained on {source_dataset}", fontsize=10)
        ax.grid(axis="y", visible=False)
        sns.despine(ax=ax)
    axes[0].set_yticks(
        np.arange(len(display_order)), [display[name] for name in display_order]
    )
    axes[0].set_ylabel("Test dataset")
    axes[0].invert_yaxis()
    metric_handles = [
        Line2D(
            [0],
            [0],
            marker=encoding["marker"],
            linestyle="none",
            color=encoding["color"],
            markeredgecolor="white",
            markersize=6,
            label=encoding["label"],
        )
        for encoding in metric_style.values()
    ]
    role_handles = [
        Patch(
            facecolor=style["selected"],
            alpha=0.10,
            edgecolor="none",
            label="Modelling target",
        ),
        Patch(
            facecolor=style["primary"],
            alpha=0.08,
            edgecolor="none",
            label="Evaluation target",
        ),
    ]
    fig.legend(
        handles=metric_handles + role_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1), w_pad=1.0)
    save_figure(fig, "pointpcapp_all_datasets.pdf")


def write_final_pointpcapp_tables(
    performance: pd.DataFrame, runtime: pd.DataFrame
) -> None:
    labels = {
        "APSIPA": r"APSIPA$^{\dagger}$",
        "ICIP2023": "ICIP2023",
        "LS-PCQA": r"LS-PCQA$^{\dagger}$",
        "SJTU-PCQA": "SJTU-PCQA",
        "UnB_PC": r"UnB\_PC",
        "WPC": r"WPC$^{\dagger}$",
        "WPC2": "WPC2",
    }
    performance = performance.set_index("dataset").loc[FINAL_DATASETS].reset_index()
    rows = []
    for row in performance.itertuples():
        rows.append(
            f"{labels[row.dataset]} & {int(row.samples)} & {int(row.reference_groups)} & "
            f"{row.srocc:.3f} [{row.srocc_ci_low:.3f}, {row.srocc_ci_high:.3f}] & "
            f"{row.plcc:.3f} [{row.plcc_ci_low:.3f}, {row.plcc_ci_high:.3f}] & "
            f"{row.rmse:.3f} [{row.rmse_ci_low:.3f}, {row.rmse_ci_high:.3f}] \\\\"
        )
    (DIRS["tables"] / "tab_pointpcapp_all_datasets.tex").write_text(
        "\n".join(rows) + "\n\\bottomrule\n"
    )

    runtime = runtime.set_index("dataset").loc[FINAL_DATASETS].reset_index()
    runtime_labels = {**labels, "APSIPA": "APSIPA"}
    runtime_rows = []
    for row in runtime.itertuples():
        runtime_rows.append(
            f"{runtime_labels[row.dataset]} & "
            f"{int(row.samples)} & {int(row.reference_groups)} & "
            f"{row.pointpcapp_median_time:.2f} [{row.pointpcapp_time_iqr:.2f}] & "
            f"{row.matlab_median_time:.2f} [{row.matlab_time_iqr:.2f}] & "
            f"{row.median_speedup:.2f} [{row.speedup_iqr:.2f}] & "
            f"[{row.speedup_ci_low:.2f}, {row.speedup_ci_high:.2f}] \\\\"
        )
    (DIRS["tables"] / "runtime_pointpcapp_integrated_rows.tex").write_text(
        "\n".join(runtime_rows) + "\n\\bottomrule\n"
    )

    apsipa = runtime[runtime["dataset"] == "APSIPA"].iloc[0]
    modality_importance = pd.read_csv(
        DIRS["summaries"] / "importance_modalities.csv"
    ).set_index("feature")
    macro_path = DIRS["tables"] / "experimental_results_macros.tex"
    lines = macro_path.read_text().splitlines()
    values = {
        "SelectedRegressor": str(
            core3_selection_manifest()["selected_regressor"]
        ).replace("ExtraTreesRegressor", "Extra Trees Regressor"),
        "PointPCAppTime": f"{apsipa['pointpcapp_median_time']:.2f}",
        "PointPCASecondTimeIQR": f"{apsipa['pointpcapp_time_iqr']:.2f}",
        "MatlabPointPCATime": f"{apsipa['matlab_median_time']:.2f}",
        "PointPCAppSpeedup": f"{apsipa['median_speedup']:.2f}",
        "PointPCAppSpeedupIQR": f"{apsipa['speedup_iqr']:.2f}",
        "PointPCAppSpeedupCILow": f"{apsipa['speedup_ci_low']:.2f}",
        "PointPCAppSpeedupCIHigh": f"{apsipa['speedup_ci_high']:.2f}",
        "ProjectionImportance": f"{modality_importance.loc['Projections', 'median_srocc_decrease']:.3f}",
        "PointPCAImportance": f"{modality_importance.loc['PointPCA3', 'median_srocc_decrease']:.3f}",
    }
    for name, value in values.items():
        prefix = f"\\newcommand{{\\{name}}}"
        lines = [
            f"{prefix}{{{value}}}" if line.startswith(prefix) else line
            for line in lines
        ]
    macro_path.write_text("\n".join(lines) + "\n")


def write_cross_dataset_transfer_macros(summary: pd.DataFrame) -> None:
    if len(summary) != 18:
        raise RuntimeError("cannot write transfer macros from an incomplete summary")

    def latex_dataset(name: str) -> str:
        return name.replace("_", r"\_")

    lowest_srocc = summary.loc[summary["srocc"].idxmin()]
    highest_srocc = summary.loc[summary["srocc"].idxmax()]
    lowest_plcc = summary.loc[summary["plcc"].idxmin()]
    highest_plcc = summary.loc[summary["plcc"].idxmax()]
    values = {
        "TransferPairCount": "18",
        "TransferPredictionCount": "9,012",
        "TransferLowestSROCC": f"{lowest_srocc['srocc']:.3f}",
        "TransferLowestSROCCSource": latex_dataset(
            str(lowest_srocc["source_dataset"])
        ),
        "TransferLowestSROCCTarget": latex_dataset(
            str(lowest_srocc["target_dataset"])
        ),
        "TransferHighestSROCC": f"{highest_srocc['srocc']:.3f}",
        "TransferHighestSROCCSource": latex_dataset(
            str(highest_srocc["source_dataset"])
        ),
        "TransferHighestSROCCTarget": latex_dataset(
            str(highest_srocc["target_dataset"])
        ),
        "TransferLowestPLCC": f"{lowest_plcc['plcc']:.3f}",
        "TransferLowestPLCCSource": latex_dataset(
            str(lowest_plcc["source_dataset"])
        ),
        "TransferLowestPLCCTarget": latex_dataset(
            str(lowest_plcc["target_dataset"])
        ),
        "TransferHighestPLCC": f"{highest_plcc['plcc']:.3f}",
        "TransferHighestPLCCSource": latex_dataset(
            str(highest_plcc["source_dataset"])
        ),
        "TransferHighestPLCCTarget": latex_dataset(
            str(highest_plcc["target_dataset"])
        ),
    }
    source_macro_names = {
        "APSIPA": "APSIPA",
        "WPC": "WPC",
        "LS-PCQA": "LSPCQA",
    }
    for source_dataset, macro_name in source_macro_names.items():
        source_rows = summary[summary["source_dataset"].eq(source_dataset)]
        values[f"Transfer{macro_name}MinSROCC"] = f"{source_rows['srocc'].min():.3f}"
        values[f"Transfer{macro_name}MaxSROCC"] = f"{source_rows['srocc'].max():.3f}"
    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()]
    (DIRS["tables"] / "cross_dataset_transfer_macros.tex").write_text(
        "\n".join(lines) + "\n"
    )


def numerical_artifact_hash() -> str:
    digest = hashlib.sha256()
    for directory_name in ["predictions", "folds", "normalized", "summaries"]:
        for path in sorted(DIRS[directory_name].rglob("*")):
            if not path.is_file():
                continue
            digest.update(str(path.relative_to(OUT)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def write_figure_metadata(
    source_run_id: str,
    render_mode: str,
    numerical_hash: str,
    rendered_names: set[str] | None = None,
) -> None:
    figure_sources = {
        "pointpca3_worker_scaling.pdf": "summaries/runtime_scaling_summary.csv",
        "projection_srocc_heatmap.pdf": "summaries/projection_case_summary.csv",
        "projection_accuracy_runtime.pdf": "summaries/projection_pipeline_ranks.csv",
        "regressor_comparison.pdf": "summaries/core3_regressor_inner.csv;summaries/core3_regressor_dataset_ranks.csv;summaries/core3_regressor_selection_summary.csv;manifests/core3_selection.json",
        "feature_importance.pdf": "summaries/importance_individual.csv;summaries/importance_iqa_blocks.csv",
        "single_iqa_selection.pdf": "summaries/single_iqa_outer_selection.csv",
        "feature_set_ablation.pdf": "summaries/core3_feature_set_fold_metrics.csv",
        "pointpcapp_modality_importance.pdf": "summaries/cross_dataset_modality_importance_summary.csv",
        "pointpcapp_runtime.pdf": "predictions/runtime_pointpcapp_integrated_by_sample.csv",
        "pointpcapp_all_datasets.pdf": "summaries/cross_dataset_transfer_summary.csv",
    }
    figure_labels = {
        "pointpca3_worker_scaling.pdf": "fig_pointpca3_worker_scaling",
        "projection_srocc_heatmap.pdf": "fig_projection_pipeline_heatmap",
        "projection_accuracy_runtime.pdf": "fig_projection_accuracy_runtime",
        "regressor_comparison.pdf": "fig_regressor_comparison",
        "feature_importance.pdf": "fig_feature_importance",
        "single_iqa_selection.pdf": "fig_single_iqa_selection",
        "feature_set_ablation.pdf": "fig_feature_set_ablation",
        "pointpcapp_modality_importance.pdf": "fig_pointpcapp_modality_importance",
        "pointpcapp_runtime.pdf": "fig_pointpcapp_runtime",
        "pointpcapp_all_datasets.pdf": "fig_pointpcapp_all_datasets",
    }
    provenance_path = DIRS["manifests"] / "artifact_provenance.csv"
    provenance = (
        pd.read_csv(provenance_path) if provenance_path.exists() else pd.DataFrame()
    )
    figure_rows = []
    for path in sorted(DIRS["figures"].glob("*.pdf")):
        if path.name not in figure_sources:
            continue
        if rendered_names is not None and path.name not in rendered_names:
            continue
        figure_rows.append(
            {
                "artifact": str(path.relative_to(REPO)),
                "run_id": source_run_id,
                "source_artifacts": figure_sources[path.name],
                "manuscript_label": figure_labels[path.name],
                "script": str(SCRIPT.relative_to(REPO)),
                "script_sha256": sha256(SCRIPT),
                "render_mode": render_mode,
                "seaborn_version": sns.__version__,
                "palette": "deep",
            }
        )
    provenance = pd.concat(
        [provenance, pd.DataFrame(figure_rows)], ignore_index=True
    ).drop_duplicates("artifact", keep="last")
    provenance.to_csv(provenance_path, index=False)

    deep_hex = sns.color_palette("deep").as_hex()
    source_hashes = {}
    for source_names in figure_sources.values():
        for source_name in source_names.split(";"):
            source_path = OUT / source_name
            source_hashes[str(source_path.relative_to(REPO))] = sha256(source_path)
    figure_hashes = {
        str(path.relative_to(REPO)): sha256(path)
        for path in sorted(DIRS["figures"].glob("*.pdf"))
        if path.name in figure_sources
    }
    write_json(
        DIRS["manifests"] / "figure_style_manifest.json",
        {
            "source_run_id": source_run_id,
            "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "render_mode": render_mode,
            "theme": {"context": "paper", "style": "whitegrid", "palette": "deep"},
            "heatmap_colormap": "crest",
            "figure_specific": {
                "regressor_comparison.pdf": {
                    "layout": "three inner-validation panels and aggregate-rank panel",
                    "points": "five inner-validation medians per regressor and dataset",
                    "interval": "first to third quartile, not a confidence interval",
                    "selection": "frequency first, then median aggregate rank",
                },
                "feature_set_ablation.pdf": {
                    "layout": "two metric rows by three dataset columns",
                    "statistics": "outer-fold median and first-to-third-quartile interval",
                    "correlations": "SROCC blue circles; PLCC orange diamonds",
                    "rmse": "independent dataset scales",
                },
                "projection_accuracy_runtime.pdf": {
                    "layout": "single-column stacked overview and zoom panels",
                    "height_ratio": [1, 2],
                    "overview_points": "all ten pipelines at source coordinates",
                    "overview_labeled_pipelines": [
                        "crop_nopad",
                        "nocrop_nopad",
                        "crop_pad_shiftmap",
                        "nocrop_pad_shiftmap",
                    ],
                    "zoom_limits": {
                        "median_time_seconds": [4.7, 16.0],
                        "median_srocc": [0.815, 0.890],
                    },
                    "zoom_labeled_pipelines": [
                        "crop_pad_navier",
                        "crop_pad_fsr",
                        "nocrop_pad_telea",
                        "nocrop_pad_navier",
                        "crop_pad_telea",
                        "nocrop_pad_fsr",
                    ],
                    "labeling": "direct pipeline names with fixed offsets; no connectors or manuscript ranking table",
                    "label_connectors": False,
                    "coordinate_displacement": False,
                    "family_encoding": "color",
                    "padder_encoding": "marker",
                    "selected_encoding": "red outline in overview and zoom panels",
                },
                "single_iqa_selection.pdf": {
                    "purpose": "nested evidence for stable single-IQA selection",
                    "layout": "five-row winner-versus-runner-up dumbbell plot",
                    "statistic": "median inner-validation SROCC within each outer training partition",
                    "selection_precedence": [
                        "highest SROCC",
                        "highest PLCC",
                        "lowest RMSE",
                        "lowest median time",
                    ],
                    "displayed_cases": "DISTS and the strongest of the remaining 21 IQAs in each partition",
                    "runner_ups": [
                        "TOPIQ-FR",
                        "HaarPSI",
                        "GMSD",
                        "DSS",
                        "WADIQAM-FR",
                    ],
                    "selected_encoding": "red diamond",
                    "alternative_encoding": "primary-blue circle",
                    "coordinate_displacement": False,
                    "inference": "selection consistency only; no equivalence or significance claim",
                },
                "pointpcapp_runtime.pdf": {
                    "purpose": "seven-dataset sample-level measured complete PointPCA++ runtime and paired speedup",
                    "layout": "4x4 full-width grid; time/speedup rows for four datasets followed by time/speedup rows for three datasets and one unused column",
                    "datasets": FINAL_DATASETS,
                    "x_definition": "(REF_NUM_POINTS + TEST_NUM_POINTS) / 1000000",
                    "x_scale": "linear, beginning at zero",
                    "y_scale": "linear, beginning at zero",
                    "trend": "second-order polynomial over the observed x range",
                    "interval": "95% multilevel bootstrap CI clustered by REF; 10000 resamples; seed 42",
                    "matlab_encoding": "neutral gray circles",
                    "pointpcapp_encoding": "primary-blue triangles",
                    "speedup_encoding": "secondary-orange circles",
                    "coordinate_displacement": False,
                    "implementation_order": "PointPCA3 followed sequentially by cubemap/DISTS",
                    "timing_scope": "artifact-reported complete feature extraction after warm-up; excludes CSV serialization and regression inference",
                },
                "pointpcapp_modality_importance.pdf": {
                    "purpose": "cross-dataset grouped modality reliance of the selected fused regressor",
                    "statistic": "median across five fold-level median SROCC decreases",
                    "interval": "IQR across the five outer folds",
                    "permutations": "50 per modality and fold, restricted within held-out REF",
                    "modalities": ["DISTS (6 views)", "PointPCA3 (40 features)"],
                    "model_selection_datasets": MODEL_SELECTION_DATASETS,
                    "evaluation_only_datasets": EVALUATION_ONLY_DATASETS,
                    "coordinate_displacement": False,
                },
                "pointpcapp_all_datasets.pdf": {
                    "purpose": "directed source-to-target transfer of the frozen 46-feature PointPCA++ design",
                    "layout": "three-panel paired forest plot, one panel per training dataset",
                    "training_datasets": MODEL_SELECTION_DATASETS,
                    "test_datasets_per_panel": 6,
                    "source_row_encoding": "neutral gray band labeled TRAINING SOURCE; no self-test result",
                    "srocc_encoding": "primary-blue circle",
                    "plcc_encoding": "secondary-orange diamond",
                    "model_selection_target_encoding": "restrained warm background labeled Modelling target",
                    "evaluation_only_target_encoding": "restrained blue background labeled Evaluation target",
                    "x_limits": [0.45, 1.00],
                    "interval": "95% bootstrap interval clustered by target REF",
                    "bootstrap_repeats": 10000,
                    "bootstrap_root_seed": SEED,
                    "coordinate_displacement": "x coordinates exact; small fixed y offsets separate SROCC and PLCC",
                    "target_preprocessing_or_calibration": False,
                }
            },
            "resolved_palette": deep_hex,
            "semantic_colors": {
                "primary": deep_hex[0],
                "secondary": deep_hex[1],
                "positive": deep_hex[2],
                "selected": deep_hex[3],
                "neutral": deep_hex[7],
            },
            "packages": {
                "matplotlib": matplotlib.__version__,
                "seaborn": sns.__version__,
            },
            "renderer": str(SCRIPT.relative_to(REPO)),
            "renderer_sha256": sha256(SCRIPT),
            "numerical_artifacts_sha256": numerical_hash,
            "source_artifact_hashes": source_hashes,
            "figure_hashes": figure_hashes,
        },
    )


def render_saved_figures(
    source_run_id: str | None = None, render_mode: str = "figures-only",
    selection_only: bool = False,
) -> None:
    before_hash = numerical_artifact_hash()
    if selection_only:
        render_core3_selection_figures()
        assert numerical_artifact_hash() == before_hash
        run = core3_selection_manifest()
        write_figure_metadata(
            run["run_id"], "figures-only:selection", before_hash,
            {"regressor_comparison.pdf", "feature_set_ablation.pdf"},
        )
        print(f"Selection figures regenerated; numerical artifacts unchanged ({before_hash})")
        return
    regressor_selection = json.loads(
        (DIRS["manifests"] / "regressor_selected.json").read_text()
    )
    run = json.loads((DIRS["manifests"] / "run_manifest.json").read_text())
    importance = pd.concat(
        [
            pd.read_csv(DIRS["summaries"] / "importance_individual.csv"),
            pd.read_csv(DIRS["summaries"] / "importance_iqa_blocks.csv"),
        ],
        ignore_index=True,
    )
    generate_figures(
        pd.read_csv(DIRS["summaries"] / "projection_case_summary.csv"),
        pd.read_csv(DIRS["summaries"] / "projection_pipeline_ranks.csv"),
        pd.read_csv(DIRS["summaries"] / "regressor_shortlist_outer.csv"),
        regressor_selection["stable_shortlist"],
        regressor_selection["selected_regressor"],
        importance,
        pd.read_csv(DIRS["summaries"] / "single_iqa_outer_selection.csv"),
        pd.read_csv(DIRS["summaries"] / "feature_set_ablation_summary.csv"),
        pd.read_csv(DIRS["summaries"] / "runtime_pointpca_summary.csv"),
        pd.read_csv(DIRS["summaries"] / "runtime_scaling_summary.csv"),
        pd.read_csv(DIRS["summaries"] / "runtime_pointpcapp_summary.csv"),
        load_pointpcapp_runtime_samples(),
    )
    if (DIRS["manifests"] / "core3_selection.json").exists():
        render_core3_selection_figures()
    modality_path = (
        DIRS["summaries"] / "cross_dataset_modality_importance_summary.csv"
    )
    if modality_path.exists():
        generate_cross_dataset_modality_importance_figure(
            pd.read_csv(modality_path)
        )
    transfer_summary_path = (
        DIRS["summaries"] / "cross_dataset_transfer_summary.csv"
    )
    if transfer_summary_path.exists():
        generate_cross_dataset_transfer_figure(pd.read_csv(transfer_summary_path))
    after_hash = numerical_artifact_hash()
    if after_hash != before_hash:
        raise RuntimeError(
            "Figure-only rendering changed a numerical artifact; refusing completion"
        )
    write_figure_metadata(source_run_id or run["run_id"], render_mode, after_hash)
    print(
        f"Regenerated publication figures without changing numerical artifacts ({after_hash})",
        flush=True,
    )


def render_final_evaluation_figures(run_id: str) -> None:
    before_numerical_hash = numerical_artifact_hash()
    protected_names = [
        "pointpca3_worker_scaling.pdf",
        "projection_srocc_heatmap.pdf",
        "projection_accuracy_runtime.pdf",
        "regressor_comparison.pdf",
        "feature_importance.pdf",
        "single_iqa_selection.pdf",
        "feature_set_ablation.pdf",
    ]
    protected_hashes = {
        name: sha256(DIRS["figures"] / name) for name in protected_names
    }
    generate_pointpcapp_runtime_figure(load_pointpcapp_runtime_samples())
    generate_cross_dataset_modality_importance_figure(
        pd.read_csv(
            DIRS["summaries"]
            / "cross_dataset_modality_importance_summary.csv"
        )
    )
    generate_cross_dataset_transfer_figure(
        pd.read_csv(DIRS["summaries"] / "cross_dataset_transfer_summary.csv")
    )
    after_numerical_hash = numerical_artifact_hash()
    if after_numerical_hash != before_numerical_hash:
        raise RuntimeError(
            "Final-evaluation rendering changed a numerical artifact; refusing completion"
        )
    for name, before_hash in protected_hashes.items():
        if sha256(DIRS["figures"] / name) != before_hash:
            raise RuntimeError(
                f"Final evaluation changed protected APSIPA figure: {name}"
            )
    write_figure_metadata(
        run_id,
        "final-evaluation-only",
        after_numerical_hash,
        rendered_names={
            "pointpcapp_modality_importance.pdf",
            "pointpcapp_runtime.pdf",
            "pointpcapp_all_datasets.pdf",
        },
    )
    print(
        "Regenerated Figures 15--17 without changing "
        f"protected APSIPA figures or numerical artifacts ({after_numerical_hash})",
        flush=True,
    )


def write_final_artifact_provenance(run_id: str) -> None:
    provenance_path = DIRS["manifests"] / "artifact_provenance.csv"
    provenance = (
        pd.read_csv(provenance_path) if provenance_path.exists() else pd.DataFrame()
    )
    artifacts = {
        DIRS["tables"] / "tab_pointpcapp_all_datasets.tex": (
            "summaries/final_pointpcapp_dataset_summary.csv",
            "tab_pointpcapp_all_datasets",
        ),
        DIRS["tables"] / "runtime_pointpcapp_integrated_rows.tex": (
            "summaries/runtime_pointpcapp_integrated_summary.csv",
            "tab_pointpcapp_integrated_runtime",
        ),
        DIRS["tables"] / "experimental_results_macros.tex": (
            "summaries/runtime_pointpcapp_integrated_summary.csv",
            "results_macros",
        ),
        DIRS["tables"] / "cross_dataset_transfer_macros.tex": (
            "summaries/cross_dataset_transfer_summary.csv",
            "cross_dataset_transfer_macros",
        ),
    }
    rows = []
    for path, (sources, label) in artifacts.items():
        rows.append(
            {
                "artifact": str(path.relative_to(REPO)),
                "run_id": run_id,
                "source_artifacts": sources,
                "manuscript_label": label,
                "script": str(SCRIPT.relative_to(REPO)),
                "script_sha256": sha256(SCRIPT),
                "render_mode": "final-evaluation-only",
                "seaborn_version": sns.__version__,
                "palette": "deep",
            }
        )
    provenance = pd.concat([provenance, pd.DataFrame(rows)], ignore_index=True)
    provenance = provenance.drop_duplicates("artifact", keep="last")
    provenance.to_csv(provenance_path, index=False)


def run_cross_dataset_transfer_only() -> None:
    start = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime(start))
    protected_paths = [
        DIRS["figures"] / "feature_importance.pdf",
        DIRS["predictions"] / "final_pointpcapp_oof_predictions.csv",
        DIRS["predictions"]
        / "cross_dataset_modality_importance_repetitions.csv",
        DIRS["predictions"] / "runtime_pointpcapp_integrated_by_sample.csv",
        DIRS["summaries"] / "final_pointpcapp_dataset_summary.csv",
        DIRS["summaries"] / "core3_feature_set_summary.csv",
        DIRS["summaries"] / "importance_individual.csv",
        DIRS["tables"] / "tab_pointpcapp_all_datasets.tex",
    ]
    protected_hashes = {str(path): sha256(path) for path in protected_paths}
    normalized_path = DIRS["normalized"] / "final_pointpcapp_46.csv"
    if not normalized_path.exists():
        raise RuntimeError("missing normalized final 46-feature table")
    selection = core3_selection_manifest()
    if selection["selected_feature_set"] != "fusion_46":
        raise RuntimeError("cross-dataset transfer requires the frozen fusion_46 set")
    selected_regressor = str(selection["selected_regressor"])

    print("[1/4] Validating frozen 46-feature inputs and selection", flush=True)
    normalized = pd.read_csv(normalized_path)
    print("[2/4] Fitting three source models and predicting 18 target pairs", flush=True)
    _, summary = cross_dataset_transfer(normalized, selected_regressor)
    print("[3/4] Writing macros, Figure 17, and provenance", flush=True)
    write_cross_dataset_transfer_macros(summary)
    generate_cross_dataset_transfer_figure(summary)
    numerical_hash = numerical_artifact_hash()
    write_figure_metadata(
        run_id,
        "cross-dataset-transfer-only",
        numerical_hash,
        rendered_names={"pointpcapp_all_datasets.pdf"},
    )
    provenance_path = DIRS["manifests"] / "artifact_provenance.csv"
    provenance = pd.read_csv(provenance_path)
    macro_path = DIRS["tables"] / "cross_dataset_transfer_macros.tex"
    provenance = pd.concat(
        [
            provenance,
            pd.DataFrame(
                [
                    {
                        "artifact": str(macro_path.relative_to(REPO)),
                        "run_id": run_id,
                        "source_artifacts": "summaries/cross_dataset_transfer_summary.csv",
                        "manuscript_label": "cross_dataset_transfer_macros",
                        "script": str(SCRIPT.relative_to(REPO)),
                        "script_sha256": sha256(SCRIPT),
                        "render_mode": "cross-dataset-transfer-only",
                        "seaborn_version": sns.__version__,
                        "palette": "deep",
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("artifact", keep="last")
    provenance.to_csv(provenance_path, index=False)

    final_manifest_path = DIRS["manifests"] / "final_pointpcapp_run_manifest.json"
    final_manifest = json.loads(final_manifest_path.read_text())
    final_manifest["cross_dataset_transfer"] = {
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": MODEL_SELECTION_DATASETS,
        "targets_per_source": 6,
        "directed_pairs": 18,
        "prediction_rows": 9012,
        "target_fitting_or_calibration": False,
        "manifest": "data/analysis/manifests/cross_dataset_transfer.json",
        "script_sha256": sha256(SCRIPT),
    }
    write_json(final_manifest_path, final_manifest)

    print("[4/4] Verifying protected experimental artifacts", flush=True)
    for path in protected_paths:
        if sha256(path) != protected_hashes[str(path)]:
            raise RuntimeError(f"transfer analysis changed protected artifact: {path}")
    print(
        "Cross-dataset transfer completed with 18 directed pairs and "
        f"9,012 predictions in {(time.time() - start) / 60:.1f} minutes",
        flush=True,
    )


def run_final_evaluation_only() -> None:
    start = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime(start))
    protected_paths = [
        DIRS["manifests"] / "projection_selected_pipeline.json",
        DIRS["manifests"] / "regressor_selected.json",
        DIRS["manifests"] / "single_iqa_selected.json",
        DIRS["summaries"] / "single_iqa_outer_selection.csv",
        DIRS["summaries"] / "feature_set_ablation_summary.csv",
    ]
    protected_hashes = {str(path): sha256(path) for path in protected_paths}
    core_selection = core3_selection_manifest()
    selected_regressor = str(core_selection["selected_regressor"])
    selections = {
        "projection": json.loads(
            (DIRS["manifests"] / "projection_selected_pipeline.json").read_text()
        )["selected_pipeline"],
        "regressor": selected_regressor,
        "iqa": json.loads(
            (DIRS["manifests"] / "single_iqa_selected.json").read_text()
        )["selected_iqa"],
        "feature_set": core_selection["selected_feature_set"],
    }
    if (
        selections["projection"] != "crop_pad_navier"
        or selections["iqa"] != "DISTSMetric"
        or selections["feature_set"] != "fusion_46"
    ):
        raise RuntimeError(f"frozen PointPCA++ selections changed: {selections}")

    print("[1/10] Auditing and normalizing seven complete datasets", flush=True)
    normalized = audit_and_normalize_final_pointpcapp()
    print("[2/10] Persisting content-disjoint grouped folds", flush=True)
    folds = create_final_pointpcapp_folds(normalized)
    print("[3/10] Recomputing APSIPA-only 172-feature importance", flush=True)
    projection_selection = pd.read_csv(
        DIRS["summaries"] / "projection_outer_selection.csv"
    )
    pipeline_by_outer = {
        int(row.outer_fold): str(row.pipeline)
        for row in projection_selection[projection_selection["selected"]].itertuples()
    }
    base, projections = load_contribution_inputs(set(pipeline_by_outer.values()))
    apsipa_outer = load_persisted_outer_folds(base)
    importance_analysis(
        base, projections, apsipa_outer, pipeline_by_outer, selected_regressor
    )
    print("[4/10] Computing selected-config OOF predictions", flush=True)
    _, _, performance, evaluation_details = evaluate_final_pointpcapp(
        normalized, folds, selected_regressor
    )
    print("[5/10] Computing directed cross-dataset transfer", flush=True)
    _, transfer_summary = cross_dataset_transfer(normalized, selected_regressor)
    print("[6/10] Computing seven-dataset grouped modality importance", flush=True)
    cross_dataset_modality_importance(normalized, folds, selected_regressor)
    print("[7/10] Computing measured complete-pipeline runtime summaries", flush=True)
    _, runtime = integrated_pointpcapp_runtime(normalized)
    print("[8/10] Writing manuscript-generated tables and macros", flush=True)
    write_final_pointpcapp_tables(performance, runtime)
    write_cross_dataset_transfer_macros(transfer_summary)
    print("[9/10] Regenerating publication figures", flush=True)
    render_final_evaluation_figures(run_id)
    write_final_artifact_provenance(run_id)

    print("[10/10] Writing provenance and checking frozen artifacts", flush=True)
    for path in protected_paths:
        if sha256(path) != protected_hashes[str(path)]:
            raise RuntimeError(f"final evaluation changed frozen artifact: {path}")
    source_hashes_path = DIRS["manifests"] / "final_pointpcapp_source_hashes.csv"
    write_json(
        DIRS["manifests"] / "final_pointpcapp_run_manifest.json",
        {
            "run_id": run_id,
            "started_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)
            ),
            "finished_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "datasets": FINAL_DATASETS,
            "qomex2019_status": "pending_missing_complete_file",
            "samples": len(normalized),
            "reference_groups": int(
                sum(value[1] for value in FINAL_EXPECTED_POPULATIONS.values())
            ),
            "features": 46,
            "feature_order": final_feature_columns(),
            "selection_frozen": True,
            "selection_roles": {
                "model_selection": MODEL_SELECTION_DATASETS,
                "evaluation_only": EVALUATION_ONLY_DATASETS,
            },
            "selected_configuration": selections,
            "seed": SEED,
            "folds": "five-fold GroupKFold independently within each dataset, grouped by REF",
            "bootstrap": {
                "cluster": "REF",
                "repeats": 10000,
                "seed": SEED,
                "invalid_replicates": evaluation_details[
                    "invalid_bootstrap_replicates"
                ],
            },
            "cross_dataset_transfer": {
                "sources": MODEL_SELECTION_DATASETS,
                "targets_per_source": 6,
                "directed_pairs": 18,
                "prediction_rows": 9012,
                "manifest": "data/analysis/manifests/cross_dataset_transfer.json",
                "target_fitting_or_calibration": False,
            },
            "resolved_model_pipeline": evaluation_details["resolved_pipeline"],
            "packages": {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "lazypredict": lazypredict.__version__,
            },
            "feature_extraction_provenance": {
                "repository": "https://gitlab.com/gpds-unb/pointpcapp",
                "commit": "a64ed60788927b15ccf4a653493d8dedea5673c7",
                "execution_order": "PointPCA3 then cubemap/DISTS (sequential)",
                "timing_scope": "point counting, point-cloud loading, complete feature extraction; excludes CSV serialization and regression inference; initialization warm-up precedes recorded rows",
                "worker_count": 32,
                "worker_count_status": "author_confirmed",
                "hardware": "AMD Ryzen Threadripper 2950X, 128 GB RAM, NVIDIA GeForce RTX 5090",
                "hardware_status": "author_confirmed",
            },
            "source_hash_manifest": str(source_hashes_path.relative_to(REPO)),
            "source_hash_manifest_sha256": sha256(source_hashes_path),
            "script": str(SCRIPT.relative_to(REPO)),
            "script_sha256": sha256(SCRIPT),
            "legacy_runtime_artifacts": {
                "status": "preserved_not_used_in_manuscript",
                "prediction": "runtime_pointpcapp_by_sample.csv",
                "summary": "runtime_pointpcapp_summary.csv",
            },
            "sota_comparison_status": "deferred",
        },
    )
    print(
        f"Seven-dataset final evaluation completed in {(time.time() - start) / 60:.1f} minutes",
        flush=True,
    )


def run_iqa_contribution_only() -> None:
    start = time.time()
    print("[1/4] Validating frozen source files and selections", flush=True)
    validate_saved_source_hashes()
    projection_selection = pd.read_csv(
        DIRS["summaries"] / "projection_outer_selection.csv"
    )
    pipeline_by_outer = {
        int(row["outer_fold"]): str(row["pipeline"])
        for _, row in projection_selection[projection_selection["selected"]].iterrows()
    }
    if set(pipeline_by_outer) != set(range(5)):
        raise ValueError("missing frozen projection selection for an outer fold")
    regressor_selection = json.loads(
        (DIRS["manifests"] / "regressor_selected.json").read_text()
    )
    iqa_selection = json.loads(
        (DIRS["manifests"] / "single_iqa_selected.json").read_text()
    )
    winner = regressor_selection["selected_regressor"]
    if winner != "RandomForestRegressor":
        raise ValueError(f"frozen regressor changed unexpectedly: {winner}")
    if iqa_selection["selected_iqa"] != "DISTSMetric":
        raise ValueError("frozen IQA selection is not DISTSMetric")
    if iqa_selection.get("outer_selection_frequency") != {"DISTSMetric": 5}:
        raise ValueError("DISTS was not selected in all five frozen partitions")

    print("[2/4] Loading persisted folds and fold-specific projection pipelines", flush=True)
    base, projections = load_contribution_inputs(set(pipeline_by_outer.values()))
    outer = load_persisted_outer_folds(base)
    prediction_path = DIRS["predictions"] / "iqa_contribution_oof_predictions.csv"
    contribution_manifest_path = DIRS["manifests"] / "iqa_contribution_manifest.json"
    reuse_predictions = prediction_path.exists()
    if reuse_predictions and contribution_manifest_path.exists():
        previous = json.loads(contribution_manifest_path.read_text())
        recorded_prediction_hash = previous.get("prediction_sha256")
        recorded_inputs = previous.get("input_artifact_hashes", {})
        current_inputs = {
            "source_hashes": sha256(DIRS["manifests"] / "source_hashes.csv"),
            "folds": sha256(DIRS["folds"] / "apsipa_nested_groupkfold.csv"),
            "projection_selection": sha256(
                DIRS["summaries"] / "projection_outer_selection.csv"
            ),
            "regressor_selection": sha256(
                DIRS["manifests"] / "regressor_selected.json"
            ),
            "iqa_selection": sha256(
                DIRS["manifests"] / "single_iqa_selected.json"
            ),
        }
        if recorded_prediction_hash is not None:
            reuse_predictions = (
                recorded_prediction_hash == sha256(prediction_path)
                and recorded_inputs == current_inputs
            )
    if reuse_predictions:
        print(
            "[3/4] Reusing OOF predictions and computing clustered intervals",
            flush=True,
        )
    else:
        print("[3/4] Fitting add-one and remove-one IQA ablations", flush=True)
    summary = iqa_contribution_analysis(
        base,
        projections,
        outer,
        pipeline_by_outer,
        winner,
        reuse_predictions=reuse_predictions,
    )
    if len(summary) != 44:
        raise RuntimeError("incomplete IQA contribution summary")

    run_manifest_path = DIRS["manifests"] / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    run_manifest["iqa_contribution_extension"] = {
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "bootstrap_repeats": 10000,
        "selected_regressor": winner,
        "selected_iqa_frozen": "DISTSMetric",
        "selection_reopened": False,
        "script_sha256": sha256(SCRIPT),
    }
    write_json(run_manifest_path, run_manifest)

    print("[4/4] Regenerating figures and provenance", flush=True)
    render_saved_figures()
    print(
        f"IQA contribution analysis completed in {(time.time() - start) / 60:.1f} minutes",
        flush=True,
    )


def manifest(start_time: float, selected_pipeline=None, selected_regressor=None, selected_iqa=None):
    values = {
        "run_id": time.strftime("%Y%m%dT%H%M%S", time.gmtime(start_time)),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "lazypredict": lazypredict.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "lazy_regressors": [name for name, _ in REGRESSORS],
        "selected_pipeline": selected_pipeline,
        "selected_regressor": selected_regressor,
        "selected_iqa": selected_iqa,
        "script_sha256": sha256(SCRIPT),
    }
    write_json(DIRS["manifests"] / "run_manifest.json", values)


def main():
    start = time.time()
    print("[1/10] Auditing inputs", flush=True)
    audit()
    print("[2/10] Building normalized tables", flush=True)
    base, projections = build_normalized()
    print("[3/10] Persisting grouped folds", flush=True)
    outer, inner_by_outer = create_folds(base)
    print("[4/10] Comparing projection pipelines", flush=True)
    selected_pipeline, projection_selection = projection_analysis(
        base, projections, outer, inner_by_outer
    )
    print(f"Selected projection pipeline: {selected_pipeline}", flush=True)
    global_fusion = build_fusion(base, projections, selected_pipeline, save=True)
    print("[5/10] Screening regressors", flush=True)
    winner, pipeline_by_outer, regressor_outer, stable = regressor_analysis(
        base, projections, projection_selection, outer, inner_by_outer
    )
    print(f"Selected regressor: {winner}", flush=True)
    print("[6/10] Computing held-out permutation importance", flush=True)
    importance = importance_analysis(base, projections, outer, pipeline_by_outer, winner)
    print("[7/10] Selecting exactly one IQA and evaluating ablations", flush=True)
    selected_iqa, ablation = single_iqa_and_ablation(
        base,
        projections,
        outer,
        inner_by_outer,
        pipeline_by_outer,
        selected_pipeline,
        winner,
    )
    print(f"Selected IQA: {selected_iqa}", flush=True)
    print("[8/10] Computing IQA add-one and remove-one contributions", flush=True)
    iqa_contribution_analysis(
        base, projections, outer, pipeline_by_outer, winner
    )
    print("[9/10] Computing runtime summaries", flush=True)
    runtime_summary, scaling_summary, complete_runtime = runtime_analysis(
        selected_pipeline, selected_iqa
    )
    print("[10/10] Generating figures", flush=True)
    projection_summary = pd.read_csv(DIRS["summaries"] / "projection_case_summary.csv")
    projection_ranks = pd.read_csv(DIRS["summaries"] / "projection_pipeline_ranks.csv")
    generate_figures(
        projection_summary,
        projection_ranks,
        regressor_outer,
        stable,
        winner,
        importance,
        pd.read_csv(DIRS["summaries"] / "single_iqa_outer_selection.csv"),
        ablation,
        runtime_summary,
        scaling_summary,
        complete_runtime,
        load_pointpcapp_runtime_samples(),
    )
    run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime(start))
    write_figure_metadata(run_id, "full-analysis", numerical_artifact_hash())
    manifest(start, selected_pipeline, winner, selected_iqa)
    (DIRS["logs"] / "failure.log").unlink(missing_ok=True)
    print(f"Analysis completed in {(time.time() - start) / 60:.1f} minutes", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-figures-only", action="store_true",
                        help="With --figures-only, render only regressor and feature comparisons")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures from saved artifacts without recomputing analyses",
    )
    modes.add_argument(
        "--iqa-contribution-only",
        action="store_true",
        help="Run only the frozen add-one/remove-one IQA contribution analysis",
    )
    modes.add_argument(
        "--final-evaluation-only",
        action="store_true",
        help="Evaluate the frozen 46-feature PointPCA++ configuration on the seven complete datasets",
    )
    modes.add_argument(
        "--core-reselection-only",
        action="store_true",
        help="Reselect the regressor and feature set across APSIPA, WPC, and LS-PCQA",
    )
    modes.add_argument(
        "--cross-dataset-transfer-only",
        action="store_true",
        help="Fit the three frozen source models and evaluate all 18 directed transfer pairs",
    )
    args = parser.parse_args()
    if args.selection_figures_only and not args.figures_only:
        parser.error("--selection-figures-only requires --figures-only")
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.figures_only:
            render_saved_figures(selection_only=args.selection_figures_only)
        elif args.iqa_contribution_only:
            run_iqa_contribution_only()
        elif args.final_evaluation_only:
            run_final_evaluation_only()
        elif args.core_reselection_only:
            run_core_reselection_only()
        elif args.cross_dataset_transfer_only:
            run_cross_dataset_transfer_only()
        else:
            main()
    except Exception:
        (DIRS["logs"] / "failure.log").write_text(traceback.format_exc())
        raise
