#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import pandas as pd
from pathlib import Path
from collections import defaultdict

# =========================
# ===== Top settings ======
# =========================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input root: recursively search prediction CSVs
BASE_DIR = Path("results/transfer/prediction/cohort_b")

# Output summary CSV
MERGE_OUT_PATH = Path("results/transfer/predict/cohort_b_ensemble_prediction.csv")

# Fusion method: 'mean' | 'median' | 'gmean' | 'logit_mean'
FUSE_METHOD = "mean"

# Decision thresholds
THRESHOLDS = [0.5]

# Allowed pretrained-model patterns
PT_PATTERNS = ("cohort_a_aug_bal", "cohort_a_aug_imb", "cohort_a_noaug")

# Optional external label table; must contain subj and label-like columns
LABEL_TABLE = None

# Supported prediction CSV names
FILENAME_CANDIDATES = ("Case_PredictLable.csv", "Case_PredictLabel.csv")

# Candidate column names
ID_CANDIDATES    = ["subj", "case", "Case", "subject", "id", "ID", "name", "fname", "file", "image", "path"]
PROB_CANDIDATES  = ["prob_mean", "prob", "prob1", "p1", "pred_prob", "predict_proba", "y_pred_proba", "prob_1", "probability"]
LABEL_CANDIDATES = ["label", "qc", "y", "y_true", "target", "truth", "gt", "gt_label", "qc_label", "true_label"]

# Decimal rounding; <0 means keep original precision
DECIMALS = -2

# String label to 0/1 mapping
LABEL_STR_MAP = {
    "pass": 0, "fail": 1, "PASS": 0, "FAIL": 1,
    "qc0": 0, "qc1": 1, "QC0": 0, "QC1": 1,
    "ok": 0, "bad": 1, "0": 0, "1": 1
}

# =========================
# ======= Parsers =========
# =========================
# Capture optional variant suffix such as _a1/_a2
GRAND_RE = re.compile(
    r"^(?P<prefix>.+)_fold_(?P<fold>\d+)_r(?P<run>\d{2})(?P<variant>_[^/]+)?$",
    re.IGNORECASE,
)

def parse_from_grandparent(grand_name: str):
    m = GRAND_RE.match(grand_name)
    if not m:
        return None
    prefix  = m.group("prefix")
    fold    = int(m.group("fold"))
    run     = m.group("run")
    variant = m.group("variant") or ""
    return prefix, fold, run, variant

def contains_any_pt(prefix: str) -> bool:
    return any(pt in prefix for pt in PT_PATTERNS) if PT_PATTERNS else True

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def to_subj_series(df: pd.DataFrame, id_col: str) -> pd.Series:
    s = df[id_col].astype(str)
    if "path" in id_col.lower():
        s = s.apply(lambda p: os.path.basename(p))
    return s.rename("subj")

def coerce_label(series: pd.Series):
    s = series.apply(lambda x: LABEL_STR_MAP.get(str(x), str(x)))
    s = pd.to_numeric(s, errors="coerce")
    s = s.apply(lambda v: int(v) if pd.notna(v) and v in (0, 1) else (0 if v == 0 else (1 if v == 1 else math.nan)))
    return s

def fuse_probs(values, method="mean"):
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return math.nan
    if method == "mean":
        return sum(vals) / len(vals)
    if method == "median":
        return float(pd.Series(vals).median())
    if method == "gmean":
        eps = 1e-8
        prod = 1.0
        for v in vals:
            v_ = max(min(v, 1 - 1e-8), eps)
            prod *= v_
        return prod ** (1.0 / len(vals))
    if method == "logit_mean":
        eps = 1e-8
        logits = []
        for v in vals:
            v_ = max(min(v, 1 - 1e-8), eps)
            logits.append(math.log(v_ / (1.0 - v_)))
        return 1.0 / (1.0 + math.exp(-sum(logits) / len(logits)))
    return sum(vals) / len(vals)

def auc_from_scores(y_true, scores):
    """Rank-based AUC without sklearn; y_true must be in {0,1}."""
    import numpy as np
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float)
    scores = pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy(dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(scores)
    y_true = y_true[mask]
    scores = scores[mask]
    if len(y_true) == 0:
        return math.nan
    pos = y_true == 1
    neg = y_true == 0
    n1 = int(pos.sum())
    n0 = int(neg.sum())
    if n1 == 0 or n0 == 0:
        return math.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    sum_pos_ranks = ranks[pos].sum()
    auc = (sum_pos_ranks - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc)

def safe_div(a, b):
    return float(a) / float(b) if b else math.nan

def confusion_and_metrics(y_true, y_pred, y_score=None):
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce")
    y_pred = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = safe_div(TP + TN, TP + TN + FP + FN)
    sen = safe_div(TP, TP + FN)
    spe = safe_div(TN, TN + FP)
    pre = safe_div(TP, TP + FP)
    f1 = safe_div(2 * pre * sen, pre + sen) if not math.isnan(pre) and not math.isnan(sen) and (pre + sen) > 0 else math.nan
    bacc = (sen + spe) / 2 if not math.isnan(sen) and not math.isnan(spe) else math.nan
    auc = math.nan
    if y_score is not None:
        auc = auc_from_scores(y_true, pd.Series(y_score)[mask])
    return {
        "n": int(TP + TN + FP + FN),
        "pos": int((y_true == 1).sum()),
        "neg": int((y_true == 0).sum()),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "acc": acc, "sen": sen, "spe": spe, "pre": pre, "f1": f1, "bacc": bacc, "auc": auc
    }

def fmt3(v):
    """Round only when DECIMALS >= 0; otherwise keep original float precision."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    try:
        fv = float(v)
    except Exception:
        return v
    if isinstance(DECIMALS, int) and DECIMALS >= 0:
        return round(fv, DECIMALS)
    return fv

# =========================
# ========= Main ==========
# =========================
def main():
    # 1) Collect candidate CSV files
    found = []
    for fname in FILENAME_CANDIDATES:
        found.extend(BASE_DIR.rglob(fname))
    found = sorted({p.resolve() for p in found if p.is_file()})

    if not found:
        print(f"[WARN] No Case_PredictLable/Label.csv found under {BASE_DIR}")
        return

    # 2) Optional external label table
    ext_lab = None
    if LABEL_TABLE is not None:
        lt = Path(LABEL_TABLE)
        if lt.exists():
            ext = pd.read_csv(lt)
            idc = find_col(ext, ID_CANDIDATES)
            labc = find_col(ext, LABEL_CANDIDATES)
            if idc is None or labc is None:
                raise ValueError("External label table must contain subj and label-like columns.")
            ext_lab = pd.DataFrame({"subj": to_subj_series(ext, idc), "label": coerce_label(ext[labc])})
            ext_lab = ext_lab.dropna(subset=["label"]).groupby("subj", as_index=False).agg({"label": "first"})

    # 3) Build groups: key = f"{prefix}_r{run}{variant}"
    groups = defaultdict(list)
    skipped = []

    for src in found:
        try:
            grand = src.parent.parent.name
        except Exception:
            skipped.append(src.as_posix())
            continue

        parsed = parse_from_grandparent(grand)
        if not parsed:
            skipped.append(src.as_posix())
            continue

        prefix, fold, run, variant = parsed
        if not contains_any_pt(prefix):
            skipped.append(src.as_posix())
            continue

        key = f"{prefix}_r{run}{variant}"
        groups[key].append((fold, src, grand))

    if not groups:
        print("[WARN] No valid groups parsed; check directory naming and PT_PATTERNS.")
        if skipped:
            print("[HINT] Skipped paths (showing up to 10):")
            for s in skipped[:10]:
                print("   -", s)
        return

    # 4) Fuse each group and compute metrics for each threshold
    merge_rows = []

    for key, items in sorted(groups.items()):
        items = sorted(items, key=lambda x: x[0])

        fold_frames = []
        label_frames = []
        folds_list = []
        fold_set = set()

        for fold, fpath, grand in items:
            df = pd.read_csv(fpath)

            id_col = find_col(df, ID_CANDIDATES)
            prob_col = find_col(df, PROB_CANDIDATES)
            label_col = find_col(df, LABEL_CANDIDATES)

            if id_col is None:
                print(f"[SKIP] {fpath.name}: no ID column found")
                continue

            # Fallback for prob_1 / prob1
            if prob_col is None:
                regex_cands = [c for c in df.columns if re.fullmatch(r"prob[_\-\s]?1", c, flags=re.IGNORECASE)]
                if regex_cands:
                    prob_col = regex_cands[0]
                else:
                    print(f"[SKIP] {fpath.name}: no probability column found")
                    continue

            subj = to_subj_series(df, id_col)
            prob = pd.to_numeric(df[prob_col], errors="coerce")

            tmp = pd.DataFrame({"subj": subj, f"prob_fold_{fold}": prob})
            tmp = tmp.groupby("subj", as_index=False).mean(numeric_only=True)
            fold_frames.append(tmp)

            folds_list.append(str(fold))
            fold_set.add(fold)

            if label_col is not None:
                lab = coerce_label(df[label_col])
                labdf = pd.DataFrame({"subj": subj, "label": lab})
                labdf = labdf.dropna(subset=["label"]).groupby("subj", as_index=False).agg({"label": "first"})
                label_frames.append(labdf)

        if not fold_frames:
            continue

        # Outer-join probabilities by subject
        merged = fold_frames[0]
        for add_df in fold_frames[1:]:
            merged = merged.merge(add_df, on="subj", how="outer")

        # Collapse duplicate prob columns for the same fold if they appear
        prob_cols = []
        for f in sorted(fold_set):
            base = f"prob_fold_{f}"
            cands = [c for c in merged.columns if (c == base) or c.startswith(base + "_")]
            if not cands:
                continue
            merged[cands] = merged[cands].apply(pd.to_numeric, errors="coerce")
            merged[base] = merged[cands].mean(axis=1, skipna=True)
            for c in cands:
                if c != base:
                    merged.drop(columns=c, inplace=True, errors="ignore")
            prob_cols.append(base)

        # Merge labels: prefer labels from prediction files, otherwise external labels
        if label_frames:
            lab_all = pd.concat(label_frames, ignore_index=True).dropna(subset=["label"])
            label_df = lab_all.groupby("subj", as_index=False)["label"].agg(lambda s: s.value_counts().index[0])
        elif ext_lab is not None:
            label_df = ext_lab.copy()
        else:
            label_df = None

        if label_df is not None:
            merged = merged.merge(label_df, on="subj", how="left")

        # Fuse probabilities once, then evaluate with each threshold
        if prob_cols:
            merged["prob_mean"] = merged[prob_cols].apply(lambda r: fuse_probs(r.values, FUSE_METHOD), axis=1)
            merged["n_folds"] = merged[prob_cols].notna().sum(axis=1).astype(int)
        else:
            merged["prob_mean"] = math.nan
            merged["n_folds"] = 0

        has_label = "label" in merged.columns
        auc_value = math.nan
        if has_label:
            auc_value = auc_from_scores(merged["label"], merged["prob_mean"])

        # Extract pt/run for summary columns
        m_pt = re.search(r"(UKB_aug(?:1|12)_r[12]|UKB_noaug)", key)
        m_run = re.search(r"(r\d{2})", key)
        pt = m_pt.group(1) if m_pt else "N/A"
        run = m_run.group(1) if m_run else "N/A"

        for thr in THRESHOLDS:
            pred_thr = (merged["prob_mean"] >= thr).astype("Int64")
            metrics = {
                "n": 0, "pos": 0, "neg": 0, "TP": 0, "FP": 0, "FN": 0, "TN": 0,
                "acc": math.nan, "sen": math.nan, "spe": math.nan, "pre": math.nan,
                "f1": math.nan, "bacc": math.nan, "auc": auc_value
            }
            if has_label:
                m2 = confusion_and_metrics(merged["label"], pred_thr, merged["prob_mean"])
                m2["auc"] = auc_value
                metrics = m2

            row = {
                "type": key,
                "pt": pt,
                "run": run,
                "n_files": len(items),
                "folds": ",".join(str(x) for x in sorted({int(f) for f in folds_list})),
                "fuse_method": FUSE_METHOD,
                "has_label": "yes" if has_label else "no",
                "THRESHOLD": float(thr),
                "n": metrics["n"], "pos": metrics["pos"], "neg": metrics["neg"],
                "TP": metrics["TP"], "FP": metrics["FP"], "FN": metrics["FN"], "TN": metrics["TN"],
                "acc": fmt3(metrics["acc"]),
                "sen": fmt3(metrics["sen"]),
                "spe": fmt3(metrics["spe"]),
                "pre": fmt3(metrics["pre"]),
                "f1": fmt3(metrics["f1"]),
                "bacc": fmt3(metrics["bacc"]),
                "auc": fmt3(metrics["auc"]),
            }
            merge_rows.append(row)

        print(f"[OK] Group done: {key}  folds=[{','.join(sorted({str(x) for x in folds_list}))}]  has_label={'yes' if has_label else 'no'}  thresholds={THRESHOLDS}")

    # Write merged summary only
    MERGE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if merge_rows:
        out_df = pd.DataFrame(merge_rows)
        out_df = out_df.sort_values(by=["pt", "run", "type", "THRESHOLD"]).reset_index(drop=True)

        # Swap TP/TN column positions only; values stay unchanged by default
        SWAP_TP_TN_VALUES = False
        cols = list(out_df.columns)
        if "TN" in cols and "TP" in cols:
            i_tn, i_tp = cols.index("TN"), cols.index("TP")
            cols[i_tn], cols[i_tp] = cols[i_tp], cols[i_tn]
            out_df = out_df[cols]
            if SWAP_TP_TN_VALUES:
                out_df[["TN", "TP"]] = out_df[["TP", "TN"]].to_numpy()

        if isinstance(DECIMALS, int) and DECIMALS >= 0:
            out_df.to_csv(MERGE_OUT_PATH, index=False, encoding="utf-8", float_format=f"%.{DECIMALS}f")
        else:
            out_df.to_csv(MERGE_OUT_PATH, index=False, encoding="utf-8")

        print(f"\n=== Done ===\nSummary: {MERGE_OUT_PATH}")
    else:
        print("[WARN] No merged metrics were generated.")

if __name__ == "__main__":
    main()