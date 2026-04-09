#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pickle
import logging
import datetime
import pandas as pd
import numpy as np

# ============================== Config ==============================
SUBJ_COL = "subj"
FOLD_KEYS = ["fold"]
IMG_KEYS = ["int16gz"]
SUBJ_ZFILL = 6

FEATURES_TYPE_DATA = {
    "qc": "cls",
    "class": "cls",
}

USED_FEATURES_LIST = ["qc"]

OUT_ROOT = "data/splits"
ENABLE_LOGGER = True

BATCH_SEARCH_ROOT = "data/splits/"
BATCH_SUBDIR = "random10"
BATCH_GLOB = "**/*.csv"
# ==================================================================


def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def save_pkl(data, path: str):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def get_logger(path: str):
    logger = logging.getLogger(path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh = logging.FileHandler(path, mode="w")
    ch = logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log(msg: str, logger=None):
    logger.info(msg) if logger else print(msg)


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if SUBJ_COL not in df.columns:
        df = df.rename(columns={df.columns[0]: SUBJ_COL})
    df[SUBJ_COL] = df[SUBJ_COL].astype(str).str.zfill(SUBJ_ZFILL)
    return df


def get_used_features(df: pd.DataFrame):
    if USED_FEATURES_LIST is None:
        return [f for f in FEATURES_TYPE_DATA if f in df.columns]
    return [f for f in USED_FEATURES_LIST if f in df.columns]


def encode_label(series: pd.Series):
    vals = sorted(series.dropna().unique())
    mapping = {i: raw for i, raw in enumerate(vals)}
    reverse = {raw: i for i, raw in mapping.items()}
    return series.map(reverse).astype(np.int32), mapping


def build_cv(csv_file: str, out_dir: str, fold_key: str, img_key: str, logger=None):
    df = read_csv(csv_file)

    used_features = get_used_features(df)
    if not used_features:
        log("No valid features found.", logger)
        return

    if fold_key not in df.columns:
        raise KeyError(f"Missing fold column: {fold_key}")
    if img_key not in df.columns:
        raise KeyError(f"Missing image column: {img_key}")

    for feat in used_features:
        log(f"Processing: {feat}", logger)

        used_df = df[[SUBJ_COL, fold_key, img_key, feat]].dropna().copy()

        if FEATURES_TYPE_DATA.get(feat, "cls") == "cls":
            used_df["label"], mapping = encode_label(used_df[feat])
            log(f"Label mapping: {mapping}", logger)
        else:
            used_df["label"] = used_df[feat]

        feat_out = os.path.join(out_dir, feat)
        mkdir(feat_out)

        csv_path = os.path.join(feat_out, f"{feat}_CV.csv")
        used_df.to_csv(csv_path, index=False)
        log(f"CSV saved: {csv_path}", logger)

        label_dic = {}
        for fold_id, part in used_df.groupby(fold_key):
            label_dic[fold_id] = {
                row[SUBJ_COL]: {
                    "img": [row[img_key]],
                    "mask": [],
                    "label": row["label"],
                    "raw_label": row[feat],
                    "name": row[SUBJ_COL],
                }
                for _, row in part.iterrows()
            }

        pkl_path = os.path.join(feat_out, "label_3d_CV.pkl")
        save_pkl(label_dic, pkl_path)
        log(f"PKL saved: {pkl_path}", logger)


def main():
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    search_dir = os.path.join(BATCH_SEARCH_ROOT, BATCH_SUBDIR)
    csv_list = sorted(
        p for p in glob.glob(os.path.join(search_dir, BATCH_GLOB), recursive=True)
        if os.path.isfile(p)
    )

    if not csv_list:
        raise FileNotFoundError(f"No CSV matched: {search_dir}/{BATCH_GLOB}")

    print(f"[Batch] Will process {len(csv_list)} CSV files:")
    for p in csv_list:
        print("  -", p)

    for csv_path in csv_list:
        rel_dir = os.path.relpath(os.path.dirname(csv_path), start=BATCH_SEARCH_ROOT)
        cv_stem = os.path.splitext(os.path.basename(csv_path))[0]

        for img_key in IMG_KEYS:
            for fold_key in FOLD_KEYS:
                out_dir = os.path.join(OUT_ROOT, rel_dir, f"{cv_stem}_{img_key}", fold_key)
                mkdir(out_dir)

                logger = get_logger(os.path.join(out_dir, f"cv_{now}.log")) if ENABLE_LOGGER else None
                log(f"Build: file={os.path.basename(csv_path)}, img_key={img_key}, fold_key={fold_key}", logger)

                build_cv(
                    csv_file=csv_path,
                    out_dir=out_dir,
                    fold_key=fold_key,
                    img_key=img_key,
                    logger=logger,
                )

                log(f"Done: {out_dir}\n", logger)


if __name__ == "__main__":
    main()