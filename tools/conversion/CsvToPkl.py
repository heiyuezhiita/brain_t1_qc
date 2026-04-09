#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import logging
import datetime
import pandas as pd
import numpy as np

# ============================== Config ==============================
LABEL_FILE_PATH = "data/splits/transfer_cohort_b_syn100.csv"
FOLD_FILE_PATH = "data/splits/transfer_cohort_b_syn100.csv"
IMG_INFO_PATH = "data/splits/transfer_cohort_b_syn100.csv"

SUBJ_COL = "subj"
FOLD_KEYS = ["fold"]
IMG_KEYS = ["int16gz"]
SUBJ_ZFILL = 6

USED_FEATURES_LIST = ["qc"]

# Feature type:
# cls -> categorical label, encoded as 0,1,2,...
# reg -> continuous value, kept as is
FEATURE_TYPE_MAP = {
    "qc": "cls"
}

OUT_ROOT = "data/splits"

ENABLE_LOGGER = True
# ==================================================================


def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def get_logger(log_path: str):
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    ch = logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log(msg: str, logger=None):
    if logger:
        logger.info(msg)
    else:
        print(msg)


def read_and_format_csv(path: str, subj_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if subj_col not in df.columns:
        df = df.rename(columns={df.columns[0]: subj_col})
    df[subj_col] = df[subj_col].astype(str).str.zfill(SUBJ_ZFILL)
    return df


def encode_label(series: pd.Series):
    uniques = sorted(series.dropna().unique())
    mapping = {i: raw for i, raw in enumerate(uniques)}
    reverse_mapping = {raw: i for i, raw in mapping.items()}
    encoded = series.map(reverse_mapping).astype(np.int32)
    return encoded, mapping


def get_used_features(label_df: pd.DataFrame):
    if USED_FEATURES_LIST is None:
        return [f for f in FEATURE_TYPE_MAP if f in label_df.columns]
    return [f for f in USED_FEATURES_LIST if f in label_df.columns]


def build_cv(label_file, fold_file, img_file, out_dir, fold_key, img_key, logger=None):
    label_df = read_and_format_csv(label_file, SUBJ_COL)
    fold_df = read_and_format_csv(fold_file, SUBJ_COL)
    img_df = read_and_format_csv(img_file, SUBJ_COL)

    used_features = get_used_features(label_df)
    if not used_features:
        log("No valid features found.", logger)
        return

    if fold_key not in fold_df.columns:
        raise KeyError(f"Missing fold column: {fold_key}")
    if img_key not in img_df.columns:
        raise KeyError(f"Missing image column: {img_key}")

    label_df = label_df.drop(columns=[fold_key], errors="ignore")
    label_df = label_df[[SUBJ_COL] + used_features]
    fold_df = fold_df[[SUBJ_COL, fold_key]]

    img_df = img_df.drop(columns=[c for c in [fold_key] + used_features if c in img_df.columns], errors="ignore")

    df = img_df.merge(label_df, on=SUBJ_COL).merge(fold_df, on=SUBJ_COL)
    log(f"Merged: {df.shape[0]} rows, {df.shape[1]} columns", logger)

    for feat in used_features:
        log(f"Processing feature: {feat}", logger)

        used_df = df[[SUBJ_COL, fold_key, img_key, feat]].dropna().copy()

        feat_type = FEATURE_TYPE_MAP.get(feat, "cls")
        if feat_type == "cls":
            used_df["label"], mapping = encode_label(used_df[feat])
            log(f"Label mapping: {mapping}", logger)
        else:
            used_df["label"] = used_df[feat]

        feat_out_dir = os.path.join(out_dir, feat)
        mkdir(feat_out_dir)

        csv_path = os.path.join(feat_out_dir, f"{feat}_CV.csv")
        used_df.to_csv(csv_path, index=False)
        log(f"CSV saved: {csv_path}", logger)

        label_dic = {}
        for fold_id, fold_part in used_df.groupby(fold_key):
            label_dic[fold_id] = {}
            for _, row in fold_part.iterrows():
                label_dic[fold_id][row[SUBJ_COL]] = {
                    "img": [row[img_key]],
                    "mask": [],
                    "label": row["label"],
                    "raw_label": row[feat],
                    "name": row[SUBJ_COL],
                }

        pkl_path = os.path.join(feat_out_dir, "label_3d_CV.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(label_dic, f)
        log(f"PKL saved: {pkl_path}", logger)


def main():
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use input filename stem as output subfolder name
    input_stem = os.path.splitext(os.path.basename(LABEL_FILE_PATH))[0]

    for img_key in IMG_KEYS:
        for fold_key in FOLD_KEYS:
            out_dir = os.path.join(OUT_ROOT, input_stem, fold_key)
            mkdir(out_dir)

            logger = None
            if ENABLE_LOGGER:
                logger = get_logger(os.path.join(out_dir, f"cv_{now}.log"))

            log(f"Start: img_key={img_key}, fold_key={fold_key}", logger)

            build_cv(
                label_file=LABEL_FILE_PATH,
                fold_file=FOLD_FILE_PATH,
                img_file=IMG_INFO_PATH,
                out_dir=out_dir,
                fold_key=fold_key,
                img_key=img_key,
                logger=logger,
            )

            log(f"Done: {out_dir}", logger)


if __name__ == "__main__":
    main()