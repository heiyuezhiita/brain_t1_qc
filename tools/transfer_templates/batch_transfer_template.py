#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Batch-generate transfer configs.
# Supports:
# - scanning multiple fold_split_dir base paths
# - dynamic syn prefix from base_dir
# - appending aug suffix (_a1 / _a2) after rXX in file stem

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
import copy
import os
import re
import glob

# === YAML setup ===
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

# Dump selected floats as scientific notation; keep 0.0 as 0.0
class SciFloat(float):
    """Dump float as %.1e in YAML, except 0.0 -> '0.0'."""
    pass

def _repr_scifloat(dumper, value):
    if float(value) == 0.0:
        txt = "0.0"
    else:
        txt = f"{float(value):.1e}"
    return dumper.represent_scalar("tag:yaml.org,2002:float", txt)

yaml.representer.add_representer(SciFloat, _repr_scifloat)

# === Base config and output template ===
BASE = "checkpoints/transfer_yaml_template/model/cohort_b/cohort_b_syn100_model.yaml"
OUT_TMPL = "configs/transfer/batch_transfer/cohort_b/{stem}.yaml"

# Output root
OUT_DIR_ROOT = "results/transfer/model/cohort_b/"

# === Stem prefix settings ===
PROJECT_TAG = "cohort_b"
SYN_DETECT_FROM_BASE = True
SYN_DEFAULT = ""
STEM_PREFIX_TEMPLATE = "{project}_{syn}"

# === fold_split_dir scan settings ===
# Supported forms:
# 1) brace expansion: .../{syn100}/{aug1,aug12}/cohort_b
# 2) glob:            .../syn*/aug*/cohort_b
# 3) list:            [".../syn100/aug1/cohort_b", ".../syn100/aug12/cohort_b"]
SET_FOLD_SPLIT_DIR = True
FOLD_SPLIT_BASES = "data/splits/random10/{syn100}/{aug}/cohort_b"
FOLD_SPLIT_SUFFIX = "fold"
AUTO_SCAN_STEMS = True
STEM_NAME_SUFFIX_FILTER = "_int16gz"

# === Pretrained model root ===
PT_MODEL_ROOT = "checkpoints/pretrained_model/cohort_a_best"

# === Dimension switches ===
USE = {
    "lr": False,
    "blocks": False,
    "wd": False,
    "es": False,
    "max_epoch": False,
    "pt_ver": True,
    "pt_fold": True,
    "fold_split_dir": True if SET_FOLD_SPLIT_DIR else False,
}

# === Full option maps ===
lr_map = {"le5": 1e-5, "le4": 1e-4, "le3": 1e-3, "3e4": 3e-4, "3e5": 3e-5}
block_sets = {
    "seg": ["conv_seg"],
    "seg_l4": ["conv_seg", "layer4"],
    "seg_l4_l3": ["conv_seg", "layer4", "layer3"],
}
wd_map = {"1e-5": 1e-5, "1e-6": 1e-6, "0": 0.0}
es_map = {"10": 10, "15": 15, "25": 25}
me_map = {"10": 10, "20": 20, "5": 5}

# Pretrained version and fold
pt_ver_map = {
    "cohort_a_r1": "cohort_a_aug_bal",
    "cohort_a_r2": "cohort_a_aug_imb",
    "cohort_a_noaug": "cohort_a_noaug",
}
pt_fold_map = {f"f{k}": k for k in range(5)}  # f0..f4 -> 0..4

def build_pt_path(ver_key: str, fold_key: str) -> str:
    """Build pretrained model path from version key and fold key."""
    ver_dir = pt_ver_map[ver_key]
    fold_id = pt_fold_map[fold_key]
    return os.path.join(PT_MODEL_ROOT, ver_dir, f"fold_{fold_id}")

# === Optional subsets; None means use all ===
SUBSET = {
    "lr": None,
    "blocks": ["seg"],
    "wd": ["0"],
    "es": ["10", "15"],
    "max_epoch": ["10", "5"],
    "pt_ver": None,
    "pt_fold": None,
}

# === Load base config ===
with open(BASE, "r") as f:
    base_cfg = yaml.load(f)

# ---------------- Helpers ----------------
def pick_keys(all_dict, use_flag, subset):
    if not use_flag:
        return [None]
    if subset:
        for k in subset:
            if k not in all_dict:
                raise ValueError(f"subset key '{k}' not in {list(all_dict.keys())}")
        return list(subset)
    return list(all_dict.keys())

def add_tag(parts, prefix, key_str):
    if key_str is None or key_str == "":
        return parts
    return parts + [f"{prefix}{key_str}"] if prefix else parts + [key_str]

def apply_fold_split_dir(cfg: dict, base: str, stem: str, suffix: str = "fold") -> str:
    """Write fold_split_dir into cfg and return a path ending with '/'."""
    path = os.path.join(base, stem, suffix)
    if not path.endswith(os.sep):
        path += os.sep
    if isinstance(cfg.get("data"), dict):
        cfg["data"]["fold_split_dir"] = path
    elif isinstance(cfg.get("train"), dict):
        cfg["train"]["fold_split_dir"] = path
    else:
        cfg["fold_split_dir"] = path
    return path

def brace_expand(pattern: str):
    """Recursively expand braces, e.g. /a/{b,c}/d -> [/a/b/d, /a/c/d]."""
    m = re.search(r"\{([^{}]+)\}", pattern)
    if not m:
        return [pattern]
    head, body, tail = pattern[:m.start()], m.group(1), pattern[m.end():]
    out = []
    for choice in body.split(","):
        out.extend(brace_expand(head + choice + tail))
    return out

def expand_base_specs(specs):
    """Expand FOLD_SPLIT_BASES from string/list with braces and glob."""
    if isinstance(specs, (str, os.PathLike)):
        specs = [str(specs)]
    paths = set()
    for s in specs:
        for pat in brace_expand(s):
            if any(ch in pat for ch in "*?[]"):
                for p in glob.glob(pat, recursive=True):
                    if os.path.isdir(p):
                        paths.add(os.path.normpath(p))
            else:
                if os.path.isdir(pat):
                    paths.add(os.path.normpath(pat))
    return sorted(paths)

def discover_fold_stems(base_dir: str, name_suffix_filter: str = "_int16gz"):
    """
    Scan one level under base_dir and keep directories that:
    1) are directories
    2) contain a 'fold' subdirectory
    3) match name_suffix_filter if provided

    Return: [(stem_name, run_suffix), ...]
    """
    stems = []
    if not os.path.isdir(base_dir):
        return stems
    for ent in os.scandir(base_dir):
        if not ent.is_dir():
            continue
        name = ent.name
        if name_suffix_filter and (name_suffix_filter not in name):
            continue
        if not os.path.isdir(os.path.join(base_dir, name, FOLD_SPLIT_SUFFIX)):
            continue
        m = re.search(r"_(r\d{2,})", name)
        run_suffix = m.group(1) if m else None
        stems.append((name, run_suffix))
    stems.sort(key=lambda x: x[0])
    return stems

def discover_fold_stems_all(base_dirs, name_suffix_filter="_int16gz"):
    """Scan all base dirs and return [(base_dir, stem_name, run_suffix), ...]."""
    results = []
    for base in base_dirs:
        for stem_name, run_suffix in discover_fold_stems(base, name_suffix_filter):
            results.append((base, stem_name, run_suffix))
    results.sort(key=lambda x: (x[0], x[1]))
    return results

def extract_syn_tag(path: str):
    """Extract syn tag like syn50/syn100/syn200 from path."""
    if not path:
        return None
    for token in os.path.normpath(path).split(os.sep):
        if re.fullmatch(r"syn\d+", token):
            return token
    return None

def extract_aug_suffix(path: str) -> str:
    """
    Check path tokens:
    aug1  -> a1
    aug12 -> a2
    else  -> ''
    """
    if not path:
        return ""
    tokens = os.path.normpath(path).split(os.sep)
    if any(tok == "aug12" for tok in tokens):
        return "a2"
    if any(tok == "aug1" for tok in tokens):
        return "a1"
    return ""

# ---------------- Active keys ----------------
lr_keys = pick_keys(lr_map, USE["lr"], SUBSET["lr"])
blk_keys = pick_keys(block_sets, USE["blocks"], SUBSET["blocks"])
wd_keys = pick_keys(wd_map, USE["wd"], SUBSET["wd"])
es_keys = pick_keys(es_map, USE["es"], SUBSET["es"])
me_keys = pick_keys(me_map, USE["max_epoch"], SUBSET["max_epoch"])
ver_keys = pick_keys(pt_ver_map, USE["pt_ver"], SUBSET["pt_ver"])
fold_keys = pick_keys(pt_fold_map, USE["pt_fold"], SUBSET["pt_fold"])

os.makedirs(os.path.dirname(OUT_TMPL), exist_ok=True)

# === Auto-discover fold stems ===
if SET_FOLD_SPLIT_DIR and AUTO_SCAN_STEMS:
    base_dirs = expand_base_specs(FOLD_SPLIT_BASES)
    if not base_dirs:
        raise FileNotFoundError(f"[fold_split_dir] No base dir found: {FOLD_SPLIT_BASES}")
    FOLD_SPLIT_STEMS = discover_fold_stems_all(base_dirs, STEM_NAME_SUFFIX_FILTER)
    if not FOLD_SPLIT_STEMS:
        raise FileNotFoundError(
            "[fold_split_dir] No valid stems found under:\n  - " + "\n  - ".join(base_dirs)
        )
    print(f"[fold_split_dir] Found {len(FOLD_SPLIT_STEMS)} stems:")
    for base_dir, n, rs in FOLD_SPLIT_STEMS:
        print(f"  - base={base_dir} | {n} (run_suffix={rs})")
else:
    FOLD_SPLIT_STEMS = []

made = 0
multiplier = max(1, len(FOLD_SPLIT_STEMS)) if USE["fold_split_dir"] else 1

# === Outer loop over discovered fold_split_dir items ===
outer_stems = FOLD_SPLIT_STEMS if (USE["fold_split_dir"] and FOLD_SPLIT_STEMS) else [(None, None, None)]

for base_dir, stem_name, run_suffix in outer_stems:
    # Dynamic syn prefix
    if SYN_DETECT_FROM_BASE:
        syn_tag = extract_syn_tag(base_dir) or SYN_DEFAULT
    else:
        syn_tag = SYN_DEFAULT

    stem_prefix = STEM_PREFIX_TEMPLATE.format(project=PROJECT_TAG, syn=syn_tag or "")
    stem_prefix = re.sub(r"_+$", "", re.sub(r"__+", "_", stem_prefix))

    for lr_k in lr_keys:
        for blk_k in blk_keys:
            for wd_k in wd_keys:
                for es_k in es_keys:
                    for me_k in me_keys:
                        for ver_k in ver_keys:
                            for fold_k in fold_keys:
                                cfg = copy.deepcopy(base_cfg)
                                stem_parts = [stem_prefix]

                                # 1) learning rate
                                if lr_k is not None:
                                    cfg.setdefault("optimizer", {})
                                    cfg["optimizer"]["learning_rate"] = SciFloat(lr_map[lr_k])
                                stem_parts = add_tag(stem_parts, "", lr_k)

                                # 2) frozen blocks
                                if blk_k is not None:
                                    cfg.setdefault("freeze", {})
                                    cfg["freeze"]["is_freezing_model"] = True
                                    seq = CommentedSeq(block_sets[blk_k])
                                    seq.fa.set_flow_style()
                                    cfg["freeze"]["training_block_name"] = seq
                                stem_parts = add_tag(stem_parts, "", blk_k)

                                # 3) weight decay
                                if wd_k is not None:
                                    cfg.setdefault("optimizer", {})
                                    cfg["optimizer"]["weight_decay"] = SciFloat(wd_map[wd_k])
                                stem_parts = add_tag(stem_parts, "_wd", wd_k)

                                # 4) early stop
                                if es_k is not None:
                                    cfg.setdefault("trainer", {})
                                    cfg["trainer"]["early_stop_epoch"] = es_map[es_k]
                                stem_parts = add_tag(stem_parts, "_es", es_k)

                                # 5) max_epoch
                                if me_k is not None:
                                    cfg.setdefault("trainer", {})
                                    cfg["trainer"]["max_epoch"] = me_map[me_k]
                                stem_parts = add_tag(stem_parts, "_ep", me_k)

                                # 6) pretrained model path
                                cfg.setdefault("transfer", {})
                                if (ver_k is not None) and (fold_k is not None):
                                    pt_path = build_pt_path(ver_k, fold_k)
                                    cfg["transfer"]["pertrained_model_dir"] = pt_path
                                    cfg["transfer"]["transfer_ckp_dir"] = pt_path
                                    stem_parts = add_tag(stem_parts, "", pt_ver_map[ver_k])
                                    stem_parts = add_tag(stem_parts, "", f"fold_{pt_fold_map[fold_k]}")

                                # 7) final stem, out_dir, fold_split_dir, write file
                                file_stem_base = "_".join([p for p in stem_parts if p])
                                file_stem = f"{file_stem_base}_{run_suffix}" if run_suffix else file_stem_base

                                aug_suffix = extract_aug_suffix(base_dir)
                                file_stem = f"{file_stem}_{aug_suffix}" if aug_suffix else file_stem

                                cfg.setdefault("train", {})
                                cfg["train"]["out_dir"] = os.path.join(OUT_DIR_ROOT, file_stem)

                                if USE["fold_split_dir"] and stem_name:
                                    fsd = apply_fold_split_dir(cfg, base_dir, stem_name, FOLD_SPLIT_SUFFIX)
                                else:
                                    fsd = None

                                out_path = OUT_TMPL.format(stem=file_stem)
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                with open(out_path, "w") as fo:
                                    yaml.dump(cfg, fo)

                                made += 1
                                print(f"✔ {out_path}")
                                print(f"  ↳ out_dir={cfg['train']['out_dir']}")
                                if fsd:
                                    print(f"  ↳ fold_split_dir={fsd}")
                                if aug_suffix:
                                    src_aug = "aug12" if aug_suffix == "a2" else "aug1"
                                    print(f"  ↳ aug_suffix=_{aug_suffix} (from {src_aug})")

print(f"\nGenerated {made} configs (multiplied by discovered fold stems × {multiplier}).")