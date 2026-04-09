#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ruamel.yaml import YAML
import copy, os, re, glob

# === YAML setup ===
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

# ===== Project / stem prefix =====
PROJECT_TAG = "cohort_b"
SYN_FALLBACK = ""
STEM_PREFIX_TEMPLATE = "{project}_{syn}"
FORCE_FIXED_STEM_PREFIX = False
STEM_PREFIX_FIXED = "cohort_b_syn100"

def make_stem_prefix(syn_tag: str | None) -> str:
    """Build stem prefix from syn_tag, or use fixed prefix if enabled."""
    if FORCE_FIXED_STEM_PREFIX:
        return STEM_PREFIX_FIXED
    syn = syn_tag if syn_tag else SYN_FALLBACK
    prefix = STEM_PREFIX_TEMPLATE.format(project=PROJECT_TAG, syn=syn)
    return re.sub(r"_+$", "", re.sub(r"__+", "_", prefix))


# === Base paths ===
BASE = 'checkpoints/transfer_yaml_template/predict/cohort_b/cohort_b_syn100_model.yaml'
OUT_TMPL = 'configs/transfer/batch_predict/cohort_b/predict_{stem}.yaml'

# Output roots
CKP_DIR_ROOT   = 'results/transfer/model/cohort_b'
PRED_DIR_ROOT  = 'results/transfer/prediction/cohort_b'

# ===== fold_split_dir scan settings =====
SET_FOLD_SPLIT_DIR = True
# Supports brace expansion and glob
FOLD_SPLIT_BASES   = "data/splits/random10/{syn100}/{aug}/cohort_b"
FOLD_SPLIT_SUFFIX  = "fold"
AUTO_SCAN_STEMS    = True
STEM_NAME_FILTER   = "_int16gz"

# ===== Switches =====
USE = {
    "lr":             False,
    "blocks":         False,
    "wd":             False,
    "es":             False,
    "max_epoch":      False,
    "pt_ver":         True,
    "pt_fold":        True,
    "fold_split_run": True,   # iterate r01..r10 and write cfg['predict']['fold_split_dir']
}

# ===== Name maps (also used for YAML values where needed) =====
lr_map = {"le5": 1e-5, "le4": 1e-4, "le3": 1e-3, "3e4": 3e-4, "3e5": 3e-5}
block_sets = {
    "seg":        ["conv_seg"],
    "seg_l4":     ["conv_seg", "layer4"],
    "seg_l4_l3":  ["conv_seg", "layer4", "layer3"],
}
wd_keys_all = ["1e-5", "1e-6", "0"]
es_map = {"10": 10, "15": 15, "25": 25}
me_map = {"5": 5, "10": 10, "20": 20}

pt_ver_map = {
    "cohort_a_r1":  "cohort_a_aug_bal",
    "cohort_a_r2":  "cohort_a_aug_imb",
    "cohort_a_noaug": "cohort_a_noaug",
}
pt_fold_map = {f"f{k}": k for k in range(5)}  # f0..f4 -> 0..4

# Run ids for fold_split_dir
fsd_run_map = {f"r{str(i).zfill(2)}": f"r{str(i).zfill(2)}" for i in range(1, 11)}

# ===== Optional subsets =====
SUBSET = {
    "lr":             None,
    "blocks":         None,
    "wd":             None,
    "es":             ["10", "15"],
    "max_epoch":      ["5","10"],
    "pt_ver":         None,
    "pt_fold":        None,
    "fold_split_run": None,
}

# === Load base config ===
with open(BASE, 'r') as f:
    base_cfg = yaml.load(f)

# ---------------- Helpers ----------------
def pick_keys(all_keys, use_flag, subset):
    if not use_flag:
        return [None]
    if subset:
        for k in subset:
            if k not in all_keys:
                raise ValueError(f"subset key '{k}' not in {list(all_keys)}")
        return list(subset)
    return list(all_keys)

lr_keys   = pick_keys(list(lr_map.keys()),      USE["lr"],             SUBSET["lr"])
blk_keys  = pick_keys(list(block_sets.keys()),  USE["blocks"],         SUBSET["blocks"])
wd_keys   = pick_keys(wd_keys_all,              USE["wd"],             SUBSET["wd"])
es_keys   = pick_keys(list(es_map.keys()),      USE["es"],             SUBSET["es"])
me_keys   = pick_keys(list(me_map.keys()),      USE["max_epoch"],      SUBSET["max_epoch"])
ver_keys  = pick_keys(list(pt_ver_map.keys()),  USE["pt_ver"],         SUBSET["pt_ver"])
fold_keys = pick_keys(list(pt_fold_map.keys()), USE["pt_fold"],        SUBSET["pt_fold"])
run_keys  = pick_keys(list(fsd_run_map.keys()), USE["fold_split_run"], SUBSET["fold_split_run"])

os.makedirs(os.path.dirname(OUT_TMPL), exist_ok=True)

def add_tag(parts, prefix, key_str):
    if key_str is None or key_str == "":
        return parts
    return parts + [f"{prefix}{key_str}"] if prefix else parts + [key_str]

def brace_expand(pattern: str):
    """Expand braces, e.g. '/a/{b,c}/d' -> ['/a/b/d', '/a/c/d'].""" 
    m = re.search(r"\{([^{}]+)\}", pattern)
    if not m:
        return [pattern]
    head, body, tail = pattern[:m.start()], m.group(1), pattern[m.end():]
    out = []
    for choice in body.split(","):
        out.extend(brace_expand(head + choice + tail))
    return out

def expand_base_specs(specs):
    """Expand base paths from string/list with braces and glob."""
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

def extract_syn_tag(path: str):
    """Extract syn tag like syn50/syn100/syn300 from path."""
    if not path:
        return None
    for token in os.path.normpath(path).split(os.sep):
        if re.fullmatch(r"syn\d+", token):
            return token
    return None

def discover_fold_stems(base_dir: str, name_suffix_filter: str = "_int16gz"):
    """
    Scan one level under base_dir and keep directories that:
      1) are directories
      2) contain a 'fold' subdir
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
        m = re.search(r'_(r\d{2,})', name)
        run_suffix = m.group(1) if m else None
        stems.append((name, run_suffix))
    stems.sort(key=lambda x: x[0])
    return stems

def apply_fold_split_dir(cfg: dict, base: str, stem: str, suffix: str = "fold") -> str:
    """Write fold_split_dir into predict/data/train/root and return path ending with '/'."""
    path = os.path.join(base, stem, suffix)
    if not path.endswith(os.sep):
        path += os.sep
    if isinstance(cfg.get("predict"), dict):
        cfg["predict"]["fold_split_dir"] = path
    elif isinstance(cfg.get("data"), dict):
        cfg["data"]["fold_split_dir"] = path
    elif isinstance(cfg.get("train"), dict):
        cfg["train"]["fold_split_dir"] = path
    else:
        cfg["fold_split_dir"] = path
    return path

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

# === Scan base dirs and group by run suffix ===
RUN_ITEMS = {}  # {'r07': [(base_dir, stem_name, syn_tag), ...], ...}

if USE["fold_split_run"]:
    if not SET_FOLD_SPLIT_DIR:
        raise RuntimeError("fold_split_run is enabled, but SET_FOLD_SPLIT_DIR=False.")

    base_dirs = expand_base_specs(FOLD_SPLIT_BASES)
    if not base_dirs:
        raise FileNotFoundError(f"[fold_split_dir] No base dir found: {FOLD_SPLIT_BASES}")

    print("[fold_split_dir] Expanded base dirs:")
    for bd in base_dirs:
        print("  -", bd)

    if AUTO_SCAN_STEMS:
        for bd in base_dirs:
            syn_tag = extract_syn_tag(bd) or "syn300"
            stems = discover_fold_stems(bd, STEM_NAME_FILTER)
            for name, run_suffix in stems:
                if run_suffix is None:
                    continue
                RUN_ITEMS.setdefault(run_suffix, []).append((bd, name, syn_tag))

        print("[fold_split_dir] Grouped by run_suffix (show up to 5 each):")
        for rk, items in sorted(RUN_ITEMS.items()):
            demo = ", ".join([f"{os.path.basename(b)}/{n}" for b, n, _ in items[:5]])
            more = " ..." if len(items) > 5 else ""
            print(f"  - {rk}: {len(items)} items, e.g. {demo}{more}")
    else:
        raise RuntimeError("fold_split_run requires AUTO_SCAN_STEMS=True.")


made = 0

for lr_k in lr_keys:
    for blk_k in blk_keys:
        for wd_k in wd_keys:
            for es_k in es_keys:
                for me_k in me_keys:
                    for ver_k in ver_keys:
                        for fold_k in fold_keys:

                            # === Build shared stem parts before run ===
                            common_parts = []

                            common_parts = add_tag(common_parts, "", lr_k)
                            common_parts = add_tag(common_parts, "", blk_k)
                            common_parts = add_tag(common_parts, "_wd", wd_k)
                            common_parts = add_tag(common_parts, "_es", es_k)
                            common_parts = add_tag(common_parts, "_ep", me_k)
                            if ver_k is not None:
                                common_parts = add_tag(common_parts, "", pt_ver_map[ver_k])
                            if fold_k is not None:
                                common_parts = add_tag(common_parts, "", f"fold_{pt_fold_map[fold_k]}")

                            # Base config before run-specific fields
                            cfg_base = copy.deepcopy(base_cfg)
                            cfg_base.setdefault("predict", {})

                            # === With run dimension ===
                            if USE["fold_split_run"]:
                                run_key_iter = run_keys if run_keys != [None] else sorted(RUN_ITEMS.keys())
                                for run_k in run_key_iter:
                                    run_suffix = fsd_run_map.get(run_k, run_k) if run_k is not None else None
                                    items = RUN_ITEMS.get(run_suffix, [])
                                    if not items:
                                        continue

                                    for base_dir, stem_name, syn_tag in items:
                                        stem_prefix_dynamic = make_stem_prefix(syn_tag)

                                        # Filename stem = prefix + shared parts + run suffix
                                        stem_parts = [stem_prefix_dynamic] + common_parts
                                        stem_parts = add_tag(stem_parts, "", run_suffix)
                                        file_stem = "_".join([p for p in stem_parts if p])

                                        # Add aug suffix after rXX
                                        aug_suffix = extract_aug_suffix(base_dir)
                                        file_stem_aug = f"{file_stem}_{aug_suffix}" if aug_suffix else file_stem

                                        cfg = copy.deepcopy(cfg_base)

                                        # Write fold_split_dir
                                        fsd = apply_fold_split_dir(cfg, base_dir, stem_name, FOLD_SPLIT_SUFFIX)

                                        # Predict paths
                                        cfg["predict"]["ckp_dir"] = os.path.join(CKP_DIR_ROOT,  file_stem_aug, 'fold_1') + os.sep
                                        cfg["predict"]["out_dir"] = os.path.join(PRED_DIR_ROOT, file_stem_aug, 'fold_2') + os.sep

                                        # max_epoch
                                        if USE["max_epoch"] and (me_k is not None):
                                            cfg.setdefault("trainer", {})
                                            cfg["trainer"]["max_epoch"] = me_map[me_k]

                                        out_path = OUT_TMPL.format(stem=file_stem_aug)
                                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                        with open(out_path, "w") as fo:
                                            yaml.dump(cfg, fo)

                                        made += 1
                                        print(f"✔ {out_path}")
                                        print(f"  ↳ predict.ckp_dir = {cfg['predict']['ckp_dir']}")
                                        print(f"  ↳ predict.out_dir = {cfg['predict']['out_dir']}")
                                        print(f"  ↳ predict.fold_split_dir = {cfg['predict'].get('fold_split_dir')}")

                                        if aug_suffix:
                                            print(f"  ↳ aug_suffix = _{aug_suffix} ({'aug12' if aug_suffix=='a2' else 'aug1'})")

                                        if USE["max_epoch"] and (me_k is not None):
                                            print(f"  ↳ trainer.max_epoch = {cfg['trainer']['max_epoch']}")

                            # === Without run dimension ===
                            else:
                                stem_prefix_dynamic = make_stem_prefix(None)
                                base_file_stem = "_".join([p for p in ([stem_prefix_dynamic] + common_parts) if p])

                                # No base_dir here, so no aug suffix is added
                                file_stem_aug = base_file_stem

                                cfg_base["predict"]["ckp_dir"] = os.path.join(CKP_DIR_ROOT,  file_stem_aug, 'fold_1') + os.sep
                                cfg_base["predict"]["out_dir"] = os.path.join(PRED_DIR_ROOT, file_stem_aug, 'fold_2') + os.sep

                                if USE["max_epoch"] and (me_k is not None):
                                    cfg_base.setdefault("trainer", {})
                                    cfg_base["trainer"]["max_epoch"] = me_map[me_k]

                                out_path = OUT_TMPL.format(stem=file_stem_aug)
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                with open(out_path, "w") as fo:
                                    yaml.dump(cfg_base, fo)

                                made += 1
                                print(f"✔ {out_path}")
                                print(f"  ↳ predict.ckp_dir = {cfg_base['predict']['ckp_dir']}")
                                print(f"  ↳ predict.out_dir = {cfg_base['predict']['out_dir']}")
                                if USE["max_epoch"] and (me_k is not None):
                                    print(f"  ↳ trainer.max_epoch = {cfg_base['trainer']['max_epoch']}")

print(f"\nGenerated {made} config files with dynamic syn prefix, rXX suffix, and aug suffix (a1/a2).")