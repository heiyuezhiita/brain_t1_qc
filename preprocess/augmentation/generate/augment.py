#!/usr/bin/env python3

import os
import random
import numpy as np


"""
TorchIO augmentation pipeline with configurable input sources.

Data usage:
- `preprocess/augmentation/generate/image_paths2.txt` should contain skull-stripped images
  used to generate synthetic noise artifacts.
- `preprocess/augmentation/generate/image_paths1.txt` should contain original images
  used to generate synthetic motion and ghosting artifacts.

Options:
- SHARE_ALL_INPUTS:
    If True, all modules use the same input txt file.
- MOTION_USE_GROUP:
    If True, motion1 and motion2 split one shared input list randomly.
    If False, motion1 and motion2 use separate txt files.
"""


# =========================
# Configuration
# =========================
SHARE_ALL_INPUTS = False
COMMON_TXT = "preprocess/augmentation/generate/image_paths1.txt"

MOTION_USE_GROUP = True
TXT_MOTION_GROUP = "preprocess/augmentation/generate/image_paths1.txt"
TXT_MOTION_1 = ""
TXT_MOTION_2 = ""

TXT_GHOSTING = "preprocess/augmentation/generate/image_paths1.txt"
TXT_NOISE = "preprocess/augmentation/generate/image_paths2.txt"
TXT_CONTRAST = "preprocess/augmentation/generate/image_paths1.txt"

OUTPUT_BASE = "data/images/raw/augmented/dataset_a"

USE_GHOSTING = True
USE_MOTION1 = True
USE_MOTION2 = True
USE_NOISE = True
USE_CONTRAST = False

RANDOM_SEED = 42


if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_BASE, exist_ok=True)


def read_txt_paths(txt_path):
    """Read non-empty lines from a txt file."""
    if not os.path.exists(txt_path):
        print(f"Warning: txt file not found: {txt_path}")
        return []
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def split_paths(paths, module_names):
    """Randomly split one path list across multiple modules."""
    if not module_names:
        return {}
    if not paths:
        return {name: [] for name in module_names}

    idxs = np.arange(len(paths))
    np.random.shuffle(idxs)
    splits = np.array_split(idxs, len(module_names))
    return {name: [paths[i] for i in split] for name, split in zip(module_names, splits)}


def get_subject_name(image_path):
    """Use parent folder name as subject identifier."""
    return os.path.basename(os.path.dirname(image_path))


def save_augmented(transform, paths, prefix, output_base=OUTPUT_BASE, name_mode="parent"):
    """Apply one transform to a batch of images and save outputs."""
    try:
        import torchio as tio
    except Exception as e:
        print(f"Warning: torchio is not available, skipping {prefix}. Error: {e}")
        return

    os.makedirs(output_base, exist_ok=True)

    for image_path in paths:
        if not os.path.exists(image_path):
            print(f"Warning: file not found: {image_path}")
            continue

        try:
            image = tio.ScalarImage(image_path)
            transformed = transform(image)

            if name_mode == "parent":
                image_name = os.path.basename(os.path.dirname(image_path))
            elif name_mode == "grandparent":
                image_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
            else:
                image_name = get_subject_name(image_path)

            output_path = os.path.join(output_base, f"{prefix}_{image_name}.nii.gz")
            transformed.save(output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def ghosting_batch(txt_file, output_base=OUTPUT_BASE):
    """Run RandomGhosting."""
    import torchio as tio
    transform = tio.transforms.RandomGhosting()
    save_augmented(transform, read_txt_paths(txt_file), "fg", output_base, name_mode="parent")


def motion1_worker(paths, output_base=OUTPUT_BASE):
    """Run RandomMotion with 3 transforms."""
    import torchio as tio
    transforms = [
        tio.transforms.RandomMotion(degrees=(3, 10), translation=(3, 10), num_transforms=3, image_interpolation="linear"),
        tio.transforms.RandomMotion(degrees=(-10, -3), translation=(3, 10), num_transforms=3, image_interpolation="linear"),
        tio.transforms.RandomMotion(degrees=(3, 10), translation=(-10, -3), num_transforms=3, image_interpolation="linear"),
        tio.transforms.RandomMotion(degrees=(-10, -3), translation=(-10, -3), num_transforms=3, image_interpolation="linear"),
    ]

    try:
        import torchio as tio  # keep local dependency behavior
    except Exception as e:
        print(f"Warning: torchio is not available, skipping motion1. Error: {e}")
        return

    os.makedirs(output_base, exist_ok=True)

    for image_path in paths:
        if not os.path.exists(image_path):
            print(f"Warning: file not found: {image_path}")
            continue
        try:
            image = tio.ScalarImage(image_path)
            transformed = random.choice(transforms)(image)
            image_name = get_subject_name(image_path)
            output_path = os.path.join(output_base, f"fm_{image_name}.nii.gz")
            transformed.save(output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def motion2_worker(paths, output_base=OUTPUT_BASE):
    """Run RandomMotion with 4 transforms."""
    import torchio as tio
    transforms = [
        tio.transforms.RandomMotion(degrees=(3, 10), translation=(3, 10), num_transforms=4, image_interpolation="linear"),
        tio.transforms.RandomMotion(degrees=(-10, -3), translation=(3, 10), num_transforms=4, image_interpolation="linear"),
        tio.transforms.RandomMotion(degrees=(3, 10), translation=(-10, -3), num_transforms=4, image_interpolation="linear"),
        tio.transforms.RandomMotion(degrees=(-10, -3), translation=(-10, -3), num_transforms=4, image_interpolation="linear"),
    ]

    try:
        import torchio as tio  # keep local dependency behavior
    except Exception as e:
        print(f"Warning: torchio is not available, skipping motion2. Error: {e}")
        return

    os.makedirs(output_base, exist_ok=True)

    for image_path in paths:
        if not os.path.exists(image_path):
            print(f"Warning: file not found: {image_path}")
            continue
        try:
            image = tio.ScalarImage(image_path)
            transformed = random.choice(transforms)(image)
            image_name = get_subject_name(image_path)
            output_path = os.path.join(output_base, f"fm_{image_name}.nii.gz")
            transformed.save(output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def noise_batch(txt_file, output_base=OUTPUT_BASE):
    """Run RandomNoise."""
    import torchio as tio
    transform = tio.transforms.RandomNoise(std=(175, 275))
    save_augmented(transform, read_txt_paths(txt_file), "fn", output_base, name_mode="grandparent")


def contrast_batch(txt_file, output_base=OUTPUT_BASE):
    """Run RandomGamma for contrast augmentation."""
    import torchio as tio
    transform = tio.transforms.RandomGamma(log_gamma=(-0.45, -0.3))
    save_augmented(transform, read_txt_paths(txt_file), "fc", output_base, name_mode="parent")


def main():
    """Run the full augmentation pipeline."""
    print("=== Augmentation pipeline started ===")

    if SHARE_ALL_INPUTS:
        ghost_txt = COMMON_TXT
        noise_txt = COMMON_TXT
        contrast_txt = COMMON_TXT
        motion_group_txt = COMMON_TXT
    else:
        ghost_txt = TXT_GHOSTING
        noise_txt = TXT_NOISE
        contrast_txt = TXT_CONTRAST
        motion_group_txt = TXT_MOTION_GROUP

    if USE_GHOSTING:
        print("Running RandomGhosting...")
        ghosting_batch(ghost_txt)
    else:
        print("RandomGhosting is disabled.")

    if USE_MOTION1 or USE_MOTION2:
        if MOTION_USE_GROUP:
            motion_paths = read_txt_paths(motion_group_txt)
            modules = []
            if USE_MOTION1:
                modules.append("motion1")
            if USE_MOTION2:
                modules.append("motion2")

            assignment = split_paths(motion_paths, modules)
            print(
                "Motion split: " +
                ", ".join(f"{m}={len(assignment.get(m, []))}" for m in modules)
            )

            if USE_MOTION1:
                motion1_worker(assignment.get("motion1", []))
            if USE_MOTION2:
                motion2_worker(assignment.get("motion2", []))
        else:
            if USE_MOTION1:
                print("Running motion1...")
                motion1_worker(read_txt_paths(TXT_MOTION_1))
            if USE_MOTION2:
                print("Running motion2...")
                motion2_worker(read_txt_paths(TXT_MOTION_2))
    else:
        print("Motion augmentation is disabled.")

    if USE_NOISE:
        print("Running RandomNoise...")
        noise_batch(noise_txt)
    else:
        print("RandomNoise is disabled.")

    if USE_CONTRAST:
        print("Running RandomGamma...")
        contrast_batch(contrast_txt)
    else:
        print("Contrast augmentation is disabled.")

    print("=== Augmentation pipeline finished ===")


if __name__ == "__main__":
    main()