#!/usr/bin/env python3
# coding: utf-8

"""
Preprocess augmented images with merged brain masks.

Supported inputs:
- fg_
- fm_
- fn_

Pipeline:
skull stripping with merged mask -> normalization -> cropping -> resizing -> int16
"""

import os
import argparse
import subprocess

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torchio as tio


DEFAULT_OUTPUT_ROOT = "data/images/preprocessed/augmented/cohort_a"
DEFAULT_MERGED_MASK_ROOT = "data/images/preprocessed/original/cohort_a"


def zscore_normalize_nonzero(input_path, output_path, scale=1000):
    """Apply z-score normalization to nonzero voxels only."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image = sitk.ReadImage(input_path, sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(image)
    mask = array != 0

    if not np.any(mask):
        print(f"Warning: all-zero image, skipped normalization: {input_path}")
        sitk.WriteImage(image, output_path)
        return

    values = array[mask]
    mean, std = np.mean(values), np.std(values)
    std = std if std != 0 else 1e-5

    out = np.zeros_like(array)
    out[mask] = (values - mean) / std * scale

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(image)
    sitk.WriteImage(out_img, output_path)
    print(f"Saved normalized image: {output_path}")


def prep_cut(input_file, output_file):
    """Crop the nonzero bounding box using FSL."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        parts = subprocess.check_output(
            f'fslstats "{input_file}" -w', shell=True
        ).decode().split()
    except subprocess.CalledProcessError:
        print(f"Failed to get bounding box: {input_file}")
        return

    if len(parts) < 6:
        print(f"Invalid fslstats output for {input_file}: {parts}")
        return

    x0, sx, y0, sy, z0, sz = map(lambda v: int(float(v)), parts[:6])
    cmd = f'fslroi "{input_file}" "{output_file}" {x0} {sx} {y0} {sy} {z0} {sz}'

    if os.system(cmd) != 0:
        print(f"fslroi failed: {input_file}")
        return

    print(f"Cropped image saved to: {output_file}")


def resize_t1(input_file, output_file, size=(160, 192, 160)):
    """Resize image to the target shape."""
    image = tio.ScalarImage(input_file)
    resized = tio.Resize(size)(image)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    resized.save(output_file)
    print(f"Saved resized image: {output_file}")


def convert_to_int16(input_file, output_file):
    """Convert image data to int16 and save."""
    img = nib.load(input_file)
    data = img.get_fdata().astype(np.int16)

    out_img = nib.Nifti1Image(data, img.affine, img.header)
    out_img.set_data_dtype(np.int16)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(out_img, output_file)
    print(f"Saved int16 image: {output_file}")


def merge_masks_and_extract_brain(input_img, merged_mask, output_img):
    """Apply a merged brain mask to extract the brain region."""
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    cmd = f'fslmaths "{input_img}" -mul "{merged_mask}" "{output_img}"'

    if os.system(cmd) != 0:
        print(f"Skull stripping failed: {input_img}")
        return False

    print(f"Saved skull-stripped image: {output_img}")
    return True


def get_prefix_and_subject_id(input_path):
    """Split filename into augmentation prefix and subject ID."""
    filename = os.path.basename(input_path)
    if "_" in filename:
        prefix, subject_id = filename.split("_", 1)
        return prefix + "_", subject_id.replace(".nii.gz", "")
    return "", filename.replace(".nii.gz", "")


def pipeline_with_merge_mask(input_path, merged_mask_root=DEFAULT_MERGED_MASK_ROOT, output_root=DEFAULT_OUTPUT_ROOT):
    """
    Process fg_/fm_/fn_ images with an existing merged mask.
    """
    prefix, subject_id = get_prefix_and_subject_id(input_path)
    merged_mask_path = os.path.join(merged_mask_root, subject_id, "T1", "merge_mask.nii.gz")
    base_out = os.path.join(output_root, f"{prefix}{subject_id}", "T1")
    os.makedirs(base_out, exist_ok=True)

    brain_path = os.path.join(base_out, "mstr_brain.nii.gz")
    norm_path = os.path.join(base_out, "T1_brain_zscore.nii.gz")
    crop_path = os.path.join(base_out, "T1_cropped.nii.gz")
    resize_path = os.path.join(base_out, "T1_resize_float32.nii.gz")
    int16_path = os.path.join(base_out, "T1_resize.nii.gz")

    print(f"\nProcessing: {input_path}")

    if not os.path.exists(merged_mask_path):
        print(f"Missing merged mask: {merged_mask_path}")
        return

    if not merge_masks_and_extract_brain(input_path, merged_mask_path, brain_path):
        return

    zscore_normalize_nonzero(brain_path, norm_path)
    prep_cut(norm_path, crop_path)
    resize_t1(crop_path, resize_path)
    convert_to_int16(resize_path, int16_path)

    print(f"Finished: {base_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess fg/fm/fn augmented images"
    )
    parser.add_argument("inputs", nargs="+", help="Input image paths")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output root directory")
    parser.add_argument("--merged-mask-root", default=DEFAULT_MERGED_MASK_ROOT, help="Merged mask root directory")
    args = parser.parse_args()

    for input_path in args.inputs:
        if not os.path.exists(input_path):
            print(f"Warning: input not found, skipped: {input_path}")
            continue

        fname = os.path.basename(input_path)

        if fname.startswith(("fg_", "fm_", "fn_")):
            pipeline_with_merge_mask(
                input_path,
                merged_mask_root=args.merged_mask_root,
                output_root=args.output_root,
            )
        else:
            print(f"Warning: unsupported input type, skipped: {input_path}")


if __name__ == "__main__":
    main()