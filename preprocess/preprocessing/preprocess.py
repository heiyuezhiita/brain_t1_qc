import os
import argparse
import subprocess

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torchio as tio
from deepbet import run_bet


def zscore_normalize_nonzero(input_path, output_path, scale=1000):
    """Apply z-score normalization on nonzero voxels only."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image = sitk.ReadImage(input_path, sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(image)

    mask = array != 0
    values = array[mask]
    mean, std = np.mean(values), np.std(values)
    std = std if std != 0 else 1e-5

    out = np.zeros_like(array)
    out[mask] = (values - mean) / std * scale

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(image)
    sitk.WriteImage(out_img, output_path)
    print(f"Saved z-score normalized image: {output_path}")


def deepbet(input_img, brain_out, mask_out, threshold=0.5, n_dilate=0, no_gpu=True):
    """Run DeepBet skull stripping."""
    run_bet(
        [input_img], [brain_out], [mask_out],
        threshold=threshold, n_dilate=n_dilate, no_gpu=no_gpu
    )
    print("DeepBet completed")


def synthstrip(input_img, brain_out, mask_out):
    """Run SynthStrip skull stripping."""
    cmd = f'mri_synthstrip -i "{input_img}" -o "{brain_out}" -m "{mask_out}"'
    os.system(cmd)
    print("SynthStrip completed")


def merge_masks_and_extract_brain(orig_img, mask1, mask2, merged_mask, brain_out):
    """Merge two masks and extract brain region."""
    os.system(f'fslmaths "{mask1}" -add "{mask2}" "{merged_mask}"')
    os.system(f'fslmaths "{merged_mask}" -bin "{merged_mask}"')
    os.system(f'fslmaths "{orig_img}" -mul "{merged_mask}" "{brain_out}"')
    print("Mask merging and brain extraction completed")


def cut(input_file, output_file):
    """Crop image to the nonzero bounding box using FSL."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        parts = subprocess.check_output(
            f'fslstats "{input_file}" -w', shell=True
        ).decode().split()
    except subprocess.CalledProcessError:
        print(f"Failed to get bounding box: {input_file}")
        return

    if len(parts) < 6:
        print(f"Expected at least 6 values, got {len(parts)}: {parts}")
        return

    x0, sx, y0, sy, z0, sz = map(lambda v: int(float(v)), parts[:6])

    cmd = f'fslroi "{input_file}" "{output_file}" {x0} {sx} {y0} {sy} {z0} {sz}'
    if os.system(cmd) != 0:
        print(f"fslroi failed: {input_file}")
        return

    print(f"Cropped image saved to: {output_file}")


def resize_t1(input_file, output_file, size=(160, 192, 160)):
    """Resize a T1 image to the target shape."""
    image = tio.ScalarImage(input_file)
    resized = tio.Resize(size)(image)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    resized.save(output_file)
    print(f"Resized image saved to: {output_file}")


def convert_to_int16(input_file, output_file):
    """Convert image data to int16 and save."""
    img = nib.load(input_file)
    data = img.get_fdata().astype(np.int16)

    out_img = nib.Nifti1Image(data, img.affine, img.header)
    out_img.set_data_dtype(np.int16)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(out_img, output_file)
    print(f"Saved int16 image to: {output_file}")


def get_subject_id(input_path):
    """Extract subject ID from input path."""
    return input_path.strip().split("/")[-2]


def process_subject(input_path, output_root):
    """Run the full preprocessing pipeline for one subject."""
    subject_id = get_subject_id(input_path)
    base_out = os.path.join(output_root, subject_id, "T1")
    os.makedirs(base_out, exist_ok=True)

    deepbet_brain = os.path.join(base_out, "T1_deepbet.nii.gz")
    deepbet_mask = os.path.join(base_out, "T1_deepbet_mask.nii.gz")
    synth_brain = os.path.join(base_out, "T1_synthstrip.nii.gz")
    synth_mask = os.path.join(base_out, "T1_synthstrip_mask.nii.gz")
    merged_mask = os.path.join(base_out, "merge_mask.nii.gz")
    merged_brain = os.path.join(base_out, "mstr_brain.nii.gz")
    zscore_path = os.path.join(base_out, "T1_brain_zscore.nii.gz")
    cropped_path = os.path.join(base_out, "T1_cropped.nii.gz")
    resized_float = os.path.join(base_out, "T1_resize_float32.nii.gz")
    resized_int16 = os.path.join(base_out, "T1_resize.nii.gz")

    print(f"\nProcessing subject: {subject_id}")

    deepbet(input_path, deepbet_brain, deepbet_mask)
    synthstrip(input_path, synth_brain, synth_mask)
    merge_masks_and_extract_brain(
        input_path, deepbet_mask, synth_mask, merged_mask, merged_brain
    )
    zscore_normalize_nonzero(merged_brain, zscore_path)
    cut(zscore_path, cropped_path)
    resize_t1(cropped_path, resized_float)
    convert_to_int16(resized_float, resized_int16)

    print(f"Finished subject: {subject_id}")


def main():
    parser = argparse.ArgumentParser(description="MRI preprocessing pipeline")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input image paths")
    parser.add_argument("--output_root", required=True, help="Output root directory")
    args = parser.parse_args()

    for input_path in args.inputs:
        process_subject(input_path, args.output_root)


if __name__ == "__main__":
    main()