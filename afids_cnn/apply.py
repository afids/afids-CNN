#!/usr/bin/env python3

from __future__ import annotations

import json
from argparse import ArgumentParser
from os import PathLike
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import skimage.measure
from numpy.typing import NDArray
from tensorflow import keras

MNI_FCSV = (
    Path(__file__).parent / "resources" / "tpl-MNI152NLin2009cAsym_res-01_T1w.fcsv"
)
MNI_IMG = (
    Path(__file__).parent / "resources" / "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
)


def load_fcsv(fcsv_path: PathLike[str] | str) -> pd.DataFrame:
    return pd.read_csv(fcsv_path, sep=",", header=2)


# utils to factor out
def get_fid(fcsv_df: pd.DataFrame, fid_label: int) -> NDArray:
    """Extract specific fiducial's spatial coordinates.

    Parameters
    ----------
    fcsv_df
        Dataframe with the FCSV data.
    fid_label
        Label (1-32) of the fiducial to grab.
    """
    return fcsv_df.loc[fid_label - 1, ["x", "y", "z"]].to_numpy(
        dtype="single",
        copy=True,
    )


def fid_voxel2world(fid_voxel: NDArray, nii_affine: NDArray) -> NDArray:
    """Transform fiducials in voxel coordinates to world coordinates."""
    # Translation
    fid_world = fid_voxel.T + nii_affine[:3, 3:4]
    # Rotation
    fid_world = np.diag(np.dot(fid_world, nii_affine[:3, :3]))

    return fid_world.astype(float)


def fid_world2voxel(
    fid_world: NDArray,
    nii_affine: NDArray,
    resample_size: int = 1,
    padding: int | None = None,
) -> NDArray:
    """Transform fiducials in world coordinates to voxel coordinates.

    Optionally, resample to match resampled image
    """
    # Translation
    fid_voxel = fid_world.T - nii_affine[:3, 3:4]
    # Rotation
    fid_voxel = np.dot(fid_voxel, np.linalg.inv(nii_affine[:3, :3]))

    # Round to nearest voxel
    fid_voxel = np.rint(np.diag(fid_voxel) * resample_size)

    if padding:
        fid_voxel = np.pad(fid_voxel, padding, mode="constant")

    return fid_voxel.astype(int)


def min_max_normalize(img: NDArray) -> NDArray:
    return (img - img.min()) / (img.max() - img.min())


def gen_patch_slices(centre: NDArray, radius: int) -> tuple[slice, slice, slice]:
    return tuple(slice(coord - radius, coord + radius + 1) for coord in centre[:3])


def slice_img(img: NDArray, centre: NDArray, radius: int) -> NDArray:
    slices = gen_patch_slices(centre, radius)
    return img[slices[0], slices[1], slices[2]]


def predict_distances(
    radius: int,
    model: keras.model,
    mni_fid: NDArray,
    img: NDArray,
) -> NDArray:
    dim = (2 * radius) + 1
    pred = np.reshape(slice_img(img, mni_fid, radius), (1, dim, dim, dim, 1))
    return model.predict(pred)


def process_distances(
    distances: NDArray,
    img: NDArray,
    mni_fid: NDArray,
    radius: int,
) -> NDArray:
    arr_dis = np.reshape(distances[0], (63, 63, 63))
    new_pred = np.full((img.shape), 100, dtype=float)
    slices = gen_patch_slices(mni_fid, radius)
    new_pred[slices[0], slices[1], slices[2]] = arr_dis
    transformed = np.exp(-0.5 * new_pred)
    thresh = np.percentile(transformed, 99.999)
    transformed[transformed < thresh] = 0
    transformed = transformed * 1000
    transformed = transformed.astype(int)
    new = skimage.measure.regionprops(transformed)
    centroids = {
        key: [centroid.centroid[idx] for centroid in new]
        for idx, key in enumerate(["x", "y", "z"])
    }
    return np.array(
        [sum(centroids[key]) / len(centroids[key]) for key in ["x", "y", "z"]],
    )


def apply_afid_model(
    img_path: PathLike[str] | str,
    model_path: PathLike[str] | str,
    mni_fid_path: PathLike[str] | str,
    mni_img_path: PathLike[str] | str,
    radius: int,
    fid_label: int,
    size: int,
    padding: int,
) -> NDArray:
    mni_img = nib.nifti1.load(mni_img_path)
    model = keras.models.load_model(model_path)
    mni_fid_world = get_fid(load_fcsv(mni_fid_path), fid_label - 1)
    mni_fid_resampled = fid_world2voxel(
        mni_fid_world,
        mni_img.affine,
        resample_size=size,
        padding=padding,
    )
    img = nib.nifti1.load(img_path)
    # need to normalize image here
    normalized = min_max_normalize(img.get_fdata())
    distances = predict_distances(
        radius,
        model,
        mni_fid_resampled,
        normalized,
    )
    fid_resampled = process_distances(
        distances,
        normalized,
        mni_fid_resampled,
        radius,
    )
    return fid_voxel2world(fid_resampled, img.affine)


def apply_model(
    img: nib.nifti1.Nifti1Image, fid_label: int, model: keras.model, radius: int,
) -> NDArray:
    mni_fid_world = get_fid(load_fcsv(MNI_FCSV), fid_label - 1)
    mni_img = nib.nifti1.load(MNI_IMG)
    mni_fid_resampled = fid_world2voxel(
        mni_fid_world,
        mni_img.affine,
        resample_size=1,
        padding=0,
    )
    normalized = min_max_normalize(img.get_fdata())
    distances = predict_distances(
        radius,
        model,
        mni_fid_resampled,
        normalized,
    )
    fid_resampled = process_distances(
        distances,
        normalized,
        mni_fid_resampled,
        radius,
    )
    return fid_voxel2world(fid_resampled, img.affine)


def gen_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("img_path")
    parser.add_argument("model_path")
    parser.add_argument("out_path")
    parser.add_argument(
        "radius",
        help="Radius of the patches that the model was trained on.",
    )
    parser.add_argument(
        "fid_label",
        help="Label (1-32) of the fiducial model to apply. E.g. AC is label 1.",
    )
    parser.add_argument(
        "--size",
        help="Size with which to resample the MNI AFID.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--padding",
        help="Number of zeroes to add to the edge of the MNI AFID.",
        default=0,
    )
    return parser


def main() -> None:
    args = gen_parser().parse_args()

    fid_world = apply_afid_model(
        args.img_path,
        args.model_path,
        Path(__file__).parent / "resources" / "tpl-MNI152NLin2009cAsym_res-01_T1w.fcsv",
        Path(__file__).parent
        / "resources"
        / "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz",
        int(args.radius),
        int(args.fid_label),
        int(args.size),
        int(args.padding),
    )
    with Path(args.out_path).open("w") as out_file:
        json.dump(list(fid_world), out_file)


if __name__ == "__main__":
    main()
