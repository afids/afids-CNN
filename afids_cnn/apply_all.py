from __future__ import annotations

import json
import tarfile
import tempfile
from os import PathLike
from pathlib import Path
from typing import IO

import nibabel as nib
from numpy.typing import NDArray
from tensorflow import keras

from afids_cnn.apply import apply_model


def apply_all(
    model_path: PathLike[str] | str,
    img: nib.nifti1.Nifti1Image,
) -> dict[int, NDArray]:
    with tarfile.open(model_path, "r:gz") as tar_file:
        config_file = extract_config(tar_file)
        radius = int(json.load(config_file)["radius"])
        afid_dict: dict[int, NDArray] = {}
        for afid_label in range(1, 33):
            with tempfile.TemporaryDirectory() as model_dir:
                model = keras.models.load_model(
                    extract_afids_model(tar_file, model_dir, afid_label),
                )
            afid_dict[afid_label] = apply_model(
                img,
                afid_label,
                model,
                radius,
            )

    return afid_dict


def extract_config(tar_file: tarfile.TarFile) -> IO[bytes]:
    try:
        config_file = tar_file.extractfile("config.json")
    except KeyError as err:
        missing_data = "config file"
        raise ArchiveMissingDataError(missing_data, tar_file) from err
    if not config_file:
        missing_data = "config file as file"
        raise ArchiveMissingDataError(missing_data, tar_file)
    return config_file


def extract_afids_model(
    tar_file: tarfile.TarFile,
    out_path: PathLike[str] | str,
    afid_label: int,
) -> Path:
    for member in tar_file.getmembers():
        if member.isdir() and f"afid-{afid_label:02}" in member.name:
            tar_file.extract(member, out_path)
            return Path(out_path) / member.name
    msg = f"AFID {afid_label:02} model"
    raise ArchiveMissingDataError(msg, tar_file)


class ArchiveMissingDataError(Exception):
    def __init__(self, missing_data: str, tar_file: tarfile.TarFile) -> None:
        super().__init__(
            f"Required data {missing_data} not found in archive {tar_file}.",
        )
