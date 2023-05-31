"""Extract one afid from an FCSV."""

from __future__ import annotations

from os import PathLike

import numpy as np
import pandas as pd


def extract_afid(fcsv: pd.DataFrame, afid_idx: int, out_path: PathLike[str] | str):
    fid_df = fcsv.loc[:, ["x", "y", "z"]].assign(value=1)
    np.savetxt(out_path, fid_df.iloc[[afid_idx]].values)


def main() -> None:
    fcsv_df = pd.read_csv(snakemake.input.fcsv, sep=",", header=2)
    afid_idx = int(snakemake.wildcards.afid)
    extract_afid(fcsv_df, afid_idx, snakemake.out.txt)
