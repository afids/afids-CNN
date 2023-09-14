"""Extract one afid from an FCSV."""

import numpy as np
import pandas as pd


def extract_afid(fcsv: pd.DataFrame, afid_label, out_path):
    """Select an AFID from an FCSV dataframe and save its values to a text file.

    Parameters
    ----------
    fcsv
        DataFrame with FCSV data in order. Must at least have columns 'x', 'y', and 'z'
    afid_label
        Label of the AFID to grab (i.e. one-indexed, between 1-32
    out_path
        File name to which to save the AFID coords.
    """
    fid_df = fcsv.loc[:, ["x", "y", "z"]].assign(value=1)
    np.savetxt(out_path, fid_df.iloc[[afid_label - 1]].values)


def main() -> None:
    fcsv_df = pd.read_csv(snakemake.input.fcsv, sep=",", header=2)
    afid_label = int(snakemake.wildcards.afid)
    extract_afid(fcsv_df, afid_label, snakemake.output.txt)


if __name__ == "__main__":
    main()
