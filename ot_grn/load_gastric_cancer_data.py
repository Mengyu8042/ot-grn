# load_gastric_cancer_data.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Tuple
from importlib import resources


def load_gastric_cancer_data(
    same_source: bool = True, paired: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load gastric cancer gene expression data.

    Parameters
    ----------
    same_source : bool, optional
        If True, normal and tumor samples are from the same source, by default True.
    paired : bool, optional
        Only relevant if `same_source` is True. If True, samples are paired, by default True.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Normal expression dataframe (genes x samples).
        - Tumor expression dataframe (genes x samples).
    """
    if same_source:
        with resources.open_text("ot_grn.data", "normal_data_source1.csv") as f:
            exp1 = pd.read_csv(f, index_col=0)
    else:
        with resources.open_text("ot_grn.data", "normal_data_source2.csv") as f:
            exp1 = pd.read_csv(f, index_col=0)
        exp1 = exp1.groupby(exp1.index).median()

    with resources.open_text("ot_grn.data", "tumor_data_source1.csv") as f:
        exp2 = pd.read_csv(f, index_col=0)

    # Filter genes
    common_genes = list(set(exp1.index).intersection(set(exp2.index)))
    exp1 = exp1.loc[common_genes]
    exp2 = exp2.loc[common_genes]

    if same_source and paired:
        exp2 = exp2[exp1.columns]

    assert (
        exp1.index.tolist() == exp2.index.tolist()
    ), "Genes in exp1 and exp2 do not match."
    return exp1, exp2
