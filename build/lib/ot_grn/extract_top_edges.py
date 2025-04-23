# extract_top_edges.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import List, Tuple


def extract_top_edges(
    y_score: np.ndarray,
    gene_names: List[str],
    num: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract the top 'num' edges based on scores and the genes connected by these edges.

    Parameters
    ----------
    y_score : np.ndarray
        Score matrix of shape (p, p), where p is the number of genes.
    gene_names : List[str]
        List of gene names of length p.
    num : int
        Number of top edges to extract.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - edges_df: Containing the top edges with columns 'from' and 'to'.
        - connected_genes: Unique genes connected by the top edges.
    """
    p = y_score.shape[0]
    assert y_score.shape == (p, p), "y_score must be a square matrix."
    assert len(gene_names) == p, "Length of gene_names must match size of y_score."

    # Create a mask to exclude self-loops (optional, remove if self-loops are allowed)
    mask = ~np.eye(p, dtype=bool)
    y_score_no_diag = y_score.copy()
    y_score_no_diag[~mask] = -np.inf  # Set diagonal to -inf to exclude self-loops

    # Flatten the score matrix and get the indices of the top 'num' scores
    flat_indices = np.argpartition(-y_score_no_diag.flatten(), num - 1)[:num]

    # Convert flat indices to 2D indices
    row_indices, col_indices = np.unravel_index(flat_indices, y_score_no_diag.shape)

    # Retrieve gene names for 'from' and 'to'
    from_genes = [gene_names[i] for i in row_indices]
    to_genes = [gene_names[j] for j in col_indices]

    # Create a DataFrame with the edge information
    edges_df = pd.DataFrame({"from": from_genes, "to": to_genes})

    # Get the list of unique genes connected by these edges
    genes_list = edges_df[["from", "to"]].values.flatten().tolist()
    genes_list = list(set(genes_list))

    return edges_df, genes_list
