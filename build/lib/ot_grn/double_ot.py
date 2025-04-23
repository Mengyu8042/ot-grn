# double_ot.py
# -*- coding: utf-8 -*-

from typing import Union, Tuple
import numpy as np
import pandas as pd
import ot
from sklearn.decomposition import PCA
import scipy.stats as stats
from scipy.spatial.distance import cdist

def double_ot(
    exp1: Union[np.ndarray, pd.DataFrame],
    exp2: Union[np.ndarray, pd.DataFrame],
    paired: bool = True,
    reg_m: float = 0.05,
    reg: Union[float, Tuple[float, float]] = (0.005, 0.05),
    s: int = None,
    n_components: int = None,
    return_alignment: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Implement the Double Optimal Transport (OT) method for inferring gene regulatory networks.

    Parameters
    ----------
    exp1 : array-like (genes x samples)
        Expression matrix for condition 1.
    exp2 : array-like (genes x samples)
        Expression matrix for condition 2.
    paired : bool, optional
        If True, assumes the samples are paired. If False, uses partial OT to align samples, by default True.
    reg_m : float, optional
        Marginal relaxation parameter for robust OT, by default 0.05.
    reg : float or (float, float), optional
        Entropic regularization; either a scalar or a tuple (reg_pot, reg_rot), by default (0.005, 0.05).
    s : int, optional
        Transport budget in partial OT, by default None (min(#samples)).
    n_components : int, optional
        Number of principal components for PCA in sample alignment, by default None (all components).
    return_alignment : bool, optional
        If True and samples are unpaired (`paired=False`), returns the sample alignment matrix (sample-level OT plan),
        by default False.
        
    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        If `return_alignment` is False, returns the gene-level OT plan matrix (genes x genes).
        If `return_alignment` is True and samples are unpaired, returns a tuple:
        - gene-level OT plan matrix (genes x genes).
        - sample-level OT plan matrix (samples_condition1 x samples_condition2).
    """
    # Convert pandas DataFrames to NumPy arrays if necessary
    if isinstance(exp1, pd.DataFrame):
        exp1 = exp1.values
    if isinstance(exp2, pd.DataFrame):
        exp2 = exp2.values
        
    # Unpack regularization parameters
    if isinstance(reg, tuple):
        reg_pot, reg_rot = reg
    else:
        reg_pot = reg_rot = reg
    
    G0 = None  # Initialize sample alignment plan

    if not paired:
        # Align samples using partial OT
        exp1, exp2, G0 = _align_samples(
            exp1, exp2, reg=reg_pot, s=s, n_components=n_components)

    # Calculate gene distance matrix
    gene_dist = _calculate_distance_matrix(exp1, exp2, method='spearman')

    # Calculate mean expression levels for each gene
    exp1_mass = np.mean(exp1, axis=1)
    exp2_mass = np.mean(exp2, axis=1)

    # Compute the unbalanced OT plan
    ot_plan = ot.unbalanced.sinkhorn_unbalanced(
        exp1_mass, exp2_mass, gene_dist, reg=reg_rot, reg_m=reg_m)

    if return_alignment and not paired:
        return ot_plan, G0
    else:
        return ot_plan

def _align_samples(
    exp1: np.ndarray,
    exp2: np.ndarray,
    reg: float = 5e-3,
    s: int = None,
    n_components: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align samples from two conditions using partial optimal transport.

    Parameters
    ----------
    exp1 : np.ndarray
        Expression matrix for condition 1 (genes x samples).
    exp2 : np.ndarray
        Expression matrix for condition 2 (genes x samples).
    reg : float, optional
        Entropic regularization for partial OT, by default 0.005.
    s : int, optional
        Total mass to transport, by default None (min(#samples)).
    n_components : int, optional
        Number of principal components for PCA, by default None (all components).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - Aligned expression matrix for condition 1 (genes x aligned_samples).
        - Aligned expression matrix for condition 2 (genes x aligned_samples).
        - Partial OT plan matrix used for sample alignment (samples_condition1 x samples_condition2).
    """
    n_samples1 = exp1.shape[1]
    n_samples2 = exp2.shape[1]
    s = min(n_samples1, n_samples2) if s is None else s

    # Perform PCA
    pca = PCA(n_components=n_components)
    data = np.hstack((exp1, exp2)).T  # Samples x Genes
    data_pca = pca.fit_transform(data)

    data1_pca = data_pca[:n_samples1]
    data2_pca = data_pca[n_samples1:]

    # Calculate cosine distance between samples
    sample_dist = _calculate_distance_matrix(data1_pca, data2_pca, method='cosine')

    # Uniform weights for samples
    weights1 = np.ones(n_samples1)
    weights2 = np.ones(n_samples2)

    # Compute the partial OT plan
    if reg == 0:
        G0 = ot.partial.partial_wasserstein(weights1, weights2, sample_dist, m=s)
    else:
        G0 = ot.partial.entropic_partial_wasserstein(
            weights1, weights2, sample_dist, reg=reg, m=s, stopThr=1e-9)
        G0 = _binarize_alignment(G0, s)
    
    # Get matching indices based on transport plan
    indices1, indices2 = np.nonzero(G0)
    exp1_aligned = exp1[:, indices1]
    exp2_aligned = exp2[:, indices2]

    return exp1_aligned, exp2_aligned, G0

def _calculate_distance_matrix(
    exp1: np.ndarray,
    exp2: np.ndarray,
    method: str = 'spearman'
) -> np.ndarray:
    """
    Calculate the gene distance matrix using specified method.

    Parameters
    ----------
    exp1 : np.ndarray
        Expression matrix for condition 1 (genes x samples).
    exp2 : np.ndarray
        Expression matrix for condition 2 (genes x samples).
    method : str, optional
        Method to define gene distance ('euclidean', 'cosine', 'l1', 'pearson', 'spearman'), by default 'spearman'.

    Returns
    -------
    np.ndarray
        Gene distance matrix (genes x genes).
    """
    if method in ['euclidean', 'cosine', 'l1']:
        gene_dist = _scipy_distance(exp1, exp2, method)
    elif method in ['pearson', 'spearman']:
        if method == 'spearman':
            cor = _spearman_correlation(exp1, exp2)
        elif method == 'pearson':
            cor = _pearson_correlation(exp1, exp2)
        gene_dist = 1 - np.abs(cor)
    else:
        raise ValueError("Invalid method. Choose from 'euclidean', 'cosine', 'l1', 'pearson', 'spearman'.")
    gene_dist /= np.max(gene_dist)
    return gene_dist

def _pearson_correlation(exp1, exp2):
    return np.corrcoef(exp1, exp2)[:exp1.shape[0], exp1.shape[0]:]

def _spearman_correlation(exp1, exp2):
    rho, _ = stats.spearmanr(exp1.T, exp2.T)
    return rho[:exp1.shape[0], exp1.shape[0]:]

def _scipy_distance(exp1, exp2, metric):
    if metric == 'euclidean':
        return cdist(exp1, exp2)**2
    elif metric == 'cosine':
        return cdist(exp1, exp2, 'cosine')
    elif metric == 'l1':
        return cdist(exp1, exp2, 'minkowski', p=1)

def _binarize_alignment(matrix, s):
    n_rows, n_cols = matrix.shape

    # Step 1: mark max values by row or column
    if n_rows <= n_cols:
        max_mask = (matrix == np.max(matrix, axis=1, keepdims=True)).astype(int)
    else:
        max_mask = (matrix == np.max(matrix, axis=0, keepdims=True)).astype(int)

    # Step 2: if already ≤ s, return directly
    if max_mask.sum() <= s:
        return max_mask

    # Step 3: keep top-s among marked entries
    scores = matrix * max_mask
    flat_scores = scores.flatten()
    top_indices = np.argsort(flat_scores)[::-1]

    threshold = flat_scores[top_indices[s]]
    binary = ((scores >= threshold) & (max_mask == 1)).astype(int)

    return binary