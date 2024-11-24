# generate_simulated_data.py
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple

def generate_simulated_data(
    p: int,
    n: int,
    diffgene: float = 0.2,
    indegree: float = 5,
    snr: float = 2,
    outlier_ratio: float = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate simulated normal and tumor gene expression matrices.

    Parameters
    ----------
    p : int
        Number of genes (rows).
    n : int
        Number of samples (columns).
    diffgene : float, optional
        Proportion of differentially expressed genes, by default 0.2.
    indegree : float, optional
        Expected number of parents for differential genes, by default 5.
    snr : float, optional
        Signal-to-noise ratio, by default 2.
    outlier_ratio : float, optional
        Proportion of outliers, by default 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - Normal expression matrix (genes x samples).
        - Tumor expression matrix (genes x samples).
        - True regulatory relationships matrix (genes x genes).
    """
    exp1 = _generate_multivariate_normal_matrix(n, p).T
    exp2, true_plan = _generate_paired_matrix(exp1, diffgene, indegree, snr, outlier_ratio)
    true_plan[true_plan != 0] = 1

    # Normalize expression matrices
    exp1 = _normalize_expression_matrix(exp1)
    exp2 = _normalize_expression_matrix(exp2)

    return exp1, exp2, true_plan

def _generate_multivariate_normal_matrix(n: int, p: int) -> np.ndarray:
    """
    Generate a random matrix with rows sampled from a multivariate normal distribution.

    Parameters
    ----------
    n : int
        Number of samples (rows).
    p : int
        Number of genes (columns).

    Returns
    -------
    np.ndarray
        Generated random matrix (n x p).
    """
    cov_matrix = np.diag(np.random.uniform(0.1, 2, p))
    mean_vector = np.random.uniform(2, 5, p)
    random_matrix = np.random.multivariate_normal(mean_vector, cov_matrix, n)
    return np.abs(random_matrix)

def _nonlinear_transform(x: np.ndarray, func_type: int) -> np.ndarray:
    """
    Apply a nonlinear function to the input array.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    func_type : int
        Type of nonlinear function to apply:
        - 1: Linear.
        - 2: Exponential.
        - 3: Square root.

    Returns
    -------
    np.ndarray
        Transformed array.
    """
    if func_type == 1:
        return x  # Linear
    elif func_type == 2:
        return 0.02 * np.exp(x)  # Exponential
    elif func_type == 3:
        return 0.5 * np.sqrt(x)  # Square root
    else:
        raise ValueError("Invalid function type. Choose 1, 2, or 3.")

def _apply_nonlinear_transform(matrix: np.ndarray) -> np.ndarray:
    """
    Apply random nonlinear functions to each row of the input matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.

    Returns
    -------
    np.ndarray
        Transformed matrix with nonlinear functions applied to each row.
    """
    transformed_matrix = matrix.copy()
    n_rows = transformed_matrix.shape[0]
    func_types = np.random.randint(1, 4, n_rows)
    for i in range(n_rows):
        transformed_matrix[i] = _nonlinear_transform(transformed_matrix[i], func_types[i])
    return transformed_matrix

def _generate_paired_matrix(
    exp1: np.ndarray,
    diffgene: float,
    indegree: float,
    snr: float,
    outlier_ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a paired expression matrix with noise and regulatory relationships between genes.

    Parameters
    ----------
    exp1 : np.ndarray
        Original expression matrix (genes x samples).
    diffgene : float
        Proportion of differentially expressed genes.
    indegree : float
        Expected number of parents for differential genes.
    snr : float
        Signal-to-noise ratio.
    outlier_ratio : float
        Proportion of outliers.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - Paired expression matrix (genes x samples).
        - True regulatory relationships matrix (genes x genes).
    """
    n_genes, n_samples = exp1.shape
    signal_level = np.std(exp1, axis=1)
    noise_level = signal_level / snr

    exp2 = np.zeros_like(exp1)
    true_plan = np.zeros((n_genes, n_genes))

    diff_indices = np.random.choice(range(n_genes), size=int(diffgene * n_genes), replace=False)

    for i in range(n_genes):
        random_noise = np.random.normal(0, noise_level[i], n_samples)

        if i in diff_indices:
            n_parents = min(np.random.poisson(indegree - 1), n_genes - 1)
            parent_indices = np.random.choice(
                diff_indices[diff_indices != i], n_parents, replace=False
            )
            parent_indices = np.append(parent_indices, i)
            weights = np.random.choice([1, -1], n_parents + 1) * np.random.uniform(0.5, 2, n_parents + 1)
            weight_vector = np.zeros(n_genes)
            weight_vector[parent_indices] = weights
            true_plan[:, i] = weight_vector
            exp2[i] = np.sum(weight_vector.reshape(n_genes, 1) * _apply_nonlinear_transform(exp1), axis=0) + random_noise
        else:
            true_plan[i, i] = 1
            exp2[i] = exp1[i] + random_noise

    exp2 = np.abs(exp2)

    # Adding outliers to exp2
    if outlier_ratio > 0:
        n_outliers = int(outlier_ratio * n_genes * n_samples)
        gene_indices = np.random.choice(n_genes, n_outliers)
        sample_indices = np.random.choice(n_samples, n_outliers)
        outlier_types = np.random.choice(['heavy_tail', 'extreme', 'bernoulli'], size=n_outliers)

        for idx, (g_idx, s_idx) in enumerate(zip(gene_indices, sample_indices)):
            if outlier_types[idx] == 'heavy_tail':
                exp2[g_idx, s_idx] += np.random.standard_t(df=2) * noise_level[g_idx]
            elif outlier_types[idx] == 'extreme':
                exp2[g_idx, s_idx] += np.random.choice([3, -3]) * noise_level[g_idx]
            elif outlier_types[idx] == 'bernoulli':
                exp2[g_idx, s_idx] = np.random.binomial(1, 0.5)

    return np.abs(exp2), true_plan

def _normalize_expression_matrix(exp: np.ndarray) -> np.ndarray:
    """
    Normalize the expression matrix by column sums.

    Parameters
    ----------
    exp : np.ndarray
        Expression matrix (genes x samples).

    Returns
    -------
    np.ndarray
        Normalized expression matrix.
    """
    column_sums = exp.sum(axis=0)
    return np.mean(column_sums) * exp / column_sums
