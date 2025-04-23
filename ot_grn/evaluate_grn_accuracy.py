# evaluate_grn_accuracy.py
# -*- coding: utf-8 -*-

from typing import Tuple
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def evaluate_grn_accuracy(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate performance metrics for the predicted gene regulatory network.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (genes x genes).
    y_score : np.ndarray
        Predicted scores (genes x genes).

    Returns
    -------
    Tuple[float, float, float]
        - AUROC: Area Under the Receiver Operating Characteristic Curve.
        - AUPR: Area Under the Precision-Recall Curve.
        - EP: Early Precision.
    """
    # mask the diagonal elements to ignore self-regulation
    n_genes = y_true.shape[0]
    mask = np.ones((n_genes, n_genes), dtype=bool)
    np.fill_diagonal(mask, False)

    top_num = int(np.sum(y_true[mask] == 1))

    auroc, aupr = _calculate_auroc_aupr(y_true, y_score, mask)
    ep = _calculate_early_precision(y_true, y_score, mask, top_num)
    return auroc, aupr, ep


def _calculate_auroc_aupr(
    y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate AUROC and AUPR for the predicted scores.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_score : np.ndarray
        Predicted scores.
    mask : np.ndarray
        Boolean mask indicating valid entries for evaluation.

    Returns
    -------
    Tuple[float, float]
        - AUROC.
        - AUPR.
    """
    y_true_flat = y_true[mask].flatten()
    y_score_flat = y_score[mask].flatten()

    fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
    auroc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true_flat, y_score_flat)
    aupr = auc(rec, prec)

    return auroc, aupr


def _calculate_early_precision(
    y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray, top_num: int
) -> float:
    """
    Calculate early precision for the top predicted scores.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_score : np.ndarray
        Predicted scores.
    mask : np.ndarray
        Boolean mask indicating valid entries for evaluation.
    top_num : int
        Number of top scores to consider for early precision.

    Returns
    -------
    float
        Early precision for the top predicted scores.
    """
    y_true_flat = y_true[mask].flatten()
    y_score_flat = y_score[mask].flatten()

    top_indices = np.argpartition(y_score_flat, -top_num)[-top_num:]
    top_true = y_true_flat[top_indices]
    early_precision = np.sum(top_true) / len(top_true)

    return early_precision
