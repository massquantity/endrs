from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def check_data_cols(
    data: Sequence[pd.DataFrame | None],
    user_col_name: str,
    item_col_name: str,
    label_col_name: str | None,
    multi_label_col_names: Sequence[str] | None,
):
    """Check if the necessary columns exist in the provided dataframes.

    Parameters
    ----------
    data : Sequence[pd.DataFrame | None]
        A sequence of dataframes (train, eval, test) to check.
    user_col_name : str
        Name of the user column.
    item_col_name : str
        Name of the item column.
    label_col_name : str or None
        Name of the label column. Can be None if using multi-label.
    multi_label_col_names : Sequence[str] or None
        Names of multiple label columns for multi-task learning. Can be None if using single label.

    Raises
    ------
    ValueError
        If any required column is missing from the dataframes.
    """
    def _checking(d: pd.DataFrame, name: str):
        if user_col_name not in d.columns:
            raise ValueError(f"`{user_col_name}` column does not exist in {name}.")
        if item_col_name not in d.columns:
            raise ValueError(f"`{item_col_name}` column does not exist in {name}.")

        assert label_col_name is not None or multi_label_col_names is not None
        if label_col_name and label_col_name not in d.columns:
            raise ValueError(f"`{label_col_name}` column does not exist in {name}.")
        if multi_label_col_names:
            if not isinstance(multi_label_col_names, list):
                raise ValueError(
                    f"multi_label_col_names must be list, got `{multi_label_col_names}`"
                )
            if len(multi_label_col_names) <= 1:
                raise ValueError(
                    f"multi_label_col_names must have length of at least 2, "
                    f"got `{multi_label_col_names}`"
                )
            for col in multi_label_col_names:
                if col not in d.columns:
                    raise ValueError(f"`{col}` column does not exist in {name}.")

    data_names = ["train_data", "eval_data", "test_data"]
    for n, d in zip(data_names, data):
        if d is not None:
            _checking(d, n)


def check_labels(task: str, labels: np.ndarray | None, neg_sampling: bool):
    """Check if labels are valid for the specified task and sampling strategy.

    Parameters
    ----------
    task : str
        The task type, e.g., 'ranking' or 'rating'.
    labels : np.ndarray or None
        The labels to check.
    neg_sampling : bool
        Whether negative sampling is being used.

    Raises
    ------
    ValueError
        If labels are None or not valid for the given task and sampling strategy.
    """
    if labels is None:
        raise ValueError("Missing label column for non-multi-task model.")
    if task == "ranking" and not neg_sampling:
        _check_binary_labels(labels)


def check_multi_labels(
    task: str, multi_labels: np.ndarray | None, neg_sampling: bool
):
    """Check if multi-labels are valid for multi-task learning.

    This function validates that the provided multi-labels array is appropriate for
    multi-task learning scenarios. It ensures that:
    1. The multi-labels array is not None
    2. For ranking tasks with negative sampling:
       - All labels must be either 0 or 1
       - Each sample must have at least one positive label (1)
    3. For ranking tasks without negative sampling:
       - Labels must be binary (0 and 1)
       - Both positive and negative labels must be present

    Parameters
    ----------
    task : str
        The task type, e.g., 'ranking' or 'rating'.
    multi_labels : np.ndarray or None
        The multi-labels to check. For multi-task learning, this is typically
        a 2D array where each row represents a sample and each column represents
        a different task/label.
    neg_sampling : bool
        Whether negative sampling is being used.

    Raises
    ------
    ValueError
        If multi_labels are None or not valid for the given task and sampling strategy.
    """
    if multi_labels is None:
        raise ValueError("Missing multi-label column for multi-task model.")
    if task == "ranking":
        if neg_sampling:
            # can be all pos or all neg labels
            if not np.all(np.logical_or(multi_labels == 0.0, multi_labels == 1.0)):
                raise ValueError(
                    f"For multi-task with negative sampling, labels in data must be 0 or 1, "
                    f"got unique labels: {np.unique(multi_labels).tolist()}"
                )
            # each sample has at least one pos label
            if not np.all(np.any(multi_labels == 1.0, axis=1)):
                raise ValueError(
                    f"For multi-task with negative sampling, each sample must have "
                    f"at least one positive label."
                )
        else:
            _check_binary_labels(multi_labels)


def _check_binary_labels(labels: np.ndarray):
    """Check if labels are binary (0 and 1).

    Parameters
    ----------
    labels : np.ndarray
        The labels to check.

    Raises
    ------
    ValueError
        If labels are not binary or don't contain both 0 and 1.
    """
    unique_labels = np.unique(labels)
    # must contain both pos and neg labels without negative sampling
    if (
        len(unique_labels) != 2
        or min(unique_labels) != 0.0
        or max(unique_labels) != 1.0
    ):
        raise ValueError(
            f"For ranking task without negative sampling, labels in data must be 0 or 1, "
            f"got unique labels: {unique_labels.tolist()}"
        )


def check_feat_cols(
    user_col_name: str,
    item_col_name: str,
    user_sparse_cols: Sequence[str] | None = None,
    item_sparse_cols: Sequence[str] | None = None,
    user_dense_cols: Sequence[str] | None = None,
    item_dense_cols: Sequence[str] | None = None
):
    """Check feature column names to prevent conflicts with user and item IDs.

    Parameters
    ----------
    user_col_name : str
        Name of the user column.
    item_col_name : str
        Name of the item column.
    user_sparse_cols : Sequence[str] or None, default: None
        Names of user sparse feature columns.
    item_sparse_cols : Sequence[str] or None, default: None
        Names of item sparse feature columns.
    user_dense_cols : Sequence[str] or None, default: None
        Names of user dense feature columns.
    item_dense_cols : Sequence[str] or None, default: None
        Names of item dense feature columns.

    Raises
    ------
    ValueError
        If user_col_name or item_col_name are included in any of the feature column lists.
    """
    cols = ["user_sparse_cols", "item_sparse_cols", "user_dense_cols", "item_dense_cols"]
    feats = [user_sparse_cols, item_sparse_cols, user_dense_cols, item_dense_cols]
    for col, feat in zip(cols, feats):
        if feat and user_col_name in feat:
            raise ValueError(f"`{user_col_name}` should not be included in `{col}`")
        if feat and item_col_name in feat:
            raise ValueError(f"`{item_col_name}` should not be included in `{col}`")


def check_feat_data(
    user_feat_data: pd.DataFrame | None,
    item_feat_data: pd.DataFrame | None,
    user_sparse_cols: Sequence[str] | None = None,
    item_sparse_cols: Sequence[str] | None = None,
    user_dense_cols: Sequence[str] | None = None,
    item_dense_cols: Sequence[str] | None = None,
    user_multi_sparse_cols: Sequence[Sequence[str]] | None = None,
    item_multi_sparse_cols: Sequence[Sequence[str]] | None = None,
):
    """Check that feature columns have corresponding feature data.

    Feature columns define what features to extract, but they require the actual
    feature data (DataFrame) to extract values from. This validation ensures that
    when feature columns are specified, the corresponding feature DataFrame is
    also provided, preventing runtime errors during feature extraction.

    Parameters
    ----------
    user_feat_data : pd.DataFrame or None
        DataFrame containing user features.
    item_feat_data : pd.DataFrame or None
        DataFrame containing item features.
    user_sparse_cols : Sequence[str] or None, default: None
        Names of user sparse feature columns.
    item_sparse_cols : Sequence[str] or None, default: None
        Names of item sparse feature columns.
    user_dense_cols : Sequence[str] or None, default: None
        Names of user dense feature columns.
    item_dense_cols : Sequence[str] or None, default: None
        Names of item dense feature columns.
    user_multi_sparse_cols : Sequence[Sequence[str]] or None, default: None
        Names of user multi-sparse feature columns.
    item_multi_sparse_cols : Sequence[Sequence[str]] or None, default: None
        Names of item multi-sparse feature columns.

    Raises
    ------
    ValueError
        If feature columns are provided but corresponding feature data is None.
    """
    if user_feat_data is None:
        if user_sparse_cols or user_dense_cols or user_multi_sparse_cols:
            raise ValueError(
                "user feature columns provided but user_feat_data is None"
            )
    if item_feat_data is None:
        if item_sparse_cols or item_dense_cols or item_multi_sparse_cols:
            raise ValueError(
                "item feature columns provided but item_feat_data is None"
            )


def check_lr_scheduler_config(
    lr_scheduler: str, lr_scheduler_config: dict[str, Any]
) -> dict[str, Any]:
    """Validate and extract required parameters for learning rate schedulers.

    Parameters
    ----------
    lr_scheduler : str
        The type of learning rate scheduler. Supported options are:
        - 'step': StepLR scheduler
        - 'exponential': ExponentialLR scheduler
        - 'cosine': CosineAnnealingLR scheduler
        - 'plateau': ReduceLROnPlateau scheduler
    lr_scheduler_config : dict[str, Any]
        Configuration dictionary containing scheduler-specific parameters.
        Required parameters vary by scheduler type:

        - For 'step': 'step_size' (int) and 'gamma' (float)
        - For 'exponential': 'gamma' (float)
        - For 'cosine': 'T_max' (int) and 'eta_min' (float)
        - For 'plateau': 'factor' (float) and 'patience' (int)

    Returns
    -------
    dict[str, Any]
        A dictionary containing only the validated required parameters for the
        specified scheduler type.

    Raises
    ------
    ValueError
        If the lr_scheduler type is not supported or if required parameters
        are missing from lr_scheduler_config for the specified scheduler type.
    """
    if lr_scheduler == "step":
        if "step_size" not in lr_scheduler_config:
            raise ValueError("step_size is required for StepLR scheduler.")
        elif "gamma" not in lr_scheduler_config:
            raise ValueError("gamma is required for StepLR scheduler.")
        return {
            "step_size": lr_scheduler_config["step_size"],
            "gamma": lr_scheduler_config["gamma"]
        }

    elif lr_scheduler == "exponential":
        if "gamma" not in lr_scheduler_config:
            raise ValueError("gamma is required for ExponentialLR scheduler.")
        return {"gamma": lr_scheduler_config["gamma"]}

    elif lr_scheduler == "cosine":
        if "T_max" not in lr_scheduler_config:
            raise ValueError("T_max is required for CosineAnnealingLR scheduler.")
        elif "eta_min" not in lr_scheduler_config:
            raise ValueError("eta_min is required for CosineAnnealingLR scheduler.")
        return {
            "T_max": lr_scheduler_config["T_max"],
            "eta_min": lr_scheduler_config["eta_min"]
        }

    elif lr_scheduler == "plateau":
        if "factor" not in lr_scheduler_config:
            raise ValueError("factor is required for ReduceLROnPlateau scheduler.")
        elif "patience" not in lr_scheduler_config:
            raise ValueError("patience is required for ReduceLROnPlateau scheduler.")
        return {
            "factor": lr_scheduler_config["factor"],
            "patience": lr_scheduler_config["patience"]
        }

    else:
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}. "
                         f"Supported options: 'step', 'exponential', 'cosine', 'plateau'")
