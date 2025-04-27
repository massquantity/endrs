from collections.abc import Sequence

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
