import itertools
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd


def extract_sparse_values(
    feat_data: pd.DataFrame,
    sparse_cols: Sequence[str],
    entity_name: Literal["user", "item"],
) -> dict[str, np.ndarray]:
    """Extract sorted unique values for each sparse column.

    Parameters
    ----------
    feat_data : pd.DataFrame
        Feature data containing sparse columns.
    sparse_cols : Sequence[str]
        Names of sparse columns to extract.
    entity_name : {'user', 'item'}
        Either "user" or "item", used for error messages.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping column names to sorted unique values.
    """
    result = {}
    for col in sparse_cols:
        if col in feat_data:
            result[col] = np.sort(feat_data[col].unique())
        else:
            raise ValueError(f"`{col}` does not exist in {entity_name} feat data.")
    return result


def extract_all_sparse_values(
    user_feat_data: pd.DataFrame | None,
    item_feat_data: pd.DataFrame | None,
    user_sparse_cols: Sequence[str] | None,
    item_sparse_cols: Sequence[str] | None,
) -> dict[str, np.ndarray]:
    """Extract unique values from both user and item sparse features.

    Parameters
    ----------
    user_feat_data : pd.DataFrame or None
        User feature data.
    item_feat_data : pd.DataFrame or None
        Item feature data.
    user_sparse_cols : Sequence[str] or None
        User sparse column names.
    item_sparse_cols : Sequence[str] or None
        Item sparse column names.

    Returns
    -------
    dict[str, np.ndarray]
        Combined dictionary of unique values from both user and item features.
    """
    user_unique = {}
    item_unique = {}

    if user_feat_data is not None and user_sparse_cols:
        user_unique = extract_sparse_values(
            user_feat_data, user_sparse_cols, "user"
        )

    if item_feat_data is not None and item_sparse_cols:
        item_unique = extract_sparse_values(
            item_feat_data, item_sparse_cols, "item"
        )

    return user_unique | item_unique


def extract_multi_sparse_values(
    feat_data: pd.DataFrame,
    multi_sparse_cols: Sequence[Sequence[str]],
    pad_val: int | str | list | tuple,
    entity_name: Literal["user", "item"],
) -> dict[str, np.ndarray]:
    """Extract sorted unique values for multi-sparse columns.

    The returned dict uses the first column name in each field (field[0])
    as the representative key.

    Parameters
    ----------
    feat_data : pd.DataFrame
        Feature data containing multi-sparse columns.
    multi_sparse_cols : Sequence[Sequence[str]]
        List of multi-sparse field groups.
    pad_val : int, str, list, or tuple
        Padding value(s) to exclude from unique values.
    entity_name : {'user', 'item'}
        Either "user" or "item", used for error messages.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping representative column names to sorted unique values.
    """
    result = {}

    for field in multi_sparse_cols:
        if not field:
            raise ValueError(
                f"{entity_name} multi_sparse_cols has invalid field: {field}"
            )

    if isinstance(pad_val, list | tuple):
        if len(multi_sparse_cols) != len(pad_val):
            raise ValueError(
                f"Length of {entity_name}_pad_val ({len(pad_val)}) must match "
                f"number of {entity_name}_multi_sparse_cols ({len(multi_sparse_cols)})"
            )
    else:
        # Scalar: auto-expand to list
        pad_val = [pad_val] * len(multi_sparse_cols)

    for i, field in enumerate(multi_sparse_cols):
        for col in field:
            if col not in feat_data:
                raise ValueError(
                    f"`{col}` does not exist in {entity_name} feat data."
                )

        values = feat_data[field].T.to_numpy().tolist()
        unique_vals = set(itertools.chain.from_iterable(values))
        if pad_val[i] in unique_vals:
            unique_vals.remove(pad_val[i])
        # use name of a field's first column as representative
        result[field[0]] = np.sort(list(unique_vals))

    return result


def extract_all_multi_sparse_values(
    user_feat_data: pd.DataFrame | None,
    item_feat_data: pd.DataFrame | None,
    user_multi_sparse_cols: Sequence[Sequence[str]] | None,
    item_multi_sparse_cols: Sequence[Sequence[str]] | None,
    user_pad_val: int | str | Sequence,
    item_pad_val: int | str | Sequence,
) -> dict[str, np.ndarray]:
    """Extract unique values from both user and item multi-sparse features.

    Parameters
    ----------
    user_feat_data : pd.DataFrame or None
        User feature data.
    item_feat_data : pd.DataFrame or None
        Item feature data.
    user_multi_sparse_cols : Sequence[Sequence[str]] or None
        User multi-sparse column groups.
    item_multi_sparse_cols : Sequence[Sequence[str]] or None
        Item multi-sparse column groups.
    user_pad_val : int, str, or Sequence
        Padding value(s) for user features.
    item_pad_val : int, str, or Sequence
        Padding value(s) for item features.

    Returns
    -------
    dict[str, np.ndarray]
        Combined dictionary of unique values from both user and item features.
    """
    user_unique = {}
    item_unique = {}

    if user_feat_data is not None and user_multi_sparse_cols:
        user_unique = extract_multi_sparse_values(
            user_feat_data, user_multi_sparse_cols, user_pad_val, "user"
        )

    if item_feat_data is not None and item_multi_sparse_cols:
        item_unique = extract_multi_sparse_values(
            item_feat_data, item_multi_sparse_cols, item_pad_val, "item"
        )

    return user_unique | item_unique
