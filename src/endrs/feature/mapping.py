from typing import Any

import numpy as np

from endrs.utils.hashing import Hasher


def build_sequential_mapping(
    unique_vals: dict[str, np.ndarray]
) -> dict[str, dict[Any, int]]:
    """Build sequential value-to-index mappings for each feature.

    Sequential mapping: value → [1, 2, 3, ...] (0 reserved for OOV).

    Parameters
    ----------
    unique_vals : dict[str, np.ndarray]
        Dictionary mapping column names to sorted unique values.

    Returns
    -------
    dict[str, dict[Any, int]]
        Nested dictionary: {column_name: {value: sequential_index}}.
        Indices start from 1, with 0 reserved for OOV.
    """
    result = {}
    for feat, vals in unique_vals.items():
        vals_list = vals.tolist()
        ids = range(1, len(vals_list) + 1)
        result[feat] = dict(zip(vals_list, ids))
    return result


def build_hash_mapping(
    hasher: Hasher,
    unique_vals: dict[str, np.ndarray],
) -> dict[str, dict[Any, int]]:
    """Build hash-based value-to-index mappings for each feature.

    Hash-based mapping: value → [1, n_hash_bins].

    Parameters
    ----------
    hasher : Hasher
        Hasher instance for computing hash values.
    unique_vals : dict[str, np.ndarray]
        Dictionary mapping column names to sorted unique values.

    Returns
    -------
    dict[str, dict[Any, int]]
        Nested dictionary: {column_name: {value: hash_index}}.
        Hash indices are in range [1, n_hash_bins].
    """
    return {
        col: hasher.to_hash_mapping(col, vals.tolist())
        for col, vals in unique_vals.items()
    }


def update_hash_mapping(
    hasher: Hasher,
    existing: dict[str, dict[Any, int]],
    new_vals: dict[str, np.ndarray],
) -> dict[str, dict[Any, int]]:
    """Update existing hash mappings with new values (for incremental training).

    Parameters
    ----------
    hasher : Hasher
        Hasher instance for computing hash values.
    existing : dict[str, dict[Any, int]]
        Existing value-to-index mappings.
    new_vals : dict[str, np.ndarray]
        New unique values to add to the mappings.

    Returns
    -------
    dict[str, dict[Any, int]]
        Updated mappings containing both existing and new values.
    """
    for col, vals in new_vals.items():
        val_to_idx = hasher.to_hash_mapping(col, vals.tolist())
        existing[col].update(val_to_idx)
    return existing
