from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from endrs.utils.constants import OOV_IDX


def build_sparse_arrays(
    feat_data: pd.DataFrame,
    sparse_cols: Sequence[str],
    val_to_idx: dict[str, dict[Any, int]],
    entity_ids: np.ndarray,
    n_entities: int,
) -> dict[str, np.ndarray]:
    """Build sparse feature arrays with OOV_IDX at position 0.

    Creates 1D arrays where position 0 holds OOV_IDX, and subsequent
    positions correspond to internal entity IDs (1, 2, 3, ...).

    Parameters
    ----------
    feat_data : pd.DataFrame
        Feature data sorted by entity ID.
    sparse_cols : Sequence[str]
        Names of sparse columns to process.
    val_to_idx : dict[str, dict[Any, int]]
        Value-to-index mappings for each column.
    entity_ids : np.ndarray
        Internal entity IDs from feat_data for position-based indexing.
    n_entities : int
        Total number of entities (used for array size alignment).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping column names to feature arrays.
    """
    feat_unique = {}
    if not sparse_cols:
        return feat_unique

    for col in sparse_cols:
        features = feat_data[col].tolist()
        mapping = val_to_idx[col]
        arr = np.full(n_entities + 1, OOV_IDX, dtype=np.int64)
        # Place actual values at correct positions based on entity_ids
        arr[entity_ids] = [mapping[v] for v in features]
        feat_unique[col] = arr

    return feat_unique


def build_dense_arrays(
    feat_data: pd.DataFrame,
    dense_cols: Sequence[str],
    entity_ids: np.ndarray,
    n_entities: int,
) -> dict[str, np.ndarray]:
    """Build dense feature arrays with median as OOV value at position 0.

    Creates 1D arrays where position 0 holds the median value (for OOV
    entities), and subsequent positions correspond to internal entity IDs.

    Parameters
    ----------
    feat_data : pd.DataFrame
        Feature data sorted by entity ID.
    dense_cols : Sequence[str]
        Names of dense columns to process.
    entity_ids : np.ndarray
        Internal entity IDs from feat_data for position-based indexing.
    n_entities : int
        Total number of entities (used for array size alignment).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping column names to feature arrays (float32).
    """
    feat_unique = {}
    if not dense_cols:
        return feat_unique

    for col in dense_cols:
        oov = feat_data[col].median()
        arr = np.full(n_entities + 1, oov, dtype=np.float32)
        # Place actual values at correct positions based on entity_ids
        arr[entity_ids] = feat_data[col].to_numpy(dtype=np.float32)
        feat_unique[col] = arr

    return feat_unique


def build_multi_sparse_arrays(
    feat_data: pd.DataFrame,
    multi_sparse_cols: Sequence[Sequence[str]],
    val_to_idx: dict[str, dict[Any, int]],
    entity_ids: np.ndarray,
    n_entities: int,
) -> dict[str, np.ndarray]:
    """Build multi-sparse feature matrices with OOV row at position 0.

    Creates 2D matrices of shape (n_entities + 1, n_fields), where row 0
    is the OOV row filled with OOV_IDX, and subsequent rows correspond
    to internal entity IDs.

    Parameters
    ----------
    feat_data : pd.DataFrame
        Feature data sorted by entity ID.
    multi_sparse_cols : Sequence[Sequence[str]]
        List of multi-sparse field groups.
    val_to_idx : dict[str, dict[Any, int]]
        Value-to-index mappings for each field.
    entity_ids : np.ndarray
        Internal entity IDs from feat_data for position-based indexing.
    n_entities : int
        Total number of entities (used for array size alignment).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping representative column names to 2D feature matrices.
    """
    feat_unique = {}
    if not multi_sparse_cols:
        return feat_unique

    for field in multi_sparse_cols:
        col_repr = field[0]
        shape = (n_entities + 1, len(field))
        data = np.full(shape, OOV_IDX)
        for i, col in enumerate(field):
            features = feat_data[col].tolist()
            mapping = val_to_idx[col_repr]
            # Place actual values at correct positions based on entity_ids
            data[entity_ids, i] = [mapping.get(v, OOV_IDX) for v in features]
        feat_unique[col_repr] = data

    return feat_unique


def build_hash_sparse_arrays(
    feat_data: pd.DataFrame,
    sparse_cols: Sequence[str],
    val_to_idx: dict[str, dict[Any, int]],
    id_col: str,
    n_bins: int,
    existing: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Build hash-indexed sparse feature arrays of size n_hash_bins + 1.

    Creates arrays of size (n_hash_bins + 1) indexed by hash values.
    This allows direct indexing via feat_array[hash_id] for any entity.

    In retrain mode, existing arrays are reused and only positions
    corresponding to new data are updated.

    Parameters
    ----------
    feat_data : pd.DataFrame
        Feature data with ID column mapped to hash values.
    sparse_cols : Sequence[str]
        Names of sparse columns to process.
    val_to_idx : dict[str, dict[Any, int]]
        Hash-based value-to-index mappings.
    id_col : str
        Name of the ID column in feat_data.
    n_bins : int
        Number of hash bins.
    existing : dict[str, np.ndarray] or None
        Existing feature arrays to update (for incremental training).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping column names to hash-indexed feature arrays.
    """
    feat_unique = existing if existing is not None else {}
    ids = feat_data[id_col].to_numpy()

    for col in sparse_cols:
        mapping = val_to_idx[col]
        data = feat_data[col].map(mapping).to_numpy()

        if existing is not None and col in existing:
            # Reuse existing array, update positions for new data
            all_hash_data = existing[col]
        else:
            # Create new array filled with OOV_IDX
            all_hash_data = np.full(n_bins + 1, OOV_IDX)

        all_hash_data[ids] = data
        feat_unique[col] = all_hash_data

    return feat_unique
