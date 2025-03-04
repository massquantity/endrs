"""Utility Functions for Splitting Data."""
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def random_split(
    data: pd.DataFrame,
    user_col_name: str,
    item_col_name: str,
    shuffle: bool = True,
    test_size: float | None = None,
    multi_ratios: Sequence[float] | None = None,
    filter_unknown: bool = True,
    pad_unknown: bool = False,
    pad_val: Any = None,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Split the data randomly.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to split.
    user_col_name : str
        User column name in original data.
    item_col_name : str
        Item column name in original data.
    shuffle : bool, default: True
        Whether to shuffle data when splitting.
    test_size : float or None, default: None
        Test data ratio.
    multi_ratios : list of float, tuple of (float,) or None, default: None
        Ratios for splitting data in multiple parts. If ``test_size`` is not None,
        ``multi_ratios`` will be ignored.
    filter_unknown : bool, default: True
        Whether to filter out users and items that don't appear in the train data
        from eval and test data. Since models can only recommend items in the train data.
    pad_unknown : bool, default: False
        Fill the unknown users/items with ``pad_val``. If ``filter_unknown`` is True,
        this parameter will be ignored.
    pad_val : any, default: None
        Pad value used in ``pad_unknown``.
    seed : int, default: 42
        Random seed.

    Returns
    -------
    multiple data : list of pandas.DataFrame
        The split data.

    Raises
    ------
    ValueError
        If neither `test_size` nor `multi_ratio` is provided.

    Examples
    --------
    >>> train, test = random_split(data, test_size=0.2)
    >>> train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])
    """
    ratios, n_splits = _check_and_convert_ratio(test_size, multi_ratios)
    if not isinstance(ratios, list):
        ratios = list(ratios)

    train_data = data.copy()
    split_data_all = []
    for _ in range(n_splits - 1):
        size = ratios.pop(-1)
        ratios = [r / math.fsum(ratios) for r in ratios]
        train_data, split_data = train_test_split(
            train_data, test_size=size, shuffle=shuffle, random_state=seed
        )
        split_data_all.insert(0, split_data)
    split_data_all.insert(0, train_data)  # insert final fold of data

    if filter_unknown:
        split_data_all = _filter_unknown_user_item(
            split_data_all, user_col_name, item_col_name
        )
    elif pad_unknown and pad_val is not None:
        split_data_all = _pad_unknown_user_item(
            split_data_all, pad_val, user_col_name, item_col_name
        )
    return split_data_all


def _filter_unknown_user_item(
    data_list: Sequence[pd.DataFrame], user_col_name: str, item_col_name: str
) -> list[pd.DataFrame]:
    train_data = data_list[0]
    unique_users = set(train_data[user_col_name].tolist())
    unique_items = set(train_data[item_col_name].tolist())
    split_data_all = [train_data]
    for test_data in data_list[1:]:
        oov_user_indices = [
            j for j, v in enumerate(test_data[user_col_name]) if v not in unique_users
        ]
        oov_item_indices = [
            j for j, v in enumerate(test_data[item_col_name]) if v not in unique_items
        ]
        oov_indices = list(set(oov_user_indices + oov_item_indices))
        mask = np.arange(len(test_data))
        test_data_clean = test_data[~np.isin(mask, oov_indices)]
        split_data_all.append(test_data_clean)
    return split_data_all


def _pad_unknown_user_item(
    data_list: Sequence[pd.DataFrame],
    pad_val: Any,
    user_col_name: str,
    item_col_name: str,
) -> list[pd.DataFrame]:
    train_data = data_list[0]
    if isinstance(pad_val, list | tuple):
        user_pad_val, item_pad_val = pad_val
    else:
        user_pad_val = item_pad_val = pad_val
    unique_users = set(train_data[user_col_name].tolist())
    unique_items = set(train_data[item_col_name].tolist())
    split_data_all = [train_data]
    for test_data in data_list[1:]:
        test_data_copy = test_data.copy()
        user_mask = ~test_data[user_col_name].isin(unique_users)
        test_data_copy.loc[user_mask, user_col_name] = user_pad_val
        item_mask = ~test_data[item_col_name].isin(unique_items)
        test_data_copy.loc[item_mask, item_col_name] = item_pad_val
        test_data = test_data_copy
        split_data_all.append(test_data)
    return split_data_all


def split_by_ratio(
    data: pd.DataFrame,
    user_col_name: str,
    item_col_name: str,
    order: bool = True,
    shuffle: bool = False,
    test_size: float | None = None,
    multi_ratios: Sequence[float] | None = None,
    filter_unknown: bool = True,
    pad_unknown: bool = False,
    pad_val: Any = None,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Assign certain ratio of items to test data for each user.

    .. NOTE::
        If a user's total # of interacted items is less than 3, these items will all been
        assigned to train data.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to split.
    user_col_name : str
        User column name in original data.
    item_col_name : str
        Item column name in original data.
    order : bool, default: True
        Whether to preserve order for user's item sequence.
    shuffle : bool, default: False
        Whether to shuffle data after splitting.
    test_size : float or None, default: None
        Test data ratio.
    multi_ratios : list of float, tuple of (float,) or None, default: None
        Ratios for splitting data in multiple parts. If ``test_size`` is not None,
        ``multi_ratios`` will be ignored.
    filter_unknown : bool, default: True
        Whether to filter out users and items that don't appear in the train data
        from eval and test data. Since models can only recommend items in the train data.
    pad_unknown : bool, default: False
        Fill the unknown users/items with ``pad_val``. If ``filter_unknown`` is True,
        this parameter will be ignored.
    pad_val : any, default: None
        Pad value used in ``pad_unknown``.
    seed : int, default: 42
        Random seed.

    Returns
    -------
    multiple data : list of pandas.DataFrame
        The split data.

    Raises
    ------
    ValueError
        If neither `test_size` nor `multi_ratio` is provided.

    See Also
    --------
    split_by_ratio_chrono
    """
    assert user_col_name in data, f"data must contains `{user_col_name}` column."
    ratios, n_splits = _check_and_convert_ratio(test_size, multi_ratios)

    n_users = data[user_col_name].nunique()
    user_indices = data[user_col_name].to_numpy()
    user_split_indices = _groupby_user(user_indices, order)

    cum_ratios = np.cumsum(ratios).tolist()[:-1]
    split_indices_all = [[] for _ in range(n_splits)]
    for u in range(n_users):
        u_data = user_split_indices[u]
        u_data_len = len(u_data)
        if u_data_len < 3:  # keep items of rare users in trainset
            split_indices_all[0].extend(u_data.tolist())
        else:
            u_split_data = np.split(
                u_data, [round(cum * u_data_len) for cum in cum_ratios]
            )
            for i in range(n_splits):
                split_indices_all[i].extend(u_split_data[i].tolist())

    if shuffle:
        np_rng = np.random.default_rng(seed)
        split_data_all = tuple(
            data.iloc[np_rng.permutation(idx)] for idx in split_indices_all
        )
    else:
        split_data_all = list(data.iloc[idx] for idx in split_indices_all)

    if filter_unknown:
        split_data_all = _filter_unknown_user_item(
            split_data_all, user_col_name, item_col_name
        )
    elif pad_unknown and pad_val is not None:
        split_data_all = _pad_unknown_user_item(
            split_data_all, pad_val, user_col_name, item_col_name
        )
    return split_data_all


def split_by_num(
    data: pd.DataFrame,
    user_col_name: str,
    item_col_name: str,
    order: bool = True,
    shuffle: bool = False,
    test_size: int = 1,
    filter_unknown: bool = True,
    pad_unknown: bool = False,
    pad_val: Any = None,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Assign a certain number of items to test data for each user.

    .. NOTE::
        If a user's total # of interacted items is less than 3, these items will all been
        assigned to train data.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to split.
    user_col_name : str
        User column name in original data.
    item_col_name : str
        Item column name in original data.
    order : bool, default: True
        Whether to preserve order for user's item sequence.
    shuffle : bool, default: False
        Whether to shuffle data after splitting.
    test_size : float or None, default: None
        Test data ratio.
    filter_unknown : bool, default: True
        Whether to filter out users and items that don't appear in the train data
        from eval and test data. Since models can only recommend items in the train data.
    pad_unknown : bool, default: False
        Fill the unknown users/items with ``pad_val``. If ``filter_unknown`` is True,
        this parameter will be ignored.
    pad_val : any, default: None
        Pad value used in ``pad_unknown``.
    seed : int, default: 42
        Random seed.

    Returns
    -------
    multiple data : list of pandas.DataFrame
        The split data.

    Raises
    ------
    ValueError
        If neither `test_size` nor `multi_ratio` is provided.

    See Also
    --------
    split_by_num_chrono
    """
    assert user_col_name in data, f"data must contains `{user_col_name}` column."
    assert isinstance(test_size, int), "test_size must be int value."
    assert 0 < test_size < len(data), "test_size must be in range (0, len(data))."

    n_users = data[user_col_name].nunique()
    user_indices = data[user_col_name].to_numpy()
    user_split_indices = _groupby_user(user_indices, order)

    train_indices = []
    test_indices = []
    for u in range(n_users):
        u_data = user_split_indices[u]
        u_data_len = len(u_data)
        if u_data_len < 3:  # keep items of rare users in trainset
            train_indices.extend(u_data)
        elif u_data_len <= test_size:
            train_indices.extend(u_data[:-1])
            test_indices.extend(u_data[-1:])
        else:
            k = test_size
            train_indices.extend(u_data[: (u_data_len - k)])
            test_indices.extend(u_data[-k:])

    if shuffle:
        np_rng = np.random.default_rng(seed)
        train_indices = np_rng.permutation(train_indices)
        test_indices = np_rng.permutation(test_indices)

    split_data_all = (data.iloc[train_indices], data.iloc[test_indices])
    if filter_unknown:
        split_data_all = _filter_unknown_user_item(
            split_data_all, user_col_name, item_col_name
        )
    elif pad_unknown and pad_val is not None:
        split_data_all = _pad_unknown_user_item(
            split_data_all, pad_val, user_col_name, item_col_name
        )
    return split_data_all


def split_by_ratio_chrono(
    data: pd.DataFrame,
    user_col_name: str,
    item_col_name: str,
    order: bool = True,
    shuffle: bool = False,
    test_size: float | None = None,
    multi_ratios: Sequence[float] | None = None,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Assign a certain ratio of items to test data for each user, where items are sorted by time first.

    .. IMPORTANT::
        This function implies the data should contain a **time** column.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to split.
    user_col_name : str
        User column name in original data.
    item_col_name : str
        Item column name in original data.
    order : bool, default: True
        Whether to preserve order for user's item sequence.
    shuffle : bool, default: False
        Whether to shuffle data after splitting.
    test_size : float or None, default: None
        Test data ratio.
    multi_ratios : list of float, tuple of (float,) or None, default: None
        Ratios for splitting data in multiple parts. If ``test_size`` is not None,
        ``multi_ratios`` will be ignored.
    seed : int, default: 42
        Random seed.

    Returns
    -------
    multiple data : list of pandas.DataFrame
        The split data.

    Raises
    ------
    ValueError
        If neither `test_size` nor `multi_ratio` is provided.

    See Also
    --------
    split_by_ratio
    """
    assert all(
        [user_col_name in data.columns, "time" in data.columns]
    ), f"data must contains {user_col_name} and time column"

    data = data.sort_values(by=["time"]).reset_index(drop=True)
    return split_by_ratio(
        data,
        user_col_name,
        item_col_name,
        order,
        shuffle,
        test_size,
        multi_ratios,
        seed=seed,
    )


def split_by_num_chrono(
    data: pd.DataFrame,
    user_col_name: str,
    item_col_name: str,
    order: bool = True,
    shuffle: bool = False,
    test_size: int = 1,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Assign a certain number of items to test data for each user, where items are sorted by time first.

    .. IMPORTANT::
        This function implies the data should contain a **time** column.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to split.
    user_col_name : str
        User column name in original data.
    item_col_name : str
        Item column name in original data.
    order : bool, default: True
        Whether to preserve order for user's item sequence.
    shuffle : bool, default: False
        Whether to shuffle data after splitting.
    test_size : float or None, default: None
        Test data ratio.
    seed : int, default: 42
        Random seed.

    Returns
    -------
    multiple data : list of pandas.DataFrame
        The split data.

    Raises
    ------
    ValueError
        If neither `test_size` nor `multi_ratio` is provided.

    See Also
    --------
    split_by_num
    """
    assert all(
        [user_col_name in data.columns, "time" in data.columns]
    ), f"data must contains {user_col_name} and time column"

    data = data.sort_values(by=["time"]).reset_index(drop=True)
    return split_by_num(
        data, user_col_name, item_col_name, order, shuffle, test_size, seed=seed
    )


def _groupby_user(user_indices: np.ndarray, order: bool) -> list[np.ndarray]:
    sort_kind = "mergesort" if order else "quicksort"
    _, user_position, user_counts = np.unique(
        user_indices, return_inverse=True, return_counts=True
    )
    user_split_indices = np.split(
        np.argsort(user_position, kind=sort_kind), np.cumsum(user_counts)[:-1]
    )
    return user_split_indices


def _check_and_convert_ratio(
    test_size: float | None, multi_ratios: Sequence[float] | None
) -> tuple[Sequence[float], int]:
    if not test_size and not multi_ratios:
        raise ValueError("must provide either 'test_size' or 'multi_ratios'")

    elif test_size is not None:
        assert isinstance(test_size, float), "test_size must be float value"
        assert 0.0 < test_size < 1.0, "test_size must be in (0.0, 1.0)"
        ratios = [1 - test_size, test_size]
        return ratios, 2

    elif isinstance(multi_ratios, list | tuple):
        assert len(multi_ratios) > 1, "multi_ratios must at least have two elements"
        assert all([r > 0.0 for r in multi_ratios]), "ratios should be positive values"
        if math.fsum(multi_ratios) != 1.0:
            ratios = [r / math.fsum(multi_ratios) for r in multi_ratios]
        else:
            ratios = multi_ratios
        return ratios, len(ratios)

    else:
        raise ValueError("multi_ratios should be list or tuple")
