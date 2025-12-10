import numpy as np
import torch

from endrs.data.data_info import IdConverter
from endrs.feature.feat_info import FeatInfo
from endrs.types import ItemId, UserId
from endrs.utils.constants import ITEM_KEY, OOV_IDX, SEQ_KEY, USER_KEY
from endrs.utils.misc import colorize


def get_user_inputs(
    users: np.ndarray,
    feat_info: FeatInfo | None,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Create input tensors for user data.

    Parameters
    ----------
    users : np.ndarray
        Array of user IDs.
    feat_info : :class:`~endrs.feature.FeatInfo` or None
        Object that contains information about features used in the model.
    device : torch.device or None, default: None
        Device to which the tensors will be moved.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary mapping input keys to their corresponding tensors.
    """
    inputs = {USER_KEY: users}
    if feat_info and feat_info.user_feats:
        for feat in feat_info.user_feats:
            inputs[feat] = feat_info.feat_unique[feat][users]
    return {k: torch.as_tensor(v, device=device) for k, v in inputs.items()}


def get_item_inputs(
    items: np.ndarray,
    feat_info: FeatInfo | None,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Create input tensors for item data.

    Parameters
    ----------
    items : np.ndarray
        Array of item IDs.
    feat_info : :class:`~endrs.feature.FeatInfo` or None
        Object that contains information about features used in the model.
    device : torch.device or None, default: None
        Device to which the tensors will be moved.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary mapping input keys to their corresponding tensors.
    """
    inputs = {ITEM_KEY: items}
    if feat_info and feat_info.item_feats:
        for feat in feat_info.item_feats:
            inputs[feat] = feat_info.feat_unique[feat][items]
    return {k: torch.as_tensor(v, device=device) for k, v in inputs.items()}


def get_seq_inputs(
    inputs: dict[str, torch.Tensor],
    seqs: np.ndarray,
    feat_info: FeatInfo | None,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Add sequence data to the input tensors.

    This function will modify the `inputs` inplace.

    Parameters
    ----------
    inputs : dict[str, torch.Tensor]
        Dictionary of existing input tensors.
    seqs : np.ndarray
        Array of sequence data.
    feat_info : :class:`~endrs.feature.FeatInfo` or None
        Object that contains information about features used in the model.
    device : torch.device or None, default: None
        Device to which the tensors will be moved.

    Returns
    -------
    dict[str, torch.Tensor]
        Updated dictionary with sequence inputs added.
    """
    inputs[SEQ_KEY] = torch.as_tensor(seqs, device=device)
    if feat_info and feat_info.item_feats:
        for feat in feat_info.item_feats:
            feat_key = SEQ_KEY + feat
            inputs[feat_key] = torch.as_tensor(
                feat_info.feat_unique[feat][seqs], device=device
            )
    return inputs


def convert_ids(
    user: UserId | list[UserId] | np.ndarray,
    item: ItemId | list[ItemId] | np.ndarray,
    id_converter: IdConverter,
    inner_id: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert external IDs to internal IDs if needed.

    Parameters
    ----------
    user : UserId or list[UserId] or np.ndarray
        User ID(s) to convert.
    item : UserId or list[UserId] or np.ndarray
        Item ID(s) to convert.
    id_converter : :class:`~endrs.data.data_info.IdConverter`
        Converter between internal and original IDs.
    inner_id : bool
        Whether the provided IDs are already internal IDs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing converted user and item ID arrays.
    """
    user = [user] if np.isscalar(user) else user
    item = [item] if np.isscalar(item) else item
    if not inner_id:
        user = [id_converter.safe_user_to_id(u) for u in user]
        item = [id_converter.safe_item_to_id(i) for i in item]
    return np.array(user), np.array(item)


def get_unknown(user: np.ndarray, item: np.ndarray) -> tuple[int, list[int]]:
    """Identify unknown users and items in the input data for further prediction.

    Parameters
    ----------
    user : np.ndarray
        Array of user IDs.
    item : np.ndarray
        Array of item IDs.

    Returns
    -------
    tuple[int, list[int]]
        Tuple containing the number of unknown IDs and their indices.
    """
    unknown_user_indices = list(np.where(user == OOV_IDX)[0])
    unknown_item_indices = list(np.where(item == OOV_IDX)[0])
    unknown_index = list(set(unknown_user_indices) | set(unknown_item_indices))
    unknown_num = len(unknown_index)
    if unknown_num > 0:
        unknown_str = (
            f"Detect {unknown_num} unknown interaction(s) during prediction, "
            f"position: {unknown_index}"
        )
        print(f"{colorize(unknown_str, 'red')}")
    return unknown_num, unknown_index


def sep_unknown_users(
    id_converter: IdConverter,
    user: UserId | list[UserId] | np.ndarray,
    inner_id: bool,
) -> tuple[list[int], list[UserId]]:
    """Separate known and unknown users for further recommendation.

    Parameters
    ----------
    id_converter : :class:`~endrs.data.data_info.IdConverter`
        Converter between internal and original IDs.
    user : UserId or list[UserId] or np.ndarray
        User ID(s) to check.
    inner_id : bool
        Whether the provided IDs are already internal IDs.

    Returns
    -------
    tuple[list[int], list[UserId]]
        Tuple containing lists of known user IDs (as internal IDs) and unknown user IDs.
    """
    known_users_ids, unknown_users = [], []
    users = [user] if np.isscalar(user) else user
    for u in users:
        if inner_id:
            if u in id_converter.id2user:
                known_users_ids.append(u)
            else:
                unknown_users.append(u)
        else:
            if u in id_converter.user2id:
                known_users_ids.append(id_converter.user2id[u])
            else:
                unknown_str = f"Detect unknown user during recommendation: {u}"
                print(f"{colorize(unknown_str, 'red')}")
                unknown_users.append(u)
    return known_users_ids, unknown_users
