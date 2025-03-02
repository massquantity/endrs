from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass

import numpy as np

from endrs.types import ItemId, UserId
from endrs.utils.constants import OOV_IDX


@dataclass
class IdConverter:
    user2id: MutableMapping[UserId, int]
    item2id: MutableMapping[ItemId, int]
    id2user: MutableMapping[int, UserId]
    id2item: MutableMapping[int, ItemId]

    def safe_user_to_id(self, user: UserId) -> int:
        return self.user2id.get(user, OOV_IDX)

    def safe_item_to_id(self, item: ItemId) -> int:
        return self.item2id.get(item, OOV_IDX)

    @property
    def unique_users(self) -> list[UserId]:
        return list(self.user2id)

    @property
    def unique_items(self) -> list[ItemId]:
        return list(self.item2id)

    def update(
        self,
        user2id: MutableMapping[UserId, int],
        item2id: MutableMapping[ItemId, int],
        id2user: MutableMapping[int, UserId],
        id2item: MutableMapping[int, ItemId],
    ):
        self.user2id.update(user2id)
        self.item2id.update(item2id)
        self.id2user.update(id2user)
        self.id2item.update(id2item)
        return self


@dataclass
class DataInfo:
    data_size: int
    user_col_name: str
    item_col_name: str
    label_col_name: str | None
    multi_label_col_name: Sequence[str] | None
    id_converter: IdConverter
    user_consumed: Mapping[int, Sequence[int]]
    item_consumed: Mapping[int, Sequence[int]]
    pop_items: list[ItemId]
    use_hash: bool = False
    n_hash_bins: int = 200_000

    @property
    def n_users(self) -> int:
        return len(self.id_converter.user2id)

    @property
    def n_items(self) -> int:
        return len(self.id_converter.item2id)

    @property
    def candidate_users(self) -> np.ndarray:
        return np.array(list(self.id_converter.id2user))

    @property
    def candidate_items(self) -> np.ndarray:
        return np.array(list(self.id_converter.id2item))

    def __repr__(self):
        density = 100 * self.data_size / (self.n_users * self.n_items)
        return (
            f"train data_size: {self.data_size}, n_users: {self.n_users}, "
            f"n_items: {self.n_items}, data density: {density:.4f} %"
        )
