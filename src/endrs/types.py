from typing import TYPE_CHECKING, TypeAlias, Union

if TYPE_CHECKING:
    from endrs.bases.dyn_embed_base import DynEmbedBase
    from endrs.bases.torch_base import TorchBase
    from endrs.bases.torch_embed_base import TorchEmbedBase

UserId: TypeAlias = Union[int, str]
ItemId: TypeAlias = Union[int, str]
RecModel: TypeAlias = Union["TorchBase", "TorchEmbedBase", "DynEmbedBase"]
