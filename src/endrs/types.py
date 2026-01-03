from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from endrs.bases.cf_base import CfBase
    from endrs.bases.dyn_embed_base import DynEmbedBase
    from endrs.bases.torch_base import TorchBase
    from endrs.bases.torch_embed_base import TorchEmbedBase
    from endrs_ext import ItemCF as RsItemCF, Swing as RsSwing, UserCF as RsUserCF

type UserId = int | str
type ItemId = int | str
type RustModel = RsUserCF | RsItemCF | RsSwing
type RecModel = TorchBase | TorchEmbedBase | DynEmbedBase | CfBase
