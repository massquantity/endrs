from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EntityFeatureConfig:
    """Feature configuration for a single entity type (user or item).

    Parameters
    ----------
    sparse_cols : Sequence[str] or None
        Names of sparse categorical columns.
    dense_cols : Sequence[str] or None
        Names of dense numerical columns.
    multi_sparse_cols : Sequence[Sequence[str]] or None
        Names of multi-value sparse column groups.
    pad_val : int, str, or Sequence
        Padding value(s) for multi-sparse features.
        If scalar, applied to all multi-sparse fields.
        If sequence, length must match number of multi_sparse_cols.
    """

    sparse_cols: Sequence[str] | None = None
    dense_cols: Sequence[str] | None = None
    multi_sparse_cols: Sequence[Sequence[str]] | None = None
    pad_val: int | str | Sequence = "missing"

    @property
    def has_sparse(self) -> bool:
        """Check if sparse columns are defined."""
        return bool(self.sparse_cols)

    @property
    def has_dense(self) -> bool:
        """Check if dense columns are defined."""
        return bool(self.dense_cols)

    @property
    def has_multi_sparse(self) -> bool:
        """Check if multi-sparse columns are defined."""
        return bool(self.multi_sparse_cols)


@dataclass(frozen=True)
class FeatureConfig:
    """Feature configuration for both user and item entities.

    Parameters
    ----------
    user : EntityFeatureConfig
        Configuration for user features.
    item : EntityFeatureConfig
        Configuration for item features.

    Examples
    --------
    >>> config = FeatureConfig(
    ...     user=EntityFeatureConfig(
    ...         sparse_cols=["gender", "occupation"],
    ...         dense_cols=["age"],
    ...     ),
    ...     item=EntityFeatureConfig(
    ...         sparse_cols=["genre"],
    ...         multi_sparse_cols=[["tag1", "tag2", "tag3"]],
    ...         pad_val="missing",
    ...     ),
    ... )
    """

    user: EntityFeatureConfig = field(default_factory=EntityFeatureConfig)
    item: EntityFeatureConfig = field(default_factory=EntityFeatureConfig)

    @classmethod
    def from_flat_params(
        cls,
        user_sparse_cols: Sequence[str] | None = None,
        item_sparse_cols: Sequence[str] | None = None,
        user_dense_cols: Sequence[str] | None = None,
        item_dense_cols: Sequence[str] | None = None,
        user_multi_sparse_cols: Sequence[Sequence[str]] | None = None,
        item_multi_sparse_cols: Sequence[Sequence[str]] | None = None,
        user_pad_val: int | str | Sequence = "missing",
        item_pad_val: int | str | Sequence = "missing",
    ) -> "FeatureConfig":
        """Create FeatureConfig from flat parameters."""
        return cls(
            user=EntityFeatureConfig(
                sparse_cols=user_sparse_cols,
                dense_cols=user_dense_cols,
                multi_sparse_cols=user_multi_sparse_cols,
                pad_val=user_pad_val,
            ),
            item=EntityFeatureConfig(
                sparse_cols=item_sparse_cols,
                dense_cols=item_dense_cols,
                multi_sparse_cols=item_multi_sparse_cols,
                pad_val=item_pad_val,
            ),
        )
