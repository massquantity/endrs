from collections.abc import Sequence
from typing import Any

import xxhash


class Hasher:
    """Feature hashing class for converting categorical features to integer indices.

    Parameters
    ----------
    num_bins : int
        Number of bins to hash into. Must be greater than 1.
    seed : int
        Random seed for the hash function to ensure reproducibility.

    Attributes
    ----------
    range_min : int
        Minimum value of the hashing range, set to 1 to skip the OOV index 0.
    range_max : int
        Maximum value of the hashing range, equal to num_bins.
    range_size : int
        Size of the range for hashing, calculated as range_max - range_min + 1.
    """

    def __init__(self, num_bins: int, seed: int):
        assert num_bins > 1
        self.range_min = 1
        self.range_max = num_bins
        self.range_size = self.range_max - self.range_min + 1
        self.seed = seed

    def str_to_int(self, feat: str, value: str) -> int:
        """Convert a feature-value pair to an integer using xxhash.

        The hash function maps the input string to an integer within the range
        [range_min, range_max] (inclusive).

        Parameters
        ----------
        feat : str
            The feature name.
        value : str
            The feature value to hash.

        Returns
        -------
        int
            The hashed integer value within the specified range.
        """
        hash_key = f"{feat}_{value}"
        hash_val = xxhash.xxh3_64_intdigest(hash_key, self.seed)
        return self.range_min + (hash_val % self.range_size)

    def to_hash_mapping(
        self, feat: str, values: Sequence[Any], include_reverse: bool = False
    ) -> dict[Any, int] | tuple[dict[Any, int], dict[int, Any]]:
        """Create a mapping from feature values to hashed integers.

        Parameters
        ----------
        feat : str
            The feature name.
        values : Sequence[Any]
            The sequence of feature values to hash.
        include_reverse : bool, default: False
            Whether to include the reverse mapping from integers back to original values.

        Returns
        -------
        dict[Any, int] or tuple[dict[Any, int], dict[int, Any]]
            If include_reverse is False, returns a dictionary mapping feature values to integers.
            If include_reverse is True, returns a tuple of two dictionaries:
            (value_to_int_mapping, int_to_value_mapping).
        """
        mapping, reverse_mapping = dict(), dict()
        for val in values:
            idx = self.str_to_int(feat, val)
            mapping[val] = idx
            reverse_mapping[idx] = val
        if include_reverse:
            return mapping, reverse_mapping
        else:
            return mapping
