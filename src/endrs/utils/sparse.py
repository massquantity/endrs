from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from endrs.data.batch import BatchData


@dataclass
class SparseMatrix:
    sparse_indices: list[int]
    sparse_indptr: list[int]
    sparse_data: list[float]


def construct_sparse(data: BatchData) -> tuple[SparseMatrix, SparseMatrix]:
    # remove duplicated lines in data
    interaction = pd.DataFrame(
        {"user": data.users, "item": data.items, "label": data.labels}
    )
    interaction = interaction.drop_duplicates(subset=["user", "item"], keep="last")
    user_indices = interaction["user"].to_numpy()
    item_indices = interaction["item"].to_numpy()
    labels = interaction["label"].to_numpy()

    user_interactions = sp.csr_array(
        (labels, (user_indices, item_indices)), dtype=np.float32
    )
    item_interactions = user_interactions.transpose().tocsr()

    user_sparse = SparseMatrix(
        user_interactions.indices.tolist(),
        user_interactions.indptr.tolist(),
        user_interactions.data.tolist(),
    )
    item_sparse = SparseMatrix(
        item_interactions.indices.tolist(),
        item_interactions.indptr.tolist(),
        item_interactions.data.tolist(),
    )

    return user_sparse, item_sparse
