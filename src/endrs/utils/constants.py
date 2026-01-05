import sys
from enum import Enum, unique
from typing import Final

USER_KEY: Final[str] = sys.intern("__users__")
ITEM_KEY: Final[str] = sys.intern("__items__")
LABEL_KEY: Final[str] = sys.intern("__labels__")
# MULTI_LABEL_KEY: Final[str] = "__multi_labels__"
SEQ_KEY: Final[str] = sys.intern("__seq__")
# HASH_KEY: Final[str] = "__hash_inputs__"
OOV_IDX: Final[int] = 0
DEFAULT_PRED: Final[float] = 0.0
DEFAULT_HASH_BINS: Final[int] = 200_000

POINTWISE_LOSS = ("cross_entropy", "focal")
PAIRWISE_LOSS = ("bpr", "max_margin")
LISTWISE_LOSS = ("softmax",)
ALL_LOSSES = POINTWISE_LOSS + PAIRWISE_LOSS + LISTWISE_LOSS

RATING_METRICS = ("loss", "rmse", "mae", "r2")
POINTWISE_METRICS = (
    "loss",
    "log_loss",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "roc_gauc",
)
LISTWISE_METRICS = ("precision", "recall", "map", "ndcg", "coverage")
RANKING_METRICS = POINTWISE_METRICS + LISTWISE_METRICS


class StrEnum(str, Enum):
    @classmethod
    def contains(cls, x):
        return x in cls.__members__.values()  # cls._member_names_


@unique
class SequenceModels(StrEnum):
    YOUTUBERETRIEVAL = "YouTubeRetrieval"
    YOUTUBERANKING = "YouTubeRanking"
    DIN = "DIN"
    RNN4REC = "RNN4Rec"
    CASER = "Caser"
    WAVENET = "WaveNet"
    TRANSFORMER = "Transformer"
    SIM = "SIM"
