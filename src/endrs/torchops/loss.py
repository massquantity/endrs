import torch
import torch.nn.functional as F


def mse_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error loss function."""
    return F.mse_loss(logits, labels)


def binary_cross_entropy_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Binary Cross Entropy loss with logits."""
    return F.binary_cross_entropy_with_logits(logits, labels)


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    mean: bool = True,
) -> torch.Tensor:
    """Focal loss for binary classification.
    
    The focal loss is designed to address class imbalance problem by 
    down-weighting easy examples so that training focuses more on hard examples.

    Parameters
    ----------
    logits : torch.Tensor
        Raw predictions from model.
    labels : torch.Tensor
        Target values (0 or 1).
    alpha : float, default: 0.25
        Weighting factor to balance positive and negative examples.
    gamma : float, default: 2.0
        Focusing parameter that adjusts the rate at which easy examples 
        are down-weighted.
    mean : bool, default: True
        Whether to return mean loss or the loss for each example.

    Returns
    -------
    torch.Tensor
        Focal loss.

    References
    ----------
    *Lin et al.* `Focal Loss for Dense Object Detection 
    <https://arxiv.org/pdf/1708.02002.pdf>`_.
    """
    weighting_factor = (labels * alpha) + ((1 - labels) * (1 - alpha))
    probs = torch.sigmoid(logits)
    p_t = (labels * probs) + ((1 - labels) * (1 - probs))
    modulating_factor = torch.pow(1.0 - p_t, gamma)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    focal = weighting_factor * modulating_factor * bce
    if mean:
        focal = torch.mean(focal)
    return focal


def softmax_cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Softmax Cross Entropy loss for multi-class classification."""
    return F.cross_entropy(logits, targets)


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss.
    
    The BPR loss is a pairwise ranking loss that aims to maximize the difference 
    between positive and negative item scores.

    Parameters
    ----------
    pos_scores : torch.Tensor
        Scores for positive examples.
    neg_scores : torch.Tensor
        Scores for negative examples.

    Returns
    -------
    torch.Tensor
        BPR loss.

    References
    ----------
    *Rendle et al.* `BPR: Bayesian Personalized Ranking from Implicit Feedback
    <https://arxiv.org/pdf/1205.2618.pdf>`_.
    """
    log_sigmoid = F.logsigmoid(pos_scores - neg_scores)
    return torch.negative(torch.mean(log_sigmoid))


def max_margin_loss(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float
) -> torch.Tensor:
    """Max Margin loss (hinge loss) for pairwise ranking.
    
    This loss function aims to enforce a margin between positive and negative scores.

    Parameters
    ----------
    pos_scores : torch.Tensor
        Scores for positive examples.
    neg_scores : torch.Tensor
        Scores for negative examples.
    margin : float
        The margin between positive and negative scores.

    Returns
    -------
    torch.Tensor
        Max margin loss.
    """
    return F.margin_ranking_loss(
        pos_scores, neg_scores, torch.ones_like(pos_scores), margin=margin
    )
