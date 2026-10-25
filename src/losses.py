import torch
import torch.nn.functional as F


def vcf_loss(
    logits_raw,
    logits_c23,
    logits_c40,
    features_raw,
    features_c23,
    features_c40,
    labels,
    lambda_pred=0.4,
):
    """
    logits_*: (B, 1), raw classifier logits (before sigmoid).
    features_*: (B, D), returned by the R3D-18 backbone (unused by this loss).
    labels: (B, 1), real=0 and fake=1.

    Classification loss is averaged across raw, c23, and c40. Prediction
    consistency is the mean squared difference between sigmoid probabilities
    for all three pairs. Since probabilities lie in [0, 1], each pairwise
    squared difference and the averaged consistency loss lie in [0, 1].
    """
    labels = labels.float()

    loss_raw = F.binary_cross_entropy_with_logits(logits_raw.float(), labels)
    loss_c23 = F.binary_cross_entropy_with_logits(logits_c23.float(), labels)
    loss_c40 = F.binary_cross_entropy_with_logits(logits_c40.float(), labels)
    loss_bce = (loss_raw + loss_c23 + loss_c40) / 3.0

    # Compute probabilities in float32 for numerical stability.
    prob_raw = torch.sigmoid(logits_raw.float())
    prob_c23 = torch.sigmoid(logits_c23.float())
    prob_c40 = torch.sigmoid(logits_c40.float())

    pred_raw_c23 = F.mse_loss(prob_raw, prob_c23)
    pred_raw_c40 = F.mse_loss(prob_raw, prob_c40)
    pred_c23_c40 = F.mse_loss(prob_c23, prob_c40)
    loss_pred = 2*(pred_raw_c23 + pred_raw_c40 + pred_c23_c40) / 3.0

    total_loss = loss_bce + lambda_pred * loss_pred
    return total_loss, {
        "bce": loss_bce.detach(),
        "pred_consistency": loss_pred.detach(),
    }

