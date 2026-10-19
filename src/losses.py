
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
    lambda_cos=0.1,
):
    """
    logits_*:  (B, 1)
    features_*: (B, 512)
    labels:     (B, 1), real=0 and fake=1
    """

    # 1. Classification loss for each compression version
    loss_raw = F.binary_cross_entropy_with_logits(
        logits_raw, labels
    )
    loss_c23 = F.binary_cross_entropy_with_logits(
        logits_c23, labels
    )
    loss_c40 = F.binary_cross_entropy_with_logits(
        logits_c40, labels
    )

    loss_bce = (loss_raw + loss_c23 + loss_c40) / 3.0

    # 2. Compression-consistency loss
    cos_raw_c23 = F.cosine_similarity(
        features_raw, features_c23, dim=1
    )
    cos_raw_c40 = F.cosine_similarity(
        features_raw, features_c40, dim=1
    )
    cos_c23_c40 = F.cosine_similarity(
        features_c23, features_c40, dim=1
    )

    loss_cos = (
        (1 - cos_raw_c23).mean()
        + (1 - cos_raw_c40).mean()
        + (1 - cos_c23_c40).mean()
    ) / 3.0

    # 3. Combined objective
    total_loss = loss_bce + lambda_cos * loss_cos

    return total_loss, {
        "bce": loss_bce.detach(),
        "cosine": loss_cos.detach(),
    }

