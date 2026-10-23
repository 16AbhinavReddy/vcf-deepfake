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
    logits_*: (B, 1)
    features_*: (B, 512)
    labels: (B, 1), real=0 and fake=1

    Uses standard BCE without a manually chosen class weight. The balanced
    dataset contains approximately 1 real : 1.5 fake videos, so a pos_weight
    of 0.25 would unnecessarily down-weight fake examples.
    """
    labels = labels.float()

    loss_raw = F.binary_cross_entropy_with_logits(logits_raw.float(), labels)
    loss_c23 = F.binary_cross_entropy_with_logits(logits_c23.float(), labels)
    loss_c40 = F.binary_cross_entropy_with_logits(logits_c40.float(), labels)
    loss_bce = (loss_raw + loss_c23 + loss_c40) / 3.0

    # Encourage features to remain similar across compression versions.
    cos_raw_c23 = F.cosine_similarity(features_raw.float(), features_c23.float(), dim=1)
    cos_raw_c40 = F.cosine_similarity(features_raw.float(), features_c40.float(), dim=1)
    cos_c23_c40 = F.cosine_similarity(features_c23.float(), features_c40.float(), dim=1)

    loss_cos = (
        (1.0 - cos_raw_c23).mean()
        + (1.0 - cos_raw_c40).mean()
        + (1.0 - cos_c23_c40).mean()
    ) / 3.0

    total_loss = loss_bce + lambda_cos * loss_cos
    return total_loss, {
        "bce": loss_bce.detach(),
        "cosine": loss_cos.detach(),
    }
