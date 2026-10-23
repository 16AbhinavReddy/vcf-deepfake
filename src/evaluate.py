
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from dataset import VCFDataset
from model import VCFDeepfakeDetector


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pt"
DATA_ROOT = ROOT / "vcf-balanced"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    dataset = VCFDataset(
        root=DATA_ROOT,
        num_frames=checkpoint["num_frames"],
        image_size=checkpoint["image_size"],
    )

    test_indices = checkpoint["test_indices"]

    if max(test_indices) >= len(dataset):
        raise ValueError(
            "Checkpoint test indices do not match the current dataset. "
            "Retrain the model using vcf-balanced."
        )

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model = VCFDeepfakeDetector(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for raw, c23, c40, labels, _group_ids in test_loader:
            raw = raw.to(device)
            c23 = c23.to(device)
            c40 = c40.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits_raw, _ = model(raw)
                logits_c23, _ = model(c23)
                logits_c40, _ = model(c40)

            # Overall prediction: average logits across all
            # three compression versions.
            avg_logits = (
                logits_raw + logits_c23 + logits_c40
            ) / 3.0

            predictions = (
                torch.sigmoid(avg_logits) >= 0.5
            ).int()

            all_labels.extend(labels.numpy().astype(int).tolist())
            all_predictions.extend(
                predictions.cpu().numpy().reshape(-1).tolist()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    print("\n===== OVERALL TEST PERFORMANCE =====")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Test samples: {len(y_true)}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        f"F1 score:  {f1_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print("Labels: 0=real, 1=fake")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Real", "Fake"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
