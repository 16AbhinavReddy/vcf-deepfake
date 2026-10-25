
from pathlib import Path
import csv

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
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model_linear_prediction_consistency_fixed_split.pt"
DATA_ROOT = ROOT / "vcf-balanced"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Dataset root: {DATA_ROOT}")

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

    # Always evaluate on the persisted test split, not indices generated
    # during a training run.
    split_file = ROOT / "splits" / "test.csv"
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Test split not found: {split_file}. "
            "Run `python src/create_splits.py` once and train using the saved splits."
        )

    index_by_path = {
        record["relative_path"]: i
        for i, record in enumerate(dataset.records)
    }

    test_indices = []
    seen_paths = set()

    with split_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"relative_path", "label", "group_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{split_file} must contain columns: "
                "relative_path,label,group_id"
            )

        for row in reader:
            relative_path = row["relative_path"]

            if relative_path in seen_paths:
                raise ValueError(
                    f"Duplicate path {relative_path!r} in {split_file}"
                )
            seen_paths.add(relative_path)

            if relative_path not in index_by_path:
                raise ValueError(
                    f"Test record {relative_path!r} is missing from the "
                    "current dataset. Check vcf-balanced and the saved split CSV."
                )

            idx = index_by_path[relative_path]
            record = dataset.records[idx]

            if int(row["label"]) != int(record["label"]):
                raise ValueError(f"Label mismatch for {relative_path!r}")
            if row["group_id"] != str(record["group_id"]):
                raise ValueError(f"Group mismatch for {relative_path!r}")

            test_indices.append(idx)

    if not test_indices:
        raise ValueError(f"No test records found in {split_file}")

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


