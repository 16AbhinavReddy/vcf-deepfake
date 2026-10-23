from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import VCFDataset
from losses import vcf_loss
from model import VCFDeepfakeDetector


SEED = 42
NUM_FRAMES = 8
IMAGE_SIZE = 112
BATCH_SIZE = 6
EPOCHS = 3
LEARNING_RATE = 1e-4
LAMBDA_COS = 1
NUM_WORKERS = 2

# Assumes this file is in <project-root>/src/train.py and the balanced
# dataset is <project-root>/vcf-balanced/{raw,c23,c40}/...
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "vcf-balanced"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_splits(dataset):
    indices = np.arange(len(dataset))
    labels = np.array([record["label"] for record in dataset.records])
    groups = np.array([record["group_id"] for record in dataset.records])

    # First reserve approximately 20% for the final test set.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_val_rel, test_rel = next(outer.split(indices, labels, groups))
    train_val_idx = indices[train_val_rel]
    test_idx = indices[test_rel]

    # Reserve approximately 10% of the full dataset for validation.
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=SEED + 1)
    train_rel, val_rel = next(
        inner.split(
            train_val_idx,
            labels[train_val_idx],
            groups[train_val_idx],
        )
    )
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    for name, idx in [
        ("Train", train_idx),
        ("Validation", val_idx),
        ("Test", test_idx),
    ]:
        n_real = int((labels[idx] == 0).sum())
        n_fake = int((labels[idx] == 1).sum())
        print(
            f"{name}: {len(idx)} samples | real={n_real} | fake={n_fake} "
            f"| groups={len(set(groups[idx]))}"
        )

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def run_epoch(model, loader, optimizer, device, scaler=None, phase="TRAIN"):
    training = optimizer is not None
    model.train(training)

    total_loss = total_bce = total_cos = 0.0
    correct = count = 0
    tp = tn = fp = fn = 0
    start_time = time.time()
    batch_times = []

    progress = tqdm(loader, desc=phase, unit="batch", dynamic_ncols=True)

    for batch_idx, (raw, c23, c40, labels, _group_ids) in enumerate(progress):
        batch_start = time.time()
        raw = raw.to(device, non_blocking=True)
        c23 = c23.to(device, non_blocking=True)
        c40 = c40.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits_raw, features_raw = model(raw)
                logits_c23, features_c23 = model(c23)
                logits_c40, features_c40 = model(c40)
                loss, parts = vcf_loss(
                    logits_raw, logits_c23, logits_c40,
                    features_raw, features_c23, features_c40,
                    labels, lambda_cos=LAMBDA_COS,
                )

            if training:
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        avg_logits = (
            logits_raw.detach() + logits_c23.detach() + logits_c40.detach()
        ) / 3.0
        predictions = (avg_logits >= 0).float()
        correct += int((predictions == labels).sum().item())
        count += labels.numel()

        pred = predictions.view(-1).to(torch.int64)
        target = labels.view(-1).to(torch.int64)
        tp += int(((pred == 1) & (target == 1)).sum().item())
        tn += int(((pred == 0) & (target == 0)).sum().item())
        fp += int(((pred == 1) & (target == 0)).sum().item())
        fn += int(((pred == 0) & (target == 1)).sum().item())

        total_loss += float(loss.detach().item())
        total_bce += float(parts["bce"].item())
        total_cos += float(parts["cosine"].item())

        elapsed = time.time() - batch_start
        batch_times.append(elapsed)
        avg_time = sum(batch_times) / len(batch_times)
        eta_seconds = avg_time * (len(loader) - batch_idx - 1)

        progress.set_postfix(
            loss=f"{total_loss / (batch_idx + 1):.4f}",
            bce=f"{total_bce / (batch_idx + 1):.4f}",
            cos=f"{total_cos / (batch_idx + 1):.4f}",
            acc=f"{correct / max(count, 1):.3f}",
            batch_s=f"{elapsed:.1f}",
            eta=f"{eta_seconds / 60:.1f}m",
        )

    real_recall = tn / max(tn + fp, 1)
    fake_recall = tp / max(tp + fn, 1)
    balanced_acc = (real_recall + fake_recall) / 2.0
    print(
        f"\n[{phase}] TN={tn}, FP={fp}, FN={fn}, TP={tp} | "
        f"real recall={real_recall:.4f}, fake recall={fake_recall:.4f}, "
        f"balanced accuracy={balanced_acc:.4f} | "
        f"elapsed={(time.time() - start_time) / 60:.1f} min"
    )

    return {
        "loss": total_loss / max(len(loader), 1),
        "bce": total_bce / max(len(loader), 1),
        "cosine": total_cos / max(len(loader), 1),
        "accuracy": correct / max(count, 1),
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "balanced_accuracy": balanced_acc,
    }


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset root:", DATA_ROOT)

    dataset = VCFDataset(
        root=DATA_ROOT,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
    )
    labels = np.array([record["label"] for record in dataset.records])
    print(
        f"Dataset records: {len(dataset)} | "
        f"real={(labels == 0).sum()} | fake={(labels == 1).sum()}"
    )

    train_idx, val_idx, test_idx = make_splits(dataset)
    print(
        f"Train: {len(train_idx)} | Validation: {len(val_idx)} | "
        f"Test: {len(test_idx)}"
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = VCFDeepfakeDetector(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, scaler, phase="TRAIN"
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, None, device, phase="VALIDATION"
            )

        print(
            f"\nEpoch {epoch}/{EPOCHS}\n"
            f"  Train: loss={train_metrics['loss']:.4f}, "
            f"BCE={train_metrics['bce']:.4f}, cos={train_metrics['cosine']:.4f}, "
            f"acc={train_metrics['accuracy']:.4f}\n"
            f"  Val:   loss={val_metrics['loss']:.4f}, "
            f"BCE={val_metrics['bce']:.4f}, cos={val_metrics['cosine']:.4f}, "
            f"acc={val_metrics['accuracy']:.4f}\n"
            f"  Val balanced accuracy={val_metrics['balanced_accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "seed": SEED,
                    "num_frames": NUM_FRAMES,
                    "image_size": IMAGE_SIZE,
                    "lambda_cos": LAMBDA_COS,
                    "data_root": str(DATA_ROOT),
                    "train_indices": train_idx,
                    "val_indices": val_idx,
                    "test_indices": test_idx,
                },
                checkpoint_path,
            )
            print("  Saved best checkpoint:", checkpoint_path)

    print("\nTraining complete.")
    print("Best validation loss:", best_val_loss)
    print("Checkpoint:", checkpoint_path)
    print("The test set remains reserved for final evaluation.")


if __name__ == "__main__":
    main()
