
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit

from dataset import VCFDataset
from model import VCFDeepfakeDetector
from losses import vcf_loss

import time
from tqdm import tqdm

from sklearn.model_selection import StratifiedGroupKFold


SEED = 42
NUM_FRAMES = 8
IMAGE_SIZE = 112
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 5e-5
LAMBDA_COS = 0.2
NUM_WORKERS = 2

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "vcf" / "vcf"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_splits(dataset):
    indices = np.arange(len(dataset))
    labels = np.array([r["label"] for r in dataset.records])
    groups = np.array([r["group_id"] for r in dataset.records])

    # Approximately 20% held out for testing.
    outer = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )
    train_val_rel, test_rel = next(
        outer.split(indices, labels, groups)
    )

    train_val_idx = indices[train_val_rel]
    test_idx = indices[test_rel]

    # Approximately 10% of the complete dataset for validation:
    # 12.5% of the remaining 80%.
    inner = StratifiedGroupKFold(
        n_splits=8, shuffle=True, random_state=SEED + 1
    )
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
        print(
            f"{name}: {len(idx)} samples, "
            f"real={(labels[idx] == 0).sum()}, "
            f"fake={(labels[idx] == 1).sum()}, "
            f"groups={len(set(groups[idx]))}"
        )

    return (
        train_idx.tolist(),
        val_idx.tolist(),
        test_idx.tolist(),
    )


def run_epoch(model, loader, optimizer, device, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_bce = 0.0
    total_cos = 0.0
    correct = 0
    count = 0
    epoch_start = time.time()
    batch_times = []

    phase = "TRAIN" if training else "VALIDATION"

    progress = tqdm(
        loader,
        desc=f"{phase}",
        unit="batch",
        dynamic_ncols=True,
    )

    tp = tn = fp = fn = 0

    for batch_idx, (raw, c23, c40, labels, group_ids) in enumerate(progress):
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
                print(
                    f"\n[{phase}] Batch {batch_idx + 1}/{len(loader)}: "
                    "running model forward passes...",
                    flush=True,
                )

                logits_raw, features_raw = model(raw)
                logits_c23, features_c23 = model(c23)
                logits_c40, features_c40 = model(c40)

                loss, parts = vcf_loss(
                    logits_raw,
                    logits_c23,
                    logits_c40,
                    features_raw,
                    features_c23,
                    features_c40,
                    labels,
                    lambda_cos=LAMBDA_COS,
                )

            if training:
                print(
                    f"[TRAIN] Batch {batch_idx + 1}: "
                    "backpropagating...",
                    flush=True,
                )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        avg_logits = (
            logits_raw.detach()
            + logits_c23.detach()
            + logits_c40.detach()
        ) / 3.0

        predictions = (avg_logits >= 0).float()
        correct += (predictions == labels).sum().item()
        count += labels.numel()

        pred = predictions.view(-1).to(torch.int64)
        target = labels.view(-1).to(torch.int64)

        tp += ((pred == 1) & (target == 1)).sum().item()
        tn += ((pred == 0) & (target == 0)).sum().item()
        fp += ((pred == 1) & (target == 0)).sum().item()
        fn += ((pred == 0) & (target == 1)).sum().item()

        total_loss += loss.detach().item()
        total_bce += parts["bce"].item()
        total_cos += parts["cosine"].item()

        batch_elapsed = time.time() - batch_start
        batch_times.append(batch_elapsed)

        avg_time = sum(batch_times) / len(batch_times)
        remaining = len(loader) - batch_idx - 1
        eta_seconds = avg_time * remaining

        gpu_memory = ""
        if device.type == "cuda":
            gpu_memory = (
                f" | GPU memory: "
                f"{torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB"
            )

        progress.set_postfix(
            loss=f"{total_loss / (batch_idx + 1):.4f}",
            bce=f"{total_bce / (batch_idx + 1):.4f}",
            cos=f"{total_cos / (batch_idx + 1):.4f}",
            acc=f"{correct / max(count, 1):.3f}",
            batch_s=f"{batch_elapsed:.2f}",
            eta=f"{eta_seconds / 60:.1f}m",
        )

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(
                f"\n[{phase}] Completed {batch_idx + 1}/{len(loader)} "
                f"batches | "
                f"Avg loss: {total_loss / (batch_idx + 1):.4f} | "
                f"Accuracy: {correct / max(count, 1):.3f} | "
                f"Elapsed: {(time.time() - epoch_start) / 60:.1f} min"
                f"{gpu_memory}",
                flush=True,
            )

    real_recall = tn / max(tn + fp, 1)
    fake_recall = tp / max(tp + fn, 1)
    balanced_acc = (real_recall + fake_recall) / 2

    print(
        f"\n[{phase}] Confusion matrix counts: "
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )
    print(
        f"[{phase}] Real recall={real_recall:.4f}, "
        f"Fake recall={fake_recall:.4f}, "
        f"Balanced accuracy={balanced_acc:.4f}"
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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = VCFDataset(
        root=DATA_ROOT,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
    )
    train_idx, val_idx, test_idx = make_splits(dataset)

    print(f"Total samples: {len(dataset)}")
    print(
        f"Train: {len(train_idx)} | "
        f"Validation: {len(val_idx)} | "
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
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(SEED),
    )

    model = VCFDeepfakeDetector(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    best_val_loss = float("inf")
    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, scaler
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, None, device
            )

        print(
            f"\nEpoch {epoch}/{EPOCHS}\n"
            f"  Train: loss={train_metrics['loss']:.4f}, "
            f"BCE={train_metrics['bce']:.4f}, "
            f"cos={train_metrics['cosine']:.4f}, "
            f"acc={train_metrics['accuracy']:.4f}\n"
            f"  Val:   loss={val_metrics['loss']:.4f}, "
            f"BCE={val_metrics['bce']:.4f}, "
            f"cos={val_metrics['cosine']:.4f}, "
            f"acc={val_metrics['accuracy']:.4f}"
        )
        print(
            f"  Val real recall={val_metrics['real_recall']:.4f}, "
            f"fake recall={val_metrics['fake_recall']:.4f}, "
            f"balanced accuracy={val_metrics['balanced_accuracy']:.4f}"
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
    print("Test set is reserved for final evaluation.")


if __name__ == "__main__":
    main()

