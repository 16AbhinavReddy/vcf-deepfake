
from pathlib import Path
import csv

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from dataset import VCFDataset


# Reproducibility
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "vcf-balanced"
SPLIT_DIR = ROOT / "splits"


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset metadata without decoding videos.
    dataset = VCFDataset(
        root=DATA_ROOT,
        num_frames=8,
        image_size=112,
    )

    records = dataset.records
    indices = np.arange(len(records))

    labels = np.array(
        [record["label"] for record in records]
    )
    groups = np.array(
        [record["group_id"] for record in records]
    )

    print(f"Total records: {len(records)}")
    print(f"Real videos: {(labels == 0).sum()}")
    print(f"Fake videos: {(labels == 1).sum()}")
    print(f"Unique source groups: {len(set(groups))}")

    if len(set(groups)) < 15:
        raise ValueError(
            "Too few unique source groups for a reliable split."
        )

    # --------------------------------------------------
    # STEP 1: Reserve approximately 25% for testing.
    # --------------------------------------------------
    outer = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=SEED,
    )

    trainval_rel, test_rel = next(
        outer.split(indices, labels, groups)
    )

    trainval_idx = indices[trainval_rel]
    test_idx = indices[test_rel]

    # --------------------------------------------------
    # STEP 2: Split the remaining 75% into approximately
    # 65.625% training and 9.375% validation.
    # --------------------------------------------------
    inner = StratifiedGroupKFold(
        n_splits=8,
        shuffle=True,
        random_state=SEED + 1,
    )

    train_rel, val_rel = next(
        inner.split(
            trainval_idx,
            labels[trainval_idx],
            groups[trainval_idx],
        )
    )

    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    # --------------------------------------------------
    # STEP 3: Verify that source groups do not overlap.
    # --------------------------------------------------
    group_sets = {
        name: set(groups[idx])
        for name, idx in splits.items()
    }

    assert group_sets["train"].isdisjoint(group_sets["val"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["val"].isdisjoint(group_sets["test"])

    # Verify that every record occurs in exactly one split.
    all_indices = np.concatenate(list(splits.values()))

    assert len(all_indices) == len(records)
    assert len(np.unique(all_indices)) == len(records)

    # --------------------------------------------------
    # STEP 4: Save permanent CSV files.
    # --------------------------------------------------
    for name, idx in splits.items():
        output_path = SPLIT_DIR / f"{name}.csv"

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "relative_path",
                    "label",
                    "group_id",
                ],
            )
            writer.writeheader()

            for i in idx:
                record = records[i]

                writer.writerow({
                    "relative_path": record["relative_path"],
                    "label": record["label"],
                    "group_id": record["group_id"],
                })

        real_count = int((labels[idx] == 0).sum())
        fake_count = int((labels[idx] == 1).sum())
        percentage = 100 * len(idx) / len(records)

        print(f"\n{name.upper()} SET")
        print(f"  Samples: {len(idx)} ({percentage:.2f}%)")
        print(f"  Real: {real_count}")
        print(f"  Fake: {fake_count}")
        print(f"  Groups: {len(group_sets[name])}")
        print(f"  Saved: {output_path}")

    print("\nAll split checks passed.")
    print("Training, validation and test CSVs have been saved.")
    print("Reuse these files for every experiment.")


if __name__ == "__main__":
    main()

