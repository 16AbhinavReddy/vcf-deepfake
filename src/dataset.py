from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


LABELS = {
    "targets": 0,
    "simswap_224": 1,
    "simswap_512": 1,
    "deeplivecam": 1,
    "deeplivecam_enhance": 1,
}

# Normalization values used by the pretrained torchvision R3D-18 weights.
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def get_label(path: Path) -> int:
    """Infer the real/fake label from the generator folder."""
    for part in path.parts:
        if part in LABELS:
            return LABELS[part]

    raise ValueError(f"Cannot determine label for: {path}")


def source_id(relative_path: Path) -> str:
    """
    Group related videos by the hash-like filename prefix.
    This helps keep related source videos in the same data split.
    """
    return relative_path.stem.split("_")[0]


def read_video(path: Path, num_frames: int = 8,
               image_size: int = 112) -> torch.Tensor:
    """
    Read evenly spaced frames and return a normalized tensor:
    (channels, frames, height, width).
    """
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {path}")

    indices = np.linspace(
        0, frame_count - 1, num=min(num_frames, frame_count)
    ).astype(int)

    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()

        if not success:
            cap.release()
            raise RuntimeError(
                f"Could not read frame {index} from {path}"
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize while preserving aspect ratio, then center-crop.
        height, width = frame.shape[:2]
        scale = image_size / min(height, width)
        new_width = max(image_size, round(width * scale))
        new_height = max(image_size, round(height * scale))

        frame = cv2.resize(
            frame, (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        top = (new_height - image_size) // 2
        left = (new_width - image_size) // 2
        frame = frame[
            top:top + image_size,
            left:left + image_size
        ]

        frame = frame.astype(np.float32) / 255.0
        frame = (frame - MEAN) / STD
        frames.append(frame)

    cap.release()

    # Repeat the last frame if the video has fewer frames than requested.
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    # (T, H, W, C) -> (C, T, H, W)
    video = np.stack(frames, axis=0)
    video = np.transpose(video, (3, 0, 1, 2)).copy()

    return torch.from_numpy(video).float()


class VCFDataset(Dataset):
    """
    Each item contains the same video in raw, c23, and c40 formats.

    Returns:
        raw_video: (3, T, H, W)
        c23_video: (3, T, H, W)
        c40_video: (3, T, H, W)
        label:     scalar float tensor
        group_id:  source identifier string
    """

    def __init__(
        self,
        root="vcf/vcf",
        num_frames=8,
        image_size=112,
        records=None,
    ):
        self.root = Path(root)
        self.num_frames = num_frames
        self.image_size = image_size

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root.resolve()}"
            )

        if records is None:
            records = self._discover_records()

        self.records = records

        if not self.records:
            raise RuntimeError(f"No paired videos found under {self.root}")

    def _discover_records(self):
        raw_root = self.root / "raw"
        records = []

        for raw_path in sorted(raw_root.rglob("*.mp4")):
            relative_path = raw_path.relative_to(raw_root)

            if not relative_path.parts:
                continue

            label = get_label(relative_path)

            paths = {
                "raw": raw_path,
                "c23": self.root / "c23" / relative_path,
                "c40": self.root / "c40" / relative_path,
            }

            missing = [
                str(path) for path in paths.values()
                if not path.is_file()
            ]

            if missing:
                raise FileNotFoundError(
                    "Missing compression counterpart(s):\n"
                    + "\n".join(missing)
                )

            records.append({
                "paths": paths,
                "label": label,
                "group_id": source_id(relative_path),
                "relative_path": str(relative_path),
            })

        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        paths = record["paths"]

        raw_video = read_video(
            paths["raw"], self.num_frames, self.image_size
        )
        c23_video = read_video(
            paths["c23"], self.num_frames, self.image_size
        )
        c40_video = read_video(
            paths["c40"], self.num_frames, self.image_size
        )

        label = torch.tensor(record["label"], dtype=torch.float32)

        return (
            raw_video,
            c23_video,
            c40_video,
            label,
            record["group_id"],
        )
