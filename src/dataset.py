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
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def get_label(relative_path: Path) -> int:
    for part in relative_path.parts:
        if part in LABELS:
            return LABELS[part]
    raise ValueError(f"Cannot determine real/fake label for: {relative_path}")


def source_id(relative_path: Path) -> str:
    # Assumes the prefix before the first underscore identifies the source.
    return relative_path.stem.split("_")[0]


def read_video(path: Path, num_frames: int = 8, image_size: int = 112) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {path}")

    indices = np.linspace(0, frame_count - 1, num=min(num_frames, frame_count)).astype(int)
    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()
        if not success:
            cap.release()
            raise RuntimeError(f"Could not read frame {index} from {path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        scale = image_size / min(height, width)
        new_width = max(image_size, round(width * scale))
        new_height = max(image_size, round(height * scale))
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        top = (new_height - image_size) // 2
        left = (new_width - image_size) // 2
        frame = frame[top:top + image_size, left:left + image_size]

        frame = frame.astype(np.float32) / 255.0
        frames.append((frame - MEAN) / STD)

    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    video = np.transpose(np.stack(frames, axis=0), (3, 0, 1, 2)).copy()
    return torch.from_numpy(video).float()


class VCFDataset(Dataset):
    """
    Each item contains raw/c23/c40 versions of one balanced-dataset video.
    Only records with all three matching relative paths are retained.
    """
    def __init__(self, root="vcf-balanced", num_frames=8, image_size=112, records=None):
        self.root = Path(root)
        self.num_frames = num_frames
        self.image_size = image_size

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root.resolve()}")

        self.records = records if records is not None else self._discover_records()
        if not self.records:
            raise RuntimeError(
                f"No videos have matching raw/c23/c40 paths under {self.root}. "
                "See the per-split filename comparison."
            )

    @staticmethod
    def _video_map(folder: Path):
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing split directory: {folder}")
        return {
            p.relative_to(folder).as_posix(): p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        }

    def _discover_records(self):
        maps = {
            split: self._video_map(self.root / split)
            for split in ("raw", "c23", "c40")
        }

        names_raw = set(maps["raw"])
        names_c23 = set(maps["c23"])
        names_c40 = set(maps["c40"])
        common = names_raw & names_c23 & names_c40

        print("Video file counts:")
        for split, video_map in maps.items():
            print(f"  {split}: {len(video_map)}")
        print(f"  common relative paths: {len(common)}")
        print(f"  raw missing c23: {len(names_raw - names_c23)}")
        print(f"  raw missing c40: {len(names_raw - names_c40)}")
        print(f"  c23 files without raw: {len(names_c23 - names_raw)}")
        print(f"  c40 files without raw: {len(names_c40 - names_raw)}")

        records = []
        for name in sorted(common):
            relative_path = Path(name)
            paths = {split: maps[split][name] for split in ("raw", "c23", "c40")}
            records.append({
                "paths": paths,
                "label": get_label(relative_path),
                "group_id": source_id(relative_path),
                "relative_path": name,
            })

        # Report class counts in the retained intersection.
        n_real = sum(record["label"] == 0 for record in records)
        n_fake = sum(record["label"] == 1 for record in records)
        print(f"Matched records retained: {len(records)} | real={n_real} | fake={n_fake}")

        if len(common) < len(names_raw):
            print(
                "WARNING: some raw records were skipped because the same relative "
                "path was absent from c23 or c40. This is a safe loader fallback, "
                "but if many records are skipped, rebuild the balanced dataset so "
                "the same selected files are present in all three compression splits."
            )

        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        raw_video = read_video(record["paths"]["raw"], self.num_frames, self.image_size)
        c23_video = read_video(record["paths"]["c23"], self.num_frames, self.image_size)
        c40_video = read_video(record["paths"]["c40"], self.num_frames, self.image_size)
        label = torch.tensor(record["label"], dtype=torch.float32)
        return raw_video, c23_video, c40_video, label, record["group_id"]
