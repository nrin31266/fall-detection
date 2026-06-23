from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FallVideoDataset(Dataset):
    """
    Dataset đọc 1 video thành 1 clip gồm 16 frame lấy đều trên toàn video.

    Output:
        clip_tensor: FloatTensor [T, C, H, W]
        label_tensor: LongTensor scalar

    Label:
        Normal = 0
        Fall = 1
    """

    def __init__(
        self,
        index_csv: str | Path,
        split: str,
        image_size: int = 224,
        transform: Optional[Callable] = None,
    ):
        self.index_csv = Path(index_csv)
        self.split = split
        self.image_size = image_size
        self.transform = transform

        self.df = pd.read_csv(self.index_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split='{split}' in {index_csv}")

        self.label_to_idx = {
            "Normal": 0,
            "Fall": 1,
        }

    def __len__(self):
        return len(self.df)

    def _read_uniform_frames(self, video_path: str, clip_length: int):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Invalid frame count: {video_path}")

        frame_indices = np.linspace(
            0,
            max(total_frames - 1, 0),
            clip_length
        ).astype(int)

        frames = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if not ret:
                if frames:
                    frames.append(frames[-1].copy())
                    continue

                cap.release()
                raise RuntimeError(f"Cannot read frame {frame_idx}: {video_path}")

            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        clip = np.stack(frames, axis=0)  # [T, H, W, C]
        clip = clip.astype(np.float32) / 255.0
        clip = np.transpose(clip, (0, 3, 1, 2))  # [T, C, H, W]

        return clip

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        clip_length = int(row["clip_length"])
        video_path = row["video_path"]

        clip = self._read_uniform_frames(video_path, clip_length)

        if self.transform is not None:
            clip = self.transform(clip)

        label = self.label_to_idx[row["label"]]

        clip_tensor = torch.from_numpy(clip).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "clip": clip_tensor,
            "label": label_tensor,
            "sample_id": row["sample_id"],
            "video_path": row["video_path"],
            "subtype": row["subtype"],
        }


if __name__ == "__main__":
    dataset = FallVideoDataset(
        index_csv="data/video_index_16.csv",
        split="train",
        image_size=224,
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]
    print("Clip shape:", sample["clip"].shape)
    print("Label:", sample["label"])
    print("Sample ID:", sample["sample_id"])
    print("Video:", sample["video_path"])