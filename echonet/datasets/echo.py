"""EchoNet-Dynamic dataset loader used by the Stage 1 experiments.

This version preserves the official EchoNet-Dynamic target API while making
large/small trace selection explicit and robust: the traced frame with the
larger LV mask area is treated as Large/end-diastolic (ED), and the frame with
the smaller LV mask area is treated as Small/end-systolic (ES).
"""

from __future__ import annotations

import collections
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import skimage.draw
import torchvision

import echonet


def _defaultdict_of_lists():
    return collections.defaultdict(list)


class Echo(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic dataset.

    Parameters are intentionally compatible with the original EchoNet loader.
    Stage 1 segmentation/multitask scripts do not use spatial augmentation
    (``pad=None``), which makes image/mask alignment unambiguous.
    """

    def __init__(
        self,
        root=None,
        split="train",
        target_type="EF",
        mean=0.0,
        std=1.0,
        length=16,
        period=2,
        max_length=250,
        clips=1,
        pad=None,
        noise=None,
        target_transform=None,
        external_test_location=None,
    ):
        if root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]

        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames: List[str] = []
        self.outcome: List[list] = []

        if self.split == "EXTERNAL_TEST":
            if self.external_test_location is None:
                raise ValueError("external_test_location is required for EXTERNAL_TEST")
            self.fnames = sorted(os.listdir(self.external_test_location))
            self.header = []
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)
            self.large_frame_index = {}
            self.small_frame_index = {}
            return

        file_list_path = os.path.join(self.root, "FileList.csv")
        data = pd.read_csv(file_list_path)
        if "Split" not in data.columns or "FileName" not in data.columns:
            raise ValueError("FileList.csv must contain FileName and Split columns")

        data["Split"] = data["Split"].astype(str).str.upper()
        if self.split != "ALL":
            data = data[data["Split"] == self.split].copy()

        self.header = data.columns.tolist()
        self.fnames = data["FileName"].astype(str).tolist()
        self.fnames = [
            fn if os.path.splitext(fn)[1] else fn + ".avi"
            for fn in self.fnames
        ]
        self.outcome = data.values.tolist()

        videos_dir = os.path.join(self.root, "Videos")
        missing = set(self.fnames) - set(os.listdir(videos_dir))
        if missing:
            first = sorted(missing)[0]
            raise FileNotFoundError(
                f"{len(missing)} listed videos are missing; first missing file: "
                f"{os.path.join(videos_dir, first)}"
            )

        self.frames = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

        tracing_path = os.path.join(self.root, "VolumeTracings.csv")
        with open(tracing_path, "r") as f:
            header = f.readline().strip().split(",")
            expected = ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
            if header != expected:
                raise ValueError(
                    f"Unexpected VolumeTracings.csv header: {header}; expected {expected}"
                )

            for line in f:
                filename, x1, y1, x2, y2, frame = line.strip().split(",")
                frame = int(frame)
                if frame not in self.trace[filename]:
                    self.frames[filename].append(frame)
                self.trace[filename][frame].append(
                    (float(x1), float(y1), float(x2), float(y2))
                )

        for filename in self.frames:
            for frame in self.frames[filename]:
                self.trace[filename][frame] = np.asarray(
                    self.trace[filename][frame], dtype=np.float32
                )

        # Only segmentation targets require ED/ES tracings. EF-only regression
        # must retain every video in the saved split, including the six videos
        # without VolumeTracings rows.
        trace_targets = {
            "LargeIndex", "SmallIndex", "LargeFrame", "SmallFrame",
            "LargeTrace", "SmallTrace",
        }
        self.requires_traces = any(t in trace_targets for t in self.target_type)
        if self.requires_traces:
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for f, k in zip(self.fnames, keep) if k]
            self.outcome = [o for o, k in zip(self.outcome, keep) if k]

        # Select Large/Small explicitly by traced LV polygon area instead of
        # sorting frame numbers. This avoids accidentally interpreting temporal
        # order as ED/ES order.
        self.large_frame_index: Dict[str, int] = {}
        self.small_frame_index: Dict[str, int] = {}
        for filename in self.fnames:
            if len(self.frames[filename]) < 2:
                continue
            areas = []
            for frame in self.frames[filename]:
                areas.append((self._trace_polygon_area(self.trace[filename][frame]), frame))
            areas.sort(key=lambda x: x[0])
            self.small_frame_index[filename] = int(areas[0][1])
            self.large_frame_index[filename] = int(areas[-1][1])

    @staticmethod
    def _trace_polygon_xy(trace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
        # Match the official EchoNet construction.
        x = np.concatenate((x1[1:], np.flip(x2[1:])))
        y = np.concatenate((y1[1:], np.flip(y2[1:])))
        return x, y

    @classmethod
    def _trace_polygon_area(cls, trace: np.ndarray) -> float:
        x, y = cls._trace_polygon_xy(trace)
        if len(x) < 3:
            return 0.0
        # Shoelace formula.
        return float(
            0.5
            * np.abs(
                np.dot(x, np.roll(y, 1))
                - np.dot(y, np.roll(x, 1))
            )
        )

    @classmethod
    def _trace_to_mask(cls, trace: np.ndarray, height: int, width: int) -> np.ndarray:
        x, y = cls._trace_polygon_xy(trace)
        rr, cc = skimage.draw.polygon(
            np.rint(y).astype(int),
            np.rint(x).astype(int),
            shape=(height, width),
        )
        mask = np.zeros((height, width), dtype=np.float32)
        mask[rr, cc] = 1.0
        return mask

    def __getitem__(self, index):
        if self.split == "EXTERNAL_TEST":
            video_path = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video_path = os.path.join(
                self.root, "ProcessedStrainStudyA4c", self.fnames[index]
            )
        else:
            video_path = os.path.join(self.root, "Videos", self.fnames[index])

        video = echonet.utils.loadvideo(video_path).astype(np.float32)

        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= np.asarray(self.mean).reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= np.asarray(self.std).reshape(3, 1, 1, 1)

        c, f, h, w = video.shape
        length = f // self.period if self.length is None else self.length
        if self.max_length is not None:
            length = min(length, self.max_length)

        if f < length * self.period:
            video = np.concatenate(
                (
                    video,
                    np.zeros(
                        (c, length * self.period - f, h, w),
                        dtype=video.dtype,
                    ),
                ),
                axis=1,
            )
            c, f, h, w = video.shape

        if self.clips == "all":
            start = np.arange(f - (length - 1) * self.period)
        else:
            start = np.random.choice(
                f - (length - 1) * self.period,
                int(self.clips),
            )

        target = []
        for target_name in self.target_type:
            key = self.fnames[index]

            if target_name == "Filename":
                target.append(key)
            elif target_name == "LargeIndex":
                target.append(np.int64(self.large_frame_index[key]))
            elif target_name == "SmallIndex":
                target.append(np.int64(self.small_frame_index[key]))
            elif target_name == "LargeFrame":
                target.append(video[:, self.large_frame_index[key], :, :])
            elif target_name == "SmallFrame":
                target.append(video[:, self.small_frame_index[key], :, :])
            elif target_name == "LargeTrace":
                frame = self.large_frame_index[key]
                target.append(self._trace_to_mask(self.trace[key][frame], h, w))
            elif target_name == "SmallTrace":
                frame = self.small_frame_index[key]
                target.append(self._trace_to_mask(self.trace[key][frame], h, w))
            else:
                if self.split in {"CLINICAL_TEST", "EXTERNAL_TEST"}:
                    target.append(np.float32(0.0))
                else:
                    if target_name not in self.header:
                        raise KeyError(
                            f"Target '{target_name}' was not found in FileList.csv columns"
                        )
                    target.append(
                        np.float32(
                            self.outcome[index][self.header.index(target_name)]
                        )
                    )

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        clips = tuple(
            video[:, s + self.period * np.arange(length), :, :]
            for s in start
        )
        if self.clips == 1:
            sampled_video = clips[0]
        else:
            sampled_video = np.stack(clips)

        if self.pad is not None:
            c, l, h, w = sampled_video.shape
            temp = np.zeros(
                (c, l, h + 2 * self.pad, w + 2 * self.pad),
                dtype=sampled_video.dtype,
            )
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = sampled_video
            i, j = np.random.randint(0, 2 * self.pad, 2)
            sampled_video = temp[:, :, i : i + h, j : j + w]

        return sampled_video, target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)
