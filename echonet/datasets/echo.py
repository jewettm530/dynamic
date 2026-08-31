"""EchoNet-Dynamic dataset loader.

This loader keeps the original target API but fixes two Stage 1 details:

1. ``VolumeTracings.csv`` is loaded only when a requested target actually
   needs ED/ES tracing information. EF-only/video-only inference can therefore
   run from ``FileList.csv`` + ``Videos/`` without ground-truth ED/ES data.
2. Large/Small traced frames are identified by traced LV polygon area rather
   than by temporal frame number.

The corrected Stage 1 B1/B2/B3 experiments use
``echonet.datasets.stage1_video.Stage1VideoDataset`` directly because it also
makes the video/segmentation separation explicit.
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


def _normalise_filename(value: str) -> str:
    value = str(value)
    return value if os.path.splitext(value)[1] else value + ".avi"


class Echo(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic dataset with an API compatible with the original loader."""

    TRACE_TARGETS = {
        "LargeIndex",
        "SmallIndex",
        "LargeFrame",
        "SmallFrame",
        "LargeTrace",
        "SmallTrace",
    }

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
        clip_start="random",
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
        self.clip_start = clip_start
        if self.clip_start not in {"random", "center"}:
            raise ValueError("clip_start must be 'random' or 'center'")

        self.requires_traces = any(t in self.TRACE_TARGETS for t in self.target_type)
        self.fnames: List[str] = []
        self.outcome: List[list] = []
        self.frames = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)
        self.large_frame_index: Dict[str, int] = {}
        self.small_frame_index: Dict[str, int] = {}

        if self.split == "EXTERNAL_TEST":
            if self.external_test_location is None:
                raise ValueError("external_test_location is required for EXTERNAL_TEST")
            self.fnames = sorted(os.listdir(self.external_test_location))
            self.header = []
            return

        file_list_path = os.path.join(self.root, "FileList.csv")
        data = pd.read_csv(file_list_path)
        if "Split" not in data.columns or "FileName" not in data.columns:
            raise ValueError("FileList.csv must contain FileName and Split columns")

        data["Split"] = data["Split"].astype(str).str.upper()
        if self.split != "ALL":
            data = data[data["Split"] == self.split].copy()

        self.header = data.columns.tolist()
        self.fnames = data["FileName"].astype(str).map(_normalise_filename).tolist()
        self.outcome = data.values.tolist()

        videos_dir = os.path.join(self.root, "Videos")
        missing = set(self.fnames) - set(os.listdir(videos_dir))
        if missing:
            first = sorted(missing)[0]
            raise FileNotFoundError(
                f"{len(missing)} listed videos are missing; first missing file: "
                f"{os.path.join(videos_dir, first)}"
            )

        # Crucial correction: video-only EF must not depend on VolumeTracings.csv.
        if self.requires_traces:
            self._load_tracings()
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for f, k in zip(self.fnames, keep) if k]
            self.outcome = [o for o, k in zip(self.outcome, keep) if k]

            for filename in self.fnames:
                areas = [
                    (self._trace_polygon_area(self.trace[filename][frame]), frame)
                    for frame in self.frames[filename]
                ]
                areas.sort(key=lambda x: x[0])
                self.small_frame_index[filename] = int(areas[0][1])
                self.large_frame_index[filename] = int(areas[-1][1])

    def _load_tracings(self) -> None:
        tracing_path = os.path.join(self.root, "VolumeTracings.csv")
        with open(tracing_path, "r") as f:
            header = f.readline().strip().split(",")
            expected = ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
            if header != expected:
                raise ValueError(
                    f"Unexpected VolumeTracings.csv header: {header}; expected {expected}"
                )
            for line in f:
                if not line.strip():
                    continue
                filename, x1, y1, x2, y2, frame = line.strip().split(",")
                filename = _normalise_filename(filename)
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

    @staticmethod
    def _trace_polygon_xy(trace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
        x = np.concatenate((x1[1:], np.flip(x2[1:])))
        y = np.concatenate((y1[1:], np.flip(y2[1:])))
        return x, y

    @classmethod
    def _trace_polygon_area(cls, trace: np.ndarray) -> float:
        x, y = cls._trace_polygon_xy(trace)
        if len(x) < 3:
            return 0.0
        return float(
            0.5
            * np.abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
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

        n_starts = f - (length - 1) * self.period
        if self.clips == "all":
            start = np.arange(n_starts)
        elif self.clip_start == "center" and int(self.clips) == 1:
            start = np.asarray([(n_starts - 1) // 2], dtype=int)
        else:
            start = np.random.choice(n_starts, int(self.clips))

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
        sampled_video = clips[0] if self.clips == 1 else np.stack(clips)

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
