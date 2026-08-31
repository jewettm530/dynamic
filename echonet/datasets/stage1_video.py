"""Datasets for the corrected Stage 1 video-based experiments.

The corrected EF task must be able to run from ``FileList.csv`` + ``Videos/``
without ``VolumeTracings.csv``.  Segmentation labels are therefore loaded only
when explicitly requested.

B1 (video EF-only)
    Uses ``include_segmentation=False`` and never opens VolumeTracings.csv.

B2 (segmentation-only)
    Uses ``include_segmentation=True`` and ``require_segmentation=True``.

B3 (video multi-task)
    Uses the full video split for EF and sparse ED/ES labels where available.
    The sampled video clip is independent of the expert ED/ES locations.
"""

from __future__ import annotations

import collections
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import skimage.draw
import torch
from torch.utils.data import Dataset

import echonet


def _normalise_filename(value: str) -> str:
    value = str(value)
    return value if os.path.splitext(value)[1] else value + ".avi"


def _defaultdict_of_lists():
    return collections.defaultdict(list)


class Stage1VideoDataset(Dataset):
    """Corrected Stage 1 dataset.

    Parameters
    ----------
    root:
        EchoNet-Dynamic dataset root containing FileList.csv and Videos/.
    split:
        train, val, or test.
    frames / period:
        A clip contains ``frames`` samples separated by ``period`` source
        frames.  ``frames`` must be > 2 for the corrected EF experiment.
    clip_sampling:
        ``random`` for training or ``center`` for deterministic validation/test.
    include_segmentation:
        If True, load sparse ED/ES tracings and return ED/ES images and masks.
        If False, VolumeTracings.csv is neither required nor opened.
    require_segmentation:
        If True, filter to videos with two traced frames.  This is used by B2
        and by segmentation evaluation.  B3 training leaves this False so its
        EF cohort remains identical to B1; segmentation loss is simply skipped
        for the very small number of untraced videos.
    include_video:
        If False, do not sample/return a video clip.  Useful for B2 and
        segmentation-only evaluation.
    """

    TRACE_HEADER = ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

    def __init__(
        self,
        root: str,
        split: str,
        *,
        frames: int = 32,
        period: int = 2,
        clip_sampling: str = "random",
        mean: float | np.ndarray = 0.0,
        std: float | np.ndarray = 1.0,
        include_segmentation: bool = False,
        require_segmentation: bool = False,
        include_video: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.split = split.upper()
        if self.split not in {"TRAIN", "VAL", "TEST"}:
            raise ValueError(f"Unsupported split: {split}")
        if frames <= 2:
            raise ValueError(
                f"Corrected Stage 1 requires a video sequence with T > 2; got frames={frames}"
            )
        if period < 1:
            raise ValueError("period must be >= 1")
        if clip_sampling not in {"random", "center"}:
            raise ValueError("clip_sampling must be 'random' or 'center'")
        if require_segmentation and not include_segmentation:
            raise ValueError("require_segmentation=True requires include_segmentation=True")

        self.frames_per_clip = int(frames)
        self.period = int(period)
        self.clip_sampling = clip_sampling
        self.mean = mean
        self.std = std
        self.include_segmentation = bool(include_segmentation)
        self.require_segmentation = bool(require_segmentation)
        self.include_video = bool(include_video)

        file_list = self.root / "FileList.csv"
        videos_dir = self.root / "Videos"
        if not file_list.exists():
            raise FileNotFoundError(file_list)
        if not videos_dir.is_dir():
            raise FileNotFoundError(videos_dir)

        data = pd.read_csv(file_list)
        required_columns = {"FileName", "Split", "EF"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"FileList.csv is missing required columns: {sorted(missing_columns)}"
            )

        data = data.copy()
        data["Split"] = data["Split"].astype(str).str.upper()
        data = data[data["Split"] == self.split].copy()
        data["_filename"] = data["FileName"].astype(str).map(_normalise_filename)

        listed = data["_filename"].tolist()
        present = set(os.listdir(videos_dir))
        missing_videos = sorted(set(listed) - present)
        if missing_videos:
            raise FileNotFoundError(
                f"{len(missing_videos)} listed videos are missing; first: "
                f"{videos_dir / missing_videos[0]}"
            )

        self.trace_frames: Dict[str, List[int]] = collections.defaultdict(list)
        self.traces: Dict[str, Dict[int, np.ndarray]] = collections.defaultdict(dict)
        self.large_frame_index: Dict[str, int] = {}
        self.small_frame_index: Dict[str, int] = {}

        if self.include_segmentation:
            self._load_tracings()
            has_trace = data["_filename"].map(
                lambda fn: len(self.trace_frames.get(fn, [])) >= 2
            )
            data["_has_segmentation"] = has_trace.astype(bool)
            if self.require_segmentation:
                data = data[data["_has_segmentation"]].copy()
        else:
            # This path intentionally does not even stat VolumeTracings.csv.
            data["_has_segmentation"] = False

        self.rows = data.reset_index(drop=True)
        self.fnames = self.rows["_filename"].astype(str).tolist()

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

    def _load_tracings(self) -> None:
        tracing_path = self.root / "VolumeTracings.csv"
        if not tracing_path.exists():
            raise FileNotFoundError(
                f"Segmentation labels were requested but {tracing_path} does not exist"
            )

        raw: Dict[str, Dict[int, List[tuple]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        with tracing_path.open("r") as f:
            header = f.readline().strip().split(",")
            if header != self.TRACE_HEADER:
                raise ValueError(
                    f"Unexpected VolumeTracings.csv header: {header}; "
                    f"expected {self.TRACE_HEADER}"
                )
            for line in f:
                if not line.strip():
                    continue
                filename, x1, y1, x2, y2, frame = line.strip().split(",")
                filename = _normalise_filename(filename)
                frame_i = int(frame)
                raw[filename][frame_i].append(
                    (float(x1), float(y1), float(x2), float(y2))
                )

        for filename, frame_map in raw.items():
            frames = sorted(frame_map.keys())
            self.trace_frames[filename] = frames
            self.traces[filename] = {
                frame: np.asarray(frame_map[frame], dtype=np.float32)
                for frame in frames
            }
            if len(frames) >= 2:
                area_frame = [
                    (self._trace_polygon_area(self.traces[filename][frame]), frame)
                    for frame in frames
                ]
                area_frame.sort(key=lambda item: item[0])
                self.small_frame_index[filename] = int(area_frame[0][1])
                self.large_frame_index[filename] = int(area_frame[-1][1])

    def _normalise_video(self, video: np.ndarray) -> np.ndarray:
        video = video.astype(np.float32, copy=False)
        if isinstance(self.mean, (float, int)):
            video = video - float(self.mean)
        else:
            video = video - np.asarray(self.mean, dtype=np.float32).reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video = video / float(self.std)
        else:
            video = video / np.asarray(self.std, dtype=np.float32).reshape(3, 1, 1, 1)
        return video

    def _sample_clip(self, video: np.ndarray) -> tuple[np.ndarray, int]:
        c, f, h, w = video.shape
        required = self.frames_per_clip * self.period
        if f < required:
            pad = np.zeros((c, required - f, h, w), dtype=video.dtype)
            video = np.concatenate((video, pad), axis=1)
            f = video.shape[1]

        n_starts = f - (self.frames_per_clip - 1) * self.period
        if n_starts <= 0:
            raise RuntimeError(
                f"Unable to sample {self.frames_per_clip} frames at period={self.period} "
                f"from video with {f} frames"
            )

        if self.clip_sampling == "random":
            start = int(np.random.randint(0, n_starts))
        else:
            start = int((n_starts - 1) // 2)

        indices = start + self.period * np.arange(self.frames_per_clip)
        clip = video[:, indices, :, :]
        return clip, start

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        filename = str(row["_filename"])
        video_path = self.root / "Videos" / filename
        video = self._normalise_video(echonet.utils.loadvideo(str(video_path)))
        c, f, h, w = video.shape

        item = {
            "filename": filename,
            "ef": torch.tensor(float(row["EF"]) / 100.0, dtype=torch.float32),
            "has_segmentation": torch.tensor(
                bool(row["_has_segmentation"]), dtype=torch.bool
            ),
        }

        if self.include_video:
            clip, clip_start = self._sample_clip(video)
            item["video"] = torch.as_tensor(clip, dtype=torch.float32)
            item["clip_start"] = torch.tensor(clip_start, dtype=torch.int64)

        if self.include_segmentation:
            has_seg = bool(row["_has_segmentation"])
            if has_seg:
                ed_index = self.large_frame_index[filename]
                es_index = self.small_frame_index[filename]
                if ed_index >= f or es_index >= f:
                    raise IndexError(
                        f"Tracing index exceeds video length for {filename}: "
                        f"ED={ed_index}, ES={es_index}, frames={f}"
                    )
                ed_frame = video[:, ed_index, :, :]
                es_frame = video[:, es_index, :, :]
                ed_mask = self._trace_to_mask(
                    self.traces[filename][ed_index], h, w
                )
                es_mask = self._trace_to_mask(
                    self.traces[filename][es_index], h, w
                )
            else:
                # Fixed-size placeholders keep DataLoader collation simple.
                # They are never sent to the segmentation branch because the
                # training loop filters with has_segmentation first.
                ed_frame = np.zeros((c, h, w), dtype=np.float32)
                es_frame = np.zeros((c, h, w), dtype=np.float32)
                ed_mask = np.zeros((h, w), dtype=np.float32)
                es_mask = np.zeros((h, w), dtype=np.float32)
                ed_index = -1
                es_index = -1

            item.update(
                {
                    "ed_image": torch.as_tensor(ed_frame, dtype=torch.float32),
                    "es_image": torch.as_tensor(es_frame, dtype=torch.float32),
                    "ed_mask": torch.as_tensor(ed_mask, dtype=torch.float32).unsqueeze(0),
                    "es_mask": torch.as_tensor(es_mask, dtype=torch.float32).unsqueeze(0),
                    "ed_index": torch.tensor(ed_index, dtype=torch.int64),
                    "es_index": torch.tensor(es_index, dtype=torch.int64),
                }
            )

        return item

    @property
    def n_with_segmentation(self) -> int:
        if "_has_segmentation" not in self.rows:
            return 0
        return int(self.rows["_has_segmentation"].sum())
