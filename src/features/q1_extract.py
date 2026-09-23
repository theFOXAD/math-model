"""Deterministic lightweight multimodal descriptors for E-problem question 1."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly


EPS = 1e-8


@dataclass
class MediaData:
    duration: float
    container_duration: float
    audio_duration: float
    video_duration: float
    audio: np.ndarray
    audio_rate: int
    video_frames: list[np.ndarray]
    video_times: np.ndarray
    source_fps: float


def make_anchors(text: str, max_length: int) -> list[str]:
    words = re.findall(r"\S+", text.strip())
    if not words:
        return []
    groups = np.array_split(np.asarray(words, dtype=object), min(len(words), max_length))
    return [" ".join(group.tolist()) for group in groups]


def proportional_boundaries(anchors: list[str], duration: float) -> np.ndarray:
    if not anchors or duration <= 0:
        return np.asarray([0.0], dtype=np.float64)
    weights = np.asarray([max(1, len(re.sub(r"\s+", "", a))) for a in anchors], dtype=np.float64)
    return np.concatenate(([0.0], duration * np.cumsum(weights) / weights.sum()))


def text_descriptor(anchor: str, left: str, right: str, position: int, total: int, dim: int) -> np.ndarray:
    value = anchor.lower()
    padded = f"^^{value}$$"
    events = [f"tri:{padded[i:i+3]}" for i in range(max(0, len(padded) - 2))]
    events += [f"tok:{token}" for token in re.findall(r"[a-z0-9']+|[^\w\s]", value)]
    events += [f"left:{left.lower()}", f"right:{right.lower()}"]
    vec = np.zeros(dim, dtype=np.float32)
    for event in events:
        digest = hashlib.sha256(event.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        vec[bucket] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(vec))
    if norm > EPS:
        vec /= norm
    stats = [
        min(len(anchor), 100) / 100.0,
        (position + 1) / max(total, 1),
        sum(ch.isupper() for ch in anchor) / max(len(anchor), 1),
        sum(ch.isdigit() for ch in anchor) / max(len(anchor), 1),
        sum(not ch.isalnum() and not ch.isspace() for ch in anchor) / max(len(anchor), 1),
    ]
    vec[: len(stats)] += np.asarray(stats, dtype=np.float32)
    vec /= max(float(np.linalg.norm(vec)), EPS)
    return vec


def decode_media(path: Path, target_audio_rate: int, target_video_fps: float) -> MediaData:
    audio_chunks: list[np.ndarray] = []
    source_audio_rate = target_audio_rate
    container_duration = 0.0
    with av.open(str(path)) as container:
        if container.duration is not None:
            container_duration = float(container.duration / av.time_base)
        if not container.streams.audio:
            audio = np.empty(0, dtype=np.float32)
        else:
            stream = container.streams.audio[0]
            for frame in container.decode(stream):
                source_audio_rate = int(frame.sample_rate)
                arr = frame.to_ndarray().astype(np.float32)
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                audio_chunks.append(arr.reshape(-1))
            audio = np.concatenate(audio_chunks) if audio_chunks else np.empty(0, dtype=np.float32)
            if audio.size:
                peak = max(float(np.max(np.abs(audio))), 1.0)
                audio = audio / peak
                if source_audio_rate != target_audio_rate:
                    divisor = math.gcd(source_audio_rate, target_audio_rate)
                    audio = resample_poly(audio, target_audio_rate // divisor, source_audio_rate // divisor)
    frames: list[np.ndarray] = []
    times: list[float] = []
    source_fps = 0.0
    video_duration = 0.0
    with av.open(str(path)) as container:
        if container.streams.video:
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate or 0.0)
            if stream.duration is not None and stream.time_base is not None:
                video_duration = float(stream.duration * stream.time_base)
            next_time = 0.0
            for index, frame in enumerate(container.decode(stream)):
                timestamp = float(frame.time) if frame.time is not None else index / max(source_fps, target_video_fps)
                frame_width = (float(frame.duration * frame.time_base)
                               if frame.duration is not None and frame.time_base is not None
                               else 1.0 / max(source_fps, target_video_fps))
                video_duration = max(video_duration, timestamp + frame_width)
                if timestamp + 1e-9 >= next_time:
                    frames.append(frame.to_ndarray(format="bgr24"))
                    times.append(timestamp)
                    next_time += 1.0 / target_video_fps
    audio_duration = audio.size / target_audio_rate if audio.size else 0.0
    bounded_durations = [value for value in (container_duration, audio_duration, video_duration) if value > 0]
    duration = min(bounded_durations) if bounded_durations else 0.0
    return MediaData(duration, container_duration, audio_duration, video_duration,
                     audio.astype(np.float32), target_audio_rate, frames,
                     np.asarray(times, dtype=np.float64), source_fps)


def _audio_frames(segment: np.ndarray, rate: int, window_sec: float, hop_sec: float) -> np.ndarray:
    width = max(1, round(window_sec * rate))
    hop = max(1, round(hop_sec * rate))
    if segment.size == 0:
        return np.empty((0, width), dtype=np.float32)
    if segment.size < width:
        segment = np.pad(segment, (0, width - segment.size))
    count = 1 + (segment.size - width) // hop
    indexes = np.arange(width)[None, :] + hop * np.arange(count)[:, None]
    return segment[indexes] * np.hanning(width).astype(np.float32)


def audio_descriptor(segment: np.ndarray, rate: int, window_sec: float, hop_sec: float,
                     mel_bins: int, mfcc_bins: int, interval_sec: float) -> tuple[np.ndarray, int]:
    frames = _audio_frames(segment, rate, window_sec, hop_sec)
    if not len(frames):
        return np.zeros(74, dtype=np.float32), 0
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    # Standard triangular Mel filterbank over the non-negative FFT bins.
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(rate / 2.0), mel_bins + 2)
    hz_points = mel_to_hz(mel_points)
    fft_freqs = np.fft.rfftfreq(frames.shape[1], 1.0 / rate)
    filters = np.zeros((mel_bins, len(fft_freqs)), dtype=np.float64)
    for band in range(mel_bins):
        left, center, right = hz_points[band:band + 3]
        filters[band] = np.maximum(0.0, np.minimum((fft_freqs - left) / max(center - left, EPS),
                                                  (right - fft_freqs) / max(right - center, EPS)))
        filters[band] /= max(filters[band].sum(), EPS)
    mel_energy = spectrum @ filters.T
    logmel = np.log(mel_energy + EPS)
    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :mfcc_bins]
    delta = np.diff(mfcc, axis=0, prepend=mfcc[:1])
    energy = np.mean(frames ** 2, axis=1)
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    freqs = np.fft.rfftfreq(frames.shape[1], 1.0 / rate)
    centroid = (spectrum * freqs).sum(axis=1) / (spectrum.sum(axis=1) + EPS)
    stats = np.asarray([energy.mean(), energy.std(), zcr.mean(), zcr.std(),
                        centroid.mean() / (rate / 2), centroid.std() / (rate / 2),
                        1.0, interval_sec], dtype=np.float32)
    result = np.concatenate([logmel.mean(axis=0), mfcc.mean(axis=0), delta.mean(axis=0), stats])
    return result.astype(np.float32), len(frames)


def visual_descriptors(frames: list[np.ndarray]) -> tuple[np.ndarray, float]:
    if not frames:
        return np.empty((0, 35), dtype=np.float32), 0.0
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    rows: list[np.ndarray] = []
    previous_gray: np.ndarray | None = None
    detected = 0
    for frame in frames:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = np.concatenate([np.histogram(hsv[..., channel], bins=8,
                              range=(0, 180) if channel == 0 else (0, 256), density=True)[0]
                              for channel in range(3)]).astype(np.float32)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(12, 12))
        if len(faces):
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            detected += 1
            face = np.asarray([1.0, (x + w / 2) / 160, (y + h / 2) / 90,
                               w / 160, h / 90, (w * h) / (160 * 90)], dtype=np.float32)
        else:
            face = np.zeros(6, dtype=np.float32)
        if previous_gray is None:
            motion = np.zeros(3, dtype=np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            magnitude = np.linalg.norm(flow, axis=2)
            motion = np.asarray([magnitude.mean(), magnitude.std(), np.quantile(magnitude, 0.9)], dtype=np.float32)
        previous_gray = gray
        luminance_edge = np.asarray([gray.mean() / 255.0,
                                     np.mean(cv2.Canny(gray, 80, 160) > 0)], dtype=np.float32)
        rows.append(np.concatenate([hist, face, motion, luminance_edge]))
    return np.stack(rows).astype(np.float32), detected / len(frames)


def extract_sample(sample, config: dict) -> dict:
    max_length = int(config["output"]["max_length"])
    anchors = make_anchors(sample.text, max_length)
    acfg, vcfg, tcfg = config["audio"], config["vision"], config["text"]
    media = decode_media(sample.source_path, int(acfg["sample_rate"]), float(vcfg["sample_fps"]))
    boundaries = proportional_boundaries(anchors, media.duration)
    text = np.zeros((max_length, int(tcfg["dimension"])), dtype=np.float32)
    audio = np.zeros((max_length, 74), dtype=np.float32)
    vision = np.zeros((max_length, 35), dtype=np.float32)
    valid_text = np.zeros(max_length, dtype=bool)
    valid_audio = np.zeros(max_length, dtype=bool)
    valid_vision = np.zeros(max_length, dtype=bool)
    padding = np.ones(max_length, dtype=bool)
    frame_desc, face_rate = visual_descriptors(media.video_frames)
    alignments = []
    for idx, anchor in enumerate(anchors):
        start, end = float(boundaries[idx]), float(boundaries[idx + 1])
        text[idx] = text_descriptor(anchor, anchors[idx - 1] if idx else "<BOS>",
                                    anchors[idx + 1] if idx + 1 < len(anchors) else "<EOS>",
                                    idx, len(anchors), text.shape[1])
        valid_text[idx] = True
        padding[idx] = False
        # Both adjacent intervals derive their endpoint from the same rounded
        # boundary, so the half-open sample ranges are contiguous and disjoint.
        a0 = max(0, min(len(media.audio), round(start * media.audio_rate)))
        a1 = max(a0, min(len(media.audio), round(end * media.audio_rate)))
        audio[idx], audio_frame_count = audio_descriptor(media.audio[a0:a1], media.audio_rate,
                                                          float(acfg["window_sec"]), float(acfg["hop_sec"]),
                                                          int(acfg["mel_bins"]), int(acfg["mfcc_bins"]), end - start)
        valid_audio[idx] = a1 > a0 and audio_frame_count > 0
        v0 = int(np.searchsorted(media.video_times, start, side="left"))
        v1 = int(np.searchsorted(media.video_times, end, side="left"))
        if idx == len(anchors) - 1:
            v1 = len(media.video_times)
        if v1 > v0:
            vision[idx] = frame_desc[v0:v1].mean(axis=0)
            valid_vision[idx] = True
        expected_audio = max(1, 1 + math.floor(max(0.0, end - start - float(acfg["window_sec"])) /
                                               float(acfg["hop_sec"])))
        expected_video = max(1, math.ceil((end - start) * float(vcfg["sample_fps"])))
        qa = min(1.0, audio_frame_count / expected_audio) if valid_audio[idx] else 0.0
        qv = min(1.0, (v1 - v0) / expected_video) if valid_vision[idx] else 0.0
        confidence = 0.35 + 0.30 * min(qa, qv) if anchor and end > start else 0.0
        alignments.append({"sample_id": sample.sample_id, "token_index": idx, "token_text": anchor,
                           "start_sec": start, "end_sec": end, "audio_start_idx": a0,
                           "audio_end_idx": a1, "video_start_idx": v0, "video_end_idx": v1,
                           "confidence": confidence, "method": "proportional", "is_fallback": True})
    return {"anchors": anchors, "duration": media.duration,
            "source_duration": media.container_duration,
            "audio_duration": media.audio_duration, "video_duration": media.video_duration,
            "source_fps": media.source_fps,
            "audio_rate": media.audio_rate, "face_detection_rate": face_rate,
            "audio_sample_count": len(media.audio), "video_frame_count": len(media.video_frames),
            "text": text, "audio": audio, "vision": vision, "valid_text": valid_text,
            "valid_audio": valid_audio, "valid_vision": valid_vision, "padding_mask": padding,
            "injected_missing_audio": np.zeros(max_length, dtype=bool),
            "injected_missing_vision": np.zeros(max_length, dtype=bool), "alignments": alignments}
