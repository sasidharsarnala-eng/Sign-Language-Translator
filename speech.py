"""Helpers for converting WebRTC audio frames into SpeechRecognition input."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import speech_recognition as sr
import streamlit as st


def frame_to_mono_pcm(frame: Any) -> tuple[bytes, int]:
    """Convert one PyAV audio frame to signed 16-bit mono PCM."""
    samples = frame.to_ndarray()
    if samples.ndim == 2:
        # AudioFrame arrays are typically channels-first.
        samples = samples.astype(np.float32).mean(axis=0)

    if np.issubdtype(samples.dtype, np.floating):
        samples = np.clip(samples, -1.0, 1.0) * 32767
    elif samples.dtype != np.int16:
        maximum = float(np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) * (32767.0 / maximum)

    pcm = samples.astype(np.int16).tobytes()
    return pcm, int(frame.sample_rate)


def transcribe_pcm(recognizer: sr.Recognizer, pcm: bytes, sample_rate: int) -> str:
    """Transcribe one captured microphone chunk using SpeechRecognition."""
    audio = sr.AudioData(pcm, sample_rate, sample_width=2)
    return recognizer.recognize_google(audio)


@st.cache_resource
def get_transcription_executor() -> ThreadPoolExecutor:
    """Return a shared single-worker executor to keep UI updates responsive."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech-recognition")
