import torch
import os
from pathlib import Path
from functools import lru_cache
from typing import Union
from src.core import config
from src.utils import logger
from whisper import load_model, Whisper


@lru_cache(maxsize=1)
def _get_s2t_model() -> Whisper:
    logger.info(f"Loading speech to text model: whisper-{config.SPEECH2TEXT_MODEL}")
    return load_model(config.SPEECH2TEXT_MODEL, device="cuda")


def _ensure_list(paths: Union[str, list[str]]) -> list[str]:
    if isinstance(paths, str):
        return [paths]
    return paths


def transcribe_audio(
    audio_paths: Union[str, list[str]],
    out_dir: str = config.TRANSCRIPT_STORAGE_PATH,
    language: str = "vi",
    use_fp16: bool = True,
) -> list[str]:

    audio_paths = _ensure_list(audio_paths)
    if not audio_paths:
        logger.warning("No valid audio files provided to transcribe_audio")
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_s2t_model()
    model = model.to(device)

    os.makedirs(out_dir, exist_ok=True)

    transcripts = []
    for audio_path in audio_paths:
        logger.info(f"Transcribing audio file: {audio_path}")

        result = model.transcribe(audio_path, language=language, fp16=use_fp16)

        transcript = ""
        for seg in result["segments"]:
            s, e, t = seg["start"], seg["end"], seg["text"].strip()
            transcript += f"[{s:.2f}s - {e:.2f}s] {t}\n"

        transcripts.append(transcript)

    for audio_path, transcript in zip(audio_paths, transcripts):
        audio_filename = Path(audio_path).stem
        transcript_path = os.path.join(out_dir, f"{audio_filename}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

    if device == "cuda":
        model.to(device="cpu")
        torch.cuda.empty_cache()

    return transcripts
