#!/usr/bin/env python3
"""VieNeu TTS-only worker for GitHub Actions.

The web app sends one short-lived job containing base64url-encoded Vietnamese
text. This worker performs TTS on the runner and writes a WAV artifact; it does
not download or render a source video.
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from vieneu import Vieneu


def decode_text(value: str) -> str:
    padded = value + "=" * (-len(value) % 4)
    raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    text = raw.decode("utf-8").strip()
    if not text:
        raise ValueError("Nội dung TTS trống.")
    if len(text) > 50_000:
        raise ValueError("Nội dung TTS vượt giới hạn 50.000 ký tự.")
    return text


def split_text(text: str, limit: int = 350) -> list[str]:
    """Keep VieNeu inputs below its stable per-item character limit."""
    parts: list[str] = []
    rest = text.strip()
    while len(rest) > limit:
        window = rest[:limit]
        matches = list(re.finditer(r"[\n.!?;:。！？]\s*", window))
        cut = matches[-1].end() if matches and matches[-1].end() > limit * 0.45 else limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return parts


def prefetch_models() -> None:
    from huggingface_hub import hf_hub_download

    model_files = [
        "vieneu_prefill.onnx",
        "vieneu_decode_step.onnx",
        "vieneu_acoustic_cached.onnx",
        "vieneu_backbone_shared.data",
        "vieneu_v3_heads.npz",
        "config.json",
        "tokenizer.json",
    ]
    tokenizer_files = [
        "moss_audio_tokenizer_decode_full.onnx",
        "moss_audio_tokenizer_decode_shared.data",
        "moss_audio_tokenizer_decode_step.onnx",
        "codec_browser_onnx_meta.json",
    ]
    for filename in model_files:
        try:
            hf_hub_download("pnnbao-ump/VieNeu-TTS-v3-Turbo", filename, subfolder="onnx_update")
        except Exception:
            pass
    for filename in tokenizer_files:
        try:
            hf_hub_download("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX", filename)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--text-b64", required=True)
    parser.add_argument("--voice", default="Minh Quân")
    parser.add_argument("--emotion", default="")
    args = parser.parse_args()

    started = time.time()
    text = decode_text(args.text_b64)
    pieces = split_text(text)
    requested_voice = args.voice.strip() or "Minh Quân"
    emotion = args.emotion.strip()
    if emotion not in {"", "[cười]", "[thở dài]", "[hắng giọng]"}:
        emotion = ""

    print(f"[TTS] Job {args.job_id}: {len(text)} ký tự, {len(pieces)} đoạn.", flush=True)
    prefetch_models()
    tts = Vieneu(mode="v3turbo", backend="onnx", device="cpu")
    available = getattr(tts, "_preset_voices", {})
    voice = requested_voice if requested_voice in available else (
        "Minh Quân" if "Minh Quân" in available else next(iter(available), requested_voice)
    )
    if voice != requested_voice:
        print(f"[TTS] Giọng '{requested_voice}' không có trên runner; dùng '{voice}'.", flush=True)

    audio_parts: list[np.ndarray] = []
    sample_rate = 48_000
    silence = np.zeros(int(sample_rate * 0.025), dtype=np.float32)
    for start in range(0, len(pieces), 16):
        batch = pieces[start : start + 16]
        inputs = [f"{part} {emotion}".strip() if emotion else part for part in batch]
        audios = tts.infer_batch(
            inputs,
            voice=voice,
            batch_size=len(inputs),
            max_new_frames=600,
            max_chars=350,
            denoise=True,
            apply_watermark=False,
        )
        if len(audios) != len(batch):
            raise RuntimeError("VieNeu trả thiếu đoạn audio.")
        for audio in audios:
            audio_parts.append(np.asarray(audio, dtype=np.float32))
            audio_parts.append(silence)
        done = min(start + len(batch), len(pieces))
        print(f"[TTS] Tiến độ: {done}/{len(pieces)} đoạn.", flush=True)

    audio = np.concatenate(audio_parts) if audio_parts else np.zeros(1, dtype=np.float32)
    output_dir = Path("workspace/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "tts.wav"
    sf.write(wav_path, audio, sample_rate, subtype="PCM_16")
    metadata = {
        "job_id": args.job_id,
        "requested_voice": requested_voice,
        "voice": voice,
        "emotion": emotion,
        "input_chars": len(text),
        "piece_count": len(pieces),
        "duration_seconds": len(audio) / sample_rate,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "tts-meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[TTS] Hoàn tất trong {metadata['elapsed_seconds']:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
