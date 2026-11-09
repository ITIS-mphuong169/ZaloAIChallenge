"""Speech to text.

Two backends with the same interface:
  ct2  faster-whisper (CTranslate2): a model name such as large-v3-turbo, or a folder holding model.bin
  hf   transformers + torch: a folder in Hugging Face format, for example the one lifemate.finetune_asr writes

Transcripts are cached in a JSONL file keyed by the audio path relative to the split folder, so a run can be stopped
and resumed and the agent never transcribes the same clip twice.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata

from .data import root


def audio_paths(sample: dict) -> list[str]:
    """Every clip of a sample: the user turns of the history in order, then the final request."""
    out = [h["audio"] for h in sample.get("history", []) if h.get("role") == "user" and h.get("audio")]
    return out + [sample["audio"]]


def load_cache(path: str, failed: set | None = None) -> dict:
    """{audio path: transcript}. Clips whose transcription failed are in the file with an empty text and an "error"
    field; pass a set as `failed` to collect their paths (the last line written for a clip wins)."""
    cache = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cache[r["audio"]] = r["text"]
                    if failed is not None:
                        (failed.add if r.get("error") else failed.discard)(r["audio"])
    return cache


class Transcriber:
    def __init__(self, model: str = "large-v3-turbo", device: str = "auto", compute_type: str = "default",
                 threads: int = 0, beam_size: int = 5, prompt: str = ""):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=threads or os.cpu_count() or 4)
        self.beam_size = beam_size
        self.prompt = prompt or None  # text shown to the decoder as preceding context: biases it towards these words

    def __call__(self, wav_path: str) -> str:
        segments, _info = self.model.transcribe(read_wav(wav_path), language="vi", beam_size=self.beam_size,
                                                initial_prompt=self.prompt,
                                                condition_on_previous_text=False, vad_filter=False)
        return " ".join(s.text.strip() for s in segments).strip()


def hf_transcribe(model, processor, audios: list, beam_size: int = 1, max_new_tokens: int = 200) -> list[str]:
    """Transcribe a batch of 16 kHz float32 arrays with a transformers Whisper model (Vietnamese, no timestamps)."""
    import torch
    feats = processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features
    feats = feats.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        try:
            ids = model.generate(input_features=feats, language="vi", task="transcribe", num_beams=beam_size, max_new_tokens=max_new_tokens)
        except ValueError:  # older checkpoints (PhoWhisper) ship a generation config without the language table
            model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
            ids = model.generate(input_features=feats, num_beams=beam_size, max_new_tokens=max_new_tokens)
    return [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]


class HFTranscriber:
    """A Whisper model in Hugging Face format, run with transformers. Written without a GPU at hand: not yet run."""

    def __init__(self, model: str, device: str = "auto", beam_size: int = 5):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model)
        m = WhisperForConditionalGeneration.from_pretrained(model).to(device)
        self.model = (m.half() if device.startswith("cuda") else m.float()).eval()
        self.beam_size = beam_size

    def __call__(self, wav_path: str) -> str:
        return hf_transcribe(self.model, self.processor, [read_wav(wav_path)], self.beam_size)[0]


def make_transcriber(model: str, device: str = "auto", compute_type: str = "default", beam_size: int = 5, prompt: str = "",
                     backend: str = "auto"):
    """Pick the backend from what the model folder contains: model.bin means CTranslate2, model.safetensors means transformers."""
    if backend == "auto":
        hf = os.path.isdir(model) and not os.path.exists(os.path.join(model, "model.bin")) and any(
            os.path.exists(os.path.join(model, f)) for f in ("model.safetensors", "pytorch_model.bin"))
        backend = "hf" if hf else "ct2"
    if backend == "hf":
        return HFTranscriber(model, device=device, beam_size=beam_size)
    return Transcriber(model, device=device, compute_type=compute_type, beam_size=beam_size, prompt=prompt)


def read_wav(path: str, rate: int = 16000):
    """PCM wav -> mono float32 at 16 kHz. The released clips are 16 kHz mono 16-bit already; reading them here keeps the
    audio decoding independent of the PyAV version faster-whisper would otherwise use."""
    import wave

    import numpy as np
    with wave.open(path, "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {width} in {path}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != rate and len(x):
        pos = np.linspace(0, len(x) - 1, int(round(len(x) * rate / sr)))
        x = np.interp(pos, np.arange(len(x)), x).astype(np.float32)
    return x


def transcribe_split(split: str, samples: list[dict], cache_path: str, tr, log=print, final_only: bool = False) -> dict:
    failed: set = set()
    cache = load_cache(cache_path, failed)
    todo = [p for s in samples for p in ([s["audio"]] if final_only else audio_paths(s)) if p not in cache or p in failed]
    todo = list(dict.fromkeys(todo))
    if todo:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        t0 = time.time()
        with open(cache_path, "a", encoding="utf-8") as f:
            for i, p in enumerate(todo, 1):
                row = {"audio": p}
                try:
                    row["text"] = tr(str(root() / split / p))
                except Exception as e:  # one unreadable clip must not stop the run: empty transcript, retried next time
                    row["text"], row["error"] = "", repr(e)[:200]
                    log(f"[asr] FAILED {p}: {row['error']}")
                cache[p] = row["text"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(todo):
                    log(f"[asr] {i}/{len(todo)} clips, {time.time() - t0:.0f}s")
    return cache


# ------------------------------------------------------------------ word error rate
def _norm_words(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"[^\w\s]", " ", s).split()


def _edit(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(pairs: list[tuple[str, str]]) -> float:
    """Word error rate over (reference, hypothesis) pairs; case and punctuation ignored."""
    err = n = 0
    for ref, hyp in pairs:
        r, h = _norm_words(ref), _norm_words(hyp)
        err += _edit(r, h)
        n += len(r)
    return err / max(1, n)
