"""Fine-tune a Whisper-family speech model on the train clips.

    python -m lifemate.finetune_asr --base openai/whisper-large-v3-turbo --out models/asr-turbo-ft
    python -m lifemate.finetune_asr --base vinai/PhoWhisper-large --out models/asr-pho-ft --batch 4 --accum 4

The train set has 5,036 clips with transcripts: the final request of each sample and the user turns of its history.
Clips of the dev samples (every 10th train sample, see lifemate.data.dev_split) are never trained on. Their word error
rate is printed before training and after every epoch, and the checkpoint with the lowest value is the one kept in --out.

Needs a CUDA GPU, torch and transformers. Written against transformers 5.22 on a machine without a GPU, so it has NOT
been run yet: start with `--max-steps 20 --dev-limit 32` to shake out problems before a full run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

from .asr import hf_transcribe, read_wav, wer
from .data import dev_split, load_samples, root


def clip_pairs(samples: list[dict]) -> list[tuple[str, str]]:
    """(audio path, transcript) of every clip of the samples, each clip once."""
    seen, out = set(), []
    for s in samples:
        turns = [(h.get("audio"), h.get("audio_text")) for h in s.get("history", []) if h.get("role") == "user"]
        for path, text in turns + [(s["audio"], s["audio_text"])]:
            if path and text and path not in seen:
                seen.add(path)
                out.append((path, text))
    return out


class Collate:
    """Turns (path, transcript) pairs into log-mel features and label ids. Padding positions of the labels are -100
    so the loss ignores them. The leading start token is dropped: the model puts it back when it shifts the labels."""

    def __init__(self, processor, audio_root, start_id: int, max_label: int = 448):
        self.processor = processor
        self.audio_root = audio_root
        self.start_id = start_id
        self.max_label = max_label

    def __call__(self, batch):
        import torch
        audio = [read_wav(str(self.audio_root / p)) for p, _t in batch]
        feats = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        rows = []
        for _p, text in batch:
            ids = list(self.processor.tokenizer(text).input_ids)
            if ids and ids[0] == self.start_id:
                ids = ids[1:]
            rows.append(ids[: self.max_label])
        labels = torch.full((len(rows), max(len(r) for r in rows)), -100, dtype=torch.long)
        for i, r in enumerate(rows):
            labels[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        return {"input_features": feats, "labels": labels}


def evaluate(model, processor, pairs, audio_root, batch: int, autocast) -> float:
    """Word error rate of greedy decoding on the given clips."""
    model.eval()
    hyps = []
    for i in range(0, len(pairs), batch):
        audio = [read_wav(str(audio_root / p)) for p, _t in pairs[i: i + batch]]
        with autocast():
            hyps += hf_transcribe(model, processor, audio, beam_size=1)
    model.train()
    return wer([(t, h) for (_p, t), h in zip(pairs, hyps)])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="lifemate.finetune_asr")
    ap.add_argument("--base", required=True, help="Hugging Face model id or folder, e.g. openai/whisper-large-v3-turbo")
    ap.add_argument("--out", required=True, help="folder for the best checkpoint")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (effective batch = batch * accum)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.1, help="share of the optimiser steps used for linear warm-up")
    ap.add_argument("--workers", type=int, default=4, help="data loading processes")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--dev-limit", type=int, default=0, help="use only the first N dev clips (0 = all)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimiser steps (0 = run all epochs)")
    ap.add_argument("--spec-augment", action="store_true", help="mask random time spans of the input while training")
    ap.add_argument("--no-amp", action="store_true", help="train in full precision")
    ap.add_argument("--grad-checkpoint", action="store_true", help="trade speed for memory if the GPU runs out of memory")
    ap.add_argument("--all-data", action="store_true", help="train on the dev clips too (for the final model, after the settings are chosen)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU found: fine-tuning a Whisper model needs one")
    device = "cuda"
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    audio_root = root() / "train"
    fit, dev = dev_split(load_samples("train"))
    dev_pairs = clip_pairs(dev)
    fit_pairs = clip_pairs(fit) + (dev_pairs if a.all_data else [])
    if a.dev_limit:
        dev_pairs = dev_pairs[: a.dev_limit]
    print(f"[asr-ft] {len(fit_pairs)} training clips, {len(dev_pairs)} dev clips, base {a.base}", flush=True)

    processor = WhisperProcessor.from_pretrained(a.base)
    processor.tokenizer.set_prefix_tokens(language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.base).float().to(device)  # full precision weights: mixed precision is applied per step
    model.generation_config.forced_decoder_ids = None
    if a.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
    if a.grad_checkpoint:
        model.gradient_checkpointing_enable()

    use_amp = not a.no_amp

    def autocast():
        return torch.autocast("cuda", dtype=torch.float16, enabled=use_amp)

    loader = DataLoader(fit_pairs, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                        collate_fn=Collate(processor, audio_root, model.config.decoder_start_token_id))
    steps_per_epoch = max(1, len(loader) // a.accum)
    total = a.max_steps or steps_per_epoch * a.epochs
    warm = max(1, int(total * a.warmup))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(a.out, exist_ok=True)
    history = []
    t0 = time.time()
    base_wer = evaluate(model, processor, dev_pairs, audio_root, a.eval_batch, autocast)
    history.append({"epoch": 0, "dev_wer": base_wer})
    print(f"[asr-ft] before training: dev word error rate {base_wer:.4f} ({time.time() - t0:.0f}s)", flush=True)

    def save(tag: str) -> None:
        model.save_pretrained(a.out)
        processor.save_pretrained(a.out)
        with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"base": a.base, "args": vars(a), "saved": tag, "history": history}, f, ensure_ascii=False, indent=1)

    best = math.inf
    step = 0
    model.train()
    opt.zero_grad(set_to_none=True)
    done = False
    for epoch in range(1, a.epochs + 1):
        running, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)  # drop the gradients of an unfinished accumulation group from the previous epoch
        for i, batch in enumerate(loader, 1):
            with autocast():
                loss = model(input_features=batch["input_features"].to(device), labels=batch["labels"].to(device)).loss
            scaler.scale(loss / a.accum).backward()
            running += float(loss.detach())
            seen += 1
            if i % a.accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                print(f"[asr-ft] epoch {epoch} step {step}/{total} loss {running / seen:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                running, seen = 0.0, 0
            if step >= total:
                done = True
                break
        w = evaluate(model, processor, dev_pairs, audio_root, a.eval_batch, autocast)
        history.append({"epoch": epoch, "step": step, "dev_wer": w})
        kept = w < best or a.all_data  # with --all-data the dev clips were trained on, so the last epoch is kept
        print(f"[asr-ft] epoch {epoch}: dev word error rate {w:.4f}{' (saved)' if kept else ''} {time.time() - t0:.0f}s", flush=True)
        if kept:
            best = min(best, w)
            save(f"epoch {epoch}")
        if done:
            break
    print(f"[asr-ft] done. best dev word error rate {best:.4f} (before training {base_wer:.4f}); model in {a.out}")
    print(f"[asr-ft] use it:  python -m lifemate.cli asr --split train --every 10 --model {a.out}")


if __name__ == "__main__":
    main()
