#!/usr/bin/env python3
"""Pianist identification by teacher-forced likelihood under every artist label.

The conditioned generator defines p(x | artist). Scoring a real test sequence
under each of the 12 artist contexts (plus a 13th "no context" condition, the
paper's ablation with the context zeroed) turns it into a generative
classifier: argmin_A NLL(x | A). Unlike the perplexity table, which compares
different models under the true label, this compares labels within one model.

Per sequence this writes the per-token NLL under every condition, so the
analysis script can derive sequence- and track-level accuracy, accuracy as a
function of prefix length, the label-vs-no-label gap, and token-resolution
"where the style lives" curves without re-running the model.

Memory: the model runs with fp16 weights (--dtype fp32 to disable), one
sequence at a time, and the cross-entropy is taken over 512-position slices of
the logits, so it fits in a few GB next to another job.

Example:
    python scripts/likelihood_classify.py \
        --checkpoint checkpoints/generator \
        --test-jsonl data/pijama12_4096/test.jsonl \
        --artist-map data/pijama12_4096/artist_to_id.json \
        --out-dir results/likelihood_classify
    python scripts/analysis/likelihood_classify_report.py --run-dir results/likelihood_classify
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from aria.config import load_model_config
from aria.model import ModelConfig
from ariautils.tokenizer import AbsTokenizer

from llama_pijama.models import ArtistEmbedding, CrossAttentionTransformerLM
from llama_pijama.training.cross_attention_dataset import CrossAttentionDataset

logger = logging.getLogger("likelihood_classify")

NO_CONTEXT = "<no context>"  # label for the zeroed-context condition


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="generator directory holding model.safetensors and "
                         "artist_embeddings.safetensors")
    ap.add_argument("--test-jsonl", type=Path, required=True)
    ap.add_argument("--artist-map", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/likelihood_classify"))
    ap.add_argument("--model-name", default="medium")
    ap.add_argument("--ca-layers", type=int, nargs="+", default=list(range(8, 16)))
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                    help="weight dtype on the device; fp16 halves the footprint")
    ap.add_argument("--ce-slice", type=int, default=512,
                    help="positions per cross-entropy slice (bounds peak memory)")
    ap.add_argument("--no-context-condition", dest="no_context", action="store_true", default=True,
                    help="also score with the artist context zeroed (ablation)")
    ap.add_argument("--skip-no-context", dest="no_context", action="store_false")
    ap.add_argument("--max-samples", type=int, default=None, help="smoke tests")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def load_state(path: Path):
    f = path / "model.safetensors" if path.is_dir() else path
    return {k.replace("_orig_mod.", ""): v for k, v in load_file(str(f)).items()}


def load_model(args, num_artists: int, device: str):
    cfg = load_model_config(args.model_name)
    cfg.setdefault("resid_dropout", 0.0)
    cfg["grad_checkpoint"] = False
    model_config = ModelConfig(**cfg)
    model = CrossAttentionTransformerLM(
        model_config, {"layers": list(args.ca_layers), "dropout": 0.0, "gate_init": 0.1})
    model.load_state_dict(load_state(args.checkpoint), strict=True)
    emb = ArtistEmbedding(num_artists=num_artists, d_model=model_config.d_model,
                          context_length=args.context_length, dropout=0.0)
    emb.load_state_dict(load_file(str(args.checkpoint / "artist_embeddings.safetensors")),
                        strict=True)
    if args.dtype == "fp16":
        model.half()
        emb.half()
    return model.to(device).eval(), emb.to(device).eval()


def per_token_nll(logits: torch.Tensor, labels: torch.Tensor, slice_len: int) -> torch.Tensor:
    """Cross-entropy at every position, in fp32, a slice at a time. -100 labels give 0."""
    out = torch.zeros(labels.numel(), dtype=torch.float32, device=labels.device)
    for s in range(0, labels.numel(), slice_len):
        e = min(s + slice_len, labels.numel())
        out[s:e] = F.cross_entropy(logits[s:e].float(), labels[s:e],
                                   ignore_index=-100, reduction="none")
    return out


@torch.no_grad()
def score_sequence(model, emb, input_ids, labels, conditions, device, use_amp, slice_len):
    """NLL per token under each condition. Returns (len(conditions), T) float32 on CPU."""
    rows = []
    for cond in conditions:
        artist_id = torch.tensor([0 if cond is None else cond], device=device)
        context, context_mask = emb(artist_id)
        if cond is None:
            context = torch.zeros_like(context)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids, context=context, context_mask=context_mask)
        rows.append(per_token_nll(logits[0], labels[0], slice_len).cpu())
        del logits
    return torch.stack(rows).numpy()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.startswith("cuda")

    artist_to_id = json.loads(args.artist_map.read_text())
    id_to_artist = {v: k for k, v in artist_to_id.items()}
    num_artists = len(artist_to_id)
    conditions = list(range(num_artists)) + ([None] if args.no_context else [])
    condition_names = [id_to_artist[c] if c is not None else NO_CONTEXT for c in conditions]

    tokenizer = AbsTokenizer()
    dataset = CrossAttentionDataset(jsonl_path=str(args.test_jsonl), artist_to_id=artist_to_id,
                                    tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    n = len(dataset) if not args.max_samples else min(args.max_samples, len(dataset))
    logger.info(f"{n} sequences x {len(conditions)} conditions on {device} ({args.dtype})")

    model, emb = load_model(args, num_artists, device)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = args.out_dir / "per_seq"
    seq_dir.mkdir(exist_ok=True)
    summary_path = args.out_dir / "summary.jsonl"

    t_start = time.time()
    with summary_path.open("w") as summary:
        for idx in range(n):
            item = dataset[idx]
            meta = dataset.entries[idx]["metadata"]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            labels = item["labels"].unsqueeze(0).to(device)
            nll = score_sequence(model, emb, input_ids, labels, conditions, device,
                                 use_amp, args.ce_slice)
            valid = (item["labels"] != -100).numpy()
            totals = nll[:, valid].sum(axis=1)
            true_id = int(item["artist_id"])

            np.savez_compressed(seq_dir / f"{idx:04d}.npz", nll=nll.astype(np.float32),
                                valid=valid, true_id=true_id)
            summary.write(json.dumps({
                "index": idx, "artist": meta["artist"], "true_id": true_id,
                "track_id": meta.get("track_id", meta.get("midi_filepath")),
                "title": meta.get("title", ""), "chunk_idx": meta.get("chunk_idx", 0),
                "n_tokens": len(dataset.entries[idx]["seq"]), "n_scored": int(valid.sum()),
                "total_nll": totals.tolist(), "pred_id": int(np.argmin(totals[:num_artists])),
            }) + "\n")
            summary.flush()

            pred = id_to_artist[int(np.argmin(totals[:num_artists]))]
            others = np.delete(totals[:num_artists], true_id)
            margin = others.min() - totals[true_id]  # positive = true label wins
            gap = totals[num_artists] - totals[true_id] if args.no_context else float("nan")
            logger.info(f"[{idx + 1}/{n}] {meta['artist'][:16]:16s} -> {pred[:16]:16s} "
                        f"nll(true)={totals[true_id]:.0f} margin={margin:+.1f} "
                        f"nocontext-true={gap:+.1f}  {(time.time() - t_start) / (idx + 1):.1f}s/seq")

    config = {"checkpoint": str(args.checkpoint), "test_jsonl": str(args.test_jsonl),
              "artist_map": str(args.artist_map), "conditions": condition_names,
              "num_artists": num_artists, "max_seq_len": args.max_seq_len,
              "dtype": args.dtype, "num_sequences": n}
    if use_amp:
        config["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"peak GPU memory {config['peak_gpu_memory_gb']:.2f} GB")
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=1))
    logger.info(f"wrote {summary_path} and {n} per-sequence files in {seq_dir}")


if __name__ == "__main__":
    main()
