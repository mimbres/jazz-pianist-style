#!/usr/bin/env python3
"""Generate the same prompt under every artist label (the label grid).

For a few held-out prompts per pianist, continue each prompt once under each of
the 12 artist contexts and once with the context zeroed (the paper's ablation).
The result is a grid of continuations that differ only in the conditioning
label, which is the input for scripts/label_grid_eval.py: can the generator's
own teacher-forced likelihood recover the label a continuation was generated
with, and does the paper's classifier agree?

Sampling follows Section 5 of the paper (temperature 0.95, top-k 50, top-p
0.95, prompt of 256 tokens). Continuations default to 2048 tokens rather than
4096 to halve the cost; pass --max-continuation 4096 for the paper's length.

Memory: the generator runs in bf16 with a bf16 KV cache, --batch-size
continuations at a time (about 0.45 GB per continuation of 2304 tokens on top
of 1.5 GB of weights), so it fits beside another job on the GPU.

Example:
    python scripts/label_grid_generate.py \
        --checkpoint-dir checkpoints/generator \
        --val-jsonl data/pijama12_4096/val.jsonl \
        --artist-map data/pijama12_4096/artist_to_id.json \
        --out results/label_grid/generations.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import jsonlines
import torch
from safetensors.torch import load_file

from aria.config import load_model_config
from aria.model import ModelConfig
from ariautils.tokenizer import AbsTokenizer

from llama_pijama.models import ArtistEmbedding, CrossAttentionInferenceLM
from llama_pijama.utils.generation import generate_tokens_kv_batched, tokens_to_ids

logger = logging.getLogger("label_grid_generate")

NO_CONTEXT = -1  # cond_id for the zeroed-context control


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--artist-map", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/label_grid/generations.jsonl"))
    ap.add_argument("--prompts-per-artist", type=int, default=2)
    ap.add_argument("--prompt-length", type=int, default=256)
    ap.add_argument("--max-continuation", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=7, help="continuations generated at once")
    ap.add_argument("--skip-no-context", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--model-name", default="medium")
    ap.add_argument("--ca-layers", type=int, nargs="*", default=list(range(8, 16)))
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--max-prompts", type=int, default=None, help="smoke tests")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def load_generator(args, num_artists, device, dtype):
    cfg = load_model_config(args.model_name)
    cfg.setdefault("resid_dropout", 0.0)
    cfg["grad_checkpoint"] = False
    model_config = ModelConfig(**cfg)
    model = CrossAttentionInferenceLM(model_config=model_config,
                                      cross_attention_config={"layers": list(args.ca_layers), "dropout": 0.0})
    model.load_state_dict(load_file(args.checkpoint_dir / "model.safetensors"), strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    model.setup_cache(batch_size=args.batch_size,
                      max_seq_len=args.prompt_length + args.max_continuation + 64, dtype=dtype)
    emb = ArtistEmbedding(num_artists=num_artists, d_model=model_config.d_model,
                          context_length=args.context_length, dropout=0.0)
    emb.load_state_dict(load_file(args.checkpoint_dir / "artist_embeddings.safetensors"), strict=True)
    return model, emb.to(device=device, dtype=dtype).eval()


def pick_prompts(val_jsonl: Path, artist_to_id, tokenizer, prompt_length, per_artist):
    """The first `per_artist` held-out sequences of each pianist, in file order."""
    by_artist = defaultdict(list)
    with jsonlines.open(val_jsonl) as reader:
        for idx, rec in enumerate(reader):
            artist = rec.get("metadata", {}).get("artist")
            if artist not in artist_to_id or len(by_artist[artist]) >= per_artist:
                continue
            ids = tokens_to_ids(rec["seq"], tokenizer)
            if len(ids) < prompt_length + 64:
                continue
            meta = rec["metadata"]
            by_artist[artist].append({"prompt_index": idx, "artist": artist,
                                      "artist_id": artist_to_id[artist],
                                      "track_id": meta.get("track_id"), "title": meta.get("title", ""),
                                      "prompt_ids": ids[:prompt_length]})
    return [p for a in sorted(by_artist) for p in by_artist[a]]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    artist_to_id = json.loads(args.artist_map.read_text())
    id_to_artist = {v: k for k, v in artist_to_id.items()}
    num_artists = len(artist_to_id)
    tokenizer = AbsTokenizer()
    eos_id = tokenizer.vocab.index(tokenizer.eos_tok)

    prompts = pick_prompts(args.val_jsonl, artist_to_id, tokenizer, args.prompt_length, args.prompts_per_artist)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    conditions = list(range(num_artists)) + ([] if args.skip_no_context else [NO_CONTEXT])
    logger.info(f"{len(prompts)} prompts x {len(conditions)} conditions, {args.max_continuation} tokens each, "
                f"batch {args.batch_size}, {args.dtype} on {device}")

    model, emb = load_generator(args, num_artists, device, dtype)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():  # resume
        with jsonlines.open(args.out) as reader:
            for r in reader:
                done.add((r["prompt_index"], r["cond_id"]))
        logger.info(f"resuming: {len(done)} continuations already in {args.out}")

    t0 = time.time()
    n_done = 0
    with jsonlines.open(args.out, mode="a") as writer:
        for pi, p in enumerate(prompts):
            todo = [c for c in conditions if (p["prompt_index"], c) not in done]
            for start in range(0, len(todo), args.batch_size):
                conds = todo[start:start + args.batch_size]
                B = len(conds)
                input_ids = torch.tensor([p["prompt_ids"]] * B, dtype=torch.long, device=device)
                ids = torch.tensor([max(c, 0) for c in conds], device=device)
                ctx, ctx_mask = emb(ids)
                zero = torch.tensor([c == NO_CONTEXT for c in conds], device=device)
                ctx = torch.where(zero[:, None, None], torch.zeros_like(ctx), ctx)
                # The cache was sized for --batch-size rows; pad a short final batch
                if B < args.batch_size:
                    pad = args.batch_size - B
                    input_ids = torch.cat([input_ids, input_ids[:1].expand(pad, -1)])
                    ctx = torch.cat([ctx, ctx[:1].expand(pad, -1, -1)])
                    ctx_mask = torch.cat([ctx_mask, ctx_mask[:1].expand(pad, *ctx_mask.shape[1:])])
                gens = generate_tokens_kv_batched(model, input_ids, args.max_continuation, eos_id,
                                                  temperature=args.temperature, top_k=args.top_k,
                                                  top_p=args.top_p, context=ctx, context_mask=ctx_mask)[:B]
                for c, g in zip(conds, gens):
                    writer.write({"prompt_index": p["prompt_index"], "artist": p["artist"],
                                  "artist_id": p["artist_id"], "track_id": p["track_id"], "title": p["title"],
                                  "cond_id": c, "cond_artist": id_to_artist[c] if c >= 0 else "<no context>",
                                  "prompt_length": len(p["prompt_ids"]), "prompt_ids": p["prompt_ids"],
                                  "continuation_ids": g, "n_generated": len(g)})
                n_done += B
                rate = (time.time() - t0) / n_done
                logger.info(f"prompt {pi + 1}/{len(prompts)} ({p['artist']}): {n_done} continuations, "
                            f"{rate:.1f}s each, ~{rate * (len(prompts) * len(conditions) - len(done) - n_done) / 60:.0f} min left")
    if device.startswith("cuda"):
        logger.info(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
