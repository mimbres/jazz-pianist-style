#!/usr/bin/env python3
"""Re-perform a real recording under every artist label: pitches fixed, timing and dynamics generated.

The score is a real test performance: its notes, in order, with the chords
(notes sharing an onset) kept together. The generator is decoded under each
artist label with its logits masked so that

  * at a note slot it may only emit that note's pitch (any velocity), or a
    time-shift <T> to move to the next 5 s segment;
  * at an onset slot it may emit any onset later than the previous note's, or,
    for a chord member, exactly the previous note's onset;
  * at a duration slot it may emit any duration.

So pitch and chord structure come from the score, while velocity, onset
(tempo, rubato, swing) and duration (articulation) come from the model under
the label. One performance per label plus a zeroed-context control, and the
original performance itself, are written in the format scripts/label_grid_eval.py
reads, with a one-token prompt (the piano prefix) so the whole performance is
judged.

Example:
    python scripts/score_fixed_generate.py \
        --checkpoint-dir checkpoints/generator \
        --test-jsonl data/pijama12_4096/test.jsonl \
        --artist-map data/pijama12_4096/artist_to_id.json \
        --out results/score_fixed/generations.jsonl
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
from llama_pijama.utils.generation import sample_next_token, tokens_to_ids

logger = logging.getLogger("score_fixed_generate")

NO_CONTEXT, ORIGINAL = -1, -2
NOTE, ONSET, DUR = 0, 1, 2
MAX_CONSECUTIVE_T = 3


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--test-jsonl", type=Path, required=True, help="4096-token chunks; chunk 0 of each track is used")
    ap.add_argument("--artist-map", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/score_fixed/generations.jsonl"))
    ap.add_argument("--scores-per-artist", type=int, default=2)
    ap.add_argument("--max-notes", type=int, default=680, help="notes of the score to re-perform (~2040 tokens)")
    ap.add_argument("--min-notes", type=int, default=300)
    ap.add_argument("--skip-no-context", action="store_true")
    ap.add_argument("--free-tempo", action="store_true",
                    help="let the model place the 5 s time shifts itself instead of taking them from the score "
                         "(default: the score's segment boundaries are kept, i.e. a coarse tempo is given)")
    ap.add_argument("--onset-tolerance-ms", type=int, default=0,
                    help="if > 0, each onset must stay within this many ms of the score's onset (micro-timing "
                         "only); 0 leaves onsets free within the segment")
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--model-name", default="medium")
    ap.add_argument("--ca-layers", type=int, nargs="*", default=list(range(8, 16)))
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--max-scores", type=int, default=None, help="smoke tests")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


class Grammar:
    """Vocab index tables for the masks."""

    def __init__(self, tokenizer, device):
        vocab = tokenizer.vocab
        self.t_id = vocab.index(tokenizer.time_tok)
        self.prefix_id = vocab.index(("prefix", "instrument", "piano"))
        self.pitch_ids = defaultdict(list)       # (instrument, pitch) -> velocity variants
        onsets, durs = [], []
        for i, tok in enumerate(vocab):
            if isinstance(tok, tuple) and len(tok) == 3 and tok[0] != "prefix":
                self.pitch_ids[(tok[0], tok[1])].append(i)
            elif isinstance(tok, tuple) and tok[0] == "onset":
                onsets.append((int(tok[1]), i))
            elif isinstance(tok, tuple) and tok[0] == "dur":
                durs.append(i)
        onsets.sort()
        self.onset_values = [v for v, _ in onsets]
        self.onset_ids_sorted = torch.tensor([i for _, i in onsets], device=device)
        self.onset_value_of = {i: v for v, i in onsets}
        self.dur_ids = torch.tensor(durs, device=device)
        self.device = device
        self.pitch_tensors = {k: torch.tensor(v, device=device) for k, v in self.pitch_ids.items()}
        self.vocab_size = len(vocab)

    def onsets_between(self, floor, ceiling):
        """ids of onset tokens with floor < value <= ceiling (ms within the segment)."""
        import bisect
        lo = bisect.bisect_right(self.onset_values, floor)
        hi = bisect.bisect_right(self.onset_values, ceiling)
        return self.onset_ids_sorted[lo:hi]


def parse_score(tokens, tokenizer, max_notes):
    """Notes of a real token stream, in order: pitch key, chord-with-previous flag,
    the 5 s segment the note falls in, and how many later notes of that segment
    still need their own (strictly later) onset."""
    special = {tokenizer.eos_tok, tokenizer.bos_tok, tokenizer.pad_tok}
    clean = [tuple(t) if isinstance(t, list) else t for t in tokens]
    clean = [t for t in clean if t not in special]
    notes, base, last_abs = [], 0, None
    pending = None
    step = tokenizer.abs_time_step_ms
    for tok in clean:
        if tok == tokenizer.time_tok:
            base += step
        elif isinstance(tok, tuple) and len(tok) == 3 and tok[0] != "prefix":
            pending = (tok[0], tok[1])
        elif isinstance(tok, tuple) and tok[0] == "onset" and pending is not None:
            abs_ms = base + int(tok[1])
            notes.append({"pitch": pending, "chord": last_abs is not None and abs_ms == last_abs,
                          "segment": base // step, "onset_ms": int(tok[1])})
            last_abs = abs_ms
            pending = None
            if len(notes) >= max_notes:
                break
    # distinct onsets still to come in the same segment after each note
    remaining = 0
    for k in range(len(notes) - 1, -1, -1):
        if k + 1 < len(notes) and notes[k + 1]["segment"] == notes[k]["segment"]:
            remaining += 0 if notes[k + 1]["chord"] else 1
        else:
            remaining = 0
        notes[k]["remaining"] = remaining
    return notes


def load_generator(args, num_artists, device, dtype, batch_size, max_seq_len):
    cfg = load_model_config(args.model_name)
    cfg.setdefault("resid_dropout", 0.0)
    cfg["grad_checkpoint"] = False
    model_config = ModelConfig(**cfg)
    model = CrossAttentionInferenceLM(model_config=model_config,
                                      cross_attention_config={"layers": list(args.ca_layers), "dropout": 0.0})
    model.load_state_dict(load_file(args.checkpoint_dir / "model.safetensors"), strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    model.setup_cache(batch_size=batch_size, max_seq_len=max_seq_len, dtype=dtype)
    emb = ArtistEmbedding(num_artists=num_artists, d_model=model_config.d_model,
                          context_length=args.context_length, dropout=0.0)
    emb.load_state_dict(load_file(args.checkpoint_dir / "artist_embeddings.safetensors"), strict=True)
    return model, emb.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def decode_score(model, score, grammar: Grammar, ctx, ctx_mask, args, device):
    """Constrained decoding of one score for B conditions at once. Returns B token-id lists."""
    B = ctx.shape[0]
    n_notes = len(score)
    max_steps = 3 * n_notes + score[-1]["segment"] + 3 * (n_notes // 40) + 32
    if hasattr(model, "reset_cache"):
        model.reset_cache()
    # per-element state
    note_idx = [0] * B
    expect = [NOTE] * B
    prev_onset_value = [None] * B      # within the current segment
    prev_onset_id = [None] * B
    new_segment = [True] * B           # no onset yet in the current segment
    t_run = [0] * B
    segment = [0] * B                  # current 5 s segment of each row
    done = [False] * B
    out = [[] for _ in range(B)]
    grid = grammar.onset_values[1] - grammar.onset_values[0] if len(grammar.onset_values) > 1 else 10
    last_value = grammar.onset_values[-1]

    input_ids = torch.full((B, 1), grammar.prefix_id, dtype=torch.long, device=device)
    input_pos = torch.arange(0, 1, device=device)
    logits = model(input_ids, input_pos, context=ctx, context_mask=ctx_mask)
    neg_inf = float("-inf")
    for step in range(max_steps):
        last = logits[:, -1, :].float()
        mask = torch.full_like(last, neg_inf)
        for b in range(B):
            if done[b]:
                mask[b, grammar.prefix_id] = 0.0   # a harmless filler for finished rows
                continue
            note = score[note_idx[b]]
            if expect[b] == NOTE:
                if args.free_tempo:
                    segment_full = (prev_onset_value[b] is not None
                                    and prev_onset_value[b] >= last_value)
                    can_shift = not note["chord"] and t_run[b] < MAX_CONSECUTIVE_T
                    if segment_full and can_shift:
                        mask[b, grammar.t_id] = 0.0       # no later onset left in this segment
                    else:
                        mask[b, grammar.pitch_tensors[note["pitch"]]] = 0.0
                        if can_shift:
                            mask[b, grammar.t_id] = 0.0
                elif note["segment"] > segment[b]:
                    mask[b, grammar.t_id] = 0.0           # the score moves to the next 5 s segment
                else:
                    mask[b, grammar.pitch_tensors[note["pitch"]]] = 0.0
            elif expect[b] == ONSET:
                if note["chord"] and prev_onset_id[b] is not None and not new_segment[b]:
                    mask[b, prev_onset_id[b]] = 0.0
                else:
                    # leave room for the later notes of this segment, each needing its own onset
                    ceiling = last_value - grid * (note["remaining"] if not args.free_tempo else 0)
                    floor = -1 if (new_segment[b] or prev_onset_value[b] is None) else prev_onset_value[b]
                    allowed = grammar.onsets_between(floor, ceiling)
                    if args.onset_tolerance_ms > 0 and not args.free_tempo:
                        tight = grammar.onsets_between(max(floor, note["onset_ms"] - args.onset_tolerance_ms - 1),
                                                       min(ceiling, note["onset_ms"] + args.onset_tolerance_ms))
                        if tight.numel() > 0:
                            allowed = tight
                    if allowed.numel() == 0:
                        allowed = grammar.onset_ids_sorted[-1:]
                    mask[b, allowed] = 0.0
            else:
                mask[b, grammar.dur_ids] = 0.0
        next_tokens = sample_next_token(last + mask, args.temperature, args.top_k, args.top_p)  # (B,1)
        flat = next_tokens[:, 0].tolist()
        for b, tok in enumerate(flat):
            if done[b]:
                continue
            out[b].append(tok)
            if expect[b] == NOTE:
                if tok == grammar.t_id:
                    t_run[b] += 1
                    segment[b] += 1
                    new_segment[b] = True
                    prev_onset_value[b], prev_onset_id[b] = None, None
                else:
                    t_run[b] = 0
                    expect[b] = ONSET
            elif expect[b] == ONSET:
                prev_onset_value[b] = grammar.onset_value_of[tok]
                prev_onset_id[b] = tok
                new_segment[b] = False
                expect[b] = DUR
            else:
                expect[b] = NOTE
                note_idx[b] += 1
                if note_idx[b] >= n_notes:
                    done[b] = True
        if all(done):
            break
        input_pos = torch.tensor([1 + step], device=device)
        logits = model(next_tokens, input_pos, context=ctx, context_mask=ctx_mask)
    return out, [note_idx[b] for b in range(B)]


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
    grammar = Grammar(tokenizer, device)

    # Scores: chunk 0 of the first --scores-per-artist test tracks of each pianist
    by_artist = defaultdict(list)
    with jsonlines.open(args.test_jsonl) as reader:
        for idx, rec in enumerate(reader):
            meta = rec["metadata"]
            artist = meta.get("artist")
            if artist not in artist_to_id or meta.get("chunk_idx", 0) != 0 or len(by_artist[artist]) >= args.scores_per_artist:
                continue
            notes = parse_score(rec["seq"], tokenizer, args.max_notes)
            if len(notes) < args.min_notes:
                continue
            by_artist[artist].append({"index": idx, "artist": artist, "artist_id": artist_to_id[artist],
                                      "track_id": meta.get("track_id"), "title": meta.get("title", ""),
                                      "notes": notes, "original_ids": tokens_to_ids(rec["seq"], tokenizer)})
    scores = [s for a in sorted(by_artist) for s in by_artist[a]]
    if args.max_scores:
        scores = scores[:args.max_scores]
    conditions = list(range(num_artists)) + ([] if args.skip_no_context else [NO_CONTEXT])
    B = len(conditions)
    longest = max((3 * len(s["notes"]) + s["notes"][-1]["segment"] + 3 * (len(s["notes"]) // 40) + 32)
                  for s in scores)
    max_seq_len = 1 + longest + 64
    logger.info(f"{len(scores)} scores x {B} conditions, up to {args.max_notes} notes each, {args.dtype} on {device}")

    model, emb = load_generator(args, num_artists, device, dtype, B, max_seq_len)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    ids = torch.tensor([max(c, 0) for c in conditions], device=device)
    ctx, ctx_mask = emb(ids)
    zero = torch.tensor([c == NO_CONTEXT for c in conditions], device=device)
    ctx = torch.where(zero[:, None, None], torch.zeros_like(ctx), ctx)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with jsonlines.open(args.out, mode="w") as writer:
        for si, s in enumerate(scores):
            gens, n_done = decode_score(model, s["notes"], grammar, ctx, ctx_mask, args, device)
            base = {"prompt_index": s["index"], "artist": s["artist"], "artist_id": s["artist_id"],
                    "track_id": s["track_id"], "title": s["title"], "prompt_length": 1,
                    "prompt_ids": [grammar.prefix_id], "n_notes": len(s["notes"])}
            # the original performance of the same notes, for reference: cut after the
            # duration token that completes the last note of the score
            orig = s["original_ids"]
            dur_set = set(grammar.dur_ids.tolist())
            cut, count, seen_onset = len(orig), 0, False
            for j, tok in enumerate(orig):
                if tok in grammar.onset_value_of:
                    seen_onset = True
                elif tok in dur_set and seen_onset:
                    seen_onset = False
                    count += 1
                    if count == len(s["notes"]):
                        cut = j + 1
                        break
            orig_body = [t for t in orig[:cut] if t != grammar.prefix_id]
            writer.write({**base, "cond_id": ORIGINAL, "cond_artist": "<original>",
                          "continuation_ids": orig_body, "n_generated": len(orig_body), "notes_done": len(s["notes"])})
            for c, g, nd in zip(conditions, gens, n_done):
                writer.write({**base, "cond_id": c, "cond_artist": id_to_artist[c] if c >= 0 else "<no context>",
                              "continuation_ids": g, "n_generated": len(g), "notes_done": nd})
            el = time.time() - t0
            logger.info(f"score {si + 1}/{len(scores)} ({s['artist']}, {len(s['notes'])} notes): "
                        f"{el / (si + 1):.0f}s per score, ~{el / (si + 1) * (len(scores) - si - 1) / 60:.0f} min left; "
                        f"notes completed min {min(n_done)}/{len(s['notes'])}")
    if device.startswith("cuda"):
        logger.info(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
