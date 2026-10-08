#!/usr/bin/env python3
"""Where does the artist label's evidence live: in the notes, the dynamics, or the timing?

Re-scores the real test sequences under every artist label (as
likelihood_classify.py does) but splits each token's log-probability by what
the token encodes. A note token (pitch, velocity) is split into the pitch
marginal, log Σ_v p(pitch, v), and the velocity given the pitch; onset and
duration tokens are kept as they are. Summing per label and per kind gives,
for each sequence, a 12-way NLL table for each kind alone, so we can ask:

  * how well does each kind identify the pianist on its own (argmin over labels)?
  * how many nats of the true-vs-runner-up margin does each kind contribute?

If most of the margin sits in the pitch marginal, fixing the score and
regenerating only timing and dynamics would leave the label little to express.

Example:
    python scripts/likelihood_by_token_type.py \
        --checkpoint checkpoints/generator \
        --test-jsonl data/pijama12_4096/test.jsonl \
        --artist-map data/pijama12_4096/artist_to_id.json \
        --out-dir results/likelihood_by_type
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ariautils.tokenizer import AbsTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from likelihood_classify import load_model  # noqa: E402
from llama_pijama.training.cross_attention_dataset import CrossAttentionDataset  # noqa: E402

logger = logging.getLogger("likelihood_by_token_type")

KINDS = ["pitch", "velocity|pitch", "onset", "dur", "other"]
NOTE, ONSET, DUR, OTHER = 0, 1, 2, 3


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--test-jsonl", type=Path, required=True)
    ap.add_argument("--artist-map", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/likelihood_by_type"))
    ap.add_argument("--model-name", default="medium")
    ap.add_argument("--ca-layers", type=int, nargs="+", default=list(range(8, 16)))
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--ce-slice", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def vocab_tables(tokenizer, device):
    """Per vocab id: token kind, and for note tokens the row of ids sharing its pitch."""
    vocab = tokenizer.vocab
    kind = torch.full((len(vocab),), OTHER, dtype=torch.long)
    groups = defaultdict(list)
    for i, tok in enumerate(vocab):
        if isinstance(tok, tuple) and len(tok) == 3 and tok[0] != "prefix":
            kind[i] = NOTE
            groups[(tok[0], tok[1])].append(i)
        elif isinstance(tok, tuple) and tok[0] == "onset":
            kind[i] = ONSET
        elif isinstance(tok, tuple) and tok[0] == "dur":
            kind[i] = DUR
    width = max(len(g) for g in groups.values())
    group_ids = torch.full((len(groups), width), -1, dtype=torch.long)
    row_of = torch.full((len(vocab),), -1, dtype=torch.long)
    for r, ids in enumerate(groups.values()):
        group_ids[r, :len(ids)] = torch.tensor(ids)
        for i in ids:
            row_of[i] = r
    logger.info(f"{len(groups)} (instrument, pitch) groups, up to {width} velocity variants each")
    return kind.to(device), row_of.to(device), group_ids.to(device)


@torch.no_grad()
def score(model, emb, input_ids, labels, num_artists, device, use_amp, slice_len, kind, row_of, group_ids):
    """Per label: NLL summed by kind -> array (num_artists, 5); plus token counts by kind."""
    T = labels.numel()
    valid = labels != -100
    lab = labels.clamp(min=0)
    tok_kind = torch.where(valid, kind[lab], torch.full_like(lab, OTHER))
    note_pos = valid & (tok_kind == NOTE)
    counts = {"pitch": int(note_pos.sum()), "velocity|pitch": int(note_pos.sum()),
              "onset": int((valid & (tok_kind == ONSET)).sum()), "dur": int((valid & (tok_kind == DUR)).sum()),
              "other": int((valid & (tok_kind == OTHER)).sum())}
    out = np.zeros((num_artists, len(KINDS)), dtype=np.float64)
    for a in range(num_artists):
        context, context_mask = emb(torch.tensor([a], device=device))
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids, context=context, context_mask=context_mask)[0]
        full = torch.zeros(T, device=device)
        pitch = torch.zeros(T, device=device)
        for s in range(0, T, slice_len):
            e = min(s + slice_len, T)
            lp = F.log_softmax(logits[s:e].float(), dim=-1)
            full[s:e] = -lp.gather(1, lab[s:e, None])[:, 0]
            npos = note_pos[s:e]
            if npos.any():
                g = group_ids[row_of[lab[s:e][npos]]]            # (n, width) ids, -1 padded
                gl = lp[npos.nonzero()[:, 0][:, None], g.clamp(min=0)]
                gl = torch.where(g >= 0, gl, torch.full_like(gl, float("-inf")))
                pitch[s:e][npos] = -torch.logsumexp(gl, dim=-1)
        del logits
        full = torch.where(valid, full, torch.zeros_like(full))
        vel = torch.where(note_pos, full - pitch, torch.zeros_like(full))
        out[a, 0] = float(pitch[note_pos].sum())
        out[a, 1] = float(vel[note_pos].sum())
        out[a, 2] = float(full[valid & (tok_kind == ONSET)].sum())
        out[a, 3] = float(full[valid & (tok_kind == DUR)].sum())
        out[a, 4] = float(full[valid & (tok_kind == OTHER)].sum())
    return out, counts


def fmt_pct(x):
    return f"{100 * x:.1f}%"


def report(rows, names, out_dir: Path):
    num_artists = len(names)
    combos = {"pitch": [0], "velocity | pitch": [1], "onset": [2], "dur": [3],
              "timing (onset + dur)": [2, 3], "velocity + timing (score fixed)": [1, 2, 3],
              "pitch + velocity (note tokens)": [0, 1], "all tokens": [0, 1, 2, 3, 4]}
    seq_true = np.array([r["true_id"] for r in rows])
    tables = np.stack([np.array(r["nll"]) for r in rows])          # (N, A, 5)
    by_track = defaultdict(list)
    for i, r in enumerate(rows):
        by_track[r["track_id"]].append(i)
    counts = {k: sum(r["counts"][k] for r in rows) for k in KINDS}

    # Runner-up by the full NLL, fixed per sequence, so every kind's margin is against the same rival
    full = tables.sum(axis=2)
    rival = np.array([int(np.delete(np.arange(num_artists), t)[np.argmin(np.delete(f, t))]) for f, t in zip(full, seq_true)])

    lines = ["# Likelihood evidence by token kind (real test recordings)", "",
             f"{len(rows)} sequences of up to 4096 tokens, {len(by_track)} tracks, {num_artists} labels (chance {fmt_pct(1 / num_artists)}). "
             "Each kind's NLL is summed over the sequence under every label; accuracy = argmin over labels using that kind alone.", "",
             "Token counts: " + ", ".join(f"{k} {counts[k]:,}" for k in KINDS) + ".", "",
             "| evidence used | seq top-1 | track top-1 | margin vs rival, nats/seq | nats/token of that kind | share of total margin |",
             "|---|---|---|---|---|---|"]
    out = {"counts": counts, "combos": {}}
    total_margin = None
    for label, cols in combos.items():
        nll = tables[:, :, cols].sum(axis=2)                      # (N, A)
        seq_acc = float((nll.argmin(axis=1) == seq_true).mean())
        t_ok = []
        for idxs in by_track.values():
            t_ok.append(int(nll[idxs].sum(axis=0).argmin()) == seq_true[idxs[0]])
        track_acc = float(np.mean(t_ok))
        margin = nll[np.arange(len(rows)), rival] - nll[np.arange(len(rows)), seq_true]
        n_tok = sum(counts[KINDS[c]] for c in cols)
        per_tok = margin.sum() / max(1, n_tok)
        if label == "all tokens":
            total_margin = margin.mean()
        out["combos"][label] = {"seq_top1": seq_acc, "track_top1": track_acc,
                                "margin_mean_nats": float(margin.mean()), "margin_median_nats": float(np.median(margin)),
                                "margin_per_token": float(per_tok), "n_tokens": int(n_tok)}
    for label, v in out["combos"].items():
        share = v["margin_mean_nats"] / total_margin if total_margin else float("nan")
        lines.append(f"| {label} | {fmt_pct(v['seq_top1'])} | {fmt_pct(v['track_top1'])} | "
                     f"{v['margin_mean_nats']:.1f} (median {v['margin_median_nats']:.1f}) | {v['margin_per_token']:.4f} | {fmt_pct(share)} |")
    lines += ["", "Margins are against the sequence's runner-up label under the full NLL, so the shares add up to 100% "
              "(the small 'other' kind, <T> and prefix tokens, is folded into 'all tokens').",
              "'velocity + timing (score fixed)' is the evidence that would remain if the pitches were held to the score.", ""]
    # per-artist, score-fixed vs pitch
    lines += ["## Per pianist: sequence accuracy from pitch alone vs velocity + timing alone", "",
              "| pianist | n | pitch | velocity + timing | all |", "|---|---|---|---|---|"]
    for a in range(num_artists):
        m = seq_true == a
        if not m.any():
            continue
        acc = lambda cols: fmt_pct((tables[m][:, :, cols].sum(axis=2).argmin(axis=1) == a).mean())
        lines.append(f"| {names[a]} | {int(m.sum())} | {acc([0])} | {acc([1, 2, 3])} | {acc([0, 1, 2, 3, 4])} |")
    text = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(text)
    (out_dir / "scores.json").write_text(json.dumps({"kinds": KINDS, "names": names, "summary": out,
                                                     "rows": rows}, indent=1))
    return text


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.startswith("cuda")
    artist_to_id = json.loads(args.artist_map.read_text())
    names = [a for a, _ in sorted(artist_to_id.items(), key=lambda kv: kv[1])]
    num_artists = len(names)
    tokenizer = AbsTokenizer()
    kind, row_of, group_ids = vocab_tables(tokenizer, device)
    dataset = CrossAttentionDataset(jsonl_path=str(args.test_jsonl), artist_to_id=artist_to_id,
                                    tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    n = len(dataset) if not args.max_samples else min(args.max_samples, len(dataset))
    model, emb = load_model(args, num_artists, device)
    logger.info(f"{n} sequences x {num_artists} labels on {device}")

    rows, t0 = [], time.time()
    for idx in range(n):
        item = dataset[idx]
        meta = dataset.entries[idx]["metadata"]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        labels = item["labels"].unsqueeze(0).to(device)
        table, counts = score(model, emb, input_ids, labels[0], num_artists, device, use_amp, args.ce_slice,
                              kind, row_of, group_ids)
        rows.append({"index": idx, "artist": meta["artist"], "true_id": int(item["artist_id"]),
                     "track_id": meta.get("track_id"), "nll": table.tolist(), "counts": counts})
        if (idx + 1) % 20 == 0 or idx + 1 == n:
            logger.info(f"  {idx + 1}/{n}  {(time.time() - t0) / (idx + 1):.1f}s/seq")
    print(report(rows, names, args.out_dir))


if __name__ == "__main__":
    main()
