#!/usr/bin/env python3
"""Report for likelihood_classify.py: can the generator's teacher-forced NLL identify the pianist?

Reads the per-sequence NLL arrays written by scripts/likelihood_classify.py and
derives, with no model in the loop:

  1. sequence- and track-level accuracy of argmin_A NLL(x | A), confusion
     matrix, and the margin NLL(runner-up) - NLL(true), set beside the
     discriminative classifier's numbers (evaluate_classifier.py output);
  2. accuracy as a function of prefix length (does evidence accumulate?);
  3. the label-vs-no-context gap (is the artist context used at all?), plus
     the token-weighted perplexity under the true label as a check against
     Table 2 of the paper;
  4. for one track, a "where the style lives" comparison with Section 7: the
     likelihood margin on the same 1024-token / stride-128 windows, z-scored
     and smoothed like characteristic_regions.py, overlaid on that script's
     classifier curve, plus a token-resolution curve.

Example:
    python scripts/analysis/likelihood_classify_report.py \
        --run-dir results/likelihood_classify \
        --real-clf-eval results/pijama12_real_clf_eval.json \
        --regions-dir results/characteristic_regions \
        --test-jsonl data/pijama12_4096/test.jsonl \
        --overlay-track "Sophisticated Lady"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from characteristic_regions import notes_with_positions, sliding_windows, slugify  # noqa: E402

WINDOW, STRIDE, SMOOTH_SIGMA = 1024, 128, 1.5  # Section 7 settings


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=Path("results/likelihood_classify"))
    ap.add_argument("--real-clf-eval", type=Path, default=Path("results/pijama12_real_clf_eval.json"),
                    help="evaluate_classifier.py output for the discriminative classifier")
    ap.add_argument("--regions-dir", type=Path, default=Path("results/characteristic_regions"),
                    help="characteristic_regions.py output (Section 7 curves)")
    ap.add_argument("--test-jsonl", type=Path, default=Path("data/pijama12_4096/test.jsonl"),
                    help="the sequences that were scored; needed for the time axis")
    ap.add_argument("--overlay-track", default="Sophisticated Lady",
                    help="title substring of the track for the Section 7 overlay")
    ap.add_argument("--prefix-lengths", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--token-sigma", type=float, default=32.0,
                    help="smoothing (tokens) for the token-resolution curve; ~10 notes")
    return ap.parse_args()


def load_run(run_dir: Path):
    config = json.loads((run_dir / "config.json").read_text())
    rows = [json.loads(l) for l in (run_dir / "summary.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        d = np.load(run_dir / "per_seq" / f"{r['index']:04d}.npz")
        r["nll"], r["valid"] = d["nll"], d["valid"]
    return config, rows


def accuracy_table(rows, num_artists, names):
    seq_pred = np.array([int(np.argmin(r["total_nll"][:num_artists])) for r in rows])
    seq_true = np.array([r["true_id"] for r in rows])
    margins = np.array([np.delete(np.array(r["total_nll"][:num_artists]), r["true_id"]).min()
                        - r["total_nll"][r["true_id"]] for r in rows])

    by_track = defaultdict(list)
    for r, p in zip(rows, seq_pred):
        by_track[r["track_id"]].append((r, p))
    track_true, track_sum, track_vote = [], [], []
    for tid, items in by_track.items():
        total = np.sum([np.array(r["total_nll"][:num_artists]) for r, _ in items], axis=0)
        track_true.append(items[0][0]["true_id"])
        track_sum.append(int(np.argmin(total)))
        votes = Counter(p for _, p in items)
        top = max(votes.values())
        tied = [a for a, c in votes.items() if c == top]
        track_vote.append(tied[0] if len(tied) == 1 else int(np.argmin(total)))  # ties by summed NLL
    track_true, track_sum, track_vote = map(np.array, (track_true, track_sum, track_vote))

    conf = np.zeros((num_artists, num_artists), dtype=int)
    for t, p in zip(seq_true, seq_pred):
        conf[t, p] += 1
    per_artist = {names[a]: {"n_seq": int((seq_true == a).sum()),
                             "seq_acc": float((seq_pred[seq_true == a] == a).mean()),
                             "n_tracks": int((track_true == a).sum()),
                             "track_acc": float((track_sum[track_true == a] == a).mean())}
                  for a in range(num_artists) if (seq_true == a).any()}
    return {"seq_acc": float((seq_pred == seq_true).mean()), "n_seq": len(rows),
            "track_acc_summed_nll": float((track_sum == track_true).mean()),
            "track_acc_majority": float((track_vote == track_true).mean()),
            "n_tracks": len(by_track),
            "margin_median": float(np.median(margins)), "margin_min": float(margins.min()),
            "margin_frac_positive": float((margins > 0).mean()),
            "confusion": conf, "per_artist": per_artist,
            "seq_true": seq_true, "seq_pred": seq_pred, "margins": margins}


def prefix_curve(rows, num_artists, lengths):
    out = {}
    for L in lengths:
        correct = 0
        for r in rows:
            part = r["nll"][:num_artists, :L].sum(axis=1)
            correct += int(np.argmin(part)) == r["true_id"]
        out[L] = correct / len(rows)
    return out


def context_gap(rows, config):
    if "<no context>" not in config["conditions"]:
        return None
    k = config["conditions"].index("<no context>")
    gaps = np.array([r["total_nll"][k] - r["total_nll"][r["true_id"]] for r in rows])
    tokens = np.array([r["n_scored"] for r in rows])
    true_nll = np.array([r["total_nll"][r["true_id"]] for r in rows])
    zero_nll = np.array([r["total_nll"][k] for r in rows])
    return {"frac_label_better": float((gaps > 0).mean()),
            "mean_gap_nats_per_seq": float(gaps.mean()),
            "mean_gap_nats_per_token": float(gaps.sum() / tokens.sum()),
            "ppl_true_label": math.exp(true_nll.sum() / tokens.sum()),
            "ppl_no_context": math.exp(zero_nll.sum() / tokens.sum())}


def load_clf_eval(path: Path, names):
    if not path or not path.exists():
        return None
    d = json.loads(path.read_text())
    conf = Counter()
    for s in d["per_sample"]:
        conf[(s["true_artist"], s["predicted_artist"])] += 1
    top_conf = defaultdict(Counter)
    for (t, p), c in conf.items():
        if t != p:
            top_conf[t][p] += c
    return {"chunk_acc": d["chunk_level"]["top1_accuracy"], "n_chunks": len(d["per_sample"]),
            "track_acc": d["track_level"]["majority_vote"]["accuracy"],
            "n_tracks": d["track_level"]["majority_vote"]["total_tracks"],
            "top_confusions": {t: c.most_common(1)[0] for t, c in top_conf.items()}}


def track_token_stream(rows, test_jsonl: Path, title_substring: str):
    """Reassemble one track's tokens and align its per-token NLL arrays to them."""
    recs = []
    with test_jsonl.open() as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            if title_substring.lower() in rec["metadata"].get("title", "").lower():
                recs.append((i, rec))
    if not recs:
        return None
    tid = recs[0][1]["metadata"]["track_id"]
    recs = sorted([r for r in recs if r[1]["metadata"]["track_id"] == tid],
                  key=lambda r: r[1]["metadata"].get("chunk_idx", 0))
    by_index = {r["index"]: r for r in rows}
    tokens, nll_cols, offset = [], [], 0
    for i, rec in recs:
        if i not in by_index:
            return None
        r = by_index[i]
        seq = rec["seq"]
        # nll[:, t] scores token t+1 of this chunk; map it onto the track stream
        n_scored = len(seq) - 1
        cols = np.full((r["nll"].shape[0], len(seq)), np.nan, dtype=np.float32)
        cols[:, 1:1 + n_scored] = r["nll"][:, :n_scored]
        tokens.extend(seq)
        nll_cols.append(cols)
        offset += len(seq)
    nll = np.concatenate(nll_cols, axis=1)
    special = {"<S>", "<E>", "<P>"}
    keep = np.array([not (isinstance(t, str) and t in special) for t in tokens])
    return {"track_id": tid, "artist": recs[0][1]["metadata"]["artist"],
            "title": recs[0][1]["metadata"]["title"], "tokens": tokens,
            "nll": nll[:, keep], "true_id": by_index[recs[0][0]]["true_id"]}


def overlay(stream, regions_dir: Path, out_path: Path, num_artists: int, names, token_sigma: float):
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import spearmanr
    from ariautils.tokenizer import AbsTokenizer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokenizer = AbsTokenizer()
    nll, true_id = stream["nll"], stream["true_id"]
    n = nll.shape[1]
    notes = notes_with_positions(stream["tokens"], tokenizer)
    tok_idx = np.array([m["token_idx"] for m in notes])
    tok_time = np.array([m["start_ms"] / 1000 for m in notes])

    # (a) window-level: same windows as Section 7, margin = log p(w|true) - max_B log p(w|B)
    windows = sliding_windows(n, STRIDE)
    filled = np.nan_to_num(nll, nan=0.0)
    cum = np.concatenate([np.zeros((nll.shape[0], 1)), np.cumsum(filled, axis=1)], axis=1)
    win_nll = np.stack([cum[:, e] - cum[:, s] for s, e in windows])  # (W, conditions)
    others = np.delete(win_nll[:, :num_artists], true_id, axis=1)
    win_margin = others.min(axis=1) - win_nll[:, true_id]
    z = (win_margin - win_margin.mean()) / (win_margin.std() + 1e-8)
    win_smoothed = gaussian_filter1d(z, SMOOTH_SIGMA)
    centres = np.array([s + (e - s) // 2 for s, e in windows])
    win_times = np.interp(centres, tok_idx, tok_time)
    runner_up = int(np.delete(np.arange(num_artists), true_id)[np.argmin(others.sum(axis=0))])

    # (b) token-level: log-ratio against the track's runner-up label, lightly smoothed
    ratio = nll[runner_up] - nll[true_id]
    ratio = np.where(np.isnan(ratio), 0.0, ratio)
    tok_smoothed = gaussian_filter1d(ratio, token_sigma)
    tok_z = (tok_smoothed - tok_smoothed.mean()) / (tok_smoothed.std() + 1e-8)
    tok_times = np.interp(np.arange(n), tok_idx, tok_time)

    # Section 7 curve
    slug = f"{slugify(stream['artist'])}__{slugify(stream['title'])}"
    ref_path = regions_dir / "curves" / f"{slug}.npz"
    ref = np.load(ref_path) if ref_path.exists() else None
    stats = {"track": stream["title"], "artist": stream["artist"], "runner_up": names[runner_up],
             "n_tokens": int(n), "n_windows": len(windows),
             "likelihood_peak_time_s": float(win_times[int(np.argmax(win_smoothed))]),
             "likelihood_trough_time_s": float(win_times[int(np.argmin(win_smoothed))]),
             "window_margin_mean_nats": float(win_margin.mean()),
             "window_margin_std_nats": float(win_margin.std())}
    if ref is not None:
        ref_on_grid = np.interp(win_times, ref["times"], ref["smoothed"])
        rho, p = spearmanr(ref_on_grid, win_smoothed)
        stats.update({"classifier_peak_time_s": float(ref["times"][int(np.argmax(ref["smoothed"]))]),
                      "classifier_trough_time_s": float(ref["times"][int(np.argmin(ref["smoothed"]))]),
                      "spearman_window_vs_classifier": float(rho), "spearman_p": float(p)})

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax = axes[0]
    if ref is not None:
        ax.plot(ref["times"], ref["smoothed"], color="#555", lw=1.8,
                label="Section 7: classifier logit margin (z, smoothed)")
    ax.plot(win_times, win_smoothed, color="#c0392b", lw=1.8,
            label="generator likelihood margin, same windows (z, smoothed)")
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_ylabel("within-track z")
    ax.legend(loc="upper right", fontsize=8)
    title = f"{stream['artist']} — {stream['title']}"
    if ref is not None:
        title += f"   Spearman ρ = {stats['spearman_window_vs_classifier']:.2f}"
    ax.set_title(title, fontsize=10)
    ax = axes[1]
    ax.plot(tok_times, tok_z, color="#2471a3", lw=1.0,
            label=f"token-level log p(x_t|{stream['artist']}) − log p(x_t|{names[runner_up]}), "
                  f"σ={token_sigma:.0f} tokens (z)")
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("within-track z")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    return stats


def fmt_pct(x):
    return f"{100 * x:.1f}%"


def main():
    args = parse_args()
    config, rows = load_run(args.run_dir)
    names = config["conditions"]
    num_artists = config["num_artists"]
    acc = accuracy_table(rows, num_artists, names)
    prefix = prefix_curve(rows, num_artists, args.prefix_lengths)
    gap = context_gap(rows, config)
    clf = load_clf_eval(args.real_clf_eval, names)

    lines = [f"# Likelihood classification report ({config['num_sequences']} sequences, "
             f"{acc['n_tracks']} tracks, {num_artists} artists; chance {fmt_pct(1 / num_artists)})", ""]
    clf_chunk = f"{fmt_pct(clf['chunk_acc'])} per 1024 chunk (n={clf['n_chunks']})" if clf else "—"
    clf_track = f"{fmt_pct(clf['track_acc'])} majority vote (n={clf['n_tracks']})" if clf else "—"
    lines += ["## 1. Accuracy of argmin_A NLL(x | A)", "",
              "| | generator likelihood | discriminative classifier (4.1) |", "|---|---|---|",
              f"| per 4096-token sequence | {fmt_pct(acc['seq_acc'])} (n={acc['n_seq']}) | — |",
              f"| per 1024-token prefix | {fmt_pct(prefix.get(1024, float('nan')))} | {clf_chunk} |",
              f"| per track, summed NLL | {fmt_pct(acc['track_acc_summed_nll'])} (n={acc['n_tracks']}) | {clf_track} |",
              f"| per track, majority vote | {fmt_pct(acc['track_acc_majority'])} | |", "",
              f"Margin NLL(runner-up) − NLL(true) per sequence: median {acc['margin_median']:.1f} nats, "
              f"min {acc['margin_min']:.1f}, positive in {fmt_pct(acc['margin_frac_positive'])} of sequences.", ""]
    lines += ["### Per artist", "", "| artist | seqs | seq acc | tracks | track acc | top confusion (likelihood) |"
              + (" top confusion (classifier) |" if clf else ""), "|---|---|---|---|---|---|" + ("---|" if clf else "")]
    for a, s in sorted(acc["per_artist"].items()):
        ai = names.index(a)
        row = acc["confusion"][ai].copy()
        row[ai] = 0
        top = f"{names[int(row.argmax())]} ({int(row.max())})" if row.sum() else "—"
        line = f"| {a} | {s['n_seq']} | {fmt_pct(s['seq_acc'])} | {s['n_tracks']} | {fmt_pct(s['track_acc'])} | {top} |"
        if clf:
            c = clf["top_confusions"].get(a)
            line += f" {c[0]} ({c[1]}) |" if c else " — |"
        lines.append(line)
    lines += ["", "## 2. Accuracy vs prefix length (per sequence)", "",
              "| tokens | " + " | ".join(str(L) for L in args.prefix_lengths) + " |",
              "|---|" + "---|" * len(args.prefix_lengths),
              "| accuracy | " + " | ".join(fmt_pct(prefix[L]) for L in args.prefix_lengths) + " |", ""]
    if gap:
        lines += ["## 3. True label vs no context (ablation)", "",
                  f"- true-label NLL lower than no-context NLL in {fmt_pct(gap['frac_label_better'])} of sequences",
                  f"- mean gap {gap['mean_gap_nats_per_seq']:.1f} nats / sequence = "
                  f"{gap['mean_gap_nats_per_token']:.4f} nats / token",
                  f"- token-weighted perplexity: true label {gap['ppl_true_label']:.2f} "
                  f"(paper Table 2: 6.82), no context {gap['ppl_no_context']:.2f}", ""]

    out = {"config": config, "accuracy": {k: v for k, v in acc.items()
                                          if k not in ("confusion", "seq_true", "seq_pred", "margins")},
           "confusion": acc["confusion"].tolist(), "prefix_curve": prefix, "context_gap": gap,
           "classifier_reference": clf}

    stream = track_token_stream(rows, args.test_jsonl, args.overlay_track) if args.test_jsonl.exists() else None
    if stream:
        png = args.run_dir / f"{slugify(stream['artist'])}__{slugify(stream['title'])}__overlay.png"
        stats = overlay(stream, args.regions_dir, png, num_artists, names, args.token_sigma)
        out["overlay"] = stats
        lines += [f"## 4. Where the style lives: {stats['artist']} — {stats['track']}", "",
                  f"- windows: {stats['n_windows']} (1024 tokens, stride 128), runner-up label: {stats['runner_up']}",
                  f"- likelihood margin per window: mean {stats['window_margin_mean_nats']:.1f} nats, "
                  f"std {stats['window_margin_std_nats']:.1f}",
                  f"- likelihood peak {stats['likelihood_peak_time_s']:.1f}s / trough {stats['likelihood_trough_time_s']:.1f}s"]
        if "spearman_window_vs_classifier" in stats:
            lines += [f"- classifier (Section 7) peak {stats['classifier_peak_time_s']:.1f}s / "
                      f"trough {stats['classifier_trough_time_s']:.1f}s",
                      f"- Spearman ρ between the two window curves: "
                      f"{stats['spearman_window_vs_classifier']:.2f} (p={stats['spearman_p']:.2g})"]
        lines += [f"- figure: `{png.name}`", ""]

    (args.run_dir / "report.md").write_text("\n".join(lines))
    (args.run_dir / "report.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
