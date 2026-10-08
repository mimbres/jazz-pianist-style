#!/usr/bin/env python3
"""Pack a side-by-side "where the style lives" comparison: classifier vs generator likelihood.

For the companion site's twelve characteristic-region tracks (one per pianist,
docs/data/characteristic.json), this joins the Section 7 classifier curve
(characteristic_regions.py) with the generator-likelihood view derived from
scripts/likelihood_classify.py's per-token NLL arrays:

  * window curve: on the same 1024-token / stride-128 windows, the likelihood
    margin log p(w | true) - max_B log p(w | B), within-track z-scored and
    smoothed exactly like the classifier's logit margin;
  * token curve: log p(x_t | true) - log p(x_t | runner-up), the per-token
    evidence, lightly smoothed, which the classifier cannot provide;
  * every note of the performance with both scores, for a coloured piano roll;
  * fifteen-second excerpts at each method's peak and trough, as playable notes.

It also measures the generator's chunk-boundary artifact: the mean per-token
margin as a function of position within each 4096-token chunk.

Example:
    python scripts/site/build_likelihood_comparison.py \
        --run-dir results/likelihood_classify \
        --out results/likelihood_classify/comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

from ariautils.tokenizer import AbsTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from characteristic_regions import notes_with_positions, sliding_windows  # noqa: E402

WINDOW, STRIDE, SMOOTH_SIGMA = 1024, 128, 1.5  # Section 7 settings
NO_CONTEXT = "<no context>"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=Path("results/likelihood_classify"))
    ap.add_argument("--test-jsonl", type=Path, default=Path("data/pijama12_4096/test.jsonl"))
    ap.add_argument("--regions-dir", type=Path, default=Path("results/characteristic_regions"))
    ap.add_argument("--site-tracks", type=Path, default=Path("docs/data/characteristic.json"),
                    help="the site's track selection (artist + title); all tracks if missing")
    ap.add_argument("--out", type=Path, default=Path("results/likelihood_classify/comparison.json"))
    ap.add_argument("--excerpt-seconds", type=float, default=15.0)
    ap.add_argument("--token-sigma", type=float, default=32.0, help="tokens; ~10 notes")
    ap.add_argument("--max-curve-points", type=int, default=1500)
    return ap.parse_args()


def load_run(run_dir: Path):
    config = json.loads((run_dir / "config.json").read_text())
    rows = {}
    for line in (run_dir / "summary.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            r["nll"] = np.load(run_dir / "per_seq" / f"{r['index']:04d}.npz")["nll"]
            rows[r["index"]] = r
    return config, rows


def load_sequences(test_jsonl: Path):
    """index -> record, and track_id -> ordered indices."""
    recs, by_track = {}, defaultdict(list)
    with test_jsonl.open() as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            recs[i] = rec
            by_track[rec["metadata"]["track_id"]].append(i)
    for tid in by_track:
        by_track[tid].sort(key=lambda i: recs[i]["metadata"].get("chunk_idx", 0))
    return recs, by_track


def track_stream(indices, recs, rows):
    """Tokens of one track with the per-token NLL aligned to them (NaN where unscored)."""
    tokens, cols, chunk_starts = [], [], []
    for i in indices:
        seq, r = recs[i]["seq"], rows[i]
        n_scored = len(seq) - 1
        c = np.full((r["nll"].shape[0], len(seq)), np.nan, dtype=np.float32)
        c[:, 1:1 + n_scored] = r["nll"][:, :n_scored]  # nll[:, t] scores token t+1
        chunk_starts.append(len(tokens))
        tokens.extend(seq)
        cols.append(c)
    nll = np.concatenate(cols, axis=1)
    keep = np.array([not (isinstance(t, str) and t in ("<S>", "<E>", "<P>")) for t in tokens])
    kept_index = np.cumsum(keep) - 1
    return {"tokens": [t for t, k in zip(tokens, keep) if k], "nll": nll[:, keep],
            "chunk_starts": [int(kept_index[s]) for s in chunk_starts],
            "true_id": rows[indices[0]]["true_id"]}


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def downsample(times, values, max_points):
    if len(values) <= max_points:
        return times, values
    idx = np.linspace(0, len(values) - 1, max_points).astype(int)
    return times[idx], values[idx]


def pack_excerpt(notes, t0_s, seconds):
    """Flat [start_ms, dur_ms, pitch, velocity], timed from t0, as the site's rolls expect."""
    flat = []
    t1 = t0_s + seconds
    for n in notes:
        start = n["start_ms"] / 1000
        if t0_s <= start < t1:
            end = min(start + n["dur_ms"] / 1000, t1)
            flat += [int(round((start - t0_s) * 1000)), int(round((end - start) * 1000)),
                     n["pitch"], n["velocity"]]
    return flat


def boundary_diagnostic(rows, num_artists, bin_size=128):
    """Mean per-token margin (true minus best other label) by position within a chunk."""
    sums = defaultdict(float)
    counts = defaultdict(int)
    for r in rows.values():
        nll = r["nll"][:num_artists]
        others = np.delete(nll, r["true_id"], axis=0).min(axis=0)
        margin = others - nll[r["true_id"]]
        n = r["n_scored"]
        for b in range(0, n, bin_size):
            seg = margin[b:min(b + bin_size, n)]
            sums[b] += float(seg.sum())
            counts[b] += len(seg)
    return [{"position": b, "mean_margin_nats": sums[b] / counts[b], "n_tokens": counts[b]}
            for b in sorted(sums)]


def main():
    args = parse_args()
    config, rows = load_run(args.run_dir)
    names = config["conditions"]
    num_artists = config["num_artists"]
    recs, by_track = load_sequences(args.test_jsonl)
    regions = json.loads((args.regions_dir / "summary.json").read_text())
    region_by_track = {t["track_id"]: t for t in regions["tracks"]}
    tokenizer = AbsTokenizer()

    wanted = None
    if args.site_tracks.exists():
        site = json.loads(args.site_tracks.read_text())["tracks"]
        wanted = {(t["artist"], t["title"]) for t in site}

    tracks = []
    for tid, indices in by_track.items():
        meta = recs[indices[0]]["metadata"]
        if wanted is not None and (meta["artist"], meta.get("title", "")) not in wanted:
            continue
        if any(i not in rows for i in indices) or tid not in region_by_track:
            print(f"skipping {meta['artist']} - {meta.get('title')}: incomplete")
            continue
        stream = track_stream(indices, recs, rows)
        nll, true_id = stream["nll"], stream["true_id"]
        n = nll.shape[1]
        notes = notes_with_positions(stream["tokens"], tokenizer)
        tok_idx = np.array([m["token_idx"] for m in notes])
        tok_time = np.array([m["start_ms"] / 1000 for m in notes])
        duration = max((m["start_ms"] + m["dur_ms"]) / 1000 for m in notes)

        # Classifier (Section 7) curve and excerpt positions
        reg = region_by_track[tid]
        z = np.load(args.regions_dir / reg["curve_file"])
        clf_times, clf_curve = z["times"].astype(float), z["smoothed"].astype(float)

        # Likelihood on the same windows
        filled = np.nan_to_num(nll, nan=0.0)
        cum = np.concatenate([np.zeros((nll.shape[0], 1)), np.cumsum(filled, axis=1)], axis=1)
        windows = sliding_windows(n, STRIDE)
        win_nll = np.stack([cum[:, e] - cum[:, s] for s, e in windows])[:, :num_artists]
        others = np.delete(win_nll, true_id, axis=1)
        win_margin = others.min(axis=1) - win_nll[:, true_id]
        lik_curve = gaussian_filter1d(zscore(win_margin), SMOOTH_SIGMA)
        centres = np.array([s + (e - s) // 2 for s, e in windows])
        lik_times = np.interp(centres, tok_idx, tok_time)
        runner_up = int(np.delete(np.arange(num_artists), true_id)[np.argmin(others.sum(axis=0))])
        total = win_nll.sum(axis=0)

        # Token-level evidence against the runner-up
        ratio = np.nan_to_num(nll[runner_up] - nll[true_id], nan=0.0)
        tok_curve = zscore(gaussian_filter1d(ratio, args.token_sigma))
        tok_times = np.interp(np.arange(n), tok_idx, tok_time)

        rho, p = spearmanr(np.interp(lik_times, clf_times, clf_curve), lik_curve)

        half = args.excerpt_seconds / 2
        excerpts = {}
        for method, times, curve in (("clf", clf_times, clf_curve), ("lik", lik_times, lik_curve)):
            for which, pick in (("peak", np.argmax), ("trough", np.argmin)):
                i = int(pick(curve))
                t0 = max(0.0, float(times[i]) - half)
                excerpts[f"{method}_{which}"] = {
                    "time_s": round(float(times[i]), 2), "z": round(float(curve[i]), 2),
                    "start_s": round(t0, 2),
                    "notes": pack_excerpt(notes, t0, args.excerpt_seconds)}

        # Every note with both scores, for the coloured roll
        clf_at_note = np.interp(tok_time, clf_times, clf_curve)
        lik_at_note = tok_curve[tok_idx]
        roll = []
        for m, cz, lz in zip(notes, clf_at_note, lik_at_note):
            roll += [int(m["start_ms"]), int(m["dur_ms"]), int(m["pitch"]), int(m["velocity"]),
                     round(float(cz), 2), round(float(lz), 2)]

        ct, cc = downsample(clf_times, clf_curve, args.max_curve_points)
        lt, lc = downsample(lik_times, lik_curve, args.max_curve_points)
        tt, tc = downsample(tok_times, tok_curve, args.max_curve_points)
        tracks.append({
            "artist": meta["artist"], "title": meta.get("title", ""), "track_id": tid,
            "duration_s": round(float(duration), 2), "n_tokens": int(n),
            "n_windows": len(windows), "chunk_starts_s": [round(float(tok_times[s]), 2)
                                                           for s in stream["chunk_starts"]],
            "runner_up": names[runner_up],
            "classifier_runner_up": None,
            "track_margin_nats": round(float(np.delete(total, true_id).min() - total[true_id]), 1),
            "window_margin_mean": round(float(win_margin.mean()), 2),
            "window_margin_std": round(float(win_margin.std()), 2),
            "classifier_margin_mean": round(float(reg["margin_mean"]), 2),
            "classifier_margin_std": round(float(reg["margin_std"]), 2),
            "spearman": round(float(rho), 2), "spearman_p": round(float(p), 3),
            "clf_times": [round(v, 2) for v in ct], "clf_curve": [round(v, 3) for v in cc],
            "lik_times": [round(v, 2) for v in lt], "lik_curve": [round(v, 3) for v in lc],
            "tok_times": [round(v, 2) for v in tt], "tok_curve": [round(v, 3) for v in tc],
            "excerpts": excerpts, "roll": roll,
        })
        print(f"{meta['artist']:18s} {meta.get('title', '')[:40]:40s} rho={rho:+.2f} "
              f"clf peak {excerpts['clf_peak']['time_s']:.0f}s / lik peak {excerpts['lik_peak']['time_s']:.0f}s  "
              f"runner-up {names[runner_up]}")

    tracks.sort(key=lambda t: t["artist"])
    payload = {"tracks": tracks, "excerpt_ms": int(args.excerpt_seconds * 1000),
               "token_sigma": args.token_sigma,
               "boundary": boundary_diagnostic(rows, num_artists),
               "config": {"window": WINDOW, "stride": STRIDE, "smooth_sigma": SMOOTH_SIGMA,
                          "run": config}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"{len(tracks)} tracks, {args.out.stat().st_size / 1024:.0f} KB -> {args.out}")
    print("chunk-boundary diagnostic (mean margin nats/token by position in chunk):")
    for b in payload["boundary"][:6] + payload["boundary"][-2:]:
        print(f"  {b['position']:5d}: {b['mean_margin_nats']:+.4f}  (n={b['n_tokens']})")


if __name__ == "__main__":
    main()
