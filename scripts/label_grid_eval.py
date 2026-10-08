#!/usr/bin/env python3
"""Score the label grid: does likelihood recover the conditioning label? Does the classifier?

Input is scripts/label_grid_generate.py's output: for each prompt, one
continuation per artist label (plus a zeroed-context control). Two judges read
every continuation, the prompt excluded:

  likelihood  the generator itself, teacher-forced under all 12 labels with the
              prompt as context; prediction = argmin NLL over the continuation;
  classifier  the paper's pianist classifier on 1024-token windows at stride
              128 (Section 5), majority vote plus per-window agreement.

Each judge is asked whether its prediction is the conditioning label, the
prompt's own pianist, or something else. Matched continuations (label = prompt
pianist) reproduce the paper's conditioned agreement; mismatched ones are its
mismatch experiment, now with a likelihood judge beside the classifier.

The two models are loaded one after the other, so the GPU holds one at a time.

Example:
    python scripts/label_grid_eval.py \
        --generations results/label_grid/generations.jsonl \
        --checkpoint checkpoints/generator --classifier checkpoints/classifier/best.pt \
        --artist-map data/pijama12_4096/artist_to_id.json --out-dir results/label_grid
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jsonlines
import numpy as np
import torch

from ariautils.tokenizer import AbsTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from likelihood_classify import load_model as load_generator, score_sequence  # noqa: E402

logger = logging.getLogger("label_grid_eval")

CLASSIFY_WINDOW, WINDOW_STRIDE = 1024, 128
NO_CONTEXT, ORIGINAL = -1, -2   # cond_id of the zeroed-context control and of the real performance


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="generator directory")
    ap.add_argument("--classifier", type=Path, required=True, help="classifier best.pt")
    ap.add_argument("--artist-map", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/label_grid"))
    ap.add_argument("--model-name", default="medium")
    ap.add_argument("--ca-layers", type=int, nargs="+", default=list(range(8, 16)))
    ap.add_argument("--context-length", type=int, default=4)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--ce-slice", type=int, default=512)
    ap.add_argument("--classifier-batch", type=int, default=8)
    ap.add_argument("--skip-likelihood", action="store_true")
    ap.add_argument("--skip-classifier", action="store_true")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def fmt_pct(x):
    return f"{100 * x:.1f}%"


@torch.no_grad()
def likelihood_pass(records, args, num_artists, device):
    model, emb = load_generator(args, num_artists, device)
    use_amp = device.startswith("cuda")
    conditions = list(range(num_artists)) + [None]
    for i, r in enumerate(records):
        ids = r["prompt_ids"] + r["continuation_ids"]
        if len(r["continuation_ids"]) < 2:
            r["lik"] = None
            continue
        input_ids = torch.tensor([ids[:-1]], device=device)
        labels = torch.tensor([ids[1:]], device=device)
        labels[0, :r["prompt_length"] - 1] = -100   # score the continuation only
        nll = score_sequence(model, emb, input_ids, labels, conditions, device, use_amp, args.ce_slice)
        valid = (labels[0] != -100).cpu().numpy()
        totals = nll[:, valid].sum(axis=1)
        order = np.argsort(totals[:num_artists])
        r["lik"] = {"total_nll": totals.tolist(), "pred": int(order[0]),
                    "margin": float(totals[order[1]] - totals[order[0]]), "n_scored": int(valid.sum())}
        if (i + 1) % 25 == 0:
            logger.info(f"  likelihood {i + 1}/{len(records)}")
    del model, emb
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


@torch.no_grad()
def classifier_pass(records, args, num_artists, device, tokenizer):
    from llama_pijama.evaluation import load_model as load_classifier
    clf, _ = load_classifier(str(args.classifier), model_name=args.model_name,
                             num_classes=num_artists, device=device)
    if args.dtype == "fp16":
        clf.half()
    eos_id = tokenizer.vocab.index(tokenizer.eos_tok)
    pad_id = tokenizer.vocab.index(tokenizer.pad_tok)
    for i, r in enumerate(records):
        cont = r["continuation_ids"]
        n = (len(cont) - CLASSIFY_WINDOW) // WINDOW_STRIDE + 1
        starts = [w * WINDOW_STRIDE for w in range(max(n, 0))]
        if not starts and len(cont) >= 64:   # short continuation: one window over what there is
            starts = [0]
        preds = []
        for b in range(0, len(starts), args.classifier_batch):
            batch, pos = [], []
            for s in starts[b:b + args.classifier_batch]:
                seq = cont[s:s + CLASSIFY_WINDOW - 1] + [eos_id]
                pos.append(len(seq) - 1)
                batch.append(seq + [pad_id] * (CLASSIFY_WINDOW - len(seq)))
            logits = clf(torch.tensor(batch, dtype=torch.long, device=device))
            preds += [int(logits[j, p].argmax()) for j, p in enumerate(pos)]
        votes = Counter(preds)
        r["clf"] = {"window_preds": preds, "positions": starts,
                    "majority": (votes.most_common(1)[0][0] if preds else None)}
        if (i + 1) % 25 == 0:
            logger.info(f"  classifier {i + 1}/{len(records)}")
    del clf
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def outcome(pred, cond_id, prompt_id):
    if pred is None:
        return "none"
    if pred == cond_id:
        return "cond"
    if pred == prompt_id:
        return "prompt"
    return "other"


def summarise(records, names, num_artists):
    groups = {"original": [r for r in records if r["cond_id"] == ORIGINAL],
              "matched": [r for r in records if r["cond_id"] == r["artist_id"]],
              "mismatched": [r for r in records if r["cond_id"] not in (r["artist_id"], NO_CONTEXT, ORIGINAL)],
              "no_context": [r for r in records if r["cond_id"] == NO_CONTEXT]}
    groups = {k: v for k, v in groups.items() if v}
    out = {"n": {k: len(v) for k, v in groups.items()}, "judges": {}}
    for judge in ("lik", "clf"):
        if not any(judge in r for r in records):
            continue
        J = out["judges"][judge] = {}
        for g, rs in groups.items():
            if not rs:
                continue
            key = "pred" if judge == "lik" else "majority"
            outs = Counter(outcome(r[judge][key] if r.get(judge) else None, r["cond_id"], r["artist_id"]) for r in rs)
            tot = max(1, len(rs))
            J[g] = {k: outs.get(k, 0) / tot for k in ("cond", "prompt", "other", "none")}
            J[g]["n"] = len(rs)
            if judge == "clf":
                # per-window agreement with the conditioning / prompt artist, by position
                pos = defaultdict(lambda: Counter())
                for r in rs:
                    if not r.get("clf"):
                        continue
                    for p, w in zip(r["clf"]["positions"], r["clf"]["window_preds"]):
                        pos[p]["n"] += 1
                        pos[p]["cond"] += int(w == r["cond_id"])
                        pos[p]["prompt"] += int(w == r["artist_id"])
                J[g]["window_curve"] = [{"position": p, "cond": c["cond"] / c["n"], "prompt": c["prompt"] / c["n"],
                                         "n": c["n"]} for p, c in sorted(pos.items())]
                allw = [(w == r["cond_id"], w == r["artist_id"]) for r in rs if r.get("clf")
                        for w in r["clf"]["window_preds"]]
                if allw:
                    J[g]["window_mean"] = {"cond": float(np.mean([a for a, _ in allw])),
                                           "prompt": float(np.mean([b for _, b in allw]))}
            else:
                margins = [r["lik"]["margin"] for r in rs if r.get("lik")]
                J[g]["margin_median_nats"] = float(np.median(margins)) if margins else None
    # confusion: conditioning label -> likelihood prediction, over conditioned continuations
    conf = np.zeros((num_artists, num_artists), dtype=int)
    for r in records:
        if r["cond_id"] >= 0 and r.get("lik"):
            conf[r["cond_id"], r["lik"]["pred"]] += 1
    out["lik_confusion_cond_to_pred"] = conf.tolist()
    per_label = {}
    for c in range(num_artists):
        rs = [r for r in records if r["cond_id"] == c and r.get("lik")]
        if rs:
            per_label[names[c]] = {"n": len(rs),
                                   "lik_recovers_label": float(np.mean([r["lik"]["pred"] == c for r in rs])),
                                   "clf_majority_is_label": float(np.mean([r["clf"]["majority"] == c for r in rs if r.get("clf")])) if any(r.get("clf") for r in rs) else None}
    out["per_conditioning_label"] = per_label
    return out


def write_report(summary, config, out_path: Path, names):
    n = summary["n"]
    L = ["# Label grid: same prompt, every artist label", "",
         f"Prompts: {config['n_prompts']} ({config['prompt_length']} tokens); "
         f"continuations: up to {config['max_continuation']} tokens; conditions per prompt: {config['n_conditions']}. "
         + "; ".join(f"{g} n={c}" for g, c in n.items()) + ". "
         "Matched = label is the prompt's pianist; mismatched = any other label; no context = zeroed artist context; "
         "original = the real performance itself (only in the score-fixed experiment).", "",
         "## Whose style does each judge hear?", "",
         "| group | judge | says the **conditioning** label | says the **prompt's** pianist | other |",
         "|---|---|---|---|---|"]
    for g in ("original", "matched", "mismatched", "no_context"):
        for judge, label in (("lik", "likelihood (generator, teacher-forced)"), ("clf", "classifier (paper), majority of windows")):
            J = summary["judges"].get(judge, {}).get(g)
            if not J:
                continue
            if g == "matched":
                L.append(f"| {g} | {label} | {fmt_pct(J['cond'])} (= prompt's pianist) | — | {fmt_pct(J['other'])} |")
            elif g in ("no_context", "original"):
                L.append(f"| {g} | {label} | — | {fmt_pct(J['prompt'])} | {fmt_pct(J['other'] + J['cond'])} |")
            else:
                L.append(f"| {g} | {label} | {fmt_pct(J['cond'])} | {fmt_pct(J['prompt'])} | {fmt_pct(J['other'])} |")
    clf = summary["judges"].get("clf", {})
    if clf:
        L += ["", "### Classifier, per window (the paper's agreement statistic)", "",
              "| group | agreement with conditioning label | agreement with prompt's pianist |", "|---|---|---|"]
        for g in ("original", "matched", "mismatched", "no_context"):
            J = clf.get(g)
            if J and "window_mean" in J:
                L.append(f"| {g} | {fmt_pct(J['window_mean']['cond'])} | {fmt_pct(J['window_mean']['prompt'])} |")
        L += ["", "Paper, Section 5 (P=256, 4096-token continuations): conditioned agreement 70%; "
              "mismatch at P=512: conditioning artist 15→24%, prompt artist 40→27% along the continuation.", ""]
    lik = summary["judges"].get("lik", {})
    if lik:
        L += ["### Likelihood margin", ""]
        for g in ("original", "matched", "mismatched", "no_context"):
            J = lik.get(g)
            if J and J.get("margin_median_nats") is not None:
                L.append(f"- {g}: median NLL(runner-up) − NLL(best) = {J['margin_median_nats']:.1f} nats")
        L.append("")
    L += ["## Per conditioning label", "", "| label | n | likelihood recovers it | classifier majority is it |", "|---|---|---|---|"]
    for name, v in summary["per_conditioning_label"].items():
        L.append(f"| {name} | {v['n']} | {fmt_pct(v['lik_recovers_label'])} | "
                 f"{fmt_pct(v['clf_majority_is_label']) if v['clf_majority_is_label'] is not None else '—'} |")
    L.append("")
    out_path.write_text("\n".join(L))
    return "\n".join(L)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    artist_to_id = json.loads(args.artist_map.read_text())
    names = [a for a, _ in sorted(artist_to_id.items(), key=lambda kv: kv[1])]
    num_artists = len(names)
    tokenizer = AbsTokenizer()
    with jsonlines.open(args.generations) as reader:
        records = list(reader)
    logger.info(f"{len(records)} continuations from {len({r['prompt_index'] for r in records})} prompts")

    if not args.skip_likelihood:
        likelihood_pass(records, args, num_artists, device)
    if not args.skip_classifier:
        classifier_pass(records, args, num_artists, device, tokenizer)

    summary = summarise(records, names, num_artists)
    config = {"n_prompts": len({r["prompt_index"] for r in records}),
              "prompt_length": records[0]["prompt_length"] if records else None,
              "max_continuation": max((r["n_generated"] for r in records), default=0),
              "n_conditions": len({r["cond_id"] for r in records}), "generations": str(args.generations)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in r.items() if k not in ("prompt_ids", "continuation_ids")} for r in records]
    (args.out_dir / "scored.json").write_text(json.dumps({"config": config, "summary": summary, "records": slim}, indent=1))
    print(write_report(summary, config, args.out_dir / "report.md", names))


if __name__ == "__main__":
    main()
