#!/usr/bin/env python
"""Produce every figure and table of the bot-detection replication.

Artefacts land in ``01-info-propagation/bot-detection-paper/results``.

    P=01-info-propagation/bot-detection-paper/.venv/bin/python
    $P scripts/run_botsage.py --quiet                 # checks, figures, baselines
    $P scripts/run_botsage.py --quiet --experiments   # + the live SVM runs

Without ``--experiments`` this needs no trained model and finishes in under a
minute: property checks, the paper's three diagrams, the timing figures, and the
label-distribution analysis that Table 5 turns on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from botsage import (Config, load_dataset, plots, results_to_frame,  # noqa: E402
                     run_all_checks, save_results, sage_embeddings,
                     suite_ablation, suite_graph_scope, suite_protocol,
                     suite_regularization_equivalence, suite_replicate,
                     suite_seed_sensitivity, suite_trained_vs_untrained,
                     twibot22_baseline_report)
from botsage.config import FIGURES, PAPER_TABLE4, PAPER_TABLE5, RESULTS  # noqa: E402
from botsage.pipeline import restrict_graph  # noqa: E402

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed" / "bot"


def banner(msg, quiet):
    if not quiet:
        print(f"\n=== {msg} ===")


def load_text(dataset: str, model: str):
    p = PROCESSED / f"{dataset}_{model}_mean_labelled.npy"
    return np.load(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--experiments", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--graph-scope", default="all", choices=["all", "labelled"])
    args = ap.parse_args()
    quiet = args.quiet

    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- 0. property checks ----
    banner("property checks (Sect. 3.5)", quiet)
    checks = run_all_checks(verbose=not quiet)
    (RESULTS / "checks.json").write_text(json.dumps(checks, indent=1))
    print(f"property checks: {len(checks)}/7 passed")

    # ---- 1. the paper's diagrams and timing tables ----
    banner("figures 1-3 and Tables 1-3", quiet)
    for name, fn in [("fig1_graphsage", plots.fig1_graphsage),
                     ("fig2_workflow", plots.fig2_workflow),
                     ("fig3_graph_construction", plots.fig3_graph_construction),
                     ("paper_timings", plots.plot_timings)]:
        p = plots.save(fn(), name)
        if not quiet:
            print(f"  {p.name}")

    # ---- 2. the TwiBot-22 baseline analysis (the headline finding) ----
    banner("TwiBot-22 label distribution vs Table 5", quiet)
    report = twibot22_baseline_report()
    (RESULTS / "twibot22_baselines.json").write_text(json.dumps(report, indent=1))
    plots.save(plots.plot_twibot22_splits(report), "twibot22_splits")
    plots.save(plots.plot_table5(baseline=report["corpus"]["majority_baseline"]),
               "table5_vs_baseline")

    base = report["corpus"]["majority_baseline"]
    below = [(m, a) for m, _, a, _, _ in PAPER_TABLE5 if a / 100 < base]
    print(f"TwiBot-22 majority baseline (corpus, the paper's 5-fold protocol): "
          f"{base:.4f}")
    print(f"  rows of Table 5 below it: {len(below)}/{len(PAPER_TABLE5)}"
          f"  -> {', '.join(m for m, _ in below)}")
    print(f"  official test-split baseline (what the leaderboard used): "
          f"{report['test']['majority_baseline']:.4f}")

    # ---- 3. Cresci-15 ----
    banner("Cresci-15", quiet)
    ds = load_dataset("cresci-15", with_tweets=False)
    print(ds.summary())

    cresci_base = ds.majority_baseline()
    plots.save(plots.plot_table4(baseline=cresci_base), "table4_vs_baseline")

    # Graph degeneracy: why the network branch contributes so little.
    lab = ds.labelled
    labset = np.zeros(len(ds), bool)
    labset[lab] = True
    ei = ds.edge_index
    both = labset[ei[0]] & labset[ei[1]]
    ei_lab = restrict_graph(ds, "labelled")
    deg_lab = np.bincount(ei_lab[1], minlength=len(ds)) if ei_lab.size else np.zeros(len(ds), int)
    degeneracy = {
        "user-user edges": int(ei.shape[1]),
        "edges joining two users with metadata": int(both.sum()),
        "fraction of such edges": float(both.mean()),
        "labelled users isolated in that subgraph": int((deg_lab[lab] == 0).sum()),
        "labelled users total": int(len(lab)),
    }
    (RESULTS / "cresci_graph_degeneracy.json").write_text(
        json.dumps(degeneracy, indent=1))
    plots.save(plots.plot_neighbour_degeneracy(degeneracy), "graph_degeneracy")
    if not quiet:
        for k, v in degeneracy.items():
            print(f"  {k:44s} {v}")

    # Rank of the embedding, on the real data.
    import torch
    x = torch.from_numpy(np.ascontiguousarray(ds.features)).float()
    x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp(min=1e-6)
    emb = sage_embeddings(x, torch.from_numpy(np.ascontiguousarray(ei)),
                          out_channels=128, seed=0).numpy()
    plots.save(plots.plot_singular_spectrum(emb[lab]), "singular_spectrum")

    if not args.experiments:
        print(f"\nfigures -> {FIGURES}\ntables  -> {RESULTS}")
        print(f"total {time.time() - t_start:.0f}s")
        return

    # ---- 4. live experiments ----
    cfg = Config()
    texts = {}
    for label, model in [("BERT", "bert-base-uncased"),
                         ("DistilBERT", "distilbert-base-uncased")]:
        te = load_text("cresci-15", model)
        if te is None:
            print(f"  [skip] {model}: run scripts/prepare_bot_embeddings.py")
        else:
            texts[label] = te
    distil = texts.get("DistilBERT")

    banner("replicating Table 4", quiet)
    r = suite_replicate(ds, cfg, texts, graph_scope=args.graph_scope, quiet=quiet)
    save_results(r, "results_replicate")
    df = results_to_frame(r)
    df.to_csv(RESULTS / "results_replicate.csv", index=False)
    plots.save(plots.plot_table4(ours=df, baseline=cresci_base),
               "table4_with_ours")

    print("\n--- ours vs Table 4 ---")
    paper = {m: (a, f) for m, _, a, f, mine in PAPER_TABLE4 if mine}
    for _, row in df.iterrows():
        key = row["variant"].replace("GraphSage+", "GraphSage+")
        p = paper.get(key)
        note = (f"  paper {p[0]:.2f}/{p[1]:.2f}  delta "
                f"{row['accuracy'] * 100 - p[0]:+.2f}") if p else ""
        print(f"  {row['variant']:24s} acc {row['accuracy'] * 100:.2f} "
              f"f1 {row['f1'] * 100:.2f}{note}")

    banner("ablation: what the 896 dimensions contribute", quiet)
    r = suite_ablation(ds, cfg, text_embeddings=distil,
                       graph_scope=args.graph_scope, quiet=quiet)
    save_results(r, "results_ablation")
    df_abl = results_to_frame(r)
    df_abl.to_csv(RESULTS / "results_ablation.csv", index=False)
    plots.save(plots.plot_ablation(df_abl, baseline=cresci_base),
               "exp_ablation")

    banner("is GraphSage[128] more than its 10 inputs?", quiet)
    r = suite_regularization_equivalence(ds, cfg, graph_scope=args.graph_scope,
                                         quiet=quiet)
    save_results(r, "results_regularization")
    df_reg = pd.DataFrame(r)
    df_reg.to_csv(RESULTS / "results_regularization.csv", index=False)
    print(f"  delta at C=1     : {df_reg.iloc[3]['delta']:+.4f}")
    print(f"  delta at C={df_reg.iloc[-1]['C']:g}  : "
          f"{df_reg.iloc[-1]['delta']:+.4f}  <- vanishes as the penalty weakens")

    banner("seed sensitivity of the untrained layer", quiet)
    r = suite_seed_sensitivity(ds, cfg, seeds=range(args.seeds),
                               graph_scope=args.graph_scope, quiet=quiet)
    save_results(r, "results_seeds")
    df_seed = results_to_frame(r)
    df_seed.to_csv(RESULTS / "results_seeds.csv", index=False)
    plots.save(plots.plot_seed_sensitivity(df_seed), "exp_seed_sensitivity")
    print(f"  accuracy across {args.seeds} seeds: "
          f"{df_seed['accuracy'].min():.4f} - {df_seed['accuracy'].max():.4f} "
          f"(spread {df_seed['accuracy'].max() - df_seed['accuracy'].min():.4f})")

    banner("trained vs untrained GraphSAGE", quiet)
    r = suite_trained_vs_untrained(ds, cfg, graph_scope=args.graph_scope,
                                   quiet=quiet)
    save_results(r, "results_trained")
    df_tr = results_to_frame(r)
    df_tr.to_csv(RESULTS / "results_trained.csv", index=False)
    plots.save(plots.plot_trained_vs_untrained(df_tr, baseline=cresci_base),
               "exp_trained_vs_untrained")

    banner("edge-definition reading", quiet)
    r = suite_graph_scope(ds, cfg, text_embeddings=None, quiet=quiet)
    save_results(r, "results_graph_scope")
    df_gs = results_to_frame(r)
    df_gs.to_csv(RESULTS / "results_graph_scope.csv", index=False)
    plots.save(plots.plot_graph_scope(df_gs, baseline=cresci_base),
               "exp_graph_scope")

    banner("protocol: 5-fold CV vs the official split", quiet)
    r = suite_protocol(ds, cfg, text_embeddings=distil,
                       graph_scope=args.graph_scope, quiet=quiet)
    save_results(r, "results_protocol")
    results_to_frame(r).to_csv(RESULTS / "results_protocol.csv", index=False)

    print(f"\nfigures -> {FIGURES}")
    print(f"tables  -> {RESULTS}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
