#!/usr/bin/env python
"""Produce every figure and table of the survey replication into results/disinfo.

    P=01-info-propagation/desinformation-paper/.venv/bin/python
    $P scripts/run_disinfo.py --quiet              # figures + meta-analysis, fast
    $P scripts/run_disinfo.py --quiet --experiments  # + the live GNN runs

Without ``--experiments`` this runs in under a minute: the paper figures and the
Tables 1-2 meta-analysis need no training. The experiment suites are the slow
part, so they are opt-in and log per-suite timings.
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

import pandas as pd  # noqa: E402

from disinfo import (Config, FeatureCache, dataset_sizes, plots, results_to_frame, run_all_checks, save_results,
                     suite_feature_ablation, suite_gnn_comparison,
                     suite_graph_comparison, suite_liar_granularity,
                     suite_pheme_split, verify_claims, verify_table3)
from disinfo.config import FIGURES, RESULTS  # noqa: E402
from disinfo.survey_data import (datasets_table, long_results,  # noqa: E402
                                 methods_table)

DATASETS = ["twitter15", "twitter16", "pheme", "ced", "liar"]


def banner(msg, quiet):
    if not quiet:
        print(f"\n=== {msg} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true",
                    help="summaries and the final table only")
    ap.add_argument("--experiments", action="store_true",
                    help="run the live GNN suites (slow)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()
    quiet = args.quiet

    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- 0. property checks on the equations ----
    banner("property checks (Eqs. 2-10)", quiet)
    checks = run_all_checks(verbose=not quiet)
    (RESULTS / "checks.json").write_text(json.dumps(checks, indent=1))
    if quiet:
        print(f"property checks: {len(checks)}/6 passed")

    # ---- 1. reproduce Figs. 1-6 ----
    banner("Figures 1-6", quiet)
    for name, fn in [("fig1_framework", plots.fig1_framework),
                     ("fig2_false_information", plots.fig2_false_information),
                     ("fig3_gnn_schematic", plots.fig3_gnn_schematic),
                     ("fig4_features", plots.fig4_features),
                     ("fig5_approaches", plots.fig5_approaches),
                     ("fig6_algorithms", plots.fig6_algorithms)]:
        p = plots.save(fn(), name)
        if not quiet:
            print(f"  {p.name}")

    # ---- 2. meta-analysis of Tables 1-3 ----
    banner("meta-analysis of Tables 1-3", quiet)
    from disinfo.survey_data import METHODS
    long_df = long_results()
    methods_table().to_csv(RESULTS / "table1_2_methods.csv", index=False)
    long_df.to_csv(RESULTS / "table1_2_results_long.csv", index=False)
    datasets_table().to_csv(RESULTS / "table3_datasets.csv", index=False)

    for name, fig in [
        ("meta_gnn_usage", plots.plot_gnn_usage(METHODS)),
        ("meta_graph_type", plots.plot_graph_type_usage(METHODS)),
        ("meta_feature_usage", plots.plot_feature_usage(METHODS)),
        ("meta_year_trend", plots.plot_year_trend(METHODS)),
        ("meta_dataset_usage", plots.plot_dataset_usage(long_df)),
        ("meta_reported_performance", plots.plot_reported_performance(long_df)),
        ("meta_performance_heatmap", plots.plot_performance_heatmap(long_df)),
    ]:
        p = plots.save(fig, name)
        if not quiet:
            print(f"  {p.name}")

    claims = verify_claims()
    claims.to_csv(RESULTS / "claim_verification.csv", index=False)
    plots.save(plots.plot_claims(claims), "meta_claims")
    n_ok = int(claims["supported"].sum())
    print(f"survey claims verified against its own tables: "
          f"{n_ok}/{len(claims)} supported")
    for _, r in claims[~claims["supported"]].iterrows():
        print(f"  CONTRADICTED  {r['claim']} (Sect. {r['section']}): {r['evidence']}")

    # ---- 3. Table 3 sizes vs the actual downloads ----
    banner("Table 3 vs downloaded data", quiet)
    t3 = verify_table3(dataset_sizes())
    t3.to_csv(RESULTS / "table3_verification.csv", index=False)
    checked = t3[t3["match"].notna()]
    print(f"Table 3 sizes reproduced: "
          f"{int(checked['match'].sum())}/{len(checked)} datasets exact")
    if not quiet:
        print(checked.to_string(index=False))

    # ---- 4. live experiments ----
    if args.experiments:
        base = Config(epochs=args.epochs, patience=25)
        cache = FeatureCache()

        banner("suite: architectures (Eqs. 2-10)", quiet)
        t0 = time.time()
        r_gnn = suite_gnn_comparison(args.datasets, base, n_seeds=args.seeds,
                                     quiet=quiet, cache=cache)
        save_results(r_gnn, "results_gnn_comparison")
        df_gnn = results_to_frame(r_gnn)
        df_gnn.to_csv(RESULTS / "results_gnn_comparison.csv", index=False)
        plots.save(plots.plot_gnn_comparison(df_gnn), "exp_gnn_comparison")
        print(f"architectures suite: {time.time() - t0:.0f}s")

        banner("suite: graph construction (Sect. 5.3)", quiet)
        t0 = time.time()
        r_graph = suite_graph_comparison(args.datasets, base,
                                         n_seeds=args.seeds, quiet=quiet,
                                         cache=cache)
        save_results(r_graph, "results_graph_comparison")
        df_graph = results_to_frame(r_graph)
        df_graph.to_csv(RESULTS / "results_graph_comparison.csv", index=False)
        plots.save(plots.plot_graph_comparison(df_graph), "exp_graph_comparison")
        print(f"graph-construction suite: {time.time() - t0:.0f}s")

        banner("suite: feature ablation (Fig. 4)", quiet)
        t0 = time.time()
        r_abl = suite_feature_ablation("twitter15", base, n_seeds=args.seeds,
                                       quiet=quiet, cache=cache)
        save_results(r_abl, "results_feature_ablation")
        df_abl = results_to_frame(r_abl)
        df_abl.to_csv(RESULTS / "results_feature_ablation.csv", index=False)
        plots.save(plots.plot_ablation(df_abl), "exp_feature_ablation")
        print(f"ablation suite: {time.time() - t0:.0f}s")

        banner("suite: LIAR granularity (Table 2 vs Sect. 7)", quiet)
        t0 = time.time()
        r_liar = suite_liar_granularity(base, n_seeds=args.seeds, quiet=quiet)
        save_results(r_liar, "results_liar_granularity")
        results_to_frame(r_liar).to_csv(RESULTS / "results_liar_granularity.csv",
                                        index=False)
        print(f"LIAR suite: {time.time() - t0:.0f}s")

        banner("suite: PHEME split protocol", quiet)
        t0 = time.time()
        r_pheme = suite_pheme_split(base, n_seeds=args.seeds, quiet=quiet)
        save_results(r_pheme, "results_pheme_split")
        results_to_frame(r_pheme).to_csv(RESULTS / "results_pheme_split.csv",
                                         index=False)
        print(f"PHEME split suite: {time.time() - t0:.0f}s")

        # ---- 5. our numbers against the published range ----
        banner("this replication vs Tables 1-2", quiet)
        ours = pd.concat([df_gnn, df_graph], ignore_index=True)
        name_map = {"twitter15": "Twitter15", "twitter16": "Twitter16",
                    "PHEME": "PHEME", "CED": "CED", "LIAR": "LIAR"}
        ours["dataset"] = ours["dataset"].map(lambda d: name_map.get(d, d))
        ours.to_csv(RESULTS / "results_all.csv", index=False)
        plots.save(plots.plot_vs_literature(ours, long_df), "exp_vs_literature")

        print("\n--- final table: best configuration per dataset ---")
        best = (ours.sort_values("accuracy", ascending=False)
                    .groupby("dataset").head(1)
                    [["dataset", "gnn", "graph", "accuracy", "accuracy_std",
                      "macro_f1"]])
        print(best.to_string(index=False))
        print(f"\n{cache.stats()}")

    print(f"\nfigures -> {FIGURES}")
    print(f"tables  -> {RESULTS}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
