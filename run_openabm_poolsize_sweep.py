#!/usr/bin/env python3
"""Run pool-size sweeps on committed OpenABM sample datasets.

This runner uses the 50 non-overlapping n=3000 samples under
openabm_sample_outputs/seed_1/samples and plots 10 samples per figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from run_poolsize_experiment import parse_int_list, run_one_seed_poolsize


DEFAULT_POOL_SIZES = [1, 5] + list(range(10, 1001, 10))


def build_runner_args(args: argparse.Namespace) -> SimpleNamespace:
    """Create the minimal namespace expected by run_one_seed_poolsize."""
    return SimpleNamespace(
        method=args.method,
        cluster_symptom_strength=args.cluster_symptom_strength,
        no_force_symptom_regeneration=args.no_force_symptom_regeneration,
        c_edge_min_weight=args.c_edge_min_weight,
        neg_covid_leak_prob=args.neg_covid_leak_prob,
        noncovid_base_prob=args.noncovid_base_prob,
        pos_noncovid_prob=args.pos_noncovid_prob,
        target_positive_symptomatic_rate=args.target_positive_symptomatic_rate,
        use_positive_pool_subproblem=args.use_positive_pool_subproblem,
        beta_symptom=args.beta_symptom,
        graph_weight=args.graph_weight,
        graph_normalization=args.graph_normalization,
        clip_graph_score=args.clip_graph_score,
        max_rounds=args.max_rounds,
        pool_assignment_seed=getattr(args, "pool_assignment_seed", None),
    )


def sample_dirs(samples_root: Path, sample_indices: list[int]) -> list[Path]:
    dirs = []
    for index in sample_indices:
        d = samples_root / f"company_n3000_sample{index:02d}_maxpos5pct_work"
        if not (d / "population.csv").exists() or not (d / "contacts.csv").exists():
            raise FileNotFoundError(f"Sample files not found in {d}")
        dirs.append(d)
    return dirs


def load_existing_results(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def result_exists(existing: pd.DataFrame, sample_index: int, pool_size: int) -> bool:
    if existing.empty:
        return False
    sub = existing[
        (existing["sample_index"].astype(int) == int(sample_index))
        & (existing["pool_size"].astype(int) == int(pool_size))
        & (existing["status"].astype(str).str.startswith("ok"))
    ]
    return not sub.empty


def append_result(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.results_dir / "poolsize_sweep_results.csv"
    pool_sizes = parse_int_list(args.pool_sizes) if args.pool_sizes else DEFAULT_POOL_SIZES
    sample_indices = parse_int_list(args.sample_indices)
    dirs = sample_dirs(args.samples_root, sample_indices)
    runner_args = build_runner_args(args)
    existing = load_existing_results(result_csv) if args.resume else pd.DataFrame()

    for d in dirs:
        sample_index = int(d.name.split("_sample", 1)[1].split("_", 1)[0])
        population = pd.read_csv(d / "population.csv")
        contacts = pd.read_csv(d / "contacts.csv")
        print(f"sample={sample_index}: n={len(population)} contacts={len(contacts)}")

        for pool_size in pool_sizes:
            if args.resume and result_exists(existing, sample_index, pool_size):
                continue
            try:
                row = run_one_seed_poolsize(
                    runner_args,
                    population=population,
                    contacts=contacts,
                    seed=sample_index,
                    pool_size=int(pool_size),
                    verbose=args.verbose,
                )
            except Exception as exc:
                row = {
                    "seed": sample_index,
                    "pool_size": int(pool_size),
                    "method": args.method,
                    "initial_pool_count": np.nan,
                    "positive_pool_constraint_count": np.nan,
                    "candidate_count": np.nan,
                    "true_positive_count": np.nan,
                    "individual_test_count": np.nan,
                    "total_test_cost": np.nan,
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
                print(f"sample={sample_index} pool_size={pool_size}: ERROR {type(exc).__name__}: {exc}")

            row.update({
                "sample_index": sample_index,
                "sample_name": d.name,
                "sample_path": str(d),
            })
            append_result(result_csv, row)

    return pd.read_csv(result_csv)


def summarize_results(results: pd.DataFrame, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = results[results["status"].astype(str).str.startswith("ok")].copy()
    ok["pool_size"] = ok["pool_size"].astype(int)
    ok["sample_index"] = ok["sample_index"].astype(int)
    ok["total_test_cost"] = pd.to_numeric(ok["total_test_cost"])
    ok["initial_pool_count"] = pd.to_numeric(ok["initial_pool_count"])
    ok["individual_test_count"] = pd.to_numeric(ok["individual_test_count"])

    summary = (
        ok.groupby("pool_size", as_index=False)
        .agg(
            n_samples=("sample_index", "nunique"),
            mean_total_tests=("total_test_cost", "mean"),
            std_total_tests=("total_test_cost", "std"),
            min_total_tests=("total_test_cost", "min"),
            max_total_tests=("total_test_cost", "max"),
            median_total_tests=("total_test_cost", "median"),
            mean_pool_tests=("initial_pool_count", "mean"),
            mean_individual_tests=("individual_test_count", "mean"),
        )
        .sort_values("pool_size")
    )
    best_by_sample = (
        ok.sort_values(["sample_index", "total_test_cost", "pool_size"])
        .groupby("sample_index", as_index=False)
        .first()[
            ["sample_index", "pool_size", "total_test_cost", "initial_pool_count", "individual_test_count", "true_positive_count"]
        ]
        .rename(columns={"pool_size": "best_pool_size", "total_test_cost": "best_total_tests"})
    )

    summary.to_csv(results_dir / "poolsize_sweep_summary_by_pool_size.csv", index=False)
    best_by_sample.to_csv(results_dir / "poolsize_sweep_best_by_sample.csv", index=False)
    return summary, best_by_sample


def save_plots(results: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    ok = results[results["status"].astype(str).str.startswith("ok")].copy()
    ok["sample_index"] = ok["sample_index"].astype(int)
    ok["pool_size"] = ok["pool_size"].astype(int)
    ok["total_test_cost"] = pd.to_numeric(ok["total_test_cost"])

    for group_start in range(1, 51, 10):
        group_end = group_start + 9
        sub = ok[(ok["sample_index"] >= group_start) & (ok["sample_index"] <= group_end)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 7))
        for sample_index, g in sub.groupby("sample_index"):
            g = g.sort_values("pool_size")
            ax.plot(g["pool_size"], g["total_test_cost"], linewidth=1.4, alpha=0.9, label=f"sample {sample_index:02d}")
        ax.set_xlabel("Pool size")
        ax.set_ylabel("Total number of tests")
        ax.set_title(f"Pool-size sweep, samples {group_start:02d}-{group_end:02d}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        out = args.results_dir / f"poolsize_sweep_samples_{group_start:02d}_{group_end:02d}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("saved:", out)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(summary["pool_size"], summary["mean_total_tests"], marker="o", markersize=3, linewidth=1.8)
        ax.fill_between(
            summary["pool_size"],
            summary["mean_total_tests"] - summary["std_total_tests"].fillna(0),
            summary["mean_total_tests"] + summary["std_total_tests"].fillna(0),
            alpha=0.18,
        )
        ax.set_xlabel("Pool size")
        ax.set_ylabel("Mean total number of tests")
        ax.set_title("Mean pool-size sweep across 50 samples")
        ax.grid(True, alpha=0.3)
        out = args.results_dir / "poolsize_sweep_mean_50_samples.png"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("saved:", out)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run pool-size sweep on OpenABM sample datasets.")
    p.add_argument("--samples-root", type=Path, default=Path("openabm_sample_outputs/seed_1/samples"))
    p.add_argument("--sample-indices", nargs="+", default=["1:50"])
    p.add_argument("--pool-sizes", nargs="+", default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results/openabm_poolsize_sweep"))
    p.add_argument("--method", default="symptom_count_plus_graph")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--cluster-symptom-strength", type=float, default=0.95)
    p.add_argument("--no-force-symptom-regeneration", action="store_true")
    p.add_argument("--target-positive-symptomatic-rate", type=float, default=0.92)
    p.add_argument("--c-edge-min-weight", type=float, default=2.0)
    p.add_argument("--neg-covid-leak-prob", type=float, default=0.0)
    p.add_argument("--noncovid-base-prob", type=float, default=0.0)
    p.add_argument("--pos-noncovid-prob", type=float, default=0.0)
    p.add_argument("--beta-symptom", type=float, default=1.0)
    p.add_argument("--graph-weight", type=float, default=1.0)
    p.add_argument("--graph-normalization", default="neighbor_symptom_sum_div_max_symptoms")
    p.add_argument("--clip-graph-score", action="store_true")
    p.add_argument("--use-positive-pool-subproblem", action="store_true", default=True)
    p.add_argument("--no-positive-pool-subproblem", dest="use_positive_pool_subproblem", action="store_false")
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--pool-assignment-seed", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    results = run_sweep(args)
    summary, best_by_sample = summarize_results(results, args.results_dir)
    print("saved:", args.results_dir / "poolsize_sweep_results.csv")
    print("saved:", args.results_dir / "poolsize_sweep_summary_by_pool_size.csv")
    print("saved:", args.results_dir / "poolsize_sweep_best_by_sample.csv")
    print(best_by_sample.head(10))
    if not args.no_plots:
        save_plots(results, summary, args)


if __name__ == "__main__":
    main()
