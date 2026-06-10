#!/usr/bin/env python3
"""Run pool-size fitting experiments from the command line.

This script is the Codex/CLI equivalent of execution_poolsizefitting_multi_seed.ipynb.
Default mode uses the built-in synthetic source generator, saves population_all.csv
and contacts_all.csv for each seed, and then reuses those files when requested.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import data_poolsizefitting as ds
from data_poolsizefitting import (
    extract_or_locate_bundle,
    generate_synthetic_company_source_data,
    export_graph_samples,
    save_bundle,
    load_sample_from_root,
    build_analysis_dataset,
    compute_prior_methods,
)
from function_poolsizefittiing import (
    get_default_params,
    restrict_dataset_to_positive_pools,
    run_sequential_sparse_reconstruction,
)


CONTACT_TYPE_WEIGHTS = {
    0: 1.0,
    1: 2.0,
    2: 3.0,
}


def parse_int_list(values: list[str]) -> list[int]:
    out: list[int] = []
    for v in values:
        for part in str(v).replace(",", " ").split():
            if ":" in part:
                pieces = [int(x) for x in part.split(":")]
                if len(pieces) == 2:
                    start, stop = pieces
                    step = 1
                elif len(pieces) == 3:
                    start, stop, step = pieces
                else:
                    raise ValueError(f"Invalid range spec: {part}")
                out.extend(range(start, stop + 1, step))
            else:
                out.append(int(part))
    return sorted(set(out))


def prepare_data_for_seed(args, seed: int, output_dir: Path) -> Path:
    data_root = output_dir / f"seed_{seed}"
    data_root.mkdir(parents=True, exist_ok=True)
    population_csv = data_root / "population_all.csv"
    contacts_csv = data_root / "contacts_all.csv"

    if args.data_source == "generate":
        if args.reuse_generated and population_csv.exists() and contacts_csv.exists():
            population_all = pd.read_csv(population_csv)
            contacts_all = pd.read_csv(contacts_csv)
            print(f"seed={seed}: reused generated CSVs from {data_root}")
        else:
            population_all, contacts_all = generate_synthetic_company_source_data(
                n_total=args.n_total,
                seed=seed,
                output_dir=data_root,
            )
            # The generator should save these, but write explicitly to make reuse guaranteed.
            population_all.to_csv(population_csv, index=False)
            contacts_all.to_csv(contacts_csv, index=False)
            print(f"seed={seed}: generated and saved population/contact CSVs to {data_root}")

    elif args.data_source == "csv":
        if not population_csv.exists() or not contacts_csv.exists():
            raise FileNotFoundError(
                f"Expected {population_csv} and {contacts_csv}. "
                "Run once with --data-source generate, or place CSVs there."
            )
        population_all = pd.read_csv(population_csv)
        contacts_all = pd.read_csv(contacts_csv)
        print(f"seed={seed}: loaded existing CSVs from {data_root}")

    elif args.data_source == "bundle":
        if not args.bundle_path:
            raise ValueError("--bundle-path is required when --data-source bundle")
        data_root = extract_or_locate_bundle(
            zip_path=Path(args.bundle_path),
            extract_dir=Path(args.extract_dir),
        )
        population_all = pd.read_csv(data_root / "population_all.csv")
        contacts_all = pd.read_csv(data_root / "contacts_all.csv")
        print(f"seed={seed}: loaded bundle from {data_root}")
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    generation_summary = export_graph_samples(
        population_all=population_all,
        contacts_all=contacts_all,
        output_dir=data_root,
        sample_sizes=[args.sample_size],
        seed=seed,
        target_age_groups=args.target_age_groups,
        workplace_contact_types=args.workplace_contact_types,
        max_positive_rate=args.max_positive_rate,
        min_positive_count=args.min_positive_count,
        target_positive_count_range=(args.target_positive_count_min, args.target_positive_count_max),
        min_positive_components=args.min_positive_components,
        min_isolated_positive=args.min_isolated_positive,
        max_largest_positive_component_ratio=args.max_largest_positive_component_ratio,
        min_positive_neighbor_ratio=args.min_positive_neighbor_ratio,
    )
    generation_summary.to_csv(data_root / "generation_summary.csv", index=False)

    if args.save_bundle:
        save_bundle(data_root, output_dir / f"openabm_bundle_seed{seed}.zip")

    return data_root


def run_one_seed_poolsize(args, population, contacts, seed: int, pool_size: int, verbose: bool = True) -> dict:
    params = get_default_params(n=len(population), pool_size=int(pool_size))

    full_dataset = build_analysis_dataset(
        population,
        contacts,
        params=params,
        pool_size=int(pool_size),
        cluster_symptom_strength=args.cluster_symptom_strength,
        seed=int(seed),
        force_symptom_regeneration=not args.no_force_symptom_regeneration,
        contact_type_weights=CONTACT_TYPE_WEIGHTS,
        strong_edge_threshold=args.c_edge_min_weight,
        neg_covid_leak_prob=args.neg_covid_leak_prob,
        noncovid_base_prob=args.noncovid_base_prob,
        pos_noncovid_prob=args.pos_noncovid_prob,
        target_positive_symptomatic_rate=args.target_positive_symptomatic_rate,
        pool_assignment_seed=getattr(args, "pool_assignment_seed", None),
    )

    analysis_dataset = (
        restrict_dataset_to_positive_pools(full_dataset)
        if args.use_positive_pool_subproblem
        else full_dataset
    )

    initial_pool_count = int(analysis_dataset.get("initial_pool_count", len(full_dataset["pools"])))
    positive_pool_count = int(analysis_dataset.get("positive_pool_count", len(full_dataset["pools"])))
    candidate_count = int(len(analysis_dataset.get("x_true", [])))

    if candidate_count == 0:
        row = {
            "seed": int(seed),
            "pool_size": int(pool_size),
            "method": args.method,
            "initial_pool_count": initial_pool_count,
            "positive_pool_constraint_count": positive_pool_count,
            "candidate_count": 0,
            "true_positive_count": 0,
            "individual_test_count": 0,
            "total_test_cost": initial_pool_count,
            "status": "ok_no_positive_pool",
        }
        if verbose:
            print(f"seed={seed} pool_size={pool_size}: total={row['total_test_cost']} candidates=0")
        return row

    priors, _, _ = compute_prior_methods(
        analysis_dataset,
        beta_symptom=args.beta_symptom,
        graph_weight=args.graph_weight,
        graph_normalization=args.graph_normalization,
        clip_graph_score=args.clip_graph_score,
    )

    if args.method not in priors:
        raise KeyError(f"{args.method} not found in priors. Available methods: {list(priors.keys())}")

    max_rounds = candidate_count if args.max_rounds is None else int(args.max_rounds)
    res = run_sequential_sparse_reconstruction(
        analysis_dataset,
        mu=priors[args.method],
        label=args.method,
        max_rounds=max_rounds,
    )

    row = {
        "seed": int(seed),
        "pool_size": int(pool_size),
        "method": args.method,
        "initial_pool_count": int(res["pool_count"]),
        "positive_pool_constraint_count": int(res.get("positive_pool_constraint_count", positive_pool_count)),
        "candidate_count": candidate_count,
        "true_positive_count": int((np.asarray(analysis_dataset["x_true"]) >= analysis_dataset["params"]["x_min_positive"]).sum()),
        "individual_test_count": int(res["inspection_count"]),
        "total_test_cost": int(res["total_test_cost"]),
        "status": "ok",
    }
    if verbose:
        print(
            f"seed={seed} pool_size={pool_size}: total={row['total_test_cost']} "
            f"candidates={row['candidate_count']} positive_pools={row['positive_pool_constraint_count']}"
        )
    return row


def save_plots(ok_df: pd.DataFrame, summary_df: pd.DataFrame, results_dir: Path, args) -> None:
    import matplotlib.pyplot as plt

    if ok_df.empty:
        print("No ok rows to plot.")
        return

    sub = ok_df.copy()
    sub["seed"] = pd.to_numeric(sub["seed"])
    sub["pool_size"] = pd.to_numeric(sub["pool_size"])
    sub["total_test_cost"] = pd.to_numeric(sub["total_test_cost"])

    fig, ax = plt.subplots(figsize=(9, 5))
    if sub["pool_size"].nunique() >= 2:
        for seed, g in sub.groupby("seed"):
            g = g.sort_values("pool_size")
            ax.plot(g["pool_size"], g["total_test_cost"], marker="o", label=f"seed={int(seed)}")
        ax.set_xlabel("Pool size")
        ax.set_title("Pool size sweep across seeds")
    else:
        g = sub.sort_values("seed")
        ax.plot(g["seed"], g["total_test_cost"], marker="o", label=f"pool_size={int(g['pool_size'].iloc[0])}")
        ax.set_xlabel("Seed")
        ax.set_title("Seed sensitivity")
    ax.set_ylabel("Total number of tests")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = results_dir / "overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", out)

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        s = summary_df.sort_values("pool_size")
        ax.errorbar(s["pool_size"], s["mean_total_tests"], yerr=s["std_total_tests"].fillna(0), marker="o", capsize=3)
        ax.set_xlabel("Pool size")
        ax.set_ylabel("Mean total number of tests")
        ax.set_title("Mean total tests by pool size")
        ax.grid(True, alpha=0.3)
        out = results_dir / "summary_by_pool_size.png"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("saved:", out)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run pool-size fitting experiments.")
    p.add_argument("--seeds", nargs="+", default=["1"], help="Seeds. Supports '1 2 3' or ranges like '1:5'.")
    p.add_argument("--pool-sizes", nargs="+", default=["10"], help="Pool sizes. Supports '1 5 10' or ranges like '1:30'.")
    p.add_argument("--sample-size", type=int, default=300)
    p.add_argument("--n-total", type=int, default=1000)
    p.add_argument("--method", default="symptom_count_plus_graph")
    p.add_argument("--data-source", choices=["generate", "csv", "bundle"], default="generate")
    p.add_argument("--reuse-generated", action="store_true", help="Reuse seed_X/population_all.csv and contacts_all.csv when present.")
    p.add_argument("--output-dir", type=Path, default=Path("company_openabm_outputs"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--bundle-path", default=None)
    p.add_argument("--extract-dir", default="openabm_bundle")
    p.add_argument("--save-bundle", action="store_true")
    p.add_argument("--target-age-groups", nargs="+", type=int, default=[2, 3, 4, 5])
    p.add_argument("--workplace-contact-types", nargs="+", type=int, default=[1])
    p.add_argument("--max-positive-rate", type=float, default=0.05)
    p.add_argument("--min-positive-count", type=int, default=30)
    p.add_argument("--target-positive-count-min", type=int, default=50)
    p.add_argument("--target-positive-count-max", type=int, default=120)
    p.add_argument("--min-positive-components", type=int, default=3)
    p.add_argument("--min-isolated-positive", type=int, default=0)
    p.add_argument("--max-largest-positive-component-ratio", type=float, default=0.60)
    p.add_argument("--min-positive-neighbor-ratio", type=float, default=1.0)
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
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = parse_int_list(args.seeds)
    pool_sizes = parse_int_list(args.pool_sizes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("seeds:", seeds)
    print("pool_sizes:", pool_sizes)
    print("sample_size:", args.sample_size)
    print("n_total:", args.n_total)
    print("data_source:", args.data_source)

    data_roots = {seed: prepare_data_for_seed(args, seed, args.output_dir) for seed in seeds}

    samples = {}
    for seed in seeds:
        population, contacts, sample_dir = load_sample_from_root(data_roots[seed], sample_size=args.sample_size)
        samples[seed] = {"population": population, "contacts": contacts, "sample_dir": sample_dir}
        print(f"seed={seed}: loaded sample n={len(population)} contacts={len(contacts)} from {sample_dir}")

    rows = []
    for seed in seeds:
        for pool_size in pool_sizes:
            try:
                row = run_one_seed_poolsize(
                    args,
                    population=samples[seed]["population"],
                    contacts=samples[seed]["contacts"],
                    seed=seed,
                    pool_size=pool_size,
                    verbose=True,
                )
            except Exception as e:
                row = {
                    "seed": int(seed),
                    "pool_size": int(pool_size),
                    "method": args.method,
                    "initial_pool_count": np.nan,
                    "positive_pool_constraint_count": np.nan,
                    "candidate_count": np.nan,
                    "true_positive_count": np.nan,
                    "individual_test_count": np.nan,
                    "total_test_cost": np.nan,
                    "status": f"error: {type(e).__name__}: {e}",
                }
                print(f"seed={seed} pool_size={pool_size}: ERROR {type(e).__name__}: {e}")
            rows.append(row)

    result_df = pd.DataFrame(rows)
    result_csv = args.results_dir / "poolsize_fitting_results.csv"
    result_df.to_csv(result_csv, index=False)
    print("saved:", result_csv)
    print(result_df)

    ok_df = result_df[result_df["status"].astype(str).str.startswith("ok")].copy()
    summary_df = pd.DataFrame()
    if not ok_df.empty:
        summary_df = (
            ok_df.groupby(["pool_size", "method"], as_index=False)
            .agg(
                n_runs=("total_test_cost", "count"),
                mean_total_tests=("total_test_cost", "mean"),
                std_total_tests=("total_test_cost", "std"),
                min_total_tests=("total_test_cost", "min"),
                max_total_tests=("total_test_cost", "max"),
                median_total_tests=("total_test_cost", "median"),
                mean_initial_pool_count=("initial_pool_count", "mean"),
                mean_individual_test_count=("individual_test_count", "mean"),
                mean_positive_pools=("positive_pool_constraint_count", "mean"),
                mean_candidate_count=("candidate_count", "mean"),
            )
            .sort_values(["mean_total_tests", "pool_size"])
        )
    summary_csv = args.results_dir / "poolsize_fitting_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("saved:", summary_csv)
    print(summary_df)

    if not args.no_plots:
        save_plots(ok_df, summary_df, args.results_dir, args)


if __name__ == "__main__":
    main()
