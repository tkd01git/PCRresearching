#!/usr/bin/env python3
"""Compare random pooling and graph-aware sparse-friendly pooling designs."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import data_poolsizefitting as ds
import function_poolsizefittiing as fn
from run_poolsize_experiment import CONTACT_TYPE_WEIGHTS, parse_int_list
from run_openabm_poolsize_sweep import sample_dirs


POOL_SIZE_CANDIDATES = [80, 100, 120, 130, 140, 150, 160, 180, 200]
METHOD_RANDOM_EXHAUSTIVE = "random_A_exhaustive_individual_testing"
METHOD_RANDOM_SPARSE = "random_A_weighted_sparse_reconstruction"
METHOD_PROPOSED_SPARSE = "proposed_A_weighted_sparse_reconstruction"


def build_base_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        method=args.method,
        cluster_symptom_strength=args.cluster_symptom_strength,
        no_force_symptom_regeneration=args.no_force_symptom_regeneration,
        c_edge_min_weight=args.c_edge_min_weight,
        neg_covid_leak_prob=args.neg_covid_leak_prob,
        noncovid_base_prob=args.noncovid_base_prob,
        pos_noncovid_prob=args.pos_noncovid_prob,
        target_positive_symptomatic_rate=args.target_positive_symptomatic_rate,
        use_positive_pool_subproblem=True,
        beta_symptom=args.beta_symptom,
        graph_weight=args.graph_weight,
        graph_normalization=args.graph_normalization,
        clip_graph_score=args.clip_graph_score,
        max_rounds=args.max_rounds,
    )


def make_random_pools(n: int, pool_size: int, seed: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    order = np.random.default_rng(int(seed)).permutation(int(n))
    return fn.make_pooling_matrix(n, pool_size=pool_size, gaps=(1,), allow_incomplete_last_pool=True, pool_order=order)


def make_graph_aware_sparse_friendly_pools(
    W: np.ndarray,
    risk_score: np.ndarray,
    pool_size: int,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    n = len(risk_score)
    pool_count = int(math.ceil(n / int(pool_size)))
    pools: list[list[int]] = [[] for _ in range(pool_count)]
    pool_sets: list[set[int]] = [set() for _ in range(pool_count)]
    risk_sums = np.zeros(pool_count, dtype=float)
    adjacency = [set(np.where(np.asarray(W[i]) > 0)[0].astype(int).tolist()) for i in range(n)]

    # Existing risk score, high first. Ties are deterministic by index.
    ordered_people = sorted(range(n), key=lambda i: (-float(risk_score[i]), int(i)))
    for person in ordered_people:
        neighbors = adjacency[int(person)]
        best_pool = None
        best_key = None
        for pool_idx, members in enumerate(pools):
            if len(members) >= pool_size:
                continue
            neighbor_count = len(neighbors.intersection(pool_sets[pool_idx]))
            key = (neighbor_count, float(risk_sums[pool_idx]), len(members), pool_idx)
            if best_key is None or key < best_key:
                best_key = key
                best_pool = pool_idx
        if best_pool is None:
            raise RuntimeError("No pool with remaining capacity")
        pools[best_pool].append(int(person))
        pool_sets[best_pool].add(int(person))
        risk_sums[best_pool] += float(risk_score[person])

    pool_tuples = [tuple(p) for p in pools if p]
    A = np.zeros((len(pool_tuples), n), dtype=float)
    for row, pool in enumerate(pool_tuples):
        A[row, list(pool)] = 1.0
    return A, pool_tuples


def with_pooling(dataset: dict, A: np.ndarray, pools: list[tuple[int, ...]], design_label: str) -> dict:
    out = dict(dataset)
    pooled_amount_true, pooled_ct, pooled_amount_est = fn.pooled_measurements_qpcr(A, np.asarray(dataset["x_true"]), dataset["params"])
    out["A"] = A
    out["pools"] = pools
    out["pooled_amount_true"] = pooled_amount_true
    out["pooled_ct"] = pooled_ct
    out["pooled_amount_est"] = pooled_amount_est
    out["pool_design"] = design_label
    return out


def positive_pool_stats(dataset: dict) -> dict:
    params = dataset["params"]
    cutoff = float(params.get("positive_cutoff", params.get("x_min_positive", 1e3)))
    x_true = np.asarray(dataset["x_true"], dtype=float)
    is_positive = x_true >= float(params.get("x_min_positive", 1e3))
    pooled_amount_est = np.asarray(dataset["pooled_amount_est"], dtype=float)
    positive_pool_indices = np.where(pooled_amount_est >= cutoff)[0]
    counts = []
    for pool_idx in positive_pool_indices:
        pool = dataset["pools"][int(pool_idx)]
        counts.append(int(is_positive[list(pool)].sum()))
    return {
        "number_of_positive_pools": int(len(positive_pool_indices)),
        "average_positive_count_per_positive_pool": float(np.mean(counts)) if counts else 0.0,
        "max_positive_count_per_positive_pool": int(max(counts)) if counts else 0,
    }


def true_positive_rank_from_measurements(res: dict, fallback: int) -> int:
    ranks = [int(m["step"]) + 1 for m in res.get("individual_measurements", []) if bool(m.get("is_true_positive"))]
    return int(max(ranks)) if ranks else int(fallback)


def evaluate_exhaustive(full_dataset: dict, sample_id: int, pool_size: int, method: str) -> dict:
    analysis_dataset = fn.restrict_dataset_to_positive_pools(full_dataset)
    stats = positive_pool_stats(full_dataset)
    candidate_count = int(len(analysis_dataset.get("x_true", [])))
    true_positive_count = int(
        (np.asarray(analysis_dataset.get("x_true", []), dtype=float) >= analysis_dataset["params"]["x_min_positive"]).sum()
    ) if candidate_count else 0
    initial_pool_tests = int(analysis_dataset.get("initial_pool_count", len(full_dataset["pools"])))
    candidate_positive_rate = true_positive_count / candidate_count if candidate_count else 0.0
    return {
        "sample_id": int(sample_id),
        "pool_size": int(pool_size),
        "method": method,
        "total_tests": int(initial_pool_tests + candidate_count),
        "initial_pool_tests": int(initial_pool_tests),
        "candidate_count": int(candidate_count),
        "individual_tests": int(candidate_count),
        "true_positive_rank": int(candidate_count),
        "number_of_positive_pools": stats["number_of_positive_pools"],
        "candidate_positive_rate": float(candidate_positive_rate),
        "average_positive_count_per_positive_pool": stats["average_positive_count_per_positive_pool"],
        "max_positive_count_per_positive_pool": stats["max_positive_count_per_positive_pool"],
    }


def evaluate_sparse(full_dataset: dict, args: argparse.Namespace, sample_id: int, pool_size: int, method: str) -> dict:
    analysis_dataset = fn.restrict_dataset_to_positive_pools(full_dataset)
    stats = positive_pool_stats(full_dataset)
    initial_pool_tests = int(analysis_dataset.get("initial_pool_count", len(full_dataset["pools"])))
    candidate_count = int(len(analysis_dataset.get("x_true", [])))
    true_positive_count = int(
        (np.asarray(analysis_dataset.get("x_true", []), dtype=float) >= analysis_dataset["params"]["x_min_positive"]).sum()
    ) if candidate_count else 0
    candidate_positive_rate = true_positive_count / candidate_count if candidate_count else 0.0
    if candidate_count == 0:
        individual_tests = 0
        total_tests = initial_pool_tests
        true_positive_rank = 0
    else:
        priors, _, _ = ds.compute_prior_methods(
            analysis_dataset,
            beta_symptom=args.beta_symptom,
            graph_weight=args.graph_weight,
            graph_normalization=args.graph_normalization,
            clip_graph_score=args.clip_graph_score,
        )
        res = fn.run_sequential_sparse_reconstruction(
            analysis_dataset,
            mu=priors[args.method],
            label=args.method,
            max_rounds=candidate_count if args.max_rounds is None else int(args.max_rounds),
        )
        individual_tests = int(res["inspection_count"])
        total_tests = int(res["total_test_cost"])
        true_positive_rank = true_positive_rank_from_measurements(res, fallback=individual_tests)
    return {
        "sample_id": int(sample_id),
        "pool_size": int(pool_size),
        "method": method,
        "total_tests": int(total_tests),
        "initial_pool_tests": int(initial_pool_tests),
        "candidate_count": int(candidate_count),
        "individual_tests": int(individual_tests),
        "true_positive_rank": int(true_positive_rank),
        "number_of_positive_pools": stats["number_of_positive_pools"],
        "candidate_positive_rate": float(candidate_positive_rate),
        "average_positive_count_per_positive_pool": stats["average_positive_count_per_positive_pool"],
        "max_positive_count_per_positive_pool": stats["max_positive_count_per_positive_pool"],
    }


def append_row(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def result_exists(existing: pd.DataFrame, sample_id: int, pool_size: int, method: str) -> bool:
    if existing.empty:
        return False
    sub = existing[
        (existing["sample_id"].astype(int) == int(sample_id))
        & (existing["pool_size"].astype(int) == int(pool_size))
        & (existing["method"].astype(str) == str(method))
    ]
    return not sub.empty


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.results_dir / "all_results_by_sample_poolsize_method.csv"
    existing = pd.read_csv(result_csv) if args.resume and result_csv.exists() else pd.DataFrame()
    pool_sizes = parse_int_list(args.pool_sizes)
    dirs = sample_dirs(args.samples_root, parse_int_list(args.sample_indices))

    for sample_dir in dirs:
        sample_id = int(sample_dir.name.split("_sample", 1)[1].split("_", 1)[0])
        population = pd.read_csv(sample_dir / "population.csv")
        contacts = pd.read_csv(sample_dir / "contacts.csv")
        print(f"sample={sample_id}: n={len(population)} contacts={len(contacts)}")

        base_params = fn.get_default_params(n=len(population), pool_size=pool_sizes[0])
        base_dataset = ds.build_analysis_dataset(
            population,
            contacts,
            params=base_params,
            pool_size=pool_sizes[0],
            cluster_symptom_strength=args.cluster_symptom_strength,
            seed=sample_id,
            force_symptom_regeneration=not args.no_force_symptom_regeneration,
            contact_type_weights=CONTACT_TYPE_WEIGHTS,
            strong_edge_threshold=args.c_edge_min_weight,
            neg_covid_leak_prob=args.neg_covid_leak_prob,
            noncovid_base_prob=args.noncovid_base_prob,
            pos_noncovid_prob=args.pos_noncovid_prob,
            target_positive_symptomatic_rate=args.target_positive_symptomatic_rate,
        )
        _, risk_df, _ = ds.compute_prior_methods(
            base_dataset,
            beta_symptom=args.beta_symptom,
            graph_weight=args.graph_weight,
            graph_normalization=args.graph_normalization,
            clip_graph_score=args.clip_graph_score,
        )
        risk_score = risk_df["combined_score"].to_numpy(dtype=float)

        for pool_size in pool_sizes:
            params = fn.derive_params({**base_dataset["params"], "n": len(population), "pool_size": int(pool_size)})
            pool_dataset_base = dict(base_dataset)
            pool_dataset_base["params"] = params
            random_seed = int(args.random_seed_base + sample_id * 1000 + pool_size)
            random_A, random_pools = make_random_pools(len(population), int(pool_size), random_seed)
            random_dataset = with_pooling(pool_dataset_base, random_A, random_pools, "random")
            proposed_A, proposed_pools = make_graph_aware_sparse_friendly_pools(
                np.asarray(base_dataset["W"], dtype=float),
                risk_score,
                int(pool_size),
            )
            proposed_dataset = with_pooling(pool_dataset_base, proposed_A, proposed_pools, "graph_aware_sparse_friendly")

            rows = [
                evaluate_exhaustive(random_dataset, sample_id, pool_size, METHOD_RANDOM_EXHAUSTIVE),
                evaluate_sparse(random_dataset, args, sample_id, pool_size, METHOD_RANDOM_SPARSE),
                evaluate_sparse(proposed_dataset, args, sample_id, pool_size, METHOD_PROPOSED_SPARSE),
            ]
            for row in rows:
                if args.resume and result_exists(existing, sample_id, pool_size, row["method"]):
                    continue
                append_row(result_csv, row)
    return pd.read_csv(result_csv)


def summarize(results: pd.DataFrame, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for col in ["sample_id", "pool_size", "total_tests", "initial_pool_tests", "candidate_count", "individual_tests"]:
        results[col] = pd.to_numeric(results[col])

    summary = (
        results.groupby(["pool_size", "method"], as_index=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            mean_total_tests=("total_tests", "mean"),
            median_total_tests=("total_tests", "median"),
            std_total_tests=("total_tests", "std"),
            mean_candidate_count=("candidate_count", "mean"),
            mean_individual_tests=("individual_tests", "mean"),
            mean_positive_pools=("number_of_positive_pools", "mean"),
            mean_candidate_positive_rate=("candidate_positive_rate", "mean"),
            mean_avg_positive_count_per_positive_pool=("average_positive_count_per_positive_pool", "mean"),
            mean_max_positive_count_per_positive_pool=("max_positive_count_per_positive_pool", "mean"),
        )
        .sort_values(["pool_size", "method"])
    )

    wide = results.pivot_table(index=["sample_id", "pool_size"], columns="method", values="total_tests", aggfunc="first").reset_index()
    inequality = pd.DataFrame({
        "sample_id": wide["sample_id"].astype(int),
        "pool_size": wide["pool_size"].astype(int),
        "proposed_total_tests": wide[METHOD_PROPOSED_SPARSE].astype(int),
        "random_sparse_total_tests": wide[METHOD_RANDOM_SPARSE].astype(int),
        "random_exhaustive_total_tests": wide[METHOD_RANDOM_EXHAUSTIVE].astype(int),
    })
    inequality["proposed_lt_random_sparse"] = inequality["proposed_total_tests"] < inequality["random_sparse_total_tests"]
    inequality["proposed_lt_random_exhaustive"] = inequality["proposed_total_tests"] < inequality["random_exhaustive_total_tests"]

    inequality_summary = (
        inequality.groupby("pool_size", as_index=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            proposed_lt_random_sparse_rate=("proposed_lt_random_sparse", "mean"),
            proposed_lt_random_exhaustive_rate=("proposed_lt_random_exhaustive", "mean"),
            mean_proposed_total_tests=("proposed_total_tests", "mean"),
            mean_random_sparse_total_tests=("random_sparse_total_tests", "mean"),
            mean_random_exhaustive_total_tests=("random_exhaustive_total_tests", "mean"),
        )
        .sort_values("pool_size")
    )

    best_poolsize = (
        summary.sort_values(["method", "mean_total_tests", "pool_size"])
        .groupby("method", as_index=False)
        .first()
    )

    results.to_csv(results_dir / "all_results_by_sample_poolsize_method.csv", index=False)
    inequality.to_csv(results_dir / "inequality_check_by_sample_poolsize.csv", index=False)
    summary.to_csv(results_dir / "summary_by_poolsize_method.csv", index=False)
    inequality_summary.to_csv(results_dir / "inequality_summary_by_poolsize.csv", index=False)
    best_poolsize.to_csv(results_dir / "best_poolsize_summary.csv", index=False)

    # Compatibility names requested in the first section.
    results.to_csv(results_dir / "poolsize_search_comparison.csv", index=False)
    best_poolsize.to_csv(results_dir / "best_method_summary.csv", index=False)
    return summary, inequality, inequality_summary, best_poolsize


def save_plots(results: pd.DataFrame, summary: pd.DataFrame, inequality_summary: pd.DataFrame, results_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pivot_order = [METHOD_RANDOM_EXHAUSTIVE, METHOD_RANDOM_SPARSE, METHOD_PROPOSED_SPARSE]
    labels = {
        METHOD_RANDOM_EXHAUSTIVE: "random A + exhaustive",
        METHOD_RANDOM_SPARSE: "random A + sparse",
        METHOD_PROPOSED_SPARSE: "proposed A + sparse",
    }
    fig, ax = plt.subplots(figsize=(12, 7))
    for method in pivot_order:
        sub = summary[summary["method"] == method].sort_values("pool_size")
        ax.plot(sub["pool_size"], sub["mean_total_tests"], marker="o", linewidth=1.8, label=labels[method])
    ax.set_xlabel("Pool size")
    ax.set_ylabel("Mean total tests")
    ax.set_title("Total tests by pool size and method")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(results_dir / "total_tests_by_poolsize_method.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(results_dir / "total_tests_by_poolsize_and_method.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metric_plots = [
        ("candidate_count", "candidate_count_by_method.png", "Candidate count"),
        ("true_positive_rank", "true_positive_rank_by_method.png", "True positive rank"),
        ("average_positive_count_per_positive_pool", "positive_count_per_positive_pool_by_method.png", "Positive count per positive pool"),
    ]
    for metric, filename, ylabel in metric_plots:
        agg = results.groupby(["pool_size", "method"], as_index=False)[metric].mean()
        fig, ax = plt.subplots(figsize=(12, 7))
        for method in pivot_order:
            sub = agg[agg["method"] == method].sort_values("pool_size")
            ax.plot(sub["pool_size"], sub[metric], marker="o", linewidth=1.8, label=labels[method])
        ax.set_xlabel("Pool size")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by method")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(results_dir / filename, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        inequality_summary["pool_size"],
        inequality_summary["proposed_lt_random_sparse_rate"],
        marker="o",
        label="proposed < random sparse",
    )
    ax.plot(
        inequality_summary["pool_size"],
        inequality_summary["proposed_lt_random_exhaustive_rate"],
        marker="o",
        label="proposed < random exhaustive",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Pool size")
    ax.set_ylabel("Success rate")
    ax.set_title("Inequality success rate by pool size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(results_dir / "inequality_success_rate_by_poolsize.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    best = summary.sort_values(["method", "mean_total_tests"]).groupby("method", as_index=False).first()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([labels[m] for m in best["method"]], best["mean_total_tests"])
    ax.set_ylabel("Best mean total tests")
    ax.set_title("Best total tests comparison")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(results_dir / "best_total_tests_comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_console_summary(summary: pd.DataFrame, inequality_summary: pd.DataFrame, best_poolsize: pd.DataFrame) -> None:
    print("\nInequality check by pool size:")
    for _, row in inequality_summary.iterrows():
        print(
            f"pool_size={int(row.pool_size)}: "
            f"proposed < random sparse rate={row.proposed_lt_random_sparse_rate:.3f}, "
            f"proposed < random exhaustive rate={row.proposed_lt_random_exhaustive_rate:.3f}, "
            f"means proposed/random_sparse/random_exhaustive="
            f"{row.mean_proposed_total_tests:.2f}/"
            f"{row.mean_random_sparse_total_tests:.2f}/"
            f"{row.mean_random_exhaustive_total_tests:.2f}"
        )

    overall = summary.groupby("method", as_index=False).agg(
        mean_total_tests=("mean_total_tests", "mean"),
        min_mean_total_tests=("mean_total_tests", "min"),
    )
    print("\nOverall method means across pool sizes:")
    print(overall.to_string(index=False))

    print("\nBest pool size by method:")
    cols = [
        "method",
        "pool_size",
        "mean_total_tests",
        "median_total_tests",
        "std_total_tests",
        "mean_candidate_count",
        "mean_individual_tests",
    ]
    print(best_poolsize[cols].to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare proposed pooling matrix A against random baselines.")
    p.add_argument("--samples-root", type=Path, default=Path("openabm_sample_outputs/seed_1/samples"))
    p.add_argument("--sample-indices", nargs="+", default=["1:50"])
    p.add_argument("--pool-sizes", nargs="+", default=[str(x) for x in POOL_SIZE_CANDIDATES])
    p.add_argument("--results-dir", type=Path, default=Path("results/pooling_matrix_design_comparison"))
    p.add_argument("--method", default="symptom_count_plus_graph")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--random-seed-base", type=int, default=20260610)
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
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    results = run_experiment(args)
    summary, inequality, inequality_summary, best_poolsize = summarize(results, args.results_dir)
    if not args.no_plots:
        save_plots(results, summary, inequality_summary, args.results_dir)
    print_console_summary(summary, inequality_summary, best_poolsize)
    print("\nSaved outputs to", args.results_dir)


if __name__ == "__main__":
    main()
