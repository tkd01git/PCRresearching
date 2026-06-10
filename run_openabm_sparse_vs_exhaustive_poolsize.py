#!/usr/bin/env python3
"""Compare sparse reconstruction vs exhaustive follow-up across pool sizes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data_poolsizefitting as ds
import function_poolsizefittiing as fn
from run_openabm_poolsize_sweep import sample_dirs
from run_poolsize_experiment import CONTACT_TYPE_WEIGHTS, parse_int_list


DEFAULT_POOL_SIZES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30,
    50, 80, 100, 120, 140, 160, 180, 190, 200, 220, 250,
]
METHOD_EXHAUSTIVE = "two_stage_exhaustive_followup"
METHOD_SPARSE = "weighted_sparse_reconstruction"


def append_row(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def result_exists(existing: pd.DataFrame, sample_index: int, assignment_seed: int, pool_size: int, method: str) -> bool:
    if existing.empty:
        return False
    sub = existing[
        (existing["sample_index"].astype(int) == int(sample_index))
        & (existing["pool_assignment_seed"].astype(int) == int(assignment_seed))
        & (existing["pool_size"].astype(int) == int(pool_size))
        & (existing["method"].astype(str) == str(method))
        & (existing["status"].astype(str).str.startswith("ok"))
    ]
    return not sub.empty


def positive_pool_stats(dataset: dict) -> dict:
    params = dataset["params"]
    cutoff = float(params.get("positive_cutoff", params.get("x_min_positive", 1e3)))
    x_true = np.asarray(dataset["x_true"], dtype=float)
    is_positive = x_true >= float(params.get("x_min_positive", 1e3))
    positive_pool_indices = np.where(np.asarray(dataset["pooled_amount_est"], dtype=float) >= cutoff)[0]
    positive_counts = [int(is_positive[list(dataset["pools"][int(idx)])].sum()) for idx in positive_pool_indices]
    return {
        "positive_pool_count": int(len(positive_pool_indices)),
        "mean_positive_count_per_positive_pool": float(np.mean(positive_counts)) if positive_counts else 0.0,
        "max_positive_count_per_positive_pool": int(max(positive_counts)) if positive_counts else 0,
    }


def with_pooling(base_dataset: dict, pool_size: int, pool_order: np.ndarray) -> dict:
    out = dict(base_dataset)
    params = fn.derive_params({**base_dataset["params"], "n": len(base_dataset["x_true"]), "pool_size": int(pool_size)})
    A, pools = fn.make_pooling_matrix(
        len(base_dataset["x_true"]),
        pool_size=int(pool_size),
        gaps=params["gaps"],
        allow_incomplete_last_pool=True,
        pool_order=pool_order,
    )
    pooled_amount_true, pooled_ct, pooled_amount_est = fn.pooled_measurements_qpcr(
        A,
        np.asarray(base_dataset["x_true"], dtype=float),
        params,
    )
    out["params"] = params
    out["A"] = A
    out["pools"] = pools
    out["pooled_amount_true"] = pooled_amount_true
    out["pooled_ct"] = pooled_ct
    out["pooled_amount_est"] = pooled_amount_est
    out["pool_assignment_seed"] = None
    return out


def evaluate_methods(dataset: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    analysis_dataset = fn.restrict_dataset_to_positive_pools(dataset)
    stats = positive_pool_stats(dataset)
    initial_pool_tests = int(analysis_dataset.get("initial_pool_count", len(dataset["pools"])))
    candidate_count = int(len(analysis_dataset.get("x_true", [])))
    true_positive_count = int(
        (np.asarray(analysis_dataset.get("x_true", []), dtype=float) >= analysis_dataset["params"]["x_min_positive"]).sum()
    ) if candidate_count else 0
    candidate_positive_rate = true_positive_count / candidate_count if candidate_count else 0.0

    common = {
        "initial_pool_tests": initial_pool_tests,
        "positive_pool_count": stats["positive_pool_count"],
        "candidate_count": candidate_count,
        "true_positive_count": true_positive_count,
        "candidate_positive_rate": float(candidate_positive_rate),
        "mean_positive_count_per_positive_pool": stats["mean_positive_count_per_positive_pool"],
        "max_positive_count_per_positive_pool": stats["max_positive_count_per_positive_pool"],
    }
    exhaustive = {
        **common,
        "method": METHOD_EXHAUSTIVE,
        "individual_tests": candidate_count,
        "total_tests": initial_pool_tests + candidate_count,
        "status": "ok",
    }

    if candidate_count == 0:
        sparse = {
            **common,
            "method": METHOD_SPARSE,
            "individual_tests": 0,
            "total_tests": initial_pool_tests,
            "status": "ok",
        }
        return exhaustive, sparse

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
    sparse = {
        **common,
        "method": METHOD_SPARSE,
        "individual_tests": int(res["inspection_count"]),
        "total_tests": int(res["total_test_cost"]),
        "status": "ok",
    }
    return exhaustive, sparse


def run(args: argparse.Namespace) -> pd.DataFrame:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.results_dir / "sparse_vs_exhaustive_poolsize_results.csv"
    existing = pd.read_csv(result_csv) if args.resume and result_csv.exists() else pd.DataFrame()
    pool_sizes = parse_int_list(args.pool_sizes)
    assignment_seeds = parse_int_list(args.assignment_seeds)
    dirs = sample_dirs(args.samples_root, parse_int_list(args.sample_indices))

    for sample_dir in dirs:
        sample_index = int(sample_dir.name.split("_sample", 1)[1].split("_", 1)[0])
        population = pd.read_csv(sample_dir / "population.csv")
        contacts = pd.read_csv(sample_dir / "contacts.csv")
        print(f"sample={sample_index}: n={len(population)} contacts={len(contacts)}")
        base_params = fn.get_default_params(n=len(population), pool_size=1)
        base_dataset = ds.build_analysis_dataset(
            population,
            contacts,
            params=base_params,
            pool_size=1,
            cluster_symptom_strength=args.cluster_symptom_strength,
            seed=sample_index,
            force_symptom_regeneration=not args.no_force_symptom_regeneration,
            contact_type_weights=CONTACT_TYPE_WEIGHTS,
            strong_edge_threshold=args.c_edge_min_weight,
            neg_covid_leak_prob=args.neg_covid_leak_prob,
            noncovid_base_prob=args.noncovid_base_prob,
            pos_noncovid_prob=args.pos_noncovid_prob,
            target_positive_symptomatic_rate=args.target_positive_symptomatic_rate,
        )

        for assignment_seed in assignment_seeds:
            pool_order = np.random.default_rng(int(assignment_seed)).permutation(len(population))
            for pool_size in pool_sizes:
                if (
                    args.resume
                    and result_exists(existing, sample_index, assignment_seed, pool_size, METHOD_EXHAUSTIVE)
                    and result_exists(existing, sample_index, assignment_seed, pool_size, METHOD_SPARSE)
                ):
                    continue
                try:
                    dataset = with_pooling(base_dataset, int(pool_size), pool_order)
                    rows = evaluate_methods(dataset, args)
                except Exception as exc:
                    rows = (
                        {
                            "method": METHOD_EXHAUSTIVE,
                            "initial_pool_tests": np.nan,
                            "positive_pool_count": np.nan,
                            "candidate_count": np.nan,
                            "true_positive_count": np.nan,
                            "candidate_positive_rate": np.nan,
                            "individual_tests": np.nan,
                            "total_tests": np.nan,
                            "status": f"error: {type(exc).__name__}: {exc}",
                        },
                        {
                            "method": METHOD_SPARSE,
                            "initial_pool_tests": np.nan,
                            "positive_pool_count": np.nan,
                            "candidate_count": np.nan,
                            "true_positive_count": np.nan,
                            "candidate_positive_rate": np.nan,
                            "individual_tests": np.nan,
                            "total_tests": np.nan,
                            "status": f"error: {type(exc).__name__}: {exc}",
                        },
                    )
                    print(
                        f"sample={sample_index} assignment_seed={assignment_seed} "
                        f"pool_size={pool_size}: ERROR {type(exc).__name__}: {exc}"
                    )

                for row in rows:
                    row.update({
                        "sample_index": sample_index,
                        "sample_name": sample_dir.name,
                        "pool_assignment_seed": int(assignment_seed),
                        "pool_size": int(pool_size),
                    })
                    if args.resume and result_exists(existing, sample_index, assignment_seed, pool_size, row["method"]):
                        continue
                    append_row(result_csv, row)
    return pd.read_csv(result_csv)


def summarize(results: pd.DataFrame, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = results[results["status"].astype(str).str.startswith("ok")].copy()
    for col in ["sample_index", "pool_assignment_seed", "pool_size"]:
        ok[col] = ok[col].astype(int)
    for col in ["total_tests", "initial_pool_tests", "individual_tests", "candidate_count"]:
        ok[col] = pd.to_numeric(ok[col])

    summary = (
        ok.groupby(["method", "pool_size"], as_index=False)
        .agg(
            n_samples=("sample_index", "nunique"),
            n_assignment_seeds=("pool_assignment_seed", "nunique"),
            n_runs=("total_tests", "count"),
            mean_total_tests=("total_tests", "mean"),
            median_total_tests=("total_tests", "median"),
            std_total_tests=("total_tests", "std"),
            q25_total_tests=("total_tests", lambda s: s.quantile(0.25)),
            q75_total_tests=("total_tests", lambda s: s.quantile(0.75)),
            mean_initial_pool_tests=("initial_pool_tests", "mean"),
            mean_individual_tests=("individual_tests", "mean"),
            mean_candidate_count=("candidate_count", "mean"),
        )
        .sort_values(["method", "mean_total_tests", "pool_size"])
    )
    best = summary.groupby("method", as_index=False).first()
    ok.to_csv(results_dir / "sparse_vs_exhaustive_poolsize_results_ok.csv", index=False)
    summary.to_csv(results_dir / "sparse_vs_exhaustive_poolsize_summary.csv", index=False)
    best.to_csv(results_dir / "sparse_vs_exhaustive_best_poolsize.csv", index=False)
    return summary, best


def print_summary(summary: pd.DataFrame, best: pd.DataFrame) -> None:
    print("\nBest pool size by method:")
    print(best[["method", "pool_size", "mean_total_tests", "median_total_tests", "std_total_tests", "n_runs"]].to_string(index=False))
    print("\nTop 8 per method:")
    cols = ["method", "pool_size", "mean_total_tests", "median_total_tests", "mean_initial_pool_tests", "mean_individual_tests", "mean_candidate_count"]
    for method, sub in summary.groupby("method"):
        print(f"\n{method}")
        print(sub.sort_values(["mean_total_tests", "pool_size"])[cols].head(8).to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare sparse vs non-sparse pooling over samples, pool sizes, and assignment seeds.")
    p.add_argument("--samples-root", type=Path, default=Path("openabm_sample_outputs/seed_1/samples"))
    p.add_argument("--sample-indices", nargs="+", default=["1:50"])
    p.add_argument("--assignment-seeds", nargs="+", default=["1", "2", "3"])
    p.add_argument("--pool-sizes", nargs="+", default=[str(x) for x in DEFAULT_POOL_SIZES])
    p.add_argument("--results-dir", type=Path, default=Path("results/sparse_vs_exhaustive_poolsize"))
    p.add_argument("--method", default="symptom_count_plus_graph")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
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
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    results = run(args)
    summary, best = summarize(results, args.results_dir)
    print_summary(summary, best)
    print("\nSaved outputs to", args.results_dir)


if __name__ == "__main__":
    main()
