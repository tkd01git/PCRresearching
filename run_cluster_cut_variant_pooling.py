#!/usr/bin/env python3
"""Evaluate cluster-cut pooling variants against random sparse pooling.

This focused runner fixes the pool size, varies the same-cluster soft cap, and
checks whether each cluster-cut design reduces sparse-reconstruction test cost
against a random pooling baseline on the 50 committed OpenABM samples.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data_poolsizefitting as ds
import function_poolsizefittiing as fn
from run_cluster_pooling_design_experiment import (
    make_cluster_cut_balanced_pools,
    pool_structure_metrics,
    pools_to_matrix,
)
from run_openabm_poolsize_sweep import sample_dirs
from run_pooling_matrix_design_comparison import (
    METHOD_RANDOM_SPARSE,
    append_row,
    evaluate_sparse,
    make_random_pools,
    result_exists,
    with_pooling,
)
from run_poolsize_experiment import CONTACT_TYPE_WEIGHTS, parse_int_list


def method_name_for_cap(cap: int) -> str:
    return f"cluster_cut_softcap_{int(cap)}_weighted_sparse_reconstruction"


METHOD_SYMPTOM_GRAPH_STRATIFIED = "symptom_graph_stratified_A_weighted_sparse_reconstruction"


def symptom_graph_personal_blocked_method_name(risk_pool_count: int) -> str:
    return f"symptom_graph_personal_blocked_riskpools_{int(risk_pool_count)}_A_weighted_sparse_reconstruction"


def symptom_graph_groups(symptom_count: np.ndarray, W: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Group people by own symptoms and graph-derived symptom exposure."""
    s = np.asarray(symptom_count, dtype=float)
    W_arr = np.asarray(W, dtype=float)
    g = W_arr @ s
    groups = {
        "A": np.where((s > 0) & (g > 0))[0].astype(int),
        "B": np.where((s > 0) & (g == 0))[0].astype(int),
        "C": np.where((s == 0) & (g > 0))[0].astype(int),
        "D": np.where((s == 0) & (g == 0))[0].astype(int),
    }
    return groups, g


def make_symptom_graph_stratified_pools(
    W: np.ndarray,
    symptom_count: np.ndarray,
    pool_size: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple[int, ...]], dict[str, int]]:
    n = len(symptom_count)
    pool_count = int(np.ceil(n / int(pool_size)))
    pools: list[list[int]] = [[] for _ in range(pool_count)]
    groups, _ = symptom_graph_groups(symptom_count, W)
    rng = np.random.default_rng(int(seed))

    cursor = 0
    for label in ["A", "B", "C", "D"]:
        members = groups[label].copy()
        rng.shuffle(members)
        for person in members.astype(int).tolist():
            placed = False
            for offset in range(pool_count):
                pool_idx = (cursor + offset) % pool_count
                if len(pools[pool_idx]) < int(pool_size):
                    pools[pool_idx].append(int(person))
                    cursor = (pool_idx + 1) % pool_count
                    placed = True
                    break
            if not placed:
                raise RuntimeError("No pool with remaining capacity")

    A, pool_tuples = pools_to_matrix(pools, n)
    group_counts = {f"group_{label}_count": int(len(groups[label])) for label in ["A", "B", "C", "D"]}
    return A, pool_tuples, group_counts


def place_round_robin(
    pools: list[list[int]],
    people: np.ndarray,
    allowed_pool_indices: list[int],
    pool_size: int,
    cursor: int,
) -> int:
    if not allowed_pool_indices:
        raise ValueError("allowed_pool_indices must not be empty")
    for person in people.astype(int).tolist():
        placed = False
        for offset in range(len(allowed_pool_indices)):
            pos = (cursor + offset) % len(allowed_pool_indices)
            pool_idx = int(allowed_pool_indices[pos])
            if len(pools[pool_idx]) < int(pool_size):
                pools[pool_idx].append(int(person))
                cursor = (pos + 1) % len(allowed_pool_indices)
                placed = True
                break
        if not placed:
            raise RuntimeError("No pool with remaining capacity among allowed pools")
    return cursor


def make_symptom_graph_personal_blocked_pools(
    W: np.ndarray,
    symptom_count: np.ndarray,
    pool_size: int,
    risk_pool_count: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple[int, ...]], dict[str, int]]:
    n = len(symptom_count)
    pool_count = int(np.ceil(n / int(pool_size)))
    pools: list[list[int]] = [[] for _ in range(pool_count)]
    groups, _ = symptom_graph_groups(symptom_count, W)
    rng = np.random.default_rng(int(seed))

    personal_count = int(len(groups["A"]) + len(groups["B"]))
    min_risk_pool_count = int(np.ceil(personal_count / max(int(pool_size), 1)))
    effective_risk_pool_count = min(pool_count, max(int(risk_pool_count), min_risk_pool_count, 1))
    risk_pool_indices = list(range(effective_risk_pool_count))
    all_pool_indices = list(range(pool_count))

    risk_cursor = 0
    for label in ["A", "B"]:
        members = groups[label].copy()
        rng.shuffle(members)
        risk_cursor = place_round_robin(pools, members, risk_pool_indices, pool_size, risk_cursor)

    all_cursor = 0
    for label in ["C", "D"]:
        members = groups[label].copy()
        rng.shuffle(members)
        all_cursor = place_round_robin(pools, members, all_pool_indices, pool_size, all_cursor)

    A, pool_tuples = pools_to_matrix(pools, n)
    group_counts = {f"group_{label}_count": int(len(groups[label])) for label in ["A", "B", "C", "D"]}
    group_counts["risk_pool_count"] = int(effective_risk_pool_count)
    return A, pool_tuples, group_counts


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.results_dir / "cluster_cut_variant_results.csv"
    structure_csv = args.results_dir / "cluster_cut_variant_structure.csv"
    if not args.resume:
        for path in [result_csv, structure_csv]:
            if path.exists():
                path.unlink()
    existing = pd.read_csv(result_csv) if args.resume and result_csv.exists() else pd.DataFrame()
    existing_structure = pd.read_csv(structure_csv) if args.resume and structure_csv.exists() else pd.DataFrame()
    dirs = sample_dirs(args.samples_root, parse_int_list(args.sample_indices))
    caps = parse_int_list(args.same_cluster_soft_caps)
    personal_blocked_risk_pool_counts = parse_int_list(args.symptom_graph_personal_blocked_risk_pool_counts)

    for sample_dir in dirs:
        sample_id = int(sample_dir.name.split("_sample", 1)[1].split("_", 1)[0])
        population = pd.read_csv(sample_dir / "population.csv")
        contacts = pd.read_csv(sample_dir / "contacts.csv")
        print(f"sample={sample_id}: n={len(population)} contacts={len(contacts)}")

        base_params = fn.get_default_params(n=len(population), pool_size=int(args.pool_size))
        base_dataset = ds.build_analysis_dataset(
            population,
            contacts,
            params=base_params,
            pool_size=int(args.pool_size),
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
        _, score_df, _ = ds.compute_prior_methods(
            base_dataset,
            beta_symptom=args.beta_symptom,
            graph_weight=args.graph_weight,
            graph_normalization=args.graph_normalization,
            clip_graph_score=args.clip_graph_score,
        )
        risk_score = score_df["combined_score"].to_numpy(dtype=float)
        symptom_count = score_df["symptom_count"].to_numpy(dtype=float)
        W = np.asarray(base_dataset.get("W_type_weighted", base_dataset["W"]), dtype=float)
        params = fn.derive_params({**base_dataset["params"], "n": len(population), "pool_size": int(args.pool_size)})
        pool_dataset_base = dict(base_dataset)
        pool_dataset_base["params"] = params

        random_seed = int(args.random_seed_base + sample_id * 1000 + int(args.pool_size))
        empty_group_counts = {f"group_{label}_count": np.nan for label in ["A", "B", "C", "D"]}
        empty_group_counts["risk_pool_count"] = np.nan
        if not (args.resume and result_exists(existing, sample_id, int(args.pool_size), METHOD_RANDOM_SPARSE)):
            random_A, random_pools = make_random_pools(len(population), int(args.pool_size), random_seed)
            random_dataset = with_pooling(pool_dataset_base, random_A, random_pools, "random")
            row = evaluate_sparse(random_dataset, args, sample_id, int(args.pool_size), METHOD_RANDOM_SPARSE)
            row["variant"] = "random"
            row["same_cluster_soft_cap"] = np.nan
            row.update(empty_group_counts)
            append_row(result_csv, row)

        if not (args.resume and result_exists(existing, sample_id, int(args.pool_size), METHOD_SYMPTOM_GRAPH_STRATIFIED)):
            strat_A, strat_pools, group_counts = make_symptom_graph_stratified_pools(
                W,
                symptom_count,
                int(args.pool_size),
                seed=random_seed + 17,
            )
            strat_dataset = with_pooling(pool_dataset_base, strat_A, strat_pools, "symptom_graph_stratified")
            row = evaluate_sparse(strat_dataset, args, sample_id, int(args.pool_size), METHOD_SYMPTOM_GRAPH_STRATIFIED)
            row["variant"] = "symptom_graph_stratified"
            row["same_cluster_soft_cap"] = np.nan
            row.update(group_counts)
            row["risk_pool_count"] = np.nan
            append_row(result_csv, row)

        for risk_pool_count in personal_blocked_risk_pool_counts:
            personal_method = symptom_graph_personal_blocked_method_name(risk_pool_count)
            if args.resume and result_exists(existing, sample_id, int(args.pool_size), personal_method):
                continue
            personal_A, personal_pools, group_counts = make_symptom_graph_personal_blocked_pools(
                W,
                symptom_count,
                int(args.pool_size),
                int(risk_pool_count),
                seed=random_seed + 47 + int(risk_pool_count),
            )
            personal_dataset = with_pooling(
                pool_dataset_base,
                personal_A,
                personal_pools,
                f"symptom_graph_personal_blocked_riskpools_{int(risk_pool_count)}",
            )
            row = evaluate_sparse(personal_dataset, args, sample_id, int(args.pool_size), personal_method)
            row["variant"] = f"symptom_graph_personal_blocked_{int(risk_pool_count)}"
            row["same_cluster_soft_cap"] = np.nan
            row.update(group_counts)
            append_row(result_csv, row)

        for cap in caps:
            method = method_name_for_cap(cap)
            if args.resume and result_exists(existing, sample_id, int(args.pool_size), method):
                continue
            A, pools, clusters = make_cluster_cut_balanced_pools(
                W,
                risk_score,
                symptom_count,
                int(args.pool_size),
                cluster_threshold=float(args.cluster_threshold),
                same_cluster_soft_cap=int(cap),
            )
            dataset = with_pooling(pool_dataset_base, A, pools, f"cluster_cut_softcap_{int(cap)}")
            row = evaluate_sparse(dataset, args, sample_id, int(args.pool_size), method)
            row["variant"] = f"softcap_{int(cap)}"
            row["same_cluster_soft_cap"] = int(cap)
            row.update(empty_group_counts)
            append_row(result_csv, row)

            if not (
                args.resume
                and not existing_structure.empty
                and (
                    (existing_structure["sample_id"].astype(int) == int(sample_id))
                    & (existing_structure["pool_size"].astype(int) == int(args.pool_size))
                    & (existing_structure["method"].astype(str) == method)
                ).any()
            ):
                structure_row = pool_structure_metrics(
                    dataset,
                    risk_score,
                    symptom_count,
                    clusters,
                    method,
                    sample_id,
                    int(args.pool_size),
                )
                structure_row["variant"] = f"softcap_{int(cap)}"
                structure_row["same_cluster_soft_cap"] = int(cap)
                append_row(structure_csv, structure_row)

    return pd.read_csv(result_csv), pd.read_csv(structure_csv)


def summarize(results: pd.DataFrame, structure: pd.DataFrame, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in [
        "sample_id",
        "pool_size",
        "total_tests",
        "candidate_count",
        "individual_tests",
        "detected_positive_count",
        "false_negative",
        "false_positive",
        "true_positive_rank",
        "group_A_count",
        "group_B_count",
        "group_C_count",
        "group_D_count",
        "risk_pool_count",
    ]:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col])
    summary = (
        results.groupby(["variant", "method"], as_index=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            mean_total_tests=("total_tests", "mean"),
            median_total_tests=("total_tests", "median"),
            std_total_tests=("total_tests", "std"),
            mean_candidate_count=("candidate_count", "mean"),
            mean_individual_tests=("individual_tests", "mean"),
            mean_false_negative=("false_negative", "mean"),
            mean_false_positive=("false_positive", "mean"),
            mean_true_positive_rank=("true_positive_rank", "mean"),
            mean_group_A_count=("group_A_count", "mean"),
            mean_group_B_count=("group_B_count", "mean"),
            mean_group_C_count=("group_C_count", "mean"),
            mean_group_D_count=("group_D_count", "mean"),
            mean_risk_pool_count=("risk_pool_count", "mean"),
        )
        .sort_values(["mean_total_tests", "variant"])
    )

    wide = results.pivot_table(index="sample_id", columns="variant", values="total_tests", aggfunc="first").reset_index()
    random_tests = wide["random"]
    comparisons = []
    for variant in [c for c in wide.columns if c not in {"sample_id", "random"}]:
        diff = wide[variant] - random_tests
        comparisons.append({
            "variant": variant,
            "n_samples": int(diff.notna().sum()),
            "mean_total_tests": float(wide[variant].mean()),
            "mean_delta_vs_random": float(diff.mean()),
            "median_delta_vs_random": float(diff.median()),
            "lt_random_rate": float((diff < 0).mean()),
            "tie_random_rate": float((diff == 0).mean()),
            "gt_random_rate": float((diff > 0).mean()),
        })
    comparison = pd.DataFrame(comparisons).sort_values(["mean_total_tests", "variant"])

    results.to_csv(results_dir / "cluster_cut_variant_results.csv", index=False)
    structure.to_csv(results_dir / "cluster_cut_variant_structure.csv", index=False)
    summary.to_csv(results_dir / "cluster_cut_variant_summary.csv", index=False)
    comparison.to_csv(results_dir / "cluster_cut_variant_vs_random.csv", index=False)
    return summary, comparison


def print_summary(summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    print("\nSummary by variant:")
    print(summary[[
        "variant",
        "n_samples",
        "mean_total_tests",
        "median_total_tests",
        "mean_individual_tests",
        "mean_candidate_count",
        "mean_false_negative",
        "mean_false_positive",
    ]].to_string(index=False))
    print("\nProposed variants vs random:")
    print(comparison.to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run cluster-cut variant comparisons at a fixed pool size.")
    p.add_argument("--samples-root", type=Path, default=Path("openabm_sample_outputs/seed_1/samples"))
    p.add_argument("--sample-indices", nargs="+", default=["1:50"])
    p.add_argument("--pool-size", type=int, default=80)
    p.add_argument("--same-cluster-soft-caps", nargs="+", default=["1", "2", "3", "5", "8"])
    p.add_argument("--symptom-graph-personal-blocked-risk-pool-counts", nargs="+", default=[])
    p.add_argument("--results-dir", type=Path, default=Path("results/cluster_cut_variant_pooling"))
    p.add_argument("--method", default="symptom_count_plus_graph")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--random-seed-base", type=int, default=20260610)
    p.add_argument("--cluster-threshold", type=float, default=1.0)
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
    results, structure = run_experiment(args)
    summary, comparison = summarize(results, structure, args.results_dir)
    print_summary(summary, comparison)
    print("\nSaved outputs to", args.results_dir)


if __name__ == "__main__":
    main()
