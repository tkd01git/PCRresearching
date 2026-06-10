#!/usr/bin/env python3
"""Compare cluster-aware pooling matrix designs for weighted sparse reconstruction."""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import data_poolsizefitting as ds
import function_poolsizefittiing as fn
from run_openabm_poolsize_sweep import sample_dirs
from run_pooling_matrix_design_comparison import (
    METHOD_PROPOSED_SPARSE,
    METHOD_RANDOM_SPARSE,
    POOL_SIZE_CANDIDATES,
    append_row,
    evaluate_sparse,
    make_graph_aware_sparse_friendly_pools,
    make_random_pools,
    result_exists,
    with_pooling,
)
from run_poolsize_experiment import CONTACT_TYPE_WEIGHTS, parse_int_list


METHOD_CLUSTER_CUT = "cluster_cut_balanced_A_weighted_sparse_reconstruction"
METHOD_CLUSTER_ANCHOR = "cluster_anchor_A_weighted_sparse_reconstruction"
METHOD_ORDER = [METHOD_RANDOM_SPARSE, METHOD_PROPOSED_SPARSE, METHOD_CLUSTER_CUT, METHOD_CLUSTER_ANCHOR]
LABELS = {
    METHOD_RANDOM_SPARSE: "random A + sparse",
    METHOD_PROPOSED_SPARSE: "neighbor-dispersed A + sparse",
    METHOD_CLUSTER_CUT: "cluster-cut balanced A + sparse",
    METHOD_CLUSTER_ANCHOR: "cluster-anchor A + sparse",
}


def build_components(W: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    n = int(W.shape[0])
    W_bin = np.asarray(W, dtype=float) >= float(threshold)
    np.fill_diagonal(W_bin, False)
    labels = np.full(n, -1, dtype=int)
    comp_id = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        stack = [int(start)]
        labels[start] = comp_id
        while stack:
            u = stack.pop()
            for v in np.where(W_bin[u])[0].astype(int).tolist():
                if labels[v] < 0:
                    labels[v] = comp_id
                    stack.append(int(v))
        comp_id += 1
    return labels


def pools_to_matrix(pools: list[list[int]] | list[tuple[int, ...]], n: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    pool_tuples = [tuple(int(x) for x in pool) for pool in pools if pool]
    A = np.zeros((len(pool_tuples), int(n)), dtype=float)
    for row, pool in enumerate(pool_tuples):
        A[row, list(pool)] = 1.0
    return A, pool_tuples


def high_risk_mask(risk_score: np.ndarray) -> np.ndarray:
    positive = np.asarray(risk_score, dtype=float)
    threshold = float(np.quantile(positive, 0.90))
    nonzero = positive[positive > 0]
    if len(nonzero) and threshold <= 0:
        threshold = float(nonzero.min())
    return positive >= threshold


def cluster_interleaved_order(cluster_labels: np.ndarray, risk_score: np.ndarray) -> list[int]:
    members_by_cluster: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cluster_labels.astype(int)):
        members_by_cluster[int(c)].append(int(i))
    for c, members in members_by_cluster.items():
        members.sort(key=lambda i: (-float(risk_score[i]), int(i)))
    cluster_order = sorted(
        members_by_cluster,
        key=lambda c: (-sum(float(risk_score[i]) for i in members_by_cluster[c]), -len(members_by_cluster[c]), int(c)),
    )
    order = []
    cursor = {c: 0 for c in cluster_order}
    remaining = True
    while remaining:
        remaining = False
        for c in cluster_order:
            idx = cursor[c]
            if idx < len(members_by_cluster[c]):
                order.append(members_by_cluster[c][idx])
                cursor[c] += 1
                remaining = True
    return order


def make_cluster_cut_balanced_pools(
    W: np.ndarray,
    risk_score: np.ndarray,
    symptom_count: np.ndarray,
    pool_size: int,
    cluster_threshold: float = 1.0,
    same_cluster_soft_cap: int = 3,
) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray]:
    n = len(risk_score)
    pool_count = int(math.ceil(n / int(pool_size)))
    pools: list[list[int]] = [[] for _ in range(pool_count)]
    pool_sets: list[set[int]] = [set() for _ in range(pool_count)]
    cluster_counts: list[Counter] = [Counter() for _ in range(pool_count)]
    risk_sums = np.zeros(pool_count, dtype=float)
    high_counts = np.zeros(pool_count, dtype=int)
    high_mask = high_risk_mask(risk_score)
    target_risk = float(np.sum(risk_score) / max(pool_count, 1))
    target_high = float(np.sum(high_mask) / max(pool_count, 1))
    cluster_labels = build_components(W, threshold=cluster_threshold)
    adjacency = [set(np.where(np.asarray(W[i]) > 0)[0].astype(int).tolist()) for i in range(n)]

    for person in cluster_interleaved_order(cluster_labels, risk_score):
        c = int(cluster_labels[person])
        best_pool = None
        best_score = None
        for pool_idx, members in enumerate(pools):
            if len(members) >= pool_size:
                continue
            next_risk = risk_sums[pool_idx] + float(risk_score[person])
            next_high = high_counts[pool_idx] + int(high_mask[person])
            same_cluster_next = cluster_counts[pool_idx][c] + 1
            neighbor_count = len(adjacency[person].intersection(pool_sets[pool_idx]))
            size_balance = len(members) / max(pool_size, 1)
            score = (
                1.00 * abs(next_risk - target_risk) / max(target_risk, 1e-9)
                + 0.80 * abs(next_high - target_high) / max(target_high, 1.0)
                + 1.60 * max(0, same_cluster_next - same_cluster_soft_cap) ** 2
                + 0.18 * neighbor_count
                + 0.05 * size_balance
            )
            key = (score, len(members), pool_idx)
            if best_score is None or key < best_score:
                best_score = key
                best_pool = pool_idx
        if best_pool is None:
            raise RuntimeError("No pool with remaining capacity")
        pools[best_pool].append(int(person))
        pool_sets[best_pool].add(int(person))
        cluster_counts[best_pool][c] += 1
        risk_sums[best_pool] += float(risk_score[person])
        high_counts[best_pool] += int(high_mask[person])
    A, pool_tuples = pools_to_matrix(pools, n)
    return A, pool_tuples, cluster_labels


def make_cluster_anchor_pools(
    W: np.ndarray,
    risk_score: np.ndarray,
    symptom_count: np.ndarray,
    neighbor_symptom_score: np.ndarray,
    pool_size: int,
    cluster_threshold: float = 1.0,
    same_cluster_soft_cap: int = 3,
) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray]:
    n = len(risk_score)
    pool_count = int(math.ceil(n / int(pool_size)))
    pools: list[list[int]] = [[] for _ in range(pool_count)]
    pool_sets: list[set[int]] = [set() for _ in range(pool_count)]
    cluster_counts: list[Counter] = [Counter() for _ in range(pool_count)]
    cluster_high_counts: list[Counter] = [Counter() for _ in range(pool_count)]
    risk_sums = np.zeros(pool_count, dtype=float)
    high_counts = np.zeros(pool_count, dtype=int)
    high_mask = high_risk_mask(risk_score)
    target_risk = float(np.sum(risk_score) / max(pool_count, 1))
    target_high = float(np.sum(high_mask) / max(pool_count, 1))
    cluster_labels = build_components(W, threshold=cluster_threshold)
    adjacency = [set(np.where(np.asarray(W[i]) > 0)[0].astype(int).tolist()) for i in range(n)]

    low_symptom_graph_signal = (np.asarray(symptom_count) <= 1) & (np.asarray(neighbor_symptom_score) > 0)
    order = sorted(
        range(n),
        key=lambda i: (
            -int(high_mask[i]),
            -float(risk_score[i]),
            -int(low_symptom_graph_signal[i]),
            int(i),
        ),
    )
    for person in order:
        c = int(cluster_labels[person])
        best_pool = None
        best_score = None
        for pool_idx, members in enumerate(pools):
            if len(members) >= pool_size:
                continue
            next_risk = risk_sums[pool_idx] + float(risk_score[person])
            next_high = high_counts[pool_idx] + int(high_mask[person])
            same_cluster_next = cluster_counts[pool_idx][c] + 1
            neighbor_count = len(adjacency[person].intersection(pool_sets[pool_idx]))
            anchor_bonus = 0.0
            if low_symptom_graph_signal[person] and cluster_high_counts[pool_idx][c] > 0:
                anchor_bonus = min(0.75, 0.25 * cluster_high_counts[pool_idx][c])
            over_cluster_penalty = max(0, same_cluster_next - same_cluster_soft_cap) ** 2
            score = (
                0.85 * abs(next_risk - target_risk) / max(target_risk, 1e-9)
                + 0.65 * abs(next_high - target_high) / max(target_high, 1.0)
                + 1.15 * over_cluster_penalty
                + 0.10 * neighbor_count
                - anchor_bonus
                + 0.04 * len(members) / max(pool_size, 1)
            )
            key = (score, len(members), pool_idx)
            if best_score is None or key < best_score:
                best_score = key
                best_pool = pool_idx
        if best_pool is None:
            raise RuntimeError("No pool with remaining capacity")
        pools[best_pool].append(int(person))
        pool_sets[best_pool].add(int(person))
        cluster_counts[best_pool][c] += 1
        if high_mask[person]:
            cluster_high_counts[best_pool][c] += 1
        risk_sums[best_pool] += float(risk_score[person])
        high_counts[best_pool] += int(high_mask[person])
    A, pool_tuples = pools_to_matrix(pools, n)
    return A, pool_tuples, cluster_labels


def pool_structure_metrics(
    dataset: dict,
    risk_score: np.ndarray,
    symptom_count: np.ndarray,
    cluster_labels: np.ndarray,
    method: str,
    sample_id: int,
    pool_size: int,
) -> dict:
    high_mask = high_risk_mask(risk_score)
    W = np.asarray(dataset.get("W_type_weighted", dataset["W"]), dtype=float)
    W_bin = W > 0
    rows = []
    for pool_id, pool in enumerate(dataset["pools"]):
        arr = np.asarray(pool, dtype=int)
        cluster_counter = Counter(cluster_labels[arr].astype(int).tolist())
        internal_edges = 0
        if len(arr) > 1:
            internal_edges = int(np.triu(W_bin[np.ix_(arr, arr)], k=1).sum())
        rows.append({
            "method": method,
            "sample_id": int(sample_id),
            "pool_size": int(pool_size),
            "pool_id": int(pool_id),
            "actual_pool_size": int(len(arr)),
            "pool_risk_sum": float(risk_score[arr].sum()) if len(arr) else 0.0,
            "pool_average_risk": float(risk_score[arr].mean()) if len(arr) else 0.0,
            "pool_high_risk_count": int(high_mask[arr].sum()) if len(arr) else 0,
            "pool_average_symptom_count": float(symptom_count[arr].mean()) if len(arr) else 0.0,
            "pool_max_same_cluster_count": int(max(cluster_counter.values())) if cluster_counter else 0,
            "pool_internal_contact_edges": int(internal_edges),
        })
    df = pd.DataFrame(rows)
    return {
        "sample_id": int(sample_id),
        "pool_size": int(pool_size),
        "method": method,
        "mean_pool_risk_sum": float(df["pool_risk_sum"].mean()),
        "std_pool_risk_sum": float(df["pool_risk_sum"].std(ddof=0)),
        "mean_pool_high_risk_count": float(df["pool_high_risk_count"].mean()),
        "std_pool_high_risk_count": float(df["pool_high_risk_count"].std(ddof=0)),
        "mean_pool_max_same_cluster_count": float(df["pool_max_same_cluster_count"].mean()),
        "mean_pool_internal_contact_edges": float(df["pool_internal_contact_edges"].mean()),
    }


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.results_dir / "cluster_design_results_by_sample_poolsize_method.csv"
    structure_csv = args.results_dir / "cluster_design_structure_by_sample_poolsize_method.csv"
    existing = pd.read_csv(result_csv) if args.resume and result_csv.exists() else pd.DataFrame()
    existing_structure = pd.read_csv(structure_csv) if args.resume and structure_csv.exists() else pd.DataFrame()
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
        _, score_df, _ = ds.compute_prior_methods(
            base_dataset,
            beta_symptom=args.beta_symptom,
            graph_weight=args.graph_weight,
            graph_normalization=args.graph_normalization,
            clip_graph_score=args.clip_graph_score,
        )
        risk_score = score_df["combined_score"].to_numpy(dtype=float)
        symptom_count = score_df["symptom_count"].to_numpy(dtype=float)
        neighbor_symptom_score = score_df["neighbor_symptom_score"].to_numpy(dtype=float)
        W = np.asarray(base_dataset.get("W_type_weighted", base_dataset["W"]), dtype=float)

        for pool_size in pool_sizes:
            params = fn.derive_params({**base_dataset["params"], "n": len(population), "pool_size": int(pool_size)})
            pool_dataset_base = dict(base_dataset)
            pool_dataset_base["params"] = params
            random_seed = int(args.random_seed_base + sample_id * 1000 + pool_size)
            random_A, random_pools = make_random_pools(len(population), int(pool_size), random_seed)
            neighbor_A, neighbor_pools = make_graph_aware_sparse_friendly_pools(W, risk_score, int(pool_size))
            cut_A, cut_pools, cut_clusters = make_cluster_cut_balanced_pools(
                W,
                risk_score,
                symptom_count,
                int(pool_size),
                cluster_threshold=args.cluster_threshold,
                same_cluster_soft_cap=args.same_cluster_soft_cap,
            )
            anchor_A, anchor_pools, anchor_clusters = make_cluster_anchor_pools(
                W,
                risk_score,
                symptom_count,
                neighbor_symptom_score,
                int(pool_size),
                cluster_threshold=args.cluster_threshold,
                same_cluster_soft_cap=args.same_cluster_soft_cap,
            )
            designs = {
                METHOD_RANDOM_SPARSE: (random_A, random_pools, build_components(W, threshold=args.cluster_threshold), "random"),
                METHOD_PROPOSED_SPARSE: (neighbor_A, neighbor_pools, cut_clusters, "neighbor_dispersed"),
                METHOD_CLUSTER_CUT: (cut_A, cut_pools, cut_clusters, "cluster_cut_balanced"),
                METHOD_CLUSTER_ANCHOR: (anchor_A, anchor_pools, anchor_clusters, "cluster_anchor"),
            }
            for method, (A, pools, clusters, design_label) in designs.items():
                if args.resume and result_exists(existing, sample_id, pool_size, method):
                    continue
                dataset = with_pooling(pool_dataset_base, A, pools, design_label)
                row = evaluate_sparse(dataset, args, sample_id, int(pool_size), method)
                append_row(result_csv, row)
                if not (
                    args.resume
                    and not existing_structure.empty
                    and (
                        (existing_structure["sample_id"].astype(int) == int(sample_id))
                        & (existing_structure["pool_size"].astype(int) == int(pool_size))
                        & (existing_structure["method"].astype(str) == method)
                    ).any()
                ):
                    append_row(
                        structure_csv,
                        pool_structure_metrics(dataset, risk_score, symptom_count, clusters, method, sample_id, int(pool_size)),
                    )
    return pd.read_csv(result_csv), pd.read_csv(structure_csv)


def summarize(results: pd.DataFrame, structure: pd.DataFrame, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_cols = ["sample_id", "pool_size", "total_tests", "candidate_count", "individual_tests", "true_positive_rank"]
    for col in numeric_cols:
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
            mean_true_positive_rank=("true_positive_rank", "mean"),
            mean_positive_pools=("number_of_positive_pools", "mean"),
            mean_positive_count_per_positive_pool=("average_positive_count_per_positive_pool", "mean"),
            mean_max_positive_count_per_positive_pool=("max_positive_count_per_positive_pool", "mean"),
        )
        .sort_values(["pool_size", "method"])
    )
    wide = results.pivot_table(index=["sample_id", "pool_size"], columns="method", values="total_tests", aggfunc="first").reset_index()
    inequality = pd.DataFrame({
        "sample_id": wide["sample_id"].astype(int),
        "pool_size": wide["pool_size"].astype(int),
        "random_sparse_total_tests": wide[METHOD_RANDOM_SPARSE].astype(int),
        "neighbor_dispersed_total_tests": wide[METHOD_PROPOSED_SPARSE].astype(int),
        "cluster_cut_total_tests": wide[METHOD_CLUSTER_CUT].astype(int),
        "cluster_anchor_total_tests": wide[METHOD_CLUSTER_ANCHOR].astype(int),
    })
    for col in ["neighbor_dispersed_total_tests", "cluster_cut_total_tests", "cluster_anchor_total_tests"]:
        prefix = col.replace("_total_tests", "")
        inequality[f"{prefix}_lt_random_sparse"] = inequality[col] < inequality["random_sparse_total_tests"]
    inequality_summary = (
        inequality.groupby("pool_size", as_index=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            neighbor_dispersed_lt_random_sparse_rate=("neighbor_dispersed_lt_random_sparse", "mean"),
            cluster_cut_lt_random_sparse_rate=("cluster_cut_lt_random_sparse", "mean"),
            cluster_anchor_lt_random_sparse_rate=("cluster_anchor_lt_random_sparse", "mean"),
            mean_random_sparse_total_tests=("random_sparse_total_tests", "mean"),
            mean_neighbor_dispersed_total_tests=("neighbor_dispersed_total_tests", "mean"),
            mean_cluster_cut_total_tests=("cluster_cut_total_tests", "mean"),
            mean_cluster_anchor_total_tests=("cluster_anchor_total_tests", "mean"),
        )
        .sort_values("pool_size")
    )
    best_poolsize = summary.sort_values(["method", "mean_total_tests", "pool_size"]).groupby("method", as_index=False).first()
    structure_summary = (
        structure.groupby(["pool_size", "method"], as_index=False)
        .agg(
            mean_pool_risk_sum_sd=("std_pool_risk_sum", "mean"),
            mean_pool_high_risk_count_sd=("std_pool_high_risk_count", "mean"),
            mean_pool_max_same_cluster_count=("mean_pool_max_same_cluster_count", "mean"),
            mean_pool_internal_contact_edges=("mean_pool_internal_contact_edges", "mean"),
        )
        .sort_values(["pool_size", "method"])
    )
    results.to_csv(results_dir / "cluster_design_results_by_sample_poolsize_method.csv", index=False)
    summary.to_csv(results_dir / "cluster_design_summary_by_poolsize_method.csv", index=False)
    inequality.to_csv(results_dir / "cluster_design_inequality_vs_random_sparse.csv", index=False)
    inequality_summary.to_csv(results_dir / "cluster_design_inequality_summary_by_poolsize.csv", index=False)
    best_poolsize.to_csv(results_dir / "cluster_design_best_poolsize_summary.csv", index=False)
    structure_summary.to_csv(results_dir / "cluster_design_structure_summary.csv", index=False)
    return summary, inequality_summary, best_poolsize


def save_plots(summary: pd.DataFrame, inequality_summary: pd.DataFrame, results_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("pool_size")
        ax.plot(sub["pool_size"], sub["mean_total_tests"], marker="o", linewidth=1.8, label=LABELS[method])
    ax.set_xlabel("Pool size")
    ax.set_ylabel("Mean total tests")
    ax.set_title("Cluster-aware A design: total tests")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(results_dir / "cluster_design_total_tests_by_poolsize_method.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(inequality_summary["pool_size"], inequality_summary["neighbor_dispersed_lt_random_sparse_rate"], marker="o", label="neighbor-dispersed < random")
    ax.plot(inequality_summary["pool_size"], inequality_summary["cluster_cut_lt_random_sparse_rate"], marker="o", label="cluster-cut < random")
    ax.plot(inequality_summary["pool_size"], inequality_summary["cluster_anchor_lt_random_sparse_rate"], marker="o", label="cluster-anchor < random")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Pool size")
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate against random A + weighted sparse")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(results_dir / "cluster_design_success_rate_vs_random_sparse.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for metric, filename, ylabel in [
        ("mean_candidate_count", "cluster_design_candidate_count_by_method.png", "Mean candidate count"),
        ("mean_true_positive_rank", "cluster_design_true_positive_rank_by_method.png", "Mean true positive rank"),
        ("mean_positive_count_per_positive_pool", "cluster_design_positive_count_per_positive_pool.png", "Positive count per positive pool"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 7))
        for method in METHOD_ORDER:
            sub = summary[summary["method"] == method].sort_values("pool_size")
            ax.plot(sub["pool_size"], sub[metric], marker="o", linewidth=1.8, label=LABELS[method])
        ax.set_xlabel("Pool size")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(results_dir / filename, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def print_summary(summary: pd.DataFrame, inequality_summary: pd.DataFrame, best_poolsize: pd.DataFrame) -> None:
    print("\nMean total tests by pool size:")
    cols = [
        "pool_size",
        "mean_random_sparse_total_tests",
        "mean_neighbor_dispersed_total_tests",
        "mean_cluster_cut_total_tests",
        "mean_cluster_anchor_total_tests",
        "cluster_cut_lt_random_sparse_rate",
        "cluster_anchor_lt_random_sparse_rate",
    ]
    print(inequality_summary[cols].to_string(index=False))
    print("\nOverall method means:")
    print(summary.groupby("method")["mean_total_tests"].mean().sort_values().to_string())
    print("\nBest pool size by method:")
    print(
        best_poolsize[
            ["method", "pool_size", "mean_total_tests", "median_total_tests", "std_total_tests", "mean_candidate_count", "mean_true_positive_rank"]
        ].to_string(index=False)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run cluster-aware pooling A design comparisons.")
    p.add_argument("--samples-root", type=Path, default=Path("openabm_sample_outputs/seed_1/samples"))
    p.add_argument("--sample-indices", nargs="+", default=["1:50"])
    p.add_argument("--pool-sizes", nargs="+", default=[str(x) for x in POOL_SIZE_CANDIDATES])
    p.add_argument("--results-dir", type=Path, default=Path("results/cluster_pooling_design_experiment"))
    p.add_argument("--method", default="symptom_count_plus_graph")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--random-seed-base", type=int, default=20260610)
    p.add_argument("--cluster-threshold", type=float, default=1.0)
    p.add_argument("--same-cluster-soft-cap", type=int, default=3)
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
    results, structure = run_experiment(args)
    summary, inequality_summary, best_poolsize = summarize(results, structure, args.results_dir)
    if not args.no_plots:
        save_plots(summary, inequality_summary, args.results_dir)
    print_summary(summary, inequality_summary, best_poolsize)
    print("\nSaved cluster design experiment to", args.results_dir)


if __name__ == "__main__":
    main()
