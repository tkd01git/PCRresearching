#!/usr/bin/env python3
"""Regenerate symptoms for one OpenABM sample and export diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data_poolsizefitting as ds
import function_poolsizefittiing as fn
from run_openabm_poolsize_sweep import sample_dirs
from run_poolsize_experiment import CONTACT_TYPE_WEIGHTS


def build_group_frame(dataset: dict, score_df: pd.DataFrame) -> pd.DataFrame:
    pop = dataset["patient_data"].copy()
    W = np.asarray(dataset.get("W_type_weighted", dataset["W"]), dtype=float)
    symptom_count = score_df["symptom_count"].to_numpy(dtype=float)
    graph_symptom_sum = W @ symptom_count
    groups = np.full(len(pop), "D", dtype=object)
    groups[(symptom_count > 0) & (graph_symptom_sum > 0)] = "A"
    groups[(symptom_count > 0) & (graph_symptom_sum == 0)] = "B"
    groups[(symptom_count == 0) & (graph_symptom_sum > 0)] = "C"
    pop["symptom_graph_group"] = groups
    pop["s_personal_symptom_count"] = symptom_count.astype(int)
    pop["g_graph_symptom_sum"] = graph_symptom_sum
    pop["combined_score"] = score_df["combined_score"].to_numpy(dtype=float)
    pop["internal_asymptomatic_positive"] = (
        (pop["y_true"].astype(int) == 1) & (pop["reported_total_symptom_count"].astype(int) == 0)
    ).astype(int)
    return pop


def summarize_sample(pop: pd.DataFrame, contacts: pd.DataFrame, dataset: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = pop["y_true"].astype(int)
    symptom_count = pop["reported_total_symptom_count"].astype(int)
    pos = y == 1
    neg = y == 0
    W = np.asarray(dataset.get("W_type_weighted", dataset["W"]), dtype=float)
    W_bin = W > 0
    pos_idx = np.where(pos.to_numpy())[0]
    pp_edges = int(np.triu(W_bin[np.ix_(pos_idx, pos_idx)], k=1).sum()) if len(pos_idx) else 0

    summary = pd.DataFrame([{
        "n": int(len(pop)),
        "positive_count": int(pos.sum()),
        "positive_rate": float(pos.mean()),
        "contact_edges": int(len(contacts)),
        "weighted_graph_edges": int(np.triu(W_bin, k=1).sum()),
        "positive_positive_edges": pp_edges,
        "positive_symptomatic_count": int((pos & (symptom_count > 0)).sum()),
        "positive_symptomatic_rate": float((symptom_count[pos] > 0).mean()) if pos.any() else 0.0,
        "asymptomatic_positive_count": int((pos & (symptom_count == 0)).sum()),
        "asymptomatic_positive_rate": float((symptom_count[pos] == 0).mean()) if pos.any() else 0.0,
        "negative_symptomatic_count": int((neg & (symptom_count > 0)).sum()),
        "negative_symptomatic_rate": float((symptom_count[neg] > 0).mean()) if neg.any() else 0.0,
        "mean_symptom_count_positive": float(symptom_count[pos].mean()) if pos.any() else 0.0,
        "mean_symptom_count_negative": float(symptom_count[neg].mean()) if neg.any() else 0.0,
    }])

    group_summary = (
        pop.groupby("symptom_graph_group", as_index=False)
        .agg(
            group_count=("person_id", "count"),
            positive_count=("y_true", "sum"),
            mean_symptom_count=("s_personal_symptom_count", "mean"),
            mean_graph_symptom_sum=("g_graph_symptom_sum", "mean"),
            mean_combined_score=("combined_score", "mean"),
        )
        .sort_values("symptom_graph_group")
    )
    group_summary["positive_rate"] = group_summary["positive_count"] / group_summary["group_count"]

    positive_group_summary = (
        pop[pos]
        .groupby("symptom_graph_group", as_index=False)
        .agg(
            positive_count=("person_id", "count"),
            asymptomatic_positive_count=("internal_asymptomatic_positive", "sum"),
            mean_graph_symptom_sum=("g_graph_symptom_sum", "mean"),
            mean_combined_score=("combined_score", "mean"),
        )
        .sort_values("symptom_graph_group")
    )
    positive_group_summary["positive_share"] = positive_group_summary["positive_count"] / max(int(pos.sum()), 1)
    positive_group_summary["asymptomatic_share_within_group"] = (
        positive_group_summary["asymptomatic_positive_count"] / positive_group_summary["positive_count"]
    )
    return summary, group_summary, positive_group_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Regenerate symptoms for one sample and export diagnostics.")
    p.add_argument("--samples-root", type=Path, default=Path("openabm_sample_outputs/seed_1/samples"))
    p.add_argument("--sample-index", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("results/symptom_rule_check_sample01"))
    p.add_argument("--pool-size", type=int, default=80)
    p.add_argument("--cluster-symptom-strength", type=float, default=0.70)
    p.add_argument("--target-positive-symptomatic-rate", type=float, default=0.33)
    p.add_argument("--neg-covid-leak-prob", type=float, default=0.01)
    p.add_argument("--noncovid-base-prob", type=float, default=0.10)
    p.add_argument("--pos-noncovid-prob", type=float, default=0.10)
    p.add_argument("--c-edge-min-weight", type=float, default=2.0)
    p.add_argument("--beta-symptom", type=float, default=1.0)
    p.add_argument("--graph-weight", type=float, default=1.0)
    p.add_argument("--graph-normalization", default="neighbor_symptom_sum_div_max_symptoms")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = sample_dirs(args.samples_root, [args.sample_index])[0]
    population = pd.read_csv(sample_dir / "population.csv")
    contacts = pd.read_csv(sample_dir / "contacts.csv")

    params = fn.get_default_params(n=len(population), pool_size=args.pool_size)
    dataset = ds.build_analysis_dataset(
        population,
        contacts,
        params=params,
        pool_size=args.pool_size,
        cluster_symptom_strength=args.cluster_symptom_strength,
        seed=int(args.sample_index),
        force_symptom_regeneration=True,
        contact_type_weights=CONTACT_TYPE_WEIGHTS,
        strong_edge_threshold=args.c_edge_min_weight,
        neg_covid_leak_prob=args.neg_covid_leak_prob,
        noncovid_base_prob=args.noncovid_base_prob,
        pos_noncovid_prob=args.pos_noncovid_prob,
        target_positive_symptomatic_rate=args.target_positive_symptomatic_rate,
    )
    _, score_df, _ = ds.compute_prior_methods(
        dataset,
        beta_symptom=args.beta_symptom,
        graph_weight=args.graph_weight,
        graph_normalization=args.graph_normalization,
    )
    pop_with_groups = build_group_frame(dataset, score_df)
    summary, group_summary, positive_group_summary = summarize_sample(pop_with_groups, contacts, dataset)

    pop_with_groups.to_csv(args.output_dir / "population_with_symptoms.csv", index=False)
    contacts.to_csv(args.output_dir / "contacts.csv", index=False)
    score_df.to_csv(args.output_dir / "risk_scores.csv", index=False)
    summary.to_csv(args.output_dir / "sample_summary.csv", index=False)
    group_summary.to_csv(args.output_dir / "symptom_graph_group_summary.csv", index=False)
    positive_group_summary.to_csv(args.output_dir / "positive_group_summary.csv", index=False)

    print("Saved symptom rule check to", args.output_dir)
    print("\nSample summary:")
    print(summary.to_string(index=False))
    print("\nGroup summary:")
    print(group_summary.to_string(index=False))
    print("\nPositive group summary:")
    print(positive_group_summary.to_string(index=False))


if __name__ == "__main__":
    main()
