#!/usr/bin/env python3
"""Regenerate symptom columns for the 50 OpenABM samples with fixed rules."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import data_poolsizefitting as ds
import function_poolsizefittiing as fn
from run_openabm_poolsize_sweep import sample_dirs
from run_poolsize_experiment import CONTACT_TYPE_WEIGHTS, parse_int_list
from run_symptom_rule_check_sample import build_group_frame, summarize_sample


def output_sample_name(sample_index: int) -> str:
    return f"company_n3000_sample{int(sample_index):02d}_maxpos5pct_work"


def regenerate_one(args: argparse.Namespace, sample_dir: Path, sample_index: int) -> dict:
    population = pd.read_csv(sample_dir / "population.csv")
    contacts = pd.read_csv(sample_dir / "contacts.csv")
    params = fn.get_default_params(n=len(population), pool_size=int(args.pool_size))
    dataset = ds.build_analysis_dataset(
        population,
        contacts,
        params=params,
        pool_size=int(args.pool_size),
        cluster_symptom_strength=float(args.cluster_symptom_strength),
        seed=int(sample_index),
        force_symptom_regeneration=True,
        contact_type_weights=CONTACT_TYPE_WEIGHTS,
        strong_edge_threshold=float(args.c_edge_min_weight),
        neg_covid_leak_prob=float(args.neg_covid_leak_prob),
        noncovid_base_prob=float(args.noncovid_base_prob),
        pos_noncovid_prob=float(args.pos_noncovid_prob),
        target_positive_symptomatic_rate=float(args.target_positive_symptomatic_rate),
    )
    _, score_df, _ = ds.compute_prior_methods(
        dataset,
        beta_symptom=float(args.beta_symptom),
        graph_weight=float(args.graph_weight),
        graph_normalization=str(args.graph_normalization),
    )
    pop_with_groups = build_group_frame(dataset, score_df)
    summary, group_summary, positive_group_summary = summarize_sample(pop_with_groups, contacts, dataset)

    out_dir = args.output_root / output_sample_name(sample_index)
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists; pass --overwrite to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    pop_with_groups.to_csv(out_dir / "population.csv", index=False)
    contacts.to_csv(out_dir / "contacts.csv", index=False)
    score_df.to_csv(out_dir / "risk_scores.csv", index=False)
    summary.to_csv(out_dir / "sample_summary.csv", index=False)
    group_summary.to_csv(out_dir / "symptom_graph_group_summary.csv", index=False)
    positive_group_summary.to_csv(out_dir / "positive_group_summary.csv", index=False)

    row = summary.iloc[0].to_dict()
    row["sample_index"] = int(sample_index)
    row["sample_dir"] = str(out_dir)
    for _, g in group_summary.iterrows():
        label = str(g["symptom_graph_group"])
        row[f"group_{label}_count"] = int(g["group_count"])
        row[f"group_{label}_positive_count"] = int(g["positive_count"])
        row[f"group_{label}_positive_rate"] = float(g["positive_rate"])
    return row


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create symptom-regenerated copies of OpenABM samples.")
    p.add_argument("--samples-root", type=Path, default=Path("openabm_sample_outputs/seed_1/samples"))
    p.add_argument("--sample-indices", nargs="+", default=["1:50"])
    p.add_argument("--output-root", type=Path, default=Path("openabm_sample_outputs/seed_1/samples_asymptomatic_2of3"))
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
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    indices = parse_int_list(args.sample_indices)
    dirs = sample_dirs(args.samples_root, indices)

    rows = []
    for sample_index, sample_dir in zip(indices, dirs, strict=True):
        row = regenerate_one(args, sample_dir, int(sample_index))
        rows.append(row)
        print(
            f"sample={sample_index:02d}: positives={int(row['positive_count'])} "
            f"asym_pos_rate={row['asymptomatic_positive_rate']:.3f} "
            f"neg_sym_rate={row['negative_symptomatic_rate']:.3f} "
            f"A/B/C/D={int(row.get('group_A_count', 0))}/"
            f"{int(row.get('group_B_count', 0))}/"
            f"{int(row.get('group_C_count', 0))}/"
            f"{int(row.get('group_D_count', 0))}"
        )

    df = pd.DataFrame(rows).sort_values("sample_index")
    df.to_csv(args.output_root / "all_sample_symptom_summary.csv", index=False)
    numeric = df.select_dtypes(include=[np.number])
    summary = numeric.agg(["min", "mean", "max"]).reset_index().rename(columns={"index": "stat"})
    summary.to_csv(args.output_root / "aggregate_symptom_summary.csv", index=False)
    print("\nSaved 50 symptom-regenerated samples to", args.output_root)
    print("\nKey aggregate:")
    cols = [
        "positive_count",
        "positive_rate",
        "positive_symptomatic_rate",
        "asymptomatic_positive_rate",
        "negative_symptomatic_rate",
        "group_A_count",
        "group_B_count",
        "group_C_count",
        "group_D_count",
    ]
    print(summary[["stat"] + [c for c in cols if c in summary.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
