#!/usr/bin/env python3
"""Run OpenABM-Covid19 and export reusable seed-wise sample CSVs.

This script is intentionally separated from run_poolsize_experiment.py:

1. Clone/build OpenABM-Covid19 when needed.
2. Run OpenABM for each requested seed.
3. Convert raw OpenABM output into population_all.csv and contacts_all.csv.
4. Extract the company/workplace sample files used by the pool-size experiment.

The output directories are designed to be committed to this repository, while
the OpenABM build directory can be ignored.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

import numpy as np
import pandas as pd

from data_poolsizefitting import (
    convert_raw_openabm_to_population_contacts,
    export_graph_samples,
    _sample_company_workplace_once,
)
from run_poolsize_experiment import parse_int_list


OPENABM_REPO_URL = "https://github.com/BDI-pathogens/OpenABM-Covid19.git"
GSL_SOURCE_URL = "https://ftp.gnu.org/gnu/gsl/gsl-{version}.tar.gz"


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def ensure_local_gsl(gsl_prefix: Path, work_dir: Path, version: str) -> Path:
    """Build GSL locally when gsl-config is not already available."""
    gsl_prefix = Path(gsl_prefix).resolve()
    gsl_config = gsl_prefix / "bin/gsl-config"
    if gsl_config.exists():
        return gsl_prefix

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    tar_path = work_dir / f"gsl-{version}.tar.gz"
    src_dir = work_dir / f"gsl-{version}"
    if not tar_path.exists():
        url = GSL_SOURCE_URL.format(version=version)
        print(f"downloading {url} -> {tar_path}")
        urllib.request.urlretrieve(url, tar_path)
    if not src_dir.exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(work_dir)

    run_cmd(["./configure", f"--prefix={gsl_prefix}"], cwd=src_dir)
    run_cmd(["make", f"-j{max(1, os.cpu_count() or 1)}"], cwd=src_dir)
    run_cmd(["make", "install"], cwd=src_dir)
    return gsl_prefix


def build_openabm(repo_dir: Path, gsl_prefix: Path | None = None) -> None:
    env = os.environ.copy()
    make_cmd = ["make", "all"]
    if gsl_prefix is not None:
        gsl_prefix = Path(gsl_prefix)
        env["PATH"] = f"{gsl_prefix / 'bin'}:{env.get('PATH', '')}"
        make_cmd += [f"INC={gsl_prefix / 'include'}", f"LIB={gsl_prefix / 'lib'}"]
    run_cmd(make_cmd, cwd=repo_dir, env=env)


def ensure_openabm_repo(
    repo_dir: Path,
    skip_build: bool = False,
    gsl_prefix: Path | None = None,
) -> Path:
    """Clone and build OpenABM-Covid19 if the executable is not available."""
    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        run_cmd(["git", "clone", OPENABM_REPO_URL, str(repo_dir)])

    exe_candidates = [
        repo_dir / "src/covid19ibm.exe",
        repo_dir / "src/COVID19",
        repo_dir / "src/covid19",
    ]
    if skip_build and any(p.exists() for p in exe_candidates):
        return repo_dir

    build_openabm(repo_dir, gsl_prefix=gsl_prefix)
    return repo_dir


def patch_openabm_parameters(
    repo_dir: Path,
    raw_output_dir: Path,
    seed: int,
    end_time: int,
    n_total: int | None,
) -> Path:
    """Create a per-run OpenABM parameter file.

    OpenABM releases differ a bit in their parameter names. This function sets
    end_time and common seed/population-size fields only when those fields exist.
    """
    base_param = repo_dir / "tests/data/baseline_parameters.csv"
    if not base_param.exists():
        raise FileNotFoundError(f"OpenABM baseline parameter file not found: {base_param}")

    raw_output_dir.mkdir(parents=True, exist_ok=True)
    params = pd.read_csv(base_param)
    updates: dict[str, int] = {}

    if "end_time" in params.columns:
        updates["end_time"] = int(end_time)
    for col in ["rng_seed", "random_seed", "seed"]:
        if col in params.columns:
            updates[col] = int(seed)
    if n_total is not None:
        for col in ["n_total", "n_total_people", "population_size", "n_people"]:
            if col in params.columns:
                updates[col] = int(n_total)

    for col, value in updates.items():
        params.loc[:, col] = value

    param_path = raw_output_dir / "parameters.csv"
    params.to_csv(param_path, index=False)
    print(f"seed={seed}: parameter overrides {updates if updates else '{}'}")
    return param_path


def find_openabm_executable(repo_dir: Path) -> Path:
    candidates = [
        repo_dir / "src/covid19ibm.exe",
        repo_dir / "src/COVID19",
        repo_dir / "src/covid19",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"OpenABM executable not found under {repo_dir}")


def run_openabm_for_seed(
    repo_dir: Path,
    seed: int,
    raw_output_dir: Path,
    end_time: int,
    n_total: int | None,
) -> Path:
    if raw_output_dir.exists():
        shutil.rmtree(raw_output_dir)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    param_path = patch_openabm_parameters(
        repo_dir=repo_dir,
        raw_output_dir=raw_output_dir,
        seed=seed,
        end_time=end_time,
        n_total=n_total,
    )
    household_path = repo_dir / "tests/data/baseline_household_demographics.csv"
    if not household_path.exists():
        raise FileNotFoundError(f"OpenABM household demographics file not found: {household_path}")

    exe = find_openabm_executable(repo_dir)
    run_cmd([str(exe), str(param_path), "1", str(raw_output_dir), str(household_path)])
    return raw_output_dir


def prepare_seed_sample(args: argparse.Namespace, repo_dir: Path, seed: int) -> Path:
    seed_root = args.output_dir / f"seed_{seed}"
    raw_dir = seed_root / "raw_openabm"
    population_csv = seed_root / "population_all.csv"
    contacts_csv = seed_root / "contacts_all.csv"
    sample_dir = seed_root / "samples" / args.sample_name.format(n=args.sample_size, sample_index=1)

    if args.reuse_converted and population_csv.exists() and contacts_csv.exists():
        print(f"seed={seed}: reused converted CSVs from {seed_root}")
        population_all = pd.read_csv(population_csv)
        contacts_all = pd.read_csv(contacts_csv)
    else:
        run_openabm_for_seed(
            repo_dir=repo_dir,
            seed=seed,
            raw_output_dir=raw_dir,
            end_time=args.end_time,
            n_total=args.n_total,
        )
        population_all, contacts_all = convert_raw_openabm_to_population_contacts(
            output_dir=raw_dir,
            target_day=args.end_time,
            contact_window=args.contact_window,
            random_seed=seed,
        )
        seed_root.mkdir(parents=True, exist_ok=True)
        population_all.to_csv(population_csv, index=False)
        contacts_all.to_csv(contacts_csv, index=False)
        print(f"seed={seed}: saved converted CSVs to {seed_root}")

    if args.samples_per_seed == 1 and args.reuse_samples and (sample_dir / "population.csv").exists() and (sample_dir / "contacts.csv").exists():
        print(f"seed={seed}: reused sample from {sample_dir}")
    elif args.samples_per_seed == 1:
        export_single_sample(args, population_all, contacts_all, seed_root, seed)
    else:
        export_nonoverlapping_samples(args, population_all, contacts_all, seed_root, seed)

    if args.delete_raw_after_convert and raw_dir.exists():
        shutil.rmtree(raw_dir)
        print(f"seed={seed}: deleted raw OpenABM output from {raw_dir}")

    return seed_root


def export_single_sample(
    args: argparse.Namespace,
    population_all: pd.DataFrame,
    contacts_all: pd.DataFrame,
    seed_root: Path,
    seed: int,
) -> pd.DataFrame:
    summary = export_graph_samples(
        population_all=population_all,
        contacts_all=contacts_all,
        output_dir=seed_root,
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
        sample_name=args.sample_name,
    )
    summary.to_csv(seed_root / "generation_summary.csv", index=False)
    sample_dir = seed_root / "samples" / args.sample_name.format(n=args.sample_size, sample_index=1)
    print(f"seed={seed}: exported sample to {sample_dir}")
    return summary


def sample_conditions_met(args: argparse.Namespace, diag: dict) -> bool:
    return (
        diag["positive_rate"] <= float(args.max_positive_rate) + 1e-12
        and diag["positive_count"] >= int(args.min_positive_count)
        and diag["positive_component_count"] >= int(args.min_positive_components)
        and diag["isolated_positive_count"] >= int(args.min_isolated_positive)
        and diag["largest_positive_component_ratio"] <= float(args.max_largest_positive_component_ratio)
        and diag["positive_neighbor_ratio"] >= float(args.min_positive_neighbor_ratio)
    )


def export_nonoverlapping_samples(
    args: argparse.Namespace,
    population_all: pd.DataFrame,
    contacts_all: pd.DataFrame,
    seed_root: Path,
    seed: int,
) -> pd.DataFrame:
    samples_dir = seed_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    used_person_ids: set[int] = set()
    rows = []
    rng_seed_base = int(seed) * 100_000

    for sample_index in range(1, int(args.samples_per_seed) + 1):
        sample_name = args.multi_sample_name.format(n=args.sample_size, sample_index=sample_index)
        sample_dir = samples_dir / sample_name

        if args.reuse_samples and (sample_dir / "population.csv").exists() and (sample_dir / "contacts.csv").exists():
            sample = pd.read_csv(sample_dir / "population.csv")
            row = pd.read_csv(sample_dir / "sample_summary.csv").iloc[0].to_dict()
            sample_ids = set(sample["person_id"].astype(int).tolist())
            overlaps_existing = bool(sample_ids & used_person_ids)
            meets_conditions = (not args.require_sampling_conditions) or bool(row.get("sampling_conditions_met", False))
            if meets_conditions and not overlaps_existing:
                used_person_ids.update(sample_ids)
                rows.append(row)
                print(f"seed={seed} sample={sample_index}: reused sample from {sample_dir}")
                continue
            reason = "overlapped previous accepted samples" if overlaps_existing else "did not meet conditions"
            print(f"seed={seed} sample={sample_index}: existing sample {reason}; drawing again")

        remaining_pop = population_all[~population_all["person_id"].astype(int).isin(used_person_ids)].copy()
        remaining_ids = set(remaining_pop["person_id"].astype(int).tolist())
        remaining_contacts = contacts_all[
            contacts_all["person_i"].astype(int).isin(remaining_ids)
            & contacts_all["person_j"].astype(int).isin(remaining_ids)
        ].copy()

        best = None
        best_score = -10**9
        for attempt in range(1, int(args.max_sample_attempts) + 1):
            rng = np.random.default_rng(rng_seed_base + sample_index * 10_000 + attempt)
            sample, con, diag, by_person = _sample_company_workplace_once(
                population_all=remaining_pop,
                contacts_all=remaining_contacts,
                n=args.sample_size,
                target_age_groups=args.target_age_groups,
                workplace_contact_types=args.workplace_contact_types,
                max_positive_rate=args.max_positive_rate,
                min_positive_count=args.min_positive_count,
                target_positive_count_range=(args.target_positive_count_min, args.target_positive_count_max),
                rng=rng,
            )
            ok = sample_conditions_met(args, diag)
            score = 0
            score += min(diag["positive_component_count"], int(args.min_positive_components)) * 10
            score += min(diag["isolated_positive_count"], int(args.min_isolated_positive)) * 10
            score += int(100 * min(diag["positive_neighbor_ratio"], float(args.min_positive_neighbor_ratio)))
            score -= int(100 * max(0.0, diag["largest_positive_component_ratio"] - float(args.max_largest_positive_component_ratio)))
            if score > best_score:
                best_score = score
                best = (sample, con, diag, by_person, attempt, ok)
            if ok:
                break

        if best is None:
            raise RuntimeError(f"seed={seed} sample={sample_index}: could not draw a sample")

        sample, con, diag, by_person, attempts_used, conditions_met = best
        if args.require_sampling_conditions and not conditions_met:
            raise RuntimeError(
                f"seed={seed} sample={sample_index}: no non-overlapping sample met conditions "
                f"after {args.max_sample_attempts} extraction seeds. "
                "Relax sampling thresholds or reduce samples-per-seed."
            )
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample.to_csv(sample_dir / "population.csv", index=False)
        con.to_csv(sample_dir / "contacts.csv", index=False)
        by_person.to_csv(sample_dir / "positive_neighbor_by_positive_person.csv", index=False)

        used_person_ids.update(sample["person_id"].astype(int).tolist())
        row = dict(diag)
        row.update({
            "seed": int(seed),
            "sample_index": int(sample_index),
            "path": str(sample_dir),
            "attempts_used": int(attempts_used),
            "sampling_conditions_met": bool(conditions_met),
            "target_age_groups": ";".join(map(str, args.target_age_groups)),
            "workplace_contact_types": ";".join(map(str, args.workplace_contact_types)),
            "max_positive_rate_condition": float(args.max_positive_rate),
            "min_positive_count_condition": int(args.min_positive_count),
            "nonoverlap_person_count_so_far": int(len(used_person_ids)),
        })
        pd.DataFrame([row]).to_csv(sample_dir / "sample_summary.csv", index=False)
        rows.append(row)
        print(f"seed={seed} sample={sample_index}: exported non-overlapping sample to {sample_dir}")

    summary = pd.DataFrame(rows)
    summary.to_csv(seed_root / "generation_summary.csv", index=False)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate reusable OpenABM sample CSVs for pool-size fitting.")
    parser.add_argument("--seeds", nargs="+", default=["1:10"], help="Seeds, e.g. '1:10' or '1 2 3'.")
    parser.add_argument("--openabm-dir", type=Path, default=Path("openabm_work/OpenABM-Covid19"))
    parser.add_argument("--gsl-prefix", type=Path, default=Path("openabm_work/gsl"))
    parser.add_argument("--gsl-version", default="2.8")
    parser.add_argument("--no-build-local-gsl", dest="build_local_gsl", action="store_false")
    parser.set_defaults(build_local_gsl=True)
    parser.add_argument("--output-dir", type=Path, default=Path("openabm_sample_outputs"))
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--sample-name", default="company_n{n}_maxpos5pct_work")
    parser.add_argument("--samples-per-seed", type=int, default=1, help="Number of non-overlapping samples to extract from each OpenABM run.")
    parser.add_argument("--multi-sample-name", default="company_n{n}_sample{sample_index:02d}_maxpos5pct_work")
    parser.add_argument("--max-sample-attempts", type=int, default=300)
    parser.add_argument("--end-time", type=int, default=40)
    parser.add_argument("--contact-window", type=int, default=7)
    parser.add_argument("--n-total", type=int, default=None, help="Best-effort population override when supported by OpenABM parameters.")
    parser.add_argument("--skip-build", action="store_true", help="Use an existing OpenABM executable without running make.")
    parser.add_argument("--reuse-converted", action="store_true", help="Reuse population_all.csv and contacts_all.csv if present.")
    parser.add_argument("--reuse-samples", action="store_true", help="Reuse sample population.csv and contacts.csv if present.")
    parser.add_argument("--delete-raw-after-convert", action="store_true", help="Delete raw OpenABM output after converted/sample CSVs are saved.")
    parser.add_argument("--allow-best-effort-samples", dest="require_sampling_conditions", action="store_false", help="Save the best sample even if diagnostics do not meet all thresholds.")
    parser.set_defaults(require_sampling_conditions=True)
    parser.add_argument("--target-age-groups", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--workplace-contact-types", nargs="+", type=int, default=[1])
    parser.add_argument("--max-positive-rate", type=float, default=0.05)
    parser.add_argument("--min-positive-count", type=int, default=30)
    parser.add_argument("--target-positive-count-min", type=int, default=50)
    parser.add_argument("--target-positive-count-max", type=int, default=120)
    parser.add_argument("--min-positive-components", type=int, default=3)
    parser.add_argument("--min-isolated-positive", type=int, default=0)
    parser.add_argument("--max-largest-positive-component-ratio", type=float, default=0.60)
    parser.add_argument("--min-positive-neighbor-ratio", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = parse_int_list(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("seeds:", seeds)
    print("openabm_dir:", args.openabm_dir)
    print("output_dir:", args.output_dir)
    print("sample_size:", args.sample_size)

    gsl_prefix = None
    if args.build_local_gsl:
        gsl_prefix = ensure_local_gsl(args.gsl_prefix, work_dir=args.openabm_dir.parent, version=args.gsl_version)

    repo_dir = ensure_openabm_repo(args.openabm_dir, skip_build=args.skip_build, gsl_prefix=gsl_prefix)
    completed = []
    for seed in seeds:
        seed_root = prepare_seed_sample(args, repo_dir=repo_dir, seed=seed)
        completed.append({"seed": int(seed), "path": str(seed_root)})

    manifest = pd.DataFrame(completed)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    print("saved:", args.output_dir / "manifest.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
