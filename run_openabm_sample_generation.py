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
    _positive_component_diagnostics,
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
        contact_start_day = int(args.contact_start_day if args.contact_start_day is not None else args.end_time - args.contact_window)
        contact_end_day = int(args.contact_end_day if args.contact_end_day is not None else args.end_time - 1)
        if contact_start_day > contact_end_day:
            raise ValueError(f"contact_start_day={contact_start_day} must be <= contact_end_day={contact_end_day}")

        target_raw_dir = raw_dir / f"end_time_{int(args.end_time)}"
        run_openabm_for_seed(
            repo_dir=repo_dir,
            seed=seed,
            raw_output_dir=target_raw_dir,
            end_time=args.end_time,
            n_total=args.n_total,
        )
        population_all, target_contacts = convert_raw_openabm_to_population_contacts(
            output_dir=target_raw_dir,
            target_day=args.end_time,
            contact_window=args.contact_window,
            contact_start_day=contact_end_day,
            contact_end_day=contact_end_day,
            random_seed=seed,
        )
        contact_parts = []
        if len(target_contacts):
            contact_parts.append(target_contacts)
        for contact_day in range(contact_start_day, contact_end_day):
            # OpenABM's interaction export is the final contact snapshot; an
            # end_time of contact_day + 1 yields contact_day interactions.
            contact_raw_dir = raw_dir / f"end_time_{contact_day + 1}"
            run_openabm_for_seed(
                repo_dir=repo_dir,
                seed=seed,
                raw_output_dir=contact_raw_dir,
                end_time=contact_day + 1,
                n_total=args.n_total,
            )
            _, day_contacts = convert_raw_openabm_to_population_contacts(
                output_dir=contact_raw_dir,
                target_day=contact_day,
                contact_window=1,
                contact_start_day=contact_day,
                contact_end_day=contact_day,
                random_seed=seed,
            )
            if len(day_contacts):
                contact_parts.append(day_contacts)
        contacts_all = pd.concat(contact_parts, ignore_index=True) if contact_parts else target_contacts.iloc[0:0].copy()
        contacts_all = contacts_all.drop_duplicates(["day", "person_i", "person_j", "contact_type"]).reset_index(drop=True)
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
        and diag["positive_count"] >= int(args.target_positive_count_min)
        and diag["positive_count"] <= int(args.target_positive_count_max)
        and diag["contact_edges"] >= int(args.min_contact_edges)
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
    if args.fast_workplace_nonoverlap:
        return export_nonoverlapping_samples_fast(args, population_all, contacts_all, seed_root, seed)

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
            score += min(diag["contact_edges"], int(args.min_contact_edges)) // 100
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
            "target_day": int(args.end_time),
            "contact_start_day": int(args.contact_start_day if args.contact_start_day is not None else args.end_time - args.contact_window),
            "contact_end_day": int(args.contact_end_day if args.contact_end_day is not None else args.end_time - 1),
            "max_positive_rate_condition": float(args.max_positive_rate),
            "min_positive_count_condition": int(args.min_positive_count),
            "min_contact_edges_condition": int(args.min_contact_edges),
            "nonoverlap_person_count_so_far": int(len(used_person_ids)),
        })
        pd.DataFrame([row]).to_csv(sample_dir / "sample_summary.csv", index=False)
        rows.append(row)
        print(f"seed={seed} sample={sample_index}: exported non-overlapping sample to {sample_dir}")

    summary = pd.DataFrame(rows)
    summary.to_csv(seed_root / "generation_summary.csv", index=False)
    return summary


def _connected_components_from_adj_local(adj: dict[int, set[int]], nodes: set[int] | None = None) -> list[list[int]]:
    if nodes is None:
        nodes = set(adj.keys())
    nodes = set(int(x) for x in nodes)
    seen = set()
    comps = []
    for seed in nodes:
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj.get(u, set()):
                if v in nodes and v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def _build_workplace_graph(
    population_all: pd.DataFrame,
    contacts_all: pd.DataFrame,
    target_age_groups: list[int],
    workplace_contact_types: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int], dict[int, set[int]], dict[int, int], list[list[int]]]:
    pop = population_all.copy()
    pop["person_id"] = pop["person_id"].astype(int)
    pop["age_group"] = pop["age_group"].astype(int)
    pop["y_true"] = pop["y_true"].astype(int)
    pop = pop[pop["age_group"].isin([int(x) for x in target_age_groups])].copy()
    allowed_ids = set(pop["person_id"].astype(int).tolist())

    con = contacts_all.copy()
    con["person_i"] = con["person_i"].astype(int)
    con["person_j"] = con["person_j"].astype(int)
    con["contact_type"] = con["contact_type"].astype(int)
    con = con[
        con["contact_type"].isin([int(x) for x in workplace_contact_types])
        & con["person_i"].isin(allowed_ids)
        & con["person_j"].isin(allowed_ids)
    ].copy()

    y_map = dict(zip(pop["person_id"].astype(int), pop["y_true"].astype(int)))
    adj: dict[int, set[int]] = {}
    degree: dict[int, int] = {}
    pp_adj: dict[int, set[int]] = {}
    for a, b in zip(con["person_i"], con["person_j"]):
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        adj.setdefault(ia, set()).add(ib)
        adj.setdefault(ib, set()).add(ia)
        degree[ia] = degree.get(ia, 0) + 1
        degree[ib] = degree.get(ib, 0) + 1
        if int(y_map.get(ia, 0)) == 1 and int(y_map.get(ib, 0)) == 1:
            pp_adj.setdefault(ia, set()).add(ib)
            pp_adj.setdefault(ib, set()).add(ia)

    pp_comps = [c for c in _connected_components_from_adj_local(pp_adj, set(pp_adj.keys())) if len(c) >= 2]
    pp_comps.sort(key=len, reverse=True)
    return pop, con, y_map, adj, degree, pp_comps


def _choose_seed_positive_ids(
    pp_comps: list[list[int]],
    used_person_ids: set[int],
    min_positive_count: int,
    target_positive_count_max: int,
    min_positive_components: int,
    max_largest_positive_component_ratio: float,
) -> set[int]:
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_component_count = 0
    largest_allowed = max(2, int(np.floor(int(target_positive_count_max) * float(max_largest_positive_component_ratio))))
    for comp in pp_comps:
        rem = [int(p) for p in comp if int(p) not in used_person_ids and int(p) not in selected_set]
        if len(rem) < 2:
            continue
        if len(rem) > largest_allowed:
            continue
        if len(selected) + len(rem) <= int(target_positive_count_max):
            selected.extend(rem)
            selected_set.update(rem)
            selected_component_count += 1
        elif len(selected) < int(min_positive_count):
            take = max(2, min(len(rem), int(target_positive_count_max) - len(selected)))
            selected.extend(rem[:take])
            selected_set.update(rem[:take])
            selected_component_count += 1
        if (
            len(selected) >= int(min_positive_count)
            and selected_component_count >= int(min_positive_components)
        ):
            break
    if selected_component_count < int(min_positive_components):
        return set()
    return set(selected)


def _expand_workplace_sample_fast(
    seed_positive_ids: set[int],
    used_person_ids: set[int],
    y_map: dict[int, int],
    adj: dict[int, set[int]],
    degree: dict[int, int],
    sample_size: int,
    max_positive_count: int,
    rng: np.random.Generator,
) -> tuple[set[int], dict]:
    selected = set(int(x) for x in seed_positive_ids)
    positive_count = sum(1 for x in selected if int(y_map.get(int(x), 0)) == 1)
    frontier = set(selected)
    rounds = 0
    while len(selected) < int(sample_size) and frontier:
        rounds += 1
        candidates = set()
        for u in frontier:
            candidates.update(
                v for v in adj.get(int(u), set())
                if int(v) not in used_person_ids and int(v) not in selected
            )
        if not candidates:
            break
        neg = [int(x) for x in candidates if int(y_map.get(int(x), 0)) == 0]
        rng.shuffle(neg)
        neg = sorted(neg, key=lambda x: degree.get(x, 0), reverse=True)
        added = []
        for pid in neg:
            if len(selected) >= int(sample_size):
                break
            selected.add(pid)
            added.append(pid)
        frontier = set(added)

    diagnostics = {
        "workplace_snowball_rounds": int(rounds),
        "workplace_seed_positive_count": int(len(seed_positive_ids)),
        "workplace_reachable_sample_size": int(len(selected)),
    }
    return set(list(selected)[:int(sample_size)]), diagnostics


def export_nonoverlapping_samples_fast(
    args: argparse.Namespace,
    population_all: pd.DataFrame,
    contacts_all: pd.DataFrame,
    seed_root: Path,
    seed: int,
) -> pd.DataFrame:
    samples_dir = seed_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    pop, con, y_map, adj, degree, pp_comps = _build_workplace_graph(
        population_all=population_all,
        contacts_all=contacts_all,
        target_age_groups=args.target_age_groups,
        workplace_contact_types=args.workplace_contact_types,
    )
    print(
        f"seed={seed}: fast graph eligible_people={len(pop)} "
        f"workplace_edges={len(con)} positive_components_ge2={len(pp_comps)}"
    )

    used_person_ids: set[int] = set()
    used_edges: set[tuple[int, int, int, int]] = set()
    rows = []
    rng = np.random.default_rng(int(seed) * 100_000 + 17)
    max_positive_count = min(int(args.target_positive_count_max), int(np.floor(args.sample_size * args.max_positive_rate)))

    for sample_index in range(1, int(args.samples_per_seed) + 1):
        sample_name = args.multi_sample_name.format(n=args.sample_size, sample_index=sample_index)
        sample_dir = samples_dir / sample_name
        accepted = None
        for attempt in range(1, int(args.max_sample_attempts) + 1):
            seed_positive_ids = _choose_seed_positive_ids(
                pp_comps=pp_comps,
                used_person_ids=used_person_ids,
                min_positive_count=args.target_positive_count_min,
                target_positive_count_max=max_positive_count,
                min_positive_components=args.min_positive_components,
                max_largest_positive_component_ratio=args.max_largest_positive_component_ratio,
            )
            if len(seed_positive_ids) < int(args.min_positive_count):
                break
            selected_ids, snowball_diag = _expand_workplace_sample_fast(
                seed_positive_ids=seed_positive_ids,
                used_person_ids=used_person_ids,
                y_map=y_map,
                adj=adj,
                degree=degree,
                sample_size=args.sample_size,
                max_positive_count=max_positive_count,
                rng=rng,
            )
            if len(selected_ids) < int(args.sample_size):
                break

            sample = pop[pop["person_id"].isin(selected_ids)].copy()
            sample["sample_source"] = np.where(
                sample["person_id"].isin(seed_positive_ids),
                "selected_positive",
                "workplace_neighbor",
            )
            sample = sample.sample(frac=1, random_state=int(rng.integers(1_000_000_000))).reset_index(drop=True)
            sample_con = con[
                con["person_i"].isin(selected_ids) & con["person_j"].isin(selected_ids)
            ].copy().reset_index(drop=True)
            diag, by_person = _positive_component_diagnostics(sample, sample_con)
            diag.update({
                "sample_size": int(len(sample)),
                "positive_count": int(sample["y_true"].astype(int).sum()),
                "positive_rate": float(sample["y_true"].astype(int).mean()),
                "contact_edges": int(len(sample_con)),
                "mean_degree_3day": float(2 * len(sample_con) / max(len(sample), 1)),
                "mean_degree_per_day": float(2 * len(sample_con) / max(len(sample), 1) / max(sample_con["day"].nunique() if "day" in sample_con.columns else 1, 1)),
                "contact_days": ";".join(map(str, sorted(sample_con["day"].astype(int).unique().tolist()))) if "day" in sample_con.columns and len(sample_con) else "",
                "contact_types": ";".join(map(str, sorted(sample_con["contact_type"].astype(str).unique().tolist()))) if len(sample_con) else "",
                "age_groups": ";".join(map(str, sorted(sample["age_group"].astype(int).unique().tolist()))),
            })
            diag.update(snowball_diag)
            if sample_conditions_met(args, diag):
                accepted = (sample, sample_con, diag, by_person, attempt)
                break
            # Reject this positive seed set permanently for this extraction pass;
            # otherwise a too-sparse component can be retried forever.
            used_person_ids.update(seed_positive_ids)

        if accepted is None:
            raise RuntimeError(
                f"seed={seed} sample={sample_index}: fast extractor could not make "
                f"a non-overlapping sample after {args.max_sample_attempts} attempts"
            )

        sample, sample_con, diag, by_person, attempts_used = accepted
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample.to_csv(sample_dir / "population.csv", index=False)
        sample_con.to_csv(sample_dir / "contacts.csv", index=False)
        by_person.to_csv(sample_dir / "positive_neighbor_by_positive_person.csv", index=False)

        sample_ids = set(sample["person_id"].astype(int).tolist())
        edge_keys = set(
            (int(row.day), int(min(row.person_i, row.person_j)), int(max(row.person_i, row.person_j)), int(row.contact_type))
            for row in sample_con.itertuples(index=False)
        )
        if sample_ids & used_person_ids:
            raise RuntimeError(f"seed={seed} sample={sample_index}: internal node-overlap check failed")
        if edge_keys & used_edges:
            raise RuntimeError(f"seed={seed} sample={sample_index}: internal edge-overlap check failed")
        used_person_ids.update(sample_ids)
        used_edges.update(edge_keys)

        row = dict(diag)
        row.update({
            "seed": int(seed),
            "sample_index": int(sample_index),
            "path": str(sample_dir),
            "attempts_used": int(attempts_used),
            "sampling_conditions_met": True,
            "target_age_groups": ";".join(map(str, args.target_age_groups)),
            "workplace_contact_types": ";".join(map(str, args.workplace_contact_types)),
            "target_day": int(args.end_time),
            "contact_start_day": int(args.contact_start_day if args.contact_start_day is not None else args.end_time - args.contact_window),
            "contact_end_day": int(args.contact_end_day if args.contact_end_day is not None else args.end_time - 1),
            "max_positive_rate_condition": float(args.max_positive_rate),
            "min_positive_count_condition": int(args.min_positive_count),
            "min_contact_edges_condition": int(args.min_contact_edges),
            "nonoverlap_person_count_so_far": int(len(used_person_ids)),
            "nonoverlap_edge_count_so_far": int(len(used_edges)),
            "fast_workplace_nonoverlap": True,
        })
        pd.DataFrame([row]).to_csv(sample_dir / "sample_summary.csv", index=False)
        rows.append(row)
        print(
            f"seed={seed} sample={sample_index}: exported fast non-overlapping sample "
            f"contacts={row['contact_edges']} positives={row['positive_count']} to {sample_dir}"
        )

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
    parser.add_argument("--contact-window", type=int, default=3, help="Fallback number of previous days for contact graph when explicit bounds are not set.")
    parser.add_argument("--contact-start-day", type=int, default=37, help="First OpenABM day included in the contact graph.")
    parser.add_argument("--contact-end-day", type=int, default=39, help="Last OpenABM day included in the contact graph.")
    parser.add_argument("--n-total", type=int, default=None, help="Best-effort population override when supported by OpenABM parameters.")
    parser.add_argument("--skip-build", action="store_true", help="Use an existing OpenABM executable without running make.")
    parser.add_argument("--reuse-converted", action="store_true", help="Reuse population_all.csv and contacts_all.csv if present.")
    parser.add_argument("--reuse-samples", action="store_true", help="Reuse sample population.csv and contacts.csv if present.")
    parser.add_argument("--delete-raw-after-convert", action="store_true", help="Delete raw OpenABM output after converted/sample CSVs are saved.")
    parser.add_argument("--allow-best-effort-samples", dest="require_sampling_conditions", action="store_false", help="Save the best sample even if diagnostics do not meet all thresholds.")
    parser.set_defaults(require_sampling_conditions=True)
    parser.add_argument("--fast-workplace-nonoverlap", action="store_true", default=True, help="Build the workplace graph once and export non-overlapping workplace-neighborhood samples quickly.")
    parser.add_argument("--no-fast-workplace-nonoverlap", dest="fast_workplace_nonoverlap", action="store_false")
    parser.add_argument("--target-age-groups", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--workplace-contact-types", nargs="+", type=int, default=[1])
    parser.add_argument("--max-positive-rate", type=float, default=0.05)
    parser.add_argument("--min-positive-count", type=int, default=30)
    parser.add_argument("--min-contact-edges", type=int, default=10000)
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
