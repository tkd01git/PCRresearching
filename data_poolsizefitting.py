"""
data_poolsizefitting.py

OpenABM出力を使った簡略版データ生成・3チェック症状・prior作成・図表作成。
地域・ワクチンは使わない。接触グラフと症状のみを用いる。
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import shutil
import zipfile
import subprocess

import numpy as np
import pandas as pd

SYMPTOM_NAMES = ["item0", "item1", "item2"]
COVID_SYMPTOM_INDICES = [0, 1, 2]
NONCOVID_SYMPTOM_INDICES = []


DEFAULT_CONTACT_TYPE_WEIGHTS = {
    # Numeric OpenABM type codes. Default interpretation for this notebook:
    #   0 = weak / transient contact       -> 1
    #   1 = middle / school-work contact   -> 2
    #   2 = close / household-like contact -> 3
    # If your OpenABM build uses a different type-code convention, override this
    # dictionary in the notebook via CONTACT_TYPE_WEIGHTS.
    0: 1.0, 1: 2.0, 2: 3.0,
    0.0: 1.0, 1.0: 2.0, 2.0: 3.0,
    "0": 1.0, "1": 2.0, "2": 3.0,
    "0.0": 1.0, "1.0": 2.0, "2.0": 3.0,
    "random": 1.0, "community": 1.0, "transient": 1.0, "interaction": 1.0,
    "occupation": 2.0, "occupational": 2.0, "school": 2.0, "work": 2.0, "workplace": 2.0,
    "household": 3.0, "home": 3.0, "house": 3.0, "hh": 3.0,
}


def contact_type_to_weight(contact_type, contact_type_weights=None):
    """Map OpenABM contact_type to an edge weight.

    Robust to int/float/string values such as 0, 1, 2, "0", "1.0".
    Unknown types fall back to 1.0.
    """
    weights = dict(DEFAULT_CONTACT_TYPE_WEIGHTS)
    if contact_type_weights:
        for k, v in contact_type_weights.items():
            weights[k] = float(v)
            weights[str(k).strip().lower()] = float(v)
            try:
                weights[int(k)] = float(v)
            except Exception:
                pass
            try:
                weights[float(k)] = float(v)
            except Exception:
                pass

    if contact_type in weights:
        return float(weights[contact_type])

    try:
        f = float(contact_type)
        if f in weights:
            return float(weights[f])
        if f.is_integer() and int(f) in weights:
            return float(weights[int(f)])
        key_num = str(int(f)) if f.is_integer() else str(f)
        if key_num in weights:
            return float(weights[key_num])
    except Exception:
        pass

    key = str(contact_type).strip().lower()
    if key in weights:
        return float(weights[key])
    if any(tok in key for tok in ["house", "home", "family", "hh"]):
        return float(weights.get("household", 3.0))
    if any(tok in key for tok in ["occupation", "school", "work", "office"]):
        return float(weights.get("occupation", 2.0))
    if any(tok in key for tok in ["random", "community", "transient", "transport", "interaction"]):
        return float(weights.get("random", 1.0))
    return 1.0


def normalize01(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def build_adjacency(contacts: pd.DataFrame) -> Dict[int, set]:
    adj = defaultdict(set)
    if contacts is None or len(contacts) == 0:
        return dict(adj)
    if not {"person_i", "person_j"}.issubset(contacts.columns):
        raise ValueError(f"contacts must include person_i/person_j. columns={contacts.columns.tolist()}")
    for i, j in zip(contacts["person_i"].astype(int), contacts["person_j"].astype(int)):
        if i == j:
            continue
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    return dict(adj)


def build_weight_matrix(contacts: pd.DataFrame, person_ids: np.ndarray, mode: str = "basic", contact_type_weights: dict | None = None) -> np.ndarray:
    """
    Build an undirected weighted contact matrix.

    mode="basic": previous behavior. Uses contacts["weight"] if present, otherwise 1.0.
                  Since converted OpenABM contact events have weight=1.0, this is mostly contact-count based.
    mode="binary": W_ij=1 if at least one contact exists.
    mode="contact_type": uses contact_type weights such as household=3, occupation=2, random=1.
    """
    ids = [int(x) for x in person_ids]
    id_to_pos = {pid: idx for idx, pid in enumerate(ids)}
    W = np.zeros((len(ids), len(ids)), dtype=float)
    if contacts is None or len(contacts) == 0:
        return W
    mode = str(mode).lower()
    weight_col = "weight" if "weight" in contacts.columns else None
    type_col = "contact_type" if "contact_type" in contacts.columns else ("type" if "type" in contacts.columns else None)
    for _, row in contacts.iterrows():
        i, j = int(row["person_i"]), int(row["person_j"])
        if i in id_to_pos and j in id_to_pos and i != j:
            base_w = float(row[weight_col]) if weight_col else 1.0
            if mode == "contact_type":
                ctype = row[type_col] if type_col else "interaction"
                # Preserve repeated-contact counts while adding contact-type intensity.
                w = base_w * contact_type_to_weight(ctype, contact_type_weights=contact_type_weights)
            elif mode == "binary":
                w = 1.0
            else:
                w = base_w
            a, b = id_to_pos[i], id_to_pos[j]
            W[a, b] += w
            W[b, a] += w
    if mode == "binary":
        W = (W > 0).astype(float)
    return W


def summarize_contact_types(contacts: pd.DataFrame, contact_type_weights: dict | None = None) -> pd.DataFrame:
    """Summarize available contact types and their assigned weights."""
    if contacts is None or len(contacts) == 0:
        return pd.DataFrame(columns=["contact_type", "count", "assigned_weight"])
    type_col = "contact_type" if "contact_type" in contacts.columns else ("type" if "type" in contacts.columns else None)
    if type_col is None:
        return pd.DataFrame({"contact_type": ["interaction"], "count": [len(contacts)], "assigned_weight": [1.0]})
    vc = contacts[type_col].astype(str).value_counts().rename_axis("contact_type").reset_index(name="count")
    vc["assigned_weight"] = vc["contact_type"].apply(lambda x: contact_type_to_weight(x, contact_type_weights=contact_type_weights))
    return vc


def edge_weight_summary(W: np.ndarray, name: str = "W") -> pd.DataFrame:
    """Return a compact summary of positive upper-triangular edge weights."""
    W = np.asarray(W, dtype=float)
    if W.size == 0:
        return pd.DataFrame(columns=["matrix", "edge_weight", "edge_count"])
    vals = W[np.triu_indices_from(W, k=1)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return pd.DataFrame(columns=["matrix", "edge_weight", "edge_count"])
    vc = pd.Series(vals).value_counts().sort_index().rename_axis("edge_weight").reset_index(name="edge_count")
    vc.insert(0, "matrix", name)
    return vc



def extract_or_locate_bundle(zip_path=None, extract_dir="/content/openabm_bundle_extracted", search_dir="/content"):
    search_dir = Path(search_dir)
    if zip_path is not None:
        zp = Path(zip_path)
        if zp.exists() and zipfile.is_zipfile(zp):
            extract_dir = Path(extract_dir)
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "r") as z:
                z.extractall(extract_dir)
            return extract_dir
    for p in search_dir.rglob("population.csv"):
        if "samples" in str(p):
            return p.parents[2] if p.parent.parent.name == "samples" else p.parent.parent
    raise FileNotFoundError("OpenABM bundle or extracted samples were not found.")


def setup_openabm_colab():
    subprocess.run("apt-get update -qq", shell=True, check=False)
    subprocess.run("apt-get install -y -qq build-essential git make gcc g++ libgsl-dev swig", shell=True, check=False)
    subprocess.run("python -m pip install -q numpy pandas scipy matplotlib", shell=True, check=False)


def clone_and_build_openabm(repo_dir="/content/OpenABM-Covid19"):
    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        subprocess.run(f"git clone https://github.com/BDI-pathogens/OpenABM-Covid19.git {repo_dir}", shell=True, check=True)
    subprocess.run("make all", shell=True, cwd=str(repo_dir), check=True)
    return repo_dir


def run_openabm_exe(repo_or_exe_path="/content/OpenABM-Covid19", output_dir="/content/openabm_real_output", n_total=None, days=None, seed=None, end_time=40, **kwargs):
    """Run OpenABM and return the output directory.

    This wrapper is intentionally permissive so older notebooks that pass
    n_total/days/seed do not fail. OpenABM's population size is primarily
    controlled by its parameter files; n_total is accepted for compatibility
    but is not guaranteed to override every OpenABM build.
    """
    path = Path(repo_or_exe_path)
    repo_dir = path if path.is_dir() else path.parent.parent
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_end_time = int(days if days is not None else end_time)
    base_param = repo_dir / "tests/data/baseline_parameters.csv"
    household = repo_dir / "tests/data/baseline_household_demographics.csv"
    param_custom = output_dir / "parameters_end.csv"
    if not base_param.exists():
        raise FileNotFoundError(f"OpenABM baseline parameter file not found: {base_param}")
    params = pd.read_csv(base_param)
    if "end_time" in params.columns:
        params.loc[0, "end_time"] = actual_end_time
    # Best-effort seed override if such columns exist in the build.
    if seed is not None:
        for col in ["rng_seed", "random_seed", "seed"]:
            if col in params.columns:
                params.loc[0, col] = int(seed)
    params.to_csv(param_custom, index=False)

    exe_candidates = [path] if path.is_file() else []
    exe_candidates += [repo_dir / "src/covid19ibm.exe", repo_dir / "src/COVID19", repo_dir / "src/covid19"]
    exe = next((p for p in exe_candidates if p.exists()), None)
    if exe is None:
        raise FileNotFoundError(f"OpenABM executable not found under {repo_dir}")
    cmd = [str(exe), str(param_custom), "1", str(output_dir), str(household)]
    subprocess.run(cmd, check=True)
    return output_dir

def _find_first(output_dir: Path, names: list[str]):
    for name in names:
        p = output_dir / name
        if p.exists():
            return p
    hits = []
    for pat in names:
        hits += list(output_dir.glob(pat))
    return hits[0] if hits else None


def convert_raw_openabm_to_population_contacts(output_dir="/content/openabm_real_output", target_day=40, contact_window=7, random_seed=472):
    output_dir = Path(output_dir)
    individual_path = _find_first(output_dir, ["individual_file_Run1.csv", "individual_Run1.csv", "*individual*Run1*.csv"])
    interactions_path = _find_first(output_dir, ["interactions_Run1.csv", "*interaction*Run1*.csv"])
    if individual_path is None:
        raise FileNotFoundError(f"individual file not found in {output_dir}")
    if interactions_path is None:
        raise FileNotFoundError(f"interactions file not found in {output_dir}")
    indiv = pd.read_csv(individual_path)
    inter = pd.read_csv(interactions_path)
    id_col = "ID" if "ID" in indiv.columns else indiv.columns[0]
    status_col = next((c for c in ["current_status", "status", "disease_state"] if c in indiv.columns), None)
    if status_col is None:
        raise ValueError(f"status column not found: {indiv.columns.tolist()}")
    status_num = pd.to_numeric(indiv[status_col], errors="coerce").fillna(0)
    pop = pd.DataFrame({
        "person_id": indiv[id_col].astype(int),
        "day": int(target_day),
        "openabm_status_code": indiv[status_col].astype(str),
        "y_true": (status_num != 0).astype(int),
    })
    if "house_no" in indiv.columns:
        pop["house_no"] = indiv["house_no"].astype(int)
    age_col = next((c for c in ["age", "age_group", "age_group_idx"] if c in indiv.columns), None)
    if age_col:
        pop["age_group"] = indiv[age_col]

    if {"ID_1", "ID_2"}.issubset(inter.columns):
        i_col, j_col = "ID_1", "ID_2"
    elif {"person_i", "person_j"}.issubset(inter.columns):
        i_col, j_col = "person_i", "person_j"
    else:
        raise ValueError(f"contact id columns not found: {inter.columns.tolist()}")
    if "time" in inter.columns:
        inter["contact_day"] = pd.to_numeric(inter["time"], errors="coerce").fillna(target_day).astype(int)
        inter = inter[(inter["contact_day"] >= target_day - contact_window) & (inter["contact_day"] <= target_day)].copy()
    else:
        inter["contact_day"] = target_day
    con = pd.DataFrame({
        "day": inter["contact_day"].astype(int),
        "person_i": inter[i_col].astype(int),
        "person_j": inter[j_col].astype(int),
        "contact_type": inter["type"].astype(str) if "type" in inter.columns else "interaction",
        "weight": 1.0,
    })
    con = con[con["person_i"] != con["person_j"]].copy()
    a = con[["person_i", "person_j"]].min(axis=1)
    b = con[["person_i", "person_j"]].max(axis=1)
    con["person_i"], con["person_j"] = a.astype(int), b.astype(int)
    con = con.groupby(["day", "person_i", "person_j", "contact_type"], as_index=False)["weight"].sum()
    pop.to_csv(output_dir / "population_all.csv", index=False)
    con.to_csv(output_dir / "contacts_all.csv", index=False)
    return pop, con


def _bfs_from_seed(seed, adj, max_count):
    out, seen = [], {seed}
    q = deque([seed])
    while q and len(out) < max_count:
        u = q.popleft()
        out.append(u)
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return out


def _sample_target_rate(candidates, n, target_positive_rate, rng):
    """Sample n rows while approximately preserving the requested positive rate."""
    if n <= 0:
        return candidates.iloc[0:0].copy().reset_index(drop=True)
    if len(candidates) <= n:
        return candidates.copy().reset_index(drop=True)
    n_pos = int(round(n * target_positive_rate))
    pos = candidates[candidates["y_true"] == 1]
    neg = candidates[candidates["y_true"] == 0]
    take_pos = min(n_pos, len(pos))
    take_neg = min(n - take_pos, len(neg))
    parts = []
    if take_pos:
        parts.append(pos.sample(take_pos, random_state=int(rng.integers(1e9))))
    if take_neg:
        parts.append(neg.sample(take_neg, random_state=int(rng.integers(1e9))))
    sample = pd.concat(parts) if parts else candidates.sample(n, random_state=int(rng.integers(1e9)))
    if len(sample) < n:
        rest = candidates[~candidates["person_id"].isin(sample["person_id"])]
        sample = pd.concat([sample, rest.sample(n-len(sample), random_state=int(rng.integers(1e9)))])
    return sample.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)


def _connected_components_from_adj(adj: dict[int, set[int]], nodes: set[int] | None = None) -> list[list[int]]:
    """Connected components for a sparse adjacency dictionary."""
    if nodes is None:
        nodes = set(adj.keys())
    nodes = set(int(x) for x in nodes)
    seen = set()
    comps = []
    for seed in nodes:
        if seed in seen:
            continue
        q = deque([seed])
        seen.add(seed)
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj.get(u, set()):
                if v in nodes and v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(comp)
    return comps


def _select_clustered_population_sample(
    population_all: pd.DataFrame,
    contacts_all: pd.DataFrame,
    n: int,
    target_positive_rate: float,
    rng: np.random.Generator,
    cluster_positive_ratio: float = 0.70,
    min_positive_component_size: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Select an n-person sample that contains positive-positive contact clusters.

    This implementation avoids expensive full-dataframe copies and uses numpy
    lookup arrays when person_id is dense, as in OpenABM outputs.
    """
    if "person_id" not in population_all.columns or "y_true" not in population_all.columns:
        raise ValueError("population_all must include person_id and y_true")
    if not {"person_i", "person_j"}.issubset(contacts_all.columns):
        raise ValueError("contacts_all must include person_i and person_j")

    target_pos = int(round(int(n) * float(target_positive_rate)))
    target_pos = max(1, min(target_pos, int(n)))
    target_neg = int(n) - target_pos

    pid = population_all["person_id"].to_numpy(dtype=np.int64, copy=False)
    y_all = population_all["y_true"].astype(int).to_numpy(copy=False)
    max_id = int(max(pid.max(), contacts_all["person_i"].max(), contacts_all["person_j"].max()))

    # Fast dense lookup: person_id -> y_true. Unknown IDs are treated as negative/unavailable.
    y_lookup = np.zeros(max_id + 1, dtype=np.int8)
    in_pop = np.zeros(max_id + 1, dtype=bool)
    y_lookup[pid] = y_all.astype(np.int8)
    in_pop[pid] = True

    ci = contacts_all["person_i"].to_numpy(dtype=np.int64, copy=False)
    cj = contacts_all["person_j"].to_numpy(dtype=np.int64, copy=False)
    valid = (ci <= max_id) & (cj <= max_id) & in_pop[ci] & in_pop[cj]
    pos_edge_mask = valid & (y_lookup[ci] == 1) & (y_lookup[cj] == 1)
    pp_i = ci[pos_edge_mask]
    pp_j = cj[pos_edge_mask]

    # Positive-positive adjacency. Enough for cluster extraction only.
    pos_adj: dict[int, set[int]] = defaultdict(set)
    for a, b in zip(pp_i, pp_j):
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        pos_adj[ia].add(ib)
        pos_adj[ib].add(ia)

    selected_pos: list[int] = []
    selected_pos_set: set[int] = set()
    clustered_target = min(target_pos, int(round(target_pos * float(np.clip(cluster_positive_ratio, 0.0, 1.0)))))

    if pos_adj and clustered_target > 0:
        # Start from high positive-positive degree nodes and BFS local clusters.
        seeds = sorted(pos_adj.keys(), key=lambda x: len(pos_adj.get(x, set())), reverse=True)
        top = seeds[: min(len(seeds), 200)]
        rng.shuffle(top)
        seeds = top + seeds[min(len(seeds), 200):]
        for seed_node in seeds:
            if len(selected_pos) >= clustered_target:
                break
            if seed_node in selected_pos_set:
                continue
            q = deque([seed_node])
            selected_pos_set.add(seed_node)
            selected_pos.append(seed_node)
            while q and len(selected_pos) < clustered_target:
                u = q.popleft()
                neigh = sorted(pos_adj.get(u, set()), key=lambda v: len(pos_adj.get(v, set())), reverse=True)
                for v in neigh:
                    if v not in selected_pos_set:
                        selected_pos_set.add(v)
                        selected_pos.append(v)
                        q.append(v)
                        if len(selected_pos) >= clustered_target:
                            break

    # Add additional positives from neighbors of selected positives, then random positives.
    if len(selected_pos) < target_pos:
        neighbor_pos = set()
        for p0 in selected_pos:
            neighbor_pos.update(pos_adj.get(p0, set()))
        neighbor_pos = list(neighbor_pos - selected_pos_set)
        rng.shuffle(neighbor_pos)
        for p0 in neighbor_pos:
            if len(selected_pos) >= target_pos:
                break
            selected_pos.append(int(p0)); selected_pos_set.add(int(p0))

    if len(selected_pos) < target_pos:
        all_pos = pid[y_all == 1]
        remaining_pos = np.array([x for x in all_pos if int(x) not in selected_pos_set], dtype=np.int64)
        if len(remaining_pos):
            take = min(target_pos - len(selected_pos), len(remaining_pos))
            add = rng.choice(remaining_pos, size=take, replace=False).astype(int).tolist()
            selected_pos.extend(add); selected_pos_set.update(add)

    selected_pos = selected_pos[:target_pos]
    selected_pos_set = set(selected_pos)

    # Negatives: prefer the neighborhood around selected positives.
    selected_mask_i = np.isin(ci, np.fromiter(selected_pos_set, dtype=np.int64))
    selected_mask_j = np.isin(cj, np.fromiter(selected_pos_set, dtype=np.int64))
    incident_mask = valid & (selected_mask_i | selected_mask_j)
    neigh_ids = np.unique(np.concatenate([ci[incident_mask], cj[incident_mask]])) if incident_mask.any() else np.array([], dtype=np.int64)
    candidate_neg = neigh_ids[(neigh_ids <= max_id) & (y_lookup[neigh_ids] == 0)]
    candidate_neg = np.array([x for x in candidate_neg if int(x) not in selected_pos_set], dtype=np.int64)
    if len(candidate_neg):
        rng.shuffle(candidate_neg)
    selected_neg = candidate_neg[:target_neg].astype(int).tolist()
    selected_neg_set = set(selected_neg)

    if len(selected_neg) < target_neg:
        all_neg = pid[y_all == 0]
        remaining_neg = np.array([x for x in all_neg if int(x) not in selected_neg_set and int(x) not in selected_pos_set], dtype=np.int64)
        if len(remaining_neg):
            take = min(target_neg - len(selected_neg), len(remaining_neg))
            add = rng.choice(remaining_neg, size=take, replace=False).astype(int).tolist()
            selected_neg.extend(add); selected_neg_set.update(add)

    selected_ids = list(dict.fromkeys(selected_pos + selected_neg))
    if len(selected_ids) < n:
        selected_set = set(selected_ids)
        remaining = pid[~np.isin(pid, np.fromiter(selected_set, dtype=np.int64))]
        take = min(int(n) - len(selected_ids), len(remaining))
        if take > 0:
            selected_ids.extend(rng.choice(remaining, size=take, replace=False).astype(int).tolist())

    selected_ids = selected_ids[:int(n)]
    selected_set = set(selected_ids)
    sample = population_all[population_all["person_id"].isin(selected_set)].copy()
    sample["sample_source"] = np.where(sample["person_id"].isin(selected_pos_set), "positive_cluster", "positive_neighborhood_or_fill")
    sample = sample.drop_duplicates("person_id").head(int(n))
    sample = sample.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)

    ids_arr = sample["person_id"].to_numpy(dtype=np.int64)
    ids_set = set(ids_arr.astype(int).tolist())
    con = contacts_all[contacts_all["person_i"].isin(ids_set) & contacts_all["person_j"].isin(ids_set)].copy()

    sample_pos = set(sample.loc[sample["y_true"].astype(int) == 1, "person_id"].astype(int))
    pp_in_sample = con[con["person_i"].isin(sample_pos) & con["person_j"].isin(sample_pos)]
    pp_adj = defaultdict(set)
    for a, b in zip(pp_in_sample["person_i"].astype(int), pp_in_sample["person_j"].astype(int)):
        pp_adj[int(a)].add(int(b)); pp_adj[int(b)].add(int(a))
    comps_sample = _connected_components_from_adj(pp_adj, set(sample_pos)) if sample_pos else []
    connected_pos = sum(1 for p0 in sample_pos if len(pp_adj.get(p0, set())) > 0)
    diagnostics = {
        "positive_positive_edges": int(len(pp_in_sample)),
        "positive_with_positive_neighbor": int(connected_pos),
        "isolated_positive_count": int(len(sample_pos) - connected_pos),
        "positive_component_count": int(len(comps_sample)),
        "largest_positive_component": int(max((len(c) for c in comps_sample), default=0)),
    }
    return sample, con, diagnostics


def export_graph_samples(
    population_all,
    contacts_all,
    output_dir,
    sample_sizes=(100, 1000, 3000),
    target_positive_rate=0.03,
    graph_core_ratio=0.70,
    seed=472,
    cluster_positive_ratio=0.70,
):
    """
    Export graph samples for analysis.

    Unlike the previous degree-core sampling, this version explicitly samples
    positive-positive contact clusters, then fills the remaining sample from
    their contact neighborhood. This matches a mass-screening scenario in a
    school/workplace where clustered infection is suspected.
    """
    output_dir = Path(output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for n in sample_sizes:
        sample, con, diag = _select_clustered_population_sample(
            population_all=population_all,
            contacts_all=contacts_all,
            n=int(n),
            target_positive_rate=float(target_positive_rate),
            rng=rng,
            cluster_positive_ratio=float(cluster_positive_ratio),
        )
        sample_dir = samples_dir / f"n{n}_pos{int(round(target_positive_rate*100))}pct_graph"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample.to_csv(sample_dir / "population.csv", index=False)
        con.to_csv(sample_dir / "contacts.csv", index=False)
        row = {
            "sample_size": int(len(sample)),
            "positive_count": int(sample["y_true"].astype(int).sum()),
            "positive_rate": float(sample["y_true"].astype(int).mean()),
            "contact_edges": int(len(con)),
            "path": str(sample_dir),
        }
        row.update(diag)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "generation_summary.csv", index=False)
    return summary


def save_bundle(output_dir, zip_path=None):
    output_dir = Path(output_dir)
    if zip_path is None:
        zip_path = output_dir.with_suffix(".zip")
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in output_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(output_dir)))
    return zip_path


def load_sample_from_root(data_root, sample_size=3000, target_positive_rate=0.03):
    data_root = Path(data_root)
    pct = int(round(target_positive_rate*100))
    candidates = [
        data_root / "samples" / f"n{sample_size}_pos{pct}pct_graph",
        data_root / f"n{sample_size}_pos{pct}pct_graph",
    ]
    for d in candidates:
        if (d/"population.csv").exists() and (d/"contacts.csv").exists():
            return pd.read_csv(d/"population.csv"), pd.read_csv(d/"contacts.csv"), d
    pops = [p for p in data_root.rglob("population.csv") if f"n{sample_size}" in str(p)]
    if not pops:
        raise FileNotFoundError(f"population.csv for n={sample_size} not found under {data_root}")
    d = pops[0].parent
    return pd.read_csv(d/"population.csv"), pd.read_csv(d/"contacts.csv"), d


def _components_from_W(W):
    n = W.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        q = deque([i]); seen[i] = True; comp=[]
        while q:
            u=q.popleft(); comp.append(u)
            for v in np.where(W[u] > 0)[0]:
                if not seen[v]:
                    seen[v]=True; q.append(v)
        comps.append(comp)
    return comps


def _generate_symptoms(
    pop,
    W,
    strength=0.90,
    seed=472,
    force=False,
    strong_edge_threshold=2.0,
    neg_covid_leak_prob=0.0,
    off_pattern_covid_prob=0.10,
    noncovid_base_prob=0.0,
    pos_noncovid_prob=0.0,
    target_positive_symptomatic_rate=0.92,
    max_symptoms=3,
):
    """
    Simplified symptom generator.

    Assumption:
      - There are exactly 3 unnamed symptom checkbox items.
      - Every observed symptom is COVID-derived.
      - Negative individuals have no reported symptoms.
      - Positive individuals are symptomatic with probability
        target_positive_symptomatic_rate.
      - Asymptomatic positives are therefore kept below about 10% by default
        when target_positive_symptomatic_rate >= 0.90.
      - Symptom patterns are locally similar within positive contact clusters.

    Output columns:
      - reported_symptom_0, reported_symptom_1, reported_symptom_2
      - reported_total_symptom_count
      - internal_covid_symptom_count
      - internal_has_covid_derived_symptom
    """
    existing = [c for c in pop.columns if c.startswith("reported_symptom_")]
    if existing and not force:
        return pop

    rng = np.random.default_rng(seed + 77)
    n = len(pop)
    y = pop["y_true"].astype(int).to_numpy()
    d = 3
    S = np.zeros((n, d), dtype=int)

    share_prob = float(np.clip(strength, 0.0, 1.0))
    target_positive_symptomatic_rate = float(np.clip(target_positive_symptomatic_rate, 0.0, 1.0))
    max_symptoms = int(max_symptoms)

    W = np.asarray(W, dtype=float)
    W_strong = np.where(W >= float(strong_edge_threshold), W, 0.0)

    pos_mask = (y == 1)
    W_pos = W_strong.copy()
    W_pos[~pos_mask, :] = 0.0
    W_pos[:, ~pos_mask] = 0.0
    comps = _components_from_W(W_pos)
    node_to_comp = {i: cid for cid, comp in enumerate(comps) for i in comp}

    # Unnamed 3-checkbox patterns. Values mean number/combination of checked symptoms,
    # not clinically named symptoms.
    symptom_patterns = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=int)
    pattern_probs = np.array([0.34, 0.18, 0.15, 0.15, 0.06, 0.06, 0.06], dtype=float)
    pattern_probs = pattern_probs / pattern_probs.sum()
    comp_pattern = {
        cid: symptom_patterns[int(rng.choice(len(symptom_patterns), p=pattern_probs))]
        for cid, _ in enumerate(comps)
    }

    pos_indices = np.where(y == 1)[0]
    # Deterministically cap asymptomatic positives according to the target rate.
    # With the default 0.92, asymptomatic positives are at most about 8%, hence below 10%.
    max_asymptomatic = int(np.floor(len(pos_indices) * max(0.0, 1.0 - target_positive_symptomatic_rate)))
    max_asymptomatic = min(max_asymptomatic, int(np.floor(len(pos_indices) * 0.10)))
    asymptomatic_set = set(rng.choice(pos_indices, size=max_asymptomatic, replace=False).astype(int).tolist()) if max_asymptomatic > 0 else set()

    for i in range(n):
        if y[i] != 1:
            continue
        # A small, capped fraction of positives remain asymptomatic.
        if i in asymptomatic_set:
            continue

        pattern = comp_pattern.get(node_to_comp.get(i, -1), symptom_patterns[0])
        if rng.random() < share_prob:
            core = pattern.copy()
            # Small individual variation while preserving at least one symptom.
            for k in range(d):
                if core[k] == 1 and rng.random() < 0.08:
                    core[k] = 0
                elif core[k] == 0 and rng.random() < float(np.clip(off_pattern_covid_prob, 0.0, 1.0)):
                    core[k] = 1
            if core.sum() == 0:
                core[int(rng.choice(np.arange(d)))] = 1
        else:
            k = int(rng.choice([1, 2, 3], p=[0.20, 0.45, 0.35]))
            core = np.zeros(d, dtype=int)
            core[rng.choice(np.arange(d), size=k, replace=False)] = 1
        S[i, :] = core

    if max_symptoms > 0:
        for i in range(n):
            idx = np.where(S[i] == 1)[0].tolist()
            if len(idx) > max_symptoms:
                keep = rng.choice(idx, size=max_symptoms, replace=False)
                S[i, :] = 0
                S[i, keep] = 1

    out = pop.copy()
    for c in [c for c in out.columns if c.startswith("reported_symptom_") or c.startswith("internal_")]:
        out = out.drop(columns=c)

    for l in range(d):
        out[f"reported_symptom_{l}"] = S[:, l]
    out["reported_total_symptom_count"] = S.sum(axis=1)
    out["internal_covid_symptom_count"] = out["reported_total_symptom_count"].astype(int)
    out["internal_has_covid_derived_symptom"] = (out["internal_covid_symptom_count"] > 0).astype(int)
    out["internal_asymptomatic_positive"] = ((out["y_true"].astype(int) == 1) & (out["reported_total_symptom_count"].astype(int) == 0)).astype(int)
    return out

def _assign_rna(pop, W, params, seed=472):
    rng = np.random.default_rng(seed + 99)
    y = pop["y_true"].astype(int).to_numpy()
    symptom = normalize01(pop["reported_total_symptom_count"].to_numpy())
    row = W.sum(axis=1)
    neigh = np.zeros_like(symptom, dtype=float)
    mask = row > 0
    neigh[mask] = (W[mask] @ symptom) / row[mask]
    neigh = normalize01(neigh)
    log_rna = np.zeros(len(pop), dtype=float)
    pos = np.where(y == 1)[0]
    base = 3.0 + 5.0 * normalize01(0.65*symptom[pos] + 0.35*neigh[pos] + rng.normal(0,0.15,len(pos)))
    log_rna[pos] = np.clip(base, 3, 8)
    x = np.zeros(len(pop), dtype=float)
    x[pos] = 10 ** log_rna[pos]
    out = pop.copy()
    out["viral_rna_load"] = x
    logx = np.zeros(len(x), dtype=float)
    mask_pos = x > 0
    logx[mask_pos] = np.log10(x[mask_pos])
    out["log10_viral_rna_load"] = logx
    return out, x


def build_analysis_dataset(
    population,
    contacts,
    params,
    pool_size=10,
    cluster_symptom_strength=0.90,
    seed=472,
    force_symptom_regeneration=True,
    contact_type_weights=None,
    strong_edge_threshold=2.0,
    neg_covid_leak_prob=0.02,
    noncovid_base_prob=0.10,
    pos_noncovid_prob=0.10,
    target_positive_symptomatic_rate=0.92,
):
    import function_poolsizefittiing as fn
    pop = population.copy().reset_index(drop=True)
    if "person_id" not in pop.columns:
        pop["person_id"] = np.arange(len(pop))
    person_ids = pop["person_id"].astype(int).to_numpy()

    # Basic graph: previous contact-count based matrix.
    W_basic = build_weight_matrix(contacts, person_ids, mode="basic")
    # Type-weighted graph: household=3, occupation=2, random=1 by default.
    W_type_weighted = build_weight_matrix(contacts, person_ids, mode="contact_type", contact_type_weights=contact_type_weights)

    # Data generation uses the contact-type-weighted graph.
    # This makes symptom similarity and RNA allocation reflect close/middle/weak contact intensity.
    pop = _generate_symptoms(
        pop,
        W_type_weighted,
        strength=cluster_symptom_strength,
        seed=seed,
        force=force_symptom_regeneration,
        strong_edge_threshold=strong_edge_threshold,
        neg_covid_leak_prob=neg_covid_leak_prob,
        noncovid_base_prob=noncovid_base_prob,
        pos_noncovid_prob=pos_noncovid_prob,
        target_positive_symptomatic_rate=target_positive_symptomatic_rate,
    )
    params = fn.derive_params({**params, "n": len(pop), "pool_size": pool_size})
    pop, x_true = _assign_rna(pop, W_type_weighted, params, seed=seed)
    n = len(pop)
    # Pool-size sweep uses pool_size=1..30. If n is not divisible by pool_size,
    # the last pool is allowed to be smaller.
    A, pools = fn.make_pooling_matrix(n, pool_size=pool_size, gaps=params["gaps"], allow_incomplete_last_pool=True)
    pooled_amount_true, pooled_ct, pooled_amount_est = fn.pooled_measurements_qpcr(A, x_true, params)
    symptom_cols = [c for c in pop.columns if c.startswith("reported_symptom_")]
    symptom_mat = pop[symptom_cols].to_numpy(dtype=int)
    return {
        "params": params,
        "patient_data": pop,
        "contacts": contacts.copy(),
        "person_ids": person_ids,
        # Main graph for analysis is the contact-type-weighted graph.
        "W": W_type_weighted,
        "W_basic": W_basic,
        "W_type_weighted": W_type_weighted,
        "contact_type_summary": summarize_contact_types(contacts, contact_type_weights=contact_type_weights),
        "symptom_mat": symptom_mat,
        "symptom_cols": symptom_cols,
        "symptom_names": SYMPTOM_NAMES.copy(),
        "true_covid_mask": np.array([1,1,1]+[0]*3, dtype=int),
        "y_true": pop["y_true"].astype(int).to_numpy(),
        "x_true": x_true,
        "A": A,
        "pools": pools,
        "pooled_amount_true": pooled_amount_true,
        "pooled_ct": pooled_ct,
        "pooled_amount_est": pooled_amount_est,
    }


def _score_symptoms(S, W, use_R=False, use_C=True, use_U=False, clip_negative_C=False, c_edge_min_weight=2.0):
    """
    Score each symptom.

    R: symptom co-occurrence with people who report many symptoms.
       This does not use graph structure.
    C: local excess consistency on strong-contact edges.
       C_m = q_m - p_m, where
         p_m = P(symptom m in the whole population)
         q_m = weighted P(neighbor has symptom m | ego has symptom m, W_ij >= c_edge_min_weight)
       Default uses the raw value q_m - p_m; negative values are not clipped.
    U: rarity/specificity = 1 - p_m. Kept for diagnostics only; default is not used.

    total_score only sums enabled components.
    """
    S = np.asarray(S, dtype=float)
    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    W_C = np.where(W >= float(c_edge_min_weight), W, 0.0)

    n, m_count = S.shape
    total_count = np.clip(S.sum(axis=1) / max(m_count, 1), 0.0, 1.0)
    row_sum = W_C.sum(axis=1)

    rows = []
    for l in range(m_count):
        x = S[:, l].astype(float)
        n_carriers = int(x.sum())
        p = float(x.mean()) if n > 0 else 0.0

        if n_carriers == 0:
            R = 0.0
            q = 0.0
            C_raw = 0.0
            C = 0.0
            U = 0.0
        else:
            R = float((x * total_count).sum() / max(float(x.sum()), 1e-12))

            # q_m: among weighted neighbors of symptom carriers,
            # how often does the neighbor also have the same symptom?
            denom = float((row_sum * x).sum())
            if denom > 0:
                numer = float(x @ W_C @ x)
                q = numer / denom
            else:
                q = 0.0

            C_raw = q - p
            C = max(C_raw, 0.0) if clip_negative_C else C_raw
            U = float(max(0.0, 1.0 - p))

        score = (R if use_R else 0.0) + (C if use_C else 0.0) + (U if use_U else 0.0)
        rows.append({
            "symptom_index": l,
            "symptom_name": SYMPTOM_NAMES[l],
            "n_carriers": n_carriers,
            "prevalence_p": p,
            "neighbor_share_q": q,
            "weighted_risk_mean_R": R,
            "local_excess_consistency_C_raw": C_raw,
            "local_excess_consistency_C": C,
            "specificity_U": U,
            # short aliases for plotting
            "R": R,
            "C": C,
            "U": U,
            "use_R": bool(use_R),
            "use_C": bool(use_C),
            "use_U": bool(use_U),
            "score_formula": "+".join([name for name, enabled in [("R", use_R), ("C", use_C), ("U", use_U)] if enabled]) or "none",
            "total_score": score,
        })
    return pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)

def _build_graph_selected_prior(S, W, top_k_symptoms=3, top_percent_symptoms=None, beta_symptom=1.0, use_R=False, use_C=True, use_U=False, c_edge_min_weight=2.0):
    """Build only the main graph-selected symptom prior."""
    score_df = _score_symptoms(S, W, use_R=use_R, use_C=use_C, use_U=use_U, clip_negative_C=False, c_edge_min_weight=c_edge_min_weight)

    if top_percent_symptoms is not None:
        p = float(top_percent_symptoms)
        if not (0 < p <= 1):
            raise ValueError("top_percent_symptoms must be in (0, 1]. Example: 0.5 selects the top 50%.")
        selected_count = int(np.ceil(S.shape[1] * p))
    else:
        selected_count = int(top_k_symptoms)
    selected_count = max(1, min(int(selected_count), S.shape[1]))

    selected = score_df.head(selected_count)["symptom_index"].astype(int).tolist()
    r_selected_symptoms = normalize01(S[:, selected].sum(axis=1)) if selected else np.zeros(S.shape[0], dtype=float)
    mu = np.exp(-beta_symptom * r_selected_symptoms)

    score_df = score_df.copy()
    score_df["selected"] = score_df["symptom_index"].isin(selected)

    eval_info = {
        "selected_indices": selected,
        "selected_names": [SYMPTOM_NAMES[i] for i in selected],
        "true_covid_indices": COVID_SYMPTOM_INDICES,
        "true_covid_names": [SYMPTOM_NAMES[i] for i in COVID_SYMPTOM_INDICES],
        "num_correct_selected": len(set(selected) & set(COVID_SYMPTOM_INDICES)),
        "selected_count": int(selected_count),
        "top_k_symptoms": None if top_percent_symptoms is not None else int(top_k_symptoms),
        "top_percent_symptoms": None if top_percent_symptoms is None else float(top_percent_symptoms),
        "score_components_enabled": {"R": bool(use_R), "C": bool(use_C), "U": bool(use_U)},
        "c_edge_min_weight": float(c_edge_min_weight),
        "score_formula": "+".join([name for name, enabled in [("R", use_R), ("C", use_C), ("U", use_U)] if enabled]) or "none",
        "C_definition": "q_m - p_m on strong-contact edges W_ij >= c_edge_min_weight; q_m is weighted neighbor symptom share among symptom carriers and p_m is population prevalence",
    }
    return mu, score_df, eval_info

def compute_prior_methods(
    dataset,
    beta_symptom=1.0,
    graph_weight=1.0,
    graph_normalization="neighbor_symptom_sum_div_max_symptoms",
    clip_graph_score=False,
    **kwargs,
):
    """
    Create the 3 simplified comparison methods.

      1. all_one
         mu_i = 1

      2. symptom_count
         Uses only the number of checked symptom items.
         symptom_score_i = symptom_count_i / number_of_symptoms
         mu_i = exp(- beta_symptom * symptom_score_i)

      3. symptom_count_plus_graph
         Uses own symptom count plus graph information.

         Default after the A-design change:
           neighbor_symptom_score_i = sum_j W_ij * symptom_count_j / number_of_symptoms

         This is intentionally not divided by the number of neighbors or by sum_j W_ij,
         because many connected symptomatic neighbors should increase risk rather than be averaged away.

         Backward-compatible option:
           graph_normalization="weighted_neighbor_average"
           neighbor_symptom_score_i = sum_j W_ij * symptom_score_j / sum_j W_ij

         mu_i = exp(- beta_symptom * combined_score_i)

    Smaller mu means higher priority in sparse reconstruction / inspection.
    """
    S = np.asarray(dataset["symptom_mat"], dtype=float)
    W = np.asarray(dataset.get("W_type_weighted", dataset["W"]), dtype=float).copy()
    np.fill_diagonal(W, 0.0)

    n_symptoms = max(S.shape[1], 1)
    symptom_count = S.sum(axis=1)
    symptom_score = np.clip(symptom_count / float(n_symptoms), 0.0, 1.0)

    row_sum = W.sum(axis=1)
    neighbor_symptom_score = np.zeros_like(symptom_score, dtype=float)
    mask = row_sum > 0
    graph_normalization = str(graph_normalization)
    if graph_normalization == "weighted_neighbor_average":
        neighbor_symptom_score[mask] = (W[mask] @ symptom_score) / row_sum[mask]
        neighbor_symptom_score = np.clip(neighbor_symptom_score, 0.0, 1.0)
    elif graph_normalization == "neighbor_symptom_sum_div_max_symptoms":
        # New r_graph design: keep neighbor count/contact intensity information.
        # Equivalent to sum_j W_ij * symptom_count_j / 3 when there are 3 symptoms.
        neighbor_symptom_score = (W @ symptom_count) / float(n_symptoms)
        if clip_graph_score:
            neighbor_symptom_score = np.clip(neighbor_symptom_score, 0.0, 1.0)
    else:
        raise ValueError(
            "graph_normalization must be 'neighbor_symptom_sum_div_max_symptoms' "
            "or 'weighted_neighbor_average'"
        )

    combined_score = symptom_score + float(graph_weight) * neighbor_symptom_score

    priors = {
        "all_one": np.ones(len(symptom_score), dtype=float),
        "symptom_count": np.exp(-float(beta_symptom) * symptom_score),
        "symptom_count_plus_graph": np.exp(-float(beta_symptom) * combined_score),
    }

    score_df = pd.DataFrame({
        "person_index": np.arange(len(symptom_score), dtype=int),
        "person_id": dataset["patient_data"]["person_id"].astype(int).to_numpy() if "person_id" in dataset["patient_data"].columns else np.arange(len(symptom_score), dtype=int),
        "y_true": dataset["patient_data"]["y_true"].astype(int).to_numpy() if "y_true" in dataset["patient_data"].columns else np.nan,
        "symptom_count": symptom_count.astype(int),
        "symptom_score": symptom_score,
        "neighbor_symptom_score": neighbor_symptom_score,
        "combined_score": combined_score,
        "neighbor_weight_sum": row_sum,
        "graph_normalization": graph_normalization,
        "mu_all_one": priors["all_one"],
        "mu_symptom_count": priors["symptom_count"],
        "mu_symptom_count_plus_graph": priors["symptom_count_plus_graph"],
    })

    selection_eval = {
        "method_count": len(priors),
        "symptom_count": int(n_symptoms),
        "graph_weight": float(graph_weight),
        "beta_symptom": float(beta_symptom),
        "graph_normalization": graph_normalization,
        "clip_graph_score": bool(clip_graph_score),
        "method_definitions": {
            "all_one": "uniform prior: mu_i = 1",
            "symptom_count": "symptom-count prior: mu_i = exp(-beta * symptom_count_i / 3)",
            "symptom_count_plus_graph": "proposed graph prior: mu_i = exp(-beta * (symptom_count_i/3 + graph_weight * sum_j W_ij * symptom_count_j/3))",
        },
        "note": "All symptoms are unnamed COVID-derived checkbox items; negatives have no symptoms by construction.",
    }
    return priors, score_df.reset_index(drop=True), selection_eval

def compute_detection_tables(dataset, priors, pool_count=None):
    import function_poolsizefittiing as fn
    if pool_count is None:
        pool_count = len(dataset["pools"])
    priority_table = fn.compute_required_tests_by_priority(dataset, priors)
    return priority_table


def create_9panel_figure(dataset, priors, sparse_summary=None, symptom_score_df=None, output_path=None):
    import matplotlib.pyplot as plt
    df = dataset["patient_data"]
    W = dataset["W"]
    fig, axes = plt.subplots(3,3,figsize=(16,13))
    fig.patch.set_facecolor("white")
    ax=axes[0,0]; ax.axis("off")
    txt=f"n={len(df)}\npositives={int(df['y_true'].sum())}\npositive_rate={df['y_true'].mean():.3f}\nedges={int(np.sum(W>0)//2)}"
    ax.text(0.05,0.9,txt,va="top",fontsize=12); ax.set_title("Overview")
    ax=axes[0,1]; df["reported_total_symptom_count"].value_counts().sort_index().plot(kind="bar", ax=ax); ax.set_title("Symptom count")
    ax=axes[0,2]; df.groupby("reported_total_symptom_count")["y_true"].mean().plot(kind="bar", ax=ax); ax.set_title("Positive rate by symptoms")
    ax=axes[1,0]
    if symptom_score_df is not None:
        tmp=symptom_score_df.sort_values("total_score", ascending=False); ax.bar(tmp["symptom_name"], tmp["total_score"]); ax.tick_params(axis='x',rotation=70); ax.set_title("Symptom selection score")
    ax=axes[1,1]; deg=(W>0).sum(axis=1); ax.hist(deg,bins=20); ax.set_title("Degree distribution")
    ax=axes[1,2]; ax.hist(np.log10(np.maximum(dataset["x_true"][dataset["x_true"]>0],1)), bins=15); ax.set_title("log10 RNA positives")
    ax=axes[2,0]
    if sparse_summary is not None and len(sparse_summary):
        tmp=sparse_summary.sort_values("inspection_count"); ax.bar(tmp["mode"], tmp["inspection_count"]); ax.tick_params(axis='x',rotation=55, labelsize=8); ax.set_title("Individual tests")
    ax=axes[2,1]
    if sparse_summary is not None and len(sparse_summary):
        tmp=sparse_summary.sort_values("total_test_cost"); ax.bar(tmp["mode"], tmp["total_test_cost"]); ax.tick_params(axis='x',rotation=55, labelsize=8); ax.set_title("Total test cost")
    ax=axes[2,2]
    if sparse_summary is not None and "detected_recall_by_individual_qpcr" in sparse_summary.columns:
        tmp=sparse_summary.sort_values("total_test_cost"); ax.bar(tmp["mode"], tmp["detected_recall_by_individual_qpcr"]); ax.tick_params(axis='x',rotation=55, labelsize=8); ax.set_ylim(0,1.05); ax.set_title("Detected recall")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    return fig

# Backward-compatible alias used by earlier notebooks
export_samples = export_graph_samples

# =====================================================================
# Final overrides: clustered sample + D-lite symptom generation
# =====================================================================

# Positive labels are no longer reassigned here. Positive clustering is created
# upstream by export_graph_samples(), so analysis uses the sampled population as-is.

def _generate_symptoms(
    pop,
    W,
    strength=0.95,
    seed=472,
    force=False,
    strong_edge_threshold=2.0,
    neg_covid_leak_prob=0.02,
    off_pattern_covid_prob=0.10,
    noncovid_base_prob=0.10,
    pos_noncovid_prob=0.10,
    target_positive_symptomatic_rate=0.92,
    max_symptoms=4,
):
    """D-lite symptom generator.

    Observed symptom = COVID-derived component OR non-COVID background component.

    - COVID-derived symptoms: fever, cough, sore_throat.
    - Non-COVID background symptoms: headache, runny_nose, fatigue.
    - About 75% of positives have at least one COVID-derived symptom.
    - About 10% of the full population has at least one non-COVID background symptom.
    - COVID-derived symptom patterns are locally shared within positive contact clusters.
    - Non-COVID background symptoms are spread independently of infection status and graph locality.
    """
    existing = [c for c in pop.columns if c.startswith("reported_symptom_")]
    if existing and not force:
        return pop

    rng = np.random.default_rng(seed + 77)
    out = pop.copy().reset_index(drop=True)
    n = len(out)
    y = out["y_true"].astype(int).to_numpy()
    S_covid = np.zeros((n, len(SYMPTOM_NAMES)), dtype=int)
    S_bg = np.zeros((n, len(SYMPTOM_NAMES)), dtype=int)

    share_prob = float(np.clip(strength, 0.0, 1.0))
    target_positive_symptomatic_rate = float(np.clip(target_positive_symptomatic_rate, 0.0, 1.0))
    neg_covid_prob = float(np.clip(neg_covid_leak_prob, 0.0, 1.0))
    noncovid_any_prob = float(np.clip(noncovid_base_prob, 0.0, 1.0))
    pos_noncovid_any_prob = float(np.clip(pos_noncovid_prob, 0.0, 1.0))

    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    W_strong = np.where(W >= float(strong_edge_threshold), W, 0.0)

    pos_mask = (y == 1)
    W_pos = W_strong.copy()
    W_pos[~pos_mask, :] = 0.0
    W_pos[:, ~pos_mask] = 0.0
    comps = [c for c in _components_from_W(W_pos) if any(pos_mask[c])]
    node_to_comp = {i: cid for cid, comp in enumerate(comps) for i in comp}

    covid_patterns = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
    ], dtype=int)
    pattern_probs = np.array([0.35, 0.22, 0.16, 0.17, 0.10], dtype=float)
    pattern_probs = pattern_probs / pattern_probs.sum()
    comp_pattern = {
        cid: covid_patterns[int(rng.choice(len(covid_patterns), p=pattern_probs))]
        for cid, _ in enumerate(comps)
    }

    bg_probs = np.ones(len(NONCOVID_SYMPTOM_INDICES), dtype=float)
    bg_probs = bg_probs / bg_probs.sum()

    for i in range(n):
        if y[i] == 1:
            if rng.random() < target_positive_symptomatic_rate:
                base_pattern = comp_pattern.get(node_to_comp.get(i, -1), covid_patterns[0])
                if rng.random() < share_prob:
                    core = base_pattern.copy()
                    # Individual variation while preserving cluster-level similarity.
                    for k in range(len(COVID_SYMPTOM_INDICES)):
                        if core[k] == 1 and rng.random() < 0.12:
                            core[k] = 0
                        elif core[k] == 0 and rng.random() < float(np.clip(off_pattern_covid_prob, 0.0, 1.0)):
                            core[k] = 1
                    if core.sum() == 0:
                        candidates = np.where(base_pattern == 1)[0]
                        if len(candidates) == 0:
                            candidates = np.arange(3)
                        core[int(rng.choice(candidates))] = 1
                else:
                    k = int(rng.choice([1, 2, 3], p=[0.25, 0.50, 0.25]))
                    core = np.zeros(3, dtype=int)
                    core[rng.choice(np.arange(3), size=k, replace=False)] = 1
                for local_k, symptom_idx in enumerate(COVID_SYMPTOM_INDICES):
                    S_covid[i, symptom_idx] = int(core[local_k])

            # Non-COVID background symptoms also occur in positives, independently.
            if rng.random() < pos_noncovid_any_prob:
                chosen = int(rng.choice(NONCOVID_SYMPTOM_INDICES, p=bg_probs))
                S_bg[i, chosen] = 1
                if rng.random() < 0.15:
                    rest = [idx for idx in NONCOVID_SYMPTOM_INDICES if idx != chosen]
                    S_bg[i, int(rng.choice(rest))] = 1

        else:
            # Rare COVID-like symptoms among negatives.
            for symptom_idx in COVID_SYMPTOM_INDICES:
                S_bg[i, symptom_idx] = int(rng.random() < neg_covid_prob)

            # Non-COVID background symptoms are spread broadly.
            if rng.random() < noncovid_any_prob:
                chosen = int(rng.choice(NONCOVID_SYMPTOM_INDICES, p=bg_probs))
                S_bg[i, chosen] = 1
                if rng.random() < 0.15:
                    rest = [idx for idx in NONCOVID_SYMPTOM_INDICES if idx != chosen]
                    S_bg[i, int(rng.choice(rest))] = 1

    S = np.maximum(S_covid, S_bg)

    if max_symptoms and max_symptoms > 0:
        for i in range(n):
            idx = np.where(S[i] == 1)[0].tolist()
            if len(idx) > max_symptoms:
                keep = rng.choice(idx, size=int(max_symptoms), replace=False)
                S[i, :] = 0
                S[i, keep] = 1
                S_covid[i, :] = S_covid[i, :] * S[i, :]
                S_bg[i, :] = S_bg[i, :] * S[i, :]

    for c in [c for c in out.columns if c.startswith("reported_symptom_") or c.startswith("internal_")]:
        out = out.drop(columns=c)
    for l, name in enumerate(SYMPTOM_NAMES):
        out[f"reported_symptom_{l}_{name}"] = S[:, l]
    out["reported_total_symptom_count"] = S.sum(axis=1)
    out["internal_covid_symptom_count"] = S_covid[:, COVID_SYMPTOM_INDICES].sum(axis=1)
    out["internal_noncovid_symptom_count"] = S_bg[:, NONCOVID_SYMPTOM_INDICES].sum(axis=1)
    out["internal_has_covid_derived_symptom"] = (out["internal_covid_symptom_count"] > 0).astype(int)
    out["internal_has_noncovid_background_symptom"] = (out["internal_noncovid_symptom_count"] > 0).astype(int)
    return out


def build_analysis_dataset(
    population,
    contacts,
    params,
    pool_size=10,
    cluster_symptom_strength=0.95,
    seed=472,
    force_symptom_regeneration=True,
    contact_type_weights=None,
    strong_edge_threshold=2.0,
    neg_covid_leak_prob=0.02,
    noncovid_base_prob=0.10,
    pos_noncovid_prob=0.10,
    target_positive_symptomatic_rate=0.75,
):
    import function_poolsizefittiing as fn
    pop = population.copy().reset_index(drop=True)
    if "person_id" not in pop.columns:
        pop["person_id"] = np.arange(len(pop))
    person_ids = pop["person_id"].astype(int).to_numpy()

    W_basic = build_weight_matrix(contacts, person_ids, mode="basic")
    W_type_weighted = build_weight_matrix(contacts, person_ids, mode="contact_type", contact_type_weights=contact_type_weights)

    pop = _generate_symptoms(
        pop,
        W_type_weighted,
        strength=cluster_symptom_strength,
        seed=seed,
        force=force_symptom_regeneration,
        strong_edge_threshold=strong_edge_threshold,
        neg_covid_leak_prob=neg_covid_leak_prob,
        noncovid_base_prob=noncovid_base_prob,
        pos_noncovid_prob=pos_noncovid_prob,
        target_positive_symptomatic_rate=target_positive_symptomatic_rate,
    )
    params = fn.derive_params({**params, "n": len(pop), "pool_size": pool_size})
    pop, x_true = _assign_rna(pop, W_type_weighted, params, seed=seed)
    n = len(pop)
    # Pool-size sweep uses pool_size=1..30. If n is not divisible by pool_size,
    # the last pool is allowed to be smaller.
    A, pools = fn.make_pooling_matrix(n, pool_size=pool_size, gaps=params["gaps"], allow_incomplete_last_pool=True)
    pooled_amount_true, pooled_ct, pooled_amount_est = fn.pooled_measurements_qpcr(A, x_true, params)
    symptom_cols = [c for c in pop.columns if c.startswith("reported_symptom_")]
    symptom_mat = pop[symptom_cols].to_numpy(dtype=int)
    return {
        "params": params,
        "patient_data": pop,
        "contacts": contacts.copy(),
        "person_ids": person_ids,
        "W": W_type_weighted,
        "W_basic": W_basic,
        "W_type_weighted": W_type_weighted,
        "contact_type_summary": summarize_contact_types(contacts, contact_type_weights=contact_type_weights),
        "symptom_mat": symptom_mat,
        "symptom_cols": symptom_cols,
        "symptom_names": SYMPTOM_NAMES.copy(),
        "true_covid_mask": np.array([1, 1, 1] + [0] * 3, dtype=int),
        "y_true": pop["y_true"].astype(int).to_numpy(),
        "x_true": x_true,
        "A": A,
        "pools": pools,
        "pooled_amount_true": pooled_amount_true,
        "pooled_ct": pooled_ct,
        "pooled_amount_est": pooled_amount_est,
    }


def create_research_summary_figure(dataset, priors, sparse_summary=None, symptom_score_df=None, output_path=None):
    """Create one consolidated figure containing all key panels."""
    import matplotlib.pyplot as plt
    df = dataset["patient_data"]
    W = dataset["W"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.patch.set_facecolor("white")

    # Overview
    ax = axes[0, 0]
    ax.axis("off")
    zeros = int((df["reported_total_symptom_count"] == 0).sum())
    txt = (
        f"n={len(df)}\n"
        f"positives={int(df['y_true'].sum())}\n"
        f"positive_rate={df['y_true'].mean():.3f}\n"
        f"asymptomatic={zeros} ({zeros/len(df):.3f})\n"
        f"edges={int(np.sum(W > 0)//2)}"
    )
    ax.text(0.05, 0.92, txt, va="top", fontsize=12)
    ax.set_title("Overview")

    # Symptom count
    ax = axes[0, 1]
    df["reported_total_symptom_count"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Symptom count")
    ax.set_xlabel("reported_total_symptom_count")

    # Positive rate by symptom count
    ax = axes[0, 2]
    df.groupby("reported_total_symptom_count")["y_true"].mean().plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Positive rate by symptoms")

    # R/C/U grouped bars
    ax = axes[1, 0]
    if symptom_score_df is not None and len(symptom_score_df):
        tmp = symptom_score_df.sort_values("total_score", ascending=False).reset_index(drop=True)
        x = np.arange(len(tmp))
        width = 0.25
        ax.bar(x - width, tmp["weighted_risk_mean_R"], width=width, label="R")
        ax.bar(x, tmp["local_excess_consistency_C"], width=width, label="C=q-p")
        ax.bar(x + width, tmp["specificity_U"], width=width, label="U")
        ax.axhline(0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(tmp["symptom_name"], rotation=55, ha="right")
        ax.legend(fontsize=8)
    ax.set_title("Diagnostic components R/C/U")

    # C ranking
    ax = axes[1, 1]
    if symptom_score_df is not None and len(symptom_score_df):
        tmp = symptom_score_df.sort_values("total_score", ascending=False).reset_index(drop=True)
        ax.bar(tmp["symptom_name"], tmp["local_excess_consistency_C"])
        ax.axhline(0, linewidth=1)
        for i, row in tmp.iterrows():
            if bool(row.get("selected", False)):
                y = float(row["local_excess_consistency_C"])
                ax.text(i, y + (0.01 if y >= 0 else -0.01), "selected", ha="center", va="bottom" if y >= 0 else "top", rotation=90, fontsize=8)
        ax.tick_params(axis="x", rotation=55)
    ax.set_title("C ranking by symptom")

    # Selected symptoms table
    ax = axes[1, 2]
    ax.axis("off")
    if symptom_score_df is not None and len(symptom_score_df):
        tmp = symptom_score_df.sort_values("total_score", ascending=False).copy()
        show = tmp[["symptom_name", "selected", "prevalence_p", "neighbor_share_q", "local_excess_consistency_C", "weighted_risk_mean_R"]].copy()
        show["prevalence_p"] = show["prevalence_p"].map(lambda v: f"{v:.3f}")
        show["neighbor_share_q"] = show["neighbor_share_q"].map(lambda v: f"{v:.3f}")
        show["local_excess_consistency_C"] = show["local_excess_consistency_C"].map(lambda v: f"{v:.3f}")
        show["weighted_risk_mean_R"] = show["weighted_risk_mean_R"].map(lambda v: f"{v:.3f}")
        table = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.2)
    ax.set_title("Symptom score table")

    # Individual tests
    ax = axes[2, 0]
    if sparse_summary is not None and len(sparse_summary):
        tmp = sparse_summary.sort_values("inspection_count")
        ax.bar(tmp["mode"], tmp["inspection_count"])
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_title("Individual tests")

    # Total test cost
    ax = axes[2, 1]
    if sparse_summary is not None and len(sparse_summary):
        tmp = sparse_summary.sort_values("total_test_cost")
        ax.bar(tmp["mode"], tmp["total_test_cost"])
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_title("Total test cost")

    # Detected recall
    ax = axes[2, 2]
    if sparse_summary is not None and len(sparse_summary) and "detected_recall_by_individual_qpcr" in sparse_summary.columns:
        tmp = sparse_summary.sort_values("total_test_cost")
        ax.bar(tmp["mode"], tmp["detected_recall_by_individual_qpcr"])
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_title("Detected recall")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    return fig

# Backward-compatible name used by the notebook.
def create_9panel_figure(dataset, priors, sparse_summary=None, symptom_score_df=None, output_path=None):
    return create_research_summary_figure(dataset, priors, sparse_summary=sparse_summary, symptom_score_df=symptom_score_df, output_path=output_path)

# ============================================================
# Company/workplace sample extraction override
# ============================================================

def _normalize_contact_columns_for_company_sampling(contacts: pd.DataFrame) -> pd.DataFrame:
    con = contacts.copy()
    rename = {}
    if "person_i" not in con.columns:
        for c in ["ID_1", "id_1", "person1", "from", "i"]:
            if c in con.columns:
                rename[c] = "person_i"; break
    if "person_j" not in con.columns:
        for c in ["ID_2", "id_2", "person2", "to", "j"]:
            if c in con.columns:
                rename[c] = "person_j"; break
    if rename:
        con = con.rename(columns=rename)
    if not {"person_i", "person_j"}.issubset(con.columns):
        raise ValueError(f"contacts must include person_i/person_j-like columns. columns={con.columns.tolist()}")
    con["person_i"] = con["person_i"].astype(int)
    con["person_j"] = con["person_j"].astype(int)
    if "contact_type" not in con.columns:
        if "type" in con.columns:
            con = con.rename(columns={"type": "contact_type"})
        else:
            con["contact_type"] = 1
    return con


def _normalize_population_columns_for_company_sampling(population: pd.DataFrame) -> pd.DataFrame:
    pop = population.copy()
    if "person_id" not in pop.columns:
        for c in ["ID", "id", "person", "node_id"]:
            if c in pop.columns:
                pop = pop.rename(columns={c: "person_id"}); break
    if "person_id" not in pop.columns:
        pop["person_id"] = np.arange(len(pop), dtype=int)
    pop["person_id"] = pop["person_id"].astype(int)
    if "age_group" not in pop.columns:
        # If age is available, convert decades to OpenABM-like age groups.
        if "age" in pop.columns:
            pop["age_group"] = (pop["age"].astype(float) // 10).astype(int)
        else:
            raise ValueError("population must include age_group or age for company/workplace sampling")
    if "y_true" not in pop.columns:
        for c in ["infected", "is_positive", "positive", "disease_status"]:
            if c in pop.columns:
                pop = pop.rename(columns={c: "y_true"}); break
    if "y_true" not in pop.columns:
        raise ValueError("population must include y_true or equivalent positive-label column")
    pop["y_true"] = pop["y_true"].astype(int)
    return pop


def _positive_component_diagnostics(sample: pd.DataFrame, contacts: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    pos_ids = set(sample.loc[sample["y_true"].astype(int) == 1, "person_id"].astype(int).tolist())
    pp = contacts[contacts["person_i"].isin(pos_ids) & contacts["person_j"].isin(pos_ids)].copy()
    adj = defaultdict(set)
    for a, b in zip(pp["person_i"].astype(int), pp["person_j"].astype(int)):
        if int(a) == int(b):
            continue
        adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    comps = _connected_components_from_adj(adj, pos_ids) if pos_ids else []
    comp_id = {}
    for k, comp in enumerate(comps):
        for p in comp:
            comp_id[int(p)] = k
    rows = []
    for p in sorted(pos_ids):
        neigh = adj.get(int(p), set())
        cid = comp_id.get(int(p), -1)
        rows.append({
            "person_id": int(p),
            "positive_neighbor_count": int(len(neigh)),
            "has_positive_neighbor": int(len(neigh) > 0),
            "positive_component_id": int(cid),
            "positive_component_size": int(len(comps[cid])) if cid >= 0 else 1,
            "is_isolated_positive": int(len(neigh) == 0),
        })
    by_person = pd.DataFrame(rows)
    largest = max((len(c) for c in comps), default=0)
    connected_pos = int(sum(1 for p in pos_ids if len(adj.get(int(p), set())) > 0))
    diag = {
        "positive_positive_edges": int(len(pp)),
        "positive_with_positive_neighbor": connected_pos,
        "isolated_positive_count": int(len(pos_ids) - connected_pos),
        "positive_component_count": int(len(comps)),
        "positive_cluster_component_count": int(sum(1 for c in comps if len(c) >= 2)),
        "largest_positive_component": int(largest),
        "largest_positive_component_ratio": float(largest / max(len(pos_ids), 1)),
        "positive_neighbor_ratio": float(connected_pos / max(len(pos_ids), 1)),
    }
    return diag, by_person


def _sample_company_workplace_once(
    population_all: pd.DataFrame,
    contacts_all: pd.DataFrame,
    n: int,
    target_age_groups: list[int],
    workplace_contact_types: list[int],
    max_positive_rate: float,
    min_positive_count: int,
    target_positive_count_range: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    pop = _normalize_population_columns_for_company_sampling(population_all)
    con = _normalize_contact_columns_for_company_sampling(contacts_all)
    pop = pop[pop["age_group"].astype(int).isin([int(x) for x in target_age_groups])].copy()
    allowed_ids = set(pop["person_id"].astype(int).tolist())
    con = con[con["person_i"].isin(allowed_ids) & con["person_j"].isin(allowed_ids)].copy()
    con = con[con["contact_type"].astype(int).isin([int(x) for x in workplace_contact_types])].copy()

    if len(pop) < n:
        raise ValueError(f"Not enough eligible age/workplace population: eligible={len(pop)}, n={n}")

    ids_in_work_edges = set(con["person_i"].astype(int)).union(set(con["person_j"].astype(int)))
    eligible = pop[pop["person_id"].isin(ids_in_work_edges)].copy()
    if len(eligible) < n:
        # Keep old structure usable: if graph is sparse, fill from age-eligible people.
        eligible = pop.copy()

    pos_pool = eligible[eligible["y_true"].astype(int) == 1]["person_id"].astype(int).to_numpy()
    neg_pool = eligible[eligible["y_true"].astype(int) == 0]["person_id"].astype(int).to_numpy()
    if len(pos_pool) < min_positive_count:
        raise ValueError(f"Not enough positives in eligible company pool: positives={len(pos_pool)}, min={min_positive_count}")
    if len(neg_pool) < n - min_positive_count:
        raise ValueError(f"Not enough negatives in eligible company pool: negatives={len(neg_pool)}")

    high = min(int(target_positive_count_range[1]), int(np.floor(n * max_positive_rate)), len(pos_pool), n)
    low = min(max(int(target_positive_count_range[0]), int(min_positive_count)), high)
    if low > high:
        low = min(int(min_positive_count), high)
    k_pos = int(rng.integers(low, high + 1)) if high >= low else int(high)
    k_pos = max(min(k_pos, int(np.floor(n * max_positive_rate))), int(min_positive_count))
    k_pos = min(k_pos, len(pos_pool), n)

    # Prefer positive-positive workplace components so every selected positive is in a cluster.
    y_map = dict(zip(pop["person_id"].astype(int), pop["y_true"].astype(int)))
    pp_adj = defaultdict(set)
    for a, b in zip(con["person_i"].astype(int), con["person_j"].astype(int)):
        if y_map.get(int(a), 0) == 1 and y_map.get(int(b), 0) == 1:
            pp_adj[int(a)].add(int(b))
            pp_adj[int(b)].add(int(a))
    pp_nodes = set(int(x) for x in pos_pool if len(pp_adj.get(int(x), set())) > 0)
    pp_comps = [c for c in _connected_components_from_adj(pp_adj, pp_nodes) if len(c) >= 2]
    rng.shuffle(pp_comps)

    pos_selected = set()
    # Add whole components; this preserves positive-positive neighbors inside the sample.
    for comp in sorted(pp_comps, key=len, reverse=True):
        comp = [int(x) for x in comp]
        if len(pos_selected) >= low:
            break
        if len(pos_selected) + len(comp) <= high:
            pos_selected.update(comp)
    # If still short, add smaller components even if the count becomes slightly larger, capped by high.
    if len(pos_selected) < low:
        for comp in pp_comps:
            for pid0 in comp:
                if len(pos_selected) < high:
                    pos_selected.add(int(pid0))
            if len(pos_selected) >= low:
                break

    if len(pos_selected) < int(min_positive_count):
        # Fallback only when the source data does not contain enough clustered positives.
        pos_selected = set(rng.choice(pos_pool, size=k_pos, replace=False).astype(int).tolist())

    neg_needed = n - len(pos_selected)
    neg_selected = set(rng.choice(neg_pool, size=neg_needed, replace=False).astype(int).tolist())
    selected = pos_selected | neg_selected
    sample = pop[pop["person_id"].isin(selected)].copy()
    sample["sample_source"] = np.where(sample["person_id"].isin(pos_selected), "selected_positive", "selected_negative")
    sample = sample.sample(frac=1, random_state=int(rng.integers(1_000_000_000))).reset_index(drop=True)
    ids = set(sample["person_id"].astype(int).tolist())
    sample_con = con[con["person_i"].isin(ids) & con["person_j"].isin(ids)].copy().reset_index(drop=True)
    diag, by_person = _positive_component_diagnostics(sample, sample_con)
    diag.update({
        "sample_size": int(len(sample)),
        "positive_count": int(sample["y_true"].astype(int).sum()),
        "positive_rate": float(sample["y_true"].astype(int).mean()),
        "contact_edges": int(len(sample_con)),
        "contact_types": ";".join(map(str, sorted(sample_con["contact_type"].astype(str).unique().tolist()))) if len(sample_con) else "",
        "age_groups": ";".join(map(str, sorted(sample["age_group"].astype(int).unique().tolist()))),
    })
    return sample, sample_con, diag, by_person


def export_graph_samples(
    population_all,
    contacts_all,
    output_dir,
    sample_sizes=(3000,),
    target_positive_rate=None,
    graph_core_ratio=0.70,
    seed=472,
    cluster_positive_ratio=0.70,
    target_age_groups=(2, 3, 4, 5),
    workplace_contact_types=(1,),
    max_positive_rate=0.05,
    min_positive_count=30,
    target_positive_count_range=(50, 120),
    min_positive_components=3,
    min_isolated_positive=1,
    max_largest_positive_component_ratio=0.60,
    min_positive_neighbor_ratio=0.40,
    max_attempts=300,
    sample_name="company_n{n}_maxpos5pct_work",
):
    """Export company/workplace samples and diagnostics.

    The function name is kept for backward compatibility. Internally, this now
    samples an age-restricted workplace-contact subpopulation:
      - age_group in target_age_groups, default [2, 3, 4, 5]
      - contact_type in workplace_contact_types, default [1]
      - positive rate <= max_positive_rate
      - at least min_positive_count positives
      - multiple positive components, some isolated positives, and no oversized
        largest positive component when feasible.
    """
    output_dir = Path(output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for n0 in sample_sizes:
        n0 = int(n0)
        best = None
        best_score = -10**9
        for attempt in range(int(max_attempts)):
            sample, con, diag, by_person = _sample_company_workplace_once(
                population_all=population_all,
                contacts_all=contacts_all,
                n=n0,
                target_age_groups=list(target_age_groups),
                workplace_contact_types=list(workplace_contact_types),
                max_positive_rate=float(max_positive_rate),
                min_positive_count=int(min_positive_count),
                target_positive_count_range=tuple(target_positive_count_range),
                rng=rng,
            )
            ok = (
                diag["positive_rate"] <= float(max_positive_rate) + 1e-12
                and diag["positive_count"] >= int(min_positive_count)
                and diag["positive_component_count"] >= int(min_positive_components)
                and diag["isolated_positive_count"] >= int(min_isolated_positive)
                and diag["largest_positive_component_ratio"] <= float(max_largest_positive_component_ratio)
                and diag["positive_neighbor_ratio"] >= float(min_positive_neighbor_ratio)
            )
            score = 0
            score += min(diag["positive_component_count"], int(min_positive_components)) * 10
            score += min(diag["isolated_positive_count"], int(min_isolated_positive)) * 10
            score += int(100 * min(diag["positive_neighbor_ratio"], float(min_positive_neighbor_ratio)))
            score -= int(100 * max(0.0, diag["largest_positive_component_ratio"] - float(max_largest_positive_component_ratio)))
            if score > best_score:
                best_score = score
                best = (sample, con, diag, by_person, attempt + 1, ok)
            if ok:
                break
        sample, con, diag, by_person, attempts_used, conditions_met = best
        sample_dir = samples_dir / sample_name.format(n=n0)
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample.to_csv(sample_dir / "population.csv", index=False)
        con.to_csv(sample_dir / "contacts.csv", index=False)
        by_person.to_csv(sample_dir / "positive_neighbor_by_positive_person.csv", index=False)
        row = dict(diag)
        row.update({
            "path": str(sample_dir),
            "attempts_used": int(attempts_used),
            "sampling_conditions_met": bool(conditions_met),
            "target_age_groups": ";".join(map(str, target_age_groups)),
            "workplace_contact_types": ";".join(map(str, workplace_contact_types)),
            "max_positive_rate_condition": float(max_positive_rate),
            "min_positive_count_condition": int(min_positive_count),
        })
        pd.DataFrame([row]).to_csv(sample_dir / "sample_summary.csv", index=False)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "generation_summary.csv", index=False)
    return summary


def load_sample_from_root(data_root, sample_size=3000, target_positive_rate=None, sample_name=None):
    data_root = Path(data_root)
    candidates = []
    if sample_name is not None:
        candidates += [data_root / "samples" / sample_name, data_root / sample_name]
    candidates += [
        data_root / "samples" / f"company_n{int(sample_size)}_maxpos5pct_work",
        data_root / f"company_n{int(sample_size)}_maxpos5pct_work",
    ]
    if target_positive_rate is not None:
        pct = int(round(float(target_positive_rate) * 100))
        candidates += [
            data_root / "samples" / f"n{int(sample_size)}_pos{pct}pct_graph",
            data_root / f"n{int(sample_size)}_pos{pct}pct_graph",
        ]
    for d in candidates:
        if (d / "population.csv").exists() and (d / "contacts.csv").exists():
            return pd.read_csv(d / "population.csv"), pd.read_csv(d / "contacts.csv"), d
    pops = [p for p in data_root.rglob("population.csv") if str(sample_size) in str(p) or "company" in str(p)]
    if not pops:
        raise FileNotFoundError(f"population.csv for sample_size={sample_size} not found under {data_root}")
    d = pops[0].parent
    return pd.read_csv(d / "population.csv"), pd.read_csv(d / "contacts.csv"), d


# ============================================================
# Source-data generation and improved company/workplace sampling
# ============================================================

def generate_synthetic_company_source_data(
    n_total: int = 10000,
    seed: int = 472,
    output_dir=None,
    positive_prevalence: float = 0.04,
    company_size_range: tuple[int, int] = (60, 220),
    within_company_edge_prob: float = 0.010,
    positive_cluster_count: int = 50,
    positive_cluster_size_range: tuple[int, int] = (4, 9),
    isolated_positive_count: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a reproducible OpenABM-like source CSV pair for this notebook.

    This is a simulation-only generator used when the notebook is run without an
    existing OpenABM build/output. It creates:
      - population_all.csv-like population data
      - contacts_all.csv-like contact data

    contact_type convention used here:
      0 = household-like contact
      1 = workplace/company contact
      2 = random/community contact

    export_graph_samples() later filters this source data to age_group 2-5 and
    contact_type 1, then samples n=3000 with <=5% positives.
    """
    rng = np.random.default_rng(int(seed))
    n_total = int(n_total)
    person_ids = np.arange(n_total, dtype=int)

    # age_group 2-5 are working-age targets, but keep some non-target groups
    age_groups = rng.choice([1, 2, 3, 4, 5, 6], size=n_total, p=[0.08, 0.18, 0.24, 0.24, 0.18, 0.08])

    # Assign company IDs, mostly for age_group 2-5.
    company_id = np.full(n_total, -1, dtype=int)
    working_ids = person_ids[np.isin(age_groups, [2, 3, 4, 5])]
    rng.shuffle(working_ids)
    companies = []
    pos = 0
    cid = 0
    while pos < len(working_ids):
        size = int(rng.integers(company_size_range[0], company_size_range[1] + 1))
        members = working_ids[pos:pos + size]
        if len(members) == 0:
            break
        company_id[members] = cid
        companies.append(members.astype(int))
        cid += 1
        pos += size

    y_true = np.zeros(n_total, dtype=int)
    working_set = set(working_ids.astype(int).tolist())

    # Workplace positive clusters: positive members connected by type-1 edges.
    forced_work_edges = []
    used_pos = set()
    candidate_companies = [m for m in companies if len(m) >= positive_cluster_size_range[1] + 5]
    rng.shuffle(candidate_companies)
    for members in candidate_companies[:int(positive_cluster_count)]:
        k = int(rng.integers(positive_cluster_size_range[0], positive_cluster_size_range[1] + 1))
        chosen = rng.choice(members, size=min(k, len(members)), replace=False).astype(int).tolist()
        for pid in chosen:
            y_true[pid] = 1
            used_pos.add(pid)
        # chain + a few extra edges so the positive subgraph has clusters
        for a, b in zip(chosen[:-1], chosen[1:]):
            forced_work_edges.append((int(a), int(b), 1, 1.0))
        if len(chosen) >= 4:
            forced_work_edges.append((int(chosen[0]), int(chosen[-1]), 1, 1.0))

    # Add isolated positives from working-age people; avoid workplace-positive links where possible.
    remaining_work = np.array([pid for pid in working_ids.astype(int) if pid not in used_pos], dtype=int)
    if len(remaining_work) > 0:
        iso_take = min(int(isolated_positive_count), len(remaining_work))
        iso = rng.choice(remaining_work, size=iso_take, replace=False).astype(int)
        y_true[iso] = 1
        used_pos.update(iso.tolist())

    # Fill prevalence with random positives if needed.
    target_pos = int(round(n_total * float(positive_prevalence)))
    if y_true.sum() < target_pos:
        remaining = np.array([pid for pid in person_ids if y_true[pid] == 0], dtype=int)
        extra = rng.choice(remaining, size=min(target_pos - int(y_true.sum()), len(remaining)), replace=False)
        y_true[extra] = 1

    pop = pd.DataFrame({
        "person_id": person_ids,
        "day": 60,
        "age_group": age_groups.astype(int),
        "company_id": company_id.astype(int),
        "openabm_status_code": np.where(y_true == 1, "infected", "susceptible"),
        "y_true": y_true.astype(int),
    })

    edges = []
    seen = set()
    def add_edge(a, b, ctype, weight=1.0):
        a, b = int(a), int(b)
        if a == b:
            return
        if a > b:
            a, b = b, a
        key = (a, b, int(ctype))
        if key in seen:
            return
        seen.add(key)
        edges.append({"day": 60, "person_i": a, "person_j": b, "contact_type": int(ctype), "weight": float(weight)})

    for a, b, ctype, weight in forced_work_edges:
        add_edge(a, b, ctype, weight)

    # Regular workplace edges inside companies.
    for members in companies:
        members = np.asarray(members, dtype=int)
        m = len(members)
        if m <= 1:
            continue
        expected = max(m - 1, int(m * (m - 1) / 2 * float(within_company_edge_prob)))
        # Connect each company weakly with a random chain, then add sparse random contacts.
        shuffled = members.copy()
        rng.shuffle(shuffled)
        for a, b in zip(shuffled[:-1], shuffled[1:]):
            if rng.random() < 0.18:
                add_edge(a, b, 1, 1.0)
        for _ in range(expected):
            a, b = rng.choice(members, size=2, replace=False)
            add_edge(a, b, 1, 1.0)

    # Household-like contacts and random contacts, which company sampling will exclude.
    for start in range(0, n_total, 3):
        hh = person_ids[start:min(start + int(rng.integers(2, 5)), n_total)]
        for a, b in zip(hh[:-1], hh[1:]):
            add_edge(a, b, 0, 1.0)
    for _ in range(max(1, n_total // 3)):
        a, b = rng.choice(person_ids, size=2, replace=False)
        add_edge(a, b, 2, 1.0)

    con = pd.DataFrame(edges)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pop.to_csv(out / "population_all.csv", index=False)
        con.to_csv(out / "contacts_all.csv", index=False)
    return pop, con


def _select_positive_ids_with_clusters(pos_pool, con, k_pos, rng, min_positive_neighbor_ratio=1.0, min_isolated_positive=0):
    """Select positives from positive-positive contact components.

    When min_positive_neighbor_ratio is 1.0, this function avoids isolated
    positives unless there are not enough clustered positives in the source.
    """
    pos_set_all = set(int(x) for x in pos_pool)
    pp = con[con["person_i"].isin(pos_set_all) & con["person_j"].isin(pos_set_all)]
    adj = defaultdict(set)
    for a, b in zip(pp["person_i"].astype(int), pp["person_j"].astype(int)):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    comps = _connected_components_from_adj(adj, set(adj.keys()))
    cluster_comps = [list(map(int, c)) for c in comps if len(c) >= 2]
    rng.shuffle(cluster_comps)
    cluster_comps = sorted(cluster_comps, key=len, reverse=True)

    selected = []
    selected_set = set()
    # Prefer whole components so every selected positive keeps a positive neighbor.
    for comp in cluster_comps:
        if len(selected) >= k_pos:
            break
        if len(selected) + len(comp) <= k_pos:
            selected.extend(comp)
            selected_set.update(comp)
    if len(selected) < k_pos:
        # Add pairs or larger chunks from remaining components. This may exceed exact
        # component completeness, but still preserves at least one positive neighbor
        # for the newly selected nodes whenever take >= 2.
        for comp in cluster_comps:
            rest = [x for x in comp if x not in selected_set]
            if len(rest) < 2:
                continue
            rng.shuffle(rest)
            remaining = k_pos - len(selected)
            if remaining <= 0:
                break
            take = min(len(rest), remaining)
            if take == 1 and len(rest) >= 2:
                take = 2 if len(selected) + 2 <= k_pos else 0
            if take >= 2:
                selected.extend(rest[:take])
                selected_set.update(rest[:take])
            if len(selected) >= k_pos:
                break

    # Fallback only if the source has too few clustered positives.
    if len(selected) < k_pos and float(min_positive_neighbor_ratio) < 1.0:
        remaining = [int(x) for x in pos_pool if int(x) not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[:max(0, k_pos - len(selected))])

    return set(int(x) for x in selected[:k_pos])

def _sample_company_workplace_once(
    population_all: pd.DataFrame,
    contacts_all: pd.DataFrame,
    n: int,
    target_age_groups: list[int],
    workplace_contact_types: list[int],
    max_positive_rate: float,
    min_positive_count: int,
    target_positive_count_range: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Company/workplace sample extraction with cluster-aware positive selection."""
    pop = _normalize_population_columns_for_company_sampling(population_all)
    con = _normalize_contact_columns_for_company_sampling(contacts_all)
    pop = pop[pop["age_group"].astype(int).isin([int(x) for x in target_age_groups])].copy()
    allowed_ids = set(pop["person_id"].astype(int).tolist())
    con = con[con["person_i"].isin(allowed_ids) & con["person_j"].isin(allowed_ids)].copy()
    con = con[con["contact_type"].astype(int).isin([int(x) for x in workplace_contact_types])].copy()

    if len(pop) < n:
        raise ValueError(f"Not enough eligible age/workplace population: eligible={len(pop)}, n={n}")

    ids_in_work_edges = set(con["person_i"].astype(int)).union(set(con["person_j"].astype(int))) if len(con) else set()
    eligible = pop[pop["person_id"].isin(ids_in_work_edges)].copy() if ids_in_work_edges else pop.copy()
    if len(eligible) < n:
        eligible = pop.copy()

    pos_pool = eligible[eligible["y_true"].astype(int) == 1]["person_id"].astype(int).to_numpy()
    neg_pool = eligible[eligible["y_true"].astype(int) == 0]["person_id"].astype(int).to_numpy()
    if len(pos_pool) < min_positive_count:
        raise ValueError(f"Not enough positives in eligible company pool: positives={len(pos_pool)}, min={min_positive_count}")
    high = min(int(target_positive_count_range[1]), int(np.floor(n * max_positive_rate)), len(pos_pool), n)
    low = max(int(target_positive_count_range[0]), int(min_positive_count))
    low = min(low, high)
    k_pos = int(rng.integers(low, high + 1)) if high >= low else int(high)
    k_pos = max(min(k_pos, int(np.floor(n * max_positive_rate))), int(min_positive_count))
    k_pos = min(k_pos, len(pos_pool), n)
    if len(neg_pool) < n - k_pos:
        raise ValueError(f"Not enough negatives in eligible company pool: negatives={len(neg_pool)}, needed={n-k_pos}")

    pos_selected = _select_positive_ids_with_clusters(
        pos_pool=pos_pool,
        con=con,
        k_pos=k_pos,
        rng=rng,
        min_positive_neighbor_ratio=1.0,
        min_isolated_positive=0,
    )
    neg_needed = n - len(pos_selected)
    neg_selected = set(rng.choice(neg_pool, size=neg_needed, replace=False).astype(int).tolist())
    selected = pos_selected | neg_selected

    sample = pop[pop["person_id"].isin(selected)].copy()
    sample["sample_source"] = np.where(sample["person_id"].isin(pos_selected), "selected_positive", "selected_negative")
    sample = sample.sample(frac=1, random_state=int(rng.integers(1_000_000_000))).reset_index(drop=True)
    ids = set(sample["person_id"].astype(int).tolist())
    sample_con = con[con["person_i"].isin(ids) & con["person_j"].isin(ids)].copy().reset_index(drop=True)
    diag, by_person = _positive_component_diagnostics(sample, sample_con)
    diag.update({
        "sample_size": int(len(sample)),
        "positive_count": int(sample["y_true"].astype(int).sum()),
        "positive_rate": float(sample["y_true"].astype(int).mean()),
        "contact_edges": int(len(sample_con)),
        "contact_types": ";".join(map(str, sorted(sample_con["contact_type"].astype(str).unique().tolist()))) if len(sample_con) else "",
        "age_groups": ";".join(map(str, sorted(sample["age_group"].astype(int).unique().tolist()))),
    })
    return sample, sample_con, diag, by_person
