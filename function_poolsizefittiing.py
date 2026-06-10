"""
function_poolsizefittiing.py

PCR/qPCR・プール行列・重み付きスパース再構成・逐次個別検査の関数群。
地域・ワクチンは使わない完成版。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_default_params(n: int = 3000, pool_size: int = 10, max_rounds: int | None = None) -> dict:
    params = {
        "n": int(n),
        "pool_size": int(pool_size),
        "gaps": (1,),
        "max_rounds": None if max_rounds is None else int(max_rounds),
        "x_min_positive": 1e3,
        "x_max": 1e8,
        "max_cycles": 40,
        "pcr_efficiency": 2.0,
        "positive_cutoff": 1e3,
        "mu_eps": 1e-12,
        "solver": "linprog",
        "verbose_solver": False,
    }
    return derive_params(params)


def derive_params(params: dict) -> dict:
    out = dict(params)
    out.setdefault("n", 3000)
    out.setdefault("pool_size", 10)
    out.setdefault("gaps", (1,))
    if isinstance(out["gaps"], list):
        out["gaps"] = tuple(out["gaps"])
    out.setdefault("max_rounds", None)
    out.setdefault("x_min_positive", 1e3)
    out.setdefault("x_max", 1e8)
    out.setdefault("max_cycles", 40)
    out.setdefault("pcr_efficiency", 2.0)
    out.setdefault("positive_cutoff", out["x_min_positive"])
    out.setdefault("mu_eps", 1e-12)
    out.setdefault("solver", "linprog")
    out.setdefault("verbose_solver", False)
    out["num_measurements_per_patient"] = len(out["gaps"]) + 1
    out["sample_fraction"] = 1.0 / out["num_measurements_per_patient"]
    out["threshold_rna"] = out["x_min_positive"] * (out["pcr_efficiency"] ** out["max_cycles"])
    out["x_scale"] = float(np.sqrt(out["x_min_positive"] * out["x_max"]))
    return out


def make_pooling_matrix(
    n: int,
    pool_size: int = 10,
    gaps: tuple = (1,),
    allow_incomplete_last_pool: bool = True,
    pool_order=None,
):
    """Create a pooling matrix.

    Default behavior after the pool-size sweep update:
      - pool_size can be any integer from 1 to n.
      - if n is not divisible by pool_size, the last pool is smaller.

    This is necessary for comparing pool_size=1..30 when n=3000.
    """
    n = int(n)
    pool_size = int(pool_size)
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    if (not allow_incomplete_last_pool) and (n % pool_size != 0):
        raise ValueError(f"n={n} must be divisible by pool_size={pool_size}")
    if pool_order is None:
        order = np.arange(n, dtype=int)
    else:
        order = np.asarray(pool_order, dtype=int)
        if len(order) != n:
            raise ValueError(f"pool_order length must be n={n}, got {len(order)}")
        if len(np.unique(order)) != n or order.min(initial=0) < 0 or order.max(initial=-1) >= n:
            raise ValueError("pool_order must be a permutation of 0..n-1")

    pools = []
    starts = list(range(0, n, pool_size))
    for gap in gaps:
        gap = int(gap)
        for start in starts:
            pool = tuple(int(order[i]) for i in (start + gap * t for t in range(pool_size)) if 0 <= i < n)
            if pool:
                pools.append(pool)
    A = np.zeros((len(pools), n), dtype=float)
    for r, pool in enumerate(pools):
        A[r, list(pool)] = 1.0
    return A, pools


def amount_to_ct_continuous(amount: float, params: dict) -> float:
    if amount <= 0:
        return np.inf
    return float(np.log(params["threshold_rna"] / amount) / np.log(params["pcr_efficiency"]))


def ct_to_amount_point(ct: float, params: dict) -> float:
    if np.isinf(ct):
        return 0.0
    return float(params["threshold_rna"] / (params["pcr_efficiency"] ** ct))


def pooled_measurements_qpcr(A: np.ndarray, x_true: np.ndarray, params: dict):
    alpha = params["sample_fraction"]
    pooled_amount_true = alpha * (A @ np.asarray(x_true, dtype=float))
    pooled_ct = np.array([amount_to_ct_continuous(a, params) for a in pooled_amount_true])
    pooled_amount_est = np.array([ct_to_amount_point(ct, params) for ct in pooled_ct])
    return pooled_amount_true, pooled_ct, pooled_amount_est


def individual_measurement_qpcr(x_i_true: float, params: dict) -> dict:
    alpha = params["sample_fraction"]
    amount_true = alpha * float(x_i_true)
    ct = amount_to_ct_continuous(amount_true, params)
    amount_est = ct_to_amount_point(ct, params)
    return {
        "amount_true": amount_true,
        "ct": ct,
        "amount_est": amount_est,
        "x_est": amount_est / alpha,
    }


def _solve_linprog(dataset: dict, mu: np.ndarray, fixed: dict[int, float] | None = None) -> np.ndarray:
    from scipy.optimize import linprog
    if fixed is None:
        fixed = {}
    params = dataset["params"]
    A = np.asarray(dataset["A"], dtype=float)
    b = np.asarray(dataset["pooled_amount_est"], dtype=float) / params["x_scale"]
    alpha = params["sample_fraction"]
    n = A.shape[1]

    mu_norm = np.asarray(mu, dtype=float)
    med = float(np.median(mu_norm)) if np.median(mu_norm) > 0 else 1.0
    c = mu_norm / med

    Aeq = alpha * A.copy()
    beq = b.copy()
    bounds = [(0.0, params["x_max"] / params["x_scale"]) for _ in range(n)]
    for i, val in fixed.items():
        zval = float(val) / params["x_scale"]
        bounds[int(i)] = (zval, zval)

    res = linprog(c=c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        # 制約が数値的に厳しい場合は、微小な推定値を返すのではなく明示的に例外
        raise RuntimeError(f"linprog failed: {res.message}")
    return np.maximum(res.x * params["x_scale"], 0.0)


def _solve_cvxpy(dataset: dict, mu: np.ndarray, fixed: dict[int, float] | None = None) -> np.ndarray:
    import cvxpy as cp
    if fixed is None:
        fixed = {}
    params = dataset["params"]
    A = np.asarray(dataset["A"], dtype=float)
    pooled_amount_est = np.asarray(dataset["pooled_amount_est"], dtype=float)
    n = A.shape[1]
    alpha = params["sample_fraction"]
    x_scale = params["x_scale"]
    mu_norm = np.asarray(mu, dtype=float) / max(float(np.median(mu)), 1e-12)

    z = cp.Variable(n, nonneg=True)
    constraints = [z <= params["x_max"] / x_scale, alpha * (A @ z) == pooled_amount_est / x_scale]
    for i, val in fixed.items():
        constraints.append(z[int(i)] == float(val) / x_scale)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(mu_norm, z))), constraints)
    last_err = None
    for solver in ["CLARABEL", "SCS"]:
        try:
            prob.solve(solver=solver, verbose=False, warm_start=True)
            if prob.status in ("optimal", "optimal_inaccurate") and z.value is not None:
                return np.maximum(np.asarray(z.value).reshape(-1) * x_scale, 0.0)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"cvxpy failed: status={prob.status}, err={last_err}")


def solve_estimation_problem_with_mu(dataset: dict, mu: np.ndarray, fixed: dict[int, float] | None = None) -> dict:
    params = dataset["params"]
    try:
        if params.get("solver", "linprog") == "cvxpy":
            x_hat = _solve_cvxpy(dataset, mu, fixed)
            solver_used = "cvxpy"
        else:
            x_hat = _solve_linprog(dataset, mu, fixed)
            solver_used = "linprog"
    except Exception:
        # fallback順序を固定
        x_hat = _solve_linprog(dataset, mu, fixed)
        solver_used = "linprog"
    return {"x_hat": x_hat, "solver_used": solver_used, "mu_used": np.asarray(mu).copy()}


def run_sequential_sparse_reconstruction(dataset: dict, mu: np.ndarray, label: str, max_rounds: int | None = None) -> dict:
    params = dataset["params"]
    x_true = np.asarray(dataset["x_true"], dtype=float)
    n = len(x_true)
    budget = min(int(max_rounds if max_rounds is not None else n), n)
    true_positive_set = set(np.where(x_true >= params["x_min_positive"])[0].tolist())

    fixed: dict[int, float] = {}
    uninspected = set(range(n))
    detected_positive_indices: list[int] = []
    detected_set: set[int] = set()
    individual_measurements = []
    final_x_hat = np.zeros(n, dtype=float)

    for step in range(budget):
        sol = solve_estimation_problem_with_mu(dataset, mu=mu, fixed=fixed)
        x_hat = sol["x_hat"]
        final_x_hat = x_hat.copy()
        if not uninspected:
            break
        cand = max(uninspected, key=lambda i: x_hat[i])
        pred = float(x_hat[cand])
        meas = individual_measurement_qpcr(float(x_true[cand]), params)
        fixed[int(cand)] = float(meas["x_est"])
        is_pos = bool(x_true[cand] >= params["x_min_positive"])
        if is_pos and cand not in detected_set:
            detected_set.add(cand)
            detected_positive_indices.append(int(cand))
        uninspected.remove(cand)
        individual_measurements.append({
            "step": int(step),
            "candidate": int(cand),
            "pred_before_test": pred,
            "x_true": float(x_true[cand]),
            "x_est": float(meas["x_est"]),
            "ct": float(meas["ct"]),
            "is_true_positive": is_pos,
            "detected_count": int(len(detected_set)),
        })
        if detected_set == true_positive_set:
            break
    return {
        "mode": label,
        "detected_positive_indices": detected_positive_indices,
        "detected_count": len(detected_positive_indices),
        "inspection_count": len(individual_measurements),
        "pool_count": int(dataset.get("initial_pool_count", len(dataset["pools"]))),
        "positive_pool_constraint_count": int(dataset.get("positive_pool_count", len(dataset["pools"]))),
        "total_test_cost": int(dataset.get("initial_pool_count", len(dataset["pools"]))) + len(individual_measurements),
        "individual_measurements": individual_measurements,
        "final_x_hat": final_x_hat,
    }


def run_priority_inspection_without_cvxpy(dataset: dict, mu: np.ndarray, label: str) -> dict:
    # 軽量確認用。主評価には使わない。
    params = dataset["params"]
    x_true = np.asarray(dataset["x_true"], dtype=float)
    order = np.argsort(mu)
    true_positive_set = set(np.where(x_true >= params["x_min_positive"])[0].tolist())
    detected = []
    measurements = []
    for step, cand in enumerate(order):
        meas = individual_measurement_qpcr(float(x_true[cand]), params)
        is_pos = bool(x_true[cand] >= params["x_min_positive"])
        if is_pos:
            detected.append(int(cand))
        measurements.append({
            "step": int(step), "candidate": int(cand), "pred_before_test": float(1.0 / max(mu[cand], 1e-12)),
            "x_true": float(x_true[cand]), "x_est": float(meas["x_est"]), "ct": float(meas["ct"]),
            "is_true_positive": is_pos, "detected_count": len(set(detected)),
        })
        if set(detected) == true_positive_set:
            break
    return {
        "mode": label,
        "detected_positive_indices": list(dict.fromkeys(detected)),
        "detected_count": len(set(detected)),
        "inspection_count": len(measurements),
        "pool_count": int(dataset.get("initial_pool_count", len(dataset["pools"]))),
        "positive_pool_constraint_count": int(dataset.get("positive_pool_count", len(dataset["pools"]))),
        "total_test_cost": int(dataset.get("initial_pool_count", len(dataset["pools"]))) + len(measurements),
        "individual_measurements": measurements,
        "final_x_hat": np.zeros_like(x_true),
    }



def restrict_dataset_to_positive_pools(dataset: dict, positive_cutoff: float | None = None) -> dict:
    """
    Keep only positive pooled-test constraints and only individuals that appear in those pools.

    Rationale:
      - A negative pool already implies all included specimens are negative under the noiseless qPCR model.
      - Therefore the sparse reconstruction problem only needs the rows of A with positive pool measurements.
      - Columns are also reduced to the union of individuals appearing in positive pools.

    The returned dataset is re-indexed locally, but keeps mappings:
      - original_indices: local index -> original index
      - original_pool_indices: local positive-pool row -> original pool row
      - initial_pool_count: total number of first-stage pool tests actually performed
    """
    params = dict(dataset["params"])
    cutoff = float(params.get("positive_cutoff", params.get("x_min_positive", 1e3)) if positive_cutoff is None else positive_cutoff)
    pooled_amount_est = np.asarray(dataset["pooled_amount_est"], dtype=float)
    A = np.asarray(dataset["A"], dtype=float)
    pools = list(dataset["pools"])

    positive_pool_mask = pooled_amount_est >= cutoff
    positive_pool_indices = np.where(positive_pool_mask)[0]

    if len(positive_pool_indices) == 0:
        # No positive pools: all individuals are determined negative by the first-stage pooling.
        out = dict(dataset)
        out["A"] = np.zeros((0, 0), dtype=float)
        out["pools"] = []
        out["pooled_amount_true"] = np.array([], dtype=float)
        out["pooled_ct"] = np.array([], dtype=float)
        out["pooled_amount_est"] = np.array([], dtype=float)
        out["patient_data"] = dataset["patient_data"].iloc[0:0].copy().reset_index(drop=True)
        out["x_true"] = np.array([], dtype=float)
        out["y_true"] = np.array([], dtype=int)
        out["W"] = np.zeros((0, 0), dtype=float)
        if "W_basic" in dataset:
            out["W_basic"] = np.zeros((0, 0), dtype=float)
        if "W_type_weighted" in dataset:
            out["W_type_weighted"] = np.zeros((0, 0), dtype=float)
        out["symptom_mat"] = np.zeros((0, np.asarray(dataset.get("symptom_mat", np.zeros((0, 0)))).shape[1]), dtype=int)
        out["original_indices"] = np.array([], dtype=int)
        out["original_pool_indices"] = np.array([], dtype=int)
        out["positive_pool_count"] = 0
        out["initial_pool_count"] = int(len(pools))
        out["excluded_by_negative_pools_count"] = int(A.shape[1])
        out["params"] = derive_params({**params, "n": 0})
        return out

    candidate_mask = A[positive_pool_indices].sum(axis=0) > 0
    candidate_indices = np.where(candidate_mask)[0]
    local_pos = {int(old): int(new) for new, old in enumerate(candidate_indices)}

    A_reduced = A[np.ix_(positive_pool_indices, candidate_indices)]
    pools_reduced = [tuple(local_pos[int(i)] for i in pools[int(r)] if int(i) in local_pos) for r in positive_pool_indices]

    out = dict(dataset)
    out["A"] = A_reduced
    out["pools"] = pools_reduced
    out["pooled_amount_true"] = np.asarray(dataset["pooled_amount_true"], dtype=float)[positive_pool_indices]
    out["pooled_ct"] = np.asarray(dataset["pooled_ct"], dtype=float)[positive_pool_indices]
    out["pooled_amount_est"] = pooled_amount_est[positive_pool_indices]
    out["patient_data"] = dataset["patient_data"].iloc[candidate_indices].copy().reset_index(drop=True)
    out["x_true"] = np.asarray(dataset["x_true"], dtype=float)[candidate_indices]
    out["y_true"] = np.asarray(dataset["y_true"], dtype=int)[candidate_indices]
    out["person_ids"] = np.asarray(dataset.get("person_ids", np.arange(A.shape[1])))[candidate_indices]
    out["symptom_mat"] = np.asarray(dataset["symptom_mat"])[candidate_indices]
    for key in ["W", "W_basic", "W_type_weighted"]:
        if key in dataset:
            W = np.asarray(dataset[key], dtype=float)
            out[key] = W[np.ix_(candidate_indices, candidate_indices)]
    out["original_indices"] = candidate_indices.astype(int)
    out["original_pool_indices"] = positive_pool_indices.astype(int)
    out["positive_pool_count"] = int(len(positive_pool_indices))
    out["initial_pool_count"] = int(len(pools))
    out["excluded_by_negative_pools_count"] = int(A.shape[1] - len(candidate_indices))
    out["params"] = derive_params({**params, "n": int(len(candidate_indices))})
    return out


def compute_method_summary(dataset: dict, result_objects: dict[str, dict]) -> pd.DataFrame:
    params = dataset["params"]
    x_true = np.asarray(dataset["x_true"], dtype=float)
    true_positive = x_true >= params["x_min_positive"]
    rows = []
    for mode, res in result_objects.items():
        detected_set = set(res.get("detected_positive_indices", []))
        detected_mask = np.zeros(len(x_true), dtype=bool)
        if detected_set:
            detected_mask[list(detected_set)] = True
        tp_detected = int(np.sum(detected_mask & true_positive))
        recall_detected = tp_detected / max(int(true_positive.sum()), 1)
        x_hat = np.asarray(res.get("final_x_hat", np.zeros_like(x_true)), dtype=float)
        pred_positive = x_hat >= params["x_min_positive"]
        tp = int(np.sum(pred_positive & true_positive))
        fp = int(np.sum(pred_positive & ~true_positive))
        fn = int(np.sum(~pred_positive & true_positive))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append({
            "mode": mode,
            "true_positives": int(true_positive.sum()),
            "detected_count": int(res["detected_count"]),
            "detected_recall_by_individual_qpcr": round(recall_detected, 4),
            "inspection_count": int(res["inspection_count"]),
            "pool_count": int(res["pool_count"]),
            "positive_pool_constraint_count": int(res.get("positive_pool_constraint_count", res["pool_count"])),
            "total_test_cost": int(res["total_test_cost"]),
            "precision_final_x_hat": round(precision, 4),
            "recall_final_x_hat": round(recall, 4),
            "f1_final_x_hat": round(f1, 4),
        })
    return pd.DataFrame(rows).sort_values("total_test_cost").reset_index(drop=True)


def compute_required_tests_by_priority(dataset: dict, priors: dict[str, np.ndarray]) -> pd.DataFrame:
    x_true = np.asarray(dataset["x_true"], dtype=float)
    params = dataset["params"]
    y = x_true >= params["x_min_positive"]
    k = int(y.sum())
    rows = []
    for name, mu in priors.items():
        order = np.argsort(mu)
        cum = np.cumsum(y[order])
        idx = np.where(cum >= k)[0]
        inspection = int(idx[0] + 1) if len(idx) else len(y)
        rows.append({"method": name, "individual_tests_to_find_all": inspection, "pool_tests": int(dataset.get("initial_pool_count", len(dataset["pools"]))), "positive_pool_constraint_count": int(dataset.get("positive_pool_count", len(dataset["pools"]))), "total_tests_to_find_all": inspection + int(dataset.get("initial_pool_count", len(dataset["pools"])))})
    return pd.DataFrame(rows).set_index("method")



def run_pool_size_sweep(
    population,
    contacts,
    base_params: dict,
    pool_sizes=range(1, 31),
    build_dataset_kwargs: dict | None = None,
    prior_kwargs: dict | None = None,
    use_positive_pool_subproblem: bool = True,
    sparse_max_rounds: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the three-method comparison for multiple pool sizes.

    Output columns:
      - pool_size
      - method
      - initial_pool_count: first-stage pool tests actually performed
      - positive_pool_constraint_count: positive pools retained in Ax=s
      - candidate_count: individuals remaining after removing negative-only pools
      - individual_test_count: sequential individual qPCR tests until all positives are found
      - total_test_cost: initial_pool_count + individual_test_count

    Negative pools are not used in the sparse reconstruction after the first-stage pooling.
    """
    import data_poolsizefitting as ds

    build_dataset_kwargs = {} if build_dataset_kwargs is None else dict(build_dataset_kwargs)
    prior_kwargs = {} if prior_kwargs is None else dict(prior_kwargs)
    rows = []

    for pool_size in list(pool_sizes):
        params = derive_params({**dict(base_params), "n": len(population), "pool_size": int(pool_size)})
        try:
            full_dataset = ds.build_analysis_dataset(
                population,
                contacts,
                params,
                pool_size=int(pool_size),
                **build_dataset_kwargs,
            )
            analysis_dataset = (
                restrict_dataset_to_positive_pools(full_dataset)
                if use_positive_pool_subproblem
                else full_dataset
            )
            priors, _, _ = ds.compute_prior_methods(analysis_dataset, **prior_kwargs)

            if len(analysis_dataset.get("x_true", [])) == 0:
                for method in priors.keys():
                    rows.append({
                        "pool_size": int(pool_size),
                        "method": method,
                        "initial_pool_count": int(analysis_dataset.get("initial_pool_count", len(full_dataset["pools"]))),
                        "positive_pool_constraint_count": int(analysis_dataset.get("positive_pool_count", 0)),
                        "candidate_count": 0,
                        "true_positive_count": 0,
                        "individual_test_count": 0,
                        "total_test_cost": int(analysis_dataset.get("initial_pool_count", len(full_dataset["pools"]))),
                        "status": "ok_no_positive_pool",
                    })
                continue

            max_rounds = len(analysis_dataset["x_true"]) if sparse_max_rounds is None else int(sparse_max_rounds)
            for method, mu in priors.items():
                res = run_sequential_sparse_reconstruction(
                    analysis_dataset,
                    mu=mu,
                    label=method,
                    max_rounds=max_rounds,
                )
                rows.append({
                    "pool_size": int(pool_size),
                    "method": method,
                    "initial_pool_count": int(res["pool_count"]),
                    "positive_pool_constraint_count": int(res.get("positive_pool_constraint_count", 0)),
                    "candidate_count": int(len(analysis_dataset["x_true"])),
                    "true_positive_count": int((np.asarray(analysis_dataset["x_true"]) >= analysis_dataset["params"]["x_min_positive"]).sum()),
                    "individual_test_count": int(res["inspection_count"]),
                    "total_test_cost": int(res["total_test_cost"]),
                    "status": "ok",
                })
            if verbose:
                done = [r for r in rows if r["pool_size"] == int(pool_size)]
                best = min(done, key=lambda x: x["total_test_cost"])
                print(f"pool_size={pool_size}: best={best['method']} total={best['total_test_cost']} candidates={best['candidate_count']} positive_pools={best['positive_pool_constraint_count']}")
        except Exception as e:
            rows.append({
                "pool_size": int(pool_size),
                "method": None,
                "initial_pool_count": np.nan,
                "positive_pool_constraint_count": np.nan,
                "candidate_count": np.nan,
                "true_positive_count": np.nan,
                "individual_test_count": np.nan,
                "total_test_cost": np.nan,
                "status": f"error: {type(e).__name__}: {e}",
            })
            if verbose:
                print(f"pool_size={pool_size}: ERROR {type(e).__name__}: {e}")

    return pd.DataFrame(rows)


def plot_pool_size_sweep(sweep_df: pd.DataFrame, output_path: str | None = None):
    """Plot total test cost by pool size for each method."""
    import matplotlib.pyplot as plt
    ok = sweep_df[sweep_df["status"].astype(str).str.startswith("ok")].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, sub in ok.groupby("method"):
        sub = sub.sort_values("pool_size")
        ax.plot(sub["pool_size"], sub["total_test_cost"], marker="o", label=str(method))
    ax.set_xlabel("Pool size")
    ax.set_ylabel("Total number of tests")
    ax.set_title("Pool size sweep: first-stage pools + individual qPCR tests")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    return fig

# ============================================================
# CSV diagnostics for method-level comparison
# ============================================================

def compute_method_comparison_individuals(dataset: dict, priors: dict[str, np.ndarray], results: dict[str, dict] | None = None) -> pd.DataFrame:
    """Create one row per individual with method ranks/scores and test outcomes."""
    if results is None:
        results = {}
    patient = dataset["patient_data"].reset_index(drop=True)
    y = np.asarray(dataset["x_true"], dtype=float) >= dataset["params"]["x_min_positive"]
    out = pd.DataFrame({
        "row_index": np.arange(len(patient), dtype=int),
        "person_id": patient["person_id"].astype(int).to_numpy() if "person_id" in patient.columns else np.arange(len(patient), dtype=int),
        "y_true": y.astype(int),
        "viral_rna_load": np.asarray(dataset["x_true"], dtype=float),
    })
    if "reported_total_symptom_count" in patient.columns:
        out["reported_total_symptom_count"] = patient["reported_total_symptom_count"].to_numpy()
    for c in [c for c in patient.columns if c.startswith("reported_symptom_")]:
        out[c] = patient[c].to_numpy()

    for method, mu in priors.items():
        mu = np.asarray(mu, dtype=float)
        priority_score = 1.0 / np.maximum(mu, 1e-12)
        order = np.argsort(-priority_score)
        rank = np.empty(len(mu), dtype=int)
        rank[order] = np.arange(1, len(mu) + 1)
        out[f"{method}_mu"] = mu
        out[f"{method}_priority_score"] = priority_score
        out[f"{method}_priority_rank"] = rank
        if method in results:
            tested_step = np.full(len(mu), np.nan)
            detected = np.zeros(len(mu), dtype=int)
            for m in results[method].get("individual_measurements", []):
                cand = int(m["candidate"])
                tested_step[cand] = int(m["step"])
                detected[cand] = int(bool(m.get("is_true_positive", False)))
            out[f"{method}_tested_step"] = tested_step
            out[f"{method}_detected_positive"] = detected
    return out


def compute_method_disagreement_summary(individual_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize rank disagreement between available method priority columns."""
    rank_cols = [c for c in individual_df.columns if c.endswith("_priority_rank")]
    rows = []
    for i, a in enumerate(rank_cols):
        for b in rank_cols[i+1:]:
            ma = a.replace("_priority_rank", "")
            mb = b.replace("_priority_rank", "")
            diff = (individual_df[a].astype(float) - individual_df[b].astype(float)).abs()
            rows.append({
                "method_a": ma,
                "method_b": mb,
                "mean_abs_rank_diff": float(diff.mean()),
                "median_abs_rank_diff": float(diff.median()),
                "max_abs_rank_diff": float(diff.max()),
                "top100_overlap": int(len(set(individual_df.nsmallest(100, a)["person_id"]) & set(individual_df.nsmallest(100, b)["person_id"]))),
            })
    return pd.DataFrame(rows)


# ============================================================
# Pool-size fitting helper functions
# ============================================================

def run_one_seed_poolsize_fitting(
    population,
    contacts,
    seed: int,
    pool_size: int,
    target_method: str = "symptom_count_plus_graph",
    build_dataset_func=None,
    compute_prior_func=None,
    contact_type_weights=None,
    cluster_symptom_strength: float = 0.95,
    force_symptom_regeneration: bool = True,
    target_positive_symptomatic_rate: float = 0.92,
    strong_edge_threshold: float = 2.0,
    neg_covid_leak_prob: float = 0.0,
    noncovid_base_prob: float = 0.0,
    pos_noncovid_prob: float = 0.0,
    beta_symptom: float = 1.0,
    graph_weight: float = 1.0,
    graph_normalization: str = "neighbor_symptom_sum_div_max_symptoms",
    clip_graph_score: bool = False,
    use_positive_pool_subproblem: bool = True,
    max_rounds: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run one seed × one pool size using only target_method.

    Notes
    -----
    - Incomplete last pools are allowed by make_pooling_matrix.
      Therefore initial_pool_count should be ceil(n / pool_size) when gaps=(1,).
    - If use_positive_pool_subproblem=True, negative pooled tests are used to remove
      negative-only individuals; the reconstruction problem is built only from
      positive pool constraints and individuals appearing in positive pools.
    """
    import math
    import data_poolsizefitting as ds

    if build_dataset_func is None:
        build_dataset_func = ds.build_analysis_dataset
    if compute_prior_func is None:
        compute_prior_func = ds.compute_prior_methods

    n = int(len(population))
    params = get_default_params(n=n, pool_size=int(pool_size))
    expected_initial_pool_count = int(math.ceil(n / int(pool_size)))
    incomplete_last_pool_size = int(n % int(pool_size))
    if incomplete_last_pool_size == 0:
        incomplete_last_pool_size = int(pool_size)

    full_dataset = build_dataset_func(
        population,
        contacts,
        params=params,
        pool_size=int(pool_size),
        cluster_symptom_strength=cluster_symptom_strength,
        seed=int(seed),
        force_symptom_regeneration=force_symptom_regeneration,
        contact_type_weights=contact_type_weights,
        strong_edge_threshold=strong_edge_threshold,
        neg_covid_leak_prob=neg_covid_leak_prob,
        noncovid_base_prob=noncovid_base_prob,
        pos_noncovid_prob=pos_noncovid_prob,
        target_positive_symptomatic_rate=target_positive_symptomatic_rate,
    )

    analysis_dataset = (
        restrict_dataset_to_positive_pools(full_dataset)
        if use_positive_pool_subproblem
        else full_dataset
    )

    actual_initial_pool_count = int(analysis_dataset.get("initial_pool_count", len(full_dataset["pools"])))
    positive_pool_count = int(analysis_dataset.get("positive_pool_count", len(full_dataset["pools"])))
    candidate_count = int(len(analysis_dataset.get("x_true", [])))
    excluded_by_negative_pools_count = int(analysis_dataset.get("excluded_by_negative_pools_count", 0))

    base_row = {
        "seed": int(seed),
        "pool_size": int(pool_size),
        "method": target_method,
        "sample_size": n,
        "expected_initial_pool_count": expected_initial_pool_count,
        "initial_pool_count": actual_initial_pool_count,
        "has_incomplete_last_pool": bool(n % int(pool_size) != 0),
        "incomplete_last_pool_size": incomplete_last_pool_size,
        "positive_pool_constraint_count": positive_pool_count,
        "candidate_count": candidate_count,
        "excluded_by_negative_pools_count": excluded_by_negative_pools_count,
    }

    if candidate_count == 0:
        row = {
            **base_row,
            "true_positive_count": 0,
            "individual_test_count": 0,
            "total_test_cost": actual_initial_pool_count,
            "detected_positive_count": 0,
            "status": "ok_no_positive_pool",
        }
        if verbose:
            print(
                f"seed={seed} pool_size={pool_size}: total={row['total_test_cost']} "
                f"initial_pools={actual_initial_pool_count} positive_pools={positive_pool_count} "
                f"candidates=0"
            )
        return row

    priors, _, _ = compute_prior_func(
        analysis_dataset,
        beta_symptom=beta_symptom,
        graph_weight=graph_weight,
        graph_normalization=graph_normalization,
        clip_graph_score=clip_graph_score,
    )

    if target_method not in priors:
        raise KeyError(f"{target_method} not found in priors. Available methods: {list(priors.keys())}")

    true_positive_count = int(
        (np.asarray(analysis_dataset["x_true"]) >= analysis_dataset["params"]["x_min_positive"]).sum()
    )
    effective_max_rounds = candidate_count if max_rounds is None else int(max_rounds)

    res = run_sequential_sparse_reconstruction(
        analysis_dataset,
        mu=priors[target_method],
        label=target_method,
        max_rounds=effective_max_rounds,
    )

    row = {
        **base_row,
        "true_positive_count": true_positive_count,
        "individual_test_count": int(res["inspection_count"]),
        "total_test_cost": int(res["total_test_cost"]),
        "detected_positive_count": int(res.get("detected_count", 0)),
        "status": "ok",
    }

    if verbose:
        print(
            f"seed={seed} pool_size={pool_size}: total={row['total_test_cost']} "
            f"initial_pools={actual_initial_pool_count} positive_pools={positive_pool_count} "
            f"candidates={candidate_count} individual={row['individual_test_count']}"
        )

    return row


def run_poolsize_fitting_experiment(
    population,
    contacts,
    seeds,
    pool_sizes,
    target_method: str = "symptom_count_plus_graph",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Run multiple seeds × multiple pool sizes for pool-size fitting."""
    rows = []
    for seed in list(seeds):
        if verbose:
            print(f"\n===== seed={seed} =====")
        for pool_size in list(pool_sizes):
            try:
                row = run_one_seed_poolsize_fitting(
                    population=population,
                    contacts=contacts,
                    seed=int(seed),
                    pool_size=int(pool_size),
                    target_method=target_method,
                    verbose=verbose,
                    **kwargs,
                )
            except Exception as e:
                row = {
                    "seed": int(seed),
                    "pool_size": int(pool_size),
                    "method": target_method,
                    "sample_size": int(len(population)),
                    "expected_initial_pool_count": np.nan,
                    "initial_pool_count": np.nan,
                    "has_incomplete_last_pool": np.nan,
                    "incomplete_last_pool_size": np.nan,
                    "positive_pool_constraint_count": np.nan,
                    "candidate_count": np.nan,
                    "excluded_by_negative_pools_count": np.nan,
                    "true_positive_count": np.nan,
                    "individual_test_count": np.nan,
                    "total_test_cost": np.nan,
                    "detected_positive_count": np.nan,
                    "status": f"error: {type(e).__name__}: {e}",
                }
                if verbose:
                    print(f"seed={seed} pool_size={pool_size}: ERROR {type(e).__name__}: {e}")
            rows.append(row)
    return pd.DataFrame(rows)
