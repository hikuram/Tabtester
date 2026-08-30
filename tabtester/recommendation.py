from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

GOAL_TYPES = ("Range", "At least", "At most", "Close to", "Maximize", "Minimize")
PRIORITY_WEIGHTS = {"Low": 0.5, "Medium": 1.0, "High": 3.0}
DOMAIN_ORDER = {"In-domain": 0, "Near edge": 1, "Extrapolation": 2}


@dataclass(frozen=True)
class SearchPlan:
    mode: str
    initial_count: int
    max_count: int
    effective_dimensions: int
    exhaustive_count: int | None
    reason: str


@dataclass
class ScoredCandidates:
    results: pd.DataFrame
    prediction_detail: dict[str, dict[str, np.ndarray]]
    objective_columns: list[str]


def _finite_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def normalize_design_space(design_space: pd.DataFrame) -> pd.DataFrame:
    required = {"Variable", "Observed min", "Observed max", "Search min", "Search max", "Step", "Active"}
    missing = required.difference(design_space.columns)
    if missing:
        raise ValueError(f"Design-space table is missing columns: {sorted(missing)}")

    table = design_space.copy()
    table["Variable"] = table["Variable"].astype(str)
    table["Active"] = table["Active"].fillna(False).astype(bool)
    for column in ["Observed min", "Observed max", "Search min", "Search max", "Step"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def validate_design_space(
    design_space: pd.DataFrame,
    mixture_variables: Sequence[str] | None = None,
    mixture_total: float | None = None,
) -> list[str]:
    table = normalize_design_space(design_space)
    errors: list[str] = []
    active = table[table["Active"]]
    if active.empty:
        errors.append("Select at least one active design variable.")
        return errors

    for row in active.itertuples(index=False):
        variable = getattr(row, "Variable")
        lower = getattr(row, "_3")  # Search min
        upper = getattr(row, "_4")  # Search max
        step = getattr(row, "Step")
        if not np.isfinite(lower) or not np.isfinite(upper):
            errors.append(f"{variable}: Search min/max must be numeric.")
        elif lower > upper:
            errors.append(f"{variable}: Search min must be <= Search max.")
        if np.isfinite(step) and step <= 0:
            errors.append(f"{variable}: Step must be blank or > 0.")

    mixture_variables = list(mixture_variables or [])
    if mixture_variables:
        active_names = set(active["Variable"])
        missing_mix = [name for name in mixture_variables if name not in active_names]
        if missing_mix:
            errors.append(f"Mixture variables must be active design variables: {missing_mix}")
        total = _finite_float(mixture_total)
        if total is None:
            errors.append("Mixture total must be numeric.")
        else:
            mix_rows = active[active["Variable"].isin(mixture_variables)]
            if len(mix_rows) >= 2:
                lower_sum = float(mix_rows["Search min"].sum())
                upper_sum = float(mix_rows["Search max"].sum())
                if total < lower_sum - 1e-9 or total > upper_sum + 1e-9:
                    errors.append(
                        f"Mixture total {total:g} is outside the feasible sum range "
                        f"[{lower_sum:g}, {upper_sum:g}]."
                    )
            elif mixture_variables:
                errors.append("A mixture constraint requires at least two variables.")
    return errors


def validate_target_spec(target_spec: pd.DataFrame) -> list[str]:
    required = {"Property", "Goal", "Lower", "Target", "Upper", "Priority", "Hard"}
    missing = required.difference(target_spec.columns)
    if missing:
        return [f"Target table is missing columns: {sorted(missing)}"]

    errors: list[str] = []
    if target_spec.empty:
        return ["Select at least one target property."]

    seen: set[str] = set()
    for row in target_spec.itertuples(index=False):
        property_name = str(row.Property)
        goal = str(row.Goal)
        lower = _finite_float(row.Lower)
        target = _finite_float(row.Target)
        upper = _finite_float(row.Upper)
        priority = str(row.Priority)
        hard = bool(row.Hard)

        if property_name in seen:
            errors.append(f"Duplicate target property: {property_name}")
        seen.add(property_name)
        if goal not in GOAL_TYPES:
            errors.append(f"{property_name}: Unknown goal '{goal}'.")
            continue
        if priority not in PRIORITY_WEIGHTS:
            errors.append(f"{property_name}: Priority must be Low, Medium, or High.")

        if goal == "Range":
            if lower is None or upper is None:
                errors.append(f"{property_name}: Range requires Lower and Upper.")
            elif lower > upper:
                errors.append(f"{property_name}: Lower must be <= Upper.")
        elif goal == "At least" and lower is None:
            errors.append(f"{property_name}: At least requires Lower.")
        elif goal == "At most" and upper is None:
            errors.append(f"{property_name}: At most requires Upper.")
        elif goal == "Close to" and target is None:
            errors.append(f"{property_name}: Close to requires Target.")

        if hard and goal == "Close to" and (lower is None or upper is None):
            errors.append(
                f"{property_name}: Hard 'Close to' requires Lower and Upper as the acceptable band."
            )
        if hard and goal in {"Maximize", "Minimize"}:
            errors.append(
                f"{property_name}: Maximize/Minimize cannot be a hard constraint; use At least/At most instead."
            )
    return errors


def _grid_values(lower: float, upper: float, step: float) -> np.ndarray:
    if upper <= lower:
        return np.array([lower], dtype=float)
    count = int(math.floor((upper - lower) / step + 1e-9)) + 1
    values = lower + np.arange(count, dtype=float) * step
    if values[-1] < upper - max(1e-10, abs(step) * 1e-9):
        values = np.append(values, upper)
    return np.clip(values, lower, upper)


def _exhaustive_count(table: pd.DataFrame) -> int | None:
    total = 1
    for row in table.itertuples(index=False):
        step = _finite_float(row.Step)
        if step is None:
            return None
        lower = float(getattr(row, "_3"))
        upper = float(getattr(row, "_4"))
        total *= len(_grid_values(lower, upper, step))
        if total > 10_000_000:
            return total
    return int(total)


def suggest_sample_count(
    design_space: pd.DataFrame,
    *,
    effort: str = "Balanced",
    mixture_variables: Sequence[str] | None = None,
    exhaustive_limit: int = 50_000,
) -> SearchPlan:
    table = normalize_design_space(design_space)
    active = table[table["Active"]].copy()
    if active.empty:
        return SearchPlan("sample", 0, 0, 0, None, "No active design variables.")

    mixture_variables = [name for name in (mixture_variables or []) if name in set(active["Variable"])]
    effective_dimensions = len(active) - (1 if len(mixture_variables) >= 2 else 0)
    effective_dimensions = max(1, effective_dimensions)
    exhaustive_count = _exhaustive_count(active)
    if exhaustive_count is not None and exhaustive_count <= exhaustive_limit:
        return SearchPlan(
            "exhaustive",
            exhaustive_count,
            exhaustive_count,
            effective_dimensions,
            exhaustive_count,
            f"All active variables are discrete and the full grid has {exhaustive_count:,} combinations.",
        )

    if effective_dimensions <= 3:
        base = 2_048
    elif effective_dimensions <= 6:
        base = 4_096
    elif effective_dimensions <= 10:
        base = 8_192
    elif effective_dimensions <= 15:
        base = 16_384
    else:
        base = 32_768

    ratios: list[float] = []
    for row in active.itertuples(index=False):
        observed_min = float(getattr(row, "_1"))
        observed_max = float(getattr(row, "_2"))
        search_min = float(getattr(row, "_3"))
        search_max = float(getattr(row, "_4"))
        observed_span = observed_max - observed_min
        search_span = search_max - search_min
        if observed_span > 0:
            ratios.append(float(np.clip(search_span / observed_span, 0.1, 2.0)))
    range_factor = math.sqrt(float(np.mean(ratios))) if ratios else 1.0
    effort_factor = {"Quick": 0.5, "Balanced": 1.0, "Thorough": 2.0}.get(effort, 1.0)
    raw = max(512, int(base * range_factor * effort_factor))
    initial_count = int(2 ** round(math.log2(raw)))
    initial_count = int(np.clip(initial_count, 512, 65_536))
    max_multiplier = {"Quick": 2, "Balanced": 4, "Thorough": 4}.get(effort, 4)
    max_count = min(131_072, initial_count * max_multiplier)
    reason = (
        f"{effective_dimensions} effective dimensions; search-range factor {range_factor:.2f}; "
        f"{effort.lower()} search effort."
    )
    return SearchPlan(
        "sample",
        initial_count,
        max_count,
        effective_dimensions,
        exhaustive_count,
        reason,
    )


def _latin_hypercube(n: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    if dimensions <= 0:
        return np.empty((n, 0), dtype=float)
    result = np.empty((n, dimensions), dtype=float)
    for dimension in range(dimensions):
        bins = (np.arange(n, dtype=float) + rng.random(n)) / n
        result[:, dimension] = bins[rng.permutation(n)]
    return result


def _apply_step(values: np.ndarray, lower: float, upper: float, step: float | None) -> np.ndarray:
    if step is None:
        return np.clip(values, lower, upper)
    snapped = lower + np.rint((values - lower) / step) * step
    return np.clip(snapped, lower, upper)


def _generate_mixture_values(
    rows: pd.DataFrame,
    n: int,
    total: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    variables = rows["Variable"].tolist()
    lowers = rows["Search min"].to_numpy(dtype=float)
    uppers = rows["Search max"].to_numpy(dtype=float)
    steps = [(_finite_float(value)) for value in rows["Step"].tolist()]
    residual = total - float(lowers.sum())
    if residual < -1e-9 or total > float(uppers.sum()) + 1e-9:
        raise ValueError("Mixture total is not feasible inside the selected bounds.")
    if residual <= 1e-12:
        return pd.DataFrame(np.tile(lowers, (n, 1)), columns=variables)

    continuous = all(step is None for step in steps)
    accepted: list[np.ndarray] = []
    needed = n
    attempts = 0
    max_attempts = 30
    while needed > 0 and attempts < max_attempts:
        batch_size = max(needed * 3, 1024)
        if continuous:
            shares = rng.dirichlet(np.ones(len(variables)), size=batch_size)
            batch = lowers + shares * residual
            valid = np.all(batch <= uppers + 1e-9, axis=1)
            batch = batch[valid]
        else:
            batch = np.empty((batch_size, len(variables)), dtype=float)
            for column_index in range(len(variables) - 1):
                lower = lowers[column_index]
                upper = uppers[column_index]
                step = steps[column_index]
                raw = rng.uniform(lower, upper, size=batch_size)
                batch[:, column_index] = _apply_step(raw, lower, upper, step)
            batch[:, -1] = total - batch[:, :-1].sum(axis=1)
            last_step = steps[-1]
            if last_step is not None:
                snapped_last = _apply_step(batch[:, -1], lowers[-1], uppers[-1], last_step)
                aligned = np.isclose(snapped_last, batch[:, -1], atol=max(last_step * 1e-6, 1e-9))
                batch[:, -1] = snapped_last
            else:
                aligned = np.ones(batch_size, dtype=bool)
            valid = aligned & np.all(batch >= lowers - 1e-9, axis=1) & np.all(batch <= uppers + 1e-9, axis=1)
            valid &= np.isclose(batch.sum(axis=1), total, atol=1e-7)
            batch = batch[valid]

        if len(batch):
            accepted.append(batch[:needed])
            needed -= min(needed, len(batch))
        attempts += 1

    if needed > 0:
        raise ValueError(
            "Could not generate enough feasible mixture candidates. Widen the ranges, relax the step, or reduce the requested sample count."
        )
    values = np.vstack(accepted)[:n]
    return pd.DataFrame(values, columns=variables)


def generate_candidates(
    design_space: pd.DataFrame,
    fixed_values: Mapping[str, object],
    count: int,
    *,
    random_state: int = 42,
    mixture_variables: Sequence[str] | None = None,
    mixture_total: float | None = None,
    exhaustive_limit: int = 50_000,
) -> tuple[pd.DataFrame, str]:
    table = normalize_design_space(design_space)
    active = table[table["Active"]].copy()
    errors = validate_design_space(table, mixture_variables, mixture_total)
    if errors:
        raise ValueError(" ".join(errors))
    mixture_variables = list(mixture_variables or [])
    rng = np.random.default_rng(random_state)

    exhaustive_count = _exhaustive_count(active)
    if exhaustive_count is not None and exhaustive_count <= exhaustive_limit:
        arrays = []
        names = []
        for row in active.itertuples(index=False):
            lower = float(getattr(row, "_3"))
            upper = float(getattr(row, "_4"))
            step = float(row.Step)
            arrays.append(_grid_values(lower, upper, step))
            names.append(row.Variable)
        grid = pd.DataFrame(product(*arrays), columns=names)
        if mixture_variables:
            total = float(mixture_total)
            grid = grid[np.isclose(grid[mixture_variables].sum(axis=1), total, atol=1e-7)]
        if grid.empty:
            raise ValueError("The discrete design grid has no candidates satisfying the mixture constraint.")
        candidates = grid.reset_index(drop=True)
        mode = "exhaustive"
    else:
        n = max(1, int(count))
        mixture_set = set(mixture_variables)
        non_mix = active[~active["Variable"].isin(mixture_set)]
        lhs = _latin_hypercube(n, len(non_mix), rng)
        sampled = pd.DataFrame(index=np.arange(n))
        for column_index, row in enumerate(non_mix.itertuples(index=False)):
            lower = float(getattr(row, "_3"))
            upper = float(getattr(row, "_4"))
            raw = lower + lhs[:, column_index] * (upper - lower)
            sampled[row.Variable] = _apply_step(raw, lower, upper, _finite_float(row.Step))
        if mixture_variables:
            mix_rows = active[active["Variable"].isin(mixture_variables)].copy()
            mix_values = _generate_mixture_values(mix_rows, n, float(mixture_total), rng)
            for variable in mixture_variables:
                sampled[variable] = mix_values[variable].to_numpy()
        candidates = sampled[active["Variable"].tolist()].drop_duplicates().reset_index(drop=True)
        mode = "space-filling"

    output = pd.DataFrame(index=candidates.index)
    for column, value in fixed_values.items():
        output[column] = value
    for column in candidates.columns:
        output[column] = candidates[column].to_numpy()
    ordered_columns = list(fixed_values.keys())
    for column in candidates.columns:
        if column not in ordered_columns:
            ordered_columns.append(column)
    return output.loc[:, ordered_columns], mode


def _property_scale(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) == 0:
        return 1.0
    span = float(np.max(numeric) - np.min(numeric))
    if span > 1e-12:
        return span
    std = float(np.std(numeric))
    return std if std > 1e-12 else 1.0


def _loss_and_hard(
    prediction: np.ndarray,
    row: pd.Series,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    goal = str(row["Goal"])
    lower = _finite_float(row["Lower"])
    target = _finite_float(row["Target"])
    upper = _finite_float(row["Upper"])
    hard = bool(row["Hard"])
    pred = np.asarray(prediction, dtype=float)
    eps_scale = max(scale, 1e-12)

    if goal == "Range":
        below = np.maximum((lower - pred) / eps_scale, 0.0)
        above = np.maximum((pred - upper) / eps_scale, 0.0)
        loss = below + above
        objective = loss.copy()
        hard_bad = (pred < lower) | (pred > upper) if hard else np.zeros(len(pred), dtype=bool)
    elif goal == "At least":
        loss = np.maximum((lower - pred) / eps_scale, 0.0)
        objective = loss.copy()
        hard_bad = pred < lower if hard else np.zeros(len(pred), dtype=bool)
    elif goal == "At most":
        loss = np.maximum((pred - upper) / eps_scale, 0.0)
        objective = loss.copy()
        hard_bad = pred > upper if hard else np.zeros(len(pred), dtype=bool)
    elif goal == "Close to":
        loss = np.abs(pred - target) / eps_scale
        objective = loss.copy()
        if hard:
            hard_bad = (pred < lower) | (pred > upper)
        else:
            hard_bad = np.zeros(len(pred), dtype=bool)
    elif goal == "Maximize":
        min_value = float(np.nanmin(pred))
        max_value = float(np.nanmax(pred))
        spread = max(max_value - min_value, 1e-12)
        loss = (max_value - pred) / spread
        objective = loss.copy()
        hard_bad = np.zeros(len(pred), dtype=bool)
    elif goal == "Minimize":
        min_value = float(np.nanmin(pred))
        max_value = float(np.nanmax(pred))
        spread = max(max_value - min_value, 1e-12)
        loss = (pred - min_value) / spread
        objective = loss.copy()
        hard_bad = np.zeros(len(pred), dtype=bool)
    else:
        raise ValueError(f"Unsupported goal: {goal}")
    return loss, hard_bad, objective


def pareto_efficient_mask(costs: np.ndarray) -> np.ndarray:
    values = np.asarray(costs, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros(values.shape[0] if values.ndim else 0, dtype=bool)
    finite = np.all(np.isfinite(values), axis=1)
    mask = finite.copy()
    indices = np.arange(len(values))
    for index in indices:
        if not mask[index]:
            continue
        active_indices = indices[mask]
        active_costs = values[mask]
        current = values[index]
        keep = np.any(active_costs < current, axis=1) | np.all(np.isclose(active_costs, current), axis=1)
        mask[active_indices] = keep
        mask[index] = True
    return mask


def _domain_diagnostics(
    candidates: pd.DataFrame,
    design_space: pd.DataFrame,
    training_design: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    table = normalize_design_space(design_space)
    active = table[table["Active"]]
    variables = active["Variable"].tolist()
    candidate_values = candidates[variables].apply(pd.to_numeric, errors="coerce")
    training_values = training_design[variables].apply(pd.to_numeric, errors="coerce")

    risks: list[str] = []
    for _, candidate in candidate_values.iterrows():
        outside = False
        near_edge = False
        for row in active.itertuples(index=False):
            variable = row.Variable
            observed_min = float(getattr(row, "_1"))
            observed_max = float(getattr(row, "_2"))
            value = float(candidate[variable])
            span = observed_max - observed_min
            if value < observed_min - 1e-9 or value > observed_max + 1e-9:
                outside = True
                break
            if span > 0:
                edge_fraction = min(value - observed_min, observed_max - value) / span
                if edge_fraction <= 0.05:
                    near_edge = True
        risks.append("Extrapolation" if outside else "Near edge" if near_edge else "In-domain")

    observed_min = active.set_index("Variable")["Observed min"].reindex(variables).to_numpy(dtype=float)
    observed_max = active.set_index("Variable")["Observed max"].reindex(variables).to_numpy(dtype=float)
    span = np.where((observed_max - observed_min) > 1e-12, observed_max - observed_min, 1.0)
    train_array = training_values.dropna().to_numpy(dtype=float)
    candidate_array = candidate_values.to_numpy(dtype=float)
    if len(train_array) == 0 or not np.all(np.isfinite(candidate_array)):
        distances = np.full(len(candidates), np.nan)
    else:
        train_scaled = (train_array - observed_min) / span
        candidate_scaled = (candidate_array - observed_min) / span
        neighbors = NearestNeighbors(n_neighbors=1)
        neighbors.fit(train_scaled)
        distances = neighbors.kneighbors(candidate_scaled, return_distance=True)[0][:, 0]
    return np.asarray(risks, dtype=object), distances


def score_candidates(
    candidates: pd.DataFrame,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    target_spec: pd.DataFrame,
    observed_targets: pd.DataFrame,
    design_space: pd.DataFrame,
    training_design: pd.DataFrame,
) -> ScoredCandidates:
    errors = validate_target_spec(target_spec)
    if errors:
        raise ValueError(" ".join(errors))
    if not predictions:
        raise ValueError("No model predictions were provided.")

    model_names = list(predictions.keys())
    results = candidates.copy().reset_index(drop=True)
    detail: dict[str, dict[str, np.ndarray]] = {name: {} for name in model_names}
    weighted_loss = np.zeros(len(results), dtype=float)
    weight_total = 0.0
    hard_violations = np.zeros(len(results), dtype=int)
    objective_values: list[np.ndarray] = []
    objective_columns: list[str] = []
    disagreement_parts: list[np.ndarray] = []

    for _, spec in target_spec.iterrows():
        property_name = str(spec["Property"])
        model_arrays = []
        for model_name in model_names:
            if property_name not in predictions[model_name]:
                continue
            values = np.asarray(predictions[model_name][property_name], dtype=float)
            if len(values) != len(results):
                raise ValueError(f"Prediction length mismatch for {model_name}/{property_name}.")
            detail[model_name][property_name] = values
            model_arrays.append(values)
        if not model_arrays:
            raise ValueError(f"No predictions available for target property '{property_name}'.")

        stack = np.vstack(model_arrays)
        consensus = np.nanmedian(stack, axis=0)
        scale = _property_scale(observed_targets[property_name])
        if stack.shape[0] > 1:
            mad = np.nanmedian(np.abs(stack - consensus), axis=0)
            disagreement = mad / max(scale, 1e-12)
        else:
            disagreement = np.zeros(len(consensus), dtype=float)
        disagreement_parts.append(disagreement)

        results[f"Pred {property_name}"] = consensus
        results[f"Disagreement {property_name}"] = disagreement
        loss, hard_bad, objective = _loss_and_hard(consensus, spec, scale)
        weight = PRIORITY_WEIGHTS[str(spec["Priority"])]
        weighted_loss += weight * loss
        weight_total += weight
        hard_violations += hard_bad.astype(int)
        objective_values.append(objective)
        objective_columns.append(property_name)

    normalized_loss = weighted_loss / max(weight_total, 1e-12)
    results.insert(0, "Target fit", 100.0 * np.exp(-normalized_loss))
    results.insert(1, "Hard violations", hard_violations)
    if disagreement_parts:
        results.insert(2, "Model disagreement", np.nanmean(np.vstack(disagreement_parts), axis=0))
    else:
        results.insert(2, "Model disagreement", 0.0)

    domain, nearest_distance = _domain_diagnostics(candidates, design_space, training_design)
    results.insert(3, "Domain", domain)
    results.insert(4, "Nearest distance", nearest_distance)

    cost_matrix = np.column_stack(objective_values)
    feasible = hard_violations == 0
    pareto_mask = np.zeros(len(results), dtype=bool)
    if bool(target_spec["Hard"].any()) and np.any(feasible):
        pareto_mask[feasible] = pareto_efficient_mask(cost_matrix[feasible])
    else:
        pareto_mask = pareto_efficient_mask(cost_matrix)
    results.insert(0, "Pareto", pareto_mask)
    results.insert(0, "Candidate", [f"C{index + 1:05d}" for index in range(len(results))])
    return ScoredCandidates(results=results, prediction_detail=detail, objective_columns=objective_columns)


def select_shortlist(
    results: pd.DataFrame,
    design_variables: Sequence[str],
    *,
    limit: int = 10,
    min_separation: float = 0.06,
) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    pool = results.copy()
    pool["__pareto_rank"] = (~pool["Pareto"].astype(bool)).astype(int)
    pool["__domain_rank"] = pool["Domain"].map(DOMAIN_ORDER).fillna(3)
    pool = pool.sort_values(
        [
            "__pareto_rank",
            "Hard violations",
            "Target fit",
            "__domain_rank",
            "Model disagreement",
            "Nearest distance",
        ],
        ascending=[True, True, False, True, True, True],
    )

    variables = list(design_variables)
    helper_columns = ["__pareto_rank", "__domain_rank"]
    if not variables:
        return pool.head(limit).drop(columns=helper_columns, errors="ignore")

    full_values = results[variables].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mins = np.nanmin(full_values, axis=0)
    maxs = np.nanmax(full_values, axis=0)
    spans = np.where((maxs - mins) > 1e-12, maxs - mins, 1.0)
    values = pool[variables].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    normalized = (values - mins) / spans

    selected: list[int] = []
    for row_index in range(len(pool)):
        if not selected:
            selected.append(row_index)
        else:
            distances = np.linalg.norm(normalized[selected] - normalized[row_index], axis=1)
            if np.nanmin(distances) >= min_separation:
                selected.append(row_index)
        if len(selected) >= limit:
            break
    if len(selected) < min(limit, len(pool)):
        for row_index in range(len(pool)):
            if row_index not in selected:
                selected.append(row_index)
            if len(selected) >= limit:
                break
    return pool.iloc[selected].drop(columns=helper_columns, errors="ignore")


def top_fit_score(results: pd.DataFrame, top_k: int = 10) -> float:
    if results.empty:
        return float("-inf")
    ordered = results.sort_values(["Hard violations", "Target fit"], ascending=[True, False]).head(top_k)
    return float(ordered["Target fit"].mean())
