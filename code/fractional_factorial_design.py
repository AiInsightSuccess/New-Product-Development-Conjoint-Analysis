#!/usr/bin/env python3
"""
Generate a mixed-level fractional factorial design from CSV feature files.

Expected inputs
---------------
Feature_specs.csv:
    Features, Features_Type, Level Number, Minimum, Maximum

    The older names Parameter Name and Parameter Type are also accepted.

Feature_template.csv:
    One column per parameter. Non-empty cells define explicit levels or labels.

The script validates the template levels against Feature_specs.csv, builds the
full factorial candidate set, and then selects a smaller D-optimal fractional
design for estimating main effects. By default, numeric levels are treated as
factor levels so each declared level can be estimated.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
import warnings
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CATEGORICAL_TYPES = {"categorical", "category", "nominal"}

SPEC_COLUMN_ALIASES = {
    "name": ("Features", "Parameter Name", "Feature", "Parameter"),
    "kind": ("Features_Type", "Parameter Type", "Feature Type", "Type"),
    "level_count": ("Level Number", "Levels", "Level Count", "Number of Levels"),
    "minimum": ("Minimum", "Min"),
    "maximum": ("Maximum", "Max"),
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    level_count: int
    minimum: float | None
    maximum: float | None
    levels: tuple[Any, ...]


def clean_header(value: Any) -> str:
    """Clean BOMs and surrounding whitespace from CSV headers."""
    return str(value).replace("\ufeff", "").strip()


def comparable_name(value: Any) -> str:
    """Normalize names enough to match specs to template columns with units."""
    text = clean_header(value).lower()
    text = text.replace("dollars", "$").replace("dollar", "$")
    text = re.sub(r"[^a-z0-9]+", "", text)

    # Drop common unit suffixes that may appear in the template file only.
    for suffix in ("usd", "days", "day"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [clean_header(col) for col in df.columns]
    return df


def normalized_column_name(value: Any) -> str:
    """Normalize CSV column names for alias matching."""
    return re.sub(r"[^a-z0-9]+", "", clean_header(value).lower())


def resolve_spec_columns(specs_df: pd.DataFrame, specs_path: Path) -> dict[str, str]:
    """Map flexible Feature_specs.csv headers onto internal field names."""
    normalized_to_original = {
        normalized_column_name(column): column for column in specs_df.columns
    }
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for field_name, aliases in SPEC_COLUMN_ALIASES.items():
        matched_column = None
        for alias in aliases:
            normalized_alias = normalized_column_name(alias)
            if normalized_alias in normalized_to_original:
                matched_column = normalized_to_original[normalized_alias]
                break

        if matched_column is None:
            missing.append("/".join(aliases))
        else:
            resolved[field_name] = matched_column

    if missing:
        available = ", ".join(specs_df.columns)
        raise ValueError(
            f"{specs_path} is missing required columns: {missing}. "
            f"Available columns: {available}"
        )

    return resolved


def optional_float(value: Any) -> float | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def normalize_kind(kind: Any) -> str:
    value = clean_header(kind).lower()
    if value in {"continuous", "numeric"}:
        return "continuous"
    if value in {"discrete", "integer"}:
        return "discrete"
    if value in CATEGORICAL_TYPES:
        return "categorical"
    raise ValueError(f"Unsupported Parameter Type: {kind!r}")


def match_template_column(parameter_name: str, template_columns: list[str]) -> str | None:
    """Return the best matching template column for a spec parameter."""
    target = comparable_name(parameter_name)
    if not target:
        return None

    scored: list[tuple[float, str]] = []
    for column in template_columns:
        candidate = comparable_name(column)
        if not candidate:
            continue
        if target == candidate:
            score = 1.0
        elif target in candidate or candidate in target:
            score = 0.95
        else:
            score = SequenceMatcher(None, target, candidate).ratio()
        scored.append((score, column))

    if not scored:
        return None

    best_score, best_column = max(scored, key=lambda item: item[0])
    return best_column if best_score >= 0.62 else None


def unique_nonblank(values: pd.Series, kind: str) -> list[Any]:
    levels: list[Any] = []
    seen: set[Any] = set()

    for value in values.tolist():
        if pd.isna(value) or str(value).strip() == "":
            continue

        parsed: Any
        if kind in {"continuous", "discrete"}:
            parsed_float = float(value)
            parsed = int(parsed_float) if parsed_float.is_integer() else parsed_float
        else:
            parsed = str(value).strip()

        if parsed not in seen:
            levels.append(parsed)
            seen.add(parsed)

    if kind in {"continuous", "discrete"}:
        levels = sorted(levels, key=float)
    return levels


def generated_numeric_levels(
    name: str,
    kind: str,
    level_count: int,
    minimum: float | None,
    maximum: float | None,
) -> list[Any]:
    if minimum is None or maximum is None:
        raise ValueError(
            f"{name!r} needs either explicit template levels or Minimum/Maximum "
            "values in Feature_specs.csv."
        )

    values = np.linspace(minimum, maximum, level_count)
    if kind == "discrete":
        values = np.rint(values)

    levels: list[Any] = []
    for value in values.tolist():
        parsed = int(value) if float(value).is_integer() else round(float(value), 6)
        if parsed not in levels:
            levels.append(parsed)

    if len(levels) != level_count:
        raise ValueError(
            f"{name!r} generated {len(levels)} unique levels, but specs require "
            f"{level_count}. Provide explicit values in Feature_template.csv."
        )
    return levels


def levels_from_template_or_specs(
    *,
    name: str,
    kind: str,
    level_count: int,
    minimum: float | None,
    maximum: float | None,
    template: pd.DataFrame | None,
    strict: bool,
) -> list[Any]:
    matched_column = None
    levels: list[Any] = []

    if template is not None:
        matched_column = match_template_column(name, list(template.columns))
        if matched_column is not None:
            levels = unique_nonblank(template[matched_column], kind)

    if len(levels) == level_count:
        return levels

    if len(levels) > level_count:
        message = (
            f"{name!r} has {len(levels)} template levels in column "
            f"{matched_column!r}, but Feature_specs.csv says {level_count}. "
            f"Using the first {level_count} levels: {levels[:level_count]!r}."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message)
        return levels[:level_count]

    if kind in {"continuous", "discrete"}:
        if levels:
            message = (
                f"{name!r} has only {len(levels)} template levels, but specs "
                f"require {level_count}. Generating levels from Minimum/Maximum."
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message)
        return generated_numeric_levels(name, kind, level_count, minimum, maximum)

    if not levels:
        raise ValueError(
            f"{name!r} is categorical and needs {level_count} labels in "
            "Feature_template.csv."
        )

    raise ValueError(
        f"{name!r} has only {len(levels)} categorical levels, but specs require "
        f"{level_count}. Add the missing labels to Feature_template.csv."
    )


def read_feature_specs(
    specs_path: Path,
    template_path: Path | None,
    strict: bool,
) -> list[FeatureSpec]:
    specs_df = read_csv(specs_path)
    spec_columns = resolve_spec_columns(specs_df, specs_path)

    template_df = read_csv(template_path) if template_path else None

    specs: list[FeatureSpec] = []
    for _, row in specs_df.iterrows():
        name = clean_header(row[spec_columns["name"]])
        kind = normalize_kind(row[spec_columns["kind"]])
        level_count = int(row[spec_columns["level_count"]])
        minimum = optional_float(row[spec_columns["minimum"]])
        maximum = optional_float(row[spec_columns["maximum"]])
        levels = levels_from_template_or_specs(
            name=name,
            kind=kind,
            level_count=level_count,
            minimum=minimum,
            maximum=maximum,
            template=template_df,
            strict=strict,
        )
        specs.append(
            FeatureSpec(
                name=name,
                kind=kind,
                level_count=level_count,
                minimum=minimum,
                maximum=maximum,
                levels=tuple(levels),
            )
        )

    return specs


def full_factorial(specs: list[FeatureSpec]) -> pd.DataFrame:
    names = [spec.name for spec in specs]
    level_lists = [spec.levels for spec in specs]
    return pd.DataFrame(itertools.product(*level_lists), columns=names)


def effect_code(series: pd.Series, levels: tuple[Any, ...]) -> pd.DataFrame:
    """Effect-code a categorical factor into k - 1 columns."""
    reference = levels[-1]
    coded = {}

    for level in levels[:-1]:
        column_name = f"{series.name}={level}"
        coded[column_name] = np.where(
            series == level,
            1.0,
            np.where(series == reference, -1.0, 0.0),
        )

    return pd.DataFrame(coded, index=series.index)


def model_matrix(
    candidates: pd.DataFrame,
    specs: list[FeatureSpec],
    numeric_coding: str,
) -> pd.DataFrame:
    """Build a main-effects model matrix for D-optimal selection."""
    matrix_parts = [pd.Series(1.0, index=candidates.index, name="Intercept")]

    for spec in specs:
        if spec.kind in {"continuous", "discrete"} and numeric_coding == "linear":
            values = pd.to_numeric(candidates[spec.name], errors="raise").astype(float)
            min_value = values.min()
            max_value = values.max()
            if np.isclose(min_value, max_value):
                coded = pd.Series(0.0, index=candidates.index, name=spec.name)
            else:
                coded = 2.0 * (values - min_value) / (max_value - min_value) - 1.0
                coded.name = spec.name
            matrix_parts.append(coded)
        else:
            matrix_parts.append(effect_code(candidates[spec.name], spec.levels))

    return pd.concat(matrix_parts, axis=1).astype(float)


def logdet_information(selected_x: np.ndarray) -> float:
    information = selected_x.T @ selected_x
    sign, value = np.linalg.slogdet(information)
    if sign <= 0:
        return -np.inf
    return float(value)


def random_start(
    x: np.ndarray,
    n_runs: int,
    rng: np.random.Generator,
    starts: int,
) -> tuple[np.ndarray, float]:
    n_candidates = x.shape[0]
    best_indices: np.ndarray | None = None
    best_score = -np.inf
    best_rank = -1

    for _ in range(starts):
        indices = rng.choice(n_candidates, size=n_runs, replace=False)
        selected_x = x[indices]
        rank = int(np.linalg.matrix_rank(selected_x))
        score = logdet_information(selected_x)

        if score > best_score or (not np.isfinite(best_score) and rank > best_rank):
            best_indices = indices
            best_score = score
            best_rank = rank

    if best_indices is None or not np.isfinite(best_score):
        raise ValueError(
            "Could not find a full-rank starting design. Increase --runs or "
            "check whether any factors have duplicate or missing levels."
        )

    return best_indices, best_score


def d_optimal_fraction(
    candidates: pd.DataFrame,
    x: pd.DataFrame,
    n_runs: int,
    seed: int,
    starts: int,
    max_iter: int,
) -> pd.DataFrame:
    x_values = x.to_numpy(dtype=float)
    n_candidates, n_terms = x_values.shape

    if n_runs > n_candidates:
        raise ValueError(
            f"--runs cannot be larger than the full factorial size ({n_candidates})."
        )
    if n_runs < n_terms:
        raise ValueError(
            f"--runs must be at least {n_terms} to estimate the main-effects model."
        )
    if n_runs == n_candidates:
        return candidates.copy()

    rng = np.random.default_rng(seed)
    selected, best_score = random_start(x_values, n_runs, rng, starts)

    selected_set = set(selected.tolist())
    all_indices = set(range(n_candidates))

    for _ in range(max_iter):
        best_swap: tuple[int, int, float, np.ndarray] | None = None
        unselected = list(all_indices - selected_set)

        for position, out_index in enumerate(selected.tolist()):
            for in_index in unselected:
                trial = selected.copy()
                trial[position] = in_index
                score = logdet_information(x_values[trial])
                if score > best_score + 1e-10:
                    best_swap = (out_index, in_index, score, trial)
                    best_score = score

        if best_swap is None:
            break

        out_index, in_index, _, selected = best_swap
        selected_set.remove(out_index)
        selected_set.add(in_index)

    design = candidates.iloc[selected].copy().reset_index(drop=True)
    design.insert(0, "run_id", np.arange(1, len(design) + 1))
    return design


def default_run_count(
    n_candidates: int,
    n_terms: int,
    specs: list[FeatureSpec],
) -> int:
    """Choose a practical default that prefers balanced level counts."""
    minimum_runs = min(n_candidates, max(2 * n_terms, n_terms + 4))

    level_lcm = 1
    for spec in specs:
        level_lcm = math.lcm(level_lcm, spec.level_count)

    balanced_runs = math.ceil(minimum_runs / level_lcm) * level_lcm
    practical_limit = math.ceil(minimum_runs * 1.5)

    if balanced_runs <= n_candidates and balanced_runs <= practical_limit:
        return balanced_runs
    return minimum_runs


def write_level_summary(specs: list[FeatureSpec], path: Path) -> None:
    rows = []
    for spec in specs:
        rows.append(
            {
                "Parameter Name": spec.name,
                "Parameter Type": spec.kind,
                "Level Number": spec.level_count,
                "Levels Used": " | ".join(map(str, spec.levels)),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a mixed-level fractional factorial design from "
            "Feature_specs.csv and Feature_template.csv."
        )
    )
    parser.add_argument(
        "--specs",
        type=Path,
        default=Path("Feature_specs.csv"),
        help="Path to Feature_specs.csv.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("Feature_template.csv"),
        help="Path to Feature_template.csv with explicit levels/labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fractional_factorial_design.csv"),
        help="Output CSV path for the selected fractional design.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help=(
            "Number of profiles/runs to select. Default is a practical "
            "balanced count at least twice the number of main-effects terms."
        ),
    )
    parser.add_argument(
        "--numeric-coding",
        choices=("factor", "linear"),
        default="factor",
        help=(
            "How continuous/discrete levels enter the design search. "
            "'factor' estimates each level; 'linear' estimates one ordered trend."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the D-optimal search.",
    )
    parser.add_argument(
        "--starts",
        type=int,
        default=200,
        help="Number of random starting designs for the D-optimal search.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum number of exchange-search iterations.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when template level counts differ from specs. This is now the default.",
    )
    parser.add_argument(
        "--allow-mismatched-levels",
        action="store_true",
        help=(
            "Warn and continue when a numeric template has too few levels or "
            "a template column has more levels than Feature_specs.csv declares."
        ),
    )
    parser.add_argument(
        "--save-full-factorial",
        type=Path,
        default=None,
        help="Optional path to save the full factorial candidate set.",
    )
    parser.add_argument(
        "--save-level-summary",
        type=Path,
        default=None,
        help="Optional path to save the final levels used for each parameter.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    strict = args.strict or not args.allow_mismatched_levels
    specs = read_feature_specs(args.specs, args.template, strict=strict)
    candidates = full_factorial(specs)
    x = model_matrix(candidates, specs, numeric_coding=args.numeric_coding)

    n_terms = x.shape[1]
    n_candidates = len(candidates)
    n_runs = (
        args.runs
        if args.runs is not None
        else default_run_count(n_candidates, n_terms, specs)
    )

    design = d_optimal_fraction(
        candidates=candidates,
        x=x,
        n_runs=n_runs,
        seed=args.seed,
        starts=args.starts,
        max_iter=args.max_iter,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(args.output, index=False)

    if args.save_full_factorial:
        args.save_full_factorial.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(args.save_full_factorial, index=False)

    if args.save_level_summary:
        args.save_level_summary.parent.mkdir(parents=True, exist_ok=True)
        write_level_summary(specs, args.save_level_summary)

    print(f"Parameters: {len(specs)}")
    print(f"Full factorial profiles: {n_candidates}")
    print(f"Main-effects terms including intercept: {n_terms}")
    print(f"Fractional design profiles written: {len(design)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
