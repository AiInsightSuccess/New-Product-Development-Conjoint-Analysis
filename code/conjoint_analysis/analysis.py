from __future__ import annotations

import argparse
import itertools
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2


OUTCOME_COLUMN = "outcome"
RUN_ID_COLUMN = "run_id"


@dataclass(frozen=True)
class LevelSpec:
    feature: str
    feature_type: str
    level_count: int
    levels: tuple[str, ...]


@dataclass(frozen=True)
class EncodedDesign:
    dataset: pd.DataFrame
    all_indicator_columns: list[str]
    model_columns: list[str]
    term_lookup: dict[str, tuple[str, str]]
    reference_levels: dict[str, str]


def clean_header(value: Any) -> str:
    return str(value).replace("\ufeff", "").strip()


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [clean_header(column) for column in df.columns]
    return df


def normalize_type(value: Any) -> str:
    text = clean_header(value).lower()
    if text in {"continuous", "numeric"}:
        return "continuous"
    if text in {"discrete", "integer"}:
        return "discrete"
    if text in {"categorical", "category", "nominal"}:
        return "categorical"
    return text


def normalize_level(value: Any, feature_type: str) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if feature_type in {"continuous", "discrete"} and text != "":
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
        return str(round(numeric, 6))

    return text


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    return text.strip("_") or "value"


def indicator_name(feature: str, level: str) -> str:
    return f"{safe_name(feature)}__{safe_name(level)}"


def load_level_specs(levels_path: Path) -> list[LevelSpec]:
    levels_df = read_csv(levels_path)
    required_columns = {
        "Parameter Name",
        "Parameter Type",
        "Level Number",
        "Levels Used",
    }
    missing = required_columns - set(levels_df.columns)
    if missing:
        raise ValueError(f"{levels_path} is missing required columns: {sorted(missing)}")

    specs: list[LevelSpec] = []
    for _, row in levels_df.iterrows():
        feature = clean_header(row["Parameter Name"])
        feature_type = normalize_type(row["Parameter Type"])
        levels = tuple(
            normalize_level(level, feature_type)
            for level in str(row["Levels Used"]).split("|")
            if str(level).strip() != ""
        )
        level_count = int(row["Level Number"])

        if len(levels) != level_count:
            raise ValueError(
                f"{feature!r} has {len(levels)} levels in {levels_path}, "
                f"but Level Number says {level_count}."
            )

        specs.append(
            LevelSpec(
                feature=feature,
                feature_type=feature_type,
                level_count=level_count,
                levels=levels,
            )
        )

    return specs


def load_design(design_path: Path, level_specs: list[LevelSpec]) -> pd.DataFrame:
    design = read_csv(design_path)
    if RUN_ID_COLUMN not in design.columns:
        raise ValueError(f"{design_path} must contain a {RUN_ID_COLUMN!r} column.")

    design[RUN_ID_COLUMN] = pd.to_numeric(design[RUN_ID_COLUMN], errors="raise").astype(int)
    if design[RUN_ID_COLUMN].duplicated().any():
        duplicates = design.loc[design[RUN_ID_COLUMN].duplicated(), RUN_ID_COLUMN].tolist()
        raise ValueError(f"{design_path} has duplicated run_id values: {duplicates}")

    for spec in level_specs:
        if spec.feature not in design.columns:
            raise ValueError(
                f"{design_path} is missing feature column {spec.feature!r} "
                f"from {Path('levels_used.csv')}."
            )
        design[spec.feature] = design[spec.feature].map(
            lambda value, feature_type=spec.feature_type: normalize_level(value, feature_type)
        )
        observed_levels = set(design[spec.feature].dropna().astype(str))
        known_levels = set(spec.levels)
        unknown_levels = sorted(observed_levels - known_levels)
        if unknown_levels:
            raise ValueError(
                f"{design_path} contains levels for {spec.feature!r} that are "
                f"not in levels_used.csv: {unknown_levels}"
            )

    return design


def find_participant_column(responses: pd.DataFrame, participant_col: str | None) -> str:
    if participant_col and participant_col in responses.columns:
        return participant_col

    aliases = ("Participant_ID", "participant_id", "Participant", "Respondent_ID")
    for alias in aliases:
        if alias in responses.columns:
            return alias

    return responses.columns[0]


def response_run_columns(responses: pd.DataFrame) -> list[str]:
    run_columns = [
        column
        for column in responses.columns
        if re.fullmatch(r"run_id_\d+", str(column))
    ]
    if not run_columns:
        raise ValueError(
            "Survey response file must contain wide response columns named "
            "run_id_1, run_id_2, ..."
        )
    return sorted(run_columns, key=lambda column: int(str(column).split("_")[-1]))


def load_responses(
    responses_path: Path,
    participant_col: str | None,
) -> tuple[pd.DataFrame, str]:
    responses = read_csv(responses_path)
    resolved_participant_col = find_participant_column(responses, participant_col)
    return responses, resolved_participant_col


def build_long_dataset(
    responses: pd.DataFrame,
    participant_col: str,
    design: pd.DataFrame,
    level_specs: list[LevelSpec],
) -> pd.DataFrame:
    run_columns = response_run_columns(responses)
    long_responses = responses.melt(
        id_vars=[participant_col],
        value_vars=run_columns,
        var_name="response_column",
        value_name=OUTCOME_COLUMN,
    )
    long_responses[RUN_ID_COLUMN] = (
        long_responses["response_column"]
        .str.extract(r"run_id_(\d+)", expand=False)
        .astype(int)
    )
    long_responses = long_responses.drop(columns=["response_column"])
    long_responses[OUTCOME_COLUMN] = pd.to_numeric(
        long_responses[OUTCOME_COLUMN], errors="raise"
    ).astype(int)

    invalid_outcomes = sorted(set(long_responses[OUTCOME_COLUMN]) - {0, 1})
    if invalid_outcomes:
        raise ValueError(
            f"Outcome columns must be binary 0/1. Found: {invalid_outcomes}"
        )

    missing_run_ids = sorted(set(long_responses[RUN_ID_COLUMN]) - set(design[RUN_ID_COLUMN]))
    if missing_run_ids:
        raise ValueError(
            f"Survey responses refer to run_id values missing from the design: "
            f"{missing_run_ids}"
        )

    feature_columns = [spec.feature for spec in level_specs]
    model_data = long_responses.merge(
        design[[RUN_ID_COLUMN, *feature_columns]],
        on=RUN_ID_COLUMN,
        how="left",
        validate="many_to_one",
    )

    return model_data.sort_values([participant_col, RUN_ID_COLUMN]).reset_index(drop=True)


def encode_dataset(
    model_data: pd.DataFrame,
    level_specs: list[LevelSpec],
) -> EncodedDesign:
    encoded = model_data.copy()
    all_indicator_columns: list[str] = []
    model_columns: list[str] = []
    term_lookup: dict[str, tuple[str, str]] = {}
    reference_levels: dict[str, str] = {}

    for spec in level_specs:
        reference_level = spec.levels[0]
        reference_levels[spec.feature] = reference_level

        for index, level in enumerate(spec.levels):
            column = indicator_name(spec.feature, level)
            encoded[column] = (encoded[spec.feature].astype(str) == level).astype(int)
            all_indicator_columns.append(column)
            term_lookup[column] = (spec.feature, level)
            if index > 0:
                model_columns.append(column)

    return EncodedDesign(
        dataset=encoded,
        all_indicator_columns=all_indicator_columns,
        model_columns=model_columns,
        term_lookup=term_lookup,
        reference_levels=reference_levels,
    )


def fit_logistic_model(
    encoded: EncodedDesign,
    participant_col: str,
    use_cluster_se: bool,
) -> tuple[Any, Any]:
    y = encoded.dataset[OUTCOME_COLUMN].astype(float)
    x = encoded.dataset[encoded.model_columns].astype(float)
    x = sm.add_constant(x, has_constant="add")

    model = sm.GLM(y, x, family=sm.families.Binomial())
    if use_cluster_se:
        result = model.fit(
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": encoded.dataset[participant_col]},
        )
    else:
        result = model.fit(maxiter=200)

    null_x = pd.DataFrame({"const": np.ones(len(y))}, index=encoded.dataset.index)
    null_model = sm.GLM(y, null_x, family=sm.families.Binomial())
    null_result = null_model.fit(maxiter=200)
    return result, null_result


def model_metrics(
    result: Any,
    null_result: Any,
    model_data: pd.DataFrame,
    participant_col: str,
    n_features: int,
) -> pd.DataFrame:
    llf = float(result.llf)
    llnull = float(null_result.llf)
    lr_stat = 2.0 * (llf - llnull)
    df_model = int(result.df_model)

    metrics = {
        "n_observations": len(model_data),
        "n_participants": model_data[participant_col].nunique(),
        "n_profiles": model_data[RUN_ID_COLUMN].nunique(),
        "n_features": n_features,
        "positive_outcomes": int(model_data[OUTCOME_COLUMN].sum()),
        "positive_rate": float(model_data[OUTCOME_COLUMN].mean()),
        "log_likelihood": llf,
        "null_log_likelihood": llnull,
        "likelihood_ratio_chi_square": lr_stat,
        "likelihood_ratio_df": df_model,
        "likelihood_ratio_p_value": float(chi2.sf(lr_stat, df_model)),
        "mcfadden_pseudo_r2": float(1.0 - (llf / llnull)),
        "aic": float(result.aic),
        "bic": float(result.bic_llf),
    }
    return pd.DataFrame([metrics])


def coefficient_table(
    result: Any,
    level_specs: list[LevelSpec],
    encoded: EncodedDesign,
    alpha: float,
) -> pd.DataFrame:
    conf_int = result.conf_int(alpha=alpha)
    rows: list[dict[str, Any]] = []

    for spec in level_specs:
        reference_level = encoded.reference_levels[spec.feature]
        for level in spec.levels:
            column = indicator_name(spec.feature, level)
            is_reference = level == reference_level

            if is_reference:
                coefficient = 0.0
                std_error = np.nan
                z_value = np.nan
                p_value = np.nan
                ci_lower = np.nan
                ci_upper = np.nan
            else:
                coefficient = float(result.params[column])
                std_error = float(result.bse[column])
                z_value = float(result.tvalues[column])
                p_value = float(result.pvalues[column])
                ci_lower = float(conf_int.loc[column, 0])
                ci_upper = float(conf_int.loc[column, 1])

            rows.append(
                {
                    "feature": spec.feature,
                    "level": level,
                    "reference_level": reference_level,
                    "is_reference": is_reference,
                    "coefficient": coefficient,
                    "std_error": std_error,
                    "z_value": z_value,
                    "p_value": p_value,
                    "odds_ratio": float(np.exp(coefficient)),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "odds_ratio_ci_lower": float(np.exp(ci_lower))
                    if not np.isnan(ci_lower)
                    else np.nan,
                    "odds_ratio_ci_upper": float(np.exp(ci_upper))
                    if not np.isnan(ci_upper)
                    else np.nan,
                    "significant": False
                    if is_reference or np.isnan(p_value)
                    else bool(p_value < alpha),
                }
            )

    return pd.DataFrame(rows)


def feature_significance_table(
    result: Any,
    level_specs: list[LevelSpec],
    alpha: float,
) -> pd.DataFrame:
    cov = result.cov_params()
    rows: list[dict[str, Any]] = []

    for spec in level_specs:
        columns = [indicator_name(spec.feature, level) for level in spec.levels[1:]]
        beta = result.params[columns].to_numpy(dtype=float)
        covariance = cov.loc[columns, columns].to_numpy(dtype=float)
        chi_square = float(beta.T @ np.linalg.pinv(covariance) @ beta)
        df = len(columns)
        p_value = float(chi2.sf(chi_square, df))
        rows.append(
            {
                "feature": spec.feature,
                "reference_level": spec.levels[0],
                "degrees_of_freedom": df,
                "wald_chi_square": chi_square,
                "p_value": p_value,
                "significant": bool(p_value < alpha),
            }
        )

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def build_all_combinations(level_specs: list[LevelSpec]) -> pd.DataFrame:
    features = [spec.feature for spec in level_specs]
    levels = [spec.levels for spec in level_specs]
    return pd.DataFrame(itertools.product(*levels), columns=features)


def add_model_columns(
    combinations: pd.DataFrame,
    level_specs: list[LevelSpec],
    model_columns: list[str],
) -> pd.DataFrame:
    encoded = combinations.copy()
    for spec in level_specs:
        for level in spec.levels[1:]:
            column = indicator_name(spec.feature, level)
            encoded[column] = (encoded[spec.feature].astype(str) == level).astype(float)

    x = encoded[model_columns].astype(float)
    return sm.add_constant(x, has_constant="add")


def predicted_combinations(
    result: Any,
    level_specs: list[LevelSpec],
    model_columns: list[str],
) -> pd.DataFrame:
    combinations = build_all_combinations(level_specs)
    x = add_model_columns(combinations, level_specs, model_columns)
    combinations["predicted_probability"] = result.predict(x)
    combinations["predicted_odds"] = (
        combinations["predicted_probability"] / (1.0 - combinations["predicted_probability"])
    )
    return combinations.sort_values(
        "predicted_probability", ascending=False
    ).reset_index(drop=True)


def write_outputs(
    output_dir: Path,
    model_data: pd.DataFrame,
    encoded: EncodedDesign,
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    feature_significance: pd.DataFrame,
    combinations: pd.DataFrame,
    top_n: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "model_dataset": output_dir / "conjoint_model_dataset.csv",
        "encoded_dataset": output_dir / "conjoint_encoded_dataset.csv",
        "metrics": output_dir / "conjoint_model_metrics.csv",
        "coefficients": output_dir / "conjoint_coefficients.csv",
        "feature_significance": output_dir / "conjoint_feature_significance.csv",
        "significant_features": output_dir / "conjoint_significant_features.csv",
        "significant_levels": output_dir / "conjoint_significant_levels.csv",
        "best_combination": output_dir / "conjoint_best_combination.csv",
        "top_combinations": output_dir / "conjoint_top_combinations.csv",
    }

    model_data.to_csv(paths["model_dataset"], index=False)

    encoded_columns = [
        column
        for column in encoded.dataset.columns
        if column not in encoded.all_indicator_columns
    ]
    encoded.dataset[[*encoded_columns, *encoded.all_indicator_columns]].to_csv(
        paths["encoded_dataset"], index=False
    )

    metrics.to_csv(paths["metrics"], index=False)
    coefficients.to_csv(paths["coefficients"], index=False)
    feature_significance.to_csv(paths["feature_significance"], index=False)
    feature_significance.loc[feature_significance["significant"]].to_csv(
        paths["significant_features"], index=False
    )
    coefficients.loc[coefficients["significant"]].to_csv(
        paths["significant_levels"], index=False
    )
    combinations.head(1).to_csv(paths["best_combination"], index=False)
    combinations.head(top_n).to_csv(paths["top_combinations"], index=False)
    return paths


def run_analysis(
    responses_path: Path,
    levels_path: Path,
    design_path: Path,
    output_dir: Path,
    participant_col: str | None = None,
    alpha: float = 0.05,
    top_n: int = 10,
    use_cluster_se: bool = True,
) -> dict[str, Any]:
    level_specs = load_level_specs(levels_path)
    design = load_design(design_path, level_specs)
    responses, resolved_participant_col = load_responses(responses_path, participant_col)
    model_data = build_long_dataset(
        responses=responses,
        participant_col=resolved_participant_col,
        design=design,
        level_specs=level_specs,
    )
    encoded = encode_dataset(model_data, level_specs)
    result, null_result = fit_logistic_model(
        encoded=encoded,
        participant_col=resolved_participant_col,
        use_cluster_se=use_cluster_se,
    )
    metrics = model_metrics(
        result=result,
        null_result=null_result,
        model_data=model_data,
        participant_col=resolved_participant_col,
        n_features=len(level_specs),
    )
    coefficients = coefficient_table(
        result=result,
        level_specs=level_specs,
        encoded=encoded,
        alpha=alpha,
    )
    feature_significance = feature_significance_table(
        result=result,
        level_specs=level_specs,
        alpha=alpha,
    )
    combinations = predicted_combinations(
        result=result,
        level_specs=level_specs,
        model_columns=encoded.model_columns,
    )
    paths = write_outputs(
        output_dir=output_dir,
        model_data=model_data,
        encoded=encoded,
        metrics=metrics,
        coefficients=coefficients,
        feature_significance=feature_significance,
        combinations=combinations,
        top_n=top_n,
    )

    return {
        "paths": paths,
        "metrics": metrics,
        "coefficients": coefficients,
        "feature_significance": feature_significance,
        "combinations": combinations,
        "participant_col": resolved_participant_col,
        "model_result": result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a logistic-regression choice-based conjoint model."
    )
    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="Path to wide Survey_response.csv with run_id_1, run_id_2, ... columns.",
    )
    parser.add_argument(
        "--levels",
        type=Path,
        default=Path("levels_used.csv"),
        help="Path to levels_used.csv.",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("fractional_factorial_design.csv"),
        help="Path to fractional_factorial_design.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conjoint_analysis_outputs"),
        help="Directory for analysis output CSV files.",
    )
    parser.add_argument(
        "--participant-col",
        default=None,
        help="Participant identifier column. Defaults to Participant_ID or first column.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for p-values.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of highest-probability combinations to save.",
    )
    parser.add_argument(
        "--no-cluster-se",
        action="store_true",
        help="Use ordinary model standard errors instead of participant-clustered SEs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        results = run_analysis(
            responses_path=args.responses,
            levels_path=args.levels,
            design_path=args.design,
            output_dir=args.output_dir,
            participant_col=args.participant_col,
            alpha=args.alpha,
            top_n=args.top_n,
            use_cluster_se=not args.no_cluster_se,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    metrics = results["metrics"].iloc[0]
    best_combination = results["combinations"].iloc[0]
    significant_features = results["feature_significance"].loc[
        results["feature_significance"]["significant"], "feature"
    ].tolist()

    print("Conjoint analysis complete")
    print(f"Model dataset rows: {int(metrics['n_observations'])}")
    print(f"Participants: {int(metrics['n_participants'])}")
    print(f"Profiles: {int(metrics['n_profiles'])}")
    print(f"McFadden pseudo R2: {metrics['mcfadden_pseudo_r2']:.6f}")
    print(f"Model LR p-value: {metrics['likelihood_ratio_p_value']:.6g}")
    if significant_features:
        print(f"Significant features: {', '.join(significant_features)}")
    else:
        print("Significant features: none at the selected alpha")
    print(f"Highest predicted probability: {best_combination['predicted_probability']:.6f}")
    print(f"Outputs written to: {args.output_dir}")
    return 0
