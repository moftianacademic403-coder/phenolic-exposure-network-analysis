"""Reproduce the phenolic-compound exposure–effect network analysis.

The workflow evaluates BPA, triclosan (TCS), and triclocarban (TCC) against
31 biological effect biomarkers in the exposed cohort. It applies within-
chemical FDR correction, bootstrap edge-stability analysis, covariate-adjusted
partial Spearman correlations, a cross-validated graphical LASSO sensitivity
analysis, and exports publication-ready figures and Supplementary Tables S4–S8.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr, t as student_t
from sklearn.covariance import GraphicalLassoCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

RENAME = {
    "age": "Age",
    "bminew": "BMI",
    "duration_job": "Job duration",
    "sigar": "Cigarette smoking",
    "hookah": "Hookah use",
    "Platelets": "PLT",
    "bilirobin_total": "Bilirubin T",
    "biliroubin_direct": "Bilirubin D",
    "IL6": "IL-6",
    "TNF_alpha": "TNF-α",
    "OHdG8": "8-OHdG",
    "Triclosane": "TCS",
    "Triclocarban": "TCC",
}

EXPOSURES = ["BPA", "TCS", "TCC"]
EFFECTS = [
    "WBC", "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils",
    "RBC", "HB", "HCT", "MCV", "MCH", "MCHC", "PLT", "RDW",
    "Creatinine", "Bilirubin T", "Bilirubin D", "AST", "ALT", "ALK",
    "Albumin", "BUN", "PT", "PTT", "T4", "T3", "TSH", "IL-6",
    "TNF-α", "8-OHdG", "CRP", "ESR",
]
COVARIATES = ["Age", "BMI", "Cigarette smoking", "Hookah use", "Job duration"]


def edge_name(exposure: str, effect: str) -> str:
    return f"{exposure} → {effect}"


def load_data(path: Path, sheet: str, group_code: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")
    data = pd.read_excel(path, sheet_name=sheet).rename(columns=RENAME)
    required = ["ID", "group", *EXPOSURES, *EFFECTS, *COVARIATES]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    for column in ["group", *EXPOSURES, *EFFECTS, *COVARIATES]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    exposed = data.loc[data["group"] == group_code].copy()
    if exposed.empty:
        raise ValueError(f"No observations found with group == {group_code}")
    return data, exposed


def calculate_correlations(exposed: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exposure in EXPOSURES:
        for effect in EFFECTS:
            pair = exposed[[exposure, effect]].dropna()
            rho, p_value = np.nan, np.nan
            if len(pair) >= 4 and pair[exposure].nunique() > 1 and pair[effect].nunique() > 1:
                rho, p_value = spearmanr(pair[exposure], pair[effect])
            rows.append({
                "Exposure": exposure,
                "Effect": effect,
                "Edge (exposure → effect)": edge_name(exposure, effect),
                "n": len(pair),
                "ρ": rho,
                "p (uncorrected)": p_value,
            })

    results = pd.DataFrame(rows)
    results["q (per-chemical FDR)"] = np.nan
    results["FDR-significant"] = False
    for _, indices in results.groupby("Exposure").groups.items():
        indices = np.asarray(list(indices))
        p_values = results.loc[indices, "p (uncorrected)"].to_numpy(float)
        valid = np.isfinite(p_values)
        rejected, q_values, _, _ = multipletests(p_values[valid], alpha=alpha, method="fdr_bh")
        results.loc[indices[valid], "q (per-chemical FDR)"] = q_values
        results.loc[indices[valid], "FDR-significant"] = rejected
    results["Nominal p<0.05"] = results["p (uncorrected)"] < 0.05
    return results.sort_values("p (uncorrected)", na_position="last").reset_index(drop=True)


def calculate_connectivity(nominal: pd.DataFrame) -> pd.DataFrame:
    rows: list[list[object]] = []
    for exposure in EXPOSURES:
        subset = nominal[nominal["Exposure"] == exposure]
        rows.append(["Exposure biomarker", exposure, 0, len(subset), subset["ρ"].abs().sum(), 0.0])
    for effect in [name for name in EFFECTS if name in set(nominal["Effect"])]:
        subset = nominal[nominal["Effect"] == effect]
        rows.append(["Effect biomarker", effect, len(subset), 0, 0.0, subset["ρ"].abs().sum()])
    return pd.DataFrame(rows, columns=[
        "Node class", "Node", "In-Degree", "Out-Degree",
        "Weighted (Out-Degree)", "Weighted (In-Degree)",
    ])


def bootstrap_edges(
    exposed: pd.DataFrame,
    nominal: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    robust_threshold: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for _, association in nominal.iterrows():
        pair = exposed[[association["Exposure"], association["Effect"]]].dropna().to_numpy(float)
        n = len(pair)
        coefficients = np.full(n_bootstrap, np.nan)
        stable_count = 0
        for index in range(n_bootstrap):
            sample = pair[rng.integers(0, n, n)]
            rho, p_value = spearmanr(sample[:, 0], sample[:, 1])
            coefficients[index] = rho
            stable_count += int(
                np.isfinite(rho)
                and np.sign(rho) == np.sign(association["ρ"])
                and p_value < 0.05
            )
        lower, upper = np.nanpercentile(coefficients, [2.5, 97.5])
        stability = stable_count / n_bootstrap
        rows.append({
            "Exposure": association["Exposure"],
            "Effect": association["Effect"],
            "Edge (exposure → effect)": association["Edge (exposure → effect)"],
            "ρ": association["ρ"],
            "95% bootstrap CI": f"{lower:.2f} to {upper:.2f}",
            "CI lower": lower,
            "CI upper": upper,
            "Stability index": stability,
            "Robust core (≥0.70)": stability >= robust_threshold,
        })
    return pd.DataFrame(rows).sort_values("Stability index", ascending=False).reset_index(drop=True)


def partial_spearman(data: pd.DataFrame, exposure: str, effect: str) -> tuple[float, float, int]:
    subset = data[[exposure, effect, *COVARIATES]].dropna()
    n = len(subset)
    k = len(COVARIATES)
    ranked = subset.apply(lambda series: rankdata(series.to_numpy(), method="average"))
    covariates = ranked[COVARIATES].to_numpy(float)
    exposure_residual = ranked[exposure].to_numpy(float) - LinearRegression().fit(
        covariates, ranked[exposure]
    ).predict(covariates)
    effect_residual = ranked[effect].to_numpy(float) - LinearRegression().fit(
        covariates, ranked[effect]
    ).predict(covariates)
    rho = float(pearsonr(exposure_residual, effect_residual).statistic)
    degrees_of_freedom = n - k - 2
    t_statistic = rho * np.sqrt(degrees_of_freedom / (1 - rho**2))
    p_value = float(2 * student_t.sf(abs(t_statistic), degrees_of_freedom))
    return rho, p_value, n


def calculate_adjusted_associations(exposed: pd.DataFrame, nominal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, association in nominal.iterrows():
        rho, p_value, n = partial_spearman(exposed, association["Exposure"], association["Effect"])
        rows.append({
            "Edge (exposure → effect)": association["Edge (exposure → effect)"],
            "Partial ρ (adj.)": rho,
            "Adjusted p": p_value,
            "Adjusted n": n,
            "Sig. after adj.": p_value < 0.05,
        })
    return pd.DataFrame(rows)


def fit_conditional_network(
    exposed: pd.DataFrame,
    nominal: pd.DataFrame,
) -> tuple[pd.DataFrame, float, list[str]]:
    effects = [effect for effect in EFFECTS if effect in set(nominal["Effect"])]
    variables = [*EXPOSURES, *effects]
    complete = exposed[variables].dropna()
    matrix = StandardScaler().fit_transform(complete.to_numpy(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = GraphicalLassoCV(alphas=10, cv=5, max_iter=1000, tol=1e-4).fit(matrix)
    precision = model.precision_
    denominator = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial_correlations = -precision / denominator
    np.fill_diagonal(partial_correlations, 1.0)
    index = {name: position for position, name in enumerate(variables)}
    rows: list[dict[str, object]] = []
    for _, association in nominal.iterrows():
        value = float(partial_correlations[index[association["Exposure"]], index[association["Effect"]]])
        rows.append({
            "Edge (exposure → effect)": association["Edge (exposure → effect)"],
            "Conditional (partial) ρ": value,
            "In GGM": abs(value) > 1e-8,
        })
    return pd.DataFrame(rows), float(model.alpha_), variables


def combine_robustness(
    nominal: pd.DataFrame,
    bootstrap: pd.DataFrame,
    adjusted: pd.DataFrame,
    conditional: pd.DataFrame,
) -> pd.DataFrame:
    combined = nominal[["Edge (exposure → effect)", "Nominal p<0.05", "FDR-significant"]]
    combined = combined.merge(adjusted, on="Edge (exposure → effect)")
    combined = combined.merge(
        bootstrap[["Edge (exposure → effect)", "Stability index", "Robust core (≥0.70)"]],
        on="Edge (exposure → effect)",
    )
    combined = combined.merge(conditional, on="Edge (exposure → effect)")
    return combined.sort_values(
        ["FDR-significant", "Robust core (≥0.70)", "Stability index"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def plot_network(nominal: pd.DataFrame, connectivity: pd.DataFrame, output: Path) -> None:
    graph = nx.DiGraph()
    for _, row in nominal.iterrows():
        graph.add_edge(row["Exposure"], row["Effect"], rho=row["ρ"])

    positions = {
        "BPA": (-1.25, 0.25), "TCS": (0.00, -0.45), "TCC": (1.25, 0.25),
        "MCH": (-2.25, 0.95), "MCHC": (-1.75, -0.75), "RDW": (-0.85, 1.30),
        "T3": (-2.35, -0.05), "Albumin": (-0.75, -1.38),
        "Eosinophils": (0.15, -1.48), "AST": (0.82, -1.30), "RBC": (1.68, -1.05),
        "8-OHdG": (0.20, 1.40), "TNF-α": (1.30, 1.25), "IL-6": (2.25, 0.62),
    }
    missing = [node for node in graph if node not in positions]
    if missing:
        for node, coordinates in nx.spring_layout(graph.subgraph(missing), seed=42).items():
            positions[node] = coordinates

    connectivity_index = connectivity.set_index("Node")
    exposures = [name for name in EXPOSURES if name in graph]
    effects = [name for name in graph if name not in EXPOSURES]
    exposure_sizes = [
        2800 + 1500 * connectivity_index.loc[name, "Weighted (Out-Degree)"] / 2.25
        for name in exposures
    ]
    effect_sizes = [
        1050 + 1800 * connectivity_index.loc[name, "Weighted (In-Degree)"] / 0.90
        for name in effects
    ]

    figure, axis = plt.subplots(figsize=(16, 11))
    figure.patch.set_facecolor("black")
    axis.set_facecolor("black")
    minimum = nominal["ρ"].abs().min()
    maximum = nominal["ρ"].abs().max()
    curvature = {"BPA": -0.035, "TCS": 0.0, "TCC": 0.035}
    for _, row in nominal.iterrows():
        width = 1.2 + 7 * (abs(row["ρ"]) - minimum) / (maximum - minimum)
        color = "#ef3b2c" if row["ρ"] > 0 else "#2b6cb0"
        nx.draw_networkx_edges(
            graph,
            positions,
            [(row["Exposure"], row["Effect"])],
            width=width,
            edge_color=color,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=24,
            connectionstyle=f"arc3,rad={curvature[row['Exposure']]}",
            min_source_margin=24,
            min_target_margin=30,
            ax=axis,
        )
    nx.draw_networkx_nodes(
        graph, positions, exposures, node_shape="h", node_color="#a6a6a6",
        edgecolors="#303030", linewidths=2.2, node_size=exposure_sizes, ax=axis,
    )
    nx.draw_networkx_nodes(
        graph, positions, effects, node_shape="o", node_color="#ff9138",
        edgecolors="#303030", linewidths=1.6, node_size=effect_sizes, ax=axis,
    )
    nx.draw_networkx_labels(graph, positions, font_size=10.5, font_weight="bold", ax=axis)
    legend = [
        mpatches.Patch(facecolor="#ff9138", label="Biomarkers"),
        mpatches.Patch(facecolor="#a6a6a6", label="Phenolic compounds"),
        mpatches.Patch(facecolor="#ef3b2c", label="Positive correlation"),
        mpatches.Patch(facecolor="#2b6cb0", label="Negative correlation"),
    ]
    axis.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4,
        facecolor="white", edgecolor="black", framealpha=1,
    )
    axis.set_xlim(-2.75, 2.75)
    axis.set_ylim(-1.9, 1.8)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_stability(bootstrap: pd.DataFrame, output: Path) -> None:
    data = bootstrap.sort_values("Stability index", ascending=False).reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(12, max(8, 0.52 * len(data) + 2)))
    positions = np.arange(len(data))
    for index, row in data.iterrows():
        positive = row["ρ"] > 0
        robust = row["Robust core (≥0.70)"]
        color = (
            "#2f7eaa" if positive else "#c43b2b"
        ) if robust else (
            "#8dbbd5" if positive else "#e7a097"
        )
        axis.plot(
            [row["CI lower"], row["CI upper"]], [index, index],
            color=color, linewidth=3 if robust else 1.5,
        )
        axis.plot(row["ρ"], index, "o", color=color, markersize=8 if robust else 4)
    axis.axvline(0, color="#666666", linestyle="--")
    axis.set_yticks(positions)
    axis.set_yticklabels(data["Edge (exposure → effect)"])
    axis.invert_yaxis()
    axis.set_xlabel("Bootstrap Spearman ρ (95% CI), 1,000 resamples")
    secondary = axis.twinx()
    secondary.set_ylim(axis.get_ylim())
    secondary.set_yticks(positions)
    secondary.set_yticklabels([f"{value:.2f}" for value in data["Stability index"]])
    secondary.set_ylabel("stability")
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_robustness_matrix(robustness: pd.DataFrame, output: Path) -> None:
    from matplotlib.colors import ListedColormap

    criteria = ["FDR-significant", "Sig. after adj.", "Robust core (≥0.70)", "In GGM"]
    matrix = robustness[criteria].astype(int).to_numpy()
    labels = robustness["Edge (exposure → effect)"].tolist()
    figure, axis = plt.subplots(figsize=(10, max(8, 0.48 * len(robustness) + 2)))
    axis.imshow(matrix, aspect="auto", cmap=ListedColormap(["#f4f8fb", "#123b75"]), vmin=0, vmax=1)
    axis.set_xticks(range(4))
    axis.set_xticklabels(
        ["FDR per-chemical", "Confounder-adj.", "Bootstrap≥0.7", "GGM conditional"],
        rotation=40,
        ha="right",
    )
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if matrix[row, column]:
                axis.text(column, row, "✓", ha="center", va="center", color="white", fontsize=13)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def export_tables(
    connectivity: pd.DataFrame,
    nominal: pd.DataFrame,
    bootstrap: pd.DataFrame,
    robustness: pd.DataFrame,
    conditional: pd.DataFrame,
    output: Path,
) -> None:
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        connectivity.to_excel(writer, sheet_name="Table S4", index=False)
        nominal[[
            "Edge (exposure → effect)", "ρ", "p (uncorrected)",
            "q (per-chemical FDR)", "FDR-significant",
        ]].to_excel(writer, sheet_name="Table S5", index=False)
        bootstrap[[
            "Edge (exposure → effect)", "ρ", "95% bootstrap CI",
            "Stability index", "Robust core (≥0.70)",
        ]].to_excel(writer, sheet_name="Table S6", index=False)
        robustness[[
            "Edge (exposure → effect)", "Nominal p<0.05", "FDR-significant",
            "Partial ρ (adj.)", "Sig. after adj.", "Stability index",
            "Robust core (≥0.70)", "In GGM",
        ]].to_excel(writer, sheet_name="Table S7", index=False)
        conditional.loc[
            conditional["In GGM"],
            ["Edge (exposure → effect)", "Conditional (partial) ρ"],
        ].to_excel(writer, sheet_name="Table S8", index=False)
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes(1, 0)
            worksheet.set_column(0, 0, 32)
            worksheet.set_column(1, 10, 20)


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    data, exposed = load_data(args.input, args.sheet, args.exposed_group)
    data[["ID", "group", "BPA"]].to_csv(
        args.output / "BPA_extracted_from_total.csv", index=False
    )

    all_tests = calculate_correlations(exposed, args.fdr_alpha)
    nominal = all_tests.loc[all_tests["Nominal p<0.05"]].copy().reset_index(drop=True)
    connectivity = calculate_connectivity(nominal)
    bootstrap = bootstrap_edges(
        exposed, nominal, args.bootstrap, args.seed, args.robust_threshold
    )
    adjusted = calculate_adjusted_associations(exposed, nominal)
    conditional, selected_alpha, variables = fit_conditional_network(exposed, nominal)
    robustness = combine_robustness(nominal, bootstrap, adjusted, conditional)

    export_tables(
        connectivity,
        nominal,
        bootstrap,
        robustness,
        conditional,
        args.output / "Phenolic_network_validation_tables.xlsx",
    )
    plot_network(
        nominal,
        connectivity,
        args.output / "Fig 2. Network visualization of phenolic compounds exposure and effect biomarkers.png",
    )
    plot_stability(
        bootstrap,
        args.output / "Fig 3. Edge-Weight stability of the exposure effect network.png",
    )
    plot_robustness_matrix(
        robustness,
        args.output / "Fig 4. robustness matrix of each edge across validation criteria.png",
    )

    print(f"Total sample size: {len(data)}")
    print(f"Exposed sample size: {len(exposed)}")
    print(f"Nominal edges: {len(nominal)}")
    print(f"FDR-significant edges: {int(nominal['FDR-significant'].sum())}")
    print(f"Bootstrap robust-core edges: {int(bootstrap['Robust core (≥0.70)'].sum())}")
    print(f"Significant after adjustment: {int(adjusted['Sig. after adj.'].sum())}")
    print(f"Nominal edges retained in GGM: {int(conditional['In GGM'].sum())}")
    print(f"Graphical LASSO selected alpha: {selected_alpha:.6f}")
    print(f"GGM variables: {variables}")
    print(f"Outputs saved to: {args.output.resolve()}")


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=repository_root / "data" / "total.xlsx")
    parser.add_argument("--sheet", default="total")
    parser.add_argument("--output", type=Path, default=repository_root / "outputs")
    parser.add_argument("--exposed-group", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--robust-threshold", type=float, default=0.70)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
