from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy.stats import spearmanr, rankdata, pearsonr, t as student_t
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLassoCV
from sklearn.exceptions import ConvergenceWarning

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent

INPUT_FILE = Path(
    os.environ.get("PHENOLIC_INPUT_FILE", PROJECT_ROOT / "data" / "total.xlsx")
)
OUTPUT_DIR = Path(
    os.environ.get("PHENOLIC_OUTPUT_DIR", PROJECT_ROOT / "outputs")
)
SHEET_NAME = "total"
EXPOSED_GROUP_CODE = 1
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
FDR_ALPHA = 0.05
ROBUST_CORE_THRESHOLD = 0.70
GGM_ZERO_TOL = 1e-8

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_RENAME = {
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
EFFECT_BIOMARKERS = [
    "WBC", "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils",
    "RBC", "HB", "HCT", "MCV", "MCH", "MCHC", "PLT", "RDW",
    "Creatinine", "Bilirubin T", "Bilirubin D", "AST", "ALT", "ALK",
    "Albumin", "BUN", "PT", "PTT", "T4", "T3", "TSH", "IL-6",
    "TNF-α", "8-OHdG", "CRP", "ESR",
]
COVARIATES = ["Age", "BMI", "Cigarette smoking", "Hookah use", "Job duration"]


def edge_name(exposure: str, effect: str) -> str:
    return f"{exposure} → {effect}"


def load_and_prepare_data(input_file: Path, sheet_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file was not found:\n{input_file}\n"
            "Set PHENOLIC_INPUT_FILE or place total.xlsx in the data directory."
        )

    raw = pd.read_excel(input_file, sheet_name=sheet_name)
    data = raw.rename(columns=COLUMN_RENAME).copy()

    required = ["ID", "group", *EXPOSURES, *EFFECT_BIOMARKERS, *COVARIATES]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"Required columns are missing: {missing}")

    numeric_columns = ["group", *EXPOSURES, *EFFECT_BIOMARKERS, *COVARIATES]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    exposed = data.loc[data["group"] == EXPOSED_GROUP_CODE].copy()
    if exposed.empty:
        raise ValueError(f"No rows found with group == {EXPOSED_GROUP_CODE}")

    print(f"Input file: {input_file}")
    print(f"Total sample size: {len(data)}")
    print(f"Exposed sample size: {len(exposed)}")
    print(f"BPA missing values in total data: {data['BPA'].isna().sum()}")
    return data, exposed


def spearman_exposure_effect(exposed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for exposure in EXPOSURES:
        for effect in EFFECT_BIOMARKERS:
            pair = exposed[[exposure, effect]].dropna()
            if len(pair) < 4 or pair[exposure].nunique() < 2 or pair[effect].nunique() < 2:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(pair[exposure], pair[effect])
            rows.append({
                "Exposure": exposure,
                "Effect": effect,
                "Edge (exposure → effect)": edge_name(exposure, effect),
                "n": len(pair),
                "ρ": float(rho) if np.isfinite(rho) else np.nan,
                "p (uncorrected)": float(p_value) if np.isfinite(p_value) else np.nan,
            })

    results = pd.DataFrame(rows)
    results["q (per-chemical FDR)"] = np.nan
    results["FDR-significant"] = False

    for exposure, idx in results.groupby("Exposure").groups.items():
        idx = list(idx)
        pvals = results.loc[idx, "p (uncorrected)"].to_numpy(float)
        valid = np.isfinite(pvals)
        if valid.any():
            reject, qvals, _, _ = multipletests(
                pvals[valid], alpha=FDR_ALPHA, method="fdr_bh"
            )
            valid_idx = np.asarray(idx)[valid]
            results.loc[valid_idx, "q (per-chemical FDR)"] = qvals
            results.loc[valid_idx, "FDR-significant"] = reject

    results["Nominal p<0.05"] = results["p (uncorrected)"] < 0.05
    return results.sort_values("p (uncorrected)", na_position="last").reset_index(drop=True)


def build_connectivity_table(nominal_edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for exposure in EXPOSURES:
        sub = nominal_edges.loc[nominal_edges["Exposure"] == exposure]
        rows.append({
            "Node class": "Exposure biomarker",
            "Node": exposure,
            "In-Degree": 0,
            "Out-Degree": int(len(sub)),
            "Weighted (Out-Degree)": float(sub["ρ"].abs().sum()),
            "Weighted (In-Degree)": 0.0,
        })

    connected_effects = [e for e in EFFECT_BIOMARKERS if e in set(nominal_edges["Effect"])]
    for effect in connected_effects:
        sub = nominal_edges.loc[nominal_edges["Effect"] == effect]
        rows.append({
            "Node class": "Effect biomarker",
            "Node": effect,
            "In-Degree": int(len(sub)),
            "Out-Degree": 0,
            "Weighted (Out-Degree)": 0.0,
            "Weighted (In-Degree)": float(sub["ρ"].abs().sum()),
        })
    return pd.DataFrame(rows)


def bootstrap_edge_stability(
    exposed: pd.DataFrame,
    nominal_edges: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output: list[dict] = []

    for _, edge in nominal_edges.iterrows():
        exposure = edge["Exposure"]
        effect = edge["Effect"]
        pair = exposed[[exposure, effect]].dropna().reset_index(drop=True)
        x = pair[exposure].to_numpy(float)
        y = pair[effect].to_numpy(float)
        n = len(pair)
        original_rho = float(edge["ρ"])

        boot_rhos = np.full(n_bootstrap, np.nan, dtype=float)
        stable_count = 0
        for b in range(n_bootstrap):
            sample_idx = rng.integers(0, n, size=n)
            rho_b, p_b = spearmanr(x[sample_idx], y[sample_idx])
            if np.isfinite(rho_b):
                boot_rhos[b] = rho_b
                same_sign = np.sign(rho_b) == np.sign(original_rho)
                if same_sign and np.isfinite(p_b) and p_b < 0.05:
                    stable_count += 1

        finite = boot_rhos[np.isfinite(boot_rhos)]
        if finite.size:
            ci_low, ci_high = np.percentile(finite, [2.5, 97.5])
        else:
            ci_low, ci_high = np.nan, np.nan

        stability = stable_count / n_bootstrap
        output.append({
            "Exposure": exposure,
            "Effect": effect,
            "Edge (exposure → effect)": edge_name(exposure, effect),
            "ρ": original_rho,
            "95% bootstrap CI": f"{ci_low:.2f} to {ci_high:.2f}",
            "CI lower": float(ci_low),
            "CI upper": float(ci_high),
            "Stability index": float(stability),
            "Robust core (≥0.70)": stability >= ROBUST_CORE_THRESHOLD,
        })

    return (
        pd.DataFrame(output)
        .sort_values("Stability index", ascending=False)
        .reset_index(drop=True)
    )


def partial_spearman(
    data: pd.DataFrame,
    x: str,
    y: str,
    covariates: list[str],
) -> tuple[float, float, int]:
    d = data[[x, y, *covariates]].dropna().copy()
    n = len(d)
    k = len(covariates)
    if n <= k + 2:
        return np.nan, np.nan, n

    ranked = d.apply(lambda s: rankdata(s.to_numpy(), method="average"))
    X_cov = ranked[covariates].to_numpy(float)
    x_rank = ranked[x].to_numpy(float)
    y_rank = ranked[y].to_numpy(float)

    x_resid = x_rank - LinearRegression().fit(X_cov, x_rank).predict(X_cov)
    y_resid = y_rank - LinearRegression().fit(X_cov, y_rank).predict(X_cov)
    rho = float(pearsonr(x_resid, y_resid).statistic)

    dfree = n - k - 2
    if not np.isfinite(rho) or abs(rho) >= 1:
        p_value = 0.0 if abs(rho) == 1 else np.nan
    else:
        t_stat = rho * np.sqrt(dfree / (1.0 - rho**2))
        p_value = float(2.0 * student_t.sf(abs(t_stat), df=dfree))
    return rho, p_value, n


def adjusted_edges(exposed: pd.DataFrame, nominal_edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, edge in nominal_edges.iterrows():
        rho, p_value, n = partial_spearman(
            exposed,
            edge["Exposure"],
            edge["Effect"],
            COVARIATES,
        )
        rows.append({
            "Edge (exposure → effect)": edge["Edge (exposure → effect)"],
            "Partial ρ (adj.)": rho,
            "Adjusted p": p_value,
            "Adjusted n": n,
            "Sig. after adj.": bool(np.isfinite(p_value) and p_value < 0.05),
        })
    return pd.DataFrame(rows)


def fit_regularised_ggm(
    exposed: pd.DataFrame,
    nominal_edges: pd.DataFrame,
) -> tuple[pd.DataFrame, float, list[str]]:
    connected_effects = [e for e in EFFECT_BIOMARKERS if e in set(nominal_edges["Effect"])]
    variables = [*EXPOSURES, *connected_effects]
    complete = exposed[variables].dropna().astype(float)

    X = StandardScaler().fit_transform(complete.to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = GraphicalLassoCV(
            alphas=10,
            cv=5,
            max_iter=1000,
            tol=1e-4,
        ).fit(X)

    precision = model.precision_
    denom = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial_corr = -precision / denom
    np.fill_diagonal(partial_corr, 1.0)

    index = {name: i for i, name in enumerate(variables)}
    rows: list[dict] = []
    nominal_pairs = set(zip(nominal_edges["Exposure"], nominal_edges["Effect"]))
    for exposure, effect in nominal_pairs:
        value = float(partial_corr[index[exposure], index[effect]])
        rows.append({
            "Exposure": exposure,
            "Effect": effect,
            "Edge (exposure → effect)": edge_name(exposure, effect),
            "Conditional (partial) ρ": value,
            "In GGM": abs(value) > GGM_ZERO_TOL,
        })

    ggm_edges = (
        pd.DataFrame(rows)
        .sort_values("Conditional (partial) ρ", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    return ggm_edges, float(model.alpha_), variables


def build_robustness_table(
    nominal_edges: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    adjusted_table: pd.DataFrame,
    ggm_table: pd.DataFrame,
) -> pd.DataFrame:
    base = nominal_edges[[
        "Edge (exposure → effect)", "Nominal p<0.05", "FDR-significant"
    ]].copy()
    out = base.merge(
        adjusted_table[[
            "Edge (exposure → effect)", "Partial ρ (adj.)", "Adjusted p", "Sig. after adj."
        ]],
        on="Edge (exposure → effect)", how="left",
    )
    out = out.merge(
        bootstrap_table[[
            "Edge (exposure → effect)", "Stability index", "Robust core (≥0.70)"
        ]],
        on="Edge (exposure → effect)", how="left",
    )
    out = out.merge(
        ggm_table[[
            "Edge (exposure → effect)", "In GGM", "Conditional (partial) ρ"
        ]],
        on="Edge (exposure → effect)", how="left",
    )
    out["In GGM"] = out["In GGM"].fillna(False).astype(bool)
    return out.sort_values(
        ["FDR-significant", "Robust core (≥0.70)", "Stability index"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def plot_network(
    nominal_edges: pd.DataFrame,
    connectivity: pd.DataFrame,
    output_path: Path,
) -> None:
    graph = nx.DiGraph()
    for _, row in nominal_edges.iterrows():
        graph.add_edge(row["Exposure"], row["Effect"], rho=row["ρ"])

    positions = {
        "BPA": (-1.25, 0.25), "TCS": (0.00, -0.45), "TCC": (1.25, 0.25),
        "MCH": (-2.25, 0.95), "MCHC": (-1.75, -0.75), "RDW": (-0.85, 1.30),
        "T3": (-2.35, -0.05), "Albumin": (-0.75, -1.38),
        "Eosinophils": (0.15, -1.48), "AST": (0.82, -1.30),
        "RBC": (1.68, -1.05), "8-OHdG": (0.20, 1.40),
        "TNF-α": (1.30, 1.25), "IL-6": (2.25, 0.62),
    }
    # Fallback positions make the function robust if a different nominal node appears.
    missing_nodes = [n for n in graph.nodes if n not in positions]
    if missing_nodes:
        fallback = nx.spring_layout(graph.subgraph(missing_nodes), seed=RANDOM_SEED)
        for n, (x, y) in fallback.items():
            positions[n] = (float(x) * 2.0, float(y) * 1.5)

    conn = connectivity.set_index("Node")
    exposure_nodes = [n for n in EXPOSURES if n in graph]
    effect_nodes = [n for n in graph.nodes if n not in EXPOSURES]
    exposure_sizes = [
        2800 + 1500 * conn.loc[n, "Weighted (Out-Degree)"] / 2.25
        for n in exposure_nodes
    ]
    special_min = {"Albumin": 3000, "Eosinophils": 3900, "8-OHdG": 2700, "TNF-α": 2400}
    effect_sizes = []
    for n in effect_nodes:
        size = 1050 + 1150 * conn.loc[n, "Weighted (In-Degree)"] / 0.90
        effect_sizes.append(max(size, special_min.get(n, 0)))

    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    abs_rho = nominal_edges["ρ"].abs()
    rho_min, rho_max = abs_rho.min(), abs_rho.max()
    for _, row in nominal_edges.iterrows():
        if rho_max > rho_min:
            width = 1.2 + 7.0 * (abs(row["ρ"]) - rho_min) / (rho_max - rho_min)
        else:
            width = 4.0
        edge_color = "#ef3b2c" if row["ρ"] > 0 else "#2b6cb0"
        curvature = {"BPA": -0.035, "TCS": 0.0, "TCC": 0.035}[row["Exposure"]]
        nx.draw_networkx_edges(
            graph, positions,
            edgelist=[(row["Exposure"], row["Effect"])],
            width=width, edge_color=edge_color, arrows=True,
            arrowstyle="-|>", arrowsize=24,
            connectionstyle=f"arc3,rad={curvature}",
            min_source_margin=24, min_target_margin=30, ax=ax,
        )

    nx.draw_networkx_nodes(
        graph, positions, nodelist=exposure_nodes, node_shape="h",
        node_color="#a6a6a6", edgecolors="#303030", linewidths=2.2,
        node_size=exposure_sizes, ax=ax,
    )
    nx.draw_networkx_nodes(
        graph, positions, nodelist=effect_nodes, node_shape="o",
        node_color="#ff9138", edgecolors="#303030", linewidths=1.6,
        node_size=effect_sizes, ax=ax,
    )
    nx.draw_networkx_labels(
        graph, positions, labels={n: n for n in graph.nodes},
        font_size=10.5, font_weight="bold", font_color="black", ax=ax,
    )

    legend_items = [
        mpatches.Patch(facecolor="#ff9138", edgecolor="#303030", label="Biomarkers"),
        mpatches.Patch(facecolor="#a6a6a6", edgecolor="#303030", label="Phenolic compounds"),
        mpatches.Patch(facecolor="#ef3b2c", label="Positive correlation"),
        mpatches.Patch(facecolor="#2b6cb0", label="Negative correlation"),
    ]
    legend = ax.legend(
        handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=4, frameon=True, facecolor="white", edgecolor="black",
        fontsize=12, framealpha=1,
    )
    for text in legend.get_texts():
        text.set_color("black")

    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-1.90, 1.80)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_bootstrap_stability(bootstrap_table: pd.DataFrame, output_path: Path) -> None:
    table = bootstrap_table.sort_values("Stability index", ascending=False).reset_index(drop=True)
    y = np.arange(len(table))
    rho = table["ρ"].to_numpy(float)
    low = table["CI lower"].to_numpy(float)
    high = table["CI upper"].to_numpy(float)
    stability = table["Stability index"].to_numpy(float)
    robust = table["Robust core (≥0.70)"].to_numpy(bool)

    fig, ax = plt.subplots(figsize=(12, max(8, 0.52 * len(table) + 2)))
    for i in range(len(table)):
        positive = rho[i] > 0
        dark = "#2f7eaa" if positive else "#c43b2b"
        light = "#8dbbd5" if positive else "#e7a097"
        color = dark if robust[i] else light
        line_width = 3.0 if robust[i] else 1.5
        marker_size = 8 if robust[i] else 4
        ax.plot([low[i], high[i]], [i, i], color=color, lw=line_width, solid_capstyle="round")
        ax.plot(rho[i], i, "o", color=color, markersize=marker_size,
                markeredgecolor="white", markeredgewidth=0.8)

    ax.axvline(0, color="#666666", linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(table["Edge (exposure → effect)"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Bootstrap Spearman ρ (95% CI), 1,000 resamples", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{x:.2f}" for x in stability], fontsize=9)
    ax2.set_ylabel("stability")
    for side in ["top", "left", "right"]:
        ax2.spines[side].set_visible(False)

    ax.legend(
        handles=[
            mpatches.Patch(color="#2f7eaa", label="positive"),
            mpatches.Patch(color="#c43b2b", label="negative"),
        ],
        loc="lower right", frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_matrix(robustness: pd.DataFrame, output_path: Path) -> None:
    criteria = [
        "FDR-significant", "Sig. after adj.",
        "Robust core (≥0.70)", "In GGM",
    ]
    matrix = robustness[criteria].astype(int).to_numpy()
    row_labels = robustness["Edge (exposure → effect)"].tolist()
    column_labels = ["FDR per-chemical", "Confounder-adj.", "Bootstrap≥0.7", "GGM conditional"]

    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(10, max(8, 0.48 * len(robustness) + 2)))
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap(["#f0f7f0", "#1b6b3c"]), vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=40, ha="right", fontsize=11)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 1:
                ax.text(j, i, "✓", ha="center", va="center", color="white", fontsize=13)

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_excel_tables(
    connectivity: pd.DataFrame,
    s5: pd.DataFrame,
    bootstrap: pd.DataFrame,
    robustness: pd.DataFrame,
    ggm_retained: pd.DataFrame,
    output_path: Path,
) -> None:
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        title_fmt = workbook.add_format({
            "bold": True, "bg_color": "#DCE6F1", "font_name": "Arial",
            "font_size": 10, "align": "left", "valign": "vcenter",
        })
        header_fmt = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#123B75",
            "font_name": "Arial", "font_size": 10,
            "align": "center", "valign": "vcenter", "text_wrap": True,
            "border": 1,
        })
        section_fmt = workbook.add_format({
            "bold": True, "bg_color": "#EEF3F8", "font_name": "Arial",
            "font_size": 10, "border": 1,
        })
        body_fmt = workbook.add_format({
            "font_name": "Arial", "font_size": 10, "border": 1,
            "valign": "vcenter",
        })
        num3_fmt = workbook.add_format({
            "font_name": "Arial", "font_size": 10, "border": 1,
            "num_format": "0.000", "align": "center",
        })
        num4_fmt = workbook.add_format({
            "font_name": "Arial", "font_size": 10, "border": 1,
            "num_format": "0.0000", "align": "center",
        })
        num2_fmt = workbook.add_format({
            "font_name": "Arial", "font_size": 10, "border": 1,
            "num_format": "0.00", "align": "center",
        })
        center_fmt = workbook.add_format({
            "font_name": "Arial", "font_size": 10, "border": 1,
            "align": "center", "valign": "vcenter",
        })

        # Table S4
        sheet = workbook.add_worksheet("Table S4")
        writer.sheets["Table S4"] = sheet
        sheet.merge_range("A1:E1", "Table S4. Connectivity measures of biomarkers in network", title_fmt)
        headers = ["", "In-Degree", "Out-Degree", "Weighted (Out-Degree)", "Weighted (In-Degree)"]
        for col, value in enumerate(headers):
            sheet.write(1, col, value, header_fmt)
        row = 2
        for node_class in ["Exposure biomarker", "Effect biomarker"]:
            sheet.merge_range(row, 0, row, 4, node_class, section_fmt)
            row += 1
            subset = connectivity.loc[connectivity["Node class"] == node_class]
            for _, r in subset.iterrows():
                sheet.write(row, 0, r["Node"], body_fmt)
                sheet.write_number(row, 1, int(r["In-Degree"]), center_fmt)
                sheet.write_number(row, 2, int(r["Out-Degree"]), center_fmt)
                sheet.write_number(row, 3, r["Weighted (Out-Degree)"], num3_fmt)
                sheet.write_number(row, 4, r["Weighted (In-Degree)"], num3_fmt)
                row += 1
        sheet.set_column("A:A", 24)
        sheet.set_column("B:E", 19)
        sheet.freeze_panes(2, 0)

        def write_dataframe_sheet(sheet_name: str, title: str, table: pd.DataFrame, formats: dict[int, object]):
            ws = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = ws
            ncols = table.shape[1]
            ws.merge_range(0, 0, 0, ncols - 1, title, title_fmt)
            for c, col_name in enumerate(table.columns):
                ws.write(1, c, col_name, header_fmt)
            for r_idx, (_, row_values) in enumerate(table.iterrows(), start=2):
                for c_idx, value in enumerate(row_values):
                    fmt = formats.get(c_idx, body_fmt)
                    if pd.isna(value):
                        ws.write_blank(r_idx, c_idx, None, fmt)
                    elif isinstance(value, (bool, np.bool_)):
                        ws.write(r_idx, c_idx, "Yes" if value else "No", center_fmt)
                    elif isinstance(value, (int, np.integer)):
                        ws.write_number(r_idx, c_idx, int(value), fmt)
                    elif isinstance(value, (float, np.floating)):
                        ws.write_number(r_idx, c_idx, float(value), fmt)
                    else:
                        ws.write(r_idx, c_idx, value, fmt)
            ws.set_column(0, 0, 31)
            ws.set_column(1, ncols - 1, 19)
            ws.set_row(0, 30)
            ws.set_row(1, 32)
            ws.freeze_panes(2, 0)

        s5_export = s5[[
            "Edge (exposure → effect)", "ρ", "p (uncorrected)",
            "q (per-chemical FDR)", "FDR-significant",
        ]].copy()
        write_dataframe_sheet(
            "Table S5",
            "Table S5. Multiple-testing correction (Benjamini–Hochberg FDR within each phenolic-compound family)",
            s5_export,
            {1: num3_fmt, 2: num4_fmt, 3: num4_fmt, 4: center_fmt},
        )

        s6_export = bootstrap[[
            "Edge (exposure → effect)", "ρ", "95% bootstrap CI",
            "Stability index", "Robust core (≥0.70)",
        ]].copy()
        write_dataframe_sheet(
            "Table S6",
            "Table S6. Non-parametric bootstrap edge stability (1,000 resamples)",
            s6_export,
            {1: num3_fmt, 2: center_fmt, 3: num2_fmt, 4: center_fmt},
        )

        s7_export = robustness[[
            "Edge (exposure → effect)", "Nominal p<0.05", "FDR-significant",
            "Partial ρ (adj.)", "Sig. after adj.", "Stability index",
            "Robust core (≥0.70)", "In GGM",
        ]].rename(columns={"FDR-significant": "FDR (per-chemical)"})
        write_dataframe_sheet(
            "Table S7",
            "Table S7. Robustness of each nominal edge across validation criteria",
            s7_export,
            {1: center_fmt, 2: center_fmt, 3: num3_fmt, 4: center_fmt,
             5: num2_fmt, 6: center_fmt, 7: center_fmt},
        )

        s8_export = ggm_retained[[
            "Edge (exposure → effect)", "Conditional (partial) ρ",
        ]].copy()
        write_dataframe_sheet(
            "Table S8",
            "Table S8. Conditional associations retained in the regularised Gaussian graphical model",
            s8_export,
            {1: num3_fmt},
        )


def main() -> None:
    data, exposed = load_and_prepare_data(INPUT_FILE, SHEET_NAME)

    bpa_output = data[["ID", "group", "BPA"]].copy()
    bpa_output.to_csv(OUTPUT_DIR / "BPA_extracted_from_total.csv", index=False, encoding="utf-8-sig")

    all_tests = spearman_exposure_effect(exposed)
    nominal = all_tests.loc[all_tests["Nominal p<0.05"]].copy().reset_index(drop=True)
    connectivity = build_connectivity_table(nominal)
    bootstrap = bootstrap_edge_stability(exposed, nominal)
    adjusted = adjusted_edges(exposed, nominal)
    ggm, ggm_alpha, ggm_variables = fit_regularised_ggm(exposed, nominal)
    robustness = build_robustness_table(nominal, bootstrap, adjusted, ggm)
    ggm_retained = ggm.loc[ggm["In GGM"]].copy().reset_index(drop=True)

    s5 = nominal.sort_values("p (uncorrected)").reset_index(drop=True)

    export_excel_tables(
        connectivity,
        s5,
        bootstrap,
        robustness,
        ggm_retained,
        OUTPUT_DIR / "Phenolic_network_validation_tables.xlsx",
    )

    plot_network(
        nominal,
        connectivity,
        OUTPUT_DIR / "Fig 2. Network visualization of phenolic compounds exposure and effect biomarkers.png",
    )
    plot_bootstrap_stability(
        bootstrap,
        OUTPUT_DIR / "Fig 3. Edge-Weight stability of the exposure effect network.png",
    )
    plot_robustness_matrix(
        robustness,
        OUTPUT_DIR / "Fig 4. robustness matrix of each edge across validation criteria.png",
    )

    print("\nAnalysis completed.")
    print(f"Nominal edges: {len(nominal)}")
    print(f"FDR-significant edges: {int(s5['FDR-significant'].sum())}")
    print(f"Bootstrap robust-core edges: {int(bootstrap['Robust core (≥0.70)'].sum())}")
    print(f"Significant after adjustment: {int(adjusted['Sig. after adj.'].sum())}")
    print(f"Nominal edges retained in GGM: {len(ggm_retained)}")
    print(f"Graphical LASSO selected alpha: {ggm_alpha:.6f}")
    print(f"GGM variables: {ggm_variables}")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")

    print("\nFDR-significant associations:")
    display_cols = [
        "Edge (exposure → effect)", "ρ", "p (uncorrected)",
        "q (per-chemical FDR)", "FDR-significant",
    ]
    print(s5.loc[s5["FDR-significant"], display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
