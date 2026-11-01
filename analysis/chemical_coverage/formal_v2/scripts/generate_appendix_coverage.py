#!/usr/bin/env python3
"""Generate reproducible appendix coverage statistics and figures.

The counting unit is one parent reaction-condition screening table (N=552).
Machine classifications are taken only from the human-reviewed final output.
Product descriptors use the benchmark product SMILES plus the documented
human-review overrides for five known extraction cross-links.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA


COLORS = {
    "blue": "#4C78A8",
    "orange": "#E6862A",
    "red": "#D95555",
    "green": "#4F956B",
    "teal": "#43A9A3",
    "purple": "#8567B1",
    "gold": "#E3AD2F",
    "navy": "#2F4B66",
    "gray": "#A7A7A7",
    "light_blue": "#DCEAF5",
    "light_orange": "#F7E2C7",
}

PRIMARY_LABELS = {
    "CROSS_COUPLING": "Cross-coupling",
    "OTHER_COUPLING": "Other coupling",
    "ADDITION": "Addition",
    "SUBSTITUTION": "Substitution",
    "ELIMINATION": "Elimination",
    "CYCLIZATION_OR_ANNULATION": "Cyclization / annulation",
    "CYCLOADDITION": "Cycloaddition",
    "C_H_FUNCTIONALIZATION": "C-H functionalization",
    "FUNCTIONAL_GROUP_INTERCONVERSION": "Functional-group interconversion",
    "OXIDATION": "Oxidation",
    "REDUCTION": "Reduction",
    "HYDROGENATION_OR_TRANSFER_HYDROGENATION": "Hydrogenation / transfer H",
    "CARBONYLATION": "Carbonylation",
    "BORATION_OR_BORYLATION": "Boration / borylation",
    "SILYLATION": "Silylation",
    "HALOGENATION": "Halogenation",
    "DEAROMATIZATION_OR_AROMATIZATION": "Dearomatization / aromatization",
    "REARRANGEMENT": "Rearrangement",
    "RESOLUTION_OR_DESYMMETRIZATION": "Resolution / desymmetrization",
    "OTHER": "Other",
}

BOND_LABELS = {
    "C-C_FORMATION": "C-C formation",
    "C-N_FORMATION": "C-N formation",
    "C-O_FORMATION": "C-O formation",
    "C-S_FORMATION": "C-S formation",
    "C-B_FORMATION": "C-B formation",
    "C-P_FORMATION": "C-P formation",
    "C-Si_FORMATION": "C-Si formation",
    "C-HALOGEN_FORMATION": "C-halogen formation",
    "N-N_FORMATION": "N-N formation",
    "OTHER_BOND_FORMATION": "Other bond formation",
    "BOND_CLEAVAGE": "Bond cleavage",
    "NO_CLEAR_NET_BOND_CHANGE": "No clear net change",
}

ENERGY_LABELS = {
    "THERMAL_HEATING": "Thermal heating",
    "PHOTOCHEMICAL": "Photochemical",
    "ELECTROCHEMICAL": "Electrochemical",
    "MECHANOCHEMICAL": "Mechanochemical",
    "OTHER_ENERGY_INPUT": "Other energy",
    "AMBIENT_OR_NOT_EXPLICIT": "Ambient / not explicit",
}

CATALYST_LABELS = {
    "TRANSITION_METAL_CATALYSIS": "Transition-metal",
    "MAIN_GROUP_METAL_CATALYSIS": "Main-group metal",
    "ORGANOCATALYSIS": "Organocatalysis",
    "LEWIS_ACID_CATALYSIS": "Lewis acid",
    "BRONSTED_ACID_CATALYSIS": "Bronsted acid",
    "BRONSTED_BASE_CATALYSIS": "Bronsted base",
    "BIOCATALYSIS": "Biocatalysis",
    "HETEROGENEOUS_NONMETAL_CATALYSIS": "Heterogeneous nonmetal",
    "NO_CATALYST": "No catalyst",
    "OTHER_CATALYSIS": "Other catalysis",
}

CONDITION_LABELS = {
    "reagents": "Reagents",
    "solvents": "Solvents",
    "catalysts": "Catalysts",
    "reaction_time": "Time",
    "reaction_temperature": "Temperature",
    "atmosphere": "Atmosphere",
    "light_condition": "Light",
    "ligand": "Ligand",
    "Electrolyte": "Electrolyte",
    "Anode(+)": "Anode",
    "Cathode(-)": "Cathode",
    "pressure": "Pressure",
}

MAJOR_CONDITIONS = list(CONDITION_LABELS)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump_tsv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.6,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax: plt.Axes, axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=axis, color="#D7D7D7", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(rf"$\bf{{{letter}}}$  {title}", loc="left", pad=5)


def add_bar_labels(
    ax: plt.Axes,
    bars: Any,
    total: int,
    fontsize: float = 5.8,
    percent_only: bool = False,
) -> None:
    maximum = max(bar.get_width() for bar in bars) if bars else 0
    ax.set_xlim(0, maximum * 1.24 if maximum else 1)
    for bar in bars:
        count = int(round(bar.get_width()))
        pct = 100 * count / total
        label = f"{pct:.1f}%" if percent_only else f"{count} ({pct:.1f}%)"
        ax.text(
            bar.get_width() + maximum * 0.018,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=fontsize,
            color="#333333",
        )


def sorted_rows(
    counts: Counter[str], labels: dict[str, str], total: int
) -> list[dict[str, Any]]:
    return [
        {
            "label": key,
            "display_label": labels.get(key, key.replace("_", " ").title()),
            "table_count": count,
            "table_pct": 100 * count / total,
        }
        for key, count in counts.most_common()
    ]


def collect_chemistry(
    final_rows: list[dict[str, Any]], output: Path
) -> dict[str, list[dict[str, Any]]]:
    primary: Counter[str] = Counter()
    bonds: Counter[str] = Counter()
    energy: Counter[str] = Counter()
    catalysts: Counter[str] = Counter()
    radicals: Counter[str] = Counter()
    for row in final_rows:
        classification = row["classification"]
        primary[classification["primary_transformation"]] += 1
        bonds.update(classification["bond_changes"])
        energy.update(classification["energy_sources"])
        catalysts.update(classification["catalyst_classes"])
        radicals[classification["radical_mechanism"]] += 1

    n = len(final_rows)
    result = {
        "primary": sorted_rows(primary, PRIMARY_LABELS, n),
        "bond": sorted_rows(bonds, BOND_LABELS, n),
        "energy": sorted_rows(energy, ENERGY_LABELS, n),
        "catalyst": sorted_rows(catalysts, CATALYST_LABELS, n),
        "radical": sorted_rows(radicals, {}, n),
    }
    for name in ("primary", "bond", "energy", "catalyst", "radical"):
        dump_tsv(
            output / f"{name}_coverage.tsv",
            ["label", "display_label", "table_count", "table_pct"],
            result[name],
        )
    return result


def collect_conditions(
    source_items: list[dict[str, Any]], output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "variable": 0,
            "constant": 0,
            "candidate_arities": [],
            "values": set(),
        }
    )
    varied_axes: list[int] = []
    legal_combinations: list[int] = []

    def value_text(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (int, float, bool)):
            return [str(value)]
        if isinstance(value, list):
            return [
                text
                for child in value
                for text in value_text(child)
            ]
        if isinstance(value, dict):
            if "content" in value:
                return value_text(value["content"])
            if "value" in value:
                return value_text(value["value"])
            return [
                text
                for key, child in sorted(value.items())
                if key not in {"SMILES", "smiles", "index"}
                for text in value_text(child)
            ]
        return []

    for item in source_items:
        query_keys = list(item.get("query_keys") or [])
        input_record = item.get("input") or {}
        constants = input_record.get("constant_conditions") or {}
        candidates = input_record.get("candidate_values") or {}
        varied_axes.append(len(query_keys))
        legal_combinations.append(
            len((item.get("meta") or {}).get("legal_combinations") or [])
        )
        for key, value in constants.items():
            records[key]["constant"] += 1
            records[key]["values"].update(x.lower() for x in value_text(value))
        for key in query_keys:
            options = candidates.get(key) or []
            records[key]["variable"] += 1
            records[key]["candidate_arities"].append(len(options))
            for option in options:
                records[key]["values"].update(
                    x.lower() for x in value_text(option.get("value"))
                )

    rows = []
    n = len(source_items)
    for key, record in records.items():
        arities = record["candidate_arities"]
        present = record["variable"] + record["constant"]
        rows.append(
            {
                "condition_key": key,
                "display_label": CONDITION_LABELS.get(key, key),
                "tables_present": present,
                "coverage_pct": 100 * present / n,
                "variable_tables": record["variable"],
                "variable_pct": 100 * record["variable"] / n,
                "constant_tables": record["constant"],
                "constant_pct": 100 * record["constant"] / n,
                "distinct_text_values": len(record["values"]),
                "candidate_arity_median": (
                    statistics.median(arities) if arities else None
                ),
                "candidate_arity_q1": (
                    float(np.percentile(arities, 25)) if arities else None
                ),
                "candidate_arity_q3": (
                    float(np.percentile(arities, 75)) if arities else None
                ),
                "candidate_arity_max": max(arities) if arities else None,
                "candidate_arities": arities,
            }
        )
    rows.sort(key=lambda row: (-row["tables_present"], row["condition_key"]))
    dump_tsv(
        output / "condition_coverage.tsv",
        [
            "condition_key",
            "display_label",
            "tables_present",
            "coverage_pct",
            "variable_tables",
            "variable_pct",
            "constant_tables",
            "constant_pct",
            "distinct_text_values",
            "candidate_arity_median",
            "candidate_arity_q1",
            "candidate_arity_q3",
            "candidate_arity_max",
        ],
        rows,
    )

    summary = {
        "condition_field_count": len(rows),
        "varied_axes_per_table": numeric_summary(varied_axes),
        "legal_combinations_per_table": numeric_summary(legal_combinations),
        "varied_axes_counts": dict(sorted(Counter(varied_axes).items())),
        "legal_combination_counts": dict(
            sorted(Counter(legal_combinations).items())
        ),
    }
    return rows, summary


def numeric_summary(values: list[float | int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "min": float(np.min(array)),
        "q1": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q3": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return max(fragments, key=lambda item: item.GetNumHeavyAtoms())


def collect_products(
    source_items: list[dict[str, Any]],
    final_by_id: dict[str, dict[str, Any]],
    overrides: dict[str, dict[str, str]],
    output: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, tuple[float, float, str]],
]:
    rows: list[dict[str, Any]] = []
    for item in source_items:
        record_id = item["meta"]["parent_table_id"]
        products = (item.get("input") or {}).get("products") or []
        raw_smiles = (item.get("meta") or {}).get("main_product_smiles")
        content = products[0].get("content", "") if products else ""
        overridden = record_id in overrides
        if overridden:
            raw_smiles = overrides[record_id]["smiles"]
            content = overrides[record_id]["content"]
        mol = Chem.MolFromSmiles(raw_smiles or "")
        if mol is None:
            rows.append(
                {
                    "record_id": record_id,
                    "content": content,
                    "input_smiles": raw_smiles or "",
                    "valid_smiles": False,
                    "human_override": overridden,
                    "primary_transformation": final_by_id[record_id][
                        "classification"
                    ]["primary_transformation"],
                }
            )
            continue

        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        fragment = largest_fragment(mol)
        scaffold = MurckoScaffold.GetScaffoldForMol(fragment)
        scaffold_smiles = (
            Chem.MolToSmiles(scaffold, isomericSmiles=False)
            if scaffold.GetNumAtoms()
            else ""
        )
        atom_rings = fragment.GetRingInfo().AtomRings()
        heterocycle = any(
            any(fragment.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in ring)
            for ring in atom_rings
        )
        chiral_centers = Chem.FindMolChiralCenters(
            fragment, includeUnassigned=True
        )
        rows.append(
            {
                "record_id": record_id,
                "content": content,
                "input_smiles": raw_smiles,
                "canonical_smiles": canonical,
                "largest_fragment_smiles": Chem.MolToSmiles(
                    fragment, isomericSmiles=True
                ),
                "valid_smiles": True,
                "human_override": overridden,
                "primary_transformation": final_by_id[record_id][
                    "classification"
                ]["primary_transformation"],
                "molecular_weight": Descriptors.MolWt(fragment),
                "clogp": Crippen.MolLogP(fragment),
                "tpsa": rdMolDescriptors.CalcTPSA(fragment),
                "hbd": Lipinski.NumHDonors(fragment),
                "hba": Lipinski.NumHAcceptors(fragment),
                "rotatable_bonds": Lipinski.NumRotatableBonds(fragment),
                "heavy_atoms": fragment.GetNumHeavyAtoms(),
                "hetero_atoms": sum(
                    atom.GetAtomicNum() not in {1, 6}
                    for atom in fragment.GetAtoms()
                ),
                "ring_count": rdMolDescriptors.CalcNumRings(fragment),
                "aromatic_ring_count": (
                    rdMolDescriptors.CalcNumAromaticRings(fragment)
                ),
                "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(fragment),
                "chiral_centers": len(chiral_centers),
                "heterocycle": heterocycle,
                "murcko_scaffold": scaffold_smiles,
            }
        )

    fieldnames = list(rows[0])
    dump_tsv(output / "product_descriptors.tsv", fieldnames, rows)

    unique_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("valid_smiles"):
            unique_groups[row["canonical_smiles"]].append(row)
    unique_rows: list[dict[str, Any]] = []
    for canonical, group in unique_groups.items():
        representative = dict(group[0])
        category_counts = Counter(
            row["primary_transformation"] for row in group
        )
        representative["primary_transformation"] = category_counts.most_common(
            1
        )[0][0]
        representative["record_count"] = len(group)
        unique_rows.append(representative)

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    )
    fingerprints = []
    bit_arrays = []
    for row in unique_rows:
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        fingerprint = generator.GetFingerprint(mol)
        fingerprints.append(fingerprint)
        bits = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, bits)
        bit_arrays.append(bits)

    pca = PCA(n_components=2, random_state=0)
    coordinates = pca.fit_transform(np.asarray(bit_arrays, dtype=float))
    pca_map: dict[str, tuple[float, float, str]] = {}
    pca_rows = []
    for row, (pc1, pc2) in zip(unique_rows, coordinates):
        pca_map[row["canonical_smiles"]] = (
            float(pc1),
            float(pc2),
            row["primary_transformation"],
        )
        pca_rows.append(
            {
                "canonical_smiles": row["canonical_smiles"],
                "pc1": float(pc1),
                "pc2": float(pc2),
                "primary_transformation": row["primary_transformation"],
            }
        )
    dump_tsv(
        output / "product_fingerprint_pca.tsv",
        ["canonical_smiles", "pc1", "pc2", "primary_transformation"],
        pca_rows,
    )

    similarities = []
    for index, fingerprint in enumerate(fingerprints):
        similarities.extend(
            DataStructs.BulkTanimotoSimilarity(
                fingerprint, fingerprints[index + 1 :]
            )
        )
    scaffold_counts = Counter(
        row["murcko_scaffold"]
        for row in unique_rows
        if row["murcko_scaffold"]
    )
    dump_tsv(
        output / "scaffold_frequencies.tsv",
        ["murcko_scaffold", "unique_product_count"],
        [
            {
                "murcko_scaffold": scaffold,
                "unique_product_count": count,
            }
            for scaffold, count in scaffold_counts.most_common()
        ],
    )

    descriptor_names = [
        "molecular_weight",
        "clogp",
        "tpsa",
        "heavy_atoms",
        "hetero_atoms",
        "ring_count",
        "aromatic_ring_count",
        "rotatable_bonds",
        "fraction_csp3",
        "chiral_centers",
    ]
    descriptor_summary = {
        name: numeric_summary([float(row[name]) for row in unique_rows])
        for name in descriptor_names
    }
    summary = {
        "product_records": len(rows),
        "valid_product_records": sum(row["valid_smiles"] for row in rows),
        "human_structure_overrides": len(overrides),
        "unique_canonical_products": len(unique_rows),
        "unique_nonempty_murcko_scaffolds": len(scaffold_counts),
        "acyclic_unique_products": sum(
            not row["murcko_scaffold"] for row in unique_rows
        ),
        "scaffold_singletons": sum(
            count == 1 for count in scaffold_counts.values()
        ),
        "scaffold_singleton_pct": (
            100
            * sum(count == 1 for count in scaffold_counts.values())
            / len(scaffold_counts)
        ),
        "largest_scaffold_family": max(scaffold_counts.values()),
        "largest_scaffold_family_pct": (
            100 * max(scaffold_counts.values()) / len(unique_rows)
        ),
        "internal_diversity": 1 - float(np.mean(similarities)),
        "heterocycle_unique_products": sum(
            row["heterocycle"] for row in unique_rows
        ),
        "chiral_unique_products": sum(
            row["chiral_centers"] > 0 for row in unique_rows
        ),
        "descriptor_summary": descriptor_summary,
        "pca_explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_
        ],
    }
    return rows, unique_rows, summary, pca_map


def plot_chemistry(
    chemistry: dict[str, list[dict[str, Any]]],
    n_tables: int,
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 7.1))
    panels = [
        ("primary", "A", "Primary transformations", COLORS["red"], 20),
        ("bond", "B", "Net bond changes (multi-label)", COLORS["blue"], 12),
        ("energy", "C", "Energy inputs (multi-label)", COLORS["green"], 8),
        (
            "catalyst",
            "D",
            "Catalyst classes (multi-label)",
            COLORS["purple"],
            10,
        ),
    ]
    for ax, (key, letter, title, color, limit) in zip(axes.flat, panels):
        rows = chemistry[key][:limit][::-1]
        bars = ax.barh(
            np.arange(len(rows)),
            [row["table_count"] for row in rows],
            color=color,
            alpha=0.9,
            height=0.72,
        )
        ax.set_yticks(
            np.arange(len(rows)),
            [row["display_label"] for row in rows],
        )
        ax.set_xlabel("Parent screening tables")
        panel_title(ax, letter, title)
        style_axes(ax)
        add_bar_labels(ax, bars, n_tables, fontsize=5.3)
    fig.subplots_adjust(
        left=0.22,
        right=0.985,
        top=0.965,
        bottom=0.07,
        hspace=0.28,
        wspace=0.58,
    )
    save_figure(fig, figure_dir / "appendix_chemical_coverage")


def plot_conditions(
    condition_rows: list[dict[str, Any]],
    condition_summary: dict[str, Any],
    n_tables: int,
    figure_dir: Path,
) -> None:
    figure = plt.figure(figsize=(7.15, 5.7))
    grid = figure.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    axes = np.asarray(
        [
            [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])],
            [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
        ]
    )
    by_key = {row["condition_key"]: row for row in condition_rows}
    selected = [
        by_key[key] for key in MAJOR_CONDITIONS if key in by_key
    ][::-1]
    y = np.arange(len(selected))
    variable = [row["variable_tables"] for row in selected]
    constant = [row["constant_tables"] for row in selected]
    axes[0, 0].barh(
        y,
        variable,
        color=COLORS["orange"],
        label="Varied / queried",
        height=0.72,
    )
    axes[0, 0].barh(
        y,
        constant,
        left=variable,
        color="#E4D2AC",
        label="Constant",
        height=0.72,
    )
    axes[0, 0].set_yticks(
        y, [row["display_label"] for row in selected]
    )
    axes[0, 0].set_xlim(0, n_tables)
    axes[0, 0].xaxis.set_major_formatter(
        PercentFormatter(xmax=n_tables, decimals=0)
    )
    axes[0, 0].set_xlabel("Parent screening tables")
    axes[0, 0].legend(frameon=False, loc="lower right")
    panel_title(axes[0, 0], "A", "Condition-field coverage")
    style_axes(axes[0, 0])

    arity_keys = [
        "reagents",
        "solvents",
        "catalysts",
        "reaction_time",
        "reaction_temperature",
        "atmosphere",
        "light_condition",
        "ligand",
    ]
    arity_data = [
        by_key[key]["candidate_arities"]
        for key in arity_keys
        if key in by_key
    ]
    arity_labels = [
        CONDITION_LABELS[key] for key in arity_keys if key in by_key
    ]
    box = axes[0, 1].boxplot(
        arity_data,
        tick_labels=arity_labels,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"linewidth": 0.7},
        capprops={"linewidth": 0.7},
        boxprops={"linewidth": 0.7},
    )
    palette = [
        COLORS["red"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["purple"],
        COLORS["gold"],
        COLORS["teal"],
        COLORS["orange"],
        COLORS["navy"],
    ]
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[0, 1].tick_params(axis="x", rotation=32)
    for label in axes[0, 1].get_xticklabels():
        label.set_ha("right")
    axes[0, 1].set_ylabel("Candidate values per varied field")
    axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    panel_title(axes[0, 1], "B", "Candidate-set arity")
    style_axes(axes[0, 1], axis="y")

    varied_counts = {
        int(key): value
        for key, value in condition_summary["varied_axes_counts"].items()
    }
    xs = list(range(min(varied_counts), max(varied_counts) + 1))
    bars = axes[1, 0].bar(
        xs,
        [varied_counts.get(value, 0) for value in xs],
        color=COLORS["blue"],
        alpha=0.88,
        width=0.78,
    )
    median_axes = condition_summary["varied_axes_per_table"]["median"]
    axes[1, 0].axvline(
        median_axes,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.1,
        label=f"Median = {median_axes:g}",
    )
    axes[1, 0].set_xlabel("Varied condition axes per table")
    axes[1, 0].set_ylabel("Number of tables")
    axes[1, 0].legend(frameon=False)
    panel_title(axes[1, 0], "C", "Optimization dimensionality")
    style_axes(axes[1, 0], axis="y")

    legal = {
        int(key): value
        for key, value in condition_summary[
            "legal_combination_counts"
        ].items()
    }
    legal_values = [
        value
        for value, count in legal.items()
        for _ in range(count)
    ]
    bins = np.arange(1.5, max(legal_values) + 1.6, 2)
    axes[1, 1].hist(
        legal_values,
        bins=bins,
        color=COLORS["green"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.4,
    )
    median_legal = condition_summary["legal_combinations_per_table"]["median"]
    axes[1, 1].axvline(
        median_legal,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.1,
        label=f"Median = {median_legal:g}",
    )
    axes[1, 1].set_xlabel("Observed condition combinations per table")
    axes[1, 1].set_ylabel("Number of tables")
    axes[1, 1].legend(frameon=False)
    panel_title(axes[1, 1], "D", "Experimental screen size")
    style_axes(axes[1, 1], axis="y")

    figure.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.96,
        bottom=0.12,
        hspace=0.48,
        wspace=0.34,
    )
    save_figure(figure, figure_dir / "appendix_condition_coverage")


def plot_products(
    unique_rows: list[dict[str, Any]],
    product_summary: dict[str, Any],
    pca_map: dict[str, tuple[float, float, str]],
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.9))
    category_counts = Counter(
        category for _, _, category in pca_map.values()
    )
    top_categories = [
        category for category, _ in category_counts.most_common(6)
    ]
    palette = [
        COLORS["red"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["purple"],
        COLORS["gold"],
        COLORS["teal"],
    ]
    for category, color in zip(top_categories, palette):
        points = [
            (x, y)
            for x, y, label in pca_map.values()
            if label == category
        ]
        coordinates = np.asarray(points)
        axes[0, 0].scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=11,
            alpha=0.68,
            color=color,
            edgecolors="none",
            label=PRIMARY_LABELS.get(category, category),
        )
    other = [
        (x, y)
        for x, y, label in pca_map.values()
        if label not in top_categories
    ]
    if other:
        coordinates = np.asarray(other)
        axes[0, 0].scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=9,
            alpha=0.28,
            color=COLORS["gray"],
            edgecolors="none",
            label="Other classes",
        )
    variance = product_summary["pca_explained_variance_ratio"]
    axes[0, 0].set_xlabel(f"PC1 ({100 * variance[0]:.1f}% variance)")
    axes[0, 0].set_ylabel(f"PC2 ({100 * variance[1]:.1f}% variance)")
    axes[0, 0].legend(
        frameon=False,
        ncol=2,
        fontsize=5.2,
        handletextpad=0.35,
        columnspacing=0.7,
    )
    panel_title(axes[0, 0], "A", "Morgan-fingerprint product space")
    axes[0, 0].grid(color="#D7D7D7", linewidth=0.5, alpha=0.55)

    scatter = axes[0, 1].scatter(
        [row["molecular_weight"] for row in unique_rows],
        [row["clogp"] for row in unique_rows],
        c=[row["tpsa"] for row in unique_rows],
        cmap="viridis",
        s=13,
        alpha=0.72,
        edgecolors="none",
    )
    axes[0, 1].set_xlabel("Molecular weight (Da)")
    axes[0, 1].set_ylabel("cLogP")
    colorbar = fig.colorbar(
        scatter, ax=axes[0, 1], fraction=0.047, pad=0.04
    )
    colorbar.set_label(r"TPSA ($\mathrm{\AA^2}$)")
    colorbar.ax.tick_params(labelsize=6)
    panel_title(axes[0, 1], "B", "Physicochemical span")
    axes[0, 1].grid(color="#D7D7D7", linewidth=0.5, alpha=0.55)

    descriptors = [
        ("heavy_atoms", "Heavy atoms"),
        ("ring_count", "Rings"),
        ("rotatable_bonds", "Rotatable bonds"),
        ("chiral_centers", "Chiral centers"),
    ]
    data = [
        [float(row[key]) for row in unique_rows]
        for key, _ in descriptors
    ]
    violin = axes[1, 0].violinplot(
        data,
        showmedians=True,
        showextrema=False,
        widths=0.82,
    )
    for body, color in zip(
        violin["bodies"],
        [
            COLORS["blue"],
            COLORS["gold"],
            COLORS["green"],
            COLORS["purple"],
        ],
    ):
        body.set_facecolor(color)
        body.set_alpha(0.72)
    violin["cmedians"].set_color("#222222")
    axes[1, 0].set_xticks(
        np.arange(1, len(descriptors) + 1),
        [label for _, label in descriptors],
        rotation=22,
        ha="right",
    )
    axes[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
    panel_title(axes[1, 0], "C", "Structural-property distributions")
    style_axes(axes[1, 0], axis="y")

    scaffold_counts = Counter(
        row["murcko_scaffold"]
        for row in unique_rows
        if row["murcko_scaffold"]
    )
    family_sizes = Counter(scaffold_counts.values())
    xs = sorted(family_sizes)
    axes[1, 1].bar(
        xs,
        [family_sizes[value] for value in xs],
        color=COLORS["red"],
        alpha=0.84,
        width=0.82,
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Unique products per Murcko scaffold")
    axes[1, 1].set_ylabel("Number of scaffold families")
    panel_title(axes[1, 1], "D", "Scaffold-family size distribution")
    style_axes(axes[1, 1], axis="y")

    fig.subplots_adjust(
        left=0.1,
        right=0.965,
        top=0.96,
        bottom=0.13,
        hspace=0.48,
        wspace=0.34,
    )
    save_figure(fig, figure_dir / "appendix_product_coverage")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve()
    formal = here.parents[1]
    workspace = formal.parent
    release_root = formal.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-results",
        type=Path,
        default=formal / "outputs/final_results_human_reviewed.jsonl",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=release_root / "data/original_json/fill_in_blank_all_blank.json",
    )
    parser.add_argument(
        "--product-overrides",
        type=Path,
        default=formal / "analysis/product_structure_overrides.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=formal / "analysis",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=workspace / "figures",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    final_rows = list(read_jsonl(args.final_results))
    source_items = json.loads(args.source.read_text(encoding="utf-8"))
    overrides = json.loads(args.product_overrides.read_text(encoding="utf-8"))
    final_by_id = {row["record_id"]: row for row in final_rows}
    source_ids = {
        item["meta"]["parent_table_id"] for item in source_items
    }
    if len(final_rows) != 552 or len(source_items) != 552:
        raise SystemExit("Expected 552 final rows and 552 source tables")
    if set(final_by_id) != source_ids:
        raise SystemExit("Final result IDs do not match source table IDs")
    if not set(overrides).issubset(source_ids):
        raise SystemExit("Product overrides contain unknown record IDs")

    chemistry = collect_chemistry(final_rows, args.output_dir)
    condition_rows, condition_summary = collect_conditions(
        source_items, args.output_dir
    )
    _, unique_products, product_summary, pca_map = collect_products(
        source_items,
        final_by_id,
        overrides,
        args.output_dir,
    )
    plot_chemistry(chemistry, len(final_rows), args.figure_dir)
    plot_conditions(
        condition_rows,
        condition_summary,
        len(final_rows),
        args.figure_dir,
    )
    plot_products(
        unique_products,
        product_summary,
        pca_map,
        args.figure_dir,
    )

    final_summary = json.loads(
        (formal / "outputs/final_summary_human_reviewed.json").read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "schema_version": "1.0.0",
        "counting_unit": "parent reaction-condition screening table",
        "parent_table_count": len(final_rows),
        "article_count": 232,
        "question_count": 4773,
        "human_reviewed_classification_count": final_summary[
            "manual_reviewed_count"
        ],
        "remaining_human_review_count": final_summary[
            "human_review_queue_count"
        ],
        "chemistry": {
            key: rows for key, rows in chemistry.items()
        },
        "conditions": condition_summary,
        "products": product_summary,
        "software": {
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (args.output_dir / "appendix_coverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
