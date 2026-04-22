import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        default="../results/deleted_feature_analysis",
        help="Root output directory created by analyze_deleted_features.py",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=60,
        help="Maximum number of feature labels to render on the dendrogram x-axis.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=28.0,
        help="Scatter point size for UMAP plots.",
    )
    return parser.parse_args()


def load_plotting_modules():
    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram
    except ImportError as exc:
        raise SystemExit(
            "This plotting script requires matplotlib. Install it in your environment "
            "to generate dendrograms and UMAP scatterplots."
        ) from exc
    return plt, dendrogram


def analysis_bundles(root_dir):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"{root_dir} does not exist")
    bundles = []
    for metadata_path in sorted(root_dir.rglob("feature_metadata.csv")):
        bundle_dir = metadata_path.parent
        linkage_path = bundle_dir / "hierarchical_linkage.csv"
        if linkage_path.exists():
            bundles.append(bundle_dir)
    return bundles


def figure_size_for_features(n_features):
    width = min(max(10, n_features * 0.18), 24)
    height = 8
    return width, height


def plot_dendrogram(bundle_dir, plt, dendrogram_fn, max_labels):
    metadata = pd.read_csv(bundle_dir / "feature_metadata.csv")
    linkage_table = pd.read_csv(bundle_dir / "hierarchical_linkage.csv")
    if metadata.empty or linkage_table.empty:
        return None

    labels = metadata["feature_name"].astype(str).tolist()
    linkage_matrix = linkage_table[["left", "right", "distance", "count"]].to_numpy(dtype=float)
    use_truncated_view = len(labels) > max_labels

    fig, ax = plt.subplots(figsize=figure_size_for_features(len(labels)))
    dendrogram_kwargs = {
        "leaf_rotation": 90,
        "leaf_font_size": 7,
        "ax": ax,
        "color_threshold": None,
    }
    if use_truncated_view:
        dendrogram_kwargs.update(
            {
                "truncate_mode": "lastp",
                "p": max_labels,
                "show_leaf_counts": True,
            }
        )
    else:
        dendrogram_kwargs["labels"] = labels
    dendrogram_fn(linkage_matrix, **dendrogram_kwargs)
    ax.set_title(f"Hierarchical Clustering: {bundle_dir.relative_to(bundle_dir.parent.parent if bundle_dir.parent.name == 'families' else bundle_dir.parent)}")
    ax.set_ylabel("Distance")
    if use_truncated_view:
        ax.set_xlabel(
            f"Truncated dendrogram showing last {max_labels} merged clusters "
            f"({len(labels)} total features)"
        )
    else:
        ax.set_xlabel("Deleted features")
    fig.tight_layout()
    output_path = bundle_dir / "hierarchical_dendrogram.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_cluster_palette(cluster_ids, plt):
    unique_clusters = sorted(pd.Series(cluster_ids).dropna().unique().tolist())
    cmap = plt.get_cmap("tab20")
    return {cluster_id: cmap(index % cmap.N) for index, cluster_id in enumerate(unique_clusters)}


def plot_umap(bundle_dir, plt, point_size):
    umap_path = bundle_dir / "umap_projection.csv"
    if not umap_path.exists():
        return None
    umap_table = pd.read_csv(umap_path)
    if umap_table.empty:
        return None

    palette = build_cluster_palette(umap_table["cluster_id"], plt)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [palette.get(cluster_id, (0.4, 0.4, 0.4, 1.0)) for cluster_id in umap_table["cluster_id"]]
    ax.scatter(
        umap_table["umap_1"],
        umap_table["umap_2"],
        c=colors,
        s=point_size,
        alpha=0.9,
        linewidths=0.2,
        edgecolors="black",
    )

    top_flip = umap_table.copy()
    if "decision_flip_count" in top_flip.columns:
        top_flip = top_flip.sort_values(["decision_flip_count", "feature_name"], ascending=[False, True]).head(12)
        for _, row in top_flip.iterrows():
            if row.get("decision_flip_count", 0) > 0:
                ax.annotate(
                    row["feature_name"],
                    (row["umap_1"], row["umap_2"]),
                    fontsize=7,
                    alpha=0.9,
                )

    ax.set_title(f"UMAP Projection: {bundle_dir.relative_to(bundle_dir.parent.parent if bundle_dir.parent.name == 'families' else bundle_dir.parent)}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    output_path = bundle_dir / "umap_scatter.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def bundle_label(bundle_dir, root_dir):
    return str(bundle_dir.relative_to(root_dir))


def main():
    args = parse_args()
    root_dir = Path(args.analysis_dir)
    plt, dendrogram_fn = load_plotting_modules()
    bundles = analysis_bundles(root_dir)
    if not bundles:
        raise SystemExit(f"No analysis bundles found under {root_dir}")

    generated_rows = []
    for bundle_dir in bundles:
        dendrogram_path = plot_dendrogram(bundle_dir, plt, dendrogram_fn, args.max_labels)
        umap_path = plot_umap(bundle_dir, plt, args.point_size)
        generated_rows.append(
            {
                "bundle": bundle_label(bundle_dir, root_dir),
                "dendrogram_png": str(dendrogram_path) if dendrogram_path else "",
                "umap_png": str(umap_path) if umap_path else "",
            }
        )

    summary_table = pd.DataFrame(generated_rows)
    summary_table.to_csv(root_dir / "plot_outputs.csv", index=False)
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
