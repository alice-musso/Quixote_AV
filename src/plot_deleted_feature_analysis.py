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
    parser.add_argument(
        "--annotate-top-n",
        type=int,
        default=20,
        help="Number of feature names to annotate on each UMAP plot.",
    )
    parser.add_argument(
        "--strip-family-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip the feature family prefix from labels in plots.",
    )
    parser.add_argument(
        "--label-max-chars",
        type=int,
        default=48,
        help="Maximum length of plotted feature labels before truncation.",
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


def linkage_matrix_from_table(linkage_table):
    return linkage_table[["left", "right", "distance", "count"]].to_numpy(dtype=float)


def format_feature_label(feature_name, strip_family_prefix, label_max_chars):
    label = str(feature_name)
    if strip_family_prefix and ":" in label:
        label = label.split(":", 1)[1]
    if len(label) > label_max_chars:
        label = label[: max(0, label_max_chars - 3)] + "..."
    return label


def descendant_cache(linkage_matrix, n_leaves):
    cache = {}

    def descendants(node_id):
        node_id = int(node_id)
        if node_id in cache:
            return cache[node_id]
        stack = [node_id]
        leaves = []
        while stack:
            current = int(stack.pop())
            if current in cache:
                leaves.extend(cache[current])
                continue
            if current < n_leaves:
                leaves.append(current)
                continue
            left = int(linkage_matrix[current - n_leaves, 0])
            right = int(linkage_matrix[current - n_leaves, 1])
            stack.append(right)
            stack.append(left)
        cache[node_id] = leaves
        return leaves

    return descendants


def representative_feature_label(node_id, labels, linkage_matrix, strip_family_prefix, label_max_chars):
    n_leaves = len(labels)
    node_id = int(node_id)
    if node_id < n_leaves:
        return format_feature_label(labels[node_id], strip_family_prefix, label_max_chars)

    descendants = descendant_cache(linkage_matrix, n_leaves)(node_id)
    representative_names = [
        format_feature_label(labels[index], strip_family_prefix, label_max_chars)
        for index in descendants[:3]
    ]
    representative_text = " | ".join(representative_names)
    if len(descendants) > 3:
        representative_text += f" | ... (n={len(descendants)})"
    return representative_text


def plot_dendrogram(bundle_dir, plt, dendrogram_fn, max_labels, strip_family_prefix, label_max_chars):
    metadata = pd.read_csv(bundle_dir / "feature_metadata.csv")
    linkage_table = pd.read_csv(bundle_dir / "hierarchical_linkage.csv")
    if metadata.empty or linkage_table.empty:
        return None

    labels = metadata["feature_name"].astype(str).tolist()
    linkage_matrix = linkage_matrix_from_table(linkage_table)
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
                "leaf_label_func": lambda node_id: representative_feature_label(
                    node_id,
                    labels,
                    linkage_matrix,
                    strip_family_prefix,
                    label_max_chars,
                ),
            }
        )
    else:
        dendrogram_kwargs["labels"] = [
            format_feature_label(label, strip_family_prefix, label_max_chars)
            for label in labels
        ]
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


def plot_umap(bundle_dir, plt, point_size, strip_family_prefix, label_max_chars):
    umap_path = bundle_dir / "umap_projection.csv"
    if not umap_path.exists():
        return None
    umap_table = pd.read_csv(umap_path)
    if umap_table.empty:
        return None

    metadata_path = bundle_dir / "feature_metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if "feature_name" in umap_table.columns:
            umap_table = umap_table.merge(
                metadata,
                on=["feature_name", "cluster_id"],
                how="left",
            )

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
    annotation_candidates = pd.DataFrame()
    if "decision_flip_count" in top_flip.columns:
        annotation_candidates = top_flip.sort_values(
            ["decision_flip_count", "contrast", "feature_name"],
            ascending=[False, False, True],
        )
        annotation_candidates = annotation_candidates[annotation_candidates["decision_flip_count"] > 0]

    if annotation_candidates.empty:
        sort_columns = [column for column in ["contrast", "document_prevalence", "feature_name"] if column in umap_table.columns]
        if sort_columns:
            ascending = [False if column != "feature_name" else True for column in sort_columns]
            annotation_candidates = umap_table.sort_values(sort_columns, ascending=ascending)
        else:
            annotation_candidates = umap_table.copy()

    for _, row in annotation_candidates.head(plot_umap.annotate_top_n).iterrows():
        ax.annotate(
            format_feature_label(row["feature_name"], strip_family_prefix, label_max_chars),
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


def plot_pca(bundle_dir, plt, point_size, strip_family_prefix, label_max_chars):
    pca_path = bundle_dir / "pca_projection.csv"
    if not pca_path.exists():
        return None
    pca_table = pd.read_csv(pca_path)
    if pca_table.empty:
        return None

    metadata_path = bundle_dir / "feature_metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if "feature_name" in pca_table.columns:
            pca_table = pca_table.merge(
                metadata,
                on="feature_name",
                how="left",
                suffixes=("", "_meta"),
            )

    cluster_column = "kmeans_cluster_id"
    if cluster_column not in pca_table.columns:
        return None

    palette = build_cluster_palette(pca_table[cluster_column], plt)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [palette.get(cluster_id, (0.4, 0.4, 0.4, 1.0)) for cluster_id in pca_table[cluster_column]]
    ax.scatter(
        pca_table["pca_1"],
        pca_table["pca_2"],
        c=colors,
        s=point_size,
        alpha=0.9,
        linewidths=0.2,
        edgecolors="black",
    )

    annotation_candidates = pd.DataFrame()
    representative_path = bundle_dir / "kmeans_top_features.csv"
    if representative_path.exists():
        representatives = pd.read_csv(representative_path)
        if not representatives.empty:
            annotation_candidates = pca_table.merge(
                representatives[["cluster_id", "rank_within_cluster", "feature_name"]],
                left_on=[cluster_column, "feature_name"],
                right_on=["cluster_id", "feature_name"],
                how="inner",
            ).sort_values(["cluster_id", "rank_within_cluster"])

    if annotation_candidates.empty:
        sort_columns = [column for column in ["decision_flip_count", "contrast", "feature_name"] if column in pca_table.columns]
        if sort_columns:
            ascending = [False if column != "feature_name" else True for column in sort_columns]
            annotation_candidates = pca_table.sort_values(sort_columns, ascending=ascending)
        else:
            annotation_candidates = pca_table.copy()

    for _, row in annotation_candidates.head(plot_pca.annotate_top_n).iterrows():
        ax.annotate(
            format_feature_label(row["feature_name"], strip_family_prefix, label_max_chars),
            (row["pca_1"], row["pca_2"]),
            fontsize=7,
            alpha=0.9,
        )

    variance_1 = pca_table["explained_variance_ratio_1"].iloc[0] if "explained_variance_ratio_1" in pca_table.columns else None
    variance_2 = pca_table["explained_variance_ratio_2"].iloc[0] if "explained_variance_ratio_2" in pca_table.columns else None
    axis_1 = "PCA 1"
    axis_2 = "PCA 2"
    if variance_1 is not None and variance_2 is not None:
        axis_1 = f"PCA 1 ({variance_1 * 100:.1f}%)"
        axis_2 = f"PCA 2 ({variance_2 * 100:.1f}%)"

    ax.set_title(f"PCA Projection (K-Means): {bundle_dir.relative_to(bundle_dir.parent.parent if bundle_dir.parent.name == 'families' else bundle_dir.parent)}")
    ax.set_xlabel(axis_1)
    ax.set_ylabel(axis_2)
    fig.tight_layout()
    output_path = bundle_dir / "pca_scatter.png"
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
    plot_umap.annotate_top_n = args.annotate_top_n
    plot_pca.annotate_top_n = args.annotate_top_n
    for bundle_dir in bundles:
        dendrogram_path = plot_dendrogram(
            bundle_dir,
            plt,
            dendrogram_fn,
            args.max_labels,
            args.strip_family_prefix,
            args.label_max_chars,
        )
        umap_path = plot_umap(
            bundle_dir,
            plt,
            args.point_size,
            args.strip_family_prefix,
            args.label_max_chars,
        )
        pca_path = plot_pca(
            bundle_dir,
            plt,
            args.point_size,
            args.strip_family_prefix,
            args.label_max_chars,
        )
        generated_rows.append(
            {
                "bundle": bundle_label(bundle_dir, root_dir),
                "dendrogram_png": str(dendrogram_path) if dendrogram_path else "",
                "umap_png": str(umap_path) if umap_path else "",
                "pca_png": str(pca_path) if pca_path else "",
            }
        )

    summary_table = pd.DataFrame(generated_rows)
    summary_table.to_csv(root_dir / "plot_outputs.csv", index=False)
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
