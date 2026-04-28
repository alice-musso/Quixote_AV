import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from authorship_verification import AuthorshipVerification
from data_preparation.data_loader import binarize_corpus, load_corpus
from quijote_classifier.quijote_experiment import QuijoteAblationExperiment
from quijote_classifier.supervised_term_weighting.tsr_functions import posneg_information_gain


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="../corpus/training")
    parser.add_argument("--positive-author", default="Cervantes")
    parser.add_argument("--target-title", default="Quijote")
    parser.add_argument("--classifier-type", choices=["lr", "svm"], default="lr")
    parser.add_argument("--hyperparams-save", default="../hyperparams/hyperparameters.pkl")
    parser.add_argument("--ablation-path", default=None)
    parser.add_argument("--decision-changes-path", default=None)
    parser.add_argument("--output-dir", default="../results/deleted_feature_analysis")
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--nn-k", type=int, default=5)
    parser.add_argument("--umap-neighbors", type=int, default=10)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument(
        "--split-by-family",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the analysis overall and separately for each feature family.",
    )
    parser.add_argument(
        "--load-hyperparams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load saved hyperparameters; use --no-load-hyperparams to recompute them.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=5000)
    return parser.parse_args()


def build_verifier_config(args):
    return SimpleNamespace(
        positive_author=args.positive_author,
        classifier_type=args.classifier_type,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        max_features=args.max_features,
    )


def resolve_hyperparams_path(args):
    base_path = Path(args.hyperparams_save)
    return base_path.parent / f"hyperparameters_posauth_{args.positive_author}.pkl"


def load_hyperparams(path):
    with Path(path).open("rb") as hyper_file:
        return pickle.load(hyper_file)


def build_topic_document_records(books, target_title):
    records = []
    documents = []
    y = []
    groups = []
    for group_id, book in enumerate(books):
        is_target = int(target_title.lower() in book.title.lower())
        documents.append(book.processed)
        y.append(is_target)
        groups.append(group_id)
        records.append(
            {
                "document_index": len(documents) - 1,
                "group_id": group_id,
                "title": book.title,
                "author": book.original_author,
                "is_target_title": is_target,
                "document_kind": "book",
                "segment_index": None,
            }
        )
        for segment_index, fragment in enumerate(book.segmented or []):
            documents.append(fragment)
            y.append(is_target)
            groups.append(group_id)
            records.append(
                {
                    "document_index": len(documents) - 1,
                    "group_id": group_id,
                    "title": book.title,
                    "author": book.original_author,
                    "is_target_title": is_target,
                    "document_kind": "segment",
                    "segment_index": segment_index,
                }
            )
    return documents, np.asarray(y, dtype=int), groups, pd.DataFrame(records)


def load_deleted_feature_table(path):
    path = Path(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def load_decision_change_table(path):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def compute_ablation_if_needed(args, verifier, verifier_artifacts, train_corpus):
    ablation_experiment = QuijoteAblationExperiment(
        target_title=args.target_title,
        positive_author=args.positive_author,
    )
    positive_author_books = ablation_experiment.cervantes_only(train_corpus)
    topic_documents, y_quijote, topic_groups, topic_metadata = build_topic_document_records(
        positive_author_books,
        args.target_title,
    )
    positive_author_train_matrix = verifier.transform_documents_with_selection(
        topic_documents,
        verifier_artifacts.feature_selection,
    )

    if args.ablation_path:
        deleted_feature_table = load_deleted_feature_table(args.ablation_path)
        deleted_feature_table = deleted_feature_table.sort_values("deleted_order").reset_index(drop=True)
        deleted_features = deleted_feature_table["feature_index"].astype(int).tolist()
        deleted_feature_names = deleted_feature_table["feature_name"].astype(str).tolist()
        return (
            positive_author_train_matrix,
            y_quijote,
            topic_metadata,
            deleted_feature_table,
            deleted_features,
            deleted_feature_names,
        )

    feature_ranking, _ = ablation_experiment.compute_feature_ranking(
        X=positive_author_train_matrix,
        y=y_quijote,
        random_state=args.random_state,
        tsr_metric=posneg_information_gain,
    )
    classifier = verifier.new_classifier().set_params(
        C=verifier_artifacts.hyperparams["C"],
        class_weight=verifier_artifacts.hyperparams["class_weight"],
    )
    ablation_artifacts = ablation_experiment.ablate(
        feature_ranking=feature_ranking,
        X=positive_author_train_matrix,
        X_test=positive_author_train_matrix,
        y=y_quijote,
        groups=topic_groups,
        classifier=classifier,
        feature_names=verifier_artifacts.feature_selection.selected_feature_names,
    )
    deleted_feature_table = pd.DataFrame(
        {
            "deleted_order": np.arange(1, len(ablation_artifacts.deleted_features) + 1),
            "feature_index": ablation_artifacts.deleted_features,
            "feature_name": ablation_artifacts.deleted_feature_names,
        }
    )
    return (
        positive_author_train_matrix,
        y_quijote,
        topic_metadata,
        deleted_feature_table,
        ablation_artifacts.deleted_features,
        ablation_artifacts.deleted_feature_names,
    )


def dense_feature_matrix(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X, dtype=float)


def sanitize_feature_vectors(feature_vectors):
    feature_vectors = np.asarray(feature_vectors, dtype=float)
    feature_vectors = np.nan_to_num(feature_vectors, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_vectors


def feature_family(feature_name):
    return feature_name.split(":", 1)[0]


def decision_flip_counts(decision_change_table, tracked_author):
    if decision_change_table is None or decision_change_table.empty:
        return pd.Series(dtype=int)
    filtered = decision_change_table.copy()
    if "classifier_author" in filtered.columns:
        filtered = filtered[filtered["classifier_author"] == tracked_author]
    filtered = filtered[filtered["decision_changed"] == True]  # noqa: E712
    if filtered.empty or "change_feature_index" not in filtered.columns:
        return pd.Series(dtype=int)
    valid = filtered["change_feature_index"].dropna().astype(int)
    if valid.empty:
        return pd.Series(dtype=int)
    return valid.value_counts().sort_index()


def attach_feature_statistics(metadata, deleted_matrix, y_quijote, flip_counts):
    nonzero = (deleted_matrix != 0).astype(float)
    metadata = metadata.copy()
    metadata["quijote_mean"] = deleted_matrix[y_quijote == 1].mean(axis=0)
    metadata["non_quijote_mean"] = deleted_matrix[y_quijote == 0].mean(axis=0)
    metadata["contrast"] = metadata["quijote_mean"] - metadata["non_quijote_mean"]
    metadata["document_prevalence"] = nonzero.mean(axis=0)
    metadata["decision_flip_count"] = metadata["feature_index"].map(flip_counts).fillna(0).astype(int)
    metadata["decision_flip_feature"] = metadata["decision_flip_count"] > 0
    return metadata


def choose_cluster_count(n_features, requested_clusters):
    if n_features <= 1:
        return 1
    return max(2, min(requested_clusters, n_features))


def choose_elbow_cluster_count(inertia_table):
    if inertia_table.empty:
        return 1
    if len(inertia_table) == 1:
        return int(inertia_table.iloc[0]["k"])

    coordinates = inertia_table[["k", "inertia"]].to_numpy(dtype=float)
    start = coordinates[0]
    end = coordinates[-1]
    baseline = end - start
    baseline_norm = np.linalg.norm(baseline)
    if baseline_norm == 0.0:
        return int(inertia_table.iloc[0]["k"])

    distances = []
    for point in coordinates:
        offset = point - start
        distance = abs(np.cross(baseline, offset)) / baseline_norm
        distances.append(float(distance))
    best_index = int(np.argmax(distances))
    return int(inertia_table.iloc[best_index]["k"])


def compute_hierarchical_labels(feature_vectors, n_clusters):
    feature_vectors = sanitize_feature_vectors(feature_vectors)
    if feature_vectors.shape[0] == 1:
        return np.array([1], dtype=int), pd.DataFrame(
            [{"left": 0, "right": 0, "distance": 0.0, "count": 1}]
        )

    distances = pdist(feature_vectors, metric="cosine")
    if not np.all(np.isfinite(distances)) or np.allclose(distances, 0.0):
        distances = pdist(feature_vectors, metric="euclidean")
    if not np.all(np.isfinite(distances)):
        distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)
    linkage_matrix = linkage(distances, method="average", optimal_ordering=True)
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    linkage_table = pd.DataFrame(
        linkage_matrix,
        columns=["left", "right", "distance", "count"],
    )
    return labels.astype(int), linkage_table


def compute_umap_projection(feature_vectors, args):
    feature_vectors = sanitize_feature_vectors(feature_vectors)
    try:
        import umap
    except ImportError:
        return None, "UMAP skipped: install umap-learn to enable this output."

    if feature_vectors.shape[0] < 2:
        return None, "UMAP skipped: need at least 2 features."

    reducer = umap.UMAP(
        n_neighbors=min(args.umap_neighbors, feature_vectors.shape[0] - 1),
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.random_state,
    )
    embedding = reducer.fit_transform(feature_vectors)
    projection = pd.DataFrame(embedding, columns=["umap_1", "umap_2"])
    return projection, "UMAP completed."


def compute_kmeans_outputs(feature_vectors, metadata, requested_clusters, random_state):
    feature_vectors = sanitize_feature_vectors(feature_vectors)
    n_features = feature_vectors.shape[0]
    if n_features == 0:
        empty = pd.DataFrame()
        return np.asarray([], dtype=int), empty, empty
    if n_features == 1:
        labels = np.array([1], dtype=int)
        inertia_table = pd.DataFrame([{"k": 1, "inertia": 0.0}])
        representatives = metadata[["feature_name"]].copy()
        representatives.insert(0, "cluster_id", 1)
        representatives.insert(1, "rank_within_cluster", 1)
        representatives["distance_to_centroid"] = 0.0
        return labels, inertia_table, representatives

    max_k = choose_cluster_count(n_features, requested_clusters)
    inertia_rows = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        model.fit(feature_vectors)
        inertia_rows.append({"k": k, "inertia": float(model.inertia_)})

    inertia_table = pd.DataFrame(inertia_rows)
    selected_k = choose_elbow_cluster_count(inertia_table)
    selected_k = max(1, min(selected_k, n_features))

    final_model = KMeans(n_clusters=selected_k, n_init=20, random_state=random_state)
    final_model.fit(feature_vectors)
    labels = final_model.labels_.astype(int) + 1

    distances = final_model.transform(feature_vectors)
    rows = []
    for cluster_index in range(selected_k):
        cluster_id = cluster_index + 1
        member_indices = np.where(labels == cluster_id)[0]
        if member_indices.size == 0:
            continue
        member_distances = distances[member_indices, cluster_index]
        ranked_positions = np.argsort(member_distances)[:10]
        for rank, local_position in enumerate(ranked_positions, start=1):
            feature_index = int(member_indices[local_position])
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "rank_within_cluster": rank,
                    "feature_name": metadata.iloc[feature_index]["feature_name"],
                    "distance_to_centroid": float(member_distances[local_position]),
                }
            )
    representatives = pd.DataFrame(rows)
    return labels, inertia_table, representatives


def compute_nearest_neighbors(feature_vectors, metadata, nn_k):
    feature_vectors = sanitize_feature_vectors(feature_vectors)
    if feature_vectors.shape[0] <= 1:
        return pd.DataFrame()
    n_neighbors = min(nn_k + 1, feature_vectors.shape[0])
    model = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
    model.fit(feature_vectors)
    distances, indices = model.kneighbors(feature_vectors)
    rows = []
    for row_index, feature_name in enumerate(metadata["feature_name"]):
        for rank, (distance, neighbor_index) in enumerate(zip(distances[row_index][1:], indices[row_index][1:]), start=1):
            rows.append(
                {
                    "feature_name": feature_name,
                    "neighbor_rank": rank,
                    "neighbor_feature_name": metadata.iloc[neighbor_index]["feature_name"],
                    "distance": float(distance),
                }
            )
    return pd.DataFrame(rows)


def cluster_flip_summary(metadata):
    if metadata.empty:
        return pd.DataFrame()
    summary = (
        metadata.groupby("cluster_id", dropna=False)
        .agg(
            cluster_size=("feature_name", "size"),
            flip_event_total=("decision_flip_count", "sum"),
            flipped_feature_count=("decision_flip_feature", "sum"),
            mean_contrast=("contrast", "mean"),
        )
        .reset_index()
    )
    summary["flipped_feature_rate"] = summary["flipped_feature_count"] / summary["cluster_size"]
    return summary


def kmeans_cluster_summary(metadata):
    if metadata.empty or "kmeans_cluster_id" not in metadata.columns:
        return pd.DataFrame()
    summary = (
        metadata.groupby("kmeans_cluster_id", dropna=False)
        .agg(
            cluster_size=("feature_name", "size"),
            flip_event_total=("decision_flip_count", "sum"),
            flipped_feature_count=("decision_flip_feature", "sum"),
            mean_contrast=("contrast", "mean"),
        )
        .reset_index()
        .rename(columns={"kmeans_cluster_id": "cluster_id"})
    )
    summary["flipped_feature_rate"] = summary["flipped_feature_count"] / summary["cluster_size"]
    return summary


def write_text(path, text):
    Path(path).write_text(text + "\n", encoding="utf-8")


def save_bundle(
    bundle_dir,
    feature_vectors,
    metadata,
    document_metadata,
    linkage_table,
    neighbors,
    umap_projection,
    umap_status,
    kmeans_inertia_table,
    kmeans_representatives,
):
    bundle_dir.mkdir(parents=True, exist_ok=True)
    document_columns = [f"doc_{index:04d}" for index in range(feature_vectors.shape[1])]
    matrix_table = pd.DataFrame(feature_vectors, columns=document_columns)
    matrix_table.insert(0, "feature_name", metadata["feature_name"].to_numpy())

    metadata.to_csv(bundle_dir / "feature_metadata.csv", index=False)
    matrix_table.to_csv(bundle_dir / "deleted_feature_matrix.csv", index=False)
    document_metadata.to_csv(bundle_dir / "document_metadata.csv", index=False)
    linkage_table.to_csv(bundle_dir / "hierarchical_linkage.csv", index=False)
    neighbors.to_csv(bundle_dir / "nearest_neighbors.csv", index=False)
    cluster_flip_summary(metadata).to_csv(bundle_dir / "cluster_flip_summary.csv", index=False)
    kmeans_cluster_summary(metadata).to_csv(bundle_dir / "kmeans_cluster_summary.csv", index=False)
    kmeans_inertia_table.to_csv(bundle_dir / "kmeans_elbow.csv", index=False)
    kmeans_representatives.to_csv(bundle_dir / "kmeans_top_features.csv", index=False)
    if umap_projection is not None:
        umap_output = pd.concat(
            [metadata[["feature_name", "cluster_id", "kmeans_cluster_id"]].reset_index(drop=True), umap_projection],
            axis=1,
        )
        umap_output.to_csv(bundle_dir / "umap_projection.csv", index=False)
    write_text(bundle_dir / "umap_status.txt", umap_status)


def analyze_feature_bundle(feature_vectors, metadata, document_metadata, args, output_dir):
    if metadata.empty:
        return
    cluster_count = choose_cluster_count(len(metadata), args.n_clusters)
    cluster_labels, linkage_table = compute_hierarchical_labels(feature_vectors, cluster_count)
    metadata = metadata.copy()
    metadata["cluster_id"] = cluster_labels
    umap_projection, umap_status = compute_umap_projection(feature_vectors, args)
    kmeans_cluster_labels, kmeans_inertia_table, kmeans_representatives = compute_kmeans_outputs(
        feature_vectors,
        metadata,
        requested_clusters=args.n_clusters,
        random_state=args.random_state,
    )
    metadata["kmeans_cluster_id"] = kmeans_cluster_labels
    neighbors = compute_nearest_neighbors(feature_vectors, metadata, args.nn_k)
    save_bundle(
        output_dir,
        feature_vectors,
        metadata,
        document_metadata,
        linkage_table,
        neighbors,
        umap_projection,
        umap_status,
        kmeans_inertia_table,
        kmeans_representatives,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verifier = AuthorshipVerification(build_verifier_config(args))
    train_corpus = binarize_corpus(load_corpus(args.train_dir), positive_author=args.positive_author)

    hyperparams_path = resolve_hyperparams_path(args)
    hyperparams = None
    if args.load_hyperparams:
        if not hyperparams_path.exists():
            raise FileNotFoundError(f"{hyperparams_path} does not exist")
        hyperparams = load_hyperparams(hyperparams_path)

    verifier_artifacts = verifier.prepare_verifier(
        train_documents=train_corpus,
        test_documents=train_corpus,
        hyperparams=hyperparams,
        save_hyper_path=None if hyperparams is not None else hyperparams_path,
    )

    (
        positive_author_train_matrix,
        y_quijote,
        topic_metadata,
        deleted_feature_table,
        deleted_features,
        deleted_feature_names,
    ) = compute_ablation_if_needed(args, verifier, verifier_artifacts, train_corpus)

    deleted_matrix = dense_feature_matrix(positive_author_train_matrix[:, deleted_features])
    feature_vectors = deleted_matrix.T

    feature_metadata = deleted_feature_table.copy()
    feature_metadata["feature_name"] = deleted_feature_names
    feature_metadata["feature_family"] = feature_metadata["feature_name"].map(feature_family)

    flip_counts = decision_flip_counts(
        load_decision_change_table(args.decision_changes_path),
        tracked_author=args.positive_author,
    )
    feature_metadata = attach_feature_statistics(feature_metadata, deleted_matrix, y_quijote, flip_counts)

    write_text(
        output_dir / "analysis_notes.txt",
        (
            "This analysis uses deleted feature columns from "
            "positive_author_train_matrix.T, where rows are deleted features and columns are "
            "positive-author topic documents."
        ),
    )

    analyze_feature_bundle(
        feature_vectors=feature_vectors,
        metadata=feature_metadata,
        document_metadata=topic_metadata,
        args=args,
        output_dir=output_dir / "overall",
    )

    if args.split_by_family:
        family_root = output_dir / "families"
        for family, family_metadata in feature_metadata.groupby("feature_family", sort=True):
            family_indices = family_metadata.index.to_numpy()
            analyze_feature_bundle(
                feature_vectors=feature_vectors[family_indices],
                metadata=family_metadata.reset_index(drop=True),
                document_metadata=topic_metadata,
                args=args,
                output_dir=family_root / family,
            )

    summary = {
        "n_deleted_features": int(len(feature_metadata)),
        "n_topic_documents": int(len(topic_metadata)),
        "families": sorted(feature_metadata["feature_family"].unique().tolist()),
        "used_decision_changes": bool(args.decision_changes_path),
        "split_by_family": bool(args.split_by_family),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
