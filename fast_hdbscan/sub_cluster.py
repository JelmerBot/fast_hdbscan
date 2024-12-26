import numba
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClusterMixin

from .hdbscan import to_numpy_rec_array
from .core_graph import core_graph_clusters, core_graph_to_edge_list


def compute_sub_clusters_in_cluster(
    cluster,
    data,
    labels,
    probabilities,
    neighbors,
    core_distances,
    min_spanning_tree,
    parent_labels,
    child_labels,
    lens_callback,
    **kwargs,
):
    # Convert to within cluster indices (-1 indicates invalid neighbor)
    points = np.nonzero(labels == cluster)[0]

    in_cluster_ids = np.full(data.shape[0], -1, dtype=np.int32)
    in_cluster_ids[points] = np.arange(len(points))
    neighbors = in_cluster_ids[neighbors[points, :]]
    core_distances = core_distances[points]
    min_spanning_tree = min_spanning_tree[
        (parent_labels == cluster) & (child_labels == cluster)
    ]
    min_spanning_tree[:, :2] = in_cluster_ids[min_spanning_tree[:, :2].astype(np.int64)]

    # Compute lens_values
    lens_values = lens_callback(
        data[points, :],
        probabilities[points],
        neighbors,
        core_distances,
        min_spanning_tree,
    )

    # Compute branches from core graph
    return (
        *core_graph_clusters(
            lens_values,
            neighbors,
            core_distances,
            min_spanning_tree,
            **kwargs,
        ),
        lens_values,
        points,
    )


def compute_sub_clusters_per_cluster(
    data,
    labels,
    probabilities,
    neighbors,
    core_distances,
    min_spanning_tree,
    lens_callback,
    num_clusters,
    **kwargs,
):
    # Loop could be parallel over clusters, but njit-compiling all called
    # functions slows down imports with a factor > 2 for small gains. Instead,
    # parts of each loop are parallel over points in the clusters.
    parent_labels = labels[min_spanning_tree[:, 0].astype(np.int64)]
    child_labels = labels[min_spanning_tree[:, 1].astype(np.int64)]
    return [
        compute_sub_clusters_in_cluster(
            cluster,
            data,
            labels,
            probabilities,
            neighbors,
            core_distances,
            min_spanning_tree,
            parent_labels,
            child_labels,
            lens_callback,
            **kwargs,
        )
        for cluster in range(num_clusters)
    ]


def update_labels(
    cluster_probabilities,
    sub_labels_list,
    sub_probabilities_list,
    lens_values_list,
    points_list,
    data_size,
):
    labels = np.full(data_size, -1, dtype=np.int64)
    probabilities = cluster_probabilities.copy()
    sub_labels = np.zeros(data_size, dtype=np.int64)
    sub_probabilities = np.zeros(data_size, dtype=np.float32)
    lens_values = np.zeros(data_size, dtype=np.float32)

    running_id = 0
    for points, _labels, _probs, _lens in zip(
        points_list,
        sub_labels_list,
        sub_probabilities_list,
        lens_values_list,
    ):
        unique_labels = np.unique(_labels)
        labels[points] = _labels + int(unique_labels[0] == -1) + running_id
        sub_labels[points] = _labels
        sub_probabilities[points] = _probs
        probabilities[points] += _probs
        probabilities[points] /= 2
        lens_values[points] = _lens
        running_id += len(unique_labels)

    return labels, probabilities, sub_labels, sub_probabilities, lens_values


def remap_results(
    labels,
    probabilities,
    cluster_labels,
    cluster_probabilities,
    sub_labels,
    sub_probabilities,
    lens_values,
    points,
    finite_index,
    num_points,
):
    new_labels = np.full(num_points, -1, dtype=labels.dtype)
    new_labels[finite_index] = labels
    labels = new_labels

    new_probabilities = np.full(num_points, 0.0, dtype=probabilities.dtype)
    new_probabilities[finite_index] = probabilities
    probabilities = new_probabilities

    new_cluster_labels = np.full(num_points, -1, dtype=cluster_labels.dtype)
    new_cluster_labels[finite_index] = cluster_labels
    cluster_labels = new_cluster_labels

    new_cluster_probabilities = np.full(
        num_points, 0.0, dtype=cluster_probabilities.dtype
    )
    new_cluster_probabilities[finite_index] = cluster_probabilities
    cluster_probabilities = new_cluster_probabilities

    new_sub_labels = np.full(num_points, 0, dtype=sub_labels.dtype)
    new_sub_labels[finite_index] = sub_labels
    sub_labels = new_sub_labels

    new_sub_probabilities = np.full(num_points, 1.0, dtype=sub_probabilities.dtype)
    new_sub_probabilities[finite_index] = sub_probabilities
    sub_probabilities = new_sub_probabilities

    new_lens_values = np.full(num_points, 0.0, dtype=lens_values.dtype)
    new_lens_values[finite_index] = lens_values
    lens_values = new_lens_values

    for pts in points:
        pts[:] = finite_index[pts]

    return (
        labels,
        probabilities,
        cluster_labels,
        cluster_probabilities,
        sub_labels,
        sub_probabilities,
        lens_values,
        points,
    )


def find_sub_clusters(
    clusterer,
    cluster_labels=None,
    cluster_probabilities=None,
    lens_callback=None,
    min_cluster_size=None,
    max_cluster_size=np.inf,
    allow_single_cluster=False,
    cluster_selection_method="eom",
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
):
    check_is_fitted(
        clusterer,
        "_min_spanning_tree",
        msg="You first need to fit the HDBSCAN model before detecting sub clusters.",
    )

    # Validate input parameters
    if cluster_labels is None:
        cluster_labels = clusterer.labels_
    elif cluster_probabilities is None:
        cluster_probabilities = np.ones(cluster_labels.shape[0], dtype=np.float32)
    if cluster_probabilities is None:
        cluster_probabilities = clusterer.probabilities_

    if min_cluster_size is None:
        min_cluster_size = clusterer.min_cluster_size

    if not (
        np.issubdtype(type(min_cluster_size), np.integer) and min_cluster_size >= 2
    ):
        raise ValueError(
            f"min_cluster_size must be an integer greater or equal "
            f"to 2,  {min_cluster_size} given."
        )
    if max_cluster_size <= 0:
        raise ValueError(
            f"max_cluster_size must be greater 0, {max_cluster_size} given."
        )
    if not (
        np.issubdtype(type(cluster_selection_persistence), np.floating)
        and cluster_selection_persistence >= 0.0
    ):
        raise ValueError(
            f"cluster_selection_persistence must be a float greater or equal to "
            f"0.0, {cluster_selection_persistence} given."
        )
    if not (
        np.issubdtype(type(cluster_selection_epsilon), np.floating)
        and cluster_selection_epsilon >= 0.0
    ):
        raise ValueError(
            f"cluster_selection_epsilon must be a float greater or equal to "
            f"0.0, {cluster_selection_epsilon} given."
        )
    if cluster_selection_method not in ("eom", "leaf"):
        raise ValueError(
            f"Invalid cluster_selection_method: {cluster_selection_method}\n"
            f'Should be one of: "eom", "leaf"\n'
        )

    # Recover finite data points
    data = clusterer._raw_data
    last_outlier = np.searchsorted(
        clusterer._condensed_tree["lambda_val"], 0.0, side="right"
    )
    if last_outlier > 0:
        finite_index = np.setdiff1d(
            np.arange(data.shape[0]), clusterer._condensed_tree["child"][:last_outlier]
        )
        data = data[finite_index]
        cluster_labels = cluster_labels[finite_index]
        cluster_probabilities = cluster_probabilities[finite_index]

    # Convert lens value array to callback
    if isinstance(lens_callback, np.ndarray):
        if len(lens_callback) != data.shape[0]:
            raise ValueError(
                "when providing values as lens_callback, they must have"
                f"the same length as the data, {len(lens_callback)} != {data.shape[0]}"
            )
        if last_outlier > 0:
            lens_values = lens_callback[finite_index]
        else:
            lens_values = lens_callback

        lens_callback = lambda a, b, c, d, e: lens_values

    # Compute per-cluster sub clusters
    num_clusters = np.max(cluster_labels) + 1
    (
        sub_labels,
        sub_probabilities,
        linkage_trees,
        condensed_trees,
        spanning_trees,
        core_graphs,
        lens_values,
        points,
    ) = zip(
        *compute_sub_clusters_per_cluster(
            data,
            cluster_labels,
            cluster_probabilities,
            clusterer._neighbors,
            clusterer._core_distances,
            clusterer._min_spanning_tree,
            lens_callback,
            num_clusters,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            allow_single_cluster=allow_single_cluster,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_persistence=cluster_selection_persistence,
        )
    )

    # Handle override labels failure cases
    condensed_trees = [
        to_numpy_rec_array(tree) if tree.parent.shape[0] > 0 else None
        for tree in condensed_trees
    ]
    linkage_trees = [tree if tree.shape[0] > 0 else None for tree in linkage_trees]

    # Aggregate the results
    (labels, probabilities, sub_labels, sub_probabilities, lens_values) = update_labels(
        cluster_probabilities,
        sub_labels,
        sub_probabilities,
        lens_values,
        points,
        data.shape[0],
    )

    # Reset for infinite data points
    if last_outlier > 0:
        (
            labels,
            probabilities,
            cluster_labels,
            cluster_probabilities,
            sub_labels,
            sub_probabilities,
            lens_values,
            points,
        ) = remap_results(
            labels,
            probabilities,
            cluster_labels,
            cluster_probabilities,
            branch_labels,
            branch_probabilities,
            lens_values,
            points,
            finite_index,
            clusterer._raw_data.shape[0],
        )

    return (
        labels,
        probabilities,
        cluster_labels,
        cluster_probabilities,
        sub_labels,
        sub_probabilities,
        core_graphs,
        condensed_trees,
        linkage_trees,
        spanning_trees,
        lens_values,
        points,
    )


class SubClusterDetector:
    """Performs a lens-value sub-cluster detection post-processing step
    on a HDBSCAN clusterer."""

    def __init__(
        self,
        lens_callback,
        min_cluster_size=None,
        max_cluster_size=np.inf,
        allow_single_cluster=False,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
        cluster_selection_persistence=0.0,
    ):
        self.lens_callback = lens_callback
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.allow_single_cluster = allow_single_cluster
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_persistence = cluster_selection_persistence

    def fit(self, clusterer, labels=None, probabilities=None):
        """labels and probabilities override the clusterer's values."""
        (
            self.labels_,
            self.probabilities_,
            self.cluster_labels_,
            self.cluster_probabilities_,
            self.sub_cluster_labels_,
            self.sub_cluster_probabilities_,
            self._approximation_graphs,
            self._condensed_trees,
            self._linkage_trees,
            self._spanning_trees,
            self.lens_values_,
            self.cluster_points_,
        ) = find_sub_clusters(clusterer, labels, probabilities, **self.get_params())
        # also store the core distances and raw data for the member functions
        self._raw_data = clusterer._raw_data
        self._core_distances = clusterer._core_distances
        return self

    def fit_predict(self, clusterer, labels=None, probabilities=None):
        self.fit(clusterer, labels, probabilities)
        return self.labels_

    @property
    def approximation_graph_(self):
        """See :class:`~hdbscan.branches.BranchDetector` for documentation."""
        from hdbscan.plots import ApproximationGraph

        check_is_fitted(
            self,
            "_approximation_graphs",
            msg="You first need to fit the BranchDetector model before accessing the approximation graphs",
        )

        edge_lists = []
        for graph, points in zip(self._approximation_graphs, self.cluster_points_):
            edges = core_graph_to_edge_list(graph)
            edges[:, 0] = points[edges[:, 0].astype(np.int64)]
            edges[:, 1] = points[edges[:, 1].astype(np.int64)]
            edge_lists.append(edges)

        return ApproximationGraph(
            edge_lists,
            self.labels_,
            self.probabilities_,
            self.cluster_labels_,
            self.cluster_probabilities_,
            1 / self.lens_values_,
            self.sub_cluster_labels_,
            self.sub_cluster_probabilities_,
            self._raw_data,
        )

    @property
    def condensed_trees_(self):
        """See :class:`~hdbscan.branches.BranchDetector` for documentation."""
        from hdbscan.plots import CondensedTree

        check_is_fitted(
            self,
            "_condensed_trees",
            msg="You first need to fit the BranchDetector model before accessing the condensed trees",
        )
        return [
            (
                CondensedTree(
                    tree,
                    np.where(
                        self.branch_labels_[points]
                        == self.branch_labels_[points].max(),
                        -1,
                        self.branch_labels_[points],
                    ),
                )
                if tree is not None
                else None
            )
            for tree, points in zip(
                self._condensed_trees,
                self.cluster_points_,
            )
        ]

    @property
    def linkage_trees_(self):
        """See :class:`~hdbscan.branches.BranchDetector` for documentation."""
        from hdbscan.plots import SingleLinkageTree

        check_is_fitted(
            self,
            "_linkage_trees",
            msg="You first need to fit the BranchDetector model before accessing the linkage trees",
        )
        return [
            SingleLinkageTree(tree) if tree is not None else None
            for tree in self._linkage_trees
        ]

    @property
    def spanning_trees_(self):
        """See :class:`~hdbscan.branches.BranchDetector` for documentation."""
        from hdbscan.plots import MinimumSpanningTree

        check_is_fitted(
            self,
            "_spanning_trees",
            msg="You first need to fit the BranchDetector model before accessing the linkage trees",
        )
        return [
            MinimumSpanningTree(tree, self._raw_data) for tree in self._spanning_trees
        ]
