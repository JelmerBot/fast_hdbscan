import numba
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClusterMixin

from .hdbscan import to_numpy_rec_array
from .core_graph import core_graph_clusters, core_graph_to_edge_list

try:
    from hdbscan.plots import (
        CondensedTree,
        SingleLinkageTree,
        ApproximationGraph,
        MinimumSpanningTree,
    )
except ImportError:
    pass


@numba.njit(fastmath=True)
def compute_centrality(data, probabilities):
    centroid = np.zeros((1, data.shape[1]), dtype=data.dtype)
    for coord, prob in zip(data, probabilities):
        centroid += coord * prob
    centroid /= probabilities.sum()
    return 1 / np.sqrt(((data - centroid) ** 2).sum(axis=1))


def compute_branches_in_cluster(
    cluster,
    data,
    labels,
    probabilities,
    neighbors,
    core_distances,
    min_spanning_tree,
    parent_labels,
    child_labels,
    **kwargs,
):
    # Compute centralities
    points = np.nonzero(labels == cluster)[0]
    centralities = compute_centrality(data[points, :], probabilities[points])

    # Convert to within cluster indices (-1 indicates invalid neighbor)
    in_cluster_ids = np.full(data.shape[0], -1, dtype=np.int32)
    in_cluster_ids[points] = np.arange(len(points))
    neighbors = in_cluster_ids[neighbors[points, :]]
    core_distances = core_distances[points]
    min_spanning_tree = min_spanning_tree[
        (parent_labels == cluster) & (child_labels == cluster)
    ]
    min_spanning_tree[:, :2] = in_cluster_ids[min_spanning_tree[:, :2].astype(np.int64)]

    # Compute branches from core graph
    return (
        *core_graph_clusters(
            centralities,
            neighbors,
            core_distances,
            min_spanning_tree,
            **{
                key.replace("branch", "cluster"): value for key, value in kwargs.items()
            },
        ),
        centralities,
        points,
    )


def parallel_branches_per_cluster(
    data,
    labels,
    probabilities,
    neighbors,
    core_distances,
    min_spanning_tree,
    num_clusters,
    **kwargs,
):
    # Loop could be parallel over clusters, but njit-compiling all called
    # functions slows down imports with a factor > 2 for small gains. Instead,
    # parts of each loop are parallel over points in the clusters.
    parent_labels = labels[min_spanning_tree[:, 0].astype(np.int64)]
    child_labels = labels[min_spanning_tree[:, 1].astype(np.int64)]
    return [
        compute_branches_in_cluster(
            cluster,
            data,
            labels,
            probabilities,
            neighbors,
            core_distances,
            min_spanning_tree,
            parent_labels,
            child_labels,
            **kwargs,
        )
        for cluster in numba.prange(num_clusters)
    ]


def update_labels(
    cluster_probabilities,
    branch_label_list,
    branch_prob_list,
    centrality_list,
    points_list,
    data_size,
    label_sides_as_branches=False,
):
    labels = np.full(data_size, -1, dtype=np.int64)
    probabilities = cluster_probabilities.copy()
    branch_labels = np.zeros(data_size, dtype=np.int64)
    branch_probs = np.zeros(data_size, dtype=np.float32)
    centralities = np.zeros(data_size, dtype=np.float32)

    running_id = 0
    branch_threshold = 1 if label_sides_as_branches else 2
    for points, _labels, _probs, _centralities in zip(
        points_list,
        branch_label_list,
        branch_prob_list,
        centrality_list,
    ):
        num_branches = np.max(_labels) + 1
        if num_branches <= branch_threshold:
            labels[points] = running_id
            running_id += 1
        else:
            noise_mask = np.nonzero(_labels == -1)[0]
            _labels[noise_mask] = num_branches
            labels[points] = _labels + running_id
            branch_labels[points] = _labels
            branch_probs[points] = _probs
            probabilities[points] += _probs
            probabilities[points] /= 2
            running_id += num_branches + int(len(noise_mask) > 0)
        centralities[points] = _centralities

    return labels, probabilities, branch_labels, branch_probs, centralities


def remap_results(
    labels,
    probabilities,
    cluster_labels,
    cluster_probabilities,
    branch_labels,
    branch_probabilities,
    centralities,
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

    new_branch_labels = np.full(num_points, 0, dtype=branch_labels.dtype)
    new_branch_labels[finite_index] = branch_labels
    branch_labels = new_branch_labels

    new_branch_probabilities = np.full(
        num_points, 1.0, dtype=branch_probabilities.dtype
    )
    new_branch_probabilities[finite_index] = branch_probabilities
    branch_probabilities = new_branch_probabilities

    new_centralities = np.full(num_points, 0.0, dtype=centralities.dtype)
    new_centralities[finite_index] = centralities
    centralities = new_centralities

    for pts in points:
        pts[:] = finite_index[pts]

    return (
        labels,
        probabilities,
        cluster_labels,
        cluster_probabilities,
        branch_labels,
        branch_probabilities,
        centralities,
        points,
    )


def detect_branches_in_clusters(
    clusterer,
    cluster_labels=None,
    cluster_probabilities=None,
    min_branch_size=None,
    max_branch_size=np.inf,
    allow_single_branch=False,
    branch_selection_method="eom",
    branch_selection_epsilon=0.0,
    branch_selection_persistence=0.0,
    label_sides_as_branches=False,
):
    check_is_fitted(
        clusterer,
        "_min_spanning_tree",
        msg="You first need to fit the HDBSCAN model before detecting branches",
    )

    # Validate input parameters
    if cluster_labels is None:
        cluster_labels = clusterer.labels_
    elif cluster_probabilities is None:
        cluster_probabilities = np.ones(cluster_labels.shape[0], dtype=np.float32)
    if cluster_probabilities is None:
        cluster_probabilities = clusterer.probabilities_

    if min_branch_size is None:
        min_branch_size = clusterer.min_cluster_size

    if not (np.issubdtype(type(min_branch_size), np.integer) and min_branch_size >= 2):
        raise ValueError(
            f"min_branch_size must be an integer greater or equal "
            f"to 2,  {min_branch_size} given."
        )
    if max_branch_size <= 0:
        raise ValueError(f"max_branch_size must be greater 0, {max_branch_size} given.")
    if not (
        np.issubdtype(type(branch_selection_persistence), np.floating)
        and branch_selection_persistence >= 0.0
    ):
        raise ValueError(
            f"branch_selection_persistence must be a float greater or equal to "
            f"0.0, {branch_selection_persistence} given."
        )
    if not (
        np.issubdtype(type(branch_selection_epsilon), np.floating)
        and branch_selection_epsilon >= 0.0
    ):
        raise ValueError(
            f"branch_selection_epsilon must be a float greater or equal to "
            f"0.0, {branch_selection_epsilon} given."
        )
    if branch_selection_method not in ("eom", "leaf"):
        raise ValueError(
            f"Invalid branch_selection_method: {branch_selection_method}\n"
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

    # Compute per-cluster branches
    num_clusters = np.max(cluster_labels) + 1
    (
        branch_labels,
        branch_probabilities,
        linkage_trees,
        condensed_trees,
        spanning_trees,
        core_graphs,
        centralities,
        points,
    ) = zip(
        *parallel_branches_per_cluster(
            data,
            cluster_labels,
            cluster_probabilities,
            clusterer._neighbors,
            clusterer._core_distances,
            clusterer._min_spanning_tree,
            num_clusters,
            min_branch_size=min_branch_size,
            max_branch_size=max_branch_size,
            allow_single_branch=allow_single_branch,
            branch_selection_method=branch_selection_method,
            branch_selection_epsilon=branch_selection_epsilon,
            branch_selection_persistence=branch_selection_persistence,
        )
    )

    # Handle override labels failure cases
    condensed_trees = [
        to_numpy_rec_array(tree) if tree.parent.shape[0] > 0 else None
        for tree in condensed_trees
    ]
    linkage_trees = [tree if tree.shape[0] > 0 else None for tree in linkage_trees]

    # Aggregate the results
    (labels, probabilities, branch_labels, branch_probabilities, centralities) = (
        update_labels(
            cluster_probabilities,
            branch_labels,
            branch_probabilities,
            centralities,
            points,
            data.shape[0],
            label_sides_as_branches=label_sides_as_branches,
        )
    )

    # Reset for infinite data points
    if last_outlier > 0:
        (
            labels,
            probabilities,
            cluster_labels,
            cluster_probabilities,
            branch_labels,
            branch_probabilities,
            centralities,
            points,
        ) = remap_results(
            labels,
            probabilities,
            cluster_labels,
            cluster_probabilities,
            branch_labels,
            branch_probabilities,
            centralities,
            points,
            finite_index,
            clusterer._raw_data.shape[0],
        )

    return (
        labels,
        probabilities,
        cluster_labels,
        cluster_probabilities,
        branch_labels,
        branch_probabilities,
        core_graphs,
        condensed_trees,
        linkage_trees,
        spanning_trees,
        centralities,
        points,
    )


class BranchDetector(BaseEstimator, ClusterMixin):
    """
    Performs a flare-detection post-processing step to detect branches within
    clusters [1]_.

    For each cluster, a graph is constructed connecting the data points based on
    their mutual reachability distances. Each edge is given a centrality value
    based on how far it lies from the cluster's center. Then, the edges are
    clustered as if that centrality was a distance, progressively removing the
    'center' of each cluster and seeing how many branches remain.

    References
    ----------
    .. [1] Bot, D. M., Peeters, J., Liesenborgs J., & Aerts, J. (2023, November).
       FLASC: A Flare-Sensitive Clustering Algorithm: Extending HDBSCAN* for
       Detecting Branches in Clusters. arXiv:2311.15887.
    """

    def __init__(
        self,
        min_branch_size=None,
        max_branch_size=np.inf,
        allow_single_branch=False,
        branch_selection_method="eom",
        branch_selection_epsilon=0.0,
        branch_selection_persistence=0.0,
        label_sides_as_branches=False,
    ):
        self.min_branch_size = min_branch_size
        self.max_branch_size = max_branch_size
        self.allow_single_branch = allow_single_branch
        self.branch_selection_method = branch_selection_method
        self.branch_selection_epsilon = branch_selection_epsilon
        self.branch_selection_persistence = branch_selection_persistence
        self.label_sides_as_branches = label_sides_as_branches

    def fit(self, clusterer, labels=None, probabilities=None):
        """labels and probabilities override the clusterer's values."""
        (
            self.labels_,
            self.probabilities_,
            self.cluster_labels_,
            self.cluster_probabilities_,
            self.branch_labels_,
            self.branch_probabilities_,
            self._approximation_graphs,
            self._condensed_trees,
            self._linkage_trees,
            self._spanning_trees,
            self.centralities_,
            self.cluster_points_,
        ) = detect_branches_in_clusters(
            clusterer, labels, probabilities, **self.get_params()
        )
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
            1 / self.centralities_,
            self.branch_labels_,
            self.branch_probabilities_,
            self._raw_data,
        )

    @property
    def condensed_trees_(self):
        """See :class:`~hdbscan.branches.BranchDetector` for documentation."""
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
        check_is_fitted(
            self,
            "_spanning_trees",
            msg="You first need to fit the BranchDetector model before accessing the linkage trees",
        )
        return [
            MinimumSpanningTree(tree, self._raw_data) for tree in self._spanning_trees
        ]
