"""
Test for the branches module.
"""

import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from fast_hdbscan import HDBSCAN, BranchDetector, detect_branches_in_clusters


def make_branches(points_per_branch=30):
    # Control points for line segments that merge three clusters
    p0 = (0.13, -0.26)
    p1 = (0.24, -0.12)
    p2 = (0.32, 0.1)
    p3 = (0.13, 0.1)

    # Noisy points along lines between three clusters
    return np.concatenate(
        [
            np.column_stack(
                (
                    np.linspace(p_start[0], p_end[0], points_per_branch),
                    np.linspace(p_start[1], p_end[1], points_per_branch),
                )
            )
            + np.random.normal(size=(points_per_branch, 2), scale=0.01)
            for p_start, p_end in [(p0, p1), (p1, p2), (p1, p3)]
        ]
    )


np.random.seed(0)
X = np.concatenate(
    (
        make_branches(),
        make_branches()[:60] + np.array([0.3, 0]),
    )
)
c = HDBSCAN(min_samples=5, min_cluster_size=10).fit(X)


def check_detected_groups(c, n_clusters=3, n_branches=6, overridden=False):
    """Checks branch_detector output for main invariants."""
    noise_mask = c.labels_ == -1
    assert np.all(np.unique(c.labels_[~noise_mask]) == np.arange(n_branches))
    assert np.all(np.unique(c.cluster_labels_[~noise_mask]) == np.arange(n_clusters))
    assert (c.branch_labels_[noise_mask] == 0).all()
    assert (c.branch_probabilities_[noise_mask] == 1.0).all()
    assert (c.probabilities_[noise_mask] == 0.0).all()
    assert (c.cluster_probabilities_[noise_mask] == 0.0).all()
    if not overridden:
        assert len(c.cluster_points_) == n_clusters
        for condensed_tree, linkage_tree in zip(c._condensed_trees, c._linkage_trees):
            assert linkage_tree is not None
            assert condensed_tree is not None


# --- Detecting Branches


def test_attributes():
    def check_attributes():
        b = BranchDetector().fit(c)
        check_detected_groups(b, n_clusters=2, n_branches=5)
        assert len(b.linkage_trees_) == 2
        assert len(b.condensed_trees_) == 2
        assert isinstance(b.condensed_trees_[0], CondensedTree)
        assert isinstance(b.linkage_trees_[0], SingleLinkageTree)
        assert isinstance(b.approximation_graph_, ApproximationGraph)

    try:
        from hdbscan.plots import ApproximationGraph, CondensedTree, SingleLinkageTree

        check_attributes()
    except ImportError:
        pass


def test_selection_method():
    b = BranchDetector(branch_selection_method="eom").fit(c)
    check_detected_groups(b, n_clusters=2, n_branches=5)

    b = BranchDetector(branch_selection_method="leaf").fit(c)
    check_detected_groups(b, n_clusters=2, n_branches=5)


def test_min_branch_size():
    b = BranchDetector(min_branch_size=7).fit(c)
    labels, counts = np.unique(b.labels_, return_counts=True)
    assert (counts[labels >= 0] >= 7).all()
    check_detected_groups(b, n_clusters=2, n_branches=5)


def test_label_sides_as_branches():
    b = BranchDetector(label_sides_as_branches=True).fit(c)
    check_detected_groups(b, n_clusters=2, n_branches=6)


def test_max_branch_size():
    b = BranchDetector(label_sides_as_branches=True, max_branch_size=25).fit(c)
    check_detected_groups(b, n_clusters=2, n_branches=4)


def test_override_cluster_labels():
    X_missing = X.copy()
    X_missing[60:80] = np.nan
    c = HDBSCAN(min_cluster_size=5).fit(X_missing)
    split_y = c.labels_.copy()
    split_y[split_y == 1] = 0
    split_y[split_y == 2] = 1
    b = BranchDetector(label_sides_as_branches=True).fit(c, split_y)
    check_detected_groups(b, n_clusters=2, n_branches=5, overridden=True)
    assert b._condensed_trees[0] is None
    assert b._linkage_trees[0] is None


def test_allow_single_branch_with_filters():
    # Without persistence, find 6 branches
    b = BranchDetector(
        min_branch_size=5,
        branch_selection_method="leaf",
    ).fit(c)
    unique_labels = np.unique(b.labels_)
    assert len(unique_labels) == 5

    # Adding persistence removes the branches
    b = BranchDetector(
        min_branch_size=5,
        branch_selection_method="leaf",
        branch_selection_persistence=0.15,
    ).fit(c)
    unique_labels = np.unique(b.labels_)
    assert len(unique_labels) == 2

    # Adding epsilon removes some branches
    b = BranchDetector(
        min_branch_size=5,
        branch_selection_method="leaf",
        branch_selection_epsilon=1 / 0.002,
    ).fit(c)
    unique_labels = np.unique(b.labels_)
    assert len(unique_labels) == 2


def test_badargs():
    c_nofit = HDBSCAN(min_cluster_size=5)

    with pytest.raises(TypeError):
        detect_branches_in_clusters("fail")
    with pytest.raises(TypeError):
        detect_branches_in_clusters(None)
    with pytest.raises(NotFittedError):
        detect_branches_in_clusters(c_nofit)
    with pytest.raises(ValueError):
        detect_branches_in_clusters(c, min_branch_size=-1)
    with pytest.raises(ValueError):
        detect_branches_in_clusters(c, min_branch_size=0)
    with pytest.raises(ValueError):
        detect_branches_in_clusters(c, min_branch_size=1)
    with pytest.raises(ValueError):
        detect_branches_in_clusters(c, min_branch_size=2.0)
    with pytest.raises(ValueError):
        detect_branches_in_clusters(c, min_branch_size="fail")
    with pytest.raises(ValueError):
        detect_branches_in_clusters(c, branch_selection_persistence=-0.1)
    with pytest.raises(ValueError):
        detect_branches_in_clusters(c, branch_selection_epsilon=-0.1)
    with pytest.raises(ValueError):
        detect_branches_in_clusters(
            c,
            branch_selection_method="something_else",
        )
