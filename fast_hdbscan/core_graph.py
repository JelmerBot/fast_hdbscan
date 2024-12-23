import numba
import numpy as np
from collections import namedtuple

from .disjoint_set import ds_rank_create
from .hdbscan import fast_hdbscan_mst_edges
from .cluster_trees import empty_condensed_tree
from .boruvka import merge_components, update_point_components

CoreGraph = namedtuple("CoreGraph", ["distances", "indices", "child_counts"])


def core_graph_clusters(
    lens,
    neighbors,
    core_distances,
    min_spanning_tree,
    **kwargs,
):
    core_graph = create_core_graph(neighbors, core_distances, min_spanning_tree, lens)
    num_components, component_labels, lensed_mst = minimum_spanning_tree(core_graph)
    if num_components > 1:
        for i, label in enumerate(np.unique(component_labels)):
            component_labels[component_labels == label] = i
        return (
            component_labels,
            np.ones(len(component_labels), dtype=np.float32),
            np.empty((0, 4)),
            empty_condensed_tree(),
            lensed_mst,
            core_graph,
        )

    return (
        *fast_hdbscan_mst_edges(lensed_mst, **kwargs),
        core_graph,
    )


@numba.njit(parallel=True)
def create_core_graph(neighbors, core_distances, min_spanning_tree, lens_values):
    """Computes union over knn and mst edges in knn-style adjacency format."""
    # count non-knn mst edges
    num_points = neighbors.shape[0]
    counts = np.zeros(num_points, dtype=np.int32)
    for parent, child, distance in min_spanning_tree:
        parent = np.intp(parent)
        child = np.intp(child)
        if distance > max(core_distances[parent], core_distances[child]):
            counts[parent] += 1

    # allocate space for all edges
    max_children = np.max(counts) + 1
    num_neighbors = neighbors.shape[1]
    graph_indices = np.empty((num_points, num_neighbors + max_children), dtype=np.int32)
    graph_distances = np.empty(
        (num_points, num_neighbors + max_children), dtype=np.float32
    )

    # fill valid knn edges
    counts[:] = 0
    for point in numba.prange(num_points):
        parent_lens = lens_values[point]
        for child in neighbors[point]:
            if child < 0:
                continue
            graph_indices[point, counts] = child
            graph_distances[point, counts] = max(parent_lens, lens_values[child])
            counts[point] += 1

    # fill non-knn mst edges
    for parent, child, distance in min_spanning_tree:
        parent = np.intp(parent)
        child = np.intp(child)
        if distance > max(core_distances[parent], core_distances[child]):
            graph_indices[parent, counts[parent]] = child
            graph_distances[parent, counts[parent]] = max(
                lens_values[parent], lens_values[child]
            )
            counts[parent] += 1

    # sort by weights and fill remaining space with -1 and inf
    max_children = np.max(counts)
    for point in numba.prange(num_points):
        order = np.argsort(graph_distances[point, : counts[point]])
        graph_indices[point, : counts[point]] = graph_indices[point, order]
        graph_distances[point, : counts[point]] = graph_distances[point, order]
        graph_indices[point, counts[point] : max_children] = -1
        graph_distances[point, counts[point] : max_children] = np.inf

    return CoreGraph(
        graph_distances[:, :max_children], graph_indices[:, :max_children], counts
    )


@numba.njit()
def minimum_spanning_tree(core_graph, overwrite=False):
    """
    Implements Boruvka on knn-style adjacency format graph. The graph may
    contain multiple connected components. (-1, inf) indicates invalid edges.
    """
    graph_indices = core_graph.indices
    graph_distances = core_graph.distances
    if not overwrite:
        graph_indices = graph_indices.copy()
        graph_distances = graph_distances.copy()

    disjoint_set = ds_rank_create(graph_distances.shape[0])
    point_components = np.arange(graph_distances.shape[0])
    n_components = len(point_components)

    edges_list = [np.empty((0, 3), dtype=np.float64) for _ in range(0)]
    while n_components > 1:
        new_edges = merge_components(
            disjoint_set,
            select_components(
                graph_distances[:, 0], graph_indices[:, 0], point_components
            ),
        )
        if new_edges.shape[0] == 0:
            break

        edges_list.append(new_edges)
        update_point_components(disjoint_set, point_components)
        update_graph_components(graph_distances, graph_indices, point_components)
        n_components = len(np.unique(point_components))

    counter = 0
    num_edges = 0
    for edges in edges_list:
        num_edges += edges.shape[0]
    result = np.empty((num_edges, 3), dtype=np.float64)
    for edges in edges_list:
        result[counter : counter + edges.shape[0]] = edges
        counter += edges.shape[0]
    return n_components, point_components, result


@numba.njit(locals={"parent": numba.types.int32})
def select_components(candidate_distances, candidate_neighbors, point_components):
    """Skips invalid edges indicated by -1 neighbors and inf distances"""
    component_edges = {
        np.int64(0): (np.int32(0), np.int32(1), np.float32(0.0)) for _ in range(0)
    }

    # Find the best edges from each component
    for parent, (distance, neighbor, from_component) in enumerate(
        zip(candidate_distances, candidate_neighbors, point_components)
    ):
        if neighbor < 0:
            continue
        if from_component in component_edges:
            if distance < component_edges[from_component][2]:
                component_edges[from_component] = (parent, neighbor, distance)
        else:
            component_edges[from_component] = (parent, neighbor, distance)

    return component_edges


@numba.njit(parallel=True)
def update_graph_components(distances, indices, point_components):
    """Sets neighbors to (-1, inf) if they connect points in the same component"""
    for point in numba.prange(point_components.shape[0]):
        count = 0
        for col, dist in zip(indices[point], distances[point]):
            if col < 0:
                break
            if point_components[col] != point_components[point]:
                indices[point, count] = col
                distances[point, count] = dist
                count += 1
        indices[point, count:] = -1
        distances[point, count:] = np.inf
