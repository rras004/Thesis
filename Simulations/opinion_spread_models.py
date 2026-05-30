"""
This file contains the update rules of the opinion spread models used in the thesis.

The implemented models are:
- standard voter model,
- noisy voter model,
- adaptive voter model.

Each function performs a single update step of the corresponding model and returns the updated graph structure.
"""

import numpy as np
from helper_functions import update_active_edges

# THE STANDARD VOTER MDOEL
def standard_voter_step(graph, rng=None):
    """
    Perform one step of the standard voter model.
    """
    if rng is None:
        rng = np.random.default_rng()

    nodes = graph["nodes"]
    vertex_to_edges = graph["vertex_to_edges"]
    opinions = graph["opinions"]

    i = rng.choice(nodes)

    # only 2-edges count as neighbors
    neighbor_edges = [e for e in vertex_to_edges[i] if len(e) == 2]

    if len(neighbor_edges) == 0:
        return graph

    neighbors = set()
    for edge in neighbor_edges:
        for v in edge:
            if v != i:
                neighbors.add(v)

    if len(neighbors) == 0:
        return graph

    j = rng.choice(list(neighbors))

    if opinions[i] != opinions[j]:
        opinions[i] = opinions[j]
        update_active_edges(graph, [i])

    return graph


# THE NOISY VOTER MODEL
def noisy_voter_step(graph, p, q, rng=None):
    """
    Perform one step of the noisy voter model.
    p : Probability of group consensus: all minority vertices adopt the majority.
    q : If no group consensus occurs, each minority vertex independently adopts the majority with probability q.
    """
    if rng is None:
        rng = np.random.default_rng()

    active_edges = graph["active_edges"]
    opinions = graph["opinions"]

    # If no active edges remain, nothing can happen
    if len(active_edges) == 0:
        return graph

    # Choose an active edge uniformly
    edge = rng.choice(list(active_edges))
    edge_vertices = list(edge)
    edge_opinions = opinions[edge_vertices]

    # Compute majority
    total = edge_opinions.sum()
    if total > 0:
        majority = 1
    elif total < 0:
        majority = -1
    else:
        majority = rng.choice([-1, 1])

    # Minority vertices
    minority_vertices = [v for v in edge_vertices if opinions[v] != majority]
    changed_vertices = []

    # With probability p: all minority vertices flip
    if rng.random() < p:
        for v in minority_vertices:
            opinions[v] = majority
            changed_vertices.append(v)

    # Otherwise each minority vertex flips independently with probability q
    else:
        for v in minority_vertices:
            if rng.random() < q:
                opinions[v] = majority
                changed_vertices.append(v)

    # Update active edges after opinion changes
    if changed_vertices:
        update_active_edges(graph, changed_vertices)
    else:
        pass

    return graph


# ADAPTIVE VOTER MODEL
from helper_functions import replace_triangles, rewire_edge, simplex_rewire


def adaptive_voter_step(graph, p, q, p_triangle=0.0, rng=None):
    """
    Perform one step of the adaptive voter model
    p : Persuasion probability.
    q : If the selected 2-edge is in a triangle:
            with probability q -> triangle majority rule
            with probability 1-q -> simplex rewiring
    p_triangle : Triangle replacement probability.
    """
    if rng is None:
        rng = np.random.default_rng()

    opinions = graph["opinions"]

    # choose only active 2-edges
    active_2_edges = [edge for edge in graph["active_edges"] if len(edge) == 2]

    if len(active_2_edges) == 0:
        return graph

    edge = rng.choice(active_2_edges)
    u, v = tuple(edge)

    changed_vertices = []
    new_edge = None
    chosen_triangle = None

    incident_triangles = graph["edge_to_triangles"].get(edge, set())

    # Case 1: edge is not part of any triangle
    if len(incident_triangles) == 0:
        if rng.random() < p:
            if rng.random() < 0.5:
                opinions[u] = opinions[v]
                changed_vertices = [u]
            else:
                opinions[v] = opinions[u]
                changed_vertices = [v]
        else:
            graph, new_edge = rewire_edge(graph, edge, rng)

            if new_edge is not None:
                local_vertices = set(new_edge)
                for x in new_edge:
                    for e in graph["vertex_to_edges"][x]:
                        if len(e) == 2:
                            local_vertices.update(e)

                replace_triangles(graph, p_triangle, rng, candidate_vertices=local_vertices)


    # Case 2: edge is part of at least one triangle
    else:
        if rng.random() < q:
            chosen_triangle = rng.choice(list(incident_triangles))
            tri_vertices = list(chosen_triangle)
            tri_opinions = opinions[tri_vertices]

            total = tri_opinions.sum()
            majority = 1 if total > 0 else -1
            minority_vertices = [x for x in tri_vertices if opinions[x] != majority]

            if rng.random() < p:
                for x in minority_vertices:
                    opinions[x] = majority
                changed_vertices = minority_vertices

        else:
            graph, new_edge = simplex_rewire(graph, edge, rng)

            if new_edge is not None:
                local_vertices = set(new_edge)
                for x in new_edge:
                    for e in graph["vertex_to_edges"][x]:
                        if len(e) == 2:
                            local_vertices.update(e)

                replace_triangles(graph, p_triangle, rng, candidate_vertices=local_vertices)

    if changed_vertices:
        update_active_edges(graph, changed_vertices)

    return graph