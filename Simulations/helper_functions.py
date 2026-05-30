"""
This file contains helper functions used throughout the opinion spread simulations.
The functions can be grouped into four categories.

Opinion initialization:
    - "generate_opinions": generates the initial distribution of opinions on the vertices

Simulation preparation:
    - "prepare_graph_for_simulation": builds all auxiliary graph structures required by the simulations
    - "initialize_active_edges": computes the initial set of active edges
    - "build_vertex_to_edges": stores the edges incident to each vertex
    - "build_vertex_to_active_edges": stores the active edges incident to each vertex

Active edge management:
    - "is_active_edge": determines whether an edge is active or inactive
    - "update_active_edges": updates the active edge structures after opinion changes

To speed up the simulations, the graph dictionary is extended with several auxiliary structures:
    - "nodes": list of vertices
    - "edges_by_size": dictionary mapping edge size to hyperedges
    - "edges": set of all hyperedges
    - "opinions": opinion distribution on the vertices
    - "active_edges": set of active hyperedges
    - "vertex_to_edges": incident edges for each vertex
    - "vertex_to_active_edges": incident active edges for each vertex
    - "edge_to_triangles": triangles containing a given 2-edge

Adaptive voter model helper functions:
    - "build_edge_to_triangles": computes which triangles contain each 2-edge
    - "rewire_edge": rewires a selected 2-edge to a new endpoint with matching opinion
    - "simplex_rewire": performs rewiring and removes all affected triangles
    - "replace_triangles": adds new triangles with probability p_triangle whenever all three 2-edges are present
"""

import numpy as np
from itertools import combinations

def generate_opinions(n, prob_positive, rng=None):
    """
    Generate the intital distribution of opinions.
    """
    if rng is None:
        rng = np.random.default_rng()

    return rng.choice([-1, 1], size=n, p=[1 - prob_positive, prob_positive])

def is_active_edge(edge, opinions):
    """
    The input is the edge and the opinions, we decide of that single edge if it is active or inactive.
    """
    vertices = list(edge)
    edge_opinions = opinions[vertices]

    return np.any(edge_opinions != edge_opinions[0])

def initialize_active_edges(graph):
    """
    Compute the set of active edges of the graph and store it in graph["active_edges"].
    """
    opinions = graph["opinions"]
    active_edges = set()

    for edge in graph["edges"]:
        if is_active_edge(edge, opinions):
            active_edges.add(edge)

    graph["active_edges"] = active_edges
    return graph

def build_vertex_to_edges(graph):
    """
    For each vertex we store which edges contain it: graph["vertex_to_edges"][v]
    """
    vertex_to_edges = {v: set() for v in graph["nodes"]}

    for edge in graph["edges"]:
        for v in edge:
            vertex_to_edges[v].add(edge)

    graph["vertex_to_edges"] = vertex_to_edges
    return graph

def build_vertex_to_active_edges(graph):
    """
    For each vertex we store which which active edges contain it: graph["vertex_to_active_edges"][v]
    """
    vertex_to_active_edges = {v: set() for v in graph["nodes"]}

    for edge in graph["active_edges"]:
        for v in edge:
            vertex_to_active_edges[v].add(edge)

    graph["vertex_to_active_edges"] = vertex_to_active_edges
    return graph

def prepare_graph_for_simulation(graph):
    build_vertex_to_edges(graph)
    initialize_active_edges(graph)
    build_vertex_to_active_edges(graph)
    build_edge_to_triangles(graph)
    return graph

def update_active_edges(graph, changed_vertices):
    """
    Update active_edges and vertex_to_active_edges after opinion changes.
    """
    opinions = graph["opinions"]
    vertex_to_edges = graph["vertex_to_edges"]
    active_edges = graph["active_edges"]
    vertex_to_active_edges = graph["vertex_to_active_edges"]

    affected_edges = set()
    for v in changed_vertices:
        affected_edges.update(vertex_to_edges[v])

    for edge in affected_edges:
        was_active = edge in active_edges
        now_active = is_active_edge(edge, opinions)

        if now_active and not was_active:
            active_edges.add(edge)
            for v in edge:
                vertex_to_active_edges[v].add(edge)

        elif was_active and not now_active:
            active_edges.discard(edge)
            for v in edge:
                vertex_to_active_edges[v].discard(edge)

    return graph

# ADAPTIVE VOTER MODEL

def build_edge_to_triangles(graph):
    """
    For each edge which triangles contain it: graph["edge_to_triangles"][edge]
    """
    edge_to_triangles = {edge: set() for edge in graph["edges_by_size"].get(2, set())}

    for triangle in graph["edges_by_size"].get(3, set()):
        for pair in combinations(triangle, 2):
            edge = frozenset(pair)
            if edge in edge_to_triangles:
                edge_to_triangles[edge].add(triangle)

    graph["edge_to_triangles"] = edge_to_triangles
    return graph


def rewire_edge(graph, edge, rng=None):
    """
    Rewire a 2-edge.
    """
    if rng is None:
        rng = np.random.default_rng()

    opinions = graph["opinions"]
    u, v = tuple(edge)

    # choose anchor endpoint
    if rng.random() < 0.5:
        anchor = u
    else:
        anchor = v

    anchor_opinion = opinions[anchor]

    # neighbors of anchor before deletion
    existing_neighbors = set()
    for e in graph["vertex_to_edges"][anchor]:
        if len(e) == 2:
            for x in e:
                if x != anchor:
                    existing_neighbors.add(x)

    # remove old edge from all structures
    if edge in graph["edges_by_size"].get(2, set()):
        graph["edges_by_size"][2].remove(edge)
    graph["edges"].discard(edge)
    graph["active_edges"].discard(edge)

    for x in edge:
        graph["vertex_to_edges"][x].discard(edge)
        graph["vertex_to_active_edges"][x].discard(edge)

    graph["edge_to_triangles"].pop(edge, None)

    # choose new endpoint
    candidates = [
        w for w in graph["nodes"]
        if w != anchor
        and opinions[w] == anchor_opinion
        and w not in existing_neighbors
    ]

    if len(candidates) == 0:
        return graph, None

    new_vertex = rng.choice(candidates)
    new_edge = frozenset({anchor, new_vertex})

    # add new edge to all structures
    graph["edges_by_size"].setdefault(2, set()).add(new_edge)
    graph["edges"].add(new_edge)

    for x in new_edge:
        graph["vertex_to_edges"][x].add(new_edge)

    graph["edge_to_triangles"][new_edge] = set()

    if is_active_edge(new_edge, opinions):
        graph["active_edges"].add(new_edge)
        for x in new_edge:
            graph["vertex_to_active_edges"][x].add(new_edge)

    return graph, new_edge

def simplex_rewire(graph, edge, rng=None):
    """
    Rewire a selected 2-edge that is part of one or more triangles.
    """
    if rng is None:
        rng = np.random.default_rng()

    incident_triangles = list(graph["edge_to_triangles"].get(edge, set()))

    graph, new_edge = rewire_edge(graph, edge, rng)

    for triangle in incident_triangles:
        if triangle in graph["edges_by_size"].get(3, set()):
            graph["edges_by_size"][3].remove(triangle)
            graph["edges"].discard(triangle)
            graph["active_edges"].discard(triangle)

            for v in triangle:
                graph["vertex_to_edges"][v].discard(triangle)
                graph["vertex_to_active_edges"][v].discard(triangle)

            for pair in combinations(triangle, 2):
                pair_edge = frozenset(pair)
                if pair_edge in graph["edge_to_triangles"]:
                    graph["edge_to_triangles"][pair_edge].discard(triangle)

    return graph, new_edge

def replace_triangles(graph, p_triangle, rng=None, candidate_vertices=None):
    """
    Add missing triangles with probability p_triangle, but only when all three 2-edges already exist.
    """
    if rng is None:
        rng = np.random.default_rng()

    if p_triangle <= 0:
        return graph

    if candidate_vertices is None:
        vertices = list(graph["nodes"])
    else:
        vertices = list(candidate_vertices)

    pair_edges = graph["edges_by_size"].get(2, set())
    triangles = graph["edges_by_size"].setdefault(3, set())
    opinions = graph["opinions"]

    for triple in combinations(vertices, 3):
        triangle = frozenset(triple)

        if triangle in triangles:
            continue

        triangle_edges = [frozenset(pair) for pair in combinations(triple, 2)]

        if all(edge in pair_edges for edge in triangle_edges):
            if rng.random() < p_triangle:
                triangles.add(triangle)
                graph["edges"].add(triangle)

                for v in triangle:
                    graph["vertex_to_edges"][v].add(triangle)

                for edge in triangle_edges:
                    graph["edge_to_triangles"].setdefault(edge, set()).add(triangle)

                if is_active_edge(triangle, opinions):
                    graph["active_edges"].add(triangle)
                    for v in triangle:
                        graph["vertex_to_active_edges"][v].add(triangle)

    return graph

