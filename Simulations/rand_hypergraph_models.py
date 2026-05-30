"""
In this file the random hypergraph models can be found, namely The generalised ER hypergraph model, the recursive generalised ER hypergraph model, and lastly the generalized SGM hypergraph model. 
(These can also be used to generate standard graphs). 
Parameters:
    - n: number of nodes 
    - prob_positive : Probability that a vertex gets opinion +1. Probability of opinion -1 is then 1 - prob_positive. ==> Generating the opinion distribution
    - probabilities : Dictionary of the form {r: p_r}, for example {2: 0.05, 3: 0.01}. ==> General ER hypergraphs
    - probabilities : ==> Generating the SBM hypergraphs
        Dictionary of the form
            {
                2: (p2, q2),
                3: (p3, q3),
                ...
            }
        where:
            - p_r = probability of adding an r-edge if all labels are the same
            - q_r = probability of adding an r-edge otherwise
    - opinions : Array of length n with entries in {-1, 1}.
    - p1 : Probability of including each edge. ==> Recursice ER for 2-simplex
    - p2 : Probability of including each triangle, conditional on all three edges existing. ==> Recursice ER for 2-simplex
    - rng : Random number generator.

Output: 
    Random Graph models
    - "nodes": list of vertices
    - "edges_by_size": dict mapping r -> set of hyperedges
    - "edges": set of all hyperedges
    - "opinions": the distribution of opinions
    """

from itertools import combinations
import numpy as np

# GENERATING THE GENERAL ER RANDOM HYPERGRAPH MODEL
def generate_general_er_hypergraph(n, probabilities, opinions, rng=None):
    
    if rng is None:
        rng = np.random.default_rng()

    nodes = list(range(n))
    edges_by_size = {}

    for r, p in probabilities.items():
    
        edges_r = set()
        for subset in combinations(nodes, r):
            if rng.random() < p:
                edges_r.add(frozenset(subset))

        edges_by_size[r] = edges_r

    all_edges = set().union(*edges_by_size.values()) if edges_by_size else set()

    return {
        "nodes": nodes,
        "edges_by_size": edges_by_size,
        "edges": all_edges,
        "opinions": opinions
    }

# GENERATING THE HIERARCHICAL ERDŐS--RÉNYI R-UNIFORM RANDOM GRAPH MODEL (since we only need it for the adaptive voter model, we will only be programming for 2-simplexes)
def generate_recursive_er_simplicial_complex(n, p1, p2, opinions, rng=None):
   
    if rng is None:
        rng = np.random.default_rng()

    nodes = list(range(n))

    edges_2 = set()
    edges_3 = set()

    # Generating edges
    for pair in combinations(nodes, 2):
        if rng.random() < p1:
            edges_2.add(frozenset(pair))

    # Generating triangles
    for triple in combinations(nodes, 3):
        triangle_edges = [frozenset(pair) for pair in combinations(triple, 2)]

        if all(edge in edges_2 for edge in triangle_edges):
            if rng.random() < p2:
                edges_3.add(frozenset(triple))

    edges_by_size = {
        2: edges_2,
        3: edges_3
    }

    # union of all hyperedges
    all_edges = edges_2.union(edges_3)

    return {
        "nodes": nodes,
        "edges_by_size": edges_by_size,
        "edges": all_edges,
        "opinions": opinions
    }

# GENERATING THE STOCHASTIC BLOCK MODEL (SBM)
def generate_generalized_hsbm(n, probabilities, opinions, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    nodes = list(range(n))
    edges_by_size = {}

    for r, (p_r, q_r) in probabilities.items():
        edges_r = set()

        for subset in combinations(nodes, r):
            subset_opinions = opinions[list(subset)]

            if np.all(subset_opinions == subset_opinions[0]):
                prob = p_r
            else:
                prob = q_r

            if rng.random() < prob:
                edges_r.add(frozenset(subset))

        edges_by_size[r] = edges_r

    all_edges = set().union(*edges_by_size.values()) if edges_by_size else set()

    return {
        "nodes": nodes,
        "edges_by_size": edges_by_size,
        "edges": all_edges,
        "opinions": opinions
    }