"""
This file contains analysis functions used to study how graph-model parameters influence the final majority-opinion ratio. The functions are all used in results4.py.

The functions can be grouped into two categories.

Parameter studies:
    - "estimate_average_majority_ratio_er_standard": varies the edge probability
      of an Erdős--Rényi graph under the standard voter model
    - "estimate_average_majority_ratio_er_hypergraph_noisy": varies a hyperedge
      probability of a generalized Erdős--Rényi hypergraph under the noisy voter model
    - "estimate_average_majority_ratio_recursive_er_adaptive": varies a parameter
      of the recursive Erdős--Rényi model under the adaptive voter model
    - "estimate_average_majority_ratio_adaptive_p_triangle": studies the effect
      of the triangle replacement probability in the adaptive voter model
    - "estimate_average_majority_ratio_hsbm_noisy_p3": studies the effect of the
      3-edge homophily parameter in the Hypergraph Stochastic Block Model

Visualization:
    - "plot_average_majority_ratio_vs_parameter": plots the average final
      majority-opinion ratio as a function of the varying parameter
"""

import numpy as np
import matplotlib.pyplot as plt

from helper_functions import generate_opinions, prepare_graph_for_simulation
from rand_hypergraph_models import (
    generate_general_er_hypergraph,
    generate_recursive_er_simplicial_complex,
    generate_generalized_hsbm
)
from opinion_spread_models import (
    standard_voter_step,
    noisy_voter_step,
    adaptive_voter_step
)
from run_simulations import run_simulation, absorbed_standard, absorbed_noisy, absorbed_adaptive
from analysis1 import majority_ratio

def estimate_average_majority_ratio_er_standard(
    n,
    p_edge_values,
    opinion_kwargs,
    num_runs=10,
    max_steps=500,
    base_seed=123
):
    """
    Estimate the average final majority-opinion ratio for the standard voter model.

    Input:
        n: number of vertices.
        p_edge_values: edge probabilities to test.
        opinion_kwargs: parameters of the opinion generator.
        num_runs: number of simulations per parameter value.
        max_steps: maximum number of simulation steps.
        base_seed: random seed.

    Output:
        Dictionary mapping each edge probability to the average final
        majority-opinion ratio.

    The simulations are carried out on Erdős--Rényi graphs.
    """
    results = {}

    for i, p_edge in enumerate(p_edge_values):
        majority_ratios = []

        for run in range(num_runs):
            rng = np.random.default_rng(base_seed + 1000 * i + run)

            opinions = generate_opinions(n=n, rng=rng, **opinion_kwargs)

            graph = generate_general_er_hypergraph(
                n=n,
                probabilities={2: p_edge},
                opinions=opinions,
                rng=rng
            )

            prepare_graph_for_simulation(graph)

            result = run_simulation(
                graph=graph,
                step_function=standard_voter_step,
                max_steps=max_steps,
                rng=rng,
                step_kwargs={},
                absorbed_function=absorbed_standard,
                track_history=False
            )

            majority_ratios.append(majority_ratio(result["graph"]))

        results[p_edge] = np.mean(majority_ratios)

    return results

def estimate_average_majority_ratio_er_hypergraph_noisy(
    n,
    fixed_probabilities,
    varying_edge_size,
    varying_values,
    noisy_kwargs,
    opinion_kwargs,
    num_runs=10,
    max_steps=500,
    base_seed=123
):
    """
    Estimate the average final majority-opinion ratio for the noisy voter model.

    Input:
        n: number of vertices.
        fixed_probabilities: fixed hyperedge probabilities.
        varying_edge_size: hyperedge size whose probability is varied.
        varying_values: tested probability values.
        noisy_kwargs: parameters of the noisy voter model.
        opinion_kwargs: parameters of the opinion generator.
        num_runs: number of simulations per parameter value.
        max_steps: maximum number of simulation steps.
        base_seed: random seed.

    Output:
        Dictionary mapping each tested probability to the average final
        majority-opinion ratio.

    The simulations are carried out on generalized Erdős--Rényi hypergraphs.
    """
    results = {}

    for i, value in enumerate(varying_values):
        majority_ratios = []

        for run in range(num_runs):
            rng = np.random.default_rng(base_seed + 1000 * i + run)

            opinions = generate_opinions(n=n, rng=rng, **opinion_kwargs)

            probabilities = dict(fixed_probabilities)
            probabilities[varying_edge_size] = value

            graph = generate_general_er_hypergraph(
                n=n,
                probabilities=probabilities,
                opinions=opinions,
                rng=rng
            )

            prepare_graph_for_simulation(graph)

            result = run_simulation(
                graph=graph,
                step_function=noisy_voter_step,
                max_steps=max_steps,
                rng=rng,
                step_kwargs=noisy_kwargs,
                absorbed_function=absorbed_noisy,
                track_history=False
            )

            majority_ratios.append(majority_ratio(result["graph"]))

        results[value] = np.mean(majority_ratios)

    return results

def estimate_average_majority_ratio_recursive_er_adaptive(
    n,
    fixed_generator_kwargs,
    varying_generator_param,
    varying_values,
    adaptive_kwargs,
    opinion_kwargs,
    num_runs=10,
    max_steps=500,
    base_seed=123
):
    """
    Estimate the average final majority-opinion ratio for the adaptive voter model.

    Input:
        n: number of vertices.
        fixed_generator_kwargs: fixed graph-generator parameters.
        varying_generator_param: parameter to be varied.
        varying_values: tested parameter values.
        adaptive_kwargs: parameters of the adaptive voter model.
        opinion_kwargs: parameters of the opinion generator.
        num_runs: number of simulations per parameter value.
        max_steps: maximum number of simulation steps.
        base_seed: random seed.

    Output:
        Dictionary mapping each tested parameter value to the average final
        majority-opinion ratio.

    The simulations are carried out on recursive Erdős--Rényi simplicial complexes.
    """
    results = {}

    for i, value in enumerate(varying_values):
        majority_ratios = []

        for run in range(num_runs):
            rng = np.random.default_rng(base_seed + 1000 * i + run)

            opinions = generate_opinions(n=n, rng=rng, **opinion_kwargs)

            generator_kwargs = dict(fixed_generator_kwargs)
            generator_kwargs[varying_generator_param] = value
            generator_kwargs["n"] = n

            graph = generate_recursive_er_simplicial_complex(
                opinions=opinions,
                rng=rng,
                **generator_kwargs
            )

            prepare_graph_for_simulation(graph)

            result = run_simulation(
                graph=graph,
                step_function=adaptive_voter_step,
                max_steps=max_steps,
                rng=rng,
                step_kwargs=adaptive_kwargs,
                absorbed_function=absorbed_adaptive,
                track_history=False
            )

            majority_ratios.append(majority_ratio(result["graph"]))

        results[value] = np.mean(majority_ratios)

    return results

def estimate_average_majority_ratio_adaptive_p_triangle(
    n,
    generator_kwargs,
    fixed_adaptive_kwargs,
    p_triangle_values,
    opinion_kwargs,
    num_runs=10,
    max_steps=500,
    base_seed=123
):
    """
    Estimate the effect of the triangle replacement probability.

    Input:
        n: number of vertices.
        generator_kwargs: graph-generator parameters.
        fixed_adaptive_kwargs: fixed adaptive-model parameters.
        p_triangle_values: tested triangle replacement probabilities.
        opinion_kwargs: parameters of the opinion generator.
        num_runs: number of simulations per parameter value.
        max_steps: maximum number of simulation steps.
        base_seed: random seed.

    Output:
        Dictionary mapping each p_triangle value to the average final
        majority-opinion ratio.

    The simulations are carried out using the adaptive voter model.
    """
    results = {}

    for i, p_triangle in enumerate(p_triangle_values):
        majority_ratios = []

        for run in range(num_runs):
            rng = np.random.default_rng(base_seed + 1000 * i + run)

            opinions = generate_opinions(n=n, rng=rng, **opinion_kwargs)

            graph = generate_recursive_er_simplicial_complex(
                opinions=opinions,
                rng=rng,
                **generator_kwargs
            )

            prepare_graph_for_simulation(graph)

            adaptive_kwargs = dict(fixed_adaptive_kwargs)
            adaptive_kwargs["p_triangle"] = p_triangle

            result = run_simulation(
                graph=graph,
                step_function=adaptive_voter_step,
                max_steps=max_steps,
                rng=rng,
                step_kwargs=adaptive_kwargs,
                absorbed_function=absorbed_adaptive,
                track_history=False
            )

            majority_ratios.append(majority_ratio(result["graph"]))

        results[p_triangle] = np.mean(majority_ratios)

    return results

def estimate_average_majority_ratio_hsbm_noisy_p3(
    n,
    fixed_probabilities,
    p3_values,
    noisy_kwargs,
    opinion_kwargs,
    num_runs=10,
    max_steps=500,
    base_seed=123
):
    """
    Estimate the effect of the 3-edge homophily parameter in the HSBM.

    Input:
        n: number of vertices.
        fixed_probabilities: fixed HSBM probabilities.
        p3_values: tested values of the 3-edge within-community probability.
        noisy_kwargs: parameters of the noisy voter model.
        opinion_kwargs: parameters of the opinion generator.
        num_runs: number of simulations per parameter value.
        max_steps: maximum number of simulation steps.
        base_seed: random seed.

    Output:
        Dictionary mapping each p3 value to the average final
        majority-opinion ratio.

    The simulations are carried out on Hypergraph Stochastic Block Models.
    """
    results = {}

    for i, p3 in enumerate(p3_values):
        majority_ratios = []

        for run in range(num_runs):
            rng = np.random.default_rng(base_seed + 1000 * i + run)

            opinions = generate_opinions(
                n=n,
                rng=rng,
                **opinion_kwargs
            )

            probabilities = dict(fixed_probabilities)

            old_p3, old_q3 = probabilities[3]
            probabilities[3] = (p3, old_q3)

            graph = generate_generalized_hsbm(
                n=n,
                probabilities=probabilities,
                opinions=opinions,
                rng=rng
            )

            prepare_graph_for_simulation(graph)

            result = run_simulation(
                graph=graph,
                step_function=noisy_voter_step,
                max_steps=max_steps,
                rng=rng,
                step_kwargs=noisy_kwargs,
                absorbed_function=absorbed_noisy,
                track_history=False
            )

            majority_ratios.append(majority_ratio(result["graph"]))

        results[p3] = np.mean(majority_ratios)

    return results

def plot_average_majority_ratio_vs_parameter(
    results,
    varying_param_name,
    title,
    figsize=(6, 4),
    show=True
):
    x = np.array(sorted(results.keys()), dtype=float)
    y = np.array([results[val] for val in x], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y, linewidth=2)

    ax.set_xlabel(varying_param_name)
    ax.set_ylabel("Average final majority-opinion ratio")
    ax.set_title(title)

    ax.set_xlim(0, x.max())
    ax.set_ylim(0.5, 1.05)

    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig, ax