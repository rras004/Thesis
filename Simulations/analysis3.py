"""
This file contains analysis functions used to study the effect of a model parameter. The functions are all used in results3.py.

The functions can be grouped into two categories.

Simulation execution:
    - "estimate_consensus_and_majority_ratio": estimates how a varying parameter
      influences consensus probability, majority-opinion ratio, and inactive-edge ratio

Visualization:
    - "plot_consensus_and_majority_vs_parameter": plots the measured quantities
      as functions of the varying parameter
"""

from matplotlib import pyplot as plt 
import numpy as np 
import copy
from analysis1 import majority_ratio
from run_simulations import has_consensus, run_simulation, count_edges, count_inactive_edges

def estimate_consensus_and_majority_ratio(
    base_graph,
    prepare_function,
    step_function,
    absorbed_function,
    fixed_param_name,
    fixed_param_value,
    varying_param_name,
    varying_values,
    num_runs=50,
    max_steps=500,
    base_seed=123,
    extra_step_kwargs=None
):
    
    """
    Estimate the effect of a model parameter on the simulation outcome.

    Input:
        base_graph: graph used as the starting point of every simulation.
        prepare_function: function preparing the graph for simulation.
        step_function: opinion update rule.
        absorbed_function: stopping criterion.
        fixed_param_name: name of the fixed parameter.
        fixed_param_value: value of the fixed parameter.
        varying_param_name: name of the varying parameter.
        varying_values: values tested for the varying parameter.
        num_runs: number of simulations per parameter value.
        max_steps: maximum number of simulation steps.
        base_seed: random seed.
        extra_step_kwargs: additional model parameters.

    Output:
        Dictionary containing, for each parameter value:
            - consensus probability
            - average final majority-opinion ratio
            - average final inactive-edge ratio

    For every parameter value, multiple simulations are carried out and
    the resulting statistics are averaged.
    """

    if extra_step_kwargs is None:
        extra_step_kwargs = {}

    results = {}

    for i, value in enumerate(varying_values):
        consensus_indicators = []
        final_majority_ratios = []
        final_inactive_edge_ratios = []

        for run in range(num_runs):
            graph = copy.deepcopy(base_graph)
            prepare_function(graph)

            step_kwargs = dict(extra_step_kwargs)
            step_kwargs[fixed_param_name] = fixed_param_value
            step_kwargs[varying_param_name] = value

            rng_sim = np.random.default_rng(base_seed + 1000 * i + run)

            result = run_simulation(
                graph=graph,
                step_function=step_function,
                max_steps=max_steps,
                rng=rng_sim,
                step_kwargs=step_kwargs,
                absorbed_function=absorbed_function,
                track_history=False
            )

            final_graph = result["graph"]

            consensus_indicators.append(int(has_consensus(final_graph)))
            final_majority_ratios.append(majority_ratio(final_graph))

            total_edges = count_edges(final_graph)
            if total_edges == 0:
                final_inactive_edge_ratios.append(np.nan)
            else:
                final_inactive_edge_ratios.append(
                    count_inactive_edges(final_graph) / total_edges
                )

        results[value] = {
            "consensus_probability": np.mean(consensus_indicators),
            "average_final_majority_ratio": np.mean(final_majority_ratios),
            "average_final_inactive_edge_ratio": np.nanmean(final_inactive_edge_ratios)
        }

    return results

def plot_consensus_and_majority_vs_parameter(
    results,
    varying_param_name,
    fixed_text,
    model_name="Model",
    figsize=(6, 4),
    show=True
):
    x = np.array(sorted(results.keys()), dtype=float)

    y_consensus = np.array(
        [results[val]["consensus_probability"] for val in x],
        dtype=float
    )

    y_majority = np.array(
        [results[val]["average_final_majority_ratio"] for val in x],
        dtype=float
    )

    y_inactive = np.array(
        [results[val]["average_final_inactive_edge_ratio"] for val in x],
        dtype=float
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y_consensus, linewidth=2, label="Consensus probability")
    ax.plot(x, y_majority, linewidth=2, label="Average final majority ratio")
    ax.plot(x, y_inactive, linewidth=2, label="Average final inactive-edge ratio")

    ax.set_xlabel(varying_param_name)
    ax.set_ylabel("Ratio / probability")
    ax.set_title(f"{model_name}: {fixed_text}")
    ax.set_xlim(0, x.max())
    ax.set_ylim(0, 1.05)

    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if show:
        plt.show()

    return fig, ax