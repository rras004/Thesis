"""
This file contains analysis functions used to compare the opinion spread models. The functions of this file are used in results1.py.
The functions can be grouped into four categories.

Simulation statistics:
    - "majority_ratio": computes the ratio of the majority opinion
    - "inactive_edge_ratio": computes the ratio of inactive edges
    - "final_state_type": classifies the final state of a simulation

Graph manipulation:
    - "extract_2_edge_graph": extracts the graph consisting only of 2-edges

Simulation execution:
    - "run_fixed_steps_ratios": runs a simulation for a fixed number of steps and records statistics
    - "compare_models_ratios": compares the standard, noisy, and adaptive voter models across multiple runs

Result summaries:
    - "print_final_state_summary": prints the distribution of final states

Visualization:
    - "plot_model_comparison_majority_ratio": plots the average majority-opinion ratio trajectories
    - "plot_all_runs_majority_ratio": plots all majority-opinion trajectories
    - "plot_model_comparison_inactive_edge_ratio": plots the average inactive-edge ratio trajectories

Additional analysis:
    - "compute_mean_absolute_deviation": computes the mean absolute deviation of the recorded trajectories
"""

import numpy as np
import copy
import matplotlib.pyplot as plt

from helper_functions import prepare_graph_for_simulation, generate_opinions
from opinion_spread_models import adaptive_voter_step, noisy_voter_step, standard_voter_step
from run_simulations import count_edges, count_inactive_edges
from rand_hypergraph_models import generate_recursive_er_simplicial_complex


def majority_ratio(graph):
    """
    Returns a value between 0.5 and 1 representing the fraction of vertices holding the majority opinion. 
    """
    opinions = graph["opinions"]
    num_pos = np.sum(opinions == 1)
    num_neg = np.sum(opinions == -1)
    n = len(opinions)

    return max(num_pos, num_neg) / n


def inactive_edge_ratio(graph):
    """
    Returns the number of inactive edges divided by the total number of edges.
    """
    total_edges = count_edges(graph)

    if total_edges == 0:
        return np.nan

    return count_inactive_edges(graph) / total_edges


def final_state_type(graph):
    """
    Classify the final state into one of three categories:
    - consensus
    - absorbed_without_consensus
    - active_edges_remain
    """
    opinions = graph["opinions"]

    has_full_consensus = np.all(opinions == opinions[0])
    no_active_edges = len(graph["active_edges"]) == 0

    if has_full_consensus:
        return "consensus"
    elif no_active_edges:
        return "absorbed_without_consensus"
    else:
        return "active_edges_remain"


def extract_2_edge_graph(graph):
    """
    Extract the graph consisting only of 2-edges.
    """
    edges_2 = copy.deepcopy(graph["edges_by_size"].get(2, set()))

    new_graph = {
        "nodes": copy.deepcopy(graph["nodes"]),
        "edges_by_size": {2: edges_2},
        "edges": set(edges_2),
        "opinions": np.array(graph["opinions"], copy=True)
    }

    return new_graph


def run_fixed_steps_ratios(
    graph,
    step_function,
    max_steps,
    step_kwargs=None,
    rng=None
):
    """
    Run a simulation for a fixed number of steps and record statistics.

    Input:
        graph: graph dictionary.
        step_function: opinion update rule.
        max_steps: number of simulation steps.
        step_kwargs: additional parameters passed to the update rule.
        rng: optional random number generator.

    Output:
        Dictionary containing the trajectory of:
            - majority-opinion ratio
            - inactive-edge ratio

    The statistics are recorded after every simulation step.
    """
    if rng is None:
        rng = np.random.default_rng()

    if step_kwargs is None:
        step_kwargs = {}

    majority_ratios = [majority_ratio(graph)]
    inactive_ratios = [inactive_edge_ratio(graph)]

    for _ in range(max_steps):
        step_function(graph, rng=rng, **step_kwargs)

        majority_ratios.append(majority_ratio(graph))
        inactive_ratios.append(inactive_edge_ratio(graph))

    return {
        "majority_ratio": np.array(majority_ratios, dtype=float),
        "inactive_edge_ratio": np.array(inactive_ratios, dtype=float)
    }


def compare_models_ratios(
    recursive_generator_kwargs,
    opinion_kwargs,
    max_steps,
    K,
    standard_kwargs=None,
    noisy_kwargs=None,
    adaptive_kwargs=None,
    base_seed=123
):
    
    """
    Compare the standard, noisy, and adaptive voter models.

    Input:
        recursive_generator_kwargs: parameters of the graph generator.
        opinion_kwargs: parameters of the opinion generator.
        max_steps: number of simulation steps.
        K: number of runs per model.
        standard_kwargs: parameters of the standard voter model.
        noisy_kwargs: parameters of the noisy voter model.
        adaptive_kwargs: parameters of the adaptive voter model.
        base_seed: random seed.

    Output:
        Dictionary containing all recorded trajectories and final-state
        statistics for each opinion model.

    Each model is simulated K times and the resulting statistics are stored
    for later analysis and visualization.
    """

    if standard_kwargs is None:
        standard_kwargs = {}

    if noisy_kwargs is None:
        noisy_kwargs = {}

    if adaptive_kwargs is None:
        adaptive_kwargs = {}

    results = {
        "standard": {
            "majority_ratio": [],
            "inactive_edge_ratio": [],
            "final_state_counts": {
                "consensus": 0,
                "absorbed_without_consensus": 0,
                "active_edges_remain": 0
            }
        },
        "noisy": {
            "majority_ratio": [],
            "inactive_edge_ratio": [],
            "final_state_counts": {
                "consensus": 0,
                "absorbed_without_consensus": 0,
                "active_edges_remain": 0
            }
        },
        "adaptive": {
            "majority_ratio": [],
            "inactive_edge_ratio": [],
            "final_state_counts": {
                "consensus": 0,
                "absorbed_without_consensus": 0,
                "active_edges_remain": 0
            }
        }
    }

    n = recursive_generator_kwargs["n"]

    model_specs = [
        ("standard", standard_voter_step, standard_kwargs),
        ("noisy", noisy_voter_step, noisy_kwargs),
        ("adaptive", adaptive_voter_step, adaptive_kwargs)
    ]

    for model_idx, (model_name, step_function, step_kwargs) in enumerate(model_specs):
        for run in range(K):
            rng = np.random.default_rng(base_seed + 10000 * model_idx + run)

            opinions = generate_opinions(
                n=n,
                rng=rng,
                **opinion_kwargs
            )

            graph = generate_recursive_er_simplicial_complex(
                opinions=opinions,
                rng=rng,
                **recursive_generator_kwargs
            )

            if model_name == "standard":
                graph = extract_2_edge_graph(graph)

            prepare_graph_for_simulation(graph)

            trajectories = run_fixed_steps_ratios(
                graph=graph,
                step_function=step_function,
                max_steps=max_steps,
                step_kwargs=step_kwargs,
                rng=rng
            )

            results[model_name]["majority_ratio"].append(
                trajectories["majority_ratio"]
            )

            results[model_name]["inactive_edge_ratio"].append(
                trajectories["inactive_edge_ratio"]
            )

            state = final_state_type(graph)
            results[model_name]["final_state_counts"][state] += 1

    for model_name in results:
        results[model_name]["majority_ratio"] = np.array(
            results[model_name]["majority_ratio"],
            dtype=float
        )

        results[model_name]["inactive_edge_ratio"] = np.array(
            results[model_name]["inactive_edge_ratio"],
            dtype=float
        )

        results[model_name]["final_state_ratios"] = {
            state: count / K
            for state, count in results[model_name]["final_state_counts"].items()
        }

    return results


def print_final_state_summary(results):
    print("Final state summary")
    print()

    for model_name in ["standard", "noisy", "adaptive"]:
        total_runs = results[model_name]["majority_ratio"].shape[0]
        counts = results[model_name]["final_state_counts"]
        ratios = results[model_name]["final_state_ratios"]

        print(model_name.capitalize())
        print(f"  Consensus: {counts['consensus']} / {total_runs} ({ratios['consensus']:.3f})")
        print(
            f"  Absorbed without consensus: "
            f"{counts['absorbed_without_consensus']} / {total_runs} "
            f"({ratios['absorbed_without_consensus']:.3f})"
        )
        print(
            f"  Active edges remain: "
            f"{counts['active_edges_remain']} / {total_runs} "
            f"({ratios['active_edges_remain']:.3f})"
        )
        print()


def plot_model_comparison_majority_ratio(results, figsize=(9, 5), show=True):
    avg_standard = np.nanmean(results["standard"]["majority_ratio"], axis=0)
    avg_noisy = np.nanmean(results["noisy"]["majority_ratio"], axis=0)
    avg_adaptive = np.nanmean(results["adaptive"]["majority_ratio"], axis=0)

    steps = np.arange(len(avg_standard))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(steps, avg_standard, label="Standard", linewidth=2)
    ax.plot(steps, avg_noisy, label="Noisy", linewidth=2)
    ax.plot(steps, avg_adaptive, label="Adaptive", linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Average majority-opinion ratio")
    ax.set_title("Comparison of opinion spread models")
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.margins(x=0)

    if show:
        plt.show()

    return fig, ax


def plot_all_runs_majority_ratio(results, figsize=(9, 5), show=True):
    steps = np.arange(results["standard"]["majority_ratio"].shape[1])

    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        "standard": "tab:blue",
        "noisy": "tab:orange",
        "adaptive": "tab:green"
    }

    labels_added = {
        "standard": False,
        "noisy": False,
        "adaptive": False
    }

    for model_name in ["standard", "noisy", "adaptive"]:
        trajectories = results[model_name]["majority_ratio"]

        for run in trajectories:
            label = model_name.capitalize() if not labels_added[model_name] else None

            ax.plot(
                steps,
                run,
                color=colors[model_name],
                alpha=0.3,
                linewidth=2,
                label=label
            )

            labels_added[model_name] = True

    ax.set_xlabel("Step")
    ax.set_ylabel("Majority-opinion ratio")
    ax.set_title("All simulation runs: majority-opinion ratio")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.margins(x=0)

    if show:
        plt.show()

    return fig, ax


def plot_model_comparison_inactive_edge_ratio(results, figsize=(9, 5), show=True):
    avg_standard = np.nanmean(results["standard"]["inactive_edge_ratio"], axis=0)
    avg_noisy = np.nanmean(results["noisy"]["inactive_edge_ratio"], axis=0)
    avg_adaptive = np.nanmean(results["adaptive"]["inactive_edge_ratio"], axis=0)

    steps = np.arange(len(avg_standard))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(steps, avg_standard, linewidth=2, label="Standard")
    ax.plot(steps, avg_noisy, linewidth=2, label="Noisy")
    ax.plot(steps, avg_adaptive, linewidth=2, label="Adaptive")

    ax.set_xlabel("Step")
    ax.set_ylabel("Average ratio of inactive edges / all edges")
    ax.set_title("Comparison of opinion spread models")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.margins(x=0)

    if show:
        plt.show()

    return fig, ax


def compute_mean_absolute_deviation(results, quantity):
    """
    quantity should be either:
    - "majority_ratio"
    - "inactive_edge_ratio"
    """
    mad_values = {}

    for model_name in ["standard", "noisy", "adaptive"]:
        data = results[model_name][quantity]

        mean_t = np.nanmean(data, axis=0)
        deviations = np.abs(data - mean_t)
        mad = np.nanmean(deviations)

        mad_values[model_name] = mad

    return mad_values