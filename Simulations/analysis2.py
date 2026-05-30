"""
This file contains analysis functions used to compare graph generating models. The functions are used in results2.py.
while keeping the noisy voter model fixed.

The functions can be grouped into three categories.

Simulation execution:
    - "run_fixed_steps_noisy_ratios": runs a noisy voter simulation and records statistics
    - "compare_graph_models_noisy_ratios": compares different graph models through repeated simulations

Simulation statistics:
    - "majority_ratio": majority-opinion ratio (imported from analysis1)
    - "inactive_edge_ratio": inactive-edge ratio (imported from analysis1)

Visualization:
    - "plot_graph_model_comparison_noisy_majority_ratio": plots average majority-opinion trajectories
    - "plot_graph_model_comparison_noisy_inactive_edge_ratio": plots average inactive-edge-ratio trajectories
"""

import numpy as np
import matplotlib.pyplot as plt

from analysis1 import majority_ratio, inactive_edge_ratio
from helper_functions import generate_opinions, prepare_graph_for_simulation
from opinion_spread_models import noisy_voter_step


def run_fixed_steps_noisy_ratios(
    graph,
    max_steps,
    noisy_kwargs,
    rng=None
):
    """
    Run one noisy-voter simulation and record both:
    - majority-opinion ratio
    - inactive-edge ratio
    from the same simulation.
    """
    if rng is None:
        rng = np.random.default_rng()

    majority_ratios = [majority_ratio(graph)]
    inactive_ratios = [inactive_edge_ratio(graph)]

    for _ in range(max_steps):
        noisy_voter_step(graph, rng=rng, **noisy_kwargs)

        majority_ratios.append(majority_ratio(graph))
        inactive_ratios.append(inactive_edge_ratio(graph))

    return {
        "majority_ratio": np.array(majority_ratios, dtype=float),
        "inactive_edge_ratio": np.array(inactive_ratios, dtype=float)
    }


def compare_graph_models_noisy_ratios(
    generator_specs,
    opinion_kwargs,
    noisy_kwargs,
    max_steps,
    K,
    base_seed=123
):
    """
    Compare graph generating models while keeping the noisy voter model fixed.

    For each graph model and each run, both quantities are recorded
    from the same simulation:
    - majority-opinion ratio
    - inactive-edge ratio

    Returns
    -------
    results : dict
        {
            model_name: {
                "majority_ratio": array of shape (K, max_steps + 1),
                "inactive_edge_ratio": array of shape (K, max_steps + 1)
            }
        }
    """
    results = {}

    for model_idx, (model_name, generator_function, generator_kwargs) in enumerate(generator_specs):
        results[model_name] = {
            "majority_ratio": [],
            "inactive_edge_ratio": []
        }

        n = generator_kwargs["n"]

        for run in range(K):
            rng = np.random.default_rng(base_seed + 10000 * model_idx + run)

            opinions = generate_opinions(
                n=n,
                rng=rng,
                **opinion_kwargs
            )

            graph = generator_function(
                opinions=opinions,
                rng=rng,
                **generator_kwargs
            )

            prepare_graph_for_simulation(graph)

            trajectories = run_fixed_steps_noisy_ratios(
                graph=graph,
                max_steps=max_steps,
                noisy_kwargs=noisy_kwargs,
                rng=rng
            )

            results[model_name]["majority_ratio"].append(
                trajectories["majority_ratio"]
            )

            results[model_name]["inactive_edge_ratio"].append(
                trajectories["inactive_edge_ratio"]
            )

        results[model_name]["majority_ratio"] = np.array(
            results[model_name]["majority_ratio"],
            dtype=float
        )

        results[model_name]["inactive_edge_ratio"] = np.array(
            results[model_name]["inactive_edge_ratio"],
            dtype=float
        )

    return results


def plot_graph_model_comparison_noisy_majority_ratio(
    results,
    figsize=(9, 5),
    show=True
):
    """
    Plot average majority-ratio trajectories for different graph models.
    """
    model_names = list(results.keys())
    steps = np.arange(next(iter(results.values()))["majority_ratio"].shape[1])

    fig, ax = plt.subplots(figsize=figsize)

    for model_name in model_names:
        avg = np.nanmean(results[model_name]["majority_ratio"], axis=0)
        ax.plot(steps, avg, linewidth=2, label=model_name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Average majority-opinion ratio")
    ax.set_title("Noisy voter model on different graph models")
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.margins(x=0)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_graph_model_comparison_noisy_inactive_edge_ratio(
    results,
    figsize=(9, 5),
    show=True
):
    """
    Plot average inactive-edge-ratio trajectories for different graph models.
    """
    model_names = list(results.keys())
    steps = np.arange(next(iter(results.values()))["inactive_edge_ratio"].shape[1])

    fig, ax = plt.subplots(figsize=figsize)

    for model_name in model_names:
        avg = np.nanmean(results[model_name]["inactive_edge_ratio"], axis=0)
        ax.plot(steps, avg, linewidth=2, label=model_name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Average ratio of inactive edges / all edges")
    ax.set_title("Noisy voter model on different graph models")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.margins(x=0)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax

