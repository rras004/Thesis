import numpy as np
import matplotlib.pyplot as plt

def magnetization(graph):
    """
    Average opinion.
    """
    return np.mean(graph["opinions"])


def absorbed_noisy(graph):
    """
    No active edge remains.
    """
    return len(graph["active_edges"]) == 0


def absorbed_standard(graph):
    """
    For the standard voter model, only active 2-edges matter.
    """
    return all(len(edge) != 2 for edge in graph["active_edges"])


def absorbed_adaptive(graph):
    """
    For the adaptive model, only active 2-edges are selectable.
    """
    return all(len(edge) != 2 for edge in graph["active_edges"])


def fraction_opinions(graph):
    """
    Fraction of nodes with opinion +1 and -1.
    """
    n = len(graph["nodes"])
    num_pos, num_neg = count_opinions(graph)
    return num_pos / n, num_neg / n


def count_edges(graph, size=None):
    if size is None:
        return len(graph["edges"])
    return len(graph["edges_by_size"].get(size, set()))
   

def count_opinions(graph):
    opinions = graph["opinions"]
    num_pos = np.sum(opinions == 1)
    num_neg = np.sum(opinions == -1)
    return int(num_pos), int(num_neg)

def count_active_edges(graph, size=None):
    if size is None:
        return len(graph["active_edges"])

    return sum(1 for edge in graph["active_edges"] if len(edge) == size)

def count_inactive_edges(graph, size=None):
    return count_edges(graph, size=size) - count_active_edges(graph, size=size)


def has_consensus(graph):
    num_pos, num_neg = count_opinions(graph)
    n = len(graph["nodes"])
    return (num_pos == n and num_neg == 0) or (num_neg == n and num_pos == 0)


def final_node_ratios(result):
    graph = result["graph"]
    num_pos, num_neg = count_opinions(graph)
    n = len(graph["nodes"])

    return {
        "ratio_pos": num_pos / n,
        "ratio_neg": num_neg / n
    }

def run_simulation(
    graph,
    step_function,
    max_steps,
    rng=None,
    step_kwargs=None,
    absorbed_function=None,
    track_history=True
):
    if rng is None:
        rng = np.random.default_rng()

    if step_kwargs is None:
        step_kwargs = {}

    if track_history:
        history = {
            "magnetization": [],
            "num_pos": [],
            "num_neg": [],
            "frac_pos": [],
            "frac_neg": [],
            "active_edges": [],
            "active_2_edges": [],
            "num_2_edges": [],
            "num_3_edges": []
        }
    else:
        history = None

    for t in range(max_steps):
        if track_history:
            num_pos, num_neg = count_opinions(graph)
            frac_pos, frac_neg = fraction_opinions(graph)

            history["magnetization"].append(magnetization(graph))
            history["num_pos"].append(num_pos)
            history["num_neg"].append(num_neg)
            history["frac_pos"].append(frac_pos)
            history["frac_neg"].append(frac_neg)
            history["active_edges"].append(count_active_edges(graph))
            history["active_2_edges"].append(count_active_edges(graph, size=2))
            history["num_2_edges"].append(count_edges(graph, size=2))
            history["num_3_edges"].append(count_edges(graph, size=3))

        if absorbed_function is not None and absorbed_function(graph):
            return {
                "graph": graph,
                "steps": t,
                "absorbed": True,
                "history": history
            }

        step_function(graph, rng=rng, **step_kwargs)

    return {
        "graph": graph,
        "steps": max_steps,
        "absorbed": False,
        "history": history
    }

def simulation_summary(result):
    graph = result["graph"]
    num_pos, num_neg = count_opinions(graph)

    return {
        "absorbed": result["absorbed"],
        "consensus": has_consensus(graph),
        "final_num_pos": num_pos,
        "final_num_neg": num_neg,
        "steps": result["steps"],
        "final_active_edges": count_active_edges(graph),
        "final_inactive_edges": count_inactive_edges(graph)
    }

def print_simulation_summary(result):
    summary = simulation_summary(result)

    print("Simulation summary")
    print(f"Absorbed: {summary['absorbed']}")
    print(f"Consensus: {summary['consensus']}")
    print(f"Steps: {summary['steps']}")
    print()

    print("Node opinions (final)")
    print(f"+1 nodes: {summary['final_num_pos']}")
    print(f"-1 nodes: {summary['final_num_neg']}")
    print()

    if "initial_active_edges" in summary:
        print("Edges")
        print(f"Initial active edges: {summary['initial_active_edges']}")
        print(f"Initial inactive edges: {summary['initial_inactive_edges']}")
        print(f"Final active edges: {summary['final_active_edges']}")
        print(f"Final inactive edges: {summary['final_inactive_edges']}")
    else:
        print("Final active edges:", summary["final_active_edges"])
        print("Final inactive edges:", summary["final_inactive_edges"])

def plot_final_node_ratios(result, figsize=(6, 4), show=True):
    """
    Plot the final ratio of +1 and -1 nodes.
    """
    ratios = final_node_ratios(result)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(["+1 nodes", "-1 nodes"], [ratios["ratio_pos"], ratios["ratio_neg"]])
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1)
    ax.set_title("Final ratio of node opinions")

    if show:
        plt.show()

    return fig, ax

def plot_active_edges(result, figsize=(8, 5), show=True):

    history = result.get("history", None)

    if history is None:
        raise ValueError("No history found. Run simulation with track_history=True.")

    if "active_edges" not in history:
        raise ValueError("History does not contain 'active_edges'.")

    values = history["active_edges"]
    steps = np.arange(len(values))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(steps, values, label="Active edges")

    ax.set_xlabel("Step")
    ax.set_ylabel("Number of active edges")
    ax.set_title("Active edges over time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show:
        plt.show()

    return fig, ax


def run_many_simulations(
    generator_function,
    generator_kwargs,
    prepare_function,
    step_function,
    step_kwargs,
    absorbed_function,
    num_runs,
    max_steps,
    opinion_generator=None,
    opinion_kwargs=None,
    base_seed=123,
    track_history=False
):
    """
    Run many independent simulations.

    Returns a list of result dictionaries.
    """
    if opinion_kwargs is None:
        opinion_kwargs = {}

    results = []

    for run in range(num_runs):
        rng = np.random.default_rng(base_seed + run)

        # generate opinions
        if opinion_generator is not None:
            opinions = opinion_generator(generator_kwargs["n"], rng=rng, **opinion_kwargs)
        else:
            opinions = None

        # generate graph
        if opinions is not None:
            graph = generator_function(rng=rng, opinions=opinions, **generator_kwargs)
        else:
            graph = generator_function(rng=rng, **generator_kwargs)

        # prepare graph
        prepare_function(graph)

        # run simulation
        sim_result = run_simulation(
            graph=graph,
            step_function=step_function,
            max_steps=max_steps,
            rng=rng,
            step_kwargs=step_kwargs,
            absorbed_function=absorbed_function,
            track_history=track_history
        )

        final_num_pos, final_num_neg = count_opinions(sim_result["graph"])
        final_frac_pos, final_frac_neg = fraction_opinions(sim_result["graph"])

        results.append({
            "run": run,
            "steps": sim_result["steps"],
            "absorbed": sim_result["absorbed"],
            "consensus": has_consensus(sim_result["graph"]),
            "final_magnetization": magnetization(sim_result["graph"]),
            "final_num_pos": final_num_pos,
            "final_num_neg": final_num_neg,
            "final_frac_pos": final_frac_pos,
            "final_frac_neg": final_frac_neg,
            "final_active_edges": count_active_edges(sim_result["graph"]),
            "final_inactive_edges": count_inactive_edges(sim_result["graph"]),
            "history": sim_result["history"]})

    return results


def plot_opinion_counts(result, figsize=(8, 5), show=True):
    history = result.get("history", None)

    if history is None:
        raise ValueError("No history found. Run simulation with track_history=True.")

    if "num_pos" not in history or "num_neg" not in history:
        raise ValueError("History does not contain 'num_pos' and 'num_neg'.")

    times = np.arange(len(history["num_pos"]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, history["num_pos"], label="Opinion +1")
    ax.plot(times, history["num_neg"], label="Opinion -1")

    ax.set_xlabel("Step")
    ax.set_ylabel("Number of nodes")
    ax.set_title("Opinion counts over time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig, ax


