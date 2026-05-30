# Opinion Spread on Random Networks

This repository contains the code accompanying the Bachelor's thesis *Opinion Spread on Random Networks*.

The thesis contains two chapters that rely on computational methods.

## Chapter 2: Random Hypergraph Models

The `Figures` folder contains the Python scripts used to generate the random graph and hypergraph figures presented in Chapter 2.

## Chapter 5: Simulations

The remaining code is related to Chapter 5, which investigates opinion spread through computer simulations. The `Simulations` folder contains the implementations of the random network models, opinion dynamics models, simulation framework, and analysis tools used throughout the chapter.

The simulation files can be divided into two groups.

### Core Implementation Files

These files contain the implementations of:

* random graph and hypergraph models,
* opinion spread models,
* simulation procedures,
* helper functions,
* analysis functions.

The main implementation files are:

* `rand_hypergraph_models.py`
* `opinion_spread_models.py`
* `run_simulations.py`
* `helper_functions.py`
* `analysis1.py`
* `analysis2.py`
* `analysis3.py`
* `analysis4.py`

### Results Notebooks

The notebooks

* `results1.ipynb`
* `results2.ipynb`
* `results3.ipynb`
* `results4.ipynb`

contain the experiments presented in Chapter 5. Running these notebooks reproduces the simulations, statistics, and figures used in the thesis.

Chapter 5 is divided into four sections, each focusing on a different aspect of the simulation results. Each section has a corresponding pair of files consisting of an analysis module and a results notebook. The analysis modules contain the functions used to perform the experiments, while the notebooks execute these functions with the parameter values used in the thesis and generate the corresponding figures.

The numbering of the files follows the order of the sections in Chapter 5:

* `analysis1.py` and `results1.ipynb`
* `analysis2.py` and `results2.ipynb`
* `analysis3.py` and `results3.ipynb`
* `analysis4.py` and `results4.ipynb`

The figures appearing in the thesis are generated directly by the corresponding results notebooks.
