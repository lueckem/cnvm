[![build](https://github.com/lueckem/cnvm/actions/workflows/build.yml/badge.svg)](https://github.com/lueckem/cnvm/actions/workflows/build.yml)

# Continuous-time noisy voter model (CNVM)
This package provides an efficient implementation of the CNVM, which is a dynamical system on a network of interacting *agents*.
It can be used to simulate how opinions about certain issues develop over time within a population of agents,
or how an infectious disease spreads.
Although the interaction rules of the CNVM are quite simple, the resulting dynamics is complex and rich.
There are numerous papers studying the behavior of the CNVM or similar voter models.

## Installation
Install the CNVM package from the PyPI repository
```
pip install cnvm
```
or get the latest version directly from GitHub:
```
pip install git+https://github.com/lueckem/cnvm
```


## About the CNVM
Let a network (undirected simple graph) of $N$ nodes be given. The nodes represent agents and the edges social interactions. 
Each node is endowed with one of $M$ discrete opinions. Thus, the system state is given by a vector $x \in \{1,\dots,M\}^N$, where $x_i$ describes the opinion of node $i$.
Each node's opinion $x_i \in \{1,\dots,M\}$ changes over time according to a continuous-time Markov chain (Markov jump process).
Given the current system state $x$, the generator matrix $Q^i$ of the continuous-time Markov chain associated with node $i$ is defined as
$$ Q^i \in \mathbb{R}^{M \times M},\quad (Q^i)_{m,n} := r_{m,n} \frac{d_{i,n}(x)}{(d_i)^\alpha} + \tilde{r}_{m,n},\ m\neq n, $$
where $d_{i,n}(x)$ denotes the number of neighbors of node $i$ with opinion $n$ and $d_i$ is the degree of node $i$. The matrices $r, \tilde{r} \in \mathbb{R}^{M \times M}$ and $\alpha \in \mathbb{R}$ are model parameters.

Thus, the transition rates $(Q^i)_{m,n}$ consist of two components. The first component $r_{m,n} \frac{d_{i,n}(x)}{(d_i)^\alpha}$ describes at which rate node $i$ gets ``infected'' by nodes in its neighborhood.
The second part $\tilde{r}_{m,n}$ describes transitions that are independent from the neighborhood.

It should be noted that after a node switches its opinion due to the above dynamics, the system state $x$ changes and hence all the generator matrices $Q^i$ may change as well.

## Basic Usage
First define the model paramaters:
```python
from cnvm import Parameters
import numpy as np
import networkx as nx

r = np.array([[0, .8], [.2, 0]])
r_tilde = np.array([[0, .1], [.2, 0]])
network = nx.erdos_renyi_graph(n=100, p=0.1)

params = Parameters(
    num_opinions=2,
    network=network,
    r=r,
    r_tilde=r_tilde,
    alpha=1,
)
```
Then simulate the model:
```python
from cnvm import CNVM

x_init = np.random.randint(0, 2, 100)
model = CNVM(params)
t, x = model.simulate(t_max=50, x_init=x_init)
```
The output `t` contains the time points of state jumps and `x` the system states after each jump.

A more detailed overview of the package can be found in the jupyter notebook [*examples/tutorial.ipynb.*](examples/tutorial.ipynb.)

## Implementation details
