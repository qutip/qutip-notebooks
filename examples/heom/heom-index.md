---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Hierarchical Equation of Motion Examples

The "hierarchical equations of motion" (HEOM) method is a powerful numerical approach to solve the dynamics and steady-state of a quantum system coupled to a non-Markovian and non-perturbative environment. Originally developed in the context of physical chemistry, it has also been extended and applied to problems in solid-state physics, optics, single-molecule electronics, and biological physics.

QuTiP's implementation of the HEOM is described in detail in https://arxiv.org/abs/2010.10806.

This collection of examples from the paper illustrates how to use QuTiP's HEOM to model and investigate the dynamics of a variety of systems coupled to bosonic or fermionic baths.

## Overview of the notebooks

* [Example 1a: Spin-Bath model (basic)](./heom-1a-spin-bath-model-basic.ipynb)

* [Example 1b: Spin-Bath model (very strong coupling)](./heom-1b-spin-bath-model-very-strong-coupling.ipynb)

* [Example 1c: Spin-Bath model (underdamped case)](./heom-1c-spin-bath-model-underdamped-sd.ipynb)

* [Example 1d: Spin-Bath model, fitting of spectrum and correlation functions](./heom-1d-spin-bath-model-ohmic-fitting.ipynb)

* [Example 1e: Spin-Bath model (pure dephasing)](./heom-1e-spin-bath-model-pure-dephasing.ipynb)

* [Example 2: Dynamics in Fenna-Mathews-Olsen complex (FMO)](./heom-2-fmo-example.ipynb)

* [Example 3: Quantum Heat Transport](./heom-3-quantum-heat-transport.ipynb)

* [Example 4: Dynamical decoupling of a non-Markovian environment](./heom-4-dynamical-decoupling.ipynb)

* [Example 5a: Fermionic single impurity model](./heom-5a-fermions-single-impurity-model.ipynb)

* [Example 5b: Discrete boson coupled to an impurity + fermionic leads](./heom-5b-fermions-discrete-boson-model.ipynb)

```{code-cell} ipython3

```
