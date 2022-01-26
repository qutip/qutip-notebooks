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

# Example 3: Quantum Heat Transport

### Setup

In this notebook, we apply the QuTiP HEOM solver to a quantum system coupled to two bosonic baths and demonstrate how to extract information about the system-bath heat currents from the auxiliary density operators (ADOs).
We consider the setup described in Ref. \[1\], which consists of two coupled qubits, each connected to its own heat bath.
The Hamiltonian of the qubits is given by
$$ \begin{aligned} H_{\text{S}} &= H_1 + H_2 + H_{12} , \quad\text{ where }\\
H_K &= \frac{\epsilon}{2} \bigl(\sigma_z^K + 1\bigr) \quad  (K=1,2) \quad\text{ and }\quad H_{12} = J_{12} \bigl( \sigma_+^1 \sigma_-^2 + \sigma_-^1 \sigma_+^2 \bigr) . \end{aligned} $$
Here, $\sigma^K_{x,y,z,\pm}$ denotes the usual Pauli matrices for the K-th qubit, $\epsilon$ is the eigenfrequency of the qubits and $J_{12}$ the coupling constant.

Each qubit is coupled to its own bath; therefore, the total Hamiltonian is
$$ H_{\text{tot}} = H_{\text{S}} + \sum_{K=1,2} \bigl( H_{\text{B}}^K + Q_K \otimes X_{\text{B}}^K \bigr) , $$
where $H_{\text{B}}^K$ is the free Hamiltonian of the K-th bath and $X_{\text{B}}^K$ its coupling operator, and $Q_K = \sigma_x^K$ are the system coupling operators.
We assume that the bath spectral densities are given by Drude distributions
$$ J_K(\omega) = \frac{2 \lambda_K \gamma_K \omega}{\omega^2 + \gamma_K^2} , $$
where $\lambda_K$ is the free coupling strength and $\gamma_K$ the cutoff frequency.

We begin by defining the system and bath parameters.
We use the parameter values from Fig. 3(a) of Ref. \[1\].
Note that we set $\hbar$ and $k_B$ to one and we will measure all frequencies and energies in units of $\epsilon$.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \[1\] Kato and Tanimura, [J. Chem. Phys. **143**, 064107](https://doi.org/10.1063/1.4928192) (2015).

```{code-cell} ipython3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import qutip as qt
from qutip.nonmarkov.heom import HEOMSolver, DrudeLorentzPadeBath, BathExponent

from ipywidgets import IntProgress
from IPython.display import display
```

```{code-cell} ipython3
# Qubit parameters
epsilon = 1

# System operators
H1   = epsilon / 2 * qt.tensor(qt.sigmaz() + qt.identity(2), qt.identity(2))
H2   = epsilon / 2 * qt.tensor(qt.identity(2), qt.sigmaz() + qt.identity(2))
H12  = lambda J12 : J12 * (qt.tensor(qt.sigmap(), qt.sigmam()) + qt.tensor(qt.sigmam(), qt.sigmap()))
Hsys = lambda J12 : H1 + H2 + H12(J12)

# Cutoff frequencies
gamma1 = 2
gamma2 = 2

# Temperatures
Tbar = 2
Delta_T = 0.01 * Tbar
T1 = Tbar + Delta_T
T2 = Tbar - Delta_T

# Coupling operators
Q1 = qt.tensor(qt.sigmax(), qt.identity(2))
Q2 = qt.tensor(qt.identity(2), qt.sigmax())
```

### Heat currents

Following Ref. \[2\], we consider two possible definitions of the heat currents from the qubits into the baths.
The so-called bath heat currents are $j_{\text{B}}^K = \partial_t \langle H_{\text{B}}^K \rangle$ and the system heat currents are $j_{\text{S}}^K = \mathrm i\, \langle [H_{\text{S}}, Q_K] X_{\text{B}}^K \rangle$.
As shown in Ref. \[2\], they can be expressed in terms of the HEOM ADOs as follows:
$$ \begin{aligned} \mbox{} \\
    j_{\text{B}}^K &= \!\!\sum_{\substack{\mathbf n\\ \text{Level 1}\\ \text{Bath $K$}}}\!\! \nu[\mathbf n] \operatorname{tr}\bigl[ Q_K \rho_{\mathbf n} \bigr] - 2 C_I^K(0) \operatorname{tr}\bigl[ Q_k^2 \rho \bigr] + \Gamma_{\text{T}}^K \operatorname{tr}\bigl[ [[H_{\text{S}}, Q_K], Q_K]\, \rho \bigr] , \\[.5em]
    j_{\text{S}}^K &= \mathrm i\!\! \sum_{\substack{\mathbf n\\ \text{Level 1}\\ \text{Bath $k$}}}\!\! \operatorname{tr}\bigl[ [H_{\text{S}}, Q_K]\, \rho_{\mathbf n} \bigr] + \Gamma_{\text{T}}^K \operatorname{tr}\bigl[ [[H_{\text{S}}, Q_K], Q_K]\, \rho \bigr] . \\ \mbox{}
\end{aligned} $$
The sums run over all level-$1$ multi-indices $\mathbf n$ with one excitation corresponding to the K-th bath, $\nu[\mathbf n]$ is the corresponding (negative) exponent of the bath auto-correlation function $C^K(t)$, and $\Gamma_{\text{T}}^K$ is the Ishizaki-Tanimura terminator (i.e., a correction term accounting for the error introduced by approximating the correlation function with a finite sum of exponential terms).
In the expression for the bath heat currents, we left out terms involving $[Q_1, Q_2]$, which is zero in this example.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \[2\] Kato and Tanimura, [J. Chem. Phys. **145**, 224105](https://doi.org/10.1063/1.4971370) (2016).

+++

In QuTiP, these currents can be conveniently calculated as follows:

```{code-cell} ipython3
def bath_heat_current(bath_tag, ado_state, hamiltonian, coupling_op, delta=0):
    """
    Bath heat current from the system into the heat bath with the given tag.
    
    Parameters
    ----------
    bath_tag : str, tuple or any other object
        Tag of the heat bath corresponding to the current of interest.
    
    ado_state : HierarchyADOsState
        Current state of the system and the environment (encoded in the ADOs).
    
    hamiltonian : Qobj
        System Hamiltonian at the current time.
    
    coupling_op : Qobj
        System coupling operator at the current time.
    
    delta : float
        The prefactor of the \delta(t) term in the correlation function (the Ishizaki-Tanimura terminator).
    """
    l1_labels = ado_state.filter(level=1, tags=[bath_tag])
    a_op = 1j * (hamiltonian * coupling_op - coupling_op * hamiltonian)

    result = 0
    cI0 = 0 # imaginary part of bath auto-correlation function (t=0)
    for label in l1_labels:
        [exp] = ado_state.exps(label)
        result += exp.vk * (coupling_op * ado_state.extract(label)).tr()

        if exp.type == BathExponent.types['I']:
            cI0 += exp.ck
        elif exp.type == BathExponent.types['RI']:
            cI0 += exp.ck2

    result -= 2 * cI0 * (coupling_op * coupling_op * ado_state.rho).tr()
    if delta != 0:
        result -= 1j * delta * ((a_op * coupling_op - coupling_op * a_op) * ado_state.rho).tr()
    return result

def system_heat_current(bath_tag, ado_state, hamiltonian, coupling_op, delta=0):
    """
    System heat current from the system into the heat bath with the given tag.
    
    Parameters
    ----------
    bath_tag : str, tuple or any other object
        Tag of the heat bath corresponding to the current of interest.
    
    ado_state : HierarchyADOsState
        Current state of the system and the environment (encoded in the ADOs).
    
    hamiltonian : Qobj
        System Hamiltonian at the current time.
    
    coupling_op : Qobj
        System coupling operator at the current time.
    
    delta : float
        The prefactor of the \delta(t) term in the correlation function (the Ishizaki-Tanimura terminator).
    """
    l1_labels = ado_state.filter(level=1, tags=[bath_tag])
    a_op = 1j * (hamiltonian * coupling_op - coupling_op * hamiltonian)

    result = 0
    for label in l1_labels:
        result += (a_op * ado_state.extract(label)).tr()

    if delta != 0:
        result -= 1j * delta * ((a_op * coupling_op - coupling_op * a_op) * ado_state.rho).tr()
    return result
```

Note that at long times, we expect $j_{\text{B}}^1 = -j_{\text{B}}^2$ and $j_{\text{S}}^1 = -j_{\text{S}}^2$ due to energy conservation. At long times, we also expect $j_{\text{B}}^1 = j_{\text{S}}^1$ and $j_{\text{B}}^2 = j_{\text{S}}^2$ since the coupling operators commute, $[Q_1, Q_2] = 0$. Hence, all four currents should agree in the long-time limit (up to a sign). This long-time value is what was analyzed in Ref. \[2\].

+++

### Simulations

+++

For our simulations, we will represent the bath spectral densities using the first term of their Padé decompositions, and we will use $7$ levels of the HEOM hierarchy.

```{code-cell} ipython3
Nk = 1
NC = 7
options = qt.Options(nsteps=1500, store_states=False, atol=1e-12, rtol=1e-12)
```

##### Time Evolution

We fix $J_{12} = 0.1 \epsilon$ (as in Fig. 3(a-ii) of Ref. \[2\]) and choose the fixed coupling strength $\lambda_1 = \lambda_2 = J_{12}\, /\, (2\epsilon)$ (corresponding to $\bar\zeta = 1$ in Ref. \[2\]).
Using these values, we will study the time evolution of the system state and the heat currents.

```{code-cell} ipython3
# fix qubit-qubit and qubit-bath coupling strengths
J12 = 0.1
lambda1 = J12 / 2
lambda2 = J12 / 2
# choose arbitrary initial state
rho0 = qt.tensor(qt.identity(2), qt.identity(2)) / 4
# simulation time span
tlist = np.linspace(0, 50, 250)
```

```{code-cell} ipython3
bath1 = DrudeLorentzPadeBath(Q1, lambda1, gamma1, T1, Nk, tag='bath 1')
bath2 = DrudeLorentzPadeBath(Q2, lambda2, gamma2, T2, Nk, tag='bath 2')

b1delta, b1term = bath1.terminator()
b2delta, b2term = bath2.terminator()
solver = HEOMSolver(qt.liouvillian(Hsys(J12)) + b1term + b2term,
                    [bath1, bath2], max_depth=NC, options=options)

result = solver.run(rho0, tlist, e_ops=[qt.tensor(qt.sigmaz(), qt.identity(2)),
                                        lambda t, ado: bath_heat_current('bath 1', ado, Hsys(J12), Q1, b1delta),
                                        lambda t, ado: bath_heat_current('bath 2', ado, Hsys(J12), Q2, b2delta),
                                        lambda t, ado: system_heat_current('bath 1', ado, Hsys(J12), Q1, b1delta),
                                        lambda t, ado: system_heat_current('bath 2', ado, Hsys(J12), Q2, b2delta)])
```

We first plot $\langle \sigma_z^1 \rangle$ to see the time evolution of the system state:

```{code-cell} ipython3
fig, axes = plt.subplots(figsize=(8,8))
axes.plot(tlist, result.expect[0], 'r', linewidth=2)
axes.set_xlabel('t', fontsize=28)
axes.set_ylabel(r"$\langle \sigma_z^1 \rangle$", fontsize=28)
pass
```

We find a rather quick thermalization of the system state. For the heat currents, however, it takes a somewhat longer time until they converge to their long-time values:

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

ax1.plot(tlist, -np.real(result.expect[1]), color='darkorange', label='BHC (bath 1 -> system)')
ax1.plot(tlist, np.real(result.expect[2]), '--', color='darkorange', label='BHC (system -> bath 2)')
ax1.plot(tlist, -np.real(result.expect[3]), color='dodgerblue', label='SHC (bath 1 -> system)')
ax1.plot(tlist, np.real(result.expect[4]), '--', color='dodgerblue', label='SHC (system -> bath 2)')

ax1.set_xlabel('t', fontsize=28)
ax1.set_ylabel('j', fontsize=28)
ax1.set_ylim((-0.05, 0.05))
ax1.legend(loc=0, fontsize=12)

ax2.plot(tlist, -np.real(result.expect[1]), color='darkorange', label='BHC (bath 1 -> system)')
ax2.plot(tlist, np.real(result.expect[2]), '--', color='darkorange', label='BHC (system -> bath 2)')
ax2.plot(tlist, -np.real(result.expect[3]), color='dodgerblue', label='SHC (bath 1 -> system)')
ax2.plot(tlist, np.real(result.expect[4]), '--', color='dodgerblue', label='SHC (system -> bath 2)')

ax2.set_xlabel('t', fontsize=28)
ax2.set_xlim((20, 50))
ax2.set_ylim((0, 0.0002))
ax2.legend(loc=0, fontsize=12)

pass
```

##### Steady-state currents

Here, we try to reproduce the HEOM curves in Fig. 3(a) of Ref. \[1\] by varying the coupling strength and finding the steady state for each coupling strength.

```{code-cell} ipython3
def heat_currents(J12, zeta_bar):
    bath1 = DrudeLorentzPadeBath(Q1, zeta_bar * J12 / 2, gamma1, T1, Nk, tag='bath 1')
    bath2 = DrudeLorentzPadeBath(Q2, zeta_bar * J12 / 2, gamma2, T2, Nk, tag='bath 2')
    b1delta, b1term = bath1.terminator()
    b2delta, b2term = bath2.terminator()
    
    solver = HEOMSolver(qt.liouvillian(Hsys(J12)) + b1term + b2term,
                        [bath1, bath2], max_depth=NC, options=options)
    
    _, steady_ados = solver.steady_state()
    return bath_heat_current('bath 1', steady_ados, Hsys(J12), Q1, b1delta), \
           bath_heat_current('bath 2', steady_ados, Hsys(J12), Q2, b2delta), \
           system_heat_current('bath 1', steady_ados, Hsys(J12), Q1, b1delta), \
           system_heat_current('bath 2', steady_ados, Hsys(J12), Q2, b2delta)
```

```{code-cell} ipython3
# Define number of points to use for final plot
plot_points = 100
```

```{code-cell} ipython3
progress = IntProgress(min=0, max=(3*plot_points))
display(progress)

zeta_bars = []
j1s = [] # J12 = 0.01
j2s = [] # J12 = 0.1
j3s = [] # J12 = 0.5

# --- J12 = 0.01 ---
NC = 7
# xrange chosen so that 20 is maximum, centered around 1 on a log scale
for zb in np.logspace(-np.log(20), np.log(20), plot_points, base=np.e):
    j1, _, _, _ = heat_currents(0.01, zb) # the four currents are identical in the steady state
    zeta_bars.append(zb)
    j1s.append(j1)
    
    progress.value += 1

# --- J12 = 0.1 ---
for zb in zeta_bars:
    # higher HEOM cut-off is necessary for large coupling strength
    if zb < 10:
        NC = 7
    else:
        NC = 12

    j2, _, _, _ = heat_currents(0.1, zb)
    j2s.append(j2)
    progress.value += 1

# --- J12 = 0.5 ---
for zb in zeta_bars:
    if zb < 5:
        NC = 7
    elif zb < 10:
        NC = 15
    else:
        NC = 20

    j3, _, _, _ = heat_currents(0.5, zb)
    j3s.append(j3)
    progress.value += 1

progress.close()
```

```{code-cell} ipython3
np.save('data/qhb_zb.npy', zeta_bars)
np.save('data/qhb_j1.npy', j1s)
np.save('data/qhb_j2.npy', j2s)
np.save('data/qhb_j3.npy', j3s)
```

### Create Plot

```{code-cell} ipython3
zeta_bars = np.load('data/qhb_zb.npy')
j1s = np.load('data/qhb_j1.npy')
j2s = np.load('data/qhb_j2.npy')
j3s = np.load('data/qhb_j3.npy')
```

```{code-cell} ipython3
matplotlib.rcParams['figure.figsize'] = (7, 5)
matplotlib.rcParams['axes.titlesize'] = 25
matplotlib.rcParams['axes.labelsize'] = 30
matplotlib.rcParams['xtick.labelsize'] = 28
matplotlib.rcParams['ytick.labelsize'] = 28
matplotlib.rcParams['legend.fontsize'] = 28
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['lines.markersize'] = 5
matplotlib.rcParams['font.family'] = 'STIXgeneral' 
matplotlib.rcParams['mathtext.fontset'] =  'stix'
matplotlib.rcParams["font.serif"] = "STIX"
matplotlib.rcParams['text.usetex'] = False
```

```{code-cell} ipython3
fig, axes = plt.subplots(figsize=(12,7))
axes.plot(zeta_bars, -1000 * 100 * np.real(j1s), 'b',   linewidth=2, label=r"$J_{12} = 0.01\, \epsilon$")
axes.plot(zeta_bars, -1000 * 10  * np.real(j2s), 'r--',  linewidth=2, label=r"$J_{12} = 0.1\, \epsilon$")
axes.plot(zeta_bars, -1000 * 2   * np.real(j3s), 'g-.', linewidth=2, label=r"$J_{12} = 0.5\, \epsilon$")

axes.set_xscale('log')
axes.set_xlabel(r"$\bar\zeta$", fontsize=30)
axes.set_xlim((zeta_bars[0], zeta_bars[-1]))

axes.set_ylabel(r"$j_{\mathrm{ss}}\; /\; (\epsilon J_{12}) \times 10^3$", fontsize=30)
axes.set_ylim((0, 2))

axes.legend(loc=0)
#fig.savefig("figures/figHeat.pdf")
pass
```

```{code-cell} ipython3
from qutip.ipynbtools import version_table

version_table()
```

```{code-cell} ipython3

```
