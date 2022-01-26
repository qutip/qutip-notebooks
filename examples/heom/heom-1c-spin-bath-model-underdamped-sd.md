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

# Example 1c: Spin-Bath model (Underdamped Case)
### Introduction

+++

The HEOM method solves the dynamics and steady state of a system and its environment, the latter of which is encoded in a set of auxiliary density matrices.

In this example we show the evolution of a single two-level system in contact with a single Bosonic environment.  The properties of the system are encoded in Hamiltonian, and a coupling operator which describes how it is coupled to the environment.

The Bosonic environment is implicitly assumed to obey a particular Hamiltonian (see paper), the parameters of which are encoded in the spectral density, and subsequently the free-bath correlation functions.

In the example below we show how to model the underdamped Brownian motion Spectral Density.

Note that in the following, we set $\hbar = k_\mathrm{B} = 1$.

###  Brownian motion (underdamped) spectral density
The underdamped spectral density is:

$$J_U = \frac{\alpha^2 \Gamma \omega}{(\omega_c^2 - \omega^2)^2 + \Gamma^2 \omega^2)}.$$

Here $\alpha$  scales the coupling strength, $\Gamma$ is the cut-off frequency, and $\omega_c$ defines a resonance frequency.  With the HEOM we must use an exponential decomposition:

The Matsubara decomposition of this spectral density is, in real and imaginary parts:



\begin{equation*}
    c_k^R = \begin{cases}
               \alpha^2 \coth(\beta( \Omega + i\Gamma/2)/2)/4\Omega & k = 0\\
               \alpha^2 \coth(\beta( \Omega - i\Gamma/2)/2)/4\Omega & k = 0\\
              -2\alpha^2\Gamma/\beta \frac{\epsilon_k }{((\Omega + i\Gamma/2)^2 + \epsilon_k^2)(\Omega - i\Gamma/2)^2 + \epsilon_k^2)}      & k \geq 1\\
           \end{cases}
\end{equation*}

\begin{equation*}
    \nu_k^R = \begin{cases}
               -i\Omega  + \Gamma/2, i\Omega  +\Gamma/2,             & k = 0\\
               {2 \pi k} / {\beta }  & k \geq 1\\
           \end{cases}
\end{equation*}




\begin{equation*}
    c_k^I = \begin{cases}
               i\alpha^2 /4\Omega & k = 0\\
                -i\alpha^2 /4\Omega & k = 0\\
           \end{cases}
\end{equation*}

\begin{equation*}
    \nu_k^I = \begin{cases}
               i\Omega  + \Gamma/2, -i\Omega  + \Gamma/2,             & k = 0\\
           \end{cases}
\end{equation*}

Note that in the above, and the following, we set $\hbar = k_\mathrm{B} = 1$.

```{code-cell} ipython3
%pylab inline
```

```{code-cell} ipython3
import contextlib
import time

import numpy as np

from qutip import *
from qutip.nonmarkov.heom import HEOMSolver, HSolverDL, BosonicBath, UnderDampedBath
```

```{code-cell} ipython3
def cot(x):
    """ Vectorized cotangent of x. """
    return 1. / np.tan(x)
```

```{code-cell} ipython3
def coth(x):
    """ Vectorized hyperbolic cotangent of x. """
    return 1. / np.tanh(x)
```

```{code-cell} ipython3
def underdamped_matsubara_params(lam, gamma, T, nk):
    """ Calculation of the real and imaginary expansions of the underdamped correlation functions.
    """
    Om = np.sqrt(w0**2 - (gamma / 2)**2)
    Gamma = gamma / 2.
    beta = 1. / T

    ckAR = [
        (lam**2 / (4*Om)) * coth(beta * (Om + 1.0j * Gamma) / 2),
        (lam**2 / (4*Om)) * coth(beta * (Om - 1.0j * Gamma) / 2),
    ]
    ckAR.extend(
        (-2 * lam**2 * gamma / beta) * (2 * np.pi * k / beta) / 
        (((Om + 1.0j * Gamma)**2 + (2 * np.pi * k / beta)**2) *
         ((Om - 1.0j * Gamma)**2 + (2 * np.pi * k / beta)**2)) + 0.j
        for k in range(1, nk + 1)
    )
    vkAR = [
        -1.0j * Om + Gamma,
        1.0j * Om + Gamma,
    ]
    vkAR.extend(
        2 * np.pi * k * T + 0.j
        for k in range(1, nk + 1)
    )

    factor = 1. / 4

    ckAI = [
        -factor * lam**2 * 1.0j / Om,
        factor * lam**2 * 1.0j / Om,
    ]
    vkAI = [
        -(-1.0j * Om - Gamma),
        -(1.0j * Om - Gamma),
    ]
    
    return ckAR, vkAR, ckAI, vkAI
```

```{code-cell} ipython3
def plot_result_expectations(plots, axes=None):
    """ Plot the expectation values of operators as functions of time.
    
        Each plot in plots consists of (solver_result, measurement_operation, color, label).
    """
    if axes is None:
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
        fig_created = True
    else:
        fig = None
        fig_created = False

    # add kw arguments to each plot if missing
    plots = [p if len(p) == 5 else p + ({},) for p in plots]
    for result, m_op, color, label, kw in plots:
        exp = np.real(expect(result.states, m_op))
        kw.setdefault("linewidth", 2)
        axes.plot(result.times, exp, color, label=label, **kw)

    if fig_created:
        axes.legend(loc=0, fontsize=12)
        axes.set_xlabel("t", fontsize=28)

    return fig
```

```{code-cell} ipython3
@contextlib.contextmanager
def timer(label):
    """ Simple utility for timing functions:
    
        with timer("name"):
            ... code to time ...
    """
    start = time.time()
    yield
    end = time.time()
    print(f"{label}: {end - start}")
```

```{code-cell} ipython3
# Defining the system Hamiltonian
eps = .5     # Energy of the 2-level system.
Del = 1.0    # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del* sigmax()
```

```{code-cell} ipython3
# Initial state of the system.
rho0 = basis(2,0) * basis(2,0).dag()  
```

```{code-cell} ipython3
# System-bath coupling (underdamed spectral density)
Q = sigmaz() # coupling operator

# Bath properties:
gamma = .1 # cut off frequency
lam = .5 # coupling strength
w0 = 1. # resonance frequency
T = 1.
beta = 1./T

# HEOM parameters
NC = 10 # cut off parameter for the bath
Nk = 2 # number of exponents to retain in the Matsubara expansion of the correlation function

# Times to solve for
tlist = np.linspace(0, 50, 1000)
```

```{code-cell} ipython3
# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresonding to groundstate
P11p = basis(2,0) * basis(2,0).dag()
P22p = basis(2,1) * basis(2,1).dag()
# 1,2 element of density matrix  - corresonding to coherence
P12p = basis(2,0) * basis(2,1).dag()
```

### First let us look at what the underdamped spectral density looks like:

```{code-cell} ipython3
def plot_spectral_density():
    """ Plot the underdamped spectral density """
    w = np.linspace(0, 5, 1000)
    J = lam**2 * gamma * w / ((w0**2 - w**2)**2 + (gamma**2) * (w**2))
    
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
    axes.plot(w, J, 'r', linewidth=2)
    axes.set_xlabel(r'$\omega$', fontsize=28)
    axes.set_ylabel(r'J', fontsize=28)

plot_spectral_density()
```

The correlation functions are now very oscillatory, because of the Lorentzian peak in the spectral density.

+++

### So next, let us plot the correlation functions themselves:

```{code-cell} ipython3
def Mk(t, k, gamma, w0, beta):
    """ Calculate the Matsubara terms for a given t and k. """
    Om = np.sqrt(w0**2 - (gamma / 2)**2)
    Gamma = gamma / 2.
    ek = 2 * pi * k / beta
    
    return (
        (-2 * lam**2 * gamma / beta) * ek * exp(-ek * abs(t))
        / (((Om + 1.0j * Gamma)**2 + ek**2) * ((Om - 1.0j * Gamma)**2 + ek**2))
    )


def c(t, Nk, lam, gamma, w0, beta):
    """ Calculate the correlation function for a vector of times, t. """
    Om = np.sqrt(w0**2 - (gamma / 2)**2)
    Gamma = gamma / 2.
    
    Cr = (
        coth(beta * (Om + 1.0j * Gamma) / 2) * np.exp(1.0j * Om * t)
        + coth(beta * (Om - 1.0j * Gamma) / 2) * np.exp(-1.0j * Om * t)
    )

    Ci = np.exp(-1.0j * Om *t) - np.exp(1.0j * Om * t)
    
    return (
        (lam**2 / (4 * Om)) * np.exp(-Gamma * np.abs(t)) * (Cr + Ci) +
        np.sum([Mk(t, k, gamma=gamma, w0=w0, beta=beta) for k in range(1, Nk + 1)], 0)
    )


def plot_correlation_function():
    """ Plot the underdamped correlation function. """
    Nk = 3
    t = np.linspace(0, 20, 1000)
    corr = c(t, Nk=3, lam=lam, gamma=gamma, w0=w0, beta=beta)

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
    axes.plot(t, np.real(corr), '-', color="black", label="Re[C(t)]")
    axes.plot(t, np.imag(corr), '-', color="red", label="Im[C(t)]")
    axes.set_xlabel(r't', fontsize=28)
    axes.set_ylabel(r'C', fontsize=28)
    axes.legend(loc=0, fontsize=12)


plot_correlation_function()
```

It is useful to look at what the Matsubara contributions do to this spectral density. We see that they modify the real part around $t=0$:

```{code-cell} ipython3
def plot_matsubara_correlation_function_contributions():
    """ Plot the underdamped correlation function. """
    t = np.linspace(0, 20, 1000)

    M_Nk3 = np.sum([Mk(t, k, gamma=gamma, w0=w0, beta=beta) for k in range(1, 2 + 1)], 0)
    M_Nk5 = np.sum([Mk(t, k, gamma=gamma, w0=w0, beta=beta) for k in range(1, 100 + 1)], 0)
    
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
    axes.plot(t, np.real(M_Nk3), '-', color="black", label="Re[M(t)] Nk=2")
    axes.plot(t, np.real(M_Nk5), '--', color="red", label="Re[M(t)] Nk=100")
    axes.set_xlabel(r't', fontsize=28)
    axes.set_ylabel(r'M', fontsize=28)
    axes.legend(loc=0, fontsize=12)


plot_matsubara_correlation_function_contributions()
```

### Solving for the dynamics as a function of time:

+++

Next we calculate the exponents using the Matsubara decompositions. Here we split them into real and imaginary parts.

The HEOM code will optimize these, and reduce the number of exponents when real and imaginary parts have the same exponent. This is clearly the case for the first term in the vkAI and vkAR lists.

```{code-cell} ipython3
ckAR, vkAR, ckAI, vkAI = underdamped_matsubara_params(lam=lam, gamma=gamma, T=T, nk=Nk)
```

Having created the lists which specify the bath correlation functions, we create a `BosonicBath` from them and pass the bath to the `HEOMSolver` class.

The solver constructs the "right hand side" (RHS) determinining how the system and auxiliary density operators evolve in time. This can then be used to solve for dynamics or steady-state.

Below we create the bath and solver and then solve for the dynamics by calling `.run(rho0, tlist)`.

```{code-cell} ipython3
options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

with timer("RHS construction time"):
    bath = BosonicBath(Q, ckAR, vkAR, ckAI, vkAI)
    HEOMMats = HEOMSolver(Hsys, bath, NC, options=options)
    
with timer("ODE solver time"):
    resultMats = HEOMMats.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Mats"),
    (resultMats, P12p, 'r', "P12 Mats"),
]);
```

In practice, one would not perform this laborious expansion for the underdamped correlation function, because
QuTiP already has a class, `UnderDampedBath`, that can construct this bath for you. Nevertheless, knowing how
to perform this expansion will allow you to construct your own baths for other spectral densities.

Below we show how to use this built-in functionality:

```{code-cell} ipython3
# Compare to built-in under-damped bath:

with timer("RHS construction time"):
    bath = UnderDampedBath(Q, lam=lam, gamma=gamma, w0=w0, T=T, Nk=Nk)
    HEOM_udbath = HEOMSolver(Hsys, bath, NC, options=options)

with timer("ODE solver time"):
    result_udbath = HEOM_udbath.run(rho0, tlist) #normal  115
```

```{code-cell} ipython3
plot_result_expectations([
    (result_udbath, P11p, 'b', "P11 (UnderDampedBath)"),
    (result_udbath, P12p, 'r', "P12 (UnderDampedBath)"),
]);
```

### We can compare these results to those of the Bloch-Redfield solver in QuTiP:

```{code-cell} ipython3
UD = (
    f" 2* {lam}**2 * {gamma} / ( {w0}**4 * {beta}) if (w==0) else"
    f" 2* ({lam}**2 * {gamma} * w / (({w0}**2 - w**2)**2 + {gamma}**2 * w**2)) * ((1 / (exp(w * {beta}) - 1)) + 1)"
)

options = Options(nsteps=15000, store_states=True, rtol=1e-12, atol=1e-12)

with timer("ODE solver time"):
    resultBR = brmesolve(Hsys, rho0, tlist, a_ops=[[sigmaz(), UD]], options=options)
```

```{code-cell} ipython3
plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Mats"),
    (resultMats, P12p, 'r', "P12 Mats"),
    (resultBR, P11p, 'g--', "P11 Bloch Redfield"),
    (resultBR, P12p, 'g--', "P12 Bloch Redfield"),
]);
```

### Lastly, let us calculate the analytical steady-state result and compare all of the results:

+++

The thermal state of a reaction coordinate (treating the environment as a single damped mode) should, at high temperatures and small gamma, tell us the steady-state:

```{code-cell} ipython3
dot_energy, dot_state = Hsys.eigenstates()
deltaE = dot_energy[1] - dot_energy[0]

gamma2 = gamma
wa = w0  # reaction coordinate frequency
g = lam / np.sqrt(2 * wa)  # coupling

NRC = 10

Hsys_exp = tensor(qeye(NRC), Hsys)
Q_exp = tensor(qeye(NRC), Q)
a = tensor(destroy(NRC), qeye(2))

H0 = wa * a.dag() * a + Hsys_exp
# interaction
H1 = (g * (a.dag() + a) * Q_exp)

H = H0 + H1

energies, states = H.eigenstates()
rhoss = 0 * states[0] * states[0].dag()
for kk, energ in enumerate(energies):
    rhoss += (states[kk] * states[kk].dag() * exp(-beta * energies[kk])) 
rhoss = rhoss / rhoss.norm()

P12RC = tensor(qeye(NRC), basis(2,0) * basis(2,1).dag())
P12RC = expect(rhoss, P12RC)

P11RC = tensor(qeye(NRC), basis(2,0) * basis(2,0).dag())
P11RC = expect(rhoss, P11RC)
```

```{code-cell} ipython3
# XXX: Decide what to do with this cell

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
# XXX: Decide what to do with this cell

fig, axes = plt.subplots(1, 1, sharex=True, figsize=(12, 7))

plt.yticks([P11RC, 0.6, 1.0], [0.38, 0.6, 1])

plot_result_expectations([
    (resultBR, P11p, 'y-.', "Bloch-Redfield"),
    (resultMats, P11p, 'b', "Matsubara $N_k=3$"),
], axes=axes)
axes.plot(tlist, [P11RC for t in tlist], color='black', linestyle="-.",linewidth=2, label="Thermal state")

axes.set_xlabel(r'$t \Delta$',fontsize=30)
axes.set_ylabel(r'$\rho_{11}$',fontsize=30)

axes.locator_params(axis='y', nbins=4)
axes.locator_params(axis='x', nbins=4)

axes.legend(loc=0)

fig.tight_layout()
# fig.savefig("figures/fig3.pdf")
```

```{code-cell} ipython3
from qutip.ipynbtools import version_table

version_table()
```

```{code-cell} ipython3

```
