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

# Example 1a: Spin-Bath model (basic)

### Introduction

+++

The HEOM method solves the dynamics and steady state of a system and its environment, the latter of which is encoded in a set of auxiliary density matrices.

In this example we show the evolution of a single two-level system in contact with a single Bosonic environment.  The properties of the system are encoded in Hamiltonian, and a coupling operator which describes how it is coupled to the environment.

The Bosonic environment is implicitly assumed to obey a particular Hamiltonian (see paper), the parameters of which are encoded in the spectral density, and subsequently the free-bath correlation functions.

In the example below we show how to model the overdamped Drude-Lorentz Spectral Density, commonly used with the HEOM. We show how to do this using the Matsubara, Pade and fitting decompositions, and compare their convergence.  

### Drude-Lorentz (overdamped) spectral density
The Drude-Lorentz spectral density is:

$$J_D(\omega)= \frac{2\omega\lambda\gamma}{{\gamma}^2 + \omega^2}$$

where $\lambda$ scales the coupling strength, and $\gamma$ is the cut-off frequency.  We use the convention,
\begin{equation*}
C(t) = \int_0^{\infty} d\omega \frac{J_D(\omega)}{\pi}[\coth(\beta\omega) \cos(\omega \tau) - i \sin(\omega \tau)]
\end{equation*}

With the HEOM we must use an exponential decomposition:

\begin{equation*}
C(t)=\sum_{k=0}^{k=\infty} c_k e^{-\nu_k t}
\end{equation*}

As an example, the Matsubara decomposition of the Drude-Lorentz spectral density is given by:

\begin{equation*}
    \nu_k = \begin{cases}
               \gamma               & k = 0\\
               {2 \pi k} / {\beta }  & k \geq 1\\
           \end{cases}
\end{equation*}

\begin{equation*}
    c_k = \begin{cases}
               \lambda \gamma (\cot(\beta \gamma / 2) - i)             & k = 0\\
               4 \lambda \gamma \nu_k / \{(nu_k^2 - \gamma^2)\beta \}    & k \geq 1\\
           \end{cases}
\end{equation*}

Note that in the above, and the following, we set $\hbar = k_\mathrm{B} = 1$.

```{code-cell} ipython3
%pylab inline
%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
import contextlib
import time

import numpy as np

from qutip import *
from qutip.nonmarkov.heom import HEOMSolver, HSolverDL, BosonicBath, DrudeLorentzBath, DrudeLorentzPadeBath
```

```{code-cell} ipython3
def cot(x):
    """ Vectorized cotangent of x. """
    return 1. / np.tan(x)
```

```{code-cell} ipython3
def dl_matsubara_params(lam, gamma, T, nk):
    """ Calculation of the real and imaginary expansions of the Drude-Lorenz correlation functions.
    """
    ckAR = [lam * gamma * cot(gamma / (2 * T))]
    ckAR.extend(
        4 * lam * gamma * T *  2 * np.pi * k * T / ((2 * np.pi * k * T)**2 - gamma**2)
        for k in range(1, nk + 1)
    )
    vkAR = [gamma]
    vkAR.extend(2 * np.pi * k * T for k in range(1, nk + 1))

    ckAI = [lam * gamma * (-1.0)]
    vkAI = [gamma]
    
    return ckAR, vkAR, ckAI, vkAI
```

```{code-cell} ipython3
def dl_corr_approx(t, nk):
    """ Drude-Lorenz correlation function approximation.
    
        Approximates the correlation function at each time t to nk exponents.
    """
    c = lam * gamma * (-1.0j + cot(gamma / (2 * T))) * np.exp(-gamma * t)
    for k in range(1, nk):
        vk = 2 * np.pi * k * T
        c += (4 * lam * gamma * T * vk / (vk**2 - gamma**2))  * np.exp(-vk * t)
    return c
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
# System-bath coupling (Drude-Lorentz spectral density)
Q = sigmaz() # coupling operator

# Bath properties:
gamma = .5 # cut off frequency
lam = .1 # coupling strength
T = 0.5
beta = 1./T

# HEOM parameters
NC = 5 # cut off parameter for the bath
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

### First of all, it is useful to look at the spectral density, to understand its magnitude and width, relative to the system properties:

```{code-cell} ipython3
def plot_spectral_density():
    """ Plot the Drude-Lorentz spectral density """
    w = np.linspace(0, 5, 1000)
    J = w * 2 * lam * gamma / (gamma**2 + w**2)

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
    axes.plot(w, J, 'r', linewidth=2)
    axes.set_xlabel(r'$\omega$', fontsize=28)
    axes.set_ylabel(r'J', fontsize=28)

plot_spectral_density()
```

Next we calculate the exponents using the Matsubara decompositions. Here we split them into real and imaginary parts.

The HEOM code will optimize these, and reduce the number of exponents when real and imaginary parts have the same
exponent. This is clearly the case for the first term in the vkAI and vkAR lists.

```{code-cell} ipython3
ckAR, vkAR, ckAI, vkAI = dl_matsubara_params(nk=Nk, lam=lam, gamma=gamma, T=T)
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

In practice, one would not perform this laborious expansion for the Drude-Lorentz correlation function, because QuTiP already has a class, `DrudeLorentzBath`, that can construct this bath for you. Nevertheless, knowing how to perform this expansion will allow you to construct your own baths for other spectral densities.

Below we show how to use this built-in functionality:

```{code-cell} ipython3
# Compare to built-in Drude-Lorentz bath:

with timer("RHS construction time"):
    bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
    HEOM_dlbath = HEOMSolver(Hsys, bath, NC, options=options)

with timer("ODE solver time"):
    result_dlbath = HEOM_dlbath.run(rho0, tlist) #normal  115
```

```{code-cell} ipython3
plot_result_expectations([
    (result_dlbath, P11p, 'b', "P11 (DrudeLorentzBath)"),
    (result_dlbath, P12p, 'r', "P12 (DrudeLorentzBath)"),
]);
```

We also provide a legacy class, `HSolverDL`, which calculates the Drude-Lorentz correlation functions automatically, to be backwards compatible with the previous HEOM solver in QuTiP:

```{code-cell} ipython3
# Compare to legacy class:

# The legacy class performs the above collation of co-oefficients automatically, based upon the
# parameters for the Drude-Lorentz spectral density.

with timer("RHS construction time"):
    HEOMlegacy = HSolverDL(Hsys, Q, lam, T, NC, Nk, gamma, options=options)

with timer("ODE solver time"):
    resultLegacy = HEOMlegacy.run(rho0, tlist) #normal  115
```

```{code-cell} ipython3
plot_result_expectations([
    (resultLegacy, P11p, 'b', "P11 Legacy"),
    (resultLegacy, P12p, 'r', "P12 Legacy"),
]);
```

## Ishizaki-Tanimura Terminator

To speed up convergence (in terms of the number of exponents kept in the Matsubara decomposition), We can treat the $Re[C(t=0)]$ component as a delta-function distribution, and include it as Lindblad correction. This is sometimes known as the Ishizaki-Tanimura Terminator.

In more detail, given

\begin{equation*}
C(t)=\sum_{k=0}^{\infty} c_k e^{-\nu_k t}
\end{equation*}
since $\nu_k=\frac{2 \pi k}{\beta }$, if $1/\nu_k$ is much much smaller than other important time-scales, we can approximate,  $ e^{-\nu_k t} \approx \delta(t)/\nu_k$, and $C(t)=\sum_{k=N_k}^{\infty} \frac{c_k}{\nu_k} \delta(t)$

It is convenient to calculate the whole sum $C(t)=\sum_{k=0}^{\infty} \frac{c_k}{\nu_k} =  2 \lambda / (\beta \gamma) - i\lambda $, and subtract off the contribution from the finite number of Matsubara terms that are kept in the hierarchy, and treat the residual as a Lindblad.

This is clearer if we plot the correlation function with a large number of Matsubara terms:

```{code-cell} ipython3
def plot_correlation_expansion_divergence(): 
    """ We plot the correlation function with a large number of Matsubara terms to show that
        the real part is slowly diverging at t = 0.
    """
    t = linspace(0, 2, 100)

    # correlation coefficients with 15k and 2 terms
    corr_15k = dl_corr_approx(t, 15_000)
    corr_2 = dl_corr_approx(t, 2)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(t, np.real(corr_2), color="b", linewidth=3, label= r"Mats = 2 real")
    ax1.plot(t, np.imag(corr_2), color="r", linewidth=3, label= r"Mats = 2 imag")
    ax1.plot(t, np.real(corr_15k), "b--", linewidth=3, label= r"Mats = 15000 real")
    ax1.plot(t, np.imag(corr_15k), "r--", linewidth=3, label= r"Mats = 15000 imag")

    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$C$")
    ax1.legend()
    
plot_correlation_expansion_divergence()
```

Let us evaluate the result including this Ishizaki-Tanimura terminator:

```{code-cell} ipython3
# Run HEOM solver and include the Ishizaki-Tanimura terminator

# Notes:
#
# * when using the built-in DrudeLorentzBath, the terminator (L_bnd) is available
#   from bath.terminator().
#
# * in the legacy HSolverDL function the terminator is included automatically if
#   the parameter bnd_cut_approx=True is used.

op = -2*spre(Q)*spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q)

approx_factr = ((2 * lam / (beta * gamma)) - 1j*lam) 

approx_factr -=  lam * gamma * (-1.0j + cot(gamma / (2 * T)))/gamma
for k in range(1,Nk+1):
    vk = 2 * np.pi * k * T
    
    approx_factr -= ((4 * lam * gamma * T * vk / (vk**2 - gamma**2))/ vk)

L_bnd = -approx_factr*op

Ltot = -1.0j*(spre(Hsys)-spost(Hsys)) + L_bnd
Ltot = liouvillian(Hsys) + L_bnd

options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

with timer("RHS construction time"):
    bath = BosonicBath(Q, ckAR, vkAR, ckAI, vkAI)
    HEOMMatsT = HEOMSolver(Ltot, bath, NC, options=options)

with timer("ODE solver time"):
    resultMatsT = HEOMMatsT.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations([
    (resultMatsT, P11p, 'b', "P11 Mats + Term"),
    (resultMatsT, P12p, 'r', "P12 Mats + Term"),
]);
```

Or using the built-in Drude-Lorentz bath we can write simply:

```{code-cell} ipython3
options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

with timer("RHS construction time"):
    bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
    _, terminator = bath.terminator()
    Ltot = liouvillian(Hsys) + terminator
    HEOM_dlbath_T = HEOMSolver(Ltot, bath, NC, options=options)

with timer("ODE solver time"):
    result_dlbath_T = HEOM_dlbath_T.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations([
    (result_dlbath_T, P11p, 'b', "P11 Mats (DrudeLorentzBath + Term)"),
    (result_dlbath_T, P12p, 'r', "P12 Mats (DrudeLorentzBath + Term)"),
]);
```

We can compare the solution obtained from the QuTiP Bloch-Redfield solver:

```{code-cell} ipython3
DL = (
    f"2*pi* 2.0 * {lam} / (pi * {gamma} * {beta}) if (w == 0) else "
    f"2*pi*(2.0*{lam}*{gamma} *w /(pi*(w**2+{gamma}**2))) * ((1/(exp((w) * {beta})-1))+1)"
)
options = Options(nsteps=15000, store_states=True,rtol=1e-12,atol=1e-12)

with timer("ODE solver time"):
    resultBR = brmesolve(Hsys, rho0, tlist, a_ops=[[sigmaz(), DL]], options=options)
```

```{code-cell} ipython3
plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Mats"),
    (resultMats, P12p, 'r', "P12 Mats"),
    (resultMatsT, P11p, 'b--', "P11 Mats + Term"),
    (resultMatsT, P12p, 'r--', "P12 Mats + Term"),
    (resultBR, P11p, 'g--', "P11 Bloch Redfield"),
    (resultBR, P12p, 'g--', "P12 Bloch Redfield"),
]);
```

```{code-cell} ipython3
# XXX: We should probably remove this at some point and make a separate notebook(s) for
# generating plots for the paper.

fig = plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Mats"),
    (resultMats, P12p, 'r', "P12 Mats"),
]);

fig.savefig("figures/docsfig1.png")
```

# Padé decomposition

+++

The Matsubara decomposition is not the only option.  We can also use the faster-converging Pade decomposition.

```{code-cell} ipython3
def deltafun(j,k):
    if j == k: 
        return 1.
    else:
        return 0.

def pade_eps(lmax):
    Alpha = np.zeros((2 * lmax, 2 * lmax))
    for j in range(2 * lmax):
        for k in range(2 * lmax):
            # fermionic (see other example notebooks):
            # Alpha[j][k] = (deltafun(j,k+1)+deltafun(j,k-1))/sqrt((2*(j+1)-1)*(2*(k+1)-1))
            # bosonic:
            Alpha[j][k] = (deltafun(j,k+1)+deltafun(j,k-1))/sqrt((2*(j+1)+1)*(2*(k+1)+1))
        
    eigvalsA = eigvalsh(Alpha)
    eps = [-2/val for val in eigvalsA[0: lmax]]
    return eps

def pade_chi(lmax):
    AlphaP = np.zeros((2 * lmax - 1, 2 * lmax - 1))
    for j in range(2 * lmax - 1):
        for k in range(2 * lmax - 1):
            # fermionic:
            # AlphaP[j][k] = (deltafun(j,k+1)+deltafun(j,k-1))/sqrt((2*(j+1)+1)*(2*(k+1)+1))
            # bosonic [this is +3 because +1 (bose) + 2*(+1)(from bm+1)]:
            AlphaP[j][k] = (deltafun(j,k+1)+deltafun(j,k-1))/sqrt((2*(j+1)+3)*(2*(k+1)+3))

    eigvalsAP = eigvalsh(AlphaP)
    chi = [-2/val for val in eigvalsAP[0: lmax - 1]]
    return chi

def pade_kappa_epsilon(lmax):
    eps = pade_eps(lmax)
    chi = pade_chi(lmax)
    
    kappa = [0]
    prefactor = 0.5 * lmax * (2 * (lmax + 1) + 1)

    for j in range(lmax):
        term = prefactor
        for k in range(lmax - 1):
            term *= (chi[k]**2 - eps[j]**2) / (eps[k]**2 - eps[j]**2 + deltafun(j, k))

        for k in range(lmax-1, lmax):
            term /= (eps[k]**2 - eps[j]**2 + deltafun(j, k))

        kappa.append(term)
        
    epsilon = [0] + eps

    return kappa, epsilon

def pade_corr(tlist, lmax):
    kappa, epsilon = pade_kappa_epsilon(lmax)
    
    eta_list = [lam * gamma * (cot(gamma * beta / 2.0) - 1.0j)]
    gamma_list = [gamma]
    
    if lmax > 0:
        for l in range(1, lmax + 1):
            eta_list.append((kappa[l]/beta)*4*lam*gamma*(epsilon[l]/beta)/((epsilon[l]**2/beta**2)-gamma**2))
            gamma_list.append(epsilon[l]/beta)
            
    c_tot = []
    for t in tlist:
        c_tot.append(sum([eta_list[l]*exp(-gamma_list[l]*t) for l in range(lmax+1)]))
    return c_tot, eta_list, gamma_list


tlist_corr = linspace(0, 2, 100)
cppLP, etapLP, gampLP = pade_corr(tlist_corr, 2)
corr_15k = dl_corr_approx(tlist_corr, 15_000)
corr_2k = dl_corr_approx(tlist_corr, 2)

fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot(tlist_corr, real(cppLP), color="b", linewidth=3, label= r"real pade 2 terms")
ax1.plot(tlist_corr, real(corr_15k), "r--", linewidth=3, label= r"real mats 15000 terms")
#ax1.plot(tlist_corr, imag(corr_15k), "r--", linewidth=3, label= r"imag mats 15000 terms")
ax1.plot( tlist_corr,real(corr_2k), "g--", linewidth=3, label= r"real mats 2 terms")
#ax1.plot(tlist_corr, imag(cppL), color="r", linewidth=3, label= r"imag mats 2 terms")

ax1.set_xlabel("t")
ax1.set_ylabel(r"$C$")
ax1.legend()

fig, ax1 = plt.subplots(figsize=(12, 7))

ax1.plot(tlist_corr, real(cppLP) - real(corr_15k), color="b", linewidth=3, label= r"pade error")
ax1.plot(tlist_corr, real(corr_2k) - real(corr_15k),"r--", linewidth=3, label= r"mats error")

ax1.set_xlabel("t")
ax1.set_ylabel(r"Error")
ax1.legend();
```

```{code-cell} ipython3
# put pade parameters in lists for heom solver
ckAR = [real(eta) +0j for eta in etapLP]
ckAI = [imag(etapLP[0]) + 0j]
vkAR = [gam +0j for gam in gampLP]
vkAI = [gampLP[0] + 0j]

options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

with timer("RHS construction time"):
    bath = BosonicBath(Q, ckAR, vkAR, ckAI, vkAI)
    HEOMPade = HEOMSolver(Hsys, bath, NC, options=options)

with timer("ODE solver time"):
    resultPade = HEOMPade.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Mats"),
    (resultMats, P12p, 'r', "P12 Mats"),
    (resultMatsT, P11p, 'y', "P11 Mats + Term"),
    (resultMatsT, P12p, 'g', "P12 Mats + Term"),
    (resultPade, P11p, 'b--', "P11 Pade"),
    (resultPade, P12p, 'r--', "P12 Pade"),
]);
```

The Padé decomposition of the Drude-Lorentz bath is also available via a built-in class, `DrudeLorentzPadeBath` bath. Like `DrudeLorentzBath`, the one can obtain the terminator by calling `bath.terminator()`.

Below we show how to use the built-in Padé Drude-Lorentz bath and its terminator (although the termintor does not provide much improvement here, because the Padé expansion already fits the correlation function well):

```{code-cell} ipython3
options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

with timer("RHS construction time"):
    bath = DrudeLorentzPadeBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
    _, terminator = bath.terminator()
    Ltot = liouvillian(Hsys) + terminator
    HEOM_dlpbath_T = HEOMSolver(Ltot, bath, NC, options=options)

with timer("ODE solver time"):
    result_dlpbath_T = HEOM_dlpbath_T.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations([
    (result_dlpbath_T, P11p, 'b', "P11 Padé (DrudeLorentzBath + Term)"),
    (result_dlpbath_T, P12p, 'r', "P12 Padé (DrudeLorentzBath + Term)"),
]);
```

### Next we do fitting of correlation functions, and compare the Matsubara and Pade decompositions

This is not efficient for this example, but can be extremely useful in situations where large number of
exponents are needed (e.g., near zero temperature).

First we collect a large sum of matsubara terms for many time steps:

```{code-cell} ipython3
tlist2 = np.linspace(0, 2, 10000)

corr_15k_t10k = dl_corr_approx(tlist2, 15_000)

corrRana = np.real(corr_15k_t10k)
corrIana = np.imag(corr_15k_t10k)
```

We then fit this sum with standard least-squares approach:

```{code-cell} ipython3
from scipy.optimize import curve_fit
def wrapper_fit_func(x, N, *args):
    a, b = list(args[0][:N]), list(args[0][N:2*N])
    return fit_func(x, a, b, N)

# actual fitting function
def fit_func(x, a, b, N):
    tot = 0
    for i in range(N):
        tot += a[i]*np.exp(b[i]*x)
    return tot


def fitter(ans, tlist, k):
    # the actual computing of fit
    popt = []
    pcov = [] 
    # tries to fit for k exponents
    for i in range(k):
        params_0 = [0]*(2*(i+1))
        upper_a = abs(max(ans, key = abs))*10
        #sets initial guess
        guess = []
        aguess = [ans[0]]*(i+1)#[max(ans)]*(i+1)
        bguess = [0]*(i+1)
        guess.extend(aguess)
        guess.extend(bguess)
        # sets bounds
        b_lower = []
        alower = [-upper_a]*(i+1)
        blower = [-np.inf]*(i+1)
        b_lower.extend(alower)
        b_lower.extend(blower)
        # sets higher bound
        b_higher = []
        ahigher = [upper_a]*(i+1)
        bhigher = [0]*(i+1)
        b_higher.extend(ahigher)
        b_higher.extend(bhigher)
        param_bounds = (b_lower, b_higher)
        p1, p2 = curve_fit(lambda x, *params_0: wrapper_fit_func(x, i+1, \
            params_0), tlist, ans, p0=guess, sigma=[0.01 for t in tlist2], bounds = param_bounds,maxfev = 1e8)
        popt.append(p1)
        pcov.append(p2)
    return popt

# function that evaluates values with fitted params at
# given inputs
def checker(tlist, vals):
    y = []
    for i in tlist:
        # print(i)
        y.append(wrapper_fit_func(i, int(len(vals)/2), vals))
    return y

# number of exponents to use for real part
k = 4
popt1 = fitter(corrRana, tlist2, k)
corrRMats = np.real(dl_corr_approx(tlist2, Nk))

for i in range(k):
    y = checker(tlist2, popt1[i])
    plt.plot(tlist2, corrRana, tlist2, y, tlist2, corrRMats)    
    plt.show()

# number of exponents for imaginary part
k1 = 1
popt2 = fitter(corrIana, tlist2, k1)
for i in range(k1):
    y = checker(tlist2, popt2[i])
    plt.plot(tlist2, corrIana, tlist2, y)
    plt.show()  
```

```{code-cell} ipython3
# Set the exponential coefficients from the fit parameters

ckAR1 = list(popt1[k-1])[:len(list(popt1[k-1]))//2]
ckAR = [x+0j for x in ckAR1]

vkAR1 = list(popt1[k-1])[len(list(popt1[k-1]))//2:]
vkAR = [-x+0j for x in vkAR1]

ckAI1 = list(popt2[k1-1])[:len(list(popt2[k1-1]))//2]
ckAI = [x+0j for x in ckAI1]

vkAI1 = list(popt2[k1-1])[len(list(popt2[k1-1]))//2:]
vkAI = [-x+0j for x in vkAI1]
```

```{code-cell} ipython3
# overwrite imaginary fit with analytical value (not much reason to use the fit for this)

ckAI = [lam * gamma * (-1.0) + 0.j]
vkAI = [gamma+0.j]
```

```{code-cell} ipython3
# The BDF ODE solver method here is faster because we have a slightly stiff problem
# We set NC=8 because we are keeping more exponents

options = Options(nsteps=1500, store_states=True, rtol=1e-12, atol=1e-12, method="bdf") 
NC = 8

with timer("RHS construction time"):
    bath = BosonicBath(Q, ckAR, vkAR, ckAI, vkAI)
    HEOMFit = HEOMSolver(Hsys, bath, NC, options=options)
    
with timer("ODE solver time"):
    resultFit = HEOMFit.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations([
    (resultFit, P11p, 'b', "P11 Fit"),
    (resultFit, P12p, 'r', "P12 Fit"),
]);
```

Here we construct a reaction coordinate inspired model to capture the steady-state behavior,
and compare to the HEOM prediction. This result is more accurate for narrow spectral densities.  Both the population and coherence from this cell are used in the final plot below.

```{code-cell} ipython3
dot_energy, dot_state = Hsys.eigenstates()
deltaE = dot_energy[1] - dot_energy[0]

gamma2 = deltaE / (2 * np.pi * gamma)
wa = 2 * np.pi * gamma2 *   gamma # reaction coordinate frequency
g = np.sqrt(np.pi * wa * lam / 2.0)  # reaction coordinate coupling
g = np.sqrt(np.pi * wa * lam / 4.0)  # reaction coordinate coupling Factor over 2 because of diff in J(w) (I have 2 lam now)
#nb = (1 / (np.exp(wa/w_th) - 1))

NRC = 10

Hsys_exp = tensor(qeye(NRC), Hsys)
Q_exp = tensor(qeye(NRC), Q)
a = tensor(destroy(NRC), qeye(2))

H0 = wa * a.dag() * a + Hsys_exp
# interaction
H1 = (g * (a.dag() + a) * Q_exp)

H = H0 + H1

#print(H.eigenstates())
energies, states = H.eigenstates()
rhoss = 0*states[0]*states[0].dag()
for kk, energ in enumerate(energies):
    rhoss += (states[kk]*states[kk].dag()*exp(-beta*energies[kk])) 

#rhoss = (states[0]*states[0].dag()*exp(-beta*energies[0]) + states[1]*states[1].dag()*exp(-beta*energies[1]))

rhoss = rhoss/rhoss.norm()

P12RC = tensor(qeye(NRC), basis(2,0) * basis(2,1).dag())

P12RC = expect(rhoss,P12RC)


P11RC = tensor(qeye(NRC), basis(2,0) * basis(2,0).dag())

P11RC = expect(rhoss,P11RC)
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
matplotlib.rcParams['text.usetex']=False
```

```{code-cell} ipython3
# XXX: Decide what to do with this cell

fig, axes = plt.subplots(2, 1, sharex=False, figsize=(12,15))

plt.sca(axes[0])
plt.yticks([np.real(P11RC), 0.6, 1.0], [0.32, 0.6, 1])

plot_result_expectations([
    (resultBR, P11p, 'y-.', "Bloch-Redfield"),
    (resultMats, P11p, 'b', "Matsubara $N_k=2$"),
    (resultMatsT, P11p, 'g--', "Matsubara $N_k=2$ & Terminator", {"linewidth": 3}),
    (resultFit, P11p, 'r', r"Fit $N_f = 4$, $N_k=15\times 10^3$", {"dashes": [3,2]}),
], axes=axes[0])
axes[0].plot(tlist, [np.real(P11RC) for t in tlist], 'black', ls='--',linewidth=2, label="Thermal")

axes[0].locator_params(axis='y', nbins=4)
axes[0].locator_params(axis='x', nbins=4)

axes[0].set_ylabel(r"$\rho_{11}$", fontsize=30)
axes[0].legend(loc=0)

axes[0].text(5, 0.9, "(a)", fontsize=30)
axes[0].set_xlim(0,50)

plt.sca(axes[1])
plt.yticks([np.real(P12RC), -0.2, 0.0, 0.2], [-0.33, -0.2, 0, 0.2])

plot_result_expectations([
    (resultBR, P12p, 'y-.', "Bloch-Redfield"),
    (resultMats, P12p, 'b', "Matsubara $N_k=2$"),
    (resultMatsT, P12p, 'g--', "Matsubara $N_k=2$ & Terminator", {"linewidth": 3}),
    (resultFit, P12p, 'r', r"Fit $N_f = 4$, $N_k=15\times 10^3$", {"dashes": [3,2]}),
], axes=axes[1])
axes[1].plot(tlist, [np.real(P12RC) for t in tlist], 'black', ls='--', linewidth=2, label="Thermal")

axes[1].locator_params(axis='y', nbins=4)
axes[1].locator_params(axis='x', nbins=4)

axes[1].text(5, 0.1, "(b)", fontsize=30)

axes[1].set_xlabel(r'$t \Delta$', fontsize=30)
axes[1].set_ylabel(r'$\rho_{01}$', fontsize=30)

axes[1].set_xlim(0,50)

fig.tight_layout()
#fig.savefig("fig1.pdf")
```

```{code-cell} ipython3
from qutip.ipynbtools import version_table

version_table()
```

```{code-cell} ipython3

```
