---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Example 1d: Spin-Bath model, fitting of spectrum and correlation functions

### Introduction

+++

The HEOM method solves the dynamics and steady state of a system and its environment, the latter of which is encoded in a set of auxiliary density matrices.

In this example we show the evolution of a single two-level system in contact with a single bosonic environment.

The properties of the system are encoded in Hamiltonian, and a coupling operator which describes how it is coupled to the environment.

The bosonic environment is implicitly assumed to obey a particular Hamiltonian (see paper), the parameters of which are encoded in the spectral density, and subsequently the free-bath correlation functions.

In the example below we show how model an Ohmic environment with exponential cut-off in two ways:

* First we fit the spectral density with a set of underdamped brownian oscillator functions.

* Second, we evaluate the correlation functions, and fit those with a certain choice of exponential functions.

In each case we will use the fit parameters to determine the correlation function expansion co-efficients needed to construct a description of the bath (i.e. a `BosonicBath` object) to supply to the `HEOMSolver` so that we can solve for the system dynamics.

```{code-cell} ipython3
%pylab inline
```

```{code-cell} ipython3
import contextlib
import time

import numpy as np

from scipy.optimize import curve_fit

from qutip import *
from qutip.nonmarkov.heom import HEOMSolver, BosonicBath

# Import mpmath functions for evaluation of gamma and zeta functions in the expression for the correlation:

from mpmath import mp

mp.dps = 15
mp.pretty = True
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
        if color == 'rand':
            axes.plot(result.times, exp, c=numpy.random.rand(3,), label=label, **kw)
        else:
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
eps = .0    # Energy of the 2-level system.
Del = .2    # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()
```

```{code-cell} ipython3
# Initial state of the system.
rho0 = basis(2,0) * basis(2,0).dag()  
```

### Analytical expressions for the Ohmic bath correlation function and spectral density

+++

Before we begin fitting, let us examine the analytic expressions for the correlation and spectral density functions and write Python equivalents. 

The correlation function is given by (see, e.g., http://www1.itp.tu-berlin.de/brandes/public_html/publications/notes.pdf for a derivation, equation 7.59, but with a factor of $\pi$ moved into the definition of the correlation function):

\begin{align}
C(t) =& \: \frac{1}{\pi}\alpha \omega_{c}^{1 - s} \beta^{- (s + 1)} \: \times \\
      & \: \Gamma(s + 1) \left[ \zeta \left(s + 1, \frac{1 + \beta \omega_c - i \omega_c t}{\beta \omega_c}\right) + \zeta \left(s + 1, \frac{1 + i \omega_c t}{\beta \omega_c}\right) \right]
\end{align}

where $\Gamma$ is the Gamma function and

\begin{equation}
\zeta(z, u) \equiv \sum_{n=0}^{\infty} \frac{1}{(n + u)^z}, \; u \neq 0, -1, -2, \ldots
\end{equation}

is the generalized Zeta function. The Ohmic case is given by $s = 1$.

The corresponding spectral density for the Ohmic case is:

\begin{equation}
J(\omega) = \omega \alpha e^{- \frac{\omega}{\omega_c}}
\end{equation}

```{code-cell} ipython3
def ohmic_correlation(t, alpha, wc, beta, s=1):
    """ The Ohmic bath correlation function as a function of t (and the bath parameters). """
   
    corr = (1/pi) * alpha * wc**(1 - s) * beta**(-(s + 1)) * mp.gamma(s + 1)
    z1_u = (1 + beta * wc - 1.0j * wc * t) / (beta * wc)
    z2_u = (1 + 1.0j * wc * t) / (beta * wc)
    # Note: the arguments to zeta should be in as high precision as possible.
    # See http://mpmath.org/doc/current/basics.html#providing-correct-input
    return np.array([
        complex(corr * (mp.zeta(s + 1, u1) + mp.zeta(s + 1, u2)))
        for u1, u2 in zip(z1_u, z2_u)
    ], dtype=np.complex128)

```

```{code-cell} ipython3
def ohmic_spectral_density(w, alpha, wc):
    """ The Ohmic bath spectral density as a function of w (and the bath parameters). """
    return w * alpha * e**(-w / wc)
```

```{code-cell} ipython3
def ohmic_power_spectrum(w, alpha, wc, beta):
    """ The Ohmic bath power spectrum as a function of w (and the bath parameters). """
    return w * alpha * e**(-abs(w) / wc) * ((1 / (e**(w * beta) - 1)) + 1) * 2
```

Finally, let's set the bath parameters we will work with and write down some measurement operators:

```{code-cell} ipython3
# Bath parameters:

Q = sigmaz()  # coupling operator

alpha = 3.25
T = 0.5
wc = 1
beta = 1 / T 
s = 1
```

```{code-cell} ipython3
# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresonding to groundstate
P11p = basis(2,0) * basis(2,0).dag()
P22p = basis(2,1) * basis(2,1).dag()
# 1,2 element of density matrix  - corresonding to coherence
P12p = basis(2,0) * basis(2,1).dag()
```

### Building the HEOM bath by fitting the spectral density

+++

We begin by fitting the spectral density, using a series of $k$ underdamped harmonic oscillators case with the Meier-Tannor form (J. Chem. Phys. 111, 3365 (1999); https://doi.org/10.1063/1.479669):

\begin{equation}
J_{\mathrm approx}(\omega; a, b, c) = \sum_{i=0}^{k-1} \frac{2 a_i b_i w}{((w + c_i)^2 + b_i^2) ((w - c_i)^2 + b_i^2)}
\end{equation}

where $a, b$ and $c$ are the fit parameters and each is a vector of length $k$.

```{code-cell} ipython3
# Helper functions for packing the paramters a, b and c into a single numpy
# array as required by SciPy's curve_fit:

def pack(a, b, c):
    """ Pack parameter lists for fitting. """
    return np.concatenate((a, b, c))
    

def unpack(params):
    """ Unpack parameter lists for fitting. """
    N = len(params) // 3
    a = params[:N]
    b = params[N:2 * N]
    c = params[2 * N:]
    return a, b, c
```

```{code-cell} ipython3
# The approximate spectral density and a helper for fitting the approximate spectral density
# to values calculated from the analytical formula:

def spectral_density_approx(w, a, b, c):
    """ Calculate the fitted value of the function for the given parameters. """
    tot = 0
    for i in range(len(a)):
        tot += 2 * a[i] * b[i] * w / (((w + c[i])**2 + b[i]**2) * ((w - c[i])**2 + b[i]**2))
    return tot


def fit_spectral_density(J, w, alpha, wc, N):
    """ Fit the spectral density with N underdamped oscillators. """
    sigma = [0.0001] * len(w)

    J_max = abs(max(J, key=abs))

    guesses = pack([J_max] * N, [wc] * N, [wc] * N)
    lower_bounds = pack([-100 * J_max] * N, [0.1 * wc] * N, [0.1 * wc] * N)
    upper_bounds = pack([100 * J_max] * N, [100 * wc] * N, [100 * wc] * N)

    params, _ = curve_fit(
        lambda x, *params: spectral_density_approx(w, *unpack(params)),
        w, J,
        p0=guesses,
        bounds=(lower_bounds, upper_bounds),
        sigma=sigma,
        maxfev=1000000000,
    )

    return unpack(params)
```

With the spectral density approximation $J_{\mathrm approx}(w; a, b, c)$ implemented above, we can now perform the fit and examine the results.

```{code-cell} ipython3
w = np.linspace(0, 25, 20000)
J = ohmic_spectral_density(w, alpha=alpha, wc=wc)

params_k = [
    fit_spectral_density(J, w, alpha=alpha, wc=wc, N=i+1)
    for i in range(4)
]
```

Let's plot the fit for each $k$ and examine how it improves with an increasing number of terms:

```{code-cell} ipython3
for k, params in enumerate(params_k):
    lam, gamma, w0 = params
    y = spectral_density_approx(w, lam, gamma, w0)
    print(f"Parameters [k={k}]: lam={lam}; gamma={gamma}; w0={w0}")
    plt.plot(w, J, w, y)
    plt.show()
```

The fit with four terms looks good. Let's take a closer look at it by plotting the contribution of each term of the fit:

```{code-cell} ipython3
# The parameters for the fit with four terms:

lam, gamma, w0 = params_k[-1]
print(f"Parameters [k={len(params_k) - 1}]: lam={lam}; gamma={gamma}; w0={w0}")
```

```{code-cell} ipython3
# Plot the components of the fit separately:

def spectral_density_ith_component(w, i, lam, gamma, w0):
    """ Return the i'th term of the approximation for the spectral density. """
    return 2 * lam[i] * gamma[i] * w / (((w + w0[i])**2 + gamma[i]**2) * ((w - w0[i])**2 + gamma[i]**2))
    

def plot_spectral_density_fit_components(J, w, lam, gamma, w0, save=True):
    """ Plot the individual components of a fit to the spectral density. """
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
    axes.plot(w, J, 'r--', linewidth=2, label="original")
    for i in range(len(lam)):
        axes.plot(
            w, spectral_density_ith_component(w, i, lam, gamma, w0),
            linewidth=2,
            label=f"fit component {i}",
    )

    axes.set_xlabel(r'$w$', fontsize=28)
    axes.set_ylabel(r'J', fontsize=28)
    axes.legend()

    if save:
        fig.savefig('noisepower.eps')


plot_spectral_density_fit_components(J, w, lam, gamma, w0, save=False)
```

And let's also compare the power spectrum of the fit and the analytical spectral density:

```{code-cell} ipython3
def plot_power_spectrum(alpha, wc, beta, lam, gamma, w0, save=True):
    """ Plot the power spectrum of a fit against the actual power spectrum. """
    w = np.linspace(-10, 10, 50000)

    s_orig = ohmic_power_spectrum(w, alpha=alpha, wc=wc, beta=beta)
    s_fit = spectral_density_approx(w, lam, gamma, w0) * ((1 / (e**(w * beta) - 1)) + 1) *2
    
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
    axes.plot(w, s_orig, 'r', linewidth=2, label="original")
    axes.plot(w, s_fit, 'b', linewidth=2, label="fit")

    axes.set_xlabel(r'$\omega$', fontsize=28)
    axes.set_ylabel(r'$S(\omega)$', fontsize=28)
    axes.legend()

    if save:
        fig.savefig('powerspectrum.eps')


plot_power_spectrum(alpha, wc, beta, lam, gamma, w0, save=False)
```

Now that we have a good fit to the spectral density, we can calculate the Matsubara expansion terms for the `BosonicBath` from them. At the same time we will calculate the Matsubara terminator for this expansion.

```{code-cell} ipython3
def matsubara_coefficients_from_spectral_fit(lam, gamma, w0, beta, Q, Nk):
    """ Calculate the Matsubara co-efficients for a fit to the spectral density. """

    terminator = 0. * spre(Q)  # initial 0 value with the correct dimensions
    terminator_max_k = 1000  # the number of matsubara expansion terms to include in the terminator
    
    ckAR = []
    vkAR = []
    ckAI = []
    vkAI = []

    for lamt, Gamma, Om in zip(lam, gamma, w0):
      
        ckAR.extend([
            (lamt / (4 * Om)) * coth(beta * (Om + 1.0j * Gamma) / 2),
            (lamt / (4 * Om)) * coth(beta * (Om - 1.0j * Gamma) / 2),
        ])
        for k in range(1, Nk + 1):
            ek = 2 * np.pi * k / beta
            ckAR.append(
                (-2 * lamt * 2 * Gamma / beta) * ek /
                (((Om + 1.0j * Gamma)**2 + ek**2) * ((Om - 1.0j * Gamma)**2 + ek**2))
            )

        terminator_factor = 0
        for k in range(Nk + 1, terminator_max_k):
            ek = 2 * pi * k / beta
            ck = (
                (-2 * lamt * 2 * Gamma / beta) * ek /
                (((Om + 1.0j * Gamma)**2 + ek**2) * ((Om - 1.0j * Gamma)**2 + ek**2))
            )
            terminator_factor += ck / ek
        terminator += terminator_factor * (
            2 * spre(Q) * spost(Q.dag()) - spre(Q.dag() * Q) - spost(Q.dag() * Q)
        )

        vkAR.extend([
            -1.0j * Om + Gamma,
            1.0j * Om + Gamma,
        ])
        vkAR.extend([
            2 * np.pi * k * T + 0.j
            for k in range(1, Nk + 1)
        ])

        ckAI.extend([
            -0.25 * lamt * 1.0j / Om,
            0.25 * lamt * 1.0j / Om,
        ])
        vkAI.extend([
            -(-1.0j * Om - Gamma),
            -(1.0j * Om - Gamma),
        ])
        
    return ckAR, vkAR, ckAI, vkAI, terminator
```

```{code-cell} ipython3
options = Options(nsteps=15000, store_states=True, rtol=1e-12, atol=1e-12, method="bdf")
# This problem is a little stiff, so we use  the BDF method to solve the ODE ^^^

def generate_spectrum_results(p_k=-1, Nk=1, max_depth=5):
    
    lam, gamma, w0 = params_k[p_k]
    ckAR, vkAR, ckAI, vkAI, terminator = matsubara_coefficients_from_spectral_fit(lam, gamma, w0, beta=beta, Q=Q, Nk=Nk)
    Ltot = liouvillian(Hsys) + terminator
    tlist = np.linspace(0, 30 * pi / Del, 600)
    
    with timer("RHS construction time"):
        bath = BosonicBath(Q, ckAR, vkAR, ckAI, vkAI)
        HEOM_spectral_fit = HEOMSolver(Ltot, bath, max_depth=max_depth, options=options)

    with timer("ODE solver time"):
        results_spectral_fit = (HEOM_spectral_fit.run(rho0, tlist))
    
    return results_spectral_fit
```

Below we generate results for different convergence parameters (number of terms in the fit, number of matsubara terms, and depth of the hierarchy).  For the parameter choices here, we need a relatively large depth of around '11', which can be a little slow.

```{code-cell} ipython3


#Generate results for different number of lorentzians in fit
Pklist = range(len(params_k))
results_spectral_fit_pk  = []
for p_k in Pklist:
    print(p_k+1)
    results_spectral_fit_pk.append(generate_spectrum_results(p_k = p_k, Nk = 1, max_depth = 11, ))

```

```{code-cell} ipython3
#generate results for different number of Matsubara terms per Lorentzian
#for max number of Lorentzians
Nklist = range(2,3)
results_spectral_fit_nk  = []
for Nk in Nklist:
    print(Nk)
    results_spectral_fit_nk.append(generate_spectrum_results(max_depth =10, Nk=Nk))
```

```{code-cell} ipython3

#Generate results for different depths
Nclist = range(3,12)
results_spectral_fit_nc  = []
for Nc in Nclist:
    print(Nc)
    results_spectral_fit_nc.append(generate_spectrum_results(max_depth = Nc,Nk = 1))

```

```{code-cell} ipython3
plot_result_expectations([
    (results_spectral_fit_pk[Kk], P11p,
    'rand',
     "P11 (spectral fit) $k_J$="+str(Kk+1)) 
    for Kk in Pklist]);
```

```{code-cell} ipython3
plot_result_expectations([
    (results_spectral_fit_nk[Kk], P11p,
    'rand',
     "P11 (spectral fit) K="+str(Kk+1)) 
    for Kk in range(len(Nklist))]);

```

```{code-cell} ipython3
plot_result_expectations([
    (results_spectral_fit_nc[Kk], P11p,
    'rand',
     "P11 (spectral fit) $N_C=$"+str(Kk+3)) 
    for Kk in range(len(Nclist))]);
```

We now combine the fitting and correlation function data into one large plot.

```{code-cell} ipython3
def correlation_approx_matsubara(t, ck, vk):
    """ Calculate the approximate real or imaginary part of the correlation function
        from the matsubara expansion co-efficients.
    """
    ck = np.array(ck)
    vk = np.array(vk)
    y = []
    for i in t:
        y.append(np.sum(ck * np.exp(-vk * i)))
    return y

```

```{code-cell} ipython3
def set_paper_figure_rcparams():
    matplotlib.rcParams['figure.figsize'] = (7, 5)
    matplotlib.rcParams['axes.titlesize'] = 25
    matplotlib.rcParams['axes.labelsize'] = 30
    matplotlib.rcParams['xtick.labelsize'] = 28
    matplotlib.rcParams['ytick.labelsize'] = 28
    matplotlib.rcParams['legend.fontsize'] = 20
    matplotlib.rcParams['axes.grid'] = False
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['lines.markersize'] = 5
    matplotlib.rcParams['font.family'] = 'STIXgeneral' 
    matplotlib.rcParams['mathtext.fontset'] =  'stix'
    matplotlib.rcParams["font.serif"] = "STIX"
    matplotlib.rcParams['text.usetex'] = False
    
set_paper_figure_rcparams()
```

```{code-cell} ipython3
def plot_matsubara_spectrum_fit_vs_actual(t, C, ckAR, vkAR, ckAI, vkAI, save = False):
    fig = plt.figure(figsize=(12, 10))
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)

    # C_R(t)
    
    yR = correlation_approx_matsubara(t, ckAR, vkAR)

    axes1 = fig.add_subplot(grid[0, 0])

    axes1.plot(t, np.real(C), "r", linewidth=3, label="Original")
    axes1.plot(t, np.real(yR), "g", dashes=[3, 3], linewidth=2, label="Reconstructed")

    axes1.legend(loc=0)
    axes1.set_ylabel(r'$C_R(t)$', fontsize=28)
    axes1.set_xlabel(r'$t\;\omega_c$', fontsize=28)
    axes1.locator_params(axis='y', nbins=4)
    axes1.locator_params(axis='x', nbins=4)
    axes1.text(2., 1.5, "(a)", fontsize=28)

    # C_I(t)

    yI = correlation_approx_matsubara(t, ckAI, vkAI)
    
    axes2 = fig.add_subplot(grid[0, 1])

    axes2.plot(t, np.imag(C), "r", linewidth=3, label="Original")
    axes2.plot(t, np.real(yI), "g", dashes=[3, 3], linewidth=2, label="Reconstructed")

    axes2.legend(loc=0)
    axes2.set_ylabel(r'$C_I(t)$', fontsize=28)
    axes2.set_xlabel(r'$t\;\omega_c$', fontsize=28)
    axes2.locator_params(axis='y', nbins=4)
    axes2.locator_params(axis='x', nbins=4)
    axes2.text(12.5, -0.2, "(b)", fontsize=28)
    
    # J(w)    

    w = np.linspace(0, 25, 20000)

    J_orig = ohmic_spectral_density(w, alpha=alpha, wc=wc)
    J_fit  = spectral_density_approx(w, lam, gamma, w0)

    axes3 = fig.add_subplot(grid[1, 0])

    axes3.plot(w, J_orig, "r", linewidth=3, label="$J(\omega)$ original")
    axes3.plot(w, J_fit, "g", dashes=[3, 3], linewidth=2, label="$J(\omega)$ Fit $k_J = 4$")
    
    axes3.legend(loc=0)
    axes3.set_ylabel(r'$J(\omega)$', fontsize=28)
    axes3.set_xlabel(r'$\omega/\omega_c$', fontsize=28)
    axes3.locator_params(axis='y', nbins=4)
    axes3.locator_params(axis='x', nbins=4)
    axes3.text(3, 1.1, "(c)", fontsize=28)

    # S(w)

    # avoid the pole in the fit around zero:
    w = np.concatenate([np.linspace(-10, -0.1, 5000), np.linspace(0.1, 10, 5000)])
    
    s_orig = ohmic_power_spectrum(w, alpha=alpha, wc=wc, beta=beta)    
    s_fit = spectral_density_approx(w, lam, gamma, w0) * ((1 / (e**(w * beta) - 1)) + 1) *2
    

    axes4 = fig.add_subplot(grid[1, 1])
    axes4.plot(w, s_orig,"r",linewidth=3,label="Original")
    axes4.plot(w, s_fit, "g", dashes=[3, 3], linewidth=2,label="Reconstructed")

    axes4.legend()
    axes4.set_ylabel(r'$S(\omega)$', fontsize=28)
    axes4.set_xlabel(r'$\omega/\omega_c$', fontsize=28)
    axes4.locator_params(axis='y', nbins=4)
    axes4.locator_params(axis='x', nbins=4)
    axes4.text(-8., 2.5, "(d)", fontsize=28)

    if save:
        fig.savefig("figFiJspec.pdf")

t = linspace(0,15,100)
C = ohmic_correlation(t, alpha=alpha, wc=wc, beta=beta)

ckAR, vkAR, ckAI, vkAI, terminator = matsubara_coefficients_from_spectral_fit(lam, gamma, w0, beta=beta, Q=Q, Nk=1)
plot_matsubara_spectrum_fit_vs_actual(t, C, ckAR, vkAR, ckAI, vkAI, save = True)
```

### Building the HEOM bath by fitting the correlation function

+++

Having successfully fitted the spectral density and used the result to calculate the Matsubara expansion and terminator for the HEOM bosonic bath, we now proceed to the second case of fitting the correlation function itself instead.

Here we fit the real and imaginary parts seperately, using the following ansatz

$$C_R^F(t) = \sum_{i=1}^{k_R} c_R^ie^{-\gamma_R^i t}\cos(\omega_R^i t)$$

$$C_I^F(t) = \sum_{i=1}^{k_I} c_I^ie^{-\gamma_I^i t}\sin(\omega_I^i t)$$

```{code-cell} ipython3
# The approximate correlation functions and a helper for fitting the approximate correlation
# function to values calculated from the analytical formula:

def correlation_approx_real(t, a, b, c):
    """ Calculate the fitted value of the function for the given parameters. """
    tot = 0
    for i in range(len(a)):
        tot += a[i] * np.exp(b[i] * t) * np.cos(c[i] * t)
    return tot      


def correlation_approx_imag(t, a, b, c):
    """ Calculate the fitted value of the function for the given parameters. """
    tot = 0
    for i in range(len(a)):
        tot += a[i] * np.exp(b[i] * t) * np.sin(c[i] * t)    
    return tot


def fit_correlation_real(C, t, wc, N):
    """ Fit the spectral density with N underdamped oscillators. """
    sigma = [0.1] * len(t)

    C_max = abs(max(C, key=abs))

    guesses = pack([C_max] * N, [-wc] * N, [wc] * N)
    lower_bounds = pack([-20 * C_max] * N, [-np.inf] * N, [0.] * N)
    upper_bounds = pack([20 * C_max] * N, [0.1] * N, [np.inf] * N)

    params, _ = curve_fit(
        lambda x, *params: correlation_approx_real(t, *unpack(params)),
        t, C,
        p0=guesses,
        bounds=(lower_bounds, upper_bounds),
        sigma=sigma,
        maxfev=1000000000,
    )

    return unpack(params)


def fit_correlation_imag(C, t, wc, N):
    """ Fit the spectral density with N underdamped oscillators. """
    sigma = [0.0001] * len(t)

    C_max = abs(max(C, key=abs))

    guesses = pack([-C_max] * N, [-2] * N, [1] * N)
    lower_bounds = pack([-5 * C_max] * N, [-100] * N, [0.] * N)
    upper_bounds = pack([5 * C_max] * N, [0.01] * N, [100] * N)

    params, _ = curve_fit(
        lambda x, *params: correlation_approx_imag(t, *unpack(params)),
        t, C,
        p0=guesses,
        bounds=(lower_bounds, upper_bounds),
        sigma=sigma,
        maxfev=1000000000,
    )

    return unpack(params)    
```

```{code-cell} ipython3

t = linspace(0,15,50000)
C = ohmic_correlation(t, alpha=alpha, wc=wc, beta=beta)

params_k_real = [
    fit_correlation_real(np.real(C), t, wc=wc, N=i+1)
    for i in range(3)
]

params_k_imag = [
    fit_correlation_imag(np.imag(C), t, wc=wc, N=i+1)
    for i in range(3)
]
```

```{code-cell} ipython3
for k, params in enumerate(params_k_real):
    lam, gamma, w0 = params
    y = correlation_approx_real(t, lam, gamma, w0)
    print(f"Parameters [k={k}]: lam={lam}; gamma={gamma}; w0={w0}")
    plt.plot(t, np.real(C), t, y)
    plt.show()
```

```{code-cell} ipython3
for k, params in enumerate(params_k_imag):
    lam, gamma, w0 = params
    y = correlation_approx_imag(t, lam, gamma, w0)
    print(f"Parameters [k={k}]: lam={lam}; gamma={gamma}; w0={w0}")
    plt.plot(t, np.imag(C), t, y)
    plt.show()
```

Now we construct the `BosonicBath` co-efficients and frequencies from the fit to the correlation function:

```{code-cell} ipython3
def matsubara_coefficients_from_corr_fit_real(lam, gamma, w0):
    """ Return the matsubara coefficients for the imaginary part of the correlation function """
    ckAR = [0.5 * x + 0j for x in lam]  # the 0.5 is from the cosine
    ckAR.extend(np.conjugate(ckAR))     # extend the list with the complex conjugates
    
    vkAR = [-x - 1.0j * y for x, y in zip(gamma, w0)]
    vkAR.extend([-x + 1.0j * y for x, y in zip(gamma, w0)])
    
    return ckAR, vkAR

def matsubara_coefficients_from_corr_fit_imag(lam, gamma, w0):
    """ Return the matsubara coefficients for the imaginary part of the correlation function. """
    ckAI = [-0.5j * x for x in lam]  # the 0.5 is from the sine
    ckAI.extend(np.conjugate(ckAI))  # extend the list with the complex conjugates
    
    vkAI = [-x - 1.0j * y for x, y in zip(gamma, w0)]
    vkAI.extend([-x + 1.0j * y for x, y in zip(gamma, w0)])
    
    return ckAI, vkAI
```

```{code-cell} ipython3
ckAR, vkAR = matsubara_coefficients_from_corr_fit_real(*params_k_real[-1])
ckAI, vkAI = matsubara_coefficients_from_corr_fit_imag(*params_k_imag[-1])
```

```{code-cell} ipython3


def corr_spectrum_approx(w, ckAR, vkAR, ckAI, vkAI):
    """ Calculates the approximate power spectrum from ck and vk. """
    S = np.zeros(len(w), dtype=np.complex128)
    for ck, vk in zip(ckAR, vkAR):
        S += 2 * ck * np.real(vk) / ((w - np.imag(vk))**2 + (np.real(vk)**2))
    for ck, vk in zip(ckAI, vkAI):
        S += 2 * 1.0j * ck * np.real(vk) / ((w - np.imag(vk))**2 + (np.real(vk)**2))
    return S
```

```{code-cell} ipython3
def plot_matsubara_correlation_fit_vs_actual(t, C, ckAR, vkAR, ckAI, vkAI, save = False):
    fig = plt.figure(figsize=(12, 10))
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)

    # C_R(t)
    
    yR = correlation_approx_matsubara(t, ckAR, vkAR)

    axes1 = fig.add_subplot(grid[0, 0])

    axes1.plot(t, np.real(C), "r", linewidth=3, label="Original")
    axes1.plot(t, np.real(yR), "g", dashes=[3, 3], linewidth=2, label="Fit $k_R = 3$")

    axes1.legend(loc=0)
    axes1.set_ylabel(r'$C_R(t)$', fontsize=28)
    axes1.set_xlabel(r'$t\;\omega_c$', fontsize=28)
    axes1.locator_params(axis='y', nbins=4)
    axes1.locator_params(axis='x', nbins=4)
    axes1.text(2., 1.4, "(a)", fontsize=28)

    # C_I(t)

    yI = correlation_approx_matsubara(t, ckAI, vkAI)
    
    axes2 = fig.add_subplot(grid[0, 1])

    axes2.plot(t, np.imag(C), "r", linewidth=3, label="Original")
    axes2.plot(t, np.real(yI), "g", dashes=[3, 3], linewidth=2, label="Fit $K_I=3$")

    axes2.legend(loc=0)
    axes2.set_ylabel(r'$C_I(t)$', fontsize=28)
    axes2.set_xlabel(r'$t\;\omega_c$', fontsize=28)
    axes2.locator_params(axis='y', nbins=4)
    axes2.locator_params(axis='x', nbins=4)
    axes2.text(12.5, -0.2, "(b)", fontsize=28)
    
    # J(w)    

    w = np.linspace(0, 25, 20000)

    J_orig = ohmic_spectral_density(w, alpha=alpha, wc=wc)
    
    J_fit = corr_spectrum_approx(w, ckAR, vkAR, ckAI, vkAI) / (((1 / (e**(w * beta) - 1)) + 1) * 2)


    axes3 = fig.add_subplot(grid[1, 0])

    axes3.plot(w, J_orig, "r", linewidth=3, label="Original")
    axes3.plot(w, J_fit, "g", dashes=[3, 3], linewidth=2, label="Reconstructed")
    
    axes3.legend(loc=0)
    axes3.set_ylabel(r'$J(\omega)$', fontsize=28)
    axes3.set_xlabel(r'$\omega/\omega_c$', fontsize=28)
    axes3.locator_params(axis='y', nbins=4)
    axes3.locator_params(axis='x', nbins=4)
    axes3.text(3, 1.1, "(c)", fontsize=28)
    #axes3.set_ylim(0,2)

    # S(w)

    # avoid the pole in the fit around zero:
    w = np.concatenate([np.linspace(-10, -0.1, 5000), np.linspace(0.1, 10, 5000)])
    
    s_orig = ohmic_power_spectrum(w, alpha=alpha, wc=wc, beta=beta)
    s_fit = corr_spectrum_approx(w, ckAR, vkAR, ckAI, vkAI) 

    axes4 = fig.add_subplot(grid[1, 1])
    axes4.plot(w, s_orig,"r",linewidth=3,label="Original")
    axes4.plot(w, s_fit, "g", dashes=[3, 3], linewidth=2,label="Reconstructed")

    axes4.legend()
    axes4.set_ylabel(r'$S(\omega)$', fontsize=28)
    axes4.set_xlabel(r'$\omega/\omega_c$', fontsize=28)
    axes4.locator_params(axis='y', nbins=4)
    axes4.locator_params(axis='x', nbins=4)
    axes4.text(-8., 2.5, "(d)", fontsize=28)

    if save:
        fig.savefig("figFiCspec.pdf")


plot_matsubara_correlation_fit_vs_actual(t, C, ckAR, vkAR, ckAI, vkAI, save = True)
```

```{code-cell} ipython3
options = Options(nsteps=15000, store_states=True, rtol=1e-12, atol=1e-12, method="bdf")
# This problem is a little stiff, so we use  the BDF method to solve the ODE ^^^

tlist = np.linspace(0, 30 * pi / Del, 600)

with timer("RHS construction time"):
    bath = BosonicBath(Q, ckAR, vkAR, ckAI, vkAI)
    HEOM_corr_fit = HEOMSolver(Hsys, bath, max_depth=5, options=options)
    
with timer("ODE solver time"):
    results_corr_fit = HEOM_corr_fit.run(rho0, tlist)
```

```{code-cell} ipython3
options = Options(nsteps=15000, store_states=True, rtol=1e-12, atol=1e-12, method="bdf")
# This problem is a little stiff, so we use  the BDF method to solve the ODE ^^^

def generate_corr_results(p_k=-1, max_depth=11):
    ckAR, vkAR = matsubara_coefficients_from_corr_fit_real(*params_k_real[p_k])
    ckAI, vkAI = matsubara_coefficients_from_corr_fit_imag(*params_k_imag[p_k])
    
    
    
    tlist = np.linspace(0, 30 * pi / Del, 600)
    
    with timer("RHS construction time"):
        bath = BosonicBath(Q, ckAR, vkAR, ckAI, vkAI)
        HEOM_corr_fit = HEOMSolver(Hsys, bath, max_depth=max_depth, options=options)

    with timer("ODE solver time"):
        results_corr_fit = (HEOM_corr_fit.run(rho0, tlist))
    
    return results_corr_fit


#Generate results for different number of lorentzians in fit
Pklist = range(len(params_k_real))
results_corr_fit_pk  = []
for p_k in Pklist:
    print(p_k+1)
    results_corr_fit_pk.append(generate_corr_results(p_k = p_k))

```

```{code-cell} ipython3
Pklist = range(len(params_k_real))
plot_result_expectations([
    (results_corr_fit_pk[Kk], P11p,
    'rand',
     "P11 (correlation fit) k_R=k_I="+str(Kk+1)) 
    for Kk in Pklist]);

```

```{code-cell} ipython3
#Nc = 5

tlist4 = np.linspace(0, 4*pi/Del, 600)
# Plot the results
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(12,7))
axes.set_yticks([0.6,0.8,1])
axes.set_yticklabels([0.6,0.8,1]) 

axes.plot(tlist4, np.real(expect(results_corr_fit_pk[0].states,P11p)), 'y', linewidth=2, label="Correlation Function Fit $k_R=k_I=1$")
axes.plot(tlist4, np.real(expect(results_corr_fit_pk[2].states,P11p)), 'y-.', linewidth=2, label="Correlation Function Fit $k_R=k_I=3$")


axes.plot(tlist4, np.real(expect(results_spectral_fit_pk[0].states,P11p)), 'b', linewidth=2, label="Spectral Density Fit $k_J=1$")
axes.plot(tlist4, np.real(expect(results_spectral_fit_pk[2].states,P11p)), 'g--', linewidth=2, label="Spectral Density Fit $k_J=3$")
axes.plot(tlist4, np.real(expect(results_spectral_fit_pk[3].states,P11p)), 'r-.', linewidth=2, label="Spectral Density Fit $k_J=4$")

axes.set_ylabel(r'$\rho_{11}$',fontsize=30)

axes.set_xlabel(r'$t\;\omega_c$',fontsize=30)
axes.locator_params(axis='y', nbins=3)
axes.locator_params(axis='x', nbins=3)
axes.legend(loc=0, fontsize=20)


fig.savefig("figFit.pdf")
```

```{code-cell} ipython3
from qutip.ipynbtools import version_table

version_table()
```

```{code-cell} ipython3

```
