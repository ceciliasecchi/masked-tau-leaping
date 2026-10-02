"""
optimal_schedules.py
=======================
Information profile, dependence density rho, the eps_fact-optimal unmasking
schedule (recursive formula, Corollary 1), and a Monte-Carlo estimate of the
factorization KL, for masked discrete diffusion with a mixture-of-products target.

Target
------
    Z ~ Cat(w),   X^j | Z = i  ~  Cat(mu[i, j, :])   independently across j
    =>  pi(x) = sum_i w_i prod_j mu[i, j, x^j].

Because the latent Z takes finitely many values, every conditional needed for the
Monte-Carlo estimators is available in closed form via the posterior over Z.

Notation follows the paper:
    f(i)   = E_{pi, sigma}[ log pi(x^{sigma_{i+1}} | x^{sigma_{<=i}} ],  i = 0..N-1
    rho(u) = sum_{i=1}^{N-1} Delta f(i) * Beta(i, N-i).pdf(u)          (dependence density)
    eps_fact(beta) = N * sum_k int_{beta_{k-1}}^{beta_k} (beta_k - u) rho(u) du   (Thm 1)

----------
    generate_mixture_parameters_dir(r, N, L, alpha, rng)
    information_profile_mc(w, mu, n_samples, rng)        -> f            (Monte Carlo)
    dependence_density(f, u)                             -> rho(u)
    total_correlations(f)                                -> (D, TC, DTC)
    factorization_error(f, beta)                         -> eps_fact      (Thm-1 value)
    linear_schedule(K)                                   -> beta
    optimal_schedule(f, K)                               -> beta          (recursive, Cor 1)
    kl_mc(w, mu, beta, n_samples, rng)                   -> (mean, stderr) (Monte Carlo)
"""

import numpy as np
from scipy.special import betainc, comb, logsumexp
from scipy.optimize import brentq, minimize


__all__ = [
    "generate_mixture_parameters_dir", "sample_pi",
    "information_profile_mc", "dependence_density", "total_correlations",
    "factorization_error", "linear_schedule", "optimal_schedule", "kl_mc",
]


# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------
def generate_mixture_parameters_dir(r, N, L, alpha, rng=None):
    """Mixture of r product distributions; component marginals ~ Dirichlet(alpha/L)."""
    if rng is None:
        rng = np.random.default_rng()
    w = np.full(r, 1.0 / r)
    mu = np.empty((r, N, L))
    av = np.full(L, alpha / L)
    for i in range(r):
        for j in range(N):
            mu[i, j] = rng.dirichlet(av)
    return w, mu


def sample_pi(w, mu, n, rng):
    """Draw n samples x ~ pi (vectorised); returns int array (n, N)."""
    r, N, L = mu.shape
    z = rng.choice(r, size=n, p=w)
    x = np.empty((n, N), int)
    for j in range(N):
        cdf = np.cumsum(mu[z, j, :], axis=1)
        x[:, j] = (rng.random(n)[:, None] < cdf).argmax(1)
    return x


# --------------------------------------------------------------------------
# (1) Information profile  f(i)  by Monte Carlo over (x, sigma)
# --------------------------------------------------------------------------

# The monotonicity argument of information_profile_mc accepts:
#   "raise"   -> stop when the raw Monte Carlo estimate is nonmonotone;
#   "project" -> use its closest nondecreasing isotonic projection.

def information_profile_mc(
    w, mu, n_samples, rng=None, return_se=False, monotonicity="raise"
):
    """
    Monte-Carlo estimate of f(i) = E[ log pi(x^{sigma_{i+1}} | x^{sigma_{<=i}}) ]
    via the conditional-entropy (plug-in / Rao-Blackwellised) estimator
 
        hat f_i = -(1 / (N - i)) sum_{j not in Z} H( p(. | x^Z) ),
 
    where Z = sigma_{<=i} is uniform among size-i subsets, and
        p(l | x^Z) = sum_z P(Z=z | x^Z) mu[z, j, l]
    is the exact one-coordinate conditional under the mixture (the perfect-learning
    plug-in p_theta^j = pi_j). The minus sign converts entropy into f, since
    f(i) = -E[H]. Averaging the entropy over ALL masked coordinates (not just the
    next one) removes the within-x sampling noise, giving lower variance than the
    single-draw log-conditional estimator at the same number of samples.
 
    Each draw samples x ~ pi and a uniform permutation sigma, walks sigma once
    updating the latent posterior incrementally, and at each position i averages the
    conditional entropy over the currently masked coordinates.

    If return_se=True, also return the pointwise Monte-Carlo standard errors of f.
    By default a nonmonotone estimate raises an error. With
    monotonicity="project", it is instead projected onto the nondecreasing cone.
    """
    if rng is None:
        rng = np.random.default_rng()
    if monotonicity not in {"raise", "project", "none"}:
        raise ValueError("monotonicity must be 'raise', 'project', or 'none'")
    if return_se and n_samples < 2:
        raise ValueError("n_samples must be at least 2 to estimate standard errors")
    r, N, L = mu.shape
    x = sample_pi(w, mu, n_samples, rng)
    sigma = np.argsort(rng.random((n_samples, N)), axis=1)      # uniform permutations
    lw = np.tile(np.log(w), (n_samples, 1))                     # latent log-posterior, (S, r)
    masked = np.ones((n_samples, N), bool)                      # masked[s, j]: j not revealed
    logmu = np.log(mu)
    acc = np.zeros(N)
    f_se = np.empty(N) if return_se else None
    S = np.arange(n_samples)
    for i in range(N):
        post = np.exp(lw - logsumexp(lw, axis=1, keepdims=True))    # P(Z | x^Z), (S, r)
        H = np.zeros((n_samples, N))                               # H(p(.|x^Z)) per coord
        for l in range(L):
            pl = post @ mu[:, :, l]                               # (S, N): pi(X^j = l | x^Z)
            H -= pl * np.log(np.clip(pl, 1e-12, None))
        contribution = -(H * masked).sum(1) / (N - i)             # one estimate per sample
        acc[i] = contribution.sum()
        if return_se:
            f_se[i] = contribution.std(ddof=1) / np.sqrt(n_samples)
        ks = sigma[:, i]                                          # reveal sigma_{i+1}
        lw += logmu[:, ks, x[S, ks]].T
        masked[S, ks] = False
    f = acc / n_samples
    if monotonicity != "none" and np.any(np.diff(f) < 0.0):
        if monotonicity == "raise":
            raise RuntimeError(
                "The Monte Carlo estimate f is not monotone nondecreasing; "
                "it requires more samples. Increase n_samples and try again."
            )
        from scipy.optimize import isotonic_regression
        f = isotonic_regression(f, increasing=True).x
    if return_se:
        return f, f_se
    return f
 

# --------------------------------------------------------------------------
# (2) Dependence density rho(u)  (Bernstein form of the beta-smoothed Delta f)
# --------------------------------------------------------------------------
def dependence_density(f, u):
    """
    rho(u) = (N-1) sum_{j=0}^{N-2} Delta f(j+1) C(N-2, j) u^j (1-u)^{N-2-j}.

    """
    f = np.asarray(f, float)
    N = len(f)
    Df = np.diff(f)
    #Df = np.clip(np.diff(f), 0.0, None)
    u = np.atleast_1d(np.asarray(u, float))
    out = np.zeros_like(u)
    for j in range(N - 1):
        out += (N - 1) * Df[j] * comb(N - 2, j) * u**j * (1 - u)**(N - 2 - j)
    return out


def total_correlations(f):
    """D, TC, DTC from the profile (Lemma: linear functionals of rho)."""
    N = len(f)
    Df = np.diff(f)
    #Df = np.clip(np.diff(f), 0.0, None)
    i = np.arange(1, N)
    return Df.sum(), ((N - i) * Df).sum(), (i * Df).sum()


def _rho_cdf(f, n_grid):
    """Grid of u, rho(u), and the cumulative integral R(u) = int_0^u rho (trapezoid)."""
    u = np.linspace(0.0, 1.0, n_grid)
    g = dependence_density(f, u)
    R = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(u))])
    return u, g, R


# --------------------------------------------------------------------------
# (3) Factorization error (Theorem 1) -- analytic, used as a reference
# --------------------------------------------------------------------------
def factorization_error(f, beta, n_grid=None):
    """Evaluate the representation of eps_fact exactly for a supplied profile.

    The optional ``n_grid`` argument is retained for backward compatibility and
    is ignored.  Each integral of a beta density is evaluated with regularized
    incomplete-beta functions, avoiding quadrature error near u=0.
    """
    N = len(f)
    beta = np.asarray(beta, float)
    if beta.ndim != 1:
        raise ValueError("beta must be one-dimensional")
    if not np.isclose(beta[0], 0.0):
        raise ValueError("beta must start at 0")
    if not np.isclose(beta[-1], 1.0):
        raise ValueError("beta must end at 1")
    if np.any(np.diff(beta) < 0):
        raise ValueError("beta must be nondecreasing")
    Df = np.diff(np.asarray(f, dtype=float))
    i = np.arange(1, N, dtype=float)
    tot = 0.0
    for k in range(1, len(beta)):
        a, b = beta[k - 1], beta[k]
        if b <= a:
            continue
        mass = betainc(i, N - i, b) - betainc(i, N - i, a)
        first_moment = (i / N) * (
            betainc(i + 1, N - i, b) - betainc(i + 1, N - i, a)
        )
        tot += np.dot(Df, b * mass - first_moment)
    return float(N * tot)


# --------------------------------------------------------------------------
# (4) Schedules
# --------------------------------------------------------------------------
def linear_schedule(K):
    """Uniform schedule beta_k = k / K."""
    return np.linspace(0.0, 1.0, K + 1)


def optimal_schedule(
    f,
    K,
    n_grid=20_000,
    n_scan=8_000,
    tol=1e-11,
    return_diagnostics=False,
):
    """
    eps_fact-optimal K-step schedule via the recursive formula of Corollary 1:

        beta_{k+1} = beta_k + Psi(beta_k, beta_k - beta_{k-1}),
        Psi(b, delta) = (1 / rho(b)) * int_{b-delta}^{b} rho(u) du,    beta_0 = 0.

    Given beta_1 the recursion fixes beta_2, ..., beta_K.  The shooting map need
    not be monotone for an estimated profile, so we scan it, refine every sign
    change with Brent's method, and retain the stationary schedule with the
    smallest factorization error.  If no shooting root is found, a constrained
    direct optimization is used as a fallback.

    ``f`` must be nondecreasing.  
    """
    f = np.asarray(f, dtype=float)

    if f.ndim != 1 or len(f) < 2:
        raise ValueError("f must be a one-dimensional profile of length at least 2")
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError("K must be a positive integer")
    if np.any(np.diff(f) < -1e-12):
        raise ValueError(
            "f must be nondecreasing; apply project_information_profile(f) "
            "before fitting the schedule"
        )
    if np.all(np.diff(f) <= 1e-15):
        beta = linear_schedule(K)
        diagnostics = {"n_roots": 0, "method": "product/constant profile"}
        return (beta, diagnostics) if return_diagnostics else beta

    u, g, R = _rho_cdf(f, n_grid)
    rho_at = lambda b: float(np.interp(b, u, g))
    R_at = lambda b: float(np.interp(b, u, R))

    def terminal(beta1):
        beta = [0.0, beta1]
        for k in range(1, K):
            b, bm = beta[k], beta[k - 1]
            if b >= 1.0:
                return 2.0, beta
            rho_b = rho_at(b)
            if not np.isfinite(rho_b) or rho_b <= np.finfo(float).tiny:
                return 2.0, beta
            psi = (R_at(b) - R_at(bm)) / rho_b
            if not np.isfinite(psi) or psi < 0.0:
                return 2.0, beta
            beta.append(b + psi)
        return beta[-1], beta

    # For large K the admissible beta_1 windows can be extremely narrow: a
    # slightly larger value may make the recursion hit 1 one step too early.
    # Increase the scan resolution with K so those windows are not skipped.
    effective_scan = max(int(n_scan), 128 * K)
    upper = 1.0 - 1e-10
    split = min(0.1, upper / 2)
    scan = np.unique(np.concatenate([
        np.geomspace(1e-10, split, max(2, 3 * effective_scan // 4)),
        np.linspace(split, upper, max(2, effective_scan // 4)),
    ]))
    values = np.array([terminal(x)[0] - 1.0 for x in scan])
    roots = []
    for left, right, f_left, f_right in zip(
        scan[:-1], scan[1:], values[:-1], values[1:]
    ):
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if f_left == 0.0:
            root = left
        elif np.signbit(f_left) != np.signbit(f_right):
            root = brentq(
                lambda x: terminal(x)[0] - 1.0,
                left,
                right,
                xtol=tol,
            )
        else:
            continue
        if not roots or min(abs(root - old) for old in roots) > 10 * tol:
            roots.append(root)

    candidates = []
    for root in roots:
        endpoint, beta = terminal(root)
        if len(beta) != K + 1 or abs(endpoint - 1.0) > 1e-7:
            continue
        beta = np.asarray(beta, dtype=float)
        beta[-1] = 1.0
        if np.all(np.diff(beta) > 0):
            candidates.append(beta)

    method = "shooting-root scan"
    if candidates:
        objectives = np.array([factorization_error(f, beta) for beta in candidates])
        beta = candidates[int(np.argmin(objectives))]
    else:
        method = "constrained optimization fallback"

        def objective(interior):
            beta_candidate = np.r_[0.0, interior, 1.0]
            increments = np.diff(beta_candidate)
            if np.any(increments <= 0.0):
                # SLSQP may probe very slightly outside its inequality
                # constraints while estimating derivatives.
                return 1e100 + 1e6 * np.square(np.minimum(increments, 0.0)).sum()
            return factorization_error(f, beta_candidate)

        constraints = {
            "type": "ineq",
            "fun": lambda x: np.diff(np.r_[0.0, x, 1.0]) - 1e-10,
        }
        result = minimize(
            objective,
            np.arange(1, K) / K,
            method="SLSQP",
            bounds=[(1e-10, 1.0 - 1e-10)] * (K - 1),
            constraints=constraints,
            options={"ftol": tol, "maxiter": 2_000},
        )
        if not result.success:
            raise RuntimeError(f"schedule optimization failed: {result.message}")
        beta = np.r_[0.0, result.x, 1.0]

    if return_diagnostics:
        print(f"optimal_schedule: {len(candidates)} shooting roots found, "
              f"method: {method}")

    return beta



# --------------------------------------------------------------------------
# (5) Monte-Carlo KL estimate
# --------------------------------------------------------------------------
def kl_mc(w, mu, beta, n_samples, rng=None):
    """
    Monte-Carlo estimate of  KL( pi(x) nu^beta(z) || p_alg(x, z) ), which under
    perfect one-coordinate learning equals the factorization error eps_fact(beta)
    and upper-bounds the marginal KL(pi || p_alg). Here the learned conditionals
    are the exact mixture conditionals, so the learning term is zero.

    Per draw: x ~ pi, reveal-step c_i ~ Cat(Delta beta) i.i.d. (tau-leaping); for each
    block z_k accumulate
        log pi(x^{z_k} | x^{z<k}) - sum_{i in z_k} log pi_i(x^i | x^{z<k}).

    Returns (mean, standard_error).
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2 to estimate the standard error")
    r, N, L = mu.shape
    dbeta = np.diff(np.asarray(beta, float))
    K = len(dbeta)
    x = sample_pi(w, mu, n_samples, rng)
    c = rng.choice(K, size=(n_samples, N), p=dbeta)
    LM = np.empty((n_samples, N, r))                           # LM[s,j,z] = log mu[z,j,x[s,j]]
    with np.errstate(divide="ignore"):
        for j in range(N):
            LM[:, j, :] = np.log(mu[:, j, x[:, j]]).T
        lw = np.tile(np.log(w), (n_samples, 1))
    gap = np.zeros(n_samples)
    for k in range(K):
        mask = (c == k)                                       # block indicator, (S, N)
        logpost = lw - logsumexp(lw, axis=1, keepdims=True)    # log pi(z | prefix)
        T = np.where(mask[:, :, None], LM, 0.0).sum(axis=1)    # log pi(x^block | z, prefix)
        log_block = logsumexp(logpost + T, axis=1)             # log pi(x^block | prefix)
        log_cond = logsumexp(logpost[:, None, :] + LM, axis=2) # log pi_i(x^i | prefix)
        log_prod = np.where(mask, log_cond, 0.0).sum(axis=1)
        gap += (log_block - log_prod)
        lw += T                                                # reveal block
    return float(gap.mean()), float(gap.std(ddof=1) / np.sqrt(n_samples))
