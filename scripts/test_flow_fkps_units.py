"""Layered unit tests for the flow-matching FKPS sampler.

Uses a 2D Gaussian mixture with the LINEAR interpolant a_t = (1-t) a0 + t a1
(a0 ~ N(0,I)), for which the velocity, score, posterior mean E[a1|a_t], and
marginal p_t are all available in closed form. This lets us check each stage of
flow_fkps independently and pinpoint the broken layer.

Layers tested (in order; fix the first failure first):
  1. score identity   : -(x - t v)/(1-t)  ==  analytic mixture score
  2. x1_hat identity  : x + (1-t) v       ==  E[a1 | a_t = x]
  3. ODE marginal     : eps=0 sampling recovers the mixture (moments + modes)
  4. SDE marginal     : eps>0 keeps the SAME marginal as eps=0 (across N, eps)
  5. FKD steering     : Q toward one mode concentrates particles there

Run:
    cd /path/to/fkps && python -m scripts.test_flow_fkps_units
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from agents.fkd_jax import FKDJax, PotentialType, fkd_sample

# ----------------------------------------------------------------------
# Analytic mixture under the LINEAR interpolant.
# ----------------------------------------------------------------------
MODES = jnp.array([[0.6, 0.6], [-0.6, 0.6], [-0.6, -0.6], [0.6, -0.6]])
MODE_STD = 0.1
W = jnp.ones(MODES.shape[0]) / MODES.shape[0]


def _var_t(t):
    # a_t | k ~ N(t mu_k, var_t I),  var_t = t^2 s^2 + (1-t)^2
    return t**2 * MODE_STD**2 + (1.0 - t) ** 2


def responsibilities(x, t):
    """Posterior over mixture components given a_t = x. x: (N,2) -> (N,K)."""
    var = _var_t(t)
    mean_k = t * MODES  # (K,2)
    sq = jnp.sum((x[:, None, :] - mean_k[None]) ** 2, axis=-1)  # (N,K)
    logc = jnp.log(W)[None] - 0.5 * sq / var - jnp.log(2 * jnp.pi * var)
    return jax.nn.softmax(logc, axis=-1)


def analytic_velocity(x, t):
    """Marginal flow velocity v(x,t) = E[a1 - a0 | a_t = x]."""
    var = _var_t(t)
    mean_k = t * MODES
    resp = responsibilities(x, t)  # (N,K)
    coef = (t * MODE_STD**2 - (1.0 - t)) / var  # scalar
    # v_k = mu_k + coef (x - t mu_k)
    v_k = MODES[None] + coef * (x[:, None, :] - mean_k[None])  # (N,K,2)
    return jnp.sum(resp[:, :, None] * v_k, axis=1)


def analytic_score(x, t):
    """Score of the marginal p_t (mixture of Gaussians)."""
    var = _var_t(t)
    mean_k = t * MODES
    resp = responsibilities(x, t)
    return jnp.sum(resp[:, :, None] * (mean_k[None] - x[:, None, :]), axis=1) / var


def analytic_post_mean_a1(x, t):
    """E[a1 | a_t = x] for the mixture."""
    var = _var_t(t)
    mean_k = t * MODES
    resp = responsibilities(x, t)
    coef = t * MODE_STD**2 / var
    e_k = MODES[None] + coef * (x[:, None, :] - mean_k[None])  # (N,K,2)
    return jnp.sum(resp[:, :, None] * e_k, axis=1)


# ----------------------------------------------------------------------
# Samplers that use ONLY the analytic velocity (no network), mirroring
# flow_fkps._flow_sde_step and _sample_actions_bc exactly.
# ----------------------------------------------------------------------
def sde_step(x, sampling_idx, rng, N, eps, clip=True):
    t = sampling_idx.astype(jnp.float32) / N
    dt = 1.0 / N
    v = analytic_velocity(x, t)
    drift = v * (1.0 + eps * t) - eps * x
    g = jnp.sqrt(jnp.maximum(2.0 * eps * (1.0 - t), 0.0))
    z = jax.random.normal(rng, x.shape)
    next_x = x + drift * dt + g * jnp.sqrt(dt) * z
    x1_hat = x + (1.0 - t) * v
    if clip:
        x1_hat = jnp.clip(x1_hat, -1, 1)
    return next_x, x1_hat


def sample_sde(rng, n, N, eps):
    rng, k = jax.random.split(rng)
    x = jax.random.normal(k, (n, 2))
    for i in range(N):
        rng, sk = jax.random.split(rng)
        x, _ = sde_step(x, jnp.int32(i), sk, N, eps)
    return x


def mode_fractions(samples, thresh=0.25):
    d = jnp.linalg.norm(samples[:, None, :] - MODES[None], axis=-1)  # (n,K)
    nearest = jnp.argmin(d, axis=-1)
    on = jnp.min(d, axis=-1) < thresh
    fr = jnp.array([jnp.mean((nearest == k) & on) for k in range(MODES.shape[0])])
    return fr, float(jnp.mean(on))


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_score_identity():
    rng = jax.random.PRNGKey(0)
    max_err = 0.0
    for t in [0.05, 0.3, 0.5, 0.7, 0.95]:
        rng, k = jax.random.split(rng)
        x = jax.random.normal(k, (256, 2)) * (1 - t) + t * MODES[0]
        v = analytic_velocity(x, t)
        score_from_v = -(x - t * v) / (1 - t)
        score_true = analytic_score(x, t)
        err = float(jnp.max(jnp.abs(score_from_v - score_true)))
        max_err = max(max_err, err)
    ok = max_err < 1e-3
    print(f"[1] score identity      max_err={max_err:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_x1hat_identity():
    rng = jax.random.PRNGKey(1)
    max_err = 0.0
    for t in [0.05, 0.3, 0.5, 0.7, 0.95]:
        rng, k = jax.random.split(rng)
        x = jax.random.normal(k, (256, 2)) * (1 - t) + t * MODES[2]
        v = analytic_velocity(x, t)
        x1_from_v = x + (1 - t) * v
        x1_true = analytic_post_mean_a1(x, t)
        err = float(jnp.max(jnp.abs(x1_from_v - x1_true)))
        max_err = max(max_err, err)
    ok = max_err < 1e-3
    print(f"[2] x1_hat identity     max_err={max_err:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_ode_marginal():
    rng = jax.random.PRNGKey(2)
    x = sample_sde(rng, 4000, N=200, eps=0.0)
    fr, on = mode_fractions(x)
    bal = float(jnp.max(jnp.abs(fr - 0.25)))
    ok = on > 0.95 and bal < 0.06
    print(f"[3] ODE marginal        on_mode={on:.3f} balance_err={bal:.3f}  "
          f"fracs={[round(float(f),3) for f in fr]}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_sde_marginal():
    """SDE must keep the SAME marginal as the ODE, across eps and N."""
    rng = jax.random.PRNGKey(3)
    ref = sample_sde(rng, 4000, N=200, eps=0.0)
    ref_fr, ref_on = mode_fractions(ref)
    all_ok = True
    for N in [10, 50, 200]:
        for eps in [0.5, 1.0, 2.0]:
            rng, k = jax.random.split(rng)
            x = sample_sde(k, 4000, N=N, eps=eps)
            fr, on = mode_fractions(x)
            # marginal match: mode fractions close to ODE reference
            drift = float(jnp.max(jnp.abs(fr - ref_fr)))
            ok = on > 0.90 and drift < 0.08
            all_ok = all_ok and ok
            print(f"    N={N:>3} eps={eps:>3} | on_mode={on:.3f} "
                  f"frac_drift_vs_ODE={drift:.3f}  {'PASS' if ok else 'FAIL'}")
    print(f"[4] SDE marginal        {'PASS' if all_ok else 'FAIL'}  "
          f"(ODE ref on_mode={ref_on:.3f})")
    return all_ok


def test_fkd_steering():
    """Q toward mode 0 should concentrate particles there."""
    goal = MODES[0]
    q_mean, q_std = 0.0, 1.0

    def reward_fn(x1_hat):
        return -(jnp.sum((x1_hat - goal) ** 2, axis=-1)) / q_std + q_mean

    N = 100
    fkd = FKDJax(
        potential_type=PotentialType.DIFF,
        lmbda=10.0,
        num_particles=512,
        adaptive_resampling=False,
        resample_frequency=1,
        resampling_t_start=0,
        resampling_t_end=N - 1,
        time_steps=N,
    )

    def step_fn(x, idx, rng):
        return sde_step(x, idx, rng, N, eps=1.0)

    rng = jax.random.PRNGKey(4)
    rng, k = jax.random.split(rng)
    init = jax.random.normal(k, (512, 2))
    out = fkd_sample(fkd, init_latents=init, step_fn=step_fn,
                     reward_fn=reward_fn, rng=rng)
    fr, on = mode_fractions(out)
    goal_frac = float(fr[0])
    ok = goal_frac > 0.6
    print(f"[5] FKD steering        goal_mode_frac={goal_frac:.3f} on_mode={on:.3f}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=" * 64)
    print("flow_fkps layered unit tests (analytic mixture, linear interpolant)")
    print("=" * 64)
    results = [
        test_score_identity(),
        test_x1hat_identity(),
        test_ode_marginal(),
        test_sde_marginal(),
        test_fkd_steering(),
    ]
    print("=" * 64)
    print(f"PASSED {sum(results)}/{len(results)}")
    print("Fix the FIRST failing layer first — later layers depend on it.")
