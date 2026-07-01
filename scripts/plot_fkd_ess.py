"""Plot the FKD effective-sample-size (ESS) curve across the sampling process.

ESS in [1, num_particles] measures particle-population diversity. A healthy FKD
run keeps ESS well above 1 while nudging it down at resampling steps; if ESS
collapses to ~1 early, all particles have degenerated to one (over-steering /
lambda too high). If ESS stays flat at num_particles, the weights are uniform
(no steering / lambda too low or Q signal dead).

Uses the analytic Gaussian-mixture flow sampler (no training) so it runs
anywhere and isolates FKD dynamics. Sweeps several lambda values.

Run:
    cd /path/to/fkps && python -m scripts.plot_fkd_ess
Outputs:
    fkd_ess_curve.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from agents.fkd_jax import FKDJax, PotentialType, fkd_sample

# Analytic mixture with the linear interpolant (shared with the unit tests).
MODES = jnp.array([[0.6, 0.6], [-0.6, 0.6], [-0.6, -0.6], [0.6, -0.6]])
MODE_STD = 0.1
W = jnp.ones(MODES.shape[0]) / MODES.shape[0]
Q_GOAL = jnp.array([0.6, 0.6])


def _var_t(t):
    return t**2 * MODE_STD**2 + (1.0 - t) ** 2


def responsibilities(x, t):
    var = _var_t(t)
    mean_k = t * MODES
    sq = jnp.sum((x[:, None, :] - mean_k[None]) ** 2, axis=-1)
    logc = jnp.log(W)[None] - 0.5 * sq / var - jnp.log(2 * jnp.pi * var)
    return jax.nn.softmax(logc, axis=-1)


def analytic_velocity(x, t):
    var = _var_t(t)
    mean_k = t * MODES
    resp = responsibilities(x, t)
    coef = (t * MODE_STD**2 - (1.0 - t)) / var
    v_k = MODES[None] + coef * (x[:, None, :] - mean_k[None])
    return jnp.sum(resp[:, :, None] * v_k, axis=1)


def make_step_fn(N, eps):
    def step_fn(x, sampling_idx, rng):
        t = sampling_idx.astype(jnp.float32) / N
        dt = 1.0 / N
        v = analytic_velocity(x, t)
        drift = v * (1.0 + eps * t) - eps * x
        g = jnp.sqrt(jnp.maximum(2.0 * eps * (1.0 - t), 0.0))
        z = jax.random.normal(rng, x.shape)
        next_x = x + drift * dt + g * jnp.sqrt(dt) * z
        x1_hat = jnp.clip(x + (1.0 - t) * v, -1, 1)
        return next_x, x1_hat

    return step_fn


def reward_fn(x1_hat):
    # normalized Q toward the goal mode
    q = -2.0 * jnp.sum((x1_hat - Q_GOAL) ** 2, axis=-1)
    return q  # already O(1); FKDJax lambda scales it


def run_ess(lmbda, N, eps, num_particles, adaptive, rng):
    fkd = FKDJax(
        potential_type=PotentialType.DIFF,
        lmbda=lmbda,
        num_particles=num_particles,
        adaptive_resampling=adaptive,
        resample_frequency=1,
        resampling_t_start=0,
        resampling_t_end=N - 1,
        time_steps=N,
    )
    rng, k = jax.random.split(rng)
    init = jax.random.normal(k, (num_particles, 2))
    _, aux = fkd_sample(
        fkd,
        init_latents=init,
        step_fn=make_step_fn(N, eps),
        reward_fn=reward_fn,
        rng=rng,
        return_aux=True,
    )
    return np.array(aux["ess"]), np.array(aux["did_resample"])


def main():
    N = 100
    eps = 1.0
    num_particles = 256
    lambdas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    rng = jax.random.PRNGKey(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    steps = np.arange(N)

    for lam in lambdas:
        rng, k = jax.random.split(rng)
        ess, did = run_ess(lam, N, eps, num_particles, adaptive=True, rng=k)
        ax1.plot(steps, ess, label=f"λ={lam}")
        # mark resample steps for the mid lambda only, to keep it readable
        if lam == 2.0:
            rs_steps = steps[did > 0.5]
            ax1.scatter(rs_steps, ess[did > 0.5], s=10, color="k", zorder=5,
                        label="resample (λ=2)")

    ax1.axhline(num_particles, ls="--", c="gray", lw=1)
    ax1.axhline(0.5 * num_particles, ls=":", c="red", lw=1,
                label="resample threshold (N/2)")
    ax1.set_xlabel("sampling step (0=noise → N=data)")
    ax1.set_ylabel("ESS")
    ax1.set_title(f"ESS vs step (adaptive, N={N}, eps={eps}, P={num_particles})")
    ax1.set_ylim(0, num_particles * 1.05)
    ax1.legend(fontsize=8)

    # Non-adaptive (resample every step) for contrast
    for lam in lambdas:
        rng, k = jax.random.split(rng)
        ess, _ = run_ess(lam, N, eps, num_particles, adaptive=False, rng=k)
        ax2.plot(steps, ess, label=f"λ={lam}")
    ax2.axhline(num_particles, ls="--", c="gray", lw=1)
    ax2.set_xlabel("sampling step")
    ax2.set_ylabel("ESS (pre-resample weights)")
    ax2.set_title("ESS vs step (non-adaptive: resample every step)")
    ax2.set_ylim(0, num_particles * 1.05)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("fkd_ess_curve.png", dpi=120)
    print("Saved fkd_ess_curve.png")

    # Console summary
    print("\nlambda |  mean ESS  | min ESS | resample steps")
    for lam in lambdas:
        rng, k = jax.random.split(rng)
        ess, did = run_ess(lam, N, eps, num_particles, adaptive=True, rng=k)
        print(f"{lam:>6} | {ess.mean():>9.1f} | {ess.min():>7.1f} | "
              f"{int(did.sum())}/{N}")


if __name__ == "__main__":
    main()
