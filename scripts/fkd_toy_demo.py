"""Toy 2D demo of FKD-steered diffusion sampling.

No training required: the "behavior policy" is a fixed mixture of Gaussians whose
diffusion score is available in closed form, so we get a *perfect* DDPM denoiser
analytically. FKD then steers the particle population toward a high-Q region using
Q(a) as the potential.

This exercises the framework-agnostic `fkd_sample` driver from agents.fkd_jax and
visualizes the particle cloud at every diffusion step, comparing:
  - plain DDPM (lambda = 0, no steering)
  - FKD-steered DDPM (lambda > 0)

Run:
    python scripts/fkd_toy_demo.py
Outputs:
    fkd_toy_process.png   — particle snapshots across denoising steps
    fkd_toy_compare.png   — final samples, unsteered vs steered
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from agents.fkd_jax import FKDJax, PotentialType, fkd_sample
from utils.diffusion import vp_beta_schedule


# ----------------------------------------------------------------------
# Toy behavior distribution: a mixture of Gaussians in 2D.
# ----------------------------------------------------------------------
MODES = jnp.array(
    [
        [0.6, 0.6],
        [-0.6, 0.6],
        [-0.6, -0.6],
        [0.6, -0.6],
    ]
)
MODE_STD = 0.08
MIX_W = jnp.ones(MODES.shape[0]) / MODES.shape[0]

# Q favors the top-right mode. Use a smooth quadratic centered there.
Q_GOAL = jnp.array([0.6, 0.6])


def q_function(a):
    """Toy critic: higher near Q_GOAL. Shape (N, 2) -> (N,)."""
    return -2.0 * jnp.sum((a - Q_GOAL) ** 2, axis=-1)


# ----------------------------------------------------------------------
# Analytic diffusion score of the mixture (the "perfect BC denoiser").
# ----------------------------------------------------------------------
def mixture_eps(a_t, alpha_hat_t):
    """Exact eps-prediction for the forward-diffused Gaussian mixture.

    Forward: a_t = sqrt(ah) a_0 + sqrt(1-ah) eps,  a_0 ~ sum_k w_k N(mu_k, s^2 I).
    Then a_t ~ sum_k w_k N(sqrt(ah) mu_k, var_k I) with
        var_k = ah * s^2 + (1 - ah).
    eps_pred = -sqrt(1-ah) * score(a_t).
    """
    ah = alpha_hat_t
    mean_k = jnp.sqrt(ah) * MODES                      # (K, 2)
    var = ah * MODE_STD**2 + (1.0 - ah)                # scalar (shared isotropic)

    # log N_k(a_t) per particle: (N, K)
    diff = a_t[:, None, :] - mean_k[None, :, :]        # (N, K, 2)
    sq = jnp.sum(diff**2, axis=-1)                      # (N, K)
    log_comp = jnp.log(MIX_W)[None, :] - 0.5 * sq / var - jnp.log(2 * jnp.pi * var)
    # responsibilities
    resp = jax.nn.softmax(log_comp, axis=-1)           # (N, K)
    # score = sum_k resp_k * (mean_k - a_t) / var
    score = jnp.sum(resp[:, :, None] * (mean_k[None] - a_t[:, None, :]), axis=1) / var
    eps_pred = -jnp.sqrt(1.0 - ah) * score
    return eps_pred


# ----------------------------------------------------------------------
# DDPM reverse step using the analytic denoiser (matches fkps._ddpm_step).
# ----------------------------------------------------------------------
def make_step_fn(alphas, alpha_hats, diffusion_steps, clip=True):
    def step_fn(current_x, sampling_idx, rng):
        t = diffusion_steps - sampling_idx
        ah_t = alpha_hats[t]
        eps_pred = mixture_eps(current_x, ah_t)

        x0_hat = (current_x - jnp.sqrt(1 - ah_t) * eps_pred) / jnp.sqrt(ah_t)
        if clip:
            x0_hat = jnp.clip(x0_hat, -1, 1)
            mean = (
                jnp.sqrt(alpha_hats[t - 1]) * (1 - alphas[t]) * x0_hat
                + jnp.sqrt(alphas[t]) * (1 - alpha_hats[t - 1]) * current_x
            ) / (1 - ah_t)
        else:
            mean = x0_hat

        z = jax.random.normal(rng, mean.shape)
        sigma_t = jnp.sqrt(1 - alphas[t])
        next_x = mean + (t > 1) * (sigma_t * z)
        return next_x, x0_hat

    return step_fn


# ----------------------------------------------------------------------
# A traced sampler that records the particle trajectory for plotting.
# ----------------------------------------------------------------------
def sample_with_trajectory(fkd, init_latents, step_fn, reward_fn, rng):
    """Like fkd_sample but returns the full (steps+1, N, 2) particle history."""
    num_steps = fkd.time_steps
    state = fkd.init_state()
    history = [init_latents]

    latents = init_latents
    for sampling_idx in range(num_steps):
        rng, step_rng, fkd_rng = jax.random.split(rng, 3)
        next_latents, x0_hat = step_fn(latents, jnp.int32(sampling_idx), step_rng)
        rs = reward_fn(x0_hat)
        latents, state, _ = fkd.resample(
            state,
            sampling_idx=jnp.int32(sampling_idx),
            latents=next_latents,
            rs_candidates=rs,
            rng=fkd_rng,
        )
        history.append(latents)
    return jnp.stack(history)  # (num_steps+1, N, 2)


def build_fkd(lmbda, num_particles, diffusion_steps, t_start=0):
    return FKDJax(
        potential_type=PotentialType.DIFF,
        lmbda=lmbda,
        num_particles=num_particles,
        adaptive_resampling=False,
        resample_frequency=1,
        resampling_t_start=t_start,
        resampling_t_end=diffusion_steps - 1,
        time_steps=diffusion_steps,
    )


def main():
    num_particles = 512
    diffusion_steps = 50
    rng = jax.random.PRNGKey(0)

    betas = jnp.array(vp_beta_schedule(diffusion_steps))
    betas = jnp.concatenate([jnp.zeros((1,)), betas])
    alphas = 1 - betas
    alpha_hats = jnp.cumprod(alphas)

    step_fn = make_step_fn(alphas, alpha_hats, diffusion_steps, clip=True)

    # Fixed Q stats for normalization (precomputed over the mode locations).
    q_vals = q_function(MODES)
    q_mean = q_vals.mean()
    q_std = jnp.maximum(q_vals.std(), 1e-6)

    def reward_fn(x0_hat):
        return (q_function(x0_hat) - q_mean) / q_std

    rng, n1, n2 = jax.random.split(rng, 3)
    init = jax.random.normal(n1, (num_particles, 2))

    # Unsteered (lambda = 0) and steered (lambda > 0), same init noise.
    fkd_plain = build_fkd(0.0, num_particles, diffusion_steps)
    fkd_steer = build_fkd(4.0, num_particles, diffusion_steps)

    rng, r_plain, r_steer = jax.random.split(rng, 3)
    hist_plain = sample_with_trajectory(fkd_plain, init, step_fn, reward_fn, r_plain)
    hist_steer = sample_with_trajectory(fkd_steer, init, step_fn, reward_fn, r_steer)

    hist_plain = np.array(hist_plain)
    hist_steer = np.array(hist_steer)

    # ---- Figure 1: particle snapshots across the denoising process ----
    snap_idx = np.linspace(0, diffusion_steps, 6, dtype=int)
    fig, axs = plt.subplots(2, len(snap_idx), figsize=(3 * len(snap_idx), 6))
    modes_np = np.array(MODES)
    goal_np = np.array(Q_GOAL)
    for col, s in enumerate(snap_idx):
        for row, (hist, title) in enumerate(
            [(hist_plain, "plain DDPM"), (hist_steer, "FKD-steered")]
        ):
            ax = axs[row, col]
            ax.scatter(hist[s, :, 0], hist[s, :, 1], s=4, alpha=0.3, color="C0")
            ax.scatter(modes_np[:, 0], modes_np[:, 1], marker="x", color="k", s=60)
            ax.scatter(goal_np[0], goal_np[1], marker="*", color="red", s=200,
                       edgecolor="k", zorder=5)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"step {s}/{diffusion_steps}")
            if col == 0:
                ax.set_ylabel(title, fontsize=12)
    fig.suptitle(
        "FKD particle steering: noise -> mixture, steered toward the red-star mode",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig("fkd_toy_process.png", dpi=120)
    print("Saved fkd_toy_process.png")

    # ---- Figure 2: final-sample comparison ----
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
    for ax, hist, title in [
        (axs2[0], hist_plain, "plain DDPM (lambda=0)"),
        (axs2[1], hist_steer, "FKD-steered (lambda=4)"),
    ]:
        final = hist[-1]
        ax.scatter(final[:, 0], final[:, 1], s=6, alpha=0.4, color="C0")
        ax.scatter(modes_np[:, 0], modes_np[:, 1], marker="x", color="k", s=80)
        ax.scatter(goal_np[0], goal_np[1], marker="*", color="red", s=250,
                   edgecolor="k", zorder=5)
        # fraction of particles near the goal mode
        frac = np.mean(np.linalg.norm(final - goal_np, axis=-1) < 0.2)
        ax.set_title(f"{title}\n{frac:.0%} on goal mode")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
    fig2.tight_layout()
    fig2.savefig("fkd_toy_compare.png", dpi=120)
    print("Saved fkd_toy_compare.png")


if __name__ == "__main__":
    main()
