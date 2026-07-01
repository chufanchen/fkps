"""Pure-JAX Feynman-Kac Diffusion (FKD) steering mechanism.

A JAX port of the PyTorch FKD class from fkd_steering.py (arXiv 2501.06848).
Fully JIT-compatible: resample() uses jax.lax.cond instead of Python if/else,
so the entire FKD loop can live inside lax.scan and vmap.

FKDJax is a pure-functional stateless object — it holds only static config
(potential type, schedule mask, hyperparameters). All per-step mutable state
lives in the FKDState namedtuple. Reward computation is done by the caller;
resample() takes pre-computed rs_candidates directly.
"""

from enum import Enum
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"
    RT = "rt"


POTENTIAL_DIFF = 0
POTENTIAL_MAX = 1
POTENTIAL_ADD = 2
POTENTIAL_RT = 3

_POTENTIAL_TO_INT = {
    PotentialType.DIFF: POTENTIAL_DIFF,
    PotentialType.MAX: POTENTIAL_MAX,
    PotentialType.ADD: POTENTIAL_ADD,
    PotentialType.RT: POTENTIAL_RT,
}


class FKDState(NamedTuple):
    population_rs: jnp.ndarray
    product_of_potentials: jnp.ndarray


class FKDStepInfo(NamedTuple):
    """Per-step diagnostics returned by resample()."""

    ess: jnp.ndarray          # effective sample size (in [1, num_particles])
    did_resample: jnp.ndarray  # 1.0 if particles were resampled this step
    logw_std: jnp.ndarray     # std of log-weights across particles (= lambda*std(diff)
    #                           for DIFF). ~0 => uniform weights => no steering.
    diff_std: jnp.ndarray     # std of the raw potential exponent across particles
    #                           (log-weight / lambda); the "are the diffs the same?"
    #                           signal, independent of lambda.
    w_max: jnp.ndarray        # largest normalized weight (1/N => uniform, 1 => collapse)


def _compute_weights(
    potential_id: int,
    lmbda: float,
    rs_candidates: jnp.ndarray,
    population_rs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute importance weights and (possibly updated) rs_candidates."""

    def _diff(args):
        rc, pr = args
        return jnp.exp(lmbda * (rc - pr)), rc

    def _max(args):
        rc, pr = args
        rc_new = jnp.maximum(rc, pr)
        return jnp.exp(lmbda * rc_new), rc_new

    def _add(args):
        rc, pr = args
        rc_new = rc + pr
        return jnp.exp(lmbda * rc_new), rc_new

    def _rt(args):
        rc, pr = args
        return jnp.exp(lmbda * rc), rc

    w, rs_out = jax.lax.switch(
        potential_id, [_diff, _max, _add, _rt], (rs_candidates, population_rs)
    )
    return w, rs_out


def _do_resample(rng, w, latents, rs_candidates, product_of_potentials):
    indices = jax.random.categorical(
        rng, jnp.log(jnp.maximum(w, 1e-30)), shape=(latents.shape[0],)
    )
    return (
        latents[indices],
        rs_candidates[indices],
        product_of_potentials[indices] * w[indices],
    )


def _no_resample(rng, w, latents, rs_candidates, product_of_potentials):
    return latents, rs_candidates, product_of_potentials


class FKDJax:
    """Pure-JAX FKD steering, fully JIT-compatible.

    Stateless: holds only static config. All per-step state lives in FKDState.
    Reward computation is the caller's responsibility — resample() takes
    pre-computed rs_candidates (shape: (num_particles,)) directly.
    """

    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_min_value: float = 0.0,
    ) -> None:
        self.potential_type = PotentialType(potential_type)
        self.potential_id = _POTENTIAL_TO_INT[self.potential_type]
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.time_steps = time_steps

        interval = np.arange(resampling_t_start, resampling_t_end + 1, resample_frequency)
        interval = np.append(interval, time_steps - 1)
        mask = np.zeros(time_steps, dtype=bool)
        for idx in interval:
            if 0 <= idx < time_steps:
                mask[idx] = True
        self.should_resample_mask = jnp.array(mask)

        self.final_step_correction = self.potential_type in (
            PotentialType.MAX, PotentialType.ADD, PotentialType.RT
        )

        self._initial_state = FKDState(
            population_rs=jnp.ones(num_particles) * reward_min_value,
            product_of_potentials=jnp.ones(num_particles),
        )

    def init_state(self) -> FKDState:
        return self._initial_state

    def resample(
        self,
        state: FKDState,
        *,
        sampling_idx: jnp.ndarray,
        latents: jnp.ndarray,
        rs_candidates: jnp.ndarray,
        rng: jax.Array,
    ) -> Tuple[jnp.ndarray, FKDState, "FKDStepInfo"]:
        """JIT-compatible resample. sampling_idx can be a traced integer.

        Args:
            state: Current FKD state.
            sampling_idx: Current step index (traced).
            latents: Particle latents to resample, shape (num_particles, ...).
            rs_candidates: Pre-computed rewards, shape (num_particles,).
            rng: PRNG key for categorical sampling.

        Returns:
            (resampled_latents, new_state, step_info) where step_info holds the
            effective sample size and whether resampling actually fired.
        """
        should_resample = self.should_resample_mask[sampling_idx]
        is_final = sampling_idx == (self.time_steps - 1)

        population_rs = state.population_rs
        product_of_potentials = state.product_of_potentials

        w, rs_out = _compute_weights(
            self.potential_id, self.lmbda, rs_candidates, population_rs
        )

        if self.final_step_correction:
            w_final = jnp.exp(self.lmbda * rs_out) / product_of_potentials
            w = jnp.where(is_final, w_final, w)

        w = jnp.clip(w, 0, 1e10)
        w = jnp.where(jnp.isnan(w), 0.0, w)

        # Effective sample size — always computed for diagnostics.
        normalized_w = w / jnp.maximum(w.sum(), 1e-30)
        ess = 1.0 / jnp.maximum((normalized_w ** 2).sum(), 1e-30)

        # Weight-spread diagnostics: is FKD actually distinguishing particles?
        logw = jnp.log(jnp.maximum(w, 1e-30))
        logw_std = logw.std()
        lam = jnp.maximum(jnp.abs(self.lmbda), 1e-8)
        diff_std = logw_std / lam
        w_max = normalized_w.max()

        if self.adaptive_resampling:
            do_resample = (ess < 0.5 * self.num_particles) | is_final
        else:
            do_resample = jnp.bool_(True)

        active = should_resample & do_resample
        resampled_latents, new_rs, new_product = jax.lax.cond(
            active,
            _do_resample,
            _no_resample,
            rng, w, latents, rs_out, product_of_potentials,
        )

        new_rs = jnp.where(should_resample, new_rs, population_rs)
        new_product = jnp.where(should_resample, new_product, product_of_potentials)

        new_state = FKDState(
            population_rs=new_rs,
            product_of_potentials=new_product,
        )
        info = FKDStepInfo(
            ess=ess,
            did_resample=active.astype(jnp.float32),
            logw_std=logw_std,
            diff_std=diff_std,
            w_max=w_max,
        )
        return resampled_latents, new_state, info


def fkd_sample(
    fkd: "FKDJax",
    *,
    init_latents: jnp.ndarray,
    step_fn,
    reward_fn,
    rng: jax.Array,
    return_aux: bool = False,
):
    """Generic FKD-steered diffusion sampling driver.

    Diffusion-framework agnostic: works with DDPM, flow matching, DDIM, etc.
    The host policy provides two callbacks; FKD owns the particle resampling.

    Args:
        fkd: A configured FKDJax instance (defines num steps via time_steps).
        init_latents: Initial particles, shape (num_particles, *latent_shape).
            For diffusion this is the prior noise.
        step_fn: Callable (latents, sampling_idx, rng) -> (next_latents, x0_hat).
            One reverse step of the host diffusion policy. `sampling_idx` is a
            traced int in [0, time_steps). `x0_hat` is the clean-sample estimate
            used to score particles (shape (num_particles, *latent_shape)).
        reward_fn: Callable (x0_hat) -> rs_candidates of shape (num_particles,).
            Typically normalized Q(s, x0_hat).
        rng: PRNG key.
        return_aux: If True, also return per-step (ess, did_resample) diagnostics.

    Returns:
        final_latents (num_particles, *latent_shape), or
        (final_latents, aux_dict) if return_aux.
    """
    num_steps = fkd.time_steps
    sampling_indices = jnp.arange(num_steps)
    init_state = fkd.init_state()

    def scan_fn(carry, sampling_idx):
        latents, fkd_st, rng_ = carry
        rng_, step_rng, fkd_rng = jax.random.split(rng_, 3)

        next_latents, x0_hat = step_fn(latents, sampling_idx, step_rng)
        rs_candidates = reward_fn(x0_hat)

        resampled, new_st, step_info = fkd.resample(
            fkd_st,
            sampling_idx=sampling_idx,
            latents=next_latents,
            rs_candidates=rs_candidates,
            rng=fkd_rng,
        )

        if return_aux:
            aux = {
                "ess": step_info.ess,
                "did_resample": step_info.did_resample,
                "logw_std": step_info.logw_std,
                "diff_std": step_info.diff_std,
                "w_max": step_info.w_max,
                "rs_mean": rs_candidates.mean(),
                "rs_std": rs_candidates.std(),
            }
        else:
            aux = None
        return (resampled, new_st, rng_), aux

    (final_latents, _, _), aux = jax.lax.scan(
        scan_fn, (init_latents, init_state, rng), sampling_indices, length=num_steps
    )

    if return_aux:
        return final_latents, aux
    return final_latents


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    num_particles = 8
    pixels = [0.1, 0.3, 0.5, 0.7, 0.9, 0.05, 0.95, 0.4]
    x0s = jnp.array(pixels, dtype=jnp.float32).reshape(num_particles, 1, 1)

    def reward_function(x):
        return -0.5 * x.sum(axis=(1, 2))

    fkd = FKDJax(
        potential_type=PotentialType.DIFF,
        lmbda=10.0,
        num_particles=num_particles,
        adaptive_resampling=False,
        resample_frequency=1,
        resampling_t_start=-1,
        resampling_t_end=100,
        time_steps=100,
    )
    state = fkd.init_state()

    rs = reward_function(x0s)
    rng = jax.random.PRNGKey(0)

    resampled_latents, state, _ = fkd.resample(
        state,
        sampling_idx=jnp.int32(0),
        latents=x0s,
        rs_candidates=rs,
        rng=rng,
    )

    print("Pixels:     ", pixels)
    print("Rewards:    ", rs.tolist())
    w = jnp.exp(10.0 * (rs - 0.0))
    print("Weights:    ", w.tolist())
    print("Normalized: ", (w / w.sum()).tolist())
    print("Resampled:  ", resampled_latents.squeeze().tolist())

    fig, axs = plt.subplots(2, num_particles)
    axs[0, 0].set_title("Initial")
    axs[1, 0].set_title("Resampled")

    for i in range(num_particles):
        axs[0, i].imshow(np.array(x0s[i]), cmap="gray", vmin=0, vmax=1)
        axs[1, i].imshow(np.array(resampled_latents[i]), cmap="gray", vmin=0, vmax=1)
        axs[1, i].axis("off")
        axs[0, i].axis("off")

    out_path = "resampled_examples_jax.png"
    plt.savefig(out_path)
    print("Saved resampled examples to:", out_path)
