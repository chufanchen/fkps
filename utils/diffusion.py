"""Diffusion model utilities for DDPM-based agents (QSM, DAC).

Adapted from ~/repos/qam/utils/diffusion.py.
"""
from functools import partial
from typing import Type
import jax
import flax.linen as nn
import jax.numpy as jnp


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    betas = jnp.linspace(
        beta_start, beta_end, timesteps
    )
    return betas

def vp_beta_schedule(timesteps):
    """Variance-preserving (VP) beta schedule."""
    t = jnp.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    betas = 1 - alpha
    return betas


class FourierFeatures(nn.Module):
    """Learnable or fixed Fourier feature encoding for timesteps."""

    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)

class OnehotEmbed(nn.Module):
    """
    use learnable null features for un-conditional generation
    """
    # used for conditional embedding
    # the scale of continuous
    # use 0 as the unconditional label
    # set the unconditional feature as zeros = jnp.zeros(self.output_size)
    num_embeddings: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):  # x.shape = (B,)
        x -= 1
        # x takes value [0, N] -> [-1, N-1], treat [-1] as unc. label
        return jax.nn.one_hot(x, self.num_embeddings)  # x.shape = (B, num_embeddings)


class LearnableEmbed(nn.Module):
    """
    use learnable null features for un-conditional generation
    """
    # used for conditional embedding
    # the scale of continuous
    # use 0 as the unconditional label
    # set the unconditional feature as zeros = jnp.zeros(self.output_size)
    output_size: int = 16
    num_embeddings: int = 40
    zero_emtpy_feature: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.zero_emtpy_feature:  # [0, 1, ... , N] -> [-1, 0, 1, ... , N-1], zero = null feature
            x -= 1
        embed = nn.Embed(num_embeddings=self.num_embeddings + int(not self.zero_emtpy_feature),
                         features=self.output_size,
                         embedding_init=orthogonal_init())(x.astype(int))  # (B, output_size)
        if self.zero_emtpy_feature:
            return jnp.where(x[:, jnp.newaxis] < 0, jnp.zeros(self.output_size), embed)
        return embed


class FourierEmbed(nn.Module):
    output_size: int = 16
    rescale: float = 1.

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = x[:, jnp.newaxis]
        # assert x.shape[-1] == 1  # (B, 1) x in [0, ..., N] (N+1 classes)
        half_dim = self.output_size // 2
        f = jnp.log(10000) / (half_dim - 1)
        f = jnp.exp(jnp.arange(half_dim) * -f)  # e.g. [1.   , 0.268, 0.072, 0.019, 0.005, 0.001, 0.   , 0.   ]
        f = x * f / self.rescale
        fourier_embed = jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)
        # assign zero embeddings to the null class
        return jnp.where(x <= 0, jnp.zeros(self.output_size), fourier_embed)

class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model for action generation.

    Takes (observation, noisy_action, timestep) and predicts the noise.
    Architecture: time -> Fourier features -> cond_encoder; then
    [noisy_action, obs, cond] -> reverse_encoder -> noise_prediction.
    """

    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray, time: jnp.ndarray):
        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff)
        reverse_input = jnp.concatenate([a, s, cond], axis=-1)
        return self.reverse_encoder_cls()(reverse_input)


# noise_pred_apply_fn is static since the network structure is immutable
@partial(jax.jit, static_argnames=('noise_pred_apply_fn', 'T', 'repeat_last_step', 'clip_sampler'))
def ddpm_sampler(rng, noise_pred_apply_fn, params, T, observations, alphas, alpha_hats, sample_temperature,
                 repeat_last_step, clip_sampler, prior: jnp.array):
    batch_size = observations.shape[0]
    input_time_proto = jnp.ones((*prior.shape[:-1], 1))

    def fn(input_tuple, t):
        current_x, rng_ = input_tuple
        # input_time = jnp.expand_dims(jnp.array([t]).repeat(current_x.shape[0]), axis=1)
        input_time = input_time_proto * t
        # noise_model(s, a, time, training=training) in DDPM

        eps_pred = noise_pred_apply_fn(params, observations, current_x, input_time, training=False)

        # re-parameterization of distribution (4) in DDPM paper
        x0_hat = 1 / jnp.sqrt(alpha_hats[t]) * (current_x - jnp.sqrt(1 - alpha_hats[t]) * eps_pred)

        if clip_sampler:
            x0_hat = jnp.clip(x0_hat, -1, 1)

        # equation (7) in DDPM paper, equivalent to (7), here using x0_hat just for clipping
        current_x = 1 / (1 - alpha_hats[t]) * (jnp.sqrt(alpha_hats[t - 1]) * (1 - alphas[t]) * x0_hat +
                                               jnp.sqrt(alphas[t]) * (1 - alpha_hats[t - 1]) * current_x)

        # alpha_1 = 1 / jnp.sqrt(alphas[t])
        # alpha_2 = ((1 - alphas[t]) / (jnp.sqrt(1 - alpha_hats[t])))
        # current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng_, key_ = jax.random.split(rng_, 2)
        z = jax.random.normal(key_, shape=(batch_size,) + current_x.shape[1:])
        z_scaled = sample_temperature * z

        # sigmas_t = jnp.sqrt((1 - alphas[t]) * (1 - alpha_hats[t - 1]) / (1 - alpha_hats[t]))
        sigmas_t = jnp.sqrt((1 - alphas[t]))  # both have similar results
        # remove the noise of t = 0
        current_x = current_x + (t > 1) * (sigmas_t * z_scaled)

        return (current_x, rng_), ()

    rng, denoise_key = jax.random.split(rng, 2)
    output_tuple, () = jax.lax.scan(fn,
                                    (prior, denoise_key),
                                    jnp.arange(T, 0, -1),  # since alphas <- cat[0, alphas]; betas <- cat[1, betas]
                                    unroll=T)  # unroll = 5

    for _ in range(repeat_last_step):
        output_tuple, () = fn(output_tuple, 0)

    action_0, rng = output_tuple
    # action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng

@partial(jax.jit, static_argnames=('noise_pred_apply_fn', 'critic_tar_apply_fn', 'T',
                                   'repeat_last_step', 'clip_sampler'))
def ddpm_sampler_with_q_guidance(rng, noise_pred_apply_fn, params, critic_tar_apply_fn, q_params, guidance_scale,
                                 T, observations, alphas, alpha_hats, sample_temperature,
                                 repeat_last_step, clip_sampler, prior: jnp.array):
    batch_size = observations.shape[0]
    input_time_proto = jnp.ones((*prior.shape[:-1], 1))
    q_grad_fn = jax.vmap(jax.grad(lambda x0, s: critic_tar_apply_fn(q_params, s, x0).mean(axis=0)))

    def fn(input_tuple, t):
        current_x, rng_ = input_tuple
        # input_time = jnp.expand_dims(jnp.array([t]).repeat(current_x.shape[0]), axis=1)
        input_time = input_time_proto * t
        # noise_model(s, a, time, training=training) in DDPM

        eps_pred = noise_pred_apply_fn(params, observations, current_x, input_time, training=False)

        # q_grad = q_grad_fn(current_x, observations)
        rng_, key_ = jax.random.split(rng_)
        q_grad = q_grad_fn(current_x, observations)

        # q_norm = jnp.abs(critic_tar_apply_fn(q_params, observations, current_x)).mean()
        # q_grad /= q_norm
        # q_grad /= (1e-3 + jnp.abs(q_grad).mean())
        # q_grad /= jnp.linalg.norm(q_grad, axis=-1, keepdims=True)

        eps_pred -= guidance_scale * jnp.sqrt(1 - alpha_hats[t]) * q_grad

        x0_hat = 1 / jnp.sqrt(alpha_hats[t]) * (current_x - jnp.sqrt(1 - alpha_hats[t]) * eps_pred)

        if clip_sampler:
            x0_hat = jnp.clip(x0_hat, -1, 1)

        # equation (7) in DDPM paper, equivalent to (7), here using x0_hat just for clipping
        current_x = 1 / (1 - alpha_hats[t]) * (jnp.sqrt(alpha_hats[t - 1]) * (1 - alphas[t]) * x0_hat +
                                               jnp.sqrt(alphas[t]) * (1 - alpha_hats[t - 1]) * current_x)

        rng_, key_ = jax.random.split(rng_)
        z = jax.random.normal(key_, shape=(batch_size,) + current_x.shape[1:])
        z_scaled = sample_temperature * z

        # sigmas_t = jnp.sqrt((1 - alphas[t]) * (1 - alpha_hats[t - 1]) / (1 - alpha_hats[t]))
        sigmas_t = jnp.sqrt((1 - alphas[t]))  # both have similar results
        # remove the noise of t = 0
        current_x = current_x + (t > 1) * (sigmas_t * z_scaled)

        # if clip_sampler:
        #     current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng_), ()

    rng, denoise_key = jax.random.split(rng, 2)
    output_tuple, () = jax.lax.scan(fn,
                                    (prior, denoise_key),
                                    jnp.arange(T, 0, -1),  # since alphas <- cat[0, alphas]; betas <- cat[1, betas]
                                    unroll=T)

    for _ in range(repeat_last_step):
        output_tuple, () = fn(output_tuple, 0)

    action_0, rng = output_tuple
    # action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng


@partial(jax.jit, static_argnames=('noise_pred_apply_fn', 'T', 'repeat_last_step', 'clip_sampler',
                                   'ddim_step', 'ddim_eta'))
def ddim_sampler(rng, noise_pred_apply_fn, params, T, observations, alphas, alpha_hats,
                 sample_temperature, repeat_last_step, clip_sampler, prior: jnp.array, ddim_step, ddim_eta=0):
    """
    dim(obs_with_prompt) = dim(obs) + 1, the prompt is one scalar value
    """
    batch_size = observations.shape[0]
    c = T // ddim_step  # jump step
    ddim_time_seq = jnp.concatenate([jnp.arange(T, 0, -c), jnp.array([0])])
    input_time_proto = jnp.ones((*prior.shape[:-1], 1))

    def fn(input_tuple, i):
        # work on the last dim
        current_x, rng_ = input_tuple

        t, prev_t = ddim_time_seq[i], ddim_time_seq[i + 1]

        input_time = input_time_proto * t

        # input_time = jnp.expand_dims(jnp.array([t]).repeat(current_x.shape[0]), axis=1)

        # if guidance_scale > 0:
        #     # use classifier-free guidance when guidance_scale > 1
        #     # treat the last dimension as the class token
        #     # observations is of dim [B*repeats, d_obs]
        #     unc_obs = jnp.concatenate([obs_with_prompt[:, :-1], jnp.zeros((batch_size, 1))], axis=-1)
        #
        #     eps_c = noise_pred_apply_fn(params, obs_with_prompt, current_x, input_time, training=False)
        #     eps_unc = noise_pred_apply_fn(params, unc_obs, current_x, input_time, training=False)
        #     eps_pred = eps_unc + guidance_scale * (eps_c - eps_unc)
        #
        #     eps_pred = rescale_noise_cfg(eps_pred, eps_c, guidance_rescale=guidance_rescale)
        #
        # else:
        eps_pred = noise_pred_apply_fn(params, observations, current_x, input_time, training=False)

        # sigmas_t = ddim_eta * jnp.sqrt((1 - alpha_hats[prev_t]) / (1 - alpha_hats[t]) * (1 - alphas[t]))
        sigmas_t = ddim_eta * jnp.sqrt((1 - alphas[t]))  # both have similar results

        alpha_1 = 1 / jnp.sqrt(alphas[t])
        alpha_2 = jnp.sqrt(1 - alpha_hats[t])
        alpha_3 = jnp.sqrt(1 - alpha_hats[prev_t] - sigmas_t ** 2)

        current_x = alpha_1 * (current_x - alpha_2 * eps_pred) + alpha_3 * eps_pred

        rng_, key_ = jax.random.split(rng_, 2)
        z = jax.random.normal(key_, shape=(batch_size,) + current_x.shape[1:])
        z_scaled = sample_temperature * z
        current_x = current_x + sigmas_t * z_scaled

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng_), ()

    rng, denoise_key = jax.random.split(rng, 2)
    output_tuple, () = jax.lax.scan(fn, (prior, denoise_key), jnp.arange(len(ddim_time_seq) - 1),
                                    unroll=T)

    for _ in range(repeat_last_step):
        output_tuple, () = fn(output_tuple, 0)

    action_0, rng = output_tuple
    # action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng