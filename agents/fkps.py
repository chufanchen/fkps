"""Feynman-Kac Particle Steering (FKPS).

Training: BC DDPM actor + IQL critic/value (decoupled, like QGF).
Inference: DDPM reverse sampling steered by FKD particle resampling using Q(s, a).

The actor never sees RL signal during training — it is a pure behavior-cloning
diffusion model. The critic provides guidance only at test time through FKD's
importance-weighted resampling of particles.
"""

import copy
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.common import aggregate_q, get_flat_batch
from agents.fkd_jax import FKDJax, PotentialType
from utils.diffusion import (
    DDPM,
    FourierFeatures,
    cosine_beta_schedule,
    vp_beta_schedule,
)
from utils.flax_utils import TrainState, expectile_loss, target_update
from utils.networks import MLP, Value


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


class FKPSAgent(flax.struct.PyTreeNode):
    """Feynman-Kac Particle Steering with DDPM actor.

    Training is fully decoupled:
      - Actor: pure BC DDPM noise-prediction loss ||eps - eps_pred||^2.
      - Critic/Value: standard IQL (separate TrainStates, separate optimizers).

    Inference:
      - DDPM ancestral sampling with FKD particle resampling at each step.
      - Q(s, a) from the target critic is used as the FKD reward signal.
    """

    rng: Any
    policy: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    config: dict = flax.struct.field(pytree_node=False)
    betas: Any
    alphas: Any
    alpha_hats: Any
    q_mean: Any
    q_std: Any

    def _aggregate_q(self, qs):
        return aggregate_q(qs, self.config)

    def _get_flat_batch(self, batch):
        return get_flat_batch(batch, self.config)

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------

    def policy_loss(self, batch, policy_params=None, rng=None):
        """Pure BC DDPM noise-prediction loss."""
        if rng is None:
            rng = self.rng
        if policy_params is None:
            policy_params = self.policy.params

        batch_actions, _, _, _, valid_w = self._get_flat_batch(batch)

        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.randint(
            t_rng, batch_actions.shape[:-1], 1, self.config["diffusion_steps"] + 1
        )
        noise = jax.random.normal(noise_rng, batch_actions.shape)

        alpha_hats = self.alpha_hats[t]
        t_expanded = jnp.expand_dims(t, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch_actions + alpha_2 * noise

        eps_pred = self.policy(
            batch["observations"], noisy_actions, t_expanded, params=policy_params
        )

        bc_loss = (jnp.square(noise - eps_pred).mean(axis=-1) * valid_w).mean()
        return bc_loss, {"bc_loss": bc_loss}

    def critic_loss(self, batch, critic_params=None):
        """IQL critic (Q) loss."""
        H = self.config.get("horizon_length", 1)
        batch_actions, next_obs, rewards, masks, valid_w = self._get_flat_batch(batch)
        next_v = self.value(next_obs)
        target_q = rewards + (self.config["discount"] ** H) * masks * next_v
        qs = self.critic(batch["observations"], batch_actions, params=critic_params)
        critic_loss = (((qs - target_q[None]) ** 2) * valid_w).mean()
        return critic_loss, {"critic_loss": critic_loss, "q": qs[0].mean()}

    def value_loss(self, batch, value_params=None):
        """IQL value (V) loss with expectile regression."""
        batch_actions, _, _, _, valid_w = self._get_flat_batch(batch)
        qs = self.target_critic(batch["observations"], batch_actions)
        q = self._aggregate_q(qs)
        v = self.value(batch["observations"], params=value_params)
        value_loss = (expectile_loss(q - v, self.config["expectile"]) * valid_w).mean()
        return value_loss, {"value_loss": value_loss, "v": v.mean()}

    @jax.jit
    def update(self, batch):
        new_rng, policy_rng = jax.random.split(self.rng, 2)

        new_policy, policy_info = self.policy.apply_loss_fn(
            loss_fn=lambda p: self.policy_loss(batch, p, rng=policy_rng)
        )
        new_critic, critic_info = self.critic.apply_loss_fn(
            loss_fn=lambda p: self.critic_loss(batch, p)
        )
        new_target_critic = target_update(
            self.critic, self.target_critic, self.config["tau"]
        )
        new_value, value_info = self.value.apply_loss_fn(
            loss_fn=lambda p: self.value_loss(batch, p)
        )

        # Update running Q statistics for FKD normalization
        batch_actions, _, _, _, _ = self._get_flat_batch(batch)
        qs = self.target_critic(batch["observations"], batch_actions)
        q = self._aggregate_q(qs)
        ema = self.config.get("q_stats_ema", 0.99)
        new_q_mean = ema * self.q_mean + (1 - ema) * q.mean()
        new_q_std = ema * self.q_std + (1 - ema) * jnp.maximum(q.std(), 1e-6)

        return self.replace(
            rng=new_rng,
            policy=new_policy,
            critic=new_critic,
            target_critic=new_target_critic,
            value=new_value,
            q_mean=new_q_mean,
            q_std=new_q_std,
        ), {**policy_info, **critic_info, **value_info}

    # ------------------------------------------------------------------
    # Inference — DDPM reverse sampling
    # ------------------------------------------------------------------

    def ddpm_sampler(self, rng, observations, noise):
        """Standard DDPM ancestral sampling (no FKD)."""
        batch_size = observations.shape[0]
        input_time_proto = jnp.ones((*noise.shape[:-1], 1))

        def fn(input_tuple, t):
            current_x, rng_ = input_tuple
            input_time = input_time_proto * t

            eps_pred = self.policy(observations, current_x, input_time)

            x0_hat = (
                1
                / jnp.sqrt(self.alpha_hats[t])
                * (current_x - jnp.sqrt(1 - self.alpha_hats[t]) * eps_pred)
            )
            if self.config["clip_sampler"]:
                x0_hat = jnp.clip(x0_hat, -1, 1)
                current_x = (
                    1
                    / (1 - self.alpha_hats[t])
                    * (
                        jnp.sqrt(self.alpha_hats[t - 1])
                        * (1 - self.alphas[t])
                        * x0_hat
                        + jnp.sqrt(self.alphas[t])
                        * (1 - self.alpha_hats[t - 1])
                        * current_x
                    )
                )
            else:
                current_x = x0_hat

            rng_, key_ = jax.random.split(rng_, 2)
            z = jax.random.normal(key_, shape=(batch_size,) + current_x.shape[1:])
            sigmas_t = jnp.sqrt(1 - self.alphas[t])
            current_x = current_x + (t > 1) * (sigmas_t * z)

            return (current_x, rng_), ()

        rng, denoise_key = jax.random.split(rng, 2)
        output_tuple, () = jax.lax.scan(
            fn,
            (noise, denoise_key),
            jnp.arange(self.config["diffusion_steps"], 0, -1),
            unroll=self.config["diffusion_steps"],
        )
        return output_tuple[0]

    def _build_fkd(self):
        diffusion_steps = self.config["diffusion_steps"]
        fkd_t_end = int(self.config["fkd_t_end"])
        if fkd_t_end < 0:
            fkd_t_end = diffusion_steps - 1
        return FKDJax(
            potential_type=PotentialType(str(self.config["fkd_potential"])),
            lmbda=float(self.config["fkd_lambda"]),
            num_particles=int(self.config["fkd_num_particles"]),
            adaptive_resampling=bool(self.config["fkd_adaptive"]),
            resample_frequency=int(self.config["fkd_resample_freq"]),
            resampling_t_start=int(self.config["fkd_t_start"]),
            resampling_t_end=fkd_t_end,
            time_steps=diffusion_steps,
        )

    def _fkd_ddpm_sample_single(self, obs, noise, rng):
        """FKD-steered DDPM reverse sampling for a single observation.

        obs: (obs_dim,), noise: (num_particles, action_dim), rng: PRNGKey.
        Returns: best action (action_dim,).
        """
        num_particles = int(self.config["fkd_num_particles"])
        diffusion_steps = self.config["diffusion_steps"]
        obs_repeated = jnp.broadcast_to(obs[None], (num_particles, *obs.shape))

        q_mean = self.q_mean
        q_std = jnp.maximum(self.q_std, 1e-6)
        fkd = self._build_fkd()
        fkd_state = fkd.init_state()
        t_schedule = jnp.arange(diffusion_steps, 0, -1)
        sampling_indices = jnp.arange(diffusion_steps)

        def scan_fn(carry, scan_inputs):
            current_x, fkd_st, rng_ = carry
            t, sampling_idx = scan_inputs

            input_time = jnp.ones((*current_x.shape[:-1], 1)) * t
            eps_pred = self.policy(obs_repeated, current_x, input_time)

            x0_hat = (
                1
                / jnp.sqrt(self.alpha_hats[t])
                * (current_x - jnp.sqrt(1 - self.alpha_hats[t]) * eps_pred)
            )
            if self.config["clip_sampler"]:
                x0_hat = jnp.clip(x0_hat, -1, 1)
                current_x = (
                    1
                    / (1 - self.alpha_hats[t])
                    * (
                        jnp.sqrt(self.alpha_hats[t - 1])
                        * (1 - self.alphas[t])
                        * x0_hat
                        + jnp.sqrt(self.alphas[t])
                        * (1 - self.alpha_hats[t - 1])
                        * current_x
                    )
                )
            else:
                current_x = x0_hat

            rng_, noise_key, fkd_key = jax.random.split(rng_, 3)
            z = jax.random.normal(noise_key, current_x.shape)
            sigmas_t = jnp.sqrt(1 - self.alphas[t])
            current_x = current_x + (t > 1) * (sigmas_t * z)

            # Compute normalized Q as reward signal for FKD
            qs = self.target_critic(obs_repeated, x0_hat)
            rs_candidates = (self._aggregate_q(qs) - q_mean) / q_std

            current_x, fkd_st = fkd.resample(
                fkd_st,
                sampling_idx=sampling_idx,
                latents=current_x,
                rs_candidates=rs_candidates,
                rng=fkd_key,
            )

            return (current_x, fkd_st, rng_), None

        (final_x, _, _), _ = jax.lax.scan(
            scan_fn,
            (noise, fkd_state, rng),
            (t_schedule, sampling_indices),
            length=diffusion_steps,
        )

        actions = jnp.clip(final_x, -1, 1)
        qs = self.target_critic(obs_repeated, actions)
        q = self._aggregate_q(qs)
        return actions[jnp.argmax(q)]

    @jax.jit
    def _sample_actions_fkd(self, observations, seed):
        """Batched FKD-steered sampling: vmap over observations."""
        num_particles = int(self.config["fkd_num_particles"])
        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"]
            if self.config.get("action_chunking", False)
            else 1
        )
        batch_size = observations.shape[0]

        keys = jax.random.split(seed, batch_size + 1)
        noise_key, sample_keys = keys[0], keys[1:]
        noise = jax.random.normal(
            noise_key, (batch_size, num_particles, full_action_dim)
        )

        return jax.vmap(self._fkd_ddpm_sample_single)(
            observations, noise, sample_keys
        )

    @jax.jit
    def _sample_actions_bc(self, observations, *, seed):
        """Standard DDPM sampling without FKD steering."""
        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"]
            if self.config.get("action_chunking", False)
            else 1
        )
        noise_key, sampler_key = jax.random.split(seed)
        noise = jax.random.normal(noise_key, (observations.shape[0], full_action_dim))
        actions = self.ddpm_sampler(sampler_key, observations, noise)
        return jnp.clip(actions, -1, 1)

    def sample_actions(self, observations, *, seed, **kwargs):
        single_obs = observations.ndim == 1
        if single_obs:
            observations = observations[None, :]

        num_particles = int(self.config.get("fkd_num_particles", 0))
        if num_particles <= 1:
            actions = self._sample_actions_bc(observations, seed=seed)
        else:
            actions = self._sample_actions_fkd(observations, seed)

        if single_obs:
            actions = actions.squeeze(axis=0)
        return actions

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, policy_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = ex_actions.shape[-1]
        H = config.get("horizon_length", 1)
        if config.get("action_chunking", False):
            ex_full_actions = jnp.concatenate([ex_actions] * H, axis=-1)
        else:
            ex_full_actions = ex_actions
        full_action_dim = ex_full_actions.shape[-1]

        config = dict(config)
        config["action_dim"] = action_dim

        activation_fn = mish

        # Actor: DDPM noise-prediction network
        preprocess_time_cls = partial(
            FourierFeatures, output_size=config["time_dim"], learnable=True
        )
        cond_model_cls = partial(
            MLP,
            hidden_dims=config["actor_hidden_dims"],
            activation=activation_fn,
            activate_final=False,
        )
        base_model_cls = partial(
            MLP,
            hidden_dims=tuple(list(config["actor_hidden_dims"]) + [full_action_dim]),
            activation=activation_fn,
            layer_norm=config.get("actor_layer_norm", False),
            activate_final=False,
        )
        policy_def = DDPM(
            time_preprocess_cls=preprocess_time_cls,
            cond_encoder_cls=cond_model_cls,
            reverse_encoder_cls=base_model_cls,
        )
        ex_times = jnp.zeros((ex_observations.shape[0], 1))
        policy_params = policy_def.init(
            policy_key, ex_observations, ex_full_actions, ex_times
        )["params"]
        policy = TrainState.create(
            policy_def, policy_params, tx=optax.adam(learning_rate=config["bc_lr"])
        )

        # Critic: IQL Q-network
        critic_def = Value(
            network_class=config.get("value_network_class", "MLP"),
            network_kwargs={
                **config["value_network_kwargs"],
            },
            num_ensembles=config["num_qs"],
        )
        critic_params = critic_def.init(critic_key, ex_observations, ex_full_actions)[
            "params"
        ]
        critic = TrainState.create(
            critic_def, critic_params, tx=optax.adam(learning_rate=config["critic_lr"])
        )
        target_critic = TrainState.create(critic_def, critic_params)

        # Value: IQL V-network
        value_def = Value(
            network_class=config.get("value_network_class", "MLP"),
            network_kwargs={
                **config["value_network_kwargs"],
            },
            num_ensembles=1,
        )
        value_params = value_def.init(value_key, ex_observations)["params"]
        value = TrainState.create(
            value_def, value_params, tx=optax.adam(learning_rate=config["value_lr"])
        )

        # Noise schedule
        beta_schedule = config["beta_schedule"]
        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(config["diffusion_steps"]))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, config["diffusion_steps"])
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(config["diffusion_steps"]))
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}")

        betas = jnp.concatenate([jnp.zeros((1,)), betas])
        alphas = 1 - betas
        alpha_hats = jnp.cumprod(alphas)

        config_dict = flax.core.FrozenDict(**config)
        return cls(
            rng=rng,
            policy=policy,
            critic=critic,
            target_critic=target_critic,
            value=value,
            config=config_dict,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hats,
            q_mean=jnp.float32(0.0),
            q_std=jnp.float32(1.0),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="fkps",
            # Common hyperparameters.
            batch_size=256,
            bc_lr=3e-4,
            critic_lr=3e-4,
            value_lr=3e-4,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_network_class="MLP",
            value_network_kwargs=dict(
                hidden_dims=(512, 512, 512, 512),
                layer_norm=True,
            ),
            # n-step returns & action chunking.
            horizon_length=1,
            action_chunking=False,
            # RL hyperparameters.
            num_qs=2,
            q_aggregation="min",
            discount=0.99,
            expectile=0.9,
            tau=0.005,
            # Diffusion hyperparameters.
            diffusion_steps=10,
            time_dim=64,
            beta_schedule="vp",
            clip_sampler=True,
            # FKD inference hyperparameters.
            fkd_num_particles=64,
            fkd_lambda=1.0,
            fkd_potential="diff",
            fkd_adaptive=True,
            fkd_resample_freq=1,
            fkd_t_start=0,
            fkd_t_end=-1,
            q_stats_ema=0.99,
        )
    )
    return config
