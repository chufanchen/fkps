"""IQL with DDPM diffusion actor.

Training: AWR-weighted DDPM noise-prediction loss (actor) + IQL critic/value.
Inference: DDPM reverse sampling (SDE) with optional clipping.
          Optionally steered by FKD particle resampling using Q(s, a) as potential.
Architecture follows DCGQL (ModuleDict, FourierFeatures, DDPM noise net).
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
import torch
from agents.fkd_steering import FKD, PotentialType
from utils.diffusion import (
    DDPM,
    FourierFeatures,
    cosine_beta_schedule,
    vp_beta_schedule,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, Value


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


class IQLDDPMAgent(flax.struct.PyTreeNode):
    """IQL with a DDPM diffusion actor.

    Critic & value are trained identically to standard IQL.
    The actor is a DDPM noise-prediction network trained with the
    advantage-weighted diffusion loss:
        L = E_t[ exp(alpha * A(s,a)) * ||eps - eps_theta(s, a_t, t)||^2 ]
    where a_t is the noised action and A = Q - V.

    Inference uses the standard DDPM reverse process (ancestral sampling).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    betas: Any
    alphas: Any
    alpha_hats: Any

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def value_loss(self, batch, grad_params):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
        else:
            batch_actions = batch["actions"][..., 0, :]
        valid_w = batch["valid"][..., -1]

        qs = self.network.select("target_critic")(batch["observations"], batch_actions)
        q = qs.min(axis=0)
        v = self.network.select("value")(batch["observations"], params=grad_params)
        value_loss = (
            self.expectile_loss(q - v, q - v, self.config["expectile"]) * valid_w
        ).mean()
        return value_loss, {"value_loss": value_loss, "v_mean": v.mean()}

    def critic_loss(self, batch, grad_params):
        H = self.config["horizon_length"]
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
        else:
            batch_actions = batch["actions"][..., 0, :]
        next_obs = batch["next_observations"][..., -1, :]
        rewards = batch["rewards"][..., -1]
        masks = batch["masks"][..., -1]
        valid_w = batch["valid"][..., -1]

        next_v = self.network.select("value")(next_obs)
        target_q = rewards + (self.config["discount"] ** H) * masks * next_v
        qs = self.network.select("critic")(
            batch["observations"], batch_actions, params=grad_params
        )
        critic_loss = (jnp.square(qs - target_q) * valid_w).mean()
        return critic_loss, {"critic_loss": critic_loss, "q_mean": qs.mean()}

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
        else:
            batch_actions = batch["actions"][..., 0, :]
        valid_w = batch["valid"][..., -1]

        # Compute AWR advantages
        qs = self.network.select(
            "target_critic" if self.config["target_extraction"] else "critic"
        )(batch["observations"], batch_actions)
        q = qs.min(axis=0)
        v = self.network.select("value")(batch["observations"])
        adv = q - v
        exp_a = jnp.exp(adv * self.config["alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Forward diffusion: noise the actions
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

        eps_pred = self.network.select("actor")(
            batch["observations"], noisy_actions, t_expanded, params=grad_params
        )

        diffusion_loss = jnp.square(noise - eps_pred).mean(axis=-1)
        actor_loss = (diffusion_loss * exp_a * valid_w).mean()

        return actor_loss, {
            "actor_loss": actor_loss,
            "diffusion_loss": (diffusion_loss * valid_w).mean(),
            "adv": adv.mean(),
            "exp_a": exp_a.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        return value_loss + critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    # ------------------------------------------------------------------
    # Inference — DDPM reverse sampling
    # ------------------------------------------------------------------

    def ddpm_sampler(self, rng, observations, noise):
        batch_size = observations.shape[0]
        input_time_proto = jnp.ones((*noise.shape[:-1], 1))

        def fn(input_tuple, t):
            current_x, rng_ = input_tuple
            input_time = input_time_proto * t

            eps_pred = self.network.select("actor")(
                observations, current_x, input_time
            )

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

    @staticmethod
    def _jax_to_torch(x):
        return torch.from_dlpack(jax.dlpack.to_dlpack(x))

    @staticmethod
    def _torch_to_jax(x):
        return jnp.from_dlpack(x.detach())

    def fkd_ddpm_sampler(
        self,
        rng,
        observations,
        noise,
        *,
        num_particles,
        fkd_lambda,
        fkd_potential,
        fkd_adaptive,
        fkd_resample_freq,
        fkd_t_start,
        fkd_t_end,
    ):
        """DDPM reverse sampling with FKD particle resampling.

        Runs the standard DDPM reverse process but maintains num_particles
        candidates per observation. At resampling steps, FKD evaluates
        Q(s, x0_hat) on the denoised action prediction and resamples the
        particle population proportionally to exp(lambda * Q).
        """
        diffusion_steps = self.config["diffusion_steps"]
        obs_repeated = jnp.broadcast_to(
            observations[None], (num_particles, *observations.shape)
        )

        def q_reward_fn(x0_preds_torch):
            a_jax = self._torch_to_jax(x0_preds_torch)
            qs = self.network.select("target_critic")(obs_repeated, a_jax)
            q = qs.min(axis=0)
            q = (q - q.mean()) / (q.std() + 1e-6)
            return self._jax_to_torch(q)

        if fkd_t_end < 0:
            fkd_t_end = diffusion_steps - 1

        fkd = FKD(
            potential_type=PotentialType(fkd_potential),
            lmbda=fkd_lambda,
            num_particles=num_particles,
            adaptive_resampling=fkd_adaptive,
            resample_frequency=fkd_resample_freq,
            resampling_t_start=fkd_t_start,
            resampling_t_end=fkd_t_end,
            time_steps=diffusion_steps,
            reward_fn=q_reward_fn,
            device=torch.device("cpu"),
        )

        current_x = noise
        rng, denoise_key = jax.random.split(rng, 2)
        input_time_proto = jnp.ones((*noise.shape[:-1], 1))

        # DDPM reverse: t from T down to 1 — sampling_idx from 0 to T-1
        for sampling_idx, t in enumerate(range(diffusion_steps, 0, -1)):
            input_time = input_time_proto * t
            eps_pred = self.network.select("actor")(
                obs_repeated, current_x, input_time
            )

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

            denoise_key, key_ = jax.random.split(denoise_key, 2)
            z = jax.random.normal(
                key_, shape=(num_particles,) + current_x.shape[1:]
            )
            sigmas_t = jnp.sqrt(1 - self.alphas[t])
            current_x = current_x + (t > 1) * (sigmas_t * z)

            # FKD resampling after the DDPM step:
            # latents = current_x (noisy actions at t-1)
            # x0_preds = x0_hat (denoised action estimate from this step)
            latents_torch = self._jax_to_torch(current_x)
            x0_preds_torch = self._jax_to_torch(x0_hat)
            resampled_latents, _ = fkd.resample(
                sampling_idx=sampling_idx,
                latents=latents_torch,
                x0_preds=x0_preds_torch,
            )
            current_x = self._torch_to_jax(resampled_latents)

        actions = jnp.clip(current_x, -1, 1)
        qs = self.network.select("target_critic")(obs_repeated, actions)
        q = qs.min(axis=0)
        return actions[jnp.argmax(q)]

    def sample_actions(self, observations, *, seed, **kwargs):
        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )

        single_obs = observations.ndim == 1
        if single_obs:
            observations = observations[None, :]

        num_particles = int(self.config.get("fkd_num_particles", 0))
        if num_particles <= 1:
            return self._sample_actions_bc(observations, seed=seed, single_obs=single_obs)

        fkd_lambda = float(self.config["fkd_lambda"])
        fkd_potential = str(self.config["fkd_potential"])
        fkd_adaptive = bool(self.config["fkd_adaptive"])
        fkd_resample_freq = int(self.config["fkd_resample_freq"])
        fkd_t_start = int(self.config["fkd_t_start"])
        fkd_t_end = int(self.config["fkd_t_end"])

        batch_size = observations.shape[0]
        all_actions = []
        for b in range(batch_size):
            obs_b = observations[b]
            seed, noise_key, sampler_key = jax.random.split(seed, 3)
            noise = jax.random.normal(
                noise_key, (num_particles, full_action_dim)
            )
            action_b = self.fkd_ddpm_sampler(
                sampler_key,
                obs_b,
                noise,
                num_particles=num_particles,
                fkd_lambda=fkd_lambda,
                fkd_potential=fkd_potential,
                fkd_adaptive=fkd_adaptive,
                fkd_resample_freq=fkd_resample_freq,
                fkd_t_start=fkd_t_start,
                fkd_t_end=fkd_t_end,
            )
            all_actions.append(action_b)

        actions = jnp.stack(all_actions, axis=0)
        if single_obs:
            actions = actions.squeeze(axis=0)
        return actions

    @jax.jit
    def _sample_actions_bc(self, observations, *, seed, single_obs):
        """Standard DDPM sampling without FKD steering."""
        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        noise_key, sampler_key = jax.random.split(seed)
        noise = jax.random.normal(noise_key, (observations.shape[0], full_action_dim))
        actions = self.ddpm_sampler(sampler_key, observations, noise)
        actions = jnp.clip(actions, -1, 1)
        if single_obs:
            actions = actions.squeeze(axis=0)
        return actions

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        action_dim = ex_actions.shape[-1]
        H = config["horizon_length"]
        if config["action_chunking"]:
            ex_full_actions = jnp.concatenate([ex_actions] * H, axis=-1)
        else:
            ex_full_actions = ex_actions
        full_action_dim = ex_full_actions.shape[-1]

        preprocess_time_cls = partial(
            FourierFeatures, output_size=config["time_dim"], learnable=True
        )
        cond_model_cls = partial(
            MLP,
            hidden_dims=config["actor_hidden_dims"],
            activation=mish,
            activate_final=False,
        )
        base_model_cls = partial(
            MLP,
            hidden_dims=tuple(list(config["actor_hidden_dims"]) + [full_action_dim]),
            activation=mish,
            layer_norm=config["actor_layer_norm"],
            activate_final=False,
        )

        actor_def = DDPM(
            time_preprocess_cls=preprocess_time_cls,
            cond_encoder_cls=cond_model_cls,
            reverse_encoder_cls=base_model_cls,
        )

        ex_times = jnp.zeros((ex_observations.shape[0], 1))
        critic_def = Value(
            network_class="MLP",
            network_kwargs=dict(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["value_layer_norm"],
            ),
            num_ensembles=config["num_qs"],
        )
        value_def = Value(
            network_class="MLP",
            network_kwargs=dict(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["value_layer_norm"],
            ),
            num_ensembles=1,
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_full_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_full_actions),
            ),
            value=(value_def, (ex_observations,)),
            actor=(actor_def, (ex_observations, ex_full_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(
            optax.clip_by_global_norm(max_norm=config["clip_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
        )
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

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

        config["action_dim"] = action_dim

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            alphas=alphas,
            alpha_hats=alpha_hats,
            betas=betas,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="iql_ddpm",
            # Common hyperparameters.
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),
            value_layer_norm=True,
            # n-step returns & action chunking.
            horizon_length=1,
            action_chunking=False,
            # RL hyperparameters.
            num_qs=2,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            # Diffusion hyperparameters.
            diffusion_steps=10,
            time_dim=64,
            beta_schedule="vp",
            clip_sampler=True,
            clip_grad_norm=1.0,
            # IQL actor hyperparameters.
            alpha=1.0,
            target_extraction=True,
            # FKD inference hyperparameters (passed to sample_actions).
            fkd_num_particles=64,
            fkd_lambda=1.0,
            fkd_potential="diff",
            fkd_adaptive=True,
            fkd_resample_freq=1,
            fkd_t_start=0,
            fkd_t_end=-1,
        )
    )
    return config
