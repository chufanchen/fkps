"""Flow-Matching Feynman-Kac Particle Steering (Flow-FKPS).

Training: BC flow-matching actor (velocity regression) + IQL critic/value
          (decoupled, like QGF).
Inference: the deterministic flow ODE is converted to a marginal-preserving SDE
          (arXiv 2410.02217), so the reverse process is stochastic. FKD particle
          resampling then steers the population using Q(s, a) as the potential.

ODE -> SDE conversion (linear interpolant a_t = (1-t) a0 + t a1, a0 ~ N(0,I)):
    score(x,t) = -(x - t v) / (1 - t)
    Choosing diffusion g(t)^2 = 2 eps (1 - t) keeps marginals AND stays finite:
        dx = [v (1 + eps t) - eps x] dt + sqrt(2 eps (1 - t)) dW
    eps = 0 recovers the deterministic ODE (pure flow); eps > 0 adds the
    stochasticity FKD needs to diversify resampled particles.
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.common import aggregate_q, get_flat_batch
from agents.fkd_jax import FKDJax, PotentialType, fkd_sample
from utils.activation import get_activation
from utils.flax_utils import TrainState, expectile_loss, target_update
from utils.networks import ActorFlowField, Value


class FlowFKPSAgent(flax.struct.PyTreeNode):
    """Flow-matching FKPS with an SDE reverse process.

    Training is fully decoupled:
      - Actor: pure BC flow-matching loss ||v - v_pred||^2.
      - Critic/Value: standard IQL (separate TrainStates, separate optimizers).

    Inference:
      - Marginal-preserving SDE sampling (churn eps) with FKD resampling.
      - Q(s, a) from the target critic is the FKD reward signal.

    Test-time guidance: supports the same eval sweep as QGF. The eval harness
    calls sample_actions(guidance_weight=w) for each w in FLAGS.guidance_weights
    and reports the best; here `guidance_weight` overrides the FKD lambda (the
    steering strength), so the sweep is over FKD steering intensity.
    """

    support_guidance = True

    rng: Any
    policy: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    config: dict = flax.struct.field(pytree_node=False)
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
        """Flow-matching BC loss (velocity regression)."""
        if rng is None:
            rng = self.rng
        if policy_params is None:
            policy_params = self.policy.params

        batch_actions, _, _, _, valid_w = self._get_flat_batch(batch)

        eps_rng, time_rng = jax.random.split(rng, 2)
        a1 = batch_actions
        a0 = jax.random.normal(eps_rng, a1.shape)
        t = (
            jax.random.randint(
                time_rng, (a1.shape[0],), 0, self.config["denoise_steps"] + 1
            ).astype(jnp.float32)
            / self.config["denoise_steps"]
        )
        tv = t[..., None]
        a_t = a0 * (1 - tv) + a1 * tv
        vel = a1 - a0

        pred_vel = self.policy(batch["observations"], a_t, t, params=policy_params)
        bc_loss = (jnp.square(vel - pred_vel).mean(axis=-1) * valid_w).mean()
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
    def total_loss(self, batch, grad_params=None, rng=None):
        if rng is None:
            rng = self.rng
        info = {}
        bc_loss, policy_info = self.policy_loss(batch, rng=rng)
        for k, v in policy_info.items():
            info[f"policy/{k}"] = v
        critic_loss, critic_info = self.critic_loss(batch)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v
        value_loss, value_info = self.value_loss(batch)
        for k, v in value_info.items():
            info[f"value/{k}"] = v
        return bc_loss + critic_loss + value_loss, info

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

        # Running Q statistics for FKD normalization.
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
    # Inference — flow SDE sampling
    # ------------------------------------------------------------------

    def _flow_sde_step(self, obs_rep, current_x, sampling_idx, rng):
        """One marginal-preserving SDE step. Returns (next_latents, x1_hat).

        sampling_idx in [0, denoise_steps): 0 is pure noise (t=0), the last
        index is nearly clean (t -> 1). x1_hat is the one-Euler clean-action
        estimate used to score particles with Q.
        """
        N = self.config["denoise_steps"]
        eps = self.config["flow_noise_eps"]
        t0 = sampling_idx.astype(jnp.float32) / N
        t1 = (sampling_idx.astype(jnp.float32) + 1.0) / N
        dt = 1.0 / N

        ti = jnp.ones((current_x.shape[0],)) * t0
        v = self.policy(obs_rep, current_x, ti)

        # Marginal-preserving SDE drift/diffusion (see module docstring):
        #   drift = v + (g^2/2) score,  score = -(x - t v)/(1 - t),
        #   with g^2 = 2 eps (1 - t)  =>  drift = v (1 + eps t) - eps x.
        drift = v * (1.0 + eps * t0) - eps * current_x
        # Exact diffusion variance over [t0, t1]: integral of g^2 = 2 eps (1 - t)
        #   = eps ((1 - t0)^2 - (1 - t1)^2). The Euler-Maruyama g(t0)^2 dt rule
        # overshoots near t -> 1 (g^2 vanishes linearly), injecting up to 2x too
        # much noise on the final step; this closed form removes that bias.
        noise_var = eps * ((1.0 - t0) ** 2 - (1.0 - t1) ** 2)
        noise_scale = jnp.sqrt(jnp.maximum(noise_var, 0.0))
        z = jax.random.normal(rng, current_x.shape)
        next_x = current_x + drift * dt + noise_scale * z

        # Clean-action estimate: x1_hat = x + (1-t) v = E[a1 | a_t = x] exactly
        # (flow analogue of DDPM's Tweedie x0_hat). Clipped to the action box.
        x1_hat = jnp.clip(current_x + (1.0 - t0) * v, -1.0, 1.0)
        return next_x, x1_hat

    def _build_fkd(self, lmbda=None):
        denoise_steps = self.config["denoise_steps"]
        fkd_t_end = int(self.config["fkd_t_end"])
        if fkd_t_end < 0:
            fkd_t_end = denoise_steps - 1
        if lmbda is None:
            lmbda = float(self.config["fkd_lambda"])
        return FKDJax(
            potential_type=PotentialType(str(self.config["fkd_potential"])),
            lmbda=lmbda,
            num_particles=int(self.config["fkd_num_particles"]),
            adaptive_resampling=bool(self.config["fkd_adaptive"]),
            resample_frequency=int(self.config["fkd_resample_freq"]),
            resampling_t_start=int(self.config["fkd_t_start"]),
            resampling_t_end=fkd_t_end,
            time_steps=denoise_steps,
        )

    def _fkd_flow_sample_single(self, obs, noise, rng, lmbda):
        """FKD-steered flow SDE sampling for a single observation.

        obs: (obs_dim,), noise: (num_particles, action_dim), rng: PRNGKey.
        lmbda: FKD steering strength (traced scalar).
        Returns: best action (action_dim,).
        """
        num_particles = int(self.config["fkd_num_particles"])
        obs_rep = jnp.broadcast_to(obs[None], (num_particles, *obs.shape))

        q_mean = self.q_mean
        q_std = jnp.maximum(self.q_std, 1e-6)
        fkd = self._build_fkd(lmbda=lmbda)

        def step_fn(current_x, sampling_idx, step_rng):
            return self._flow_sde_step(obs_rep, current_x, sampling_idx, step_rng)

        def reward_fn(x1_hat):
            qs = self.target_critic(obs_rep, x1_hat)
            return (self._aggregate_q(qs) - q_mean) / q_std

        final_x = fkd_sample(
            fkd,
            init_latents=noise,
            step_fn=step_fn,
            reward_fn=reward_fn,
            rng=rng,
        )

        actions = jnp.clip(final_x, -1, 1)
        qs = self.target_critic(obs_rep, actions)
        q = self._aggregate_q(qs)
        return actions[jnp.argmax(q)]

    @jax.jit
    def _sample_actions_fkd(self, observations, seed, lmbda):
        """Batched FKD-steered sampling: vmap over observations.

        lmbda is shared across the batch (in_axes=None), so the eval harness can
        sweep FKD steering strength without rebuilding/retracing per observation.
        """
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
        return jax.vmap(self._fkd_flow_sample_single, in_axes=(0, 0, 0, None))(
            observations, noise, sample_keys, lmbda
        )

    def _fkd_aux_single(self, obs, noise, rng, lmbda):
        """Run the FKD sampler for one obs, return per-step (ess, did_resample)."""
        num_particles = int(self.config["fkd_num_particles"])
        obs_rep = jnp.broadcast_to(obs[None], (num_particles, *obs.shape))
        q_mean = self.q_mean
        q_std = jnp.maximum(self.q_std, 1e-6)
        fkd = self._build_fkd(lmbda=lmbda)

        def step_fn(current_x, sampling_idx, step_rng):
            return self._flow_sde_step(obs_rep, current_x, sampling_idx, step_rng)

        def reward_fn(x1_hat):
            qs = self.target_critic(obs_rep, x1_hat)
            return (self._aggregate_q(qs) - q_mean) / q_std

        _, aux = fkd_sample(
            fkd,
            init_latents=noise,
            step_fn=step_fn,
            reward_fn=reward_fn,
            rng=rng,
            return_aux=True,
        )
        return aux["ess"], aux["did_resample"], aux["rs_std"]

    @jax.jit
    def fkd_ess_curve(self, observations, seed):
        """Batch-averaged FKD diagnostics per sampling step.

        Returns a dict of arrays of shape (denoise_steps,):
            ess          - effective sample size, averaged over the batch
            did_resample - fraction of batch that resampled at each step
            rs_std       - std of the particle rewards (steering signal strength)
        Use during training/eval to watch how FKD behaves as Q improves.
        """
        num_particles = int(self.config["fkd_num_particles"])
        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"]
            if self.config.get("action_chunking", False)
            else 1
        )
        batch_size = observations.shape[0]
        lmbda = jnp.float32(self.config["fkd_lambda"])

        keys = jax.random.split(seed, batch_size + 1)
        noise_key, sample_keys = keys[0], keys[1:]
        noise = jax.random.normal(
            noise_key, (batch_size, num_particles, full_action_dim)
        )
        ess, did, rs_std = jax.vmap(self._fkd_aux_single, in_axes=(0, 0, 0, None))(
            observations, noise, sample_keys, lmbda
        )
        return {
            "ess": ess.mean(axis=0),
            "did_resample": did.mean(axis=0),
            "rs_std": rs_std.mean(axis=0),
        }

    @jax.jit
    def _sample_actions_bc(self, observations, *, seed):
        """Deterministic flow ODE sampling (no FKD, no churn)."""
        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"]
            if self.config.get("action_chunking", False)
            else 1
        )
        N = self.config["denoise_steps"]
        a = jax.random.normal(seed, (observations.shape[0], full_action_dim))
        dt = 1.0 / N

        def step(a, t_idx):
            ti = jnp.ones((a.shape[0],)) * (t_idx / N)
            v = self.policy(observations, a, ti)
            return a + v * dt, None

        a, _ = jax.lax.scan(step, a, jnp.arange(N), length=N)
        return jnp.clip(a, -1, 1)

    def sample_actions(
        self, observations, *, seed, guidance_weight=None,
        rejection_sampling=1, **kwargs
    ):
        single_obs = observations.ndim == 1
        if single_obs:
            observations = observations[None, :]

        num_particles = int(self.config.get("fkd_num_particles", 0))
        if num_particles <= 1:
            actions = self._sample_actions_bc(observations, seed=seed)
        else:
            # guidance_weight (from the eval sweep) overrides the FKD lambda.
            lmbda = (
                float(self.config["fkd_lambda"])
                if guidance_weight is None
                else float(guidance_weight)
            )
            actions = self._sample_actions_fkd(
                observations, seed, jnp.float32(lmbda)
            )

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

        activation_fn = get_activation(config["activation"])
        mlp_kwargs = dict(activation=activation_fn, layer_norm=config["use_layer_norm"])
        ex_t = jnp.zeros(ex_observations.shape[0])

        # Actor: flow-matching velocity field.
        policy_def = ActorFlowField(
            config["actor_hidden_dims"], full_action_dim, mlp_kwargs=mlp_kwargs
        )
        policy_params = policy_def.init(
            policy_key, ex_observations, ex_full_actions, ex_t
        )["params"]
        policy = TrainState.create(
            policy_def, policy_params, tx=optax.adam(learning_rate=config["bc_lr"])
        )

        # Critic / value: IQL.
        critic_def = Value(
            network_class=config["value_network_class"],
            network_kwargs={
                **config["value_network_kwargs"],
                "activation": activation_fn,
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

        value_def = Value(
            network_class=config["value_network_class"],
            network_kwargs={
                **config["value_network_kwargs"],
                "activation": activation_fn,
            },
            num_ensembles=1,
        )
        value_params = value_def.init(value_key, ex_observations)["params"]
        value = TrainState.create(
            value_def, value_params, tx=optax.adam(learning_rate=config["value_lr"])
        )

        config_dict = flax.core.FrozenDict(**config)
        return cls(
            rng=rng,
            policy=policy,
            critic=critic,
            target_critic=target_critic,
            value=value,
            config=config_dict,
            q_mean=jnp.float32(0.0),
            q_std=jnp.float32(1.0),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="flow_fkps",
            # Common hyperparameters.
            batch_size=256,
            bc_lr=3e-4,
            critic_lr=3e-4,
            value_lr=3e-4,
            actor_hidden_dims=(512, 512, 512, 512),
            use_layer_norm=1,
            activation="gelu",
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
            # Flow / SDE hyperparameters.
            # NOTE: the marginal-preserving SDE is only faithful if Euler-Maruyama
            # is fine enough. Verified via scripts/test_flow_fkps_units.py:
            #   denoise_steps=10  is faithful only for flow_noise_eps<=0.5;
            #   flow_noise_eps>=1.0 needs denoise_steps>=50.
            # Too-coarse + too-much-churn pushes actions off the data manifold and
            # breaks BOTH the actor output and FKD's Q scoring.
            denoise_steps=100,
            # Churn: 0 = deterministic ODE, >0 = marginal-preserving SDE noise.
            flow_noise_eps=1.0,
            # FKD inference hyperparameters.
            fkd_num_particles=64,
            fkd_lambda=1.0,
            fkd_potential="diff",
            fkd_adaptive=True,
            fkd_resample_freq=1,
            fkd_t_start=20,
            fkd_t_end=-1,
            q_stats_ema=0.99,
        )
    )
    return config
