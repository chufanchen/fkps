import copy
from collections import defaultdict
from functools import partial
from typing import List

import jax
import numpy as np
import tqdm


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def _is_test_time_guidance_agent(agent) -> bool:
    return getattr(agent, "support_guidance", False)


class SingleEnvBatchAdapter:
    """Adapter to present a single Gymnasium env as a batched (num_envs=1) env.

    Accepts batched actions with leading batch dimension 1 and always returns
    batched outputs, emulating vector env API sufficiently for evaluation.
    """

    def __init__(self, env):
        self._env = env
        self.num_envs = 1

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return np.expand_dims(obs, axis=0), info

    def step(self, actions):
        action = (
            actions[0]
            if isinstance(actions, np.ndarray)
            and actions.ndim >= 1
            and actions.shape[0] == 1
            else actions
        )
        obs, reward, terminated, truncated, info = self._env.step(action)
        return (
            np.expand_dims(obs, axis=0),
            np.array([reward], dtype=np.float32),
            np.array([terminated], dtype=np.bool_),
            np.array([truncated], dtype=np.bool_),
            info,
        )

    def render(self):
        return self._env.render()


def _convert_batched_infos_to_list(vector_infos, num_envs):
    """Unbatch a Gymnasium vector-env info dict into per-env info dicts.

    Args:
        vector_infos: Batched info from ``AsyncVectorEnv`` / ``SyncVectorEnv``.
            Each data key ``k`` is paired with an optional mask key ``_k`` (bool
            array of length ``num_envs``) indicating which envs set that key.
            Leaf values are length-``num_envs`` arrays. Gymnasium 1.0+ may nest
            this structure for dict-valued info (e.g. ``episode``, ``total``).

            Example (Gymnasium 0.29, ``num_envs=2``)::

                {
                    "success": array([1.0, 0.0]),
                    "_success": array([True, True]),
                }

            Example (Gymnasium 1.0+, ``num_envs=2``)::

                {
                    "total": {
                        "timesteps": array([10, 20]),
                        "_timesteps": array([True, True]),
                    },
                    "_total": array([True, True]),
                }

        num_envs: Number of parallel environments in the vector env.

    Returns:
        A list of length ``num_envs``. Each entry is a plain single-env info dict
        in the same shape a non-vectorized env would return, e.g.::

            [
                {"success": 1.0, "total": {"timesteps": 10}},
                {"success": 0.0, "total": {"timesteps": 20}},
            ]

        Keys whose mask is False for an env are omitted from that env's dict.
    """
    per_env = [dict() for _ in range(num_envs)]
    for key, value in vector_infos.items():
        # Keys prefixed with "_" are per-env presence masks, not data.
        if key.startswith("_"):
            continue

        mask = vector_infos.get(f"_{key}")
        if isinstance(value, dict):
            # Gymnasium 1.0+: nested info dicts are batched recursively.
            nested = _convert_batched_infos_to_list(value, num_envs)
            if mask is None:
                for idx, nested_info in enumerate(nested):
                    per_env[idx][key] = nested_info
            else:
                for idx, (nested_info, has_info) in enumerate(zip(nested, mask)):
                    if has_info:
                        per_env[idx][key] = nested_info
        elif isinstance(value, np.ndarray):
            if mask is None:
                mask = np.ones(num_envs, dtype=bool)
            for idx in range(num_envs):
                if mask[idx]:
                    per_env[idx][key] = value[idx]
        elif mask is None:
            for idx in range(num_envs):
                per_env[idx][key] = value
        else:
            for idx, has_info in enumerate(mask):
                if has_info:
                    per_env[idx][key] = value
    return per_env


def _vector_infos_to_list(infos, num_envs):
    """Convert env info to a list of per-env dicts."""
    if not isinstance(infos, dict):
        raise TypeError(f"Expected info dict, got {type(infos).__name__}")
    # Vector envs (AsyncVectorEnv) tag batched info with per-key "_<key>" masks.
    # Single-env dicts (via SingleEnvBatchAdapter) have no such keys.
    # Empty vector info ({}) also has no "_" keys; replicating it is correct.
    if any(k.startswith("_") for k in infos):
        return _convert_batched_infos_to_list(infos, num_envs)
    return [infos for _ in range(num_envs)]


def _prepare_actor(agent, guidance_weight, rejection_sampling):
    rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
    sample_actions = partial(supply_rng(agent.sample_actions, rng=rng))

    if _is_test_time_guidance_agent(agent):
        return partial(
            sample_actions,
            guidance_weight=guidance_weight,
            rejection_sampling=rejection_sampling,
        )

    # Standard action sampling for other agents
    assert (
        guidance_weight is None
    ), "guidance_weight is only supported for test time guidance agents"
    return partial(
        sample_actions,
        rejection_sampling=rejection_sampling,
    )


def run_episodes(
    agent,
    env,
    task_id=None,
    eval_gaussian=None,
    guidance_weight=None,
    should_render=False,
    video_frame_skip=3,
    rejection_sampling=1,
):
    """Shared rollout for sequential and vectorized environments (batch-first).

    Always treat inputs/outputs as batched. If `env` is single, use SingleEnvBatchAdapter.
    """

    if not hasattr(env, "num_envs"):
        env = SingleEnvBatchAdapter(env)
    if env.num_envs > 1 and should_render:
        raise ValueError("Rendering is only supported for single environments.")

    actor_fn = _prepare_actor(
        agent,
        guidance_weight=guidance_weight,
        rejection_sampling=rejection_sampling,
    )

    # Detect action chunking from agent config. `horizon_length` can be used for
    # n-step targets without implying that the policy emits an action chunk.
    horizon_length = int(agent.config.get("horizon_length", 1))
    action_chunking = bool(agent.config.get("action_chunking", False))
    action_dim = agent.config.get("action_dim", None)
    if action_dim is not None:
        action_dim = int(action_dim)
    rollout_horizon = horizon_length if action_chunking else 1

    observations, _ = env.reset(options=dict(task_id=task_id))

    num_envs = env.num_envs

    # Per-env state
    active = np.ones(num_envs, dtype=bool)
    returns = np.zeros(num_envs, dtype=np.float32)
    lengths = np.zeros(num_envs, dtype=np.int32)
    trajectories = [defaultdict(list) for _ in range(num_envs)]
    renders = [[] for _ in range(num_envs)]

    rng = np.random.default_rng()

    # Per-env action queues for action chunking (H > 1).
    action_queues = [[] for _ in range(num_envs)]

    while not np.all(~active):
        # Determine which envs need a new action chunk.
        # Keep stepping inactive vector sub-envs too. Gymnasium vector envs
        # auto-reset completed sub-envs, and we ignore their transitions below.
        need_chunk = [i for i in range(num_envs) if not action_queues[i]]
        if need_chunk:
            subset_obs = observations[need_chunk]
            raw = actor_fn(observations=subset_obs)
            raw = np.atleast_2d(np.array(raw))

            if rollout_horizon > 1:
                # raw: (len(need_chunk), H * action_dim) -> (len(need_chunk), H, action_dim)
                if action_dim is None:
                    if raw.shape[-1] % rollout_horizon != 0:
                        raise ValueError(
                            f"Cannot infer per-step action_dim: raw.shape[-1]={raw.shape[-1]}, "
                            f"rollout_horizon={rollout_horizon}"
                        )
                    action_dim = raw.shape[-1] // rollout_horizon
                chunks = raw.reshape(len(need_chunk), rollout_horizon, action_dim)
                for j, idx in enumerate(need_chunk):
                    action_queues[idx].extend(chunks[j])
            else:
                for j, idx in enumerate(need_chunk):
                    action_queues[idx].append(raw[j])

        # Pop one action per env from the queue.
        actions = np.array([action_queues[i].pop(0) for i in range(num_envs)])

        if eval_gaussian is not None:
            actions = rng.normal(loc=actions, scale=eval_gaussian)
        actions = np.clip(actions, -1, 1)

        next_observations, rewards, terminations, truncations, step_infos = env.step(
            actions
        )
        infos_per_step = _vector_infos_to_list(step_infos, num_envs)
        done_now = np.logical_or(terminations, truncations)

        for idx in range(num_envs):
            reward = rewards[idx]
            info = infos_per_step[idx]
            next_observation = next_observations[idx]

            if active[idx]:
                lengths[idx] += 1
                returns[idx] += reward

                if done_now[idx]:
                    if "final_observation" in info:
                        next_observation = info["final_observation"]
                    if "final_info" in info:
                        info = copy.deepcopy(info["final_info"])
                transition = dict(
                    observation=observations[idx],
                    next_observation=next_observation,
                    action=actions[idx],
                    reward=reward,
                    done=done_now[idx],
                    info=info,
                )
                add_to(trajectories[idx], copy.deepcopy(transition))

                if should_render and (
                    lengths[idx] % video_frame_skip == 0 or done_now[idx]
                ):
                    frame = env.render().copy()
                    renders[idx].append(frame)

                if done_now[idx]:
                    action_queues[idx] = []  # clear queue on episode end
                    active[idx] = False

            observations[idx] = np.asarray(next_observation).reshape(-1)

    if should_render:
        renders = [np.array(r) for r in renders]

    return trajectories, renders, returns, lengths


def eval_standard(
    agent,
    env,
    vec_eval_env=None,
    task_id=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_gaussian=None,
    guidance_weight=None,
    rejection_sampling=1,
):
    """Evaluate the agent in the environment with optimized execution.

    Returns a tuple: (stats, trajectories, renders)
    """
    trajs = []
    stats = defaultdict(list)
    renders = []

    batch_size = vec_eval_env.num_envs if vec_eval_env is not None else 1
    assert num_eval_episodes % batch_size == 0
    total_batches = num_eval_episodes // batch_size

    for _ in range(total_batches):
        traj_batch, _, returns, lengths = run_episodes(
            agent,
            vec_eval_env if vec_eval_env is not None else env,
            task_id,
            eval_gaussian,
            guidance_weight,
            should_render=False,
            video_frame_skip=video_frame_skip,
            rejection_sampling=rejection_sampling,
        )
        for idx in range(batch_size):
            info_flat = {
                "episode.return": returns[idx],
                "episode.length": lengths[idx],
                **flatten(traj_batch[idx]["info"][-1]),
            }
            add_to(stats, info_flat)
        trajs.extend(traj_batch)

    for _ in range(num_video_episodes):
        _, render_batch, _, _ = run_episodes(
            agent,
            env,
            task_id,
            eval_gaussian,
            guidance_weight,
            should_render=True,
            video_frame_skip=video_frame_skip,
            rejection_sampling=rejection_sampling,
        )
        renders.append(render_batch[0])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def eval_with_test_time_guidance(
    agent,
    eval_env,
    vec_eval_env,
    *,
    num_eval_episodes: int,
    rejection_sampling: int,
    guidance_weights: List[str],
    num_video_episodes: int,
):
    """Rollout over multiple guidance weights."""
    renders = []
    eval_metrics = {}
    max_return = -np.inf
    guidance_weights = [float(x) for x in guidance_weights]

    w_results = {}
    for w in tqdm.tqdm(guidance_weights, desc="Evaluating various guidance weights"):
        eval_info, _, cur_renders = eval_standard(
            agent=agent,
            env=eval_env,
            vec_eval_env=vec_eval_env,
            num_eval_episodes=num_eval_episodes,
            num_video_episodes=num_video_episodes,
            guidance_weight=w,
            rejection_sampling=rejection_sampling,
        )
        renders.extend(cur_renders)
        w_results[w] = eval_info
        max_return = max(max_return, eval_info["episode.return"])

    eval_metrics["evaluation/episode.return"] = max_return
    if "success" in w_results[guidance_weights[0]]:
        eval_metrics["evaluation/success"] = max(
            w_results[w]["success"] for w in guidance_weights
        )
    eval_metrics["evaluation/episode_length"] = min(
        w_results[w]["episode.length"] for w in guidance_weights
    )

    best_w = None
    for w in guidance_weights:
        if w_results[w]["episode.return"] == max_return:
            best_w = w
            break

    eval_metrics["evaluation/best_guidance_weight"] = best_w

    for w in guidance_weights:
        result = w_results[w]
        prefix = f"evaluation_guidance_weight_{w}"
        eval_metrics[f"{prefix}/episode_return"] = result["episode.return"]
        if "success" in result:
            eval_metrics[f"{prefix}/success"] = result["success"]
        eval_metrics[f"{prefix}/episode_length"] = result["episode.length"]

    return eval_metrics, renders
