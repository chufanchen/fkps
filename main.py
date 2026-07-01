from __future__ import annotations

import gc
import glob
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass

import gymnasium as gym
import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from ml_collections import config_flags
from utils.datasets import Dataset, ReplayBuffer, load_replay_buffer
from utils.evaluation import eval_standard, eval_with_test_time_guidance, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS


# wandb options
flags.DEFINE_string("wandb_run_group", "Debug", "Run group.")
flags.DEFINE_string("wandb_project", "qgf", "Wandb project name.")
flags.DEFINE_boolean("wandb_offline", False, "Whether to run wandb in offline mode.")
flags.DEFINE_multi_string("wandb_tags", None, "Wandb tags.")

# experiment
flags.DEFINE_boolean("debug", False, "Whether to run in debug mode.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "env_name", "cube-double-play-singletask-v0", "Environment (dataset) name."
)
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", 0.0, "Reward bias.")
flags.DEFINE_boolean(
    "sparse",
    False,
    "If True, transform into sparse rewards: -1 each step, 0 at the end.",
)

# save and restore
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", 0, "Restore epoch.")

# training
flags.DEFINE_integer("buffer_size", 2000000, "Replay buffer size.")
flags.DEFINE_integer("offline_steps", 500000, "Number of offline steps.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", 100000, "Saving interval.")

# evaluation
flags.DEFINE_boolean(
    "eval_only",
    False,
    "Run evaluation and exit (no training). Useful for re-evaluating test-time guidance methods.",
)
flags.DEFINE_integer("eval_episodes", 30, "Number of evaluation episodes.")
flags.DEFINE_integer("eval_vecenv_size", 5, "Evaluation vectorized environment size.")
flags.DEFINE_integer("video_episodes", 0, "Number of video episodes for each task.")
flags.DEFINE_list(
    "guidance_weights",
    "0.0,1.0,1.5,3.0,5.0",
    "Guidance weights to evaluate for test-time guidance agents.",
)
flags.DEFINE_list("bfn_values", "1", "N value for best-of-n action sampling.")

# online RL
flags.DEFINE_float(
    "online_explorative_guidance_weight",
    1.0,
    "Guidance weight used for online exploration.",
)
flags.DEFINE_integer("online_steps", 0, "Number of online steps.")
flags.DEFINE_integer(
    "balanced_sampling", 0, "Whether to use balanced sampling for online fine-tuning."
)
flags.DEFINE_boolean(
    "save_online_buffer", False, "Whether to save the online replay buffer."
)
flags.DEFINE_integer(
    "online_buffer_save_interval",
    500000,
    "Interval (in steps) to save the online replay buffer.",
)

# agent
config_flags.DEFINE_config_file("agent", "agents/qgf.py", lock_config=False)

# large OGBench dataset (100M) flags
flags.DEFINE_string(
    "ogbench_dataset_dir", None, "Directory of OGBench 100M dataset .npz slices."
)
flags.DEFINE_integer(
    "dataset_replace_interval",
    1000,
    "Steps between dataset slice swaps (0 = disabled).",
)


@dataclass
class DataSetup:
    dataset_action_clip_eps: float
    dataset_idx: int
    dataset_paths: list[str]
    env: object
    eval_env: object
    train_dataset: object
    val_dataset: object | None
    replay_buffer: object
    vec_eval_env: object | None
    example_batch: dict


def _is_test_time_guidance_agent(agent) -> bool:
    return getattr(agent, "support_guidance", False)


def _remap_sparse_env_reward(reward):
    """Dense -> sparse step reward for online replay."""
    r = np.asarray(reward)
    out = np.where(r != 0.0, -1.0, 0.0).astype(r.dtype)
    return out.item() if out.ndim == 0 else out


def _setup_experiment(config):
    """Create experiment names and save directory."""
    agent_name = config["agent_name"]
    env_short = re.sub(r"-(singletask|v0|v1|v2)", "", FLAGS.env_name)
    exp_hash = hashlib.sha256(json.dumps(config.to_dict()).encode()).hexdigest()[:8]
    exp_name = f"{FLAGS.wandb_run_group}_{agent_name}_{env_short}_seed{FLAGS.seed:02d}_{exp_hash}"

    save_subpath = [FLAGS.wandb_project, FLAGS.wandb_run_group, exp_name]
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, *save_subpath)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    return agent_name, exp_name


def _setup_data(config):
    # Make offline datasets
    dataset_action_clip_eps = config.get("dataset_action_clip_eps", 1e-5)
    if FLAGS.ogbench_dataset_dir is not None:
        # 100M OGBench datasets
        assert (
            FLAGS.dataset_replace_interval != 0
        ), "dataset_replace_interval must be nonzero for large OGBench datasets"
        dataset_idx = 0
        dataset_paths = sorted(
            [
                f
                for f in glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")
                if "-val.npz" not in f
            ]
        )
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
            action_clip_eps=dataset_action_clip_eps,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
            sparse=FLAGS.sparse,
        )
    else:
        dataset_idx = 0
        dataset_paths = []
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
            FLAGS.env_name,
            action_clip_eps=dataset_action_clip_eps,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
            sparse=FLAGS.sparse,
        )

    # Eval envs (vectorized version available, but video rendering needs single process env)
    if FLAGS.eval_vecenv_size > 1:
        assert (
            FLAGS.eval_episodes % FLAGS.eval_vecenv_size == 0
        ), "eval_episodes must be divisible by eval_vecenv_size"
        env_fns = [
            lambda: make_env_and_datasets(
                FLAGS.env_name,
                action_clip_eps=dataset_action_clip_eps,
                reward_scale=FLAGS.reward_scale,
                reward_bias=FLAGS.reward_bias,
                eval_env_only=True,
            )
            for _ in range(FLAGS.eval_vecenv_size)
        ]
        vec_eval_env = gym.vector.AsyncVectorEnv(env_fns)
    else:
        vec_eval_env = None

    if FLAGS.video_episodes > 0:
        assert (
            "singletask" in FLAGS.env_name
        ), "Rendering is currently only supported for OGBench environments."
    if FLAGS.online_steps > 0:
        assert (
            "visual" not in FLAGS.env_name
        ), "Online fine-tuning is currently not supported for visual environments."

    train_dataset = Dataset.create(**train_dataset)

    # Create replay buffer
    if FLAGS.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        # Use the training dataset as the replay buffer.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
        gc.collect()

    # Restore replay buffer if resuming from an online RL checkpoint.
    if FLAGS.restore_epoch > FLAGS.offline_steps and not FLAGS.eval_only:
        replay_buffer = load_replay_buffer(FLAGS.restore_path, FLAGS.restore_epoch)
        print(
            f"Restored replay buffer from step {FLAGS.restore_epoch} "
            f"(size: {replay_buffer.size})"
        )

    # Create example batch for agent initialization
    example_batch = train_dataset.sample(1)

    return DataSetup(
        dataset_action_clip_eps=dataset_action_clip_eps,
        dataset_idx=dataset_idx,
        dataset_paths=dataset_paths,
        env=env,
        eval_env=eval_env,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        replay_buffer=replay_buffer,
        vec_eval_env=vec_eval_env,
        example_batch=example_batch,
    )


def _setup_agents(config, agent_name, example_batch):
    # RL agent
    agent = agents[agent_name].create(
        FLAGS.seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )

    # Restore agent
    if FLAGS.restore_path:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        print(
            f"Restored agent from {FLAGS.restore_path} at epoch {FLAGS.restore_epoch}"
        )

    return agent


def _evaluate_agent(
    agent,
    *,
    eval_env,
    vec_eval_env,
    num_eval_episodes,
    rejection_sampling,
):
    # Special evaluation for diffusion agents with multiple guidance weights
    if _is_test_time_guidance_agent(agent):
        return eval_with_test_time_guidance(
            agent,
            eval_env,
            vec_eval_env,
            num_eval_episodes=num_eval_episodes,
            rejection_sampling=rejection_sampling,
            guidance_weights=FLAGS.guidance_weights,
            num_video_episodes=FLAGS.video_episodes,
        )

    # Standard evaluation for other agents
    eval_info, _, renders = eval_standard(
        agent=agent,
        env=eval_env,
        vec_eval_env=vec_eval_env,
        num_eval_episodes=num_eval_episodes,
        num_video_episodes=FLAGS.video_episodes,
        rejection_sampling=rejection_sampling,
    )
    return {f"evaluation/{k}": v for k, v in eval_info.items()}, renders


def _finish_training(agent, train_logger, eval_logger, vec_eval_env, step):
    save_agent(agent, FLAGS.save_dir, step)
    train_logger.close()
    eval_logger.close()
    _close_vec_eval_env(vec_eval_env)


def main(_):
    # Set seeds
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Config
    config = FLAGS.agent
    agent_name, exp_name = _setup_experiment(config)

    # Dataset and environments
    data = _setup_data(config)
    dataset_action_clip_eps = data.dataset_action_clip_eps
    dataset_idx = data.dataset_idx
    dataset_paths = data.dataset_paths
    env = data.env
    eval_env = data.eval_env
    train_dataset = data.train_dataset
    val_dataset = data.val_dataset
    replay_buffer = data.replay_buffer
    vec_eval_env = data.vec_eval_env
    example_batch = data.example_batch

    # Agent
    agent = _setup_agents(config, agent_name, example_batch)

    # Wandb set up
    run = setup_wandb(
        project=FLAGS.wandb_project,
        group=FLAGS.wandb_run_group,
        tags=FLAGS.wandb_tags,
        name=exp_name,
        hyperparam_dict=config.to_dict(),
        mode="disabled"
        if FLAGS.debug
        else ("offline" if FLAGS.wandb_offline else "online"),
    )
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(get_flag_dict(), f)

    # Eval-only mode: run one evaluation pass and exit without any training.
    if FLAGS.eval_only:
        assert FLAGS.restore_path, "--eval_only requires --restore_path"
        eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
        for rejection_sampling in [int(float(x.strip())) for x in FLAGS.bfn_values]:
            eval_metrics, cur_renders = _evaluate_agent(
                agent,
                eval_env=eval_env,
                vec_eval_env=vec_eval_env,
                num_eval_episodes=FLAGS.eval_episodes,
                rejection_sampling=rejection_sampling,
            )
            
            if rejection_sampling > 1:
                eval_metrics = {
                    f"best_of_{rejection_sampling}_{k}": v
                    for k, v in eval_metrics.items()
                }

            wandb.log(eval_metrics, step=FLAGS.restore_epoch)
            eval_logger.log(eval_metrics, step=FLAGS.restore_epoch)

        eval_logger.close()
        wandb.finish()
        return

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    action_queue = []  # action chunk queue for action chunking support
    action_dim = example_batch["actions"].shape[-1]
    for i in tqdm.tqdm(
        range(FLAGS.restore_epoch + 1, FLAGS.offline_steps + FLAGS.online_steps + 1),
        smoothing=0.1,
        dynamic_ncols=True,
    ):
        #########################################################
        # Offline RL
        #########################################################

        if i <= FLAGS.offline_steps:

            # Rotate to the next dataset slice for large OGBench datasets.
            if (
                FLAGS.ogbench_dataset_dir is not None
                and FLAGS.dataset_replace_interval != 0
                and i % FLAGS.dataset_replace_interval == 0
            ):
                dataset_idx = (dataset_idx + 1) % len(dataset_paths)
                print(
                    f"Swapping to dataset slice: {dataset_paths[dataset_idx]}",
                    flush=True,
                )

                # free memory
                train_dataset = None
                gc.collect()

                train_dataset, _ = make_ogbench_env_and_datasets(
                    FLAGS.env_name,
                    dataset_path=dataset_paths[dataset_idx],
                    compact_dataset=False,
                    dataset_only=True,
                    cur_env=env,
                    action_clip_eps=dataset_action_clip_eps,
                    reward_scale=FLAGS.reward_scale,
                    reward_bias=FLAGS.reward_bias,
                    sparse=FLAGS.sparse,
                )

            # sample batch and update
            batch = train_dataset.sample_sequence(
                config["batch_size"],
                sequence_length=config["horizon_length"],
                discount=config["discount"],
            )
            agent, update_info = agent.update(batch)
        else:

            #########################################################
            # Online fine-tuning
            #########################################################

            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()
                action_queue = []

            if len(action_queue) == 0:
                if _is_test_time_guidance_agent(agent):
                    sampled_action = agent.sample_actions(
                        observations=ob,
                        seed=key,
                        guidance_weight=FLAGS.online_explorative_guidance_weight,
                    )
                else:
                    sampled_action = agent.sample_actions(observations=ob, seed=key)
                action_chunk = np.array(sampled_action).reshape(-1, action_dim)
                action_queue.extend(action_chunk)
            action = action_queue.pop(0)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            if FLAGS.sparse:
                reward = _remap_sparse_env_reward(reward)

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {
                    f"exploration/{k}": np.mean(v) for k, v in flatten(info).items()
                }

            step += 1

            # Update agent.
            if FLAGS.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer.
                dataset_batch = train_dataset.sample_sequence(
                    config["batch_size"] // 2,
                    sequence_length=config["horizon_length"],
                    discount=config["discount"],
                )
                replay_batch = replay_buffer.sample_sequence(
                    config["batch_size"] // 2,
                    sequence_length=config["horizon_length"],
                    discount=config["discount"],
                )
                batch = {
                    k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0)
                    for k in dataset_batch
                }
            else:
                batch = replay_buffer.sample_sequence(
                    config["batch_size"],
                    sequence_length=config["horizon_length"],
                    discount=config["discount"],
                )

            agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            train_metrics["training/rewards"] = batch["rewards"].mean()
            train_metrics["training/rewards_max"] = batch["rewards"].max()
            train_metrics["training/rewards_min"] = batch["rewards"].min()
            if val_dataset is not None:
                val_batch = val_dataset.sample_sequence(
                    config["batch_size"],
                    sequence_length=config["horizon_length"],
                    discount=config["discount"],
                )
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update(
                    {f"validation/{k}": v for k, v in val_info.items()}
                )

            # FKD effective-sample-size curve (for FKD/particle-steering agents).
            if (
                hasattr(agent, "fkd_ess_curve")
                and int(agent.config.get("fkd_num_particles", 0)) > 1
                and val_dataset is not None
            ):
                # A small obs subset keeps this diagnostic cheap.
                obs = np.asarray(val_batch["observations"])[:16]
                aux = agent.fkd_ess_curve(obs, jax.random.PRNGKey(i))
                ess = np.asarray(aux["ess"])
                did = np.asarray(aux["did_resample"])
                rs_std = np.asarray(aux["rs_std"])
                train_metrics["fkd/ess_mean"] = float(ess.mean())
                train_metrics["fkd/ess_min"] = float(ess.min())
                train_metrics["fkd/resample_frac"] = float(did.mean())
                train_metrics["fkd/rs_std_mean"] = float(rs_std.mean())
                curve = wandb.Table(
                    data=[[k, float(e)] for k, e in enumerate(ess)],
                    columns=["step", "ess"],
                )
                train_metrics["fkd/ess_curve"] = wandb.plot.line(
                    curve, "step", "ess", title="FKD ESS vs sampling step"
                )
            train_metrics["time/epoch_time"] = (
                time.time() - last_time
            ) / FLAGS.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0:
            for rejection_sampling in [int(float(x.strip())) for x in FLAGS.bfn_values]:
                eval_metrics, renders = _evaluate_agent(
                    agent,
                    eval_env=eval_env,
                    vec_eval_env=vec_eval_env,
                    num_eval_episodes=FLAGS.eval_episodes,
                    rejection_sampling=rejection_sampling,
                )

                if FLAGS.video_episodes > 0:
                    video = get_wandb_video(renders=renders)
                    eval_metrics["video"] = video

                if rejection_sampling > 1:
                    eval_metrics = {
                        f"best_of_{rejection_sampling}_{k}": v
                        for k, v in eval_metrics.items()
                    }

                wandb.log(eval_metrics, step=i)
                eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

        # Save online replay buffer.
        if (
            FLAGS.save_online_buffer
            and i > FLAGS.offline_steps
            and (i - FLAGS.offline_steps) % FLAGS.online_buffer_save_interval == 0
        ):
            replay_buffer.save(FLAGS.save_dir, i)

    _finish_training(agent, train_logger, eval_logger, vec_eval_env, i)


def _close_vec_eval_env(vec_eval_env):
    if vec_eval_env is not None:
        for p in vec_eval_env.processes:
            if p.is_alive():
                p.kill()
        for p in vec_eval_env.processes:
            p.join(timeout=5)
        vec_eval_env.closed = True


if __name__ == "__main__":
    app.run(main)
