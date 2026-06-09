# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare a live RSL-RL checkpoint against a LEAPP export on the same env state.

This is a debugging helper for LEAPP deployment mismatches.  It drives one
standard Isaac Lab/RSL-RL environment, evaluates both the live PyTorch policy
and the LEAPP InferenceManager from the same simulator tensors, then steps the
reference environment with the live RSL action.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata as metadata
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare live RSL-RL and LEAPP policy outputs.")
    parser.add_argument("--task", type=str, required=True, help="Registered Isaac Lab task.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RSL-RL checkpoint.")
    parser.add_argument("--leapp_model", type=str, required=True, help="Path to LEAPP YAML pipeline.")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config registry key.")
    parser.add_argument("--num_steps", type=int, default=8, help="Number of comparison steps.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs. Keep this at 1 for LEAPP export.")
    parser.add_argument("--seed", type=int, default=None, help="Optional env seed override.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV output path for per-step diff metrics.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import onnx
import torch
import yaml
from leapp import InferenceManager
from onnx import numpy_helper
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs.leapp_deployment_env import _resolve_joint_ids
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_RSL_RL_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "rsl_rl"
if str(_RSL_RL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_RSL_RL_SCRIPTS_DIR))


def log(*args):
    print(*args, flush=True)


def policy_module(policy):
    """Return the module behind an inference callable when available."""
    return getattr(policy, "__self__", policy)


def is_recurrent_policy(policy) -> bool:
    module = policy_module(policy)
    return bool(getattr(module, "is_recurrent", False) or hasattr(module, "memory_a"))


def _clone_state(state):
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(t.clone() for t in state)
    return state.clone()


def clone_hidden_state(policy):
    if not is_recurrent_policy(policy):
        return None
    module = policy_module(policy)
    if hasattr(module, "get_hidden_state"):
        state = module.get_hidden_state()
    elif hasattr(module, "get_hidden_states"):
        state, _ = module.get_hidden_states()
    elif hasattr(module, "memory_a"):
        state = module.memory_a.hidden_state
    else:
        return None
    return _clone_state(state)


def restore_hidden_state(policy, state):
    if not is_recurrent_policy(policy):
        return
    module = policy_module(policy)
    if hasattr(module, "memory_a"):
        module.memory_a.hidden_state = _clone_state(state)
    elif state is None:
        module.reset()
    else:
        module.reset(hidden_state=_clone_state(state))


def tensor_preview(tensor: torch.Tensor, limit: int = 6) -> str:
    flat = tensor.detach().flatten().cpu()
    values = ", ".join(f"{float(v):+.5f}" for v in flat[:limit])
    suffix = ", ..." if flat.numel() > limit else ""
    return f"[{values}{suffix}]"


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().to(device="cpu", dtype=torch.float32)
    b = b.detach().to(device="cpu", dtype=torch.float32)
    diff = (a - b).abs()
    return {
        "max": float(diff.max().item()),
        "mean": float(diff.mean().item()),
        "rms": float(torch.sqrt(torch.mean(diff * diff)).item()),
    }


def _extract_observation_term(env, obs: dict[str, torch.Tensor], group_name: str, term_name: str) -> torch.Tensor:
    """Extract one observation term from a live observation buffer."""
    if group_name not in obs:
        raise KeyError(f"Observation group '{group_name}' not found in live obs keys: {list(obs.keys())}")

    observation_manager = env.unwrapped.observation_manager
    group_obs = obs[group_name]
    if not observation_manager.group_obs_concatenate[group_name]:
        return group_obs[term_name].detach().clone()

    group_term_names = observation_manager._group_obs_term_names[group_name]
    group_term_dims = observation_manager._group_obs_term_dim[group_name]
    term_index = group_term_names.index(term_name)
    concat_dim = observation_manager._group_obs_concatenate_dim[group_name]
    term_dim = concat_dim - 1 if concat_dim > 0 else concat_dim
    start = sum(dims[term_dim] for dims in group_term_dims[:term_index])
    length = group_term_dims[term_index][term_dim]
    return group_obs.narrow(dim=concat_dim, start=start, length=length).detach().clone()


def resolve_leapp_inputs(
    env, inference: InferenceManager, yaml_desc: dict, obs: dict[str, torch.Tensor] | None = None
) -> dict[str, torch.Tensor]:
    inputs = {}
    pipeline_inputs = yaml_desc["pipeline"]["inputs"]
    for node_name, input_names in pipeline_inputs.items():
        node = inference.nodes[node_name]
        desc_by_name = {d["name"]: d for d in node.input_descriptions}
        for input_name in input_names:
            desc = desc_by_name[input_name]
            connection = desc.get("isaaclab_connection")
            if connection is None:
                continue
            parts = connection.split(":")
            if parts[0] == "state":
                entity_name, prop_name = parts[1], parts[2]
                entity = env.unwrapped.scene[entity_name]
                value = getattr(entity.data, prop_name).torch
                joint_ids = _resolve_joint_ids(desc.get("element_names"), entity)
                if joint_ids is not None:
                    value = value[:, joint_ids]
            elif parts[0] == "observation":
                if obs is None:
                    raise RuntimeError(f"Live obs is required to resolve LEAPP input: {connection}")
                group_name, term_name = parts[1], parts[2]
                value = _extract_observation_term(env, obs, group_name, term_name)
            else:
                raise NotImplementedError(f"Unsupported input connection in debug helper: {connection}")
            inputs[f"{node_name}/{input_name}"] = value.detach().clone()
    return inputs


def get_model_output(outputs: dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
    matches = [value for key, value in outputs.items() if key.endswith(f"/{suffix}")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one LEAPP output ending in '/{suffix}', found {len(matches)}")
    return matches[0]


def get_onnx_constant0(yaml_path: str, yaml_desc: dict) -> torch.Tensor | None:
    models = yaml_desc.get("models", {})
    if not models:
        return None
    model_desc = next(iter(models.values()))
    model_path = os.path.join(os.path.dirname(yaml_path), model_desc["parameters"]["model_path"])
    graph = onnx.load(model_path).graph
    for init in graph.initializer:
        if init.name == "_tensor_constant0":
            return torch.from_numpy(numpy_helper.to_array(init).copy()).to(dtype=torch.float32)
    return None


def with_baked_shaft_pos(obs, baked_shaft_pos: torch.Tensor | None):
    obs_copy = copy.deepcopy(obs)
    if baked_shaft_pos is not None and baked_shaft_pos.shape[-1] == 3:
        obs_copy["policy"][:, 12:15] = baked_shaft_pos.to(device=obs_copy["policy"].device, dtype=obs_copy["policy"].dtype)
    return obs_copy


def evaluate_policy_with_state(policy, state, obs):
    restore_hidden_state(policy, state)
    return policy(copy.deepcopy(obs))


def expected_relative_joint_target(env, actions: torch.Tensor, clip_actions: float | None) -> torch.Tensor:
    action_term = env.unwrapped.action_manager._terms["arm_action"]
    arm_joint_ids = action_term._joint_ids
    scale = action_term._scale
    if not torch.is_tensor(scale):
        scale = torch.tensor(scale, device=actions.device, dtype=actions.dtype)
    clipped = torch.clamp(actions, -clip_actions, clip_actions) if clip_actions is not None else actions
    processed = clipped[:, : len(arm_joint_ids)] * scale
    current_joint_pos = env.unwrapped.scene["robot"].data.joint_pos.torch[:, arm_joint_ids]
    return current_joint_pos + processed


def maybe_write_csv_header(path: str | None):
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "step,rsl_vs_leapp_max,rsl_vs_leapp_mean,rsl_vs_leapp_rms,"
            "zero_vs_leapp_max,zero_vs_leapp_mean,zero_vs_leapp_rms,"
            "rsl_vs_zero_max,rsl_vs_zero_mean,rsl_vs_zero_rms\n"
        )


def append_csv(path: str | None, step: int, rsl_leapp: dict, zero_leapp: dict, rsl_zero: dict):
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"{step},{rsl_leapp['max']:.9g},{rsl_leapp['mean']:.9g},{rsl_leapp['rms']:.9g},"
            f"{zero_leapp['max']:.9g},{zero_leapp['mean']:.9g},{zero_leapp['rms']:.9g},"
            f"{rsl_zero['max']:.9g},{rsl_zero['mean']:.9g},{rsl_zero['rms']:.9g}\n"
        )


def main():
    task_name = args_cli.task.split(":")[-1]
    log("[COMPARE] loading env config")
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    log("[COMPARE] loading agent config")
    agent_cfg = load_cfg_from_registry(task_name, args_cli.agent)

    # Match the play.py config path closely.
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = os.path.dirname(args_cli.checkpoint)

    log("[COMPARE] creating gym env")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    log("[COMPARE] wrapping env for RSL-RL")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log("[COMPARE] constructing RSL-RL runner")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    log("[COMPARE] loading checkpoint")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    log("[COMPARE] loading LEAPP inference manager")
    inference = InferenceManager(args_cli.leapp_model)
    with open(args_cli.leapp_model, encoding="utf-8") as f:
        leapp_yaml = yaml.safe_load(f)
    baked_shaft_pos = get_onnx_constant0(args_cli.leapp_model, leapp_yaml)

    log("[COMPARE] task:", args_cli.task)
    log("[COMPARE] checkpoint:", args_cli.checkpoint)
    log("[COMPARE] leapp_model:", args_cli.leapp_model)
    log("[COMPARE] recurrent policy:", is_recurrent_policy(policy))
    log("[COMPARE] yaml feedback_flow:", leapp_yaml.get("pipeline", {}).get("feedback_flow"))
    log("[COMPARE] yaml inputs:", leapp_yaml.get("pipeline", {}).get("inputs"))
    log("[COMPARE] yaml outputs:", leapp_yaml.get("pipeline", {}).get("outputs"))
    log(
        "[COMPARE] onnx _tensor_constant0:",
        None if baked_shaft_pos is None else f"shape={tuple(baked_shaft_pos.shape)} {tensor_preview(baked_shaft_pos, 3)}",
    )

    maybe_write_csv_header(args_cli.csv)

    obs = env.get_observations()
    try:
        for step in range(args_cli.num_steps):
            with torch.inference_mode():
                # Evaluate a zero-memory version without disturbing the carried policy state.
                carried_state = clone_hidden_state(policy)
                zero_actions = evaluate_policy_with_state(policy, None, obs)
                zero_target = expected_relative_joint_target(env, zero_actions, agent_cfg.clip_actions)
                zero_baked_actions = evaluate_policy_with_state(policy, None, with_baked_shaft_pos(obs, baked_shaft_pos))
                zero_baked_target = expected_relative_joint_target(env, zero_baked_actions, agent_cfg.clip_actions)
                carried_baked_actions = evaluate_policy_with_state(
                    policy, carried_state, with_baked_shaft_pos(obs, baked_shaft_pos)
                )
                carried_baked_target = expected_relative_joint_target(
                    env, carried_baked_actions, agent_cfg.clip_actions
                )
                restore_hidden_state(policy, carried_state)

                # Evaluate the actual carried-memory policy.
                rsl_actions = policy(obs)
                rsl_target = expected_relative_joint_target(env, rsl_actions, agent_cfg.clip_actions)

                leapp_inputs = resolve_leapp_inputs(env, inference, leapp_yaml, obs)
                leapp_outputs = inference.run_policy(leapp_inputs)
                leapp_target = get_model_output(leapp_outputs, "arm_action").to(rsl_target.device)

                rsl_leapp = diff_stats(rsl_target, leapp_target)
                zero_leapp = diff_stats(zero_target, leapp_target)
                rsl_zero = diff_stats(rsl_target, zero_target)
                zero_baked_leapp = diff_stats(zero_baked_target, leapp_target)
                carried_baked_leapp = diff_stats(carried_baked_target, leapp_target)

                policy_obs = obs["policy"].detach().clone()
                shaft_pos = policy_obs[:, 12:15]
                shaft_quat = policy_obs[:, 15:19]

                log(
                    f"[COMPARE][step {step:03d}] "
                    f"rsl-vs-leapp max={rsl_leapp['max']:.6g} mean={rsl_leapp['mean']:.6g} rms={rsl_leapp['rms']:.6g} | "
                    f"zero-vs-leapp max={zero_leapp['max']:.6g} mean={zero_leapp['mean']:.6g} rms={zero_leapp['rms']:.6g} | "
                    f"rsl-vs-zero max={rsl_zero['max']:.6g} mean={rsl_zero['mean']:.6g} rms={rsl_zero['rms']:.6g}"
                )
                log(
                    f"[COMPARE][step {step:03d}] "
                    f"zero_baked-vs-leapp max={zero_baked_leapp['max']:.6g} "
                    f"mean={zero_baked_leapp['mean']:.6g} rms={zero_baked_leapp['rms']:.6g} | "
                    f"carried_baked-vs-leapp max={carried_baked_leapp['max']:.6g} "
                    f"mean={carried_baked_leapp['mean']:.6g} rms={carried_baked_leapp['rms']:.6g}"
                )
                log(f"[COMPARE][step {step:03d}] obs_policy={tensor_preview(policy_obs, 19)}")
                log(
                    f"[COMPARE][step {step:03d}] shaft_pos={tensor_preview(shaft_pos, 3)} "
                    f"shaft_quat={tensor_preview(shaft_quat, 4)}"
                )
                log(f"[COMPARE][step {step:03d}] rsl_target={tensor_preview(rsl_target, 6)}")
                log(f"[COMPARE][step {step:03d}] leapp_target={tensor_preview(leapp_target, 6)}")

                append_csv(args_cli.csv, step, rsl_leapp, zero_leapp, rsl_zero)

                obs, _, dones, _ = env.step(rsl_actions)
                if getattr(policy, "is_recurrent", False):
                    policy.reset(dones)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        try:
            main()
        except Exception:
            traceback.print_exc()
            raise
    finally:
        simulation_app.close()
