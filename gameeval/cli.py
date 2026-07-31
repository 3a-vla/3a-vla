"""GameEval CLI for task-driven VLA evaluation on Windows.

The VLA agent receives screenshots and task instructions. GP-State and
GP-Visual are separate protocols: the former has evaluator-only state, while
the latter is judged post-hoc from the recorded visual rollout.

Examples
--------
::

    gameeval --game gp --protocol gp-visual --task-dir examples/tasks/gp --list-tasks
    gameeval --game gp --protocol gp-visual --tasks gp_kill --task-dir examples/tasks/gp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from gameeval import GameEvalEnv, __version__
from gameeval.core.action_space import Action
from gameeval.core.game_adapter import AdapterConfig
from gameeval.core.observation import ObservationConfig
from gameeval.evaluators import EpisodeContext, StateEvaluator, VLMJudgeEvaluator
from gameeval.metrics.agreement import calculate_agreement
from gameeval.metrics.calculator import MetricsCalculator
from gameeval.metrics.report import ReportGenerator
from gameeval.paths import CONFIG_ROOT
from gameeval.protocols import GP_STATE, GP_VISUAL, expected_evaluator, resolve_protocol
from gameeval.recorder.artifact import EpisodeArtifactWriter
from gameeval.tasks.task_registry import TaskRegistry
from gameeval.utils.logger import setup_logger

logger = logging.getLogger("gameeval.cli")


def resolve_cli_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def parse_key_value_pairs(items: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Malformed override '{item}' (expected key=value)")
        key, value = item.split("=", 1)
        result[key.strip()] = resolve_cli_value(value.strip())
    return result


def resolve_runtime_config(obj: Any) -> Any:
    """Resolve ``${ENV_VAR}`` placeholders recursively."""
    if isinstance(obj, dict):
        return {key: resolve_runtime_config(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [resolve_runtime_config(value) for value in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        return os.getenv(obj[2:-1], "")
    return obj


SUPPORTED_GAMES = ("csgo", "gta5", "gp")


def create_runtime(game: str, protocol: str, server_config: dict[str, Any]):
    """Create a built-in runtime or an explicitly configured private plugin."""
    if game not in SUPPORTED_GAMES:
        raise ValueError(f"Unsupported game: {game}")
    game_config = server_config.get(game, {}) or {}
    protocol_config = game_config
    if game == "gp":
        protocol_config = game_config.get(
            "state" if protocol == GP_STATE else "visual", {}
        ) or {}
    plugin_name = protocol_config.get("adapter_plugin")
    if plugin_name:
        from gameeval.plugins import load_adapter_plugin

        return load_adapter_plugin(str(plugin_name), protocol_config)
    if game == "csgo":
        from gameeval.adapters.csgo.adapter import CSGOGameAdapter

        return CSGOGameAdapter()
    if game == "gta5":
        from gameeval.adapters.gta5.adapter import GTA5GameAdapter

        return GTA5GameAdapter()
    if protocol == GP_STATE:
        from gameeval.adapters.gp.adapter import GPBridgeAdapter

        return GPBridgeAdapter()
    if protocol == GP_VISUAL:
        from gameeval.adapters.gp.visual import GPVisualAdapter

        return GPVisualAdapter()
    raise ValueError(f"Unsupported GP protocol: {protocol}")


def _as_size(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def create_runtime_config(
    game: str,
    protocol: str,
    server_config: dict[str, Any],
) -> AdapterConfig:
    """Build a runtime config while keeping private values out of artifacts."""
    if game not in SUPPORTED_GAMES:
        raise ValueError(f"Unsupported game: {game}")
    cfg = server_config.get(game, {}) or {}
    input_cfg = cfg.get("input", {}) or {}
    capture_cfg = cfg.get("screenshot", {}) or {}
    gp_protocol_cfg = cfg.get(
        "state" if protocol == GP_STATE else "visual", {}
    ) if game == "gp" else {}
    if cfg.get("adapter_plugin"):
        return AdapterConfig(game=game, extra=dict(cfg))
    if isinstance(gp_protocol_cfg, dict) and gp_protocol_cfg.get("adapter_plugin"):
        return AdapterConfig(game=game, extra=dict(gp_protocol_cfg))
    if game == "csgo":
        gsi = cfg.get("gsi", {}) or {}
        return AdapterConfig(
            game=game,
            extra={
                "gsi_host": gsi.get("host", "0.0.0.0"),
                "gsi_port": int(gsi.get("port", 3000)),
                "gsi_auth_token": gsi.get("auth_token", "gameeval_local"),
                "screenshot_region": capture_cfg.get("region"),
                "screenshot_crop_size": _as_size(capture_cfg.get("crop_size")),
                "screenshot_target_size": _as_size(capture_cfg.get("target_size")),
                "screenshot_process_name": capture_cfg.get("process"),
                "screenshot_debug": bool(capture_cfg.get("debug", False)),
                "enable_input_injection": bool(input_cfg.get("enable", False)),
                "mouse_scale": float(input_cfg.get("mouse_scale", 1.0)),
                "hold_left_mouse": bool(input_cfg.get("hold_left_mouse", False)),
            },
        )
    if game == "gta5":
        return AdapterConfig(
            game=game,
            extra={
                "screenshot_region": capture_cfg.get("region"),
                "monitor_index": int(capture_cfg.get("monitor_index", 1)),
                "enable_input": bool(input_cfg.get("enable", False)),
                "mouse_scale": float(input_cfg.get("mouse_scale", 1.0)),
                "frame_duration_ms": int(input_cfg.get("frame_duration_ms", 50)),
                "use_key": str(input_cfg.get("use_key", "f")),
                "step_delay": float(cfg.get("step_delay", 0.05)),
                "reset_delay": float(cfg.get("reset_delay", 0.0)),
            },
        )
    if protocol == GP_STATE:
        state_cfg = cfg.get("state", {}) or {}
        bridge = state_cfg.get("bridge", {}) or {}
        return AdapterConfig(
            game=game,
            extra={
                "base_url": bridge.get("base_url"),
                "token": bridge.get("token"),
                "timeout": float(bridge.get("timeout", 30.0)),
            },
        )
    if protocol == GP_VISUAL:
        visual_cfg = cfg.get("visual", {}) or {}
        input_cfg = visual_cfg.get("input", {}) or {}
        capture_cfg = visual_cfg.get("screenshot", {}) or {}
        return AdapterConfig(
            game=game,
            extra={
                "screenshot_region": capture_cfg.get("region"),
                "monitor_index": int(capture_cfg.get("monitor_index", 1)),
                "enable_input": bool(input_cfg.get("enable", False)),
                "mouse_scale": float(input_cfg.get("mouse_scale", 1.0)),
                "use_key": str(input_cfg.get("use_key", "f")),
                "step_delay": float(visual_cfg.get("step_delay", 0.05)),
                "reset_delay": float(visual_cfg.get("reset_delay", 5.0)),
            },
        )
    raise ValueError(f"Unsupported GP protocol: {protocol}")


def _create_vlm_evaluator(
    evaluator_cfg: dict[str, Any],
    server_config: dict[str, Any],
) -> VLMJudgeEvaluator:
    judge_defaults = server_config.get("vlm_judge", {}) or {}
    from gameeval.utils.vlm_client import VLMClient

    client = VLMClient(
        model=evaluator_cfg.get("model") or judge_defaults.get("model", "gpt-4o"),
        api_key=judge_defaults.get("api_key"),
        base_url=judge_defaults.get("base_url"),
        temperature=0.0,
        max_tokens=int(judge_defaults.get("max_tokens", 512)),
        timeout=float(judge_defaults.get("timeout", 60.0)),
    )
    return VLMJudgeEvaluator(
        client,
        max_frames=int(evaluator_cfg.get("sample_frames", 8)),
        min_confidence=float(evaluator_cfg.get("min_confidence", 0.5)),
    )


def create_evaluator(task: dict[str, Any], server_config: dict[str, Any]):
    """Create the task's one authoritative judge backend."""
    evaluator_cfg = task.get("evaluator", {}) or {}
    evaluator_type = str(evaluator_cfg.get("type", "state")).lower()
    protocol = str(task.get("protocol", "")).lower()
    required = expected_evaluator(protocol)
    if evaluator_type != required:
        raise ValueError(f"{protocol} requires evaluator.type: {required}")
    if evaluator_type == "state":
        return StateEvaluator()
    if evaluator_type == "vlm":
        return _create_vlm_evaluator(evaluator_cfg, server_config)
    raise ValueError(f"Unsupported evaluator type: {evaluator_type}")


def load_agent(agent_type: str, agent_config: dict[str, Any] | None = None):
    cfg = dict(agent_config or {})
    if agent_type == "manual":
        from gameeval.agents.manual_agent import ManualAgent

        return ManualAgent(
            step_interval=float(cfg.get("step_interval", 0.1)),
            start_delay=float(cfg.get("start_delay", 3.0)),
            verbose=bool(cfg.get("verbose", True)),
        )
    if agent_type == "open-p2p-150m":
        from gameeval.agents.open_p2p_agent import OpenP2P150MAgent

        return OpenP2P150MAgent(**cfg)
    if agent_type == "open-p2p-150m-local":
        from gameeval.agents.open_p2p_agent import OpenP2P150MAgent
        from gameeval.agents.open_p2p_local import InProcessOpenP2PTransport

        transport_keys = {
            "open_p2p_root",
            "config_path",
            "checkpoint_path",
            "device",
            "max_virtual_steps",
            "compile_model",
            "model_size",
        }
        transport_config = {
            key: cfg.pop(key) for key in list(cfg) if key in transport_keys
        }
        if "open_p2p_root" not in transport_config:
            raise ValueError(
                "open-p2p-150m-local requires "
                "--agent-override open_p2p_root=C:\\path\\to\\open-p2p"
            )
        transport = InProcessOpenP2PTransport(**transport_config)
        cfg.setdefault("width", transport.frame_width)
        cfg.setdefault("height", transport.frame_height)
        return OpenP2P150MAgent(
            transport=transport,
            check_health=False,
            **cfg,
        )
    from gameeval.plugins import load_agent_plugin

    return load_agent_plugin(agent_type, cfg)


def _collect_episode_metrics(ep_data: dict[str, Any], info: dict[str, Any]) -> None:
    """Collect only explicitly exposed numeric runtime metrics."""
    values = info.get("metrics", {})
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            ep_data[str(key)] = value


def run_evaluation(
    *,
    game: str,
    protocol: str | None,
    task_ids: list[str],
    episodes_per_task: int,
    agent_type: str,
    server_config: dict[str, Any],
    output_dir: Path,
    max_steps: int | None = None,
    agent_config: dict[str, Any] | None = None,
    episode_delay: float = 2.0,
    task_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    """Run tasks and return aggregated metrics keyed by task id."""
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = resolve_protocol(game, protocol)
    if episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps safety cap must be positive")

    registry = TaskRegistry()
    for task_dir in task_dirs or []:
        registry.scan(task_dir)

    if task_ids == ["all"]:
        task_ids = registry.list_task_ids(game=game, protocol=protocol)
    else:
        available = set(registry.list_task_ids(game=game, protocol=protocol))
        missing = [task_id for task_id in task_ids if task_id not in available]
        if missing:
            raise ValueError(f"Unknown {game} task(s): {', '.join(missing)}")
    if not task_ids:
        raise ValueError(f"No tasks registered for game '{game}'")

    runtime = create_runtime(game, protocol, server_config)
    runtime_config = create_runtime_config(game, protocol, server_config)
    general_cfg = server_config.get("general", {}) or {}
    env = GameEvalEnv(
        adapter=runtime,
        adapter_config=runtime_config,
        obs_config=ObservationConfig(
            screenshot_width=int(general_cfg.get("screenshot_width", 640)),
            screenshot_height=int(general_cfg.get("screenshot_height", 360)),
            include_state=False,
        ),
        max_steps=max_steps or 1000,
    )
    artifact_cfg = server_config.get("artifacts", {}) or {}
    recorder = EpisodeArtifactWriter(
        output_dir / "artifacts",
        state_retention=artifact_cfg.get("state_retention", "none"),
        video_fps=float(artifact_cfg.get("video_fps", 8.0)),
    )
    calculator = MetricsCalculator()
    resolved_agent_config = dict(agent_config or {})
    resolved_agent_config.setdefault("game", game)
    agent = load_agent(agent_type, resolved_agent_config)

    all_results: dict[str, dict[str, Any]] = {}
    agreement_records: list[dict[str, Any]] = []
    total_start = time.time()
    logger.info(
        "Starting GameEval: game=%s tasks=%d episodes=%d agent=%s",
        game,
        len(task_ids),
        episodes_per_task,
        agent_type,
    )

    try:
        for task_id in task_ids:
            task = registry.get_task(task_id, game=game, protocol=protocol).to_dict()
            evaluator = create_evaluator(task, server_config)
            task_max_steps = int(task.get("max_steps", 1000))
            env.max_steps = (
                task_max_steps
                if max_steps is None
                else min(int(max_steps), task_max_steps)
            )
            task_episodes: list[dict[str, Any]] = []
            logger.info(
                "Task %s: evaluator=%s episodes=%d",
                task_id,
                evaluator.evaluator_type.value,
                episodes_per_task,
            )

            if hasattr(agent, "set_task_context"):
                agent.set_task_context(
                    game=game,
                    task=task_id,
                    instruction=task.get("description"),
                )

            for episode_index in range(1, episodes_per_task + 1):
                ep_start = time.time()
                recorder.start_episode(task=task, game=game, episode_id=episode_index)
                if hasattr(agent, "reset"):
                    agent.reset()
                observation = env.reset(task_config=task)
                initial_state = env.oracle_state or None
                done = False
                step = 0
                info: dict[str, Any] = {}
                ep_data: dict[str, Any] = {
                    "task_id": task_id,
                    "episode": episode_index,
                }

                time_limit = float(task.get("time_limit", 0.0))
                while not done:
                    if time_limit > 0 and time.time() - ep_start >= time_limit:
                        info.update({"done": True, "timeout": True})
                        done = True
                        break

                    action = agent.act(observation)
                    if isinstance(action, dict):
                        action = Action.from_dict(action)
                    observation, reward, done, info = env.step(action)
                    step += 1
                    _collect_episode_metrics(ep_data, info)

                    recorder.record_step(
                        step_index=step,
                        action=action.to_dict(),
                        frame=observation.screenshot,
                        state=env.oracle_state,
                        done=done,
                        info=info,
                    )

                state = env.oracle_state or None
                video_path = recorder.seal_rollout()
                episode_context = EpisodeContext(
                    task=task,
                    step_index=step,
                    initial_state=initial_state,
                    state=state,
                    frames=env.episode_screenshots,
                    video_path=video_path,
                    actions_path=recorder.actions_path,
                    episode_dir=recorder.episode_dir,
                    runtime_info=info,
                )
                result = evaluator.evaluate(episode_context)

                agreement_cfg = task.get("agreement", {}) or {}
                if protocol == GP_STATE and agreement_cfg.get("vlm"):
                    vlm_cfg = {"type": "vlm", **dict(agreement_cfg["vlm"])}
                    vlm_task = {**task, "evaluator": vlm_cfg}
                    vlm_result = _create_vlm_evaluator(vlm_cfg, server_config).evaluate(
                        EpisodeContext(
                            task=vlm_task,
                            step_index=step,
                            initial_state=None,
                            state=None,
                            frames=episode_context.frames,
                            video_path=episode_context.video_path,
                            actions_path=episode_context.actions_path,
                            episode_dir=episode_context.episode_dir,
                            runtime_info=episode_context.runtime_info,
                        )
                    )
                    state_label = result.status.value if result.status.value != "error" else None
                    vlm_label = vlm_result.status.value if vlm_result.status.value != "error" else None
                    agreement_record = {
                        "task_id": task_id,
                        "episode": episode_index,
                        "state": state_label,
                        "vlm": vlm_label,
                        "human": None,
                    }
                    agreement_records.append(agreement_record)
                    recorder.write_sidecar(
                        "agreement.json",
                        {
                            "schema_version": "1.0",
                            "episode": {
                                "task_id": task_id,
                                "episode": episode_index,
                                "protocol": protocol,
                            },
                            "labels": {
                                "state": state_label,
                                "vlm": vlm_label,
                                "human": None,
                            },
                            "vlm_evaluation": vlm_result.to_dict(),
                            "note": "Calibration labels do not replace the GP-State authoritative result.",
                        },
                    )
                ep_data.update(
                    {
                        "success": result.success,
                        "evaluation_status": result.status.value,
                        "total_steps": step,
                        "duration": time.time() - ep_start,
                    }
                )
                for key, value in result.metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        ep_data[key] = value
                if result.score is not None and "progress" not in ep_data:
                    ep_data["progress"] = float(result.score)

                episode_metrics = calculator.compute_episode(ep_data)
                recorder.finalize(
                    result=result,
                    total_steps=step,
                    metrics=episode_metrics,
                )
                task_episodes.append(ep_data)
                logger.info(
                    "Episode %d/%d: %s steps=%d duration=%.1fs -- %s",
                    episode_index,
                    episodes_per_task,
                    result.status.value.upper(),
                    step,
                    ep_data["duration"],
                    result.reason[:120],
                )
                if episode_index < episodes_per_task and episode_delay > 0:
                    time.sleep(episode_delay)

            all_results[task_id] = calculator.aggregate(
                task_episodes,
                scheduled_episodes=episodes_per_task,
            )
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
    finally:
        env.close()
        recorder.close()

    total_time = time.time() - total_start
    metadata = {
        "framework": "GameEval",
        "version": __version__,
        "game": game,
        "protocol": protocol,
        "agent": agent_type,
        "evaluator_selection": "task",
        "episodes_per_task": episodes_per_task,
        "total_tasks": len(task_ids),
        "total_time_seconds": round(total_time, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_files = ReportGenerator(output_dir).generate(
        all_results, metadata
    )
    for fmt, path in report_files.items():
        logger.info("%s report: %s", fmt, path)
    with open(output_dir / "raw_results.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": metadata, "per_task": all_results},
            handle,
            indent=2,
            default=str,
        )
    if agreement_records:
        with open(output_dir / "agreement_state_vlm.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": calculate_agreement(agreement_records),
                    "records": agreement_records,
                    "note": "Join expert labels with gameeval-agreement for three-way agreement.",
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
    return all_results


def _list_tasks(game: str, protocol: str, task_dirs: list[Path]) -> int:
    """Validate task YAML files and print tasks matching one protocol."""
    registry = TaskRegistry()
    for task_dir in task_dirs:
        registry.scan(task_dir)
    tasks = registry.list_tasks(game=game, protocol=protocol)
    if not tasks:
        raise ValueError(
            f"No tasks found for {game}:{protocol}; pass one or more --task-dir values"
        )
    for task in tasks:
        print(f"{task.task_id}\t{task.game}\t{task.protocol}\t{task.description}")
    return len(tasks)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gameeval",
        description="Evaluate VLA agents in CSGO, GTA5, or GP",
    )
    parser.add_argument("--version", action="version", version=f"GameEval {__version__}")
    parser.add_argument("--game", "-g", default="csgo", choices=SUPPORTED_GAMES)
    parser.add_argument(
        "--protocol",
        choices=(GP_STATE, GP_VISUAL),
        help="Required for GP; GP-State and GP-Visual cannot be mixed in one run",
    )
    task_mode = parser.add_mutually_exclusive_group(required=True)
    task_mode.add_argument(
        "--tasks",
        "-t",
        help="Comma-separated task ids or 'all'",
    )
    task_mode.add_argument(
        "--list-tasks",
        action="store_true",
        help="Validate and list matching task YAML files without starting a game",
    )
    parser.add_argument("--episodes", "-n", type=int, default=10)
    parser.add_argument(
        "--agent",
        "-a",
        default="manual",
        help=(
            "Built-ins: manual, open-p2p-150m, open-p2p-150m-local; "
            "otherwise an installed entry-point name"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional safety cap; task YAML remains the default step budget",
    )
    parser.add_argument("--episode-delay", type=float, default=2.0)
    parser.add_argument("--output", "-o", default="results/run")
    parser.add_argument("--config", "-c", default=str(CONFIG_ROOT / "server.yaml"))
    parser.add_argument("--agent-override", action="append", default=[])
    parser.add_argument(
        "--task-dir",
        action="append",
        default=[],
        help="Project task directory, repeatable (no fixed suite is bundled)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return resolve_runtime_config(yaml.safe_load(handle) or {})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    level = logging.ERROR if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    setup_logger(level=level)

    server_config = _load_yaml(Path(args.config))
    model_config = _load_yaml(CONFIG_ROOT / "model.yaml")
    game_config = _load_yaml(CONFIG_ROOT / args.game / "server.yaml")
    server_config[args.game] = {**server_config.get(args.game, {}), **game_config}
    server_config["vlm_judge"] = {
        **server_config.get("vlm_judge", {}),
        **model_config.get("judge", {}),
    }
    agent_config = parse_key_value_pairs(args.agent_override)

    try:
        protocol = resolve_protocol(args.game, args.protocol)
        task_dirs = [Path(value) for value in args.task_dir]
        if args.list_tasks:
            count = _list_tasks(args.game, protocol, task_dirs)
            logger.info("Validated %d task(s)", count)
            return 0
        results = run_evaluation(
            game=args.game,
            protocol=protocol,
            task_ids=[item.strip() for item in args.tasks.split(",")],
            episodes_per_task=args.episodes,
            agent_type=args.agent,
            server_config=server_config,
            output_dir=Path(args.output),
            max_steps=args.max_steps,
            agent_config=agent_config,
            episode_delay=args.episode_delay,
            task_dirs=task_dirs,
        )
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        return 1

    for task_id, metrics in results.items():
        logger.info(
            "%s: success=%.1f%% coverage=%.1f%%",
            task_id,
            metrics.get("success_rate", 0.0),
            metrics.get("coverage_rate", 0.0),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
