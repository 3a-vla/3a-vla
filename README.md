# 3A-VLA

### Abstraction-Aligned Action Learning for Vision-Language Agents in 3D Game Worlds

[Game for Peace Dataset](https://huggingface.co/datasets/3a-vla/gp)

3A-VLA studies vision-language-action agents in dynamic 3D games. This
repository contains **GameEval**, a compact Python library for connecting an
agent to a real Windows game, recording its rollout, and evaluating task
completion.

The agent receives game pixels and emits ordinary keyboard and mouse actions.
Evaluator-only signals are kept outside the policy observation.

## Features

- Real Windows game capture and input control.
- YAML task definitions with fixed setup, budgets, and success criteria.
- Shared observation and action interfaces across supported games.
- Replayable episode artifacts with actions, frames, verdicts, and metrics.
- Plugin interfaces for external agents and private game adapters.

## Kill-task examples

| Game | Objective | Task file |
|---|---|---|
| Game for Peace | Eliminate one opponent | [`examples/tasks/gp/kill.yaml`](examples/tasks/gp/kill.yaml) |
| GTA5 | Eliminate one hostile NPC | [`examples/tasks/gta5/kill.yaml`](examples/tasks/gta5/kill.yaml) |
| CSGO | Eliminate one opponent | [`examples/tasks/csgo/kill.yaml`](examples/tasks/csgo/kill.yaml) |

These files are integration templates rather than a fixed benchmark suite.
Replace their build, resolution, scene, spawn, loadout, and target placeholders
with the exact setup used in an experiment.

## Installation

Game runtimes target **Windows 11** and require **Python 3.10+**.

```powershell
git clone https://github.com/3a-vla/3a-vla.git
cd 3a-vla
py -m pip install -e ".[windows]"
```

Game binaries, model checkpoints, and commercial game assets are not included.

## Run an evaluation

First validate and discover the bundled task for a game without starting it:

```powershell
gameeval --game gp --protocol gp-visual `
  --task-dir examples/tasks/gp --list-tasks
```

Then start one episode with the built-in manual-control agent:

```powershell
gameeval --game gp --protocol gp-visual `
  --task-dir examples/tasks/gp --tasks gp_kill `
  --agent manual --episodes 1 --output results/gp_kill
```

Use the corresponding `examples/tasks/gta5` or `examples/tasks/csgo`
directory for the other games. Run `gameeval --help` for agent, runtime, and
override options.

## Load tasks in Python

```python
from gameeval.tasks.task_registry import TaskRegistry

registry = TaskRegistry("examples/tasks")
for task in registry:
    print(task.game, task.task_id, task.description)
```

The bundled task IDs are `gp_kill`, `gta5_kill`, and `csgo_kill`.

## Connect an agent

An agent only needs an `act(observation)` method. `reset()` and
`set_task_context(...)` are optional.

```python
class MyAgent:
    def act(self, observation):
        frame = observation.image_for_vlm
        return {
            "action_chunk": "0 0 w ; w,shift ; L",
            "duration_ms": 150,
        }
```

An action chunk uses this format:

```text
mouse_dx mouse_dy frame_1 ; frame_2 ; ... ; frame_n
```

Frame inputs include movement keys, `space`, `ctrl`, `shift`, mouse buttons
`L`/`R`, interaction keys, reload, and weapon slots `1`–`4`.

## Episode artifacts

```text
<output>/<task_id>/episode_0000/
├── task.yaml
├── actions.jsonl
├── observations.jsonl
├── rollout.mp4          # or frames/
├── evaluation.json
└── manifest.json
```

GameEval reports each task independently. Success Rate uses all pre-declared
episodes as its denominator; unknown, error, and missing runs cannot improve it.

This compact release intentionally omits development tests, generated results,
private services, training code, credentials, and game assets.

## Responsible use and security

This is a source-available research release, not an unrestricted open-source
release. Use is limited to academic, security, and defensive research under the
[3A-VLA Research Use License](LICENSE). Competitive deployment, commercial
automation, automated aiming, resource farming, bulk-account control,
anti-cheat evasion, social impersonation, unlawful surveillance, and real-world
weaponization are prohibited.

Report security, privacy, or platform-safety concerns privately as described in
[SECURITY.md](SECURITY.md). No fine-tuned checkpoint is currently released; any
future checkpoint release will follow the stated delay and review safeguards.

## License

3A-VLA Research Use License v1.0. See [LICENSE](LICENSE).
