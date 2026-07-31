"""Unified keyboard, mouse, and timed action-chunk definitions.

GameEval supports two compatible action representations:

* **Single-tick actions**: discrete actions + mouse deltas.
* **Timed action chunks**: total mouse delta plus an arbitrary number
  of per-frame keyboard / mouse button slots within a chunk.

The canonical chunk text format is:

``mX mY f1 ; f2 ; ... ; fN``

where ``mX`` and ``mY`` are total mouse deltas for the chunk and each
frame slot contains ``.`` or comma-separated input tokens such as
``w``, ``a``, ``shift``, ``L``, ``R``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any


class DiscreteAction(IntEnum):
    """Common discrete actions used by keyboard/mouse game runtimes."""

    NOOP = 0
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    JUMP = auto()
    CROUCH = auto()
    SHOOT = auto()
    RELOAD = auto()
    WEAPON_SLOT_1 = auto()
    WEAPON_SLOT_2 = auto()
    WEAPON_SLOT_3 = auto()
    USE = auto()  # interact / plant bomb / defuse


ACTION_INPUT_TOKENS = {
    ".",
    "w",
    "s",
    "a",
    "d",
    "space",
    "ctrl",
    "shift",
    "1",
    "2",
    "3",
    "4",
    "q",
    "z",
    "r",
    "e",
    "f",
    "L",
    "R",
    "M",
}

TOKEN_TO_DISCRETE: dict[str, DiscreteAction] = {
    "w": DiscreteAction.MOVE_FORWARD,
    "s": DiscreteAction.MOVE_BACKWARD,
    "a": DiscreteAction.MOVE_LEFT,
    "d": DiscreteAction.MOVE_RIGHT,
    "space": DiscreteAction.JUMP,
    "ctrl": DiscreteAction.CROUCH,
    "L": DiscreteAction.SHOOT,
    "r": DiscreteAction.RELOAD,
    "e": DiscreteAction.USE,
    "f": DiscreteAction.USE,
    "1": DiscreteAction.WEAPON_SLOT_1,
    "2": DiscreteAction.WEAPON_SLOT_2,
    "3": DiscreteAction.WEAPON_SLOT_3,
}


@dataclass
class ActionFrame:
    """One sub-frame inside a timed action chunk."""

    inputs: list[str] = field(default_factory=list)

    @classmethod
    def from_value(cls, value: str | list[str] | tuple[str, ...]) -> "ActionFrame":
        if isinstance(value, str):
            text = value.strip()
            if not text or text == ".":
                return cls([])
            inputs = [token.strip() for token in text.split(",") if token.strip()]
        else:
            inputs = [str(token).strip() for token in value if str(token).strip()]

        for token in inputs:
            if token not in ACTION_INPUT_TOKENS - {"."}:
                raise ValueError(f"Unsupported action input token: {token}")
        return cls(inputs)

    def to_string(self) -> str:
        return "." if not self.inputs else ",".join(self.inputs)

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": list(self.inputs)}


@dataclass
class Action:
    """A single-tick or chunked action in a game environment.

    Attributes
    ----------
    discrete : set[DiscreteAction]
        Active discrete actions this tick (can be simultaneous, e.g.
        MOVE_FORWARD + SHOOT).
    mouse_dx : float
        Horizontal mouse delta (yaw).  Positive = turn right.
    mouse_dy : float
        Vertical mouse delta (pitch).  Positive = look up (game-dependent).
    raw : dict[str, Any]
        Optional pass-through for game-specific overrides.
    duration_ms : int | None
        Optional total duration for a chunk action.
    frames : list[ActionFrame]
        Optional sub-frame inputs for the timed chunk protocol.
    """

    discrete: set[DiscreteAction] = field(default_factory=set)
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None
    frames: list[ActionFrame] = field(default_factory=list)

    # ---- Convenience constructors -------------------------------------------

    @classmethod
    def noop(cls) -> Action:
        """Return a no-operation action."""
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> Action:
        """Construct from a plain dictionary.

        Expected keys::

            {
                "discrete": ["MOVE_FORWARD", "SHOOT"],  # list of names
                "mouse_dx": 1.5,
                "mouse_dy": -0.3,
            }
        """
        if "action_chunk" in d:
            action = cls.from_chunk_string(
                d["action_chunk"], duration_ms=d.get("duration_ms")
            )
            action.raw.update(d.get("raw", {}))
            return action

        discrete = set()
        for name in d.get("discrete", []):
            if isinstance(name, str):
                discrete.add(DiscreteAction[name.upper()])
            elif isinstance(name, int):
                discrete.add(DiscreteAction(name))
            else:
                discrete.add(name)

        frames = [
            ActionFrame.from_value(frame.get("inputs", []))
            if isinstance(frame, dict)
            else ActionFrame.from_value(frame)
            for frame in d.get("frames", [])
        ]
        if frames and not discrete:
            discrete = cls._discrete_from_frames(frames)

        return cls(
            discrete=discrete,
            mouse_dx=float(d.get("mouse_dx", 0.0)),
            mouse_dy=float(d.get("mouse_dy", 0.0)),
            raw=d.get("raw", {}),
            duration_ms=(int(d["duration_ms"]) if d.get("duration_ms") is not None else None),
            frames=frames,
        )

    @classmethod
    def from_chunk_string(
        cls,
        text: str,
        duration_ms: int | None = None,
        raw: dict[str, Any] | None = None,
    ) -> Action:
        """Parse canonical action-chunk text.

        Format: ``mX mY f1 ; f2 ; ... ; fN``.
        """
        text = text.strip()
        if not text:
            raise ValueError("Empty action string")

        parts = text.split(maxsplit=2)
        if len(parts) < 2:
            raise ValueError(f"Invalid action string: {text}")

        mouse_dx = float(parts[0])
        mouse_dy = float(parts[1])
        frame_part = parts[2] if len(parts) >= 3 else "."
        frames = [ActionFrame.from_value(slot.strip()) for slot in frame_part.split(";")]
        discrete = cls._discrete_from_frames(frames)
        return cls(
            discrete=discrete,
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
            raw=raw or {},
            duration_ms=duration_ms,
            frames=frames,
        )

    @staticmethod
    def _discrete_from_frames(frames: list[ActionFrame]) -> set[DiscreteAction]:
        discrete = set()
        for frame in frames:
            for token in frame.inputs:
                action = TOKEN_TO_DISCRETE.get(token)
                if action is not None:
                    discrete.add(action)
        return discrete

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary."""
        payload = {
            "discrete": sorted(a.name for a in self.discrete),
            "mouse_dx": self.mouse_dx,
            "mouse_dy": self.mouse_dy,
            "raw": self.raw,
        }
        if self.duration_ms is not None:
            payload["duration_ms"] = self.duration_ms
        if self.frames:
            payload["frames"] = [frame.to_dict() for frame in self.frames]
            payload["action_chunk"] = self.to_chunk_string()
        return payload

    def to_chunk_string(self) -> str:
        """Serialize to canonical action-chunk text."""
        frame_text = " ; ".join(frame.to_string() for frame in self.frames) if self.frames else "."
        mouse_dx = int(round(self.mouse_dx)) if float(self.mouse_dx).is_integer() else self.mouse_dx
        mouse_dy = int(round(self.mouse_dy)) if float(self.mouse_dy).is_integer() else self.mouse_dy
        return f"{mouse_dx} {mouse_dy} {frame_text}"


class GameActionSpace:
    """Describes the full action space and provides encode/decode utilities.

    This is *not* a Gymnasium Space object (to avoid hard dependency), but
    mirrors the concept: it tells agents what actions are legal and converts
    between our :class:`Action` and flat vectors / game-specific commands.
    """

    ALL_DISCRETE = list(DiscreteAction)

    def __init__(
        self,
        enabled_discrete: list[DiscreteAction] | None = None,
        mouse_dx_range: tuple[float, float] = (-180.0, 180.0),
        mouse_dy_range: tuple[float, float] = (-90.0, 90.0),
    ):
        self.enabled_discrete = enabled_discrete or list(DiscreteAction)
        self.mouse_dx_range = mouse_dx_range
        self.mouse_dy_range = mouse_dy_range

    @property
    def num_discrete(self) -> int:
        return len(self.enabled_discrete)

    @property
    def num_continuous(self) -> int:
        return 2  # mouse_dx, mouse_dy

    @property
    def dim(self) -> int:
        """Total dimensionality (discrete flags + continuous)."""
        return self.num_discrete + self.num_continuous

    # ---- Encode / Decode ---------------------------------------------------

    def encode(self, action: Action) -> list[float]:
        """Encode an :class:`Action` to a flat float vector.

        The first *N* values are binary flags for each discrete action,
        followed by [mouse_dx, mouse_dy].
        """
        vec: list[float] = []
        for da in self.enabled_discrete:
            vec.append(1.0 if da in action.discrete else 0.0)
        vec.append(action.mouse_dx)
        vec.append(action.mouse_dy)
        return vec

    def decode(self, vec: list[float]) -> Action:
        """Decode a flat float vector back to an :class:`Action`."""
        discrete = set()
        for i, da in enumerate(self.enabled_discrete):
            if vec[i] > 0.5:
                discrete.add(da)
        mouse_dx = vec[self.num_discrete]
        mouse_dy = vec[self.num_discrete + 1]
        return Action(discrete=discrete, mouse_dx=mouse_dx, mouse_dy=mouse_dy)

    def describe(self) -> dict:
        """Return a human-readable description of the action space."""
        return {
            "discrete_actions": [a.name for a in self.enabled_discrete],
            "mouse_dx_range": self.mouse_dx_range,
            "mouse_dy_range": self.mouse_dy_range,
            "total_dim": self.dim,
        }
