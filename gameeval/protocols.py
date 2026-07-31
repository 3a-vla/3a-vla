"""Named evaluation protocols and their judge boundaries."""

from __future__ import annotations

CSGO_STATE = "csgo-state"
GTA5_VISUAL = "gta5-visual"
GP_STATE = "gp-state"
GP_VISUAL = "gp-visual"

PROTOCOLS_BY_GAME: dict[str, frozenset[str]] = {
    "csgo": frozenset({CSGO_STATE}),
    "gta5": frozenset({GTA5_VISUAL}),
    "gp": frozenset({GP_STATE, GP_VISUAL}),
}
DEFAULT_PROTOCOL_BY_GAME = {
    "csgo": CSGO_STATE,
    "gta5": GTA5_VISUAL,
}
EVALUATOR_BY_PROTOCOL = {
    CSGO_STATE: "state",
    GTA5_VISUAL: "vlm",
    GP_STATE: "state",
    GP_VISUAL: "vlm",
}


def resolve_protocol(game: str, protocol: str | None = None) -> str:
    """Validate a protocol, requiring an explicit choice for GP."""
    normalized_game = str(game).strip().lower()
    if normalized_game not in PROTOCOLS_BY_GAME:
        allowed = ", ".join(sorted(PROTOCOLS_BY_GAME))
        raise ValueError(f"Unsupported game '{normalized_game}'; expected one of: {allowed}")

    normalized_protocol = str(protocol or "").strip().lower()
    if not normalized_protocol:
        if normalized_game == "gp":
            raise ValueError("GP tasks and runs must explicitly select gp-state or gp-visual")
        normalized_protocol = DEFAULT_PROTOCOL_BY_GAME[normalized_game]

    allowed_protocols = PROTOCOLS_BY_GAME[normalized_game]
    if normalized_protocol not in allowed_protocols:
        allowed = ", ".join(sorted(allowed_protocols))
        raise ValueError(
            f"Protocol '{normalized_protocol}' is invalid for {normalized_game}; expected: {allowed}"
        )
    return normalized_protocol


def expected_evaluator(protocol: str) -> str:
    try:
        return EVALUATOR_BY_PROTOCOL[protocol]
    except KeyError as exc:
        raise ValueError(f"Unsupported evaluation protocol: {protocol}") from exc


__all__ = [
    "CSGO_STATE",
    "GTA5_VISUAL",
    "GP_STATE",
    "GP_VISUAL",
    "PROTOCOLS_BY_GAME",
    "resolve_protocol",
    "expected_evaluator",
]
