"""CSGO game state data classes.

Parsed from the JSON payload sent by CSGO's Game State Integration (GSI).
See: https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Game_State_Integration
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Vector3:
    """3D position / velocity."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_str(cls, s: str) -> Vector3:
        """Parse from CSGO format ``'x, y, z'``."""
        parts = [float(p.strip()) for p in s.split(",")]
        return cls(x=parts[0], y=parts[1], z=parts[2] if len(parts) > 2 else 0.0)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class WeaponState:
    """Current weapon information."""

    name: str = ""
    type: str = ""  # "Rifle", "Pistol", "Knife", "SniperRifle", ...
    ammo_clip: int = 0
    ammo_clip_max: int = 0
    ammo_reserve: int = 0
    state: str = ""  # "active", "holstered"

    @classmethod
    def from_dict(cls, d: dict) -> WeaponState:
        return cls(
            name=d.get("name", ""),
            type=d.get("type", ""),
            ammo_clip=int(d.get("ammo_clip", 0)),
            ammo_clip_max=int(d.get("ammo_clip_max", 0)),
            ammo_reserve=int(d.get("ammo_reserve", 0)),
            state=d.get("state", ""),
        )


@dataclass
class PlayerState:
    """Player-level game state."""

    steam_id: str = ""
    name: str = ""
    team: str = ""  # "T" or "CT"
    health: int = 0
    armor: int = 0
    helmet: bool = False
    money: int = 0
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    mvps: int = 0
    score: int = 0
    # Per-round counters from ``player.state``; CSGO resets them each round.
    round_kills: int = 0
    round_killhs: int = 0
    position: Vector3 = field(default_factory=Vector3)
    forward: Vector3 = field(default_factory=Vector3)
    weapons: list[WeaponState] = field(default_factory=list)
    active_weapon: WeaponState | None = None
    is_alive: bool = True

    @classmethod
    def from_gsi(cls, player_data: dict) -> PlayerState:
        """Parse from GSI player payload."""
        state = player_data.get("state", {})
        match_stats = player_data.get("match_stats", {})

        # Parse weapons
        weapons_raw = player_data.get("weapons", {})
        weapons = []
        active_weapon = None
        for _key, w in weapons_raw.items():
            ws = WeaponState.from_dict(w)
            weapons.append(ws)
            if w.get("state") == "active":
                active_weapon = ws

        # Parse position (if available)
        pos = Vector3()
        fwd = Vector3()
        if "position" in player_data:
            pos = Vector3.from_str(player_data["position"])
        if "forward" in player_data:
            fwd = Vector3.from_str(player_data["forward"])

        return cls(
            steam_id=player_data.get("steamid", ""),
            name=player_data.get("name", ""),
            team=player_data.get("team", ""),
            health=int(state.get("health", 0)),
            armor=int(state.get("armor", 0)),
            helmet=state.get("helmet", False),
            money=int(state.get("money", 0)),
            kills=int(match_stats.get("kills", 0)),
            deaths=int(match_stats.get("deaths", 0)),
            assists=int(match_stats.get("assists", 0)),
            mvps=int(match_stats.get("mvps", 0)),
            score=int(match_stats.get("score", 0)),
            round_kills=int(state.get("round_kills", 0)),
            round_killhs=int(state.get("round_killhs", 0)),
            position=pos,
            forward=fwd,
            weapons=weapons,
            active_weapon=active_weapon,
            is_alive=int(state.get("health", 0)) > 0,
        )

    def to_dict(self) -> dict:
        return {
            "steam_id": self.steam_id,
            "name": self.name,
            "team": self.team,
            "health": self.health,
            "armor": self.armor,
            "helmet": self.helmet,
            "money": self.money,
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "score": self.score,
            "round_kills": self.round_kills,
            "round_killhs": self.round_killhs,
            "position": self.position.to_dict(),
            "forward": self.forward.to_dict(),
            "is_alive": self.is_alive,
            "active_weapon": self.active_weapon.name if self.active_weapon else None,
            "ammo_clip": self.active_weapon.ammo_clip if self.active_weapon else 0,
            "ammo_reserve": self.active_weapon.ammo_reserve if self.active_weapon else 0,
        }


@dataclass
class BombState:
    """Bomb-related state."""

    state: str = ""  # "carried", "planted", "defused", "exploded", ""
    position: Vector3 = field(default_factory=Vector3)
    countdown: float = 0.0

    @classmethod
    def from_gsi(cls, bomb_data: dict) -> BombState:
        pos = Vector3()
        if "position" in bomb_data:
            pos = Vector3.from_str(bomb_data["position"])
        return cls(
            state=bomb_data.get("state", ""),
            position=pos,
            countdown=float(bomb_data.get("countdown", 0)),
        )


@dataclass
class RoundState:
    """Round-level information."""

    phase: str = ""  # "live", "freezetime", "over", "warmup"
    win_team: str = ""
    bomb: str = ""  # "planted", "exploded", "defused", ""
    round_number: int = 0

    @classmethod
    def from_gsi(cls, round_data: dict, map_data: dict | None = None) -> RoundState:
        rnd_num = 0
        if map_data:
            ct_score = int(map_data.get("team_ct", {}).get("score", 0))
            t_score = int(map_data.get("team_t", {}).get("score", 0))
            rnd_num = ct_score + t_score + 1
        return cls(
            phase=round_data.get("phase", ""),
            win_team=round_data.get("win_team", ""),
            bomb=round_data.get("bomb", ""),
            round_number=rnd_num,
        )


@dataclass
class MapState:
    """Map and score information."""

    name: str = ""
    phase: str = ""  # "warmup", "live", "intermission", "gameover"
    mode: str = ""  # "competitive", "casual", "deathmatch"
    ct_score: int = 0
    t_score: int = 0
    num_matches_to_win_series: int = 0

    @classmethod
    def from_gsi(cls, map_data: dict) -> MapState:
        return cls(
            name=map_data.get("name", ""),
            phase=map_data.get("phase", ""),
            mode=map_data.get("mode", ""),
            ct_score=int(map_data.get("team_ct", {}).get("score", 0)),
            t_score=int(map_data.get("team_t", {}).get("score", 0)),
            num_matches_to_win_series=int(map_data.get("num_matches_to_win_series", 0)),
        )


@dataclass
class GameState:
    """Complete CSGO game state assembled from a GSI payload.

    This is the top-level container that holds all parsed information.
    """

    player: PlayerState = field(default_factory=PlayerState)
    map_state: MapState = field(default_factory=MapState)
    round_state: RoundState = field(default_factory=RoundState)
    bomb: BombState = field(default_factory=BombState)
    all_players: list[PlayerState] = field(default_factory=list)
    timestamp: float = 0.0

    @classmethod
    def from_gsi_payload(cls, payload: dict) -> GameState:
        """Parse a complete GSI JSON payload into a GameState."""
        import time

        provider_steamid = str(payload.get("provider", {}).get("steamid", ""))

        # Map
        map_data = payload.get("map", {})
        map_state = MapState.from_gsi(map_data) if map_data else MapState()

        # Round
        round_data = payload.get("round", {})
        round_state = RoundState.from_gsi(round_data, map_data) if round_data else RoundState()

        # Bomb
        bomb_data = payload.get("bomb", {})
        bomb_state = BombState.from_gsi(bomb_data) if bomb_data else BombState()

        # Player (observer's own player data)
        player_data = payload.get("player", {})
        player = PlayerState.from_gsi(player_data) if player_data else PlayerState()

        # All players (if available in spectator mode)
        all_players_data = payload.get("allplayers", {})
        all_players = []
        for _pid, pdata in all_players_data.items():
            # `_pid` is the steamid key; inject it so PlayerState can
            # carry it even when individual player dicts omit "steamid".
            if "steamid" not in pdata:
                pdata = {**pdata, "steamid": _pid}
            all_players.append(PlayerState.from_gsi(pdata))

        # After death, CSGO may put the spectated entity in ``player``. Keep
        # the normalized subject tied to the local client when allplayers is
        # available, without adding task- or episode-specific logic.
        if provider_steamid and player.steam_id != provider_steamid:
            local_player = next(
                (item for item in all_players if item.steam_id == provider_steamid),
                None,
            )
            if local_player is not None:
                player = local_player

        return cls(
            player=player,
            map_state=map_state,
            round_state=round_state,
            bomb=bomb_state,
            all_players=all_players,
            timestamp=time.time(),
        )

    def to_dict(self) -> dict:
        """Convert to a flat dictionary suitable for observation state."""
        return {
            "player_health": self.player.health,
            "player_armor": self.player.armor,
            "player_position": self.player.position.to_dict(),
            "player_forward": self.player.forward.to_dict(),
            "player_team": self.player.team,
            "player_is_alive": self.player.is_alive,
            "current_weapon": self.player.active_weapon.name if self.player.active_weapon else "",
            "ammo_clip": self.player.active_weapon.ammo_clip if self.player.active_weapon else 0,
            "ammo_reserve": (
                self.player.active_weapon.ammo_reserve if self.player.active_weapon else 0
            ),
            "player_kills": self.player.kills,
            "player_deaths": self.player.deaths,
            "map_name": self.map_state.name,
            "map_phase": self.map_state.phase,
            "round_phase": self.round_state.phase,
            "round_number": self.round_state.round_number,
            "ct_score": self.map_state.ct_score,
            "t_score": self.map_state.t_score,
            "bomb_state": self.round_state.bomb or self.bomb.state,
            "bomb_position": self.bomb.position.to_dict(),
            "num_players": len(self.all_players),
            "enemies": [
                p.to_dict()
                for p in self.all_players
                if p.team != self.player.team and p.is_alive
            ],
            "teammates": [
                p.to_dict()
                for p in self.all_players
                if p.team == self.player.team
                and p.steam_id != self.player.steam_id
                and p.is_alive
            ],
        }
