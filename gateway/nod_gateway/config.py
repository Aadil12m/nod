"""Gateway configuration (design 4.2.2).

Every field can be set from the environment with the `NOD_` prefix, e.g.
`NOD_MODE=mock`, `NOD_HB_TIMEOUT_MS=500`. The Token Factory key is also read from
`NEBIUS_API_KEY`, the name the Nebius docs use.

`mode` picks a default for each dependency (planner, ASR, Cosmos, Home, sim); each
choice can be overridden on its own, e.g. `NOD_MODE=mock NOD_PLANNER=nemotron` to try
the real planner against the mock sim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

PlannerKind = Literal["nemotron", "fake"]
AsrKind = Literal["parakeet", "fake"]
CosmosKind = Literal["vllm", "fake", "none"]
HomeKind = Literal["hass", "fake"]
StructuredMode = Literal["auto", "json_schema", "tools", "json_object"]


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOD_", env_file=None, extra="ignore", populate_by_name=True)

    mode: Literal["live", "mock"] = "mock"
    dev: bool = False  # NOD_DEV=1 enables /api/dev/inject in live mode

    # ---- Nemotron on Token Factory (7.1)
    tf_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("NOD_TF_API_KEY", "NEBIUS_API_KEY", "tf_api_key")
    )
    tf_fast_url: str = "https://api.tokenfactory.nebius.com/v1/"
    tf_plan_url: str = "https://api.tokenfactory.us-central1.nebius.com/v1/"
    fast_model: str = "nvidia/Nemotron-3_5-Lightning"
    plan_model: str = "nvidia/nemotron-3-super-120b-a12b"
    fast_timeout_s: float = 2.5
    plan_timeout_s: float = 8.0
    planner_structured: StructuredMode = "auto"  # G3 settles which mechanism works

    # ---- sim (3.3)
    sim_ctl: str = "tcp://sim:5560"
    sim_estop: str = "tcp://sim:5561"
    sim_pub: str = "tcp://sim:5562"
    sim_pose_pub: str = "tcp://sim:5563"
    scene_id: str = "libero_goal_v1"
    init_index: int = 0
    sim_ack_timeout_s: float = 3.0
    sim_status_timeout_s: float = 5.0  # no status for the run in flight for this long: fail the run

    # ---- Cosmos, Home Assistant
    cosmos_url: str | None = None  # e.g. http://127.0.0.1:8001/v1
    cosmos_model: str = "nvidia/Cosmos-Reason2-8B"
    cosmos_timeout_s: float = 4.0
    cosmos_cameras: list[str] = Field(default_factory=lambda: ["frontview"])
    hass_url: str | None = None  # e.g. http://homeassistant:8123
    hass_token: str | None = None
    hass_light_entity: str = "light.ceiling_lights"  # HA demo integration light
    hass_notify_service: str = "persistent_notification"  # notify.<service>

    # ---- safety parameters (5.2, 5.6)
    hb_timeout_ms: int = 500
    face_lost_ms: int = 1000
    proposal_ttl_s: int = 30
    arm_delay_ms: int = 300
    focus_dwell_ms: int = 300
    min_confirm_hold_ms: int = 600  # lower end of the calibrated long-blink range (6.1)
    countdown_s: float = 1.0
    plan_countdown_s: float = 3.0
    spacing_ms: int = 2000
    near_user_scale: float = 0.6
    estop_ack_wait_ms: int = 1000  # wait this long for sim.estop_ack before acking without sim fields
    caregiver_enabled: bool = False  # R7; can be toggled by /api/dev/inject
    plan_mode: Literal["per_step", "whole"] = "per_step"
    preferred_name: str | None = None
    caregiver_name: str | None = None

    # ---- dependency choices (None = chosen by mode)
    planner: PlannerKind | None = None
    asr: AsrKind | None = None
    cosmos: CosmosKind | None = None
    home: HomeKind | None = None
    mock_runner: bool | None = None  # start the in-process MockRunner (default: on in mock mode)

    # ---- mock runner
    mock_hz: int = 20
    mock_steps: int = 60  # steps until a mock skill "succeeds"
    mock_failure_rate: float = 0.0
    mock_seed: int = 0
    mock_hb_timeout_ms: int | None = 500  # sim-side watchdog; None disables

    # ---- paths
    registry_path: Path = REPO_ROOT / "registry" / "skills.yaml"
    stats_path: Path | None = REPO_ROOT / "registry" / "stats" / "latest.json"
    pwa_dir: Path = REPO_ROOT / "pwa"
    runs_dir: Path = Path("runs")
    recordings_dir: Path = REPO_ROOT / "recordings"
    canned_dir: Path = Path(__file__).resolve().parent / "planner" / "canned"
    fake_transcripts: list[str] = Field(default_factory=lambda: ["I'm a bit hungry"])

    @model_validator(mode="after")
    def _mock_addresses(self) -> Config:
        # In mock mode the sim is the in-process MockRunner on localhost, not the compose host "sim".
        if self.mode == "mock":
            for name, port in (("sim_ctl", 5560), ("sim_estop", 5561), ("sim_pub", 5562), ("sim_pose_pub", 5563)):
                if getattr(self, name) == f"tcp://sim:{port}":
                    setattr(self, name, f"tcp://127.0.0.1:{port}")
        return self

    # ---- resolved choices
    @property
    def planner_kind(self) -> PlannerKind:
        return self.planner or ("fake" if self.mode == "mock" else "nemotron")

    @property
    def asr_kind(self) -> AsrKind:
        return self.asr or ("fake" if self.mode == "mock" else "parakeet")

    @property
    def cosmos_kind(self) -> CosmosKind:
        if self.cosmos:
            return self.cosmos
        if self.mode == "mock":
            return "fake"
        return "vllm" if self.cosmos_url else "none"

    @property
    def home_kind(self) -> HomeKind:
        return self.home or ("fake" if self.mode == "mock" or not self.hass_url else "hass")

    @property
    def run_mock_runner(self) -> bool:
        return self.mode == "mock" if self.mock_runner is None else self.mock_runner

    @property
    def dev_enabled(self) -> bool:
        return self.mode == "mock" or self.dev
