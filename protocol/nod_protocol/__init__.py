"""nod_protocol: the shared message and registry contract for Nod (design sections 3.5 and 4.2).

Import everything from the package root, e.g.

    from nod_protocol import UiConfirm, Proposal, RunSkill, WorldState, SkillSpec, parse_ws, dumps, loads
"""

from .arrays import F32Array
from .atoms import Lit, Rollout, apply_effects, atoms_of, holds, parse_atom, parse_literal, rollout, unmet
from .base import (
    PROTOCOL_VERSION,
    ConfirmPolicy,
    CosmosAnswer,
    CosmosDecision,
    Envelope,
    Fault,
    HomeAction,
    NodModel,
    OptionConfirm,
    OptionKind,
    PostDecision,
    Risk,
    RunPhase,
    ScriptName,
    SessionMode,
    SkillKind,
    StrictModel,
    ZmqMessage,
)
from .codec import (
    ALL_TYPES,
    ANY,
    CTL,
    GATEWAY_OUT,
    PHONE_IN,
    PRESENTER_IN,
    SAFETY_IN,
    SIM_OUT,
    TYPE_TO_MODEL,
    WS,
    WS_TYPES,
    ZMQ_TYPES,
    dumps,
    loads,
    packb,
    parse_ws,
    to_ws,
    unpackb,
)
from .cosmos import CosmosCheck, CosmosReply, CosmosResult, FrameRef, HomeResult, Metrics, OracleAtom, Trace
from .gateway import (
    Badge,
    Check,
    CosmosVerdict,
    EStopAck,
    FixedOption,
    LatMs,
    Notice,
    Option,
    Pong,
    Precheck,
    Proposal,
    RunUpdate,
    State,
    Success,
    Thumb,
    Transcript,
    Welcome,
)
from .ids import new_id, new_ulid, now_ms, ulid_time_ms, is_ulid
from .phone import (
    Calib,
    EStop,
    EStopReset,
    EStopSource,
    Gesture,
    GestureName,
    Heartbeat,
    Hello,
    Ping,
    UiCancel,
    UiConfirm,
    UiFocus,
    UiMenu,
    UtterEnd,
    UtterStart,
    UtterText,
)
from .planner import (
    MENU_NO_SPEECH,
    FailureInfo,
    FeasibleSkill,
    HistoryItem,
    InfeasibleSkill,
    Intent,
    PlannerOption,
    PlannerOutput,
    PlanRequest,
    Profile,
    planner_response_format,
)
from .sim import (
    DEFAULT_EEF_BOX,
    Abort,
    Capture,
    CtlEStop,
    CtlHeartbeat,
    Frame,
    Limits,
    LoadScene,
    ModelPatch,
    Pause,
    Pose,
    Resume,
    RunScripted,
    RunSkill,
    SimAck,
    SimEpisodeHeader,
    SimEStopAck,
    SimFrames,
    SimPhase,
    SimStatus,
    SimWorld,
    WorldRequest,
)
from .skills import (
    ChainStats,
    CosmosQuestions,
    HomeSpec,
    PolicySpec,
    ScriptedSpec,
    SkillSpec,
    SkillStats,
    StatBlock,
    StatsFile,
    apply_stats,
    check_registry,
    load_skills,
    load_stats,
    plan_rollout,
    registry_atoms,
    registry_objects,
    wilson_lower,
)
from .world import Eef, ObjPose, WorldState

__version__ = "0.1.0"

__all__ = [name for name in dir() if not name.startswith("_")]
