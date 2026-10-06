"""SafetyChannel unit tests (design 5.5, 5.6): the e-stop fast path, acks, de-dup, heartbeat and watchdog."""

from __future__ import annotations

import inspect

import pytest
from nod_protocol import EStop, EStopAck, EStopReset, Heartbeat, SimEStopAck

from nod_gateway.config import Config
from nod_gateway.metrics import MetricsHub
from nod_gateway.peers import Peer
from nod_gateway.safety import HbState, SafetyChannel
from nod_gateway.simlink import SimError


class Clock:
    def __init__(self, t: int = 1_000_000) -> None:
        self.t = t

    def __call__(self) -> int:
        return self.t


class FakeSim:
    def __init__(self, log: list, fail: bool = False) -> None:
        self.log = log
        self.fail = fail
        self.hbs: list[int] = []

    def estop(self, seq: int) -> None:
        self.log.append(("sim.estop", seq))
        if self.fail:
            raise SimError("not connected")

    def send_hb(self, t: int, phone_ok: bool = True) -> bool:
        self.hbs.append(t)
        return True


class FakeSession:
    def __init__(self, log: list) -> None:
        self.log = log
        self.mode = "EXECUTING"
        self.epochs: list[int] = []
        self.pauses: list[str] = []

    def latch_estop(self, reason: str) -> None:
        self.log.append(("session.latch", reason))
        self.mode = "ESTOPPED"

    def reset_estop(self, msg: EStopReset) -> bool:
        if self.mode != "ESTOPPED":
            return False
        self.mode = "IDLE"
        return True

    def protective_pause(self, reason: str) -> None:
        self.pauses.append(reason)
        self.mode = "PAUSED"

    def on_estop_seq(self, seq: int) -> None:
        self.log.append(("session.seq", seq))

    def on_sim_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


@pytest.fixture()
def rig():
    log: list = []
    clock = Clock()
    sim, sess, hb = FakeSim(log), FakeSession(log), HbState()
    metrics = MetricsHub()
    ch = SafetyChannel(Config(mode="mock"), sim, sess, hb, metrics=metrics, clock=clock)
    peer = Peer(None, "safety", "s_1")
    return ch, sim, sess, hb, clock, peer, log, metrics


def estop(seq: int = 3, ts: int = 1_000_000 - 40) -> EStop:
    return EStop(estop_seq=seq, source="jaw_hold", t_detect=ts - 5, held_ms=310, ts=ts)


def test_on_estop_is_synchronous():
    assert not inspect.iscoroutinefunction(SafetyChannel.on_estop)
    assert not inspect.iscoroutinefunction(SafetyChannel.on_hb)
    assert not inspect.iscoroutinefunction(SafetyChannel.watchdog_tick)


def test_estop_forwards_to_sim_before_latching(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    ch.on_estop(estop(), peer)
    assert log[0] == ("sim.estop", 1)  # the gateway's own seq goes to the sim, first
    # The client's number is recorded before the latch, so the ESTOPPED `state` carries it.
    assert log[1] == ("session.seq", 3)
    assert log[2] == ("session.latch", "jaw_hold")
    assert sess.mode == "ESTOPPED"


def test_ack_after_sim_ack_with_latency(rig):
    ch, sim, sess, hb, clock, peer, log, metrics = rig
    ch.on_estop(estop(seq=3, ts=clock.t - 40), peer)
    assert peer.sent == []  # not acked until the sim confirms the halt
    ch.on_sim_estop_ack(SimEStopAck(estop_seq=1, epoch=7, stopped_at_step=143, t_rx=clock.t + 2, t_halt=clock.t + 20))
    (ack,) = peer.sent
    assert isinstance(ack, EStopAck) and ack.estop_seq == 3 and ack.stopped_at_step == 143
    assert ack.lat_ms.phone_gw == 40 and ack.lat_ms.gw_sim == 2 and ack.lat_ms.sim_halt == 18
    assert sess.epochs == [7]
    assert metrics.h["estop_ms"].pct(50) == 60


def test_duplicate_estop_seq_ignored_and_ack_resent(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    ch.on_estop(estop(seq=3), peer)
    ch.on_estop(estop(seq=3), peer)  # resend before the ack
    assert [e for e in log if e[0] == "sim.estop"] == [("sim.estop", 1)]
    assert peer.sent == []
    ch.on_sim_estop_ack(SimEStopAck(estop_seq=1, epoch=1, stopped_at_step=5, t_rx=clock.t, t_halt=clock.t))
    ch.on_estop(estop(seq=3), peer)  # resend crossing the ack: same ack again, no new stop
    assert len(peer.sent) == 2 and peer.sent[0].estop_seq == peer.sent[1].estop_seq == 3
    assert [e for e in log if e[0] == "sim.estop"] == [("sim.estop", 1)]


def test_new_seq_is_a_new_stop(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    ch.on_estop(estop(seq=3), peer)
    ch.on_estop(estop(seq=4), peer)
    assert [e for e in log if e[0] == "sim.estop"] == [("sim.estop", 1), ("sim.estop", 2)]


def test_same_seq_after_reset_is_processed(rig):
    """A phone that reuses an old estop_seq after a reset must still stop the robot."""
    ch, sim, sess, hb, clock, peer, log, _ = rig
    ch.on_estop(estop(seq=3), peer)
    ch.on_sim_estop_ack(SimEStopAck(estop_seq=1, epoch=1, t_rx=clock.t, t_halt=clock.t))
    assert ch.on_estop_reset(EStopReset(estop_seq=3, gesture="long_blink"), peer)
    sess.mode = "EXECUTING"
    ch.on_estop(estop(seq=3), peer)
    assert [e for e in log if e[0] == "sim.estop"] == [("sim.estop", 1), ("sim.estop", 2)]


def test_same_seq_from_other_peer_is_processed(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    ch.on_estop(estop(seq=1), peer)
    presenter = Peer(None, "presenter", "s_1")
    ch.on_estop(EStop(estop_seq=1, source="presenter", t_detect=clock.t), presenter)
    assert len([e for e in log if e[0] == "sim.estop"]) == 2


def test_estop_latches_even_if_sim_unreachable(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    sim.fail = True
    ch.on_estop(estop(seq=9), peer)
    assert sess.mode == "ESTOPPED"
    assert peer.sent == []
    clock.t += 999
    ch.watchdog_tick()
    assert peer.sent == []
    clock.t += 1
    ch.watchdog_tick()  # no sim ack within estop_ack_wait_ms: ack without sim fields
    (ack,) = peer.sent
    assert ack.estop_seq == 9 and ack.lat_ms.gw_sim is None and ack.lat_ms.sim_halt is None
    assert ack.stopped_at_step is None


def test_no_hb_relay_while_estop_unconfirmed(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    ch.on_hb(Heartbeat(face_present=True, track_conf=0.9, fps=24, latched=False), peer)
    ch.on_estop(estop(), peer)
    sim.hbs.clear()
    for _ in range(10):
        clock.t += 50
        ch.on_hb(Heartbeat(face_present=True, track_conf=0.9, fps=24, latched=True), peer)
        ch.watchdog_tick()
    assert sim.hbs == []  # the sim's own watchdog would stop it if the e-stop was lost
    ch.on_sim_estop_ack(SimEStopAck(estop_seq=1, epoch=1, t_rx=clock.t, t_halt=clock.t))
    clock.t += 100
    ch.watchdog_tick()
    assert len(sim.hbs) == 1


def test_late_sim_ack_after_timeout_clears_unconfirmed(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    ch.on_estop(estop(), peer)
    clock.t += 1500
    ch.watchdog_tick()
    assert len(peer.sent) == 1 and ch.unacked()
    ch.on_sim_estop_ack(SimEStopAck(estop_seq=1, epoch=1, t_rx=clock.t, t_halt=clock.t))
    assert not ch.unacked() and len(peer.sent) == 1


def test_hb_relay_only_while_fresh_every_100ms(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    sess.mode = "IDLE"
    ch.watchdog_tick()
    assert sim.hbs == []  # no phone heartbeat yet
    ch.on_hb(Heartbeat(face_present=True, track_conf=0.9, fps=24, latched=False), peer)
    for _ in range(4):  # 200 ms of 50 ms ticks
        ch.watchdog_tick()
        clock.t += 50
    assert len(sim.hbs) == 2
    clock.t += 500  # now 700 ms since the last phone hb: stale
    ch.watchdog_tick()
    assert len(sim.hbs) == 2


def test_hb_loss_pauses_execution(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    ch.on_hb(Heartbeat(face_present=True, track_conf=0.9, fps=24, latched=False), peer)
    clock.t += 500
    ch.watchdog_tick()
    assert sess.pauses == []  # exactly 500 ms is still fresh
    clock.t += 1
    ch.watchdog_tick()
    assert sess.pauses == ["hb_lost"]


def test_face_lost_pauses_after_1000ms(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    for i in range(12):
        ch.on_hb(Heartbeat(face_present=False, track_conf=0.0, fps=24, latched=False), peer)
        ch.watchdog_tick()
        if i < 10:
            assert sess.pauses == [], i
        clock.t += 100
    assert sess.pauses == ["face_lost"]


def test_face_found_resets_timer(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    for i in range(30):
        ch.on_hb(Heartbeat(face_present=(i % 8 == 0), track_conf=0.5, fps=24, latched=False), peer)
        ch.watchdog_tick()
        clock.t += 100
    assert sess.pauses == []


def test_no_pause_when_idle(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    sess.mode = "IDLE"
    clock.t += 5000
    ch.watchdog_tick()
    assert sess.pauses == []


def test_latched_heartbeat_latches_gateway(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    ch.on_hb(Heartbeat(face_present=True, track_conf=0.9, fps=24, latched=True), peer)
    assert sess.mode == "ESTOPPED"
    assert ("sim.estop", 1) in log
    assert peer.sent == []  # no client seq: nothing to ack


def test_latched_heartbeat_ignored_just_after_reset(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    ch.on_estop(estop(), peer)
    ch.on_sim_estop_ack(SimEStopAck(estop_seq=1, epoch=1, t_rx=clock.t, t_halt=clock.t))
    assert ch.on_estop_reset(EStopReset(estop_seq=3, gesture="long_blink"), peer)
    clock.t += 100
    ch.on_hb(Heartbeat(face_present=True, track_conf=0.9, fps=24, latched=True), peer)  # in flight
    assert sess.mode == "IDLE"
    clock.t += 600
    ch.on_hb(Heartbeat(face_present=True, track_conf=0.9, fps=24, latched=True), peer)
    assert sess.mode == "ESTOPPED"


def test_reset_only_when_estopped(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    sess.mode = "IDLE"
    assert not ch.on_estop_reset(EStopReset(estop_seq=1, gesture="touch"), peer)


def test_phone_gw_latency_with_skewed_clock(rig):
    ch, sim, sess, hb, clock, peer, log, _ = rig
    # Phone clock 5 s behind: the ping estimate corrects it (offset + min uplink delay).
    ch.on_ping_sample(t0_phone=clock.t - 5000 - 30, t_rx=clock.t)
    ch.on_estop(estop(seq=1, ts=clock.t - 5000 - 30), peer)
    ch.on_sim_estop_ack(SimEStopAck(estop_seq=1, epoch=1, t_rx=clock.t, t_halt=clock.t))
    assert peer.sent[0].lat_ms.phone_gw == 0  # bounded by the estimate; never absurd
    # Without an estimate, an implausible value is dropped rather than reported.
    ch2 = SafetyChannel(Config(mode="mock"), FakeSim([]), FakeSession([]), HbState(), clock=clock)
    p2 = Peer(None, "safety", "s")
    ch2.on_estop(estop(seq=1, ts=clock.t - 50_000), p2)
    ch2.on_sim_estop_ack(SimEStopAck(estop_seq=1, epoch=1, t_rx=clock.t, t_halt=clock.t))
    assert p2.sent[0].lat_ms.phone_gw is None
