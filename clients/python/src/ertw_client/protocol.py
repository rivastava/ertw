"""ERTW protocol v4 framing and lockstep client."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import socket
import struct
from typing import Any, Mapping, Sequence

PROTOCOL_VERSION = 4
WIRE_MAGIC = int.from_bytes(b"ERTW", "little")
HEADER_WORDS = 14
HEADER_BYTES = HEADER_WORDS * 4
MAX_FRAME_BYTES = 16 * 1024 * 1024

FRAME_HELLO = 1
FRAME_OBSERVATION = 2
FRAME_ACTION = 3
FRAME_METADATA = 4
FRAME_LIFECYCLE = 5
FRAME_RESUME = 6
FRAME_SNAPSHOT = 7
FRAME_OBSERVATION_EXTENSION = 8

SELF_STRIDE = 8
FIELD_COUNT = 3
NEIGHBOR_STRIDE = 15
ACTION_STRIDE = 7

_HEADER = struct.Struct("<14I")
_ACTION = struct.Struct("<7f")
_JSON_KINDS = {
    FRAME_METADATA,
    FRAME_LIFECYCLE,
    FRAME_RESUME,
    FRAME_SNAPSHOT,
    FRAME_OBSERVATION_EXTENSION,
}
_ACTION_SEMANTICS = {"continuous", "level", "edge", "target"}
_LIFECYCLE_KINDS = {
    "entity_alive",
    "entity_died",
    "entity_reproduced",
    "entity_replaced",
    "world_terminated",
    "session_attached",
    "session_detached",
}


class ProtocolError(ValueError):
    """A peer sent a frame that violates the negotiated wire contract."""


@dataclass(frozen=True)
class FrameHeader:
    version: int
    kind: int
    frame_bytes: int
    step: int = 0
    entity_id: int = 0
    max_neighbors: int = 0
    neighbor_count: int = 0
    field_samples: int = 0
    field_channels: int = 0
    payload_floats: int = 0

    @classmethod
    def from_bytes(cls, data: bytes) -> FrameHeader:
        if len(data) != HEADER_BYTES:
            raise ProtocolError(f"header must be {HEADER_BYTES} bytes")
        words = _HEADER.unpack(data)
        if words[0] != WIRE_MAGIC:
            raise ProtocolError("invalid wire magic")
        if words[1] != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version {words[1]}")
        if not HEADER_BYTES <= words[3] <= MAX_FRAME_BYTES:
            raise ProtocolError(f"invalid frame length {words[3]}")
        if words[13] != 0:
            raise ProtocolError("reserved header word must be zero")
        return cls(
            version=words[1],
            kind=words[2],
            frame_bytes=words[3],
            step=words[4] | words[5] << 32,
            entity_id=words[6] | words[7] << 32,
            max_neighbors=words[8],
            neighbor_count=words[9],
            field_samples=words[10],
            field_channels=words[11],
            payload_floats=words[12],
        )

    def to_bytes(self) -> bytes:
        if self.version != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version {self.version}")
        if not 0 <= self.step < 1 << 64 or not 0 <= self.entity_id < 1 << 64:
            raise ProtocolError("step and entity ID must fit u64")
        if not HEADER_BYTES <= self.frame_bytes <= MAX_FRAME_BYTES:
            raise ProtocolError(f"invalid frame length {self.frame_bytes}")
        words = {
            "kind": self.kind,
            "max_neighbors": self.max_neighbors,
            "neighbor_count": self.neighbor_count,
            "field_samples": self.field_samples,
            "field_channels": self.field_channels,
            "payload_floats": self.payload_floats,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 1 << 32
            for value in words.values()
        ):
            raise ProtocolError("header words must fit u32")
        return _HEADER.pack(
            WIRE_MAGIC,
            self.version,
            self.kind,
            self.frame_bytes,
            self.step & 0xFFFF_FFFF,
            self.step >> 32,
            self.entity_id & 0xFFFF_FFFF,
            self.entity_id >> 32,
            self.max_neighbors,
            self.neighbor_count,
            self.field_samples,
            self.field_channels,
            self.payload_floats,
            0,
        )


@dataclass(frozen=True)
class Frame:
    header: FrameHeader
    payload: bytes


@dataclass(frozen=True)
class Neighbor:
    relative_position: tuple[float, float]
    relative_velocity: tuple[float, float]
    mass: float
    structure: float
    energy: float
    tags: int
    conductivity: float
    oscillator_frequency: float
    oscillator_phase: float
    valid: bool


@dataclass(frozen=True)
class Observation:
    step: int
    entity_id: int
    self_state: tuple[float, ...]
    fields: tuple[float, ...]
    neighbors: tuple[Neighbor, ...]


@dataclass(frozen=True)
class Action:
    force_x: float = 0.0
    force_y: float = 0.0
    torque: float = 0.0
    clamp: float = 0.0
    fabricate: float = 0.0
    oscillator_frequency: float = 0.0
    oscillator_phase: float = 0.0

    def values(self) -> tuple[float, ...]:
        values = (
            self.force_x,
            self.force_y,
            self.torque,
            self.clamp,
            self.fabricate,
            self.oscillator_frequency,
            self.oscillator_phase,
        )
        if not all(math.isfinite(value) for value in values):
            raise ProtocolError("actions must contain only finite values")
        return values


@dataclass(frozen=True)
class ResumeCredentials:
    session_id: int
    resume_token: str
    stable_agent_id: int


@dataclass(frozen=True)
class Decision:
    observation: Observation
    extension: Mapping[str, Any] | None


def _recv_exact(stream: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.recv(size - len(chunks))
        if not chunk:
            raise EOFError("connection closed during frame")
        chunks.extend(chunk)
    return bytes(chunks)


def read_frame(stream: socket.socket) -> Frame:
    header = FrameHeader.from_bytes(_recv_exact(stream, HEADER_BYTES))
    payload = _recv_exact(stream, header.frame_bytes - HEADER_BYTES)
    frame = Frame(header, payload)
    _validate_frame(frame)
    return frame


def _validate_frame(frame: Frame) -> None:
    header = frame.header
    if len(frame.payload) != header.frame_bytes - HEADER_BYTES:
        raise ProtocolError("payload length does not match header")
    if header.kind in _JSON_KINDS and header.payload_floats != 0:
        raise ProtocolError("JSON frames cannot declare tensor floats")
    if header.kind == FRAME_HELLO and frame.payload:
        raise ProtocolError("hello frame cannot contain a payload")
    if header.kind == FRAME_ACTION:
        if header.payload_floats != ACTION_STRIDE or len(frame.payload) != _ACTION.size:
            raise ProtocolError("invalid action tensor length")
    if header.kind == FRAME_OBSERVATION:
        expected = (
            SELF_STRIDE
            + FIELD_COUNT * header.field_samples * header.field_channels
            + header.max_neighbors * NEIGHBOR_STRIDE
        )
        if header.neighbor_count > header.max_neighbors:
            raise ProtocolError("valid neighbor count exceeds configured maximum")
        if header.payload_floats != expected or len(frame.payload) != expected * 4:
            raise ProtocolError("invalid observation tensor length")


def _encode_frame(header: FrameHeader, payload: bytes) -> bytes:
    if header.frame_bytes != HEADER_BYTES + len(payload):
        raise ProtocolError("payload length does not match header")
    frame = Frame(header, payload)
    _validate_frame(frame)
    return header.to_bytes() + payload


def encode_action(step: int, entity_id: int, action: Action) -> bytes:
    payload = _ACTION.pack(*action.values())
    return _encode_frame(
        FrameHeader(
            version=PROTOCOL_VERSION,
            kind=FRAME_ACTION,
            frame_bytes=HEADER_BYTES + len(payload),
            step=step,
            entity_id=entity_id,
            payload_floats=ACTION_STRIDE,
        ),
        payload,
    )


def encode_json_frame(kind: int, step: int, entity_id: int, value: object) -> bytes:
    if kind not in _JSON_KINDS:
        raise ProtocolError(f"frame kind {kind} is not a JSON frame")
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return _encode_frame(
        FrameHeader(
            version=PROTOCOL_VERSION,
            kind=kind,
            frame_bytes=HEADER_BYTES + len(payload),
            step=step,
            entity_id=entity_id,
        ),
        payload,
    )


def decode_json(frame: Frame) -> Mapping[str, Any]:
    if frame.header.kind not in _JSON_KINDS:
        raise ProtocolError("frame does not contain JSON")
    try:
        value = json.loads(frame.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("invalid JSON payload") from error
    if not isinstance(value, dict):
        raise ProtocolError("JSON frame payload must be an object")
    return value


def validate_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate the negotiated dimensions needed to decode later frames."""
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "action_floats": ACTION_STRIDE,
        "self_stride": SELF_STRIDE,
        "neighbor_stride": NEIGHBOR_STRIDE,
        "field_count": FIELD_COUNT,
        "transport_mode": "lockstep",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ProtocolError(f"incompatible metadata field {key}")
    max_neighbors = _metadata_int(metadata, "max_neighbors", 32)
    field_samples = _metadata_int(metadata, "field_samples", 32)
    field_channels = _metadata_int(metadata, "field_channels", 32)
    observation_floats = _metadata_int(metadata, "observation_floats", 32)
    physics_ticks = _metadata_int(metadata, "physics_ticks_per_decision", 32)
    session_id = _metadata_int(metadata, "session_id", 128)
    stable_agent_id = _metadata_int(metadata, "stable_agent_id", 64)
    _metadata_int(metadata, "schema_version", 16, minimum=1)
    _metadata_int(metadata, "snapshot_schema_version", 16, minimum=1)
    _metadata_int(metadata, "world_seed", 64)
    _metadata_int(metadata, "world_tick", 64)
    _metadata_int(metadata, "world_id", 128)
    expected_observation = (
        SELF_STRIDE
        + FIELD_COUNT * field_samples * field_channels
        + max_neighbors * NEIGHBOR_STRIDE
    )
    if min(max_neighbors, field_samples, field_channels, physics_ticks) <= 0:
        raise ProtocolError("metadata dimensions and tick hold must be positive")
    if observation_floats != expected_observation:
        raise ProtocolError("metadata observation length is inconsistent")
    if not isinstance(metadata.get("resume_token"), str) or not metadata["resume_token"]:
        raise ProtocolError("metadata resume token is missing")
    fixed_timestep = metadata.get("fixed_timestep_seconds")
    sensor_radius = metadata.get("sensor_radius")
    if not _positive_finite(fixed_timestep) or not _positive_finite(sensor_radius):
        raise ProtocolError("metadata timestep and sensor radius must be positive finite values")
    action_min = _finite_number_list(metadata, "action_min", ACTION_STRIDE)
    action_max = _finite_number_list(metadata, "action_max", ACTION_STRIDE)
    if any(minimum > maximum for minimum, maximum in zip(action_min, action_max)):
        raise ProtocolError("metadata action bounds are inverted")
    semantics = metadata.get("action_semantics")
    if (
        not isinstance(semantics, list)
        or len(semantics) != ACTION_STRIDE
        or any(value not in _ACTION_SEMANTICS for value in semantics)
    ):
        raise ProtocolError("metadata action semantics are invalid")
    capabilities = metadata.get("capabilities")
    if not isinstance(capabilities, list) or not all(
        isinstance(capability, str) for capability in capabilities
    ):
        raise ProtocolError("metadata capabilities must be a string list")


def _metadata_int(
    metadata: Mapping[str, Any], key: str, bits: int, minimum: int = 0
) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"metadata field {key} must be an integer")
    if not minimum <= value < 1 << bits:
        raise ProtocolError(f"metadata field {key} is out of range")
    return value


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _finite_number_list(
    metadata: Mapping[str, Any], key: str, length: int
) -> tuple[float, ...]:
    values = metadata.get(key)
    if not isinstance(values, list) or len(values) != length:
        raise ProtocolError(f"metadata field {key} must have {length} entries")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ProtocolError(f"metadata field {key} must contain finite numbers")
    return tuple(float(value) for value in values)


def validate_lifecycle(event: Mapping[str, Any]) -> None:
    """Validate a lifecycle event before exposing it to agent code."""
    _metadata_int(event, "sequence", 64, minimum=1)
    _metadata_int(event, "world_tick", 64)
    _metadata_int(event, "subject_id", 64)
    if event.get("kind") not in _LIFECYCLE_KINDS:
        raise ProtocolError("lifecycle kind is invalid")
    related_id = event.get("related_id")
    if related_id is not None:
        _metadata_int(event, "related_id", 64)
    reason = event.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ProtocolError("lifecycle reason must be a string or null")
    lineage_id = event.get("lineage_id")
    if lineage_id is not None:
        _metadata_int(event, "lineage_id", 64)
    generation = event.get("generation")
    if generation is not None:
        _metadata_int(event, "generation", 32)


def validate_extension(extension: Mapping[str, Any]) -> None:
    """Validate optional decision metadata and physical deltas."""
    _metadata_int(extension, "decision_sequence", 64, minimum=1)
    delta = extension.get("delta")
    if delta is None:
        return
    if not isinstance(delta, dict) or set(delta) != {"energy", "structure", "mass"}:
        raise ProtocolError("physical delta has an invalid shape")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in delta.values()
    ):
        raise ProtocolError("physical delta must contain finite numbers")


def decode_observation(frame: Frame) -> Observation:
    if frame.header.kind != FRAME_OBSERVATION:
        raise ProtocolError("expected observation frame")
    count = frame.header.payload_floats
    values = struct.unpack(f"<{count}f", frame.payload)
    field_count = FIELD_COUNT * frame.header.field_samples * frame.header.field_channels
    field_end = SELF_STRIDE + field_count
    neighbors = []
    for index in range(frame.header.max_neighbors):
        base = field_end + index * NEIGHBOR_STRIDE
        raw = values[base : base + NEIGHBOR_STRIDE]
        tag_words = tuple(_exact_u16(value) for value in raw[7:11])
        neighbors.append(
            Neighbor(
                relative_position=(raw[0], raw[1]),
                relative_velocity=(raw[2], raw[3]),
                mass=raw[4],
                structure=raw[5],
                energy=raw[6],
                tags=sum(word << (16 * offset) for offset, word in enumerate(tag_words)),
                conductivity=raw[11],
                oscillator_frequency=raw[12],
                oscillator_phase=raw[13],
                valid=raw[14] >= 0.5,
            )
        )
    if sum(neighbor.valid for neighbor in neighbors) != frame.header.neighbor_count:
        raise ProtocolError("valid neighbor flags do not match header")
    return Observation(
        step=frame.header.step,
        entity_id=frame.header.entity_id,
        self_state=tuple(values[:SELF_STRIDE]),
        fields=tuple(values[SELF_STRIDE:field_end]),
        neighbors=tuple(neighbors),
    )


def _exact_u16(value: float) -> int:
    if not math.isfinite(value):
        raise ProtocolError("relation-tag chunks must be exact u16 values")
    integer = int(value)
    if value != integer or not 0 <= integer <= 0xFFFF:
        raise ProtocolError("relation-tag chunks must be exact u16 values")
    return integer


class LockstepClient:
    """Stateful client for one ERTW lockstep session."""

    def __init__(self, stream: socket.socket, address: tuple[str, int]):
        self._stream: socket.socket | None = stream
        self._address = address
        self.metadata: Mapping[str, Any] | None = None
        self.lifecycle: deque[Mapping[str, Any]] = deque()
        self._pending_decision: Decision | None = None

    @classmethod
    def connect(
        cls, address: tuple[str, int], timeout: float | None = None
    ) -> LockstepClient:
        stream = socket.create_connection(address, timeout)
        client = cls(stream, address)
        try:
            client._receive_metadata()
            return client
        except BaseException:
            client.close()
            raise

    @property
    def resume_credentials(self) -> ResumeCredentials:
        if self.metadata is None:
            raise ProtocolError("session metadata has not been received")
        return ResumeCredentials(
            session_id=int(self.metadata["session_id"]),
            resume_token=str(self.metadata["resume_token"]),
            stable_agent_id=int(self.metadata["stable_agent_id"]),
        )

    def next_decision(self) -> Decision:
        if self._pending_decision is not None:
            raise ProtocolError("send the pending action before requesting another decision")
        while True:
            frame = read_frame(self._socket())
            if frame.header.kind == FRAME_LIFECYCLE:
                self._store_lifecycle(decode_json(frame))
                continue
            if frame.header.kind == FRAME_METADATA:
                self._store_metadata(decode_json(frame))
                continue
            if frame.header.kind != FRAME_OBSERVATION:
                raise ProtocolError(f"unexpected frame kind {frame.header.kind}")
            observation = decode_observation(frame)
            extension = None
            if self._has_capability("physical_deltas"):
                extension_frame = read_frame(self._socket())
                if extension_frame.header.kind != FRAME_OBSERVATION_EXTENSION:
                    raise ProtocolError("expected observation extension")
                if extension_frame.header.step != observation.step:
                    raise ProtocolError("observation extension step does not match observation")
                extension = decode_json(extension_frame)
                validate_extension(extension)
            decision = Decision(observation, extension)
            self._pending_decision = decision
            return decision

    def send_action(self, decision: Decision, action: Action) -> None:
        if decision != self._pending_decision:
            raise ProtocolError("action does not correspond to the pending decision")
        observation = decision.observation
        self._socket().sendall(
            encode_action(observation.step, observation.entity_id, action)
        )
        self._pending_decision = None

    def reconnect(self, timeout: float | None = None) -> None:
        credentials = self.resume_credentials
        self.close()
        self._pending_decision = None
        stream = socket.create_connection(self._address, timeout)
        self._stream = stream
        try:
            stream.sendall(
                encode_json_frame(
                    FRAME_RESUME,
                    0,
                    credentials.stable_agent_id,
                    {
                        "session_id": credentials.session_id,
                        "resume_token": credentials.resume_token,
                    },
                )
            )
            self._receive_metadata()
            if self.resume_credentials != credentials:
                raise ProtocolError("resumed session credentials changed")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> LockstepClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _receive_metadata(self) -> None:
        while True:
            frame = read_frame(self._socket())
            if frame.header.kind == FRAME_METADATA:
                self._store_metadata(decode_json(frame))
                return
            if frame.header.kind == FRAME_LIFECYCLE:
                self._store_lifecycle(decode_json(frame))
                continue
            raise ProtocolError("metadata must precede observations")

    def _socket(self) -> socket.socket:
        if self._stream is None:
            raise ProtocolError("client is disconnected")
        return self._stream

    def _store_metadata(self, metadata: Mapping[str, Any]) -> None:
        validate_metadata(metadata)
        if self.metadata is not None:
            previous = dict(self.metadata)
            current = dict(metadata)
            previous.pop("world_tick", None)
            current.pop("world_tick", None)
            if current != previous:
                raise ProtocolError("session metadata changed after negotiation")
        self.metadata = metadata

    def _store_lifecycle(self, event: Mapping[str, Any]) -> None:
        validate_lifecycle(event)
        self.lifecycle.append(event)

    def _has_capability(self, name: str) -> bool:
        if self.metadata is None:
            return False
        capabilities = self.metadata.get("capabilities", [])
        return (
            isinstance(capabilities, Sequence)
            and not isinstance(capabilities, (str, bytes))
            and name in capabilities
        )
