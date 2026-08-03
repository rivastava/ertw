# ERTW protocol v4

All integers and floats are little-endian. Every TCP message starts with fourteen
`u32` words. Tensor frames continue with IEEE-754 `f32` values; metadata,
lifecycle, resume, and extension frames continue with UTF-8 JSON bytes.

| Word | Meaning |
| ---: | --- |
| 0 | Magic: little-endian `ERTW` |
| 1 | Protocol version (`4`) |
| 2 | Frame kind; see below |
| 3 | Total frame length in bytes |
| 4–5 | Simulation step, low then high 32 bits |
| 6–7 | Entity ID, low then high 32 bits |
| 8 | Configured maximum neighbors |
| 9 | Valid neighbors in this observation |
| 10 | Field sample count |
| 11 | Channels per field sample |
| 12 | Tensor payload float count; zero for JSON frames |
| 13 | Reserved; must be zero |

The server sends a hello frame after connection. Observation and action frames
are self-delimiting and validate their magic, version, kind, length, and payload
size. Implementations must reject frames larger than 16 MiB.

Protocol v4 adds JSON payload frames while retaining the fixed binary header:

| Kind | Frame |
| ---: | --- |
| 1 | Legacy hello |
| 2 | Observation tensor |
| 3 | Action tensor |
| 4 | Negotiated protocol metadata |
| 5 | Generic lifecycle event |
| 6 | Resume request |
| 7 | Reserved for canonical snapshot transfer |
| 8 | Optional observation extension/deltas |

Lockstep mode begins with metadata and `session_attached` lifecycle frames.
After every decision-boundary observation it sends an optional extension frame,
then waits without a deadline for the exactly matching action.

## Observation payload

The payload is:

```text
self[8] ++ fields[3 * field_samples * field_channels]
        ++ neighbors[max_neighbors * 15]
```

Self state contains local velocity X/Y, mass, structure, stored energy,
oscillator frequency, oscillator phase, and reproductive energy surplus.
Absolute position is never exposed.

For each Kinetic, Thermal, and Electromagnetic field, every probe contains the
field value and its X/Y gradient in the agent's local frame. Probe zero is at
the agent; remaining probes form an egocentric ring at the sensor radius.

Each neighbor contains local relative position X/Y, local relative velocity X/Y,
mass, structure, energy, four exact 16-bit chunks of the 64-bit relation mask,
conductivity, oscillator frequency, oscillator phase, and a valid flag. Four
16-bit chunks are used because every `u16` is exactly representable by `f32`.

## Action payload

Actions contain force X/Y, torque, clamp, fabricate, oscillator frequency, and
oscillator phase. The server sanitizes non-finite and out-of-range values.

Socket I/O runs on a background thread. Simulation never waits for a client.
Queued observations may be dropped under backpressure; actions older than two
simulation steps resolve to a no-op.

This behavior applies to real-time mode. Lockstep mode deliberately pauses the
world at a decision boundary until a matching action arrives. It then retains
continuous, level, and target commands for `physics_ticks_per_decision` ticks.
Edge-triggered fabrication is pulsed only on the first held tick.

## Lifecycle and persistence

Lifecycle frames report generic physical/session events: entity alive, death,
reproduction, explicit replacement, world termination, session attachment, and
detachment. They contain no reward, objective, hint, or evaluator output.

Metadata exposes the world ID, session ID, stable public agent ID, and opaque
resume token. On transport loss the lockstep world remains paused. A new socket
may send a resume frame containing the session ID and token, after which the
same world and physical entity continue where possible.

Canonical snapshots are valid at decision boundaries. They include the seed,
simulation tick, field clock, stable entity identities, physical state,
lineage/tuning state, active clamp relationships, RNG state, and an opaque
external-agent checkpoint reference. ERTW does not serialize an arbitrary
external agent's cognition. Snapshot schema v2 is currently saved and loaded by
the server process; frame kind 7 is reserved for a future negotiated byte
transfer.

## Reference server

Run the first-party single-agent server with:

```text
cargo run -p ertw_server --bin ertw-server -- ADDRESS STEPS SEED
```

For example, `127.0.0.1:9000 600 12648430` listens locally, advances 600
fixed ticks at 60 Hz after a client connects, and uses seed `12648430`.

Run the slow-agent lockstep server with:

```text
cargo run -p ertw_server --bin ertw-lockstep -- ADDRESS DECISIONS SEED PHYSICS_TICKS_PER_DECISION [SNAPSHOT_PATH]
```

Transport session IDs and resume tokens come from operating-system randomness
and do not affect deterministic world physics. TCP transport is not encrypted
or authenticated beyond the opaque resume token; bind to loopback or place ERTW
behind an authenticated secure tunnel on untrusted networks.

## Python reference client

The dependency-free client in `clients/python` validates protocol metadata and
frame dimensions before exposing observations. Its opt-in interoperability test
launches the Rust lockstep server, exchanges actions across three decision
boundaries, disconnects, and resumes the same session:

```text
PYTHONPATH=clients/python/src ERTW_RUN_RUST_INTEROP=1 \
  python3 -m unittest discover -s clients/python/tests -v
```
