# ERTW protocol v3

All integers and floats are little-endian. Every TCP message starts with fourteen
`u32` words followed by `payload_floats` IEEE-754 `f32` values.

| Word | Meaning |
| ---: | --- |
| 0 | Magic: little-endian `ERTW` |
| 1 | Protocol version (`3`) |
| 2 | Frame kind: hello `1`, observation `2`, action `3` |
| 3 | Total frame length in bytes |
| 4–5 | Simulation step, low then high 32 bits |
| 6–7 | Entity ID, low then high 32 bits |
| 8 | Configured maximum neighbors |
| 9 | Valid neighbors in this observation |
| 10 | Field sample count |
| 11 | Channels per field sample |
| 12 | Payload float count |
| 13 | Reserved; must be zero |

The server sends a hello frame after connection. Observation and action frames
are self-delimiting and validate their magic, version, kind, length, and payload
size. Implementations must reject frames larger than 16 MiB.

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

## Reference server

Run the first-party single-agent server with:

```text
cargo run -p ertw_server --bin ertw-server -- ADDRESS STEPS SEED
```

For example, `127.0.0.1:9000 600 12648430` listens locally, advances 600
fixed ticks at 60 Hz after a client connects, and uses seed `12648430`.
