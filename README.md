# ERTW — Extensible Relational Tensor World

A **zero-reward, agent-agnostic 2D physics simulation** in Rust/Bevy. Agents
perceive only an egocentric float tensor and act via continuous low-level
actuators. The world exposes **no reward, no score, no goals** — only
continuous physical consequence under three global fields (Kinetic / Thermal /
Electromagnetic). This is a substrate for emergent-behavior research, not a
game with win conditions.

The public system boundaries are documented in `docs/ARCHITECTURE.md`, and the
language-neutral agent wire contract is documented in `docs/PROTOCOL.md`.

## Quick start

The repository uses stable Rust with `rustfmt` and Clippy. Windows GNU builds
also require MinGW-w64 on `PATH`.

```powershell
# Headless: run the world with 8 random agents, no window
cargo run -p ertw_core --example headless_sim

# Native observer: Bevy window + gizmos + egui HUD
cargo run -p ertw_render --bin ertw --features render

# External agent server: address, simulated steps, seed
cargo run -p ertw_server --bin ertw-server -- 127.0.0.1:9000 600 12648430

# Slow-agent lockstep: address, decisions, seed, physics ticks per decision
cargo run -p ertw_server --bin ertw-lockstep -- 127.0.0.1:9000 600 12648430 4
```

The native executable is written to `target/debug/ertw` (or `ertw.exe` on
Windows). The older `rendered_sim` example remains as a compatibility wrapper.
The server advances at 60 Hz after one external agent connects; clients use the
versioned tensor protocol documented in `docs/PROTOCOL.md`.

## Implementation status

ERTW is pre-1.0 research software. The eleven design systems are connected
end-to-end and covered by unit, property, scenario, replay, benchmark, or soak
gates. The remaining pre-1.0 work requires public CI history, multi-platform
performance evidence, and signed packaging.

| Phase | System | Status |
| ----- | ------ | ------ |
| 1 | Entities & component model (property + relation-tag mixes) | ✅ |
| 2 | Global substrate & three fields (Kinetic/Thermal/EM) | ✅ |
| 3 | Thermodynamics & energy economy (drain, vents, death) | ✅ |
| 4 | Structural failure & fragmentation into daughter nodes | ✅ |
| 5 | Universal interface contract (observation/action tensors, spatial hash) | ✅ |
| 6 | Actuators: locomotion, clamp, oscillator, fabrication | ✅ |
| 7 | Procedural genesis & chunk streaming | ✅ |
| 8 | Lineage & population dynamics | ✅ |
| 9 | Visualization: gizmo overlay + **egui HUD** | ✅ |
| 10 | Network/IPC bridge (`ertw_server`, framed protocol v4) | ✅ |
| 11 | External evaluator (`ertw_evaluator`, historical competence ranking) | ✅ |

## Crates

- **`ertw_core`** — the world: components, fields, systems, physics glue,
  genesis, fragmentation, lineage, actuation. Exposes `ErtwWorld`,
  `configure_world`, and the `SimulationSet` (named `SystemSet` gating all
  fixed-step sim systems, used by the HUD to pause/step).
- **`ertw_interface`** — `ObservationTensor` / `ActionTensor` + the `Agent`
  trait + the wire header. It has no Bevy dependency, keeping external adapters
  lightweight.
- **`ertw_server`** — optional TCP bridge. It supports non-blocking real-time
  exchange and slow-agent lockstep with action hold, lifecycle events, resume,
  physical deltas, and canonical snapshots.
- **`ertw_render`** — native `ertw` observer binary with Bevy gizmos and an
  `bevy_egui` HUD. Feature-gated (`render`).
- **`ertw_evaluator`** — `CompetenceRecord` + `rank`, plus an `evaluate`
  example that ranks live and historical agents by survival, generation, and
  accumulated incoming energy.
- **`agents/random_policy`** — reference in-process agent.

## The interface contract

Every agent, regardless of architecture (random policy, PPO/DQN, wrapped
LLM/VLM), receives the same `ObservationTensor` and returns the same
continuous `ActionTensor`:

- **Observation**: egocentric self-state + radial field samples + up to
  `max_neighbors` distance-sorted neighbors (padded with zero-state ghost nodes
  and a `valid` flag).
- **Action**: continuous `force` (Vec2), `torque`, `clamp`, `fabricate`,
  `osc_freq`, `osc_phase`. Every actuator call costs stored energy — there is no
  free action.

Protocol v4 uses self-describing length-prefixed little-endian frames with full
64-bit step/entity IDs and contiguous `f32` payloads. See
`docs/PROTOCOL.md` for the exact layout.

## Native observer

The egui HUD is a **non-participant observer**: it shows a pause checkbox, a
single-step button, the seed, live agent/node counts, and the Kinetic/Thermal/
EM field sample at the origin plus the field clock. It never influences the
simulation beyond pausing/stepping it. WASD/arrow keys pan, Q/E zoom, and the
entity inspector exposes physical state, tags, generation, and lineage.

## Determinism

Fixed timestep (60 Hz) via Bevy `FixedUpdate` plus strictly seeded noise/RNG.
Absolute cross-platform FP determinism is not guaranteed with avian2d, but
procedural genesis and field drift are reproducible per-machine — which is what
the external evaluator needs. The seed drives both the field sampler and the
genesis chunk distribution.

## Build / verify

```powershell
cargo build --workspace --all-targets --all-features
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check

# Explicit performance and long-run gates
cargo bench -p ertw_core --bench simulation
cargo test -p ertw_core bounded_state_soak -- --ignored
```

The benchmark defaults to 32 agents and 300 ticks. Override it with
`ERTW_BENCH_AGENTS` and `ERTW_BENCH_STEPS`. A July 2026 audit run on Apple
Silicon measured approximately 38k, 23k, and 34k agent-steps/s at 32, 64, and
128 agents respectively. These are local smoke measurements, not portable
performance guarantees; geometry and physics contacts materially affect them.

## Releases

Tagged releases publish the native observer and lockstep server for Linux
x86-64, macOS Apple Silicon, and Windows x86-64, together with SHA-256
checksums. Download them from the
[GitHub releases page](https://github.com/rivastava/ertw/releases).

## Research-preview limitations

- The protected CI gate is validated on Linux, macOS, and Windows.
- Large-population throughput still needs profiler traces and repeatable
  multi-machine baselines; the included harness is a scale smoke test.
- The evaluator is an initial lexicographic comparison, not a validated general
  intelligence score.
- The TCP bridge has been tested end-to-end with an external Python NEAT
  controller, but broader architecture-diverse benchmark suites remain future
  work.
- Release binaries are not yet signed or notarized.

## License

MIT OR Apache-2.0.
