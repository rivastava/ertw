# ERTW architecture

## Trust boundaries

ERTW has three explicit boundaries:

1. `ertw_core` owns physical truth and exposes no reward, score, objective, or
   evaluator output.
2. `ertw_interface` is the complete agent-visible contract. Policies receive an
   `ObservationTensor` and return an `ActionTensor`.
3. `ertw_evaluator` reads historical world outcomes after the fact. Its output
   never enters observations or simulation systems.

Rendering and the HUD are read-only observers except for gating the entire fixed
simulation tick during pause and single-step.

## Fixed tick

Every simulated tick is ordered as follows:

1. Advance seeded fields.
2. Rebuild the physical-entity spatial index and remove orphaned clamp joints.
3. Build egocentric observations using reusable scratch storage and collect
   bounded actions.
4. Apply locomotion, joints, fabrication, oscillators, and field forces.
5. Apply thermodynamic drain and energy-transfer channels.
6. Run Avian broad phase, contact generation, solver, and writeback.
7. Convert solved impulses into structural stress and damage.
8. Credit finishing-blow consumption, fragment failed nodes, and update active
   chunks after physics commands have settled. Depleted agents lose controller
   identity and become inert physical matter before later chunk reclamation.
9. Advance the authoritative simulation clock and record lineage history.

All steps, including Avian physics, run in `FixedUpdate`. A paused observer gates
both ERTW systems and Avian's physics sets.

## Determinism

The simulation uses a fixed 60 Hz timestep, a single `SimClock`, seeded Perlin
fields, per-chunk seeds, and per-node mutation streams. Ordered collections are
used where traversal affects spawning. Exact cross-platform floating-point
identity is not promised; replay equivalence is expected on the same target and
build configuration.

## Conservation

Fabrication and reproduction transfer mass and stored energy rather than
creating them. Fragmentation limits daughter count to the available viable mass
and structure, then divides remaining mass and energy exactly. Undersized failed
nodes decay instead of creating minimum-sized daughters. Only the contact
impulse that caused structural failure may receive consumption transfer;
ambient failure has no attacker. Thermal fields and vents are explicit
environmental sources/sinks. All production energy mutation passes through the
`EnergyLedger` transaction API.

## Controller lifecycle

Each live agent entity owns a controller ID in `WorldAgents`. Dead controller
objects are removed. Reproduction calls `Agent::spawn_child`; successful births
receive an independent controller and a deterministically mutated tuning. A
controller that cannot reproduce returns `None`.

## Durable sessions

Protocol v4 offers real-time and lockstep transports without changing the
observation/action schema. Lockstep holds continuous actions for a configured
number of physics ticks and pauses at decision boundaries while disconnected.
Stable public identities, lifecycle events, opaque resume tokens, and canonical
snapshot schema v2 support durable sessions. Restoring a snapshot requires the
caller to provide each external agent controller or checkpoint separately;
ERTW restores world physics, lineage, RNG state, and active clamp relationships
but cannot serialize arbitrary agent cognition.

## Spatial continuum

The chunk manager keeps a deterministic one-chunk halo around each live agent.
Inactive non-agent state is discarded and regenerated from its coordinate seed
when revisited. Streaming never injects agents; population changes only through
explicit initial spawning and reproduction.
