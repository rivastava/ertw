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
2. Rebuild the spatial index.
3. Build egocentric observations and collect bounded actions.
4. Apply locomotion, joints, fabrication, oscillators, and field forces.
5. Apply thermodynamic drain and energy-transfer channels.
6. Run Avian broad phase, contact generation, solver, and writeback.
7. Convert solved impulses into structural stress and damage.
8. Credit finishing-blow consumption, fragment failed nodes, and update active
   chunks after physics commands have settled.
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
creating them. Fragmentation divides remaining mass and energy among rubble,
after any externally credited consumption share. Thermal fields and vents are
explicit environmental sources/sinks. `EnergyLedger` distinguishes dissipation
and actuation costs from energy transferred into fabrication, offspring, or
other nodes.

## Controller lifecycle

Each live agent entity owns a controller ID in `WorldAgents`. Dead controller
objects are removed. Reproduction calls `Agent::spawn_child`; successful births
receive an independent controller and a deterministically mutated tuning. A
controller that cannot reproduce returns `None`.

## Spatial continuum

The chunk manager keeps a deterministic one-chunk halo around each live agent.
Inactive non-agent state is discarded and regenerated from its coordinate seed
when revisited. Streaming never injects agents; population changes only through
explicit initial spawning and reproduction.
