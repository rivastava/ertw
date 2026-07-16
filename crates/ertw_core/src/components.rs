//! Core ECS components describing every node in the world.
//!
//! Structure is *not* a single scalar that hits zero and disappears (spec item 6):
//! it depletes under cumulative force impulses exceeding a yield threshold.

use crate::tags::CustomTags;
use bevy::prelude::*;

/// Physical properties shared by every node.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct Physical {
    /// Inertial / gravitational mass. Also gates energy economy scaling.
    pub mass: f32,
    /// Current structural integrity. Fractures at 0; yields under impulse > yield.
    pub structure: f32,
    /// Reserve energy. Drains continuously; refilled by vents/harvest/consumption.
    /// Every action also costs energy. At 0 the node dies.
    pub energy: f32,
}

/// Tag-defined yield threshold: cumulative force impulse above this accumulates
/// structural damage (spec item 6).
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct Yield(pub f32);

/// Thermal conductivity. Low conductivity (shelter material) reduces Thermal
/// field drain (spec items 4, 5). 0 = inert, 1 = fully exposed.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct Conductivity(pub f32);

/// Relation mask.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct Tags(pub CustomTags);

/// Internal oscillator state broadcast to nearby nodes (spec items 9, 10).
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct Oscillator {
    /// Current frequency (rad/s).
    pub freq: f32,
    /// Current phase (rad).
    pub phase: f32,
    /// Baseline frequency inherited at spawn; mutation only on lineage.
    pub baseline_freq: f32,
}

/// Accumulated force impulse since last rest; compared against [`Yield`]. When it
/// exceeds yield, structure takes damage proportional to the overshoot.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct ImpulseAccum {
    pub value: f32,
    pub source: Option<Entity>,
}

/// Tracks the duration of a sustained reproductive surplus and the cooldown
/// after a successful birth.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct ReproductionState {
    pub surplus_seconds: f32,
    pub cooldown_seconds: f32,
}

/// Marks an entity as a controllable agent and carries its lineage id. The
/// world treats agents identically to any other node for physics/economy; this
/// only routes observations and actions.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct AgentMarker {
    /// Lineage depth: 0 for genesis agents, N for Nth-generation offspring.
    pub generation: u32,
    /// Lineage identifier shared by an ancestor chain.
    pub lineage: u64,
    /// ID of the controller stored in [`crate::agents::WorldAgents`].
    pub controller: u64,
}

/// Per-agent observation/action tuning inherited at spawn and fed to the
/// interface contract. Evolution mutates this (spec item 10).
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct AgentTuning {
    pub sensor_radius: f32,
    pub max_neighbors: usize,
    pub yield_threshold: f32,
    pub osc_baseline: f32,
}

impl Default for AgentTuning {
    fn default() -> Self {
        Self {
            sensor_radius: 12.0,
            max_neighbors: 16,
            yield_threshold: 8.0,
            osc_baseline: 1.0,
        }
    }
}

/// Bookkeeping for the energy economy: how much energy this node has harvested
/// or consumed, used by the external evaluator (spec item 5, philosophy).
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct EnergyLedger {
    pub harvested: f32,
    pub consumed_from_others: f32,
    pub vented: f32,
    pub dissipated: f32,
    pub spent_on_actuation: f32,
    pub invested_in_fabrication: f32,
    pub invested_in_offspring: f32,
    pub transferred_out: f32,
    pub born_step: u32,
}

/// Clamp actuator state (spec item 9). When engaged the agent applies a spring
/// force toward its current clamp target, fusing their kinetic motion. Costs
/// energy every step it is held.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct ClampState {
    /// Entity currently clamped to, or `None` when released.
    pub target: Option<Entity>,
    /// Constraint entity joining the two bodies.
    pub joint: Option<Entity>,
    /// Remaining cooldown (seconds) before the clamp can re-engage.
    pub cooldown: f32,
}

/// Fabrication actuator state (spec item 3, "Fabricate low-conductivity
/// structure"). A cooldown gates repeated fabrication so it cannot be spammed
/// for free.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct FabricateCooldown {
    pub remaining: f32,
}

/// Per-node RNG seed stream used for emergent, reproducible mutation on
/// fragmentation and lineage. Distinct from the global field seed.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct NodeRng(pub u64);

/// Bundle grouping every component a controllable agent entity carries. Defined
/// explicitly because the full set exceeds Bevy's 15-component tuple `Bundle`
/// limit, and it keeps agent spawning consistent across genesis, fragmentation,
/// and external injection.
#[derive(bevy::prelude::Bundle)]
pub struct AgentBundle {
    pub transform: bevy::prelude::Transform,
    pub rigid_body: avian2d::prelude::RigidBody,
    pub collider: avian2d::prelude::Collider,
    pub mass: avian2d::prelude::Mass,
    pub physical: Physical,
    pub yield_thresh: Yield,
    pub conductivity: Conductivity,
    pub tags: Tags,
    pub oscillator: Oscillator,
    pub impulse: ImpulseAccum,
    pub ledger: EnergyLedger,
    pub marker: AgentMarker,
    pub tuning: AgentTuning,
    pub clamp: ClampState,
    pub fabricate: FabricateCooldown,
    pub reproduction: ReproductionState,
    pub node_rng: NodeRng,
}
