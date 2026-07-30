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

/// Auditable source or sink used by [`EnergyLedger`] transactions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnergyFlow {
    Harvested,
    Consumed,
    Vented,
    Dissipated,
    Actuation,
    Fabrication,
    Offspring,
    Transferred,
}

impl EnergyLedger {
    /// Credit a finite non-negative amount and record its physical source.
    pub fn credit(&mut self, physical: &mut Physical, flow: EnergyFlow, amount: f32) -> f32 {
        if !matches!(
            flow,
            EnergyFlow::Harvested | EnergyFlow::Consumed | EnergyFlow::Vented
        ) {
            return 0.0;
        }
        let amount = valid_energy_amount(amount);
        physical.energy += amount;
        self.record(flow, amount);
        amount
    }

    /// Debit the full amount only when the reserve can fund it.
    pub fn debit_exact(&mut self, physical: &mut Physical, flow: EnergyFlow, amount: f32) -> bool {
        let amount = valid_energy_amount(amount);
        if !is_debit(flow) || physical.energy < amount {
            return false;
        }
        physical.energy -= amount;
        self.record(flow, amount);
        true
    }

    /// Debit as much as remains available, without making the reserve negative.
    pub fn debit_available(
        &mut self,
        physical: &mut Physical,
        flow: EnergyFlow,
        amount: f32,
    ) -> f32 {
        if !is_debit(flow) {
            return 0.0;
        }
        let amount = valid_energy_amount(amount).min(physical.energy.max(0.0));
        physical.energy -= amount;
        self.record(flow, amount);
        amount
    }

    /// Apply an environmental drain. Unlike a voluntary spend, this may cross
    /// zero so the death system observes depletion during the same tick.
    pub fn drain(&mut self, physical: &mut Physical, amount: f32) -> f32 {
        let amount = valid_energy_amount(amount);
        physical.energy -= amount;
        self.record(EnergyFlow::Dissipated, amount);
        amount
    }

    fn record(&mut self, flow: EnergyFlow, amount: f32) {
        match flow {
            EnergyFlow::Harvested => self.harvested += amount,
            EnergyFlow::Consumed => self.consumed_from_others += amount,
            EnergyFlow::Vented => self.vented += amount,
            EnergyFlow::Dissipated => self.dissipated += amount,
            EnergyFlow::Actuation => self.spent_on_actuation += amount,
            EnergyFlow::Fabrication => self.invested_in_fabrication += amount,
            EnergyFlow::Offspring => self.invested_in_offspring += amount,
            EnergyFlow::Transferred => self.transferred_out += amount,
        }
    }
}

fn valid_energy_amount(amount: f32) -> f32 {
    if amount.is_finite() {
        amount.max(0.0)
    } else {
        0.0
    }
}

fn is_debit(flow: EnergyFlow) -> bool {
    matches!(
        flow,
        EnergyFlow::Dissipated
            | EnergyFlow::Actuation
            | EnergyFlow::Fabrication
            | EnergyFlow::Offspring
            | EnergyFlow::Transferred
    )
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

/// Ownership record attached to every dynamically created clamp joint.
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct ClampJoint {
    pub owner: Entity,
    pub target: Entity,
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

/// Durable public identity used by lifecycle events and canonical snapshots.
/// It is intentionally independent of Bevy's allocator-specific [`Entity`].
#[derive(Component, Clone, Copy, Debug, Default, Reflect, PartialEq, Eq, PartialOrd, Ord)]
#[reflect(Component)]
pub struct StableId(pub u64);

/// Deterministic chunk that generated an inert terrain node. Ownership remains
/// stable even when physics moves the node across a coordinate boundary.
#[derive(Component, Clone, Copy, Debug, Reflect, PartialEq, Eq)]
#[reflect(Component)]
pub struct ChunkOrigin {
    pub x: i32,
    pub y: i32,
}

/// Marks a depleted node whose controller and actuator identity have ended.
/// The physical body remains as inert matter.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct DeadNode;

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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{prop_assert, prop_assert_eq, proptest};

    proptest! {
        #[test]
        fn energy_transactions_match_the_ledger(
            initial in 0.0_f32..1_000.0,
            incoming in 0.0_f32..1_000.0,
            requested in 0.0_f32..1_000.0,
        ) {
            let mut physical = Physical {
                energy: initial,
                ..Default::default()
            };
            let mut ledger = EnergyLedger::default();
            let credited = ledger.credit(&mut physical, EnergyFlow::Harvested, incoming);
            let debited =
                ledger.debit_available(&mut physical, EnergyFlow::Actuation, requested);

            prop_assert_eq!(ledger.harvested, credited);
            prop_assert_eq!(ledger.spent_on_actuation, debited);
            prop_assert!(physical.energy >= 0.0);
            let expected = initial + credited - debited;
            prop_assert!((physical.energy - expected).abs() <= expected.max(1.0) * f32::EPSILON);
        }
    }

    #[test]
    fn energy_transactions_reject_non_finite_amounts() {
        let mut physical = Physical {
            energy: 10.0,
            ..Default::default()
        };
        let mut ledger = EnergyLedger::default();
        assert_eq!(
            ledger.credit(&mut physical, EnergyFlow::Harvested, f32::NAN),
            0.0
        );
        assert_eq!(
            ledger.debit_available(&mut physical, EnergyFlow::Actuation, f32::INFINITY),
            0.0
        );
        assert_eq!(physical.energy, 10.0);
        assert_eq!(ledger.harvested, 0.0);
        assert_eq!(ledger.spent_on_actuation, 0.0);
    }
}
