//! Lineage & evaluator bridge (spec items 8, 10, 11).
//!
//! Lineage depth/generation is carried on [`crate::components::AgentMarker`] and
//! incremented when a node fragments (see [`crate::fragmentation`]). This module
//! provides the read-side helpers that turn live world state into plain snapshot
//! records the offline evaluator ranks by. No reward is computed here.

use crate::components::{
    AgentBundle, AgentMarker, AgentTuning, ClampState, EnergyFlow, EnergyLedger, FabricateCooldown,
    ImpulseAccum, NodeRng, Oscillator, Physical, ReproductionState, Tags, Yield,
};
use bevy::prelude::*;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};

pub const REPRODUCTION_ENERGY_THRESHOLD: f32 = 32.0;
pub const REPRODUCTION_SURPLUS_SECONDS: f32 = 5.0;
pub const REPRODUCTION_ENERGY_COST: f32 = 12.0;
pub const REPRODUCTION_MASS_COST: f32 = 1.5;
pub const REPRODUCTION_COOLDOWN_SECONDS: f32 = 20.0;

/// Spawns independently controlled offspring after a sustained energy surplus.
/// Mass and energy are transferred from parent to child, and physical tuning is
/// deterministically mutated from the parent's per-node seed stream.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn reproduce_agents(
    mut commands: Commands,
    mut controllers: ResMut<crate::agents::WorldAgents>,
    clock: Res<crate::SimClock>,
    mut agents: Query<(
        Entity,
        &Transform,
        &mut Physical,
        &mut avian2d::prelude::Mass,
        &AgentMarker,
        &AgentTuning,
        &mut ReproductionState,
        &mut NodeRng,
        &mut EnergyLedger,
    )>,
) {
    for (
        _entity,
        transform,
        mut physical,
        mut mass,
        marker,
        tuning,
        mut reproduction,
        mut node_rng,
        mut ledger,
    ) in agents.iter_mut()
    {
        reproduction.cooldown_seconds =
            (reproduction.cooldown_seconds - crate::economy::FIXED_DT).max(0.0);
        let eligible = physical.energy >= REPRODUCTION_ENERGY_THRESHOLD
            && physical.mass >= REPRODUCTION_MASS_COST + 0.5
            && reproduction.cooldown_seconds <= 0.0;
        if !eligible {
            reproduction.surplus_seconds = 0.0;
            continue;
        }
        reproduction.surplus_seconds += crate::economy::FIXED_DT;
        if reproduction.surplus_seconds < REPRODUCTION_SURPLUS_SECONDS {
            continue;
        }

        let child_seed = node_rng.0 ^ clock.step.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let Some(controller) = controllers.spawn_child(marker.controller, child_seed) else {
            reproduction.surplus_seconds = 0.0;
            reproduction.cooldown_seconds = REPRODUCTION_COOLDOWN_SECONDS;
            continue;
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(child_seed);
        let child_tuning = AgentTuning {
            sensor_radius: (tuning.sensor_radius * rng.gen_range(0.95..1.05)).clamp(4.0, 32.0),
            max_neighbors: tuning.max_neighbors.clamp(1, 64),
            yield_threshold: (tuning.yield_threshold * rng.gen_range(0.95..1.05)).max(1.0),
            osc_baseline: (tuning.osc_baseline + rng.gen_range(-0.15..0.15)).clamp(-16.0, 16.0),
        };

        if !ledger.debit_exact(
            &mut physical,
            EnergyFlow::Offspring,
            REPRODUCTION_ENERGY_COST,
        ) {
            reproduction.surplus_seconds = 0.0;
            continue;
        }
        physical.mass -= REPRODUCTION_MASS_COST;
        mass.0 = physical.mass.max(0.05);
        reproduction.surplus_seconds = 0.0;
        reproduction.cooldown_seconds = REPRODUCTION_COOLDOWN_SECONDS;
        node_rng.0 = child_seed.rotate_left(17);

        let birth_angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let position = transform.translation.truncate() + Vec2::from_angle(birth_angle) * 2.0;
        commands.spawn(AgentBundle {
            transform: Transform::from_translation(position.extend(0.0)),
            rigid_body: avian2d::prelude::RigidBody::Dynamic,
            collider: avian2d::prelude::Collider::circle(0.5),
            mass: avian2d::prelude::Mass(REPRODUCTION_MASS_COST),
            physical: Physical {
                mass: REPRODUCTION_MASS_COST,
                structure: child_tuning.yield_threshold,
                energy: REPRODUCTION_ENERGY_COST,
            },
            yield_thresh: Yield(child_tuning.yield_threshold),
            conductivity: crate::components::Conductivity(0.6),
            tags: Tags(crate::tags::CustomTags::from_bits(
                crate::tags::CustomTags::AGENT
                    | crate::tags::CustomTags::CLAMP_CAPABLE
                    | crate::tags::CustomTags::OSCILLATOR,
            )),
            oscillator: Oscillator {
                freq: child_tuning.osc_baseline,
                phase: 0.0,
                baseline_freq: child_tuning.osc_baseline,
            },
            impulse: ImpulseAccum::default(),
            ledger: EnergyLedger {
                born_step: clock.step.min(u32::MAX as u64) as u32,
                ..Default::default()
            },
            marker: AgentMarker {
                generation: marker.generation.saturating_add(1),
                lineage: marker.lineage,
                controller,
            },
            tuning: child_tuning,
            clamp: ClampState::default(),
            fabricate: FabricateCooldown::default(),
            reproduction: ReproductionState::default(),
            node_rng: NodeRng(child_seed),
        });
    }
}

/// Plain snapshot of one live agent's competence-relevant state at a step. The
/// external evaluator (`ertw_evaluator`) consumes these; this crate stays free
/// of an evaluator dependency to avoid a circular crate graph.
#[derive(Clone, Copy, Debug, Default)]
pub struct AgentSnapshot {
    pub alive: bool,
    pub entity: u64,
    pub generation: u32,
    pub lineage: u64,
    pub step: u32,
    pub harvested: f32,
    pub consumed_from_others: f32,
    pub vented: f32,
    pub born_step: u32,
}

#[derive(Clone, Copy)]
struct ActiveAgentRecord {
    marker: AgentMarker,
    ledger: EnergyLedger,
}

#[derive(Resource, Default)]
pub struct AgentHistory {
    active: HashMap<Entity, ActiveAgentRecord>,
    pub completed: Vec<AgentSnapshot>,
}

/// Maintains an append-only outcome record so dead agents remain visible to the
/// external evaluator.
pub fn record_agent_history(
    clock: Res<crate::SimClock>,
    agents: Query<(Entity, &AgentMarker, &EnergyLedger)>,
    mut history: ResMut<AgentHistory>,
) {
    let mut current = HashSet::new();
    for (entity, marker, ledger) in agents.iter() {
        current.insert(entity);
        history.active.insert(
            entity,
            ActiveAgentRecord {
                marker: *marker,
                ledger: *ledger,
            },
        );
    }
    let removed = history
        .active
        .keys()
        .filter(|entity| !current.contains(entity))
        .copied()
        .collect::<Vec<_>>();
    for entity in removed {
        if let Some(record) = history.active.remove(&entity) {
            history.completed.push(snapshot_from_record(
                entity,
                record,
                clock.step.min(u32::MAX as u64) as u32,
            ));
        }
    }
}

fn snapshot_from_record(entity: Entity, record: ActiveAgentRecord, step: u32) -> AgentSnapshot {
    AgentSnapshot {
        alive: false,
        entity: entity.to_bits(),
        generation: record.marker.generation,
        lineage: record.marker.lineage,
        step,
        harvested: record.ledger.harvested,
        consumed_from_others: record.ledger.consumed_from_others,
        vented: record.ledger.vented,
        born_step: record.ledger.born_step,
    }
}

/// Snapshot every live agent's competence-relevant state at `step`. Pure read of
/// the world; the returned records carry no reward, only comparable metrics.
pub fn collect_competence(world: &mut World, step: u32) -> Vec<AgentSnapshot> {
    let mut out = world
        .get_resource::<AgentHistory>()
        .map(|history| history.completed.clone())
        .unwrap_or_default();
    let mut query = world.query::<(Entity, &AgentMarker, &EnergyLedger)>();
    for (e, marker, ledger) in query.iter(world) {
        out.push(AgentSnapshot {
            alive: true,
            entity: e.to_bits(),
            generation: marker.generation,
            lineage: marker.lineage,
            step,
            harvested: ledger.harvested,
            consumed_from_others: ledger.consumed_from_others,
            vented: ledger.vented,
            born_step: ledger.born_step,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ertw_interface::{ActionTensor, Agent, ObservationTensor};

    struct ReproducingAgent;

    impl Agent for ReproducingAgent {
        fn act(&mut self, _observation: &ObservationTensor) -> ActionTensor {
            ActionTensor::default()
        }

        fn spawn_child(&mut self, _seed: u64) -> Option<Box<dyn Agent>> {
            Some(Box::new(Self))
        }
    }

    #[test]
    fn sustained_surplus_creates_independent_conserving_offspring() {
        let mut simulation = crate::ErtwWorld::new(91);
        let parent = simulation.spawn_agent(Box::new(ReproducingAgent), Vec2::ZERO);
        simulation.step(1);
        {
            let world = simulation.app().world_mut();
            world.get_mut::<Physical>(parent).expect("parent").energy = 60.0;
        }
        simulation.step((REPRODUCTION_SURPLUS_SECONDS * 60.0) as u32 + 2);

        let world = simulation.app().world_mut();
        let mut query = world.query::<(&Physical, &AgentMarker)>();
        let agents = query
            .iter(world)
            .map(|(physical, marker)| (*physical, *marker))
            .collect::<Vec<_>>();
        assert_eq!(agents.len(), 2);
        assert!(agents.iter().any(|(_, marker)| marker.generation == 1));
        assert_ne!(agents[0].1.controller, agents[1].1.controller);
        let total_mass = agents
            .iter()
            .map(|(physical, _)| physical.mass)
            .sum::<f32>();
        assert!((total_mass - AgentTuning::default().yield_threshold).abs() < 1.0e-4);
    }
}
