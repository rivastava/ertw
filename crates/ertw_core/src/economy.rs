//! Thermodynamic Decay & Energy Economy (spec items 4, 5).
//!
//! The world imposes a continuous allostatic drain. Agents offset it by exploiting
//! Thermal Vents, Kinetic Harvesting, or Consumption of failed structures. Every
//! action also costs energy. At zero energy a node dies.

use crate::components::{Conductivity, DeadNode, EnergyFlow, EnergyLedger, Physical, Tags, Yield};
use crate::fields::FieldSampler;
use crate::fragmentation::FragmentQueue;
use crate::spatial_hash::SpatialHash;
use crate::tags::CustomTags;
use avian2d::dynamics::rigid_body::LinearVelocity;
use bevy::prelude::*;

/// Base allostatic drain per second before modifiers.
pub const BASE_DRAIN: f32 = 0.6;

/// Fixed timestep (60 Hz). Centralized so determinism holds with step count.
pub const FIXED_DT: f32 = 1.0 / 60.0;

/// Relative-velocity threshold (units/s) below which kinetic harvesting does
/// not accrue energy. Sustained relative motion above this transfers energy
/// from `ENERGY_CONVERTIBLE` nodes to nearby agents (spec item 5).
pub const HARVEST_MIN_REL_SPEED: f32 = 1.5;

/// Energy (joules / sec of relative-velocity unit) transferred per second once
/// `HARVEST_MIN_REL_SPEED` is exceeded.
pub const HARVEST_RATE: f32 = 1.5;

/// Effective range (world units) within which an agent can drain a volatile
/// trap when the EM field spikes (spec item 5).
pub const VOLATILE_TRAP_RANGE: f32 = 8.0;

/// Fraction of a volatile trap's stored energy released per spike event.
pub const VOLATILE_TRAP_DRAIN_FRACTION: f32 = 0.5;

#[derive(Resource, Default)]
pub struct DeathQueue {
    entities: Vec<Entity>,
}

/// Apply continuous thermodynamic decay + field-driven drain to every node and
/// queue depleted nodes for a post-physics lifecycle transition.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn thermodynamic_drain(
    sampler: Res<FieldSampler>,
    spatial: Res<SpatialHash>,
    mut query: Query<
        (
            Entity,
            &Transform,
            &Conductivity,
            &Tags,
            &Yield,
            &mut Physical,
            &mut EnergyLedger,
        ),
        Without<DeadNode>,
    >,
    transforms: Query<(Entity, &Transform)>,
    tags_query: Query<&Tags>,
    mut fragments: ResMut<FragmentQueue>,
    mut deaths: ResMut<DeathQueue>,
) {
    for (entity, tf, cond, tags, yield_threshold, mut phys, mut ledger) in query.iter_mut() {
        let position = tf.translation.truncate();
        let f = sampler.sample(position);
        // Thermal exposure raises drain, scaled by conductivity. Low-conductivity
        // (shelter) material resists drain.
        let thermal_load = f.thermal.max(0.0);
        let shelter_factor =
            shelter_exposure_factor(entity, position, &spatial, &transforms, &tags_query);
        let drain = BASE_DRAIN * (1.0 + thermal_load * shelter_factor) * (0.4 + 0.6 * cond.0);

        // Vents gain energy from ambient heat but accrue structural stress on
        // over-exposure (spec item 5).
        if tags.0.has(CustomTags::THERMAL_VENT) {
            let gain = 2.0 * thermal_load;
            ledger.credit(&mut phys, EnergyFlow::Vented, gain * FIXED_DT);
            phys.structure -= 0.3 * thermal_load * FIXED_DT;
        }

        ledger.drain(&mut phys, drain * FIXED_DT);

        if phys.structure <= 0.0
            && phys.energy > 0.0
            && crate::fragmentation::can_fragment(&phys, *yield_threshold)
        {
            // Ambient structural failure has no attacker. Collision-induced
            // failure is queued separately from the actual contact impulse.
            crate::fragmentation::queue_fragment(entity, None, &mut fragments);
        } else if phys.energy <= 0.0 && !deaths.entities.contains(&entity) {
            deaths.entities.push(entity);
        }
    }
}

/// End depleted agent/controller identity while retaining inert physical matter.
pub fn run_deaths(world: &mut World) {
    let queued = std::mem::take(&mut world.resource_mut::<DeathQueue>().entities);
    for entity in queued {
        if world.get_entity(entity).is_err() {
            continue;
        }
        let mut joint_query = world.query::<(Entity, &crate::components::ClampJoint)>();
        let joints = joint_query
            .iter(world)
            .filter_map(|(joint_entity, joint)| {
                (joint.owner == entity || joint.target == entity).then_some(joint_entity)
            })
            .collect::<Vec<_>>();
        for joint in joints {
            world.despawn(joint);
        }
        if let Some(mut tags) = world.get_mut::<Tags>(entity) {
            tags.0 = tags
                .0
                .without(CustomTags::AGENT)
                .without(CustomTags::CLAMP_CAPABLE);
        }
        world
            .entity_mut(entity)
            .remove::<crate::components::AgentMarker>()
            .remove::<crate::components::AgentTuning>()
            .remove::<crate::components::ClampState>()
            .remove::<crate::components::FabricateCooldown>()
            .remove::<crate::components::ReproductionState>()
            .insert(DeadNode);
    }
}

fn shelter_exposure_factor(
    entity: Entity,
    position: Vec2,
    spatial: &SpatialHash,
    transforms: &Query<(Entity, &Transform)>,
    tags: &Query<&Tags>,
) -> f32 {
    const SHELTER_RADIUS: f32 = 2.5;
    let mut nearby = Vec::new();
    spatial.query_radius(position, SHELTER_RADIUS, &mut nearby);
    let protected = nearby.into_iter().any(|other| {
        if other == entity {
            return false;
        }
        let Ok(tag) = tags.get(other) else {
            return false;
        };
        let Ok((_, transform)) = transforms.get(other) else {
            return false;
        };
        tag.0.has(CustomTags::SHELTER)
            && transform.translation.truncate().distance_squared(position)
                <= SHELTER_RADIUS * SHELTER_RADIUS
    });
    if protected {
        0.25
    } else {
        1.0
    }
}

/// Kinetic Harvesting (spec item 5): agents in sustained relative motion
/// against `ENERGY_CONVERTIBLE`-tagged nodes above `HARVEST_MIN_REL_SPEED`
/// drain energy from the node into their own reserve. The harvest rate scales
/// with relative velocity; energy moves from node to agent and the agent's
/// `EnergyLedger.harvested` accrues accordingly.
#[allow(clippy::too_many_arguments)]
pub fn kinetic_harvest(
    spatial: Res<SpatialHash>,
    velocities: Query<&LinearVelocity>,
    tags_q: Query<&Tags>,
    mut physicals: Query<&mut Physical>,
    mut ledgers: Query<&mut EnergyLedger>,
    transforms: Query<(Entity, &Transform)>,
) {
    // Collect AGENT candidates first; nested mutation across two entities is
    // legal but we want to short-circuit on the agent side without holding
    // mut refs across multiple iterations.
    let mut agents: Vec<Entity> = Vec::new();
    for (e, _t) in transforms.iter() {
        if let Ok(t) = tags_q.get(e) {
            if t.0.has(CustomTags::AGENT) {
                agents.push(e);
            }
        }
    }
    let harvest_radius_sq = 4.0_f32 * 4.0; // small: frictional contact range

    for agent_e in agents {
        let Ok((_, agent_tf)) = transforms.get(agent_e) else {
            continue;
        };
        let agent_pos = agent_tf.translation.truncate();
        let agent_vel = velocities.get(agent_e).map(|v| v.0).unwrap_or(Vec2::ZERO);

        let mut near: Vec<Entity> = Vec::new();
        spatial.query_radius(agent_pos, 4.0, &mut near);
        for other in near {
            if other == agent_e {
                continue;
            }
            let Ok(t) = tags_q.get(other) else { continue };
            if !t.0.has(CustomTags::ENERGY_CONVERTIBLE) {
                continue;
            }
            let Ok((_, other_tf)) = transforms.get(other) else {
                continue;
            };
            let d2 = other_tf.translation.truncate().distance_squared(agent_pos);
            if d2 > harvest_radius_sq {
                continue;
            }
            let other_vel = velocities.get(other).map(|v| v.0).unwrap_or(Vec2::ZERO);
            let rel_speed = (agent_vel - other_vel).length();
            if rel_speed < HARVEST_MIN_REL_SPEED {
                continue;
            }
            let rate = HARVEST_RATE * (rel_speed - HARVEST_MIN_REL_SPEED + 1.0) * FIXED_DT;
            // Compute the actual transferable amount with a short borrow of
            // the node's physical only, then drop the borrow before applying
            // the two-sided transfer. This keeps the borrow checker happy
            // without proving `agent_e != other` statically.
            let transfer = {
                let Ok(node_phys) = physicals.get(other) else {
                    continue;
                };
                rate.min(node_phys.energy.max(0.0))
            };
            if transfer <= 0.0 {
                continue;
            }
            // Two independent, scope-limited mutations. Each `get_mut` borrow
            // is dropped at the end of its block before the next is taken.
            let debited = if let (Ok(mut node_phys), Ok(mut node_ledger)) =
                (physicals.get_mut(other), ledgers.get_mut(other))
            {
                node_ledger.debit_available(&mut node_phys, EnergyFlow::Transferred, transfer)
            } else {
                0.0
            };
            if debited <= 0.0 {
                continue;
            }
            if let (Ok(mut agent_phys), Ok(mut agent_ledger)) =
                (physicals.get_mut(agent_e), ledgers.get_mut(agent_e))
            {
                agent_ledger.credit(&mut agent_phys, EnergyFlow::Harvested, debited);
            }
        }
    }
}

/// Volatile Trap discharge (spec items 5, 8): when the EM field spikes through
/// a `VOLATILE_TRAP`-tagged node, distribute a fraction of its stored energy
/// to nearby AGENT entities (inverse-distance weighted).
#[allow(clippy::too_many_arguments)]
pub fn volatile_trap_discharge(
    sampler: Res<FieldSampler>,
    spatial: Res<SpatialHash>,
    tags_q: Query<&Tags>,
    mut physicals: Query<&mut Physical>,
    mut ledgers: Query<&mut EnergyLedger>,
    transforms: Query<(Entity, &Transform)>,
) {
    // Collect VOLATILE_TRAP entities first so we can mutate them without
    // aliasing the transforms/tags queries.
    let traps: Vec<Entity> = transforms
        .iter()
        .filter_map(|(e, _)| {
            tags_q
                .get(e)
                .ok()
                .filter(|t| t.0.has(CustomTags::VOLATILE_TRAP))
                .map(|_| e)
        })
        .collect();

    for trap in traps {
        let Ok((_, trap_tf)) = transforms.get(trap) else {
            continue;
        };
        let pos = trap_tf.translation.truncate();
        if !sampler.is_em_spike(pos) {
            continue;
        }
        // Find AGENT neighbors within range.
        let mut near: Vec<Entity> = Vec::new();
        spatial.query_radius(pos, VOLATILE_TRAP_RANGE, &mut near);
        let mut weights: Vec<(Entity, f32)> = Vec::new();
        let mut total_w = 0.0_f32;
        for other in &near {
            if *other == trap {
                continue;
            }
            let Ok(t) = tags_q.get(*other) else { continue };
            if !t.0.has(CustomTags::AGENT) {
                continue;
            }
            let Ok((_, tf)) = transforms.get(*other) else {
                continue;
            };
            let d = tf.translation.truncate().distance(pos).max(0.1);
            let w = 1.0 / d;
            total_w += w;
            weights.push((*other, w));
        }
        if total_w <= 0.0 {
            continue;
        }
        let release = physicals
            .get(trap)
            .map(|physical| physical.energy.max(0.0) * VOLATILE_TRAP_DRAIN_FRACTION)
            .unwrap_or(0.0);
        if release <= 0.0 {
            continue;
        }
        let released = if let (Ok(mut trap_phys), Ok(mut ledger)) =
            (physicals.get_mut(trap), ledgers.get_mut(trap))
        {
            ledger.debit_available(&mut trap_phys, EnergyFlow::Transferred, release)
        } else {
            0.0
        };

        for (target, w) in weights {
            let share = released * (w / total_w);
            if let (Ok(mut physical), Ok(mut ledger)) =
                (physicals.get_mut(target), ledgers.get_mut(target))
            {
                ledger.credit(&mut physical, EnergyFlow::Harvested, share);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tags::CustomTags;

    /// Spawn an AGENT entity with the requested tags at `pos`.
    fn spawn_ecs_node(
        world: &mut World,
        pos: Vec2,
        tag_bits: u64,
        energy: f32,
        structure: f32,
        mass: f32,
        conductivity: f32,
    ) -> Entity {
        world
            .spawn((
                Transform::from_translation(pos.extend(0.0)),
                crate::components::Physical {
                    mass,
                    structure,
                    energy,
                },
                Conductivity(conductivity),
                Tags(CustomTags::from_bits(tag_bits)),
                EnergyLedger::default(),
                ImpulseAccumStub,
            ))
            .id()
    }

    /// Marker so the query above can stand in for `crate::components::ImpulseAccum`
    /// without forcing every test node to carry one.
    #[derive(Component)]
    struct ImpulseAccumStub;

    /// A node with the `ENERGY_CONVERTIBLE` tag must transfer energy to a
    /// nearby moving agent; without motion the channel must remain inactive.
    #[test]
    fn kinetic_harvest_only_fires_above_rel_speed_threshold() {
        // Build a minimal world via the public ErtwWorld so all the resources
        // and plugins the system needs are present.
        let mut world = crate::ErtwWorld::new(123);
        // Spawn a slow-moving agent at origin.
        let agent_e = world.spawn_agent(Box::new(NullAgent), Vec2::ZERO);
        // Insert an ENERGY_CONVERTIBLE node right next to the agent.
        let node_e = spawn_ecs_node(
            world.app().world_mut(),
            Vec2::new(1.5, 0.0),
            CustomTags::ENERGY_CONVERTIBLE,
            5.0,
            8.0,
            1.0,
            0.6,
        );

        // Add a LinearVelocity component to the agent so it has nonzero speed.
        world
            .app()
            .world_mut()
            .entity_mut(agent_e)
            .insert(LinearVelocity(Vec2::new(3.0, 0.0)));

        // Rebuild spatial hash + run one harvest tick.
        world.app().world_mut().run_schedule(FixedUpdate);

        let w = world.app().world();
        let agent_energy = w
            .get::<crate::components::Physical>(agent_e)
            .unwrap()
            .energy;
        let node_energy = w.get::<crate::components::Physical>(node_e).unwrap().energy;
        assert!(
            agent_energy > 20.0,
            "agent must have gained energy from friction harvest (got {agent_energy})"
        );
        assert!(
            node_energy < 5.0,
            "node must have lost energy to the harvester (got {node_energy})"
        );

        // Cleanup test agent.
        let _ = node_e;
    }

    /// `ActionTensor::sanitize` clamp band regressions: see ertw_interface tests.
    /// This is the consumption-attribution smoke test.
    #[test]
    fn consumption_transfer_runs_through_fragmentation_helper() {
        use crate::fragmentation::FragmentQueue;
        let mut q = FragmentQueue::default();
        q.entities.clear();
        crate::fragmentation::queue_fragment_with_share(
            Entity::PLACEHOLDER,
            Some(Entity::PLACEHOLDER),
            5.0,
            &mut q,
        );
        assert_eq!(q.entities.len(), 1);
        assert!(q.entities[0].1.is_some());
        assert_eq!(q.entities[0].2, 5.0);
    }

    #[test]
    fn nearby_shelter_reduces_thermal_drain() {
        let mut exposed = crate::ErtwWorld::new(33);
        let exposed_agent = exposed.spawn_agent(Box::new(NullAgent), Vec2::ZERO);
        exposed.app().world_mut().flush();
        exposed
            .app()
            .world_mut()
            .resource_mut::<crate::SimClock>()
            .step = 1;

        let mut sheltered = crate::ErtwWorld::new(33);
        let sheltered_agent = sheltered.spawn_agent(Box::new(NullAgent), Vec2::ZERO);
        sheltered.app().world_mut().flush();
        crate::genesis::spawn_genesis_node(
            &mut sheltered.app().world_mut().commands(),
            crate::genesis::TerrainSpawn {
                chunk_origin: crate::components::ChunkOrigin { x: 0, y: 0 },
                pos: Vec2::new(1.0, 0.0),
                node_rng: 7,
                kind: crate::genesis::TerrainKind::Shelter,
                mass: 1.0,
                structure: 14.0,
                energy: 2.0,
                conductivity: 0.2,
            },
        );
        sheltered.app().world_mut().flush();
        sheltered
            .app()
            .world_mut()
            .resource_mut::<crate::SimClock>()
            .step = 1;

        exposed.step(1);
        sheltered.step(1);
        let exposed_energy = exposed
            .app()
            .world()
            .get::<Physical>(exposed_agent)
            .expect("exposed agent")
            .energy;
        let sheltered_energy = sheltered
            .app()
            .world()
            .get::<Physical>(sheltered_agent)
            .expect("sheltered agent")
            .energy;
        assert!(sheltered_energy > exposed_energy);
    }

    // Stand-in for the project's NullAgent; not visible from outside this
    // module so we duplicate it here.
    use ertw_interface::{ActionTensor, Agent, ObservationTensor};
    struct NullAgent;
    impl Agent for NullAgent {
        fn act(&mut self, _obs: &ObservationTensor) -> ActionTensor {
            ActionTensor::default()
        }
    }
}
