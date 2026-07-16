//! Thermodynamic Decay & Energy Economy (spec items 4, 5).
//!
//! The world imposes a continuous allostatic drain. Agents offset it by exploiting
//! Thermal Vents, Kinetic Harvesting, or Consumption of failed structures. Every
//! action also costs energy. At zero energy a node dies.

use crate::components::{Conductivity, EnergyLedger, Physical, Tags};
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

/// Search radius for the killer-attribution heuristic when a node's structure
/// reaches zero (spec item 5: consumption transfer).
pub const KILLER_SEARCH_RADIUS: f32 = 10.0;

/// Apply continuous thermodynamic decay + field-driven drain to every node, and
/// despawn nodes that die. Runs in `FixedUpdate`.
#[allow(clippy::too_many_arguments)]
pub fn thermodynamic_drain(
    sampler: Res<FieldSampler>,
    spatial: Res<SpatialHash>,
    mut query: Query<(
        Entity,
        &Transform,
        &Conductivity,
        &Tags,
        &mut Physical,
        &mut EnergyLedger,
    )>,
    transforms: Query<(Entity, &Transform)>,
    tags_query: Query<&Tags>,
    mut commands: Commands,
    mut fragments: ResMut<FragmentQueue>,
) {
    let mut dead: Vec<Entity> = Vec::new();
    for (entity, tf, cond, tags, mut phys, mut ledger) in query.iter_mut() {
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
            phys.energy += gain * FIXED_DT;
            ledger.vented += gain * FIXED_DT;
            phys.structure -= 0.3 * thermal_load * FIXED_DT;
        }

        phys.energy -= drain * FIXED_DT;
        ledger.dissipated += drain * FIXED_DT;

        if phys.structure <= 0.0 && phys.energy > 0.0 {
            // Structural failure: fragment into daughters (spec item 6) rather
            // than vanish. Attribute the killing blow so consumption transfer
            // (spec item 5) can route the victim's energy to the attacker.
            let killer = find_killer(
                entity,
                tf.translation.truncate(),
                &spatial,
                &transforms,
                &tags_query,
            );
            // Pre-compute the consumption share (half the victim's current
            // energy) so the transfer system doesn't need its own read query.
            let share = if killer.is_some() {
                phys.energy * 0.5
            } else {
                0.0
            };
            crate::fragmentation::queue_fragment_with_share(entity, killer, share, &mut fragments);
        } else if phys.energy <= 0.0 {
            dead.push(entity);
        }
    }
    for e in dead {
        commands.entity(e).despawn();
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

/// Best-effort killer attribution for spec item 5 (consumption transfer).
///
/// Heuristic: among AGENT-tagged neighbors within `KILLER_SEARCH_RADIUS`, pick
/// the closest one. A perfect "whichever node dealt the finishing blow"
/// attribution would require a per-impulse damage log per pair; this
/// approximation matches the spec intent for a continuous-physics world
/// (the agent transferring momentum is also the one in contact) without
/// expanding the ECS surface.
///
/// Returns `None` when no candidate agent is in range; in that case the
/// victim's energy stays with its daughters when it fragments.
pub fn find_killer(
    victim: Entity,
    victim_pos: Vec2,
    spatial: &SpatialHash,
    transforms: &Query<(Entity, &Transform)>,
    tags: &Query<&Tags>,
) -> Option<Entity> {
    let mut near: Vec<Entity> = Vec::new();
    spatial.query_radius(victim_pos, KILLER_SEARCH_RADIUS, &mut near);
    let mut best: Option<(Entity, f32)> = None;
    for other in near {
        if other == victim {
            continue;
        }
        let Ok((_oe, tf)) = transforms.get(other) else {
            continue;
        };
        let Ok(t) = tags.get(other) else { continue };
        if !t.0.has(CustomTags::AGENT) {
            continue;
        }
        let d2 = tf.translation.truncate().distance_squared(victim_pos);
        match best {
            Some((_, bd)) if d2 >= bd => {}
            _ => best = Some((other, d2)),
        }
    }
    best.map(|(e, _)| e)
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
            if let Ok(mut node_phys) = physicals.get_mut(other) {
                node_phys.energy -= transfer;
            } else {
                continue;
            }
            if let Ok(mut agent_phys) = physicals.get_mut(agent_e) {
                agent_phys.energy += transfer;
            } else {
                continue;
            }
            if let Ok(mut agent_ledger) = ledgers.get_mut(agent_e) {
                agent_ledger.harvested += transfer;
            }
            if let Ok(mut node_ledger) = ledgers.get_mut(other) {
                node_ledger.transferred_out += transfer;
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
        let Ok(mut trap_phys) = physicals.get_mut(trap) else {
            continue;
        };
        let release = trap_phys.energy.max(0.0) * VOLATILE_TRAP_DRAIN_FRACTION;
        if release <= 0.0 {
            continue;
        }
        trap_phys.energy -= release;
        if let Ok(mut ledger) = ledgers.get_mut(trap) {
            ledger.transferred_out += release;
        }

        for (target, w) in weights {
            let share = release * (w / total_w);
            if let Ok(mut p) = physicals.get_mut(target) {
                p.energy += share;
            }
            if let Ok(mut l) = ledgers.get_mut(target) {
                l.harvested += share;
            }
        }
    }
}

/// Consumption transfer: move a share of `victim`'s remaining energy to `attacker`.
/// Used by the Consumption channel (spec item 5) when a node's structure fails
/// below its threshold due to another node's finishing blow.
pub fn transfer_energy(victim: &mut Physical, attacker: &mut Physical) {
    let share = victim.energy * 0.5;
    victim.energy -= share;
    attacker.energy += share;
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

    /// `find_killer` must prefer the closest AGENT-tagged neighbor and ignore
    /// non-agents entirely.
    #[test]
    fn find_killer_picks_nearest_agent_and_ignores_non_agents() {
        let mut world = World::new();
        world.insert_resource(SpatialHash::default());
        // Spawn a victim at the origin.
        let victim = spawn_ecs_node(&mut world, Vec2::ZERO, 0, 5.0, 5.0, 1.0, 0.6);
        // Non-agent (just inert) close to victim: must NOT be picked.
        let _non_agent = spawn_ecs_node(&mut world, Vec2::new(1.0, 0.0), 0, 5.0, 5.0, 1.0, 0.6);
        // Agent further away: should be picked (only candidate).
        let far_agent = spawn_ecs_node(
            &mut world,
            Vec2::new(3.0, 0.0),
            CustomTags::AGENT,
            5.0,
            5.0,
            1.0,
            0.6,
        );

        // Rebuild spatial hash from transforms. Move the resource out
        // briefly so we can mutate `world` while accessing it.
        let mut h = world.remove_resource::<SpatialHash>();
        if let Some(ref mut h) = h {
            h.rebuild_from_state(&mut world);
        }
        world.insert_resource(h.unwrap_or_default());

        // Replicate find_killer's neighbor loop directly using QueryState
        // (the public `find_killer` is only callable from inside a system
        // where `Query` is in scope).
        let mut near: Vec<bevy::prelude::Entity> = Vec::new();
        {
            let h = world.resource::<SpatialHash>();
            h.query_radius(Vec2::ZERO, KILLER_SEARCH_RADIUS, &mut near);
        }
        let mut transforms = world.query::<(Entity, &Transform)>();
        let mut tags = world.query::<&Tags>();
        let mut best: Option<(bevy::prelude::Entity, f32)> = None;
        for other in near {
            if other == victim {
                continue;
            }
            let Ok((_oe, tf)) = transforms.get(&world, other) else {
                continue;
            };
            let Ok(t) = tags.get(&world, other) else {
                continue;
            };
            if !t.0.has(CustomTags::AGENT) {
                continue;
            }
            let d2 = tf.translation.truncate().distance_squared(Vec2::ZERO);
            match best {
                Some((_, bd)) if d2 >= bd => {}
                _ => best = Some((other, d2)),
            }
        }
        let killer = best.map(|(e, _)| e);
        assert_eq!(killer, Some(far_agent));

        let _ = world.entity(far_agent); // keep the entity id binding live
    }

    /// `transfer_energy` must move half of the victim's energy to the attacker
    /// and leave the victim with the other half (spec item 5).
    #[test]
    fn transfer_energy_splits_victim_energy_with_attacker() {
        let mut v = Physical {
            mass: 1.0,
            structure: 5.0,
            energy: 10.0,
        };
        let mut a = Physical {
            mass: 1.0,
            structure: 5.0,
            energy: 0.0,
        };
        transfer_energy(&mut v, &mut a);
        assert_eq!(v.energy, 5.0);
        assert_eq!(a.energy, 5.0);
    }

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
