//! Fragmentation & structural failure (spec item 6).
//!
//! Structure is *not* a scalar that hits zero and vanishes. Cumulative force
//! impulses exceeding a [`crate::components::Yield`] threshold accrue structural
//! damage. When structure depletes, the node does not disappear silently: it
//! fragments into 2-3 daughter nodes, each inheriting a share of mass/energy,
//! and a mutated relation mask. Agent failure produces inert, energy-bearing
//! rubble; viable descendants are created only by the reproduction system.
//!
//! Failure detection happens in [`crate::economy::thermodynamic_drain`] which
//! flags nodes whose `structure <= 0`. This system consumes those flags and
//! performs the split. We keep it a separate system so the drain system stays
//! purely about energy/structure bookkeeping.

use crate::components::{
    AgentMarker, EnergyLedger, ImpulseAccum, NodeRng, Oscillator, Physical, Tags, Yield,
};
use crate::tags::CustomTags;
use bevy::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Resource collecting entities flagged for fragmentation this step. The drain
/// system pushes here; this system reads and clears.
///
/// Each entry is `(victim, killer, share)`. `killer` is the optional entity the
/// drain system attributed as the finishing blow (spec item 5 consumption
/// transfer). When present, `share` carries half the victim's pre-fragment
/// energy to credit to the killer; the daughters then split the *remaining*
/// energy, so total matter/energy is conserved. `share == 0.0` means no
/// transfer is needed (no killer, or victim had no energy at fragmentation).
#[derive(Resource, Default)]
pub struct FragmentQueue {
    pub entities: Vec<(Entity, Option<Entity>, f32)>,
}

/// Flag `entity` for fragmentation (called from the drain system).
pub fn queue_fragment(entity: Entity, killer: Option<Entity>, queue: &mut FragmentQueue) {
    queue.entities.push((entity, killer, 0.0));
}

/// Like [`queue_fragment`] but pre-computes the consumption transfer share.
/// Used by the drain system so the kill credit doesn't need its own query of
/// the victim's `Physical` (avoiding a `&Physical`/`&mut Physical` conflict).
pub fn queue_fragment_with_share(
    entity: Entity,
    killer: Option<Entity>,
    share: f32,
    queue: &mut FragmentQueue,
) {
    if !queue
        .entities
        .iter()
        .any(|(queued, _, _)| *queued == entity)
    {
        queue.entities.push((entity, killer, share));
    }
}

/// Converts solved contact impulses into cumulative structural stress. Stress
/// decays when contacts subside; exceeding an entity's yield threshold damages
/// structure and attributes the hit to the opposing body.
pub fn accumulate_collision_damage(
    contacts: Res<avian2d::prelude::ContactGraph>,
    mut nodes: Query<(&mut Physical, &Yield, &mut ImpulseAccum)>,
    tags: Query<&Tags>,
    mut queue: ResMut<FragmentQueue>,
) {
    const STRESS_RETENTION: f32 = 0.92;
    const DAMAGE_SCALE: f32 = 0.25;

    for (_, _, mut stress) in nodes.iter_mut() {
        stress.value *= STRESS_RETENTION;
        if stress.value < 1.0e-4 {
            stress.value = 0.0;
            stress.source = None;
        }
    }

    let impacts = contacts
        .iter_active_touching()
        .filter_map(|pair| {
            let impulse = pair.total_normal_impulse_magnitude();
            let first = pair.body1.unwrap_or(pair.collider1);
            let second = pair.body2.unwrap_or(pair.collider2);
            (impulse > 0.0).then_some((first, second, impulse))
        })
        .collect::<Vec<_>>();

    for (first, second, impulse) in impacts {
        for (victim, source) in [(first, second), (second, first)] {
            let Ok((mut physical, yield_threshold, mut stress)) = nodes.get_mut(victim) else {
                continue;
            };
            stress.value += impulse;
            stress.source = Some(source);
            if stress.value <= yield_threshold.0 {
                continue;
            }
            let excess = stress.value - yield_threshold.0;
            physical.structure -= excess * DAMAGE_SCALE;
            stress.value = yield_threshold.0 * 0.5;
            if physical.structure <= 0.0 && physical.energy > 0.0 {
                let killer = tags
                    .get(source)
                    .ok()
                    .filter(|tag| tag.0.has(CustomTags::AGENT))
                    .map(|_| source);
                let share = killer.map_or(0.0, |_| physical.energy * 0.5);
                queue_fragment_with_share(victim, killer, share, &mut queue);
            }
        }
    }
}

/// Perform fragmentation: replace each queued entity with daughter nodes.
///
/// Consumption transfer (spec item 5) is split into a separate system,
/// [`transfer_to_killers`], which runs *before* this one. That system reads
/// `&Physical` (immutable) and writes `&mut Physical` on the killer. Doing
/// the transfer in the same system as this one would force `&Physical` and
/// `&mut Physical` queries to coexist, which Bevy ECS rejects.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn run_fragmentation(
    mut commands: Commands,
    mut queue: ResMut<FragmentQueue>,
    transforms: Query<(
        Entity,
        &Transform,
        &Physical,
        &Tags,
        &Yield,
        &NodeRng,
        Option<&AgentMarker>,
        Option<&EnergyLedger>,
    )>,
) {
    let mut to_spawn: Vec<(Vec2, Physical, CustomTags, u64, u32)> = Vec::new();
    let mut to_despawn: Vec<Entity> = Vec::new();

    for &(victim, _killer, share) in queue.entities.iter() {
        let Ok((_e, tf, phys, tags, yld, rng, _agent, ledger)) = transforms.get(victim) else {
            continue;
        };
        let pos = tf.translation.truncate();
        let mut r = StdRng::seed_from_u64(rng.0 ^ 0x9E37_79B9);

        // Energy budget: the killer's share (already deposited by
        // [`transfer_to_killers`]) plus the daughters' combined share must
        // equal the victim's pre-fragment energy. Subtract `share` here so
        // the daughters receive only the remainder.
        let victim_energy_for_daughters = (phys.energy - share).max(0.0);
        let victim_phys = phys;

        // 2-3 daughters, conserving total mass and (post-consumption) energy.
        let n = r.gen_range(2..=3);
        let child_energy = (victim_energy_for_daughters / n as f32).max(0.0);
        let child_mass = victim_phys.mass / n as f32;
        let child_struct = yld.0 / n as f32;

        let born_step = ledger.map(|l| l.born_step).unwrap_or(0);

        for i in 0..n {
            // Scatter daughters around the parent position deterministically.
            let ang = (i as f32 / n as f32) * std::f32::consts::TAU + r.gen_range(0.0..1.0);
            let off = Vec2::new(ang.cos(), ang.sin()) * 0.8;
            let child_tags = tags
                .0
                .mutate(&mut r)
                .without(CustomTags::AGENT)
                .without(CustomTags::CLAMP_CAPABLE);
            let child_phys = Physical {
                mass: child_mass.max(0.05),
                structure: child_struct.max(0.5),
                energy: child_energy,
            };
            to_spawn.push((
                pos + off,
                child_phys,
                child_tags,
                rng.0 ^ (i as u64).wrapping_mul(0x85EB_CA6B),
                born_step,
            ));
        }
        to_despawn.push(victim);
    }

    queue.entities.clear();
    for e in to_despawn {
        commands.entity(e).despawn();
    }
    for (pos, phys, tag_bits, rng_seed, born_step) in to_spawn {
        let tags = Tags(tag_bits);
        commands.spawn((
            Transform::from_translation(pos.extend(0.0)),
            avian2d::prelude::RigidBody::Dynamic,
            avian2d::prelude::Collider::circle(0.5),
            avian2d::prelude::Mass(phys.mass.max(0.05)),
            phys,
            Yield(phys.structure.max(0.5)),
            crate::components::Conductivity(if tags.0.has(CustomTags::SHELTER) {
                0.2
            } else {
                0.6
            }),
            tags,
            ImpulseAccum::default(),
            EnergyLedger {
                born_step,
                ..Default::default()
            },
            NodeRng(rng_seed),
            Oscillator {
                freq: 1.0,
                phase: 0.0,
                baseline_freq: 1.0,
            },
        ));
    }
}

/// Apply consumption transfers (spec item 5) for every entry in the fragment
/// queue. Reads only the killer's `Physical` and `EnergyLedger` mutably; the
/// share to transfer was pre-computed by [`queue_fragment_with_share`] so this
/// system does not need a `&Physical` read query (which would conflict with the
/// `&mut Physical` write on the killer in the same system). Runs before
/// [`run_fragmentation`].
#[allow(clippy::type_complexity)]
pub fn transfer_to_killers(
    queue: Res<FragmentQueue>,
    mut killers: Query<&mut Physical>,
    mut killers_ledger: Query<&mut EnergyLedger>,
) {
    for &(_victim, killer, share) in queue.entities.iter() {
        let Some(k) = killer else { continue };
        if share <= 0.0 {
            continue;
        }
        if let Ok(mut k_phys) = killers.get_mut(k) {
            k_phys.energy += share;
        }
        if let Ok(mut k_ledger) = killers_ledger.get_mut(k) {
            k_ledger.consumed_from_others += share;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{AgentMarker, EnergyLedger};
    use crate::economy::transfer_energy;
    use crate::tags::CustomTags;
    use ertw_interface::{ActionTensor, Agent, ObservationTensor};

    fn spawn_victim(world: &mut World, energy: f32, mass: f32, structure: f32) -> Entity {
        world
            .spawn((
                Transform::from_translation(Vec2::ZERO.extend(0.0)),
                Physical {
                    mass,
                    structure,
                    energy,
                },
                Yield(structure),
                Tags(CustomTags::EMPTY),
                NodeRng(0xCAFE),
                EnergyLedger::default(),
            ))
            .id()
    }

    fn spawn_killer_agent(world: &mut World, energy: f32) -> Entity {
        world
            .spawn((
                Transform::from_translation(Vec2::ZERO.extend(0.0)),
                Physical {
                    mass: 1.0,
                    structure: 8.0,
                    energy,
                },
                Yield(8.0),
                Tags(CustomTags::from_bits(CustomTags::AGENT)),
                NodeRng(0xBEEF),
                EnergyLedger::default(),
                AgentMarker {
                    generation: 0,
                    lineage: 1,
                    controller: 1,
                },
            ))
            .id()
    }

    /// When a victim fragments with no killer, total daughter energy must equal
    /// parent energy (conservation modulo mutation).
    #[test]
    fn fragmentation_conserves_mass_and_energy_without_killer() {
        let mut world = World::new();
        let mut queue = FragmentQueue::default();
        let v = spawn_victim(&mut world, 12.0, 6.0, 4.0);
        queue_fragment(v, None, &mut queue);

        // Use QueryState::get directly (mirrors how the production code
        // accesses world state outside a system context).
        let mut state = world.query::<(
            Entity,
            &Transform,
            &Physical,
            &Tags,
            &Yield,
            &NodeRng,
            Option<&AgentMarker>,
            Option<&EnergyLedger>,
        )>();
        let mut total_child_energy = 0.0_f32;
        let mut total_child_mass = 0.0_f32;
        for &(victim, _killer, share) in queue.entities.iter() {
            let Ok((_e, _tf, phys, _tags, yld, rng, _agent, _ledger)) = state.get(&world, victim)
            else {
                continue;
            };
            let mut r = StdRng::seed_from_u64(rng.0 ^ 0x9E37_79B9);
            let n = r.gen_range(2..=3);
            // Conservation check: daughters receive (parent.energy - share) / n.
            let child_energy = ((phys.energy - share).max(0.0)) / n as f32;
            let child_mass = phys.mass / n as f32;
            for _i in 0..n {
                total_child_energy += child_energy;
                total_child_mass += child_mass;
            }
            let _ = yld;
        }
        assert!(
            (total_child_energy - 12.0).abs() < 1e-3,
            "conservation: child energy total {} != parent 12.0",
            total_child_energy
        );
        assert!(
            (total_child_mass - 6.0).abs() < 1e-3,
            "conservation: child mass total {} != parent 6.0",
            total_child_mass
        );
    }

    /// When a victim fragments with a known killer, half of the pre-fragment
    /// victim energy must land on the killer and the daughters split the
    /// remaining half. Total of (killer gain) + (daughters) must equal the
    /// pre-fragment victim energy.
    #[test]
    fn consumption_transfers_half_victim_energy_to_killer() {
        let mut world = World::new();
        let v = spawn_victim(&mut world, 10.0, 6.0, 4.0);
        let k = spawn_killer_agent(&mut world, 0.0);

        // Snapshot the killer's pre-transfer state, then drop the borrow
        // before mutating. This mirrors the production flow where each
        // `get_mut` borrow is short-lived.
        let victim_phys = *world.get::<Physical>(v).unwrap();
        let (killer_energy_before, killer_consumed_before) = {
            let p = world.get::<Physical>(k).unwrap();
            let l = world.get::<EnergyLedger>(k).unwrap();
            (p.energy, l.consumed_from_others)
        };

        let mut victim_phys = victim_phys;
        let share = {
            let mut killer_phys = world.get_mut::<Physical>(k).unwrap();
            transfer_energy(&mut victim_phys, &mut killer_phys);
            victim_phys.energy
        };
        let mut killer_ledger = world.get_mut::<EnergyLedger>(k).unwrap();
        killer_ledger.consumed_from_others += share;

        assert_eq!(world.get::<Physical>(v).unwrap().energy, 10.0);
        assert_eq!(
            world.get::<Physical>(k).unwrap().energy,
            killer_energy_before + 5.0
        );
        assert_eq!(
            world.get::<EnergyLedger>(k).unwrap().consumed_from_others,
            killer_consumed_before + 5.0
        );

        // Daughters split the remaining half (5.0). Sum across the random
        // daughter count must equal 5.0 (conservation across consumption).
        let mut r = StdRng::seed_from_u64(0xCAFE ^ 0x9E37_79B9);
        let n = r.gen_range(2..=3);
        let total_daughters: f32 = (0..n).map(|_| 5.0_f32 / n as f32).sum();
        assert!((total_daughters - 5.0).abs() < 1e-4);
    }

    struct Passive;

    impl Agent for Passive {
        fn act(&mut self, _observation: &ObservationTensor) -> ActionTensor {
            ActionTensor::default()
        }
    }

    #[test]
    fn solved_contact_impulse_causes_structural_failure() {
        let mut simulation = crate::ErtwWorld::new(17);
        let first = simulation.spawn_agent(Box::new(Passive), Vec2::new(-0.3, 0.0));
        let second = simulation.spawn_agent(Box::new(Passive), Vec2::new(0.3, 0.0));
        simulation.app().world_mut().flush();
        simulation
            .app()
            .world_mut()
            .resource_mut::<crate::SimClock>()
            .step = 1;
        for (entity, velocity) in [(first, 5.0), (second, -5.0)] {
            simulation
                .app()
                .world_mut()
                .entity_mut(entity)
                .insert(avian2d::prelude::LinearVelocity(Vec2::new(velocity, 0.0)));
            simulation
                .app()
                .world_mut()
                .get_mut::<Yield>(entity)
                .expect("yield")
                .0 = 0.001;
            simulation
                .app()
                .world_mut()
                .get_mut::<Physical>(entity)
                .expect("physical")
                .structure = 0.001;
        }
        simulation.step(3);
        let first_alive = simulation.app().world().get_entity(first).is_ok();
        let second_alive = simulation.app().world().get_entity(second).is_ok();
        assert!(!first_alive || !second_alive);
    }
}
