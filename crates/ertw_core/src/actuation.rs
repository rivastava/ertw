//! Actuators (spec items 3, 9, 10).
//!
//! The locomotion force/torque is applied in `crate::apply_agent_actions`. This
//! module layers the remaining actuators on top of the queued [`ActionTensor`]:
//!
//! - **Clamp** (`clamp > 0.5`): create a fixed two-body joint with the nearest
//!   physical neighbor. Holding the joint drains energy and release starts a
//!   cooldown.
//! - **Fabricate** (`fabricate > 0.5`): spawn a low-conductivity inert structure
//!   node at the agent's position, spending mass + energy. Gated by a cooldown
//!   (spec item 3: "fabricate low-conductivity structure").
//! - **Oscillator** (`osc_freq`/`osc_phase`): drive the agent's internal
//!   oscillator broadcast toward the commanded values (spec items 9, 10).
//!
//! The world interprets and bounds every actuator; the agent only ever emits
//! continuous tensor values.

use crate::components::{
    ClampJoint, ClampState, EnergyFlow, FabricateCooldown, NodeRng, Oscillator, Physical, Tags,
};
use crate::spatial_hash::SpatialHash;
use crate::tags::CustomTags;
use crate::PendingActions;
use bevy::prelude::*;
use ertw_interface::ActionTensor;

/// Energy cost per second of holding a clamp.
pub const CLAMP_DRAIN: f32 = 0.4;
/// Cooldown (seconds) after releasing a clamp before it can re-engage.
pub const CLAMP_COOLDOWN: f32 = 1.5;
/// Fabrication cost (energy) to spawn one structure node.
pub const FABRICATE_ENERGY_COST: f32 = 3.0;
/// Fabrication cost (mass) drawn from the agent.
pub const FABRICATE_MASS_COST: f32 = 0.5;
/// Cooldown (seconds) between fabrications.
pub const FABRICATE_COOLDOWN: f32 = 3.0;
/// How quickly the commanded oscillator value is approached (per second).
pub const OSC_RESPONSE: f32 = 4.0;

/// Queue joint entities before queueing their physical endpoints for despawn.
/// Avian relationship hooks require this ordering within one command buffer.
pub fn despawn_physical_entities(
    commands: &mut Commands,
    entities: &[Entity],
    joints: &Query<(Entity, &ClampJoint)>,
) {
    for (joint_entity, joint) in joints.iter() {
        if entities.contains(&joint.owner) || entities.contains(&joint.target) {
            commands.entity(joint_entity).despawn();
        }
    }
    for entity in entities {
        commands.entity(*entity).despawn();
    }
}

/// Remove orphaned joint entities and reset clamp state after either endpoint
/// disappears. This runs every simulation tick independently of agent actions.
pub fn cleanup_clamp_joints(
    mut commands: Commands,
    physicals: Query<(), With<Physical>>,
    joints: Query<(Entity, &ClampJoint)>,
    mut clamps: Query<&mut ClampState>,
) {
    for (joint_entity, joint) in joints.iter() {
        let owner_matches = clamps
            .get(joint.owner)
            .is_ok_and(|clamp| clamp.joint == Some(joint_entity));
        if physicals.get(joint.owner).is_err() || !owner_matches {
            commands.entity(joint_entity).despawn();
        }
    }

    for mut clamp in clamps.iter_mut() {
        let target_exists = clamp
            .target
            .is_none_or(|target| physicals.get(target).is_ok());
        let joint_exists = clamp.joint.is_none_or(|joint| joints.get(joint).is_ok());
        if !target_exists || !joint_exists {
            if let Some(joint) = clamp.joint.take() {
                commands.entity(joint).despawn();
            }
            clamp.target = None;
            clamp.cooldown = CLAMP_COOLDOWN;
        }
    }
}

/// Apply the non-locomotion actuators for one fixed step. Split from the force
/// apply pass only to keep borrow sets simple; runs inside `FixedUpdate`.
#[allow(clippy::too_many_arguments)]
pub fn apply_actuators(
    mut commands: Commands,
    pending: Res<PendingActions>,
    spatial: Res<SpatialHash>,
    transforms: Query<(Entity, &Transform)>,
    mut physicals: Query<&mut Physical>,
    mut masses: Query<&mut avian2d::prelude::Mass>,
    mut ledgers: Query<&mut crate::components::EnergyLedger>,
    tags: Query<&Tags>,
    mut oscillators: Query<&mut Oscillator>,
    mut clamps: Query<&mut ClampState>,
    mut fabricate_cds: Query<&mut FabricateCooldown>,
) {
    for (e, action) in pending.items.iter() {
        let e = *e;
        apply_one(
            &mut commands,
            e,
            action,
            &spatial,
            &transforms,
            &mut physicals,
            &mut masses,
            &mut ledgers,
            &tags,
            &mut oscillators,
            &mut clamps,
            &mut fabricate_cds,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_one(
    commands: &mut Commands,
    e: Entity,
    action: &ActionTensor,
    spatial: &SpatialHash,
    transforms: &Query<(Entity, &Transform)>,
    physicals: &mut Query<&mut Physical>,
    masses: &mut Query<&mut avian2d::prelude::Mass>,
    ledgers: &mut Query<&mut crate::components::EnergyLedger>,
    tags: &Query<&Tags>,
    oscillators: &mut Query<&mut Oscillator>,
    clamps: &mut Query<&mut ClampState>,
    fabricate_cds: &mut Query<&mut FabricateCooldown>,
) {
    let dt = crate::economy::FIXED_DT;
    let (Ok((_, self_tf)), _) = (transforms.get(e), physicals.get(e)) else {
        return;
    };
    let self_pos = self_tf.translation.truncate();

    // --- Oscillator drive ---
    if let Ok(mut osc) = oscillators.get_mut(e) {
        let k = (OSC_RESPONSE * dt).clamp(0.0, 1.0);
        let osc_cost = (action.osc_freq - osc.freq).abs() * 0.005;
        let mut can_drive = osc_cost == 0.0;
        if let (Ok(mut physical), Ok(mut ledger)) = (physicals.get_mut(e), ledgers.get_mut(e)) {
            can_drive = ledger.debit_exact(&mut physical, EnergyFlow::Actuation, osc_cost);
        }
        if can_drive {
            osc.freq += (action.osc_freq - osc.freq) * k;
            osc.phase = action.osc_phase;
        }
    }

    // --- Clamp ---
    if let Ok(mut clamp) = clamps.get_mut(e) {
        if clamp.cooldown > 0.0 {
            clamp.cooldown -= dt;
        }
        let want_clamp = action.clamp > 0.5;
        if want_clamp && clamp.cooldown <= 0.0 {
            // Acquire nearest valid neighbor if not already clamped.
            if clamp.target.is_none() {
                if let Some(target) = nearest_neighbor(e, self_pos, spatial, transforms, tags) {
                    clamp.target = Some(target);
                    clamp.joint = Some(
                        commands
                            .spawn((
                                avian2d::prelude::FixedJoint::new(e, target),
                                avian2d::prelude::JointCollisionDisabled,
                                ClampJoint { owner: e, target },
                            ))
                            .id(),
                    );
                }
            }
        } else if !want_clamp {
            if let Some(joint) = clamp.joint.take() {
                commands.entity(joint).despawn();
                clamp.cooldown = CLAMP_COOLDOWN;
            }
            clamp.target = None;
        }

        if let Some(target) = clamp.target {
            // Release if target gone.
            if transforms.get(target).is_err() {
                if let Some(joint) = clamp.joint.take() {
                    commands.entity(joint).despawn();
                }
                clamp.target = None;
                clamp.cooldown = CLAMP_COOLDOWN;
            } else if let (Ok(mut phys), Ok(mut ledger)) =
                (physicals.get_mut(e), ledgers.get_mut(e))
            {
                let cost = CLAMP_DRAIN * dt;
                ledger.debit_available(&mut phys, EnergyFlow::Actuation, cost);
                if phys.energy <= 0.0 {
                    if let Some(joint) = clamp.joint.take() {
                        commands.entity(joint).despawn();
                    }
                    clamp.target = None;
                    clamp.cooldown = CLAMP_COOLDOWN;
                }
            }
        }
    }

    // --- Fabricate ---
    if let Ok(mut cd) = fabricate_cds.get_mut(e) {
        if cd.remaining > 0.0 {
            cd.remaining -= dt;
        }
        if action.fabricate > 0.5 && cd.remaining <= 0.0 {
            if let (Ok(mut phys), Ok(mut ledger)) = (physicals.get_mut(e), ledgers.get_mut(e)) {
                if phys.energy >= FABRICATE_ENERGY_COST && phys.mass >= FABRICATE_MASS_COST {
                    let funded = ledger.debit_exact(
                        &mut phys,
                        EnergyFlow::Fabrication,
                        FABRICATE_ENERGY_COST,
                    );
                    if !funded {
                        return;
                    }
                    phys.mass -= FABRICATE_MASS_COST;
                    if let Ok(mut mass) = masses.get_mut(e) {
                        mass.0 = phys.mass.max(0.05);
                    }
                    cd.remaining = FABRICATE_COOLDOWN;
                    spawn_structure(
                        commands,
                        self_pos + Vec2::X,
                        FABRICATE_MASS_COST,
                        FABRICATE_ENERGY_COST,
                    );
                }
            }
        }
    }
}

/// Spawn an inert low-conductivity structure node (fabricated shelter).
fn spawn_structure(commands: &mut Commands, pos: Vec2, mass: f32, energy: f32) {
    commands.spawn((
        Transform::from_translation(pos.extend(0.0)),
        avian2d::prelude::RigidBody::Dynamic,
        avian2d::prelude::Collider::circle(0.5),
        avian2d::prelude::Mass(mass),
        Physical {
            mass,
            structure: 12.0,
            energy,
        },
        crate::components::Yield(12.0),
        crate::components::Conductivity(0.2), // low conductivity = shelter
        Tags(CustomTags::from_bits(CustomTags::SHELTER)),
        crate::components::ImpulseAccum::default(),
        crate::components::EnergyLedger::default(),
        NodeRng(0),
    ));
}

/// Find the nearest real neighbor entity (excluding self) within sensor radius.
fn nearest_neighbor(
    self_e: Entity,
    self_pos: Vec2,
    spatial: &SpatialHash,
    transforms: &Query<(Entity, &Transform)>,
    tags: &Query<&Tags>,
) -> Option<Entity> {
    let mut near: Vec<Entity> = Vec::new();
    spatial.query_radius(self_pos, 12.0, &mut near);
    let mut best: Option<(Entity, f32)> = None;
    for other in near {
        if other == self_e {
            continue;
        }
        // SpatialHash contains every Transform, including cameras and other
        // observer-only entities in rendered worlds. Clamp only to physical
        // nodes, all of which carry Tags.
        if tags.get(other).is_err() {
            continue;
        }
        if let Ok((_oe, tf)) = transforms.get(other) {
            let d2 = tf.translation.truncate().distance_squared(self_pos);
            if d2 > 12.0 * 12.0 {
                continue;
            }
            match best {
                Some((_, bd)) if d2 >= bd => {}
                _ => best = Some((other, d2)),
            }
        }
    }
    best.map(|(e, _)| e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ertw_interface::{Agent, ObservationTensor, Vec2Lite};

    struct Fabricator;

    impl Agent for Fabricator {
        fn act(&mut self, _observation: &ObservationTensor) -> ActionTensor {
            ActionTensor {
                force: Vec2Lite::ZERO,
                fabricate: 1.0,
                osc_freq: 1.0,
                ..Default::default()
            }
        }
    }

    struct ClampOnce {
        first: bool,
    }

    impl Agent for ClampOnce {
        fn act(&mut self, _observation: &ObservationTensor) -> ActionTensor {
            let clamp = if self.first { 1.0 } else { 0.0 };
            self.first = false;
            ActionTensor {
                clamp,
                osc_freq: 1.0,
                ..Default::default()
            }
        }
    }

    struct Passive;

    impl Agent for Passive {
        fn act(&mut self, _observation: &ObservationTensor) -> ActionTensor {
            ActionTensor {
                osc_freq: 1.0,
                ..Default::default()
            }
        }
    }

    struct ClampHold;

    impl Agent for ClampHold {
        fn act(&mut self, _observation: &ObservationTensor) -> ActionTensor {
            ActionTensor {
                clamp: 1.0,
                osc_freq: 1.0,
                ..Default::default()
            }
        }
    }

    #[test]
    fn fabrication_transfers_exact_mass_and_energy() {
        let mut simulation = crate::ErtwWorld::new(11);
        let parent = simulation.spawn_agent(Box::new(Fabricator), Vec2::ZERO);
        simulation.app().world_mut().flush();
        simulation
            .app()
            .world_mut()
            .resource_mut::<crate::SimClock>()
            .step = 1;
        simulation.step(1);

        let world = simulation.app().world_mut();
        let parent_physical = *world.get::<Physical>(parent).expect("parent");
        assert!((parent_physical.mass - 7.5).abs() < 1.0e-4);
        let mut shelters = world.query::<(&Physical, &Tags)>();
        let fabricated = shelters
            .iter(world)
            .find(|(_, tags)| tags.0.has(CustomTags::SHELTER))
            .map(|(physical, _)| *physical)
            .expect("fabricated shelter");
        assert!((fabricated.mass - FABRICATE_MASS_COST).abs() < 1.0e-4);
        assert!(fabricated.energy <= FABRICATE_ENERGY_COST);
        assert!(fabricated.energy > FABRICATE_ENERGY_COST - 0.05);
        assert!((parent_physical.mass + fabricated.mass - 8.0).abs() < 1.0e-4);
    }

    #[test]
    fn clamp_creates_and_releases_physical_joint_with_cooldown() {
        let mut simulation = crate::ErtwWorld::new(12);
        let clamping =
            simulation.spawn_agent(Box::new(ClampOnce { first: true }), Vec2::new(-0.75, 0.0));
        simulation.spawn_agent(Box::new(Passive), Vec2::new(0.75, 0.0));
        simulation.app().world_mut().flush();
        simulation
            .app()
            .world_mut()
            .resource_mut::<crate::SimClock>()
            .step = 1;

        simulation.step(1);
        let engaged = *simulation
            .app()
            .world()
            .get::<ClampState>(clamping)
            .expect("clamp state");
        assert!(engaged.target.is_some());
        assert!(engaged.joint.is_some());

        simulation.step(1);
        let released = *simulation
            .app()
            .world()
            .get::<ClampState>(clamping)
            .expect("clamp state");
        assert!(released.target.is_none());
        assert!(released.joint.is_none());
        assert!(released.cooldown > 0.0);
    }

    #[test]
    fn clamp_cleanup_handles_target_and_owner_despawn() {
        let mut target_case = crate::ErtwWorld::new(13);
        let owner = target_case.spawn_agent(Box::new(ClampHold), Vec2::new(-0.75, 0.0));
        let target = target_case.spawn_agent(Box::new(Passive), Vec2::new(0.75, 0.0));
        target_case.step(1);
        let joint = target_case
            .app()
            .world()
            .get::<ClampState>(owner)
            .and_then(|state| state.joint)
            .expect("engaged joint");
        target_case.app().world_mut().despawn(target);
        target_case.step(1);
        let state = target_case
            .app()
            .world()
            .get::<ClampState>(owner)
            .expect("owner");
        assert!(state.target.is_none());
        assert!(state.joint.is_none());
        assert!(target_case.app().world().get_entity(joint).is_err());

        let mut owner_case = crate::ErtwWorld::new(14);
        let owner = owner_case.spawn_agent(Box::new(ClampHold), Vec2::new(-0.75, 0.0));
        owner_case.spawn_agent(Box::new(Passive), Vec2::new(0.75, 0.0));
        owner_case.step(1);
        let joint = owner_case
            .app()
            .world()
            .get::<ClampState>(owner)
            .and_then(|state| state.joint)
            .expect("engaged joint");
        owner_case.app().world_mut().despawn(owner);
        owner_case.step(1);
        assert!(owner_case.app().world().get_entity(joint).is_err());
    }
}
