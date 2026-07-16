//! In-process agent registry (spec item 3, "Both: trait + optional server").
//!
//! Each controllable agent owns a boxed [`Agent`] trait object. The world gathers
//! an [`ObservationTensor`] per agent each fixed step, calls `act`, and applies
//! the resulting [`ActionTensor`]. The trait object never sees any world type
//! beyond the tensors, keeping the world agent-agnostic.

use crate::components::{AgentMarker, AgentTuning};
use crate::fields::FieldSampler;
use crate::spatial_hash::SpatialHash;
use avian2d::dynamics::rigid_body::LinearVelocity;
use bevy::prelude::*;
use ertw_interface::{
    Agent as AgentTrait, InterfaceConfig, NeighborView, ObservationTensor, Vec2Lite,
};
use std::collections::HashMap;

/// Minimum energy reserve that an agent should hold above the thermodynamic
/// drain rate. Surplus above this threshold is the implicit reproduction signal
/// (spec item 10); agents cannot reproduce without sustained net inflow. Also
/// used as the `energy_surplus` slot of the self-state tensor (index 9) so any
/// downstream policy sees a meaningful value rather than a raw energy level.
pub const REPRODUCTION_THRESHOLD: f32 = crate::lineage::REPRODUCTION_ENERGY_THRESHOLD;

/// Owner record linking a Bevy entity to its controlling [`Agent`] trait object.
#[derive(Resource, Default)]
pub struct WorldAgents {
    /// Maps the agent controller id (stored in [`AgentMarker::controller`]) to the
    /// boxed trait object.
    controllers: HashMap<u64, Box<dyn AgentTrait>>,
    next_id: u64,
}

impl WorldAgents {
    /// Register an agent controller and return its assigned controller id.
    pub fn register(&mut self, agent: Box<dyn AgentTrait>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.controllers.insert(id, agent);
        id
    }

    pub fn spawn_child(&mut self, parent: u64, seed: u64) -> Option<u64> {
        let mut child = self.controllers.get_mut(&parent)?.spawn_child(seed)?;
        child.on_reset(seed);
        Some(self.register(child))
    }

    pub fn retain(&mut self, active: &std::collections::HashSet<u64>) {
        self.controllers.retain(|id, _| active.contains(id));
    }

    pub fn get(&self, id: u64) -> Option<&dyn AgentTrait> {
        self.controllers
            .get(&id)
            .map(|b| b.as_ref() as &dyn AgentTrait)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut dyn AgentTrait> {
        self.controllers
            .get_mut(&id)
            .map(|b| b.as_mut() as &mut dyn AgentTrait)
    }
}

/// Spawn a controllable agent entity wired with physics + interface components,
/// register its controller, and return the Bevy entity.
#[allow(clippy::too_many_arguments)]
pub fn spawn_agent(
    commands: &mut Commands,
    world_agents: &mut WorldAgents,
    agent: Box<dyn AgentTrait>,
    pos: Vec2,
    tuning: AgentTuning,
) -> Entity {
    let controller = world_agents.register(agent);
    spawn_with_bundle(commands, controller, pos, tuning)
}

/// Spawn an agent entity given an already-registered `controller` id. Used by
/// [`ErtwWorld::spawn_agent`] to avoid aliasing the `WorldAgents` resource while
/// issuing commands.
pub fn spawn_with_id(
    commands: &mut Commands,
    controller: u64,
    pos: Vec2,
    tuning: AgentTuning,
) -> Entity {
    spawn_with_bundle(commands, controller, pos, tuning)
}

/// Build the standard agent entity from a registered controller id.
fn spawn_with_bundle(
    commands: &mut Commands,
    controller: u64,
    pos: Vec2,
    tuning: AgentTuning,
) -> Entity {
    commands
        .spawn(crate::components::AgentBundle {
            transform: Transform::from_translation(pos.extend(0.0)),
            rigid_body: avian2d::prelude::RigidBody::Dynamic,
            collider: avian2d::prelude::Collider::circle(0.5),
            mass: avian2d::prelude::Mass(tuning.yield_threshold.max(0.1)),
            physical: crate::components::Physical {
                mass: tuning.yield_threshold.max(0.1),
                structure: tuning.yield_threshold,
                energy: 20.0,
            },
            yield_thresh: crate::components::Yield(tuning.yield_threshold),
            conductivity: crate::components::Conductivity(0.6),
            tags: crate::components::Tags(crate::tags::CustomTags::from_bits(
                crate::tags::CustomTags::AGENT
                    | crate::tags::CustomTags::CLAMP_CAPABLE
                    | crate::tags::CustomTags::OSCILLATOR,
            )),
            oscillator: crate::components::Oscillator {
                freq: tuning.osc_baseline,
                phase: 0.0,
                baseline_freq: tuning.osc_baseline,
            },
            impulse: crate::components::ImpulseAccum::default(),
            ledger: crate::components::EnergyLedger {
                born_step: 0,
                ..Default::default()
            },
            marker: AgentMarker {
                generation: 0,
                lineage: controller ^ 0xABCD,
                controller,
            },
            tuning,
            clamp: crate::components::ClampState::default(),
            fabricate: crate::components::FabricateCooldown::default(),
            reproduction: crate::components::ReproductionState::default(),
            node_rng: crate::components::NodeRng(controller.wrapping_mul(0x9E3779B1) ^ 0xABCD),
        })
        .id()
}

/// Build the egocentric observation for one agent entity from current world
/// state. Neighbors are gathered via the spatial hash, sorted by ascending
/// distance, and padded to `max_neighbors` with ghost nodes.
#[allow(clippy::too_many_arguments)]
pub fn build_observation(
    entity: Entity,
    tune: &AgentTuning,
    clock: &crate::SimClock,
    sampler: &FieldSampler,
    spatial: &SpatialHash,
    transforms: &Query<(Entity, &Transform)>,
    velocities: &Query<&LinearVelocity>,
    physicals: &Query<&crate::components::Physical>,
    tags: &Query<&crate::components::Tags>,
    conductivities: &Query<&crate::components::Conductivity>,
    oscillators: &Query<&crate::components::Oscillator>,
) -> Option<ObservationTensor> {
    let config = InterfaceConfig {
        max_neighbors: tune.max_neighbors,
        sensor_radius: tune.sensor_radius,
        field_samples: 4,
        field_channels: 3,
    };
    let mut obs = ObservationTensor::new(config);
    obs.step = clock.step;
    obs.entity_id = entity.to_bits();

    let (_e, self_tf) = transforms.get(entity).ok()?;
    let self_pos = self_tf.translation.truncate();
    let self_phys = physicals.get(entity).ok()?;
    let self_osc = oscillators.get(entity).ok()?;
    let self_vel = velocities.get(entity).map(|v| v.0).unwrap_or(Vec2::ZERO);
    let inverse_rotation = self_tf.rotation.inverse();
    let local_velocity = (inverse_rotation * self_vel.extend(0.0)).truncate();

    obs.self_state[0] = local_velocity.x;
    obs.self_state[1] = local_velocity.y;
    obs.self_state[2] = self_phys.mass;
    obs.self_state[3] = self_phys.structure;
    obs.self_state[4] = self_phys.energy;
    obs.self_state[5] = self_osc.freq;
    obs.self_state[6] = self_osc.phase;
    obs.self_state[7] = (self_phys.energy - REPRODUCTION_THRESHOLD).max(0.0);

    // Sample the center and an egocentric ring. Each field contributes value,
    // local gradient X, and local gradient Y.
    const GRADIENT_EPSILON: f32 = 0.05;
    for s in 0..config.field_samples {
        let local_probe = if s == 0 || config.field_samples == 1 {
            Vec2::ZERO
        } else {
            let angle = (s - 1) as f32 / (config.field_samples - 1) as f32 * std::f32::consts::TAU;
            Vec2::from_angle(angle) * config.sensor_radius
        };
        let world_probe = (self_tf.rotation * local_probe.extend(0.0)).truncate();
        let probe = self_pos + world_probe;
        let (f, gradients) = sampler.sample_with_gradient(probe, GRADIENT_EPSILON);
        let fields = [
            (f.kinetic, gradients[0]),
            (f.thermal, gradients[1]),
            (f.em, gradients[2]),
        ];
        for (field_index, (value, world_gradient)) in fields.into_iter().enumerate() {
            let local_gradient = (inverse_rotation * world_gradient.extend(0.0)).truncate();
            let base = (field_index * config.field_samples + s) * config.field_channels;
            obs.field[base] = value;
            obs.field[base + 1] = local_gradient.x;
            obs.field[base + 2] = local_gradient.y;
        }
    }

    // Neighbors via spatial hash within sensor radius.
    let mut near: Vec<Entity> = Vec::new();
    spatial.query_radius(self_pos, config.sensor_radius, &mut near);
    let mut views: Vec<(f32, NeighborView)> = Vec::new();
    for other in near {
        if other == entity {
            continue;
        }
        let Ok((_oe, tf)) = transforms.get(other) else {
            continue;
        };
        let Ok(phys) = physicals.get(other) else {
            continue;
        };
        let Ok(t) = tags.get(other) else { continue };
        let world_rel_pos = tf.translation.truncate() - self_pos;
        let dist2 = world_rel_pos.length_squared();
        if dist2 > config.sensor_radius * config.sensor_radius {
            continue;
        }
        let other_vel = velocities.get(other).map(|v| v.0).unwrap_or(Vec2::ZERO);
        let world_rel_vel = other_vel - self_vel;
        let rel_pos = (inverse_rotation * world_rel_pos.extend(0.0)).truncate();
        let rel_vel = (inverse_rotation * world_rel_vel.extend(0.0)).truncate();
        let cond = conductivities.get(other).map(|c| c.0).unwrap_or(0.6);
        let osc = oscillators.get(other).ok();
        views.push((
            dist2,
            NeighborView {
                rel_pos: Vec2Lite::new(rel_pos.x, rel_pos.y),
                rel_vel: Vec2Lite::new(rel_vel.x, rel_vel.y),
                mass: phys.mass,
                structure: phys.structure,
                energy: phys.energy,
                tags: t.0 .0,
                conductivity: cond,
                osc_freq: osc.map(|o| o.freq).unwrap_or(0.0),
                osc_phase: osc.map(|o| o.phase).unwrap_or(0.0),
                valid: true,
            },
        ));
    }
    views.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for (i, (_, v)) in views.into_iter().take(config.max_neighbors).enumerate() {
        obs.neighbors[i] = v;
    }
    Some(obs)
}
