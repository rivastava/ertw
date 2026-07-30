//! Canonical decision-boundary snapshots for durable external-agent sessions.
//!
//! Snapshots contain ERTW physical state, clocks, seeded state, public entity
//! identities, and an opaque external-agent checkpoint reference. They never
//! serialize an agent implementation or evaluator output.

use crate::components::*;
use crate::fields::{FieldSampler, SimSeed};
use crate::{ErtwWorld, SimClock};
use avian2d::prelude::{AngularVelocity, LinearVelocity, Mass, RigidBody};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

pub const SNAPSHOT_SCHEMA_VERSION: u16 = 2;

#[derive(Resource, Default, Debug)]
pub struct StableIdAllocator {
    next: u64,
}

impl StableIdAllocator {
    pub fn allocate(&mut self) -> StableId {
        self.next = self.next.saturating_add(1).max(1);
        StableId(self.next)
    }

    fn observe(&mut self, value: u64) {
        self.next = self.next.max(value);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CanonicalNode {
    pub stable_id: u64,
    pub position: [f32; 2],
    pub rotation: f32,
    pub linear_velocity: [f32; 2],
    pub angular_velocity: f32,
    pub physical: [f32; 3],
    pub yield_threshold: f32,
    pub conductivity: f32,
    pub tags: u64,
    pub oscillator: Option<[f32; 3]>,
    pub node_rng: u64,
    pub chunk_origin: Option<[i32; 2]>,
    pub dead: bool,
    pub energy_ledger: [f32; 8],
    pub born_step: u32,
    pub agent: Option<CanonicalAgent>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CanonicalAgent {
    pub generation: u32,
    pub lineage: u64,
    pub sensor_radius: f32,
    pub max_neighbors: usize,
    pub yield_threshold: f32,
    pub oscillator_baseline: f32,
    pub reproduction: [f32; 2],
    pub fabricate_cooldown: f32,
    pub clamp_target: Option<u64>,
    pub clamp_cooldown: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WorldSnapshot {
    pub schema_version: u16,
    pub seed: u64,
    pub simulation_tick: u64,
    pub field_time: f32,
    pub agent_checkpoint: Option<String>,
    pub active_chunks: Vec<[i32; 2]>,
    pub nodes: Vec<CanonicalNode>,
}

impl WorldSnapshot {
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub fn content_hash(&self) -> Result<String, serde_json::Error> {
        Ok(blake3::hash(&self.canonical_bytes()?).to_hex().to_string())
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<String, String> {
        let bytes = self.canonical_bytes().map_err(|error| error.to_string())?;
        std::fs::write(path, &bytes).map_err(|error| error.to_string())?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())
    }
}

pub fn capture(
    world: &mut World,
    agent_checkpoint: Option<String>,
) -> Result<WorldSnapshot, String> {
    ensure_stable_ids(world);
    let seed = world.resource::<SimSeed>().0;
    let simulation_tick = world.resource::<SimClock>().step;
    let field_time = world.resource::<FieldSampler>().time;
    let active_chunks = world
        .resource::<crate::genesis::ChunkManager>()
        .active_chunks()
        .map(|(x, y)| [x, y])
        .collect();
    let stable_ids = {
        let mut stable_id_query = world.query::<(Entity, &StableId)>();
        stable_id_query
            .iter(world)
            .map(|(entity, stable_id)| (entity, stable_id.0))
            .collect::<HashMap<_, _>>()
    };
    let clamp_states = {
        let mut clamp_query = world.query::<(Entity, &ClampState)>();
        clamp_query
            .iter(world)
            .filter_map(|(entity, clamp)| {
                stable_ids.get(&entity).map(|stable_id| {
                    (
                        *stable_id,
                        (
                            clamp
                                .target
                                .and_then(|target| stable_ids.get(&target).copied()),
                            clamp.cooldown,
                        ),
                    )
                })
            })
            .collect::<HashMap<_, _>>()
    };
    let chunk_origins = {
        let mut chunk_query = world.query::<(Entity, &ChunkOrigin)>();
        chunk_query
            .iter(world)
            .filter_map(|(entity, origin)| {
                stable_ids
                    .get(&entity)
                    .map(|stable_id| (*stable_id, [origin.x, origin.y]))
            })
            .collect::<HashMap<_, _>>()
    };
    let dead_nodes = {
        let mut dead_query = world.query_filtered::<Entity, With<DeadNode>>();
        dead_query
            .iter(world)
            .filter_map(|entity| stable_ids.get(&entity).copied())
            .collect::<HashSet<_>>()
    };
    let mut query = world.query::<(
        &StableId,
        &Transform,
        Option<&LinearVelocity>,
        Option<&AngularVelocity>,
        &Physical,
        &Yield,
        &Conductivity,
        &Tags,
        Option<&Oscillator>,
        Option<&NodeRng>,
        Option<&EnergyLedger>,
        Option<&AgentMarker>,
        Option<&AgentTuning>,
        Option<&ReproductionState>,
        Option<&FabricateCooldown>,
    )>();
    let mut nodes = query
        .iter(world)
        .map(
            |(
                id,
                transform,
                velocity,
                angular,
                physical,
                yield_threshold,
                conductivity,
                tags,
                oscillator,
                rng,
                ledger,
                marker,
                tuning,
                reproduction,
                fabricate,
            )| {
                let (_, _, rotation) = transform.rotation.to_euler(EulerRot::XYZ);
                CanonicalNode {
                    stable_id: id.0,
                    position: [transform.translation.x, transform.translation.y],
                    rotation,
                    linear_velocity: velocity.map_or([0.0; 2], |value| [value.x, value.y]),
                    angular_velocity: angular.map_or(0.0, |value| value.0),
                    physical: [physical.mass, physical.structure, physical.energy],
                    yield_threshold: yield_threshold.0,
                    conductivity: conductivity.0,
                    tags: tags.0 .0,
                    oscillator: oscillator
                        .map(|value| [value.freq, value.phase, value.baseline_freq]),
                    node_rng: rng.map_or(0, |value| value.0),
                    chunk_origin: chunk_origins.get(&id.0).copied(),
                    dead: dead_nodes.contains(&id.0),
                    energy_ledger: ledger.map_or([0.0; 8], |value| {
                        [
                            value.harvested,
                            value.consumed_from_others,
                            value.vented,
                            value.dissipated,
                            value.spent_on_actuation,
                            value.invested_in_fabrication,
                            value.invested_in_offspring,
                            value.transferred_out,
                        ]
                    }),
                    born_step: ledger.map_or(0, |value| value.born_step),
                    agent: marker.zip(tuning).map(|(marker, tuning)| CanonicalAgent {
                        generation: marker.generation,
                        lineage: marker.lineage,
                        sensor_radius: tuning.sensor_radius,
                        max_neighbors: tuning.max_neighbors,
                        yield_threshold: tuning.yield_threshold,
                        oscillator_baseline: tuning.osc_baseline,
                        reproduction: reproduction.map_or([0.0; 2], |value| {
                            [value.surplus_seconds, value.cooldown_seconds]
                        }),
                        fabricate_cooldown: fabricate.map_or(0.0, |value| value.remaining),
                        clamp_target: clamp_states.get(&id.0).and_then(|(target, _)| *target),
                        clamp_cooldown: clamp_states
                            .get(&id.0)
                            .map_or(0.0, |(_, cooldown)| *cooldown),
                    }),
                }
            },
        )
        .collect::<Vec<_>>();
    nodes.sort_by_key(|node| node.stable_id);
    Ok(WorldSnapshot {
        schema_version: SNAPSHOT_SCHEMA_VERSION,
        seed,
        simulation_tick,
        field_time,
        agent_checkpoint,
        active_chunks,
        nodes,
    })
}

/// Restore a canonical physical world. Controllers are supplied separately by
/// stable ID because arbitrary agent implementations are outside ERTW state.
pub fn restore(
    snapshot: &WorldSnapshot,
    mut controllers: BTreeMap<u64, Box<dyn ertw_interface::Agent>>,
) -> Result<ErtwWorld, String> {
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION {
        return Err("unsupported snapshot schema".to_owned());
    }
    let mut simulation = ErtwWorld::new(snapshot.seed);
    let world = simulation.app().world_mut();
    world.resource_mut::<SimClock>().step = snapshot.simulation_tick;
    world
        .resource_mut::<FieldSampler>()
        .set_time(snapshot.field_time);
    world
        .resource_mut::<crate::genesis::ChunkManager>()
        .restore_active_chunks(
            snapshot
                .active_chunks
                .iter()
                .map(|coordinate| (coordinate[0], coordinate[1])),
        );

    let mut restored_entities = HashMap::new();
    for node in &snapshot.nodes {
        let physical = Physical {
            mass: node.physical[0],
            structure: node.physical[1],
            energy: node.physical[2],
        };
        let base = (
            StableId(node.stable_id),
            Transform::from_translation(Vec3::new(node.position[0], node.position[1], 0.0))
                .with_rotation(Quat::from_rotation_z(node.rotation)),
            RigidBody::Dynamic,
            avian2d::prelude::Collider::circle(0.5),
            Mass(physical.mass.max(0.05)),
            LinearVelocity(Vec2::from_array(node.linear_velocity)),
            AngularVelocity(node.angular_velocity),
            physical,
            Yield(node.yield_threshold),
            Conductivity(node.conductivity),
            Tags(crate::tags::CustomTags::from_bits(node.tags)),
            ImpulseAccum::default(),
            EnergyLedger {
                harvested: node.energy_ledger[0],
                consumed_from_others: node.energy_ledger[1],
                vented: node.energy_ledger[2],
                dissipated: node.energy_ledger[3],
                spent_on_actuation: node.energy_ledger[4],
                invested_in_fabrication: node.energy_ledger[5],
                invested_in_offspring: node.energy_ledger[6],
                transferred_out: node.energy_ledger[7],
                born_step: node.born_step,
            },
            NodeRng(node.node_rng),
        );
        let entity = world.spawn(base).id();
        restored_entities.insert(node.stable_id, entity);
        if let Some(values) = node.oscillator {
            world.entity_mut(entity).insert(Oscillator {
                freq: values[0],
                phase: values[1],
                baseline_freq: values[2],
            });
        }
        if let Some([x, y]) = node.chunk_origin {
            world.entity_mut(entity).insert(ChunkOrigin { x, y });
        }
        if node.dead {
            world.entity_mut(entity).insert(DeadNode);
        }
        if let Some(agent) = &node.agent {
            if let Some(controller) = controllers.remove(&node.stable_id) {
                let controller_id = world
                    .resource_mut::<crate::agents::WorldAgents>()
                    .register(controller);
                world.entity_mut(entity).insert((
                    AgentMarker {
                        generation: agent.generation,
                        lineage: agent.lineage,
                        controller: controller_id,
                    },
                    AgentTuning {
                        sensor_radius: agent.sensor_radius,
                        max_neighbors: agent.max_neighbors,
                        yield_threshold: agent.yield_threshold,
                        osc_baseline: agent.oscillator_baseline,
                    },
                    ClampState {
                        cooldown: agent.clamp_cooldown,
                        ..Default::default()
                    },
                    FabricateCooldown {
                        remaining: agent.fabricate_cooldown,
                    },
                    ReproductionState {
                        surplus_seconds: agent.reproduction[0],
                        cooldown_seconds: agent.reproduction[1],
                    },
                ));
            }
        }
        world
            .resource_mut::<StableIdAllocator>()
            .observe(node.stable_id);
    }
    world.flush();
    for node in &snapshot.nodes {
        let Some(agent) = &node.agent else { continue };
        let Some(target_id) = agent.clamp_target else {
            continue;
        };
        let Some(&owner) = restored_entities.get(&node.stable_id) else {
            continue;
        };
        let Some(&target) = restored_entities.get(&target_id) else {
            continue;
        };
        let joint = world
            .spawn((
                avian2d::prelude::FixedJoint::new(owner, target),
                avian2d::prelude::JointCollisionDisabled,
                ClampJoint { owner, target },
            ))
            .id();
        if let Some(mut clamp) = world.get_mut::<ClampState>(owner) {
            clamp.target = Some(target);
            clamp.joint = Some(joint);
        }
    }
    world.flush();
    Ok(simulation)
}

pub fn ensure_stable_ids(world: &mut World) {
    let highest_existing = {
        let mut existing = world.query::<&StableId>();
        existing.iter(world).map(|id| id.0).max().unwrap_or(0)
    };
    world
        .resource_mut::<StableIdAllocator>()
        .observe(highest_existing);
    let mut query = world.query_filtered::<Entity, (With<Physical>, Without<StableId>)>();
    let mut entities = query.iter(world).collect::<Vec<_>>();
    entities.sort_by_key(|entity| entity.to_bits());
    let mut assigned = HashMap::new();
    for entity in entities {
        let id = world.resource_mut::<StableIdAllocator>().allocate();
        assigned.insert(entity, id);
    }
    for (entity, id) in assigned {
        world.entity_mut(entity).insert(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ertw_interface::{ActionTensor, Agent, ObservationTensor};

    struct Passive;
    impl Agent for Passive {
        fn act(&mut self, _: &ObservationTensor) -> ActionTensor {
            ActionTensor::default()
        }
    }

    struct ClampHold;
    impl Agent for ClampHold {
        fn act(&mut self, _: &ObservationTensor) -> ActionTensor {
            ActionTensor {
                clamp: 1.0,
                osc_freq: 1.0,
                ..Default::default()
            }
        }
    }

    #[test]
    fn snapshot_hash_and_restore_are_canonical() {
        let mut simulation = ErtwWorld::new(77);
        simulation.spawn_agent(Box::new(Passive), Vec2::new(1.0, 2.0));
        simulation.step(3);
        let snapshot = capture(
            simulation.app().world_mut(),
            Some("agent-checkpoint-1".into()),
        )
        .unwrap();
        assert_eq!(
            snapshot.content_hash().unwrap(),
            snapshot.content_hash().unwrap()
        );
        let stable_id = snapshot
            .nodes
            .iter()
            .find(|node| node.agent.is_some())
            .unwrap()
            .stable_id;
        let mut restored = restore(
            &snapshot,
            BTreeMap::from([(stable_id, Box::new(Passive) as Box<dyn Agent>)]),
        )
        .unwrap();
        assert_eq!(
            restored.app.world().resource::<SimClock>().step,
            snapshot.simulation_tick
        );
        let recaptured = capture(
            restored.app().world_mut(),
            snapshot.agent_checkpoint.clone(),
        )
        .unwrap();
        assert_eq!(recaptured, snapshot);
        let path = std::env::temp_dir().join(format!(
            "ertw-snapshot-{}-{}.json",
            std::process::id(),
            snapshot.simulation_tick
        ));
        let saved_hash = snapshot.save(&path).unwrap();
        let loaded = WorldSnapshot::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(loaded, snapshot);
        assert_eq!(saved_hash, snapshot.content_hash().unwrap());
    }

    #[test]
    fn allocator_never_reuses_an_explicit_stable_id() {
        let mut simulation = ErtwWorld::new(78);
        let explicit = simulation.spawn_agent(Box::new(Passive), Vec2::ZERO);
        simulation
            .app()
            .world_mut()
            .entity_mut(explicit)
            .insert(StableId(1));
        simulation.spawn_agent(Box::new(Passive), Vec2::X);
        ensure_stable_ids(simulation.app().world_mut());
        let world = simulation.app().world_mut();
        let mut query = world.query::<&StableId>();
        let mut ids = query.iter(world).map(|id| id.0).collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn snapshot_restores_active_clamp_relationship() {
        let mut simulation = ErtwWorld::new(79);
        simulation.spawn_agent(Box::new(ClampHold), Vec2::new(-0.75, 0.0));
        simulation.spawn_agent(Box::new(Passive), Vec2::new(0.75, 0.0));
        simulation.step(1);
        let snapshot = capture(simulation.app().world_mut(), None).unwrap();
        assert!(snapshot.nodes.iter().any(|node| {
            node.agent
                .as_ref()
                .is_some_and(|agent| agent.clamp_target.is_some())
        }));
        let controllers = snapshot
            .nodes
            .iter()
            .filter(|node| node.agent.is_some())
            .map(|node| {
                (
                    node.stable_id,
                    Box::new(Passive) as Box<dyn ertw_interface::Agent>,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut restored = restore(&snapshot, controllers).unwrap();
        let recaptured = capture(restored.app().world_mut(), None).unwrap();
        assert_eq!(recaptured, snapshot);
    }
}
