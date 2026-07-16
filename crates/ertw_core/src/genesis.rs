//! Genesis & chunk streaming (spec item 7).
//!
//! The world is unbounded in principle (only disk, not memory, bounds it), but
//! entity management is organized into fixed-size spatial chunks so genesis and
//! streaming spawning stay local and reproducible. This module provides a
//! [`ChunkManager`] that spawns a reproducible initial population distributed
//! across chunks, and tops up population in under-filled chunks over time.
//!
//! No reward is involved; "genesis" only places nodes. Lineage/depth is assigned
//! at spawn (see [`crate::components::AgentMarker`]).

use crate::components::{
    AgentMarker, AgentTuning, ClampState, FabricateCooldown, ImpulseAccum, NodeRng, Oscillator,
    Physical, Tags,
};
use crate::tags::CustomTags;
use bevy::prelude::*;
use ertw_interface::{ActionTensor, Agent, ObservationTensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeSet;

/// Side length of one spatial chunk in world units.
pub const CHUNK_SIZE: f32 = 32.0;
/// Agents to maintain per chunk before streaming tops up.
pub const AGENTS_PER_CHUNK: usize = 4;

/// Minimal placeholder agent used for genesis-seeded population. External testers
/// replace these with real policies; the world only needs *an* agent object to
/// route observations. It inherits the no-op (passive) behavior.
pub struct GenesisAgent;

impl Agent for GenesisAgent {
    fn act(&mut self, _obs: &ObservationTensor) -> ActionTensor {
        ActionTensor::default()
    }

    fn spawn_child(&mut self, _seed: u64) -> Option<Box<dyn Agent>> {
        Some(Box::new(Self))
    }
}

/// Tracks which chunks have been seeded and the current global step for streaming.
#[derive(Resource)]
pub struct ChunkManager {
    active: BTreeSet<(i32, i32)>,
    seed: u64,
}

impl Default for ChunkManager {
    fn default() -> Self {
        Self {
            active: BTreeSet::new(),
            seed: 1,
        }
    }
}

impl ChunkManager {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    /// Compute the deterministic spawn plan for an initial population across a
    /// `grid_x` by `grid_y` block of chunks, placing `AGENTS_PER_CHUNK` agents in
    /// each. Returns `(position, per-node rng seed)` pairs. Pure read of `self`.
    pub fn spawn_plan(&mut self, grid_x: i32, grid_y: i32) -> Vec<(Vec2, u64)> {
        let mut out = Vec::new();
        let mut r = StdRng::seed_from_u64(self.seed);
        for cx in 0..grid_x {
            for cy in 0..grid_y {
                self.active.insert((cx, cy));
                let base_x = cx as f32 * CHUNK_SIZE;
                let base_y = cy as f32 * CHUNK_SIZE;
                for i in 0..AGENTS_PER_CHUNK {
                    let pos = Vec2::new(
                        base_x + r.gen_range(1.0..CHUNK_SIZE - 1.0),
                        base_y + r.gen_range(1.0..CHUNK_SIZE - 1.0),
                    );
                    let node_rng = self.seed
                        ^ (i as u64).wrapping_mul(0x9E3779B1)
                        ^ ((cx as u64) << 32)
                        ^ (cy as u64);
                    out.push((pos, node_rng));
                }
            }
        }
        out
    }

    /// Compute the deterministic terrain plan for an initial grid of chunks.
    /// Each chunk receives a fixed mix of inert terrain nodes that supply the
    /// three refill channels the brief requires (spec items 5, 8):
    ///
    /// - `ENERGY_CONVERTIBLE` × 2 — drained by agents in sustained relative
    ///   motion (kinetic harvesting).
    /// - `THERMAL_VENT` × 1 — passively converts ambient heat into reserve
    ///   energy at structural-stress risk.
    /// - `VOLATILE_TRAP` × 1 — releases stored energy to nearby agents when
    ///   the EM field spikes.
    /// - `SHELTER` × 1 — low-conductivity inert structure that reduces the
    ///   thermal drain for any agent on top of it.
    pub fn terrain_plan(&self, grid_x: i32, grid_y: i32) -> Vec<TerrainSpawn> {
        let mut out = Vec::new();
        for cx in 0..grid_x {
            for cy in 0..grid_y {
                out.extend(self.terrain_plan_for_chunk(cx, cy));
            }
        }
        out
    }

    pub fn terrain_plan_for_chunk(&self, cx: i32, cy: i32) -> Vec<TerrainSpawn> {
        let base = Vec2::new(cx as f32 * CHUNK_SIZE, cy as f32 * CHUNK_SIZE);
        let chunk_seed = self.seed ^ ((cx as u64) << 32) ^ cy as u64;
        let mut rng = StdRng::seed_from_u64(chunk_seed ^ 0xDEAD_BEEF);
        let entries = [
            (Vec2::new(0.25, 0.25), TerrainKind::EnergyConvertible),
            (Vec2::new(0.75, 0.25), TerrainKind::EnergyConvertible),
            (Vec2::new(0.25, 0.75), TerrainKind::ThermalVent),
            (Vec2::new(0.75, 0.75), TerrainKind::VolatileTrap),
            (Vec2::new(0.5, 0.5), TerrainKind::Shelter),
        ];
        entries
            .into_iter()
            .enumerate()
            .map(|(index, (fraction, kind))| {
                let (mass, structure, energy, conductivity) = match kind {
                    TerrainKind::EnergyConvertible => (1.0, 8.0, 6.0, 0.6),
                    TerrainKind::ThermalVent => (2.0, 10.0, 4.0, 0.6),
                    TerrainKind::VolatileTrap => (1.5, 8.0, 12.0, 0.6),
                    TerrainKind::Shelter => (1.0, 14.0, 2.0, 0.2),
                };
                TerrainSpawn {
                    pos: base
                        + fraction * CHUNK_SIZE
                        + Vec2::new(rng.gen_range(-0.5..0.5), rng.gen_range(-0.5..0.5)),
                    node_rng: chunk_seed ^ (index as u64).wrapping_mul(0x9E37_79B1),
                    kind,
                    mass,
                    structure,
                    energy,
                    conductivity,
                }
            })
            .collect()
    }
}

/// Keeps a deterministic one-chunk halo active around every live agent.
/// Inactive non-agent nodes are discarded and recreate from the seed when the
/// chunk becomes active again.
pub fn stream_chunks(
    mut commands: Commands,
    clock: Res<crate::SimClock>,
    mut chunks: ResMut<ChunkManager>,
    agents: Query<&Transform, With<AgentMarker>>,
    nodes: Query<(Entity, &Transform, Has<AgentMarker>), With<Physical>>,
) {
    if !clock.step.is_multiple_of(120) {
        return;
    }
    let mut desired = BTreeSet::new();
    for transform in agents.iter() {
        let position = transform.translation.truncate();
        let center = (
            (position.x / CHUNK_SIZE).floor() as i32,
            (position.y / CHUNK_SIZE).floor() as i32,
        );
        for dx in -1..=1 {
            for dy in -1..=1 {
                desired.insert((center.0 + dx, center.1 + dy));
            }
        }
    }

    for &(cx, cy) in desired.difference(&chunks.active) {
        for spawn in chunks.terrain_plan_for_chunk(cx, cy) {
            spawn_genesis_node(&mut commands, spawn);
        }
    }
    let inactive = chunks
        .active
        .difference(&desired)
        .copied()
        .collect::<BTreeSet<_>>();
    for (entity, transform, is_agent) in nodes.iter() {
        if is_agent {
            continue;
        }
        let position = transform.translation.truncate();
        let coordinate = (
            (position.x / CHUNK_SIZE).floor() as i32,
            (position.y / CHUNK_SIZE).floor() as i32,
        );
        if inactive.contains(&coordinate) {
            commands.entity(entity).despawn();
        }
    }
    chunks.active = desired;
}

/// Spawn one genesis-seeded agent at `pos` with a deterministic per-node rng
/// seed, given an already-registered `controller` id.
pub fn spawn_genesis(
    commands: &mut Commands,
    controller: u64,
    pos: Vec2,
    node_rng: u64,
    born_step: u32,
) -> Entity {
    let tuning = AgentTuning::default();
    commands
        .spawn(crate::components::AgentBundle {
            transform: Transform::from_translation(pos.extend(0.0)),
            rigid_body: avian2d::prelude::RigidBody::Dynamic,
            collider: avian2d::prelude::Collider::circle(0.5),
            mass: avian2d::prelude::Mass(tuning.yield_threshold.max(0.1)),
            physical: Physical {
                mass: tuning.yield_threshold.max(0.1),
                structure: tuning.yield_threshold,
                energy: 20.0,
            },
            yield_thresh: crate::components::Yield(tuning.yield_threshold),
            conductivity: crate::components::Conductivity(0.6),
            tags: Tags(CustomTags::from_bits(
                CustomTags::AGENT | CustomTags::CLAMP_CAPABLE | CustomTags::OSCILLATOR,
            )),
            oscillator: Oscillator {
                freq: tuning.osc_baseline,
                phase: 0.0,
                baseline_freq: tuning.osc_baseline,
            },
            impulse: ImpulseAccum::default(),
            ledger: crate::components::EnergyLedger {
                born_step,
                ..Default::default()
            },
            marker: AgentMarker {
                generation: 0,
                lineage: controller ^ 0xABCD,
                controller,
            },
            tuning,
            clamp: ClampState::default(),
            fabricate: FabricateCooldown::default(),
            reproduction: crate::components::ReproductionState::default(),
            node_rng: NodeRng(node_rng),
        })
        .id()
}

/// Kind of inert terrain node spawned by [`ChunkManager::terrain_plan`] and
/// [`spawn_genesis_node`]. The tag bits on each variant match the constants in
/// [`crate::tags::CustomTags`] so the world's existing economy/effect systems
/// react to them with no additional wiring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainKind {
    EnergyConvertible,
    ThermalVent,
    VolatileTrap,
    Shelter,
}

impl TerrainKind {
    pub fn tag_bits(self) -> u64 {
        match self {
            TerrainKind::EnergyConvertible => CustomTags::ENERGY_CONVERTIBLE,
            TerrainKind::ThermalVent => CustomTags::THERMAL_VENT,
            TerrainKind::VolatileTrap => CustomTags::VOLATILE_TRAP,
            TerrainKind::Shelter => CustomTags::SHELTER,
        }
    }
}

/// Pre-computed inert terrain spawn. Output of [`ChunkManager::terrain_plan`]
/// and input to [`spawn_genesis_node`].
#[derive(Clone, Copy, Debug)]
pub struct TerrainSpawn {
    pub pos: Vec2,
    pub node_rng: u64,
    pub kind: TerrainKind,
    pub mass: f32,
    pub structure: f32,
    pub energy: f32,
    pub conductivity: f32,
}

/// Spawn one inert terrain node from a [`TerrainSpawn`] plan entry. Inert nodes
/// carry the shared 12 components (no AgentMarker, no Tuning, no Clamp, no
/// FabricateCooldown) — they participate in physics, fields, and the economy
/// systems but are not agents.
pub fn spawn_genesis_node(commands: &mut Commands, spawn: TerrainSpawn) -> Entity {
    commands
        .spawn((
            Transform::from_translation(spawn.pos.extend(0.0)),
            avian2d::prelude::RigidBody::Dynamic,
            avian2d::prelude::Collider::circle(0.5),
            avian2d::prelude::Mass(spawn.mass.max(0.05)),
            Physical {
                mass: spawn.mass,
                structure: spawn.structure,
                energy: spawn.energy,
            },
            crate::components::Yield(spawn.structure),
            crate::components::Conductivity(spawn.conductivity),
            Tags(CustomTags::from_bits(spawn.kind.tag_bits())),
            ImpulseAccum::default(),
            crate::components::EnergyLedger::default(),
            NodeRng(spawn.node_rng),
            Oscillator::default(),
        ))
        .id()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_terrain_is_order_independent_and_reproducible() {
        let manager = ChunkManager::new(1234);
        let first = manager.terrain_plan_for_chunk(-3, 7);
        let _other = manager.terrain_plan_for_chunk(10, -2);
        let second = manager.terrain_plan_for_chunk(-3, 7);
        assert_eq!(first.len(), second.len());
        for (left, right) in first.iter().zip(second.iter()) {
            assert_eq!(left.pos, right.pos);
            assert_eq!(left.node_rng, right.node_rng);
            assert_eq!(left.kind, right.kind);
            assert_eq!(left.mass, right.mass);
            assert_eq!(left.energy, right.energy);
        }
    }

    #[test]
    fn inactive_chunks_unload_and_recreate_around_agent() {
        let mut simulation = crate::ErtwWorld::new(222);
        let agent = simulation.spawn_agent(Box::new(GenesisAgent), Vec2::ZERO);
        simulation.app().world_mut().flush();
        simulation.step(1);

        simulation
            .app()
            .world_mut()
            .get_mut::<Transform>(agent)
            .expect("agent transform")
            .translation
            .x = CHUNK_SIZE * 4.0;
        simulation
            .app()
            .world_mut()
            .resource_mut::<crate::SimClock>()
            .step = 120;
        simulation.step(1);

        let world = simulation.app().world_mut();
        let mut nodes = world.query_filtered::<(&Transform, Has<AgentMarker>), With<Physical>>();
        let inert_chunks = nodes
            .iter(world)
            .filter(|(_, is_agent)| !*is_agent)
            .map(|(transform, _)| (transform.translation.x / CHUNK_SIZE).floor() as i32)
            .collect::<Vec<_>>();
        assert_eq!(inert_chunks.len(), 45);
        assert!(inert_chunks.iter().all(|cx| (3..=5).contains(cx)));
    }
}
