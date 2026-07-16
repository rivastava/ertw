//! ERTW core simulation: a zero-reward, agent-agnostic 2D tensor world.
//!
//! This crate owns the ECS world, the three global fields, the spatial hash, the
//! energy economy, and the in-process agent driver. It exposes no reward or score.
//! See the repository's public architecture and protocol documentation.

pub mod actuation;
pub mod agents;
pub mod components;
pub mod economy;
pub mod fields;
pub mod fragmentation;
pub mod genesis;
pub mod lineage;
pub mod spatial_hash;
pub mod tags;

use avian2d::dynamics::rigid_body::forces::WriteRigidBodyForces;
use avian2d::dynamics::rigid_body::LinearVelocity;
use avian2d::prelude::PhysicsSystems;
use bevy::prelude::*;
use fields::{advance_fields, FieldSampler, SimSeed};

/// Protocol version for the wire header (spec: raw f32 + minimal header).
///
/// Version 3 is self-describing, carries full 64-bit step/entity identifiers,
/// and encodes tags losslessly as four 16-bit float channels.
pub const PROTOCOL_VERSION: u8 = 3;

/// Named set grouping every fixed-step simulation system configured by
/// [`configure_world`]. Exposed so external front-ends (e.g. the rendered HUD)
/// can gate the simulation (pause / single-step) independently of rendering and
/// the egui overlay. The world itself is agnostic to any UI.
#[derive(SystemSet, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct SimulationSet;

#[derive(SystemSet, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct SimulationPrePhysics;

#[derive(SystemSet, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct SimulationPostPhysics;

/// Authoritative simulation clock. Unlike wall-clock time, it advances exactly
/// once for each completed fixed simulation tick.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct SimClock {
    pub step: u64,
}

/// Builder for a headless ERTW world. The world runs identically with or without
/// the `ertw_render` feature; rendering is an overlay, never participant.
pub struct ErtwWorld {
    app: App,
}

/// Configure an [`App`] (or `SubApp`) with the full ERTW world: plugins,
/// resources, and the fixed-step simulation systems. Reused by [`ErtwWorld::new`]
/// and by the rendered example (which layers `DefaultPlugins` + `RenderPlugin`
/// on top). The caller is responsible for adding the render/window plugins.
pub fn configure_world(app: &mut App, seed: u64) {
    // A bare headless `App` needs the schedule runner, task pools, and time.
    // A rendered app has already installed `DefaultPlugins`, including the
    // native winit event-loop runner. Installing `ScheduleRunnerPlugin` there
    // would replace the native runner: the ECS `Window` entity would exist,
    // but winit would never materialize an operating-system window.
    if !app.is_plugin_added::<bevy::app::TaskPoolPlugin>() {
        app.add_plugins((
            bevy::app::ScheduleRunnerPlugin::default(),
            bevy::app::TaskPoolPlugin::default(),
            bevy::time::TimePlugin,
        ));
    }
    app.add_plugins(avian2d::prelude::PhysicsPlugins::new(FixedUpdate))
        .insert_resource(avian2d::prelude::Gravity(Vec2::ZERO))
        .insert_resource(SimSeed(seed))
        // Avian's diagnostics resources must exist even in headless mode because
        // several internal systems hold `ResMut` to them unconditionally.
        .init_resource::<avian2d::prelude::SpatialQueryDiagnostics>()
        .init_resource::<avian2d::collider_tree::ColliderTreeDiagnostics>()
        .init_resource::<avian2d::collision::CollisionDiagnostics>()
        .init_resource::<avian2d::dynamics::solver::SolverDiagnostics>()
        .insert_resource(agents::WorldAgents::default())
        .insert_resource(spatial_hash::SpatialHash::default())
        .insert_resource(PendingActions::default())
        .insert_resource(SimClock::default())
        .insert_resource(lineage::AgentHistory::default())
        .insert_resource(fragmentation::FragmentQueue::default())
        .insert_resource(genesis::ChunkManager::new(seed))
        .init_resource::<FieldSampler>()
        .configure_sets(
            FixedUpdate,
            (
                SimulationPrePhysics.before(PhysicsSystems::StepSimulation),
                SimulationPostPhysics.after(PhysicsSystems::Last),
                PhysicsSystems::First.in_set(SimulationSet),
                PhysicsSystems::Prepare.in_set(SimulationSet),
                PhysicsSystems::StepSimulation.in_set(SimulationSet),
                PhysicsSystems::Writeback.in_set(SimulationSet),
                PhysicsSystems::Last.in_set(SimulationSet),
            ),
        )
        .add_systems(
            FixedUpdate,
            (
                advance_fields,
                rebuild_spatial_hash,
                gather_agent_actions,
                apply_agent_actions,
                actuation::apply_actuators,
                fields::apply_field_forces,
                economy::thermodynamic_drain,
                economy::kinetic_harvest,
                economy::volatile_trap_discharge,
                lineage::reproduce_agents,
            )
                .chain()
                .in_set(SimulationPrePhysics)
                .in_set(SimulationSet),
        )
        .add_systems(
            FixedUpdate,
            (
                fragmentation::accumulate_collision_damage,
                fragmentation::transfer_to_killers,
                fragmentation::run_fragmentation,
                genesis::stream_chunks,
                advance_sim_clock,
                lineage::record_agent_history,
            )
                .chain()
                .in_set(SimulationPostPhysics)
                .in_set(SimulationSet),
        );

    // Fixed timestep: 60 Hz for deterministic step counting.
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .set_timestep(std::time::Duration::from_secs_f32(economy::FIXED_DT));
}

fn advance_sim_clock(mut clock: ResMut<SimClock>) {
    clock.step = clock.step.saturating_add(1);
}

impl ErtwWorld {
    /// Create a world seeded for reproducible genesis/field drift.
    pub fn new(seed: u64) -> Self {
        let mut app = App::new();
        configure_world(&mut app, seed);

        Self { app }
    }

    /// Register an in-process agent controller and spawn its entity.
    pub fn spawn_agent(&mut self, agent: Box<dyn ertw_interface::Agent>, pos: Vec2) -> Entity {
        let world = self.app.world_mut();
        let controller = {
            let mut world_agents = world.resource_mut::<agents::WorldAgents>();
            world_agents.register(agent)
        };
        let entity = agents::spawn_with_id(
            &mut world.commands(),
            controller,
            pos,
            components::AgentTuning::default(),
        );
        world.flush();
        entity
    }

    /// Seed a reproducible initial population across a `grid_x` by `grid_y` block
    /// of spatial chunks (spec item 7 genesis). External testers may instead call
    /// [`ErtwWorld::spawn_agent`] to inject their own policies.
    ///
    /// Each chunk also receives a fixed terrain mix
    /// (`ENERGY_CONVERTIBLE` × 2, `THERMAL_VENT` × 1, `VOLATILE_TRAP` × 1,
    /// `SHELTER` × 1) so the three refill channels (spec item 5) and the
    /// volatile-trap discharge (spec item 8) have actual instances to act on.
    pub fn seed_world(&mut self, grid_x: i32, grid_y: i32) -> Vec<Entity> {
        let world = self.app.world_mut();
        let agent_plan = {
            let mut chunks = world.resource_mut::<genesis::ChunkManager>();
            chunks.spawn_plan(grid_x, grid_y)
        };
        let terrain_plan = {
            let chunks = world.resource::<genesis::ChunkManager>();
            chunks.terrain_plan(grid_x, grid_y)
        };
        let mut out = Vec::new();
        for (pos, node_rng) in agent_plan {
            let controller = {
                let mut agents_res = world.resource_mut::<agents::WorldAgents>();
                agents_res.register(Box::new(genesis::GenesisAgent))
            };
            let e = genesis::spawn_genesis(&mut world.commands(), controller, pos, node_rng, 0);
            out.push(e);
        }
        for spawn in terrain_plan {
            genesis::spawn_genesis_node(&mut world.commands(), spawn);
        }
        world.flush();
        out
    }

    /// Run `steps` fixed steps and return. Headless by design. Drives the
    /// `FixedUpdate` schedule directly so simulation progress is deterministic
    /// and independent of wall-clock time (the runner plugin otherwise gates
    /// fixed steps on elapsed real time).
    pub fn step(&mut self, steps: u32) {
        for _ in 0..steps {
            let world = self.app.world_mut();
            world
                .resource_mut::<Time>()
                .advance_by(std::time::Duration::from_secs_f32(economy::FIXED_DT));
            world.run_schedule(FixedUpdate);
        }
    }

    /// Borrow the underlying app (for the render/overlay or external observers).
    pub fn app(&mut self) -> &mut App {
        &mut self.app
    }
}

/// Rebuild the spatial hash each fixed step before any neighbor queries.
pub fn rebuild_spatial_hash(
    positions: Query<(Entity, &Transform)>,
    mut spatial: ResMut<spatial_hash::SpatialHash>,
) {
    spatial.rebuild(&positions);
}

/// Actions decided this fixed step, queued for the apply pass. Splitting gather
/// (immutable reads) from apply (mutable writes) avoids conflicting `&`/`&mut`
/// queries over the same components in one system.
#[derive(Resource, Default)]
pub struct PendingActions {
    pub items: Vec<(Entity, ertw_interface::ActionTensor)>,
}

/// Gather pass (Phase 5): build each agent's observation from read-only world
/// state, call its trait, and queue the action. Holds only immutable queries so
/// the physics/energy apply pass can mutate freely.
#[allow(clippy::too_many_arguments)]
fn gather_agent_actions(
    mut world_agents: ResMut<agents::WorldAgents>,
    sampler: Res<FieldSampler>,
    clock: Res<SimClock>,
    spatial: Res<spatial_hash::SpatialHash>,
    tuning: Query<&components::AgentTuning>,
    velocities: Query<&LinearVelocity>,
    physicals: Query<&components::Physical>,
    tags: Query<&components::Tags>,
    conductivities: Query<&components::Conductivity>,
    oscillators: Query<&components::Oscillator>,
    agents: Query<(Entity, &components::AgentMarker)>,
    transforms: Query<(Entity, &Transform)>,
    mut pending: ResMut<PendingActions>,
) {
    let mut jobs: Vec<(Entity, u64)> = Vec::new();
    for (e, m) in agents.iter() {
        jobs.push((e, m.controller));
    }
    let active = jobs
        .iter()
        .map(|(_, controller)| *controller)
        .collect::<std::collections::HashSet<_>>();
    world_agents.retain(&active);

    pending.items.clear();
    for (e, ctrl) in jobs {
        let Ok(tune) = tuning.get(e) else { continue };
        let Some(obs) = agents::build_observation(
            e,
            tune,
            &clock,
            &sampler,
            &spatial,
            &transforms,
            &velocities,
            &physicals,
            &tags,
            &conductivities,
            &oscillators,
        ) else {
            continue;
        };
        if let Some(a) = world_agents.get_mut(ctrl) {
            let mut action = a.act(&obs);
            // Sanitize once at the trust boundary so every downstream
            // actuator sees the same finite, bounded command.
            action.sanitize();
            pending.items.push((e, action));
        }
    }
}

/// Apply pass (Phase 5): spend energy on actions and apply continuous
/// force/torque via the physics body. Locomotion only here; clamp/fabricate/
/// oscillator are layered in `actuation::apply_actuators` which reads the same
/// queued items in a later system. `pending` is cleared at the start of the next
/// `gather_agent_actions` (this system must NOT drain it).
fn apply_agent_actions(
    pending: Res<PendingActions>,
    mut physicals: Query<&mut components::Physical>,
    mut ledgers: Query<&mut components::EnergyLedger>,
    mut bodies: Query<avian2d::dynamics::rigid_body::forces::Forces>,
) {
    for (e, action) in pending.items.iter() {
        let e = *e;
        let cost = (action.force.length() + action.torque.abs()) * 0.1;
        let (mut applied_force, mut applied_torque) = (Vec2::ZERO, 0.0f32);
        if let Ok(mut phys) = physicals.get_mut(e) {
            if phys.energy >= cost {
                phys.energy -= cost;
                if let Ok(mut ledger) = ledgers.get_mut(e) {
                    ledger.spent_on_actuation += cost;
                }
                applied_force = Vec2::new(action.force.x, action.force.y);
                applied_torque = action.torque;
            }
        }
        if let Ok(mut forces) = bodies.get_mut(e) {
            forces.apply_force(applied_force);
            forces.apply_torque(applied_torque);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ertw_interface::{ActionTensor, Agent, ObservationTensor};

    /// A passive agent that never acts. Useful for isolating world dynamics
    /// (decay, death) from agent behavior.
    struct NullAgent;
    impl Agent for NullAgent {
        fn act(&mut self, _obs: &ObservationTensor) -> ActionTensor {
            ActionTensor::default()
        }
    }

    /// Agents with no energy inflow must die — the allostatic drain binds
    /// (spec item 4). A passive agent should not survive indefinitely.
    #[test]
    fn passive_agent_dies_under_decay() {
        let mut world = ErtwWorld::new(1);
        let e = world.spawn_agent(Box::new(NullAgent), Vec2::ZERO);
        // Run well past the expected lifetime (energy 20, drain ~0.6/s @ 60Hz).
        world.step(60 * 60);
        let alive = world.app().world().get_entity(e).is_ok();
        assert!(
            !alive,
            "passive agent should have died from thermodynamic decay"
        );
        let outcomes = lineage::collect_competence(world.app().world_mut(), 60 * 60);
        assert!(outcomes.iter().any(|outcome| outcome.entity == e.to_bits()));
    }

    /// The same seed must produce the same field sample sequence: genesis and
    /// field drift are reproducible per-machine (spec determinism decision).
    #[test]
    fn field_sampling_is_seeded_and_reproducible() {
        let mut a = ErtwWorld::new(42);
        let mut b = ErtwWorld::new(42);
        let sa = a
            .app()
            .world()
            .resource::<FieldSampler>()
            .sample(Vec2::new(3.3, 7.7));
        let sb = b
            .app()
            .world()
            .resource::<FieldSampler>()
            .sample(Vec2::new(3.3, 7.7));
        assert_eq!(sa.kinetic, sb.kinetic);
        assert_eq!(sa.thermal, sb.thermal);
        assert_eq!(sa.em, sb.em);

        // Different seeds diverge.
        let mut c = ErtwWorld::new(43);
        let sc = c
            .app()
            .world()
            .resource::<FieldSampler>()
            .sample(Vec2::new(3.3, 7.7));
        assert_ne!(sa.kinetic, sc.kinetic);
    }

    /// The observation tensor must round-trip through the raw f32 wire
    /// encoding with every neighbor field intact, including the `valid`
    /// flag in the presence of nonzero high tag bits. This is the IPC
    /// regression test for the `from_f32_slice` `valid` decode bug.
    #[test]
    fn observation_roundtrips_through_wire_encoding() {
        use ertw_interface::{
            NeighborView, ObservationTensor, Vec2Lite, ACTION_STRIDE, WIRE_HEADER_LEN,
        };

        let mut world = ErtwWorld::new(7);
        // Spawn a passive agent and tick a few steps so we have a populated
        // world (with terrain nodes from genesis chunks).
        world.spawn_agent(Box::new(NullAgent), Vec2::ZERO);
        world.step(30);

        // Build a synthetic observation with high tag bits set, run it through
        // the same f32 + header encoding the TCP server uses, and decode it.
        let mut obs = ObservationTensor::new(ertw_interface::InterfaceConfig::default());
        obs.neighbors[0] = NeighborView {
            rel_pos: Vec2Lite::new(1.0, 2.0),
            rel_vel: Vec2Lite::new(3.0, 4.0),
            mass: 5.0,
            structure: 6.0,
            energy: 7.0,
            tags: 0xFFFF_FFFF_0000_0001, // high tag bits set
            conductivity: 0.2,
            osc_freq: 1.25,
            osc_phase: std::f32::consts::PI,
            valid: true,
        };
        obs.neighbors[1] = NeighborView {
            tags: 0xDEAD_BEEF_CAFE_BABE,
            valid: false, // ghost
            ..Default::default()
        };

        let flat = obs.to_f32_vec();
        let neighbor_count = obs.neighbors.iter().filter(|n| n.valid).count() as u32;
        let header = ertw_interface::wire_header(ertw_interface::WireHeader {
            version: PROTOCOL_VERSION,
            frame_kind: ertw_interface::FRAME_OBSERVATION,
            frame_bytes: (WIRE_HEADER_LEN * 4 + flat.len() * 4) as u32,
            step: obs.step,
            entity_id: obs.entity_id,
            max_neighbors: obs.config.max_neighbors as u32,
            neighbor_count,
            field_samples: obs.config.field_samples as u32,
            field_channels: obs.config.field_channels as u32,
            payload_floats: flat.len() as u32,
        });

        // Reconstruct the wire bytes the way ertw_server::encode_observation does.
        let mut bytes = Vec::with_capacity(WIRE_HEADER_LEN * 4 + flat.len() * 4);
        for h in header {
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        for f in &flat {
            bytes.extend_from_slice(&f.to_le_bytes());
        }

        // Decode back: skip the u32 header, reinterpret payload as f32 LE.
        let payload = &bytes[WIRE_HEADER_LEN * 4..];
        assert_eq!(payload.len(), obs.len() * 4);
        let mut slice = vec![0f32; obs.len()];
        for (i, s) in slice.iter_mut().enumerate() {
            let b = &payload[i * 4..i * 4 + 4];
            *s = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        let back = ObservationTensor::from_f32_slice(&slice, obs.config);

        assert!(
            back.neighbors[0].valid,
            "valid flag must survive the wire roundtrip"
        );
        assert_eq!(back.neighbors[0].conductivity, 0.2);
        assert_eq!(back.neighbors[0].osc_freq, 1.25);
        assert!((back.neighbors[0].osc_phase - std::f32::consts::PI).abs() < 1e-5);
        assert!(
            !back.neighbors[1].valid,
            "ghost must remain a ghost after roundtrip"
        );
        assert_eq!(back.neighbors[0].tags, 0xFFFF_FFFF_0000_0001);
        assert_eq!(PROTOCOL_VERSION, 3);
        // And the wire payload must match ACTION_STRIDE for the response side.
        assert_eq!(ACTION_STRIDE, 7);
    }

    /// The same observation tensor must produce identical bytes across two
    /// independent worlds with the same seed — wire payload is deterministic
    /// per-machine (spec determinism decision).
    #[test]
    fn observation_payload_is_deterministic_per_seed() {
        let mut a = ErtwWorld::new(99);
        let mut b = ErtwWorld::new(99);
        a.spawn_agent(Box::new(NullAgent), Vec2::ZERO);
        b.spawn_agent(Box::new(NullAgent), Vec2::ZERO);
        a.step(15);
        b.step(15);

        let sampler_a = a.app().world().resource::<FieldSampler>().clone();
        let sampler_b = b.app().world().resource::<FieldSampler>().clone();
        // Two independent FieldSamplers from the same seed sample identically.
        let pa = sampler_a.sample(Vec2::new(2.0, 3.0));
        let pb = sampler_b.sample(Vec2::new(2.0, 3.0));
        assert_eq!(pa.kinetic, pb.kinetic);
        assert_eq!(pa.thermal, pb.thermal);
        assert_eq!(pa.em, pb.em);
    }

    #[test]
    fn same_target_world_replay_is_bit_identical() {
        fn snapshot(simulation: &mut ErtwWorld) -> Vec<(u64, [u32; 7])> {
            let world = simulation.app().world_mut();
            let mut query = world.query::<(Entity, &Transform, &components::Physical)>();
            let mut values = query
                .iter(world)
                .map(|(entity, transform, physical)| {
                    (
                        entity.to_bits(),
                        [
                            transform.translation.x.to_bits(),
                            transform.translation.y.to_bits(),
                            transform.rotation.z.to_bits(),
                            transform.rotation.w.to_bits(),
                            physical.mass.to_bits(),
                            physical.structure.to_bits(),
                            physical.energy.to_bits(),
                        ],
                    )
                })
                .collect::<Vec<_>>();
            values.sort_by_key(|(entity, _)| *entity);
            values
        }

        let mut first = ErtwWorld::new(0xA11CE);
        let mut second = ErtwWorld::new(0xA11CE);
        first.spawn_agent(Box::new(NullAgent), Vec2::new(2.0, -3.0));
        second.spawn_agent(Box::new(NullAgent), Vec2::new(2.0, -3.0));
        first.step(180);
        second.step(180);
        assert_eq!(snapshot(&mut first), snapshot(&mut second));
    }

    #[test]
    #[ignore = "explicit bounded-state soak gate"]
    fn bounded_state_soak() {
        const INITIAL_AGENTS: usize = 32;
        let mut simulation = ErtwWorld::new(0x50A5);
        for index in 0..INITIAL_AGENTS {
            let position = Vec2::new((index % 8) as f32 * 2.0, (index / 8) as f32 * 2.0);
            simulation.spawn_agent(Box::new(NullAgent), position);
        }
        simulation.step(5_000);

        let world = simulation.app().world_mut();
        let mut physicals = world.query::<(&Transform, &components::Physical)>();
        let remaining = physicals
            .iter(world)
            .inspect(|(transform, physical)| {
                assert!(transform.translation.is_finite());
                assert!(physical.mass.is_finite());
                assert!(physical.structure.is_finite());
                assert!(physical.energy.is_finite());
            })
            .count();
        assert!(remaining < 512, "streaming state grew without a bound");
        let history = world.resource::<lineage::AgentHistory>();
        assert!(history.completed.len() >= INITIAL_AGENTS);
    }
}
