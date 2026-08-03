use avian2d::collision::CollisionDiagnostics;
use avian2d::dynamics::solver::SolverDiagnostics;
use ertw_core::components::{AgentMarker, Physical};
use ertw_core::genesis::{collision_free_benchmark_position, ChunkManager};
use ertw_core::ErtwWorld;
use ertw_interface::{ActionTensor, Agent, ObservationTensor};
use serde::Serialize;
use std::error::Error;
use std::path::Path;
use std::time::Instant;

struct Passive;

impl Agent for Passive {
    fn act(&mut self, _observation: &ObservationTensor) -> ActionTensor {
        ActionTensor::default()
    }

    fn spawn_child(&mut self, _seed: u64) -> Option<Box<dyn Agent>> {
        Some(Box::new(Self))
    }
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum WorldMode {
    Streamed,
    AgentsOnly,
}

impl WorldMode {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        match std::env::var("ERTW_BENCH_WORLD")
            .unwrap_or_else(|_| "streamed".to_owned())
            .as_str()
        {
            "streamed" => Ok(Self::Streamed),
            "agents_only" => Ok(Self::AgentsOnly),
            value => Err(format!(
                "ERTW_BENCH_WORLD must be `streamed` or `agents_only`, got `{value}`"
            )
            .into()),
        }
    }
}

#[derive(Serialize)]
struct BenchmarkResult {
    agents_requested: usize,
    steps: u32,
    seed: u64,
    world_mode: WorldMode,
    elapsed_ns: u128,
    agent_steps_per_second: f64,
    live_agents: usize,
    physical_bodies: usize,
    active_chunks: usize,
    contact_samples: u64,
    contact_constraint_samples: u64,
    broad_phase_ns: u128,
    narrow_phase_ns: u128,
    solver_ns: u128,
}

fn env_value<T>(name: &str, default: T) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let agents = env_value("ERTW_BENCH_AGENTS", 32usize)?;
    let steps = env_value("ERTW_BENCH_STEPS", 300u32)?;
    let seed = env_value("ERTW_BENCH_SEED", 0xBEEF_BEEFu64)?;
    let world_mode = WorldMode::from_env()?;
    if matches!(world_mode, WorldMode::AgentsOnly) && steps > 120 {
        return Err("agents_only is a one-stream-period ablation; use at most 120 steps".into());
    }

    let mut simulation = ErtwWorld::new(seed);
    for index in 0..agents {
        simulation.spawn_agent(Box::new(Passive), collision_free_benchmark_position(index));
    }
    if matches!(world_mode, WorldMode::AgentsOnly) {
        simulation
            .app()
            .world_mut()
            .resource_mut::<ChunkManager>()
            .restore_active_chunks((-1..=1).flat_map(|x| (-1..=1).map(move |y| (x, y))));
    }

    let mut contact_samples = 0u64;
    let mut contact_constraint_samples = 0u64;
    let mut broad_phase_ns = 0u128;
    let mut narrow_phase_ns = 0u128;
    let mut solver_ns = 0u128;
    let started = Instant::now();
    for _ in 0..steps {
        simulation.step(1);
        let world = simulation.app().world();
        let collision = world.resource::<CollisionDiagnostics>();
        let solver = world.resource::<SolverDiagnostics>();
        contact_samples += u64::from(collision.contact_count);
        contact_constraint_samples += u64::from(solver.contact_constraint_count);
        broad_phase_ns += collision.broad_phase.as_nanos();
        narrow_phase_ns += collision.narrow_phase.as_nanos();
        solver_ns += solver.prepare_constraints.as_nanos()
            + solver.update_velocity_increments.as_nanos()
            + solver.integrate_velocities.as_nanos()
            + solver.warm_start.as_nanos()
            + solver.solve_constraints.as_nanos()
            + solver.integrate_positions.as_nanos()
            + solver.relax_velocities.as_nanos()
            + solver.apply_restitution.as_nanos()
            + solver.finalize.as_nanos()
            + solver.store_impulses.as_nanos()
            + solver.swept_ccd.as_nanos();
    }
    let elapsed = started.elapsed();

    let world = simulation.app().world_mut();
    let live_agents = world.query::<&AgentMarker>().iter(world).count();
    let physical_bodies = world.query::<&Physical>().iter(world).count();
    let active_chunks = world.resource::<ChunkManager>().active_chunks().count();
    let result = BenchmarkResult {
        agents_requested: agents,
        steps,
        seed,
        world_mode,
        elapsed_ns: elapsed.as_nanos(),
        agent_steps_per_second: agents as f64 * steps as f64 / elapsed.as_secs_f64(),
        live_agents,
        physical_bodies,
        active_chunks,
        contact_samples,
        contact_constraint_samples,
        broad_phase_ns,
        narrow_phase_ns,
        solver_ns,
    };
    let json = serde_json::to_string(&result)?;
    println!("{json}");
    if let Ok(output) = std::env::var("ERTW_BENCH_OUTPUT") {
        let path = Path::new(&output);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, format!("{json}\n"))?;
    }
    Ok(())
}
