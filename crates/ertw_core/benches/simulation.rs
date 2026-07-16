use ertw_core::ErtwWorld;
use ertw_interface::{ActionTensor, Agent, ObservationTensor};
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

fn main() {
    let agents = std::env::var("ERTW_BENCH_AGENTS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32usize);
    let steps = std::env::var("ERTW_BENCH_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(300u32);
    let mut simulation = ErtwWorld::new(0xBEEFBEEF);
    for index in 0..agents {
        let x = (index % 8) as f32 * 2.0;
        let y = (index / 8) as f32 * 2.0;
        simulation.spawn_agent(Box::new(Passive), bevy::math::Vec2::new(x, y));
    }
    let started = Instant::now();
    simulation.step(steps);
    let elapsed = started.elapsed();
    println!(
        "agents={agents} steps={steps} elapsed_ms={} agent_steps_per_second={:.0}",
        elapsed.as_millis(),
        agents as f64 * steps as f64 / elapsed.as_secs_f64()
    );
}
