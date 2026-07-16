//! Reference in-process agent: a bounded random policy (spec item 3, agent-agnostic).
//!
//! It implements the `Agent` trait exactly like any PPO/DQN/LLM adapter would,
//! proving the interface shape is identical for every architecture. It receives
//! only the egocentric observation tensor and returns a continuous action.

use ertw_interface::{ActionTensor, Agent, ObservationTensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Bounded random locomotion + occasional clamp/fabricate/oscillator commands.
pub struct RandomPolicy {
    rng: StdRng,
    seed: u64,
}

impl RandomPolicy {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }
}

impl Agent for RandomPolicy {
    fn act(&mut self, _obs: &ObservationTensor) -> ActionTensor {
        ActionTensor {
            force: ertw_interface::Vec2Lite::new(
                self.rng.gen_range(-1.0..1.0),
                self.rng.gen_range(-1.0..1.0),
            ),
            torque: self.rng.gen_range(-1.0..1.0),
            clamp: if self.rng.gen_bool(0.02) { 1.0 } else { 0.0 },
            fabricate: if self.rng.gen_bool(0.01) { 1.0 } else { 0.0 },
            osc_freq: self.rng.gen_range(-2.0..2.0),
            osc_phase: self.rng.gen_range(0.0..std::f32::consts::TAU),
        }
    }

    fn on_reset(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = StdRng::seed_from_u64(seed);
    }

    fn spawn_child(&mut self, seed: u64) -> Option<Box<dyn Agent>> {
        Some(Box::new(Self::new(seed)))
    }
}
