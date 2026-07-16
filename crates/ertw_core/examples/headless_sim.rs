//! Headless simulation: run the world with N random agents for a fixed number of
//! steps, then print survival/economy stats. No window, no rendering.
//!
//! Run: `cargo run -p ertw_core --example headless_sim`

use ertw_core::ErtwWorld;
use random_policy::RandomPolicy;

fn main() {
    let mut world = ErtwWorld::new(0xC0FFEE);
    for i in 0..8u64 {
        let pos = bevy::math::Vec2::new((i as f32 - 3.5) * 3.0, (i as f32 % 2.0) * 2.0);
        world.spawn_agent(Box::new(RandomPolicy::new(0x1000 + i)), pos);
    }
    let steps = 600;
    world.step(steps);
    println!("completed {steps} steps headless");
}
