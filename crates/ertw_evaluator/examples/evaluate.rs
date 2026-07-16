//! External evaluator demo (spec design philosophy, Phase 11).
//!
//! The world exposes no reward or score. This example runs the headless world
//! with several random agents, then reads the *same raw simulation state* to
//! build [`ertw_evaluator::CompetenceRecord`]s and rank them by survival and
//! energy accumulated. The ranking never reaches the agents.
//!
//! Run: `cargo run -p ertw_evaluator --example evaluate`

use ertw_core::lineage;
use ertw_core::ErtwWorld;
use ertw_evaluator::{rank, CompetenceRecord};
use random_policy::RandomPolicy;

fn main() {
    let mut world = ErtwWorld::new(0xBADC0DE);
    for i in 0..6u64 {
        let pos = bevy::math::Vec2::new((i as f32 - 2.5) * 4.0, (i as f32 % 3.0) * 3.0);
        world.spawn_agent(Box::new(RandomPolicy::new(0x2000 + i)), pos);
    }

    let steps = 60 * 40; // 40 simulated seconds
    world.step(steps);

    // Collect competence from the live world state at the final step.
    let records: Vec<CompetenceRecord> = {
        let w = world.app().world_mut();
        lineage::collect_competence(w, steps)
            .into_iter()
            .map(|s| CompetenceRecord {
                alive: s.alive,
                step: s.step,
                entity: s.entity,
                generation: s.generation,
                lineage: s.lineage,
                survival_steps: s.step.saturating_sub(s.born_step),
                energy_accumulated: s.vented + s.harvested + s.consumed_from_others,
                energy_harvested: s.harvested,
                energy_consumed: s.consumed_from_others,
            })
            .collect()
    };

    let ordering = rank(&records);
    println!("step={steps}  recorded_agents={}", records.len());
    println!("rank  entity  gen  survival  energy_acc");
    for (i, &idx) in ordering.iter().enumerate() {
        let r = &records[idx];
        println!(
            "{:>3}  {:>6}  {:>3}  {:>8}  {:>10.2}",
            i + 1,
            r.entity,
            r.generation,
            r.survival_steps,
            r.energy_accumulated
        );
    }
}
