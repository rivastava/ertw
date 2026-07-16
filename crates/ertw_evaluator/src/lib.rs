//! External Evaluator (spec design philosophy, Phase 11).
//!
//! The world exposes no reward or score. This separate system reads the *same raw
//! simulation state* — survival duration, energy accumulated, and lineage depth
//! — and computes comparative competence offline. The signal never reaches the
//! agent.
//!
//! This module defines the metric record and a pure aggregation helper so downstream
//! code can consume it without depending on the live world.

use ertw_core::components::EnergyLedger;

/// Snapshot of an agent's competence-relevant state at a given step. None of
/// these are rewards; they are observations an offline evaluator ranks by.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompetenceRecord {
    pub alive: bool,
    pub step: u32,
    pub entity: u64,
    pub generation: u32,
    pub lineage: u64,
    pub survival_steps: u32,
    pub energy_accumulated: f32,
    pub energy_harvested: f32,
    pub energy_consumed: f32,
}

impl CompetenceRecord {
    pub fn from_ledger(
        entity: u64,
        generation: u32,
        lineage: u64,
        step: u32,
        l: &EnergyLedger,
    ) -> Self {
        Self {
            alive: true,
            step,
            entity,
            generation,
            lineage,
            survival_steps: step.saturating_sub(l.born_step),
            energy_accumulated: l.vented + l.harvested + l.consumed_from_others,
            energy_harvested: l.harvested,
            energy_consumed: l.consumed_from_others,
        }
    }
}

/// Rank records by survival then by energy accumulated. Returns indices sorted
/// best-first. Pure: no world access.
pub fn rank(records: &[CompetenceRecord]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..records.len()).collect();
    idx.sort_by(|&a, &b| {
        records[b]
            .survival_steps
            .cmp(&records[a].survival_steps)
            .then(records[b].generation.cmp(&records[a].generation))
            .then(
                finite_metric(records[b].energy_accumulated)
                    .total_cmp(&finite_metric(records[a].energy_accumulated)),
            )
    });
    idx
}

fn finite_metric(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        f32::MIN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_is_stable_for_dead_records_and_non_finite_metrics() {
        let records = [
            CompetenceRecord {
                alive: false,
                survival_steps: 100,
                energy_accumulated: 2.0,
                ..Default::default()
            },
            CompetenceRecord {
                alive: true,
                survival_steps: 80,
                energy_accumulated: 100.0,
                ..Default::default()
            },
            CompetenceRecord {
                alive: false,
                survival_steps: 100,
                energy_accumulated: f32::NAN,
                ..Default::default()
            },
        ];
        assert_eq!(rank(&records), vec![0, 2, 1]);
    }
}
