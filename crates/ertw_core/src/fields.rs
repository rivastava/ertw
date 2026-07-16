//! Global Substrate: three continuous vector fields (Kinetic/Friction, Thermal/
//! Weather, Electromagnetic/Signal Noise) sampled over position and time
//! (spec items 1, 8). All noise is seeded from [`SimSeed`] so genesis and field
//! drift are reproducible per-machine.

use avian2d::dynamics::rigid_body::{forces::WriteRigidBodyForces, LinearVelocity};
use bevy::prelude::*;
use noise::{NoiseFn, Perlin, Seedable};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Master seed threading into every noise function and per-chunk RNG. This is the
/// single source of determinism for the world (spec determinism decision).
#[derive(Resource, Clone, Copy, Debug)]
pub struct SimSeed(pub u64);

/// Three overlapping continuous fields, sampled at `(pos, t)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct FieldSample {
    /// Kinetic / friction field: tangential drag magnitude (0..~1).
    pub kinetic: f32,
    /// Thermal / weather field: ambient heat load (can be negative = cold).
    pub thermal: f32,
    /// Electromagnetic / signal-noise field: signed noise (-1..1).
    pub em: f32,
}

/// Global slow drift parameters (diurnal/seasonal baseline shift). Layered with
/// stochastic local spikes/inversions per chunk (spec item 8).
#[derive(Resource, Clone, Debug)]
pub struct FieldSampler {
    kinetic: Perlin,
    thermal: Perlin,
    em: Perlin,
    /// Master simulation time in seconds (advanced by fixed timestep).
    pub time: f32,
    /// Drift periods (seconds) for each field's global baseline.
    pub thermal_drift_period: f32,
    pub kinetic_drift_period: f32,
    pub em_drift_period: f32,
    /// Amplitude of local stochastic spikes (volatility).
    pub volatility: f32,
    kinetic_scale: f32,
    thermal_offset: f32,
    em_offset: f32,
}

impl FromWorld for FieldSampler {
    fn from_world(world: &mut World) -> Self {
        let SimSeed(seed) = *world.resource::<SimSeed>();
        // Distinct sub-seeds keep the three fields decorrelated but reproducible.
        let mut r = ChaCha8Rng::seed_from_u64(seed);
        let mut k = ChaCha8Rng::seed_from_u64(r.gen());
        let mut t = ChaCha8Rng::seed_from_u64(r.gen());
        let mut e = ChaCha8Rng::seed_from_u64(r.gen());
        // Perlin is seeded via `Seedable::set_seed(u32)`. Derive stable, distinct
        // seeds for the three fields from the rng streams.
        let kp = Perlin::default().set_seed(k.gen());
        let tp = Perlin::default().set_seed(t.gen());
        let ep = Perlin::default().set_seed(e.gen());
        Self {
            kinetic: kp,
            thermal: tp,
            em: ep,
            time: 0.0,
            thermal_drift_period: 120.0,
            kinetic_drift_period: 200.0,
            em_drift_period: 90.0,
            volatility: 0.6,
            kinetic_scale: 1.0,
            thermal_offset: 0.0,
            em_offset: 0.0,
        }
    }
}

impl FieldSampler {
    /// Sample the three fields at a world position, given the current time.
    pub fn sample(&self, pos: Vec2) -> FieldSample {
        // Offset by half a cell so exact integer world coordinates don't land on
        // Perlin lattice corners (where the gradient evaluates to ~0 for every
        // seed). Keeps the field continuous and seed-diverse everywhere.
        let p = [(pos.x + 0.5) as f64, (pos.y + 0.5) as f64];

        // Kinetic: low-frequency friction field plus drift.
        let kinetic = (self.kinetic.get(p) as f32 * 0.5 + 0.5) * self.kinetic_scale;

        // Thermal: field value plus diurnal drift; can go below zero (cold).
        let thermal = self.thermal.get(p) as f32 * 0.5 + self.thermal_offset;

        // EM: signed noise plus slow drift; spikes handled by noise gradient.
        let em = self.em.get(p) as f32 + self.em_offset;

        FieldSample {
            kinetic: kinetic.clamp(0.0, 2.0),
            thermal: thermal.max(-1.0), // cold allowed, never unbounded below
            em: em.clamp(-1.5, 1.5),
        }
    }

    /// Sample values and deterministic forward-difference gradients with only
    /// three field evaluations. Drift is spatially uniform, so it cancels from
    /// the gradient while remaining present in every returned value.
    pub fn sample_with_gradient(&self, pos: Vec2, epsilon: f32) -> (FieldSample, [Vec2; 3]) {
        let center = self.sample(pos);
        let x = self.sample(pos + Vec2::X * epsilon);
        let y = self.sample(pos + Vec2::Y * epsilon);
        let inverse_epsilon = epsilon.recip();
        (
            center,
            [
                Vec2::new(x.kinetic - center.kinetic, y.kinetic - center.kinetic) * inverse_epsilon,
                Vec2::new(x.thermal - center.thermal, y.thermal - center.thermal) * inverse_epsilon,
                Vec2::new(x.em - center.em, y.em - center.em) * inverse_epsilon,
            ],
        )
    }

    fn refresh_drift(&mut self) {
        let t = self.time;
        let thermal = (t * (std::f32::consts::TAU / self.thermal_drift_period)).sin();
        let kinetic = (t * (std::f32::consts::TAU / self.kinetic_drift_period)).sin();
        let em = (t * (std::f32::consts::TAU / self.em_drift_period)).sin();
        self.kinetic_scale = 1.0 - 0.4 * kinetic;
        self.thermal_offset = 0.4 * thermal;
        self.em_offset = 0.25 * em;
    }

    /// True if the EM field at `pos` is spiking enough to trigger a
    /// [`crate::tags::CustomTags::VOLATILE_TRAP`] energy release (spec item 8).
    pub fn is_em_spike(&self, pos: Vec2) -> bool {
        let f = self.sample(pos);
        f.em > 1.0 + 0.3 * self.volatility || f.em < -(1.0 + 0.3 * self.volatility)
    }
}

/// Fixed-step system advancing the field sampler clock. Runs inside the
/// `FixedUpdate` schedule so the world is deterministic w.r.t. step count. It
/// advances by the fixed timestep (not wall-clock `Time`) so the simulation is
/// reproducible regardless of how the schedule is driven.
pub fn advance_fields(mut sampler: ResMut<FieldSampler>) {
    sampler.time += crate::economy::FIXED_DT;
    sampler.refresh_drift();
}

/// Applies the substrate's direct physical effects. The kinetic channel acts
/// as spatially varying drag, while magnetic nodes couple through the signed
/// electromagnetic field. Pair forces are equal and opposite.
pub fn apply_field_forces(
    sampler: Res<FieldSampler>,
    spatial: Res<crate::spatial_hash::SpatialHash>,
    nodes: Query<(
        Entity,
        &Transform,
        &crate::components::Physical,
        &crate::components::Tags,
    )>,
    mut dynamics: ParamSet<(
        Query<&LinearVelocity>,
        Query<avian2d::dynamics::rigid_body::forces::Forces>,
    )>,
) {
    const DRAG_COEFFICIENT: f32 = 0.35;
    const MAGNETIC_RANGE: f32 = 10.0;
    const MAGNETIC_STRENGTH: f32 = 1.2;

    for (entity, transform, physical, _) in nodes.iter() {
        let pos = transform.translation.truncate();
        let velocity = dynamics.p0().get(entity).map(|v| v.0).unwrap_or(Vec2::ZERO);
        let drag = -velocity * sampler.sample(pos).kinetic * DRAG_COEFFICIENT * physical.mass;
        if let Ok(mut forces) = dynamics.p1().get_mut(entity) {
            forces.apply_force(drag);
        }
    }

    let magnetic = nodes
        .iter()
        .filter(|(_, _, _, tags)| tags.0.has(crate::tags::CustomTags::MAGNETIC))
        .map(|(entity, transform, _, _)| (entity, transform.translation.truncate()))
        .collect::<Vec<_>>();

    for (entity, pos) in magnetic {
        let mut nearby = Vec::new();
        spatial.query_radius(pos, MAGNETIC_RANGE, &mut nearby);
        for other in nearby {
            if entity.to_bits() >= other.to_bits() {
                continue;
            }
            let Ok((_, other_transform, _, other_tags)) = nodes.get(other) else {
                continue;
            };
            if !other_tags.0.has(crate::tags::CustomTags::MAGNETIC) {
                continue;
            }
            let delta = other_transform.translation.truncate() - pos;
            let distance_sq = delta.length_squared().max(0.25);
            if distance_sq > MAGNETIC_RANGE * MAGNETIC_RANGE {
                continue;
            }
            let midpoint = pos + delta * 0.5;
            let polarity = sampler.sample(midpoint).em.signum();
            let force =
                delta.normalize_or_zero() * polarity * (MAGNETIC_STRENGTH / distance_sq.max(1.0));
            if let Ok(mut first) = dynamics.p1().get_mut(entity) {
                first.apply_force(force);
            }
            if let Ok(mut second) = dynamics.p1().get_mut(other) {
                second.apply_force(-force);
            }
        }
    }
}

/// Deterministic per-entity/per-chunk RNG derived from the master seed and a
/// spatial key. Used for procedural genesis and fragmentation mutation.
pub fn seeded_rng(seed: SimSeed, key: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed.0 ^ key.wrapping_mul(0x9E3779B97F4A7C15))
}
