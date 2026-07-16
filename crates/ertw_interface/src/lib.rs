//! ERTW Universal Interface Contract.
//!
//! This crate defines the *only* surface the world exposes to an agent: a fixed
//! observation tensor and a symmetric continuous action tensor. The schema is
//! identical regardless of what agent architecture is plugged in (random policy,
//! PPO/DQN, wrapped LLM/VLM). The world owes the agent nothing and tells it
//! nothing beyond raw, egocentric, floating-point physical truth.
//!
//! No reward, no score, no goals. See the repository's public architecture and
//! protocol documentation.

/// Self-contained 2D vector so the interface crate stays free of Bevy grid
/// dependencies for downstream consumers (external adapters need not pull Bevy).
/// The world converts this to/from `bevy::math::Vec2` at the boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2Lite {
    pub x: f32,
    pub y: f32,
}

impl Vec2Lite {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }
    pub fn clamp_length_max(self, max: f32) -> Self {
        let l = self.length();
        if l > max && l > 0.0 {
            let s = max / l;
            Self::new(self.x * s, self.y * s)
        } else {
            self
        }
    }
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

/// Per-agent configuration that shapes the observation tensor.
///
/// The world streams an egocentric window of `max_neighbors` entities (padded
/// with zero-state ghost nodes) plus `field_samples` samples of each global
/// field across a bounded `sensor_radius`.
#[derive(Clone, Copy, Debug)]
pub struct InterfaceConfig {
    /// Hard cap on how many neighbors are reported. Real neighbors within the
    /// sensor radius are sorted by ascending distance then padded with ghost
    /// nodes up to this count. Must be > 0.
    pub max_neighbors: usize,
    /// Bounded egocentric radius (world units) within which neighbors and field
    /// samples are gathered.
    pub sensor_radius: f32,
    /// Number of radial samples per field, evenly spaced from 0..=`sensor_radius`.
    pub field_samples: usize,
    /// Number of scalar channels per field sample (e.g. magnitude + gradient dir).
    pub field_channels: usize,
}

impl Default for InterfaceConfig {
    fn default() -> Self {
        Self {
            max_neighbors: 16,
            sensor_radius: 12.0,
            field_samples: 4,
            field_channels: 3,
        }
    }
}

/// Named offsets into a single neighbor's flattened f32 block. Centralizing
/// these keeps encoding and decoding aligned as the schema evolves.
pub const REL_X: usize = 0;
pub const REL_Y: usize = 1;
pub const REL_VX: usize = 2;
pub const REL_VY: usize = 3;
pub const MASS: usize = 4;
pub const STRUCTURE: usize = 5;
pub const ENERGY: usize = 6;
pub const TAG_0: usize = 7;
pub const TAG_1: usize = 8;
pub const TAG_2: usize = 9;
pub const TAG_3: usize = 10;
pub const CONDUCTIVITY: usize = 11;
pub const OSC_FREQ: usize = 12;
pub const OSC_PHASE: usize = 13;
pub const VALID: usize = 14;

/// Number of floats describing a single neighbor in the observation tensor.
///
/// Layout:
///   [0]  rel_pos.x
///   [1]  rel_pos.y
///   [2]  rel_vel.x
///   [3]  rel_vel.y
///   [4]  mass
///   [5]  structure
///   [6]  energy
///   [7..11] four exact 16-bit chunks of `CustomTags`
///   [11] conductivity  (shelter detectability, spec items 3, 4, 5)
///   [12] osc_freq      (neighbor oscillator broadcast, spec item 9)
///   [13] osc_phase
///   [14] valid         (1.0 = real neighbor, 0.0 = padding ghost)
pub const NEIGHBOR_STRIDE: usize = 15;

/// Number of scalar fields reported. Kinetic, Thermal, Electromagnetic.
pub const FIELD_COUNT: usize = 3;

/// Number of floats in the self-state prefix of the observation tensor.
///
/// Layout: [local_vel_x, local_vel_y, mass, structure, energy, osc_freq,
/// osc_phase, energy_surplus]. Absolute position is intentionally absent.
pub const SELF_STRIDE: usize = 8;

impl InterfaceConfig {
    /// Total length of the flattened observation tensor in f32s.
    pub fn observation_len(&self) -> usize {
        // self-state + field samples + neighbor block
        SELF_STRIDE
            + FIELD_COUNT * self.field_samples * self.field_channels
            + self.max_neighbors * NEIGHBOR_STRIDE
    }

    /// Total length of the flattened action tensor in f32s.
    pub fn action_len(&self) -> usize {
        ACTION_STRIDE
    }
}

/// Number of floats in the action tensor.
///
/// Layout:
///   [0] force_x     — continuous locomotion force (world units / s^2 baked via mass)
///   [1] force_y     — continuous locomotion force
///   [2] torque      — continuous angular force
///   [3] clamp       — 1.0 = attempt clamp to nearest neighbor, 0.0 = release
///   [4] fabricate   — 1.0 = trigger fabrication of low-conductivity structure
///   [5] osc_freq    — commanded oscillator frequency
///   [6] osc_phase   — commanded oscillator phase
pub const ACTION_STRIDE: usize = 7;

/// Continuous action emitted *every* step. Every actuator call costs stored
/// energy — there is no free action. The world interprets and bounds these.
#[derive(Clone, Copy, Debug, Default)]
pub struct ActionTensor {
    pub force: Vec2Lite,
    pub torque: f32,
    /// `>0.5` means the agent requests a clamp to the nearest neighbor.
    /// `<0.5` means release any current clamp.
    pub clamp: f32,
    /// >0.5 means trigger fabrication (costs stored mass + energy).
    pub fabricate: f32,
    /// Commanded oscillator frequency (rad/s).
    pub osc_freq: f32,
    /// Commanded oscillator phase (rad).
    pub osc_phase: f32,
}

impl ActionTensor {
    /// Clamp all components to their physical ranges so downstream code never
    /// sees NaN/inf or out-of-band values.
    pub fn sanitize(&mut self) {
        // NaN/inf inputs collapse to safe defaults BEFORE the clamp bands run
        // because Rust's `f32::clamp` propagates NaN.
        self.torque = if self.torque.is_finite() {
            self.torque
        } else {
            0.0
        };
        self.clamp = if self.clamp.is_finite() {
            self.clamp
        } else {
            0.0
        };
        self.fabricate = if self.fabricate.is_finite() {
            self.fabricate
        } else {
            0.0
        };
        self.osc_freq = if self.osc_freq.is_finite() {
            self.osc_freq
        } else {
            0.0
        };
        self.osc_phase = if self.osc_phase.is_finite() {
            self.osc_phase
        } else {
            0.0
        };
        self.force = if self.force.is_finite() {
            self.force.clamp_length_max(1.0)
        } else {
            Vec2Lite::ZERO
        };
        self.torque = self.torque.clamp(-1.0, 1.0);
        self.clamp = self.clamp.clamp(0.0, 1.0);
        self.fabricate = self.fabricate.clamp(0.0, 1.0);
        self.osc_freq = self.osc_freq.clamp(-16.0, 16.0);
        self.osc_phase = self.osc_phase.rem_euclid(std::f32::consts::TAU);
    }

    /// Flatten into the canonical wire layout.
    pub fn to_f32(&self) -> [f32; ACTION_STRIDE] {
        [
            self.force.x,
            self.force.y,
            self.torque,
            self.clamp,
            self.fabricate,
            self.osc_freq,
            self.osc_phase,
        ]
    }

    /// Parse from a flat slice in the canonical wire layout. Panics if `s.len() != ACTION_STRIDE`.
    pub fn from_f32(s: &[f32]) -> Self {
        assert_eq!(s.len(), ACTION_STRIDE, "action slice wrong length");
        let mut a = Self {
            force: Vec2Lite::new(s[0], s[1]),
            torque: s[2],
            clamp: s[3],
            fabricate: s[4],
            osc_freq: s[5],
            osc_phase: s[6],
        };
        a.sanitize();
        a
    }
}

/// A single egocentric neighbor entry (before flattening). `valid == false`
/// represents a padding ghost node.
///
/// `conductivity` lets the agent detect low-conductivity shelter material
/// (spec items 3, 4, 5). `osc_freq`/`osc_phase` carry the neighbor's current
/// oscillator broadcast so the agent can hear ω signals from nearby nodes
/// (spec item 9).
#[derive(Clone, Copy, Debug, Default)]
pub struct NeighborView {
    pub rel_pos: Vec2Lite,
    pub rel_vel: Vec2Lite,
    pub mass: f32,
    pub structure: f32,
    pub energy: f32,
    pub tags: u64,
    pub conductivity: f32,
    pub osc_freq: f32,
    pub osc_phase: f32,
    pub valid: bool,
}

/// Egocentric observation the world hands to an agent. Build with
/// [`ObservationTensor::builder`]; flatten with [`ObservationTensor::to_f32_vec`]
/// for the wire protocol or external adapters.
#[derive(Clone, Debug)]
pub struct ObservationTensor {
    pub step: u64,
    pub entity_id: u64,
    pub config: InterfaceConfig,
    pub self_state: [f32; SELF_STRIDE],
    /// `field_samples * FIELD_COUNT * field_channels` floats.
    pub field: Vec<f32>,
    /// Exactly `max_neighbors` entries (real neighbors first, then ghosts).
    pub neighbors: Vec<NeighborView>,
}

impl ObservationTensor {
    pub fn new(config: InterfaceConfig) -> Self {
        assert!(config.max_neighbors > 0, "max_neighbors must be > 0");
        let field_len = FIELD_COUNT * config.field_samples * config.field_channels;
        let neighbors = vec![NeighborView::default(); config.max_neighbors];
        Self {
            step: 0,
            entity_id: 0,
            config,
            self_state: [0.0; SELF_STRIDE],
            field: vec![0.0; field_len],
            neighbors,
        }
    }

    /// Total flattened length.
    pub fn len(&self) -> usize {
        self.config.observation_len()
    }

    /// `ObservationTensor` is always populated to its configured length; this
    /// method exists only to satisfy clippy's `len_without_is_empty` rule.
    /// The observation tensor is never empty in the meaningful sense.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Flatten into the canonical wire layout:
    /// `[self_state (SELF_STRIDE)] ++ [field] ++ [neighbors packed NEIGHBOR_STRIDE each]`.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len());
        out.extend_from_slice(&self.self_state);
        out.extend_from_slice(&self.field);
        for n in &self.neighbors {
            out.push(n.rel_pos.x);
            out.push(n.rel_pos.y);
            out.push(n.rel_vel.x);
            out.push(n.rel_vel.y);
            out.push(n.mass);
            out.push(n.structure);
            out.push(n.energy);
            out.push((n.tags & 0xFFFF) as f32);
            out.push(((n.tags >> 16) & 0xFFFF) as f32);
            out.push(((n.tags >> 32) & 0xFFFF) as f32);
            out.push(((n.tags >> 48) & 0xFFFF) as f32);
            out.push(n.conductivity);
            out.push(n.osc_freq);
            out.push(n.osc_phase);
            out.push(if n.valid { 1.0 } else { 0.0 });
        }
        out
    }

    /// Parse from a flat slice in canonical layout. Panics if length mismatches.
    pub fn from_f32_slice(s: &[f32], config: InterfaceConfig) -> Self {
        assert_eq!(
            s.len(),
            config.observation_len(),
            "observation slice wrong length"
        );
        let mut obs = Self::new(config);
        obs.self_state.copy_from_slice(&s[..SELF_STRIDE]);
        let field_end = SELF_STRIDE + obs.field.len();
        obs.field.copy_from_slice(&s[SELF_STRIDE..field_end]);
        for (i, n) in obs.neighbors.iter_mut().enumerate() {
            let base = field_end + i * NEIGHBOR_STRIDE;
            let raw = &s[base..base + NEIGHBOR_STRIDE];
            let tag_0 = raw[TAG_0].clamp(0.0, u16::MAX as f32) as u16 as u64;
            let tag_1 = raw[TAG_1].clamp(0.0, u16::MAX as f32) as u16 as u64;
            let tag_2 = raw[TAG_2].clamp(0.0, u16::MAX as f32) as u16 as u64;
            let tag_3 = raw[TAG_3].clamp(0.0, u16::MAX as f32) as u16 as u64;
            *n = NeighborView {
                rel_pos: Vec2Lite::new(raw[REL_X], raw[REL_Y]),
                rel_vel: Vec2Lite::new(raw[REL_VX], raw[REL_VY]),
                mass: raw[MASS],
                structure: raw[STRUCTURE],
                energy: raw[ENERGY],
                tags: tag_0 | (tag_1 << 16) | (tag_2 << 32) | (tag_3 << 48),
                conductivity: raw[CONDUCTIVITY],
                osc_freq: raw[OSC_FREQ],
                osc_phase: raw[OSC_PHASE],
                valid: raw[VALID] >= 0.5,
            };
        }
        obs
    }
}

pub const WIRE_MAGIC: u32 = u32::from_le_bytes(*b"ERTW");
pub const FRAME_HELLO: u32 = 1;
pub const FRAME_OBSERVATION: u32 = 2;
pub const FRAME_ACTION: u32 = 3;
pub const WIRE_HEADER_LEN: usize = 14;

#[derive(Clone, Copy, Debug, Default)]
pub struct WireHeader {
    pub version: u8,
    pub frame_kind: u32,
    pub frame_bytes: u32,
    pub step: u64,
    pub entity_id: u64,
    pub max_neighbors: u32,
    pub neighbor_count: u32,
    pub field_samples: u32,
    pub field_channels: u32,
    pub payload_floats: u32,
}

pub fn wire_header(header: WireHeader) -> [u32; WIRE_HEADER_LEN] {
    [
        WIRE_MAGIC,
        header.version as u32,
        header.frame_kind,
        header.frame_bytes,
        header.step as u32,
        (header.step >> 32) as u32,
        header.entity_id as u32,
        (header.entity_id >> 32) as u32,
        header.max_neighbors,
        header.neighbor_count,
        header.field_samples,
        header.field_channels,
        header.payload_floats,
        0,
    ]
}

/// The contract every agent (in-process or remote) satisfies. The world calls
/// `act` once per fixed step with an observation; the agent returns a continuous
/// action. The trait is intentionally free of any world types beyond the tensors.
pub trait Agent: Send + Sync {
    /// Produce an action for the given observation. Called once per fixed step.
    fn act(&mut self, observation: &ObservationTensor) -> ActionTensor;

    /// Optional hook called once when the agent is (re)spawned, carrying its
    /// inherited physical tuning seed. Default is a no-op.
    fn on_reset(&mut self, _seed: u64) {}

    /// Create an independent controller for an offspring. Policies that cannot
    /// reproduce, such as a single remote socket, return `None`.
    fn spawn_child(&mut self, _seed: u64) -> Option<Box<dyn Agent>> {
        None
    }
}

/// Helper to build an observation field-view from a sampler without pulling the
/// whole world into this crate.
pub mod field_view {
    /// One sampled scalar triple for a field at a given radius.
    /// `[magnitude, gradient_x, gradient_y]` (gradient optional, may be 0).
    pub type FieldSample = [f32; 3];
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> InterfaceConfig {
        InterfaceConfig {
            max_neighbors: 4,
            sensor_radius: 10.0,
            field_samples: 3,
            field_channels: 3,
        }
    }

    #[test]
    fn observation_len_matches_layout() {
        let c = cfg();
        let expected = SELF_STRIDE
            + FIELD_COUNT * c.field_samples * c.field_channels
            + c.max_neighbors * NEIGHBOR_STRIDE;
        let obs = ObservationTensor::new(c);
        assert_eq!(obs.len(), expected);
        assert_eq!(obs.to_f32_vec().len(), expected);
    }

    #[test]
    fn padding_neighbors_are_ghosts() {
        let obs = ObservationTensor::new(cfg());
        assert_eq!(obs.neighbors.len(), cfg().max_neighbors);
        assert!(obs.neighbors.iter().all(|n| !n.valid));
        // Default rel_pos/rel_vel are zero and energy is zero; valid flag is the discriminator.
        let flat = obs.to_f32_vec();
        let base = SELF_STRIDE + FIELD_COUNT * cfg().field_samples * cfg().field_channels;
        for i in 0..cfg().max_neighbors {
            assert_eq!(flat[base + i * NEIGHBOR_STRIDE + VALID], 0.0);
        }
    }

    #[test]
    fn roundtrip_preserves_all_neighbor_fields() {
        let mut obs = ObservationTensor::new(cfg());
        obs.neighbors[0] = NeighborView {
            rel_pos: Vec2Lite::new(1.5, -2.25),
            rel_vel: Vec2Lite::new(0.7, 1.3),
            mass: 2.5,
            structure: 7.0,
            energy: 11.5,
            tags: 0xFEDC_BA98_7654_3210,
            conductivity: 0.2,
            osc_freq: 1.75,
            osc_phase: 4.2,
            valid: true,
        };
        let flat = obs.to_f32_vec();
        let back = ObservationTensor::from_f32_slice(&flat, obs.config);
        let a = &obs.neighbors[0];
        let b = &back.neighbors[0];
        assert_eq!(a.rel_pos, b.rel_pos);
        assert_eq!(a.rel_vel, b.rel_vel);
        assert_eq!(a.mass, b.mass);
        assert_eq!(a.structure, b.structure);
        assert_eq!(a.energy, b.energy);
        assert_eq!(a.tags, b.tags);
        assert_eq!(a.conductivity, b.conductivity);
        assert_eq!(a.osc_freq, b.osc_freq);
        assert_eq!(a.osc_phase, b.osc_phase);
        assert_eq!(a.valid, b.valid);
    }

    #[test]
    fn arbitrary_tags_and_valid_flags_roundtrip_exactly() {
        let mut obs = ObservationTensor::new(cfg());
        obs.neighbors[0] = NeighborView {
            tags: 0xFFFF_FFFF_0000_0001,
            conductivity: 0.5,
            osc_freq: 0.0,
            osc_phase: 0.0,
            valid: true,
            ..Default::default()
        };
        obs.neighbors[1] = NeighborView {
            tags: 0xFFFF_FFFF_DEAD_BEEF,
            valid: false,
            ..Default::default()
        };
        let back = ObservationTensor::from_f32_slice(&obs.to_f32_vec(), obs.config);
        assert!(
            back.neighbors[0].valid,
            "real neighbor with high tag bits must roundtrip as valid"
        );
        assert!(
            !back.neighbors[1].valid,
            "ghost neighbor must roundtrip as invalid"
        );
        assert_eq!(back.neighbors[0].tags, 0xFFFF_FFFF_0000_0001);
        assert_eq!(back.neighbors[1].tags, 0xFFFF_FFFF_DEAD_BEEF);
    }

    #[test]
    fn action_sanitize_nan_inf() {
        let mut a = ActionTensor {
            force: Vec2Lite::new(f32::NAN, f32::INFINITY),
            torque: f32::NAN,
            clamp: f32::NAN,
            fabricate: f32::NAN,
            osc_freq: f32::NAN,
            osc_phase: f32::NAN,
        };
        a.sanitize();
        assert_eq!(a.force, Vec2Lite::ZERO);
        assert_eq!(a.torque, 0.0);
        assert_eq!(a.clamp, 0.0);
        assert_eq!(a.fabricate, 0.0);
        assert_eq!(a.osc_freq, 0.0);
        assert_eq!(a.osc_phase, 0.0);
    }

    #[test]
    fn action_sanitize_clamps_out_of_band() {
        let mut a = ActionTensor {
            force: Vec2Lite::new(10.0, 0.0),
            torque: 5.0,
            clamp: 2.0,
            fabricate: -1.0,
            osc_freq: 99.0,
            osc_phase: -99.0,
        };
        a.sanitize();
        assert!(
            a.force.length() <= 1.0 + 1e-5,
            "force must be length-clamped to 1.0"
        );
        assert_eq!(a.torque, 1.0);
        assert_eq!(a.clamp, 1.0);
        assert_eq!(a.fabricate, 0.0);
        assert_eq!(a.osc_freq, 16.0);
        assert!(a.osc_phase >= 0.0 && a.osc_phase <= std::f32::consts::TAU);
    }

    #[test]
    fn action_roundtrip_preserves_values() {
        let a = ActionTensor {
            force: Vec2Lite::new(0.3, -0.4),
            torque: -0.7,
            clamp: 0.5,
            fabricate: 0.0,
            osc_freq: 2.5,
            osc_phase: 1.2,
        };
        let flat = a.to_f32();
        let b = ActionTensor::from_f32(&flat);
        assert_eq!(a.force, b.force);
        assert_eq!(a.torque, b.torque);
        assert_eq!(a.clamp, b.clamp);
        assert_eq!(a.fabricate, b.fabricate);
        assert_eq!(a.osc_freq, b.osc_freq);
        assert_eq!(a.osc_phase, b.osc_phase);
    }

    #[test]
    fn wire_header_is_stable() {
        let h = wire_header(WireHeader {
            version: 3,
            frame_kind: FRAME_OBSERVATION,
            frame_bytes: 1024,
            step: 0x1234_5678_9ABC_DEF0,
            entity_id: 0xFEDC_BA98_7654_3210,
            max_neighbors: 16,
            neighbor_count: 3,
            field_samples: 4,
            field_channels: 3,
            payload_floats: 280,
        });
        assert_eq!(h[0], WIRE_MAGIC);
        assert_eq!(h[1], 3);
        assert_eq!(h[2], FRAME_OBSERVATION);
        assert_eq!(h[3], 1024);
        assert_eq!(h[4], 0x9ABC_DEF0);
        assert_eq!(h[5], 0x1234_5678);
        assert_eq!(h[6], 0x7654_3210);
        assert_eq!(h[7], 0xFEDC_BA98);
        assert_eq!(WIRE_HEADER_LEN, 14);
    }

    #[test]
    fn relation_mask_encoding_is_exact_for_many_values() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..4096 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut observation = ObservationTensor::new(cfg());
            observation.neighbors[0] = NeighborView {
                tags: state,
                valid: true,
                ..Default::default()
            };
            let decoded =
                ObservationTensor::from_f32_slice(&observation.to_f32_vec(), observation.config);
            assert_eq!(decoded.neighbors[0].tags, state);
        }
    }
}
