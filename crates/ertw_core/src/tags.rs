//! Relation mask (`custom_tags`): 64 bits of emergent interaction flags.
//!
//! Objects are not hardcoded classes. They are created by mixing physical
//! properties with these bits, allowing emergent interactions like magnetic
//! attraction or volatile thermal traps. See spec item 2.

use bevy::prelude::*;

/// 64-bit relation mask packed into a Bevy component.
#[derive(Component, Clone, Copy, Debug, Default, Reflect, PartialEq, Eq)]
#[reflect(Component)]
pub struct CustomTags(pub u64);

impl CustomTags {
    pub const EMPTY: Self = CustomTags(0);

    /// Energy is released under sustained relative motion / friction.
    pub const ENERGY_CONVERTIBLE: u64 = 1 << 0;
    /// Can clamp onto other nodes to combine kinetic forces (spec item 9).
    pub const CLAMP_CAPABLE: u64 = 1 << 1;
    /// Low-conductivity material — used for fabricated shelters; resists the
    /// Thermal field's drain (spec item 3, 4).
    pub const SHELTER: u64 = 1 << 2;
    /// Fixed node in the Thermal field that passively converts ambient heat into
    /// reserve energy at structural-stress risk (spec item 5).
    pub const THERMAL_VENT: u64 = 1 << 3;
    /// When the passing EM field spikes, triggers a rapid energy release. The
    /// mechanism behind a "volatile thermal trap" (spec item 8).
    pub const VOLATILE_TRAP: u64 = 1 << 4;
    /// Attracts neighbors through the Electromagnetic field (magnetic behavior).
    pub const MAGNETIC: u64 = 1 << 5;
    /// Treated as a controllable agent rather than inert terrain.
    pub const AGENT: u64 = 1 << 6;
    /// Inherits/holds an active oscillator broadcast (spec items 9, 10).
    pub const OSCILLATOR: u64 = 1 << 7;

    #[inline]
    pub fn has(self, flag: u64) -> bool {
        self.0 & flag != 0
    }

    /// Build a mask from raw bits.
    #[inline]
    pub fn from_bits(bits: u64) -> Self {
        CustomTags(bits)
    }

    #[inline]
    pub fn with(mut self, flag: u64) -> Self {
        self.0 |= flag;
        self
    }

    #[inline]
    pub fn without(mut self, flag: u64) -> Self {
        self.0 &= !flag;
        self
    }

    /// Small per-child mutation applied on fragmentation/fragment spawns.
    pub fn mutate(mut self, rng: &mut impl rand::Rng) -> Self {
        if rng.gen_bool(0.05) {
            let bit = 1u64 << rng.gen_range(0..64);
            if rng.gen_bool(0.5) {
                self = self.with(bit);
            } else {
                self = self.without(bit);
            }
        }
        self
    }
}
