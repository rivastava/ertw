//! Human Visualization overlay (spec item 11).
//!
//! Agents operate "blind" in a purely mathematical space; this crate overlays a
//! high-performance 2D primitive rendering pipeline (shapes and lines) tied to the
//! world's coordinate system so researchers can observe emergent behavior. It is a
//! non-participant: it reads state, never influences it.

use bevy::prelude::*;
use ertw_core::components::{Physical, Tags};
use ertw_core::tags::CustomTags;

#[cfg(feature = "render")]
mod app;

#[cfg(feature = "render")]
pub use app::run_rendered_sim;

/// Render plugin. Add to a [`bevy::prelude::App`] after the core world systems.
/// Gated behind the `render` feature so headless builds stay dependency-light.
pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw_nodes);
    }
}

/// Draw each node as a circle whose color encodes its relation tags, with a ring
/// whose radius reflects remaining structure.
fn draw_nodes(mut gizmos: Gizmos, query: Query<(Entity, &Transform, &Physical, &Tags)>) {
    for (_e, tf, phys, tags) in query.iter() {
        let pos = tf.translation.truncate();
        let color = tag_color(tags.0);
        let radius = 0.5 + 0.4 * (phys.structure.max(0.0) / 10.0).clamp(0.0, 1.0);
        gizmos.circle_2d(pos, radius, color);
        // Energy pip: a short line whose length encodes stored energy.
        let energy_len = (phys.energy * 0.05).clamp(0.0, 3.0);
        gizmos.line_2d(
            pos,
            pos + Vec2::new(energy_len, 0.0),
            Color::srgb(0.2, 1.0, 0.2),
        );
        let _ = CustomTags::AGENT;
    }
}

fn tag_color(tags: CustomTags) -> Color {
    if tags.has(CustomTags::AGENT) {
        Color::srgb(0.3, 0.6, 1.0)
    } else if tags.has(CustomTags::THERMAL_VENT) {
        Color::srgb(1.0, 0.4, 0.1)
    } else if tags.has(CustomTags::VOLATILE_TRAP) {
        Color::srgb(1.0, 0.1, 0.6)
    } else {
        Color::srgb(0.6, 0.6, 0.6)
    }
}
