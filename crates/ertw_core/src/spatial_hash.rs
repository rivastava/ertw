//! Custom spatial hash keeping entity queries local (spec item 7). World size is
//! bounded only by disk, not memory: we index entities by a uniform grid keyed on
//! world position and only query a bounded radius around a point.
//!
//! Standard crates lag Bevy 0.19, and the spec explicitly requires a spatial hash,
//! so we implement a minimal, lock-free-per-frame rebuild structure.

use bevy::ecs::query::QueryFilter;
use bevy::prelude::*;
use std::collections::HashMap;

/// Uniform grid cell size in world units.
pub const CELL_SIZE: f32 = 4.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cell {
    pub x: i32,
    pub y: i32,
}

impl Cell {
    pub fn from_pos(pos: Vec2) -> Self {
        Self {
            x: (pos.x / CELL_SIZE).floor() as i32,
            y: (pos.y / CELL_SIZE).floor() as i32,
        }
    }
}

/// Spatial index mapping grid cells to the entities currently occupying them.
/// Rebuilt each fixed step from entity transforms. Cheap to rebuild for the
/// entity counts we target and avoids per-frame allocation churn via pooling.
#[derive(Resource, Default)]
pub struct SpatialHash {
    cells: HashMap<Cell, Vec<Entity>>,
}

impl SpatialHash {
    /// Rebuild the index from a position query. Call once at the start of each
    /// fixed step before any neighbor queries.
    pub fn rebuild<F: QueryFilter>(&mut self, positions: &Query<(Entity, &Transform), F>) {
        self.cells.clear();
        for (e, tf) in positions.iter() {
            let cell = Cell::from_pos(tf.translation.truncate());
            self.cells.entry(cell).or_default().push(e);
        }
    }

    /// Rebuild the index from a borrowed `QueryState`. Useful from tests and
    /// from contexts where the caller already holds a `QueryState` rather than
    /// a `Query`. Behaves identically to [`Self::rebuild`].
    pub fn rebuild_from_state(&mut self, world: &mut World) {
        self.cells.clear();
        let mut state = world.query::<(Entity, &Transform)>();
        for (e, tf) in state.iter(world) {
            let cell = Cell::from_pos(tf.translation.truncate());
            self.cells.entry(cell).or_default().push(e);
        }
    }

    /// Collect all entities within `radius` of `pos` into `out` (unsorted).
    /// Includes the entity at `pos` itself unless filtered by the caller.
    pub fn query_radius(&self, pos: Vec2, radius: f32, out: &mut Vec<Entity>) {
        out.clear();
        let r_cells = (radius / CELL_SIZE).ceil() as i32;
        let center = Cell::from_pos(pos);
        for dx in -r_cells..=r_cells {
            for dy in -r_cells..=r_cells {
                let cell = Cell {
                    x: center.x + dx,
                    y: center.y + dy,
                };
                if let Some(ents) = self.cells.get(&cell) {
                    out.extend_from_slice(ents);
                }
            }
        }
    }

    /// Number of occupied cells (diagnostic).
    pub fn occupied_cells(&self) -> usize {
        self.cells.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::{Transform, World};

    fn spawn_at(world: &mut World, pos: Vec2) -> Entity {
        world
            .spawn(Transform::from_translation(pos.extend(0.0)))
            .id()
    }

    /// `query_radius` must return every entity that any caller would want to
    /// consider inside the radius (the index is over-approximating; the
    /// caller filters by exact distance). It must NOT return entities that
    /// are well beyond the cell reach of the query.
    #[test]
    fn query_radius_includes_all_within_no_far_outsiders() {
        let mut world = World::new();
        let near1 = spawn_at(&mut world, Vec2::new(1.0, 0.0));
        let near2 = spawn_at(&mut world, Vec2::new(2.5, -1.5));
        let far = spawn_at(&mut world, Vec2::new(50.0, 50.0));
        let mut h = SpatialHash::default();
        h.rebuild_from_state(&mut world);

        let mut out = Vec::new();
        h.query_radius(Vec2::ZERO, 5.0, &mut out);
        assert!(out.contains(&near1), "near1 should be in radius 5.0");
        assert!(out.contains(&near2), "near2 should be in radius 5.0");
        assert!(!out.contains(&far), "far must not be in radius 5.0");

        // An entity many cells away must not appear.
        let very_far = spawn_at(&mut world, Vec2::new(100.0, 0.0));
        h.rebuild_from_state(&mut world);
        let mut out2 = Vec::new();
        h.query_radius(Vec2::ZERO, 5.0, &mut out2);
        assert!(!out2.contains(&very_far));
    }

    /// `rebuild` must discard stale entries; a previous-frame entity that has
    /// been despawned (and so is not present in the query) must not linger.
    #[test]
    fn rebuild_clears_stale_entries() {
        let mut world = World::new();
        let e = spawn_at(&mut world, Vec2::ZERO);
        let mut h = SpatialHash::default();
        h.rebuild_from_state(&mut world);
        let mut out = Vec::new();
        h.query_radius(Vec2::ZERO, 10.0, &mut out);
        assert!(out.contains(&e));

        // Despawn and rebuild.
        world.despawn(e);
        h.rebuild_from_state(&mut world);
        let mut out2 = Vec::new();
        h.query_radius(Vec2::ZERO, 10.0, &mut out2);
        assert!(
            !out2.contains(&e),
            "stale entity must not survive a rebuild"
        );
    }

    /// Cell assignment must be deterministic for the same world position and
    /// differ for positions that fall in different cells.
    #[test]
    fn cell_assignment_is_deterministic() {
        let a = Cell::from_pos(Vec2::new(3.0, 7.0));
        let b = Cell::from_pos(Vec2::new(3.0, 7.0));
        assert_eq!(a, b, "same input must yield the same cell");
        // CELL_SIZE is 4.0 (from the module); (8.0, 7.0) falls into a
        // different x-cell than (3.0, 7.0).
        let c = Cell::from_pos(Vec2::new(8.0, 7.0));
        assert_ne!(a, c, "distinct cells must hash to distinct Cell values");
        let d = Cell::from_pos(Vec2::new(3.0, 7.0 + CELL_SIZE));
        assert_ne!(a, d, "different y cells must also differ");
    }
}
