//! Backward-compatible example wrapper for the native `ertw` binary.
//!
//! Prefer: `cargo run -p ertw_render --bin ertw --features render`.

#[cfg(feature = "render")]
fn main() {
    ertw_render::run_rendered_sim();
}

#[cfg(not(feature = "render"))]
fn main() {
    eprintln!("Enable rendering with --features render");
}
