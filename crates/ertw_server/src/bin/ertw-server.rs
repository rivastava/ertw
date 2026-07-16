use ertw_core::ErtwWorld;
use ertw_server::RemoteAgent;
use std::net::TcpListener;
use std::time::{Duration, Instant};

fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);
    let address = args.next().unwrap_or_else(|| "127.0.0.1:9000".to_owned());
    let steps = args
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(600);
    let seed = args
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0xC0FFEE);

    let listener = TcpListener::bind(&address)?;
    println!(
        "ERTW protocol v{0} listening on {address}",
        ertw_core::PROTOCOL_VERSION
    );
    let (stream, peer) = listener.accept()?;
    println!("agent connected from {peer}");

    let mut world = ErtwWorld::new(seed);
    world.spawn_agent(Box::new(RemoteAgent::new(stream)), bevy::math::Vec2::ZERO);

    let tick = Duration::from_secs_f32(ertw_core::economy::FIXED_DT);
    for _ in 0..steps {
        let started = Instant::now();
        world.step(1);
        if let Some(remaining) = tick.checked_sub(started.elapsed()) {
            std::thread::sleep(remaining);
        }
    }
    let snapshots = ertw_core::lineage::collect_competence(world.app().world_mut(), steps);
    println!(
        "completed {steps} steps; recorded {} agent outcome(s)",
        snapshots.len()
    );
    Ok(())
}
