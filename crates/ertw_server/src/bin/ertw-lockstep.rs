use ertw_core::components::StableId;
use ertw_core::snapshot;
use ertw_core::ErtwWorld;
use ertw_interface::LifecycleKind;
use ertw_server::lockstep::{LockstepConfig, LockstepSession};
use rand::{rngs::OsRng, RngCore};
use std::collections::BTreeSet;
use std::net::TcpListener;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let address = args.next().unwrap_or_else(|| "127.0.0.1:9000".to_owned());
    let decisions = parse_next(&mut args, 600u32);
    let seed = parse_next(&mut args, 0xC0FFEEu64);
    let physics_ticks_per_decision = parse_next(&mut args, 1u32).max(1);
    let snapshot_path = args.next().map(PathBuf::from);

    let listener = TcpListener::bind(&address)?;
    println!(
        "ERTW protocol v{} lockstep listening on {address}",
        ertw_core::PROTOCOL_VERSION
    );
    let (stream, peer) = listener.accept()?;
    println!("agent connected from {peer}");

    let mut operating_system_rng = OsRng;
    let session_id = random_u128(&mut operating_system_rng);
    let world_id = random_u128(&mut operating_system_rng);
    let mut token_material = [0_u8; 32];
    operating_system_rng.fill_bytes(&mut token_material);
    let resume_token = blake3::hash(&token_material).to_hex().to_string();
    let stable_agent_id = 1;
    let config = LockstepConfig {
        physics_ticks_per_decision,
        world_seed: seed,
        world_id,
        session_id,
        stable_agent_id,
        resume_token,
        deltas: true,
        interface_config: ertw_interface::InterfaceConfig::default(),
    };
    let (session, agent) = LockstepSession::new(listener, stream, config)?;
    let mut world = ErtwWorld::new(seed);
    let entity = world.spawn_agent(Box::new(agent), bevy::math::Vec2::ZERO);
    world
        .app()
        .world_mut()
        .entity_mut(entity)
        .insert(StableId(stable_agent_id));

    session.emit(
        0,
        LifecycleKind::EntityAlive,
        None,
        Some("initial controlled entity".into()),
    );
    let mut known_agents = BTreeSet::from([stable_agent_id]);
    let total_ticks = decisions.saturating_mul(physics_ticks_per_decision);
    for tick in 0..total_ticks {
        world.step(1);
        snapshot::ensure_stable_ids(world.app().world_mut());
        let current_agents = {
            let ecs = world.app().world_mut();
            let mut query = ecs.query::<(&StableId, &ertw_core::components::AgentMarker)>();
            query
                .iter(ecs)
                .map(|(id, marker)| (id.0, marker.generation))
                .collect::<Vec<_>>()
        };
        for (id, generation) in current_agents.iter().copied() {
            if !known_agents.contains(&id) && generation > 0 {
                session.emit(
                    u64::from(tick) + 1,
                    LifecycleKind::EntityReproduced,
                    Some(id),
                    Some("new entity in controlled lineage".into()),
                );
            }
        }
        known_agents = current_agents.into_iter().map(|(id, _)| id).collect();
        if world
            .app()
            .world()
            .get::<ertw_core::components::AgentMarker>(entity)
            .is_none()
        {
            session.emit(
                u64::from(tick) + 1,
                LifecycleKind::EntityDied,
                None,
                Some("physical entity no longer exists".into()),
            );
            break;
        }
    }
    let final_tick = world.app().world().resource::<ertw_core::SimClock>().step;
    if let Some(path) = snapshot_path {
        let state = snapshot::capture(
            world.app().world_mut(),
            Some(format!("session:{session_id}")),
        )?;
        let hash = state.save(&path)?;
        println!("snapshot={} hash={hash}", path.display());
    }
    session.emit(
        final_tick,
        LifecycleKind::WorldTerminated,
        None,
        Some("configured decision budget exhausted or entity died".into()),
    );
    println!("completed {final_tick} physics ticks");
    Ok(())
}

fn parse_next<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, default: T) -> T {
    args.next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn random_u128(rng: &mut impl RngCore) -> u128 {
    u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64())
}
