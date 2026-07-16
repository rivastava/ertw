//! Non-blocking TCP bridge for the ERTW protocol.

use ertw_core::ErtwWorld;
use ertw_interface::{
    wire_header, ActionTensor, Agent, InterfaceConfig, ObservationTensor, WireHeader,
    ACTION_STRIDE, FRAME_ACTION, FRAME_HELLO, FRAME_OBSERVATION, WIRE_HEADER_LEN, WIRE_MAGIC,
};
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::Mutex;

const HEADER_BYTES: usize = WIRE_HEADER_LEN * std::mem::size_of::<u32>();
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
const MAX_ACTION_AGE: u64 = 2;

pub fn encode_hello(version: u8, config: InterfaceConfig) -> Vec<u8> {
    encode_header(wire_header(WireHeader {
        version,
        frame_kind: FRAME_HELLO,
        frame_bytes: HEADER_BYTES as u32,
        max_neighbors: config.max_neighbors as u32,
        field_samples: config.field_samples as u32,
        field_channels: config.field_channels as u32,
        ..Default::default()
    }))
}

pub fn encode_observation(version: u8, obs: &ObservationTensor) -> Vec<u8> {
    let payload = obs.to_f32_vec();
    let frame_bytes = HEADER_BYTES + payload.len() * std::mem::size_of::<f32>();
    let header = wire_header(WireHeader {
        version,
        frame_kind: FRAME_OBSERVATION,
        frame_bytes: frame_bytes as u32,
        step: obs.step,
        entity_id: obs.entity_id,
        max_neighbors: obs.config.max_neighbors as u32,
        neighbor_count: obs
            .neighbors
            .iter()
            .filter(|neighbor| neighbor.valid)
            .count() as u32,
        field_samples: obs.config.field_samples as u32,
        field_channels: obs.config.field_channels as u32,
        payload_floats: payload.len() as u32,
    });
    let mut bytes = encode_header(header);
    bytes.reserve(payload.len() * 4);
    for value in payload {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn encode_action(version: u8, step: u64, entity_id: u64, action: ActionTensor) -> Vec<u8> {
    let frame_bytes = HEADER_BYTES + ACTION_STRIDE * std::mem::size_of::<f32>();
    let header = wire_header(WireHeader {
        version,
        frame_kind: FRAME_ACTION,
        frame_bytes: frame_bytes as u32,
        step,
        entity_id,
        payload_floats: ACTION_STRIDE as u32,
        ..Default::default()
    });
    let mut bytes = encode_header(header);
    for value in action.to_f32() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn decode_action(bytes: &[u8]) -> Option<(u64, u64, ActionTensor)> {
    if bytes.len() < HEADER_BYTES {
        return None;
    }
    let header = decode_header(&bytes[..HEADER_BYTES])?;
    if header[0] != WIRE_MAGIC
        || header[1] != ertw_core::PROTOCOL_VERSION as u32
        || header[2] != FRAME_ACTION
        || header[3] as usize != bytes.len()
        || header[12] as usize != ACTION_STRIDE
        || bytes.len() != HEADER_BYTES + ACTION_STRIDE * 4
    {
        return None;
    }
    let step = header[4] as u64 | ((header[5] as u64) << 32);
    let entity = header[6] as u64 | ((header[7] as u64) << 32);
    let mut action_values = [0.0; ACTION_STRIDE];
    for (index, value) in action_values.iter_mut().enumerate() {
        let offset = HEADER_BYTES + index * 4;
        *value = f32::from_le_bytes(bytes[offset..offset + 4].try_into().ok()?);
    }
    Some((step, entity, ActionTensor::from_f32(&action_values)))
}

fn encode_header(header: [u32; WIRE_HEADER_LEN]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_BYTES);
    for value in header {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_header(bytes: &[u8]) -> Option<[u32; WIRE_HEADER_LEN]> {
    if bytes.len() != HEADER_BYTES {
        return None;
    }
    let mut header = [0; WIRE_HEADER_LEN];
    for (index, value) in header.iter_mut().enumerate() {
        let offset = index * 4;
        *value = u32::from_le_bytes(bytes[offset..offset + 4].try_into().ok()?);
    }
    Some(header)
}

fn read_frame(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut header_bytes = [0; HEADER_BYTES];
    stream.read_exact(&mut header_bytes)?;
    let header = decode_header(&header_bytes)
        .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "invalid frame header"))?;
    let frame_bytes = header[3] as usize;
    if header[0] != WIRE_MAGIC || !(HEADER_BYTES..=MAX_FRAME_BYTES).contains(&frame_bytes) {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "invalid frame size or magic",
        ));
    }
    let mut frame = Vec::with_capacity(frame_bytes);
    frame.extend_from_slice(&header_bytes);
    frame.resize(frame_bytes, 0);
    stream.read_exact(&mut frame[HEADER_BYTES..])?;
    Ok(frame)
}

/// A remote controller whose socket I/O runs on a background thread. Simulation
/// ticks never wait for the network; stale or missing actions resolve to no-op.
pub struct RemoteAgent {
    observations: SyncSender<Vec<u8>>,
    actions: Mutex<Receiver<(u64, u64, ActionTensor)>>,
    last_action: Option<(u64, ActionTensor)>,
}

impl RemoteAgent {
    pub fn new(mut stream: TcpStream) -> Self {
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(100)));
        let _ = stream.set_write_timeout(Some(std::time::Duration::from_millis(100)));
        let (observation_tx, observation_rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let (action_tx, action_rx) = mpsc::channel::<(u64, u64, ActionTensor)>();
        std::thread::spawn(move || {
            let hello = encode_hello(ertw_core::PROTOCOL_VERSION, InterfaceConfig::default());
            if stream.write_all(&hello).is_err() {
                return;
            }
            while let Ok(observation) = observation_rx.recv() {
                if stream.write_all(&observation).is_err() {
                    break;
                }
                match read_frame(&mut stream) {
                    Ok(frame) => {
                        if let Some(action) = decode_action(&frame) {
                            let _ = action_tx.send(action);
                        }
                    }
                    Err(error)
                        if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
                    Err(_) => break,
                }
            }
        });
        Self {
            observations: observation_tx,
            actions: Mutex::new(action_rx),
            last_action: None,
        }
    }
}

impl Agent for RemoteAgent {
    fn act(&mut self, observation: &ObservationTensor) -> ActionTensor {
        let _ = self
            .observations
            .try_send(encode_observation(ertw_core::PROTOCOL_VERSION, observation));
        if let Ok(actions) = self.actions.lock() {
            while let Ok((step, entity, action)) = actions.try_recv() {
                if entity == observation.entity_id && step <= observation.step {
                    self.last_action = Some((step, action));
                }
            }
        }
        self.last_action
            .filter(|(step, _)| observation.step.saturating_sub(*step) <= MAX_ACTION_AGE)
            .map(|(_, action)| action)
            .unwrap_or_default()
    }
}

pub fn serve_one(world: &mut ErtwWorld, addr: &str, steps: u32) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    let (stream, _) = listener.accept()?;
    world.spawn_agent(Box::new(RemoteAgent::new(stream)), bevy::math::Vec2::ZERO);
    world.step(steps);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_frame_round_trips_with_full_ids() {
        let action = ActionTensor {
            force: ertw_interface::Vec2Lite::new(0.25, -0.5),
            torque: 0.75,
            clamp: 1.0,
            fabricate: 0.0,
            osc_freq: 3.0,
            osc_phase: 1.5,
        };
        let bytes = encode_action(
            ertw_core::PROTOCOL_VERSION,
            0x1234_5678_9ABC_DEF0,
            0xFEDC_BA98_7654_3210,
            action,
        );
        let (step, entity, decoded) = decode_action(&bytes).expect("valid action frame");
        assert_eq!(step, 0x1234_5678_9ABC_DEF0);
        assert_eq!(entity, 0xFEDC_BA98_7654_3210);
        assert_eq!(decoded.to_f32(), action.to_f32());
    }

    #[test]
    fn malformed_action_frames_are_rejected() {
        let valid = encode_action(ertw_core::PROTOCOL_VERSION, 1, 2, ActionTensor::default());
        assert!(decode_action(&valid[..valid.len() - 1]).is_none());
        let mut bad_magic = valid.clone();
        bad_magic[0] ^= 0xFF;
        assert!(decode_action(&bad_magic).is_none());
        let mut bad_kind = valid;
        bad_kind[8..12].copy_from_slice(&FRAME_OBSERVATION.to_le_bytes());
        assert!(decode_action(&bad_kind).is_none());
    }

    #[test]
    fn remote_agent_drops_backpressure_and_rejects_stale_actions() {
        let (observation_tx, _observation_rx) = mpsc::sync_channel(1);
        let (action_tx, action_rx) = mpsc::channel();
        let mut remote = RemoteAgent {
            observations: observation_tx,
            actions: Mutex::new(action_rx),
            last_action: None,
        };
        let mut observation = ObservationTensor::new(InterfaceConfig::default());
        observation.step = 10;
        observation.entity_id = 7;
        let requested = ActionTensor {
            force: ertw_interface::Vec2Lite::new(0.5, -0.25),
            ..Default::default()
        };

        action_tx.send((7, 7, requested)).expect("receiver exists");
        assert_eq!(
            remote.act(&observation).to_f32(),
            ActionTensor::default().to_f32()
        );

        action_tx.send((8, 7, requested)).expect("receiver exists");
        assert_eq!(remote.act(&observation).to_f32(), requested.to_f32());

        // The observation queue is still full; this call must drop the new
        // observation rather than waiting for a network consumer.
        assert_eq!(remote.act(&observation).to_f32(), requested.to_f32());
    }
}
