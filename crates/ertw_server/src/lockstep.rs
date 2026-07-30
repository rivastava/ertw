//! Slow-agent lockstep transport with action hold, lifecycle frames, and resume.

use crate::{decode_action, encode_header, encode_observation, read_frame, HEADER_BYTES};
use ertw_interface::{
    wire_header, ActionTensor, Agent, InterfaceConfig, LifecycleEvent, LifecycleKind,
    ObservationTensor, PhysicalDelta, ProtocolMetadata, TransportMode, TriggerSemantics,
    WireHeader, ACTION_STRIDE, FIELD_COUNT, FRAME_LIFECYCLE, FRAME_METADATA,
    FRAME_OBSERVATION_EXTENSION, FRAME_RESUME, NEIGHBOR_STRIDE, SELF_STRIDE,
};
use serde::{Deserialize, Serialize};
use std::io::{ErrorKind, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct LockstepConfig {
    pub physics_ticks_per_decision: u32,
    pub world_seed: u64,
    pub world_id: u128,
    pub session_id: u128,
    pub stable_agent_id: u64,
    pub resume_token: String,
    pub deltas: bool,
    pub interface_config: InterfaceConfig,
}

impl LockstepConfig {
    pub fn metadata(&self, world_tick: u64) -> ProtocolMetadata {
        let interface = self.interface_config;
        ProtocolMetadata {
            protocol_version: ertw_core::PROTOCOL_VERSION,
            schema_version: 1,
            fixed_timestep_seconds: ertw_core::economy::FIXED_DT,
            physics_ticks_per_decision: self.physics_ticks_per_decision.max(1),
            observation_floats: interface.observation_len(),
            action_floats: ACTION_STRIDE,
            self_stride: SELF_STRIDE,
            neighbor_stride: NEIGHBOR_STRIDE,
            field_count: FIELD_COUNT,
            max_neighbors: interface.max_neighbors,
            field_samples: interface.field_samples,
            field_channels: interface.field_channels,
            sensor_radius: interface.sensor_radius,
            action_min: [-1.0, -1.0, -1.0, 0.0, 0.0, -16.0, 0.0],
            action_max: [1.0, 1.0, 1.0, 1.0, 1.0, 16.0, std::f32::consts::TAU],
            action_semantics: [
                TriggerSemantics::Continuous,
                TriggerSemantics::Continuous,
                TriggerSemantics::Continuous,
                TriggerSemantics::Level,
                TriggerSemantics::Edge,
                TriggerSemantics::Target,
                TriggerSemantics::Target,
            ],
            transport_mode: TransportMode::Lockstep,
            world_seed: self.world_seed,
            world_tick,
            world_id: self.world_id,
            session_id: self.session_id,
            resume_token: self.resume_token.clone(),
            stable_agent_id: self.stable_agent_id,
            snapshot_schema_version: ertw_core::snapshot::SNAPSHOT_SCHEMA_VERSION,
            capabilities: vec![
                "lockstep".into(),
                "action_hold".into(),
                "lifecycle_events".into(),
                "reconnect_resume".into(),
                "canonical_snapshots".into(),
                "physical_deltas".into(),
            ],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResumeRequest {
    pub session_id: u128,
    pub resume_token: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservationExtension {
    pub decision_sequence: u64,
    pub delta: Option<PhysicalDelta>,
}

struct SessionTransport {
    listener: TcpListener,
    stream: Option<TcpStream>,
    config: LockstepConfig,
    lifecycle_sequence: u64,
    pending_lifecycle: Vec<LifecycleEvent>,
}

impl SessionTransport {
    fn attach(&mut self, mut stream: TcpStream, world_tick: u64) -> std::io::Result<()> {
        stream.set_read_timeout(None)?;
        stream.set_write_timeout(None)?;
        write_json_frame(
            &mut stream,
            FRAME_METADATA,
            world_tick,
            self.config.stable_agent_id,
            &self.config.metadata(world_tick),
        )?;
        for event in self.pending_lifecycle.drain(..) {
            write_json_frame(
                &mut stream,
                FRAME_LIFECYCLE,
                event.world_tick,
                event.subject_id,
                &event,
            )?;
        }
        self.stream = Some(stream);
        self.emit_lifecycle(
            world_tick,
            LifecycleKind::SessionAttached,
            None,
            Some("session attached".into()),
        )
    }

    fn accept_resume(&mut self, world_tick: u64) -> std::io::Result<()> {
        loop {
            let (mut stream, _) = self.listener.accept()?;
            let frame = read_frame(&mut stream)?;
            let header = crate::decode_header(&frame[..HEADER_BYTES]).ok_or_else(|| {
                std::io::Error::new(ErrorKind::InvalidData, "invalid resume header")
            })?;
            if header[2] != FRAME_RESUME {
                continue;
            }
            let request: ResumeRequest = serde_json::from_slice(&frame[HEADER_BYTES..])?;
            if request.session_id == self.config.session_id
                && request.resume_token == self.config.resume_token
            {
                return self.attach(stream, world_tick);
            }
        }
    }

    fn emit_lifecycle(
        &mut self,
        world_tick: u64,
        kind: LifecycleKind,
        related_id: Option<u64>,
        reason: Option<String>,
    ) -> std::io::Result<()> {
        self.lifecycle_sequence = self.lifecycle_sequence.saturating_add(1);
        let event = LifecycleEvent {
            sequence: self.lifecycle_sequence,
            world_tick,
            kind,
            subject_id: self.config.stable_agent_id,
            related_id,
            lineage_id: None,
            generation: None,
            reason,
        };
        if let Some(stream) = &mut self.stream {
            write_json_frame(
                stream,
                FRAME_LIFECYCLE,
                world_tick,
                self.config.stable_agent_id,
                &event,
            )?;
        } else {
            self.pending_lifecycle.push(event);
        }
        Ok(())
    }

    fn exchange(
        &mut self,
        observation: &ObservationTensor,
        extension: &ObservationExtension,
    ) -> std::io::Result<ActionTensor> {
        loop {
            if self.stream.is_none() {
                self.accept_resume(observation.step)?;
            }
            let result = (|| {
                let stream = self.stream.as_mut().expect("stream checked");
                stream.write_all(&encode_observation(
                    ertw_core::PROTOCOL_VERSION,
                    observation,
                ))?;
                write_json_frame(
                    stream,
                    FRAME_OBSERVATION_EXTENSION,
                    observation.step,
                    self.config.stable_agent_id,
                    extension,
                )?;
                let frame = read_frame(stream)?;
                let (step, entity, action) = decode_action(&frame).ok_or_else(|| {
                    std::io::Error::new(ErrorKind::InvalidData, "invalid lockstep action")
                })?;
                if step != observation.step || entity != observation.entity_id {
                    return Err(std::io::Error::new(
                        ErrorKind::InvalidData,
                        "action does not match observation tick",
                    ));
                }
                Ok(action)
            })();
            match result {
                Ok(action) => return Ok(action),
                Err(error) if is_disconnect(&error) => {
                    self.stream = None;
                    self.emit_lifecycle(
                        observation.step,
                        LifecycleKind::SessionDetached,
                        None,
                        Some("transport disconnected; world paused".into()),
                    )?;
                }
                Err(error) if error.kind() == ErrorKind::InvalidData => continue,
                Err(error) => return Err(error),
            }
        }
    }
}

pub struct LockstepAgent {
    transport: Arc<Mutex<SessionTransport>>,
    held_action: ActionTensor,
    remaining_ticks: u32,
    decision_sequence: u64,
    previous_physical: Option<[f32; 3]>,
}

#[derive(Clone)]
pub struct LockstepSession {
    transport: Arc<Mutex<SessionTransport>>,
}

impl LockstepSession {
    pub fn new(
        listener: TcpListener,
        initial_stream: TcpStream,
        config: LockstepConfig,
    ) -> std::io::Result<(Self, LockstepAgent)> {
        let transport = Arc::new(Mutex::new(SessionTransport {
            listener,
            stream: None,
            config,
            lifecycle_sequence: 0,
            pending_lifecycle: Vec::new(),
        }));
        transport
            .lock()
            .map_err(poisoned)?
            .attach(initial_stream, 0)?;
        Ok((
            Self {
                transport: transport.clone(),
            },
            LockstepAgent {
                transport,
                held_action: ActionTensor::default(),
                remaining_ticks: 0,
                decision_sequence: 0,
                previous_physical: None,
            },
        ))
    }

    pub fn emit(
        &self,
        world_tick: u64,
        kind: LifecycleKind,
        related_id: Option<u64>,
        reason: Option<String>,
    ) {
        if let Ok(mut transport) = self.transport.lock() {
            let _ = transport.emit_lifecycle(world_tick, kind, related_id, reason);
        }
    }
}

impl Agent for LockstepAgent {
    fn act(&mut self, observation: &ObservationTensor) -> ActionTensor {
        if self.remaining_ticks > 0 {
            self.remaining_ticks -= 1;
            let mut held = self.held_action;
            held.fabricate = 0.0;
            return held;
        }
        let physical = [
            observation.self_state[4],
            observation.self_state[3],
            observation.self_state[2],
        ];
        let delta = self.previous_physical.map(|previous| PhysicalDelta {
            energy: physical[0] - previous[0],
            structure: physical[1] - previous[1],
            mass: physical[2] - previous[2],
        });
        self.previous_physical = Some(physical);
        self.decision_sequence = self.decision_sequence.saturating_add(1);
        let extension = ObservationExtension {
            decision_sequence: self.decision_sequence,
            delta: self
                .transport
                .lock()
                .ok()
                .filter(|transport| transport.config.deltas)
                .and(delta),
        };
        let action = self
            .transport
            .lock()
            .ok()
            .and_then(|mut transport| transport.exchange(observation, &extension).ok())
            .unwrap_or_default();
        self.held_action = action;
        let ticks = self.transport.lock().ok().map_or(1, |transport| {
            transport.config.physics_ticks_per_decision.max(1)
        });
        self.remaining_ticks = ticks.saturating_sub(1);
        action
    }
}

pub fn write_json_frame<T: Serialize>(
    stream: &mut TcpStream,
    kind: u32,
    step: u64,
    entity_id: u64,
    value: &T,
) -> std::io::Result<()> {
    let payload = serde_json::to_vec(value)?;
    let header = encode_header(wire_header(WireHeader {
        version: ertw_core::PROTOCOL_VERSION,
        frame_kind: kind,
        frame_bytes: (HEADER_BYTES + payload.len()) as u32,
        step,
        entity_id,
        ..Default::default()
    }));
    stream.write_all(&header)?;
    stream.write_all(&payload)
}

fn is_disconnect(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        ErrorKind::BrokenPipe
            | ErrorKind::ConnectionAborted
            | ErrorKind::ConnectionReset
            | ErrorKind::UnexpectedEof
    )
}

fn poisoned<T>(_: std::sync::PoisonError<T>) -> std::io::Error {
    std::io::Error::other("lockstep session mutex poisoned")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ertw_interface::FRAME_OBSERVATION;
    use std::thread;

    fn next_kind(stream: &mut TcpStream) -> (u32, Vec<u8>) {
        let frame = read_frame(stream).expect("frame");
        let header = crate::decode_header(&frame[..HEADER_BYTES]).expect("header");
        (header[2], frame)
    }

    fn read_until(stream: &mut TcpStream, expected: u32) -> Vec<u8> {
        loop {
            let (kind, frame) = next_kind(stream);
            if kind == expected {
                return frame;
            }
        }
    }

    #[test]
    fn action_hold_pulses_edges_and_session_resumes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let config = LockstepConfig {
            physics_ticks_per_decision: 2,
            world_seed: 9,
            world_id: 10,
            session_id: 11,
            stable_agent_id: 12,
            resume_token: "resume-secret".into(),
            deltas: true,
            interface_config: InterfaceConfig::default(),
        };
        let client = thread::spawn(move || {
            let mut first = TcpStream::connect(address).unwrap();
            read_until(&mut first, FRAME_METADATA);
            read_until(&mut first, FRAME_LIFECYCLE);
            let observation = read_until(&mut first, FRAME_OBSERVATION);
            let header = crate::decode_header(&observation[..HEADER_BYTES]).unwrap();
            read_until(&mut first, FRAME_OBSERVATION_EXTENSION);
            first
                .write_all(&crate::encode_action(
                    ertw_core::PROTOCOL_VERSION,
                    header[4] as u64 | ((header[5] as u64) << 32),
                    header[6] as u64 | ((header[7] as u64) << 32),
                    ActionTensor {
                        force: ertw_interface::Vec2Lite::new(0.5, 0.0),
                        fabricate: 1.0,
                        ..Default::default()
                    },
                ))
                .unwrap();
            drop(first);

            let mut resumed = TcpStream::connect(address).unwrap();
            write_json_frame(
                &mut resumed,
                FRAME_RESUME,
                2,
                12,
                &ResumeRequest {
                    session_id: 11,
                    resume_token: "resume-secret".into(),
                },
            )
            .unwrap();
            read_until(&mut resumed, FRAME_METADATA);
            read_until(&mut resumed, FRAME_LIFECYCLE);
            let observation = read_until(&mut resumed, FRAME_OBSERVATION);
            let header = crate::decode_header(&observation[..HEADER_BYTES]).unwrap();
            read_until(&mut resumed, FRAME_OBSERVATION_EXTENSION);
            resumed
                .write_all(&crate::encode_action(
                    ertw_core::PROTOCOL_VERSION,
                    header[4] as u64 | ((header[5] as u64) << 32),
                    header[6] as u64 | ((header[7] as u64) << 32),
                    ActionTensor::default(),
                ))
                .unwrap();
        });

        let (initial, _) = listener.accept().unwrap();
        let (_session, mut agent) = LockstepSession::new(listener, initial, config).unwrap();
        let mut observation = ObservationTensor::new(InterfaceConfig::default());
        observation.entity_id = 99;
        let first = agent.act(&observation);
        assert_eq!(first.force.x, 0.5);
        assert_eq!(first.fabricate, 1.0);
        observation.step = 1;
        let held = agent.act(&observation);
        assert_eq!(held.force.x, 0.5);
        assert_eq!(held.fabricate, 0.0);
        observation.step = 2;
        let resumed = agent.act(&observation);
        assert_eq!(resumed.to_f32(), ActionTensor::default().to_f32());
        client.join().unwrap();
    }
}
