use avian2d::prelude::{Collider, Mass, RigidBody};
use bevy::camera::ScalingMode;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use ertw_core::{agents, components, configure_world, SimulationSet};
use random_policy::RandomPolicy;

const DEFAULT_SEED: u64 = 0xC0FFEE;

/// Run the native ERTW observer application.
///
/// The observer can pause and single-step the fixed simulation schedule, but it
/// never supplies observations, actions, rewards, or state mutations to agents.
pub fn run_rendered_sim() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "ERTW — Extensible Relational Tensor World".into(),
            resolution: (1280, 800).into(),
            ..Default::default()
        }),
        ..Default::default()
    }))
    .add_plugins(crate::RenderPlugin)
    .add_plugins(EguiPlugin::default());

    configure_world(&mut app, DEFAULT_SEED);
    spawn_initial_population(&mut app);

    app.insert_resource(SimControl {
        paused: false,
        step_once: false,
        seed: DEFAULT_SEED,
    })
    .insert_resource(ObserverState::default())
    .configure_sets(FixedUpdate, SimulationSet.run_if(should_simulate))
    .add_systems(
        FixedUpdate,
        consume_step.after(SimulationSet).run_if(onestep_active),
    )
    .add_systems(Startup, spawn_camera)
    .add_systems(Update, camera_controls)
    .add_systems(EguiPrimaryContextPass, hud_system)
    .run();
}

fn spawn_initial_population(app: &mut App) {
    let controllers = {
        let mut world_agents = app.world_mut().resource_mut::<agents::WorldAgents>();
        (0..8u64)
            .map(|i| {
                (
                    world_agents.register(Box::new(RandomPolicy::new(0x1000 + i))),
                    Vec2::new((i as f32 - 3.5) * 3.0, (i as f32 % 2.0) * 2.0),
                )
            })
            .collect::<Vec<_>>()
    };

    for (controller, pos) in controllers {
        app.world_mut().commands().spawn(components::AgentBundle {
            transform: Transform::from_translation(pos.extend(0.0)),
            rigid_body: RigidBody::Dynamic,
            collider: Collider::circle(0.5),
            mass: Mass(0.8),
            physical: components::Physical {
                mass: 0.8,
                structure: 8.0,
                energy: 20.0,
            },
            yield_thresh: components::Yield(8.0),
            conductivity: components::Conductivity(0.6),
            tags: components::Tags(ertw_core::tags::CustomTags::from_bits(
                ertw_core::tags::CustomTags::AGENT
                    | ertw_core::tags::CustomTags::CLAMP_CAPABLE
                    | ertw_core::tags::CustomTags::OSCILLATOR,
            )),
            oscillator: components::Oscillator {
                freq: 1.0,
                phase: 0.0,
                baseline_freq: 1.0,
            },
            impulse: components::ImpulseAccum::default(),
            ledger: components::EnergyLedger::default(),
            marker: components::AgentMarker {
                generation: 0,
                lineage: controller ^ 0xABCD,
                controller,
            },
            tuning: components::AgentTuning::default(),
            clamp: components::ClampState::default(),
            fabricate: components::FabricateCooldown::default(),
            reproduction: components::ReproductionState::default(),
            node_rng: components::NodeRng(controller.wrapping_mul(0x9E3779B1)),
        });
    }
    app.world_mut().flush();
}

#[derive(Resource)]
struct SimControl {
    paused: bool,
    step_once: bool,
    seed: u64,
}

#[derive(Resource, Default)]
struct ObserverState {
    selected: Option<Entity>,
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 40.0,
            },
            ..OrthographicProjection::default_2d()
        }),
    ));
}

fn should_simulate(sim: Res<SimControl>) -> bool {
    !sim.paused || sim.step_once
}

fn onestep_active(sim: Res<SimControl>) -> bool {
    sim.step_once
}

fn consume_step(mut sim: ResMut<SimControl>) {
    sim.step_once = false;
}

fn camera_controls(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut camera: Query<(&mut Transform, &mut Projection), With<Camera2d>>,
) {
    let Ok((mut transform, mut projection)) = camera.single_mut() else {
        return;
    };
    let mut direction = Vec2::ZERO;
    if keys.pressed(KeyCode::ArrowLeft) || keys.pressed(KeyCode::KeyA) {
        direction.x -= 1.0;
    }
    if keys.pressed(KeyCode::ArrowRight) || keys.pressed(KeyCode::KeyD) {
        direction.x += 1.0;
    }
    if keys.pressed(KeyCode::ArrowUp) || keys.pressed(KeyCode::KeyW) {
        direction.y += 1.0;
    }
    if keys.pressed(KeyCode::ArrowDown) || keys.pressed(KeyCode::KeyS) {
        direction.y -= 1.0;
    }
    transform.translation += (direction.normalize_or_zero() * 20.0 * time.delta_secs()).extend(0.0);
    if let Projection::Orthographic(orthographic) = projection.as_mut() {
        if keys.pressed(KeyCode::KeyQ) {
            orthographic.scale = (orthographic.scale * (1.0 + time.delta_secs())).min(8.0);
        } else if keys.pressed(KeyCode::KeyE) {
            orthographic.scale = (orthographic.scale * (1.0 - time.delta_secs() * 0.5)).max(0.2);
        }
    }
}

fn hud_system(
    mut contexts: EguiContexts,
    mut sim: ResMut<SimControl>,
    mut observer: ResMut<ObserverState>,
    sampler: Res<ertw_core::fields::FieldSampler>,
    agents: Query<&components::AgentMarker>,
    bodies: Query<(
        Entity,
        &components::Physical,
        &components::Tags,
        &Transform,
        Option<&components::AgentMarker>,
    )>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("ERTW HUD").show(ctx, |ui| {
        ui.heading("ERTW — non-participant observer");
        ui.checkbox(&mut sim.paused, "Paused");
        if ui.button("Step once").clicked() {
            sim.paused = true;
            sim.step_once = true;
        }
        ui.label(format!("Seed: 0x{:X}", sim.seed));
        ui.separator();
        ui.label(format!("Agents: {}", agents.iter().count()));
        ui.label(format!("Nodes:  {}", bodies.iter().count()));
        ui.separator();
        let f = sampler.sample(Vec2::ZERO);
        ui.label(format!(
            "Kinetic {:.3}  Thermal {:.3}  EM {:.3}",
            f.kinetic, f.thermal, f.em
        ));
        ui.label(format!("Field clock t: {:.1}", sampler.time));
        ui.label("Camera: WASD/arrows, Q/E zoom");
        ui.separator();
        ui.heading("Entity inspector");
        egui::ScrollArea::vertical()
            .max_height(150.0)
            .show(ui, |ui| {
                for (entity, physical, _, _, marker) in bodies.iter().take(64) {
                    let kind = if marker.is_some() { "agent" } else { "node" };
                    if ui
                        .selectable_label(
                            observer.selected == Some(entity),
                            format!("{} {kind} E={:.2}", entity.to_bits(), physical.energy),
                        )
                        .clicked()
                    {
                        observer.selected = Some(entity);
                    }
                }
            });
        if let Some(selected) = observer.selected {
            if let Ok((entity, physical, tags, transform, marker)) = bodies.get(selected) {
                ui.separator();
                ui.label(format!("Entity: {}", entity.to_bits()));
                ui.label(format!(
                    "Position: {:.2}, {:.2}",
                    transform.translation.x, transform.translation.y
                ));
                ui.label(format!(
                    "Mass {:.2}  Structure {:.2}  Energy {:.2}",
                    physical.mass, physical.structure, physical.energy
                ));
                ui.label(format!("Tags: 0x{:016X}", tags.0 .0));
                if let Some(marker) = marker {
                    ui.label(format!(
                        "Generation {}  Lineage {}",
                        marker.generation, marker.lineage
                    ));
                }
            } else {
                observer.selected = None;
            }
        }
    });
}
