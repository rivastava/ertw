# Changelog

All notable changes are documented here. ERTW follows Semantic Versioning once
public versions are tagged.

## Unreleased

- Added the native `ertw` observer binary and fixed native event-loop startup.
- Enabled embedded egui fonts and gizmo rendering, and configured a world-scale
  orthographic camera so the HUD text and simulation nodes are visible.
- Unified ERTW and Avian physics under one gated fixed-step schedule.
- Added direct kinetic and electromagnetic forces and contact-impulse damage.
- Made clamp a physical two-body joint and made fabrication conserve mass and
  energy.
- Added spatial shelter protection and sustained-surplus reproduction with
  independent child controllers.
- Added deterministic active-chunk loading and unloading without population
  injection.
- Added historical dead-agent outcomes to external evaluation.
- Introduced protocol v3 with lossless tags, full identifiers, framed messages,
  egocentric observations, field gradients, and non-blocking remote I/O.
- Added the `ertw-server` executable for fixed-rate external-agent integration
  and validated it end-to-end with an independently sourced Python NEAT agent.
