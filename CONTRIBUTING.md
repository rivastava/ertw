# Contributing to ERTW

ERTW is a zero-reward research substrate. Contributions must preserve the
separation between simulation state, agent observations, and external
evaluation. The world must never provide a reward, score, objective, or hidden
privileged channel to an agent.

## Development workflow

1. Open an issue for changes to physics semantics, tensor schemas, the wire
   protocol, determinism, or evaluator methodology.
2. Keep changes focused and add tests for every new invariant or bug fix.
3. Run the complete gate before opening a pull request:

   ```text
   cargo build --workspace --all-targets --all-features
   cargo test --workspace --all-features
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   cargo fmt --all --check
   ```

4. Update `README.md`, `docs/ARCHITECTURE.md`, and `docs/PROTOCOL.md` when
   behavior or public contracts change.

Pull requests should explain the causal mechanism being added, its conservation
or determinism implications, and the evidence used to verify it. Generated
build output, editor state, logs, and credentials must not be committed.

By contributing, you agree that your contribution is licensed under the
project's MIT OR Apache-2.0 terms.
