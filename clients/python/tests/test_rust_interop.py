import os
from pathlib import Path
import socket
import subprocess
import time
import unittest

from ertw_client import Action, LockstepClient


@unittest.skipUnless(
    os.environ.get("ERTW_RUN_RUST_INTEROP") == "1",
    "set ERTW_RUN_RUST_INTEROP=1 to launch the Rust reference server",
)
class RustInteropTests(unittest.TestCase):
    def test_lockstep_decisions_survive_reconnect(self):
        repository = Path(__file__).resolve().parents[3]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        address = ("127.0.0.1", port)
        process = subprocess.Popen(
            [
                "cargo",
                "run",
                "--quiet",
                "-p",
                "ertw_server",
                "--bin",
                "ertw-lockstep",
                "--",
                f"{address[0]}:{address[1]}",
                "3",
                "42",
                "1",
            ],
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        client = None
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    output = process.stdout.read() if process.stdout else ""
                    self.fail(f"reference server exited during startup:\n{output}")
                try:
                    client = LockstepClient.connect(address, timeout=1.0)
                    break
                except OSError:
                    time.sleep(0.05)
            if client is None:
                self.fail("reference server did not accept a connection")

            first = client.next_decision()
            client.send_action(first, Action(force_x=0.25))
            credentials = client.resume_credentials
            client.reconnect(timeout=10.0)
            self.assertEqual(client.resume_credentials, credentials)

            second = client.next_decision()
            client.send_action(second, Action(force_y=-0.25))
            third = client.next_decision()
            client.send_action(third, Action())
            self.assertEqual(
                [
                    first.observation.step,
                    second.observation.step,
                    third.observation.step,
                ],
                [0, 1, 2],
            )
            client.close()
            client = None

            output, _ = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, output)
            self.assertIn("completed 3 physics ticks", output)
        finally:
            if client is not None:
                client.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
