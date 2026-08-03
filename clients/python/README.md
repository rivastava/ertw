# ERTW Python client

This dependency-free client implements the ERTW protocol v4 lockstep transport.
It validates framed messages, decodes observations and lifecycle events, encodes
actions, and retains the opaque credentials needed to resume a disconnected
session.

```python
from ertw_client import Action, LockstepClient

with LockstepClient.connect(("127.0.0.1", 9000)) as client:
    while True:
        decision = client.next_decision()
        client.send_action(decision, Action())
```

The world supplies no reward, objective, or agent-specific semantics. Client
policies receive the same observation tensor and emit the same action tensor as
every in-process ERTW agent.

Run the conformance tests from the repository root:

```text
PYTHONPATH=clients/python/src python3 -m unittest discover -s clients/python/tests -v
```
