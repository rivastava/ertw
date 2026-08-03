# ERTW Python client

This dependency-free client implements the ERTW protocol v4 lockstep transport.
It validates framed messages, decodes observations and lifecycle events, encodes
actions, and retains the opaque credentials needed to resume a disconnected
session. Python 3.9 or newer is required.

Install the package from a repository checkout:

```text
python3 -m pip install ./clients/python
```

```python
from ertw_client import Action, LockstepClient

with LockstepClient.connect(("127.0.0.1", 9000)) as client:
    while True:
        decision = client.next_decision()
        client.send_action(decision, Action())
```

`next_decision()` enforces lockstep ordering: the pending action must be sent
before another observation can be requested. Lifecycle events encountered while
waiting are validated and appended to `client.lifecycle`. After a transport
failure, `client.reconnect()` uses the negotiated opaque credentials to resume
the same server session; credentials must never be logged or shared.

Protocol violations raise `ProtocolError`. Socket closure raises `EOFError`,
and operating-system connection failures retain their standard Python socket
exceptions so callers can distinguish transport loss from malformed peer data.

The world supplies no reward, objective, or agent-specific semantics. Client
policies receive the same observation tensor and emit the same action tensor as
every in-process ERTW agent.

Run the conformance tests from the repository root:

```text
PYTHONPATH=clients/python/src python3 -m unittest discover -s clients/python/tests -v
```
