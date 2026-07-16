# Security policy

ERTW is pre-1.0 research software. Security fixes are supported on the current
main branch.

Do not open a public issue for a vulnerability that could enable code execution,
denial of service across a network boundary, protocol memory exhaustion, or
exposure of private experiment data. Report it with a private GitHub Security
Advisory for the repository. Include affected versions, reproduction steps,
impact, and any proposed mitigation.

The TCP bridge is not an authentication or authorization boundary. Run it only
on a trusted interface or behind an appropriately configured secure proxy.
