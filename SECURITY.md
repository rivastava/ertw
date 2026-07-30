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

## Accepted informational advisory

`RUSTSEC-2024-0436` marks `paste` as unmaintained; it does not report a
vulnerability. ERTW receives `paste` transitively through
Avian → Parry → Simba, and that compatible dependency line has no patched
release. The automated audit ignores only this advisory. Revisit the exception
when upgrading the physics stack or when an affected release becomes available.
