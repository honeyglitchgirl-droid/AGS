# AGS-Sci Security Model

## Security target

AGS-Sci v102 uses a fail-closed security model for scientific experiments, AI/plugin boundaries, provenance, and self-evolution. The experiment sandbox is an application-level isolation boundary; it is not a substitute for OS/container isolation against fully hostile code.

## Core controls

- 1D–4D experiments pass dimension/resource admission before execution.
- Experiment workers are isolated and recyclable; timeouts terminate workers.
- Network access is disabled by default and, when explicitly enabled, is HTTPS/domain/IP constrained.
- Untrusted AI inputs and outputs are bounded, JSON-like, finite, and schema validated.
- Discovery simulation budgets are bounded and fail closed.
- Experiment source, AST complexity, stdout, result, and artifact egress are bounded.
- Provenance uses deterministic SHA-256 fingerprints.
- Self-evolution proposals cannot directly modify protected security/evolution/sandbox/release-policy files.
- Experiment state is ephemeral; persistent knowledge is a separate concern.

## Threat boundary

For hostile multi-tenant workloads, run AGS-Sci inside a dedicated OS/container boundary with a non-privileged account, read-only code mounts, cgroups, seccomp/AppArmor/SELinux where available, and an egress firewall.

## Incident principle

A policy violation, malformed scientific result, resource exhaustion, or failed experiment is a contained failure—not permission to relax a security control automatically.
