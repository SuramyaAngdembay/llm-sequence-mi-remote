# Magnolia Bubblewrap Note

On Magnolia login nodes, some sandboxed Codex commands fail with:

```text
bwrap: Creating new namespace failed, likely because the kernel does not support user namespaces.
```

This is not a repo bug. It is a host/runtime restriction:

- the Codex sandbox uses `bubblewrap`
- `bubblewrap` needs unprivileged user namespaces, or a setuid-root `bwrap`
- Magnolia currently does not permit that in this environment

## What This Means

- normal shell commands may still work
- some Codex tool invocations that require sandbox setup will fail before the command itself runs
- this cannot be fixed from project code alone

## Practical Workarounds

1. Use escalated Codex command execution when needed.
2. Prefer repo-local scripts and batch jobs over many tiny sandboxed commands.
3. For remote GitHub/Anvil work, do the real execution on Anvil and sync back small CSV/MD artifacts.
4. If locale warnings are distracting, prefer:

```bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

or, if that locale is unavailable:

```bash
export LANG=C
export LC_ALL=C
```

This only cleans locale warnings. It does not fix the namespace failure.

## Real Host-Level Fixes

A system admin would need one of:

- enable unprivileged user namespaces
- install `bwrap` setuid-root
- run Codex with sandboxing disabled or in a different execution mode

## Recommendation For This Project

Treat Magnolia as a partially sandbox-hostile control node:

- do local file edits in the repo
- use escalated commands when the sandbox wrapper blocks execution
- use Anvil for heavier or more reproducible execution paths
