# Security Policy

## Supported versions

SCEMonteCarlo.jl is a numerical Julia package. Only the latest released minor
version (or, before the first release, `main`) receives fixes.

## Reporting a vulnerability

If you believe you have found a security issue (e.g., a vulnerability in a
dependency that affects users of this package, code execution from untrusted
input files, or similar), please **do not** open a public GitHub issue.

Instead, email the maintainer:

- **T. Tanaka** — `tomonori.tanaka.academic@gmail.com`

Please include a description of the issue and its impact, steps to reproduce
(ideally a minimal example), and the affected version(s). You should expect an
acknowledgement within a few business days; we will agree on a coordinated
disclosure timeline before any public discussion.

## Scope

SCEMonteCarlo.jl reads fitted-model TOML files (through SCEFitting.jl) and its
own JLD2 checkpoint files. Issues that allow code
execution, denial of service, or data corruption from these inputs are in scope.

Out of scope: general Julia language vulnerabilities, issues in upstream
dependencies (please report those to the respective projects), and
correctness / numerical bugs (please file a regular GitHub issue).
