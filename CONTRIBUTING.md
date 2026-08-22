# Contributing to SCEMonteCarlo.jl

Thanks for your interest in contributing. SCEMonteCarlo.jl runs classical spin
Monte Carlo (Metropolis + overrelaxation, replica exchange, ground-state
search, CPU and GPU) on fitted SCE models from the sibling SCEFitting.jl.
Numerical correctness, bit-reproducibility, and physical consistency take
precedence over stylistic refactoring.

## Reporting issues

- **Bugs**: open a GitHub issue using the *Bug report* template. Please
  include a minimal reproducer (the model or how it was built, `dims`, the
  temperature, the seed) and the SCEFitting.jl commit.
- **Feature requests**: use the *Feature request* template.
- **Security issues**: see [SECURITY.md](SECURITY.md).

## Development workflow

1. Clone this repository **and** SCEFitting.jl as siblings (the dependency is
   a path-dev during development; `make setup` wires it).
2. Fork / branch from `main` (`fix/<slug>`, `feat/<slug>`, …).
3. For non-trivial work we use spec folders under `docs/specs/`; a template
   is at [docs/specs/_template/](docs/specs/_template/). The flat files
   there (`hamiltonian-tiling.md`, `updates-stationarity.md`, …) are the
   standing decision records — update the one your change touches.
4. Add or update tests. Numerical changes must come with a regression or
   validation test whose expected value is **independent of the
   implementation**; statistical gates state their tolerance as measured σ
   with headroom. The bitwise contracts (serial ≡ parallel, PT thread
   independence, resume ≡ uninterrupted, CPU ≡ GPU) are never loosened.
5. Run the local checks before opening a PR:
   ```bash
   make test-all      # unit + Aqua + JET (4 threads — required)
   make docs          # strict Documenter build (executes the guide examples)
   ```
   `make test-ci` runs both; `make ci-local` reproduces CI from a cold
   environment. GPU device validation runs on a CUDA node
   (`bench/bench_gpu.jl`, `bench/bench_gpu_pt.jl`) and is recorded in
   `.claude/bench_log.md`.
6. Update documentation as needed: user-facing changes → `docs/src/guide/`;
   new public API → `SPEC.md` and `docs/src/api.md`; `CHANGELOG.md`
   `[Unreleased]`.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/):
`<type>(<scope>): <subject>` — types `feat` / `fix` / `docs` / `test` /
`refactor` / `perf` / `chore` / `style`; imperative lowercase subject;
`BREAKING CHANGE: ...` in the body. Backports from SLCEMonteCarlo.jl cite the
upstream SHA.

## Style

- US English in source, comments, docstrings, commit messages, and PRs.
- See [STYLE_GUIDE.md](STYLE_GUIDE.md) (mutated argument first; `H` / `st` /
  `sc`; "site" vs "atom") on top of the shared Julia style.
- Hot-path guidance in [CLAUDE.md](CLAUDE.md) "Performance guidelines";
  benchmark fixtures and the bottleneck procedure in
  [bench/README.md](bench/README.md).

## Physics conventions

Easy to break silently — confirm before touching the algorithm:

- Spins are unit vectors; the `(4π)^(body/2)` scale is applied exactly once
  in the `TiledHamiltonian` constructor; `j0` is excluded everywhere.
- `temperature` [K] xor `kT` [model units]; β only in accept steps.
- Each directed member is one plain summand — no ½ or 1/N factors.
- Detailed balance of every update; adaptation frozen during measurement.
- Inactive sites: skipped by sweeps, excluded from observables, bitwise frozen.

The full list, and the coupled code sites that move together, are in
[CLAUDE.md](CLAUDE.md).

## Pull requests

- One logical change per PR. Fill in the PR template.
- CI must pass: tests on both operating systems (both are load-bearing —
  libm-dependent statistics) and the docs build.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
