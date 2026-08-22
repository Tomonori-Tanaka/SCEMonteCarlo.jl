<!-- Thanks for contributing to SCEMonteCarlo.jl. -->

## Summary

<!-- 1-3 sentences: what does this PR do and why. -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would change existing behavior,
      the checkpoint schema, or a dependent-package-facing name)
- [ ] Docs / tests / chore only

## Numerical correctness and determinism

If this PR changes any numerical result or a determinism contract:

- [ ] I have explained *why* the result changes.
- [ ] I have added a regression or validation test whose expected value is
      independent of the implementation (exact identity, hand calculation, or
      a labeled pin); statistical tolerances are measured σ with headroom.
- [ ] The bitwise gates still hold: serial ≡ parallel, PT thread-count
      independence, resume ≡ uninterrupted (mid-measure), CPU ≡ GPU.
- [ ] I have updated the relevant decision record under `docs/specs/`.

If this PR touches a coupled code site (see "Coupled code sites" in
[CLAUDE.md](../CLAUDE.md)):

- [ ] I have updated every site in that bullet together.

## Checks

- [ ] `make test-all` passes locally (4 threads).
- [ ] `make docs` builds (strict).
- [ ] Commit messages follow
      [Conventional Commits](https://www.conventionalcommits.org/).
- [ ] Public API changes are reflected in `SPEC.md` and `docs/src/api.md`.
- [ ] `CHANGELOG.md` `[Unreleased]` updated.

## Related issues

<!-- Closes #123, refs #456. -->
