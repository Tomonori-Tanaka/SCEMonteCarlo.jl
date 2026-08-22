# Development targets for SCEMonteCarlo.jl. The package depends on the sibling
# checkout ../SCEFitting.jl by path (`make setup` once per clone). Every test
# target pins JULIA_NUM_THREADS=4: the serial-vs-parallel and PT determinism
# gates are vacuous at one thread, and CI pins the same value.

JULIA ?= julia
THREADS ?= 4
export JULIA_NUM_THREADS := $(THREADS)

.PHONY: setup test-all test-unit test-aqua test-jet test-ci ci-local docs \
        bench-setup bench-kernels bench-sweeps bench-tiling bench-run \
        bench-minimize bench-profile

# Develop the sibling SCEFitting.jl into this environment (once per clone, or
# after the Manifest is removed).
setup:
	$(JULIA) --project -e 'using Pkg; Pkg.develop(path = "../SCEFitting.jl"); Pkg.instantiate()'

# ---- core suite (runtests.jl dispatches on TEST_MODE) -----------------------

test-all:
	TEST_MODE=all $(JULIA) --project -e 'using Pkg; Pkg.test()'

test-unit:
	TEST_MODE=unit $(JULIA) --project -e 'using Pkg; Pkg.test()'

test-aqua:
	TEST_MODE=aqua $(JULIA) --project -e 'using Pkg; Pkg.test()'

test-jet:
	TEST_MODE=jet $(JULIA) --project -e 'using Pkg; Pkg.test()'

# Strict Documenter build (checkdocs = :exports); executes the guide examples
# (-t 4: the PT examples sweep their lanes over threads). SCEFitting must sit
# at ../SCEFitting.jl (docs/Project.toml resolves it by path).
docs:
	$(MAKE) -C docs build

# ---- CI parity ---------------------------------------------------------------

# Exactly the jobs GitHub Actions runs (tests on both OSes collapse to this
# one, plus the docs build). Run before a release or version bump. GPU device
# validation is NOT in CI: bench/bench_gpu.jl / bench_gpu_pt.jl on a CUDA
# node, recorded in .claude/bench_log.md.
test-ci: test-all docs

# Cold-start reproduction of CI: no cached Manifest, juliaup `release`
# channel. `juliaup add release` if missing.
ci-local:
	rm -f Manifest.toml
	$(JULIA) +release --project -e 'using Pkg; Pkg.develop(path = "../SCEFitting.jl"); Pkg.instantiate()'
	TEST_MODE=all $(JULIA) +release --project -e 'using Pkg; Pkg.test()'
	$(MAKE) docs

# ---- benchmarks (bench/README.md documents fixtures, arguments, and how to
#      localize a bottleneck) ---------------------------------------------------

bench-setup:
	$(JULIA) --project=bench -e 'using Pkg; Pkg.develop([PackageSpec(path = "."), PackageSpec(path = "../SCEFitting.jl")]); Pkg.instantiate()'

bench-kernels:
	$(JULIA) --project=bench bench/bench_kernels.jl

bench-sweeps:
	$(JULIA) --project=bench bench/bench_sweeps.jl

bench-tiling:
	$(JULIA) --project=bench bench/bench_tiling.jl

# PT thread scaling needs more than the default 4 threads: THREADS=8 make bench-run
bench-run:
	$(JULIA) --project=bench bench/bench_run.jl

bench-minimize:
	$(JULIA) --project=bench bench/bench_minimize.jl

# Line-level hotspots: make bench-profile PROFILE_ARGS="sweep 2141 10"
bench-profile:
	$(JULIA) --project=bench bench/bench_profile.jl $(PROFILE_ARGS)
