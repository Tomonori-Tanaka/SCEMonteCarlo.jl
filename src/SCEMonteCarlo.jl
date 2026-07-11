"""
    SCEMonteCarlo

Classical spin Monte Carlo for fitted SCE (symmetry-adapted cluster expansion) models
from `SCEFitting.jl`: tile the fitted training-cell Hamiltonian onto an
`N₁ × N₂ × N₃` supercell (`TiledHamiltonian`) and sample it with single-spin
Metropolis (adaptive step) and overrelaxation sweeps — single temperature, annealing
sweeps (`run_mc`), or replica exchange over threads (`run_pt`) — with composable
observables, autocorrelation-aware binning errors, and bit-reproducible
checkpoint/restart.

The fitted model is read **only** through `SCEFitting`'s public introspection surface
(`multipole_terms`, `n_atoms`, `intercept`, `SCEFitting.Harmonics`); the per-term
`(4π)^(body/2)` scale is applied exactly once, in the `TiledHamiltonian` constructor.
Temperatures are absolute, under exactly one of two keywords: `temperature` [kelvin,
converted with [`KB_EV`](@ref)] or `kT` [model energy units].
"""
module SCEMonteCarlo

using LinearAlgebra
using Printf: @sprintf, @printf
using Random: Random, AbstractRNG, Xoshiro
using StaticArrays
using Statistics: Statistics, mean

using SCEFitting: SCEPredictor, MultipoleTerm, multipole_terms, n_atoms, intercept,
                  Lattice, Crystal, cartesian_positions
import SCEFitting.Harmonics

include("units.jl")
include("hamiltonian.jl")
include("energy.jl")
include("binning.jl")
include("observables.jl")
include("state.jl")
include("updates.jl")
include("run.jl")

export KB_EV
export TiledHamiltonian, n_sites, total_energy
export Observable, Evaluable, ObservableStat, standard_observables,
       standard_evaluables
export run_mc, MCResult, TempResult

public resolve_kt
public ScaledTerm, SpinConfig, site_index, site_atom
public site_coeffs!, delta_energy, site_gradient
public LogBinner, BinStore, jackknife, std_error, tau_int, bin_means
public ChainState, SweepScratch, metropolis_sweep!, overrelaxation_sweep!

end # module SCEMonteCarlo
