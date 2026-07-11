# Shared fixtures for the unit suite. `MC` aliases the package so internal (non-exported)
# names resolve as `MC._name`.

using SCEMonteCarlo
using SCEFitting
using LinearAlgebra
using Random
using StaticArrays
using Statistics: mean, std
using Test

const MC = SCEMonteCarlo

# Classical Langevin function L(x) = coth(x) − 1/x.
_langevin(x) = coth(x) - 1 / x
