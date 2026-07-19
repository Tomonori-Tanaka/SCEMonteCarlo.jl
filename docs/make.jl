using SCEMonteCarlo
using SCEFitting   # the SCE fitting core, for the executed `@example` model builds
using Documenter

DocMeta.setdocmeta!(SCEMonteCarlo, :DocTestSetup, :(using SCEMonteCarlo);
                    recursive = true)

makedocs(;
    sitename = "SCEMonteCarlo.jl",
    modules = [SCEMonteCarlo],
    # The SCEFitting dependency is a path-dev without a resolvable remote in this
    # build, so per-line source/edit links stay disabled; the navbar links to the
    # repository (private: github.com/Tomonori-Tanaka/SCEMonteCarlo.jl).
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.MathJax3(),
        edit_link = nothing,
        repolink = "https://github.com/Tomonori-Tanaka/SCEMonteCarlo.jl",
        footer = "Built with [Documenter.jl](https://documenter.juliadocs.org).",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Tutorials" => [
            "tutorials/cubic_heisenberg.md",
        ],
        "Guide" => [
            "guide/running.md",
            "guide/parallel_tempering.md",
            "guide/ground_states.md",
            "guide/parallelism.md",
            "guide/gpu.md",
            "guide/observables.md",
            "guide/checkpointing.md",
        ],
        "Theory" => [
            "theory/updates.md",
            "theory/binning.md",
        ],
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
    doctest = false,
)
