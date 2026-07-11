using SCEMonteCarlo
using SCEFitting   # the SCE fitting core, for the executed `@example` model builds
using Documenter

DocMeta.setdocmeta!(SCEMonteCarlo, :DocTestSetup, :(using SCEMonteCarlo);
                    recursive = true)

makedocs(;
    sitename = "SCEMonteCarlo.jl",
    modules = [SCEMonteCarlo],
    # Local-only build: there is no published remote yet, so do not try to resolve
    # "edit on GitHub" / source links. Add a `repolink`/`deploydocs` when a remote exists.
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.MathJax3(),
        edit_link = nothing,
        repolink = "",
        footer = "Built with [Documenter.jl](https://documenter.juliadocs.org).",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Guide" => [
            "guide/running.md",
            "guide/parallel_tempering.md",
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
