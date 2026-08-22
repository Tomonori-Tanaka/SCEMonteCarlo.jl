# MCP server setup notes (Claude Code only)

`/.mcp.json` at the repository root declares three project-scoped MCP
servers. These notes cover setup. CI is unaffected.

## 1. GitHub MCP (`@modelcontextprotocol/server-github`)

Structured access to PRs / issues / releases / tags / actions.

- **Requires**: `npx` (Node.js 18+) and a Personal Access Token (PAT).
- **Scopes**: `repo` (read/write) plus `read:org` is usually enough.
  Add `workflow` if you need release / tag operations.
- **Setup**: export `GITHUB_PERSONAL_ACCESS_TOKEN=<token>` in your shell
  (e.g., `.zshrc` / `.bashrc`, or via `direnv`). `.mcp.json` references it
  via `${...}` expansion.
- **When to use**: "review PR #N", "triage issues", "draft release notes". The
  `gh` CLI works too, but the MCP server returns structured responses.

## 2. Context7 (`@upstash/context7-mcp`)

Structured **up-to-date API documentation** for dependencies (Julia /
StaticArrays / KernelAbstractions / JLD2 / Documenter / CUDA / ...).

- **Requires**: `npx` only. No account needed.
- **When to use**: the latest signature of a dependency API; what a
  Documenter option does; a breaking-change scan before a Julia upgrade.

## 3. arXiv MCP (`arxiv-mcp-server`)

Fetches paper abstracts / sections for references.

- **Requires**: `uvx` (`pip install uv`), or `pip install arxiv-mcp-server`.
- **When to use**: confirm an equation in Drautz & Fähnle 2004 (PRB 69,
  104404) or Drautz 2020 (PRB 102, 024104); keep `docs/src/theory/` and the decision records aligned
  with the papers. Notes on each source live under `references/`.

## Skipping servers

Leave `.mcp.json` as-is and reject individual servers at the Claude Code
approval prompt. To disable persistently, add the server name to
`.claude/settings.local.json` under `disableMcpServers`.

## Removal policy

If a PAT is leaked, revoke it on GitHub first and `unset` the shell
environment variable. Never write tokens into `.mcp.json` — always use
`${...}` expansion.
