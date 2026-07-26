# dimma_lib

> **Don't delete `CLAUDE.md`.** It looks redundant — it contains only the line
> `@AGENTS.md` — but Claude Code does not discover `AGENTS.md` on its own
> (verified on 2.1.220: the same file was read under the name `CLAUDE.md` and
> ignored under the name `AGENTS.md`). It fails silently, with no warning, so
> removing `CLAUDE.md` makes everything below invisible to Claude Code while
> appearing to work. Other agents (Codex, Cursor, Copilot) read this file
> directly.

## Agent skills

### Issue tracker

Issues and PRDs live as GitHub issues in `Larraguibel/dimma_lib`, managed with
the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label named after itself. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` and `docs/adr/` at the repo root. See
`docs/agents/domain.md`.
