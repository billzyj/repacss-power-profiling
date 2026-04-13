# AGENTS.md

This file defines repository-level interaction rules for assistant sessions in this repository.

## Shared Rules

### Scope

These rules define the shared baseline for assistant sessions across REPACSS stack projects and paper projects.

Each repository must keep its own runtime-visible `AGENTS.md`.
This file is the canonical runtime source for this repository.

### Conversation Language Rule

1. If the user writes in Chinese, assistant replies may be in Chinese.
2. If the user writes in English, assistant replies should be in English.
3. Chinese is allowed for interactive planning and discussion in chat when that helps the user review intermediate work.

### Generated Content Language Rule (Strict)

All assistant-generated project content must be in English unless the user explicitly requests non-English file content.

This includes:

1. Source code comments and docstrings
2. Markdown documents such as `README`, `docs/*`, reports, notes, and analysis drafts
3. Config descriptions, inline help text, and usage examples
4. Commit messages created by the assistant
5. CLI or log text templates written to files
6. Any other content written into the repository or project workspace for later sharing or reuse

### Repo Material Language Rule

Anything persisted into the project workspace should default to English if it may be read, shared, reused, reviewed, or versioned later.

Do not write Chinese into repository or project files unless the user explicitly asks for that exact Chinese output.

### Correction Rule

If any generated project file content is accidentally non-English:

1. Rewrite it to English immediately in the same session.
2. Prefer updating the existing file rather than creating a duplicate.

## Repo-Specific Rules

### Temporary Refactor Plan Rule

1. Temporary refactor planning documents may live under `docs/_working/`.
2. The canonical temporary plan path is `docs/_working/refactor-plan.md`.
3. Temporary planning documents under `docs/_working/` may use mixed Chinese and English when that improves collaboration between the user, Codex, and Claude Code.
4. Temporary planning documents must be clearly written as working artifacts, not stable user-facing documentation.

### Permanent Documentation Rule

1. All permanent repository documentation must remain in English.
2. Any information that should survive after implementation must be moved into English documentation such as `README.md` or `docs/*`.
3. Do not leave mixed-language planning content in stable documentation files.

### Refactor Execution Cleanup Rule

1. Before a refactor branch is considered complete, update the relevant English `README.md` and any affected English docs.
2. After the durable English documentation has been updated, delete the temporary plan file from `docs/_working/`.
3. The final repository state should not depend on temporary planning files for user-facing understanding.

### Validation Rule

1. Refactor plans should include at least one fresh-clone validation path that can run on a remote machine after `git clone`.
2. Prefer a no-credentials smoke test as the minimum validation path when possible.
3. Use repository-native test entrypoints for validation so both Codex and Claude Code can run the same commands.
