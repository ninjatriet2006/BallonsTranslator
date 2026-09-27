# AGENTS.md

## Project Shape

BallonsTranslator is a PyQt/qtpy desktop app for comic image translation.

Important areas:
- `ballontranslator/launch.py`: startup, dependency checks, Qt setup, headless mode.
- `ballontranslator/ui/`: Qt UI, canvas, panels, module manager, worker threads.
- `ballontranslator/modules/`: pluggable detector/OCR/translator/inpainter implementations.
- `ballontranslator/utils/proj_imgtrans.py`: project persistence and image/textblock state.
- `ballontranslator/utils/textblock.py`: central TextBlock domain object.
- `ballontranslator/utils/config.py`: persistent config and module settings.

## Refactoring Rules

- Preserve behavior unless explicitly asked to change it.
- Prefer small, reviewable refactors over broad rewrites.
- Do not change public project JSON shape without migration/backward compatibility.
- Be careful with Qt signal/thread behavior in `ballontranslator/ui/module_manager.py`.
- Do not rename registered module keys unless compatibility aliases are added.
- Keep model-loading lazy/eager behavior intact.
- Keep module selection lazy/config-only. Data needed by the config UI before module initialization must come from lazy metadata or `SafeEval`-compatible pure helpers, not from `__init__`, `_setup_*`, `update_*`, `flush`, model loading, downloads, or network calls.
- Avoid adding dependencies unless approved.

## New Feature Rules

- Start with the existing architecture. Prefer extending `ballontranslator/ui/`, `ballontranslator/modules/`, `ballontranslator/utils/config.py`, and `ballontranslator/utils/proj_imgtrans.py` patterns before introducing new frameworks or global services.
- Keep features behind explicit config, UI controls, or module parameters when behavior may surprise existing users.
- Preserve project JSON compatibility. If a feature adds saved fields, provide defaults for old projects and avoid breaking older project files.
- For new automation modules, use the existing registry pattern and stable module keys. Do not rename existing keys without compatibility aliases.
- Keep UI work responsive. Long-running OCR, translation, inpainting, IO, downloads, and model loading must not block the Qt main thread.
- Respect headless mode. If a feature affects the translation pipeline, make sure it works or safely no-ops under `--headless`.
- Avoid mandatory new dependencies. Optional integrations should fail gracefully with a clear error or setup message.
- Keep model/download behavior explicit. Do not download large files or contact online services without an existing module/config path or user action.
- Add focused tests or import checks for non-UI logic. For UI-heavy changes, document the manual verification performed.
- Keep user data safe. Do not overwrite source images, existing translations, masks, or project JSON without following existing save/backup behavior.
- Preserve localization. New visible UI strings should use Qt translation patterns already used in the surrounding code.
- Prefer incremental delivery. Large features should be split into domain/config, pipeline, UI, and persistence changes where practical.

## Code Comment Rules
- Include a standard Python >>> doctest snippet in the docstring of core classes and complex functions.
- Add the minimum comments needed to make code review efficient.
- Comment non-obvious intent, invariants, compatibility constraints, and failure modes.
- For Qt threading, signals, model loading, project JSON compatibility, and file IO, add short comments when the ordering or side effect is important.
- Prefer comments that explain why code is structured a certain way, not what each line does.
- Do not add boilerplate comments, redundant docstrings, or comments that merely repeat function or variable names.
- When refactoring complex logic, add a brief comment before the extracted block if it preserves a subtle behavior from the old implementation.

## Done Criteria For Features

- Existing workflows still run.
- New behavior is configurable or clearly discoverable.
- Old projects load without errors.
- Relevant checks were run, or limitations are stated.

## Verification

For narrow Python changes:
- Run targeted import checks where possible.
- Run relevant tests if available.
- For UI/threading changes, explain what was not practically verified.

Use `rg` for repo search.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **BallonsTranslator** (20828 symbols, 39626 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/BallonsTranslator/context` | Codebase overview, check index freshness |
| `gitnexus://repo/BallonsTranslator/clusters` | All functional areas |
| `gitnexus://repo/BallonsTranslator/processes` | All execution flows |
| `gitnexus://repo/BallonsTranslator/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
