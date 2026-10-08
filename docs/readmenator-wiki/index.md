# Second Brain

*Last synthesized: 2026-10-07 | 7 files | 3 concept pages | offline, zero tokens*

> Raw sources -> readmenator wiki -> links (Karpathy LLM Wiki Pattern, deterministic).
> Start here, then open one community page. Prefer grep over full reads.

## Vault Overview

The codebase centres on `agi.py`, `unificado.py`, `voz.py`. Architecturally it is 1 layers, dominant utility (7 files) across 3 import-based communities. Recorded risk surface: 0 security findings and 0 dependency cycles.

Surprising tissue lives between root: agi, root: voz, orphans: 1 extracted cross-community imports and 2 inferred bridges. Follow `connections.json` sorted by strength before refactoring.

Open work clusters around documentation (86% file coverage), 0 security findings, 0 taint paths, and 5 suggested exploration questions in `queries.md`.

## Stats

| Metric | Value |
|--------|-------|
| Files | 7 |
| Symbols | 65 |
| Resolved imports | 8 |
| Languages | py, sh |
| Communities | 3 |
| Doc coverage | 86% (6/7 files) |
| Security findings | 0 |
| Estimated read cost | ~1385 tokens (chars/4, offline so $0) |

## Reading Order

1. Skim Stats and God Nodes below for blast radius.
2. Open the largest community page first, then follow Connections.
3. Use `queries.md` for the next question; log the answer there.

```
grep -rn '<keyword>' index.md community_*.md
readmenator query "<question>" --target readmenator_agi_7xcl_9xo
```

## Concept Wiki

- [root: agi (3 files, cohesion 0.40)](./community_0_root_agi.md)
- [root: voz (3 files, cohesion 0.40)](./community_1_root_voz.md)
- [orphans (1 files, cohesion 0.00)](./community_2_orphans.md)

## God Nodes

| File | Score |
|------|-------|
| `agi.py` | 12.5 |
| `unificado.py` | 6.5 |
| `voz.py` | 5.2 |
| `uni.py` | 4.2 |
| `super_casette.py` | 3.5 |

## Strongest Connections

- 1 -> 0: depends_on (strength 0.9, EXTRACTED)
- 0 -> 2: shares_context (strength 0.5, INFERRED)
- 1 -> 2: shares_context (strength 0.5, INFERRED)

## Navigation Tips

- Obsidian Graph View works: every community page links back here.
- `connections.json` is machine-readable for GraphRAG pipelines.
- `REPORT.md` states what was extracted vs inferred and current limits.
- Regenerate offline: `readmenator . --rebuild` (no network, no tokens).
