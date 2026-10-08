# orphans

*Community 2 | 1 files | cohesion 0.00*

## Definition

This community groups 1 file(s) rooted at `root` with dominant language sh (cohesion 0.00). Central symbols: no extracted symbols.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `install.sh` | sh | utility | 0 | no |

## Key Symbols

- No symbols extracted in this community.

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 2 (strength 0.5): Inferred shared context (layer utility) with no import path between community 0 (root: agi) and community 2 (orphans).
- [INFERRED] shares_context community 1 <-> 2 (strength 0.5): Inferred shared context (layer utility) with no import path between community 1 (root: voz) and community 2 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `install.sh`)? What purpose do they serve?
- What would break if the most connected file in orphans changed?
- Should orphans be split, given cohesion 0.00?

## Sources

- `install.sh`
