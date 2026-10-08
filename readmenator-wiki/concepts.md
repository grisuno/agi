# Concepts

Second-brain semantic layer: nouns map atomically to file sets (EXTRACTED); verbs aggregate structural edges (INFERRED).

| Concept | Files | Mentions | Top Files |
|---------|-------|----------|-----------|
| `agi` | 5 | 9 | `agi.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py` |
| `con` | 4 | 11 | `agi.py`, `app.py`, `uni.py`, `voz.py` |
| `kepler` | 4 | 6 | `agi.py`, `app.py`, `super_casette.py`, `uni.py` |
| `forward` | 3 | 9 | `agi.py`, `super_casette.py`, `unificado.py` |
| `grokkit` | 3 | 6 | `agi.py`, `app.py`, `unificado.py` |
| `generate` | 3 | 5 | `agi.py`, `app.py`, `voz.py` |
| `para` | 3 | 4 | `app.py`, `super_casette.py`, `voz.py` |
| `parity` | 3 | 4 | `agi.py`, `app.py`, `uni.py` |
| `pendulum` | 3 | 4 | `agi.py`, `app.py`, `uni.py` |
| `wave` | 3 | 4 | `agi.py`, `app.py`, `uni.py` |
| `del` | 3 | 3 | `agi.py`, `super_casette.py`, `voz.py` |
| `demo` | 3 | 3 | `agi.py`, `uni.py`, `voz.py` |
| `dominio` | 3 | 3 | `agi.py`, `app.py`, `uni.py` |
| `get` | 3 | 3 | `agi.py`, `super_casette.py`, `voz.py` |
| `load` | 3 | 3 | `agi.py`, `super_casette.py`, `unificado.py` |
| `cassette` | 2 | 6 | `agi.py`, `app.py` |
| `data` | 2 | 4 | `agi.py`, `app.py` |
| `los` | 2 | 4 | `app.py`, `voz.py` |
| `que` | 2 | 4 | `app.py`, `voz.py` |
| `una` | 2 | 4 | `app.py`, `voz.py` |
| `unified` | 2 | 4 | `agi.py`, `unificado.py` |
| `app` | 2 | 3 | `app.py`, `super_casette.py` |
| `dataset` | 2 | 3 | `agi.py`, `app.py` |
| `expert` | 2 | 3 | `super_casette.py`, `voz.py` |
| `final` | 2 | 3 | `super_casette.py`, `voz.py` |
| `grokked` | 2 | 3 | `agi.py`, `app.py` |
| `robusta` | 2 | 3 | `unificado.py`, `voz.py` |
| `script` | 2 | 3 | `app.py`, `uni.py` |
| `sticas` | 2 | 3 | `agi.py`, `voz.py` |
| `usando` | 2 | 3 | `unificado.py`, `voz.py` |
| `using` | 2 | 3 | `agi.py`, `app.py` |
| `basado` | 2 | 2 | `agi.py`, `voz.py` |
| `bits` | 2 | 2 | `app.py`, `voz.py` |
| `demostraci` | 2 | 2 | `app.py`, `voz.py` |
| `domain` | 2 | 2 | `uni.py`, `voz.py` |
| `input` | 2 | 2 | `agi.py`, `voz.py` |
| `problemas` | 2 | 2 | `uni.py`, `voz.py` |
| `real` | 2 | 2 | `app.py`, `voz.py` |
| `resuelve` | 2 | 2 | `app.py`, `uni.py` |
| `weights` | 2 | 2 | `agi.py`, `super_casette.py` |

## Verb Edges

| Source | Verb | Target | Strength |
|--------|------|--------|----------|
| `agi` | `depends_on` | `forward` | 1.00 |
| `agi` | `depends_on` | `grokkit` | 1.00 |
| `agi` | `depends_on` | `load` | 1.00 |
| `agi` | `depends_on` | `unified` | 1.00 |
| `con` | `depends_on` | `agi` | 0.83 |
| `con` | `depends_on` | `forward` | 0.83 |
| `con` | `depends_on` | `grokkit` | 0.83 |
| `con` | `depends_on` | `load` | 0.83 |
| `con` | `depends_on` | `unified` | 0.83 |
| `agi` | `depends_on` | `basado` | 0.67 |
| `agi` | `depends_on` | `cassette` | 0.67 |
| `agi` | `depends_on` | `con` | 0.67 |
| `agi` | `depends_on` | `data` | 0.67 |
| `agi` | `depends_on` | `dataset` | 0.67 |
| `agi` | `depends_on` | `del` | 0.67 |
| `agi` | `depends_on` | `demo` | 0.67 |
| `agi` | `depends_on` | `dominio` | 0.67 |
| `agi` | `depends_on` | `generate` | 0.67 |
| `agi` | `depends_on` | `get` | 0.67 |
| `agi` | `depends_on` | `grokked` | 0.67 |
| `agi` | `depends_on` | `input` | 0.67 |
| `agi` | `depends_on` | `kepler` | 0.67 |
| `agi` | `depends_on` | `parity` | 0.67 |
| `agi` | `depends_on` | `pendulum` | 0.67 |
| `agi` | `depends_on` | `sticas` | 0.67 |
| `agi` | `depends_on` | `using` | 0.67 |
| `agi` | `depends_on` | `wave` | 0.67 |
| `agi` | `depends_on` | `weights` | 0.67 |
| `demo` | `depends_on` | `agi` | 0.67 |
| `demo` | `depends_on` | `forward` | 0.67 |
| `demo` | `depends_on` | `grokkit` | 0.67 |
| `demo` | `depends_on` | `load` | 0.67 |
| `demo` | `depends_on` | `unified` | 0.67 |
| `domain` | `depends_on` | `agi` | 0.67 |
| `domain` | `depends_on` | `forward` | 0.67 |
| `domain` | `depends_on` | `grokkit` | 0.67 |
| `domain` | `depends_on` | `load` | 0.67 |
| `domain` | `depends_on` | `unified` | 0.67 |
| `kepler` | `depends_on` | `agi` | 0.67 |
| `kepler` | `depends_on` | `forward` | 0.67 |
| `kepler` | `depends_on` | `grokkit` | 0.67 |
| `kepler` | `depends_on` | `load` | 0.67 |
| `kepler` | `depends_on` | `unified` | 0.67 |
| `para` | `depends_on` | `agi` | 0.67 |
| `para` | `depends_on` | `forward` | 0.67 |
| `para` | `depends_on` | `grokkit` | 0.67 |
| `para` | `depends_on` | `load` | 0.67 |
| `para` | `depends_on` | `unified` | 0.67 |
| `problemas` | `depends_on` | `agi` | 0.67 |
| `problemas` | `depends_on` | `forward` | 0.67 |

## Dialectic Prompts

- Thesis: `agi` centralizes 5 files; Antithesis: `basado` pulls 2 files with 2 shared (Jaccard 0.40); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `agi` centralizes 5 files; Antithesis: `con` pulls 4 files with 3 shared (Jaccard 0.50); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `agi` centralizes 5 files; Antithesis: `del` pulls 3 files with 3 shared (Jaccard 0.60); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `agi` centralizes 5 files; Antithesis: `demo` pulls 3 files with 3 shared (Jaccard 0.60); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `agi` centralizes 5 files; Antithesis: `domain` pulls 2 files with 2 shared (Jaccard 0.40); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `agi` centralizes 5 files; Antithesis: `dominio` pulls 3 files with 2 shared (Jaccard 0.33); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `agi` centralizes 5 files; Antithesis: `expert` pulls 2 files with 2 shared (Jaccard 0.40); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `agi` centralizes 5 files; Antithesis: `final` pulls 2 files with 2 shared (Jaccard 0.40); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `agi` centralizes 5 files; Antithesis: `forward` pulls 3 files with 3 shared (Jaccard 0.60); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `agi` centralizes 5 files; Antithesis: `generate` pulls 3 files with 2 shared (Jaccard 0.33); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
