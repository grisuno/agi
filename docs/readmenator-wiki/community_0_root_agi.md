# root: agi

*Community 0 | 3 files | cohesion 0.40*

## Definition

This community groups 3 file(s) rooted at `root` with dominant language py (cohesion 0.40). Central symbols: `ComplexityAnalyzer`, `FusedAGIBrain`, `Grokkit`, `GrokkitRouter`, `KeplerCassette`, `ParityCassette`, `PendulumCassette`, `SuperpositionSAE`. Core file: `agi.py` (25 symbols). Documented purpose: Agentic Grokked Integratted v0.1 - Unified Algorithmic Cassette Model  A modular, composable library for transplanting grokked algorithmic primitives into unifi.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `agi.py` | py | utility | 25 | yes |
| `app.py` | py | utility | 6 | yes |
| `super_casette.py` | py | utility | 15 | yes |

## Key Symbols

- `get_parity_dataset` (function, `agi.py:31`) `def get_parity_dataset(n_bits, k, size)`
- `generate_wave_data` (function, `agi.py:36`) `def generate_wave_data(N, T, c, dt, L, seed)`
- `step` (method, `agi.py:42`) `def step(u_t, u_tm1)`
- `generate_kepler_data` (function, `agi.py:61`) `def generate_kepler_data(num_samples, seed)`
- `generate_and_save_chaotic_pendulum_dataset` (function, `agi.py:82`) `def generate_and_save_chaotic_pendulum_dataset(n_samples, seed)`
- `ParityCassette` (class, `agi.py:88`) `class ParityCassette(Module)`
- `__init__` (method, `agi.py:89`) `def __init__(self, input_dim, hidden_dim)`
- `forward` (method, `agi.py:97`) `def forward(self, x)`
- `WaveCassette` (class, `agi.py:107`) `class WaveCassette(Module)`
- `__init__` (method, `agi.py:108`) `def __init__(self, hidden_dim)`
- `forward` (method, `agi.py:116`) `def forward(self, x)`
- `KeplerCassette` (class, `agi.py:127`) `class KeplerCassette(Module)`
- `__init__` (method, `agi.py:128`) `def __init__(self, hidden_dim)`
- `forward` (method, `agi.py:138`) `def forward(self, x)`
- `PendulumCassette` (class, `agi.py:145`) `class PendulumCassette(Module)`
- `__init__` (method, `agi.py:146`) `def __init__(self, hidden_dim)`
- `forward` (method, `agi.py:157`) `def forward(self, x)`
- `GrokkitRouter` (class, `agi.py:163`) `class GrokkitRouter(Module)`
- `__init__` (method, `agi.py:164`) `def __init__(self, num_domains)`
- `forward` (method, `agi.py:170`) `def forward(self, x)` - Router determinístico basado en características infalibles del input.
- `Grokkit` (class, `agi.py:196`) `class Grokkit(Module)`
- `__init__` (method, `agi.py:197`) `def __init__(self, load_weights)`
- `load_pretrained_weights` (method, `agi.py:213`) `def load_pretrained_weights(self)`
- `forward` (method, `agi.py:237`) `def forward(self, x)`
- `demo_grokkit` (method, `agi.py:244`) `def demo_grokkit()`
- `test_wave` (function, `app.py:25`) `def test_wave(grokkit)` - Test de la ecuación de onda en una malla más fina (N=256) de la que se entrenó (N=32).
- `test_parity` (function, `app.py:41`) `def test_parity(grokkit)` - Test de paridad con inputs de 64 bits (muy más allá de los 3 usados para entrenar).
- `test_kepler` (function, `app.py:65`) `def test_kepler(grokkit)` - Test using the SAME data generation logic as the original training script.
- `generate_kepler_test` (function, `app.py:70`) `def generate_kepler_test(n_samples, seed)`
- `test_pendulum` (function, `app.py:108`) `def test_pendulum(grokkit)` - Test using the REAL saved dataset, not mock data.

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 3
- Cross-boundary resolved imports (EXTRACTED): 3

## Connections

- [EXTRACTED] depends_on community 1 <-> 0 (strength 0.9): Extracted import edge crosses communities: uni.py imports agi.py.
- [INFERRED] shares_context community 0 <-> 2 (strength 0.5): Inferred shared context (layer utility) with no import path between community 0 (root: agi) and community 2 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- What would break if the most connected file in root: agi changed?
- Should root: agi be split, given cohesion 0.40?

## Sources

- `agi.py`
- `app.py`
- `super_casette.py`
