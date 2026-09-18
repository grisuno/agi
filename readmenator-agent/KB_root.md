# Subsystem: root

## agi.py
- Layer: utility
- Doc: Agentic Grokked Integratted v0.1 - Unified Algorithmic Cassette Model  A modular, composable library for transplanting g
- Language: py
- Symbols:
  - `get_parity_dataset` (function, line 31) `def get_parity_dataset(n_bits, k, size)`
  - `generate_wave_data` (function, line 36) `def generate_wave_data(N, T, c, dt, L, seed)`
  - `generate_kepler_data` (function, line 61) `def generate_kepler_data(num_samples, seed)`
  - `generate_and_save_chaotic_pendulum_dataset` (function, line 82) `def generate_and_save_chaotic_pendulum_dataset(n_samples, seed)`
  - `ParityCassette` (class, line 88) `class ParityCassette(Module)`
  - `WaveCassette` (class, line 107) `class WaveCassette(Module)`
  - `KeplerCassette` (class, line 127) `class KeplerCassette(Module)`
  - `PendulumCassette` (class, line 145) `class PendulumCassette(Module)`
  - `GrokkitRouter` (class, line 163) `class GrokkitRouter(Module)`
  - `Grokkit` (class, line 196) `class Grokkit(Module)`
  - `demo_grokkit` (method, line 244) `def demo_grokkit()`
  - `step` (method, line 42) `def step(u_t, u_tm1)`
  - `__init__` (method, line 89) `def __init__(self, input_dim, hidden_dim)`
  - `forward` (method, line 97) `def forward(self, x)`
  - `__init__` (method, line 108) `def __init__(self, hidden_dim)`
  - `forward` (method, line 116) `def forward(self, x)`
  - `__init__` (method, line 128) `def __init__(self, hidden_dim)`
  - `forward` (method, line 138) `def forward(self, x)`
  - `__init__` (method, line 146) `def __init__(self, hidden_dim)`
  - `forward` (method, line 157) `def forward(self, x)`
  - `__init__` (method, line 164) `def __init__(self, num_domains)`
  - `forward` (method, line 170) `def forward(self, x)`
  - `__init__` (method, line 197) `def __init__(self, load_weights)`
  - `load_pretrained_weights` (method, line 213) `def load_pretrained_weights(self)`
  - `forward` (method, line 237) `def forward(self, x)`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

## app.py
- Layer: utility
- Doc: Demostración Definitiva de Éxito de Grokking en Grokkit  Este script prueba que cada cassette, con sus pesos grokked, re
- Language: py
- Symbols:
  - `test_wave` (function, line 25) `def test_wave(grokkit)`
  - `test_parity` (function, line 41) `def test_parity(grokkit)`
  - `test_kepler` (function, line 65) `def test_kepler(grokkit)`
  - `test_pendulum` (function, line 108) `def test_pendulum(grokkit)`
  - `main` (function, line 134) `def main()`
  - `generate_kepler_test` (function, line 70) `def generate_kepler_test(n_samples, seed)`
- Depends on: `agi.py`

## install.sh
- Layer: utility
- Language: sh

## super_casette.py
- Layer: utility
- Doc: FusedAGI - HYBRIDO FINAL (Arquitectura Casette + Entrenamiento App) Inyección del truco Superposición+LC para resucitar 
- Language: py
- Symbols:
  - `SuperpositionSAE` (class, line 32) `class SuperpositionSAE(Module)`
  - `ComplexityAnalyzer` (class, line 56) `class ComplexityAnalyzer`
  - `FusedAGIBrain` (class, line 78) `class FusedAGIBrain(Module)`
  - `SurgicalFusion` (class, line 121) `class SurgicalFusion`
  - `recovery_fine_tuning` (method, line 193) `def recovery_fine_tuning(brain)`
  - `main` (method, line 276) `def main()`
  - `__init__` (method, line 34) `def __init__(self, d_model, d_sae)`
  - `forward` (method, line 41) `def forward(self, x)`
  - `get_metrics` (method, line 46) `def get_metrics(self, z)`
  - `measure_lc` (method, line 58) `def measure_lc(model, x, epsilon)`
  - `__init__` (method, line 79) `def __init__(self, hidden_dim)`
  - `forward` (method, line 91) `def forward(self, x)`
  - `__init__` (method, line 122) `def __init__(self, weights_dir)`
  - `load_expert_weights` (method, line 131) `def load_expert_weights(self, domain)`
  - `transplant` (method, line 137) `def transplant(self, brain)`
- Depends on: `agi.py`

## uni.py
- Layer: utility
- Doc: AGI v0.1 - Demo Multi-Dominio Resuelve Parity, Wave, Kepler y Pendulum en un solo script.
- Language: py
- Symbols:
  - `batch_multi_domain` (function, line 15) `def batch_multi_domain()`
  - `main` (function, line 38) `def main()`
- Depends on: `agi.py`, `unificado.py`

## unificado.py
- Layer: utility
- Doc: Grokkit v0.1 - Unified Agent with Explicit Path Mapping No parsing, no magic, just works.
- Language: py
- Symbols:
  - `UnifiedGrokkitAgent` (class, line 17) `class UnifiedGrokkitAgent(Module)`
  - `__init__` (method, line 18) `def __init__(self, cassette_paths)`
  - `_load_cassettes` (method, line 33) `def _load_cassettes(self, paths)`
  - `forward` (method, line 68) `def forward(self, x)`
  - `__call__` (method, line 80) `def __call__(self, x)`
- Depends on: `agi.py`
- Imported by: `uni.py`, `voz.py`

## voz.py
- Layer: utility
- Doc: AGI Voice Layer v2.0 - Capa de lenguaje robusta para sistema de expertos AGI Corrección de problemas de routing y genera
- Language: py
- Symbols:
  - `AGIVoiceLayer` (class, line 33) `class AGIVoiceLayer`
  - `demo_voice_layer` (method, line 584) `def demo_voice_layer()`
  - `__init__` (method, line 36) `def __init__(self, model_name, use_unified, debug)`
  - `_extract_binary_number` (method, line 91) `def _extract_binary_number(self, text)`
  - `_extract_k_value` (method, line 114) `def _extract_k_value(self, text)`
  - `_domain_routing_heuristic` (method, line 135) `def _domain_routing_heuristic(self, question)`
  - `_prepare_input_for_expert` (method, line 166) `def _prepare_input_for_expert(self, expert_name, parameters, question)`
  - `_interpret_technical_result` (method, line 242) `def _interpret_technical_result(self, expert_name, result, parameters, question)`
  - `_get_expert_analysis` (method, line 311) `def _get_expert_analysis(self, question)`
  - `_generate_final_response` (method, line 388) `def _generate_final_response(self, question, expert_result, expert_name)`
  - `respond_to_question` (method, line 480) `def respond_to_question(self, question)`
  - `interactive_mode` (method, line 554) `def interactive_mode(self)`
- Depends on: `agi.py`, `unificado.py`
