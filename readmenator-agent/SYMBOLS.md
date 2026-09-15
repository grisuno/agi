# Symbols

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `Grokkit` | class | `agi.py:196` | `class Grokkit(Module)` |
| `GrokkitRouter` | class | `agi.py:163` | `class GrokkitRouter(Module)` |
| `KeplerCassette` | class | `agi.py:127` | `class KeplerCassette(Module)` |
| `ParityCassette` | class | `agi.py:88` | `class ParityCassette(Module)` |
| `PendulumCassette` | class | `agi.py:145` | `class PendulumCassette(Module)` |
| `WaveCassette` | class | `agi.py:107` | `class WaveCassette(Module)` |
| `__init__` | method | `agi.py:89` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `agi.py:108` | `def __init__(self, hidden_dim)` |
| `__init__` | method | `agi.py:128` | `def __init__(self, hidden_dim)` |
| `__init__` | method | `agi.py:146` | `def __init__(self, hidden_dim)` |
| `__init__` | method | `agi.py:164` | `def __init__(self, num_domains)` |
| `__init__` | method | `agi.py:197` | `def __init__(self, load_weights)` |
| `demo_grokkit` | method | `agi.py:244` | `def demo_grokkit()` |
| `forward` | method | `agi.py:97` | `def forward(self, x)` |
| `forward` | method | `agi.py:116` | `def forward(self, x)` |
| `forward` | method | `agi.py:138` | `def forward(self, x)` |
| `forward` | method | `agi.py:157` | `def forward(self, x)` |
| `forward` | method | `agi.py:170` | `def forward(self, x)` |
| `forward` | method | `agi.py:237` | `def forward(self, x)` |
| `generate_and_save_chaotic_pendulum_dataset` | function | `agi.py:82` | `def generate_and_save_chaotic_pendulum_dataset(n_samples, seed)` |
| `generate_kepler_data` | function | `agi.py:61` | `def generate_kepler_data(num_samples, seed)` |
| `generate_wave_data` | function | `agi.py:36` | `def generate_wave_data(N, T, c, dt, L, seed)` |
| `get_parity_dataset` | function | `agi.py:31` | `def get_parity_dataset(n_bits, k, size)` |
| `load_pretrained_weights` | method | `agi.py:213` | `def load_pretrained_weights(self)` |
| `step` | method | `agi.py:42` | `def step(u_t, u_tm1)` |
| `generate_kepler_test` | function | `app.py:70` | `def generate_kepler_test(n_samples, seed)` |
| `main` | function | `app.py:134` | `def main()` |
| `test_kepler` | function | `app.py:65` | `def test_kepler(grokkit)` |
| `test_parity` | function | `app.py:41` | `def test_parity(grokkit)` |
| `test_pendulum` | function | `app.py:108` | `def test_pendulum(grokkit)` |
| `test_wave` | function | `app.py:25` | `def test_wave(grokkit)` |
| `ComplexityAnalyzer` | class | `super_casette.py:56` | `class ComplexityAnalyzer` |
| `FusedAGIBrain` | class | `super_casette.py:78` | `class FusedAGIBrain(Module)` |
| `SuperpositionSAE` | class | `super_casette.py:32` | `class SuperpositionSAE(Module)` |
| `SurgicalFusion` | class | `super_casette.py:121` | `class SurgicalFusion` |
| `__init__` | method | `super_casette.py:34` | `def __init__(self, d_model, d_sae)` |
| `__init__` | method | `super_casette.py:79` | `def __init__(self, hidden_dim)` |
| `__init__` | method | `super_casette.py:122` | `def __init__(self, weights_dir)` |
| `forward` | method | `super_casette.py:41` | `def forward(self, x)` |
| `forward` | method | `super_casette.py:91` | `def forward(self, x)` |
| `get_metrics` | method | `super_casette.py:46` | `def get_metrics(self, z)` |
| `load_expert_weights` | method | `super_casette.py:131` | `def load_expert_weights(self, domain)` |
| `main` | method | `super_casette.py:276` | `def main()` |
| `measure_lc` | method | `super_casette.py:58` | `def measure_lc(model, x, epsilon)` |
| `recovery_fine_tuning` | method | `super_casette.py:193` | `def recovery_fine_tuning(brain)` |
| `transplant` | method | `super_casette.py:137` | `def transplant(self, brain)` |
| `batch_multi_domain` | function | `uni.py:15` | `def batch_multi_domain()` |
| `main` | function | `uni.py:38` | `def main()` |
| `UnifiedGrokkitAgent` | class | `unificado.py:17` | `class UnifiedGrokkitAgent(Module)` |
| `__call__` | method | `unificado.py:80` | `def __call__(self, x)` |
| `__init__` | method | `unificado.py:18` | `def __init__(self, cassette_paths)` |
| `_load_cassettes` | method | `unificado.py:33` | `def _load_cassettes(self, paths)` |
| `forward` | method | `unificado.py:68` | `def forward(self, x)` |
| `AGIVoiceLayer` | class | `voz.py:33` | `class AGIVoiceLayer` |
| `__init__` | method | `voz.py:36` | `def __init__(self, model_name, use_unified, debug)` |
| `_domain_routing_heuristic` | method | `voz.py:135` | `def _domain_routing_heuristic(self, question)` |
| `_extract_binary_number` | method | `voz.py:91` | `def _extract_binary_number(self, text)` |
| `_extract_k_value` | method | `voz.py:114` | `def _extract_k_value(self, text)` |
| `_generate_final_response` | method | `voz.py:388` | `def _generate_final_response(self, question, expert_result, expert_name)` |
| `_get_expert_analysis` | method | `voz.py:311` | `def _get_expert_analysis(self, question)` |
| `_interpret_technical_result` | method | `voz.py:242` | `def _interpret_technical_result(self, expert_name, result, parameters, question)` |
| `_prepare_input_for_expert` | method | `voz.py:166` | `def _prepare_input_for_expert(self, expert_name, parameters, question)` |
| `demo_voice_layer` | method | `voz.py:584` | `def demo_voice_layer()` |
| `interactive_mode` | method | `voz.py:554` | `def interactive_mode(self)` |
| `respond_to_question` | method | `voz.py:480` | `def respond_to_question(self, question)` |
