# API

## agi.py
Imported by: `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`
- `get_parity_dataset` (function) `agi.py:31` `def get_parity_dataset(n_bits, k, size)`
- `generate_wave_data` (function) `agi.py:36` `def generate_wave_data(N, T, c, dt, L, seed)`
- `step` (method) `agi.py:42` `def step(u_t, u_tm1)`
- `generate_kepler_data` (function) `agi.py:61` `def generate_kepler_data(num_samples, seed)`
- `generate_and_save_chaotic_pendulum_dataset` (function) `agi.py:82` `def generate_and_save_chaotic_pendulum_dataset(n_samples, seed)`
- `ParityCassette.__init__` (method) `agi.py:89` `def __init__(self, input_dim, hidden_dim)`
- `ParityCassette.forward` (method) `agi.py:97` `def forward(self, x)`
- `WaveCassette.__init__` (method) `agi.py:108` `def __init__(self, hidden_dim)`
- `WaveCassette.forward` (method) `agi.py:116` `def forward(self, x)`
- `KeplerCassette.__init__` (method) `agi.py:128` `def __init__(self, hidden_dim)`
- `KeplerCassette.forward` (method) `agi.py:138` `def forward(self, x)`
- `PendulumCassette.__init__` (method) `agi.py:146` `def __init__(self, hidden_dim)`
- `PendulumCassette.forward` (method) `agi.py:157` `def forward(self, x)`
- `GrokkitRouter.__init__` (method) `agi.py:164` `def __init__(self, num_domains)`
- `GrokkitRouter.forward` (method) `agi.py:170` `def forward(self, x)` -- Router determinístico basado en características infalibles del input.
- `Grokkit.__init__` (method) `agi.py:197` `def __init__(self, load_weights)`
- `Grokkit.load_pretrained_weights` (method) `agi.py:213` `def load_pretrained_weights(self)`
- `Grokkit.forward` (method) `agi.py:237` `def forward(self, x)`
- `Grokkit.demo_grokkit` (method) `agi.py:244` `def demo_grokkit()`

## app.py
Depends on: `agi.py`
- `test_wave` (function) `app.py:25` `def test_wave(grokkit)` -- Test de la ecuación de onda en una malla más fina (N=256) de la que se entrenó (N=32).
- `test_parity` (function) `app.py:41` `def test_parity(grokkit)` -- Test de paridad con inputs de 64 bits (muy más allá de los 3 usados para entrenar).
- `test_kepler` (function) `app.py:65` `def test_kepler(grokkit)` -- Test using the SAME data generation logic as the original training script.
- `generate_kepler_test` (function) `app.py:70` `def generate_kepler_test(n_samples, seed)`
- `test_pendulum` (function) `app.py:108` `def test_pendulum(grokkit)` -- Test using the REAL saved dataset, not mock data.
- `main` (function) `app.py:134` `def main()`

## super_casette.py
Depends on: `agi.py`
- `SuperpositionSAE.__init__` (method) `super_casette.py:34` `def __init__(self, d_model, d_sae)`
- `SuperpositionSAE.forward` (method) `super_casette.py:41` `def forward(self, x)`
- `SuperpositionSAE.get_metrics` (method) `super_casette.py:46` `def get_metrics(self, z)`
- `ComplexityAnalyzer.measure_lc` (method) `super_casette.py:58` `def measure_lc(model, x, epsilon)` -- Mide la complejidad de circuito (neuronas muertas/activas).
- `FusedAGIBrain.__init__` (method) `super_casette.py:79` `def __init__(self, hidden_dim)`
- `FusedAGIBrain.forward` (method) `super_casette.py:91` `def forward(self, x)`
- `SurgicalFusion.__init__` (method) `super_casette.py:122` `def __init__(self, weights_dir)`
- `SurgicalFusion.load_expert_weights` (method) `super_casette.py:131` `def load_expert_weights(self, domain)`
- `SurgicalFusion.transplant` (method) `super_casette.py:137` `def transplant(self, brain)`
- `SurgicalFusion.recovery_fine_tuning` (method) `super_casette.py:193` `def recovery_fine_tuning(brain)`
- `SurgicalFusion.main` (method) `super_casette.py:276` `def main()`

## uni.py
Depends on: `agi.py`, `unificado.py`
- `batch_multi_domain` (function) `uni.py:15` `def batch_multi_domain()` -- Crea un batch con 4 problemas distintos
- `main` (function) `uni.py:38` `def main()`

## unificado.py
Depends on: `agi.py`
Imported by: `uni.py`, `voz.py`
- `UnifiedGrokkitAgent.__init__` (method) `unificado.py:18` `def __init__(self, cassette_paths)`
- `UnifiedGrokkitAgent.forward` (method) `unificado.py:68` `def forward(self, x)`
- `UnifiedGrokkitAgent.__call__` (method) `unificado.py:80` `def __call__(self, x)`

## voz.py
Depends on: `agi.py`, `unificado.py`
- `AGIVoiceLayer.__init__` (method) `voz.py:36` `def __init__(self, model_name, use_unified, debug)` -- Inicializa la capa de voz con el modelo de lenguaje y los expertos
- `AGIVoiceLayer.respond_to_question` (method) `voz.py:480` `def respond_to_question(self, question)` -- Proceso completo robusto con múltiples fallbacks
- `AGIVoiceLayer.interactive_mode` (method) `voz.py:554` `def interactive_mode(self)` -- Modo interactivo mejorado con manejo de errores
- `AGIVoiceLayer.demo_voice_layer` (method) `voz.py:584` `def demo_voice_layer()` -- Demostración con ejemplos corregidos
