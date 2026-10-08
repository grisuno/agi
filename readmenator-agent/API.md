# API

## agi.py

### get_parity_dataset (function) `def get_parity_dataset(n_bits, k, size)`
- Defined: `agi.py:31`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### generate_wave_data (function) `def generate_wave_data(N, T, c, dt, L, seed)`
- Defined: `agi.py:36`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### generate_kepler_data (function) `def generate_kepler_data(num_samples, seed)`
- Defined: `agi.py:61`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### generate_and_save_chaotic_pendulum_dataset (function) `def generate_and_save_chaotic_pendulum_dataset(n_samples, seed)`
- Defined: `agi.py:82`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### demo_grokkit (method) `def demo_grokkit()`
- Defined: `agi.py:244`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### step (method) `def step(u_t, u_tm1)`
- Defined: `agi.py:42`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### __init__ (method) `def __init__(self, input_dim, hidden_dim)`
- Defined: `agi.py:89`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### forward (method) `def forward(self, x)`
- Defined: `agi.py:97`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### __init__ (method) `def __init__(self, hidden_dim)`
- Defined: `agi.py:108`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### forward (method) `def forward(self, x)`
- Defined: `agi.py:116`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### __init__ (method) `def __init__(self, hidden_dim)`
- Defined: `agi.py:128`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### forward (method) `def forward(self, x)`
- Defined: `agi.py:138`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### __init__ (method) `def __init__(self, hidden_dim)`
- Defined: `agi.py:146`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### forward (method) `def forward(self, x)`
- Defined: `agi.py:157`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### __init__ (method) `def __init__(self, num_domains)`
- Defined: `agi.py:164`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### forward (method) `def forward(self, x)`
- Defined: `agi.py:170`
- Doc: Router determinístico basado en características infalibles del input.
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### __init__ (method) `def __init__(self, load_weights)`
- Defined: `agi.py:197`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### load_pretrained_weights (method) `def load_pretrained_weights(self)`
- Defined: `agi.py:213`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

### forward (method) `def forward(self, x)`
- Defined: `agi.py:237`
- Imported by: `app.py`, `app.py`, `super_casette.py`, `uni.py`, `unificado.py`, `voz.py`

## app.py

### test_wave (function) `def test_wave(grokkit)`
- Defined: `app.py:25`
- Doc: Test de la ecuación de onda en una malla más fina (N=256) de la que se entrenó (N=32).
- Depends on: `agi.py`

### test_parity (function) `def test_parity(grokkit)`
- Defined: `app.py:41`
- Doc: Test de paridad con inputs de 64 bits (muy más allá de los 3 usados para entrenar).
- Depends on: `agi.py`

### test_kepler (function) `def test_kepler(grokkit)`
- Defined: `app.py:65`
- Doc: Test using the SAME data generation logic as the original training script.
- Depends on: `agi.py`

### test_pendulum (function) `def test_pendulum(grokkit)`
- Defined: `app.py:108`
- Doc: Test using the REAL saved dataset, not mock data.
- Depends on: `agi.py`

### main (function) `def main()`
- Defined: `app.py:134`
- Depends on: `agi.py`

### generate_kepler_test (function) `def generate_kepler_test(n_samples, seed)`
- Defined: `app.py:70`
- Depends on: `agi.py`

## super_casette.py

### recovery_fine_tuning (method) `def recovery_fine_tuning(brain)`
- Defined: `super_casette.py:193`
- Depends on: `agi.py`

### main (method) `def main()`
- Defined: `super_casette.py:276`
- Depends on: `agi.py`

### __init__ (method) `def __init__(self, d_model, d_sae)`
- Defined: `super_casette.py:34`
- Depends on: `agi.py`

### forward (method) `def forward(self, x)`
- Defined: `super_casette.py:41`
- Depends on: `agi.py`

### get_metrics (method) `def get_metrics(self, z)`
- Defined: `super_casette.py:46`
- Depends on: `agi.py`

### measure_lc (method) `def measure_lc(model, x, epsilon)`
- Defined: `super_casette.py:58`
- Doc: Mide la complejidad de circuito (neuronas muertas/activas).
- Depends on: `agi.py`

### __init__ (method) `def __init__(self, hidden_dim)`
- Defined: `super_casette.py:79`
- Depends on: `agi.py`

### forward (method) `def forward(self, x)`
- Defined: `super_casette.py:91`
- Depends on: `agi.py`

### __init__ (method) `def __init__(self, weights_dir)`
- Defined: `super_casette.py:122`
- Depends on: `agi.py`

### load_expert_weights (method) `def load_expert_weights(self, domain)`
- Defined: `super_casette.py:131`
- Depends on: `agi.py`

### transplant (method) `def transplant(self, brain)`
- Defined: `super_casette.py:137`
- Depends on: `agi.py`

## uni.py

### batch_multi_domain (function) `def batch_multi_domain()`
- Defined: `uni.py:15`
- Doc: Crea un batch con 4 problemas distintos
- Depends on: `agi.py`, `unificado.py`

### main (function) `def main()`
- Defined: `uni.py:38`
- Depends on: `agi.py`, `unificado.py`

## unificado.py

### __init__ (method) `def __init__(self, cassette_paths)`
- Defined: `unificado.py:18`
- Depends on: `agi.py`
- Imported by: `uni.py`, `voz.py`

### _load_cassettes (method) `def _load_cassettes(self, paths)`
- Defined: `unificado.py:33`
- Doc: Carga usando la lógica robusta de agi.py
- Depends on: `agi.py`
- Imported by: `uni.py`, `voz.py`

### forward (method) `def forward(self, x)`
- Defined: `unificado.py:68`
- Depends on: `agi.py`
- Imported by: `uni.py`, `voz.py`

### __call__ (method) `def __call__(self, x)`
- Defined: `unificado.py:80`
- Depends on: `agi.py`
- Imported by: `uni.py`, `voz.py`

## voz.py

### demo_voice_layer (method) `def demo_voice_layer()`
- Defined: `voz.py:584`
- Doc: Demostración con ejemplos corregidos
- Depends on: `agi.py`, `unificado.py`

### __init__ (method) `def __init__(self, model_name, use_unified, debug)`
- Defined: `voz.py:36`
- Doc: Inicializa la capa de voz con el modelo de lenguaje y los expertos
- Depends on: `agi.py`, `unificado.py`

### _extract_binary_number (method) `def _extract_binary_number(self, text)`
- Defined: `voz.py:91`
- Doc: Extrae el número binario de una pregunta usando regex
- Depends on: `agi.py`, `unificado.py`

### _extract_k_value (method) `def _extract_k_value(self, text)`
- Defined: `voz.py:114`
- Doc: Extrae el valor k (número de bits a sumar)
- Depends on: `agi.py`, `unificado.py`

### _domain_routing_heuristic (method) `def _domain_routing_heuristic(self, question)`
- Defined: `voz.py:135`
- Doc: Routing basado en heurísticas de palabras clave (más robusto que el LLM)
- Depends on: `agi.py`, `unificado.py`

### _prepare_input_for_expert (method) `def _prepare_input_for_expert(self, expert_name, parameters, question)`
- Defined: `voz.py:166`
- Doc: Prepara los datos de entrada según el experto necesario, con fallbacks robustos
- Depends on: `agi.py`, `unificado.py`

### _interpret_technical_result (method) `def _interpret_technical_result(self, expert_name, result, parameters, question)`
- Defined: `voz.py:242`
- Doc: Convierte el resultado técnico en una descripción precisa en lenguaje natural
- Depends on: `agi.py`, `unificado.py`

### _get_expert_analysis (method) `def _get_expert_analysis(self, question)`
- Defined: `voz.py:311`
- Doc: Análisis robusto con fallback a heurísticas si el LLM falla
- Depends on: `agi.py`, `unificado.py`

### _generate_final_response (method) `def _generate_final_response(self, question, expert_result, expert_name)`
- Defined: `voz.py:388`
- Doc: Genera la respuesta final natural basada en el resultado real del experto
- Depends on: `agi.py`, `unificado.py`

### respond_to_question (method) `def respond_to_question(self, question)`
- Defined: `voz.py:480`
- Doc: Proceso completo robusto con múltiples fallbacks
- Depends on: `agi.py`, `unificado.py`

### interactive_mode (method) `def interactive_mode(self)`
- Defined: `voz.py:554`
- Doc: Modo interactivo mejorado con manejo de errores
- Depends on: `agi.py`, `unificado.py`
