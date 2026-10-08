# root: voz

*Community 1 | 3 files | cohesion 0.40*

## Definition

This community groups 3 file(s) rooted at `root` with dominant language py (cohesion 0.40). Central symbols: `AGIVoiceLayer`, `UnifiedGrokkitAgent`, `__call__`, `__init__`, `_domain_routing_heuristic`, `_extract_binary_number`, `_extract_k_value`, `_generate_final_response`. Core file: `voz.py` (12 symbols). Documented purpose: AGI v0.1 - Demo Multi-Dominio Resuelve Parity, Wave, Kepler y Pendulum en un solo script..

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `uni.py` | py | utility | 2 | yes |
| `unificado.py` | py | utility | 5 | yes |
| `voz.py` | py | utility | 12 | yes |

## Key Symbols

- `batch_multi_domain` (function, `uni.py:15`) `def batch_multi_domain()` - Crea un batch con 4 problemas distintos
- `main` (function, `uni.py:38`) `def main()`
- `UnifiedGrokkitAgent` (class, `unificado.py:17`) `class UnifiedGrokkitAgent(Module)`
- `__init__` (method, `unificado.py:18`) `def __init__(self, cassette_paths)`
- `_load_cassettes` (method, `unificado.py:33`) `def _load_cassettes(self, paths)` - Carga usando la lógica robusta de agi.py
- `forward` (method, `unificado.py:68`) `def forward(self, x)`
- `__call__` (method, `unificado.py:80`) `def __call__(self, x)`
- `AGIVoiceLayer` (class, `voz.py:33`) `class AGIVoiceLayer` - Capa de lenguaje robusta que articula respuestas usando los expertos AGI
- `__init__` (method, `voz.py:36`) `def __init__(self, model_name, use_unified, debug)` - Inicializa la capa de voz con el modelo de lenguaje y los expertos
- `_extract_binary_number` (method, `voz.py:91`) `def _extract_binary_number(self, text)` - Extrae el número binario de una pregunta usando regex
- `_extract_k_value` (method, `voz.py:114`) `def _extract_k_value(self, text)` - Extrae el valor k (número de bits a sumar)
- `_domain_routing_heuristic` (method, `voz.py:135`) `def _domain_routing_heuristic(self, question)` - Routing basado en heurísticas de palabras clave (más robusto que el LLM)
- `_prepare_input_for_expert` (method, `voz.py:166`) `def _prepare_input_for_expert(self, expert_name, parameters, question)` - Prepara los datos de entrada según el experto necesario, con fallbacks robustos
- `_interpret_technical_result` (method, `voz.py:242`) `def _interpret_technical_result(self, expert_name, result, parameters, question)` - Convierte el resultado técnico en una descripción precisa en lenguaje natural
- `_get_expert_analysis` (method, `voz.py:311`) `def _get_expert_analysis(self, question)` - Análisis robusto con fallback a heurísticas si el LLM falla
- `_generate_final_response` (method, `voz.py:388`) `def _generate_final_response(self, question, expert_result, expert_name)` - Genera la respuesta final natural basada en el resultado real del experto
- `respond_to_question` (method, `voz.py:480`) `def respond_to_question(self, question)` - Proceso completo robusto con múltiples fallbacks
- `interactive_mode` (method, `voz.py:554`) `def interactive_mode(self)` - Modo interactivo mejorado con manejo de errores
- `demo_voice_layer` (method, `voz.py:584`) `def demo_voice_layer()` - Demostración con ejemplos corregidos

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 2
- Cross-boundary resolved imports (EXTRACTED): 3

## Connections

- [EXTRACTED] depends_on community 1 <-> 0 (strength 0.9): Extracted import edge crosses communities: uni.py imports agi.py.
- [INFERRED] shares_context community 1 <-> 2 (strength 0.5): Inferred shared context (layer utility) with no import path between community 1 (root: voz) and community 2 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- What would break if the most connected file in root: voz changed?
- Should root: voz be split, given cohesion 0.40?

## Sources

- `uni.py`
- `unificado.py`
- `voz.py`
