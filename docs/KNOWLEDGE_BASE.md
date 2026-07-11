# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis.

**Total Files Parsed:** 7 | **Total Symbols Extracted:** 65 | **Total Imports:** 43

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray: 5 5,color:#aaa;
    super_casette_py["super_casette.py (py)"]
    class super_casette_py mod;
    super_casette_py_SuperpositionSAE["SuperpositionSAE"]
    class super_casette_py_SuperpositionSAE cls;
    super_casette_py --> super_casette_py_SuperpositionSAE
    super_casette_py_ComplexityAnalyzer["ComplexityAnalyzer"]
    class super_casette_py_ComplexityAnalyzer cls;
    super_casette_py --> super_casette_py_ComplexityAnalyzer
    super_casette_py_FusedAGIBrain["FusedAGIBrain"]
    class super_casette_py_FusedAGIBrain cls;
    super_casette_py --> super_casette_py_FusedAGIBrain
    super_casette_py_SurgicalFusion["SurgicalFusion"]
    class super_casette_py_SurgicalFusion cls;
    super_casette_py --> super_casette_py_SurgicalFusion
    super_casette_py_recovery_fine_tuning["recovery_fine_tuning"]
    class super_casette_py_recovery_fine_tuning fn;
    super_casette_py --> super_casette_py_recovery_fine_tuning
    voz_py["voz.py (py)"]
    class voz_py mod;
    voz_py_AGIVoiceLayer["AGIVoiceLayer"]
    class voz_py_AGIVoiceLayer cls;
    voz_py --> voz_py_AGIVoiceLayer
    voz_py_demo_voice_layer["demo_voice_layer"]
    class voz_py_demo_voice_layer fn;
    voz_py --> voz_py_demo_voice_layer
    voz_py___init__["__init__"]
    class voz_py___init__ fn;
    voz_py --> voz_py___init__
    voz_py__extract_binary_number["_extract_binary_number"]
    class voz_py__extract_binary_number fn;
    voz_py --> voz_py__extract_binary_number
    voz_py__extract_k_value["_extract_k_value"]
    class voz_py__extract_k_value fn;
    voz_py --> voz_py__extract_k_value
    agi_py["agi.py (py)"]
    class agi_py mod;
    agi_py_get_parity_dataset["get_parity_dataset"]
    class agi_py_get_parity_dataset fn;
    agi_py --> agi_py_get_parity_dataset
    agi_py_generate_wave_data["generate_wave_data"]
    class agi_py_generate_wave_data fn;
    agi_py --> agi_py_generate_wave_data
    agi_py_generate_kepler_data["generate_kepler_data"]
    class agi_py_generate_kepler_data fn;
    agi_py --> agi_py_generate_kepler_data
    agi_py_generate_and_save_chaotic_pendulum_dataset["generate_and_save_chaotic_pendulum_dataset"]
    class agi_py_generate_and_save_chaotic_pendulum_dataset fn;
    agi_py --> agi_py_generate_and_save_chaotic_pendulum_dataset
    agi_py_ParityCassette["ParityCassette"]
    class agi_py_ParityCassette cls;
    agi_py --> agi_py_ParityCassette
    app_py["app.py (py)"]
    class app_py mod;
    app_py_test_wave["test_wave"]
    class app_py_test_wave fn;
    app_py --> app_py_test_wave
    app_py_test_parity["test_parity"]
    class app_py_test_parity fn;
    app_py --> app_py_test_parity
    app_py_test_kepler["test_kepler"]
    class app_py_test_kepler fn;
    app_py --> app_py_test_kepler
    app_py_test_pendulum["test_pendulum"]
    class app_py_test_pendulum fn;
    app_py --> app_py_test_pendulum
    app_py_main["main"]
    class app_py_main fn;
    app_py --> app_py_main
    unificado_py["unificado.py (py)"]
    class unificado_py mod;
    unificado_py_UnifiedGrokkitAgent["UnifiedGrokkitAgent"]
    class unificado_py_UnifiedGrokkitAgent cls;
    unificado_py --> unificado_py_UnifiedGrokkitAgent
    unificado_py___init__["__init__"]
    class unificado_py___init__ fn;
    unificado_py --> unificado_py___init__
    unificado_py__load_cassettes["_load_cassettes"]
    class unificado_py__load_cassettes fn;
    unificado_py --> unificado_py__load_cassettes
    unificado_py_forward["forward"]
    class unificado_py_forward fn;
    unificado_py --> unificado_py_forward
    unificado_py___call__["__call__"]
    class unificado_py___call__ fn;
    unificado_py --> unificado_py___call__
    uni_py["uni.py (py)"]
    class uni_py mod;
    uni_py_batch_multi_domain["batch_multi_domain"]
    class uni_py_batch_multi_domain fn;
    uni_py --> uni_py_batch_multi_domain
    uni_py_main["main"]
    class uni_py_main fn;
    uni_py --> uni_py_main
    install_sh["install.sh (sh)"]
    class install_sh mod;
    ext___future__["__future__"]
    class ext___future__ ext;
    agi_py -.->|imports| ext___future__
    ext_os["os"]
    class ext_os ext;
    agi_py -.->|imports| ext_os
    ext_torch["torch"]
    class ext_torch ext;
    agi_py -.->|imports| ext_torch
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    agi_py -.->|imports| ext_torch_nn
    ext_torch_nn_functional["torch.nn.functional"]
    class ext_torch_nn_functional ext;
    agi_py -.->|imports| ext_torch_nn_functional
    ext_numpy["numpy"]
    class ext_numpy ext;
    agi_py -.->|imports| ext_numpy
    ext_typing["typing"]
    class ext_typing ext;
    agi_py -.->|imports| ext_typing
    ext_pathlib["pathlib"]
    class ext_pathlib ext;
    agi_py -.->|imports| ext_pathlib
    app_py -.->|imports| ext_torch
    app_py -.->|imports| ext_torch_nn_functional
    app_py -.->|imports| ext_numpy
    ext_agi["agi"]
    class ext_agi ext;
    app_py -.->|imports| ext_agi
    ext_sys["sys"]
    class ext_sys ext;
    app_py -.->|imports| ext_sys
    app_py -.->|imports| ext_pathlib
    app_py -.->|imports| ext_agi
    super_casette_py -.->|imports| ext_os
    super_casette_py -.->|imports| ext_sys
    super_casette_py -.->|imports| ext_torch
    super_casette_py -.->|imports| ext_torch_nn
    super_casette_py -.->|imports| ext_torch_nn_functional
    ext_math["math"]
    class ext_math ext;
    super_casette_py -.->|imports| ext_math
    super_casette_py -.->|imports| ext_pathlib
    ext_copy["copy"]
    class ext_copy ext;
    super_casette_py -.->|imports| ext_copy
    super_casette_py -.->|imports| ext_agi
    uni_py -.->|imports| ext_torch_nn_functional
    uni_py -.->|imports| ext_torch
    ext_time["time"]
    class ext_time ext;
    uni_py -.->|imports| ext_time
    ext_unificado["unificado"]
    class ext_unificado ext;
    uni_py -.->|imports| ext_unificado
    uni_py -.->|imports| ext_agi
    unificado_py -.->|imports| ext_torch
    unificado_py -.->|imports| ext_torch_nn
    unificado_py -.->|imports| ext_pathlib
    unificado_py -.->|imports| ext_typing
    unificado_py -.->|imports| ext_agi
    voz_py -.->|imports| ext_torch
    ext_json["json"]
    class ext_json ext;
    voz_py -.->|imports| ext_json
    voz_py -.->|imports| ext_time
    ext_ollama["ollama"]
    class ext_ollama ext;
    voz_py -.->|imports| ext_ollama
    ext_re["re"]
    class ext_re ext;
    voz_py -.->|imports| ext_re
    voz_py -.->|imports| ext_typing
    voz_py -.->|imports| ext_agi
    voz_py -.->|imports| ext_unificado
    voz_py -.->|imports| ext_sys
```

---

## Architecture Reference

### PY (6 files)

#### `agi.py`
**Path:** `agi.py`

**Classs:**
- `ParityCassette` (line 88)
- `WaveCassette` (line 107)
- `KeplerCassette` (line 127)
- `PendulumCassette` (line 145)
- `GrokkitRouter` (line 163)
- `Grokkit` (line 196)

**Functions:**
- `get_parity_dataset` (line 31)
- `generate_wave_data` (line 36)
- `generate_kepler_data` (line 61)
- `generate_and_save_chaotic_pendulum_dataset` (line 82)
- `demo_grokkit` (line 244)
- `step` (line 42)
- `__init__` (line 89)
- `forward` (line 97)
- `__init__` (line 108)
- `forward` (line 116)
- `__init__` (line 128)
- `forward` (line 138)
- `__init__` (line 146)
- `forward` (line 157)
- `__init__` (line 164)
- `forward` (line 170) - *Router determinístico basado en características infalibles del input.
Devuelve probabilidades softmax con 100% en el dominio correcto.*
- `__init__` (line 197)
- `load_pretrained_weights` (line 213)
- `forward` (line 237)

#### `app.py`
**Path:** `app.py`

**Functions:**
- `test_wave` (line 25) - *Test de la ecuación de onda en una malla más fina (N=256) de la que se entrenó (N=32).*
- `test_parity` (line 41) - *Test de paridad con inputs de 64 bits (muy más allá de los 3 usados para entrenar).*
- `test_kepler` (line 65) - *Test using the SAME data generation logic as the original training script.*
- `test_pendulum` (line 108) - *Test using the REAL saved dataset, not mock data.*
- `main` (line 134)
- `generate_kepler_test` (line 70)

#### `super_casette.py`
**Path:** `super_casette.py`

**Classs:**
- `SuperpositionSAE` (line 32) - *Sparse Autoencoder para forzar la estructura geométrica en el espacio latente.*
- `ComplexityAnalyzer` (line 56)
- `FusedAGIBrain` (line 78)
- `SurgicalFusion` (line 121)

**Functions:**
- `recovery_fine_tuning` (line 193)
- `main` (line 276)
- `__init__` (line 34)
- `forward` (line 41)
- `get_metrics` (line 46)
- `measure_lc` (line 58) - *Mide la complejidad de circuito (neuronas muertas/activas).*
- `__init__` (line 79)
- `forward` (line 91)
- `__init__` (line 122)
- `load_expert_weights` (line 131)
- `transplant` (line 137)

#### `uni.py`
**Path:** `uni.py`

**Functions:**
- `batch_multi_domain` (line 15) - *Crea un batch con 4 problemas distintos*
- `main` (line 38)

#### `unificado.py`
**Path:** `unificado.py`

**Classs:**
- `UnifiedGrokkitAgent` (line 17)

**Functions:**
- `__init__` (line 18)
- `_load_cassettes` (line 33) - *Carga usando la lógica robusta de agi.py*
- `forward` (line 68)
- `__call__` (line 80)

#### `voz.py`
**Path:** `voz.py`

**Classs:**
- `AGIVoiceLayer` (line 33) - *Capa de lenguaje robusta que articula respuestas usando los expertos AGI*

**Functions:**
- `demo_voice_layer` (line 584) - *Demostración con ejemplos corregidos*
- `__init__` (line 36) - *Inicializa la capa de voz con el modelo de lenguaje y los expertos*
- `_extract_binary_number` (line 91) - *Extrae el número binario de una pregunta usando regex*
- `_extract_k_value` (line 114) - *Extrae el valor k (número de bits a sumar)*
- `_domain_routing_heuristic` (line 135) - *Routing basado en heurísticas de palabras clave (más robusto que el LLM)*
- `_prepare_input_for_expert` (line 166) - *Prepara los datos de entrada según el experto necesario, con fallbacks robustos*
- `_interpret_technical_result` (line 242) - *Convierte el resultado técnico en una descripción precisa en lenguaje natural*
- `_get_expert_analysis` (line 311) - *Análisis robusto con fallback a heurísticas si el LLM falla*
- `_generate_final_response` (line 388) - *Genera la respuesta final natural basada en el resultado real del experto*
- `respond_to_question` (line 480) - *Proceso completo robusto con múltiples fallbacks*
- `interactive_mode` (line 554) - *Modo interactivo mejorado con manejo de errores*

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
