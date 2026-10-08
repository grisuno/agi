# Architecture

## Internal Dependencies

- `app.py` -> `agi.py`
- `super_casette.py` -> `agi.py`
- `uni.py` -> `agi.py`
- `uni.py` -> `unificado.py`
- `unificado.py` -> `agi.py`
- `voz.py` -> `agi.py`
- `voz.py` -> `unificado.py`

## External Imports

- `agi.py` -> __future__, numpy, os, pathlib, torch, torch.nn, torch.nn.functional, typing
- `app.py` -> numpy, pathlib, sys, torch, torch.nn.functional
- `super_casette.py` -> copy, math, os, pathlib, sys, torch, torch.nn, torch.nn.functional
- `uni.py` -> time, torch, torch.nn.functional
- `unificado.py` -> pathlib, torch, torch.nn, typing
- `voz.py` -> json, ollama, re, sys, time, torch, typing
