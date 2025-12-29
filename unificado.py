#!/usr/bin/env python3
"""
Grokkit v0.1 - Unified Agent with Explicit Path Mapping
No parsing, no magic, just works.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

# Importar desde agi.py (tu código funcional)
from agi import (
    GrokkitRouter, ParityCassette, WaveCassette, KeplerCassette, PendulumCassette,
    WEIGHTS_DIR
)

class UnifiedGrokkitAgent(nn.Module):
    def __init__(self, cassette_paths: Dict[str, str]):
        super().__init__()
        self.router = GrokkitRouter()
        
        # Mapeo domain → clase
        self.cassette_classes = {
            'parity': ParityCassette,
            'wave': WaveCassette,
            'kepler': KeplerCassette,
            'pendulum': PendulumCassette,
        }
        
        self.cassettes = nn.ModuleDict()
        self._load_cassettes(cassette_paths)
    
    def _load_cassettes(self, paths: Dict[str, str]):
        """Carga usando la lógica robusta de agi.py"""
        for domain, filename in paths.items():
            path = WEIGHTS_DIR / filename
            
            if path.exists():
                print(f"🧠 Cargando cassette {domain} desde {path}")
                checkpoint = torch.load(path, map_location='cpu')
                
                # Extraer state_dict
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Determinar dims desde el state_dict
                first_layer = list(state_dict.keys())[0]
                hidden_dim = state_dict[first_layer].shape[0]
                
                if domain == 'parity':
                    input_dim = state_dict['fc1.weight'].shape[1]
                    model = ParityCassette(input_dim=input_dim, hidden_dim=hidden_dim)
                else:
                    model = self.cassette_classes[domain](hidden_dim=hidden_dim)
                
                model.load_state_dict(state_dict)
            else:
                print(f"⚠️  No se encontró {path}, usando dummy")
                model = self.cassette_classes[domain]()  # default params
            
            # Freeze y registrar
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self.cassettes[domain] = model
        
        print(f"✅ Cargados {len(self.cassettes)} cassettes")
    
    def forward(self, x: torch.Tensor) -> tuple:
        probs = self.router(x)
        domain_idx = torch.argmax(probs).item()
        
        domain_map = {0: 'parity', 1: 'wave', 2: 'kepler', 3: 'pendulum'}
        domain = domain_map[domain_idx]
        
        with torch.no_grad():
            output = self.cassettes[domain](x)
        
        return output, domain, probs
    
    def __call__(self, x: torch.Tensor) -> tuple:
        return self.forward(x)

# --- USO ---
if __name__ == "__main__":
    # Mapeo explícito (copiado de agi.py)
    CASSETTES = {
        'parity': "grok_model_stage4_n64_d1024_adaptive.pth",
        'wave': "wave_grok_cnn_physics_cassette.pth",
        'kepler': "kepler_base_model.pth",
        'pendulum': "symplectic_double_pendulum_grok_cassette.pth"
    }
    
    agent = UnifiedGrokkitAgent(CASSETTES)
    
    # Test
    batch_parity = (torch.randn(5, 64) > 0.5).float()
    out, domain, _ = agent(batch_parity)
    print(f"Parity: {domain} → {out.argmax(1)}")