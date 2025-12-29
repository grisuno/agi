#!/usr/bin/env python3
"""
UnifiedGrokkitAgent: Agente que usa todos los cassettes sin entrenar conjuntamente
"""
import os
import re
import torch
import torch.nn as nn
from typing import Dict, Any

class WaveCassette(nn.Module):
    def forward(self, x: torch.Tensor):
        # Si x es [B, N], reshapéalo a [B, 1, N]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.core_forward(x)

class CassetteLoader:
    """Carga cassettes grokked desde state_dict inferiendo arquitectura del nombre"""
    
    # Mapeo de tipo → clase
    ARCH_MAP = {
        'parity': 'GrokkingTransformer',
        'wave': 'WaveCassette',
        'kepler': 'KeplerCassette',
        'pendulum': 'PendulumCassette',
    }
    
    # Mapeo de clase → constructor
    # (asumiendo que todas siguen patrón d_in, d_h)
    # Nota: Asegúrate de que estas clases (GrokkingTransformer, etc.) estén importadas o definidas
    CONSTRUCTORS = {
        'GrokkingTransformer': lambda d_in, d_h: GrokkingTransformer(d_in=d_in, d_h=d_h),
        'WaveCassette': lambda d_in, d_h: WaveCassette(d_in=d_in, d_h=d_h),
        'KeplerCassette': lambda d_in, d_h: KeplerCassette(d_in=d_in, d_h=d_h),
        'PendulumCassette': lambda d_in, d_h: PendulumCassette(d_in=d_in, d_h=d_h),
    }

    @classmethod
    def load(cls, path: str, domain: str) -> nn.Module:
        """
        path: ruta al .pth
        domain: 'parity', 'wave', etc
        """
        # 1. Parsear nombre para extraer dims
        # Ej: "grok_model_stage4_n64_d1024_adaptive.pth"
        filename = os.path.basename(path)
        match = re.search(r'n(\d+)_d(\d+)', filename)
        if not match:
            # Fallback por si el nombre no tiene el formato exacto pero conocemos d_h
            match_d = re.search(r'_d(\d+)', filename)
            if match_d:
                d_h = int(match_d.group(1))
                d_in = 64 # Valor por defecto común si no se encuentra
            else:
                raise ValueError(f"No se pudo parsear dims de {filename}")
        else:
            d_in, d_h = map(int, match.groups())

        # 2. Instanciar arquitectura
        arch_name = cls.ARCH_MAP[domain]
        constructor = cls.CONSTRUCTORS[arch_name]
        model = constructor(d_in, d_h)

        # 3. Cargar pesos grokked
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()

        # 4. Freeze
        for p in model.parameters():
            p.requires_grad = False
        return model

class DomainRouter(nn.Module):
    """Router que NO entrena. Infiere dominio desde estadísticas del input."""
    def __init__(self):
        super().__init__()
        # Estos pesos son *handcrafted*, no aprendidos
        self.register_buffer('wave_detector', torch.tensor([1, -2, 1])) # Kernel Laplaciano
        self.register_buffer('parity_detector', torch.tensor([0.5, -0.5])) # XOR pattern

    def forward(self, x: torch.Tensor) -> str:
        """Devuelve 'parity', 'wave', 'kepler', 'pendulum', o 'unknown'"""
        shape = x.shape
        
        # Heurística basada en shape + autocorrelación
        if len(shape) == 2 and shape[1] > 32:
            return 'parity'
        elif len(shape) == 3 and shape[2] > 32: # [B, 1, N]
            # Detecta si es onda vs caótico viendo la FFT
            fft = torch.fft.rfft(x)
            if fft.abs().max() > 0.5: # Pico en frecuencia → onda
                return 'wave'
            else:
                return 'pendulum'
        elif len(shape) == 2 and shape[1] == 4: # [B, 4] (x,y,vx,vy)
            return 'kepler'
        
        return 'unknown'

class UnifiedGrokkitAgent:
    def __init__(self, cassette_paths: Dict[str, str]):
        self.router = DomainRouter().eval()
        self.cassettes = nn.ModuleDict()
        
        for domain, path in cassette_paths.items():
            # Carga inteligente
            model = CassetteLoader.load(path, domain=domain)
            self.cassettes[domain] = model
            print(f"✅ Cargados {len(self.cassettes)} cassettes: {list(self.cassettes.keys())}")

    def forward(self, x: torch.Tensor) -> Any:
        domain = self.router(x)
        if domain == 'unknown':
            raise ValueError(f"No se detectó dominio para shape {x.shape}")
        
        cassette = self.cassettes[domain]
        with torch.no_grad():
            return cassette(x)

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """Interfaz estándar de agente"""
        return self.forward(observation)

# --- USO ---
if __name__ == "__main__":
    # Cassettes pre-grokked (Asegúrate de que estos archivos existan en tu carpeta)
    CASSETTES = {
        'parity': "grok_model_stage4_n64_d1024_adaptive.pth",
        'wave': "wave_cassette_n256_d512.pth",
        'kepler': "kepler_cassette_d64.pth",
        'pendulum': "pendulum_cassette_d128.pth"
    }

    # Intentar inicializar (Fallará si los .pth no están presentes)
    try:
        agent = UnifiedGrokkitAgent(CASSETTES)
        
        # Test multi-dominio
        batch_parity = (torch.randn(10, 64) > 0.5).float()
        batch_wave = torch.randn(10, 1, 256)
        batch_kepler = torch.randn(10, 4)

        print("Parity:", agent.act(batch_parity).argmax(1))
        # Ajuste en los prints para evitar errores de índice si la salida no es una lista
        res_wave = agent.act(batch_wave)
        print("Wave output shape:", res_wave.shape)
        
        res_kepler = agent.act(batch_kepler)
        print("Kepler output shape:", res_kepler.shape)
        
    except Exception as e:
        print(f"❌ Error durante el test: {e}")
        print("\nNota: Asegúrate de definir las clases GrokkingTransformer, KeplerCassette, etc. antes de ejecutar.")
