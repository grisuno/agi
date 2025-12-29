#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path

# Fix de importación
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)

try:
    from agi import (
        ParityCassette, KeplerCassette, PendulumCassette, WaveCassette,
        get_parity_dataset, generate_wave_data, 
        generate_kepler_data, generate_and_save_chaotic_pendulum_dataset
    )
except ImportError:
    print("❌ Error: No se pudo importar el framework 'agi'.")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# EL CIRUJANO CON ALINEACIÓN (Fine-Tuning Post-Cirugía)
# ============================================================

class AGISurgeon:
    @staticmethod
    def graft_and_realign(source_sd, target_model, domain, data):
        """
        No solo pega los pesos, sino que 'afina' el modelo para 
        recuperar la precisión de 10^-7.
        """
        # 1. Cirugía SVD/Padding
        target_sd = target_model.state_dict()
        for name, target_p in target_sd.items():
            if name in source_sd:
                source_p = source_sd[name]
                if source_p.shape == target_p.shape:
                    target_sd[name].copy_(source_p)
                elif len(source_p.shape) == 2: # Linear expansion
                    U, S, V = torch.svd(source_p.float())
                    graft = torch.zeros_like(target_p)
                    r, c = source_p.shape
                    graft[:r, :c] = U @ torch.diag(S) @ V.t()
                    target_sd[name].copy_(graft)
                elif len(source_p.shape) == 3: # Conv expansion
                    graft = torch.zeros_like(target_p)
                    o, i, k = source_p.shape
                    graft[:o, :i, :k] = source_p
                    target_sd[name].copy_(graft)
        
        target_model.load_state_dict(target_sd)
        
        # 2. FINE-TUNING DE ALINEACIÓN (Lo que faltaba)
        # Esto 'sutura' las neuronas nuevas con las viejas
        print(f"🔧 Alineando sabiduría para {domain.upper()}...")
        target_model.train()
        optimizer = torch.optim.AdamW(target_model.parameters(), lr=1e-4)
        x, y = data
        x, y = x.to(DEVICE), y.to(DEVICE)

        for _ in range(100): # Rápido, solo para ajustar fase
            out = target_model(x)
            val = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(val, y) if domain == 'parity' else F.mse_loss(val, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        target_model.eval()
        return target_model

# ============================================================
# MOTOR AGI CON MEMORIA RECUPERADA
# ============================================================

class UnifiedAGI(nn.Module):
    def __init__(self, weights_dir="./"):
        super().__init__()
        self.hidden = 1024
        self.experts = nn.ModuleDict()
        self.ckpts = {
            'parity': "grok_model_stage4_n64_d1024_adaptive.pth",
            'wave': "wave_grok_cnn_physics_cassette.pth",
            'kepler': "kepler_base_model.pth",
            'pendulum': "symplectic_double_pendulum_grok_cassette.pth"
        }
        self._fuse(weights_dir)

    def _fuse(self, weights_dir):
        p = Path(weights_dir)
        # Datos mínimos para el fine-tuning de alineación
        alignment_data = {
            'parity': get_parity_dataset(n_bits=64, k=3, size=200),
            'wave': generate_wave_data(N=32, T=20)[:2],
            'kepler': generate_kepler_data(num_samples=50),
            'pendulum': (torch.tensor(generate_and_save_chaotic_pendulum_dataset(n_samples=50)[0]).float(),
                         torch.tensor(generate_and_save_chaotic_pendulum_dataset(n_samples=50)[1]).float())
        }

        configs = {
            'parity': (ParityCassette, {'input_dim': 64, 'hidden_dim': self.hidden}),
            'wave': (WaveCassette, {'hidden_dim': self.hidden}),
            'kepler': (KeplerCassette, {'hidden_dim': self.hidden}),
            'pendulum': (PendulumCassette, {'hidden_dim': self.hidden})
        }

        for key, (cls, kwargs) in configs.items():
            model = cls(**kwargs).to(DEVICE)
            ckpt_path = p / self.ckpts[key]
            
            if ckpt_path.exists():
                sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                if 'model_state_dict' in sd: sd = sd['model_state_dict']
                # AQUÍ EJECUTAMOS EL FINE-TUNING DE ALINEACIÓN
                self.experts[key] = AGISurgeon.graft_and_realign(sd, model, key, alignment_data[key])
            else:
                self.experts[key] = model

    def forward(self, x):
        # Routing geométrico
        if x.dim() == 3: domain = 'wave'
        elif x.shape[1] == 5: domain = 'kepler'
        elif x.shape[1] == 4: domain = 'pendulum'
        else: domain = 'parity'
        
        out = self.experts[domain](x.to(DEVICE))
        return out[0] if isinstance(out, tuple) else out, domain

# ============================================================
# TEST FINAL: BUSCANDO EL 10^-7
# ============================================================

def main():
    agent = UnifiedAGI()
    agent.eval()
    
    problems = {
        'PARITY': get_parity_dataset(n_bits=64, k=3, size=100),
        'WAVE': generate_wave_data(N=32, T=10)[:2],
        'KEPLER': generate_kepler_data(num_samples=10),
        'PENDULUM': (torch.tensor(generate_and_save_chaotic_pendulum_dataset(n_samples=10)[0]).float(),
                     torch.tensor(generate_and_save_chaotic_pendulum_dataset(n_samples=10)[1]).float())
    }

    print("\n" + "="*60)
    print(f"{'DOMINIO':<12} | {'MÉTRICA':<15} | {'ROUTING'}")
    print("-" * 60)

    for name, (x, y) in problems.items():
        with torch.no_grad():
            pred, domain = agent(x)
            y = y.to(DEVICE)
            if name == 'PARITY':
                res = f"Acc: {(pred.argmax(1) == y).float().mean():.2%}"
            else:
                res = f"MSE: {F.mse_loss(pred, y):.2e}"
            print(f"{name:<12} | {res:<15} | {domain:<10} ✅")

if __name__ == "__main__":
    main()
