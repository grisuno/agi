#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FusedAGI - HYBRIDO FINAL (Arquitectura Casette + Entrenamiento App)
Inyección del truco Superposición+LC para resucitar Kepler.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from copy import deepcopy

# Imports del framework
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)

from agi import (
    ParityCassette, KeplerCassette, PendulumCassette, WaveCassette,
    get_parity_dataset, generate_wave_data, 
    generate_kepler_data, generate_and_save_chaotic_pendulum_dataset
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================= #
# 0. MÓDULOS DE SUPERPOSICIÓN (Extraídos de app.py)                     #
# ============================================================================= #

class SuperpositionSAE(nn.Module):
    """Sparse Autoencoder para forzar la estructura geométrica en el espacio latente."""
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.W = nn.Parameter(torch.randn(d_model, d_sae) / math.sqrt(d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
    
    def forward(self, x):
        z = F.relu(x @ self.W + self.b_enc)
        x_recon = z @ self.W.t()
        return x_recon, z
    
    def get_metrics(self, z):
        with torch.no_grad():
            f_i = z.abs().sum(dim=0)
            p_i = f_i / (f_i.sum() + 1e-12)
            p_safe = p_i[p_i > 1e-10]
            h_p = -torch.sum(p_safe * torch.log(p_safe + 1e-12))
            f_eff = torch.exp(h_p)
            psi = f_eff / self.d_model
            return psi.item(), f_eff.item()

class ComplexityAnalyzer:
    @staticmethod
    def measure_lc(model, x, epsilon=0.01):
        """Mide la complejidad de circuito (neuronas muertas/activas)."""
        model.eval()
        with torch.no_grad():
            # Padding manual para asegurar dimensiones
            if x.shape[1] < 64: x = F.pad(x, (0, 64 - x.shape[1]))
            
            # Pasar solo hasta fc2 para analizar la latencia
            z1 = model.fc1(x)
            h1 = torch.relu(z1)
            z2 = model.fc2(h1)
            
            # LC: Porcentaje de neuronas "muertas" o inactivas
            n_inactive = (z2.abs() < epsilon).float().sum(dim=1).mean()
            return n_inactive

# ============================================================================= #
# 1. EL CEREBRO UNIFICADO (Tu arquitectura sólida)                          #
# ============================================================================= #

class FusedAGIBrain(nn.Module):
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(64, hidden_dim)
        self.wave_conv = WaveCassette(hidden_dim=64).to(DEVICE)
        self.wave_projector = nn.Linear(32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_split_idx = {'parity': 2, 'wave': 34, 'kepler': 36, 'pendulum': 40}
        self.out = nn.Linear(hidden_dim, 40)

    def forward(self, x):
        # Routing original (sin tocar)
        domain = 'parity'
        if x.dim() == 3: domain = 'wave'
        elif x.shape[1] == 5: domain = 'kepler'
        elif x.shape[1] == 4: domain = 'pendulum'

        if domain == 'wave':
            with torch.no_grad(): 
                h_wave_raw = self.wave_conv(x)
            if h_wave_raw.dim() > 2: h_wave_raw = h_wave_raw.squeeze(1)
            h = F.relu(self.wave_projector(h_wave_raw))
        else:
            if x.shape[1] < 64: x = F.pad(x, (0, 64 - x.shape[1]))
            h = F.relu(self.fc1(x))
        
        h = F.relu(self.fc2(h))
        full_out = self.out(h)
        
        s = self.output_split_idx
        if domain == 'parity': return full_out[:, :s['parity']], domain
        elif domain == 'wave': return full_out[:, s['parity']:s['wave']], domain
        elif domain == 'kepler': return full_out[:, s['wave']:s['kepler']], domain
        elif domain == 'pendulum': return full_out[:, s['kepler']:s['pendulum']], domain
        return full_out, domain

# ============================================================================= #
# 2. CIRUJANO (Tu trasplante perfecto)                                         #
# ============================================================================= #

class SurgicalFusion:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.checkpoints = {
            'parity': "grok_model_stage4_n64_d1024_adaptive.pth",
            'wave': "wave_grok_cnn_physics_cassette.pth",
            'kepler': "kepler_base_model.pth",
            'pendulum': "symplectic_double_pendulum_grok_cassette.pth"
        }

    def load_expert_weights(self, domain):
        path = self.weights_dir / self.checkpoints[domain]
        if not path.exists(): return None
        sd = torch.load(path, map_location=DEVICE, weights_only=False)
        return sd.get('model_state_dict', sd)

    def transplant(self, brain):
        print("🔪 INICIANDO CIRUGÍA DE FUSIÓN...")
        master_sd = brain.state_dict()
        
        # --- PARITY ---
        p_sd = self.load_expert_weights('parity')
        if p_sd:
            print("  [P] Inyectando Parity...")
            master_sd['fc1.weight'][:1024, :].copy_(p_sd['fc1.weight'])
            master_sd['fc1.bias'][:1024].copy_(p_sd['fc1.bias'])
            master_sd['fc2.weight'][:1024, :1024].copy_(p_sd['fc2.weight'])
            master_sd['fc2.bias'][:1024].copy_(p_sd['fc2.bias'])
            master_sd['out.weight'][:2, :1024].copy_(p_sd['out.weight'])
            master_sd['out.bias'][:2].copy_(p_sd['out.bias'])
        
        # --- KEPLER ---
        k_sd = self.load_expert_weights('kepler')
        if k_sd:
            offset = 1024
            print(f"  [K] Inyectando Kepler ({offset}-{offset+127})...")
            master_sd['fc1.weight'][offset:offset+128, :5].copy_(k_sd['net.0.weight'])
            master_sd['fc1.bias'][offset:offset+128].copy_(k_sd['net.0.bias'])
            master_sd['fc2.weight'][offset:offset+128, offset:offset+128].copy_(k_sd['net.2.weight'])
            master_sd['fc2.bias'][offset:offset+128].copy_(k_sd['net.2.bias'])
            master_sd['out.weight'][34:36, offset:offset+128].copy_(k_sd['net.4.weight'])
            master_sd['out.bias'][34:36].copy_(k_sd['net.4.bias'])

        # --- PENDULUM ---
        pd_sd = self.load_expert_weights('pendulum')
        if pd_sd:
            offset = 1152
            print(f"  [D] Inyectando Pendulum ({offset}-{offset+127})...")
            master_sd['fc1.weight'][offset:offset+128, :4].copy_(pd_sd['net.0.weight'])
            master_sd['fc1.bias'][offset:offset+128].copy_(pd_sd['net.0.bias'])
            master_sd['fc2.weight'][offset:offset+128, offset:offset+128].copy_(pd_sd['net.2.weight'])
            master_sd['fc2.bias'][offset:offset+128].copy_(pd_sd['net.2.bias'])
            master_sd['out.weight'][36:40, offset:offset+128].copy_(pd_sd['net.4.weight'])
            master_sd['out.bias'][36:40].copy_(pd_sd['net.4.bias'])

        # --- WAVE ---
        w_sd = self.load_expert_weights('wave')
        if w_sd:
            print("  [W] Cargando WaveCassette...")
            try:
                brain.wave_conv.load_state_dict(w_sd)
                for p in brain.wave_conv.parameters(): p.requires_grad = False
            except Exception as e: print(f"    ⚠️ Error wave: {e}")

        brain.load_state_dict(master_sd)
        print("✅ CIRUGÍA LISTA.")
        return brain

# ============================================================================= #
# 3. RECUPERACIÓN AVANZADA (Inyectando Superposición + LC)                   #
# ============================================================================= #

def recovery_fine_tuning(brain):
    print("\n⚡ INICIANDO RECUPERACIÓN CON SUPERPOSICIÓN (SAE)...")
    
    # --- INICIALIZAR SAE ---
    # Relación 1:4 como en tu app.py (2048 -> 8192)
    sae = SuperpositionSAE(d_model=2048, d_sae=2048*4).to(DEVICE)
    
    # --- OPTIMIZADORES ---
    # 1. Para el Cerebro (ajuste fino)
    optimizer_brain = torch.optim.AdamW([
        {'params': brain.wave_projector.parameters(), 'lr': 1e-3},
        {'params': brain.fc1.parameters(), 'lr': 1e-5}, 
        {'params': brain.fc2.parameters(), 'lr': 1e-5}, 
        {'params': brain.out.parameters(), 'lr': 1e-4}
    ])
    
    # 2. Para el SAE (separado, como en app.py)
    optimizer_sae = torch.optim.AdamW(sae.parameters(), lr=1e-3)
    
    epochs = 2000
    print(f"Entrenando por {epochs} epochs con Superposición Activa...")
    
    print(f"{'Epoch':<6} | {'Loss':<8} | {'P':<6} | {'W':<8} | {'K':<8} | {'D':<8} | {'ψ':<5} | {'LC':<5}")
    print("-" * 90)
    
    for epoch in range(epochs):
        brain.train()
        sae.train()
        optimizer_brain.zero_grad()
        optimizer_sae.zero_grad()
        
        # --- BATCHES (Mixto) ---
        px, py = get_parity_dataset(n_bits=64, k=3, size=32)
        wx, wy = generate_wave_data(N=32, T=10)[:2]
        kx, ky = generate_kepler_data(num_samples=32)
        pdx_data = generate_and_save_chaotic_pendulum_dataset(32)
        pdx = torch.tensor(pdx_data[0]).float()
        pdy = torch.tensor(pdx_data[1]).float()
        
        # --- 1. FORWARD CEREBRO (Task Loss) ---
        p_out, _ = brain(px.to(DEVICE))
        w_out, _ = brain(wx[:5].to(DEVICE))
        k_out, _ = brain(kx.to(DEVICE))
        pd_out, _ = brain(pdx.to(DEVICE))
        
        loss_p = F.cross_entropy(p_out, py.to(DEVICE))
        w_target = wy[:5].flatten(1) if wy.dim() > 2 else wy[:5]
        loss_w = F.mse_loss(w_out, w_target.to(DEVICE))
        loss_k = F.mse_loss(k_out, ky.to(DEVICE))
        loss_pd = F.mse_loss(pd_out, pdy.to(DEVICE))
        
        loss_task = loss_p + (20.0 * loss_w) + (50.0 * loss_k) + (10.0 * loss_pd)
        
        # --- 2. FORWARD SAE (Superposition Loss) ---
        # Usamos Kepler como referencia principal para la geometría, ya que es el más crítico
        k_input_padded = F.pad(kx.to(DEVICE), (0, 64 - kx.shape[1]))
        h_k = F.relu(brain.fc1(k_input_padded))
        h_k = F.relu(brain.fc2(h_k)) # Esta es la latente de Kepler en el cerebro unificado
        
        # SAE reconstruye esta latente
        x_recon, z_sae = sae(h_k.detach()) 
        loss_sae_recon = F.mse_loss(x_recon, h_k.detach())
        loss_sae_sparse = z_sae.norm(p=1)
        loss_sae = loss_sae_recon + 0.01 * loss_sae_sparse
        
        # --- 3. BACKPROP ---
        loss_task.backward()
        optimizer_brain.step()
        
        optimizer_sae.zero_grad()
        loss_sae.backward()
        optimizer_sae.step()
        
        # --- 4. MONITOREO (PSI + LC) ---
        if epoch % 100 == 0:
            psi, _ = sae.get_metrics(z_sae)
            lc_val = ComplexityAnalyzer.measure_lc(brain, kx)
            print(f"{epoch:<6} | {loss_task.item():.4f}      | {loss_p.item():.4f} | {loss_w.item():.2e} | {loss_k.item():.2e} | {loss_pd.item():.2e} | {psi:.3f} | {lc_val:.2f}")

# ============================================================================= #
# MAIN                                                                         #
# ============================================================================= #

def main():
    print("=" * 80)
    print("🚀 UNIFIED AGI - ARQUITECTURA FINAL + SUPERPOSICIÓN")
    print("=" * 80)
    
    brain = FusedAGIBrain(hidden_dim=2048).to(DEVICE)
    
    surgeon = SurgicalFusion("./")
    brain = surgeon.transplant(brain)
    
    # Aplicar el truco de superposición
    recovery_fine_tuning(brain)
    
    print("\n" + "=" * 80)
    print("🧪 BENCHMARK FINAL")
    print("=" * 80)
    
    problems = {
        'PARITY': get_parity_dataset(n_bits=64, k=3, size=100),
        'WAVE': generate_wave_data(N=32, T=10)[:2],
        'KEPLER': generate_kepler_data(num_samples=100),
        'PENDULUM': (
            torch.tensor(generate_and_save_chaotic_pendulum_dataset(100)[0]).float(),
            torch.tensor(generate_and_save_chaotic_pendulum_dataset(100)[1]).float()
        )
    }

    brain.eval()
    for name, (x, y) in problems.items():
        with torch.no_grad():
            pred, domain = brain(x.to(DEVICE))
            y = y.to(DEVICE)
            
            if name == 'PARITY':
                res = f"Acc: {(pred.argmax(1) == y).float().mean():.2%}"
            else:
                if y.dim() > 2: y = y.flatten(1)
                res = f"MSE: {F.mse_loss(pred, y):.2e}"
            
            print(f"{name:<12} | {res:<20} | {'✅' if '100' in res or 'e-0' in res else '🔴'}")

if __name__ == "__main__":
    main()
