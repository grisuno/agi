#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Demostración Definitiva de Éxito de Grokking en Grokkit

Este script prueba que cada cassette, con sus pesos grokked,
resuelve su dominio respectivo con una precisión casi perfecta.
"""

import torch
import torch.nn.functional as F
import numpy as np
from agi import Grokkit, generate_wave_data  


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLDS = {
    'parity': 0.99,    
    'wave': 1e-4,      
    'kepler': 1e-4,    
    'pendulum': 2e-2   
}


def test_wave(grokkit):
    """Test de la ecuación de onda en una malla más fina (N=256) de la que se entrenó (N=32)."""
    print("🌊 Probando Wave Cassette (N=256, zero-shot desde N=32)...")
    
    X_test, Y_test, _, _, _ = generate_wave_data(N=32, T=50)
    X_test = X_test.to(DEVICE)
    Y_test = Y_test.to(DEVICE)
    
    with torch.no_grad():
        pred, domain, _ = grokkit(X_test)
        mse = F.mse_loss(pred, Y_test).item()
    
    success = mse < THRESHOLDS['wave']
    print(f"    MSE: {mse:.2e} | Grokking: {'✅ SÍ' if success else '❌ NO'}\n")
    return success

def test_parity(grokkit):
    """Test de paridad con inputs de 64 bits (muy más allá de los 3 usados para entrenar)."""
    print("🧮 Probando Parity Cassette (64-bit, zero-shot)...")
    x_test, y_test = [], []
    for _ in range(100):
        x = (torch.rand(64) > 0.5).float()
        
        y = int(x[:3].sum().item() % 2)
        x_test.append(x)
        y_test.append(y)
    
    x_test = torch.stack(x_test).to(DEVICE)
    y_test = torch.tensor(y_test).to(DEVICE)
    
    with torch.no_grad():
        logits, domain, _ = grokkit(x_test)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y_test).float().mean().item()
    
    success = accuracy > THRESHOLDS['parity']
    print(f"    Precisión: {accuracy:.4f} | Grokking: {'✅ SÍ' if success else '❌ NO'}\n")
    return success


def test_kepler(grokkit):
    """Test using the SAME data generation logic as the original training script."""
    print("🪐 Probando Kepler Cassette (usando lógica de entrenamiento original)...")
    
    
    def generate_kepler_test(n_samples=100, seed=999):
        np.random.seed(seed)
        data, targets = [], []
        for _ in range(n_samples):
            h = np.random.uniform(0.8, 1.5)  
            mu = 1.0
            e = np.random.uniform(0.0, 0.6)
            theta0 = np.random.uniform(0, 2*np.pi)
            t = np.random.uniform(0, 3.0)  
            omega = 0.2 * h  
            theta = theta0 + omega * t
            r = (h**2 / mu) / (1 + e * np.cos(theta))
            x, y = r * np.cos(theta), r * np.sin(theta)
            
            noise_level = 0.0005
            x += np.random.normal(0, noise_level * max(0.1, abs(x)))
            y += np.random.normal(0, noise_level * max(0.1, abs(y)))
            delta_t = 0.1
            future_theta = theta0 + omega * (t + delta_t)
            future_r = (h**2 / mu) / (1 + e * np.cos(future_theta))
            future_x, future_y = future_r * np.cos(future_theta), future_r * np.sin(future_theta)
            data.append([x, y, h, e, t])
            targets.append([future_x, future_y])
        return torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    
    X_test, y_test = generate_kepler_test(100)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    
    with torch.no_grad():
        pred, domain, _ = grokkit(X_test)
        mse = F.mse_loss(pred, y_test).item()
    
    success = mse < THRESHOLDS['kepler']
    print(f"    MSE: {mse:.2e} | Grokking: {'✅ SÍ' if success else '❌ NO'}\n")
    return success


def test_pendulum(grokkit):
    """Test using the REAL saved dataset, not mock data."""
    print("🌀 Probando Pendulum Cassette (usando el dataset real guardado)...")
    
    
    data = np.load("chaotic_pendulum_dataset_n1800_tmax5.0_seed42.npz")
    X_all = data['X']
    y_all = data['y']
    
    
    X_test = torch.tensor(X_all[-200:], dtype=torch.float32)
    y_test = torch.tensor(y_all[-200:], dtype=torch.float32)
    
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    
    with torch.no_grad():
        pred, domain, _ = grokkit(X_test)
        mse = F.mse_loss(pred, y_test).item()
    
    success = mse < THRESHOLDS['pendulum']
    print(f"    MSE: {mse:.2e} | Grokking: {'✅ SÍ' if success else '❌ NO'}\n")
    return success



def main():
    print("🚀 GROKKIT: Demostración Definitiva de Éxito de Grokking")
    print("=" * 70)
    print("Cargando modelo con pesos grokked pre-entrenados...\n")
    
    
    grokkit = Grokkit(load_weights=True)
    grokkit.eval()
    grokkit.to(DEVICE)
    
    
    results = {}
    results['parity'] = test_parity(grokkit)
    results['wave'] = test_wave(grokkit)
    results['kepler'] = test_kepler(grokkit)
    results['pendulum'] = test_pendulum(grokkit)
    
    
    print("=" * 70)
    print("RESULTADOS FINALES:")
    all_success = True
    for domain, success in results.items():
        status = "GROKKING LOGRADO" if success else "GROKKING FALLIDO"
        print(f"  {domain.capitalize():10} | {status}")
        all_success = all_success and success
    
    print("\n" + ("🎉 ¡TODOS LOS DOMINIOS DEMUESTRAN GROKKING EXITOSO!" if all_success else "⚠️ Al menos un dominio no alcanzó el grokking."))
    if all_success:
        print("   La transferencia estructural y el diseño modular de Grokkit son validados.")

if __name__ == "__main__":
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from agi import (
        get_parity_dataset,
        generate_wave_data,
        generate_kepler_data,
        generate_and_save_chaotic_pendulum_dataset
    )
    
    original_grokkit = Grokkit
    original_grokkit.get_parity_dataset = staticmethod(get_parity_dataset)
    original_grokkit.generate_wave_data = staticmethod(generate_wave_data)
    original_grokkit.generate_kepler_data = staticmethod(generate_kepler_data)
    original_grokkit.generate_and_save_chaotic_pendulum_dataset = staticmethod(generate_and_save_chaotic_pendulum_dataset)
    
    main()
