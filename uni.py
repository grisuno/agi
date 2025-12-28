#!/usr/bin/env python3
"""
AGI v0.1 - Demo Multi-Dominio
Resuelve Parity, Wave, Kepler y Pendulum en un solo script.
"""
import torch.nn.functional as F
import torch
import time
from unificado import UnifiedGrokkitAgent  # Tu agente funcional
from agi import (  # Importamos generadores de datos
    get_parity_dataset, generate_wave_data, 
    generate_kepler_data, generate_and_save_chaotic_pendulum_dataset
)

def batch_multi_domain():
    """Crea un batch con 4 problemas distintos"""
    # 1. Parity: 5 ejemplos de 64 bits
    parity_x, parity_y = get_parity_dataset(n_bits=64, k=3, size=5)
    
    # 2. Wave: 5 ejemplos N=32 steps
    wave_x, wave_y, _, _, _ = generate_wave_data(N=32, T=5)
    
    # 3. Kepler: 5 órbitas
    kepler_x, kepler_y = generate_kepler_data(num_samples=5)
    
    # 4. Pendulum: 5 estados caóticos
    pend_x, pend_y = generate_and_save_chaotic_pendulum_dataset(n_samples=5)
    pend_x = torch.tensor(pend_x)
    pend_y = torch.tensor(pend_y)
    
    return {
        'parity': (parity_x.float(), parity_y, "Paridad Binaria"),
        'wave': (wave_x.float(), wave_y, "Ecuación de Onda"),
        'kepler': (kepler_x.float(), kepler_y, "Órbita Kepleriana"),
        'pendulum': (pend_x.float(), pend_y, "Péndulo Caótico")
    }

def main():
    print("Aentic Grokked Integrated v0.1 - Demo Multi-Domain")
    print("=" * 60)
    
    # Inicializar agente
    CASSETTES = {
        'parity': "weights/grok_model_stage4_n64_d1024_adaptive.pth",
        'wave': "weights/wave_grok_cnn_physics_cassette.pth",
        'kepler': "weights/kepler_base_model.pth",
        'pendulum': "weights/symplectic_double_pendulum_grok_cassette.pth"
    }
    
    agent = UnifiedGrokkitAgent(CASSETTES)
    agent.eval()
    
    # Generar problemas
    problems = batch_multi_domain()
    
    # Métricas
    total_time = 0
    correct_routing = 0
    results = {}
    
    for domain, (x, y_true, desc) in problems.items():
        print(f"\n{'='*40}")
        print(f"Problem: {desc}")
        print(f"{'='*40}")
        
        # Medir tiempo de inferencia
        start = time.time()
        with torch.no_grad():
            output, predicted_domain, confidence = agent(x)
        elapsed = time.time() - start
        total_time += elapsed
        
        # Verificar routing
        is_correct = (predicted_domain == domain)
        correct_routing += int(is_correct)
        
        # Métricas específicas por dominio
        if domain == 'parity':
            pred = output.argmax(dim=1)
            accuracy = (pred == y_true).float().mean().item()
            print(f"  ✓ Predictions: {pred.tolist()}")
            print(f"  ✓ Ground truth: {y_true.tolist()}")
            print(f"  ✓ Accuracy: {accuracy:.2%}")
            
        elif domain == 'wave':
            mse = F.mse_loss(output, y_true).item()
            print(f"  ✓ MSE: {mse:.2e}")
            print(f"  ✓ Output shape: {output.shape}")
            
        elif domain == 'kepler':
            mse = F.mse_loss(output, y_true).item()
            print(f"  ✓ MSE: {mse:.2e}")
            print(f"  ✓ Orbits predicted: {output[:2].tolist()}...")
            
        elif domain == 'pendulum':
            mse = F.mse_loss(output, y_true).item()
            print(f"  ✓ MSE: {mse:.2e}")
            print(f"  ✓ State predicted: {output[:2].tolist()}...")
        
        print(f"  ✓ Domain Detected: {predicted_domain} {'✅' if is_correct else '❌'}")
        print(f"  ✓ Confidence: {confidence.max().item():.1%}")
        print(f"  ✓ Time: {elapsed*1000:.2f} ms")
        
        results[domain] = {
            'correct_routing': is_correct,
            'mse' if domain != 'parity' else 'accuracy': mse if domain != 'parity' else accuracy
        }
    
    # Resumen final
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"✓ Routing Accuracy: {correct_routing}/{len(problems)} ({correct_routing/len(problems):.2%})")
    print(f"✓ Time total inference: {total_time*1000:.2f} ms")
    print(f"✓ Time per problem: {total_time/len(problems)*1000:.2f} ms")
    
    for domain, metrics in results.items():
        metric_name = 'MSE' if domain != 'parity' else 'Accuracy'
        value = metrics['mse' if domain != 'parity' else 'accuracy']
        if domain == 'parity':
            print(f"  - {domain}: {metric_name} = {value:.2%}")
        else:
            print(f"  - {domain}: {metric_name} = {value:.2e}")
    
    

if __name__ == "__main__":
    torch.manual_seed(42)
    main()
