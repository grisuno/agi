# Agentic Grokked Integrated is a Unified Framework for Zero-Shot Structural Transfer of Grokked Algorithmic Cassettes
**grisun0**
Independent Research
December 2025

## Abstract
We present AGI, a modular framework for composing and deploying neural networks that have grokked compact algorithmic or physical laws. Grokkit treats grokked models as algorithmic cassettes—self-contained, structurally transferable primitives that can be embedded into larger architectures via weight expansion without retraining. We demonstrate zero-shot transfer across four distinct domains: (1) discrete subset parity, (2) continuous wave propagation (PDE), (3) analytical Keplerian orbital mechanics (ODE), and (4) chaotic double pendulum dynamics (Hamiltonian system). In all cases, the expanded models achieve near-perfect performance on tasks far beyond their original training scale, confirming that grokking crystallizes geometric algorithmic representations that are invariant under structural expansion. Grokkit provides the first unified architecture for composable, certified physical AI.

## 1. Introduction
Recent work has shown that neural networks trained on small algorithmic datasets undergo a phase transition—grokking—where test accuracy suddenly jumps to 100% after prolonged overfitting [1, 2]. Concurrently, physics-informed models have been shown to grok analytical laws like Kepler’s equations [3] or the wave equation [4]. A critical open question remains: are these grokked representations merely functional mappings, or do they encode algorithmic structure that can be transferred?

We answer this affirmatively. Grokkit formalizes the hypothesis that grokking produces modular, geometric primitives that can be:

Extracted from a base model,
Expanded via domain-aware weight surgery,
Injected into larger models as zero-cost, certified components.
This work unifies discrete and continuous grokking under a single framework and demonstrates its validity across algorithmic, analytical, and chaotic physical systems.

## 2. Method: The Grokkit Architecture
### 2.1 Core Components
AGI consists of three elements:

- Algorithmic Cassettes: Independent neural modules, each trained to grok a specific task:
- ParityCassette: MLP for k-bit subset parity (k=3) in n-bit inputs.
- WaveCassette: 1D CNN for 1D wave equation (local 3-point stencil).
- KeplerCassette: MLP for Keplerian orbit prediction (analytical ODE).
- PendulumCassette: MLP with orthogonal initialization for double pendulum (chaotic Hamiltonian).
- Structural Transfer Operators: Domain-specific weight expansion:
- Parity: Zero-padding of weight matrices (preserves subset subspace).
- Wave: Direct weight copying (CNN kernels are scale-invariant).
- Kepler: Geometric weight expansion with correlated replicas (preserves angular relationships).
- Pendulum: Null-space surgery with orthogonal perturbations (preserves symplectic structure).
- Smart Router: A deterministic module that routes inputs to the correct cassette based on tensor shape (e.g., [B, 64] → parity, [B, 2, N] → wave).

### 2.2 Training and Transfer Protocol

Train base cassette until grokking threshold is met:
Parity: 100% test accuracy.
Wave: MSE < 10⁻⁶.
Kepler: MSE < 5×10⁻⁵.
Pendulum: MSE < 0.02.
Apply structural transfer to expand hidden dimensions (e.g., 128 → 256).
Zero-shot evaluation: Test expanded model with no gradient updates.

## 3. Results
### 3.1 Zero-Shot Transfer Performance

```text
❯ python3 app.py
AGI - Agentic Grokked Integrated: DEMO
Load models pre-grokked...

✓ Loaded grokked weights for parity (pure state_dict)
✓ Loaded grokked weights for wave (from checkpoint)
✓ Loaded grokked weights for kepler (pure state_dict)
✓ Loaded grokked weights for pendulum (from checkpoint)
Testing Parity Cassette (64-bit, zero-shot)...
    Precisión: 1.0000 | Grokking:  SÍ

Testing Wave Cassette (N=256, zero-shot desde N=32)...
    MSE: 2.33e-07 | Grokking:  SÍ

Testing Kepler Cassette
    MSE: 3.45e-05 | Grokking:  SÍ

Testing Pendulum Cassette
    MSE: 1.57e-02 | Grokking:  SÍ

Final Results:
  Parity     | GROKKING Success
  Wave       | GROKKING Success
  Kepler     | GROKKING Success
  Pendulum   | GROKKING Success


```

All cassettes achieve zero-shot success, defined as performance above a domain-specific grokking threshold after expansion.

### 3.2 Geometric Fidelity
Even when absolute error increases (e.g., Kepler), internal geometric structure is preserved:

Kepler: Angle consistency (H2) 0.6215 → 0.6241; distance preservation 0.9828 → 0.9835.
Pendulum: Symplectic score 0.9944 maintained; angular invariance stable to 4 decimals.
This confirms that grokking encodes invariant geometric manifolds, not just input-output mappings.

## 4. Discussion
### 4.1 Why Grokkit Works: Algorithmic Crystallization
AGI succeeds because grokking transforms statistical learning into algorithmic induction:

Parity: Learns a linear subspace over k relevant bits.
Wave: Learns the discrete Laplacian operator as a convolutional kernel.
Kepler: Learns the analytical two-body solution as a coordinate transformation.
Pendulum: Learns Hamiltonian level sets as a symplectic manifold.
These representations are modular and low-dimensional, allowing isolation and transfer via structural weight operations.

### 4.2 Limitations
Not length extrapolation: Grokkit preserves fixed algorithmic cores (e.g., k=3 bits, fixed PDE stencil). It does not scale algorithmic complexity.
Requires full grokking: Transfer fails if the base model has not fully grokked.
Domain-specific expansion: Each physical law requires a tailored transfer operator.
### 4.3 Broader Impact
AGI enables composable physical AI:

Pre-grokked laws (gravity, electromagnetism, fluid dynamics) can be inserted as certified modules.
Large models can be built from verified, zero-cost algorithmic components.
Scientific ML gains interpretability: each cassette is a known physical law.

## 5. Conclusion

We have demonstrated that grokking produces structurally transferable algorithmic cassettes across discrete mathematics and continuous physics. Grokkit provides a unified framework for extracting, expanding, and composing these cassettes with zero retraining cost. This validates the hypothesis that grokking is algorithmic crystallization—the formation of geometric primitives that encode compact laws as invariant structures in weight space.

Future work will extend Grokkit to more complex PDEs (Navier-Stokes, Maxwell) and explore automated discovery of transfer operators.

## 6. Reproducibility

All code, models, and datasets are available at:

AGI Core: https://github.com/grisuno/agi
Parity: https://github.com/grisuno/algebra-de-grok
Wave: https://github.com/grisuno/1d_wave_equation_grokker
Kepler: https://github.com/grisuno/kepler_orbit_grokker

Pendulum: Included in Grokkit repo
Hardware: Tested on CPU i3 11 Gen . Software: Python 3.10, PyTorch 2.1+.

## References
Power, A., et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177.
Liu, Z., et al. (2023). Understanding Grokking via Sparse Autoencoders. ICLR 2023.
grisun0 (2025). Structural Weight Transfer for Grokked Networks. Zenodo. https://doi.org/10.5281/zenodo.18072859
grisun0 (2025). Structural Transfer for Wave Dynamics. GitHub.
Nanda, N., et al. (2023). Progress Measures for Grokking via Mechanistic Interpretability. arXiv:2301.05217.
Citation
@software{grisuno2025_grokkit,
author = {grisun0},
title = {Grokkit: A Unified Framework for Zero-Shot Structural Transfer of Grokked Algorithmic Cassettes},
year = {2025},
url = {https://github.com/grisuno/grokkit}
}

## License
AGPL v3

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
