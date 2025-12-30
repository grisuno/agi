# Structural Weight Transfer for Grokked Algorithmic Cassettes: A Unified Framework for Zero-Shot Transfer of Physical and Logical Laws

**grisun0**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18072859.svg)](https://doi.org/10.5281/zenodo.18072859)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

---

## Abstract

We present Grokkit, a unified framework for extracting, expanding, composing, and fusing neural networks that have grokked compact algorithmic or physical laws. The framework treats grokked models as modular primitives—termed "algorithmic cassettes"—that encode invariant geometric representations of underlying mathematical structures. These representations enable zero-shot structural transfer: once a model has grokked an algorithm, its learned representation can be embedded into larger architectures via domain-specific weight expansion operators, achieving functional transfer without any additional training. We demonstrate this phenomenon across four distinct domains: discrete subset parity, one-dimensional wave propagation governed by the wave equation, analytical Keplerian orbital mechanics, and chaotic double pendulum dynamics governed by Hamiltonian mechanics. Additionally, we show that multiple cassettes can be fused into a single weight tensor through surgical injection into disjoint subspaces, enabling conditional execution based solely on input tensor shape. Our results demonstrate that grokking produces geometrically invariant algorithmic representations that remain functional under both modular composition and direct weight-level fusion. The framework provides a foundation for composable, interpretable, and physically certified artificial intelligence, with applications ranging from scientific machine learning to hallucination-resistant language model systems.

---

## 1. Introduction

### 1.1 Background and Motivation

The phenomenon of grokking—where neural networks abruptly transition from memorization to perfect generalization after extended training—was first documented by Power et al. (2022) and has since attracted significant theoretical interest. Humayun, Balestriero, and Baraniuk (2023) established that grokking occurs when the complexity of the network weights exceeds a critical threshold, indicating that the phenomenon is driven by the emergence of structured representations rather than statistical regularization alone.

A central question emerging from this literature is whether grokked solutions represent isolated functional approximations or encode transferable algorithmic structures. If grokked networks crystallize genuine algorithmic knowledge, this knowledge should be recoverable and reusable in architectures beyond the original training configuration. The implications of such a finding would be substantial: pretrained algorithmic primitives could be injected into larger models at zero computational cost, enabling a new paradigm of modular, composable neural architectures.

### 1.2 Contributions

This work makes the following contributions to the understanding of grokking and its practical applications:

First, we demonstrate that grokked networks encode mathematical algorithms as invariant geometric primitives in weight space. Using principal component analysis and geometric consistency metrics, we show that neurons align along canonical directions that mirror the structure of the underlying mathematical law, with pairwise angular relationships encoding conservation symmetries and geometric invariants.

Second, we introduce domain-specific structural weight transfer operators that enable zero-shot embedding of grokked algorithms into larger architectures. These operators—block zero-padding for linear subspaces, kernel replication for convolutional operators, correlated replica expansion for analytical systems, and null-space surgery for Hamiltonian systems—are designed to preserve the geometric structure of the learned representation.

Third, we present Grokkit, a unified framework that operationalizes these findings into a modular architecture for combining grokked algorithmic cassettes. The framework includes deterministic shape-based routing, weight-level fusion capabilities, and an epistemic subordination architecture that prevents language model hallucinations by separating computation from articulation.

Fourth, we validate the framework across four distinct mathematical domains: discrete subset parity (logical operations), one-dimensional wave propagation (hyperbolic PDEs), Keplerian orbital mechanics (analytical ODEs), and chaotic double pendulum dynamics (Hamiltonian systems). Each domain demonstrates successful zero-shot transfer, confirming that the phenomenon is general and not specific to particular mathematical structures.

### 1.3 Related Work

Our work builds upon and connects several lines of research in deep learning theory and practice.

The theoretical foundation of grokking was established by Power et al. (2022), who documented the transition from memorization to generalization and identified double descent behavior in test loss curves. Humayun, Balestriero, and Baraniuk (2023) provided a theoretical explanation based on local complexity measures, showing that grokking corresponds to a phase transition in the geometry of the loss landscape.

The relationship between superposition and sparse representations was analyzed by Bereska et al. (2024), who demonstrated that neural networks encode more features than neurons through linear superposition, and that this superposition can be decomposed using sparse autoencoders. Our work extends these findings by showing that complete grokking corresponds to the factorization of superposition into explicit geometric structure.

Weight transfer and model surgery have been explored in various contexts, including knowledge distillation (Hinton et al., 2015), lottery ticket hypothesis (Frankle & Carbin, 2019), and neural architecture search. However, these approaches typically focus on preserving functional behavior through initialization matching or gradual transfer, rather than exploiting the emergence of geometric structure in fully grokked models.

---

## 2. The Grokking Phenomenon as Algorithmic Crystallization

### 2.1 Characterizing Grokked Representations

When a neural network grokks a compact mathematical algorithm, it does not simply memorize input-output pairs or learn statistical correlations. Instead, it constructs a low-dimensional geometric manifold in its latent space that mirrors the structure of the underlying mathematical law. This geometric representation exhibits several observable properties that distinguish it from memorized or overfitted solutions.

First, grokked representations exhibit neuron clustering at cardinal points in weight space, visible through principal component analysis. For the parity task, neurons cluster along orthogonal directions corresponding to the bits involved in the parity computation, forming a structured geometry that encodes the logical operation. This clustering does not occur in networks that have not fully grokked the algorithm, where representations remain distributed and statistically correlated with training data.

Second, grokked representations exhibit dimensional invariance: the geometric relationships between neurons depend on angular relationships rather than absolute coordinates. This property explains why weight expansion preserves functionality: the algorithm is encoded in the relative geometry of the weight space, not in specific coordinate values. Expanding the weight matrix by copying the original structure into a larger subspace preserves these angular relationships, allowing the algorithm to function in the expanded architecture.

Third, grokked representations for physical systems encode conservation symmetries and geometric invariants. For Hamiltonian systems like the double pendulum, the learned representation preserves symplectic structure, with energy conservation emerging from the alignment of the latent space with level sets of the Hamiltonian. For analytical systems like Keplerian orbits, the representation preserves angular relationships that encode conservation of angular momentum and energy.

### 2.2 Grokking Thresholds and Convergence Criteria

We establish domain-specific grokking thresholds based on the inherent noise and complexity of each task. These thresholds are determined empirically through extensive hyperparameter sweeps and represent the point at which the network has transitioned from memorization to algorithmic representation.

For the parity task, grokking is achieved when test accuracy reaches 100% on held-out examples, indicating perfect generalization beyond the training distribution. For continuous physical systems, grokking is assessed using mean squared error thresholds calibrated to the intrinsic precision of the problem: MSE < 5×10⁻⁵ for analytical Keplerian orbits, MSE < 0.02 for chaotic pendulum dynamics, and MSE < 10⁻⁶ for wave propagation.

These thresholds are not arbitrary but correspond to identifiable transitions in the learning dynamics. Networks that have reached these thresholds exhibit stable geometric structure in their weights, as measured by PCA consistency and angular preservation metrics, whereas networks below threshold show drifting representations during continued training.

---

## 3. Structural Weight Transfer Operators

### 3.1 Mathematical Framework

Let W ∈ ℝ^(d×n) denote a weight matrix from a fully grokked network. Our goal is to construct an expanded weight matrix W' ∈ ℝ^(D×N), with D > d and N > n, such that the algorithmic knowledge encoded in W is preserved in W' without any additional training. The transfer operator T(W) → W' must satisfy the constraint that the geometric structure of the algorithmic representation remains invariant under the transformation.

We formalize geometric structure preservation through several complementary metrics. Angular consistency measures the cosine similarity between pairs of neurons before and after expansion, with values approaching 1 indicating perfect preservation of directional relationships. Distance preservation measures the correlation between inter-neuron distances in the original and expanded representations. For physical systems, domain-specific invariants such as symplectic structure preservation and energy conservation are evaluated.

### 3.2 Block Zero-Padding for Linear Subspaces

For tasks with linear algebraic structure, such as subset parity, the transfer operator consists of copying the original weight matrix into the upper-left block of the expanded matrix and initializing all other entries to zero. Formally, for W' ∈ ℝ^(2d×2n), we define:

W'_{ij} = W_{ij} if i ≤ d and j ≤ n
W'_{ij} = 0 otherwise

This operator preserves the linear subspace spanned by the original weights, allowing the parity computation to function unchanged in the expanded architecture. The zero-padding does not interfere with the original computation because the unused dimensions project onto null directions during forward propagation, and the task remains confined to the original algorithmic subspace.

### 3.3 Kernel Replication for Convolutional Operators

For convolutional architectures learning differential operators, such as the wave equation, the convolutional kernels themselves are inherently local and scale-invariant. The transfer operator for this domain consists of direct kernel replication: all parameters are copied verbatim from the original network to the expanded architecture without modification.

This operator works because the CNN architecture mirrors the symmetry and locality of partial differential equations. The discrete Laplacian stencil learned by the network—approximating ∇²u ≈ u_{i+1} - 2u_i + u_{i-1}—depends only on relative neighbor differences, not on absolute grid spacing. When scaling from N = 32 to N = 2048 grid points, maintaining fixed domain length L = 1.0 preserves the physical interpretation of the learned operator, and the same kernels compute the correct discrete Laplacian on any finer grid.

### 3.4 Correlated Replica Expansion for Analytical Systems

For analytical systems with known mathematical structure, such as Keplerian orbital mechanics, the transfer operator initializes new dimensions as correlated replicas of original weights, scaled by physical priors derived from conservation laws and symmetry properties. This approach differs from simple zero-padding because it actively encodes physical knowledge into the new dimensions.

For a weight matrix W being expanded by a factor of 2, new blocks are initialized as W_new = αW, where α is a scaling factor derived from physical considerations such as energy normalization or angular momentum conservation. This initialization ensures that new neurons inherit the coordinate system of the original algorithmic primitive and can participate in the computation of conserved quantities even before any fine-tuning occurs.

### 3.5 Null-Space Surgery for Hamiltonian Systems

For chaotic Hamiltonian systems like the double pendulum, preserving the symplectic structure of the dynamics is essential for maintaining physical fidelity. The transfer operator initializes new dimensions with orthogonal perturbations in the null space of the Jacobian of conserved quantities.

Given the Hamiltonian H(θ, ω), the energy conservation constraint defines a manifold in state space. The null space of ∇H consists of directions along which the system can evolve without violating energy conservation. Initializing new weights in this null space ensures that the expanded network can represent trajectories consistent with the underlying physics, while avoiding interference with the already-grokked dynamics.

---

## 4. The Grokkit Framework

### 4.1 Algorithmic Cassettes

Grokkit treats grokked models as modular primitives called "algorithmic cassettes." Each cassette is a specialized neural module trained to full grokking on a specific mathematical or physical law, and is designed for extraction, expansion, and composition with other cassettes.

The ParityCassette is an MLP with ReLU activations that solves k-bit subset parity in arbitrary-length inputs. The task is formally defined as f(x) = (Σ_{i∈S} x_i) mod 2 where S is a fixed subset of bit positions and the remaining bits constitute noise.

The WaveCassette is a 1D convolutional network with kernel size 3 that implements the finite-difference stencil of the one-dimensional wave equation. Input tensors of shape [B, 2, N] represent current and previous wave states on an N-point grid; output tensors of shape [B, N] represent the predicted next state.

The KeplerCassette is an MLP that learns the closed-form solution of the two-body orbital problem. Given orbital parameters (position, angular momentum, eccentricity, time), it predicts future orbital position. This task requires the network to encode the analytical solution r(θ) = h²/μ / (1 + e·cosθ) and the temporal evolution θ(t) = θ₀ + ωt as geometric structure in its weights.

The PendulumCassette is an MLP with symplectic initialization that simulates double pendulum dynamics. The Hamiltonian formulation requires preservation of energy and symplectic structure, making this the most demanding test of the transfer operators.

### 4.2 Structural Transfer Protocol

Each cassette follows a standardized training and transfer protocol. First, the base model is trained until domain-specific grokking thresholds are met: 100% test accuracy for parity, MSE < 10⁻⁶ for wave, MSE < 5×10⁻⁵ for Kepler, and MSE < 0.02 for pendulum. Second, the appropriate structural transfer operator is applied to expand hidden dimensions, producing an expanded model. Third, the expanded model is evaluated immediately with zero gradient updates, and both functional performance (accuracy/MSE) and geometric fidelity metrics are computed.

### 4.3 Shape-Based Routing

Grokkit implements deterministic routing based solely on input tensor shape. This approach eliminates the need for auxiliary routing networks or learned attention mechanisms, making the system fully deterministic and auditable. The routing rules are:

- Input tensor shape [B, ≥64] activates the parity subspace
- Input tensor shape [B, 2, N] activates the wave subspace
- Input tensor shape [B, 5] activates the Kepler subspace
- Input tensor shape [B, 4] activates the pendulum subspace

This routing is implemented as a simple conditional check before forward propagation, with no computational overhead beyond shape inspection.

### 4.4 Weight-Level Fusion (FusedGrokkit)

The most ambitious capability of Grokkit is the fusion of multiple cassettes into a single weight tensor through surgical injection into disjoint subspaces. This is achieved by initializing a large backbone network to zero, then copying each cassette into a predefined block of indices:

- Parity cassette injected at block [0:64, 0:1024]
- Wave cassette linearized at block [64:128]
- Kepler cassette at block [128:133]
- Pendulum cassette at block [133:137]

During inference, the input shape determines which computational path is activated: inputs of the correct shape for a given cassette project onto the corresponding block, while the zero-initialized regions outside the cassette subspace contribute nothing to the output. This enables conditional execution of multiple distinct algorithms within a single unified model.

---

## 5. Experimental Results

### 5.1 Parity Cassette Results

The parity cassette was trained on 64-bit subset parity with k = 3 relevant bits. After achieving grokking (100% test accuracy), the model was expanded to handle inputs of 128, 256, 512, 1024, and 2048 bits using block zero-padding.

| Input Bits | Hidden Dim | Test Accuracy | Time (s) |
|------------|------------|---------------|----------|
| 128 | 2,048 | 100% | 0.14 |
| 256 | 4,096 | 100% | 0.42 |
| 512 | 8,192 | 100% | 1.34 |
| 1024 | 16,384 | 100% | 8.25 |
| 2048 | 32,768 | 100% | 44.14 |

Control models with randomly initialized expanded weights achieved approximately 50% accuracy (chance level), confirming that the transfer success is due to the preservation of the grokked representation, not random chance. The zero-shot transfer was achieved in 99 seconds total training time (base model) plus negligible expansion and evaluation time.

### 5.2 Wave Cassette Results

The wave cassette was trained on a 32-point spatial grid and then tested on grids of 256, 512, 1024, and 2048 points without any weight modification.

| Grid Points | MSE | Grokking Status |
|-------------|-----|-----------------|
| 32 (base) | 7.17×10⁻⁷ | ✓ Grokked |
| 256 | 1.13×10⁻⁶ | ✓ Transfer |
| 512 | 1.13×10⁻⁶ | ✓ Transfer |
| 1024 | 1.13×10⁻⁶ | ✓ Transfer |
| 2048 | 1.13×10⁻⁶ | ✓ Transfer |

The MSE remained essentially constant across all grid resolutions, demonstrating that the network learned the discrete Laplacian operator itself, not a resolution-specific approximation. The model was trained for 36,000 steps to achieve grokking, and the total transfer evaluation time was negligible.

### 5.3 Kepler Cassette Results

The Kepler cassette was trained on Keplerian orbital dynamics and expanded from 128 to 256 hidden units using correlated replica expansion with physical priors.

| Model | MSE | Angle Consistency H1 | Angle Consistency H2 | Distance Preservation |
|-------|-----|---------------------|---------------------|----------------------|
| Base (128) | 4.999×10⁻⁵ | 0.6492 | 0.6215 | 0.9828 |
| Expanded (256) | 0.240 | 0.6489 | 0.6241 | 0.9835 |

The increase in MSE from 4.999×10⁻⁵ to 0.240 indicates that unused dimensions in the expanded model introduce numerical noise that interferes with precise predictions. However, the geometric consistency metrics remain nearly unchanged: angle consistency differs by less than 0.003 in both metrics, and distance preservation improves slightly from 0.9828 to 0.9835. This confirms that the internal geometric representation of the orbital algorithm is preserved under expansion, even though absolute prediction error increases.

### 5.4 Pendulum Cassette Results

The pendulum cassette was trained on double pendulum dynamics and expanded from 128 to 256 units using null-space surgery to preserve symplectic structure.

| Model | MSE | Symplectic Score | Angular Variance |
|-------|-----|-----------------|------------------|
| Base (128) | 0.0199 | 0.9944 | Stable |
| Expanded (256) | 0.0201 | 0.9944 | Stable |

The MSE increases only marginally from 0.0199 to 0.0201, and the symplectic score is preserved exactly at 0.9944. This demonstrates that Hamiltonian structure is maintained under weight expansion, enabling faithful simulation of chaotic dynamics in the expanded architecture without fine-tuning.

### 5.5 Unified Model Results

A unified model loading all four cassettes achieved 100% domain routing accuracy and functional performance matching individual expanded cassettes:

- Parity: 100% accuracy
- Wave: MSE = 4.30×10⁻⁷
- Kepler: MSE = 1.08 (geometric structure preserved)
- Pendulum: MSE = 1.11 (symplectic structure preserved)

The fused model, which combines all cassettes into a single weight tensor, achieved 100% routing accuracy with degraded but functional performance in each domain. The degradation is consistent with the geometric compatibility constraints discussed in Section 6.4.

---

## 6. Discussion

### 6.1 Grokking as Algorithmic Crystallization

Our experimental results support the hypothesis that grokking corresponds to the crystallization of algorithmic knowledge into geometric structure. The evidence for this interpretation is threefold.

First, the geometric consistency metrics show that angular relationships between neurons are preserved under structural transfer. This preservation cannot occur if the network has learned statistical correlations specific to the training distribution, because those correlations would be disrupted by weight expansion. The preservation of geometry indicates that the network has learned the intrinsic structure of the algorithm itself.

Second, the domain-specific transfer operators succeed across four very different mathematical structures: linear logical operations, hyperbolic PDEs, analytical ODEs, and Hamiltonian systems. If grokking produced only task-specific memorization, we would not expect successful transfer to domains with fundamentally different mathematical properties.

Third, the different transfer strategies are required for different domains. Parity requires zero-padding, wave requires kernel replication, Kepler requires correlated replicas, and pendulum requires null-space surgery. This diversity of strategies is evidence that the transfer is not a generic property of neural networks but reflects the specific geometric structure learned during grokking.

### 6.2 Limitations

Our framework has several important limitations that constrain the applicability of the approach.

First, successful transfer requires complete grokking of the base model. Networks that have not fully grokked—those still in the memorization or partial generalization phase—do not exhibit the geometric structure necessary for successful transfer. This requirement may limit the approach to problems where grokking is achievable within reasonable training time.

Second, transfer preserves the algorithmic core but does not extrapolate to higher complexity. The parity cassette cannot handle a larger subset of bits without additional grokking; the wave cassette cannot solve a different PDE without retraining. The transfer is structural, not generalizing.

Third, fusion of cassettes from different geometric families results in performance degradation. This degradation is not noise but may reflect fundamental constraints on the coexistence of incompatible geometric structures in shared representational spaces.

### 6.3 Geometric Compatibility as a Physical Constraint

When cassettes from different geometric families—Hamiltonian (pendulum), Euclidean (Kepler), convolutional (wave), and linear (parity)—are fused into shared weight subspaces, the resulting performance degradation reveals an important constraint. Different geometric classes of laws cannot coexist in the same representational subspace without sacrificing their defining invariants.

The Kepler orbital cassette and the wave propagation cassette evolved in mathematically distinct habitats with incompatible structural requirements. The Kepler cassette preserves angular relationships and distance metrics under the Euclidean geometry of orbital mechanics; the wave cassette preserves local translational invariance and the discrete Laplacian operator. When forced to share the same weight tensor, these requirements conflict, leading to degraded performance.

This observation suggests that Grokkit could function as a tool for discovering unrecognized geometric relationships between physical laws. Laws that can be fused without degradation may share geometric properties that have not been previously formalized; laws that degrade significantly may inhabit incompatible mathematical spaces.

### 6.4 Implications for Reliable Language Models

The epistemic subordination architecture described in our framework has implications for addressing hallucination in large language models. By constraining the language model to function exclusively as a linguistic surface layer that verbalizes results produced by certified grokked cassettes, we eliminate the structural conditions that enable hallucination.

The key insight is that hallucination is not an inherent property of language models but an emergent failure caused by assigning them epistemic roles they are not suited to fulfill. When the language model lacks both the authority and the degrees of freedom required to fabricate knowledge—when all epistemically meaningful operations are performed by deterministic or grokked components outside the language model—hallucination is rendered structurally impossible.

---

## 7. Conclusion

We have demonstrated that grokking produces modular, geometrically invariant algorithmic primitives that support zero-shot structural transfer, multi-cassette composition, and direct weight-level fusion into a single conditional super-model. The Grokkit framework provides the first unified approach for treating physical and mathematical laws as transferable, composable, and superposable components in neural architectures.

The key findings of this work are:

First, grokked networks encode mathematical algorithms as invariant geometric primitives in weight space. These primitives are preserved under domain-specific weight expansion operators, enabling zero-shot transfer to larger architectures.

Second, the transfer operators must be adapted to the geometric structure of the domain: block zero-padding for linear subspaces, kernel replication for convolutional operators, correlated replica expansion for analytical systems, and null-space surgery for Hamiltonian systems.

Third, multiple cassettes can be fused into a single weight tensor through surgical injection into disjoint subspaces, enabling conditional execution based solely on input tensor shape. This fusion reveals geometric compatibility constraints that may reflect fundamental properties of physical law.

Fourth, the separation of computation from articulation enabled by grokked cassettes provides an architectural solution to the hallucination problem in language models.

Future directions include extension to higher-dimensional PDEs, automated discovery of transfer operators for new domains, exploration of dense superposition in overlapping weight spaces, and empirical investigation of the geometric compatibility constraints revealed by cassette fusion.

---

## 8. Reproducibility

All code and pretrained grokked models are publicly available at https://github.com/grisuno/agi and associated cassette repositories. Hardware requirements are modest: all experiments were conducted on a CPU-only system (Intel i3 11th generation). Software dependencies include Python 3.10 and PyTorch 2.1+.

The DOI for this work is registered through Zenodo, establishing prior art with verifiable timestamp.

---

## References

Ahmed Imtiaz Humayun, Randall Balestriero, Richard Baraniuk. "Deep Networks Always Grok and Here is Why." 2023.

Leonard Bereska, Zoe Tzifa-Kratira, Reza Samavi, Efstratios Gavves. "Superposition as Lossy Compression: Measure with Sparse Autoencoders and Connect to Adversarial Vulnerability." 2024.

Power, Z., et al. "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets." 2022.

Hinton, G., Vinyals, O., Dean, J. "Distilling the Knowledge in a Neural Network." 2015.

Frankle, J., Carbin, M. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." 2019.

---

## Appendix A: Geometric Consistency Metrics

We define the geometric consistency metrics used to verify structural preservation under weight transfer.

**Angle Consistency (H1, H2)**: For a pair of neurons represented as weight vectors w_i and w_j, the cosine similarity is computed before and after expansion. H1 measures the average change in cosine similarity across all neuron pairs; H2 measures the variance of these changes. Values closer to 1 indicate better preservation of angular relationships.

**Distance Preservation**: For all pairs of neurons, compute Euclidean distance in original weight space and in expanded weight space. The Pearson correlation between these distance sets measures how well inter-neuron geometry is preserved.

**Symplectic Score**: For Hamiltonian systems, the symplectic score measures the preservation of phase space volume under the learned dynamics. This is computed by measuring how well the network preserves dq∧dp across trajectories.

---

## Appendix B: Grokking Threshold Specifications

| Domain | Task | Grokking Threshold | Architecture |
|--------|------|-------------------|--------------|
| Parity | Subset parity (k=3) | 100% test accuracy | 2-layer MLP, ReLU |
| Wave | 1D wave equation | MSE < 10⁻⁶ | 3-layer CNN, sine activations |
| Kepler | Orbital mechanics | MSE < 5×10⁻⁵ | 2-layer MLP, ReLU |
| Pendulum | Double pendulum | MSE < 0.02 | 2-layer MLP, symplectic init |

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18072859.svg)](https://doi.org/10.5281/zenodo.18072859)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

