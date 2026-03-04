# Paper 2: Exactly Where LHFGSO Works in the Pipeline

## Paper: Hybrid Optimization Enabled Deep-Learning for Prostate Cancer Detection
## Journal: Sensing and Imaging (2024)

---

## Overview

LHFGSO is used at **two distinct points** in the pipeline:

1. **Training the multi-objective SegNet** (segmentation stage)
2. **Training the DMN** (cancer detection stage)

These are two separate applications of the same optimizer on two different networks with two different fitness functions.

**Critical difference from Paper 1:** In Paper 1, HFGSO optimized only the FC layer of DRN (a deep network with conv+residual+FC). In Paper 2, LHFGSO optimizes **all weights and biases of DMN** — because DMN is a pure multi-layer classifier with no convolutional layers. This is a fundamental architectural difference.

---

## 1. LHFGSO in SegNet Training (Segmentation)

### What is SegNet?

Same encoder-decoder architecture as Paper 1:

```
SegNet Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ENCODER (13 Conv layers)              DECODER (13 layers)      │
│  ┌────────┐  ┌────────┐  ┌────┐  ┌────────┐  ┌────────┐       │
│  │ Conv+BN│→ │ Conv+BN│→ │... │→ │Upsample│→ │Upsample│→ ...  │
│  │ +Pool  │  │ +Pool  │  │    │  │ +Conv  │  │ +Conv  │       │
│  └────────┘  └────────┘  └────┘  └────────┘  └────────┘       │
│       │                              ↑                          │
│       └──── pooling indices ─────────┘                          │
│                                                                 │
│  Input: Pre-processed MRI (B_e)    Output: Segmentation map     │
│                                    (pixel-wise cancer mask C_t) │
│                                                                 │
│  Final layer: Pixel-wise classification (softmax)               │
│               ↑                                                 │
│         THIS is where LHFGSO targets                            │
└─────────────────────────────────────────────────────────────────┘
```

### What does LHFGSO optimize in SegNet?

**From paper Section 3.3.3:** *"The LHFGSO model, an optimization technique, is used to train the created multi-objective SegNet model."*

Same as Paper 1 — the encoder's 13 convolutional layers learn features through standard forward-backward passes. LHFGSO targets the **trainable parameters at the decoder's final pixel-classification layer** — the layer that assigns each pixel its class label (cancer / non-cancer).

### What is the fitness function?

**The fused Dice + Cross-Entropy loss (Equation 4 in paper):**

```
E(a, b) = (1 − σ) × [Σ_d a_d log(b_d)] − σ × log[(2Σ(b_d × a_d) + F) / (Σb_d + Σa_d + F)]
                      \_________________/         \________________________________________/
                       Cross-entropy part                    Dice coefficient part

Where:  σ = 0.75  (75% weight to Dice)
        F = 1e-15 (smoothing to avoid numerical instability)
        a_d = predicted label for category d
        b_d = true label for category d
```

### What extra does LHFGSO bring over HFGSO (Paper 1)?

The SegNet training follows the same population-based search pattern as Paper 1, but LHFGSO adds the **Light Spectrum Optimizer (LSO)** component:

```
HFGSO (Paper 1):                    LHFGSO (Paper 2):
┌──────────────────┐                ┌──────────────────────────────┐
│ Firefly Algorithm │                │ Firefly Algorithm             │
│ (attraction-based │                │ (attraction-based search)     │
│  search)          │                │                               │
│        +          │                │        +                      │
│ HGSO              │                │ HGSO                          │
│ (gas solubility   │                │ (gas solubility dynamics)     │
│  dynamics)        │                │                               │
└──────────────────┘                │        +                      │
                                    │ LSO (NEW)                     │
                                    │ (light-spectrum-inspired      │
                                    │  differential perturbation)   │
                                    └──────────────────────────────┘
```

**LSO adds:** From paper Section 3.3.3 — *"The rainbow effect, which occurs when sunlight passes through a drop of water, serves as the foundation for the LSO metaheuristic algorithm. LSO appears to have struck a good balance between exploitation and exploration."*

The LSO update introduces a **differential evolution-style perturbation** using random vectors and solutions from the current population:

```
q_new = (q_s1 + GT5 × (q_s2 − q_s3)) × t_vec + (1 − t_vec) × q_current

Where: q_s1, q_s2, q_s3 = randomly selected solutions from population
       GT5 = normally distributed scalar
       t_vec = random vector with values in (0, 1)
```

This additional mechanism diversifies the search beyond what FA + HGSO alone achieve.

---

## 2. LHFGSO in DMN Training (Cancer Detection)

### What is DMN?

DMN (Deep Maxout Network) is fundamentally different from DRN. It is a **pure multi-layer classifier** — it has NO convolutional layers, NO pooling, NO residual blocks. It is entirely composed of maxout layers:

```
DMN Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Input: Extracted features F = {LBP, SLBT, mean, variance,     │
│         kurtosis, skewness, entropy}                            │
│         ↓                                                       │
│  ╔══════════════════════════════╗                                │
│  ║ Maxout Layer 1               ║                               │
│  ║ p1_{o,k} = max_{k∈[1,a1]}   ║  ← LHFGSO optimizes          │
│  ║   (F · N_{ok} + m_{ok})     ║     weights N and bias m      │
│  ╠══════════════════════════════╣                                │
│  ║ Maxout Layer 2               ║                               │
│  ║ p2_{o,k} = max_{k∈[1,a2]}   ║  ← LHFGSO optimizes          │
│  ║   (p1 · N_{ok} + m_{ok})    ║     weights N and bias m      │
│  ╠══════════════════════════════╣                                │
│  ║ ...                          ║                               │
│  ╠══════════════════════════════╣                                │
│  ║ Maxout Layer h (final)       ║                               │
│  ║ ph_{ok} = max_{k∈[1,an]}    ║  ← LHFGSO optimizes          │
│  ║   (f^{h-1} · N_{ok} + m_{ok})║    weights N and bias m      │
│  ╚══════════════╤═══════════════╝                                │
│                 ↓                                                │
│            Output I_o: Cancer / No Cancer                        │
│                                                                 │
│  ★ LHFGSO OPTIMIZES ALL LAYERS — every N (weight) and m (bias) │
│                                                                 │
│  Total: 230,931 params (902 KB) — small enough for              │
│         population-based optimization                           │
└─────────────────────────────────────────────────────────────────┘
```

### What does LHFGSO optimize in DMN?

**From paper Section 3.6.2:** *"The developed LHFGSO model is used to train DMN."*

**From paper Section 3.6.1:** *"h is DMN's total number of layers, m is the bias of the network, and N is the weight of the network"*

LHFGSO optimizes **ALL weights N and biases m across ALL h layers of DMN**. This is possible because:

1. **DMN has no convolutional layers** — it is purely a multi-layer maxout classifier
2. **DMN takes extracted features as input** (not raw images) — the heavy feature extraction is already done by SegNet + LBP/SLBT/stats
3. **The total model is only 230,931 parameters** — small enough for population-based search

### This is fundamentally different from Paper 1

| Aspect | Paper 1 (HFGSO on DRN) | Paper 2 (LHFGSO on DMN) |
|--------|------------------------|-------------------------|
| Network type | Deep CNN with conv+residual+FC | Pure multi-layer maxout classifier |
| What HFGSO/LHFGSO optimizes | **Only FC layer** {P, U} | **ALL layers** {N, m} across h layers |
| What uses backpropagation | Conv, BatchNorm, Residual blocks | Nothing — DMN is fully optimized by LHFGSO |
| Why this scope is possible | DRN has millions of conv params — too many for population search | DMN has ~230K total params — feasible for population search |
| Input to the classifier | Raw augmented images | Extracted features (7 handcrafted features) |

### What is the fitness function?

**MSE — Mean Squared Error (Equation 36 in paper):**

```
MSE = (1/h) × Σ_{o=1}^{h} (I*_o − I_o)²

Where:  I*_o = expected output (ground truth label)
        I_o  = DMN's predicted output
        h    = number of samples
```

### How does the optimization work step by step?

```
Step 1: Initialize a POPULATION of candidate DMN configurations
        Each candidate = ALL weights and biases across ALL maxout layers

        Candidate 1: {N_layer1=[...], m_layer1=[...],
                      N_layer2=[...], m_layer2=[...],
                      ...
                      N_layerH=[...], m_layerH=[...]}

        Candidate 2: {N_layer1=[...], m_layer1=[...], ...}
        ...

Step 2: For EACH candidate:
        → Load ALL its weights (N) and biases (m) into DMN
        → Feed extracted features F through DMN:
          Layer 1: p1 = max(F · N_1 + m_1)
          Layer 2: p2 = max(p1 · N_2 + m_2)
          ...
          Layer h: output = max(p^{h-1} · N_h + m_h)
        → Compute MSE between DMN output and ground truth
        → This MSE = fitness of that candidate

Step 3: Apply LHFGSO update rules on ALL weight/bias values:

        Firefly component:
        → Candidates with worse MSE are attracted toward better candidates
        → Weight values of worse candidates shift toward better ones

        HGSO component:
        → Henry's coefficient controls exploration-exploitation over iterations
        → Solubility dynamics guide how far each weight value moves
        → Temperature decay: early iterations = big moves, later = refinement

        LSO component (NEW in Paper 2):
        → Light-spectrum perturbation: pick 3 random candidates (s1, s2, s3)
        → Differential update: q_new = (q_s1 + GT5 × (q_s2 − q_s3)) × t
        → This adds diversification to prevent premature convergence

        Escape mechanism:
        → If candidates stagnate, worst are repositioned randomly

Step 4: Repeat Steps 2-3 for max iterations (up to 30 epochs)

Step 5: Best candidate's complete {N, m} across all layers → final DMN

RESULT: Final output I_o (cancer / no-cancer classification)
```

### Why optimizing ALL layers of DMN is feasible (but wasn't for DRN)

```
Paper 1 — DRN:
┌────────────────────────────────────┐
│ Conv layers: ~millions of params   │ ← Too many for population search
│ Residual blocks: ~thousands        │ ← Too many for population search
│ FC layer: ~hundreds to thousands   │ ← HFGSO optimizes ONLY this
└────────────────────────────────────┘
Total candidate size: SMALL (just FC)

Paper 2 — DMN:
┌────────────────────────────────────┐
│ Maxout layer 1: weights + biases   │ ← LHFGSO optimizes this
│ Maxout layer 2: weights + biases   │ ← LHFGSO optimizes this
│ ...                                │
│ Maxout layer h: weights + biases   │ ← LHFGSO optimizes this
└────────────────────────────────────┘
Total candidate size: 230,931 params (ALL of DMN)
Each candidate in the population encodes the ENTIRE network.

Why this works:
- DMN has NO conv layers (no spatial feature extraction inside DMN)
- Feature extraction is EXTERNAL (LBP, SLBT, stats done before DMN)
- 230K params is small — a population of 30 candidates means
  30 × 230K = ~7M values to maintain, feasible on 8GB RAM
```

---

## 3. How Maxout Layers Differ from Conv+ReLU (Why DMN is Fully Optimizable)

Understanding why LHFGSO can optimize all DMN layers requires understanding what maxout is:

```
Standard Conv+ReLU (DRN):
┌───────────────────────────────┐
│ Input image (2D spatial data) │
│        ↓                      │
│ Convolution (kernel slides     │   Conv layers: many params
│ across spatial dimensions)     │   (kernel size × channels × filters)
│        ↓                      │
│ ReLU: max(0, x)               │   Fixed — nothing to optimize
│        ↓                      │
│ ... many such layers ...       │
│        ↓                      │
│ FC: P × r + U                  │   HFGSO optimizes {P, U}
└───────────────────────────────┘

Maxout (DMN):
┌───────────────────────────────┐
│ Input: feature vector          │
│ (LBP, SLBT, stats — NOT       │
│  a 2D image)                   │
│        ↓                      │
│ Maxout: max(x·W1+b1,           │   N and m are trainable
│              x·W2+b2,           │   LHFGSO optimizes ALL of them
│              ...,               │
│              x·Wa+ba)           │
│        ↓                      │
│ ... h such layers ...           │
│        ↓                      │
│ Output: cancer / no-cancer      │
└───────────────────────────────┘

Key differences:
- DMN input is a 1D feature vector, not a 2D image → no need for convolutions
- Each maxout unit has 'a' sets of weights (a ≥ 2), making the activation trainable
- Entire network is just stacked maxout layers → ALL params are classification params
- No spatial hierarchy to learn → no need for backprop-friendly conv structure
```

---

## Summary: Where LHFGSO Works in Paper 2

```
Full Pipeline with LHFGSO locations marked:

MRI Input
    ↓
Adaptive Median Filter  ← No optimization (classical signal processing)
    ↓
ROI Extraction          ← No optimization (geometric operation)
    ↓
┌───────────────────────────────────────┐
│ SegNet Segmentation                   │
│                                       │
│  Encoder (13 Conv layers) ← Backprop  │
│  Decoder (13 layers)      ← Backprop  │
│  Classification layer     ← ★ LHFGSO ★│
│                                       │
│  Fitness = Dice+CE loss (Eq. 4)       │
└───────────────────────────────────────┘
    ↓
Data Augmentation       ← No optimization (rotation, crop, flip, erase)
    ↓
Feature Extraction      ← No optimization (LBP, SLBT, stats are formulas)
    ↓
┌───────────────────────────────────────┐
│ DMN Cancer Detection                  │
│                                       │
│  Maxout Layer 1 {N, m} ← ★ LHFGSO ★ │
│  Maxout Layer 2 {N, m} ← ★ LHFGSO ★ │
│  ...                                  │
│  Maxout Layer h {N, m} ← ★ LHFGSO ★ │
│                                       │
│  ALL 230,931 params optimized         │
│                                       │
│  Fitness = MSE (Eq. 36)              │
└───────────────────────────────────────┘
    ↓
Output: Cancer / No Cancer
```

---

## Paper 1 vs Paper 2: Optimization Scope Comparison

```
Paper 1 (HFGSO-DRN):                 Paper 2 (LHFGSO-DMN):

SegNet:                               SegNet:
  13 Conv encoder ← backprop            13 Conv encoder ← backprop
  13 Decoder      ← backprop            13 Decoder      ← backprop
  Classifier      ← ★ HFGSO             Classifier      ← ★ LHFGSO

DRN:                                  DMN:
  Conv layers     ← backprop            Maxout layer 1  ← ★ LHFGSO
  Pooling         ← no params           Maxout layer 2  ← ★ LHFGSO
  ReLU            ← no params           ...
  Batch Norm      ← backprop            Maxout layer h  ← ★ LHFGSO
  Residual blocks ← backprop
  FC layer {P,U}  ← ★ HFGSO           (ALL layers optimized by LHFGSO)

Optimizer scope: NARROW               Optimizer scope: FULL NETWORK
(just FC layer of DRN)                (all of DMN)

Why the difference?                   Why this is better?
DRN has conv layers with              DMN is a pure classifier with no
millions of params — too many         conv layers. 230K total params
for population search.                is feasible for population search.
                                      The optimizer sees the ENTIRE
                                      decision-making process, not just
                                      the final layer.
```

### This is the key insight for why Paper 2 performs better

In Paper 1, HFGSO could only optimize the final FC layer of DRN. The conv/residual layers were trained by standard backprop, meaning local minima in those layers couldn't be addressed.

In Paper 2, the architecture was redesigned so that:
1. **Feature extraction moved OUTSIDE the classifier** (LBP, SLBT, stats)
2. **The classifier (DMN) became a pure maxout network** with no conv layers
3. **LHFGSO could optimize the ENTIRE classifier** — every weight, every bias, every layer

This means the optimizer has **full control over the classification decision**, not just the final layer. Combined with the additional LSO search mechanism, this explains the improvement from 92.63% to 94.63% accuracy and from 91.30% to 95.72% specificity.
