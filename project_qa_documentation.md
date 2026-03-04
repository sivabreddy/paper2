# Conference-Ready Explanation and Q&A

## Paper Covered

**Title:** Hybrid Optimization Enabled Deep-Learning for Prostate Cancer Detection  
**Venue/Year:** Sensing and Imaging, 2024, Volume 25, Article 52  
**DOI:** https://doi.org/10.1007/s11220-024-00495-0  
**Authors:** Siva Kumar Reddy, Kalaivani Kathirvelu  
**Affiliation:** Department of Computer Science and Engineering, Vels Institute of Science Technology and Advanced Studies (VISTAS), Chennai, Tamil Nadu, India  
**Received:** 11 December 2023 | **Revised:** 13 June 2024 | **Accepted:** 16 July 2024 | **Published:** 17 August 2024

---

## 1) Clear, Complete Explanation of the Paper

### 1.1 Why this paper exists

The paper addresses **prostate cancer (PCa) detection from MRI images**.  
The authors motivate the work by saying:

- Prostate cancer is the second most frequent malignancy in men and the fifth most common cause of cancer-related mortality globally.
- In 2022, PCa was predicted to cause 268,490 new cases and 34,500 deaths in the United States alone.
- Current diagnosis methods like biopsy are invasive and can result in bleeding or infection.
- MRI is non-invasive and widely used, but manual reading is difficult and time-consuming.
- Existing AI pipelines can still miss cases, overfit, or require high computation.
- Elevated PSA (Prostate-Specific Antigen) screening puts significant numbers of men at risk of overdiagnosis.
- Distinguishing between clinically significant PCa and indolent PCa remains difficult.

So the core goal is: **build a better AI pipeline that improves detection performance while reducing computational cost and error tolerance issues**.

---

### 1.2 Main idea in one line

The paper proposes a pipeline called **LHFGSO-DMN**, where:

- MRI images are preprocessed and segmented,
- features are extracted (both structural and statistical),
- cancer is classified using a Deep Maxout Network (DMN),
- and both segmentation/classification training are guided by a new hybrid optimizer called **LHFGSO**.

---

### 1.3 End-to-end pipeline (step by step)

The full pipeline is:

1. **Input MRI acquisition** from a prostate MRI dataset.
2. **Preprocessing**:
   - Adaptive median filtering (remove impulse noise while preserving sharpness and fine details)
   - ROI extraction (focus on relevant prostate region, defined by pixel intensity rate)
3. **Segmentation** using a **multi-objective SegNet** (encoder-decoder with 13 Conv layers each).
4. **Optimization-based training of SegNet** using **LHFGSO**.
5. **Data augmentation** (4 techniques):
   - Rotation (−50° to 30°)
   - Cropping (30% on one side)
   - Flipping (vertical, horizontal, or both axes)
   - Random erasing (randomly erase one square region)
6. **Feature extraction**:
   - Structural texture features: LBP (Local Binary Pattern), SLBT (Shape-and-Local-Binary-Texture)
   - Statistical features: mean, variance, kurtosis, skewness, entropy
7. **Classification** using **DMN** (Deep Maxout Network), trained with LHFGSO.
8. **Evaluation** with accuracy, sensitivity, specificity.

---

### 1.4 Segmentation contribution: multi-objective SegNet

The segmentation model is not plain SegNet; it is modified with a combined objective:

- **Pixel-wise cross-entropy:** D(a,b) = Σ_d a_d log(b_d)
- **Dice coefficient:** B(a,b) = 2Σ(b_d × b_d) / (Σb_d + Σa_d)
- **Combined (fused) loss:** E(a,b) = (1−σ)(Σ a_d log(b_d)) − σ log((2Σ(b_d × a_d) + F) / (Σb_d + Σa_d + F))

Where σ = 0.75 and F (smooth) = 1e−15 to avoid numerical instability.

This combined objective is intended to improve segmentation of cancer regions, especially where class imbalance exists. The σ=0.75 weighting gives 75% emphasis to the Dice component, prioritizing spatial overlap quality.

---

### 1.5 What is LHFGSO and why they built it

The authors propose **Light Henry Firefly Gas Solubility Optimization (LHFGSO)**, a hybrid of:

- **HFGSO** + **LSO**
- where **HFGSO** itself combines:
  - HGSO (Henry Gas Solubility Optimization)
  - Firefly Algorithm (FA)

So this is a **multi-level hybrid metaheuristic**: FA + HGSO → HFGSO, then HFGSO + LSO → LHFGSO.

Intuition of each part:

- **Firefly idea:** Move candidate solutions toward brighter (better) solutions. Attractiveness is proportional to brightness, and all fireflies are treated as unisex. However, FA alone struggles with multi-objective optimization.
- **HGSO idea:** Based on Henry's law from chemistry — the amount of gas dissolved in a liquid at constant temperature is proportional to partial pressure. Uses gas-solubility-inspired dynamics for exploration/exploitation.
- **LSO (Light Spectrum Optimizer) idea:** Inspired by the rainbow effect when sunlight passes through water droplets. Models the dispersion of light at various angles to achieve a good balance between exploitation and exploration in continuous optimization problems.

Why combine all three?

- FA alone struggles in multi-objective settings.
- HFGSO (FA+HGSO) improves search but can still benefit from additional diversification.
- LSO adds another layer of exploration-exploitation balance via light-spectrum-inspired search.
- The optimizer is used to tune training behavior (weights/parameters) for both SegNet and DMN.

### LHFGSO algorithm steps

1. **Initialization** — Initialize total gases and their locations: G_i(f+1) = G_min + g × (G_max − G_min). Initialize Henry's constant H_i(f) = h1 × rand(0,1), partial pressure I_{i,c} = h2 × rand(0,1), constants h1=5E-02, h2=100, h3=1E-02.
2. **Clustering** — Group population agents into equal-sized clusters with same gas types, each associated with Henry's constant.
3. **Fitness evaluation** — Compute fitness using the combined loss function (Eq. 4).
4. **Henry's coefficient update** — H_i(f+1) = H_i(f) × exp(−M_d × (1/N(f) − 1/N_ψ)); N(f) = exp(−f/j), N_ψ = 298.15.
5. **Solubility update** — O_{i,c}(f) = P × H_i(f+1) × I_{i,c}(f).
6. **Position update** — Uses the hybridized LHFGSO equation integrating HGSO dynamics, FA movement terms, and **LSO light-spectrum-inspired updates** using random vectors and differential evolution-style position perturbation.
7. **Escape local optimum** — Rank and reposition worst search agents: Q_k = Q × rand(l2−l1) + l1; l1=0.1, l2=0.2.
8. **Update search agent positions** — R_{i,c} = R_min + ν × (R_max − R_min).
9. **Feasibility evaluation** — Compute fitness for each agent; lowest value is optimal.
10. **Termination** — Repeat until maximum iterations reached.

---

### 1.6 Why DMN (Deep Maxout Network)

DMN uses **maxout units**, which can approximate complex nonlinear activations and may avoid saturation issues seen in some activations. Key characteristics:

- Multi-layer maxout networks (MMN) have trainable activation functions
- Activation: p^b_{i,o,k} = max_{k∈[1,a]} f^{n-1}_{o,k} · W_{ok} + m_{ok}
- When parameter a ≥ 2, DMN can accurately approximate absolute value rectifier and rectified linear rectifier
- Increasing parameter a allows approximation of arbitrarily complex nonlinear activation functions
- DMN partially prevents hidden units from entering dormant states

The authors use DMN as the final cancer detector after feature extraction, with LHFGSO-based training to improve optimization quality.

---

### 1.7 Feature extraction details

The paper uses a **hybrid feature extraction** approach combining structural and statistical features:

**Structural features:**
- **LBP (Local Binary Pattern):** A global texture-enabled feature descriptor that labels a pixel's value by thresholding the 8 neighbors around each pixel and interpreting the result as a binary number: LBP(β_m, θ_m) = Σ_{e=0}^{7} D(χ_e − χ_m) × 2^e, where D(q) = 1 if q≥0, else 0
- **SLBT (Shape-and-Local-Binary-Texture):** Projects shape-free patch LBP feature histograms into eigenface space, capturing both global shape variation and local texture variation

**Statistical features (computed from structural features):**
- **Mean (σ):** Concentration of distribution's data: σ = Σ_{o=0}^{S-1} o × D(o)
- **Variance (μ2):** Grey level deviation from mean: μ2 = Σ_{o=0}^{S-1} (o−σ)² × D(o)
- **Kurtosis (τ1):** Fourth normalized moment, estimates distribution balance: τ1 = σ^{-4} Σ_{o=0}^{S-1} (o−σ)⁴ × D(o)
- **Skewness (τ2):** Degree of asymmetry around mean: τ2 = σ^{-3} Σ_{o=0}^{S-1} (o−σ)³ × D(o)
- **Entropy (B):** Measures system disorder: B = Σ_{o=0}^{S-1} D(o) × log2[D(o)]

---

### 1.8 Dataset and setup reported

The paper reports:

- **Dataset:** Prostate MRI dataset (https://prostate-mri-database.com/)
- Database contains records of **230 patients** with examination type, description, and date
- Experiments use **20 patients' records**
- **Model parameters:** Total: 230,931 (902.07 KB); Trainable: 230,611 (900.82 KB); Non-trainable: 320 (1.25 KB)
- Implementation in **Python**

### Hyperparameters table

| Parameter | DCNN | Panoptic | Focal-Net | ResNet | HFGSO-DRN | **Proposed LHFGSO-DMN** |
|-----------|------|----------|-----------|--------|-----------|------------------------|
| Epochs | 20 | 20 | 20 | 20 | 25 | **30** |
| Batch size | 64 | 32 | 64 | 32 | 32 | **32** |
| Learning rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | **0.001** |
| Optimizer | SGD | Adam | Adam | RMSprop | HFGSO | **LHFGSO** |

---

### 1.9 Metrics used

- **Accuracy:** Ω = (I_q + I_k) / (I_q + I_k + Q_q + Q_k) — overall correctness
- **Specificity:** ℘ = I_q / (I_q + Q_k) — true positive detection rate
- **Sensitivity:** λ = I_k / (I_k + Q_q) — true negative identification rate

Where I_q = True Positive, I_k = True Negative, Q_q = False Positive, Q_k = False Negative.

In clinical screening settings, sensitivity is usually critical (missing cancer is costly), while specificity matters to reduce false alarms and unnecessary procedures.

---

### 1.10 Reported results (core numbers)

**At 90% training data, proposed LHFGSO-DMN reports:**

- **Accuracy: 94.63%**
- **Sensitivity: 93.46%**
- **Specificity: 95.72%**

**Under K-fold analysis (K=9), proposed method reports:**

- **Accuracy: 94.06%**
- **Sensitivity: 93.13%**
- **Specificity: 94.99%**

### Iteration-wise progression (at 90% training data)
| Iteration | Accuracy (%) | Sensitivity (%) | Specificity (%) |
|-----------|-------------|----------------|-----------------|
| 5         | 88.43       | 86.43          | 89.42           |
| 10        | 89.42       | 87.96          | 90.53           |
| 15        | 91.86       | 90.58          | 92.56           |
| 20        | 92.78       | 91.67          | 93.56           |

### Comparative results (training data — 90%)

| Method | Accuracy (%) | Sensitivity (%) | Specificity (%) |
|--------|-------------|----------------|-----------------|
| DCNN | 76.05 | 75.65 | 78.01 |
| Panoptic model | 79.71 | 78.37 | 80.43 |
| Focal-Net | 87.93 | 86.48 | 88.95 |
| ResNet | 89.40 | 88.47 | 90.76 |
| HFGSO-based DRN | 92.63 | 91.57 | 93.88 |
| **Proposed LHFGSO-DMN** | **94.63** | **93.46** | **95.72** |

### Comparative results (K-Fold — K=7)

| Method | Accuracy (%) | Sensitivity (%) | Specificity (%) |
|--------|-------------|----------------|-----------------|
| DCNN | 76.42 | 75.89 | 77.54 |
| Panoptic model | 78.80 | 77.05 | 79.12 |
| Focal-Net | 85.50 | 84.51 | 87.06 |
| ResNet | 87.91 | 86.64 | 88.56 |
| HFGSO-based DRN | 91.92 | 91.05 | 92.86 |
| **Proposed LHFGSO-DMN** | **94.06** | **93.13** | **94.99** |

### Improvement margins (training data — 90%)
- Over DCNN: ~18.58% accuracy improvement
- Over Panoptic: ~14.92% accuracy improvement
- Over Focal-Net: ~6.70% accuracy improvement
- Over ResNet: ~5.23% accuracy improvement
- Over HFGSO-based DRN: ~2.00% accuracy improvement

The paper concludes that the proposed hybrid optimization + deep learning design outperforms listed baselines.

---

### 1.11 Convergence analysis

The paper provides a convergence curve (Fig. 9) comparing optimizer fitness values at iteration 25:

| Optimizer | Fitness Value |
|-----------|--------------|
| SGD | 0.0571 |
| Adam | 0.0458 |
| RMSprop | 0.0314 |
| HFGSO | 0.0205 |
| **Proposed LHFGSO** | **0.0252** |

This shows the metaheuristic optimizers (HFGSO and LHFGSO) achieve competitive fitness values compared to standard gradient-based optimizers.

---

### 1.12 What this paper contributes

Main claimed contributions:

1. A hybrid optimizer (**LHFGSO = HFGSO + LSO**) for model training.
2. Multi-objective SegNet using Dice + cross-entropy.
3. End-to-end pipeline from preprocessing to feature extraction to classification.
4. Explicit feature extraction (LBP, SLBT, statistical features) combined with deep learning.
5. Comparative experiments against multiple baselines including confusion matrix analysis.

---

### 1.13 Practical strengths

- Uses MRI (non-invasive modality).
- Combines segmentation quality objective (Dice + CE).
- Tries to reduce overfitting with augmentation (4 techniques) and optimization.
- Includes explicit feature extraction that can capture texture cues even with limited data.
- Reports strong values on all three core metrics.
- Provides hyperparameter table for reproducibility.
- Includes confusion matrix and convergence curve analysis.

---

### 1.14 Critical reading (important for conference presentation)

When presenting, also mention limitations transparently:

1. **Reproducibility detail is limited**  
   Full implementation-level details, exact splits, and training protocol are not fully explicit in a way that guarantees one-click reproduction. No public code repository is provided.

2. **Dataset reporting is somewhat inconsistent**  
   The manuscript mentions records for 230 patients but states 20 patients were used for experiments. This discrepancy should be clarified.

3. **External validation is missing**  
   No clear multi-center or external hospital validation is shown, which is crucial for clinical translation.

4. **Ablation depth is limited**  
   More ablations would help isolate how much each block (LHFGSO vs HFGSO, multi-objective loss, feature set, DMN vs other classifiers) contributes independently.

5. **Metaheuristic complexity vs practical deployment**  
   Hybrid optimizers can improve metrics but may increase computational complexity and training time. The convergence curve shows LHFGSO fitness (0.0252) is slightly higher than HFGSO (0.0205), which warrants discussion.

6. **No statistical significance testing**  
   Confidence intervals, p-values, or bootstrap tests are not reported.

---

### 1.15 Challenges addressed (from literature review)

The paper identifies specific challenges in existing PCa detection methods:

| Challenge | Source |
|-----------|--------|
| Stacking ensemble learning failed on large datasets | Wang et al. [12] |
| DCNN could not perform external validation | Alsadoon et al. [14] |
| Panoptic model couldn't incorporate distinct semantic branches into instance segmentation | Heidenreich et al. [3] |
| Focal-Net didn't permit precise registration of whole-mount slices with mp-MRI | Turkbey et al. [8] |
| CDBN-EHO required external validation and modest sample size | Mary et al. [18] |
| Extracting hidden features for higher performance remains challenging | General |

---

### 1.16 Final takeaway for audience

This paper is a **methodology-focused improvement paper**: it proposes a sophisticated multi-level hybrid optimization framework (LHFGSO) around segmentation + feature extraction + classification for prostate MRI detection, and reports strong performance numbers that outperform both classical and optimization-based baselines.  
For real-world adoption, the next major step is **strong external validation, reproducible benchmarking, and deeper ablation studies**.

---

## 2) Conference-Style Expected Questions and Detailed Answers

### Q1) What is the single biggest idea of this paper?
**Answer:** The central idea is to improve prostate cancer detection by combining deep learning with a multi-level hybrid optimizer (LHFGSO = HFGSO + LSO). Instead of relying only on standard gradient-based training, the paper introduces an optimizer inspired by physical/nature processes (gas solubility, firefly attraction, light spectrum) to guide training for both segmentation and classification.

### Q2) Why is MRI used here instead of biopsy directly?
**Answer:** Biopsy is invasive and can cause discomfort, bleeding, and infection. MRI is non-invasive and provides rich anatomical and tissue information. AI on MRI can support earlier and safer screening workflows, though biopsy remains the final diagnostic standard in many clinical settings. PSA screening also puts many men at risk of overdiagnosis, making MRI-based detection a complementary approach.

### Q3) What exactly does the pipeline do from start to finish?
**Answer:** It takes MRI images, cleans them using adaptive median filtering, extracts the prostate-focused ROI region, segments possible cancer regions using an optimized SegNet, augments data with four techniques (rotation, cropping, flipping, random erasing), extracts handcrafted texture/statistical features (LBP, SLBT, mean, variance, kurtosis, skewness, entropy), and then classifies with a Deep Maxout Network optimized by LHFGSO.

### Q4) Why combine Dice and cross-entropy losses?
**Answer:** Cross-entropy is strong for pixel-wise classification, but Dice directly optimizes overlap quality between predicted masks and true masks. In medical segmentation, overlap quality is crucial, especially with class imbalance (small lesion regions), so combining both often gives more robust segmentation. The weight σ=0.75 gives 75% emphasis to Dice.

### Q5) Why is segmentation done before classification?
**Answer:** Segmentation helps focus the model on lesion-relevant regions rather than irrelevant background. That can improve signal quality for downstream classification and reduce noise in extracted features. The segmented output is also augmented to increase data diversity.

### Q6) What is novel about LHFGSO compared with standard optimizers like Adam?
**Answer:** Adam is gradient-based and local in update behavior. LHFGSO is a multi-level hybrid metaheuristic intended to improve global search and avoid poor local minima. The novelty is in combining three optimization mechanisms (HGSO + Firefly + LSO) into one training strategy, where each adds a different search capability.

### Q7) Is this replacing backpropagation completely?
**Answer:** The paper positions LHFGSO as an optimizer for model training/tuning, but practical implementations often still rely on gradient computations in deep learning stacks. A key presentation point is: the method augments optimization behavior beyond simple off-the-shelf optimizers.

### Q8) Why use DMN (maxout network) instead of plain CNN or DRN?
**Answer:** Maxout units can represent richer nonlinear functions and can reduce activation saturation issues. Unlike ReLU which is a fixed function, maxout activation is **trainable** — when parameter a ≥ 2, it can approximate absolute value rectifier, rectified linear rectifier, and more complex functions. DMN also partially prevents hidden units from entering dormant states, which can happen with ReLU (dead neurons).

### Q9) Why include handcrafted features like LBP and SLBT in a deep-learning paper?
**Answer:** This paper uses a hybrid philosophy. Handcrafted features can capture texture cues explicitly and may help when data volume is limited (only 20 patients). LBP captures local texture patterns by comparing each pixel with its 8 neighbors, while SLBT combines shape and texture information by projecting LBP histograms into eigenface space. The authors combine these with statistical features and deep models to improve robustness.

### Q10) What are the strongest reported results?
**Answer:** At 90% training data: 94.63% accuracy, 93.46% sensitivity, and 95.72% specificity. Under K-fold (K=7): 94.06% accuracy, 93.13% sensitivity, and 94.99% specificity. These outperform all listed baselines.

### Q11) Which metric matters most clinically: accuracy, sensitivity, or specificity?
**Answer:** For cancer detection, sensitivity is often prioritized first because missed cancers are high-risk. Specificity is also important to reduce unnecessary follow-ups and biopsies. Accuracy alone can hide class imbalance, so sensitivity and specificity should always be shown together. This paper reports all three.

### Q12) Did the paper show external validation?
**Answer:** Not clearly at a strong multi-center level. That is a major future requirement for real clinical confidence.

### Q13) How robust is this method across scanners/hospitals?
**Answer:** The paper does not fully establish cross-site robustness. Domain shift (different scanners, acquisition protocols, patient populations) is a real risk. A solid next step is external cohort testing and harmonization experiments.

### Q14) Is there risk of overfitting despite good results?
**Answer:** Yes, always possible in medical imaging with limited data (20 patients). The paper uses 4-type augmentation and optimization to reduce overfitting, but independent external validation is still needed to confirm true generalization.

### Q15) Why does the paper compare with multiple baselines?
**Answer:** Multi-baseline comparison helps show that gains are not accidental. It positions the method against different families of models: DCNN, panoptic, focal, residual, and optimization-based baselines (HFGSO-DRN). Notably, HFGSO-DRN is the author's own previous work (Paper 1), and LHFGSO-DMN improves upon it.

### Q16) What are the computational costs of this approach?
**Answer:** Hybrid metaheuristics add training overhead. The proposed method uses 30 epochs vs 20-25 for baselines, suggesting more training time. The model has ~230K parameters (small by modern standards). In practice, training time, memory, and energy cost should be benchmarked against Adam/RMSprop baselines before deployment.

### Q17) Could this run in real-time in hospitals?
**Answer:** Inference should be feasible since the model is small (~230K parameters, 902KB). Training complexity is less relevant once the model is frozen. For deployment, clinically acceptable latency and PACS workflow integration are needed.

### Q18) How does this compare to end-to-end transformers or modern foundation models?
**Answer:** This paper is not based on transformers/foundation models; it follows a hybrid handcrafted + deep + metaheuristic route. Modern comparisons with transformer-based medical segmentation/classification (like ViT, Swin-UNETR, SAM) would strengthen current relevance.

### Q19) Did the paper include explainability?
**Answer:** Not deeply. For clinical acceptance, explainability tools like saliency maps, Grad-CAM, lesion overlays, and uncertainty estimates should be added in future work.

### Q20) Is this model intended to replace radiologists?
**Answer:** No. It is best positioned as a decision-support tool to assist radiologists, improve consistency, and prioritize suspicious cases.

### Q21) What is the role of data augmentation in this work?
**Answer:** Augmentation creates plausible input variations using four techniques: (1) rotation (−50° to 30°), (2) cropping (30% on one side), (3) flipping (vertical/horizontal/both), (4) random erasing (randomly erase square regions). This increases effective training diversity, reduces overfitting, and is critical given the small 20-patient dataset.

### Q22) Could class imbalance affect results?
**Answer:** Yes. Prostate lesion datasets are often imbalanced. The Dice term in the loss function helps handle segmentation-level imbalance, but full imbalance handling should also include careful split strategy and class-wise reporting.

### Q23) What are the main risks before clinical adoption?
**Answer:** External validity, reproducibility, bias across populations, and workflow integration are main risks. Prospective evaluation is needed before using this in decision-critical pathways.

### Q24) What ablations would you ask the authors to add?
**Answer:** At least:
- LHFGSO vs HFGSO (isolate LSO contribution)
- With/without multi-objective loss (Dice+CE vs CE alone)
- DMN vs DRN vs simple classifier on same features
- Handcrafted features vs end-to-end deep features only
- Different augmentation subset combinations
- Statistical features only vs LBP/SLBT only vs combined

### Q25) Are results statistically significant?
**Answer:** The paper reports comparative percentages but rigorous statistical testing details (p-values, confidence intervals, bootstrap tests) are limited. Adding significance tests would improve scientific strength.

### Q26) Is this reproducible from the paper alone?
**Answer:** Partially, but not fully guaranteed. The paper provides the hyperparameter table and model size, which helps. However, reproducibility would improve with public code, exact preprocessing scripts, fixed random seeds, and complete train/val/test protocol details.

### Q27) How would you improve this work in a follow-up paper?
**Answer:** Add external datasets, stronger ablations, calibration analysis, explainability (Grad-CAM), uncertainty estimation, and prospective reader-study evaluation with radiologists. Also compare against modern transformer-based architectures.

### Q28) Why is specificity high in this paper?
**Answer:** The combined segmentation objective, feature extraction, and optimizer-guided training may produce cleaner lesion localization and better decision boundaries, reducing false positives. The 95.72% specificity is the highest among all three metrics, suggesting good discrimination of non-cancer cases. External testing is needed to confirm consistency.

### Q29) What is your balanced one-minute verdict as a presenter?
**Answer:** This is a strong engineering-style contribution that combines multiple optimization ideas with deep learning and handcrafted features for better reported MRI prostate cancer detection metrics. It extends the prior HFGSO-DRN work by adding LSO, DMN, feature extraction, and more augmentation. The method is promising, but clinical translation requires stronger reproducibility and external validation evidence.

### Q30) If participants ask, "Can we trust this model clinically today?" what should we answer?
**Answer:** We should say: "It is promising for research and decision support, but not yet sufficient as a stand-alone clinical decision tool. More external and prospective validation is needed before routine clinical use."

---

## Additional Conference Questions

### Q31) What is the difference between this paper (LHFGSO-DMN) and your earlier paper (HFGSO-DRN)?
**Answer:** This paper extends the earlier work (published in Multimedia Tools and Applications, 2024) in several ways:

| Aspect | Paper 1 (HFGSO-DRN) | Paper 2 (LHFGSO-DMN) |
|--------|---------------------|----------------------|
| Optimizer | HFGSO (HGSO + FA) | LHFGSO (HFGSO + LSO) |
| Classifier | Deep Residual Network (DRN) | Deep Maxout Network (DMN) |
| Preprocessing | T2FCS filter | Adaptive median filter |
| Feature extraction | None (end-to-end) | LBP, SLBT, statistical features |
| Augmentation | 2 techniques (rotation, cropping) | 4 techniques (+flipping, random erasing) |
| Accuracy | 92.63% | 94.63% (+2.00%) |
| Sensitivity | 93.67% | 93.46% (−0.21%) |
| Specificity | 91.30% | 95.72% (+4.42%) |
| Epochs | Not specified | 30 |

The biggest improvement is in specificity (+4.42%), while sensitivity is slightly lower. Overall accuracy improved by 2%.

### Q32) What is LSO (Light Spectrum Optimizer) and what does it add?
**Answer:** LSO is inspired by the **rainbow effect** — when sunlight passes through a water droplet, it disperses into a spectrum of colors at various angles. This natural phenomenon is abstracted into an optimization algorithm that:
- Models light dispersion as a search mechanism
- Uses the spectrum spread to balance exploration (broad search via different wavelengths/angles) and exploitation (converging on promising regions)
- Adds differential evolution-style position perturbation to the search
In the context of LHFGSO, LSO provides an additional mechanism to escape local optima beyond what HFGSO alone offers.

### Q33) Why switch from T2FCS (Paper 1) to adaptive median filter (Paper 2)?
**Answer:** The adaptive median filter is specifically designed to preserve sharpness while removing mixed impulses with high probability of occurrence. It works better with high-density collaborative impulsive and non-impulsive noise while preserving fine details. T2FCS uses Type-2 fuzzy + Cuckoo Search which adds optimization complexity. The switch to adaptive median filtering simplifies the preprocessing while maintaining noise removal quality, allowing the optimization effort to focus on the main pipeline components.

### Q34) What are the specific statistical features and why are they useful?
**Answer:**
- **Mean:** Shows the global average intensity — indicates overall tissue brightness
- **Variance:** Measures grey level deviation from mean — captures texture roughness
- **Kurtosis:** Fourth moment — measures how peaked or flat the distribution is compared to normal. Cancerous tissue often has different kurtosis than healthy tissue
- **Skewness:** Third moment — measures asymmetry. Asymmetric intensity distributions can indicate abnormal tissue
- **Entropy:** Measures disorder/randomness — higher entropy indicates more heterogeneous tissue, which can correlate with cancer

These features complement the deep learning features by providing explicit, interpretable texture and distribution information.

### Q35) What is a maxout unit mathematically?
**Answer:** A maxout unit takes the maximum over k linear functions: h(x) = max_{k∈[1,a]} (x · W_k + b_k). Unlike ReLU which is fixed as max(0, x), the maxout activation is **trainable** — the network learns which linear piece to use. When a ≥ 2, it can represent ReLU, absolute value, and any piecewise linear function. The trade-off is increased parameters (a weight matrices per layer instead of 1), but with better expressiveness.

### Q36) Why does the proposed model use 30 epochs while baselines use 20-25?
**Answer:** The proposed LHFGSO-DMN uses 30 epochs, which is higher than baselines (20 for DCNN/Panoptic/Focal-Net/ResNet, 25 for HFGSO-DRN). This could be because: (1) the metaheuristic optimizer may need more iterations to converge, (2) the DMN architecture with maxout units may benefit from longer training, (3) the feature extraction pipeline provides richer inputs that need more epochs to fully exploit. However, this means training-time comparisons should account for this difference.

### Q37) What does the confusion matrix tell us about the model?
**Answer:** The paper includes confusion matrices (Fig. 8) for all compared methods. These show the distribution of True Positives, True Negatives, False Positives, and False Negatives for each model. The proposed LHFGSO-DMN should show the highest TP and TN counts with lowest FP and FN, consistent with its superior accuracy/sensitivity/specificity values. Confusion matrices provide more detail than aggregate metrics alone.

### Q38) How does the convergence curve compare optimizers?
**Answer:** At iteration 25, the fitness values are: SGD=0.0571, Adam=0.0458, RMSprop=0.0314, HFGSO=0.0205, LHFGSO=0.0252. Interestingly, HFGSO achieves a slightly lower fitness than LHFGSO (0.0205 vs 0.0252), yet LHFGSO-DMN achieves better classification metrics. This suggests that lower training fitness doesn't always mean better generalization — LHFGSO may find solutions that generalize better despite slightly higher training loss.

### Q39) What is the model size and is it practical for deployment?
**Answer:** The model has 230,931 total parameters (902.07 KB), with 230,611 trainable and 320 non-trainable. This is extremely small by modern standards (GPT models have billions of parameters, even ResNet-50 has ~25M). This makes the model very practical for deployment: fast inference, low memory, and suitable for edge devices or hospital workstations.

### Q40) Why not use more modern architectures like U-Net, nnU-Net, or transformer-based models?
**Answer:** The paper focuses on the **optimization contribution** (LHFGSO) rather than architecture innovation. SegNet and DMN serve as the base architectures to demonstrate that metaheuristic optimization can improve training. Future work could apply the same optimization framework to modern architectures like U-Net++, nnU-Net, TransUNet, or Swin-UNETR to see if the benefits transfer.

### Q41) How do LBP features work in practice for cancer detection?
**Answer:** LBP compares each pixel with its 8 neighbors: if a neighbor's intensity is greater than or equal to the center pixel, it gets a 1; otherwise 0. These 8 binary values form an 8-bit number (0-255). The histogram of these LBP values across the image captures local texture patterns. Cancerous tissue often has different texture characteristics (more heterogeneous, irregular boundaries) than healthy tissue, making LBP a useful discriminative feature even without deep learning.

### Q42) What data augmentation techniques are new compared to Paper 1?
**Answer:** Paper 1 used only rotation (−50° to 30°) and cropping (30%). This paper adds:
- **Flipping:** Rotates the image around vertical, horizontal, or both axes — effectively doubling/tripling the dataset without geometric distortion
- **Random erasing:** Randomly removes a square region from the image — forces the model to learn from partial information and improves robustness to occlusion
These two additional techniques increase data diversity significantly, which helps with the limited 20-patient dataset.

### Q43) What are the implications of using only 20 patients?
**Answer:** Using only 20 patients out of 230 available raises several concerns:
- **Statistical power** is limited — results may not generalize
- **Patient-level variation** may not be fully captured
- **Overfitting risk** is high, even with augmentation
- **Confidence intervals** would likely be wide
- The paper doesn't explain why only 20 were selected
For clinical credibility, future work must use the full 230-patient dataset or larger external cohorts.

### Q44) How does this paper handle the preprocessing order (filtering then ROI vs ROI then filtering)?
**Answer:** This paper uses **adaptive median filtering first, then ROI extraction** — the opposite order from Paper 1 (which did ROI extraction then T2FCS filtering). The rationale is that noise should be removed from the full image first before extracting the region of interest, ensuring cleaner ROI boundaries. Both approaches are valid, but the order can affect downstream segmentation quality.

### Q45) What would a prospective clinical study look like for this method?
**Answer:** A proper prospective study would involve: (1) Collecting new MRI scans from multiple hospitals, (2) Running the LHFGSO-DMN pipeline blind to clinical diagnosis, (3) Comparing AI predictions against biopsy-confirmed ground truth, (4) Measuring sensitivity/specificity with confidence intervals, (5) Conducting a reader study where radiologists use the tool vs work without it, (6) Assessing impact on clinical outcomes (earlier detection, fewer unnecessary biopsies), (7) Evaluating across patient subgroups for fairness.

---

## 3) Suggested Closing Statement for Your Presentation

"This paper demonstrates that carefully combining segmentation objectives, handcrafted feature design, deep maxout classification, and a multi-level hybrid metaheuristic optimizer (LHFGSO) can produce strong prostate MRI detection metrics. It builds upon and improves the earlier HFGSO-DRN work by adding the Light Spectrum Optimizer, Deep Maxout Network, explicit feature extraction, and richer augmentation. The next leap is not only better accuracy, but stronger trust: reproducibility, external validation, and clinically interpretable deployment."
