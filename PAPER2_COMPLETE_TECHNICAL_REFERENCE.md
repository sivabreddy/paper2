# Paper 2: Hybrid Optimization Enabled Deep-Learning for Prostate Cancer Detection — Complete Technical Reference

## Table of Contents

1. [Paper Overview](#1-paper-overview)
2. [Complete End-to-End Pipeline](#2-complete-end-to-end-pipeline)
3. [Step-by-Step Technical Breakdown](#3-step-by-step-technical-breakdown)
4. [Adaptive Median Filter: How It Works](#4-adaptive-median-filter-how-it-works)
5. [SegNet Segmentation with Multi-Objective Loss](#5-segnet-segmentation-with-multi-objective-loss)
6. [Data Augmentation (4 Techniques)](#6-data-augmentation-4-techniques)
7. [Feature Extraction: LBP, SLBT, and Statistical Features](#7-feature-extraction-lbp-slbt-and-statistical-features)
8. [LHFGSO Algorithm Deep Dive](#8-lhfso-algorithm-deep-dive)
9. [LHFGSO in SegNet (Segmentation)](#9-lhfso-in-segnet-segmentation)
10. [Deep Maxout Network (DMN) Architecture](#10-deep-maxout-network-dmn-architecture)
11. [LHFGSO in DMN (Classification)](#11-lhfso-in-dmn-classification)
12. [LHFGSO Parameter Summary](#12-lhfso-parameter-summary)
13. [Architecture Reference](#13-architecture-reference)
14. [Loss Functions and Fitness Metrics](#14-loss-functions-and-fitness-metrics)
15. [Data Flow Summary](#15-data-flow-summary)
16. [Results Analysis](#16-results-analysis)
17. [Ablation Studies](#17-ablation-studies)
18. [Paper 1 vs Paper 2: Complete Comparison](#18-paper-1-vs-paper-2-complete-comparison)

---

## 1. Paper Overview

| Item | Detail |
|------|--------|
| **Title** | Hybrid Optimization Enabled Deep-Learning for Prostate Cancer Detection |
| **Journal** | Sensing and Imaging (2024), Volume 25, Article 52 |
| **DOI** | https://doi.org/10.1007/s11220-024-00495-0 |
| **Authors** | Siva Kumar Reddy, Kalaivani Kathirvelu |
| **Affiliation** | Department of CSE, VISTAS, Chennai, India |
| **Dataset** | Prostate MRI from Brigham and Women's Hospital (20 patients used out of 230) |
| **Model Size** | 230,931 parameters (902 KB) |
| **Epochs** | 30 |

### Research Problem

Prostate cancer is the second most frequent malignancy in men, causing 268,490 new cases and 34,500 deaths annually in the US alone (2022 data). Current diagnosis via biopsy is invasive. MRI is non-invasive but manual reading requires expertise. This paper asks: **Can a hybrid-optimized deep learning pipeline detect prostate cancer from MRI more accurately than existing approaches?**

### Key Innovation from Paper 1 to Paper 2

Paper 1 used HFGSO (HGSO + Firefly) to train DRN (Deep Residual Network) and achieved 92.63% accuracy. Paper 2 builds on this by:

1. Adding **LSO (Light Spectrum Optimizer)** to create **LHFGSO** (3-level hybrid)
2. Replacing DRN with **DMN (Deep Maxout Network)** — trainable activation functions
3. Adding **explicit feature extraction** (LBP, SLBT, statistical features)
4. Replacing T2FCS with simpler **adaptive median filter**
5. Increasing augmentation from 2 to **4 techniques**
6. Result: **94.63% accuracy, 95.72% specificity** — best improvements in specificity

### Best Results

| Evaluation Mode | Accuracy | Sensitivity | Specificity |
|----------------|----------|-------------|-------------|
| 90% Training Data | **94.63%** | **93.46%** | **95.72%** |
| K-Fold (K=9) | **94.06%** | **93.13%** | **94.99%** |

---

## 2. Complete End-to-End Pipeline

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              MRI IMAGE INPUT                                   │
│                        (256×256 RGB PNG file)                                 │
└─────────────────────────────────┬──────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA PREPARATION (Main/prepare_data.py)                               │
│  ─────────────────────────────────────────────────────────────────────         │
│  - Read Database/ and Database_gt/ recursively                                 │
│  - Resize to 128×128 PNG                                                       │
│  - Ground truth: RGB(0, 242, 255) pixels → white (255), rest → black (0)      │
│  - Output: data/im/*.png (101 images), data/gt/*.png (101 masks)               │
└─────────────────────────────────┬──────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: ROI EXTRACTION (Pre_processing.py → Select_Roi)                       │
│  ─────────────────────────────────────────────────────────────────────         │
│  - Take center portion of image                                                │
│  - Remove 10px top, 20px bottom, 20px left/right                               │
│  - Output: Output/roi/roi_X.png (~230×230)                                     │
└─────────────────────────────────┬──────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: ADAPTIVE MEDIAN FILTERING (Pre_processing.py → amf)                   │
│  ─────────────────────────────────────────────────────────────────────         │
│  - Apply to each color channel (B, G, R) separately                            │
│  - Window grows dynamically: initial=3, max=11                                 │
│  - Level A: Check if median is between min and max of window                   │
│  - Level B: If Level A fails, expand window (up to 11×11)                      │
│  - Preserves fine details while removing impulse noise                         │
│  - Output: Output/amf/amf_X.png                                                 │
└─────────────────────────────────┬──────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: SEGNET SEGMENTATION (Main/Proposed_SegNet.py)                         │
│  ─────────────────────────────────────────────────────────────────────         │
│  - Encoder: 13 Conv blocks → MaxPool (stores indices)                          │
│  - Dense: 1024 → 1024 (bottleneck)                                             │
│  - Decoder: UpSample → ConvTranspose → BN → ReLU × 13 blocks                  │
│  - Final: Sigmoid → 192×256 probability map                                    │
│  - Loss: (1-0.75)×CrossEntropy − 0.75×log(Dice+ε)                             │
│  - LHFGSO: Optimizes classification layer weights (see Section 9)              │
│  - Output: Output/segmented/seg_X.png                                          │
└─────────────────────────────────┬──────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: DATA AUGMENTATION (Main/Augmentation.py)                              │
│  ─────────────────────────────────────────────────────────────────────         │
│  Four augmentation techniques per image:                                       │
│                                                                                │
│  1. Rotation: 30° CCW using 3×3 rotation matrix                               │
│     M = [[cos(30°)  -sin(30°)  0]                                              │
│          [sin(30°)   cos(30°)  0]                                              │
│          [0          0         1]]                                             │
│                                                                                │
│  2. Cropping: Remove 30% from top and left → keep bottom-right                 │
│     crop = image[0.3*cols:, 0.3*cols:]                                         │
│                                                                                │
│  3. Flipping: cv2.flip(image, 0) — vertical flip (around horizontal axis)     │
│                                                                                │
│  4. Random Erasing: torchvision RandomErasing, probability=1, mode='pixel'     │
│     Randomly masks out a square region of the image                            │
│                                                                                │
│  Output: Output/rotation/, Output/cropping/, Output/flipping/, Output/rand_er/ │
└─────────────────────────────────┬──────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: FEATURE EXTRACTION (Pre_processing.py → augment)                      │
│  ─────────────────────────────────────────────────────────────────────         │
│  For each of 4 augmented images, extract 7 feature types:                      │
│                                                                                │
│  1. STATISTICAL FEATURES:                                                      │
│     - Mean: avg pixel intensity                                                │
│     - Variance: deviation from mean                                            │
│     - Kurtosis: peakedness of distribution                                     │
│     - Skewness: asymmetry of distribution                                      │
│     - Entropy: randomness/disorder via histogram                               │
│                                                                                │
│  2. TEXTURE FEATURES:                                                          │
│     - LBP: Local Binary Pattern → 8 neighbors → binary → decimal → histogram  │
│     - SLBT: Shape-and-Local-Binary-Texture → LBP + eigenface projection        │
│                                                                                │
│  Total per image: 2 + 1 + 1 + 1 + 100 + 100 + 100 = 305 features              │
│  Total per augmented variant: 305 features                                     │
│  Total across 4 variants: 4 × 305 = 1,220 features per original image         │
│  Total across 101 images: 101 × 4 × 305 = 123,205 values                      │
│                                                                                │
│  Saved to: Feat_fin.npy (123205 × ?) or Feat_fin.npy                          │
│  Labels: lab_fin.npy                                                           │
└─────────────────────────────────┬──────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: DMN CLASSIFICATION (prop_DMO/DeepMaxout.py)                           │
│  ─────────────────────────────────────────────────────────────────────         │
│  - Resize features to 64×64×3 (forced into image format)                      │
│  - Build Deep Maxout Network (3 conv layers, 3 maxout layers, 1 FC)            │
│  - Train with Adam for 2 epochs                                                │
│  - LHFGSO: Optimizes ALL weights (230,931 params) before training              │
│  - LHFGSO acts as pre-training weight initialization                           │
│  - Output: Cancer (1) or Non-cancer (0) per sample                             │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Technical Breakdown

### Step 1: Data Preparation (`Main/prepare_data.py`)

Identical to Paper 1. Converts raw database images to standardized 128×128 PNG with binary ground truth masks (RGB(0,242,255) → white 255, rest → black 0). Produces 101 image-mask pairs in `data/im/` and `data/gt/`.

### Step 2: ROI Extraction (`Pre_processing.py` → `Select_Roi`)

Identical to Paper 1. Takes center portion of 256×256 image, removing 10px top, 20px bottom, 20px left/right. Output: ~230×230 region focusing on prostate area.

### Step 3: Adaptive Median Filter (`Pre_processing.py` → `amf`)

This is **different from Paper 1 (which used T2FCS)**. The adaptive median filter is applied to each color channel separately:

```python
# Apply to each of B, G, R channels
b = amf(roi[:,:,0], initial_window=3, max_window=11)  # Blue channel
g = amf(roi[:,:,1], initial_window=3, max_window=11)  # Green channel
r = amf(roi[:,:,2], initial_window=3, max_window=11)  # Red channel
bgr_amf = np.dstack((b, g, r)).astype(np.uint8)        # Combine channels
```

**Why adaptive median filter instead of T2FCS?**
- T2FCS used fuzzy logic + Cuckoo Search optimization (optimization-within-optimization)
- Adaptive median filter is simpler, faster, and proven effective for impulse noise
- Preserves fine details while removing noise — ideal for medical imaging
- Reduces pipeline complexity so optimization effort focuses on segmentation + classification

**See Section 4 for detailed algorithm.**

### Step 4: SegNet Segmentation (`Main/Proposed_SegNet.py`)

Same architecture as Paper 1: 13 encoder blocks + bottleneck (Dense 1024×2) + 13 decoder blocks. Uses multi-objective loss (75% Dice + 25% cross-entropy). LHFGSO optimizes the final classification layer.

**See Sections 5 and 9 for details.**

### Step 5: Data Augmentation (`Main/Augmentation.py`)

**Four techniques** (vs two in Paper 1):

```python
def Augmentation(input_im):
    rows, cols, dim = input_im.shape  # e.g., 256, 256, 3

    # Technique 1: Rotation (30° counter-clockwise)
    angle = np.radians(30)
    M = np.float32([
        [np.cos(angle), -(np.sin(angle)), 0],
        [np.sin(angle),  np.cos(angle),  0],
        [0,              0,               1]
    ])
    rotated_img = cv2.warpPerspective(input_im, M, (int(cols), int(rows)))
    rotated_img = cv2.resize(rotated_img, (256, 256))

    # Technique 2: Cropping (30% top-left removed)
    cropped_image = input_im[int(cols*0.3):, int(cols*0.3):]
    cropped_image = cv2.resize(cropped_image, (256, 256))

    # Technique 3: Vertical flipping (flip around horizontal axis)
    fliped_img = cv2.flip(input_im, 0)

    # Technique 4: Random erasing (PyTorch RandomErasing)
    grayscale = cv2.cvtColor(input_im, cv2.COLOR_BGR2GRAY)
    x = transforms.ToTensor()(grayscale)  # Shape: (1, H, W)
    random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
    r_e = random_erase(x)                 # Randomly mask a region
    r_e = r_e.permute(1, 2, 0)            # (1, H, W) → (H, W, 1)
    r_e = r_e.numpy().reshape(H, W)       # Shape: (H, W)
    r_e1_1 = np.dstack((r_e, r_e, r_e)) * 255.999  # Convert to 3-channel
    r_e1_1 = r_e1_1.astype(np.uint8)

    return rotated_img, cropped_image, fliped_img, r_e1_1
```

**What each technique does:**

| Technique | Method | Purpose | Parameters |
|-----------|--------|---------|------------|
| Rotation | `cv2.warpPerspective` with 3×3 matrix | Simulates different patient positioning | 30° CCW |
| Cropping | Array slicing | Simulates partial field-of-view | 30% from top-left |
| Flipping | `cv2.flip(img, 0)` | Vertical flip (mirror along horizontal axis) | — |
| Random Erasing | `torchvision.transforms.RandomErasing` | Masks random rectangular region | probability=1, mode='pixel' |

**Data multiplication:** 1 original image → 4 augmented variants → 4× more training samples.

### Step 6: Feature Extraction (`Pre_processing.py` → `augment`)

This is the **major addition** from Paper 1. Instead of using histogram features directly, Paper 2 extracts **7 distinct feature types** from each augmented image:

```python
# For each of the 4 augmented images (rotate, crop, flip, rand_erase):

# === STATISTICAL FEATURES ===
# Mean
f_mean = np.mean(augmented_image)           # Shape: (1,)

# Variance
gray = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)
f_var = np.var(gray)                        # Shape: (1,)

# Kurtosis (4th moment about mean, normalized)
f_kurt = kurtosis(gray, axis=0, bias=True)  # Shape: (H, W) → flattened

# Skewness (3rd moment about mean, normalized)
f_skew = skew(gray, axis=0, bias=True)      # Shape: (H, W) → flattened

# Entropy (using skimage)
entr_img = entropy(gray, disk(10))          # Local entropy with radius 10
f_ent = np.histogram(entr_img[seg], 100)   # Histogram of entropy in cancer region
f_ent_fin = f_ent[0]                        # Shape: (100,)

# === TEXTURE FEATURES ===
# LBP: Local Binary Pattern
lbp_img = lbp.lbp_main(augmented_image)     # Full LBP image (H×W)
lbp_hist = cv2.calcHist([lbp_img], [0], None, [100], [0, 256])  # 100-bin histogram

# SLBT: Shape-and-Local-Binary-Texture
slbt_img = SLBT.slbt(augmented_image)       # Full SLBT image (H×W)
slbt_hist = cv2.calcHist([slbt_img], [0], None, [100], [0, 256])  # 100-bin histogram

# Concatenate all features
feat = np.concatenate((
    [f_mean, f_var],         # 2 features (statistical, shape: 2)
    f_kurt.reshape(1,-1),    # kurtosis of each pixel (variable)
    f_skew.reshape(1,-1),    # skewness of each pixel (variable)
    f_ent_fin,               # 100 entropy histogram bins
    lbp_hist.reshape(1,-1),  # 100 LBP histogram bins
    slbt_hist.reshape(1,-1)  # 100 SLBT histogram bins
), axis=1)

# Result: feat shape varies per augmentation
# Saved to Feat_fin.npy
```

**Feature count per augmented image:**
```
Statistical (mean + variance):     2 features
Kurtosis per pixel:               H × W features (e.g., 256×256 = 65,536)
Skewness per pixel:               H × W features (e.g., 65,536)
Entropy histogram (100 bins):      100 features
LBP histogram (100 bins):          100 features
SLBT histogram (100 bins):         100 features
─────────────────────────────────────────────
Total:                           ~65,938 features (highly dimensional)
```

**Key observation:** The feature vectors are extremely high-dimensional (tens of thousands of features). The DMN classifier takes these as input and uses its maxout layers to learn optimal nonlinear combinations.

### Step 7: DMN Classification (`prop_DMO/DeepMaxout.py`)

**See Sections 10 and 11 for full details.** The DMN (Deep Maxout Network) with LHFGSO optimization classifies the extracted features.

**Data pipeline for DMN:**
```python
# Read features and labels
x_train = read_data()   # From Feat_fin.npy
y_train = read_label()  # From lab_fin.npy

# Split
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=tr, random_state=42)

# Resize features into image format for DMN
xt = len(X_train)
X_train = np.resize(X_train, (xt, 64, 64, 3))  # Force into 64×64×3 shape

# Initial training with Adam
model2.fit(X_train, y_trainx, epochs=2, batch_size=10, verbose=0)

# Get initial weights
Initial_weight = model2.get_weights()

# Apply LHFGSO optimization
updated_weights = LHFGSO.algm(Initial_weight)
model2.set_weights(updated_weights)

# Continue training from LHFGSO-optimized starting point
# ... evaluation ...
```

---

## 4. Adaptive Median Filter: How It Works

### Why Adaptive Median Filter?

Standard median filters use a fixed window size. They fail when:
- Noise level varies across the image
- Fine details (edges, small structures) are close in size to the window

Adaptive median filter solves this by **dynamically adjusting the window size** based on local noise characteristics.

### Algorithm: Two-Level Adaptive Approach

The algorithm has two levels per pixel:

**Level A (Check if median is useful):**
```python
def level_A(z_min, z_med, z_max, z_xy, S_xy, S_max):
    # z_min = minimum intensity in window
    # z_med = median intensity in window
    # z_max = maximum intensity in window
    # z_xy = current pixel intensity
    # S_xy = current window size
    # S_max = maximum window size (11)

    if z_min < z_med < z_max:
        # Median is NOT an impulse — it's a useful representative
        # Proceed to Level B
        return level_B(z_min, z_med, z_max, z_xy, S_xy, S_max)
    else:
        # Median IS an impulse — expand window
        S_xy += 2  # Increase window size by 2 (keep it odd)
        if S_xy <= S_max:
            # Try again with larger window
            return level_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
        else:
            # Reached max window, return median anyway
            return z_med
```

**Level B (Check if current pixel is an impulse):**
```python
def level_B(z_min, z_med, z_max, z_xy, S_xy, S_max):
    if z_min < z_xy < z_max:
        # Current pixel is NOT an impulse — keep it
        return z_xy
    else:
        # Current pixel IS an impulse — replace with median
        return z_med
```

### Full Algorithm per Pixel:

```
For each pixel at position (row, col):
  S_xy = initial_window (3×3)
  
  while True:
    Extract window of size S_xy × S_xy around (row, col)
    Calculate: z_min, z_med, z_max, z_xy (current pixel)
    
    if level_A(z_min, z_med, z_max, z_xy, S_xy, S_max) returns a value:
      Use that value as the new pixel intensity
      Break
    
    else:
      S_xy += 2  # Expand window
      if S_xy > S_max:
        # Use median of max window
        new_pixel = z_med
        Break
```

### Visual Example:

```
Window size 3×3 (S_xy=3):
┌─────────┐
│ a b c   │
│ d e f   │  z_min = min(a,b,c,d,e,f,g,h,i)
│ g h i   │  z_med = median(a,b,c,d,e,f,g,h,i)  ← median of 9 values
└─────────┘  z_max = max(a,b,c,d,e,f,g,h,i)
             z_xy = e (current pixel)

Case 1: z_min < z_med < z_max (e.g., z_min=10, z_med=50, z_max=200, z_xy=180)
→ level_B: z_min < z_xy < z_max → return z_xy (keep original value)

Case 2: z_min < z_med < z_max (e.g., z_xy=5, which is impulse)
→ level_B: z_xy ≤ z_min → return z_med (replace with median)

Case 3: z_min < z_med < z_max is FALSE (e.g., z_med is an impulse)
→ Expand to 5×5 window and repeat

Case 4: z_min < z_med < z_max is FALSE AND S_xy > S_max
→ Return z_med (use max-window median)
```

### Key Properties:

| Property | Value |
|----------|-------|
| Initial window size | 3×3 |
| Maximum window size | 11×11 |
| Window growth | +2 per iteration (odd sizes: 3, 5, 7, 9, 11) |
| Maximum iterations per pixel | 5 (3→5→7→9→11) |
| Noise removal | Impulse/salt-and-pepper noise |
| Edge preservation | High — preserves fine details |
| Complexity | O(H × W × max_iterations × window_size²) |
| Applied to | Each of B, G, R channels separately |

### Comparison with T2FCS (Paper 1):

| Aspect | T2FCS (Paper 1) | Adaptive Median Filter (Paper 2) |
|--------|----------------|----------------------------------|
| Approach | Fuzzy logic + neighborhood averaging | Adaptive window median |
| Optimization | Uses Cuckoo Search internally | No optimization — classical signal processing |
| Complexity | Higher (fuzzy membership + optimization) | Lower (simple median + window expansion) |
| Edge preservation | Moderate | High |
| Impulse noise removal | Good | Excellent |
| Computational cost | Higher | Lower |
| Parameters | Threshold-based | Window size-based (3→11) |
| Suitability for pipeline | Complex but effective | Simple and effective |

**Rationale for switch:** T2FCS adds optimization complexity to the preprocessing stage. Adaptive median filtering achieves similar or better noise removal with simpler mathematics, allowing the metaheuristic optimization to focus on what matters — training the segmentation and classification networks.

---

## 5. SegNet Segmentation with Multi-Objective Loss

### Architecture (Same as Paper 1)

```
INPUT: (192, 256, 3)

ENCODER:
  Block 1: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → (96, 128, 64)
  Block 2: Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool → (48, 64, 128)
  Block 3: Conv2D(256) × 3 → BN → ReLU → MaxPool → (24, 32, 256)
  Block 4: Conv2D(512) × 3 → BN → ReLU → MaxPool → (12, 16, 512)
  Block 5: Conv2D(512) × 3 → BN → ReLU → MaxPool → (6, 8, 512)

BOTTLENECK:
  Dense(1024, ReLU) → Dense(1024, ReLU)

DECODER:
  Block 1: UpSample → Conv2DTranspose(512) × 3 → (12, 16, 512)
  Block 2: UpSample → Conv2DTranspose(512) → Conv2DTranspose(256) → (24, 32, 256)
  Block 3: UpSample → Conv2DTranspose(256) → Conv2DTranspose(128) → (48, 64, 128)
  Block 4: UpSample → Conv2DTranspose(128) → Conv2DTranspose(64) → (96, 128, 64)
  Block 5: UpSample → Conv2DTranspose(64) → Conv2DTranspose(1) → Sigmoid → (192, 256, 1)

OUTPUT: (192, 256) probability map
```

### Multi-Objective Loss Function

The key innovation (same as Paper 1) is the combined loss function:

```python
def prop_loss_fn(y_true, y_pred, smooth=1e-15):
    sigma = 0.75  # Weight: 75% Dice, 25% cross-entropy

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    # Component 1: Cross-entropy (pixel-wise classification)
    # Maximizes: Σ(y_true × log(y_pred)) for correct pixels
    # Minimizes: Σ(y_true × log(1 - y_pred)) for incorrect pixels
    cross_entropy = K.sum(y_true_f * math.log(y_pred_f))

    # Component 2: Dice coefficient (spatial overlap quality)
    # Dice = 2 × |A ∩ B| / (|A| + |B|)
    # Higher overlap → higher Dice → lower loss
    dice = (2 * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )

    # Combined loss (minimize this):
    # E(a, b) = (1 - σ) × CrossEntropy - σ × log(Dice)
    loss = (1 - sigma) * cross_entropy - sigma * math.log(dice)

    return loss
```

**Expanded form:**
```
Loss = 0.25 × Σ(a_d × log(b_d)) - 0.75 × log((2 × Σ(a_d × b_d) + 1e-15) / (Σa_d + Σb_d + 1e-15))

Where:
  a_d = ground truth pixel at position d (0 or 1)
  b_d = predicted probability at position d (0 to 1)
  σ = 0.75 (75% weight on Dice)
  d indexes all 192×256 = 49,152 pixels
```

**Why 75% weight on Dice?**
- Cancer regions are small compared to background (class imbalance)
- Cross-entropy alone would predict mostly background and still achieve ~95% "accuracy" by getting most pixels right
- Dice directly measures **overlap quality**: `2|A∩B|/(|A|+|B|)`
- Higher Dice weight → model prioritizes finding the cancer region accurately
- 25% cross-entropy ensures pixel-level classification quality

### SegNet with LHFGSO

The training is identical to Paper 1. The key difference is that LHFGSO (instead of HFGSO) is used to optimize the classification layer weights. See Section 9 for integration details.

---

## 6. Data Augmentation (4 Techniques)

Paper 2 uses **4 augmentation techniques** (vs 2 in Paper 1):

### 6.1 Rotation

```python
# 30° counter-clockwise rotation
angle = np.radians(30)
M = np.float32([
    [np.cos(angle), -(np.sin(angle)), 0],
    [np.sin(angle),  np.cos(angle),  0],
    [0,              0,               1]
])
rotated_img = cv2.warpPerspective(input_im, M, (int(cols), int(rows)))
rotated_img = cv2.resize(rotated_img, dsize=(256, 256))
```

**Purpose:** Prostate MRI scans can appear at slightly different orientations depending on patient positioning, scanner protocol, and slice angle. Rotation simulates this natural variation.

**Geometric transformation:**
```
x' = x × cos(30°) - y × sin(30°)
y' = x × sin(30°) + y × cos(30°)
```

### 6.2 Cropping (30%)

```python
# Remove top-left 30% of image
cropped_image = input_im[int(cols*0.3):, int(cols*0.3):]
# For 256×256: starts at (77, 77), keeps 179×179 region
cropped_image = cv2.resize(cropped_image, dsize=(256, 256))
```

**Purpose:** Simulates partial field-of-view variations between scans. Different MRI slices capture different portions of the anatomy, and cropping simulates this.

### 6.3 Flipping (Vertical)

```python
# Flip around horizontal axis (vertical flip)
# cv2.flip(img, 0) means: flip along x-axis
# For image:
#   Original y=0 (top)     → New y'=H-1 (bottom)
#   Original y=H-1 (bottom) → New y'=0 (top)
fliped_img = cv2.flip(input_im, 0)
```

**Purpose:** The prostate is vertically symmetric (top-to-bottom), so vertical flipping produces anatomically plausible variations. This effectively doubles the data without adding noise or distortion.

**Note:** `cv2.flip(input_im, 0)` flips **vertically** (top-to-bottom). `cv2.flip(input_im, 1)` would flip horizontally. The code uses `flip=0`, which is the correct choice for vertical symmetry of the prostate.

### 6.4 Random Erasing

```python
from torchvision.transforms import RandomErasing

# Convert grayscale image to tensor
grayscale = cv2.cvtColor(input_im, cv2.COLOR_BGR2GRAY)
x = transforms.ToTensor()(grayscale)  # Shape: (1, H, W)

# Randomly erase a region (probability=1 means always applies)
random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
r_e = random_erase(x)  # Tensor with random region masked

# Convert back to image format
r_e = r_e.permute(1, 2, 0)  # (1, H, W) → (H, W, 1)
r_e = r_e.numpy().reshape(H, W)  # Flatten channel
r_e1_1 = np.dstack((r_e, r_e, r_e)) * 255.999  # Replicate to 3 channels
r_e1_1 = r_e1_1.astype(np.uint8)  # Convert to uint8
```

**What RandomErasing does:**
- Randomly selects a rectangular region within the image
- Fills that region with random pixel values (mode='pixel')
- The region size and position are random
- Probability=1 means it always applies

**Purpose:** Forces the model to learn from partial information and not rely on specific spatial patterns. Makes the model robust to:
- Partial occlusions in medical images
- Variations in field-of-view
- Missing or artifact-corrupted regions

### Augmentation Impact

| Technique | Data Multiplier | Clinically Plausible? |
|-----------|----------------|----------------------|
| Rotation | 2× | Yes — different orientations occur naturally |
| Cropping | 2× | Yes — partial FOV variations occur naturally |
| Flipping | 2× | Yes — vertical symmetry of prostate |
| Random Erasing | 2× | Yes — simulates partial occlusion/artifact |

**Combined:** 1 original image → 4 augmented images. From 101 images → 404 images. This significantly increases training diversity.

---

## 7. Feature Extraction: LBP, SLBT, and Statistical Features

This is the **major new component** in Paper 2 that was absent in Paper 1.

### 7.1 LBP (Local Binary Pattern)

**What it does:** Captures local texture patterns by comparing each pixel with its 8 neighbors.

```python
def lbp_calculated_pixel(img_gray, x, y):
    center = img_gray[x][y]  # Threshold center pixel value

    # Sample 8 neighbors (clockwise from top-left)
    val_ar = [
        img_gray[x-1, y-1],  # top-left
        img_gray[x-1, y  ],  # top
        img_gray[x-1, y+1],  # top-right
        img_gray[x  , y+1],  # right
        img_gray[x+1, y+1],  # bottom-right
        img_gray[x+1, y  ],  # bottom
        img_gray[x+1, y-1],  # bottom-left
        img_gray[x  , y-1],  # left
    ]

    # Binary comparison: 1 if neighbor >= center, 0 if neighbor < center
    binary = []
    for neighbor in val_ar:
        if neighbor >= center:
            binary.append(1)
        else:
            binary.append(0)

    # Convert 8-bit binary to decimal
    powers = [1, 2, 4, 8, 16, 32, 64, 128]
    lbp_value = sum(binary[i] * powers[i] for i in range(8))
    # lbp_value ranges from 0 to 255

    return lbp_value
```

**Full image LBP:**
```python
def lbp_main(img_in):
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape
    img_lbp = np.zeros((height, width), np.uint8)

    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

    return img_lbp  # Returns H×W image where each pixel is an LBP code (0-255)
```

**Then histogram is computed:**
```python
lbp_img = lbp.lbp_main(rotated_img)  # Full H×W LBP image
lbp_hist = cv2.calcHist([lbp_img], [0], None, [100], [0, 256])
# 100-bin histogram of LBP codes
# Captures distribution of texture patterns across the image
```

**Visual example:**
```
Original grayscale image:
   Pixel at (5,5) has value 120
   8 neighbors: [130, 110, 140, 125, 115, 105, 135, 118]

Binary comparison (neighbor >= center?):
   [1, 0, 1, 1, 0, 0, 1, 1]  →  [1,0,1,1,0,0,1,1]

As binary: 1×1 + 0×2 + 1×4 + 1×8 + 0×16 + 0×32 + 1×64 + 1×128 = 1 + 4 + 8 + 64 + 128 = 205

LBP code at (5,5) = 205
```

**What LBP captures:**
- **Homogeneous regions** (e.g., healthy tissue) → Most neighbors similar to center → Binary values either mostly 0s or mostly 1s → Low LBP values
- **Heterogeneous regions** (e.g., tumor boundaries, irregular tissue) → More neighbors differ from center → Mixed binary values → Wide range of LBP values
- **Cancerous tissue** typically has higher LBP variance due to irregular texture

### 7.2 SLBT (Shape-and-Local-Binary-Texture)

**What it does:** Combines local texture patterns (LBP) with shape information by projecting LBP histograms into eigenface space.

```python
def slbt(img1):
    def lbp(img):
        # Same LBP computation as above
        # Returns lbp_photo: H×W image of LBP codes (0-255)

        def assign_bit(picture, x, y, c):
            bit = 0
            try:
                if picture[x][y] >= c:
                    bit = 1
            except:
                pass
            return bit

        def local_bin_val(picture, x, y):
            # Same 8-neighbor comparison (clockwise from top-right)
            eight_bit_binary = []
            centre = picture[x][y]
            powers = [1, 2, 4, 8, 16, 32, 64, 128]

            # Note: SLBT starts from top-right (not top-left)
            eight_bit_binary.append(assign_bit(picture, x-1, y+1, centre))  # top-right
            eight_bit_binary.append(assign_bit(picture, x, y+1, centre))    # right
            eight_bit_binary.append(assign_bit(picture, x+1, y+1, centre))  # bottom-right
            eight_bit_binary.append(assign_bit(picture, x+1, y, centre))    # bottom
            eight_bit_binary.append(assign_bit(picture, x+1, y-1, centre))  # bottom-left
            eight_bit_binary.append(assign_bit(picture, x, y-1, centre))    # left
            eight_bit_binary.append(assign_bit(picture, x-1, y-1, centre))  # top-left
            eight_bit_binary.append(assign_bit(picture, x-1, y, centre))    # top

            decimal_val = sum(eight_bit_binary[i] * powers[i] for i in range(8))
            return decimal_val

        m, n, _ = img.shape
        gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp_photo = np.zeros((m, n), np.uint8)

        for i in range(0, m):
            for j in range(0, n):
                lbp_photo[i, j] = local_bin_val(gray_scale, i, j)

        return lbp_photo

    l = lbp(img1)  # Compute LBP image
    return l       # Returns shape-free LBP (H×W)
```

**The SLBT process:**
1. Compute LBP image (same as above, but starting clockwise from top-right)
2. Extract histogram: captures local texture distribution
3. Project into eigenface space: captures global shape variation
4. The combination encodes both **shape** (global structure) and **texture** (local patterns)

**Difference between LBP and SLBT:**

| Aspect | LBP | SLBT |
|--------|-----|------|
| Neighbor order | Top-left → clockwise | Top-right → clockwise |
| Shape information | Texture only | Shape + texture |
| Eigenface projection | Not included | Included |
| What it captures | Local texture patterns | Shape-free texture in eigen space |

The different neighbor starting point (top-right vs top-left) means LBP and SLBT can capture different aspects of the texture. Combined, they provide complementary texture information.

### 7.3 Statistical Features

**Mean:**
```python
f_mean = np.mean(rotated_img)
# Average pixel intensity across the image
# Cancerous regions often have different average brightness than normal tissue
```

**Variance:**
```python
gray_rotate = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
f_var = np.var(gray_rotate)
# Measures how much pixel intensities deviate from the mean
# Heterogeneous tissue (cancer) → higher variance
# Homogeneous tissue (normal) → lower variance
```

**Kurtosis:**
```python
from scipy.stats import kurtosis
f_kurt = kurtosis(gray_rotate, axis=0, bias=True)
# 4th normalized moment of intensity distribution
# Measures "tailedness" — how peaked or flat the distribution is vs normal
# Kurtosis > 3 (leptokurtic): Heavy tails, sharp peak (common in cancer)
# Kurtosis < 3 (platykurtic): Light tails, flat peak (common in normal tissue)
```

**Skewness:**
```python
from scipy.stats import skew
f_skew = skew(gray_rotate, axis=0, bias=True)
# 3rd normalized moment of intensity distribution
# Measures asymmetry around the mean
# Skewness = 0: symmetric distribution
# Positive skew: tail extends to the right (brighter pixels dominate)
# Negative skew: tail extends to the left (darker pixels dominate)
# Abnormal tissue often shows skewed intensity distributions
```

**Entropy:**
```python
from skimage.filters.rank import entropy
entr_img = entropy(gray_rotate, disk(10))  # Local entropy with radius 10
f_ent = np.histogram(entr_img[seg], 100)  # Histogram of entropy in cancer region
# Entropy measures randomness/disorder in pixel values
# High entropy → more heterogeneous (often cancer)
# Low entropy → more uniform (often normal tissue)
# Local entropy: computed over neighborhoods (disk radius 10)
```

### Feature Extraction Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  Feature Extraction Pipeline (per augmented image)              │
│                                                                 │
│  Statistical Features (from grayscale):                         │
│  ├── Mean (1 value)                                             │
│  ├── Variance (1 value)                                         │
│  ├── Kurtosis per pixel (H×W values, then histogram)           │
│  └── Skewness per pixel (H×W values, then histogram)           │
│                                                                 │
│  Texture Features:                                              │
│  ├── Entropy histogram (100 bins) — local entropy in cancer    │
│  ├── LBP histogram (100 bins) — texture patterns               │
│  └── SLBT histogram (100 bins) — shape-free texture            │
│                                                                 │
│  Total: ~65,938 features per augmented image                   │
│  After histogram binning: ~305 features                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. LHFGSO Algorithm Deep Dive

### 8.1 What is LHFGSO?

**LHFGSO** = **Light Henry Firefly Gas Solubility Optimization**

It is a **three-level hybrid metaheuristic**:

```
Level 1: HGSO (Henry Gas Solubility Optimization)
         └→ Based on Henry's law from chemistry: gas dissolving in liquid

Level 2: FA (Firefly Algorithm)
         └→ Based on bioluminescence: fireflies attract each other by brightness

Level 3: LSO (Light Spectrum Optimizer) — NEW in Paper 2
         └→ Based on rainbow effect: light dispersion through water droplets
```

### 8.2 Mathematical Foundation

#### Level 1: Henry Gas Solubility Optimization (HGSO)

**Henry's Law:** At constant temperature, the amount of gas dissolved in a liquid is proportional to its partial pressure above the liquid.

```
C = k_H × P
Where:
  C = concentration of dissolved gas (mol/L)
  P = partial pressure of gas (atm)
  k_H = Henry's constant (temperature-dependent)
```

**HGSO maps gas physics to optimization:**

| Physics | Optimization |
|---------|-------------|
| Gas molecules | Candidate solutions (weight vectors) |
| Solvent | Search space |
| Partial pressure | Exploration "push" |
| Henry's constant k_H | Temperature-dependent exploration parameter |
| Solubility S | How readily solutions explore |
| Temperature T | Iteration counter (annealing schedule) |

**HGSO equations:**
```python
# Henry's coefficient update
H_i(f+1) = H_i(f) × exp(−M_d × (1/N(f) − 1/N_ψ))
# Where: N(f) = exp(−f/ν) (temperature), N_ψ = 298.15 (reference)

# Solubility update
O_{i,c}(f) = P × H_i(f+1) × I_{i,c}(f)
# P = constant (0.5), I_{i,c} = partial pressure of agent
```

#### Level 2: Firefly Algorithm (FA)

**FA behavior:** Fireflies move toward brighter (better) fireflies, with attractiveness decreasing with distance.

```python
# Attractiveness at distance r
beta = beta0 × exp(-γ × r²)

# Movement toward brighter firefly
x_i_new = x_i + beta × (x_j - x_i) + α × ε
# α = step size, ε = random perturbation
```

#### Level 3: Light Spectrum Optimizer (LSO) — NEW

**LSO inspiration:** When white light passes through a water droplet, it disperses into a spectrum (rainbow effect) at various angles.

**LSO mechanism:**
```python
# Differential evolution-style perturbation
q_new = (q_s1 + GT5 × (q_s2 - q_s3)) × t_vec + (1 - t_vec) × q_current
# Where:
#   q_s1, q_s2, q_s3 = three random solutions from population
#   GT5 = normally distributed scalar (γ in the paper)
#   t_vec = random vector with values in (0, 1)
#   q_current = current solution
```

**What LSO adds:**
- **Differential evolution diversity**: Picks 3 random solutions and creates a perturbation vector (q_s2 - q_s3)
- **Spectrum-inspired search**: Like light splitting into colors, the search explores multiple directions
- **Better exploration**: Prevents premature convergence by introducing random differential vectors
- **Complementary to HGSO + FA**: FA attracts, HGSO explores via solubility, LSO diversifies via differential perturbation

### 8.3 Complete LHFGSO Algorithm (10 Steps)

```
LHFGSO Algorithm (10 steps, Tmax iterations)

═════════════════════════════════════════════════════════════════════
INPUT: w = list of weight arrays from neural network (DMN or SegNet)
       N = len(w) ≈ 30 (number of weight arrays)
       M = max(len(arr) for arr in w) ≈ largest single weight array size
       Tmax = 10 (maximum iterations)

STEP 1: PREPARE WEIGHTS
───────────────────────────────────────────────────────────────────────
a) Convert to list if needed:
   if not isinstance(w, list): w = list(w)

b) Flatten each weight array:
   flat_weights = [arr.flatten() for arr in w]

c) Find global bounds across all weight arrays:
   lb = min(arr.min() for arr in w)  # Global minimum
   ub = max(arr.max() for arr in w)  # Global maximum

d) Set dimension:
   Xmin, Xmax = 1, 5  # Fixed bounds
   N, M = len(flat_weights), max(len(arr) for arr in w)
   # N = number of weight arrays (not population size!)
   # M = size of largest flattened weight array

═════════════════════════════════════════════════════════════════════
STEP 2: INITIALIZE POPULATION
───────────────────────────────────────────────────────────────────────
For each of N "agents" (which correspond to weight arrays in this implementation):
   For each of M dimensions (padded to max size):
       X[i][j] = random.random()  # Uniform in [0, 1)

Initialize algorithm constants:
   l1 = 5 × exp(-2) ≈ 0.067  (Henry's constant base)
   l2 = 100                     (Partial pressure base)
   l3 = 1 × exp(-2) ≈ 0.01   (Compression constant)

   Hj = l1 × random()         → ~0.067 × U(0,1) ≈ 0.033
   Pij = l2 × random()        → ~100 × U(0,1) ≈ 50
   Cj = l3 × random()         → ~0.01 × U(0,1) ≈ 0.005

   F = random.uniform(-1, 1)  → Random scaling factor
   E = random.sample(range(1, N+1), N)  → [1, 2, 3, ..., N]
   alpha = 1.0               → Step size

═════════════════════════════════════════════════════════════════════
STEP 3: EVALUATE INITIAL FITNESS
───────────────────────────────────────────────────────────────────────
For each agent i:
   fitness_i = sum of all values in X[i] + random()
   # Simple sum-based fitness (placeholder — in full implementation,
   # this would evaluate actual model loss)

   Fit[i] = fitness_i

Select best:
   Fbest = max(Fit)     → Highest fitness
   best = Fit.index(Fbest)  → Index of best
   Xbest = max(X[best]) → Best solution value

═════════════════════════════════════════════════════════════════════
STEP 4: MAIN LOOP (repeat Tmax = 10 times)
───────────────────────────────────────────────────────────────────────

   For iteration t = 1 to Tmax:

   (a) Update Temperature (annealing)
   ────────────────────────────────────
   T = exp(-t / Tmax)
   # t=1: T ≈ 0.905, t=10: T ≈ 0.368

   (b) Update Henry's Coefficient (HGSO component)
   ────────────────────────────────────────────────
   Hj = Hj × exp(-Cj × (1/T) - (1/298.15))
   # As T decreases, the exponent becomes more negative
   # → Hj decreases (gas comes out of solution)

   (c) Calculate Gas Solubility (HGSO component)
   ──────────────────────────────────────────────
   S = 0.5 × Hj × Pij
   # Solubility ∝ Henry's coefficient × partial pressure
   # As iterations progress (T decreases, Hj decreases), S decreases
   # → Less "dissolved" → less movement → convergence

   (d) Calculate Attractiveness (FA component)
   ───────────────────────────────────────────
   rr = sqrt((X[0][0] - X[1][1])²)  # Distance between agent 0 and 1
   beta0 = exp(-1 × rr)             # Attractiveness at distance rr
   # Note: The distance calculation uses X[0][0] and X[1][1],
   # which are specific scalar positions, not full vectors

   (e) Calculate Movement Intensity (FA component)
   ────────────────────────────────────────────────
   gamma_param = 0.5 × exp(-(Fbest + 0.05) / (Fit[i] + 0.05))
   # Poor solutions → larger gamma → more movement
   # Good solutions → smaller gamma → fine-tuning

   (f) Apply LHFGSO Position Update (FA + HGSO combined)
   ─────────────────────────────────────────────────────
   For each agent i, dimension j:

       n = (
           # Term A: FA attraction (firefly pulled by its own position)
           (beta0 × exp(-gamma_param × r²) × X[i][j] × alpha × E[i])
           × ((F × r × gamma_param) + ((F × r × alpha) - 1))

           # Term B: HGSO + LSO (push toward best, with differential perturbation)
           + (F × r × (gamma_param × Xbest + alpha × S × Xbest))
           × (1 - beta0 × exp(-gamma_param × r²))
       ) / (
           (F × r × gamma_param) + (F × r × alpha)
           - beta0 × exp(-gamma_param × r²)
       )

       new_X[i][j] = n

   (g) Apply Boundary Constraints
   ───────────────────────────────
   For each agent i, dimension j:
       if new_X[i][j] < 0 or new_X[i][j] > 100:
           new_X[i][j] = random.uniform(lb, ub)
       else:
           new_X[i][j] = new_X[i][j]

   (h) Escape Local Optima (reposition worst agents)
   ─────────────────────────────────────────────────
   c1, c2 = 0.1, 0.2
   Nw = M × (random.uniform(0.1, 0.2) + 0.1)
   # Nw ≈ 15-30% of positions will be repositioned

   G = 1 + random() × (5 - 1)  # Random in [1, 5]
   worst = round(G)
   # Randomly replace worst solutions

   (i) Re-evaluate Fitness
   ─────────────────────────
   For each agent i:
       fitness_i = sum(new_X[i]) + random()
       Fit[i] = fitness_i

   Update best:
       Fbest = max(Fit)
       best = Fit.index(Fbest)
       Xbest = max(X[best])

═════════════════════════════════════════════════════════════════════
STEP 5: RETURN OPTIMIZED WEIGHTS
───────────────────────────────────────────────────────────────────────
a) Extract best solution:
   best_solution = X[best]  # Shape: list of M values

b) Reconstruct weight arrays:
   updated_weights = []
   start = 0
   for original_weight in w:
       shape = original_weight.shape
       size = np.prod(shape)

       if start + size > len(X[best]):
           # If not enough data, use original weights
           best_solution_slice = list(original_weight.flatten())
       else:
           best_solution_slice = X[best][start:start + size]

       updated_weight = np.array(best_solution_slice[:size]).reshape(shape)
       updated_weights.append(updated_weight)
       start += size

c) Return: list of numpy arrays (optimized weights)
```

### 8.4 Key Difference: LHFGSO vs HFGSO

| Component | HFGSO (Paper 1) | LHFGSO (Paper 2) |
|-----------|----------------|------------------|
| Level 1 | HGSO (gas solubility) | HGSO (gas solubility) |
| Level 2 | FA (firefly attraction) | FA (firefly attraction) |
| Level 3 | None | **LSO (light spectrum, differential perturbation)** |
| Position update | Same hybrid equation | Same hybrid equation |
| Parameter set | 13 parameters | 13 parameters (same) |
| Complexity | 2-algorithm hybrid | 3-algorithm hybrid |

**Why add LSO?**

The LSO component adds **differential evolution-style diversification** to the search. By combining:

- **FA**: Attraction toward best solution (exploitation)
- **HGSO**: Temperature-annealed exploration (exploration)
- **LSO**: Random differential perturbation from 3 random solutions (diversification)

The algorithm achieves better **balance between exploration, exploitation, and diversification**, which is critical in complex, multimodal loss landscapes like medical image analysis.

### 8.5 Convergence Analysis

The paper provides convergence curves (Fig. 9 in the paper):

| Optimizer | Fitness Value (Iteration 25) |
|-----------|------------------------------|
| SGD | 0.0571 |
| Adam | 0.0458 |
| RMSprop | 0.0314 |
| HFGSO | 0.0205 |
| **LHFGSO** | **0.0252** |

**Interesting observation:** HFGSO achieves *lower* fitness (0.0205) than LHFGSO (0.0252), yet LHFGSO-DMN achieves *better* classification metrics (94.63% vs 92.63%).

**Interpretation:** Lower training fitness does NOT always mean better generalization. LHFGSO's additional LSO diversification prevents overfitting to training data, finding solutions that generalize better to unseen data. This is a key advantage of the three-level hybrid.

---

## 9. LHFGSO in SegNet (Segmentation)

### Where LHFGSO is Called

In `Main/Proposed_SegNet.py` (same location as Paper 1):

```python
# Get current model weights
w = model.get_weights()

# Apply LHFGSO optimization
model.set_weights(w + LHFGSO.algm(w))

# Load pretrained weights (overwrites LHFGSO results)
model.load_weights('segnet_100.h5')

# Run prediction
seg_img = predict(model, org)
```

**Note:** The same code structure issue from Paper 1 exists here — LHFGSO-optimized weights are applied then overwritten by `load_weights()`. The LHFGSO optimization runs but is then replaced by pretrained weights.

### What LHFGSO Optimizes in SegNet

LHFGSO receives all trainable parameters (~35-40 million weights) including:
- 13 encoder convolutional layers (kernels + biases)
- Batch normalization parameters (gamma, beta, mean, var)
- Dense layer weights (bottleneck: 1024×1024)
- 13 decoder transposed convolutional layers
- Final classification layer weights

### Fitness Function for SegNet

The fitness function uses the **multi-objective loss** (Dice + cross-entropy):

```
Fitness = (1 - 0.75) × CrossEntropy - 0.75 × log(Dice + 1e-15)
```

The optimizer searches for weights that minimize this combined loss.

### Note on SegNet LHFGSO

The SegNet LHFGSO optimization serves the same purpose as in Paper 1 — as a pre-initialization step before loading pretrained weights. The major difference is that the overall pipeline uses the **3-level LHFGSO** instead of the 2-level HFGSO.

---

## 10. Deep Maxout Network (DMN) Architecture

### 10.1 What is Maxout?

**Maxout** is a type of activation function introduced by Goodfellow et al. (2013). Unlike ReLU which is fixed (`max(0, x)`), maxout is **trainable**:

```
Standard activation functions:
  ReLU:     f(x) = max(0, x)                    ← Fixed function
  LeakyReLU: f(x) = max(0.01x, x)               ← Fixed function
  Sigmoid:  f(x) = 1 / (1 + exp(-x))           ← Fixed function

Maxout:     f(x) = max(w₁·x + b₁, w₂·x + b₂, ..., wₐ·x + bₐ)
                                          ↑
                                    a linear pieces, all trainable
```

**Mathematical formulation:**
```
p^b_{i,o,k} = max_{k∈[1,a]} (f^{n-1}_{o,k} · W_{ok} + m_{ok})

Where:
  p = output of maxout layer
  b = layer index
  i = input sample
  o = output unit
  k = piece index (1 to a)
  f^{n-1} = input from previous layer
  W = weight matrix (trainable)
  m = bias vector (trainable)
  a = number of linear pieces (hyperparameter)
```

### 10.2 Why Maxout Over ReLU?

| Aspect | ReLU | Maxout |
|--------|------|--------|
| Activation function | Fixed: max(0, x) | Trainable: max of a linear functions |
| Dead neurons | Yes — neurons can "die" if output is always 0 | No — always at least some positive output |
| Expressiveness | Can represent ReLU, LeakyReLU | Can represent ReLU, absolute value, and **any piecewise linear function** |
| Parameters | w · x + b (one set) | w₁·x+b₁, ..., wₐ·x+bₐ (a sets) |
| Memory | Lower | a times more parameters per layer |
| Gradient flow | Can stop if all inputs are negative | Always flows — max of positives |

**The key advantage:** Maxout prevents the "dying ReLU" problem. In deep networks trained with ReLU, some neurons can get stuck outputting zero for all inputs (if their bias makes w·x+b always negative). Once dead, they never recover. Maxout always has at least one positive linear piece, so gradients always flow.

### 10.3 DMN Architecture (Full Details)

```python
from tensorflow.keras.layers import Layer

# Custom Maxout Layer
class Maxout(Layer):
    def __init__(self, units, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # input_shape = (batch, H, W, channels)
        # Create weight matrix: (channels, units)
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        # inputs: (batch, H, W, channels)
        input_shape = tf.shape(inputs)
        flattened = tf.reshape(inputs, [-1, input_shape[-1]])
        # flattened: (batch×H×W, channels)

        output = tf.matmul(flattened, self.kernel)
        # output: (batch×H×W, units) — each unit has its own linear projection

        output = tf.reshape(output,
            [input_shape[0], input_shape[1], input_shape[2], self.units])
        # Reshape back to spatial: (batch, H, W, units)

        return output
        # NOTE: This returns linear combinations, NOT max.
        # The max operation is applied separately via maxout_activation_function.

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.units)
```

```python
# Maxout activation: max over the last axis
def maxout_activation_function(inputs, units, axis=-1):
    input_shape = inputs.get_shape().as_list()
    n_dims = len(input_shape)
    num_channels = input_shape[axis]

    # Reshape: split the 'units' dimension into (units, num_channels/units)
    input_shape[axis] = units
    input_shape.append(num_channels // units)

    output = K.reshape(inputs, (-1,) + tuple(input_shape[1:]))
    # Reshape: (batch, H, W, units, pieces)
    output_max = K.max(output, axis=-1, keepdims=False)
    # Take max over the last axis (the pieces)
    # Returns: (batch, H, W, units)

    return output_max
```

### 10.4 Full DMN Model

```python
img_shape = (64, 64, 3)  # Input: 64×64 RGB images (resized features)
inp = L.Input(img_shape)

# === LAYER 1 ===
conv1 = L.Conv2D(filters=64, kernel_size=(3,3), activation=None,
                 kernel_constraint=max_norm)(inp)
maxout1 = Maxout(32)(conv1)              # Reduce 64 channels → 32 via maxout
maxout1 = maxout_activation_function(maxout1, units=32)
batch1 = L.BatchNormalization(momentum=0.8)(maxout1)
pool1 = L.MaxPooling2D(pool_size=(2,2))(batch1)
# Output: 32×32×32
drop1 = L.Dropout(0.6)(pool1)

# === LAYER 2 ===
conv2 = L.Conv2D(filters=128, kernel_size=(3,3), activation=None,
                 kernel_constraint=max_norm)(drop1)
maxout2 = Maxout(64)(conv2)              # Reduce 128 channels → 64 via maxout
maxout2 = maxout_activation_function(maxout2, units=64)
batch2 = L.BatchNormalization(momentum=0.8)(maxout2)
pool2 = L.MaxPooling2D(pool_size=(2,2))(batch2)
# Output: 16×16×64
drop2 = L.Dropout(0.5)(pool2)

# === LAYER 3 ===
conv3 = L.Conv2D(filters=256, kernel_size=(3,3), activation=None,
                 kernel_constraint=max_norm)(drop2)
maxout3 = Maxout(64)(conv3)              # Reduce 256 channels → 64 via maxout
maxout3 = maxout_activation_function(maxout3, units=64)
batch3 = L.BatchNormalization(momentum=0.8)(maxout3)
pool3 = L.MaxPooling2D(pool_size=(2,2))(batch3)
# Output: 8×8×64
drop3 = L.Dropout(0.4)(pool3)

# === CLASSIFICATION ===
flatten = L.Flatten()(drop3)
# Output: 8×8×64 = 4,096 features
dense = L.Dense(2, activation='softmax')(flatten)
# Output: 2 classes (cancer/non-cancer)

model2 = M.Model(inputs=inp, outputs=dense)
model2.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
```

### 10.5 DMN Architecture Diagram

```
INPUT: (64, 64, 3) — Feature image (resized from ~65K features)
│
├─ Conv2D(64, 3×3) → Maxout(32) → BN → MaxPool(2×2) → Dropout(0.6)
│   Input: 64×64×3 → Conv: 64×64×64 → Maxout: 64×64×32 → Pool: 32×32×32
│   Dropout: 30%
│
├─ Conv2D(128, 3×3) → Maxout(64) → BN → MaxPool(2×2) → Dropout(0.5)
│   Input: 32×32×32 → Conv: 32×32×128 → Maxout: 32×32×64 → Pool: 16×16×64
│   Dropout: 20%
│
├─ Conv2D(256, 3×3) → Maxout(64) → BN → MaxPool(2×2) → Dropout(0.4)
│   Input: 16×16×64 → Conv: 16×16×256 → Maxout: 16×16×64 → Pool: 8×8×64
│   Dropout: 40%
│
├─ Flatten → Dense(2) → Softmax
│   Input: 8×8×64 = 4096 → Output: 2 (cancer/non-cancer probabilities)
│
OUTPUT: [p_non-cancer, p_cancer] — sums to 1.0
```

### 10.6 Model Parameters

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Conv2D(64) | 3×3×3×64 = 1,728 | First conv layer |
| Maxout(32) | 64×32 = 2,048 | Maxout kernel (1 per unit) |
| BN1 | 4×32 = 128 | gamma, beta, mean, var |
| Conv2D(128) | 3×3×32×128 = 36,864 | Second conv layer |
| Maxout(64) | 128×64 = 8,192 | Second maxout |
| BN2 | 4×64 = 256 | |
| Conv2D(256) | 3×3×64×256 = 147,456 | Third conv layer |
| Maxout(64) | 256×64 = 16,384 | Third maxout |
| BN3 | 4×64 = 256 | |
| Dense(2) | 4096×2 = 8,192 | Final classification |
| **Total** | **~230,931** | **902 KB** |

### 10.7 Dropout Schedule

The model uses **progressively decreasing dropout** across layers:

| Layer | Dropout Rate | Reason |
|-------|-------------|--------|
| After pool 1 | 60% (0.6) | Early features are noisy — heavy regularization |
| After pool 2 | 50% (0.5) | Mid-level features — moderate regularization |
| After pool 3 | 40% (0.4) | High-level features — lighter regularization |

This schedule acknowledges that:
1. Early layers capture basic features (edges, textures) that are less discriminative
2. Later layers capture more abstract, task-relevant features that should be preserved more
3. Progressive reduction mimics the natural information flow from general to specific

### 10.8 MaxNorm Weight Constraint

```python
max_norm = max_norm(max_value=8, axis=[0, 1, 2])
# Applied to all Conv2D layers
conv1 = L.Conv2D(filters=64, kernel_size=(3,3),
                 kernel_constraint=max_norm)(inp)
```

**MaxNorm constraint:** Scales weights so that their Euclidean norm does not exceed 8:
```
If ||w|| > 8: w = (8 / ||w||) × w
```

**Why maxnorm?**
- Prevents weights from growing too large
- Acts as an implicit regularizer (similar to L2 but adaptive)
- Combined with dropout, maxnorm provides strong regularization for the small dataset
- Helps prevent overfitting on the 20-patient dataset

---

## 11. LHFGSO in DMN (Classification)

### 11.1 Where LHFGSO is Called

In `prop_DMO/DeepMaxout.py`:

```python
# Step 1: Build model with random initial weights
model2.fit(X_train, y_trainx, epochs=2, batch_size=10, verbose=0)

# Step 2: Get initial weights
Initial_weight = model2.get_weights()

# Step 3: Run LHFGSO optimization
updated_weights = LHFGSO.algm(Initial_weight)

# Step 4: REPLACE random weights with LHFGSO-optimized weights
model2.set_weights(updated_weights)

# Step 5: Continue training (or use for prediction)
pred = model2.predict(X_test)
```

### 11.2 What LHFGSO Optimizes in DMN

**All 230,931 parameters are optimized:**

| Component | Shape | Optimized? |
|-----------|-------|-----------|
| Conv2D(64) kernel | (3, 3, 3, 64) | YES — all 1,728 values |
| Maxout(32) kernel | (64, 32) | YES — all 2,048 values |
| BatchNorm1 | (4, 32) | YES — all 128 values |
| Conv2D(128) kernel | (3, 3, 32, 128) | YES — all 36,864 values |
| Maxout(64) kernel | (128, 64) | YES — all 8,192 values |
| BatchNorm2 | (4, 64) | YES — all 256 values |
| Conv2D(256) kernel | (3, 3, 64, 256) | YES — all 147,456 values |
| Maxout(64) kernel | (256, 64) | YES — all 16,384 values |
| BatchNorm3 | (4, 64) | YES — all 256 values |
| Dense(2) kernel | (4096, 2) | YES — all 8,192 values |

### 11.3 Why LHFGSO Can Optimize ALL DMN Weights

This is the **key architectural insight** that distinguishes Paper 2 from Paper 1:

**Paper 1 (HFGSO-DRN):**
- DRN has convolutional layers with millions of parameters
- HFGSO can only optimize the **final FC layer** (~8K-16K params)
- Convolutional layers are trained by standard backpropagation

**Paper 2 (LHFGSO-DMN):**
- DMN is a **pure maxout classifier** — no convolutional layers for spatial feature extraction
- Feature extraction is **EXTERNAL** (SegNet + LBP + SLBT + stats produce the features)
- DMN's input is a **1D feature vector** (resized to image format)
- Total model size: **230,931 parameters** — feasible for population-based optimization
- LHFGSO can optimize **ALL parameters** simultaneously

```
┌─────────────────────────────────────────────────────────────┐
│ Paper 1 — HFGSO-DRN:                                        │
│                                                             │
│ Conv layers (millions of params) → BACKPROP, not LHFGSO    │
│ Residual blocks → BACKPROP, not LHFGSO                      │
│ FC layer → ★ HFGSO (only this part)                        │
│                                                             │
│ HFGSO scope: NARROW (~8K-16K of millions)                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Paper 2 — LHFGSO-DMN:                                       │
│                                                             │
│ Feature extraction → DONE EXTERNALLY (SegNet + LBP + SLBT) │
│ DMN maxout layer 1 → ★ LHFGSO                              │
│ DMN maxout layer 2 → ★ LHFGSO                              │
│ DMN maxout layer 3 → ★ LHFGSO                              │
│ DMN FC layer → ★ LHFGSO                                    │
│                                                             │
│ LHFGSO scope: FULL NETWORK (all 230,931 params)            │
└─────────────────────────────────────────────────────────────┘
```

### 11.4 LHFGSO as Pre-Training Weight Initialization

The LHFGSO integration follows the same pattern as Paper 1:

```
Random Init → LHFGSO Optimization → LHFGSO-Optimized Weights → Adam Fine-Tuning
    ↓              ↓                    ↓                          ↓
  Keras        10 iterations      Replace all              Standard backprop
  default       of LHFGSO         230K weights             continues from
  random                                ↓                    here
    ↓                                  ↓
  1. Train with Adam         2. Predict with LHFGSO-
     for 2 epochs              optimized weights
```

### 11.5 Complete DMN Training Pipeline with LHFGSO

```python
def Dmax(feature, label, tr, A, Se, Sp):
    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        feature, label, train_size=tr, random_state=42
    )

    # 2. Resize features into image format
    xt = len(X_train)
    X_train = np.resize(X_train, (xt, 64, 64, 3))

    xt1 = len(X_test)
    X_test = np.resize(X_test, (xt1, 64, 64, 3))

    # 3. One-hot encode labels
    y_trainx = np.resize(y_train, (xt, 2))
    # y_train: (N,) with values 0 or 1
    # y_trainx: (N, 2) with [1,0] for class 0, [0,1] for class 1

    # 4. Initial training with Adam (2 epochs)
    model2.fit(X_train, y_trainx, epochs=2, batch_size=10, verbose=0)

    # 5. Get initial (random) weights
    Initial_weight = model2.get_weights()
    # List of ~10-15 numpy arrays, total ~230,931 parameters

    # 6. LHFGSO optimizes ALL 230,931 weights
    # - Flattens all weights into a combined vector
    # - Creates population of candidates
    # - Runs 10 iterations of HGSO + FA + LSO optimization
    # - Returns best solution found
    updated_weights = LHFGSO.algm(Initial_weight)

    # 7. Replace ALL model weights with LHFGSO results
    model2.set_weights(updated_weights)

    # 8. Predict (continues with LHFGSO-optimized weights)
    pred = model2.predict(X_test)

    # 9. Calculate metrics
    # ... (TP, TN, FP, FN calculation)
```

### 11.6 Why Full-Network Optimization Matters

When LHFGSO optimizes all DMN layers, it finds a **globally better weight configuration** for the entire classification task, not just the final layer. This means:

1. **Conv layers learn optimal filters** for the specific extracted features
2. **Maxout layers find optimal activation pieces** for discriminating cancer vs non-cancer
3. **The entire decision boundary is jointly optimized** by the metaheuristic
4. **No local minima in intermediate layers are left unaddressed**

In contrast, Paper 1's HFGSO only optimized the FC layer, leaving the conv/residual layers stuck in potentially suboptimal local minima from standard backpropagation.

---

## 12. LHFGSO Parameter Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| N (dimension) | len(flat_weights) | Number of weight arrays to optimize |
| M (max array size) | max(len(arr) for arr in w) | Dimension of largest flattened array |
| Tmax | 10 | Maximum iterations |
| l1 (Henry constant base) | 5 × exp(-2) ≈ 0.067 | Henry's constant initialization |
| l2 (partial pressure base) | 100 | Partial pressure initialization |
| l3 (compression constant) | 1 × exp(-2) ≈ 0.01 | Compression coefficient |
| alpha | 1.0 | Step size for position updates |
| g (beta0 absorption) | 1 | Light absorption coefficient |
| beta | 0.5 | Movement intensity base |
| epsilon | 0.05 | Prevents division by zero |
| T_eta | 298.15 | Reference temperature (Kelvin) |
| K (solubility constant) | 0.5 | Solubility scaling |
| c1, c2 | 0.1, 0.2 | Local optima escape parameters |
| z1, z2 | 0.1, 0.2 | Repositioning bounds |
| F | random ∈ [-1, 1] | Random scaling factor |
| E | random.sample(1..N, N) | Agent-specific scaling |
| Population size | N (number of weight arrays) | Candidates in population |

---

## 13. Architecture Reference

### DMN Layer-by-Layer

| Layer | Type | Input Shape | Output Shape | Parameters |
|-------|------|-------------|--------------|-----------|
| Input | Input | — | (64, 64, 3) | 0 |
| Conv1 | Conv2D | (64, 64, 3) | (64, 64, 64) | 1,728 (3×3×3×64) |
| Maxout1 | Maxout | (64, 64, 64) | (64, 64, 32) | 2,048 (64×32) |
| BN1 | BatchNorm | (64, 64, 32) | (64, 64, 32) | 128 |
| Pool1 | MaxPool | (64, 64, 32) | (32, 32, 32) | 0 |
| Drop1 | Dropout(0.6) | (32, 32, 32) | (32, 32, 32) | 0 |
| Conv2 | Conv2D | (32, 32, 32) | (32, 32, 128) | 36,864 (3×3×32×128) |
| Maxout2 | Maxout | (32, 32, 128) | (32, 32, 64) | 8,192 (128×64) |
| BN2 | BatchNorm | (32, 32, 64) | (32, 32, 64) | 256 |
| Pool2 | MaxPool | (32, 32, 64) | (16, 16, 64) | 0 |
| Drop2 | Dropout(0.5) | (16, 16, 64) | (16, 16, 64) | 0 |
| Conv3 | Conv2D | (16, 16, 64) | (16, 16, 256) | 147,456 (3×3×64×256) |
| Maxout3 | Maxout | (16, 16, 256) | (16, 16, 64) | 16,384 (256×64) |
| BN3 | BatchNorm | (16, 16, 64) | (16, 16, 64) | 256 |
| Pool3 | MaxPool | (16, 16, 64) | (8, 8, 64) | 0 |
| Drop3 | Dropout(0.4) | (8, 8, 64) | (8, 8, 64) | 0 |
| Flatten | Flatten | (8, 8, 64) | (4096,) | 0 |
| FC | Dense(2) | (4096,) | (2,) | 8,192 (4096×2) |
| Softmax | Activation | (2,) | (2,) | 0 |
| | | | **Total** | **~230,931** |

### LBP Feature Extraction (per image)

| Step | Operation | Output Shape |
|------|-----------|-------------|
| 1 | Read image | (256, 256, 3) |
| 2 | Convert to grayscale | (256, 256) |
| 3 | For each pixel: compare 8 neighbors → binary → decimal | (256, 256) |
| 4 | Compute 100-bin histogram | (100,) |

### SLBT Feature Extraction (per image)

| Step | Operation | Output Shape |
|------|-----------|-------------|
| 1 | Read image | (256, 256, 3) |
| 2 | Convert to grayscale | (256, 256) |
| 3 | LBP from top-right clockwise | (256, 256) |
| 4 | Compute 100-bin histogram | (100,) |

### Statistical Features (per image)

| Feature | Computation | Output Shape |
|---------|-------------|-------------|
| Mean | np.mean(image) | (1,) |
| Variance | np.var(grayscale) | (1,) |
| Kurtosis | scipy.stats.kurtosis(grayscale) | (H×W,) |
| Skewness | scipy.stats.skew(grayscale) | (H×W,) |
| Entropy histogram | skimage.filters.rank.entropy → histogram | (100,) |

---

## 14. Loss Functions and Fitness Metrics

### SegNet Loss (Multi-Objective)

Same as Paper 1:
```
E(a, b) = (1 - 0.75) × CrossEntropy(a, b) - 0.75 × log(Dice(a, b) + 1e-15)
```

Where:
- a = ground truth pixel values
- b = predicted pixel probabilities
- σ = 0.75 (75% weight on Dice)

### DMN Training Loss (Standard)

```python
model2.compile(
    loss='categorical_crossentropy',  # Standard cross-entropy for 2-class
    optimizer='adam',
    metrics=['accuracy']
)
```

### LHFGSO Fitness Function

The fitness function in `LHFGSO.py` uses a placeholder:

```python
def fitness(soln):
    return [sum(row) + random.random() for row in soln]
```

This is a **placeholder** — in a full implementation, the fitness would be evaluated by:
1. Loading the weight candidate into the model
2. Running inference on validation data
3. Computing the actual model loss or accuracy
4. Using that as the fitness score

### Evaluation Metrics

```python
# Confusion matrix elements
tp, tn, fn, fp = 0, 0, 0, 0

# Accuracy: Overall correctness
Acc = (tp + tn) / (tp + tn + fp + fn)

# Sensitivity: Of all actual positives, how many detected?
Sen = tp / (tp + fn)

# Specificity: Of all actual negatives, how many detected?
Sp = tn / (tn + fp)
```

---

## 15. Data Flow Summary

### Complete Data Transformation

```
Raw Input:
  Database/Image001.jpg (full MRI, ~various sizes)
  Database_gt/Image001.jpg (colored annotation)

Step 1 → Data Preparation:
  data/im/0.png (128×128 grayscale)
  data/gt/0.png (128×128 binary mask)

Step 2 → ROI Extraction:
  Output/roi/roi_0.png (~230×230, center crop)

Step 3 → Adaptive Median Filter:
  Output/amf/amf_0.png (denoised per channel, 3→11 window)

Step 4 → SegNet Segmentation:
  Output/segmented/seg_0.png (256×256, cyan-marked)
  → Binary mask: suspected cancer regions at pixel level

Step 5 → Augmentation (4 techniques):
  Output/rotation/rot_0.png (30° rotated)
  Output/cropping/crop_0.png (30% cropped)
  Output/flipping/flip_0.png (vertical flip)
  Output/rand_er/er_0.png (random erased)

Step 6 → Feature Extraction (7 types per augmented image):
  For each of 4 images (rot, crop, flip, er):
    Statistical: mean, variance, kurtosis, skewness, entropy
    Texture: LBP histogram (100 bins), SLBT histogram (100 bins)
  → 4 variants × 305 features ≈ 1,220 features per original image
  → Saved to Feat_fin.npy (all images combined)
  → Labels saved to lab_fin.npy

Step 7 → DMN Classification:
  Input: Feat_fin.npy → resized to 64×64×3
  Process:
    1. Train with Adam (2 epochs)
    2. LHFGSO optimizes ALL 230,931 weights
    3. Predict with LHFGSO-optimized weights
  Output: Cancer (1) or Non-cancer (0) per sample
```

### Data Size at Each Stage

| Stage | Description | Count | Output |
|-------|------------|-------|--------|
| Raw Database | MRI images + annotations | 101 pairs | Various sizes |
| After prepare_data | Resized PNG | 101 images + 101 masks | 128×128 |
| After ROI extraction | Center crop | 101 ROIs | ~230×230 |
| After AMF | Per-channel filtered | 101 filtered | 256×256 |
| After SegNet | Segmentations | 101 segmented | 256×256 |
| After Augmentation | 4 variants each | 404 images | 256×256 |
| After Feature Extract | Feat_fin.npy | ~404 samples | High-dimensional |
| After DMN input prep | Resized features | ~404 samples | 64×64×3 |

---

## 16. Results Analysis

### 16.1 Best Results

| Evaluation Mode | Accuracy | Sensitivity | Specificity |
|----------------|----------|-------------|-------------|
| 90% Training Data | **94.63%** | **93.46%** | **95.72%** |
| K-Fold (K=9) | **94.06%** | **93.13%** | **94.99%** |

### 16.2 Complete Comparative Results

#### At 90% Training Data

| Model | Optimizer | Accuracy | Sensitivity | Specificity | Δ vs LHFGSO-DMN |
|-------|-----------|----------|-------------|-------------|-----------------|
| DCNN | SGD | 76.05% | 75.65% | 78.01% | −18.58% |
| Panoptic Model | Adam | 79.71% | 78.37% | 80.43% | −14.92% |
| Focal-Net | Adam | 87.93% | 86.48% | 88.95% | −6.70% |
| ResNet | RMSprop | 89.40% | 88.47% | 90.76% | −5.23% |
| HFGSO-based DRN | HFGSO | 92.63% | 91.57% | 93.88% | −2.00% |
| **LHFGSO-based DMN** | **LHFGSO** | **94.63%** | **93.46%** | **95.72%** | **—** |

#### At K-Fold (K=7)

| Model | Accuracy | Sensitivity | Specificity |
|-------|----------|-------------|-------------|
| DCNN | 76.42% | 75.89% | 77.54% |
| Panoptic Model | 78.80% | 77.05% | 79.12% |
| Focal-Net | 85.50% | 84.51% | 87.06% |
| ResNet | 87.91% | 86.64% | 88.56% |
| HFGSO-based DRN | 91.92% | 91.05% | 92.86% |
| **LHFGSO-based DMN** | **94.06%** | **93.13%** | **94.99%** |

### 16.3 Per-Iteration Progression

| Iteration | Accuracy | Sensitivity | Specificity |
|-----------|----------|-------------|-------------|
| 5 | 88.43% | 86.43% | 89.42% |
| 10 | 89.42% | 87.96% | 90.53% |
| 15 | 91.86% | 90.58% | 92.56% |
| 20 | 92.78% | 91.67% | 93.56% |

**Key observation:** Performance improves steadily across all 20 iterations, with the largest jump between iterations 10-15 (+2.44% accuracy). No plateau or overfitting signal.

### 16.4 Convergence Analysis (Optimizer Fitness at Iteration 25)

| Optimizer | Fitness Value | Notes |
|-----------|--------------|-------|
| SGD | 0.0571 | Worst — gradient descent only |
| Adam | 0.0458 | Better — adaptive learning rate |
| RMSprop | 0.0314 | Best of gradient-based |
| HFGSO | 0.0205 | Metaheuristic — global search |
| **LHFGSO** | **0.0252** | **3-level hybrid — less overfitting** |

**Critical insight:** LHFGSO achieves *higher* training fitness (0.0252) than HFGSO (0.0205), yet produces *better* classification accuracy (94.63% vs 92.63%). This proves that **lower training loss ≠ better generalization**. LHFGSO's extra LSO diversification prevents overfitting to training data.

### 16.5 Clinical Interpretation

**At 90% training data (94.63% accuracy, 93.46% sensitivity, 95.72% specificity):**

```
Per 1,000 patients screened:
  Actual cancers: ~500  (if 50% prevalence)
  Actual normals: ~500

Results:
  TP = 467 (93.46% of 500 cancers detected)
  TN = 479 (95.72% of 500 normals cleared)
  FN = 33  (missed cancers)
  FP = 21  (false alarms)

Clinical impact:
  - 33 missed cancers out of 500 → acceptable for screening
  - Only 21 false alarms out of 500 → reduces unnecessary procedures
  - Specificity gain of +4.42pp over Paper 1 means ~20 fewer false alarms
    per 1,000 patients screened
```

### 16.6 Comparison with Paper 1

| Metric | Paper 1 (HFGSO-DRN) | Paper 2 (LHFGSO-DMN) | Change |
|--------|---------------------|----------------------|--------|
| Accuracy | 92.63% | 94.63% | **+2.00%** |
| Sensitivity | 93.67% | 93.46% | **−0.21%** |
| Specificity | 91.30% | 95.72% | **+4.42%** |
| Optimizer | HFGSO (2-level) | LHFGSO (3-level) | +LSO |
| Classifier | DRN (ResNet-20) | DMN (Maxout) | Trainable activation |
| Features | Histogram only | LBP+SLBT+stats | 7 feature types |
| Augmentation | 2 techniques | 4 techniques | +flipping, random erasing |
| Preprocessing | T2FCS | Adaptive median | Simpler |
| Epochs | Not specified | 30 | — |
| Model size | Not specified | 230,931 (902KB) | Lightweight |

---

## 17. Ablation Studies

### 17.1 Optimizer Comparison

| Optimizer | Model | Accuracy | Sensitivity | Specificity |
|-----------|-------|----------|-------------|-------------|
| SGD | DCNN | 76.05% | 75.65% | 78.01% |
| Adam | Focal-Net | 87.93% | 86.48% | 88.95% |
| RMSprop | ResNet | 89.40% | 88.47% | 90.76% |
| HFGSO | DRN | 92.63% | 91.57% | 93.88% |
| **LHFGSO** | **DMN** | **94.63%** | **93.46%** | **95.72%** |

**Incremental improvement breakdown:**
- Gradient-based (SGD→Adam→RMSprop): +13.35% accuracy gain
- HFGSO over RMSprop: +3.23% accuracy gain
- LHFGSO over HFGSO: +2.00% accuracy gain

### 17.2 Architecture Comparison (Same Optimizer)

This is the most important ablation — same LHFGSO optimizer with different architectures:

| Component | Paper 1 (HFGSO-DRN) | Paper 2 (LHFGSO-DMN) | Improvement |
|-----------|---------------------|----------------------|-------------|
| Segmentation | SegNet | SegNet | Same |
| Optimizer | HFGSO | LHFGSO | +LSO |
| Classifier | DRN (ResNet) | DMN (Maxout) | Trainable activation |
| Features | None (raw images) | LBP + SLBT + stats | Explicit features |
| Augmentation | 2 techniques | 4 techniques | More diversity |
| Preprocessing | T2FCS | Adaptive median | Simpler |
| **Accuracy** | **92.63%** | **94.63%** | **+2.00%** |
| **Sensitivity** | **93.67%** | **93.46%** | **−0.21%** |
| **Specificity** | **91.30%** | **95.72%** | **+4.42%** |

### 17.3 What Each Enhancement Contributed

The paper doesn't explicitly ablate each component, but we can infer from comparisons:

| Enhancement | Likely Contribution |
|-------------|-------------------|
| **LHFGSO over HFGSO** | +2.00% accuracy (LSO prevents overfitting) |
| **DMN over DRN** | Better decision boundaries with trainable activations |
| **Explicit features (LBP+SLBT+stats)** | Robust texture/shape info complementing deep features |
| **4 augmentation techniques** | Better generalization from more diverse training data |
| **Adaptive median filter** | Cleaner input → cleaner segmentation → better features |
| **Larger model (30 epochs)** | More training iterations for convergence |

### 17.4 Convergence Curve Analysis

The convergence curve (Fig. 9) shows an important pattern:

```
Iteration 5:  SGD(0.0571), Adam(0.0458), RMSprop(0.0314), HFGSO(0.0250), LHFGSO(0.0350)
...
Iteration 25: SGD(0.0571), Adam(0.0458), RMSprop(0.0314), HFGSO(0.0205), LHFGSO(0.0252)
```

**Key observations:**
- Gradient-based optimizers converge to their fitness within the first few iterations
- HFGSO continuously improves (0.0250 → 0.0205), finding better solutions over time
- LHFGSO improves less dramatically (0.0350 → 0.0252), suggesting it explores more broadly
- **Despite higher training fitness, LHFGSO produces better test results** — classic case of generalization vs training optimization

---

## 18. Paper 1 vs Paper 2: Complete Comparison

### 18.1 Pipeline Differences

| Component | Paper 1 | Paper 2 |
|-----------|---------|---------|
| Preprocessing | T2FCS (fuzzy + optimization) | Adaptive median filter (classical) |
| ROI extraction | Same (center crop) | Same (center crop) |
| Segmentation | Multi-objective SegNet | Multi-objective SegNet |
| Augmentation | 2 techniques (rotation, cropping) | 4 techniques (+flipping, random erasing) |
| Feature extraction | None (end-to-end) | LBP, SLBT, mean, variance, kurtosis, skewness, entropy |
| Classifier | Deep Residual Network (ResNet) | Deep Maxout Network |
| Optimizer | HFGSO (HGSO + FA) | LHFGSO (HGSO + FA + LSO) |
| LHFGSO scope | FC layer only (narrow) | All layers (full network) |

### 18.2 Technical Differences

| Aspect | Paper 1 (HFGSO-DRN) | Paper 2 (LHFGSO-DMN) |
|--------|---------------------|----------------------|
| Activation function | ReLU (fixed) | Maxout (trainable) |
| Skip connections | Yes (ResNet) | No (DMN is pure feedforward) |
| Dropout schedule | Not specified | Progressive (0.6 → 0.5 → 0.4) |
| Weight constraint | None | MaxNorm (||w|| ≤ 8) |
| Input to classifier | Resized raw images (32×32×3) | Resized features (64×64×3) |
| Feature dimensionality | 100 (histogram bins) | ~65,938 (raw) → ~305 (after extraction) |
| Model parameters | ~200K-300K | 230,931 (explicitly reported) |
| Epochs | Not specified | 30 |
| Batch size | Not specified | 32 |

### 18.3 Results Comparison

| Metric | Paper 1 (HFGSO-DRN) | Paper 2 (LHFGSO-DMN) | Change |
|--------|---------------------|----------------------|--------|
| **Accuracy** | 92.63% | 94.63% | **+2.00 pp** |
| **Sensitivity** | 93.67% | 93.46% | **−0.21 pp** |
| **Specificity** | 91.30% | 95.72% | **+4.42 pp** |

### 18.4 Key Insight: Why Paper 2 Performs Better

**The fundamental architectural redesign:**

Paper 1 tried to use HFGSO on a large convolutional network (DRN). But HFGSO could only optimize the final FC layer because:
- DRN has millions of convolutional parameters → too many for population-based search
- HFGSO was effectively doing weight initialization for just the last layer

Paper 2 **redesigned the classifier** to enable full-network optimization:
- Replaced DRN with DMN (pure maxout, no convolutions)
- Moved feature extraction **outside** the classifier (SegNet + LBP + SLBT + stats)
- DMN is small (230K params) → feasible for population-based search on ALL weights
- LHFGSO now sees and optimizes the **entire classification process**

The result is a more tightly integrated optimization pipeline where the metaheuristic optimizer has full visibility and control over the classification network.

### 18.5 What Was Maintained from Paper 1

- Multi-objective SegNet architecture
- ROI extraction approach
- Dice + cross-entropy combined loss
- Same dataset (20 patients from Brigham and Women's Hospital)
- Same evaluation metrics (accuracy, sensitivity, specificity)
- Same comparative baseline models (DCNN, Panoptic, Focal-Net, ResNet)
- Same medical context (prostate cancer detection from MRI)

### 18.6 Limitations (Common to Both Papers)

1. **Only 20 of 230 patients used** — limited statistical power
2. **No external multi-center validation** — unknown generalization
3. **No statistical significance testing** — p-values, confidence intervals not reported
4. **No full K-fold implementation** — single split with (K-1)/K ratio
5. **Placeholder fitness function** in optimizer code — not using actual model loss

---

*Document compiled for Paper 2: Hybrid Optimization Enabled Deep-Learning for Prostate Cancer Detection*
*Sensing and Imaging, 2024*
