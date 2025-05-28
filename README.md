# Deep Learning Architectures: Autoencoders, Transformers & LLM Apps

A collection of end‑to‑end experiments demonstrating how to build, train, evaluate and deploy a variety of deep learning models—from custom autoencoders for anomaly detection, to transformers and pretrained LLMs for NLP, to CNNs and VGG‑13 for image classification.

---

## Part I: Tabular Binary Classification

### Dataset Overview
- **Samples:** 766 → cleaned to 760  
- **Features (f1–f7):** numeric (integers & floats)  
- **Target:** binary (0/1), imbalance (~65% zeros)  
- **Summary (mean ± std, min–max):**  
  - f1: 3.8 ± 3.5 (0–17)  
  - f2: 120.9 ± 32.7 (44–199)  
  - …  
  - f7: 0.47 ± 0.34 (0.08–2.42)  

### Data Preprocessing
1. **Invalid Character Removal:** replaced [c, f, a, b, d, e] → NaN → dropped rows  
2. **Type Conversion:** cast object columns → numeric (errors='coerce') → drop new NaNs  
3. **Train/Validation/Test Split:**  
   - Train: 532 samples  
   - Val: 76 samples  
   - Test: 152 samples  
4. **Scaling:** StandardScaler (mean = 0, std = 1)

### Exploratory Visualization
- **Histograms:** skewed distributions for f1, f5; near‑normal for f3, f6, f7  
- **Correlation Matrix:** f2 most correlated with target (r = 0.46)  
- **Pairwise Plots:** weak linear relationships → motivates non‑linear models

### Neural Network Model
- **Architecture:**  
  - Input: 7 features  
  - Hidden 1: 64 ReLU → Dropout(0.5)  
  - Hidden 2: 64 ReLU → Dropout(0.5)  
  - Output: 1 Sigmoid  
- **Training:** 30 epochs, Adam (lr=0.001), BCE loss  
- **Results:**  
  - Test accuracy: 84.2%  
  - Precision: 0.76 Recall: 0.70 F1: 0.73  
  - AUC: 0.88  

---

## Part II: Hyperparameter Optimization

### Dropout Rate
| Rate | Test Accuracy |
|------|---------------|
| 0.0  | 80.3%         |
| 0.2  | 79.5%         |
| 0.3  | 78.8%         |

### Weight Initialization
| Method  | Test Accuracy |
|---------|---------------|
| Default | 80.3%         |
| Xavier  | 80.1%         |
| Kaiming | 82.1%         |

### Batch Size
| Size | Test Accuracy |
|------|---------------|
| 16   | 80.6%         |
| 32   | 78.9%         |
| 64   | 78.4%         |

### Depth (Hidden Layers)
| Layers | Test Accuracy |
|--------|---------------|
| 2      | 80.6%         |
| 3      | 82.3%         |
| 4      | 84.2%         |

### Improvements & Best Model
- **Batch Normalization** after each hidden layer  
- **Gradient Accumulation** for larger effective batch sizes  
- **Data Augmentation** (noise injection)  
- **K‑Fold Cross‑Validation**  
- **Best Model:** 4 hidden layers, Dropout=0.2, Kaiming init, Batch size=16 → **85.4%** test accuracy, F1 = 0.73

---

## Part III: CNN for Alphanumeric Classification

### Dataset
- **36 classes** (0–9, A–Z), 28×28 images, RGB or grayscale  
- **Samples:** 100 800 (2 800 per class)  
- **Split:** 80% train, 10% val, 10% test  

### Preprocessing
- Resize → 28×28, normalize (mean=0.5, std=0.5)  
- Augment: random rotations, horizontal flips  

### CharacterCNN Architecture
1. **Convs:** 3 blocks (Conv → BatchNorm → ReLU → MaxPool)  
2. **FC:** 128 ReLU → Dropout(0.5)  
3. **Output:** 36 units → LogSoftmax  

### Training & Metrics
- **Optimizer:** Adam (lr=0.001) + LR scheduler (halve every 5 epochs)  
- **Epochs:** 10, **Batch:** 64  
- **Test Accuracy:** 88.3%  
- **Precision/Recall/F1:** ≈ 88%  
- **AUC:** per-class 0.75–0.99  

---

## Part IV: VGG‑13 Baseline

- **5 convolutional blocks** (2×Conv 3×3 + ReLU + MaxPool)  
- **3 FC layers:** 4096 units each + Dropout(0.5)  
- **Comparison:**  
  - VGG‑13 has far more parameters and deeper conv stacks  
  - CharacterCNN is ~0.9 M parameters (≈ 27 MB) vs. VGG‑13’s >130 M  

---

## Contribution

| Team Member               | Parts                                 | %   |
|---------------------------|---------------------------------------|-----|
| Saroja Vuluvabeeti        | I, II, III, IV + Bonus                | 50% |
| Sri Sakthi Thirumagal     | I, II, III, IV + Bonus                | 50% |

---

## References
1. PyTorch Beginner Tutorial  
2. torch.nn.Module documentation  
3. pandas.read_csv reference  
4. scikit‑learn preprocessing & metrics  
5. PyTorch Ignite early stopping  
6. K‑Fold Cross Validation guide  
7. ROC & AUC concepts (Google ML Crash Course)  
