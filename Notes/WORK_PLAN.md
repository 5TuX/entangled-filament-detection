# Comprehensive Work Plan: Filamentous Instance Detection & Annotation

## Project Overview

**Objective**: Explore methodologies for detecting and annotating highly entangled filamentous instances in images with partial annotations.

**Constraints**:
- Thin, continuous structures with crossings and partial occlusions
- Partial annotations expressed as polylines (centerlines)
- Non-exhaustive annotations - no reliable negative background
- Never penalize for failing to detect unannotated filaments

---

## 1. IDENTIFIED DATASETS

### Primary: Biological/Microscopy Datasets

| Dataset | Description | Annotation Type | License | Relevance |
|---------|-------------|----------------|---------|-----------|
| **FISBe** (CVPR 2024) | FlyLight neuron instance segmentation in 3D light microscopy (~600 instances) | Instance segmentation masks | CC BY-SA 2.0 | **Highly relevant** - exactly thin filamentous neurons with crossings/occlusions |
| **CREMI** | EM images of Drosophila brain for circuit reconstruction | Neuron instance IDs, synaptic clefts | Public (Janelia) | Good for instance segmentation methods |
| **Fluorescent Neuronal Cells v2** | Multi-task, multi-format microscopy annotations | Polylines, masks, points | CC BY 4.0 | Has polyline centerline annotations |
| **SNEMI3D** | EM neurite segmentation | Instance masks | Public | Benchmark for neuron segmentation |

### Secondary: Geometric/Remote Sensing

| Dataset | Description | Annotation Type | License |
|---------|-------------|----------------|---------|
| **DeepGlobe Road Extraction** | Satellite road extraction | Binary masks + centerlines | Public |
| **SAM-Road++** (CVPR 2025) | Global road graph extraction | Polylines/graphs | CC BY 4.0 |
| **RoadNet** | Multi-task road detection | Road surface + centerline | Public |

### Synthetic Generation Resources

| Resource | Description |
|----------|-------------|
| **CS-Sim** (GitHub) | Synthetic curvilinear structures in 2D/3D |
| **SyntheticFiberGenerator** (UW Loci) | Fiber network generation |
| **Yarn-Fiber-Generation** | Yarn/fiber synthetic images |

---

## 2. PROPOSED METHODOLOGIES

### 2.1 Direct Polyline Regression
- **Approach**: CNN backbone + sequential point prediction (RNN/Transformer decoder)
- **References**: PolyLaneNet, PRNet, YOLinO
- **Adaptation for partial annotations**: 
  - Use only annotated points in loss computation
  - Hungarian matching on annotated segments only
- **Pros**: End-to-end, outputs polylines directly
- **Cons**: Struggles with branching

### 2.2 DETR-like Set Prediction
- **Approach**: Transformer with learnable queries predicting polylines/segments
- **References**: DETR, MaskFormer, TREXplorer
- **Adaptation**: 
  - Modify query design for filament proposals
  - Use soft-NMS variant to avoid over-suppression
- **Pros**: Handles variable number of instances, global context
- **Cons**: Requires careful query design for thin structures

### 2.3 Segmentation + Instance Embedding
- **Approach**: 
  - Shared encoder → two heads: (1) foreground segmentation, (2) embedding predictions
  - Embeddings clustered via DBSCAN or learned grouping
- **References**: DeepBranchTracer, embedding-based cell tracking
- **Adaptation**: 
  - Use partial loss - only compute embedding loss on annotated regions
  - Margin-based contrastive loss for separation
- **Pros**: Robust to crossings via embedding space
- **Cons**: Requires post-processing, embedding collapse issues

### 2.4 PU (Positive-Unlabeled) Learning
- **Approach**: 
  - Treat annotated filaments as positive (P)
  - Unannotated regions as unlabeled (U) - may contain missed filaments
  - Use PU-loss variants to avoid penalizing undetected filaments
- **References**: PU-learning surveys, selective labeling works
- **Adaptation**: 
  - Two-component model: classifier + density estimator
  - Penalize false positives on unlabeled regions more than false negatives
- **Pros**: Aligns with your constraint - never penalized for missing unannotated filaments
- **Cons**: Requires careful prior estimation

---

## 3. SYNTHETIC DATA GENERATION PIPELINE

### Core Algorithm: Entangled Spaghetti Generator

**Parameters**:
```python
config = {
    "image_size": (1024, 1024),
    "num_filaments": (50, 200),
    "num_control_points": (20, 100),
    "amplitude_range": (10, 100),
    "frequency_range": (0.01, 0.05),
    "width_range": (2, 8),
    "blur_sigma_range": (0.5, 3.0),      # Blur variance (KEY PARAMETER)
    "convergence_zones": (2, 5),
    "convergence_radius": (50, 150),
    "convergence_multiplier": (3, 6),
    "background_noise": (0, 50),
    "partial_annotation_ratio": (0.3, 0.7),
    "partial_length_ratio": (0.4, 1.0),
}
```

### Generation Steps
1. Control Point Sampling: Random walk with momentum
2. Convergence Zones: Place high-density regions
3. B-Spline Fitting: Interpolate to smooth polylines
4. Rendering: Draw thick lines with variable width
5. Blurring: Apply Gaussian blur with random sigma
6. Noise Injection: Poisson/Gaussian noise
7. Partial Annotation: Randomly mask portions

---

## 4. DETAILED EXPERIMENT PLAN

### Phase 1: Data Preparation (Week 1-2)
- 1.1: Download & explore FISBe dataset (2 days)
- 1.2: Preprocess: crop, normalize, convert annotations (3 days)
- 1.3: Create synthetic dataset generator (3 days)
- 1.4: Generate synthetic pilot set (500 images) (2 days)
- 1.5: Create data loaders for each format (2 days)

### Phase 2: Baseline Models (Week 3-5)
- 2.1: Implement polyline regression baseline (5 days)
- 2.2: Implement DETR-style filament detector (5 days)
- 2.3: Implement segmentation + embedding approach (5 days)
- 2.4: Implement PU-learning variant (5 days)

### Phase 3: Training & Evaluation (Week 6-9)
- 3.1: Train on synthetic data (rapid iteration) (5 days)
- 3.2: Evaluate with partial annotation metrics (3 days)
- 3.3: Fine-tune on FISBe real data (5 days)
- 3.4: Cross-domain transfer experiments (3 days)

### Phase 4: Analysis & Visualization (Week 10-11)
- 4.1: Generate comparison visualizations (3 days)
- 4.2: Metric analysis (per-method) (3 days)
- 4.3: Error analysis and ablation (3 days)
- 4.4: Compile final report (2 days)

---

## 5. EVALUATION METRICS

### For Partial Annotations (Primary)
| Metric | Description |
|--------|-------------|
| **Annotated Recall** | % of annotated filaments detected |
| **Unannotated Filament Tolerance** | Penalize only false positives on unannotated regions |
| **Geometric Accuracy** | Chamfer/Fréchet distance to annotated segments |
| **Structural Error** | Miss rate, fragmentation |

### Standard Metrics (Secondary)
| Metric | Description |
|--------|-------------|
| **Precision** | TP / (TP + FP) on annotated regions |
| **IoU** | For segmentation approaches |
| **AP** | Average precision |

---

## 6. PROJECT STRUCTURE

```
entangled-filament-detection/
├── data/
│   ├── fisbe/
│   ├── synthetic/
│   └── preprocessing.py
├── src/
│   ├── models/
│   ├── losses/
│   ├── utils/
│   └── train.py
├── synthetic/
│   └── generator.py
├── configs/
├── notebooks/
├── pyproject.toml
└── README.md
```

---

## 7. CRITICAL DECISIONS

1. **Annotation format**: Convert all to centerline polylines
2. **PU-learning prior**: Estimate fraction of unannotated filaments
3. **Metric priority**: Detecting ALL annotated filaments more important than precision
4. **Synthetic realism**: Match blur/noise to real data domain
