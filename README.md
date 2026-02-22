# Entangled Filament Detection

Exploring methodologies for detecting and annotating highly entangled filamentous instances in images with partial annotations.

## Project Overview

This project investigates different approaches for detecting thin, continuous filamentous structures (like neurons, fibers, or roads) that exhibit:
- Crossings and occlusions
- Partial annotations (non-exhaustive centerline annotations)
- No reliable negative background

## Key Constraints

- Never penalize for failing to detect unannotated filaments
- Work with polyline/centerline annotations
- Handle variable filament thickness and blur

## Installation

```bash
# Clone the repository
git clone https://github.com/5TuX/entangled-filament-detection.git
cd entangled-filament-detection

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -e .
```

## Quick Start

### Generate Synthetic Data

```python
from synthetic.generator import FilamentGenerator, GeneratorConfig

config = GeneratorConfig(
    image_size=(512, 512),
    num_filaments=(50, 100),
    blur_sigma_range=(0.5, 2.0),
    convergence_zones=(2, 5),
    seed=42
)

generator = FilamentGenerator(config)
result = generator.generate_image_with_annotations()

# Access generated data
image = result['image']
annotated_polylines = result['annotated_polylines']
unannotated_polylines = result['unannotated_polylines']
```

### Run Visualizer

```bash
streamlit run notebooks/visualizer_app.py
```

## Datasets

### Identified Public Datasets

| Dataset | Domain | Annotation Type |
|---------|--------|----------------|
| FISBe (CVPR 2024) | Neuron microscopy | Instance masks |
| CREMI | EM neurites | Instance IDs |
| DeepGlobe Roads | Satellite imagery | Centerlines |
| Fluorescent Neuronal Cells v2 | Microscopy | Polylines, masks |

### Synthetic Data Generator

The `FilamentGenerator` class creates realistic entangled filament images with:
- Configurable blur and noise
- Convergence zones (high-density areas)
- Partial annotation simulation
- Variable filament thickness

## Methodologies

1. **Direct Polyline Regression** - CNN + sequential point prediction
2. **DETR-like Set Prediction** - Transformer with learnable queries
3. **Segmentation + Embedding** - Shared encoder with clustering
4. **PU Learning** - Positive-Unlabeled learning for partial annotations

## Project Structure

```
entangled-filament-detection/
├── data/
│   ├── synthetic/           # Generated synthetic data
│   └── fisbe/              # Downloaded FISBe data
├── src/
│   ├── models/             # Model implementations
│   ├── losses/             # Custom loss functions
│   └── utils/              # Utilities
├── synthetic/
│   └── generator.py        # Synthetic data generator
├── notebooks/
│   └── visualizer_app.py   # Streamlit visualizer
├── configs/                 # Configuration files
└── pyproject.toml          # Dependencies
```

## Evaluation Metrics

- **Annotated Recall** - Detection rate on annotated filaments
- **Geometric Accuracy** - Chamfer/Fréchet distance
- **Structural Error** - Miss rate, fragmentation

## License

MIT License

## References

- FISBe: A Real-World Benchmark Dataset for Instance Segmentation of Long-Range Thin Filamentous Structures (CVPR 2024)
- TREXplorer: Recurrent DETR for Topologically Correct Tree Centerline Tracking (MICCAI 2024)
- PU Learning surveys and selective labeling methods
