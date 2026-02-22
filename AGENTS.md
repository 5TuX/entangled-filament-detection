# Agent Instructions

## Development Commands

### Install dependencies
```bash
uv venv
uv pip install -e .
```

### Generate synthetic data
```bash
.venv/Scripts/python -c "
from synthetic.generator import FilamentGenerator, GeneratorConfig
from pathlib import Path
config = GeneratorConfig(image_size=(512, 512), num_filaments=(50, 100), seed=42)
generator = FilamentGenerator(config)
generator.generate_batch(10, Path('data/synthetic/batch_test'))
"
```

### Run visualizer
```bash
.venv/Scripts/streamlit run notebooks/visualizer_app.py
```

### Linting
```bash
uv pip install ruff
ruff check .
```

## Key Files

- `synthetic/generator.py` - Synthetic filament generation
- `notebooks/visualizer_app.py` - Streamlit web app
- `Notes/WORK_PLAN.md` - Detailed work plan

## GPU

- RTX 4080 SUPER with 16GB VRAM
- CUDA 12.1 compatible
