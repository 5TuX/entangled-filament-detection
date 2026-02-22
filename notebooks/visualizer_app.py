"""
Streamlit Web App for Filament Data Visualization

Visualizes:
- Synthetic generated filament images with annotations
- Downloaded real datasets (FISBe format)
- Comparison between ground truth and predictions
"""

import streamlit as st
import numpy as np
import cv2
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import asdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetic.generator import FilamentGenerator, GeneratorConfig


# Page configuration
st.set_page_config(
    page_title="Filament Detection Visualizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_synthetic_data(data_dir: Path) -> List[Dict]:
    """Load synthetic dataset from directory."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    
    annotations = []
    for ann_file in sorted(data_dir.glob("annotation_*.json")):
        with open(ann_file, 'r') as f:
            annotations.append(json.load(f))
    
    return annotations


def load_image_from_annotation(ann: Dict) -> np.ndarray:
    """Load image from annotation path."""
    img_path = Path(ann['image_path'])
    if img_path.exists():
        return cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    return None


def load_fisbe_sample(sample_dir: Path) -> Dict:
    """Load FISBe dataset sample (placeholder - adapt to actual format)."""
    # FISBe stores data as HDF5 or numpy arrays
    # This is a placeholder implementation
    return {}


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def draw_polylines(image: np.ndarray, polylines: List, 
                   color: Tuple[int, int, int] = (0, 255, 0),
                   width: int = 2, annotations: bool = False) -> np.ndarray:
    """Draw polylines on image."""
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    for polyline in polylines:
        pts = np.array(polyline, dtype=np.int32)
        cv2.polylines(vis_image, [pts], False, color, width, lineType=cv2.LINE_AA)
    
    return vis_image


def create_comparison_view(image: np.ndarray, 
                           annotated: List,
                           unannotated: List = None,
                           predictions: List = None) -> np.ndarray:
    """Create side-by-side or overlay comparison visualization."""
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    # Draw annotated (ground truth) in green
    vis = draw_polylines(vis, annotated, color=(0, 255, 0), width=2)
    
    # Draw unannotated in yellow (dashed effect - approximated with thinner line)
    if unannotated:
        vis = draw_polylines(vis, unannotated, color=(255, 255, 0), width=1)
    
    # Draw predictions in red
    if predictions:
        vis = draw_polylines(vis, predictions, color=(0, 0, 255), width=2)
    
    return vis


def plot_statistics(annotations: List[Dict]) -> Dict:
    """Compute and return statistics about the dataset."""
    if not annotations:
        return {}
    
    num_images = len(annotations)
    total_annotated = sum(len(a.get('annotated_polylines', [])) for a in annotations)
    total_unannotated = sum(len(a.get('unannotated_polylines', [])) for a in annotations)
    total_full = sum(len(a.get('full_polylines', [])) for a in annotations)
    
    # Compute polyline lengths
    all_lengths = []
    for ann in annotations:
        for poly in ann.get('annotated_polylines', []):
            lengths = np.sqrt(np.sum(np.diff(poly, axis=0)**2, axis=1))
            all_lengths.append(np.sum(lengths))
    
    return {
        'num_images': num_images,
        'total_filaments': total_full,
        'annotated_filaments': total_annotated,
        'unannotated_filaments': total_unannotated,
        'avg_filaments_per_image': total_full / num_images if num_images > 0 else 0,
        'avg_polyline_length': np.mean(all_lengths) if all_lengths else 0,
    }


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.title("🔬 Controls")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Generate Synthetic", "Load Synthetic Data", "Load FISBe"]
    )
    
    # Generation parameters (for synthetic)
    if data_source == "Generate Synthetic":
        st.sidebar.header("Generation Parameters")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            img_size = st.sidebar.slider("Image Size", 256, 2048, 512, step=64)
        with col2:
            num_filaments = st.sidebar.slider("Num Filaments", 10, 200, 50)
        
        col3, col4 = st.sidebar.columns(2)
        with col3:
            blur_sigma = st.sidebar.slider("Blur Sigma", 0.0, 5.0, 1.0)
        with col4:
            noise_level = st.sidebar.slider("Noise Level", 0, 100, 20)
        
        col5, col6 = st.sidebar.columns(2)
        with col5:
            num_zones = st.sidebar.slider("Convergence Zones", 0, 10, 3)
        with col6:
            partial_ratio = st.sidebar.slider("Partial Annotation %", 0.0, 1.0, 0.5)
        
        seed = st.sidebar.number_input("Random Seed", value=42)
        
        gen_config = GeneratorConfig(
            image_size=(img_size, img_size),
            num_filaments=(num_filaments, num_filaments + 10),
            blur_sigma_range=(blur_sigma, blur_sigma + 0.5),
            background_noise=(noise_level, noise_level),
            convergence_zones=(num_zones, num_zones + 1),
            partial_annotation_ratio=(partial_ratio, partial_ratio + 0.1),
            seed=seed
        )
        
        return data_source, {'config': gen_config, 'generated': None}
    
    elif data_source == "Load Synthetic Data":
        data_dir = st.sidebar.text_input("Data Directory", "data/synthetic")
        return data_source, {'data_dir': data_dir}
    
    else:  # FISBe
        fisbe_dir = st.sidebar.text_input("FISBe Directory", "data/fisbe")
        return data_source, {'data_dir': fisbe_dir}


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    st.title("🔬 Filament Detection Visualizer")
    st.markdown("""
    Explore and visualize filamentous structure datasets with partial annotations.
    This tool helps analyze data characteristics for developing detection algorithms.
    """)
    
    # Render sidebar and get selection
    data_source, params = render_sidebar()
    
    # Initialize session state
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'sample_index' not in st.session_state:
        st.session_state.sample_index = 0
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Data Viewer", "📈 Statistics", "⚙️ Generator Config"])
    
    # =========================================================================
    # TAB 1: DATA VIEWER
    # =========================================================================
    with tab1:
        col_main, col_controls = st.columns([4, 1])
        
        with col_controls:
            st.header("Navigation")
            
            if data_source == "Generate Synthetic":
                if st.button("🔄 Generate New", type="primary"):
                    with st.spinner("Generating..."):
                        generator = FilamentGenerator(params['config'])
                        result = generator.generate_image_with_annotations()
                        st.session_state.current_data = result
                        st.session_state.current_image = result['image']
                        st.session_state.sample_index = 0
                        st.rerun()
            
            elif data_source == "Load Synthetic Data":
                data_dir = Path(params.get('data_dir', 'data/synthetic'))
                if data_dir.exists():
                    annotations = load_synthetic_data(data_dir)
                    if annotations:
                        num_samples = len(annotations)
                        idx = st.slider("Sample Index", 0, num_samples - 1, 
                                       st.session_state.sample_index)
                        st.session_state.sample_index = idx
                        
                        ann = annotations[idx]
                        img = load_image_from_annotation(ann)
                        st.session_state.current_data = ann
                        st.session_state.current_image = img
                    else:
                        st.warning("No annotations found")
                else:
                    st.warning(f"Directory not found: {data_dir}")
            
            # Display options
            st.header("Display Options")
            show_annotated = st.checkbox("Show Annotated", value=True)
            show_unannotated = st.checkbox("Show Unannotated", value=True)
            show_predictions = st.checkbox("Show Predictions", value=False)
            
            # Prediction input (for future use)
            if show_predictions:
                pred_file = st.file_uploader("Upload Predictions (JSON)", 
                                           type=['json'])
    
    # =========================================================================
    # Render the visualization
    # =========================================================================
    
    with col_main:
        if st.session_state.current_data is not None:
            data = st.session_state.current_data
            image = st.session_state.current_image
            
            if image is not None and len(image) > 0:
                # Prepare polylines
                annotated = data.get('annotated_polylines', [])
                unannotated = data.get('unannotated_polylines', [])
                
                # Create visualization
                vis_image = image.copy()
                if len(vis_image.shape) == 2:
                    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
                
                if show_annotated:
                    vis_image = draw_polylines(vis_image, annotated, 
                                              color=(0, 255, 0), width=2)
                if show_unannotated:
                    vis_image = draw_polylines(vis_image, unannotated,
                                              color=(255, 255, 0), width=1)
                
                # Display
                st.image(vis_image, channels='BGR', use_container_width=True)
                
                # Info
                st.caption(f"Annotated filaments: {len(annotated)} | "
                          f"Unannotated filaments: {len(unannotated)}")
            else:
                st.warning("Could not load image")
        else:
            # Show demo/placeholder
            st.info("👈 Select a data source and generate/load data to begin")
            
            # Show sample generated image on startup
            if data_source == "Generate Synthetic":
                st.markdown("*Click 'Generate New' to create synthetic filament data*")
    
    # =========================================================================
    # TAB 2: STATISTICS
    # =========================================================================
    with tab2:
        st.header("Dataset Statistics")
        
        if data_source == "Load Synthetic Data" and st.session_state.current_data:
            data_dir = Path(params.get('data_dir', 'data/synthetic'))
            annotations = load_synthetic_data(data_dir)
            
            if annotations:
                stats = plot_statistics(annotations)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Images", stats.get('num_images', 0))
                with col2:
                    st.metric("Total Filaments", stats.get('total_filaments', 0))
                with col3:
                    st.metric("Annotated", stats.get('annotated_filaments', 0))
                with col4:
                    st.metric("Avg Length", f"{stats.get('avg_polyline_length', 0):.1f}")
                
                # Visualization
                st.subheader("Annotation Coverage")
                annotated_pct = (stats.get('annotated_filaments', 0) / 
                                 max(stats.get('total_filaments', 1), 1) * 100)
                unannotated_pct = (stats.get('unannotated_filaments', 0) / 
                                  max(stats.get('total_filaments', 1), 1) * 100)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(['Annotated', 'Unannotated'], 
                      [annotated_pct, unannotated_pct],
                      color=['green', 'yellow'])
                ax.set_ylabel('Percentage (%)')
                ax.set_ylim(0, 100)
                st.pyplot(fig)
        else:
            st.info("Load a dataset to see statistics")
    
    # =========================================================================
    # TAB 3: GENERATOR CONFIG
    # =========================================================================
    with tab3:
        st.header("Synthetic Generator Configuration")
        
        st.markdown("""
        The synthetic generator creates realistic images of entangled filamentous
        structures with configurable parameters:
        """)
        
        # Display current config
        if data_source == "Generate Synthetic":
            config = params['config']
            st.json(asdict(config))
            
            st.markdown("""
            ### Parameter Descriptions
            
            - **Image Size**: Output image dimensions (width, height)
            - **Num Filaments**: Range of filament count per image
            - **Blur Sigma**: Gaussian blur strength (higher = more blur)
            - **Noise Level**: Background Gaussian noise intensity
            - **Convergence Zones**: Areas where filaments cluster
            - **Partial Annotation %**: Fraction of filaments with incomplete annotations
            """)
        
        else:
            st.info("Select 'Generate Synthetic' to see configuration")


if __name__ == "__main__":
    main()
