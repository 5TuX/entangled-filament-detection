"""
Synthetic Entangled Filament Generator

Generates realistic images of entangled filamentous structures (like spaghetti)
with controllable parameters for blur, density, and partial annotations.
"""

import numpy as np
from scipy import interpolate, ndimage
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from pathlib import Path
import json
import cv2


@dataclass
class GeneratorConfig:
    """Configuration for synthetic filament generation."""
    image_size: Tuple[int, int] = (1024, 1024)
    num_filaments: Tuple[int, int] = (50, 200)
    num_control_points: Tuple[int, int] = (20, 100)
    amplitude_range: Tuple[float, float] = (10, 100)
    frequency_range: Tuple[float, float] = (0.01, 0.05)
    width_range: Tuple[float, float] = (2, 8)
    blur_sigma_range: Tuple[float, float] = (0.5, 3.0)
    convergence_zones: Tuple[int, int] = (2, 5)
    convergence_radius: Tuple[int, int] = (50, 150)
    convergence_multiplier: Tuple[float, float] = (3, 6)
    background_noise: Tuple[float, float] = (0, 50)
    partial_annotation_ratio: Tuple[float, float] = (0.3, 0.7)
    partial_length_ratio: Tuple[float, float] = (0.4, 1.0)
    seed: Optional[int] = None


class FilamentGenerator:
    """Generate synthetic entangled filament images."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
    
    def generate_control_points(self, num_points: int, 
                                 convergence_zones: List[Tuple[float, float, float]]
                                 ) -> np.ndarray:
        """Generate smooth control points using random walk with momentum."""
        points = np.zeros((num_points, 2))
        
        # Start from random position
        points[0] = np.random.uniform(0, self.config.image_size[1]), \
                    np.random.uniform(0, self.config.image_size[0])
        
        # Random walk with momentum
        velocity = np.zeros(2)
        amplitude = np.random.uniform(*self.config.amplitude_range)
        frequency = np.random.uniform(*self.config.frequency_range)
        
        for i in range(1, num_points):
            # Add influence from convergence zones
            zone_attraction = np.zeros(2)
            for cx, cy, strength in convergence_zones:
                to_zone = np.array([cx, cy]) - points[i-1]
                dist = np.linalg.norm(to_zone)
                if dist > 0:
                    zone_attraction += (to_zone / dist) * strength / (dist + 1)
            
            # Random direction with momentum
            noise = np.random.randn(2) * amplitude
            velocity = velocity * 0.8 + noise * 0.2 + zone_attraction * 0.3
            
            # Apply sinusoidal variation
            angle = np.arctan2(velocity[1], velocity[0])
            angle += np.sin(i * frequency) * 0.5
            
            step_size = np.random.uniform(5, 20)
            new_point = points[i-1] + np.array([np.cos(angle), np.sin(angle)]) * step_size
            
            # Keep within bounds
            new_point[0] = np.clip(new_point[0], 10, self.config.image_size[1] - 10)
            new_point[1] = np.clip(new_point[1], 10, self.config.image_size[0] - 10)
            
            points[i] = new_point
        
        return points
    
    def points_to_polyline(self, points: np.ndarray, num_samples: int = 200) -> np.ndarray:
        """Convert control points to smooth polyline using B-spline interpolation."""
        # Calculate cumulative distances
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
        
        # Create parameterization
        total_length = cumulative[-1]
        if total_length < 1:
            return points
        
        t = cumulative / total_length
        t_new = np.linspace(0, 1, num_samples)
        
        # B-spline interpolation
        try:
            spl = interpolate.splprep([points[:, 0], points[:, 1]], 
                                      k=min(3, len(points)-1), s=0.01)
            interpolated = interpolate.splev(t_new, spl[0])
            polyline = np.column_stack(interpolated)
        except:
            # Fallback to linear interpolation
            polyline = np.zeros((num_samples, 2))
            for i in range(num_samples):
                t_val = i / (num_samples - 1)
                idx = np.searchsorted(t, t_val)
                if idx >= len(t):
                    idx = len(t) - 1
                if idx == 0:
                    polyline[i] = points[0]
                else:
                    t0, t1 = t[idx-1], t[idx]
                    if t1 - t0 > 0:
                        alpha = (t_val - t0) / (t1 - t0)
                    else:
                        alpha = 0
                    polyline[i] = (1 - alpha) * points[idx-1] + alpha * points[idx]
        
        return polyline
    
    def create_convergence_zones(self) -> List[Tuple[float, float, float]]:
        """Create high-density convergence zones."""
        num_zones = np.random.randint(*self.config.convergence_zones)
        zones = []
        
        for _ in range(num_zones):
            x = np.random.uniform(0.2, 0.8) * self.config.image_size[1]
            y = np.random.uniform(0.2, 0.8) * self.config.image_size[0]
            strength = np.random.uniform(*self.config.convergence_multiplier)
            zones.append((x, y, strength))
        
        return zones
    
    def draw_filament(self, polyline: np.ndarray, width: float, 
                      canvas: np.ndarray) -> np.ndarray:
        """Draw a single filament on the canvas."""
        # Convert to integer coordinates
        pts = polyline.astype(np.int32)
        
        # Draw thick line
        cv2.polylines(canvas, [pts], False, 255, int(width), lineType=cv2.LINE_AA)
        
        return canvas
    
    def generate_image_with_annotations(self) -> Dict:
        """Generate a single image with annotations."""
        # Create convergence zones
        convergence_zones = self.create_convergence_zones()
        
        # Determine number of filaments
        num_filaments = np.random.randint(*self.config.num_filaments)
        
        # Create blank canvas
        canvas = np.zeros(self.config.image_size, dtype=np.float32)
        
        # Generate filaments
        polylines = []
        widths = []
        
        for _ in range(num_filaments):
            num_control = np.random.randint(*self.config.num_control_points)
            control_points = self.generate_control_points(num_control, convergence_zones)
            polyline = self.points_to_polyline(control_points)
            width = np.random.uniform(*self.config.width_range)
            
            polylines.append(polyline)
            widths.append(width)
            
            # Draw on canvas
            canvas = self.draw_filament(polyline, width, canvas)
        
        # Apply blur
        blur_sigma = np.random.uniform(*self.config.blur_sigma_range)
        blurred = gaussian_filter(canvas, blur_sigma)
        
        # Add noise
        noise_level = np.random.uniform(*self.config.background_noise)
        if noise_level > 0:
            noise = np.random.randn(*self.config.image_size) * noise_level
            blurred = blurred + noise
        
        # Normalize to 0-255
        if blurred.max() > 0:
            blurred = (blurred / blurred.max() * 255).astype(np.uint8)
        else:
            blurred = (blurred * 255).astype(np.uint8)
        
        # Create partial annotations
        partial_ratio = np.random.uniform(*self.config.partial_annotation_ratio)
        partial_length_ratio = np.random.uniform(*self.config.partial_length_ratio)
        
        annotated_polylines = []
        unannotated_polylines = []
        
        for i, polyline in enumerate(polylines):
            if np.random.random() < partial_ratio:
                # Partial annotation - keep only a portion
                start_idx = 0
                end_idx = int(len(polyline) * partial_length_ratio)
                if end_idx > start_idx:
                    annotated_polylines.append(polyline[start_idx:end_idx])
                    if end_idx < len(polyline):
                        unannotated_polylines.append(polyline[end_idx:])
            else:
                # Full annotation
                annotated_polylines.append(polyline)
        
        return {
            'image': blurred,
            'full_polylines': polylines,
            'annotated_polylines': annotated_polylines,
            'unannotated_polylines': unannotated_polylines,
            'widths': widths,
            'config': {
                'blur_sigma': blur_sigma,
                'noise_level': noise_level,
                'partial_ratio': partial_ratio,
            }
        }
    
    def generate_batch(self, num_images: int, output_dir: Path) -> List[Dict]:
        """Generate multiple images and save to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i in range(num_images):
            if self.config.seed is not None:
                np.random.seed(self.config.seed + i)
            
            result = self.generate_image_with_annotations()
            
            # Save image
            img_path = output_dir / f"image_{i:04d}.png"
            cv2.imwrite(str(img_path), result['image'])
            
            # Save annotations in multiple formats
            annotation = {
                'image_path': str(img_path),
                'image_id': i,
                'annotated_polylines': [p.tolist() for p in result['annotated_polylines']],
                'unannotated_polylines': [p.tolist() for p in result['unannotated_polylines']],
                'full_polylines': [p.tolist() for p in result['full_polylines']],
                'widths': result['widths'],
                'generation_params': result['config']
            }
            
            ann_path = output_dir / f"annotation_{i:04d}.json"
            with open(ann_path, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            results.append(annotation)
        
        # Save dataset metadata
        metadata = {
            'num_images': num_images,
            'config': {k: str(v) if isinstance(v, tuple) else v 
                      for k, v in self.config.__dict__.items()},
            'generator_class': 'FilamentGenerator'
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return results


def generate_demo_samples(num_samples: int = 10, seed: int = 42):
    """Generate demo samples for visualization."""
    config = GeneratorConfig(
        image_size=(512, 512),
        num_filaments=(30, 80),
        blur_sigma_range=(0.5, 2.5),
        convergence_zones=(1, 3),
        seed=seed
    )
    
    generator = FilamentGenerator(config)
    return generator.generate_image_with_annotations()


if __name__ == "__main__":
    # Demo generation
    import os
    
    config = GeneratorConfig(
        image_size=(512, 512),
        num_filaments=(40, 60),
        num_control_points=(30, 50),
        blur_sigma_range=(0.8, 2.0),
        convergence_zones=(2, 3),
        partial_annotation_ratio=(0.4, 0.8),
        seed=42
    )
    
    generator = FilamentGenerator(config)
    
    # Generate single sample
    result = generator.generate_image_with_annotations()
    
    print(f"Generated image shape: {result['image'].shape}")
    print(f"Number of filaments: {len(result['full_polylines'])}")
    print(f"Annotated polylines: {len(result['annotated_polylines'])}")
    print(f"Unannotated polylines: {len(result['unannotated_polylines'])}")
    
    # Save demo image
    cv2.imwrite("data/synthetic/demo_filaments.png", result['image'])
    
    # Also generate batch
    print("\nGenerating batch of 5 images...")
    generator.generate_batch(5, Path("data/synthetic/batch_001"))
    print("Done!")
