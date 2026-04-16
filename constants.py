"""Shared constants for PEZ reproduction pipeline."""

import numpy as np

# Dataset specs
N_FRAMES = 16
FPS = 24
DURATION = N_FRAMES / FPS  # 0.6667 seconds
RESOLUTION = 256
BALL_RADIUS_M = 0.3

# Velocity dataset
DIRECTIONS_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
SPEEDS = [1, 2, 3, 4, 5, 6, 7]  # m/s
N_START_POSITIONS = 7
SEED = 42

# Acceleration dataset
ACCELERATIONS = [2, 4, 6, 8, 10]  # m/s^2

# Camera/rendering
WORLD_EXTENT = 8.0  # visible area: [-8, 8]^2 meters
PIXELS_PER_METER = RESOLUTION / (2 * WORLD_EXTENT)  # 16 px/m
BALL_RADIUS_PX = max(3, int(BALL_RADIUS_M * PIXELS_PER_METER))  # ~5 pixels
BALL_COLOR_BGR = (220, 120, 60)  # blue-ish in BGR for OpenCV
FLOOR_COLOR = 128  # gray

# Paths
DATA_ROOT = "/home/solee/pez/artifacts/data/kubric_data"
FEATURES_ROOT = "/home/solee/pez/artifacts/features"
RESULTS_ROOT = "/home/solee/pez/artifacts/results"
VJEPA2_ROOT = "/home/solee/vjepa2"
VJEPA2_SRC = "/home/solee/vjepa2/src"
CHECKPOINT_DIR = "/mnt/md1/solee/checkpoints/vjepa2"

# V-JEPA 2 preprocessing (ImageNet stats)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VJEPA2_INPUT_SIZE = 224  # paper spec: 14x14 grid from 224/16


def get_start_positions():
    """Generate 7 start positions with seed 42, uniform in [-2, 2]^2."""
    rng = np.random.RandomState(SEED)
    return rng.uniform(-2, 2, size=(N_START_POSITIONS, 2))


def world_to_pixel(x_world, y_world):
    """Convert world coords (meters) to pixel coords in 256x256 image.
    World origin (0,0) -> image center (128,128).
    World +X -> pixel right, World +Y -> pixel up (so flip Y).
    """
    px = RESOLUTION / 2 + x_world * PIXELS_PER_METER
    py = RESOLUTION / 2 - y_world * PIXELS_PER_METER
    return px, py
