"""Shared constants for PEZ reproduction pipeline."""

import os

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

# Paths. Artifacts live under ./artifacts in this checkout; the V-JEPA 2 source
# and checkpoints live elsewhere and can be set with environment variables.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "artifacts")
DATA_ROOT = os.path.join(ARTIFACTS_ROOT, "data", "kubric_data")
FEATURES_ROOT = os.path.join(ARTIFACTS_ROOT, "features")
RESULTS_ROOT = os.path.join(ARTIFACTS_ROOT, "results")
INTPHYS_ROOT = os.path.join(ARTIFACTS_ROOT, "intphys")


def _env_path(name, default):
    return os.path.abspath(os.path.expanduser(os.environ.get(name) or default))


VJEPA2_ROOT = _env_path("VJEPA2_ROOT", "/home/solee/vjepa2")
VJEPA2_SRC = os.path.join(VJEPA2_ROOT, "src")
CHECKPOINT_DIR = _env_path("VJEPA2_CHECKPOINT_DIR", "/mnt/md1/solee/checkpoints/vjepa2")

# V-JEPA 2 preprocessing (ImageNet stats)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VJEPA2_INPUT_SIZE = 256  # Matches local V-JEPA 2 pretrain/checkpoint family (256px)

VJEPA2_MODEL_SPECS = {
    "large": {
        "repo_name": "vjepa2_L",
        "checkpoint": "vitl.pt",
        "factory": "vit_large",
        "embed_dim": 1024,
        "depth": 24,
    },
    "huge": {
        "repo_name": "vjepa2_H",
        "checkpoint": "vith.pt",
        "factory": "vit_huge",
        "embed_dim": 1280,
        "depth": 32,
    },
    "giant": {
        "repo_name": "vjepa2_G",
        "checkpoint": "vitg.pt",
        "factory": "vit_giant_xformers",
        "embed_dim": 1408,
        "depth": 40,
    },
}


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
