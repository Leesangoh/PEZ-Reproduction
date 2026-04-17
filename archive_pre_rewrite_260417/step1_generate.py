"""Step 1: Generate synthetic ball datasets for PEZ reproduction.

Supports two rendering backends:
- `kubric`: Blender-based Kubric rendering inside a Blender Python process
- `pyrender`: fallback renderer for plain Python environments

The rendering stack follows the paper's Kubric/Blender setup. Motion trajectories
are simulated in PyBullet at 240 Hz (10 substeps per rendered frame) before
being rendered framewise.

Usage:
  Blender / Kubric:
    PYTHONPATH=/home/solee/kubric blender --background \
      --python /home/solee/pez/step1_generate.py -- --backend kubric

  Fallback / plain Python:
    /isaac-sim/python.sh /home/solee/pez/step1_generate.py --backend pyrender
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/solee/pez")
from constants import (
    ACCELERATIONS,
    DATA_ROOT,
    DIRECTIONS_DEG,
    DURATION,
    FPS,
    N_FRAMES,
    N_START_POSITIONS,
    RESOLUTION,
    SEED,
    SPEEDS,
)


BALL_RADIUS = 0.3
FLOOR_SIZE_M = 8.0
CAMERA_Z = 10.0
SIM_STEP_RATE = 240
STEPS_PER_FRAME = SIM_STEP_RATE // FPS
SIM_DT = 1.0 / SIM_STEP_RATE


def perspective_yfov_for_floor(camera_z: float, floor_size_m: float) -> float:
    return 2.0 * np.arctan((floor_size_m / 2.0) / camera_z)


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, rgb.astype(np.uint8))


def load_rgb_png(path: Path) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def sample_start_positions_by_pair(n_secondary_conditions: int, seed: int) -> np.ndarray:
    """Sample 7 fresh start positions for every (direction, condition) pair."""
    rng = np.random.RandomState(seed)
    return rng.uniform(
        -2.0,
        2.0,
        size=(len(DIRECTIONS_DEG), n_secondary_conditions, N_START_POSITIONS, 2),
    )


def init_pybullet_scene(x0: float, y0: float):
    import pybullet as pb

    client = pb.connect(pb.DIRECT)
    pb.setTimeStep(SIM_DT, physicsClientId=client)
    pb.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
    pb.setPhysicsEngineParameter(
        restitutionVelocityThreshold=0.0,
        warmStartingFactor=0.0,
        useSplitImpulse=True,
        contactSlop=0.0,
        enableConeFriction=False,
        deterministicOverlappingPairs=True,
        physicsClientId=client,
    )

    plane_shape = pb.createCollisionShape(pb.GEOM_PLANE, physicsClientId=client)
    plane_id = pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=plane_shape, physicsClientId=client)
    pb.changeDynamics(
        plane_id,
        -1,
        lateralFriction=0.0,
        rollingFriction=0.0,
        spinningFriction=0.0,
        restitution=0.0,
        physicsClientId=client,
    )

    sphere_shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=BALL_RADIUS, physicsClientId=client)
    sphere_id = pb.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=sphere_shape,
        basePosition=(x0, y0, BALL_RADIUS),
        useMaximalCoordinates=True,
        physicsClientId=client,
    )
    pb.changeDynamics(
        sphere_id,
        -1,
        lateralFriction=0.0,
        rollingFriction=0.0,
        spinningFriction=0.0,
        restitution=0.0,
        linearDamping=0.0,
        angularDamping=0.0,
        contactProcessingThreshold=0.0,
        physicsClientId=client,
    )
    return pb, client, sphere_id


def sample_pybullet_state(pb, sphere_id, client):
    position, _ = pb.getBasePositionAndOrientation(sphere_id, physicsClientId=client)
    velocity, _ = pb.getBaseVelocity(sphere_id, physicsClientId=client)
    return np.asarray(position, dtype=np.float64), np.asarray(velocity, dtype=np.float64)


def simulate_velocity_trajectory(x0: float, y0: float, vx: float, vy: float):
    pb, client, sphere_id = init_pybullet_scene(x0, y0)
    pb.resetBaseVelocity(
        sphere_id,
        linearVelocity=(vx, vy, 0.0),
        angularVelocity=(0.0, 0.0, 0.0),
        physicsClientId=client,
    )

    positions = []
    velocities = []
    for frame_idx in range(N_FRAMES):
        if frame_idx > 0:
            for _ in range(STEPS_PER_FRAME):
                pb.stepSimulation(physicsClientId=client)
        pos, vel = sample_pybullet_state(pb, sphere_id, client)
        positions.append(pos)
        velocities.append(vel)

    pb.disconnect(client)
    return np.stack(positions), np.stack(velocities)


def simulate_acceleration_trajectory(x0: float, y0: float, ax: float, ay: float):
    pb, client, sphere_id = init_pybullet_scene(x0, y0)
    force = np.asarray([ax, ay, 0.0], dtype=np.float64)  # m=1.0 kg, so F = a

    positions = []
    velocities = []
    for frame_idx in range(N_FRAMES):
        if frame_idx > 0:
            for _ in range(STEPS_PER_FRAME):
                pb.applyExternalForce(
                    sphere_id,
                    -1,
                    forceObj=force.tolist(),
                    posObj=[0.0, 0.0, 0.0],
                    flags=pb.WORLD_FRAME,
                    physicsClientId=client,
                )
                pb.stepSimulation(physicsClientId=client)
        pos, vel = sample_pybullet_state(pb, sphere_id, client)
        positions.append(pos)
        velocities.append(vel)

    pb.disconnect(client)
    return np.stack(positions), np.stack(velocities)


class BaseRenderer:
    def __init__(self, resolution=256, floor_size_m=8.0, camera_z=10.0):
        self.resolution = resolution
        self.floor_size_m = floor_size_m
        self.camera_z = camera_z
        self.yfov = perspective_yfov_for_floor(camera_z, floor_size_m)

    def project_point(self, x, y, z):
        depth = self.camera_z - z
        scale = np.tan(self.yfov / 2.0) * depth
        x_ndc = x / scale
        y_ndc = y / scale
        px = (x_ndc + 1.0) * 0.5 * self.resolution
        py = (1.0 - y_ndc) * 0.5 * self.resolution
        return float(px), float(py)

    def pixel_scale(self, z):
        depth = self.camera_z - z
        return self.resolution / (2.0 * np.tan(self.yfov / 2.0) * depth)

    def render_frame(self, ball_x, ball_y):
        raise NotImplementedError

    def render_frames(self, xy_positions):
        return [self.render_frame(x, y) for x, y in xy_positions]

    def close(self):
        return None


class PBRRenderer(BaseRenderer):
    """Fallback renderer used outside Blender/Kubric."""

    def __init__(self, resolution=256, floor_size_m=8.0, camera_z=10.0):
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender
        import trimesh

        super().__init__(resolution=resolution, floor_size_m=floor_size_m, camera_z=camera_z)
        self._pyrender = pyrender
        self.scene = pyrender.Scene(
            ambient_light=[0.10, 0.10, 0.10],
            bg_color=[0.40, 0.40, 0.40, 1.0],
        )

        floor_geom = trimesh.creation.box(extents=[floor_size_m, floor_size_m, 0.02])
        floor_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.55, 0.53, 0.50, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.90,
        )
        self.scene.add(
            pyrender.Mesh.from_trimesh(floor_geom, material=floor_mat),
            pose=self._translate(0, 0, -0.01),
        )

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.5)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, 0, 15]
        self.scene.add(light, pose=light_pose)

        camera = pyrender.PerspectiveCamera(
            yfov=self.yfov,
            aspectRatio=1.0,
            znear=0.1,
            zfar=100.0,
        )
        cam_pose = np.eye(4)
        cam_pose[2, 3] = camera_z
        self.scene.add(camera, pose=cam_pose)

        sphere_geom = trimesh.creation.icosphere(subdivisions=4, radius=BALL_RADIUS)
        ball_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.15, 0.40, 0.85, 1.0],
            metallicFactor=0.2,
            roughnessFactor=0.35,
        )
        ball_mesh = pyrender.Mesh.from_trimesh(sphere_geom, material=ball_mat)
        self.ball_node = self.scene.add(ball_mesh)

        self.renderer = pyrender.OffscreenRenderer(resolution, resolution)
        self.render_flags = pyrender.RenderFlags.SHADOWS_DIRECTIONAL

    @staticmethod
    def _translate(x, y, z):
        pose = np.eye(4)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z
        return pose

    def render_frame(self, ball_x, ball_y):
        pose = self._translate(ball_x, ball_y, BALL_RADIUS)
        self.scene.set_pose(self.ball_node, pose)
        color, _ = self.renderer.render(self.scene, flags=self.render_flags)
        return color

    def close(self):
        self.renderer.delete()


class KubricRenderer(BaseRenderer):
    """Kubric + Blender renderer used when running inside Blender."""

    def __init__(self, resolution=256, floor_size_m=8.0, camera_z=10.0):
        import kubric as kb
        from kubric.renderer.blender import Blender as KubricBlender

        super().__init__(resolution=resolution, floor_size_m=floor_size_m, camera_z=camera_z)
        self.kb = kb
        self.scratch_dir = Path(tempfile.mkdtemp(prefix="pez_kubric_render_"))

        scene = kb.Scene(resolution=(resolution, resolution), frame_start=0, frame_end=N_FRAMES - 1)
        scene.frame_rate = FPS
        scene.step_rate = 240
        scene.ambient_illumination = kb.Color(0.10, 0.10, 0.10)
        scene.background = kb.Color(0.40, 0.40, 0.40)

        self.renderer = KubricBlender(
            scene,
            scratch_dir=self.scratch_dir,
            adaptive_sampling=False,
            use_denoising=False,
            samples_per_pixel=16,
        )

        floor_material = kb.PrincipledBSDFMaterial(
            color=kb.Color(0.55, 0.53, 0.50),
            roughness=0.90,
            metallic=0.0,
            specular=0.0,
        )
        floor = kb.Cube(
            name="floor",
            scale=(floor_size_m / 2.0, floor_size_m / 2.0, 0.01),
            position=(0, 0, -0.01),
            material=floor_material,
            static=True,
            background=True,
            friction=0.0,
            restitution=0.0,
        )

        light = kb.DirectionalLight(
            name="sun",
            position=(0, 0, 15),
            look_at=(0, 0, 0),
            intensity=4.5,
            color=kb.get_color("white"),
        )

        ball_material = kb.PrincipledBSDFMaterial(
            color=kb.Color(0.15, 0.40, 0.85),
            metallic=0.2,
            roughness=0.35,
        )
        ball = kb.Sphere(
            name="ball",
            scale=BALL_RADIUS,
            position=(0, 0, BALL_RADIUS),
            material=ball_material,
            static=True,
            mass=1.0,
            friction=0.0,
            restitution=0.0,
        )

        scene += floor
        scene += light
        scene += ball

        scene.camera = kb.PerspectiveCamera(position=(0, 0, camera_z), look_at=(0, 0, 0))
        scene.camera.field_of_view = self.yfov

        self.scene = scene
        self.ball = ball

    def _clear_render_scratch(self):
        for subdir in ["exr", "images"]:
            clear_dir(self.scratch_dir / subdir)

    def render_frames(self, xy_positions):
        self._clear_render_scratch()
        self.ball.keyframes.clear()
        for frame_idx, (x, y) in enumerate(xy_positions):
            self.ball.position = (x, y, BALL_RADIUS)
            self.ball.keyframe_insert("position", frame_idx)

        result = self.renderer.render(
            frames=range(len(xy_positions)),
            ignore_missing_textures=True,
            return_layers=("rgba",),
        )
        return [frame[..., :3].astype(np.uint8) for frame in result["rgba"]]

    def close(self):
        shutil.rmtree(self.scratch_dir, ignore_errors=True)


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if "bpy" in sys.modules:
        return "kubric"
    return "pyrender"


def make_renderer(backend: str):
    if backend == "kubric":
        return KubricRenderer(
            resolution=RESOLUTION,
            floor_size_m=FLOOR_SIZE_M,
            camera_z=CAMERA_Z,
        )
    if backend == "pyrender":
        return PBRRenderer(
            resolution=RESOLUTION,
            floor_size_m=FLOOR_SIZE_M,
            camera_z=CAMERA_Z,
        )
    raise ValueError(f"Unknown backend: {backend}")


def video_complete(video_dir: Path) -> bool:
    return all((video_dir / f"frame_{frame_idx:02d}.png").exists() for frame_idx in range(N_FRAMES))


def velocity_rows(renderer, start_positions_by_pair, resume=False):
    out_dir = Path(DATA_ROOT) / "velocity" / "videos"
    gt_rows = []
    video_idx = 0

    for d_idx, direction_deg in enumerate(DIRECTIONS_DEG):
        theta = np.radians(direction_deg)
        for s_idx, speed in enumerate(SPEEDS):
            vx = speed * np.cos(theta)
            vy = speed * np.sin(theta)
            start_positions = start_positions_by_pair[d_idx, s_idx]
            for p_idx, (x0, y0) in enumerate(start_positions):
                video_id = f"vel_dir{d_idx}_spd{s_idx}_pos{p_idx}"
                video_dir = out_dir / video_id
                video_dir.mkdir(parents=True, exist_ok=True)
                already_done = resume and video_complete(video_dir)

                positions, velocities = simulate_velocity_trajectory(x0, y0, vx, vy)
                xy_positions = [(float(pos[0]), float(pos[1])) for pos in positions]
                frames = None if already_done else renderer.render_frames(xy_positions)

                for frame_idx, ((x, y), pos, vel) in enumerate(zip(xy_positions, positions, velocities)):
                    z = float(pos[2])
                    px, py = renderer.project_point(x, y, z)
                    px_scale = renderer.pixel_scale(z)
                    vx_px = px_scale * float(vel[0]) / FPS
                    vy_px = -px_scale * float(vel[1]) / FPS
                    speed_px = np.sqrt(vx_px**2 + vy_px**2)
                    direction_px = 0.0 if speed_px < 1e-12 else np.arctan2(vy_px, vx_px)
                    if frames is not None:
                        save_rgb_png(video_dir / f"frame_{frame_idx:02d}.png", frames[frame_idx])
                    gt_rows.append({
                        "video_id": video_id,
                        "video_idx": video_idx,
                        "frame_idx": frame_idx,
                        "pos_x_world": x,
                        "pos_y_world": y,
                        "pos_x_px": px,
                        "pos_y_px": py,
                        "vx_world": float(vel[0]),
                        "vy_world": float(vel[1]),
                        "vx_px": vx_px,
                        "vy_px": vy_px,
                        "speed": speed_px,
                        "direction_rad": direction_px,
                    })

                video_idx += 1
                if video_idx % 25 == 0:
                    print(f"  Velocity: {video_idx}/392 videos")

    df = pd.DataFrame(gt_rows)
    df.to_parquet(Path(DATA_ROOT) / "velocity" / "gt_velocity.parquet", index=False)
    print(f"Velocity dataset: {video_idx} videos, {len(df)} GT rows")
    return df


def acceleration_rows(renderer, start_positions_by_pair, resume=False):
    out_dir = Path(DATA_ROOT) / "acceleration" / "videos"
    gt_rows = []
    video_idx = 0

    for d_idx, direction_deg in enumerate(DIRECTIONS_DEG):
        theta = np.radians(direction_deg)
        for a_idx, accel in enumerate(ACCELERATIONS):
            ax = accel * np.cos(theta)
            ay = accel * np.sin(theta)
            start_positions = start_positions_by_pair[d_idx, a_idx]
            for p_idx, (x0, y0) in enumerate(start_positions):
                video_id = f"acc_dir{d_idx}_acc{a_idx}_pos{p_idx}"
                video_dir = out_dir / video_id
                video_dir.mkdir(parents=True, exist_ok=True)
                already_done = resume and video_complete(video_dir)

                positions, velocities = simulate_acceleration_trajectory(x0, y0, ax, ay)
                xy_positions = [(float(pos[0]), float(pos[1])) for pos in positions]
                frames = None if already_done else renderer.render_frames(xy_positions)
                vx_world_series = velocities[:, 0]
                vy_world_series = velocities[:, 1]
                ax_world_series = np.diff(vx_world_series) * FPS
                ay_world_series = np.diff(vy_world_series) * FPS
                measured_ax_world = float(ax_world_series.mean()) if len(ax_world_series) else 0.0
                measured_ay_world = float(ay_world_series.mean()) if len(ay_world_series) else 0.0

                vx_px_series = []
                vy_px_series = []
                for pos, vel in zip(positions, velocities):
                    px_scale = renderer.pixel_scale(float(pos[2]))
                    vx_px_series.append(px_scale * float(vel[0]) / FPS)
                    vy_px_series.append(-px_scale * float(vel[1]) / FPS)
                vx_px_series = np.asarray(vx_px_series, dtype=np.float64)
                vy_px_series = np.asarray(vy_px_series, dtype=np.float64)
                ax_px_series = np.diff(vx_px_series)
                ay_px_series = np.diff(vy_px_series)
                measured_ax_px = float(ax_px_series.mean()) if len(ax_px_series) else 0.0
                measured_ay_px = float(ay_px_series.mean()) if len(ay_px_series) else 0.0
                measured_accel_px = float(np.sqrt(measured_ax_px**2 + measured_ay_px**2))

                for frame_idx, ((x, y), pos, vel, vx_px, vy_px) in enumerate(
                    zip(xy_positions, positions, velocities, vx_px_series, vy_px_series)
                ):
                    z = float(pos[2])
                    cur_speed = np.sqrt(vx_px**2 + vy_px**2)
                    direction_px = theta if cur_speed < 1e-12 else np.arctan2(vy_px, vx_px)
                    px, py = renderer.project_point(x, y, z)
                    if frames is not None:
                        save_rgb_png(video_dir / f"frame_{frame_idx:02d}.png", frames[frame_idx])
                    gt_rows.append({
                        "video_id": video_id,
                        "video_idx": video_idx,
                        "frame_idx": frame_idx,
                        "pos_x_world": x,
                        "pos_y_world": y,
                        "pos_x_px": px,
                        "pos_y_px": py,
                        "vx_world": float(vel[0]),
                        "vy_world": float(vel[1]),
                        "vx_px": vx_px,
                        "vy_px": vy_px,
                        "speed": cur_speed,
                        "direction_rad": direction_px,
                        "ax_world": measured_ax_world,
                        "ay_world": measured_ay_world,
                        "ax_px": measured_ax_px,
                        "ay_px": measured_ay_px,
                        "accel_magnitude": measured_accel_px,
                    })

                video_idx += 1
                if video_idx % 25 == 0:
                    print(f"  Acceleration: {video_idx}/280 videos")

    df = pd.DataFrame(gt_rows)
    df.to_parquet(Path(DATA_ROOT) / "acceleration" / "gt_acceleration.parquet", index=False)
    print(f"Acceleration dataset: {video_idx} videos, {len(df)} GT rows")
    return df


def sanity_checks(vel_df, acc_df):
    print("\n=== SANITY CHECKS ===")

    n_vel = vel_df["video_id"].nunique()
    n_acc = acc_df["video_id"].nunique()
    print(f"[1] Video counts: velocity={n_vel}, acceleration={n_acc}")
    assert n_vel == 392
    assert n_acc == 280

    max_speed_diff = 0.0
    for vid in vel_df["video_id"].unique()[:50]:
        sub = vel_df[vel_df["video_id"] == vid]
        diff = sub["speed"].diff().abs().max()
        if not np.isnan(diff):
            max_speed_diff = max(max_speed_diff, diff)
    print(f"[2] Max speed diff across velocity frames: {max_speed_diff:.2e}")
    assert max_speed_diff < 1e-4

    max_acc_diff = 0.0
    for vid in acc_df["video_id"].unique()[:50]:
        sub = acc_df[acc_df["video_id"] == vid]
        diff = sub["accel_magnitude"].diff().abs().max()
        if not np.isnan(diff):
            max_acc_diff = max(max_acc_diff, diff)
    print(f"[3] Max accel diff across acceleration frames: {max_acc_diff:.2e}")
    assert max_acc_diff < 1e-6

    max_dir_diff = 0.0
    for vid in vel_df["video_id"].unique()[:50]:
        sub = vel_df[vel_df["video_id"] == vid]
        diff = sub["direction_rad"].diff().abs().max()
        if not np.isnan(diff):
            max_dir_diff = max(max_dir_diff, diff)
    print(f"[4] Max direction diff across velocity frames: {max_dir_diff:.2e}")
    assert max_dir_diff < 1e-4

    sanity_dir = Path(DATA_ROOT) / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    sample_vids = vel_df["video_id"].unique()[:5]
    for vid in sample_vids:
        frames = []
        for frame_idx in range(N_FRAMES):
            frames.append(load_rgb_png(Path(DATA_ROOT) / "velocity" / "videos" / vid / f"frame_{frame_idx:02d}.png"))
        imageio.mimsave(sanity_dir / f"{vid}.gif", frames, fps=FPS, loop=0)
    print(f"[5] Saved {len(sample_vids)} sample GIFs to {sanity_dir}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vel_per_video = vel_df[vel_df["frame_idx"] == 0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(vel_per_video["speed"], bins=20, edgecolor="black")
        axes[0].set_xlabel("Speed (px/frame)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Speed distribution")

        axes[1].hist(vel_per_video["direction_rad"], bins=20, edgecolor="black")
        axes[1].set_xlabel("Direction (rad)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Direction distribution")
        plt.tight_layout()
        plt.savefig(sanity_dir / "distributions.png", dpi=150)
        plt.close()
        print(f"[6] Distribution plot saved to {sanity_dir / 'distributions.png'}")
    except Exception as exc:
        print(f"[6] Could not generate distribution plot: {exc}")

    print("=== ALL SANITY CHECKS PASSED ===\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["auto", "kubric", "pyrender"], default="auto")
    parser.add_argument("--resume", action="store_true")
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]
    return parser.parse_args(argv)


def main():
    import time

    args = parse_args()
    backend = choose_backend(args.backend)

    print("=" * 60)
    print(f"PEZ Step 1: Synthetic Ball Video Generation ({backend})")
    print("=" * 60)

    data_root = Path(DATA_ROOT)
    for dataset_name in ["velocity", "acceleration"]:
        videos_dir = data_root / dataset_name / "videos"
        if args.resume:
            videos_dir.mkdir(parents=True, exist_ok=True)
        else:
            clear_dir(videos_dir)

    velocity_start_positions = sample_start_positions_by_pair(len(SPEEDS), seed=SEED)
    acceleration_start_positions = sample_start_positions_by_pair(len(ACCELERATIONS), seed=SEED + 1)
    legacy_start_positions = data_root / "start_positions.npy"
    if legacy_start_positions.exists():
        legacy_start_positions.unlink()
    np.save(data_root / "velocity_start_positions.npy", velocity_start_positions)
    np.save(data_root / "acceleration_start_positions.npy", acceleration_start_positions)
    np.savez(
        data_root / "start_positions.npz",
        velocity=velocity_start_positions,
        acceleration=acceleration_start_positions,
    )
    print(f"Random seed: {SEED}")
    print(
        "Using 7 freshly sampled start positions per (direction, speed) and "
        "(direction, acceleration) pair"
    )

    renderer = make_renderer(backend)
    print("Renderer ready.")

    t0 = time.time()
    print("\nGenerating velocity dataset (392 videos)...")
    vel_df = velocity_rows(renderer, velocity_start_positions, resume=args.resume)
    t1 = time.time()
    print(f"  Velocity render time: {t1 - t0:.1f}s")

    print("\nGenerating acceleration dataset (280 videos)...")
    acc_df = acceleration_rows(renderer, acceleration_start_positions, resume=args.resume)
    t2 = time.time()
    print(f"  Acceleration render time: {t2 - t1:.1f}s")

    renderer.close()

    metadata = {
        "n_frames": N_FRAMES,
        "fps": FPS,
        "duration": DURATION,
        "resolution": RESOLUTION,
        "ball_radius_m": BALL_RADIUS,
        "friction": 0.0,
        "restitution": 0.0,
        "seed": SEED,
        "directions_deg": DIRECTIONS_DEG,
        "speeds_m_per_s": SPEEDS,
        "accelerations_m_per_s2": ACCELERATIONS,
        "n_start_positions": N_START_POSITIONS,
        "floor_size_m": FLOOR_SIZE_M,
        "camera_z": CAMERA_Z,
        "rendering": backend,
        "renderer_details": {
            "camera": f"perspective_topdown(position=(0,0,{CAMERA_Z}), yfov={renderer.yfov:.6f})",
            "gt_units": "pixels_per_frame",
            "motion_generation": "pybullet_3.2.5_trajectories_rendered_framewise",
            "simulation_step_rate_hz": SIM_STEP_RATE,
            "substeps_per_rendered_frame": STEPS_PER_FRAME,
            "start_position_sampling": "fresh_uniform_samples_per_(direction,condition)_pair",
        },
    }
    if backend == "kubric":
        metadata["renderer_details"]["stack"] = "Kubric + Blender (container-local adaptation)"
        metadata["note"] = (
            "Kubric rendering enabled in this container. Motion trajectories are simulated in "
            "PyBullet at 240 Hz with 10 substeps per rendered frame before rendering."
        )
    else:
        metadata["renderer_details"]["stack"] = "pyrender fallback"
        metadata["note"] = "Kubric unavailable in this runtime; using fallback renderer."

    for dataset_name in ["velocity", "acceleration"]:
        with open(data_root / dataset_name / "metadata.json", "w") as fp:
            json.dump(metadata, fp, indent=2)

    sanity_checks(vel_df, acc_df)
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"Output: {DATA_ROOT}")


if __name__ == "__main__":
    main()
