from typing import Any, Dict, Union
from pathlib import Path
import os.path as osp
import glob
import os

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat
from sapien.render import RenderBodyComponent
from scipy.spatial.transform import Rotation as R

from mani_skill.agents.robots.panda import Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig


# Load table texture files for domain randomization
TABLE_TEXTURE_DIR = Path(os.getcwd()) / "assets/curated_table_textures"
TABLE_TEXTURES = sorted(glob.glob(str(TABLE_TEXTURE_DIR / "*.png")))
print(f"### Loaded {len(TABLE_TEXTURES)} curated table textures")

EXRS_DOME_LIGHTING_DIR = Path(os.getcwd()) / "assets/dome_light_textures"
EXRS_DOME_LIGHTINGS = sorted(glob.glob(str(EXRS_DOME_LIGHTING_DIR / "*.exr")))
print(f"### Loaded {len(EXRS_DOME_LIGHTINGS)} curated dome lighting textures")

# Load real camera calibration data
camera_data = Path(os.getcwd()) / "assets/calibration_data.npz"
camera_data = np.load(camera_data)
REAL_POSE = np.eye(4)
REAL_POSE[:3, 3] = camera_data["translations"]
REAL_POSE[:3, :3] = (R.from_matrix(camera_data["rotations"]) * R.from_euler('zyx', [90, 0, 90], degrees=True)).as_matrix()

camera_data = Path(os.getcwd()) / "assets/calibration_data.npz"
camera_data = np.load(camera_data)
REAL_POSE = np.eye(4)
REAL_POSE[:3, 3] = camera_data["translations"]
REAL_POSE[:3, :3] = (R.from_matrix(camera_data["rotations"]) * R.from_euler('zyx', [90, 0, 90], degrees=True)).as_matrix()
INTRINSICS_REAL = camera_data["K"]

INTRINSICS_REAL[0, -1] -= (1280 - 720) / 2
INTRINSICS_REAL[:-1] *= 224.0 / 720.0 # adjustment factor after centercropping


def _build_box_with_rectangular_hole(
    scene: ManiSkillScene,
    box_half_size,  # cube half-size (all 3 dimensions equal)
    hole_half_h,  # hole half-size in Y (with clearance already added)
    hole_half_w,  # hole half-size in Z (with clearance already added)
    hole_depth,  # how deep the hole goes in X direction
    hole_center=(0, 0),  # (y, z) offset of hole center from box center
    material=None
):
    """
    Build a cube with a rectangular hole on one face.
    The hole opens at x=0 and extends in the +X direction for hole_depth.
    The box is centered at origin with half-size = box_half_size in all dims.
    
    x-axis is the insertion direction.
    """
    builder = scene.create_actor_builder()
    
    if material is None:
        material = sapien.render.RenderMaterial(
            base_color=[0.9, 0.9, 0.9, 1.0], roughness=0.5, specular=0.5
        )
    
    L = box_half_size  # cube half-size
    hh = hole_half_h  # hole half-height (Y)
    hw = hole_half_w  # hole half-width (Z)
    d = hole_depth    # hole depth (X)
    cy, cz = hole_center 
    
    # The hole opens at x = -L and extends to x = -L + d
    # The solid back is from x = -L + d to x = L
    
    # Part 1: Solid back wall (behind the hole)
    back_half_x = (2 * L - d) / 2  # half-size in x
    back_center_x = -L + d + back_half_x  # center position
    builder.add_box_collision(
        sapien.Pose([back_center_x, 0, 0]),
        [back_half_x, L, L]
    )
    builder.add_box_visual(
        sapien.Pose([back_center_x, 0, 0]),
        [back_half_x, L, L],
        material=material
    )
    
    # Part 2: Four walls around the hole (in the region x = -L to x = -L + d)
    hole_center_x = -L + d / 2  # center of hole region in X
    
    # Top wall (positive Y side)
    # From y = cy + hh to y = L
    top_thickness = (L - (cy + hh)) / 2
    top_center_y = cy + hh + top_thickness
    if top_thickness > 0:
        builder.add_box_collision(
            sapien.Pose([hole_center_x, top_center_y, 0]),
            [d / 2, top_thickness, L]
        )
        builder.add_box_visual(
            sapien.Pose([hole_center_x, top_center_y, 0]),
            [d / 2, top_thickness, L],
            material=material
        )
    
    # Bottom wall (negative Y side)
    # From y = -L to y = cy - hh
    bot_thickness = ((cy - hh) + L) / 2
    bot_center_y = -L + bot_thickness
    if bot_thickness > 0:
        builder.add_box_collision(
            sapien.Pose([hole_center_x, bot_center_y, 0]),
            [d / 2, bot_thickness, L]
        )
        builder.add_box_visual(
            sapien.Pose([hole_center_x, bot_center_y, 0]),
            [d / 2, bot_thickness, L],
            material=material
        )
    
    # Right wall (positive Z side)
    # From z = cz + hw to z = L, but only in the Y range where there's no top/bottom wall
    right_thickness = (L - (cz + hw)) / 2
    right_center_z = cz + hw + right_thickness
    if right_thickness > 0:
        builder.add_box_collision(
            sapien.Pose([hole_center_x, cy, right_center_z]),
            [d / 2, hh, right_thickness]
        )
        builder.add_box_visual(
            sapien.Pose([hole_center_x, cy, right_center_z]),
            [d / 2, hh, right_thickness],
            material=material
        )
    
    # Left wall (negative Z side)
    # From z = -L to z = cz - hw
    left_thickness = ((cz - hw) + L) / 2
    left_center_z = -L + left_thickness
    if left_thickness > 0:
        builder.add_box_collision(
            sapien.Pose([hole_center_x, cy, left_center_z]),
            [d / 2, hh, left_thickness]
        )
        builder.add_box_visual(
            sapien.Pose([hole_center_x, cy, left_center_z]),
            [d / 2, hh, left_thickness],
            material=material
        )
    
    return builder


@register_env("PegInsertionSide-v1", max_episode_steps=150)
class PegInsertionSideEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a black peg and insert it into the white box with a rectangular hole.

    **Randomizations:**
    - Box is a cube with side length randomized between 0.085m and 0.125m (during reconfiguration)
    - Peg dimensions: length = box_side/2, height and width randomized between 0.015m and 0.025m (during reconfiguration)
    - Hole depth = peg length (half of box), with 3mm clearance on height/width
    - Hole center offset: uniformly randomized ±25mm in Y and Z
    - Peg is laid flat on table and has its xy position and z-axis rotation randomized
    - Box is laid flat on table and has its xy position and z-axis rotation randomized
    - Table texture is randomized per environment
    - Dome lighting is randomized per environment
    - Camera pose has small perturbations (±1cm position, ±3° rotation)
    - Peg color slightly randomized (shades of black, 0.0-0.06 gray)
    - Box color slightly randomized (shades of white, 0.85-1.0 gray)

    **Success Conditions:**
    - The peg is inserted 90% of the hole depth (0.9 * peg_length).
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionSide-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]
    _clearance = 0.002  # 2mm radial clearance for hole

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        num_envs=1,
        reconfiguration_freq=None,
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig()

    # @property
    # def _default_sensor_configs(self):
    #     # Camera is mounted on cam_mount with identity local pose
    #     return CameraConfig(
    #         "base_camera",
    #         sapien.Pose(),  # Identity local pose; world pose set via cam_mount
    #         224, 224,
    #         intrinsic=torch.from_numpy(INTRINSICS_REAL),
    #         near=0.01, far=100,
    #         mount=self.cam_mount,
    #     )

    @property
    def _default_human_render_camera_configs(self):
        # Human camera positioned to view the robot arm, peg, and box area
        # Robot is at (-0.615, 0, 0), workspace is around the table
        pose = sapien_utils.look_at(
            eye=[0.4, 1.6, 0.8],  # Elevated, to the right side of workspace
            target=[-0.2, 0.6, 0.1]  # Center of action
        )
        return CameraConfig("render_camera", pose, 512, 512, fov=1, near=0.01, far=100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            #
            #  Create camera mount actor for episodic camera pose randomization Robot initial pose from _load_agent (link poses not available yet during _load_scene)
            robot_initial_pose = sapien.Pose(p=[-0.615, 0, 0])
            cam_initial_pose = robot_initial_pose * sapien.Pose(REAL_POSE)
            cam_mount_builder = self.scene.create_actor_builder()
            cam_mount_builder.initial_pose = cam_initial_pose
            self.cam_mount = cam_mount_builder.build_kinematic("camera_mount")

            # Table configuration
            model_dir = Path(osp.dirname(__file__)).parent.parent.parent / "utils" / "scene_builder" / "table" / "assets"
            table_model_file = str(model_dir / "table.glb")
            scale = 1.75
            # Base table dimensions (before scaling)
            base_table_length = 58 * 0.0254  # 1.4732m
            base_table_width = 40 * 0.0254   # 0.7112m
            base_table_height = 0.9196429
            # Scale collision box to match visual model scale
            table_length = base_table_length * scale  # 2.578m
            table_width = base_table_width * scale    # 1.245m
            table_height = base_table_height * scale  # 1.609m
            table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
            table_initial_pose = sapien.Pose(p=[-0.12, 0, -table_height], q=euler2quat(0, 0, np.pi / 2))

            # Build separate tables for each environment
            tables = []
            for i in range(self.num_envs):
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(
                    pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
                    half_size=(table_length / 2, table_width / 2, 0.9196429 / 2),
                )
                builder.add_visual_from_file(
                    filename=table_model_file, scale=[scale] * 3, pose=table_pose
                )
                builder.initial_pose = table_initial_pose
                builder.set_scene_idxs([i])
                table = builder.build_kinematic(name=f"table-workspace_{i}")
                self.remove_from_state_dict_registry(table)
                tables.append(table)

            self.table = Actor.merge(tables, name="table-workspace")
            self.add_to_state_dict_registry(self.table)

            # Apply random textures to each table
            for i, obj in enumerate(self.table._objs):
                # Sample a random texture for this table
                texture_idx = self._batched_episode_rng[i].randint(0, len(TABLE_TEXTURES))
                texture_path = TABLE_TEXTURES[texture_idx]
                texture = sapien.render.RenderTexture2D(
                    filename=texture_path,
                    mipmap_levels=1,
                )

                # Apply texture to all render parts of the table
                render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
                if render_body_component is not None:
                    for render_shape in render_body_component.render_shapes:
                        for part in render_shape.parts:
                            part.material.set_base_color([1, 1, 1, 1])  # Reset base color to white so texture shows properly
                            part.material.set_base_color_texture(texture)
                            # Reset material properties to avoid dark appearance from original .glb materials
                            part.material.set_metallic(0.0)  # Non-metallic for diffuse texture appearance
                            part.material.set_roughness(0.8)  # Mostly matte surface
                            part.material.set_specular(0.0)  # Reduce specular highlights
                            part.material.set_normal_texture(None)
                            part.material.set_metallic_texture(None)
                            part.material.set_roughness_texture(None)
                            part.material.set_emission_texture(None)

            # Compute table dimensions from first table for reference
            aabb = (
                tables[0]._objs[0]
                .find_component_by_type(sapien.render.RenderBodyComponent)
                .compute_global_aabb_tight()
            )
            self.table_length = aabb[1, 0] - aabb[0, 0]
            self.table_width = aabb[1, 1] - aabb[0, 1]
            self.table_height = aabb[1, 2] - aabb[0, 2]

            # Build ground (shared across all environments) - collision only, no visual
            floor_width = 100
            if self.scene.parallel_in_single_scene:
                floor_width = 500
            self.ground = build_ground(
                self.scene, floor_width=floor_width, altitude=-self.table_height
            )
            # Hide the ground visual (grid texture) - keep collision for physics
            for obj in self.ground._objs:
                render_comp = obj.find_component_by_type(RenderBodyComponent)
                if render_comp is not None:
                    obj.remove_component(render_comp)

            # Store table scene reference for compatibility and use TableSceneBuilder for robot initialization
            self.table_scene = TableSceneBuilder(
                self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            # Override table_scene's table and ground with our per-env versions
            self.table_scene.table = self.table
            self.table_scene.ground = self.ground
            self.table_scene.table_height = self.table_height
            self.table_scene.table_length = self.table_length
            self.table_scene.table_width = self.table_width
            self.table_scene.scene_objects = [self.table, self.ground]

            # Randomize peg and box dimensions
            # Box is a cube with side length in [0.085, 0.125]
            # Hole size is randomized first (to match printed parts), then peg = hole - clearance
            # Hole: 18mm x 18mm -> 31mm x 31mm (half-size: 9mm -> 15.5mm)
            box_sides = self._batched_episode_rng.uniform(0.1, 0.125)  # full box side length
            peg_lengths = box_sides / 2  
            hole_hw = self._batched_episode_rng.uniform(0.012, 0.0155)  # hole half-size (square): 24-31mm full
            peg_hw = hole_hw - self._clearance  # peg = hole - 2mm clearance
            peg_heights = peg_hw
            peg_widths = peg_hw
            
            # Hole center offset: uniformly ±25mm in Y and Z
            hole_centers = self._batched_episode_rng.uniform(-0.025, 0.025, size=(2,))

            # save some useful values for use later
            self.peg_half_sizes = common.to_tensor(np.vstack([peg_lengths, peg_heights, peg_widths])).T
            self.box_half_sizes = common.to_tensor(box_sides / 2)
            
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]  # peg head at +x
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            # Hole is on the -X face of the box, offset by hole_center in Y and Z
            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 0] = -self.box_half_sizes  # hole opens at -x face
            box_hole_offsets[:, 1:] = common.to_tensor(hole_centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            
            # Store hole dimensions (randomized first, peg derived from hole - clearance)
            self.hole_half_h = common.to_tensor(hole_hw)
            self.hole_half_w = common.to_tensor(hole_hw)
            self.hole_depth = common.to_tensor(box_sides / 2)  # hole depth = half the box (halfway through)
            
            # Success threshold: insert black half of peg (peg_lengths = half the full peg = black section length)
            self.success_insertion_depth = common.to_tensor(peg_lengths) * 0.9

            # in each parallel env we build a different box with a hole and peg
            pegs = []
            boxes = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                peg_half_size = peg_lengths[i]
                section_half_size = peg_half_size / 2
                peg_h = peg_heights[i]
                peg_w = peg_widths[i]
                box_half = box_sides[i] / 2
                hole_d = box_half

                black_gray = self._batched_episode_rng[i].uniform(0.0, 0.06)
                black_color = [black_gray, black_gray, black_gray, 1.0]
                
                white_gray = self._batched_episode_rng[i].uniform(0.92, 1.0)
                white_color = [white_gray, white_gray, white_gray, 1.0]

                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[peg_half_size, peg_h, peg_w])
                
                black_mat = sapien.render.RenderMaterial(
                    base_color=black_color,
                    metallic=0.0,
                    roughness=self._batched_episode_rng[i].uniform(0.5, 0.8),
                )
                builder.add_box_visual(
                    sapien.Pose([section_half_size, 0, 0]),
                    half_size=[section_half_size, peg_h, peg_w],
                    material=black_mat,
                )
                
                white_mat = sapien.render.RenderMaterial(
                    base_color=white_color,
                    metallic=0.0,
                    roughness=self._batched_episode_rng[i].uniform(0.5, 0.8),
                )
                builder.add_box_visual(
                    sapien.Pose([-section_half_size, 0, 0]),
                    half_size=[section_half_size, peg_h, peg_w],
                    material=white_mat,
                )
                
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)

                # Box color: white with slight variation (same as peg white half)
                box_color = white_color  # reuse same white color for box
                box_material = sapien.render.RenderMaterial(
                    base_color=box_color, roughness=0.5, specular=0.5
                )

                # Build box with rectangular hole
                builder = _build_box_with_rectangular_hole(
                    self.scene,
                    box_half_size=box_half,
                    hole_half_h=hole_hw[i],
                    hole_half_w=hole_hw[i],
                    hole_depth=hole_d,  # hole goes halfway through the box
                    hole_center=hole_centers[i],
                    material=box_material
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")
                self.remove_from_state_dict_registry(box)
                pegs.append(peg)
                boxes.append(box)

            self.peg = Actor.merge(pegs, "peg")
            self.box = Actor.merge(boxes, "box_with_hole")

            # to support heterogeneous simulation state dictionaries we register merged versions
            # of the parallel actors
            self.add_to_state_dict_registry(self.peg)
            self.add_to_state_dict_registry(self.box)

    def _load_lighting(self, options: Dict):
        print("Loading EXR Dome lighting with ambient randomization preset")
        for i in range(self.num_envs):
            self.scene.sub_scenes[i].set_environment_map(EXRS_DOME_LIGHTINGS[self._batched_episode_rng[i].randint(0, len(EXRS_DOME_LIGHTINGS))])
            self.scene.sub_scenes[i].render_system.ambient_light = (
                np.array([1, 1, 1]) * self._batched_episode_rng[i].uniform(0.05, 0.2)
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            env_idx = env_idx.to(self.device)
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Camera pose randomization
            link0_pose = self.agent.robot.links[0].pose[env_idx]

            # Apply nominal camera pose (REAL_POSE) relative to link_0
            nominal_pose = Pose.create(sapien.Pose(REAL_POSE))
            cam_pose = link0_pose * nominal_pose

            # Apply random perturbation: position ±1cm, rotation up to ±3 degrees
            max_angle = np.deg2rad(3)
            axes = np.random.randn(b, 3)
            axes /= np.linalg.norm(axes, axis=1, keepdims=True)
            angles = np.random.uniform(-max_angle, max_angle, size=(b, 1))
            rotvecs = axes * angles
            delta_quats = R.from_rotvec(rotvecs).as_quat()
            delta_quats = np.roll(delta_quats, 1, axis=1)
            delta_positions = np.random.uniform(-0.01, 0.01, size=(b, 3))
            # Create perturbation pose and apply to camera pose
            perturbation = Pose.create_from_pq(
                p=torch.from_numpy(delta_positions).float().to(self.device),
                q=torch.from_numpy(delta_quats).float().to(self.device),
            )
            cam_pose = cam_pose * perturbation
            self.cam_mount.set_pose(cam_pose)

            # Randomize wrist camera if using 
            if self.robot_uids == "panda_wristcam":
                self.agent.randomize_wrist_camera_pose(env=self, env_idx=env_idx)

            # Robot base position for spawn calculations
            # Robot base is at: x = -0.615 + 7*0.0254 = -0.4372, y = 1.200032 - 14*0.0254 = 0.8444
            robot_base_x = -0.615 + 7 * 0.0254   # -0.4372m
            robot_base_y = 1.200032 - 14 * 0.0254  # 0.8444m

            # Peg spawn: In front and slightly to the side of robot
            # Original spawn region was [-0.1, -0.3] to [0.1, 0] - a 0.2 x 0.3 region
            # Now spawn relative to robot position
            peg_spawn_center_x = robot_base_x + 14 * 0.0254  # 14" in front of robot (+X direction)
            peg_spawn_center_y = robot_base_y - 6 * 0.0254   # 6" to the side (-Y direction)
            peg_spawn_half_x = 0.1  # ±10cm in X
            peg_spawn_half_y = 0.15  # ±15cm in Y

            xy = torch.zeros((b, 2))
            xy[:, 0] = peg_spawn_center_x + (torch.rand((b,)) * 2 - 1) * peg_spawn_half_x
            xy[:, 1] = peg_spawn_center_y + (torch.rand((b,)) * 2 - 1) * peg_spawn_half_y
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 2]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
            )
            self.peg.set_pose(Pose.create_from_pq(pos, quat))

            # Box spawn: Further in front of robot (beyond peg)
            # Original spawn region was [-0.05, 0.2] to [0.05, 0.4] - a 0.1 x 0.2 region
            box_spawn_center_x = robot_base_x + 23 * 0.0254  # 20" in front of robot (+X direction)
            box_spawn_center_y = robot_base_y  # Same Y as robot base
            box_spawn_half_x = 0.05  # ±5cm in X
            box_spawn_half_y = 0.1   # ±10cm in Y

            xy = torch.zeros((b, 2))
            xy[:, 0] = box_spawn_center_x + (torch.rand((b,)) * 2 - 1) * box_spawn_half_x
            xy[:, 1] = box_spawn_center_y + (torch.rand((b,)) * 2 - 1) * box_spawn_half_y
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.box_half_sizes[env_idx]  # box is a cube, use its half-size
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(0, np.pi / 2),
            )
            self.box.set_pose(Pose.create_from_pq(pos, quat))

            # Override robot qpos: gripper hovering above peg spawn center
            # TODO: check peg is always in frame at start
            qpos = np.array([
                -0.36,          
                -0.01745329,         
                0.0,         
                -2.3387412,          
                0.0,          
                2.26892803,          
                -1.13446401,   
                0.035,         
                0.035,         
            ])
            qpos = self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.035  # keep gripper open
            self.agent.robot.set_qpos(qpos)

    # save some commonly used attributes
    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets

    @property
    def box_hole_pose(self):
        return self.box.pose * self.box_hole_offsets

    @property
    def goal_pose(self):
        # NOTE (stao): this is fixed after each _initialize_episode call. You can cache this value
        # and simply store it after _initialize_episode or set_state_dict calls.
        return self.box.pose * self.box_hole_offsets * self.peg_head_offsets.inv()

    def has_peg_inserted(self):
        # Check if peg head has been inserted 90% of hole depth
        # Peg head position relative to box hole opening
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        
        # x-axis is the insertion direction (positive = deeper into hole)
        # Success: peg head is at least 0.9 * hole_depth inside
        x_flag = peg_head_pos_at_hole[:, 0] >= self.success_insertion_depth
        
        # Y and Z: peg head must be within the hole bounds (with clearance)
        y_flag = (-self.hole_half_h <= peg_head_pos_at_hole[:, 1]) & (
            peg_head_pos_at_hole[:, 1] <= self.hole_half_h
        )
        z_flag = (-self.hole_half_w <= peg_head_pos_at_hole[:, 2]) & (
            peg_head_pos_at_hole[:, 2] <= self.hole_half_w
        )
        return (
            x_flag & y_flag & z_flag,
            peg_head_pos_at_hole,
        )

    def evaluate(self):
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_half_size=self.peg_half_sizes,
                box_hole_pose=self.box_hole_pose.raw_pose,
                hole_half_h=self.hole_half_h,
                hole_half_w=self.hole_half_w,
                hole_depth=self.hole_depth,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Stage 1: Encourage gripper to be rotated to be lined up with the peg

        # Stage 2: Encourage gripper to move close to peg tail and grasp it
        gripper_pos = self.agent.tcp.pose.p
        tgt_gripper_pose = self.peg.pose
        
        offset_p = torch.zeros((self.num_envs, 3), device=self.device)
        offset_p[:, 0] = -self.peg_half_sizes[:, 0] * 0.75
        offset = Pose.create_from_pq(p=offset_p)
        tgt_gripper_pose = tgt_gripper_pose * offset
        gripper_to_peg_dist = torch.linalg.norm(
            gripper_pos - tgt_gripper_pose.p, axis=1
        )

        reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)

        # check with max_angle=20 to ensure gripper isn't grasping peg at an awkward pose
        is_grasped = self.agent.is_grasping(self.peg, max_angle=20)
        reward = reaching_reward + is_grasped

        # Stage 3: Orient the grasped peg properly towards the hole

        # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_head_wrt_goal_yz_dist = torch.linalg.norm(
            peg_head_wrt_goal.p[:, 1:], axis=1
        )
        peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
        peg_wrt_goal_yz_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

        pre_insertion_reward = 3 * (
            1
            - torch.tanh(
                0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
                + 4.5 * torch.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
            )
        )
        reward += pre_insertion_reward * is_grasped
        # stage 3 passes if peg is correctly oriented in order to insert into hole easily
        pre_inserted = (peg_head_wrt_goal_yz_dist < 0.01) & (
            peg_wrt_goal_yz_dist < 0.01
        )

        # Stage 4: Insert the peg into the hole once it is grasped and lined up
        peg_head_wrt_goal_inside_hole = self.box_hole_pose.inv() * self.peg_head_pose
        
        insertion_error = peg_head_wrt_goal_inside_hole.p.clone()
        insertion_error[:, 0] = insertion_error[:, 0] - self.success_insertion_depth  # distance to goal depth
        
        insertion_reward = 5 * (
            1
            - torch.tanh(
                5.0 * torch.linalg.norm(insertion_error, axis=1)
            )
        )
        reward += insertion_reward * (is_grasped & pre_inserted)

        reward[info["success"]] = 10

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10
