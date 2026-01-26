from typing import Any, Dict, Union
from pathlib import Path
import os.path as osp
import glob

import numpy as np
import sapien
import sapien.physx
import sapien.render
import os
import torch
from transforms3d.euler import euler2quat
from sapien.render import RenderBodyComponent

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from scipy.spatial.transform import Rotation as R


# Load table texture files for domain randomization
TABLE_TEXTURE_DIR = Path(os.getcwd()) / "assets/curated_table_textures"
TABLE_TEXTURES = sorted(glob.glob(str(TABLE_TEXTURE_DIR / "*.png")))
print(f"### Loaded {len(TABLE_TEXTURES)} curated table textures")

EXRS_DOME_LIGHTING_DIR = Path(os.getcwd()) / "assets/dome_light_textures"
EXRS_DOME_LIGHTINGS = sorted(glob.glob(str(EXRS_DOME_LIGHTING_DIR / "*.exr")))
print(f"### Loaded {len(EXRS_DOME_LIGHTINGS)} curated dome lighting textures")

camera_data = Path(os.getcwd()) / "assets/calibration_data.npz"
camera_data = np.load(camera_data)
REAL_POSE = np.eye(4)
REAL_POSE[:3, 3] = camera_data["translations"]
REAL_POSE[:3, :3] = (R.from_matrix(camera_data["rotations"]) * R.from_euler('zyx', [90, 0, 90], degrees=True)).as_matrix()
INTRINSICS_REAL = camera_data["K"]

INTRINSICS_REAL[0, -1] -= (1280 - 720) / 2
INTRINSICS_REAL[:-1] *= 224.0 / 720.0 # adjustment factor after centercropping



@register_env("PickCube-v1", max_episode_steps=75)
class PickCubeEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
    capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100]
    cube_half_size = 0.02
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # pose = sapien_utils.look_at(
        #     eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        # )
        # return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]
        
        # Camera is mounted on cam_mount (kinematic actor) with identity local pose.
        # The cam_mount pose is set in _initialize_episode to be relative to link_0
        # with random perturbations applied each episode.
        return CameraConfig(
            "base_camera", 
            sapien.Pose(),  # Identity local pose; world pose set via cam_mount
            224, 224, 
            intrinsic=torch.from_numpy(INTRINSICS_REAL),
            near=0.01, far=100,
            mount=self.cam_mount, 
        )


    @property
    def _default_human_render_camera_configs(self):
        # Human camera positioned to view the robot arm, cube, and goal area
        # Robot is at (-0.615, 0, 0), workspace is around (-0.08, 0.84)
        # Camera placed at elevated position looking at the workspace center
        pose = sapien_utils.look_at(
            eye=[0.4, 1.6, 0.8],  # Elevated, to the right side of workspace
            target=[-0.2, 0.6, 0.1]  # Center of action (between robot and cube area)
        )
        return CameraConfig("render_camera", pose, 512, 512, fov=1, near=0.01, far=100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # Create camera mount actor for episodic camera pose randomization
        # Robot initial pose from _load_agent (link poses not available yet during _load_scene)
        robot_initial_pose = sapien.Pose(p=[-0.615, 0, 0])
        cam_initial_pose = robot_initial_pose * sapien.Pose(REAL_POSE)
        cam_mount_builder = self.scene.create_actor_builder()
        cam_mount_builder.initial_pose = cam_initial_pose
        self.cam_mount = cam_mount_builder.build_kinematic("camera_mount")
        
        # Build tables and cubes separately per environment for domain randomization support
        # This allows each parallel environment to have different physical/visual materials
        
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
        
        cube_material = sapien.physx.PhysxMaterial(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0
        )
        
        # Build separate cubes for each environment
        cubes = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[self.cube_half_size] * 3, material=cube_material)
            builder.add_box_visual(
                half_size=[self.cube_half_size] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=[0, 0, 0, 1],
                ),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
            builder.set_scene_idxs([i])
            cube = builder.build(name=f"cube_{i}")
            self.remove_from_state_dict_registry(cube)
            cubes.append(cube)
        
        self.cube = Actor.merge(cubes, name="cube")
        self.add_to_state_dict_registry(self.cube)
        
        # Goal site (shared, as it's just a visual marker)
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        # TODO: need to add randomization of background image by segmenting the background 
        self._hidden_objects.append(self.goal_site)

    def _load_lighting(self, options: Dict):
        print("Loading EXR Dome lighting with ambient randomization preset")
        for i in range(self.num_envs):
            self.scene.sub_scenes[i].set_environment_map(EXRS_DOME_LIGHTINGS[self._batched_episode_rng[i].randint(0, len(EXRS_DOME_LIGHTINGS))])
        # self.scene.set_ambient_light(np.array([1,1,1])*0.05)
        
        self.scene.set_ambient_light(np.array([1,1,1]) * np.random.uniform(0.05, 0.2))
            # [0.3, 0.3, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=2048
        # )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # TODO: randomize lighting a bit more


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

            # TODO: randomize quick physics parameters (friction, coef resitution, intertia, mass, etc.)

            #             
            # Spawn cube on the table, 14" in front of robot base
            # Robot base is at: x = -0.615 + 7*0.0254 = -0.4372, y = 1.200032 - 14*0.0254 = 0.8444
            # Robot faces +X direction (toward table center), so 14" in front = +X offset
            robot_base_x = -0.615 + 7 * 0.0254   # -0.4372m
            robot_base_y = 1.200032 - 14 * 0.0254  # 0.8444m
            cube_spawn_center_x = robot_base_x + 23 * 0.0254  # 14" in front of robot (+X direction)
            cube_spawn_center_y = robot_base_y  # Same Y as robot base
            
            xyz = torch.zeros((b, 3))
            cube_spawn_half_size = 0.1  # ±10cm range (matches original ManiSkill ±0.1m randomization)
            xyz[:, 0] = cube_spawn_center_x + (torch.rand((b,)) * 2 - 1) * cube_spawn_half_size
            xyz[:, 1] = cube_spawn_center_y + (torch.rand((b,)) * 2 - 1) * cube_spawn_half_size
            xyz[:, 2] = self.cube_half_size  # On table surface
            
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # Fixed goal position: 14" in front of robot, 300mm above table, same Y as robot base
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = robot_base_x + 16 * 0.0254  # 16" in front of robot (+X direction)
            goal_xyz[:, 1] = robot_base_y  # Same Y as robot base
            goal_xyz[:, 2] = 0.15  # 300mm above table surface
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube, max_angle=30)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        if self.robot_uids == "panda":
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        # Orientation reward: encourage vertical approach (gripper Z aligned with -world Z)
        tcp_pose_mat = self.agent.tcp_pose.to_transformation_matrix()
        gripper_z_axis = tcp_pose_mat[..., :3, 2]  # Z column of rotation matrix
        world_down = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        orientation_alignment = (gripper_z_axis * world_down).sum(dim=-1)
        approach_orientation_reward = (orientation_alignment + 1) / 2 * (~is_grasped) * 0.5 #[0, 0.5] 
        reward += approach_orientation_reward

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


@register_env("PickCubeSO100-v1", max_episode_steps=50)
class PickCubeSO100Env(PickCubeEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCubeSO100-v1_rt.mp4"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="so100", **kwargs)
