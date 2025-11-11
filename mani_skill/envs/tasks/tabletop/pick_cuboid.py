from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cuboid_cfgs import PICK_CUBOID_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from pathlib import Path
import os.path as osp
from transforms3d.euler import euler2quat


class HighFrictionTableSceneBuilder(TableSceneBuilder):
    """Table scene builder with extremely high friction to prevent sliding"""
    
    def build(self):
        builder = self.scene.create_actor_builder()
        model_dir = Path(osp.dirname(__file__)) / ".." / ".." / ".." / "utils" / "scene_builder" / "table" / "assets"
        table_model_file = str(model_dir / "table.glb")
        scale = 1.75

        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        
        # Create extremely high friction material for the table
        # Using very high friction values to prevent any sliding
        table_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=100.0,   # Extremely high static friction
            dynamic_friction=100.0,  # Extremely high dynamic friction
            restitution=1.0,          # perfectly elastic
        )
        
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
            half_size=(2.418 / 2, 1.209 / 2, 0.9196429 / 2),
            material=table_material,
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        builder.initial_pose = sapien.Pose(
            p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )
        table = builder.build_kinematic(name="table-workspace")
        aabb = (
            table._objs[0]
            .find_component_by_type(sapien.render.RenderBodyComponent)
            .compute_global_aabb_tight()
        )
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1]
        self.table_height = aabb[1, 2] - aabb[0, 2]
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        
        # Also give the ground high friction
        from mani_skill.utils.building.ground import build_ground
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        self.table = table
        self.scene_objects = [self.table, self.ground]


@register_env("PickCuboid-v1", max_episode_steps=50)
class PickCuboidEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cuboid and move it to a target goal position. Baseline task for MDPO

    **Randomizations:**
    - the cuboid's size along the x and y axis is also randomized.
    - the cuboid's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cuboid's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cuboid has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cuboid position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    SUPPORTED_ROBOTS = [
        "panda",
        # "fetch",
        # "xarm6_robotiq",
        # "so100",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100]
    cuboid_half_sizes = [0.03, 0.01, 0.02]
    goal_thresh = 0.03
    cuboid_spawn_half_size = 0.05
    cuboid_spawn_center = (0, 0)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBOID_CONFIGS:
            cfg = PICK_CUBOID_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBOID_CONFIGS["panda"]
        self.cuboid_half_sizes = cfg["cuboid_half_sizes"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cuboid_spawn_half_size = cfg["cuboid_spawn_half_size"]
        self.cuboid_spawn_center = cfg["cuboid_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = HighFrictionTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Create high friction material for cuboid
        cuboid_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=100.0,  
            dynamic_friction=100.0,  
            restitution=0.0,         # No bouncing
        )
        
        # Create per-environment cuboids with randomized dimensions
        base_half_sizes = self.cuboid_half_sizes.copy()
        self._cuboids: List[Actor] = []
        self._cuboid_half_sizes_list = []
        
        for i in range(self.num_envs):
            # Randomize dimensions for this environment
            scale_x = np.random.uniform(0.9, 5.0)
            scale_y = np.random.uniform(0.9, 5.0)
            cuboid_half_sizes = [
                base_half_sizes[0] * scale_x,
                base_half_sizes[1] * scale_y,
                base_half_sizes[2],
            ]
            
            # Ensure at least one dimension is small enough to grasp
            if cuboid_half_sizes[0] >= 0.04 and cuboid_half_sizes[1] >= 0.04:
                # index = np.random.choice([0, 1])
                # cuboid_half_sizes[index] = 0.03
                min_dim = 0 if cuboid_half_sizes[0] < cuboid_half_sizes[1] else 1
                cuboid_half_sizes[min_dim] = 0.03
                    
            self._cuboid_half_sizes_list.append(cuboid_half_sizes)
            
            # Create cuboid for this specific environment with high friction
            builder = self.scene.create_actor_builder()
            # builder.add_box_collision(half_size=cuboid_half_sizes)
            builder.add_box_collision(half_size=cuboid_half_sizes, material=cuboid_material)
            builder.add_box_visual(
                half_size=cuboid_half_sizes,
                material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, cuboid_half_sizes[2]])
            builder.set_scene_idxs([i])
            self._cuboids.append(builder.build(name=f"cuboid-{i}"))
            self.remove_from_state_dict_registry(self._cuboids[-1])
        
        # Merge all cuboids into a single batched actor
        self.cuboid = Actor.merge(self._cuboids, name="cuboid")
        self.add_to_state_dict_registry(self.cuboid)
        
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _after_reconfigure(self, options: dict):
        # Convert dimensions to tensor after device is set up
        self._cuboid_half_sizes_tensor = common.to_tensor(
            self._cuboid_half_sizes_list, device=self.device
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cuboid_spawn_half_size * 2
                - self.cuboid_spawn_half_size
            )
            xyz[:, 0] += self.cuboid_spawn_center[0]
            xyz[:, 1] += self.cuboid_spawn_center[1]
            xyz[:, 2] = self._cuboid_half_sizes_tensor[env_idx, 2]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cuboid.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cuboid_spawn_half_size * 2
                - self.cuboid_spawn_half_size
            )
            goal_xyz[:, 0] += self.cuboid_spawn_center[0]
            goal_xyz[:, 1] += self.cuboid_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
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
                obj_pose=self.cuboid.pose.raw_pose,
                tcp_to_obj_pos=self.cuboid.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cuboid.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cuboid.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cuboid)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cuboid.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cuboid.pose.p, axis=1
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

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5