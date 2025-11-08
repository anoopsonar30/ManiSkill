"""
PickCuboid-v1 is a basic/common task which defaults to using the panda robot.
"""
PICK_CUBOID_CONFIGS = {
    "panda": {
        "cuboid_half_sizes": [0.025, 0.025, 0.02],
        "goal_thresh": 0.03,
        "cuboid_spawn_half_size": 0.1,
        "cuboid_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [
            0.3,
            0,
            0.6,
        ],  # sensor cam is the camera used for visual observation generation
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [
            0.6,
            0.7,
            0.6,
        ],  # human cam is the camera used for human rendering (i.e. eval videos)
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    # "fetch": {
    #     "cube_half_size": 0.02,
    #     "goal_thresh": 0.025,
    #     "cube_spawn_half_size": 0.1,
    #     "cube_spawn_center": (0, 0),
    #     "max_goal_height": 0.3,
    #     "sensor_cam_eye_pos": [0.3, 0, 0.6],
    #     "sensor_cam_target_pos": [-0.1, 0, 0.1],
    #     "human_cam_eye_pos": [0.6, 0.7, 0.6],
    #     "human_cam_target_pos": [0.0, 0.0, 0.35],
    # },
    # "xarm6_robotiq": {
    #     "cube_half_size": 0.02,
    #     "goal_thresh": 0.025,
    #     "cube_spawn_half_size": 0.1,
    #     "cube_spawn_center": (0, 0),
    #     "max_goal_height": 0.3,
    #     "sensor_cam_eye_pos": [0.3, 0, 0.6],
    #     "sensor_cam_target_pos": [-0.1, 0, 0.1],
    #     "human_cam_eye_pos": [0.6, 0.7, 0.6],
    #     "human_cam_target_pos": [0.0, 0.0, 0.35],
    # },
    # "so100": {
    #     "cube_half_size": 0.0125,
    #     "goal_thresh": 0.0125 * 1.25,
    #     "cube_spawn_half_size": 0.05,
    #     "cube_spawn_center": (-0.46, 0),
    #     "max_goal_height": 0.08,
    #     "sensor_cam_eye_pos": [-0.27, 0, 0.4],
    #     "sensor_cam_target_pos": [-0.56, 0, -0.25],
    #     "human_cam_eye_pos": [-0.1, 0.3, 0.4],
    #     "human_cam_target_pos": [-0.46, 0.0, 0.1],
    # },
}
