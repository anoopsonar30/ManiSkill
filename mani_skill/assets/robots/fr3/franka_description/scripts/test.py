# import pybullet as p
# import pybullet_data

# # Connect to GUI
# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# # Load ground plane
# p.loadURDF("plane.urdf")

# # Load FR3
# robot = p.loadURDF(
#     "/home/ankile/franka_description/urdfs/fr3.urdf",
#     [0, 0, 0],
#     useFixedBase=True,
# )

# # Set better camera view
# p.resetDebugVisualizerCamera(
#     cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0.5]
# )

# # Get joint info
# num_joints = p.getNumJoints(robot)
# print(f"Robot has {num_joints} joints")

# # Keep window open
# print("PyBullet GUI opened. Close the window or press Ctrl+C to exit.")
# try:
#     while True:
#         p.stepSimulation()
#         p.getCameraImage(320, 240)  # Keep rendering
# except KeyboardInterrupt:
#     pass

# p.disconnect()


import pybullet as p
import pybullet_data
import time

# Connect to GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load ground and robot
p.loadURDF("plane.urdf")
robot = p.loadURDF(
    "/home/ankile/franka_description/urdfs/fr3.urdf", [0, 0, 0], useFixedBase=True
)

# Get info about all links
num_joints = p.getNumJoints(robot)
print("Link indices and names:")
for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    link_name = info[12].decode("utf-8")
    print(f"Link {i}: {link_name}")

# Find fr3_link8 index (should be around index 16-17)
target_link_index = -1
for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    link_name = info[12].decode("utf-8")
    print(f"Link {i}: {link_name}")
    if link_name == "calibration_target":
        target_link_index = i
        break


print(f"\nfr3_link8 is at index: {target_link_index}")

# Set better camera view
p.resetDebugVisualizerCamera(1.0, 45, -30, [0, 0, 0.5])

# Visualize coordinate frames
while True:
    # Draw coordinate frame for base
    p.addUserDebugLine(
        [0, 0, 0], [0.1, 0, 0], [1, 0, 0], lineWidth=3, lifeTime=0.1
    )  # X-axis (red)
    p.addUserDebugLine(
        [0, 0, 0], [0, 0.1, 0], [0, 1, 0], lineWidth=3, lifeTime=0.1
    )  # Y-axis (green)
    p.addUserDebugLine(
        [0, 0, 0], [0, 0, 0.1], [0, 0, 1], lineWidth=3, lifeTime=0.1
    )  # Z-axis (blue)

    # Draw coordinate frame for fr3_link8 (flange)
    if target_link_index >= 0:
        link_state = p.getLinkState(robot, target_link_index)
        pos = link_state[4]  # World position of link frame
        orn = link_state[5]  # World orientation of link frame

        # Convert quaternion to rotation matrix
        rot_mat = p.getMatrixFromQuaternion(orn)

        # Extract axis vectors
        x_axis = [rot_mat[0], rot_mat[3], rot_mat[6]]
        y_axis = [rot_mat[1], rot_mat[4], rot_mat[7]]
        z_axis = [rot_mat[2], rot_mat[5], rot_mat[8]]

        # Draw axes (scaled to 0.05m = 5cm)
        scale = 0.05
        p.addUserDebugLine(
            pos,
            [
                pos[0] + scale * x_axis[0],
                pos[1] + scale * x_axis[1],
                pos[2] + scale * x_axis[2],
            ],
            [1, 0, 0],
            lineWidth=3,
            lifeTime=0.1,
        )  # X-axis (red)
        p.addUserDebugLine(
            pos,
            [
                pos[0] + scale * y_axis[0],
                pos[1] + scale * y_axis[1],
                pos[2] + scale * y_axis[2],
            ],
            [0, 1, 0],
            lineWidth=3,
            lifeTime=0.1,
        )  # Y-axis (green)
        p.addUserDebugLine(
            pos,
            [
                pos[0] + scale * z_axis[0],
                pos[1] + scale * z_axis[1],
                pos[2] + scale * z_axis[2],
            ],
            [0, 0, 1],
            lineWidth=3,
            lifeTime=0.1,
        )  # Z-axis (blue)

        p.addUserDebugText(
            "X",
            [
                pos[0] + scale * x_axis[0],
                pos[1] + scale * x_axis[1],
                pos[2] + scale * x_axis[2],
            ],
            [1, 0, 0],
            textSize=1.5,
            lifeTime=0.1,
        )
        p.addUserDebugText(
            "Y",
            [
                pos[0] + scale * y_axis[0],
                pos[1] + scale * y_axis[1],
                pos[2] + scale * y_axis[2],
            ],
            [0, 1, 0],
            textSize=1.5,
            lifeTime=0.1,
        )
        p.addUserDebugText(
            "Z",
            [
                pos[0] + scale * z_axis[0],
                pos[1] + scale * z_axis[1],
                pos[2] + scale * z_axis[2],
            ],
            [0, 0, 1],
            textSize=1.5,
            lifeTime=0.1,
        )

    p.stepSimulation()
    time.sleep(0.01)
