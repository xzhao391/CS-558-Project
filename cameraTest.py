import numpy as np
import pybullet as p
import pybullet_data
import math

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -10)
p.setTimeStep(0.01)

# Add plane
plane_id = p.loadURDF("plane.urdf")

# Add kuka bot
start_pos = [0, 0, 0.001]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
kuka_id = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation)

fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
distance = 100000
img_w, img_h = 1000, 1000
def kuka_camera():
    # # Center of mass position and orientation (of link-7)
    # com_p, com_o = p.getLinkState(kuka_id, 6, computeForwardKinematics=True)[:2]
    # rot_matrix = p.getMatrixFromQuaternion(com_o)
    # rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # # Initial vectors
    # init_camera_vector = (0, 0, 1) # z-axis
    # init_up_vector = (0, 1, 0) # y-axis
    # # Rotated vectors
    # camera_vector = rot_matrix.dot(init_camera_vector)
    # up_vector = rot_matrix.dot(init_up_vector)
    # view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
    # img = p.getCameraImage(1000, 1000, view_matrix, projection_matrix)
    # return img

    agent_pos, agent_orn = p.getLinkState(kuka_id, 6, computeForwardKinematics=True)[:2]
    yaw = p.getEulerFromQuaternion(agent_orn)[-1]
    xA, yA, zA = agent_pos
    zA = zA + 0.3 # make the camera a little higher than the robot

    # compute focusing point of the camera
    xB = xA + math.cos(yaw) * distance
    yB = yA + math.sin(yaw) * distance
    zB = zA

    view_matrix = p.computeViewMatrix(
                        cameraEyePosition=[xA, yA, zA],
                        cameraTargetPosition=[xB, yB, zB],
                        cameraUpVector=[0, 0, 1.0]
                    )

    projection_matrix = p.computeProjectionMatrixFOV(
                            fov=90, aspect=1.5, nearVal=0.02, farVal=10)

    imgs = p.getCameraImage(img_w, img_h,
                            view_matrix,
                            projection_matrix, shadow=True,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return imgs

# Main loop
while True:
    p.stepSimulation()
    kuka_camera()