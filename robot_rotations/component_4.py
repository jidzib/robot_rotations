import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import plot_arm_path, forward_kinematics
#region interpolate_arm
def interpolate_arm(start_angles: np.array, goal_angles: np.array) -> np.array:
    path = [start_angles]
    n = 20 
    incrementTheta1 = (goal_angles[0] - start_angles[0]) / n
    incrementTheta2 = (goal_angles[1] - start_angles[1]) / n

    theta1, theta2 = start_angles

    for i in range(n):
        theta1 += incrementTheta1
        theta2 += incrementTheta2
        path.append([theta1, theta2])
    
    path = np.array(path)
    return path

#Testing
start_angles = np.array([0, np.pi / 4])  # Initial 
goal_angles = np.array([np.pi/2, np.pi / 2])  # Final

arm_path = interpolate_arm(start_angles, goal_angles)
plot_arm_path(arm_path, saveString="interpolate_arm_Test.png")

#endregion

#region forward_propagate_arm
def forward_propagate_arm(start_pose: np.array, plan: list) -> np.array:
    path = [start_pose]
    theta1, theta2 = start_pose

    for velocity_tuple in plan:
        v_theta1 = velocity_tuple[0][0] 
        v_theta2 = velocity_tuple[0][1] 
        duration = velocity_tuple[1] 

        for i in range(duration):
            theta1 += v_theta1
            theta2 += v_theta2
            pose = [theta1, theta2]
            path.append(pose)

    path = np.array(path)
    return path

start_pose = np.array([0, 0])  # Initial joint angles

plan = [
    [(np.pi/4, 0), 4], 
    [(0, np.pi/4), 4] 
]

arm_path = forward_propagate_arm(start_pose, plan)
#Plotting
plot_arm_path(arm_path, saveString="forward_propagate_arm_Test.png")

#endregion

#region visualize_arm_path

def visualize_arm_path(path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')

    link1, = ax.plot([], [], 'o-', lw=4, color='blue', label="Link 1")
    link2, = ax.plot([], [], 'o-', lw=4, color='red', label="Link 2")

    def init():
        link1.set_data([], [])
        link2.set_data([], [])
        return link1, link2

    def update(frame):
        theta1, theta2 = path[frame]

        (x1, y1), (x2, y2) = forward_kinematics(theta1, theta2)

        link1.set_data([0, x1], [0, y1])
        link2.set_data([x1, x2], [y1, y2])

        return link1, link2

    ani = FuncAnimation(fig, update, frames=len(path), init_func=init, interval=200, blit=True)
    plt.legend()
    plt.grid(True)
    plt.show()
    ani.save('movementOfaArm_test.gif', writer='imagemagick')
#Testing
start_pose = np.array([0, 0])
plan = [
    ([np.pi/4, 0], 4),  
    ([0, np.pi/4], 4), 
]

path = forward_propagate_arm(start_pose, plan)

visualize_arm_path(path)

#endregion
