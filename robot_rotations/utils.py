from matplotlib import pyplot as plt
import numpy as np
import math

#Used for some tests in component_1
def create_se2_matrix(theta, tx, ty):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    t = np.array([tx, ty])
    se2 = np.array([
        [R[0, 0], R[0, 1], t[0]],
        [R[1, 0], R[1, 1], t[1]],
        [0, 0, 1]
    ])
    
    return se2

def create_se3_matrix(theta_x, theta_y, theta_z, tx, ty, tz):
    R_x = create_R_x(theta_x)
    
    R_y = create_R_y(theta_y)
    
    R_z = create_R_z(theta_z)
    
    R = R_z @ R_y @ R_x

    se3 = np.array([
        [R[0, 0], R[0, 1], R[0, 2], tx],
        [R[1, 0], R[1, 1], R[1, 2], ty],
        [R[2, 0], R[2, 1], R[2, 2], tz],
        [0,       0,       0,       1]
    ])
    
    return se3

#Helper 3D rotation matrices
def create_R_x(theta_x):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
def create_R_y(theta_y):
    return np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
def create_R_z(theta_z):
    return np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
#From textbook not used (works lol)
def uniform_random_euler_angles():
    theta = 2 * np.pi * np.random.rand() - np.pi
    
    phi = np.arccos(1 - 2 * np.random.rand()) + (np.pi / 2)
    
    if np.random.rand() < 0.5:
        if phi < np.pi:
            phi = phi + np.pi
        else:
            phi = phi - np.pi
    eta = 2 * np.pi * np.random.rand() - np.pi
    
    return theta, phi, eta

#Func to plot arm
def plot_arm_path(arm_path, link1_length=2, link2_length=1.5, saveString=""):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    for i, (theta1, theta2) in enumerate(arm_path):
        x1 = link1_length * np.cos(theta1)
        y1 = link1_length * np.sin(theta1)

        x2 = x1 + link2_length * np.cos(theta1 + theta2)
        y2 = y1 + link2_length * np.sin(theta1 + theta2)

        if i == 0:  # Start pose
            ax.plot([0, x1, x2], [0, y1, y2], marker='o', color='green', label='Start Pose', zorder=5)
        elif i == len(arm_path) - 1:  # End pose
            ax.plot([0, x1, x2], [0, y1, y2], marker='o', color='red', label='End Pose', zorder=5)
        else:  # Intermediate poses
            ax.plot([0, x1, x2], [0, y1, y2], marker='o', color='blue', alpha=0.5)

    ax.set_aspect('equal')
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.grid(True)
    plt.title('2-Link Arm Path')

    plt.legend()
    plt.show()
    plt.savefig(saveString)

# Helper func to find x,y positions of the arm's joints
def forward_kinematics(theta1, theta2, L1=2, L2=1.5):
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    
    return (x1, y1), (x2, y2)


    