import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import create_R_z, create_R_y, create_R_x
#Function
def random_rotation_matrix(naive: bool) -> np.array :
    if (naive):
        yaw = np.random.uniform(0, 2 * np.pi)   # Rotation around z-axis
        pitch = np.random.uniform(0, np.pi) # Rotation around y-axis
        roll = np.random.uniform(0, 2 * np.pi)  # Rotation around x-axis
        R_x = create_R_x(roll)
        R_y = create_R_y(pitch)
        R_z = create_R_z(yaw)
        R = R_z @ R_y @ R_x
        return R
    else:
        x1, x2, x3 = np.random.uniform(0, 1, 3)
        theta = 2*math.pi * x1
        rho = 2*math.pi * x2
        z = x3
        v= np.array([
            [np.cos(rho)*math.sqrt(z)],[np.sin(rho)*math.sqrt(z)],[math.sqrt(1-z)]
        ])
        VV_T = 2 * np.dot(v, v.T) - np.identity(3)
        R_z = create_R_z(theta)
        M = np.dot(VV_T, R_z)
        return M
    

#Helper Function to graph Random Rotations
def generate_rotation_visualization(num_samples, naive, pngNum):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    v0 = np.array([0, 0, 1])  # North Pole
    epsilon = 0.1
    v1 = np.array([0, epsilon, 0]) + v0 

    for _ in range(num_samples):
        rotation_matrix = random_rotation_matrix(naive=naive)

        v0_prime = rotation_matrix @ v0
        v1_prime = rotation_matrix @ (v1 - v0)

        ax.quiver(v0_prime[0], v0_prime[1], v0_prime[2], 
          v1_prime[0], v1_prime[1], v1_prime[2], 
          color='red', alpha=0.7, lw=0.3)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    
    plt.show()
    if pngNum == 1:
        plt.savefig('naiveSampling.png')
    if pngNum == 2:
        plt.savefig('uniformSampling.png')


#Naive
generate_rotation_visualization(num_samples=1000, naive=True, pngNum=1)
#Uniform
generate_rotation_visualization(num_samples=1000, naive=False, pngNum=2)