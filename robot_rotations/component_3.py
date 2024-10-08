import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

#region interpolate_rigid_body
def interpolate_rigid_body(start_pose: np.array, goal_pose: np.array) -> np.array:
    path = [start_pose]
    n = 20
    incrementX = (goal_pose[0]- start_pose[0]) / n 
    incrementY = (goal_pose[1]-start_pose[1]) / n
    incrementTheta = (goal_pose[2]-start_pose[2]) / n
    poseX , poseY, poseTheta = start_pose

    for i in range(n):
        poseX += incrementX
        poseY += incrementY
        poseTheta += incrementTheta
        pose = [poseX,poseY,poseTheta]
        path.append(pose)
    path = np.array(path)
    return path

#Test interpolate_rigid_body
start_pose_interpolate_rigid_body = [-6, 5, 0]
goal_pose = [7, -3, math.pi/2]
interpolate_path = interpolate_rigid_body(start_pose_interpolate_rigid_body, goal_pose)
#Plotting
xvalues = interpolate_path[:,0]
yvalues = interpolate_path[:,1]
thetavalues = interpolate_path[:,2]

plt.figure(figsize=(10,10))
ax = plt.gca()

ax.plot(xvalues, yvalues, label="Path")

dx = np.cos(thetavalues)
dy = np.sin(thetavalues)
ax.quiver(xvalues, yvalues, dx, dy, angles = 'xy', scale_units = 'xy', scale = 0.5, color='blue',alpha=0.7, lw=0.3)
ax.set_aspect('equal')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)

ax.scatter(xvalues[0], yvalues[0], color='green', s=100, label='Start', zorder = 5)
ax.scatter(xvalues[-1], yvalues[-1], color='red', s=100, label='End', zorder = 5)

plt.legend()
plt.grid(True)
plt.show()
plt.savefig('interpolate_rigid_body_Test.png')

#endregion

#region forward_propagate_rigid_body
def forward_propagate_rigid_body(start_pose: np.array, plan: list) ->np.array:
    path = [start_pose]
    newX, newY, newTheta = start_pose

    for velocity_tuple in plan:
        v_X = velocity_tuple[0][0]  # Velocity in X direction
        v_Y = velocity_tuple[0][1]  # Velocity in Y direction
        v_Theta = velocity_tuple[0][2]  # Angular velocity (Theta)
        duration = velocity_tuple[1]  # Duration for this velocity plan

        for i in range(duration):
            newX += v_X
            newY += v_Y
            newTheta += v_Theta
            pose = [newX, newY, newTheta]
            path.append(pose)
    
    path = np.array(path)
    return path


T = [([[2,2,math.pi/2],5]), ([[-2,0,math.pi / 4],5])]
start_pose_forward_propagate_rigid_body = [-6, -4, 0]
forward_path = forward_propagate_rigid_body(start_pose_forward_propagate_rigid_body,T)
#Plotting
xvalues = forward_path[:,0]
yvalues = forward_path[:,1]
thetavalues = forward_path[:,2]

plt.figure(figsize=(10,10))
ax = plt.gca()

ax.plot(xvalues, yvalues, label="Path")

dx = np.cos(thetavalues)
dy = np.sin(thetavalues)
ax.quiver(xvalues, yvalues, dx, dy, angles = 'xy', scale_units = 'xy', scale = 0.5, color='blue')
ax.set_aspect('equal')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)

ax.scatter(xvalues[0], yvalues[0], color='green', s=100, label='Start', zorder = 5)
ax.scatter(xvalues[-1], yvalues[-1], color='red', s=100, label='End', zorder = 5)

plt.legend()
plt.grid(True)
plt.show()
plt.savefig('forward_propagate_rigid_body_Test.png')

#endregion 

#region visualize_path
ani = None
def visualize_path(path):
    global ani
    fig, ax = plt.subplots(figsize=(10, 10))

    x_vals = [pose[0] for pose in path]
    y_vals = [pose[1] for pose in path]
    theta_vals = [pose[2] for pose in path]

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')

    ax.plot(x_vals, y_vals, label="Path", color='gray', linestyle='--')

    robot_rect = Rectangle((0, 0), 0.5, 0.3, fc='red', ec='black', angle=0)
    ax.add_patch(robot_rect)

    direction_arrow = ax.quiver([],[],[],[], scale=20, color='blue')

    def init():
            robot_rect.set_xy([x_vals[0], y_vals[0]]) 
            robot_rect.angle = np.degrees(theta_vals[0]) 

            dx = np.cos(theta_vals[0]) * 0.5
            dy = np.sin(theta_vals[0]) * 0.5
            direction_arrow.set_offsets([x_vals[0],y_vals[0]])
            direction_arrow.set_UVC(dx, dy)

            return robot_rect, direction_arrow

    def update(frame):
            robot_rect.set_xy([x_vals[frame], y_vals[frame]]) 
            robot_rect.angle = np.degrees(theta_vals[frame]) 

            dx = np.cos(theta_vals[frame]) * 0.5
            dy = np.sin(theta_vals[frame]) * 0.5
            direction_arrow.set_offsets([x_vals[frame], y_vals[frame]])
            direction_arrow.set_UVC(dx, dy)

            return robot_rect, direction_arrow


    ani = FuncAnimation(fig, update, frames=len(path), init_func=init, interval=1000, repeat=True)

    ani.save('rigid_body_path_test.gif', writer='imagemagick')

    plt.legend()
    plt.grid(True)
    plt.show()

visualize_path(forward_path)

#endregion

