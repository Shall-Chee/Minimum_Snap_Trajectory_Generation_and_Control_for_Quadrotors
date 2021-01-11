import inspect
import json
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import time

from scipy.spatial.transform import Rotation as R

from flightsim.animate import animate
from flightsim.axes3ds import Axes3Ds
from flightsim.crazyflie_params import quad_params
from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World

from proj1_4.code.occupancy_map import OccupancyMap
from proj1_4.code.se3_control import SE3Control
from proj1_4.code.world_traj import WorldTraj

import rosbag

# Put the path to your bagfile here! It is easiest to put the bag files in the
# same directory you run the script in.
bagfile = 'map1.bag'

# Improve figure display on high DPI screens.
# mpl.rcParams['figure.dpi'] = 200

# Choose a test example file. You should write your own example files too!
filename = '../util/test_lab_1.json'

# Load the test example.
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
world = World.from_file(file)          # World boundary and obstacles.
start  = world.world['start']          # Start point, shape=(3,)
goal   = world.world['goal']           # Goal point, shape=(3,)

# fig = plt.figure()
# ax = Axes3Ds(fig)
# world.draw(ax)
# plt.show()

# This object defines the quadrotor dynamical model and should not be changed.
quadrotor = Quadrotor(quad_params)
robot_radius = 0.25

# Your SE3Control object (from project 1-1).
my_se3_control = SE3Control(quad_params)

# Your MapTraj object. This behaves like the trajectory function you wrote in
# project 1-1, except instead of giving it waypoints you give it the world,
# start, and goal.
planning_start_time = time.time()
my_world_traj = WorldTraj(world, start, goal)
planning_end_time = time.time()

# Set simulation parameters.
t_final = 60
initial_state = {'x': start,
                 'v': (0, 0, 0),
                 'q': (0, 0, 0, 1), # [i,j,k,w]
                 'w': (0, 0, 0)}

# Perform simulation.
#
# This function performs the numerical simulation.  It returns arrays reporting
# the quadrotor state, the control outputs calculated by your controller, and
# the flat outputs calculated by you trajectory.

print()
print('Simulate.')
(sim_time, state, control, flat, exit) = simulate(initial_state,
                                              quadrotor,
                                              my_se3_control,
                                              my_world_traj,
                                              t_final)
print(exit.value)

# Load the flight data from bag.
rosstate = {}
roscontrol = {}
with rosbag.Bag(bagfile) as bag:
    odometry = np.array([
        np.array([t.to_sec() - bag.get_start_time(),
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
        msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z,
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
            for (_, msg, t) in bag.read_messages(topics=['odom'])])
    vicon_time = odometry[:, 0]
    rosstate['x'] = odometry[:, 1:4]
    rosstate['v'] = odometry[:, 4:7]
    rosstate['w'] = odometry[:, 7:10]
    rosstate['q'] = odometry[:, 10:15]

    commands = np.array([
        np.array([t.to_sec() - bag.get_start_time(),
        msg.linear.z, msg.linear.y, msg.linear.x])
            for (_, msg, t) in bag.read_messages(topics=['so3cmd_to_crazyflie/cmd_vel_fast'])])
    command_time = commands[:, 0]
    c1 = -0.6709 # Coefficients to convert thrust PWM to Newtons.
    c2 = 0.1932
    c3 = 13.0652
    roscontrol['cmd_thrust'] = (((commands[:, 1]/60000 - c1) / c2)**2 - c3)/1000*9.81
    roscontrol['cmd_q'] = R.from_euler('zyx', np.transpose([commands[:, 2], commands[:, 3],
                                                         np.zeros(commands[:, 2].shape)]), degrees=True).as_quat()

# Print results.
#
# Only goal reached, collision test, and flight time are used for grading.

collision_pts = world.path_collisions(state['x'], robot_radius)

stopped_at_goal = (exit == ExitStatus.COMPLETE) and np.linalg.norm(state['x'][-1] - goal) <= 0.05
no_collision = collision_pts.size == 0
flight_time = sim_time[-1]
flight_distance = np.sum(np.linalg.norm(np.diff(state['x'], axis=0),axis=1))
planning_time = planning_end_time - planning_start_time

print()
print(f"Results:")
print(f"  No Collision:    {'pass' if no_collision else 'FAIL'}")
print(f"  Stopped at Goal: {'pass' if stopped_at_goal else 'FAIL'}")
print(f"  Flight time:     {flight_time:.1f} seconds")
print(f"  Flight distance: {flight_distance:.1f} meters")
print(f"  Planning time:   {planning_time:.1f} seconds")
if not no_collision:
    print()
    print(f"  The robot collided at location {collision_pts[0]}!")

# Plot Results
#
# You will need to make plots to debug your quadrotor.
# Here are some example of plots that may be useful.

# Visualize the original dense path from A*, your sparse waypoints, and the
# smooth trajectory.
fig = plt.figure('Waypoints, Trajectory, and Actual Path')
ax = Axes3Ds(fig)
world.draw(ax)
ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
# if hasattr(my_world_traj, 'path'):
#     if my_world_traj.path is not None:
#         world.draw_line(ax, my_world_traj.path, color='red', linewidth=1)
# else:
#     print("Have you set \'self.path\' in WorldTraj.__init__?")
if hasattr(my_world_traj, 'points'):
    if my_world_traj.points is not None:
        world.draw_points(ax, my_world_traj.points, color='purple', markersize=8)
else:
    print("Have you set \'self.points\' in WorldTraj.__init__?")
world.draw_line(ax, flat['x'], color='black', linewidth=2)
world.draw_line(ax, rosstate['x'], color='green', linewidth=2)
ax.legend(handles=[
    # Line2D([], [], color='red', linewidth=1, label='Dense A* Path'),
    Line2D([], [], color='purple', linestyle='', marker='.', markersize=8, label='Sparse Waypoints'),
    Line2D([], [], color='black', linewidth=2, label='Trajectory'),
    Line2D([], [], color='green', linewidth=2, label='Actual Path')],
    loc='lower right')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_title('Map 1; Code 1 - Waypoints, Trajectory, and Actual Path')

# Real Position and Velocity vs. Time
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position vs Time')
x_in = rosstate['x']
t1 = 650
t2 = 2350
x = x_in[t1:t2,:]
v_time = vicon_time[t1:t2] - vicon_time[t1]
x_des = flat['x']
ax = axes[0]
ax.plot(sim_time, x_des[:,0], 'r', sim_time, x_des[:,1], 'g', sim_time, x_des[:,2], 'b')
ax.plot(v_time, x[:,0], 'r.',    v_time, x[:,1], 'g.',    v_time, x[:,2], 'b.')
# ax.legend(('x', 'y', 'z'), loc='upper right')
ax.legend(handles=[
    Line2D([], [], color='red', linestyle='', marker='.', markersize=6, label='Actual x'),
    Line2D([], [], color='green', linestyle='', marker='.', markersize=6, label='Actual y'),
    Line2D([], [], color='blue', linestyle='', marker='.', markersize=6, label='Actual z'),
    Line2D([], [], color='red', linewidth=2, label='Trajectory x'),
    Line2D([], [], color='green', linewidth=2, label='Trajectory y'),
    Line2D([], [], color='blue', linewidth=2, label='Trajectory z')],
    loc='lower right')
ax.set_ylabel('position (m)')
ax.grid('major')
ax.set_title('Map1; Code 1 - Position and Velocity')
v_in = rosstate['v']
v = v_in[t1:t2,:]
v_des = flat['x_dot']
ax = axes[1]
ax.plot(sim_time, v_des[:,0], 'r', sim_time, v_des[:,1], 'g', sim_time, v_des[:,2], 'b')
ax.plot(v_time, v[:,0], 'r.',    v_time, v[:,1], 'g.',    v_time, v[:,2], 'b.')
# ax.legend(('x', 'y', 'z'), loc='upper right')
ax.legend(handles=[
    Line2D([], [], color='red', linestyle='', marker='.', markersize=6, label='Actual x'),
    Line2D([], [], color='green', linestyle='', marker='.', markersize=6, label='Actual y'),
    Line2D([], [], color='blue', linestyle='', marker='.', markersize=6, label='Actual z'),
    Line2D([], [], color='red', linewidth=2, label='Trajectory x'),
    Line2D([], [], color='green', linewidth=2, label='Trajectory y'),
    Line2D([], [], color='blue', linewidth=2, label='Trajectory z')],
    loc='upper right')
ax.set_ylabel('velocity (m/s)')
ax.set_xlabel('time (s)')
ax.grid('major')

plt.show()
