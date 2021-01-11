import inspect
import json
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import time

from flightsim.animate import animate
from flightsim.axes3ds import Axes3Ds
from flightsim.crazyflie_params import quad_params
from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World

from proj1_3.code.occupancy_map import OccupancyMap
from proj1_3.code.se3_control import SE3Control
from proj1_3.code.world_traj import WorldTraj

# Improve figure display on high DPI screens.
# mpl.rcParams['figure.dpi'] = 200

# Choose a test example file. You should write your own example files too!
# filename = '../util/test_window.json'
# filename = '../util/test_maze.json'
# filename = '../util/test_over_under.json'

filename = '../util/test_lab_1.json'
# Load the test example.
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
world = World.from_file(file)          # World boundary and obstacles.
start  = world.world['start']          # Start point, shape=(3,)
goal   = world.world['goal']           # Goal point, shape=(3,)

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

# Help debug issues you may encounter with your choice of resolution and margin
# by plotting the occupancy grid after inflation by margin. THIS IS VERY SLOW!!
# fig = plt.figure('world')
# ax = Axes3Ds(fig)
# world.draw(ax)
# fig = plt.figure('occupancy grid')
# ax = Axes3Ds(fig)
# resolution = SET YOUR RESOLUTION HERE
# margin = SET YOUR MARGIN HERE
# oc = OccupancyMap(world, resolution, margin)
# oc.draw(ax)
# ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=10, markeredgewidth=3, markerfacecolor='none')
# ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=10, markeredgewidth=3, markerfacecolor='none')
# plt.show()

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
fig = plt.figure('A* Path, Waypoints, and Trajectory')
ax = Axes3Ds(fig)
world.draw(ax)
ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
if hasattr(my_world_traj, 'path'):
    if my_world_traj.path is not None:
        world.draw_line(ax, my_world_traj.path, color='red', linewidth=1)
else:
    print("Have you set \'self.path\' in WorldTraj.__init__?")
if hasattr(my_world_traj, 'points'):
    if my_world_traj.points is not None:
        world.draw_points(ax, my_world_traj.points, color='purple', markersize=8)
else:
    print("Have you set \'self.points\' in WorldTraj.__init__?")
world.draw_line(ax, flat['x'], color='black', linewidth=2)
ax.legend(handles=[
    Line2D([], [], color='red', linewidth=1, label='Dense A* Path'),
    Line2D([], [], color='purple', linestyle='', marker='.', markersize=8, label='Sparse Waypoints'),
    Line2D([], [], color='black', linewidth=2, label='Trajectory')],
    loc='upper right')

# Position and Velocity vs. Time
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position vs Time')
x = state['x']
x_des = flat['x']
ax = axes[0]
ax.plot(sim_time, x_des[:,0], 'r', sim_time, x_des[:,1], 'g', sim_time, x_des[:,2], 'b')
ax.plot(sim_time, x[:,0], 'r.',    sim_time, x[:,1], 'g.',    sim_time, x[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('Position')
ax.grid('major')
ax.set_title('Position, derivatives')
v = state['v']
v_des = flat['x_dot']
ax = axes[1]
ax.plot(sim_time, v_des[:,0], 'r', sim_time, v_des[:,1], 'g', sim_time, v_des[:,2], 'b')
ax.plot(sim_time, v[:,0], 'r.',    sim_time, v[:,1], 'g.',    sim_time, v[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('Velocity')
ax.set_xlabel('time, s')
ax.grid('major')
# Acceleration and Jerk vs. Time
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='acc vs Time')
# acc = state['x']
acc_des = flat['x_ddot']
ax = axes[0]
# ax.plot(sim_time, acc_des[:,0], 'r', sim_time, acc_des[:,1], 'g', sim_time, x_des[:,2], 'b')
ax.plot(sim_time, acc_des[:,0], 'r.',    sim_time, acc_des[:,1], 'g.',    sim_time, acc_des[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('Acceleration, m')
ax.grid('major')
# v = state['v']
j_des = flat['x_dddot']
ax = axes[1]
# ax.plot(sim_time, v_des[:,0], 'r', sim_time, v_des[:,1], 'g', sim_time, v_des[:,2], 'b')
ax.plot(sim_time, j_des[:,0], 'r.',    sim_time, j_des[:,1], 'g.',    sim_time, j_des[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('Jerk')
ax.set_xlabel('time, s')
ax.grid('major')

# SNAP
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='snap vs Time')
# acc = state['x']
s_des = flat['x_ddddot']
ax = axes[0]
# ax.plot(sim_time, acc_des[:,0], 'r', sim_time, acc_des[:,1], 'g', sim_time, x_des[:,2], 'b')
ax.plot(sim_time, s_des[:,0], 'r.',    sim_time, s_des[:,1], 'g.',    sim_time, s_des[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('Snap')
ax.grid('major')


# Orientation and Angular Velocity vs. Time
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time')
q_des = control['cmd_q']
q = state['q']
ax = axes[0]
ax.plot(sim_time, q_des[:,0], 'r', sim_time, q_des[:,1], 'g', sim_time, q_des[:,2], 'b', sim_time, q_des[:,3], 'k')
ax.plot(sim_time, q[:,0], 'r.',    sim_time, q[:,1], 'g.',    sim_time, q[:,2], 'b.',    sim_time, q[:,3],     'k.')
ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
ax.set_ylabel('quaternion')
ax.set_xlabel('time, s')
ax.grid('major')
w = state['w']
ax = axes[1]
ax.plot(sim_time, w[:,0], 'r.', sim_time, w[:,1], 'g.', sim_time, w[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('angular velocity, rad/s')
ax.set_xlabel('time, s')
ax.grid('major')

# Commands vs. Time
(fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Commands vs Time')
s = control['cmd_motor_speeds']
ax = axes[0]
ax.plot(sim_time, s[:,0], 'r.', sim_time, s[:,1], 'g.', sim_time, s[:,2], 'b.', sim_time, s[:,3], 'k.')
ax.legend(('1', '2', '3', '4'), loc='upper right')
ax.set_ylabel('motor speeds, rad/s')
ax.grid('major')
ax.set_title('Commands')
M = control['cmd_moment']
ax = axes[1]
ax.plot(sim_time, M[:,0], 'r.', sim_time, M[:,1], 'g.', sim_time, M[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('moment, N*m')
ax.grid('major')
T = control['cmd_thrust']
ax = axes[2]
ax.plot(sim_time, T, 'k.')
ax.set_ylabel('thrust, N')
ax.set_xlabel('time, s')
ax.grid('major')

# 3D Paths
fig = plt.figure('3D Path')
ax = Axes3Ds(fig)
world.draw(ax)
ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
world.draw_line(ax, flat['x'], color='black', linewidth=2)
world.draw_points(ax, state['x'], color='blue', markersize=4)
if collision_pts.size > 0:
    ax.plot(collision_pts[0,[0]], collision_pts[0,[1]], collision_pts[0,[2]], 'rx', markersize=36, markeredgewidth=4)
ax.legend(handles=[
    Line2D([], [], color='black', linewidth=2, label='Trajectory'),
    Line2D([], [], color='blue', linestyle='', marker='.', markersize=4, label='Flight')],
    loc='upper right')


# Animation (Slow)
#
# Instead of viewing the animation live, you may provide a .mp4 filename to save.

R = Rotation.from_quat(state['q']).as_dcm()
animate(sim_time, state['x'], R, world=world, filename=None, show_axes=True)



plt.show()
