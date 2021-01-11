import numpy as np
from scipy.spatial.transform import Rotation
from numpy.linalg import inv
class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2
        # print('3')

        # STUDENT CODE HERE

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE

        # K_d=np.diag(np.array([7,7,10]))  #define parameters
        # K_p=np.diag(np.array([9,9,20]))
        # K_R=np.diag(np.array([70,70,70]))
        # K_w=np.diag(np.array([8,8,8]))  

        # K_d=np.diag(np.array([4,4,20]))  #define parameters
        # K_p=np.diag(np.array([5,5,40]))
        # K_R=np.diag(np.array([70,70,70]))
        # K_w=np.diag(np.array([8,8,8]))  

        K_d=np.diag(np.array([8.5,8.5,8]))  #define parameters
        K_p=np.diag(np.array([5,5,5]))
        K_R=np.diag(np.array([2500,2500,400]))
        K_w=np.diag(np.array([60,60,50]))  

        acc_des=flat_output['x_ddot']-np.dot(K_d,(state['v']-flat_output['x_dot']))-np.dot(K_p,(state['x']-flat_output['x']))
              #commanded acceleration
        F_des=np.array(self.mass*acc_des+[0,0,self.mass*self.g]) # total commanded force
        R=Rotation.from_quat(state['q']).as_matrix()  #rotation matrix
        b3=np.dot(R,[0,0,1])  #quadrotor’s axis
        u_1=np.dot(b3,F_des)  #input u1
        b3_des=F_des/np.linalg.norm(F_des)  #b3_des should be oriented along the desired thrust
        a_psi=np.array([np.cos(flat_output['yaw']),np.sin(flat_output['yaw']),0])  #defines the yaw direction in the plane (a1, a2) plane
        b2_des=np.cross(b3_des,a_psi)/np.linalg.norm(np.cross(b3_des,a_psi))
        R_des=np.array([np.cross(b2_des,b3_des),b2_des,b3_des]).T  #desired rotation matrix
        e_R_beforev=0.5*(np.dot(R_des.T,R)-np.dot(R.T,R_des))
        e_R=[e_R_beforev[2,1],e_R_beforev[0,2],e_R_beforev[1,0]] # error vector
        e_w=state['w']
        u_2=np.dot(self.inertia,(-np.dot(K_R,e_R)-np.dot(K_w,e_w)))  #the control input
        u=np.array([u_1,u_2[0],u_2[1],u_2[2]])
        gamma=self.k_drag/self.k_thrust
        matrix_from_F_to_u=np.array([[1,1,1,1],[0,self.arm_length,0,-self.arm_length],\
            [-self.arm_length,0,self.arm_length,0],[gamma,-gamma,gamma,-gamma]])
        F=np.dot(inv(matrix_from_F_to_u),u)
        for i in range(0,4):  #whenF[i]<0 F[i]=0
            if F[i]<0:
                F[i]=0
        cmd_motor_speeds=np.sqrt(F/self.k_thrust)
        cmd_thrust=u_1
        # print(cmd_thrust)
        cmd_moment=[u_2[0],u_2[1],u_2[2]]
        cmd_q=Rotation.from_matrix(R_des).as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
