import numpy as np

from proj1_3.code.graph_search import graph_search
from proj1_3.code.occupancy_map import OccupancyMap # Recommended.

# from graph_search import graph_search
# from occupancy_map import OccupancyMap 

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.125, 0.125, 0.125])
        self.margin = 0.3

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        # print('4')
        self.path = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1,3)) # shape=(n_pts,3)


        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        self.time_nodes=[0]
        self.direction=[0,0,0]
        self.acc_value = 1   #the acceleration is a constant 4-55.37 3-43.6 4.8-10 5.5-58.36 
        #6-58.79 10-60 20-86.7 55-fail-10 100-fail-0 35-0 25-19 20.5-50 20.2-50 20.1-86.3 18-20 19-81.7 19.5-50
        # self.points_list=points
        # print(self.path)
        self.points_list=self.path

        print((self.points_list).shape[0])
        for i in range((self.points_list).shape[0]-1):
            self.time_nodes.append(self.time_nodes[-1]+2*np.sqrt(np.linalg.norm(self.points_list[i+1]-self.points_list[i])/self.acc_value))  #get time_nodes, each time section is 2*length/acceleration
            self.direction=np.vstack((self.direction,(self.points_list[i+1]-self.points_list[i])/np.linalg.norm(self.points_list[i+1]-self.points_list[i]))) #get direction, norm is 1
        
        
        # STUDENT CODE HERE
        

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        n = len(self.points_list)-1
        # p=Ac
        A = np.zeros((6*n,6*n,3))
        c = np.zeros((6*n,3))
        b = np.zeros((6*n,3))
        # self.points_list = np.array(self.points_list)
        # print(self.time_nodes)
        # snap continue
        for i in range(3):
            cons = -1
            for j in range(n-1):  #q
                # print(j)
                cons = cons+1
                A[cons,6*j:6*(j+1),i] = [self.time_nodes[j+1]**5,self.time_nodes[j+1]**4,self.time_nodes[j+1]**3,self.time_nodes[j+1]**2,self.time_nodes[j+1],1]
                b[cons,i] = self.points_list[j+1,i]
                cons = cons+1
                A[cons,6*(j+1):6*(j+2),i] = [self.time_nodes[j+1]**5,self.time_nodes[j+1]**4,self.time_nodes[j+1]**3,self.time_nodes[j+1]**2,self.time_nodes[j+1],1]
                b[cons,i] = self.points_list[j+1,i]
                # print(self.points_list[j+1,i])
                # print(b[cons,i])
            # print(A)
            for j in range(n-1):  # velocity continue
                cons = cons+1
                A[cons,6*j:6*(j+1),i] = [5*self.time_nodes[j+1]**4,4*self.time_nodes[j+1]**3,3*self.time_nodes[j+1]**2,2*self.time_nodes[j+1],1,0]
                # print(A[cons,6*j:6*(j+1),i])
                # print((-1)*np.array([5*self.time_nodes[j]**4,4*self.time_nodes[j]**3,3*self.time_nodes[j]**2,2*self.time_nodes[j],1,0]))
                A[cons,6*(j+1):6*(j+2),i] = [-5*self.time_nodes[j+1]**4,-4*self.time_nodes[j+1]**3,-3*self.time_nodes[j+1]**2,-2*self.time_nodes[j+1],-1,0]
                b[cons,i] = 0
            for j in range(n-1):  # acc continue
                cons = cons+1
                A[cons,6*j:6*(j+1),i] = [20*self.time_nodes[j+1]**3,12*self.time_nodes[j+1]**2,6*self.time_nodes[j+1],2,0,0]
                A[cons,6*(j+1):6*(j+2),i] = [-20*self.time_nodes[j+1]**3,-12*self.time_nodes[j+1]**2,-6*self.time_nodes[j+1],-2,0,0]
                b[cons,i] = 0
            for j in range(n-1):  # jerk continue
                cons = cons+1
                A[cons,6*j:6*(j+1),i] = [60*self.time_nodes[j+1]**2,24*self.time_nodes[j+1],6,0,0,0]
                A[cons,6*(j+1):6*(j+2),i] = [-60*self.time_nodes[j+1]**2,-24*self.time_nodes[j+1],-6,0,0,0]
                b[cons,i] = 0                       
            for j in range(n-1):  # snap continue
                cons = cons+1
                A[cons,6*j:6*(j+1),i] = [120*self.time_nodes[j+1],24,0,0,0,0]
                A[cons,6*(j+1):6*(j+2),i] = [-120*self.time_nodes[j+1],-24,0,0,0,0]
                b[cons,i] = 0      
            #start
            cons = cons+1
            A[cons,0:6,i] = [self.time_nodes[0]**5,self.time_nodes[0]**4,self.time_nodes[0]**3,self.time_nodes[0]**2,self.time_nodes[0],1]
            b[cons,i] = self.points_list[0,i]
            cons = cons+1
            A[cons,0:6,i] = [5*self.time_nodes[0]**4,4*self.time_nodes[0]**3,3*self.time_nodes[0]**2,2*self.time_nodes[0],1,0]
            b[cons,i] = 0
            cons = cons+1
            A[cons,0:6,i] = [20*self.time_nodes[0]**3,12*self.time_nodes[0]**2,6*self.time_nodes[0],2,0,0]
            b[cons,i] = 0

            #goal
            cons = cons+1
            A[cons,6*(n-1):6*n,i] = [self.time_nodes[-1]**5,self.time_nodes[-1]**4,self.time_nodes[-1]**3,self.time_nodes[-1]**2,self.time_nodes[-1],1]
            b[cons,i] = self.points_list[n,i]
            cons = cons+1
            A[cons,6*(n-1):6*n,i] = [5*self.time_nodes[-1]**4,4*self.time_nodes[-1]**3,3*self.time_nodes[-1]**2,2*self.time_nodes[-1],1,0]
            b[cons,i] = 0
            cons = cons+1
            A[cons,6*(n-1):6*n,i] = [20*self.time_nodes[-1]**3,12*self.time_nodes[-1]**2,6*self.time_nodes[-1],2,0,0]
            b[cons,i] = 0

            # print(A)
            # print(b)
            A = np.array(A)
            b = np.array(b)
            c[:,i] = np.linalg.inv(A[:,:,i])@b[:,i]
        # print(A.shape)
        # print('11222')
        # print(b.shape)
        if t<self.time_nodes[-1]:  #if the time is less than the end time_nodes
            for i in range(len(self.points_list)-1):  #find the time in which section
                if t>=self.time_nodes[i] and t<self.time_nodes[i+1]:
                #     if t<=((self.time_nodes[i+1]-self.time_nodes[i])/2+self.time_nodes[i]):  #if in the first half section
                #         x=self.points_list[i]+1/2*self.acc_value*(t-self.time_nodes[i])**2*self.direction[i+1]
                #         x_dot=self.acc_value*self.direction[i+1]*(t-self.time_nodes[i])
                #         x_ddot=self.acc_value*self.direction[i+1]
                #     else: #if in the second half section
                #         x=self.points_list[i+1]-1/2*self.acc_value*(self.time_nodes[i+1]-t)**2*self.direction[i+1]
                #         x_dot=self.acc_value*self.direction[i+1]*(self.time_nodes[i+1]-t)
                #         x_ddot=-self.acc_value*self.direction[i+1]
                    x = np.array([t**5,t**4,t**3,t**2,t,1]).T@c[6*i:6*(i+1),:]
                    x_dot = np.array([5*t**4,4*t**3,3*t**2,2*t,1,0]).T@c[6*i:6*(i+1),:]
                    x_ddot = np.array([20*t**3,12*t**2,6*t,2,0,0]).T@c[6*i:6*(i+1),:]
                    x_dddot = np.array([60 * t ** 2, 24 * t, 6 , 0, 0, 0]).T @ c[6 * i:6 * (i + 1), :]
                    x_ddddot = np.array([120 *t, 24, 0, 0, 0, 0]).T @ c[6 * i:6 * (i + 1), :]
        else:
            x=self.points_list[-1]

        # STUDENT CODE HERE
        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        # print(flat_output['x'])
        return flat_output
