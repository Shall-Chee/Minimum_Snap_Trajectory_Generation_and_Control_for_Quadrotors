import heapq as hp
# from heapq import heappush, heappop  # Recommended.
import numpy as np
# from numpy import linalg as LA

from flightsim.world import World


from proj1_3.code.occupancy_map import OccupancyMap # Recommended.
# from occupancy_map import OccupancyMap # Recommended.

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
    """
    # print('5')
    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    # print('c')
    occ_map._init_map_from_world
    # print('ac')
    start_index = (occ_map.metric_to_index(start))
    goal_index = (occ_map.metric_to_index(goal))
    # print(start_index)
    start_index =[start_index[0],start_index[1],start_index[2]]
    goal_index = [goal_index[0],goal_index[1],goal_index[2]]
    parents=np.zeros((occ_map.map.shape[0],occ_map.map.shape[1],occ_map.map.shape[2],3))
    cost_to_come=np.full((occ_map.map.shape[0],occ_map.map.shape[1],occ_map.map.shape[2]),np.inf)
    h_cost_to_come=np.full((occ_map.map.shape[0],occ_map.map.shape[1],occ_map.map.shape[2]),np.inf)
    cost_to_come[start_index[0]][start_index[1]][start_index[2]]=0
    h_cost_to_come[start_index[0]][start_index[1]][start_index[2]]=0
    Q=[]
    # x = np.array([[1,2,3], [4,5,6]])
    # print(x[tuple([1,0])])
    # print(x[1,0])
    if (not (occ_map.is_valid_index(start_index))) or occ_map.is_occupied_index(start_index):
        print('12')
        return None
    if (not (occ_map.is_valid_index(goal_index))) or occ_map.is_occupied_index(goal_index):
        print('22')
        return None    
    # print(start_index)
    hp.heappush(Q, (0,start_index))
    hp.heappush(Q, (0,start_index))
    hp.heappush(Q, (np.inf,goal_index))
    # goal_index_center=occ_map.index_to_metric_center(goal_index)
    # iter=0
    u=hp.heappop(Q)
    # print('b')
    # print(type(start_index))
    # print('before if')
    if astar==False: #Dijk
        while (np.array(u[1])!=np.array(goal_index)).any():
            # iter=iter+1
            if len(Q)==0: #impossible situation
                return None
            u=hp.heappop(Q)
            neighbor_index = np.array([u[1][0]+[-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
                u[1][1]+[-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1],
                u[1][2]+[-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1]])
            valid = occ_map.is_valid_index_2(neighbor_index)
            neighbor_valid_index_valid = neighbor_index[:,valid]            
            occupied = occ_map.is_occupied_index(neighbor_valid_index_valid)
            neighbor_valid_index = neighbor_valid_index_valid[:,~occupied]
            
            distance_between_new_old_matrix = [[resolution[0]],[resolution[1]],[resolution[2]]]*(neighbor_valid_index-np.array([u[1]]).T)
            distance_between_new_old = np.linalg.norm(distance_between_new_old_matrix,axis=0)
            # d = cost_to_come[[u[1][0]],[u[1][1]],[u[1][2]]]+distance_between_new_old.flatten()
            d = cost_to_come[tuple(u[1])]+distance_between_new_old.flatten()
            change_index = neighbor_valid_index[:,d<cost_to_come[tuple(neighbor_valid_index)]]
            change_d = d[d<cost_to_come[tuple(neighbor_valid_index)]]
            cost_to_come[tuple(change_index)] = change_d
            parents[tuple(change_index)] = u[1]
            for i in range(change_index.shape[1]):
                hp.heappush(Q,(change_d[i],[change_index[0][i],change_index[1][i],change_index[2][i]]))
    else: #A*
        while (np.array(u[1])!=np.array(goal_index)).any():
            # print(u[1])
            # iter=iter+1
            # print(iter)
            if len(Q)==0:#impossible situation
                return None
            u=hp.heappop(Q)
            neighbor_index = np.array([u[1][0]+[-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
                u[1][1]+[-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1],
                u[1][2]+[-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1]])
            valid = occ_map.is_valid_index_2(neighbor_index)
            neighbor_valid_index_valid = neighbor_index[:,valid]            
            occupied = occ_map.is_occupied_index(neighbor_valid_index_valid)
            neighbor_valid_index = neighbor_valid_index_valid[:,~occupied]

            distance_between_new_old_matrix = [[resolution[0]],[resolution[1]],[resolution[2]]]*(neighbor_valid_index-np.array([u[1]]).T)
            distance_between_new_goal_matrix = [[resolution[0]],[resolution[1]],[resolution[2]]]*(neighbor_valid_index-np.array([goal_index]).T)
            distance_between_new_old = np.linalg.norm(distance_between_new_old_matrix,axis=0)
            distance_between_new_goal = np.linalg.norm(distance_between_new_goal_matrix,axis=0)
            d = cost_to_come[tuple(u[1])]+distance_between_new_old.flatten()
            fv = cost_to_come[tuple(u[1])]+distance_between_new_old.flatten()+distance_between_new_goal.flatten()
            judge = d<cost_to_come[tuple(neighbor_valid_index)]
            change_index = neighbor_valid_index[:,judge]
            change_d = d[judge]
            change_fv = fv[judge]
            cost_to_come[tuple(change_index)] = change_d
            parents[tuple(change_index)] = u[1]
            for i in range(change_index.shape[1]):
                hp.heappush(Q,(change_fv[i],[change_index[0][i],change_index[1][i],change_index[2][i]]))
            
    path = []
    path = [goal[0],goal[1],goal[2]]
    path = np.vstack((occ_map.index_to_metric_center(tuple(occ_map.metric_to_index(goal))),path))
    # path_index = goal_index
    # print(path)
    # print(parents)
    # path_length=np.linalg.norm(path[1]-path[0])
    # path_find=[int(parents[goal_index[0],goal_index[1],goal_index[2]][0]),int(parents[goal_index[0],goal_index[1],goal_index[2]][1]),\
    #     int(parents[goal_index[0],goal_index[1],goal_index[2]][2])]
    # print(parents)
    path_find = parents[tuple(goal_index)]
    path_find = path_find.astype(int)
    while (path_find!=np.array(start_index)).any():
        path = np.vstack((occ_map.index_to_metric_center(tuple(path_find)),path))
        # path_index = np.vstack((path_find),path_index))
        # path_length=path_length+np.linalg.norm(occ_map.index_to_metric_center(path_find)-path[1])
        path_find = parents[tuple(path_find)]
        path_find = path_find.astype(int)
    # path_length=path_length+np.linalg.norm(occ_map.index_to_metric_center(start_index)-path[0])
    path = np.vstack((occ_map.index_to_metric_center(tuple(occ_map.metric_to_index(start))),path))
    # path_length=path_length+np.linalg.norm([start[0],start[1],start[2]]-path[0])
    path = np.vstack(([start[0],start[1],start[2]],path))
    # print(iter)
    # print(path[0])
    # print(path.shape[0])

    #simplify map
    path_iter = path[0]
    path_final = path[np.array([0])]
    print(path)
    # print(path_final)
    # print(path_final)
    for i in range(1,path.shape[0]-4):
        # print(path)
        diff_deci = (abs(path[i]-path_iter)/resolution)
        diff = diff_deci.astype(int)
        diff_max = np.amax(diff) 
        test_point_T = np.linspace(path_iter,path[i],num=diff_max)
        # print(test_point_T)
        test_point_ind_T = occ_map.metric_to_index(path_iter)
        for j in range(2,test_point_T.shape[0]):
            # print(occ_map.metric_to_index(path[j]))
            test_point_ind_T = np.vstack((test_point_ind_T,occ_map.metric_to_index(test_point_T[j])))
        coll_1 = occ_map.is_occupied_index(test_point_ind_T.T)

        diff_deci_2 = (abs(path[i+1]-path_iter)/resolution)
        diff_2 = diff_deci_2.astype(int)
        diff_max_2 = np.amax(diff_2) 
        test_point_T_2 = np.linspace(path_iter,path[i+1],num=diff_max_2)
        # print(test_point_T)
        test_point_ind_T_2 = occ_map.metric_to_index(path_iter)
        for k in range(2,test_point_T_2.shape[0]):
            # print(occ_map.metric_to_index(path[j]))
            test_point_ind_T_2 = np.vstack((test_point_ind_T_2,occ_map.metric_to_index(test_point_T_2[k])))
        coll_2 = occ_map.is_occupied_index(test_point_ind_T_2.T)

        if(coll_1[coll_1>0].shape[0]==0 and coll_2[coll_2>0].shape[0]!=0):
            path_final = np.vstack((path_final,(path[i])))
            path_iter = path[i]
    # path_final = np.vstack((path_final,(occ_map.index_to_metric_center(tuple(occ_map.metric_to_index(goal))))))
    path_final = np.vstack((path_final,(goal)))
    print(path_final)
                     

        # points_number = int()
    # return path
    return path_final