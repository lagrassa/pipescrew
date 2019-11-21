from nav_env import NavEnv
from history import History
from simple_model import train_policy
import numpy as np
from motion_planners.rrt_connect import birrt

distance = lambda x,y: 1

def closest_pt_index(path, state):
    dists = np.linalg.norm(path-state,axis=1)
    min_dist_pt = np.argmin(dists)
    return min_dist_pt

def select_imitation_trajs(state, history):
    #which point is the closest distance 
    min_dists = []
    best_k = 2
    for path in history.paths:
        min_dist_pt = closest_pt_index(path, state)
        min_dists.append(min_dist_pt)
    sorted_by_dist_idxs = np.argsort(min_dists)
    closest_pt_idxs = [closest_pt_index(history.paths[sorted_by_dist_idx],state) for sorted_by_dist_idx in sorted_by_dist_idxs]
    best_path_idxs = sorted_by_dist_idxs[:best_k]
    best_paths = [np.vstack(history.paths[dist_idx][pt_idx:]) for dist_idx, pt_idx in zip(best_path_idxs, closest_pt_idxs)]
    #pad them
    longest_path = max(best_paths, key=lambda x: x.shape[0]).shape[0]
    trajs = np.zeros((len(best_paths),)+(2,longest_path) )
    for i in range(len(best_paths)):
        path = best_paths[i]
        diff = longest_path-path.shape[0]
        if diff != 0:
            padding = np.ones((diff,2))*path[-1]
            best_paths[i] = np.vstack([path,padding])
        
        trajs[i,:] = best_paths[i].T
    return trajs
    


def moving_average(a, n=5) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def imitate_history(history, state, goal):
    import pydmps
    #find trajectory with the closest state to this one, form a DMP starting from that state to imitate it
    dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=20, ay=np.ones(2)*25.0, dt =1/186.)
    imitation_trajs = select_imitation_trajs(state, history)
    #imitation_traj = np.vstack([moving_average(np.array(imitation_traj)[:,i] ) for i in range(0,2)])
    y_des = imitation_trajs
    
    dmp.imitate_path(y_des=y_des, plot=False)
    import ipdb; ipdb.set_trace()
    y0 = state
    goal = goal 
    dmp.reset_state(goal=goal, y0=y0)
    y_track, dy_track, ddy_track = dmp.rollout(timesteps=210)
    return y_track

def sample():
    x_rand = np.random.randint(low=0,high=ne.gridsize[0])
    y_rand = np.random.randint(low=0,high=ne.gridsize[0])
    return x_rand, y_rand

def extend(last_config, s):
    configs = [last_config.copy()]
    curr_config = last_config.copy().astype(np.float32)
    dx = s[0] - last_config[0]
    dy = s[1] - last_config[1]
    theta = np.arctan2(dy,dx)
    x_vec = np.linalg.norm([dx,dy])*np.cos(theta) 
    y_vec = np.linalg.norm([dx,dy])*np.sin(theta) 
    dt = 0.01
    vel = 5*dt
    threshold = 0.05
    while(np.linalg.norm(curr_config-s) > threshold):
        curr_config[0] += vel*np.cos(theta)
        curr_config[1] += vel*np.sin(theta)
        configs.append(curr_config.copy())
    return configs 

def collision(pt):
    ret =  ne.collision_fn(pt)
    return ret

def interp(history, s0):
    old_xs = [pt[0] for pt in history.paths]
    old_ys = [pt[1] for pt in history.paths]
    x = np.interp(s0[0],ts,old_xs)
    y = np.interp(s0[1],ts,old_ys)
    path = []
    for i in range(10):
        path.append((x,y))
        x = np.interp(x,old_xs, old_ys)
        y = np.interp(y,old_xs, old_ys)
        
    return path

def print_path_stats(path):
    max_x_diff = np.max(np.diff(np.array(path)[:,0]))
    max_y_diff = np.max(np.diff(np.array(path)[:,1]))

    print("diff", abs(max_x_diff-max_y_diff))
    print("max x diff", max_x_diff)
    print("may y diff", max_y_diff)

starts = [(0.2,0.1),(0.4, 0.5)]
history = History()
thresh = 3
timeout = 15

def follow_path(ne, path):
    kp=5#50
    delay = 3
    kd=0.4
    for pt in path:
        #xdd = 2*(pt-ne.get_pos()-ne.get_vel()*ne.dt)/(ne.dt**2) #inverse dynamics here
        for _ in range(timeout):
            traj_dist =  np.linalg.norm(ne.get_pos()-pt)
            #print("traj dist", traj_dist)
            if traj_dist < 0.01:
                break
            xdd =  kp*(pt-ne.get_pos())-kd*(ne.get_vel())
            for i in range(delay):
                ne.step(*xdd)
        if traj_dist > 0.1:
            print("High traj dist at", traj_dist)
    print("Goal distance", np.linalg.norm(ne.get_pos()-goal))

for i in range(len(starts)):
    #ne.agent.position = np.array(starts[i])
    #ne.agent.start = np.array(starts[i])
    start = np.array(starts[i])
    goal = np.array((1,0.67) )
    ne = NavEnv(start,goal, slip=False)
    path = birrt(start, goal, distance, sample, extend, collision)
    print("Found path")
    ne.desired_pos_history=path
    follow_path(ne,path)
    print_path_stats(path)

    
    if np.linalg.norm(ne.get_pos()-goal) < thresh:
        print("Found one close enough to add")
        history.starts.append(start)
        history.paths.append(path)
start = np.array((0.2,0.1))
goal = np.array((1,0.67))
ne = NavEnv(start, goal, slip=False)
ne.path_color = (190,0,0,1)
imitation_path = imitate_history(history, ne.get_pos(), goal)
ne.plot_path(imitation_path)
print_path_stats(imitation_path)
import ipdb; ipdb.set_trace()
follow_path(ne, path)



