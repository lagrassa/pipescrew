from nav_env import NavEnv
from history import History
from simple_model import train_policy
import numpy as np
from motion_planners.rrt_connect import birrt
ne = NavEnv(slip=True)

distance = lambda x,y: 1

def closest_pt_index(path, state):
    dists = np.linalg.norm(path-state,axis=1)
    min_dist_pt = np.argmin(dists)
    return min_dist_pt

def select_imitation_traj(state, history):
    #which point is the closest distance 
    min_dists = []
    for path in history.paths:
        min_dist_pt = closest_pt_index(path, state)
        min_dists.append(min_dist_pt)
    best_path_idx = np.argmin(min_dists)
    closest_pt_idx = closest_pt_index(history.paths[best_path_idx],state)
    return history.paths[best_path_idx][closest_pt_idx:]


def moving_average(a, n=5) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def imitate_history(history, state):
    import pydmps
    #find trajectory with the closest state to this one, form a DMP starting from that state to imitate it
    dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=20, ay=np.ones(2)*25.0, dt =1/186.)
    imitation_traj = select_imitation_traj(state, history)
    imitation_traj = np.vstack([moving_average(np.array(imitation_traj)[:,i] ) for i in range(0,2)])
    y_des = imitation_traj
    
    dmp.imitate_path(y_des=y_des, plot=False)
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
    vel = 1*dt
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

start = ne.start
goal = ne.goal
starts = [(0.2,0.1),(0.4, 0.5)]
history = History()
thresh = 3
timeout = 15
for i in range(len(starts)):
    #ne.agent.position = np.array(starts[i])
    #ne.agent.start = np.array(starts[i])
    start = np.array(ne.start).copy()
    
    path = birrt(start, goal, distance, sample, extend, collision)
    print("Found path")
    ne.desired_pos_history=path
    kp=100
    kd=0
    for pt in path:
        #xdd = 2*(pt-ne.get_pos()-ne.get_vel()*ne.dt)/(ne.dt**2) #inverse dynamics here
        for _ in range(timeout):
            traj_dist =  np.linalg.norm(ne.get_pos()-pt)
            #print("traj dist", traj_dist)
            if traj_dist < 0.01:
                break
            xdd =  kp*(pt-ne.get_pos())-kd*(ne.get_vel())
            ne.step(*xdd)
        if traj_dist > 0.1:
            print("High traj dist at", traj_dist)
    print("Goal distance", np.linalg.norm(ne.get_pos()-goal))
    print_path_stats(path)

    
    if np.linalg.norm(ne.pos-goal) < thresh:
        print("Found one close enough to add")
        history.starts.append(start)
        history.paths.append(path)

ne.pos = np.array(starts[i])
ne.path_color = (190,0,0,1)
imitation_path = imitate_history(history, ne.pos)
print_path_stats(imitation_path)
for pt in imitation_path:
    diff = (np.array(pt)-ne.pos)/ne.k1
    ne.step(*diff)

#policy = train_policy(history)
#for i in range(300):
#    action = policy.predict(ne.pos)
#    ne.step(*action)


