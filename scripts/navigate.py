from nav_env import NavEnv
from history import History
from simple_model import train_policy
import numpy as np
from motion_planners.rrt_connect import birrt
ne = NavEnv(slip=True)

distance = lambda x,y: 1
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
    vel = 2
    threshold = 5
    while(np.linalg.norm(curr_config-s) > threshold):
        curr_config[0] += vel*np.cos(theta)
        curr_config[1] += vel*np.sin(theta)
        configs.append(curr_config.copy())
    return configs 

def collision(pt):
    return ne.collision_fn(pt)

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



start = ne.pos
goal = ne.goal
starts = [(0,3),(1,1),(2,4),(5,6),(4,6),(9,10)]
history = History()
for i in range(len(starts)):
    ne.pos = np.array(starts[i])
    start = np.array(starts[i]).copy()
    path = birrt(start, goal, distance, sample, extend, collision)
    for pt in path:
        diff = (np.array(pt)-ne.pos)/ne.k1
        ne.step(*diff)
    history.starts.append(start)
    history.paths.append(path)
policy = train_policy(history)
for i in range(300):
    action = policy.predict(ne.pos)
    ne.step(*action)
import ipdb; ipdb.set_trace()


