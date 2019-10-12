from nav_env import NavEnv
import numpy as np
from motion_planners.rrt_connect import birrt
ne = NavEnv()

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
    vel = 1
    threshold = 5
    while(np.linalg.norm(curr_config-s) > threshold):
        curr_config[0] += vel*np.cos(theta)
        curr_config[1] += vel*np.sin(theta)
        configs.append(curr_config.copy())
    return configs 

def collision(pt):
    return ne.collision_fn(pt)
start = ne.pos
goal = ne.goal
path = birrt(start, goal, distance, sample, extend, collision)
[ne.go_to(pt[0],pt[1]) for pt in path]
import ipdb; ipdb.set_trace()


