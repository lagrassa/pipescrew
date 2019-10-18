from nav_env import NavEnv
from simple_model import train_policy
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

class History():
    def __init__(self):
        self.starts = []
        self.paths = []


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
    [ne.go_to(pt[0], pt[1]) for pt in path]
    import ipdb; ipdb.set_trace()
    history.starts.append(start)
    history.paths.append(path)
policy = train_policy(history)
test_start = np.array((1,2)).reshape((1,2))
current_state = test_start.copy()
for i in range(300):
    current_state = policy.predict(current_state)
    ne.go_to(current_state[0][0],current_state[0][1])
    print(current_state.round(2))
import ipdb; ipdb.set_trace()


