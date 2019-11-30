from history import History
import numpy as np
from motion_planners.rrt_connect import birrt
from dmp_traj_cluster import DMPTrajCluster
from tslearn.clustering import TimeSeriesKMeans
from screwpipe import PipeGraspEnv
"""
Can make decisions based on a nav_env
"""


class PipeAgent:
    def __init__(self, show_training = False):
        self.history = History()
        self.show_training = show_training
        self.grasp = np.array((0,0,0.067)) # acquire autonomously

    def follow_path(self, pe, path):
        kp = 10  # 50
        delay = 3
        kd = 0.4
        pe.total_timeout= 5
        target_quat = (1,0.5,0,0)
        for pt in path:
            target_point = pt +self.grasp
            pe.go_to_pose((target_point, target_quat), attachments = [pe.pipe_attach], maxForce = 100, cart_traj = True)

    def collect_planning_history(self, starts=None, goals = None):
        shifts = (0, 0.001, 0.01)#, (0.21, 0.11)]
        thresh = 0.08
        target_quat = (1,0.5,0,0) #get whatever it is by default


        env = PipeGraspEnv(visualize=False, shift=0)
        for i in range(len(shifts)):
            start = env.get_pos()
            goal = np.array([0,0,0.1])
            path = self.plan_path(env,start, goal)
            if path is None:
                print("No path found")
            else:
                env.desired_pos_history = path
                self.follow_path(env,path)
                if env.is_pipe_in_hole():
                    print("Found one close enough to add")
                    self.history.starts.append(start)
                    self.history.execution_times.append(env.steps_taken)
                    self.history.paths.append(np.vstack(path))
            env.restore_state()
            
        self.history.paths = pad_paths(self.history.paths) #put it in a nicer form
    ''' list of pipe sequences'''
    def plan_path(self, env, start, goal):
        """

        :rtype: np.array
        """
        distance = lambda x, y: ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5

        def extend(last_config, s):
            configs = [last_config.copy()]
            curr_config = last_config.copy().astype(np.float32)
            dt = env.dt_pose
            vel = 0.1
            diff =  np.linalg.norm(last_config-s)
            diff_comp =  s-last_config
            num_steps_req = int(np.ceil(diff/(vel*dt)))
            for i in range(num_steps_req):
                curr_config[:] += diff_comp*dt
                configs.append(curr_config.copy())
            return configs

        def sample():
            return 0.1*np.random.uniform((3,))

        def collision(pt):
            ret = env.collision_fn(pt)
            return ret

        return birrt(start, goal, distance, sample, extend, collision)
    def cluster_planning_history(self,k=2):

        self.dmp_traj_clusters = []
        kmeans = TimeSeriesKMeans(n_clusters=k).fit(self.history.paths)
        #sort into groups, make them into a dmp_traj_cluster, all into one cluster for now
        for i in range(k):
            center = kmeans.cluster_centers_[i]
            relevant_labels = np.where(kmeans.labels_ == i)[0]
            execution_times = np.array(self.history.execution_times)[relevant_labels]
            path_data = self.history.paths[relevant_labels]
            formatted_path_data = np.zeros((path_data.shape[0], path_data.shape[2], path_data.shape[1])) #gets ugly if there's only one now
            n_training_paths = path_data.shape[0]
            for i in range(n_training_paths):
                formatted_path_data[i] = path_data[i].T

            new_dmp_traj_cluster = DMPTrajCluster(formatted_path_data, center, execution_times=execution_times)
            self.dmp_traj_clusters.append(new_dmp_traj_cluster)

    def dmp_plan(self, start, goal):
        #select cluster where first point is closest to start
        best_idx = np.argmin([dmp_traj_cluster.distance_from_center(start) for dmp_traj_cluster in self.dmp_traj_clusters])
        return self.dmp_traj_clusters[best_idx].rollout(start, goal)
        """
        min_dists = []
        best_k = 2
        for path in self.history.paths:
            min_dist_pt = closest_pt_index(path, start)
            min_dists.append(min_dist_pt)
        sorted_by_dist_idxs = np.argsort(min_dists)
        closest_pt_idxs = [closest_pt_index(self.history.paths[sorted_by_dist_idx], start) for sorted_by_dist_idx in
                           sorted_by_dist_idxs]
        best_path_idxs = sorted_by_dist_idxs[:best_k]
        best_paths = [np.vstack(self.history.paths[dist_idx][pt_idx:]) for dist_idx, pt_idx in
                      zip(best_path_idxs, closest_pt_idxs)]
        """



def print_path_stats(path):
    max_x_diff = np.max(np.diff(np.array(path)[:, 0]))
    max_y_diff = np.max(np.diff(np.array(path)[:, 1]))
    print("diff", abs(max_x_diff - max_y_diff))
    print("max x diff", max_x_diff)
    print("may y diff", max_y_diff)


def closest_pt_index(path, state):
    dists = np.linalg.norm(path - state, axis=1)
    min_dist_pt = np.argmin(dists)
    return min_dist_pt

def pad_paths(paths):
    longest_path = max(paths, key=lambda x: x.shape[0]).shape[0]
    trajs = np.zeros((len(paths),) + (longest_path,paths[0].shape[1]))
    for i in range(len(paths)):
        path = paths[i]
        diff = longest_path - path.shape[0]
        if diff != 0:
            padding = np.ones((diff, 2)) * path[-1]
            paths[i] = np.vstack([path, padding])

        trajs[i, :] = paths[i]
    return trajs
