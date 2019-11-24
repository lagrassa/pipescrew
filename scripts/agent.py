from history import History
import numpy as np
from nav_env import NavEnv
from motion_planners.rrt_connect import birrt
from dmp_traj_cluster import DMPTrajCluster
from tslearn.clustering import TimeSeriesKMeans
"""
Can make decisions based on a nav_env
"""


class Agent:
    def __init__(self):
        self.history = History()

    def follow_path(self, ne, path):
        kp = 10  # 50
        delay = 3
        kd = 0.4
        timeout = 20
        for pt in path:
            # xdd = 2*(pt-ne.get_pos()-ne.get_vel()*ne.dt)/(ne.dt**2) #inverse dynamics here
            for _ in range(timeout):
                traj_dist = np.linalg.norm(ne.get_pos() - pt)
                if traj_dist < 0.01:
                    break
                xdd = kp * (pt - ne.get_pos()) - kd * (ne.get_vel())
                for i in range(delay):
                    ne.step(*xdd)
            if traj_dist > 0.1:
                print("High traj dist at", traj_dist)

    def collect_planning_history(self, starts=None, goals = None):
        if starts is None:
            starts = [(0.2, 0.1)]#, (0.21, 0.11)]
        if goals is None:
            goals = (np.array((1, 0.67)),)
        thresh = 3

        for i in range(len(starts)):
            for j in range(len(goals)):
                start = np.array(starts[i])
                goal = goals[j]
                ne = NavEnv(start, goal, slip=False)
                path = self.plan_path(ne,start, goal)
                if path is None:
                    print("No path found")
                else:
                    ne.desired_pos_history = path
                    self.follow_path(ne, path)
                    print_path_stats(path)

                    if np.linalg.norm(ne.get_pos() - goal) < thresh:
                        print("Found one close enough to add")
                        self.history.starts.append(start)
                        self.history.paths.append(np.vstack(path))
        import ipdb; ipdb.set_trace()
        self.history.paths = pad_paths(self.history.paths) #put it in a nicer form
    def plan_path(self, ne, start, goal):
        """

        :rtype: np.array
        """
        distance = lambda x, y: ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5

        def extend(last_config, s):
            configs = [last_config.copy()]
            curr_config = last_config.copy().astype(np.float32)
            dx = s[0] - last_config[0]
            dy = s[1] - last_config[1]
            theta = np.arctan2(dy, dx)
            dt = 0.01
            vel = 3 * dt
            threshold = 0.05
            while np.linalg.norm(curr_config - s) > threshold:
                curr_config[0] += vel * np.cos(theta)
                curr_config[1] += vel * np.sin(theta)
                configs.append(curr_config.copy())
            return configs

        def sample():
            x_rand = np.random.randint(low=0, high=ne.gridsize[0])
            y_rand = np.random.randint(low=0, high=ne.gridsize[0])
            return x_rand, y_rand

        def collision(pt):
            ret = ne.collision_fn(pt)
            return ret

        return birrt(start, goal, distance, sample, extend, collision)
    def cluster_planning_history(self,k=2):

        self.dmp_traj_clusters = []
        kmeans = TimeSeriesKMeans(n_clusters=k).fit(self.history.paths)
        #sort into groups, make them into a dmp_traj_cluster, all into one cluster for now
        for i in range(k):
            center = kmeans.cluster_centers_[i]
            relevant_labels = np.where(kmeans.labels_ == i)[0]
            path_data = self.history.paths[relevant_labels]
            formatted_path_data = np.zeros((path_data.shape[0], path_data.shape[2], path_data.shape[1])) #gets ugly if there's only one now
            n_training_paths = path_data.shape[0]
            for i in range(n_training_paths):
                formatted_path_data[i] = path_data[i].T

            import ipdb; ipdb.set_trace()
            new_dmp_traj_cluster = DMPTrajCluster(formatted_path_data, center)
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
    trajs = np.zeros((len(paths),) + (longest_path,2))
    for i in range(len(paths)):
        path = paths[i]
        diff = longest_path - path.shape[0]
        if diff != 0:
            padding = np.ones((diff, 2)) * path[-1]
            paths[i] = np.vstack([path, padding])

        trajs[i, :] = paths[i]
    return trajs

