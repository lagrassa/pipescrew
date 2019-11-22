import numpy as np
import pydmps
class DMPTrajCluster:
    """
    :param paths: list of paths
    :type paths list
    """
    def __init__(self, paths):
        self.training_trajs = pad_paths(paths)
        self.dmp = self.fit_dmp()

    def fit_dmp(self):
        dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 10.0)
        dmp.imitate_path(self.training_trajs)
        return dmp

    def rollout(self, start, goal):
         self.dmp.reset_state(y0=start, goal=goal)
         y_track, dy_track, ddy_track = self.dmp.rollout()
         return y_track

def pad_paths(paths):
    longest_path = max(paths, key=lambda x: x.shape[0]).shape[0]
    trajs = np.zeros((len(paths),) + (2, longest_path))
    for i in range(len(paths)):
        path = paths[i]
        diff = longest_path - path.shape[0]
        if diff != 0:
            padding = np.ones((diff, 2)) * path[-1]
            paths[i] = np.vstack([path, padding])

        trajs[i, :] = paths[i].T
    return trajs
