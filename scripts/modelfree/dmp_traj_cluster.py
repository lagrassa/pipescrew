import numpy as np
np.seterr(all='raise')

import pydmps
class DMPTrajCluster:
    """
    :param paths: list of paths
    :type paths list
    """
    def __init__(self, paths, center, execution_times, rl = False):
        self.training_trajs = paths
        self.center = center
        self.execution_times = execution_times
        self.dmp = self.fit_dmp()
        if rl:
            from bolero.optimizer import REPSOptimizer
            self.opt = REPSOptimizer(self.dmp.w.flatten(), train_freq = 1, variance = 0.1*np.var(self.dmp.w))
            self.opt.init(len(self.dmp.w.flatten()))

        print("Check weights")


    def fit_dmp(self):
        ndims = self.training_trajs.shape[1]
        dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=ndims, n_bfs=20, ay=np.ones(ndims) * 10.0)
        dmp.imitate_path(self.training_trajs)
        return dmp
    def reps_update_dmp(self, result, explore=False):
        if self.opt.params is not None:
            import ipdb; ipdb.set_trace()
            self.opt.set_evaluation_feedback(result)
        self.opt.get_next_parameters(self.dmp.w.flatten(), explore=explore)
        self.dmp.w = self.opt.params.reshape(self.dmp.w.shape)


    def rollout(self, start, goal):
        self.dmp.reset_state(y0=start, goal=goal)
        y_track, dy_track, ddy_track = self.dmp.rollout()
        return y_track
    def distance_from_center(self, pt):
        return np.linalg.norm(self.center-pt)




