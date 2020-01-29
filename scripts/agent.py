from numpy.core._multiarray_umath import ndarray
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from history import History
from belief import Belief
from sklearn import mixture
import numpy as np
from nav_env import NavEnv
from motion_planners.rrt_connect import birrt
from dmp_traj_cluster import DMPTrajCluster
from scipy.integrate import solve_ivp
from rmpflow.rmp import RMPRoot
from vae import make_vae, train_vae
from rmpflow.rmp_leaf import GoalAttractorUni

"""
Can make decisions based on a nav_env
"""


class Agent:
    def __init__(self, show_training=False):
        self.history = History()
        self.belief = Belief(mu = [0,0], cov=1) #we know nothing
        self.autoencoder = None
        self.show_training = show_training
        self.goal_threshold = 0.02
        self.max_xdot = 2
        self.vae_fn = "models/vae.h5y"
        self.guapo_eps = 0.9
        self.rmp = None
        self.cluster_planning_history = self.dmp_cluster_planning_history
        self.pd_errors = []
    """
    Given a concerrt policy, follows it until it detects that there's a model error
    """
    def follow_policy(self, policy, ne, goal_belief):
        """
        Selects actions and executes them
        :return:
        """
        while not ne.goal_condition_met():
            curr_belief = self.get_curr_belief(ne)
            action = policy(curr_belief, goal_belief) # policy doesn't care about velocity
            success = self.execute_action(action)
            if not success:
                print("Detected model error")
                return
        print("Achieved goal using MB policy")
    def off_path(self, expected_next, obs):
        """
        True if observed a low probability event
        """
        return False
    """
    Executes action, True if succeeds ands False if not
    actions are high level, comes with the controller that agent has
    """
    def execute_action(self, ne, action):
        #move in delta size steps toward q_rand.
        ext_force = ne.ext_mu*ne.mass
        control, expected_next = action.get_control(ne.get_obs_low_dim(), ne.dt, ext_force)
        ne.step(control, rl=False)
        return not self.off_path(expected_next, ne.get_obs_low_dim())

    def follow_path(self, ne, path, force=None):
        kp = 10  # 50
        delay = 2
        kd = 0.4
        timeout = 6
        for pt in path:
            # xdd = 2*(pt-ne.get_pos()-ne.get_vel()*ne.dt)/(ne.dt**2) #inverse dynamics here
            for _ in range(timeout):
                traj_dist = np.linalg.norm(ne.get_pos() - pt)
                if traj_dist < 0.01:
                    break
                xdd = kp * (pt - ne.get_pos()) - kd * (ne.get_vel())
                if np.linalg.norm(xdd) > self.max_xdot:
                    xdd = self.max_xdot * xdd / (np.linalg.norm(xdd))
                # print(np.linalg.norm(xdd))
                for i in range(delay):
                    ne.step(xdd)
            if self.show_training and traj_dist > 0.1:
                print("High traj dist at", traj_dist)

    def do_rl(self, N=300):
        # sample point, pick cluster, do RL on that cluster
        for i in range(N):
            start = np.random.uniform(size=2, low=0, high=0.7)
            goal = np.array((1, 0.67))
            dmp_traj, best_idx = self.dmp_plan(start, goal, ret_cluster=True)  # type: (object, ndarray[int])
            ne = NavEnv(start, goal, slip=False, visualize=self.show_training)
            self.follow_path(ne, dmp_traj)
            # print("Goal dist", np.linalg.norm(ne.get_pos()-goal))
            result = -np.sign(np.linalg.norm(ne.get_pos() - goal))
            self.do_rl_on_cluster(self.dmp_traj_clusters[best_idx], result)

    def do_rl_on_cluster(self, cluster, result):
        cluster.reps_update_dmp(result, explore=True)

    def collect_planning_history(self, starts=None, goals=None, N=5):
        if starts is None:
            starts = np.random.uniform(size=(N, 2), low=0, high=0.7)
        if goals is None:
            goals = (np.array((1, 0.67)),)
        thresh = 0.08
        new_paths = []
        while len(new_paths) < N:
            for i in range(len(starts)):
                for j in range(len(goals)):
                    start = np.array(starts[i])
                    goal = goals[j]
                    ne = NavEnv(start, goal, slip=False, visualize=self.show_training)
                    path = self.plan_path(ne, start, goal)
                    if path is None:
                        print("No path found")
                    else:
                        ne.desired_pos_history = path
                        self.follow_path(ne, path)
                        if np.linalg.norm(ne.get_pos() - goal) < thresh:
                            print("Found one close enough to add")
                            self.history.starts.append(start)
                            self.history.execution_times.append(ne.steps_taken)
                            assert (np.linalg.norm(path) != 0)
                            new_paths.append(np.vstack(path))
                            if len(new_paths) >= N:
                                break
                if len(new_paths) >= N:
                    break
        if len(new_paths) > 0:
            self.history.paths = pad_and_add_paths(new_paths, self.history.paths)  # put it in a nicer form
        else:
            print("out of ", len(starts), "things to iterate over, I got none")

    """
    Assumes we are already in contact with the object

    """

    def plan_path(self, ne, start, goal):
        """

        :rtype: np.array
        """
        distance = lambda x, y: ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5
        dt = 0.01

        # vel = 0.6+2*np.random.random()
        vel = 0.9 + 2 * np.random.random()

        def extend(last_config, s):
            configs = [last_config.copy()]
            curr_config = last_config.copy().astype(np.float32)
            dt = 0.01
            diff = np.linalg.norm(last_config - s)
            diff_comp = s - last_config
            num_steps_req = int(np.ceil(diff / (vel * dt)))
            for i in range(num_steps_req):
                curr_config[:] += diff_comp / num_steps_req
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

    def traj_cluster_planning_history(self, k=2):

        self.dmp_traj_clusters = []

        from tslearn.clustering import TimeSeriesKMeans
        kmeans = TimeSeriesKMeans(n_clusters=k).fit(self.history.paths)
        # sort into groups, make them into a dmp_traj_cluster, all into one cluster for now
        for i in range(k):
            center = kmeans.cluster_centers_[i]
            relevant_labels = np.where(kmeans.labels_ == i)[0]
            if len(relevant_labels) == 0:
                print("Empty cluster")
                continue
            execution_times = np.array(self.history.execution_times)[relevant_labels]
            path_data = self.history.paths[relevant_labels]
            formatted_path_data = np.zeros(
                (path_data.shape[0], path_data.shape[2], path_data.shape[1]))  # gets ugly if there's only one now
            n_training_paths = path_data.shape[0]
            for i in range(n_training_paths):
                formatted_path_data[i] = path_data[i].T

            new_dmp_traj_cluster = DMPTrajCluster(formatted_path_data, center, execution_times=execution_times, rl=True)
            self.dmp_traj_clusters.append(new_dmp_traj_cluster)

    def dmp_cluster_planning_history(self, k=2):
        self.dmp_traj_clusters = []
        from tslearn.clustering import TimeSeriesKMeans
        kmeans = TimeSeriesKMeans(n_clusters=k).fit(self.history.paths)
        # fit all of these to different DMPS then cluster
        for i in range(self.history.paths.shape[0]):
            path_data = self.history.paths[i]
            formatted_path_data = path_data.reshape((1, path_data.shape[1], path_data.shape[0]))
            center = formatted_path_data[0]
            new_dmp_traj_cluster = DMPTrajCluster(formatted_path_data, center,
                                                  execution_times=[self.history.execution_times[i]], rl=True)
            self.dmp_traj_clusters.append(new_dmp_traj_cluster)
        # cluster based on weights
        weight_list = [dmp.dmp.w for dmp in self.dmp_traj_clusters]
        weights_to_cluster = np.zeros((len(self.dmp_traj_clusters), weight_list[0].shape[1] * weight_list[0].shape[0]))
        for i in range(len(weight_list)):
            weights_to_cluster[i] = weight_list[i].flatten()
        self.gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(weights_to_cluster)
        import ipdb;
        ipdb.set_trace()
        # plot these somehow

    def dmp_plan(self, start, goal, ret_cluster=False):
        # select cluster where first point is closest to start
        best_idx = np.argmin(
            [dmp_traj_cluster.distance_from_center(start) for dmp_traj_cluster in self.dmp_traj_clusters])
        if ret_cluster:
            return self.dmp_traj_clusters[best_idx].rollout(start, goal), best_idx
        else:
            return self.dmp_traj_clusters[best_idx].rollout(start, goal)

    def is_in_s_uncertain(self, ne, verbose = False):
        width = 0.1  # user set
        pos = ne.get_pos()
        p_in_s_uncertain = self.belief.in_s_uncertain.cdf(pos + width) - self.belief.in_s_uncertain.cdf(pos - width)
        if verbose:
            print(p_in_s_uncertain)
        return p_in_s_uncertain > self.guapo_eps

    """
    Probability distribution of action given state
    1. determine if s \in s_hat_uncertain
    2. choose appropriate policy
    """

    def sample_policy(self, ne, obs):
        if self.is_in_s_uncertain(ne):
            return self.model_free_policy(ne.get_obs(), ne, load_model=True)
        else:
            return self.model_based_policy(obs, ne)

    def random_policy(self, ne):
        return np.random.uniform(low=ne.action_space.low, high=ne.action_space.high)

    def model_based_trajectory(self, s, ne, nsteps=10):
        centroid = self.belief.in_s_uncertain.mean
        if self.rmp is None:
            r = RMPRoot("root")
            x_g = centroid
            leaf2 = GoalAttractorUni("goal_attractor", r, x_g)
            self.rmp = r

            def dynamics(t, state):
                state = state.reshape(2, -1)
                x = state[0]
                x_dot = state[1]
                x_ddot = r.solve(x, x_dot)
                state_dot = np.concatenate((x_dot, x_ddot), axis=None)
                return state_dot

            self.dynamics = dynamics

        sol = solve_ivp(self.dynamics, [0.001, nsteps], s)
        return sol.y[0:2, :]

    """
    get into s_hat_uncertain of ne, just the nearest centroids of the gaussians if you're going to do that
    """

    def model_based_policy(self, s, ne, nsteps=10, recompute_rmp=True):
        kp = 15
        kd = 0.001
        if recompute_rmp:
            self.model_based_trajectory(s, ne, nsteps=nsteps)
        xdd = kp * self.rmp.solve(s[0:2], s[2:]).flatten()
        return xdd

    """
    Collects N samples that would be useful to the autoencoder, which means samples in the "uncertain" region. 
    Easiest thing to do is to use the MB policy to bring the agent into the uncertain region and then keep acting randomly
    as long as its still in the uncertain region. 
    """

    def collect_autoencoder_data(self, ne, n_data=10):
        sample_obs = ne.get_obs()
        samples = np.zeros((n_data,) + (sample_obs.shape[0] * sample_obs.shape[1],))
        s = np.array([ne.get_pos(), ne.get_vel()]).flatten()
        i = 0
        while i < n_data:
            if self.is_in_s_uncertain(ne):
                samples[i, :] = ne.get_obs().flatten()
                if i % 2 == 0:
                   action = self.random_policy(ne)
                   dt = ne.dt*3 # to get a bit more movement
                else:
                    action = self.model_based_policy(s, ne)
                    dt = ne.dt
                ne.step(action, dt=dt)  # longer
                i += 1
            else:
                action = self.model_based_policy(s, ne)
                ne.step(action)
        return samples

    def train_autoencoder(self, ne, n_epochs=50, n_data=5):
        training_data = self.collect_autoencoder_data(ne, n_data=n_data)
        sample_obs = ne.get_obs()
        image_size = sample_obs.shape[1]
        n_train = int(len(training_data) * 4 / 5.)
        vae, encoder, decoder, inputs, outputs, output_tensors= make_vae(image_size = image_size)
        train_vae(vae, training_data, n_train, inputs, outputs,output_tensors, n_epochs=n_epochs)
        vae.save_weights(self.vae_fn)

    def setup_autoencoder(self, obs):
        image_size = obs.shape[1]
        vae, encoder, decoder, inputs, outputs , _= make_vae(image_size = image_size)
        self.autoencoder = encoder
        vae.load_weights(self.vae_fn)

    def autoencode(self, obs):
        if self.autoencoder is None:
            try:
                self.setup_autoencoder(obs)
            except FileNotFoundError:
                print("File not found")
        processed_obs = obs.flatten()
        _, _, z = self.autoencoder.predict(processed_obs.reshape((1,processed_obs.shape[0])), batch_size=1)
        return z.flatten()

    """
    do RL where the agent collects information about the world to update its model free policy, just only for states that are in s_hat_uncertain
    """
    def model_free_policy(self, ne, n_epochs=1, train=True, load_model=False):
        if self.autoencoder is None:
            self.setup_autoencoder(ne.get_obs())
            assert(self.autoencoder) is not None
        if ne.autoencoder is None:
            ne.set_autoencoder(self.autoencode)
            ne.autoencoder = self.autoencode
        if train:
            fn = "models/model1.h5"
            self.mf_policy = PPO2(env=ne, policy=MlpPolicy, n_steps=40, verbose=2, noptepochs=10, learning_rate=3e-4,
                                  ent_coef=0.1, gamma=0.1)
            if load_model:
                self.mf_policy.load(fn, env=make_vec_env(lambda: ne))
            else:
                self.mf_policy.learn(total_timesteps=n_epochs * 40)
                self.mf_policy.save(fn)
        encoded_obs = ne.rl_obs()
        return self.mf_policy.step([encoded_obs], deterministic=True)[0].flatten()

    """
Main control loop,
    uses mb policy to get into s_hat_uncertain of ne, and then mf to maximize reward
    executes policy to get feedback and update state
    obstacle prior is a rough idea of where the obstacle is
    """

    def achieve_goal(self, ne, goal, N=1):
        # are we in s_hat_uncertain?
        state = np.concatenate([ne.get_pos(), ne.get_vel()])
        actions = []
        states = []
        for i in range(N):
            action = self.sample_policy(ne, ne.get_obs())
            state = ne.step(action)
            actions.append(action)
            states.append(state)
        return states, actions


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


def pad_and_add_paths(paths, old_path_history):
    longest_path = max(paths, key=lambda x: x.shape[0]).shape[0]
    new_trajs = np.zeros((len(paths),) + (longest_path, 2))
    for i in range(len(paths)):
        path = paths[i]
        diff = longest_path - path.shape[0]
        if diff != 0:
            padding = np.ones((diff, 2)) * path[-1]
            paths[i] = np.vstack([path, padding])

        new_trajs[i, :] = paths[i]
    if len(old_path_history) == 0:
        return new_trajs
    else:
        combined_trajs = np.zeros(
            (len(paths) + len(old_path_history), max(new_trajs.shape[1], old_path_history.shape[1]), 2))
        combined_trajs[:len(old_path_history), :old_path_history.shape[1]] = old_path_history
        combined_trajs[len(paths):, :new_trajs.shape[1]] = new_trajs
        return combined_trajs
