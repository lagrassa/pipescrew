#from .planning import Planner
from itertools import product

import numpy as np
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
import matplotlib.pyplot as plt
class Agent:

    def __init__(self, agent_name="agent1"):
        self.agent_cfg = {}
        self.agent_cfg["data_path"] = "data/{}/".format(agent_name)
    def collect_transition_data(self, vec_env, plans=[]):
        """
        In the original method, random and then use the planner to collect data. Pass them in for now so you know theyre' valid plans
        """
        # in original method, plan here
        for plan in plans:
            self.execute_plan(plan, vec_env)

    def execute_plan(self, plan, vec_env):
        """
        Execute series of operators. Results are saved.
        """
        for op in plan:
            op.cfg = self.agent_cfg
            op.monitor_execution(vec_env)


    def train_classifier(self, plot_classification =True):
        #load data, train classifier. Try KLR
        operator_name = "PushInDir"
        fn = self.agent_cfg["data_path"]+operator_name+".npy"
        data = np.load(fn, allow_pickle=True).all()
        actual_states = data["actual_next_states"]
        expected_states = data["expected_next_states"]
        state = data["states"]
        actions = data["actions"]
        input_vec = np.hstack([state, actions])
        std_vec = np.std(input_vec, axis=0)
        std_thresh = 0.01
        varying_input_vec = input_vec[:,std_vec > std_thresh]
        varying_input_vec = varying_input_vec[:,[1,0]] #for plotting, will give worse results
        scaler = preprocessing.StandardScaler().fit(varying_input_vec)
        input_vec_scaled = scaler.transform(varying_input_vec)
        deviations = np.linalg.norm(actual_states[:, :3] - expected_states[:, :3], axis=1)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(10,(0.1, 10), nu=1.5)
        #kernel = C(1.0, (1e-3, 1e3)) * RBF(20, (1e-3, 1e2))
        #[0 and 1 ] of the varying version will be the interesting part
        gp = GPR(kernel=kernel, n_restarts_optimizer=9)
        gp.fit(input_vec_scaled, deviations)
        deviations_pred, sigma = gp.predict(input_vec_scaled, return_std=True)
        #plot using 2D
        print("Error", np.linalg.norm(deviations_pred-deviations))
        # Input space
        if plot_classification:
            from autolab_core import YamlConfig
            import matplotlib.patches
            cfg = YamlConfig("cfg/franka_block_push_two_d.yaml")
            self._board_names = [name for name in cfg.keys() if "boardpiece" in name]
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            for _board_name in self._board_names:
                color = cfg[_board_name]['rb_props']["color"]
                pose = [cfg[_board_name]["pose"]["x"], cfg[_board_name]["pose"]["y"]]
                dims = [cfg[_board_name]["dims"]["depth"], cfg[_board_name]["dims"]["width"]]
                #draw them.
                pose[0] -= (dims[0]/2.)
                pose[1] -= (dims[1]/2.)
                patch = matplotlib.patches.Rectangle(pose, dims[0], dims[1], fill=False, edgecolor=color+[0.8,], linewidth=20)
                ax.add_patch(patch)
            X = varying_input_vec
            y = deviations
            num_pts = 100
            x1 = np.linspace(X[:, 0].min()-0.05, X[:, 0].max()+0.05, num_pts)  # p
            x2 = np.linspace(X[:, 1].min()-0.05, X[:, 1].max()+0.05, num_pts)  # q
            x = (np.array([x1, x2])).T
            x1x2 = np.array(list(product(x1, x2)))
            y_pred, MSE = gp.predict(scaler.transform(x1x2), return_std=True)
            X0p, X1p = x1x2[:,0].reshape(num_pts,num_pts), x1x2[:,1].reshape(num_pts,num_pts)
            Zp = np.reshape(y_pred,(num_pts,num_pts))

            # alternative way to generate equivalent X0p, X1p, Zp
            # X0p, X1p = np.meshgrid(x1, x2)
            # Zp = [gp.predict([(X0p[i, j], X1p[i, j]) for i in range(X0p.shape[0])]) for j in range(X0p.shape[1])]
            # Zp = np.array(Zp).T

            block_shift = 0#0.07/2
            ax.pcolormesh(X0p-block_shift, X1p-block_shift, Zp, cmap="Greens", vmin=-0.01)
            #ax.scatter(X0p, X1p, c=Zp, cmap="Greens", vmin=-0.1)
            plot = ax.scatter(X[:,0]-block_shift, X[:,1]-block_shift, c=deviations, cmap="Greens", vmin=-0.01, edgecolor="k")
            #ax.scatter(X[:,0], X[:,1], c=deviations_pred, cmap="Greens", vmin=-0.05)
            x_max = 0.3
            x_min = -0.3
            z_max = 0.6
            z_min = 0.1
            plt.xlim([x_max, x_min])
            plt.ylim([z_max, z_min])
            plt.colorbar(plot)
            #fig.colorbar(ax)
            plt.show()



