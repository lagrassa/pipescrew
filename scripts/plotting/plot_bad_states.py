import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
bad_model_states = np.load("../real_robot/data/bad_model_states.npy")
good_model_states = np.load("../real_robot/data/good_model_states.npy")
import ipdb; ipdb.set_trace()
ax.scatter(bad_model_states[:,0],bad_model_states[:,1], bad_model_states[:,2], color='r', zdir='z', s=20, c=None, depthshade=True, label="bad points")
ax.scatter(good_model_states[:,0],good_model_states[:,1], good_model_states[:,2], color='g', zdir='z', s=20, c=None, depthshade=True, label="good points")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
