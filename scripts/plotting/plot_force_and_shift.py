import numpy as np
import matplotlib.pyplot as plt
import rl_plot.make_plots as mp
plt.rcParams['font.size']=30
data = np.load("shift_to_force_to_success_3.npy")
data = data[:40]
#shifts, collapse along 0th axis
mean = np.mean(data, axis=0) 
import ipdb; ipdb.set_trace()
mean  = mp.moving_average(mean, n=3)
std = 0.02*np.std(data, axis=0) 
std  = mp.moving_average(std, n=3)
xs = np.load("forces_3.npy")[:-2]
mp.plot_line(mean, std, xaxis=xs)
plt.xlabel("Constraint force")
plt.ylabel("Success probability")
plt.rcParams['font.size']=40
plt.show()

