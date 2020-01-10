import numpy as np
import matplotlib.pyplot as plt
import rl_plot.make_plots as mp
plt.rcParams['font.size']=30
data = np.load("data/dmp_successes.npy")
data_baseline = np.load("data/plan_successes.npy")
#shifts, collapse along 0th axis
mean = np.mean(data, axis=0) 
import ipdb; ipdb.set_trace()
mean  = mp.moving_average(mean, n=3)
std = np.std(data, axis=0) 
std  = mp.moving_average(std, n=3)
xs = np.linspace(0, 1000, 50)
mp.plot_line(mean, std, xaxis=xs)
plt.plot(xs, np.mean(data_baseline))
plt.xlabel("Samples")
plt.ylabel("Success probability")
plt.rcParams['font.size']=40
plt.show()

