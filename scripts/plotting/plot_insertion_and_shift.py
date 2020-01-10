import matplotlib.pyplot as plt
import rl_plot.make_plots as mp
import numpy as np
for exp, color in zip(["40", "60", "80", "90"], ('r', 'g', 'b', 'y')):
    ins = np.load("rewards_"+exp+".npy")
    shifts = np.load("shifts_"+exp+".npy")
    mp.plot_line(np.mean(ins, axis=0), np.std(ins, axis=0),xaxis=shifts, label=exp, color = color)
plt.legend()
plt.show()
