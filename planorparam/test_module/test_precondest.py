import matplotlib.pyplot as plt
def my_contour(*args, **kwargs):
    res = plt.contour(*args, **kwargs)
    plt.clabel(res)
    return res
plt.contour = my_contour
from agent import precond_estimation as pe
import numpy as np
from agent.skills import test_skill
from env.peginsertenv import PegInsertEnv

def test_active_learn():
    env = PegInsertEnv()
    gp = pe.active_learn(env, test_skill, 20, N_init=20)
    expected_mean_good = gp.predict(np.array([0,0,0.05]).reshape(1,-1))[0]
    expected_mean_bad = gp.predict(np.array([6,0,0.13]).reshape(1,-1))[0]
    print("expected mean good", expected_mean_good, "expected mean bad", expected_mean_bad)
    #assert(abs(expected_mean_bad) < 0.01)
    #assert(abs(expected_mean_good) > 0.1)
    print("Tests passed")

    gp.plot(visible_dims=[0,2], free_dims=[1])
    #plt.clabel(inline=1, fontsize=10)
    plt.legend()
    plt.colormaps()
    plt.show()

def test_forcefeedback():
    env = PegInsertEnv()
    for i in range(3):
        env.step([0,0,-0.05, 50])
    for force_feedback_type in [None, "cont", "binary", "discrete"]:
        env.force_feedback_type = force_feedback_type
        state = env.observe_state()
        print(force_feedback_type, state[3:])


        #make sure it's pressing down


#test_active_learn()
test_forcefeedback()



