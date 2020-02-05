from nav_env import NavEnv
import numpy as np
def test_step():
    ne1 = NavEnv([0.1,0.1], [0.4,0.6])
    original_pos = ne1.agent.position.copy()
    print("original pos", original_pos)
    for i in range(500):
        ne1.step(5.,5.)
    new_pos = ne1.agent.position

    print("new pos", new_pos)
    assert not np.allclose(original_pos, new_pos)


test_step()
