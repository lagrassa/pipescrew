class BlockPushPolicy():
    def __init__(self):
        pass
    """
    :param goal desired block pose in np array
    :return [(state,action)] tuple for each env
    states and action is  each a 3D array where [env_id, state, t] 
    ex) the state for env 1 at time 3 = states[1,:,3]
    """
    def plan(self, start, goal):
        return states, actions

    def monitored_execution(self, vec_env, states, action, tol = 0.01):
        """
        :param states, actions states and high level actions to execute
        :return list of states where the agent deviated, amount of each deviation
        """
