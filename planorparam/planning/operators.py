class GoToSide():
    deviations = []
    def __init__(self, sidenum):
        self.sidenum = sidenum
    def execute_prim(self, env):
        env.goto_side(self.sidenum)

class PushInDir:
    states = []
    actions = []
    expected_next_states = [] #maybe files instead of static vars
    actual_next_states = []
    def __init__(self, sidenum, amount, T):
        self.sidenum = sidenum
        self.amount = amount
        self.T = T
        self.transition_function =[]
    def execute_prim(self, env):
        time, res = env.push_in_dir(self.sidenum, self.amount, self.T)




