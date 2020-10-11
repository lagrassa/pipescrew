import time
from planning.robot_push_controller import RobotPushController
from odium.utils.env_utils.make_env import make_env
from odium.utils.simple_utils import set_global_seed
from planning.push_rts_agent import push_rts_agent
from odium.agents.fetch_agents.arguments import get_args
from autolab_core import YamlConfig
import numpy as np
def get_env_params(args, env):
    # Construct environment parameters
    obs = env.reset()
    params = {'obs': env.observation_space.shape[0],
              'goal': env.observation_space.shape[0],
              'action': env.action_space.shape,
              'num_actions': env.num_discrete_actions,
              'num_features': env.num_features,
              'max_timesteps': args.planning_rollout_length,
              'offline_max_timesteps': 50,
              'qpos':env.env.get_states().shape[0],
              'qvel':env.env.get_vels().shape[0]
              }
    return params


def integration_test(args):
    env_name = "GymFrankaBlockPushEnv-v0"
    cfg = YamlConfig("cfg/franka_block_push.yaml")
    env_id = 0
    planning_env_id = 1
    env = make_env(env_name=env_name,
                   env_id=env_id,
                   discrete=True,
                   cfg=cfg)
    #env.seed(args.seed)
    #set_global_seed(args.seed)
    if args.deterministic:
        env.make_deterministic()
    delta_goal =  np.array([-0.1])
    controller = RobotPushController(env, delta_goal)
    env_params = get_env_params(args, env)
    trainer = push_rts_agent(args, env_params, env, planning_env_id, controller)
    n_steps = trainer.learn_online_in_real_world(args.max_timesteps)
    print('REACHED GOAL in', n_steps, 'by agent', args.agent)
    time.sleep(5)
    return n_steps


if __name__ == '__main__':
    args = get_args()
    integration_test(args)
