from carbongym import gymapi
from carbongym_utils.math_utils import RigidTransform_to_transform, transform_to_RigidTransform
import numpy as np
from autolab_core import RigidTransform
class BlockPushPolicy():
    def __init__(self):
        pass
    """
    :param goal desired block pose in np array
    :return [(state,action)] tuple for each env
    states and action is  each a 3D array where [env_id, state, t] 
    ex) the state for env 1 at time 3 = states[1,:,3]
    """
    def plan(self, start, goal,  horizon = 5):
        states = np.zeros(start.shape + (horizon,))
        action_shape = (8,)
        actions = np.zeros((start.shape[0],)+action_shape+(horizon-1,))
        for i in range(start.shape[0]):
           states[i,:,:] = np.linspace(start[i,:], goal[i,:], horizon).T
        stiffness = 50
        for t in range(states.shape[-1]-1):
            actions[:, 3:, t] = np.array([0, 0, 0, 1, stiffness]).reshape(actions[:, 3:, t].shape)
            actions[:,:3,t] = (states[:,:3,t+1]-states[:,:3, t])
        return states, actions

    def monitored_execution(self, vec_env, states, action, tol = 0.01):
        """
        :param states, actions states and high level actions to execute
        :return list of states where the agent deviated, amount of each deviation
        """
        deviations = np.zeros(states.shape[0], 1)
        for t in range(states.shape[-1]-1):
            next_state = vec_env.step(action[:,:,t])
            expected_state = states[:,:,t+1]
            new_deviations = np.linalg.norm(next_state-expected_state, axis=1)
            deviations = np.hstack([deviations, new_deviations])
        return deviations

    def go_to_push_start(self, vec_env, z_offset=0.09):
        for env_index, env_ptr in enumerate(vec_env._scene.env_ptrs):
            block_ah = vec_env._scene.ah_map[env_index][vec_env._block_name]
            block_transform = vec_env._block.get_rb_transforms(env_ptr, block_ah)[0]
            ee_transform = vec_env._franka.get_ee_transform(env_ptr, vec_env._franka_name)
            grasp_transform = gymapi.Transform(p=block_transform.p, r=ee_transform.r)
            pre_grasp_transfrom = gymapi.Transform(p=grasp_transform.p, r=grasp_transform.r)
            pre_grasp_transfrom.p.z += z_offset
            pre_grasp_transfrom.p.y -= 0.02
            rt = transform_to_RigidTransform(pre_grasp_transfrom)
            rot_matrix = RigidTransform.y_axis_rotation(np.pi/2)
            rt.rotation = np.dot(rot_matrix, rt.rotation)
            pre_grasp_transfrom = RigidTransform_to_transform(rt)
            stiffness = 1e3
            vec_env._franka.set_attractor_props(env_index, env_ptr, vec_env._franka_name,
                                             {
                                                 'stiffness': stiffness,
                                                 'damping': 6 * np.sqrt(stiffness)
                                             })
            vec_env._franka.set_ee_transform(env_ptr, env_index, vec_env._franka_name, pre_grasp_transfrom)
        #for i in range(50):
        #    vec_env.step([0.1,0.1,0.1,0,0,0,1])
    def go_to_block(self, vec_env):
        for env_index, env_ptr in enumerate(vec_env._scene.env_ptrs):
            block_ah = vec_env._scene.ah_map[env_index][vec_env._block_name]
            block_transform = vec_env._block.get_rb_transforms(env_ptr, block_ah)[0]
            ee_transform = vec_env._franka.get_ee_transform(env_ptr, vec_env._franka_name)
            grasp_transform = gymapi.Transform(p=block_transform.p, r=ee_transform.r)
            pre_grasp_transfrom = gymapi.Transform(p=grasp_transform.p, r=grasp_transform.r)
            pre_grasp_transfrom.p.z += vec_env._cfg['block']['dims']['width']/2.

            pre_grasp_transfrom.p.y -=0.01
            stiffness = 50
            vec_env._franka.set_attractor_props(env_index, env_ptr, vec_env._franka_name,
                                                {
                                                    'stiffness': stiffness,
                                                    'damping': 2 * np.sqrt(stiffness)
                                                })
            vec_env._franka.set_ee_transform(env_ptr, env_index, vec_env._franka_name, pre_grasp_transfrom)

