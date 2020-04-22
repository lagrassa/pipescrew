"""
Uses a planner to push a block
"""
class BlockPushPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name

        self._time_horizon = 1000

        self.reset()

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []

    def __call__(self, scene, env_index, t_step, _):
        env_ptr = scene.env_ptrs[env_index]
        ah = scene.ah_map[env_index][self._franka_name]

        if t_step == 20:
            block_ah = scene.ah_map[env_index][self._block_name]
            block_transform = self._block.get_rb_transforms(env_ptr, block_ah)[0]

            ee_transform = self._franka.get_ee_transform(env_ptr, self._franka_name)
            self._init_ee_transforms.append(ee_transform)

            grasp_transform = gymapi.Transform(p=block_transform.p, r=ee_transform.r)
            pre_grasp_transfrom = gymapi.Transform(p=grasp_transform.p, r=grasp_transform.r)
            pre_grasp_transfrom.p.y += 0.2

            self._grasp_transforms.append(grasp_transform)
            self._pre_grasp_transforms.append(pre_grasp_transfrom)

            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 200:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

        if t_step == 300:
            self._franka.close_grippers(env_ptr, ah)

        if t_step == 400:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 500:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

        if t_step == 600:
            self._franka.open_grippers(env_ptr, ah)

        if t_step == 700:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 800:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._init_ee_transforms[env_index])