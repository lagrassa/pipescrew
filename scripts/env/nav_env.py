import pygame
from obstacle import line_world_obstacles, Obstacle, two_openings_obstacles, Quicksand
import numpy as np
from gym.spaces import Box
from Box2D import *
pygame.init()

# convert to Box2D

"""
2D gridworld. obstacles are represented as Obstacle
They need to be boxes. Origin is the top right
300x400 grid
"""
LIGHT_GREEN = (172, 255, 192)
LIGHT_BLUE = (158, 242, 254)


class NavEnv:
    def __init__(self, start: object, goal: object, slip: object = True, visualize=False,
                 observation_space=None,
                 autoencoder = None, shift: object = None,
                 obstacles: object = two_openings_obstacles(),
                 gridsize: object = np.array([300, 400])) -> object:
        self.slip = slip
        self.gridsize = gridsize
        self.autoencoder = autoencoder
        self.m = 1
        self.steps_taken = 0
        self.visualize = visualize
        self.metadata = {}
        self.mu = -1
        self.ppm = 100
        self.view_wid = 4
        self.reward_range = Box(low=np.array([-100]),high=np.array([0]))
        self.joint = None
        self.obs_width = 12
        obs_size = (2 * self.obs_width + 1) ** 2
        self.observation_space = Box(low=np.zeros(obs_size), high=np.ones(obs_size) * 255) if observation_space is None else observation_space
        self.observation_space = Box(low=np.zeros(4), high=np.ones(4) * 100)
        self.action_space = Box(low=np.array([0, -2]), high=np.array([0, 2]))
        self.ice_boundary_x = 150. / self.ppm

        self.quicksand_ring = Quicksand((0.4, 0.4), 0.00)
        self.goal = goal
        self.start = start
        self.max_force = 4
        self.mass = 1
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.robot_width = 0.01
        self.agent = self.world.CreateDynamicBody(
            position=self.start,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(0.01, 0.01)),
                density=self.mass/(0.01*0.01)
            )
        )
        self.ground = self.world.CreateStaticBody(
            shapes=create_box2d_box(self.gridsize[1] / self.ppm, self.gridsize[0] / self.ppm)
        )
        self.check_and_set_mu(dirty_bit=True)
        self.pos_history = []
        self.desired_pos_history = []
        self.path_color = (1, 0, 0, 1)
        self.obstacles = obstacles
        for obstacle in self.obstacles:
            self.world.CreateStaticBody(
                position=obstacle.origin,
                shapes=create_box2d_box(obstacle.y, obstacle.x)
            )

        self.dt = 0.05
        self.visualize = visualize
        self.belief_screen = pygame.display.set_mode(self.gridsize)
        self.belief_screen.fill((255, 255, 255))
        self.render(belief_only=True, flip=True)
        if self.visualize:
            self.setup_visuals()
            self.render()

    def get_pos(self):
        return np.array(self.agent.position.tuple)

    def setup_visuals(self):
        self.world_screen = pygame.display.set_mode(self.gridsize)
        self.world_screen.fill((255, 255, 255))
        pygame.display.flip()

    def get_vel(self):
        return np.array(self.agent.GetLinearVelocityFromLocalPoint((0, 0)))

    def close(self):
        pass

    def check_and_set_mu(self, dirty_bit=False):
        old_pos = np.array(self.agent.position.tuple)
        no_ice_mu = 0.6
        self.ext_mu = no_ice_mu
        ice_mu = 0.02
        quicksand_mu = 3
        if old_pos[0] < self.ice_boundary_x:
            if self.mu != no_ice_mu:
                self.mu = no_ice_mu
                dirty_bit = True
        else:
            if self.mu != ice_mu:
                self.mu = ice_mu
                dirty_bit = True
        if self.quicksand_ring.in_collision(old_pos):
            self.mu = quicksand_mu
            dirty_bit = True
        if dirty_bit:
            if self.joint is not None:
                self.world.DestroyJoint(self.joint)
            self.joint = self.world.CreateFrictionJoint(bodyA=self.agent, bodyB=self.ground, maxForce=self.m * self.mu)

    '''
    Rolls dynamical system 1 dt according to x'', y''
    implements openai gym interface
    '''
    def step(self, action, dt = None, rl=True):
        old_pos = np.array(self.agent.position.tuple)
        if dt is None: dt = self.dt
        self.pos_history.append(old_pos)
        self.check_and_set_mu()
        if np.linalg.norm(action) > self.max_force:
            action = (action/ np.linalg.norm(action))*self.max_force
        move = action
        self.agent.ApplyForceToCenter(force=move.tolist(), wake=True)
        self.world.Step(dt, 6, 2)
        self.world.ClearForces()
        self.steps_taken += 1
        if self.visualize:
            self.render()
        done = (self.goal_distance() <= 0.01) or (self.goal_distance() > 0.15)
        rew_scale = 3
        if rl:
            return self.rl_obs(), -rew_scale * self.goal_distance(), done, {}
        else:
            return self.get_obs_low_dim() -rew_scale * self.goal_distance(), done, {}

    def rl_obs(self):
        #return np.hstack([self.autoencoder(self.get_obs()), self.get_pos(), self.get_vel()]).flatten()
        return np.hstack([self.autoencoder(self.get_obs()), self.get_vel()]).flatten()
    def get_state(self):
        return np.hstack([self.get_pos(), self.get_vel()])
    def plot_path(self, path):
        self.render(flip=False)
        pygame.draw.lines(self.world_screen, (0, 0, 255), False, self.ppm * path, 6)
        pygame.display.flip()

    def goal_distance(self):
        return np.linalg.norm(self.get_pos() - self.goal)

    def goal_condition_met(self):
        ret = self.goal_distance() < 0.025
        return ret

    def reset(self):
        assert(self.autoencoder is not None)
        autoencoder = self.autoencoder
        self.__init__(start=self.start, obstacles=self.obstacles,
                      goal=self.goal, gridsize=self.gridsize, autoencoder = autoencoder,
                      observation_space=self.observation_space,
                      visualize=self.visualize)
        assert(self.autoencoder is not None)
        return self.rl_obs()

    def set_autoencoder(self, fn):
        assert fn is not None
        self.autoencoder = fn
        obs_shape = self.rl_obs()
        self.observation_space = Box(low = -np.inf*np.ones(obs_shape.shape), high = np.inf*np.ones(obs_shape.shape))

    def get_obs_low_dim(self):
     return np.array([self.agent.position, self.agent.GetLinearVelocityFromLocalPoint((0, 0))]).flatten()

    """
    2D occupancy grid @param width units in pix away from the agent
    with the agent centered. 
    
    """
    def get_obs(self):
        self.belief_screen.fill((255,255,255))
        self.render(belief_only=True, flip=True)
        grid = np.zeros((2 * self.obs_width + 1, 2 * self.obs_width + 1))
        i_range =  range(int(self.ppm*(self.agent.position[0])) - self.obs_width, int(self.ppm * (self.agent.position[0])) + self.obs_width + 1)
        j_range = range(int(self.ppm*(self.agent.position[1])) - self.obs_width, int(self.ppm * (self.agent.position[1])) + self.obs_width + 1)
        for grid_i, i in zip(range(len(i_range)), i_range):
            for grid_j, j in zip(range(len(j_range)),j_range):
                i = np.clip(i, 0,self.gridsize[0]-1)
                j = np.clip(j, 0,self.gridsize[1]-1)
                grid[grid_i,grid_j] = np.mean(self.belief_screen.get_at((i,j))[0:3])

        return grid.T #n
        # .flatten()
        #scalar = 1.0
        #return scalar * np.array([self.agent.position, self.agent.GetLinearVelocityFromLocalPoint((0, 0))]).flatten()

        # returns image of area around agent

    def render_start_goal(self):
        start_rect = pygame.Rect(self.ppm * self.start[0] - self.view_wid / 2,
                                 self.ppm * self.start[1] - self.view_wid / 2, self.view_wid, self.view_wid)
        goal_rect = pygame.Rect(self.ppm * self.goal[0] - self.view_wid / 2.,
                                self.ppm * self.goal[1] - self.view_wid / 2., self.view_wid, self.view_wid)
        pygame.draw.rect(self.world_screen, (170, 0, 0, 1), start_rect, 0)
        pygame.draw.rect(self.world_screen, (0, 170, 0, 1), goal_rect, 0)

    def render(self, flip=True, belief_only=False):
        # draw green for normal, light blue for the ice
        if not belief_only:
            green_rect = pygame.Rect(0, 0, self.gridsize[0], self.ice_boundary_x * self.ppm)
            ice_rect = pygame.Rect(0, self.ice_boundary_x * self.ppm, self.gridsize[0], self.gridsize[1])
            pygame.draw.rect(self.world_screen, LIGHT_GREEN, green_rect, 0)
            pygame.draw.rect(self.world_screen, LIGHT_BLUE, ice_rect, 0)
            self.render_start_goal()

        robot_rect = pygame.Rect(self.ppm * self.agent.position[0] - self.view_wid / 2.,
                                 self.ppm * self.agent.position[1] - self.view_wid / 2., self.view_wid, self.view_wid)
        pygame.draw.rect(self.belief_screen, (10, 0, 200, 1), robot_rect, 0)
        if not belief_only:
            pygame.draw.rect(self.world_screen, (10, 0, 200, 1), robot_rect, 0)
            for obs in self.obstacles:
                obs.render(self.world_screen, ppm=self.ppm)
        for obs in self.obstacles:
            obs.render(self.belief_screen, ppm=self.ppm)
        if not belief_only:
            self.quicksand_ring.render(self.world_screen, ppm=self.ppm)
            for i in range(len(self.pos_history) - 1):
                pygame.draw.line(self.world_screen, self.path_color, (self.ppm * self.pos_history[i]).astype(np.int32),
                                 (self.ppm * self.pos_history[i + 1]).astype(np.int32), 8)
            for i in range(len(self.desired_pos_history) - 1):
                pygame.draw.line(self.world_screen, (0, 100, 0, 1),
                                 (self.ppm * self.desired_pos_history[i]).astype(np.int32),
                                 (self.ppm * self.desired_pos_history[i + 1]).astype(np.int32), 2)
            # if np.random.randint(2) == 2:
        if flip:
            pygame.display.flip()

    def collision_fn(self, pt):
        ret = not (not np.array([obs.get_particles_in_collision(pt) for obs in self.obstacles]).any() and not (
                pt > self.gridsize / self.ppm).any()) or (pt < 0).any()
        return ret


def create_box2d_box(h, w):
    return b2PolygonShape(vertices=[(0, 0), (0, h), (w, h), (w, 0), (0, 0)])


if __name__ == "__main__":
    ne = NavEnv(np.array([0., 0.]), np.array([0.5, 0.5]))
    ne.render()
    ne.collision_fn((2, 0.1 + 150 / ne.ppm))
    for i in range(50):
        ne.step(0, 3)
