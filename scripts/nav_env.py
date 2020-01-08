import pygame
from obstacle import line_world_obstacles, Obstacle, two_openings_obstacles, Quicksand
import numpy as np
from Box2D import *

#convert to Box2D

"""
2D gridworld. obstacles are represented as Obstacle
They need to be boxes. Origin is the top right
300x400 grid
"""
LIGHT_GREEN=(172, 255, 192)
LIGHT_BLUE = (158,242,254)
class NavEnv():
    def __init__(self, start,goal,slip=True , visualize=True, shift=None, obstacles = two_openings_obstacles()):
        self.slip=slip
        self.gridsize = np.array([300,400])
        self.m = 1
        self.steps_taken = 0
        self.mu = -1
        self.ppm = 200.
        self.joint = None
        self.ice_boundary_x = 150./self.ppm
        self.quicksand_ring = Quicksand((0.4,0.4), 0.00)
        self.goal = goal
        self.start = start 
        self.world = b2World(gravity=(0,0), doSleep=True)
        self.agent = self.world.CreateDynamicBody(
            position=self.start,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(0.02, 0.02)),
            )
        )
        self.ground = self.world.CreateStaticBody(
            shapes=create_box2d_box(self.gridsize[1]/self.ppm, self.gridsize[0]/self.ppm)        
        )
        self.check_and_set_mu(dirty_bit = True)
        self.pos_history = []
        self.desired_pos_history=[]
        self.path_color = (1,0,0,1)
        self.obstacles = obstacles
        for obstacle in self.obstacles:
            self.world.CreateStaticBody(
            position=obstacle.origin,
            shapes=create_box2d_box(obstacle.y, obstacle.x)
        )

        self.dt = 0.01
        self.visualize = visualize
        if self.visualize:
            pygame.init()
            self.screen = pygame.display.set_mode(self.gridsize)
            self.screen.fill((255,255,255))
            pygame.display.flip()
            self.render()

    def get_pos(self):
        return np.array(self.agent.position.tuple)

    def get_vel(self):
        return np.array(self.agent.GetLinearVelocityFromLocalPoint((0,0)))
    def close(self):
        pass

    def check_and_set_mu(self,dirty_bit = False):
        old_pos = np.array(self.agent.position.tuple)
        no_ice_mu = 0.6
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
            self.joint = self.world.CreateFrictionJoint(bodyA = self.agent, bodyB = self.ground,maxForce = self.m*self.mu )

    '''
    Rolls dynamical system 1 dt according to x'', y''
    ''' 
    def step(self,x,y):
        old_pos = np.array(self.agent.position.tuple)
        self.pos_history.append(old_pos)
        self.check_and_set_mu()
        move = np.array((x,y))
        #import ipdb; ipdb.set_trace()
        self.agent.ApplyForceToCenter(force=move,wake = True)
        #import ipdb; ipdb.set_trace()
        self.world.Step(self.dt, 16, 10)
        new_pos = np.array(self.agent.position.tuple)
        self.world.ClearForces()
        self.steps_taken +=1
        if self.visualize:
            self.render()
    def plot_path(self,path):
        self.render(flip=False)
        pygame.draw.lines(self.screen, (0,0,255),False,self.ppm*path, 6)
        pygame.display.flip()

    def goal_condition_met(self):
        return np.linalg.norm(self.get_pos()-self.goal) < 0.02

    def render(self, flip = True):
        view_wid =10
        #draw green for normal, light blue for the ice
        green_rect = pygame.Rect(0, 0, self.gridsize[0], self.ice_boundary_x*self.ppm)
        ice_rect = pygame.Rect(0,self.ice_boundary_x*self.ppm, self.gridsize[0],self.gridsize[1] )
        pygame.draw.rect(self.screen, LIGHT_GREEN,green_rect,0)
        pygame.draw.rect(self.screen, LIGHT_BLUE,ice_rect,0)
        start_rect = pygame.Rect(self.ppm*self.start[0], self.ppm*self.start[1], view_wid, view_wid)
        goal_rect = pygame.Rect(self.ppm*self.goal[0],self.ppm*self.goal[1], view_wid, view_wid)
        pygame.draw.rect(self.screen, (170,0,0,1),start_rect,0)
        pygame.draw.rect(self.screen, (0,170,0,1),goal_rect,0)
        for obs in self.obstacles:
            obs.render(self.screen, ppm = self.ppm)
        self.quicksand_ring.render(self.screen,ppm = self.ppm)
        for i in range(len(self.pos_history)-1):
            pygame.draw.line(self.screen,self.path_color,(self.ppm*self.pos_history[i]).astype(np.int32),(self.ppm*self.pos_history[i+1]).astype(np.int32), 8)
        for i in range(len(self.desired_pos_history)-1):
            pygame.draw.line(self.screen,(0,100,0,1),(self.ppm*self.desired_pos_history[i]).astype(np.int32),(self.ppm*self.desired_pos_history[i+1]).astype(np.int32), 2)
        #if np.random.randint(2) == 2:
        if flip:
            pygame.display.flip()

    def collision_fn(self, pt):
        ret =  np.array([obs.in_collision(pt) for obs in self.obstacles]).any() or (pt > self.gridsize/self.ppm).any() or (pt < 0).any()
        return ret


def create_box2d_box(h, w):
    return b2PolygonShape(vertices = [(0,0), (0,h), (w, h),(w,0), (0,0)] )



if __name__ == "__main__":
    ne = NavEnv(np.array([0.,0.]),np.array([0.5,0.5])) 
    ne.render()
    ne.collision_fn((2,0.1+150/ne.ppm))
    for i in range(50):
        ne.step(0,3)
        
