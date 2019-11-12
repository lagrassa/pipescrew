import pygame
import numpy as np
from Box2D import *
#convert to Box2D

"""
2D gridworld. obstacles are represented as Obstacle
They need to be boxes. Origin is the top right
300x400 grid
"""
class NavEnv():
    def __init__(self, slip=True ):
        self.slip=slip
        self.gridsize = [300,400]
        self.m = 1
        self.mu = -1
        self.ppm = 10.
        self.joint = None
        self.ice_boundary_x = 150./self.ppm
        pygame.init()
        self.screen = pygame.display.set_mode(self.gridsize)
        self.screen.fill((255,255,255))
        pygame.display.flip()
        self.goal = np.array((200,300))
        self.start = np.array((0,5)) + 5*np.random.random((2,))
        self.world = b2World(gravity=(0,0), doSleep=True)
        self.agent = self.world.CreateDynamicBody(
            position=self.start,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(3, 3))
            )
        )
        self.ground = self.world.CreateStaticBody(
            shapes=b2PolygonShape(box=(self.gridsize[0]/self.ppm, self.gridsize[1]/self.ppm))
        )
        self.check_and_set_mu(dirty_bit = True)
        self.pos_history = []
        self.path_color = (1,0,0,1)
        gap = 25
        length = 90
        dist_down=150
        self.obstacles = [Obstacle(np.array((1,dist_down)),length,10),
                           Obstacle(np.array((length+gap,dist_down)),length,10),
                           Obstacle(np.array((2*length+2*gap,dist_down)),length,10),
        ]
        self.dt = 0.1
        for obs in self.obstacles:
            obs.render(self.screen)

    def check_and_set_mu(self,dirty_bit = False):
        old_pos = np.array(self.agent.position.tuple)
        no_ice_mu = 0.9
        ice_mu = 0.05
        if old_pos[0] < self.ice_boundary_x:
            if self.mu != no_ice_mu:
                self.mu = no_ice_mu
                dirty_bit = True
        else:
            if self.mu != ice_mu:
                self.mu = ice_mu
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
        self.check_and_set_mu()
        move = np.array((x,y))
        self.agent.ApplyForceToCenter(force=move,wake = True) 
        self.world.Step(self.dt, 6, 2)
        new_pos = np.array(self.agent.position.tuple)
        print(new_pos)
        self.world.ClearForces()
        self.pos_history.append(new_pos)
        import ipdb; ipdb.set_trace()
        self.render()



    def render(self):
        view_wid =10
        start_rect = pygame.Rect(self.start[0], self.start[1], view_wid, view_wid)
        goal_rect = pygame.Rect(self.goal[0],self.goal[1], view_wid, view_wid)
        pygame.draw.rect(self.screen, (170,0,0,1),start_rect,0)
        pygame.draw.rect(self.screen, (0,170,0,1),goal_rect,0)
        for i in range(len(self.pos_history)-1):
            pygame.draw.line(self.screen,self.path_color,(self.ppm*self.pos_history[i]).astype(np.int32),(self.ppm*self.pos_history[i+1]).astype(np.int32), 5)
        pygame.display.flip()

class Obstacle():
    def __init__(self, origin, w,y):
        self.origin = origin
        self.x = w
        self.y = y
    def render(self,screen):
        py_rect = pygame.Rect(self.origin[0], self.origin[1],self.x,self.y)
        pygame.draw.rect(screen, (50,0,0,1),py_rect,0)
    def in_collision(self,pt):
        if (pt > self.origin).all() and (pt < self.origin + np.array(self.x, self.y)).all():
            return True
        return False

if __name__ == "__main__":
    ne = NavEnv() 
    for i in range(50):
        ne.step(0,3)
        
