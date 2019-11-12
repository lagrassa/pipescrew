import pygame
import numpy as np
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
        pygame.init()
        self.screen = pygame.display.set_mode(self.gridsize)
        self.screen.fill((255,255,255))
        pygame.display.flip()
        self.goal = np.array((200,300))
        self.start = np.array((0,5)) + 5*np.random.random((2,))
        self.pos = self.start[:]
        self.vel = np.array([0,0])
        self.pos_history = []
        self.path_color = (1,0,0,1)
        self.obstacles = [(Obstacle(np.array((20,20)),50))]
        self.m = 1
        self.mu = 0.5
        self.dt = 0.01
        for obs in self.obstacles:
            obs.render(self.screen)
    '''
    Rolls dynamical system 1 dt according to x'', y''
    ''' 
    def step(self,x,y):
        old_pos = self.pos[:]
        move = np.array((x,y))
        if np.linalg.norm(move) > self.mu*self.m:
            net_force = np.array([0,0])
        else:
            net_force = move - np.sign(move)*(self.mu*self.m)
        new_pos = old_pos + self.vel*self.dt + 0.5*net_force*self.dt**2
        self.vel = self.vel+ move*self.dt
        if not self.collision_fn(new_pos):
            self.pos = new_pos
        else:
            self.vel = 0
        self.pos_history.append(new_pos[:])
        pygame.draw.line(self.screen,self.path_color,old_pos.astype(np.int32),new_pos.astype(np.int32), 5)
        self.render()

    def go_to(self, x,y):
        self.step(*np.array([x,y])-self.pos)

    def collision_fn(self,pt):
        if pt[0] < 0 or pt[1] < 0:
            return True
        if pt[1] > self.gridsize[1] or pt[0] > self.gridsize[0]:
            return True
        return np.array([obs.in_collision(pt) for obs in self.obstacles]).any()


    def render(self):
        view_wid =10
        start_rect = pygame.Rect(self.start[0], self.start[1], view_wid, view_wid)
        goal_rect = pygame.Rect(self.goal[0],self.goal[1], view_wid, view_wid)
        pygame.draw.rect(self.screen, (170,0,0,1),start_rect,0)
        pygame.draw.rect(self.screen, (0,170,0,1),goal_rect,0)
        pygame.display.flip()

class Obstacle():
    def __init__(self, origin, w):
        self.origin = origin
        self.w = w
    def render(self,screen):
        py_rect = pygame.Rect(self.origin[0], self.origin[1],self.w,self.w)
        pygame.draw.rect(screen, (50,0,0,1),py_rect,0)
    def in_collision(self,pt):
        if (pt > self.origin).all() and (pt < self.origin + self.w).all():
            return True
        return False

if __name__ == "__main__":
    ne = NavEnv() 
    import ipdb; ipdb.set_trace()
    ne.step(0,0)
        
