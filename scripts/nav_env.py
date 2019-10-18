import pygame
import numpy as np

"""
2D gridworld. obstacles are represented as Obstacle
They need to be boxes. Origin is the top right
300x400 grid
"""
class NavEnv():
    def __init__(self):
        self.gridsize = [300,400]
        pygame.init()
        self.screen = pygame.display.set_mode(self.gridsize)
        self.screen.fill((255,255,255))
        pygame.display.flip()
        self.goal = np.array((200,300))
        self.start = np.array((0,5))
        self.pos = self.start[:]
        self.obstacles = [(Obstacle(np.array((20,20)),50))]

    def step(self,x,y):
        old_pos = self.pos[:]
        new_pos = np.array((x,y))+self.pos
        if not self.collision_fn(new_pos):
            self.pos = new_pos
        pygame.draw.line(self.screen,(1,0,0,1),old_pos.astype(np.int32),new_pos.astype(np.int32), 5)
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
        for obs in self.obstacles:
            obs.render(self.screen)
        pygame.display.flip()

class Obstacle():
    def __init__(self, origin, w):
        self.origin = origin
        self.w = w
    def render(self,screen):
        py_rect = pygame.Rect(self.origin[0], self.origin[1],self.w,self.w)
        pygame.draw.rect(screen, (0,1,0,1),py_rect,0)
    def in_collision(self,pt):
        if (pt > self.origin).all() and (pt < self.origin + self.w).all():
            return True
        return False
        
