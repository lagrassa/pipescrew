import pygame
import numpy as np
class Obstacle():
    def __init__(self, origin, w,y):
        self.origin = origin
        self.x = w
        self.y = y
    def render(self,screen, ppm = 1):
        py_rect = pygame.Rect(ppm*self.origin[0], ppm*self.origin[1],ppm*self.x,ppm*self.y) 
        pygame.draw.rect(screen, (50,0,0,1),py_rect,0)

    def in_collision(self,pt):
        if (pt > self.origin).all() and (pt < self.origin + np.array((self.x, self.y))).all():
            return True
        return False


class Quicksand():
    def __init__(self, origin, r):
        self.origin = origin
        self.r = r

    def render(self,screen, ppm = 1):
        pygame.draw.circle(screen, (242,229,194,1),(int(ppm*self.origin[0]), int(ppm*self.origin[1])), int(ppm*self.r),0)

    def in_collision(self,pt):
        if np.linalg.norm(np.subtract(pt,self.origin)) < self.r :
            print("In quicksand")
            return True
        return False
def two_openings_obstacles():
    ppm = 100.
    gap = 35/ppm
    length = 80/ppm
    dist_down=150/ppm
    print("dist_down", dist_down)
    thickness  = 20/ppm
    obstacles = [Obstacle(np.array((.001,dist_down)),length,thickness),
                       Obstacle(np.array((length+gap,dist_down)),length,thickness),
                       Obstacle(np.array((2*length+2*gap,dist_down)),length,thickness)]
    return obstacles
"""
line centered a few cm below the goal in the y direction, centered in the x direction
"""
def line_world_obstacles(goal):
    length = 0.15
    thickness = 0.03
    offset = 0.02
    x = goal[0]-length/2.
    y = goal[1] + offset
    return [Obstacle(np.array((x,y)), length, thickness)]


