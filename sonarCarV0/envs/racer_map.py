import numpy as np
import random
from pygame.color import THECOLORS
import pygame, pymunk
from pymunk.vec2d import Vec2d


class RacerMap():
    """
    Map for a racer
    """
    def __init__(self,
        space,
        field_wid = 1000,
        field_hei = 1000
        ):
        super().__init__()

        self.static = [
            pymunk.Segment(space.static_body,(0, 1), (0, field_hei), 1),
            pymunk.Segment(space.static_body,(1, field_hei), (field_wid,field_hei), 1),
            pymunk.Segment(space.static_body,(field_wid-1, field_hei),(field_wid-1, 1), 1),
            pymunk.Segment(space.static_body,(1, 1), (field_wid, 1), 1)]
        for s in self.static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        

class RacerCat():
    """
    Cat for a racer to avoid over fitting
    """
    def __init__(self,
        pos_x=100,
        pos_y=100,
        radius=30,
        ):
        super().__init__()

        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.c_body = pymunk.Body(1, inertia)
        self.c_body.position = pos_x, pos_y
        self.c_shape = pymunk.Circle(self.c_body, radius)
        self.c_shape.color = THECOLORS["orange"]
        self.c_shape.elasticity = 1.0
        self.c_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.c_body.angle)

        self.pos_x = pos_x
        self.pos_y = pos_y  
    
    def move(self):
        speed = random.randint(2, 20)
        self.c_body.angle -= 0.5*random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.c_body.angle)
        self.c_body.velocity = speed * direction
    

    def reset(self, pos_x=-1, pos_y=-1 ):
        if pos_x <0 or pos_y < 0 :
            self.c_body.position = self.pos_x, self.pos_y
        else:
            self.c_body.position = pos_x, pos_y



class RacerObs():
    """ 
    collection of obstacles for a racer
    """
    def __init__(self,
        pos_x=100,
        pos_y=100,
        radius=0.2,
        ):
        super().__init__()
        self.c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        self.c_shape = pymunk.Circle(self.c_body, radius)
        self.c_shape.elasticity = 1.0
        self.c_body.position = pos_x, pos_y
        self.c_shape.color = THECOLORS["pink"]

        self.pos_x = pos_x
        self.pos_y = pos_y  
    
    def move(self):
        speed = random.randint(0, 5)
        direction = Vec2d(1, 0).rotated(random.randint(-2, 2))
        self.c_body.velocity = speed * direction

    def reset(self):
        self.c_body.position = self.pos_x, self.pos_y
        
class RacerGoal():
    """ 
    collection of obstacles for a racer
    """
    def __init__(self,
        pos_x=100,
        pos_y=100,
        radius=0.2,
        ):
        super().__init__()
        self.c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        self.c_shape = pymunk.Circle(self.c_body, radius)
        self.c_shape.elasticity = 1.0
        self.c_body.position = pos_x, pos_y
        self.c_shape.color = THECOLORS["blue"]

        self.pos_x = pos_x
        self.pos_y = pos_y  
    
    def reset(self, x, y):
        self.c_body.position = x, y

class RacerObs2():
    """ 
    collection of obstacles for a racer
    """
    def __init__(self,
        pos_x=100,
        pos_y=100,
        radius=0.2,
        ):
        super().__init__()
        self.c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        vertices = [(0, 0), (60, 0), (30, 60)]
        self.c_shape = pymunk.Poly(self.c_body, vertices, radius = radius)
        self.c_shape.elasticity = 1.0
        self.c_body.position = pos_x, pos_y
        self.c_shape.color = THECOLORS["pink"]

        self.pos_x = pos_x
        self.pos_y = pos_y  
    
    def move(self):
        speed = random.randint(0, 5)
        direction = Vec2d(1, 0).rotated(random.randint(-1, 1))
        self.c_body.velocity = speed * direction

    def reset(self):
        self.c_body.position = self.pos_x, self.pos_y

