import numpy as np
import math
from scipy.integrate import solve_ivp
from sonarCarV0.envs.BicycleModel import LinearBicycleModel, NonLinearBicycleModel, normalize_angle
pi = math.pi

from pygame.color import THECOLORS
import pygame, pymunk
from pymunk.vec2d import Vec2d


class RacerCar():
    def __init__(
        self,
        pos_x=100,
        pos_y=100,
        direction=0.2,
        speed= 100,
        vehicle_model = LinearBicycleModel,
    ):

        super().__init__()

        self.carmodel = vehicle_model(pos_x, pos_y, direction, speed)

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.direction = direction  # in radians
        self.speed = speed

        self.tspan = [0, 0.05]
        self.drag_coeff = 0.5
        self.curr_state = [self.pos_x, self.pos_y, self.direction, self.speed]

        self.car_body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 14, (0, 0)))
        self.car_body.position = self.pos_x, self.pos_y
        self.car_shape = pymunk.Circle(self.car_body, 25)
        self.car_shape.color = THECOLORS["yellow"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = self.direction
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = self.speed*driving_direction


    def step(self, action):
        """Perform the action

        Two continous action spaces:
            1) Throttle 
            2) Steer
        """

        # Using direct Update eqn
        # self.carmodel.update(action[0][0], action[0][1], dt = self.tspan[-1])
        # dY = self.carmodel.get_pose()

        #Using ODE Solvers
        sol = solve_ivp(self.carmodel.derivative, self.tspan,
            self.curr_state, args = (action[0][0], action[0][1]))
        dY = sol.y.T
        dY = dY[-1]
        
        # move the car
        self.pos_x = dY[0]
        self.pos_y = dY[1]
        self.direction = dY[2]
        self.speed = dY[3]
        self.curr_state = [self.pos_x, self.pos_y, self.direction, self.speed]

        # update pymunk model
        self.car_body.position = self.pos_x, self.pos_y
        self.car_body.angle = self.direction
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = np.asscalar(self.speed)*driving_direction


    def reset(self, pos_x, pos_y, direction=0, speed = 10):
        """
        Reset the car state
        """
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.direction = direction  # in radians
        self.speed = speed
        self.curr_state = [self.pos_x, self.pos_y, self.direction, self.speed]

        self.car_body.position = self.pos_x, self.pos_y
        self.car_body.angle = self.direction
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = self.speed*driving_direction
        
