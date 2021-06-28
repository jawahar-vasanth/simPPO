import random, math
import numpy as np
import gym
from gym import spaces

import pygame
from pygame.color import THECOLORS
import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

from sonarCarV0.envs.racer_car import RacerCar
from sonarCarV0.envs.racer_map import RacerGoal, RacerMap, RacerObs, RacerCat, RacerGoal, RacerObs2
from sonarCarV0.envs.BicycleModel import normalize_angle



class RacerEnv(gym.Env):    
    metadata = {"render.modes": [True, False]}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, render_mode=True, sensor_display = False):
 
        # racing field dimensions
        self.field_wid = 1000
        self.field_hei = 700
        self.field_size = (self.field_wid, self.field_hei)
        # sidebar info dimensions
        self.sidebar_wid = 300
        self.sidebar_hei = self.field_hei
        self.sidebar_size = (self.sidebar_wid, self.sidebar_hei)

        self.total_wid = self.sidebar_wid + self.field_wid
        self.total_hei = self.field_hei
        self.total_size = (self.total_wid, self.total_hei)
        self.tot_ray_num = 3


        self.show_sensors = sensor_display
        self.render_mode = render_mode
        self.crashed = False
        self.draw_screen = True
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        self._setup_pygame()
        self.clock = pygame.time.Clock()

        # setup the car
        self.racer_car = RacerCar(pos_x=100, pos_y=100, direction=0.0, speed= 100)
        self.space.add(self.racer_car.car_body, self.racer_car.car_shape)

        self.goal = RacerGoal(500,500,50)
        self.space.add(self.goal.c_body, self.goal.c_shape)

        # setup the obstacle map
        self.racer_map = RacerMap(self.space, self.field_wid, self.field_hei)
        self.space.add(self.racer_map.static)
        
        self.obstacles = []
        self.obstacles.append(RacerObs(200, 150, 75))
        # self.obstacles.append(RacerObs(350, 300, 60))
        # self.obstacles.append(RacerObs2(200, 350, 75))
        self.obstacles.append(RacerObs(700, 200, 60))
        self.obstacles.append(RacerObs(600, 500, 45))
        self.obstacles.append(RacerObs(300, 600, 50))
        for obs in self.obstacles:
            self.space.add(obs.c_body, obs.c_shape)
        
        self.cat = []
        self.cat.append(RacerCat(50, 500, 30))
        for cat in self.cat:
            self.space.add(cat.c_body, cat.c_shape)

        # Define action and observation space
        self._setup_action_obs_space()
        self.reward = 0
        self.reset()


    def _setup_action_obs_space(self):
        self.action_space = spaces.Box(np.array([-1000, -0.2]), np.array([1000, 0.2]), dtype=np.float32)
        HEIGHT = self.tot_ray_num
        self.observation_space = spaces.Box(-1, 1, (HEIGHT,), dtype=np.float32)

    def render(self):
        """
        Render the environment to the screen
        """
        if self.render_mode == True:
            self.screen.fill(THECOLORS["black"])
            draw(self.screen, self.space)
            self.space.step(1./10)
            self._update_dynamic_sidebar(reward= self.reward)
            pygame.display.flip()
        self.clock.tick()


    def reset(self):
        """Reset the state of the environment to an initial state
        """
        self.crashed = False

        #  pick a random segment of the map and place the car there
        pos_x, pos_y = random.randint(50,self.field_wid-50), random.randint(50,self.field_hei-50)
        dir = random.uniform(-math.pi,math.pi)
        spd = random.uniform(10,100)
        self.racer_car.reset(pos_x, pos_y, direction= dir, speed= spd)

        for obst in self.obstacles:
            obst.reset()
        for cat in self.cat:
            pos_x, pos_y = random.randint(50,self.field_wid-50), random.randint(50,self.field_hei-50)
            cat.reset(pos_x, pos_y)

        jitt = random.randint(50,150)
        p_x, p_y = random.randint(0,1), random.randint(0,1)
        if p_x^p_y == 0:
            x = (self.field_wid - jitt) if self.field_wid/2 > pos_x else jitt
            y = (self.field_hei - jitt) if self.field_hei/2 > pos_y else jitt
        elif p_x == 0:
            x = pos_x
            y = (self.field_hei - jitt) if self.field_hei/2 > pos_y else jitt
        else:
            x = (self.field_wid - jitt) if self.field_wid/2 > pos_x else jitt
            y = pos_y

        self.goal.reset(x, y)
        # self.racer_car.reset(550, 350, direction= 1.5, speed= spd)
        obs, _ = self._collide_analyze()
        return obs

    def step(self, action):
        """Perform the action

        Steer: change steering
        Throttle: accelerate/brake
        combination of the above
        do nothing

        ----------
        This method steps the game forward one step
        Parameters
        ----------
        action : str
            MAYBE should be int anyway
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing the
                state of the environment.
            reward (float) :
                amount of reward achieved by the previous action.
            episode_over (bool) :
                whether it's time to reset the environment again.
            info (dict) :
                diagnostic information useful for debugging.
        """
        #set up obstacle movement
        if np.random.binomial(n = 1, p = 0.01):
            for obs in self.obstacles:
                obs.move()
        for cat in self.cat:
            cat.move()

        # SAR

        self.racer_car.step(action)
        obs, self.reward, done = self._compute_reward()

        # print(action, obs, self.reward, done)
        
        # create recap of env state
        info = {
            "car_pos_x": self.racer_car.pos_x,
            "car_pos_y": self.racer_car.pos_y,
            "car_dir": self.racer_car.direction,
            "car_speed": self.racer_car.speed,
            "crash_info": self.crashed,
        }

        return obs, self.reward, done, info
    
    def _compute_reward(self):
        """Compute reward function
        """
        done = False

        # Set the reward.
        obs, readings = self._collide_analyze()
        
        if self.goal_reached(readings):
            reward = 500
            done = True 
        elif self.car_is_crashed(readings):
            self.crashed = True
            reward = -200
            done = True
        else:
            reward = (-3 + int(sum(readings)/40))/10

        return obs, reward, done

    def _collide_analyze(self):
        '''
        '''
        x, y = self.racer_car.car_body.position
        readings = self.get_sonar_readings(x, y, self.racer_car.car_body.angle)
        normalized_readings = [(x-40.0)/40.0 for x in readings] 
        obs = np.array([normalized_readings])
        return obs, readings

    def get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm_left = self.make_sonar_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left

        # Rotate them and get readings.
        readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))
        # print(readings)
        if self.show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(x, y, point[0], point[1], angle + offset)

            # Check if we've hit something. Return the current i (distance)
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= self.field_wid or rotated_p[1] >= self.field_hei:
                return i  # Sensor is off the screen.
            else:
                obs = self.screen.get_at(rotated_p)
                read = self.get_track_or_not(obs) 
                if read == 0:
                    return i
                if read == 0.5:
                    return 2*len(arm)-i
            if self.show_sensors and self.render_mode:
                pygame.draw.circle(self.screen, (255, 255, 255), (rotated_p), 2)

        return i

    def make_sonar_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))
        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = self.field_hei - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading != THECOLORS['black'] and reading != THECOLORS['blue']:
            return 0
        elif reading == THECOLORS['blue']:
            return 0.5
        else:
            return 1

    def car_is_crashed(self, readings):
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
            return True
        else:
            return False

    def goal_reached(self, readings):
        if readings[0] > 75 or readings[1] > 75 or readings[2] >75:
            return True
        else:
            return False

    def _setup_pygame(self):
        """
        """
        if self.render_mode == True:
            # start pygame
            pygame.init()
            self.screen = pygame.display.set_mode(self.total_size)
            pygame.display.set_caption("Racer")
            self.screen.set_alpha(None)


            # create the background that will be redrawn each iteration
            self.background = pygame.Surface(self.total_size)
            self.background = self.background.convert()

            # Create the playing field
            self.field = pygame.Surface(self.field_size)
            self.field = self.field.convert()
            self.field.fill((0, 0, 0))

            # where the info will be
            self._setup_sidebar()   

        elif self.render_mode == False:
            pass

        else:
            raise ValueError(f"Unknown render mode {self.render_mode}")


    def _setup_sidebar(self):
        # setup fonts to display info
        
        self._setup_font()
        self.sidebar_back_color = (80, 80, 80)
        self.font_info_color = (255, 255, 255)
        self.side_space = 50

        # create the sidebar surface
        self.sidebar_surf = pygame.Surface(self.sidebar_size)
        self.sidebar_surf = self.sidebar_surf.convert()
        self.sidebar_surf.fill(self.sidebar_back_color)

        # add titles
        self.speed_text_hei = 200
        text_speed = self.main_font.render("Speed:", 1, self.font_info_color)
        textpos_speed = text_speed.get_rect(
            midleft=(self.side_space, self.speed_text_hei)
        )
        self.sidebar_surf.blit(text_speed, textpos_speed)

        self.direction_text_hei = 260
        text_direction = self.main_font.render("Direction:", 1, self.font_info_color)
        textpos_direction = text_direction.get_rect(
            midleft=(self.side_space, self.direction_text_hei)
        )
        self.sidebar_surf.blit(text_direction, textpos_direction)

        self.reward_text_hei = 320
        text_reward = self.main_font.render("Reward:", 1, self.font_info_color)
        textpos_reward = text_reward.get_rect(
            midleft=(self.side_space, self.reward_text_hei)
        )
        self.sidebar_surf.blit(text_reward, textpos_reward)

        # setup positions for dynamic info: blit the text on a secondary
        # surface, then blit that on the screen in the specified position
        self.speed_val_pos = self.sidebar_wid - self.side_space, self.speed_text_hei
        self.direction_val_pos = (
            self.sidebar_wid - self.side_space,
            self.direction_text_hei,
        )
        self.reward_val_pos = self.sidebar_wid - self.side_space, self.reward_text_hei

        # create the dynamic sidebar surface
        self.side_dyn_surf = pygame.Surface(self.sidebar_size)
        self.side_dyn_surf = self.side_dyn_surf.convert()
        black = (0, 0, 0)
        self.side_dyn_surf.fill(black)
        self.side_dyn_surf.set_colorkey(black)

        # draw the static sidebar on the background
        self.background.blit(self.sidebar_surf, (self.field_wid, 0))

    def _update_dynamic_sidebar(self, reward=None):
        """fill the info values in the sidebar
        """
        # reset the Surface
        black = (0, 0, 0)
        self.side_dyn_surf.fill(black)

        # speed text
        text_info_speed = self.main_font.render(
            f"{self.racer_car.speed:.2f}", 1, self.font_info_color, self.sidebar_back_color,
        )
        textpos_speed_info = text_info_speed.get_rect(midright=self.speed_val_pos)
        self.side_dyn_surf.blit(text_info_speed, textpos_speed_info)

        # direction text
        text_info_direction = self.main_font.render(
            f"{180*normalize_angle(self.racer_car.direction)/3.14:.2f}",
            1,
            self.font_info_color,
            self.sidebar_back_color,
        )
        textpos_direction_info = text_info_direction.get_rect(
            midright=self.direction_val_pos
        )
        self.side_dyn_surf.blit(text_info_direction, textpos_direction_info)

        # reward text
        if not reward is None:
            reward_val = f"{reward:.2f}"
        else:
            reward_val = f"-"
        text_info_reward = self.main_font.render(
            reward_val, 1, self.font_info_color, self.sidebar_back_color,
        )
        textpos_reward_info = text_info_reward.get_rect(midright=self.reward_val_pos)
        self.side_dyn_surf.blit(text_info_reward, textpos_reward_info)

        # draw the filled surface
        self.screen.blit(self.side_dyn_surf, (self.field_wid, 0))

    def _setup_font(self):
        """
        """
        if not pygame.font:
            raise RuntimeError("You need fonts to put text on the screen")
        # create a new Font object (from a file if you want)
        #  self.main_font = pygame.font.Font(None, 36)
        #  self.main_font = pygame.font.Font(pygame.font.match_font("hack"), 16)
        self.main_font = pygame.font.SysFont("arial", 26)


