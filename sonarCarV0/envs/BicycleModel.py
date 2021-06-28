import numpy as np
import math
import matplotlib.pyplot as plt
from math import atan, tan

max_steer = math.pi/15  # max steering angle
max_acc = 5
L = 2.9  # [m] Wheel base of vehicle
Lr = L / 2.0  # [m]
Lf = L - Lr


class LinearBicycleModel(object):
    """
    Class representing the state of a vehicle.
    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, throttle, delta, dt = 0.1):
        """
        Update the state of the vehicle.
        :param Throttle: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)
        throttle = np.clip(throttle, -max_acc, max_acc)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.v += throttle * dt

    def derivative(self, t, y, throttle, delta):
        # delta = np.clip(delta, -max_steer, max_steer)
        # delta = normalize_angle(delta)
        # throttle = np.clip(throttle, -max_acc, max_acc)

        beta = math.atan((Lr/L)*math.tan(delta))
        dx = y[3]*math.cos(y[2] + beta)
        dy = y[3]*math.sin(y[2] + beta)
        dyaw = (y[3]/Lr)*math.sin(beta)
        dv = throttle
        return [dx,dy,dyaw,dv]

    def get_pose(self):
        return [self.x, self.y, self.yaw, self.v]

    def set_pose(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
    


Cf = 1600.0 * 2.0  # N/rad
Cr = 1700.0 * 2.0  # N/rad
Iz = 2250.0  # kg/m2
m = 1500.0  # kg

# non-linear lateral bicycle model
class NonLinearBicycleModel():
    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0, omega=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.omega = omega
        # Aerodynamic and friction coefficients
        self.c_a = 1.36
        self.c_r1 = 0.01
    
    def derivative(self, t, y, a, delta):
        Ffy = -Cf*math.atan2(((y[4] + Lf*y[5])/y[3]-delta), 1.0)
        Fry = -Cr*math.atan2((y[4] - Lr*y[5])/y[3], 1.0)
        F_load = self.c_a*y[3]**2 + self.c_r1*y[3]
     
        dx = y[3]*math.cos(y[2]) - y[4]*math.sin(y[2])
        dy = y[4]*math.cos(y[2]) + y[3]*math.sin(y[2])
        dyaw = y[5]
        dvx = a + y[5]*y[4] - (1.0/m)*(Ffy*math.sin(delta) - F_load)
        dvy = -y[5]*y[3] + (1.0/m)*(Ffy*math.cos(delta)+Fry)
        domega = (1.0/Iz)*(Lf*Ffy*math.cos(delta) - Lr*Fry)
        return [dx, dy, dyaw, dvx, dvy, domega]

    def get_pose(self):
        return [self.x, self.y, self.yaw, self.vx, self.vy, self.omega]

    def set_pose(self, x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0, omega=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.omega = omega

    def update(self, throttle, delta, dt):
        delta = np.clip(delta, -max_steer, max_steer)
        self.x = self.x + self.vx * math.cos(self.yaw) * dt - self.vy * math.sin(self.yaw) * dt
        self.y = self.y + self.vx * math.sin(self.yaw) * dt + self.vy * math.cos(self.yaw) * dt
        self.yaw = self.yaw + self.omega * dt
        self.yaw = normalize_angle(self.yaw)
        Ffy = -Cf * math.atan2(((self.vy + Lf * self.omega) / self.vx - delta), 1.0)
        Fry = -Cr * math.atan2((self.vy - Lr * self.omega) / self.vx, 1.0)
        R_x = self.c_r1 * self.vx
        F_aero = self.c_a * self.vx ** 2
        F_load = F_aero + R_x
        self.vx = self.vx + (throttle - Ffy * math.sin(delta) / m - F_load/m + self.vy * self.omega) * dt
        self.vy = self.vy + (Fry / m + Ffy * math.cos(delta) / m - self.vx * self.omega) * dt
        self.omega = self.omega + (Ffy * Lf * math.cos(delta) - Fry * Lr) / Iz * dt


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    return angle