# Tester file

import gym
import time
import numpy as np
import sonarCarV0

if __name__ == "__main__":
    racer_env = gym.make("sonarCar-v0",
                        render_mode = True,
                        sensor_display = True
                        )
    going = True
    i = 0
    num_frames = 4000
    while going:
        action = [[200,0.01]]
        obs, reward, done, info = racer_env.step(action)
        # time.sleep(0.05)
        racer_env.render()
        if done:
            racer_env.reset()
        # print(info)
        # print(i, reward)
        if num_frames > 0:
            i += 1
            if i == num_frames:
                going = False