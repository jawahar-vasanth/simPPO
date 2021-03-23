from gym.envs.registration import register  # type: ignore

# define the name to call the env
# gym.make("racer-v0")
register(id="Lidarcar-v0", entry_point="gym_racer.envs:RacerEnv")
