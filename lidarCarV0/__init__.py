from gym.envs.registration import register  # type: ignore

# gym.make("racer-v0")
register(id="Lidarcar-v0", entry_point="lidarCarV0.envs:RacerEnv")
