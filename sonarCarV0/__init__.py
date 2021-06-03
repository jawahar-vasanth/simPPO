from gym.envs.registration import register  # type: ignore

register(id="sonarCar-v0", entry_point="sonarCarV0.envs:RacerEnv")
