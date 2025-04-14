from gym.envs.registration import register

register(
    id='TrafficLight-v0',
    entry_point='envs.traffic_env:TrafficLightEnv',
)
