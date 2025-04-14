import gym
from gym import spaces
import numpy as np

class TrafficLightEnv(gym.Env):
    """
    A custom Gym environment for simulating traffic light control at a single 4-way intersection.
    """
    def __init__(self):
        super(TrafficLightEnv, self).__init__()
        
        # Observation: queue length in 4 directions (N, E, S, W)
        self.observation_space = spaces.Box(low=0, high=20, shape=(4,), dtype=np.int32)
        
        # Action: switch light to one of two phases (0 = NS green, 1 = EW green)
        self.action_space = spaces.Discrete(2)
        
        self.max_queue = 20
        self.reset()

    def reset(self):
        self.queues = np.random.randint(0, 10, size=4)
        self.current_phase = 0
        self.time = 0
        return self.queues

    def step(self, action):
        self.time += 1
        self.current_phase = action
        
        # Simulate traffic dynamics
        if action == 0:  # NS green
            self.queues[0] = max(0, self.queues[0] - np.random.randint(1, 5))
            self.queues[2] = max(0, self.queues[2] - np.random.randint(1, 5))
        else:  # EW green
            self.queues[1] = max(0, self.queues[1] - np.random.randint(1, 5))
            self.queues[3] = max(0, self.queues[3] - np.random.randint(1, 5))
        
        # Add random new cars
        self.queues += np.random.randint(0, 3, size=4)
        self.queues = np.clip(self.queues, 0, self.max_queue)

        reward = -np.sum(self.queues)  # Goal: minimize total queue

        done = self.time >= 100  # End after 100 steps

        return self.queues, reward, done, {}

    def render(self, mode='human'):
        print(f"Time: {self.time} | Queues: {self.queues} | Phase: {'NS' if self.current_phase == 0 else 'EW'}")
