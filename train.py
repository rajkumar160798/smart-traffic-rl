import gym
import torch
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from envs.traffic_env import TrafficLightEnv
from agents.dqn_agent import DQN

env = TrafficLightEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
lr = 1e-3
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 500
batch_size = 64
memory = deque(maxlen=10000)

model = DQN(state_dim, action_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

def act(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    q_values = model(state)
    return torch.argmax(q_values).item()

def replay():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.BoolTensor(dones).to(device)

    q_values = model(states).gather(1, actions).squeeze()
    next_q_values = model(next_states).max(1)[0]
    expected = rewards + gamma * next_q_values * (~dones)

    loss = loss_fn(q_values, expected.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

rewards_history = []

for ep in range(episodes):# Declare epsilon as global before modifying it
    state = env.reset()
    total_reward = 0

    done = False
    while not done:
        action = act(state)
        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        replay()

    rewards_history.append(total_reward)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if ep % 10 == 0:
        print(f"Episode {ep} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")

# Save model
torch.save(model.state_dict(), "models/traffic_dqn.pt")
print("✅ Training complete and model saved!")

# Plot
plt.plot(rewards_history)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.savefig("reports/rewards.png")
plt.show()

