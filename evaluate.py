import torch
import numpy as np
from envs.traffic_env import TrafficLightEnv
from agents.dqn_agent import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TrafficLightEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = DQN(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("models/traffic_dqn.pt"))
model.eval()

episodes = 10
total_rewards = []

for ep in range(episodes):
    state = env.reset()
    done = False
    ep_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.argmax(model(state_tensor)).item()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        ep_reward += reward

    total_rewards.append(ep_reward)
    print(f"Episode {ep} — Total Reward: {ep_reward}")

avg = np.mean(total_rewards)
print(f"\n✅ Average Evaluation Reward over {episodes} episodes: {avg:.2f}")
print("Evaluation complete.")
# Plotting the rewards
import matplotlib.pyplot as plt
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Evaluation Rewards')
plt.show()
# Save the plot
plt.savefig("reports/evaluation_rewards.png")

# Save the model
torch.save(model.state_dict(), "models/traffic_dqn_evaluation.pt")
print("✅ Evaluation model saved!")


