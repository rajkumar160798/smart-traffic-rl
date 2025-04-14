import streamlit as st
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.traffic_env import TrafficLightEnv
from agents.dqn_agent import DQN

# App Config
st.set_page_config(page_title="Smart Traffic Light RL", layout="centered")
st.title("🚦 Smart Traffic Light Controller (RL Demo)")
st.markdown("This demo uses a trained RL agent to optimize traffic signal switching at a 4-way intersection.")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TrafficLightEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
model = DQN(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("models/traffic_dqn.pt", map_location=device))
model.eval()

# Session state
if "state" not in st.session_state:
    st.session_state.state = env.reset()
    st.session_state.total_reward = 0
    st.session_state.step = 0
    st.session_state.done = False

# Visual Layout
st.markdown("### 🛣️ Intersection Queues")
cols = st.columns(4)
labels = ["North", "East", "South", "West"]
for i in range(4):
    cols[i].metric(label=labels[i], value=int(st.session_state.state[i]))

st.markdown("---")

# Predict & Step
if not st.session_state.done:
    state_tensor = torch.FloatTensor(st.session_state.state).unsqueeze(0).to(device)
    with torch.no_grad():
        action = torch.argmax(model(state_tensor)).item()
    st.markdown(f"🚦 Current Phase: {'North-South Green' if action == 0 else 'East-West Green'}")

    if st.button("🚗 Next Step"):
        next_state, reward, done, _ = env.step(action)
        st.session_state.state = next_state
        st.session_state.total_reward += reward
        st.session_state.step += 1
        st.session_state.done = done
else:
    st.success("✅ Simulation Complete!")
    st.write(f"**Total Reward:** {st.session_state.total_reward:.2f}")
    if st.button("🔄 Reset Simulation"):
        st.session_state.state = env.reset()
        st.session_state.total_reward = 0
        st.session_state.step = 0
        st.session_state.done = False

# Footer
st.markdown("---")
st.markdown("### 📊 Simulation Stats" )
st.write(f"**Total Steps:** {st.session_state.step}")
st.write(f"**Total Reward:** {st.session_state.total_reward:.2f}")
st.markdown("### 📈 Model Performance")
st.write("This demo uses a DQN model trained on simulated traffic data. The model learns to minimize total queue length at the intersection.")
st.markdown("### 📜 About")
st.write("This project is part of a research initiative to improve urban traffic management using reinforcement learning.")
st.markdown("### 📧 Contact")
st.write("For inquiries or collaboration, please reach out to [Raj Kumar Myakala](https://github.com/rajkumar160798).")