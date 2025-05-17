import torch

from models.actor_critic import ActorCritic

import gymnasium as gym

import os

env = gym.make("CartPole-v1", render_mode="human")

num_episodes = 100

current_dir = os.path.dirname(os.path.abspath(__file__))

policy = ActorCritic(env.observation_space.shape[0], env.action_space.n)
policy.load_state_dict(torch.load(f"{current_dir}/weights/cartpole.pth"))

total_rewards = []
soups = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    
    done = False
    episode_reward = 0
    soup_count = 0

    # the environment will end automatically when horizon is reached
    step = 0
    while not done:
        step += 1
        actions = []
        logits = policy(torch.FloatTensor(obs).to("cpu"))
        action = torch.argmax(logits).item()

        
        obs, reward, done, _, info = env.step(action)

        soup_count += reward // 20
        episode_reward += reward

    total_rewards.append(episode_reward)
    soups.append(soup_count)

print(f"Average Reward: {sum(total_rewards) / num_episodes}")
