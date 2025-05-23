import torch
import numpy as np
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from models.actor_critic import ActorCritic

import os

device = "cpu"

base_mdp = OvercookedGridworld.from_layout_name("cramped_room", old_dynamics=True)
base_env = OvercookedEnv.from_mdp(base_mdp, horizon=400, info_level=0)
env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

visualizer = StateVisualizer()

num_episodes = 100

current_dir = os.path.dirname(os.path.abspath(__file__))

policy = ActorCritic(env.observation_space.shape[0], env.action_space.n)
policy.load_state_dict(torch.load(f"{current_dir}/weights/overcooked-ppo.pth"))

total_rewards = []
soups = []

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    soup_count = 0

    os.makedirs(f"{current_dir}/visualizations", exist_ok=True)

    # the environment will end automatically when horizon is reached
    step = 0
    while not done:
        step += 1
        actions = []
        states = torch.FloatTensor(np.array(obs["both_agent_obs"])).to(device)
        logits = policy(states)
        actions = torch.argmax(logits, dim=1).tolist()
        #dist = torch.distributions.Categorical(logits=logits)
        #actions = dist.sample().tolist()
        
        obs, reward, done, info = env.step(actions)

        soup_count += reward // 20
        episode_reward += reward

    total_rewards.append(episode_reward)
    soups.append(soup_count)
    print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward} - Soup Count: {soup_count}")

print(f"Average Reward: {sum(total_rewards) / num_episodes}")
print(f"average soup count: {sum(soups) / num_episodes}")
