import torch
import numpy as np

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from models.actor_critic import ActorCritic

import pygame
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

trajectory = []
hud = []

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    soup_count = 0

    os.makedirs(f"{current_dir}/visualizations", exist_ok=True)

    # the environment will end automatically when horizon is reached
    step = 0
    while not done:
        if episode == num_episodes - 1:
            trajectory.append(obs["overcooked_state"])
            hud.append({"score": episode_reward, "soups": soup_count})

        step += 1
        actions = []
        states = torch.FloatTensor(np.array(obs["both_agent_obs"])).to(device)
        logits = policy(states)
        actions = torch.argmax(logits, dim=1).tolist()
        # dist = torch.distributions.Categorical(logits=logits)
        # actions = dist.sample().tolist()

        obs, reward, done, info = env.step(actions)

        soup_count += reward // 20
        episode_reward += reward

    total_rewards.append(episode_reward)
    soups.append(soup_count)
    print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward} - Soup Count: {soup_count}")

print(f"Average Reward: {sum(total_rewards) / num_episodes}")
print(f"average soup count: {sum(soups) / num_episodes}")


frames = [
    visualizer.render_state(
        state, grid=base_mdp.terrain_mtx,
        hud_data=hud
    ) for state, hud in zip(trajectory, hud)
]
    
frame_rate = 10
# milliseconds per frame
frame_duration = 1000 // frame_rate

# initialize pygame to show the last episode
pygame.init()
# initialize the display to the first frame
frame = frames[0]

window = pygame.display.set_mode(frame.get_size(), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)

frame_count = 0
running = True
last_update = pygame.time.get_ticks()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            print("episode ended")

    now = pygame.time.get_ticks()

    if now - last_update > frame_duration:
        window.blit(frame, (0, 0))
        pygame.display.flip()
        last_update = pygame.time.get_ticks()

        frame_count += 1
        if frame_count < len(frames):
            frame = frames[frame_count]

pygame.quit()