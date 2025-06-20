import torch
import numpy as np
import os
import pygame
import random
import argparse

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from models.feed_forward import FeedForward
from utils import make_env


class Tester:
    def __init__(
        self,
        layouts: str | list[str],
        agent_paths,
        frame_rate=10,
    ):
        if isinstance(layouts, str):
            layouts = [layouts]

        self.layouts = layouts
        self.frame_rate = frame_rate
        self.frame_duration = 1000 // frame_rate
        self.device = torch.device("cpu")

        # Set up dummy environment to get observation and action spaces
        dummy_env = make_env("cramped_room")

        self.agents = []

        # Load agents from the provided paths
        for i, path in enumerate(agent_paths):
            if path != 'random':
                self.agents.append(FeedForward(
                    dummy_env.observation_space.shape[-1],
                    dummy_env.action_space.n
                ))
                self.agents[i].load_state_dict(torch.load(path))
            else:
                self.agents.append(None)

        self.visualizer = StateVisualizer()

    def test_layout(self, layout, num_episodes=100):
        """
        Test the agent for multiple episodes and return average metrics.

        Args:
            layout: Name of the layout to test
            num_episodes: Number of episodes to test for each layout

        Returns:
            dict with average reward and soup count for each layout
        """

        env = make_env(layout)

        total_rewards = []
        soups = []

        print(f"Testing layout: {layout}")

        for episode in range(num_episodes):
            other_idx = None
            # Ensure the order of the agents follows that of the paths
            while other_idx != 1:
                obs = env.reset()
                other_idx = obs["other_agent_env_idx"]

            done = False
            episode_reward = 0
            soup_count = 0
            steps = 0

            # The environment will end automatically when horizon is reached
            while not done:
                states = obs["both_agent_obs"]
                states = torch.FloatTensor(np.array(states)).to(self.device)

                actions = []
                for i, agent in enumerate(self.agents):
                    if agent is not None:
                        # Get the logits from the policy 
                        logits = agent(states[i])
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample().item() 
                        actions.append(action)
                    else:
                        # If the agent is None, use a random action
                        actions.append(random.randint(0, env.action_space.n - 1))
                    
                obs, reward, done, _ = env.step(actions)

                soup_count += reward // 20
                episode_reward += reward
                steps += 1

            total_rewards.append(episode_reward)
            soups.append(soup_count)
            print(
                f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward} - Soup Count: {soup_count}"
            )

        print()  # New line after the last episode
        avg_reward = sum(total_rewards) / num_episodes
        avg_soup = sum(soups) / num_episodes
        print(f"Average Reward for {layout}: {avg_reward}")
        print(f"Average Soup Count for {layout}: {avg_soup}")

        return avg_reward, avg_soup

    def test(self, num_episodes=100):
        results = {}

        for layout in self.layouts:
            results[layout] = self.test_layout(layout, num_episodes)

        print("Evaluation Results:")
        for layout, (avg_reward, avg_soup) in results.items():
            print(
                f"Layout: {layout} - Average Reward: {avg_reward}, Average Soup Count: {avg_soup}"
            )

    def collect_trajectory(self, layout):
        """
        Run a single episode in the given layout and collect the state trajectory.

        Returns:
            Tuple of (trajectory, HUD data, episode reward, soup count)
        """
        env = make_env(layout)

        # Ensure the order of the agents follows that of the paths
        other_idx = None
        while other_idx != 1:
            obs = env.reset()
            other_idx = obs["other_agent_env_idx"]
        done = False

        trajectory = []
        hud = []
        shaped_rewards = [0, 0]

        episode_reward = 0
        soup_count = 0
        steps = 0

        while not done:
            trajectory.append(obs["overcooked_state"])
            hud.append({"score": episode_reward, "soups": soup_count, "shaped_rewards": shaped_rewards})

            states = obs["both_agent_obs"]
            states = torch.FloatTensor(np.array(states)).to(self.device)

            actions = []
            for i, agent in enumerate(self.agents):
                if agent is not None:
                    # Get the logits from the policy 
                    logits = agent(states[i])
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
                    actions.append(action)
                else:
                    # If the agent is None, use a random action
                    actions.append(random.randint(0, env.action_space.n - 1))

            obs, reward, done, info = env.step(actions)
            shaped_rewards = info["shaped_r_by_agent"]

            soup_count += reward // 20
            episode_reward += reward
            steps += 1

        return trajectory, hud, episode_reward, soup_count

    def render_trajectories(self):
        """
        Render a trajectory for each layout using pygame.
        If no trajectory is provided, collect one first.
        """
        for layout in self.layouts:

            trajectory, hud, reward, soups = self.collect_trajectory(layout)
            print(
                f"Rendering: Layout: {layout} - Final Reward: {reward} - Soup Count: {soups}"
            )

            base_mdp = OvercookedGridworld.from_layout_name(
                layout,
                old_dynamics=True
            )
            # Generate frames
            frames = [
                self.visualizer.render_state(
                    state, grid=base_mdp.terrain_mtx, hud_data=hud_data
                )
                for state, hud_data in zip(trajectory, hud)
            ]

            # Initialize pygame to show the episode
            pygame.init()
            # Initialize the display to the first frame
            frame = frames[0]

            window = pygame.display.set_mode(
                frame.get_size(), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
            )
            pygame.display.set_caption(f"Overcooked AI - {layout}")

            frame_count = 0
            running = True
            last_update = pygame.time.get_ticks()

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                now = pygame.time.get_ticks()

                if now - last_update > self.frame_duration:
                    window.blit(frame, (0, 0))
                    pygame.display.flip()
                    last_update = pygame.time.get_ticks()

                    frame_count += 1
                    if frame_count < len(frames):
                        frame = frames[frame_count]
                    # wait for 2 seconds after the last frame before quitting
                    elif frame_count >= len(frames) + self.frame_rate * 2:
                        running = False

        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agents",
        nargs='+',
        type=str,
        help="List of agent weights files (max 2) to load from the weights directory",
        default=['random', 'random']
    )
    parser.add_argument(
        "--layouts",
        nargs='+',
        required=True,
        help="List of layouts to test on"
    )
    args = parser.parse_args()

    if len(args.agents) > 2:
        raise ValueError("You can only specify a maximum of 2 agents.")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Convert agent names to full paths
    agent_paths = []
    for agent_name in args.agents:
        if agent_name != 'random' and agent_name.endswith(".pth"):
            path = f"{current_dir}/weights/{agent_name}"
        elif agent_name != 'random':
            path = f"{current_dir}/weights/{agent_name}.pth"
        else:
            path = agent_name  # 'random' agent does not have a path

        agent_paths.append(path)

    # Ensure we have exactly 2 agents, filling with 'random' if necessary
    agent_names = args.agents
    if len(agent_paths) == 1:
        agent_paths.append('random')
        agent_names.append('random')

    layouts = args.layouts

    print(f"Agents: {args.agents}")
    print(f"Layouts: {layouts}")

    # Example usage
    tester = Tester(layouts=layouts, agent_paths=agent_paths, frame_rate=10)

    tester.test(num_episodes=100)
    tester.render_trajectories()
