import torch
import numpy as np
import os
import pygame
import gymnasium as gym

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from models.feed_forward import FeedForward
from utils import make_env


class Tester:
    def __init__(
        self,
        layouts: str | list[str],
        model_path,
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

        self.policy = FeedForward(
            dummy_env.observation_space.shape[0],
            dummy_env.action_space.n
        )
        self.policy.load_state_dict(torch.load(model_path))

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
            obs = env.reset()
            done = False
            episode_reward = 0
            soup_count = 0

            # The environment will end automatically when horizon is reached
            while not done:
                states = torch.FloatTensor(np.array(obs["both_agent_obs"]))\
                    .to(self.device)
                logits = self.policy(states)

                # Sample actions from the distribution
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample().tolist()

                obs, reward, done, _ = env.step(actions)

                soup_count += reward // 20
                episode_reward += reward

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

        obs = env.reset()
        done = False

        trajectory = []
        hud = []

        episode_reward = 0
        soup_count = 0

        while not done:
            trajectory.append(obs["overcooked_state"])
            hud.append({"score": episode_reward, "soups": soup_count})

            states = torch.FloatTensor(np.array(obs["both_agent_obs"])).to(self.device)
            logits = self.policy(states)

            # Sample actions from the distribution
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample().tolist()

            obs, reward, done, _ = env.step(actions)

            soup_count += reward // 20
            episode_reward += reward

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

    current_dir = os.path.dirname(os.path.abspath(__file__))

    path = f"{current_dir}/weights/asymmetric_advantages_ppo.pth"

    layouts = "asymmetric_advantages"

    # Example usage
    tester = Tester(layouts=layouts, model_path=path, frame_rate=10)

    tester.test(num_episodes=100)
    tester.render_trajectories()
