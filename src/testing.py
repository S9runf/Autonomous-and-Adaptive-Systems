import torch
import numpy as np
import os
import pygame
import argparse

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from agents.ppoAgent import PPOAgent
from utils import GeneralizedOvercooked


class Tester:
    def __init__(
        self,
        env,
        agent_paths=['random', 'random'],
        frame_rate=10,
        device="cpu"
    ):
        self.env = env
        self.frame_rate = frame_rate
        self.frame_duration = 1000 // frame_rate
        self.device = torch.device(device)

        self.agents = []
        input_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        # Ensure we have exactly 2 agents, duplicating the first if necessary
        if len(agent_paths) == 1:
            agent_paths.append(agent_paths[0])

        print([os.path.basename(str(path)) for path in agent_paths])

        # Load agents from the provided paths
        for path in agent_paths:
            if path not in ['random', 'idle']:
                agent = PPOAgent(
                        input_dim,
                        action_dim,
                        training=False,
                    )
                agent.load_actor(path)
                self.agents.append(agent)
            else:
                self.agents.append(path)
        self.visualizer = StateVisualizer()

    def test_layout(self, num_episodes=100, verbose=False):
        """
        Test the agent for multiple episodes and return average metrics.
        """

        total_rewards = []
        soups = []

        print(f"Testing layout: {self.env.layout_name}")

        for episode in range(num_episodes):
            _, _, episode_reward, soup_count = self.play_episode()

            total_rewards.append(episode_reward)
            soups.append(soup_count)
            if verbose:
                print(
                    f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward} Soup Count: {soup_count}\n"
                )
            else:
                print(f"episode {episode+1}/{num_episodes}", end="\r")
        avg_reward = np.mean(total_rewards)
        avg_soup = np.mean(soups)
        std_reward = np.std(total_rewards)

        return avg_reward, std_reward, avg_soup

    def test(self, num_episodes=100, verbose=False):
        results = {}

        last_layout = False
        while not last_layout:
            results[self.env.layout_name] = self.test_layout(num_episodes, verbose=verbose)
            try:
                self.env.next_layout()
            except IndexError:
                last_layout = True 

        print("Evaluation Results:")
        for layout, (avg_reward, std_reward, avg_soup) in results.items():
            print(
                f"Layout: {layout} - Average Reward: {avg_reward} , Standard Deviation: {std_reward:.2f}, Average Soup Count: {avg_soup}\n"
            )

    def play_episode(self):
        """
        Run a single episode in the current layout and collect the state trajectory.
        """
        # Ensure the order of the agents follows that of the paths
        other_idx = None
        while other_idx != 1:
            obs = self.env.reset()
            other_idx = obs["other_agent_env_idx"]
        done = False

        trajectory = []
        hud = []

        episode_reward = 0
        soup_count = 0
        steps = 0

        while not done:
            trajectory.append(obs["overcooked_state"])
            hud.append({"score": episode_reward, "soups": soup_count})

            states = obs["both_agent_obs"]
            states = torch.FloatTensor(np.array(states)).to(self.device)

            actions = []
            for i, agent in enumerate(self.agents):
                if isinstance(agent, PPOAgent):
                    action, _ = agent.get_actions(states[i])
                    actions.append(action.item())
                elif agent == 'random':
                    # If the agent is 'random', use a random action
                    actions.append(self.env.action_space.sample())
                elif agent == 'idle':
                    actions.append(4)  # Idle action

            obs, reward, done, info = self.env.step(actions)

            soup_count += reward // 20
            episode_reward += reward
            steps += 1

        return trajectory, hud, episode_reward, soup_count

    def render_trajectories(self):
        """
        Render a trajectory for each layout using pygame.
        """
        last_layout = False
        while not last_layout:
            trajectory, hud, reward, soups = self.play_episode()

            print(
                f"Rendering: Layout: {self.env.layout_name} - Final Reward: {reward} - Soup Count: {soups}"
            )

            base_mdp = self.env.curr_env.base_env.mdp

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
            pygame.display.set_caption(f"Overcooked AI - {self.env.layout_name}")

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
            try:
                self.env.next_layout()
            except IndexError:
                last_layout = True
        pygame.quit()


def get_paths(agents):  

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Convert agent names to full paths
    agent_paths = []
    weights_dir = os.path.join(current_dir, "weights")
    for agent_name in agents:
        if agent_name not in ['random', 'idle']:
            # Check if agent_name exists as a file in the weights directory
            # (with or without .pth)
            candidate_path = os.path.join(weights_dir, agent_name)
            if not candidate_path.endswith(".pth"):
                candidate_path = candidate_path + ".pth"

            if os.path.isfile(candidate_path):
                path = candidate_path
            else:
                raise FileNotFoundError(f"Agent weights file not found for '{agent_name}'")
        else:
            # 'random' agent does not have a path
            path = agent_name  

        agent_paths.append(path)

    return agent_paths


def test_agents(agents, layouts, num_episodes=100, render=False, verbose=False):
    # convert agents and layouts to lists if they are strings
    if type(agents) is str:
        agents = [agents]
    if type(layouts) is str:
        layouts = [layouts]

    if len(agents) == 0:
        raise ValueError("You must specify at least one agent.")
    if len(agents) > 2:
        raise ValueError("You can only specify a maximum of 2 agents.")

    env = GeneralizedOvercooked(layouts=layouts, randomize=False)

    agent_paths = get_paths(agents)
    tester = Tester(env=env, agent_paths=agent_paths, frame_rate=10)

    tester.test(num_episodes=num_episodes, verbose=verbose)
    env.reset_layouts()
    if render:
        tester.render_trajectories()


def render_episodes(agents, layouts):

    env = GeneralizedOvercooked(layouts=layouts, randomize=False)
    agent_paths = get_paths(agents)
    tester = Tester(env=env, agent_paths=agent_paths, frame_rate=10)

    tester.render_trajectories()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agents",
        nargs='+',
        type=str,
        help="List of agent weights files (max 2) to load from the weights directory",
        default=[]
    )
    parser.add_argument(
        "--layouts",
        nargs='+',
        required=True,
        help="List of layouts to test on"
    )
    parser.add_argument(
        "--no-test",
        action='store_true',
        help="If set, only render the trajectories without testing the agents"
    )
    parser.add_argument(
        "--render",
        action='store_true',
        help="Render the trajectories using pygame"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Print the results of each episode"
    )

    args = parser.parse_args()

    layouts = args.layouts

    if args.no_test:
        render_episodes(agents=args.agents, layouts=layouts)
    else:
        test_agents(
            agents=args.agents,
            layouts=layouts,
            num_episodes=100,
            render=args.render,
            verbose=args.verbose
        )
