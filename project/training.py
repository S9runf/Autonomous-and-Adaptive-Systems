import torch
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gymnasium as gym

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked

from agents.ppoAgent import PPOAgent
from utils import make_env

class AgentTrainer: 

    def __init__(
        self,
        agent,
        layouts: str | list[str],
        batch_eps=10,
        gamma=0.95,
        lam=1
    ):
        self.device = torch.device("cpu")
        self.agent = agent
        self.layouts = layouts
        self.batch_eps = batch_eps
        self.gamma = gamma
        self.lam = lam
        self.env = None
        

        if isinstance(self.layouts, str):
            self.env = make_env(self.layouts)

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

        self.mean_reward = 0

    def store(self, state, action, log_prob, reward, done):
        """ Store a single transition in the trajectory """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """ Clear the stored trajectory """
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

    def random_layout(self):
        """ create an Overcooked environment with a random layout from the list of layouts """

        assert len(self.layouts) > 0, "No layouts provided for the Overcooked environment."

        layout_name = random.choice(self.layouts)

        return make_env(layout_name)

    def get_trajectories(self):
        """ Collect a batch of trajectories by running the agent in the environment """

        # ensure the memory is empty before starting a new batch
        self.clear_memory()

        total_rewards = []
        env = self.env
        for ep in range(self.batch_eps):
            if self.env is None:
                # get a random layout for each episode
                env = self.random_layout()

            # reset the environment
            obs = env.reset()
            done = False
            ep_reward = 0

            # run the episode until complete
            while not done:

                # get the observations of both agents
                state = obs["both_agent_obs"]

                actions, log_probs = self.agent.get_actions(state)

                next_obs, reward, done, info = env.step(actions.tolist())

                shaped_rewards = torch.FloatTensor(info["shaped_r_by_agent"])

                # keep track of the total episode reward for logging
                ep_reward += reward + shaped_rewards.sum().item()

                # keep track of agent rewards separately for training
                agent_rewards = reward + shaped_rewards

                # store the transition in the trajectory
                self.store(state, actions, log_probs, agent_rewards, done)
                obs = next_obs

            total_rewards.append(ep_reward)

        self.mean_reward = sum(total_rewards) / len(total_rewards)

    def gae(self, rewards, V, dones):
        """ Compute the Generalized Advantage Estimation (GAE) for the given rewards and value function estimates """
        advantages = []
        returns = []

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = V[t + 1].detach() * (1 - dones[t])
            expected_return = rewards[t] + self.gamma * next_value
            delta = expected_return - V[t].detach()
            gae = delta + self.gamma * self.lam * gae

            advantages.insert(0, gae)
            returns.insert(0, expected_return)


        advantages = torch.stack(advantages).to(self.device)
        returns = torch.stack(returns).to(self.device)

        return advantages, returns

    def plot_results(self, actor_losses, critic_losses, mean_rewards):
        """ Plot the training results """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.plot(actor_losses, label='Actor Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Actor Loss Over Time')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(critic_losses, label='Critic Loss', color='orange')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Critic Loss Over Time')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(mean_rewards, label='Mean Reward', color='green')
        plt.xlabel('Training Steps')
        plt.ylabel('Mean Reward')
        plt.title('Mean Reward Over Time')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_models(self, layout):
        """ Save the actor and critic models to disk """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(f"{current_dir}/weights", exist_ok=True)

        torch.save(self.agent.actor.state_dict(), f"{current_dir}/weights/{layout}_ppo.pth")
        torch.save(self.agent.critic.state_dict(), f"{current_dir}/weights/{layout}_critic_ppo.pth")

        print(f"Models saved in {current_dir}/weights/{layout}_ppo.pth")

    def train(self, total_episodes=1000):
        """ Train the agent for the specified number of episodes """
        current_episode = 0
        actor_losses = []
        critic_losses = []
        mean_rewards = []

        pbar = tqdm(total=total_episodes, desc="PPO Training", unit="step")
        while current_episode < total_episodes:
            # collect a batch of trajectories
            self.get_trajectories()
            current_episode += self.batch_eps

            # convert the stored trajectories to tensors
            states = torch.FloatTensor(np.array(self.states)).to(self.device)
            actions = torch.stack(self.actions).to(self.device)
            log_probs = torch.stack(self.log_probs).to(self.device)
            rewards = torch.stack(self.rewards).to(self.device)
            dones = torch.FloatTensor(self.dones).to(self.device)

            # compute the advantages and expected returns
            V, _ = self.agent.evaluate(states, actions)
            adv, expected_returns = self.gae(rewards, V, dones)

            # normalize the adv
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)

            # update the agent
            losses = self.agent.learn(
                states,
                actions,
                log_probs,
                adv,
                expected_returns
            )

            # log losses and mean rewards for the current batch
            actor_losses.append(losses[0])
            critic_losses.append(losses[1])
            mean_rewards.append(self.mean_reward)

            pbar.update(self.batch_eps)
            pbar.set_postfix({"mean_reward": self.mean_reward})

        self.plot_results(actor_losses, critic_losses, mean_rewards)
        if isinstance(self.layouts, str):
            self.save_models(self.layouts)
        else:
            self.save_models("generalized")


if __name__ == "__main__":

    dummy_env = make_env("cramped_room")
    input_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    layouts = ["cramped_room", "coordination_ring", "asymmetric_advantages"]

    agent = PPOAgent(input_dim=input_dim, action_dim=action_dim)

    trainer = AgentTrainer(agent, layouts=layouts)

    trainer.train(total_episodes=1000)
