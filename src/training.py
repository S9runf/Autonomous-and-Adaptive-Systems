import torch
import os
import numpy as np
from tqdm import tqdm
import random
import argparse

from agents.ppoAgent import PPOAgent
from utils import GeneralizedOvercooked


class AgentTrainer:

    def __init__(
        self,
        agent,
        env,
        layouts,
        batch_eps=10,
        gamma=0.95,
        lam=0.99,
        random_agent_prob=0.0,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.agent = agent
        self.batch_eps = batch_eps
        self.gamma = gamma
        self.lam = lam
        self.env = env
        self.layouts = layouts
        self.random_agent_prob = random_agent_prob

        self.random_idx = None
        self.random_agent_mask = []

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

        self.last_stage = False

        self.total_rewards = []
        self.mean_reward = 0


        print(f"Training on layouts: {layouts}")
        print(f"Random agent probability: {self.random_agent_prob}")

    def store(self, state, action, log_prob, reward, done):
        """Store a single transition in the trajectory"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """Clear the stored trajectory"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.random_agent_mask.clear()

    def get_trajectories(self):
        """Collect a batch of trajectories by running the agent in the environment"""

        # ensure the memory is empty before starting a new batch
        self.clear_memory()

        total_rewards = []
        for _ in range(self.batch_eps):

            # reset the environment
            obs = self.env.reset()
            done = False
            ep_reward = 0
            self.current_episode += 1

            self.random_idx = None
            if random.random() < self.random_agent_prob:
                self.random_idx = random.randint(0, 1)

            # run the episode until complete
            while not done:
                # build a step-wise mask for the random agent
                if self.random_idx is not None:
                    self.random_agent_mask.append(
                        [i != self.random_idx for i in range(2)]
                    )
                else:
                    self.random_agent_mask.append([True, True])

                # get the observations of both agents
                state = obs["both_agent_obs"]

                actions, log_probs = self.agent.get_actions(state)

                # if the agent is random,  choose a random action for the random agent
                if self.random_idx is not None and random.random() < 0.4:
                    # choose a random action for the random agent
                    actions[self.random_idx] = self.env.curr_env.action_space.sample()
                # force the random agent to stay idle 60% of the time
                elif self.random_idx is not None:
                    actions[self.random_idx] = 4

                next_obs, reward, done, info = self.env.step(actions.tolist())

                shaped_rewards = info["shaped_r_by_agent"]

                # convert the shaped rewards to a tensor
                shaped_rewards = torch.FloatTensor(shaped_rewards).to(self.device)

                # keep track of the total episode reward for logging
                ep_reward += reward

                # keep track of agent rewards separately for training
                agent_rewards = reward + shaped_rewards

                # store the transition in the trajectory
                self.store(state, actions, log_probs, agent_rewards, done)
                obs = next_obs

            total_rewards.append(ep_reward)

        self.total_rewards.extend(total_rewards)
        self.mean_reward = sum(total_rewards) / len(total_rewards)

    def gae(self, rewards, V, dones):
        """Compute the Generalized Advantage Estimation (GAE) for the given rewards and value function estimates"""
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

    def save_models(self, layout):
        """Save the actor and critic models to disk"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(f"{current_dir}/weights", exist_ok=True)

        torch.save(self.agent.actor.state_dict(), f"{current_dir}/weights/{layout}.pth")
        torch.save(
            self.agent.critic.state_dict(), f"{current_dir}/weights/{layout}_critic.pth"
        )

        print(f"Models saved in {current_dir}/weights/{layout}.pth")

    def train(self, total_episodes=1000, model_name=None):
        """Train the agent for the specified number of episodes"""
        self.current_episode = 0
        actor_losses = []
        critic_losses = []
        mean_rewards = []

        pbar = tqdm(total=total_episodes, desc="PPO Training", unit="step")
        while self.current_episode < total_episodes:
            self.get_trajectories()

            # convert the stored trajectories to tensors
            states = torch.FloatTensor(np.array(self.states)).to(self.device)
            actions = torch.stack(self.actions).to(self.device)
            log_probs = torch.stack(self.log_probs).to(self.device)
            rewards = torch.stack(self.rewards).to(self.device)
            dones = torch.FloatTensor(self.dones).to(self.device)
            random_agent_mask = torch.tensor(self.random_agent_mask).to(self.device)

            # filter the actions for the random agent
            actions = actions[random_agent_mask]
            log_probs = log_probs[random_agent_mask]

            # compute the advantages and expected returns
            V, _, _ = self.agent.evaluate(
                states, actions, random_agent_mask=random_agent_mask
            )

            adv, expected_returns = self.gae(rewards, V, dones)
            adv = adv[random_agent_mask]

            # normalize the advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)

            joint_mask = random_agent_mask.all(dim=1)

            # update the agent
            losses = self.agent.learn(
                states,
                actions,
                log_probs,
                adv,
                expected_returns,
                random_agent_mask,
                joint_mask,
            )

            # log losses and mean rewards for the current batch
            actor_losses.append(losses[0])
            critic_losses.append(losses[1])
            mean_rewards.append(self.mean_reward)
            pbar.update(self.batch_eps)
            pbar.set_postfix({"mean_reward": self.mean_reward})

        if model_name is not None:
            self.save_models(model_name)
        elif len(self.layouts) > 1:
            self.save_models("generalized")
        else:
            self.save_models(self.env.layout_name)

        return {
            "episode_rewards": self.total_rewards,
            "mean_rewards": mean_rewards
        }


def train_agent(
        layouts,
        episodes,
        random_prob=0,
        entropy_coef=0.05,
        model_name=None):
    # Make sure layouts is a list
    if type(layouts) is str:
        layouts = [layouts]

    env = GeneralizedOvercooked(layouts=layouts)

    # get the last dimension of the observation space
    # this allows to handle different featurization methods if needed
    input_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.n

    agent = PPOAgent(input_dim=input_dim, action_dim=action_dim, entropy_coef=entropy_coef)

    trainer = AgentTrainer(agent, env, layouts=layouts, random_agent_prob=random_prob)

    return trainer.train(total_episodes=episodes, model_name=model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layouts",
        nargs="+",
        help="List of layouts to train on",
        default=["cramped_room"],
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Total number of episodes to train for",
    )
    parser.add_argument(
        "--random_prob",
        type=float,
        default=0.0,
        help="Probability of using a random agent for each episode",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model to save",
    )


    args = parser.parse_args()

    layouts = args.layouts
    random_prob = args.random_prob

    train_agent(
        layouts=layouts,
        episodes=args.episodes,
        random_prob=random_prob,
        model_name=args.model_name,
    )