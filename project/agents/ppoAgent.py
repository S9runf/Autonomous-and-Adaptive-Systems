import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from models.actor_critic import ActorCritic

from tqdm import tqdm

class PPOAgent:
    def __init__(
        self,
        env,
        batch_eps=6, 
        it_updates=10,
        gamma=0.95,
        eps=0.2,
        lr=1e-3,
    ):
        self.device = torch.device("cpu")

        # environment parameters
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # create the actor and critic networks
        self.actor = ActorCritic(self.input_dim, self.action_dim)\
            .to(self.device)
        # critic takes the observations of both agents as input
        self.critic = ActorCritic(2 * self.input_dim, 1)\
            .to(self.device)

        # define the optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # PPO hyperparameters
        self.batch_eps = batch_eps
        self.episode_steps = self.env.base_env.horizon
        self.it_updates = it_updates
        self.gamma = gamma
        self.eps = eps

    def learn(self, total_episodes=1000):
        current_ep = 0
        current_it = 0

        pbar = tqdm(total=total_episodes, desc="PPO Training", unit="step")

        while current_ep < total_episodes:
            # run the environment for a number of steps
            states, actions, log_probs_old, rewards, dones = self.trajectories()

            current_ep += len(states) // self.episode_steps
            current_it += 1

            V, _ = self.evaluate(states, actions)

            # compute the advantages
            adv, expected_returns = self.gae(rewards, V, dones)
            # normalize the advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)

            for _ in range(self.it_updates):
                # get the new state values and log probabilities
                V, log_probs = self.evaluate(states, actions)

                # compute the ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(log_probs - log_probs_old)

                # compute the surrogate loss
                unclipped = ratios * adv
                clipped = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * adv

                actor_loss = -torch.min(unclipped, clipped).mean()
                V = V.unsqueeze(1).repeat(1, 2)
                critic_loss = nn.MSELoss()(V, expected_returns)

                # update the actor and critic networks
                self.actor_optimizer.zero_grad()
                # keep the graph to allow backpropagation through the critic
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            
            # update the progress bar
            pbar.update(self.batch_eps)

    def trajectories(self):
        """build the batch for the current ppo iteration

        Returns:
            states (torch.Tensor): the observations of both agents
            actions (torch.Tensor): the actions taken by both agents
            log_probs (torch.Tensor): the log probabilities of the actions
            rewards (torch.Tensor): the rewards received by both agents
            dones (torch.Tensor): the done flags for both agents
        """
        
        # TODO: implement better memory
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        ep_rewards = []

        for _ in range(self.batch_eps):

            # reset the environment
            obs = self.env.reset()
            done = False
            ep_reward = 0

            # run the episode until complete
            while not done:

                # get the observations of both agents
                obs = obs["both_agent_obs"]

                states.append(obs)

                with torch.no_grad():
                    actions_t, log_probs_t = self.get_actions(obs)

                actions.append(actions_t)
                log_probs.append(log_probs_t)

                obs, reward, done, info = self.env.step(actions_t.tolist())

                # add shaped rewards to the episode reward
                shaped_rewards = torch.FloatTensor(info["shaped_r_by_agent"])\
                    .to(self.device)
                shaped_rewards += reward
                ep_reward += shaped_rewards.sum().item()

                rewards.append(shaped_rewards)
                dones.append(done)

            ep_rewards.append(ep_reward)

        # convert the lists to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.stack(actions).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        print(f"average episode reward: {np.mean(ep_rewards):.2f}")

        return states, actions, log_probs, rewards, dones

    def get_actions(self, obs):
        # convert the observations to a tensor
        states = torch.FloatTensor(np.array(obs)).to(self.device)

        # sample the action from the actor policy
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        actions = dist.sample()

        log_prob = dist.log_prob(actions)

        return actions, log_prob

    def evaluate(self, states, actions):

        # concatenate the states of both agents
        joint_states = states.view(states.shape[0], -1)
        # get the state values from the critic and remove the last dimension
        V = self.critic(joint_states).squeeze()

        # get the state values and log probabilities of the actions
        logits = self.actor(states)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)

        return V, log_probs

    def gae(self, rewards, V, dones):
        advantages = []
        expected_returns = []

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = V[t + 1].detach() * (1 - dones[t])

            expected_return = rewards[t] + self.gamma * next_value
            delta = expected_return - V[t].detach()
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)
            expected_returns.append(expected_return)

        advantages = torch.stack(advantages).to(self.device)
        expected_returns = torch.stack(expected_returns).to(self.device)

        return advantages, expected_returns
