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
    batch_steps=2048, 
    episode_steps=400, 
    it_updates=10, 
    gamma=0.95,
    eps=0.2,
    lr=1e-3
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
        self.batch_steps = batch_steps
        self.episode_steps = episode_steps
        self.it_updates = it_updates
        self.gamma = gamma
        self.eps = eps

    def learn(self, total_timesteps=400000):
        current_step = 0
        current_it = 0

        self.pbar = tqdm(total=total_timesteps, desc="PPO Training", unit="step")

        while current_step < total_timesteps:
            # run the environment for a number of steps
            states, actions, log_probs_old, rewards, dones = self.trajectories()

            current_step += states.shape[0]
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
                critic_loss = nn.MSELoss()(V, expected_returns)

                # optimize the actor
                self.actor_optimizer.zero_grad()
                # keep the graph to allow backpropagation through the critic
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                # optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

    def trajectories(self):
        # TODO: implement better memory
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        ep_rewards = []

        t = 0
        while t < self.batch_steps:
            # reset the environment
            obs = self.env.reset()
            done = False
            ep_reward = 0

            while not done:
                t += 1

                # obtain state representation from the observation
                obs = obs["both_agent_obs"]

                states.append(obs)
                # get the actions for both agents
                actions_t, log_probs_t = self.get_actions(obs)

                # take a step in the environment
                obs, reward, done, info = self.env.step(actions_t)

                # add the shaped rewards to the sparse rewards
                for shaped in info["shaped_r_by_agent"]:
                    shaped_reward = reward + shaped
                    rewards.append(shaped_reward)
                    ep_reward += shaped_reward

                ep_reward -= reward

                for action, log_prob in zip(actions_t, log_probs_t):
                    actions.append(action)
                    log_probs.append(log_prob)

                # TODO: find a better way to handle the done flag
                dones.append(done)
                dones.append(done)

            ep_rewards.append(ep_reward)

        self.pbar.update(t)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        log_probs = torch.FloatTensor(np.array(log_probs)).to(self.device)

        print(f"mean episode reward: {np.mean(ep_rewards)}")

        return states, actions, log_probs, rewards, dones

    def get_actions(self, states):
        actions = []
        log_probs = []

        for state in states:
            # convert the states to tensors
            state = torch.FloatTensor(state).to(self.device)

            # get the action logits from the actors
            logits = self.actor(state)

            # create a categorical distribution from the logits
            dist = Categorical(logits=logits)
            # sample an action from the distribution
            action = dist.sample()

            # get the log probability of the action
            log_probs.append(dist.log_prob(action).item())
            actions.append(action.item())

        return actions, log_probs

    def evaluate(self, states, actions):

        # concatenate the states of both agents
        joint_states = states.view(states.shape[0], -1)
        # get the values for the states from the critic
        V = self.critic(joint_states).squeeze()
        # repeat the values to match the shape of the returns
        V = V.repeat_interleave(2)

        # flatten the second dimension of the states
        states = states.view(-1, self.input_dim)

        # get the action logits from the actor
        logits = self.actor(states)
        # create a categorical distribution from the logits
        dist = Categorical(logits=logits)
        # get the log probabilities of the actions
        log_probs = dist.log_prob(actions)

        return V, log_probs

    def gae(self, rewards, V, dones):
        advantages = []
        returns = []
        # add a zero at the end for the the terminal state
        values = torch.cat([V, torch.zeros(1).to(self.device)], dim=0).detach()
        gae = 0
        for t in reversed(range(len(rewards))):
            # if the state is terminal next_value is 0
            next_value = values[t + 1] * (1 - dones[t])
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + V.detach()

        return advantages, returns
