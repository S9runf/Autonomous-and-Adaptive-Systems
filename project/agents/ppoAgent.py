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

            # calculate the advantages
            adv, expected_returns = self.gae(rewards, V, dones)
            # normalize the advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)

            for _ in range(self.it_updates):
                # get the new state values and log probabilities
                V, log_probs = self.evaluate(states, actions)

                # calculate the ratio of the new and old probabilities
                ratios = torch.exp(log_probs - log_probs_old)

                # calculate the surrogate loss
                unclipped = ratios * adv
                clipped = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * adv

                # calculate the actor loss
                actor_loss = -torch.min(unclipped, clipped).mean()
                # calculate the critic loss
                V = V.unsqueeze(1)
                critic_loss = nn.MSELoss()(V, expected_returns)

                # update the actor and critic networks
                self.actor_optimizer.zero_grad()
                # keep the graph to allow backpropagation through the critic
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

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

            # run an episode until complete
            while not done:
                t += 1

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
                ep_reward += reward + shaped_rewards.sum()

                rewards.append(shaped_rewards)
                dones.append(done)
            
            ep_rewards.append(ep_reward)

        # update the progress bar
        self.pbar.update(t)

        # convert the lists to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.stack(actions).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        print(f"average episode reward: {np.mean(ep_rewards)}")

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
