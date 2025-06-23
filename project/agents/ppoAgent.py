import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from models.feed_forward import FeedForward

class PPOAgent:
    def __init__(
        self,
        input_dim,
        action_dim,
        training=True,
        it_updates=12,
        eps=0.2,
        lr=8e-4,
        entropy_coef=0.05,
        device="cpu",
    ):
        self.device = torch.device(device)

        self.actor = FeedForward(input_dim, action_dim).to(self.device)

        # the critic is only used during training
        if training:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
             # critic takes the joint observations of both agents as input
            self.critic = FeedForward(2 * input_dim, 1).to(self.device)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

            self.mse = nn.MSELoss()

            # PPO hyperparameters
            self.it_updates = it_updates
            self.eps = eps
            self.entropy_coef = entropy_coef

    def learn(self, states, actions, log_probs_old, adv, expected_returns, random_agent_mask, joint_mask):

        for _ in range(self.it_updates):
            # get the new state values and log probabilities
            V, log_probs, entropy = self.evaluate(states, actions, random_agent_mask)

            # compute the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - log_probs_old)

            # compute the surrogate loss
            unclipped = ratios * adv
            clipped = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * adv

            entropy = entropy.mean()
            actor_loss = -torch.min(unclipped, clipped)
            actor_loss = torch.mean(actor_loss) - self.entropy_coef * entropy
            # Only consider the values of states without the random agent
            V = V[joint_mask]
            target_values = expected_returns[joint_mask]
            # repeat the values for both agents to match the expected returns
            V = V.unsqueeze(1).repeat(1, 2)

            critic_loss = self.mse(V, target_values)

            # update the actor and critic networks
            self.actor_optimizer.zero_grad()
            # keep the graph to allow backpropagation through the critic
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    # inference function, no gradients needed
    @torch.no_grad()
    def get_actions(self, obs):
        # convert the observations to a tensor
        states = torch.FloatTensor(np.array(obs)).to(self.device)

        # sample the action from the actor policy
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        actions = dist.sample()

        log_prob = dist.log_prob(actions)

        return actions, log_prob

    def evaluate(self, states, actions, random_agent_mask):
        # concatenate the states of both agents
        joint_states = states.view(states.shape[0], -1)
        # get the state values from the critic and remove the last dimension
        V = self.critic(joint_states).squeeze()

        # get the state values and log probabilities of the actions
        states_batch = states[random_agent_mask]
        states_batch = states_batch.view(-1, states_batch.shape[-1])
        logits = self.actor(states_batch)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)

        return V, log_probs, dist.entropy()
    
    def load_actor(self, path):
        """ Load the weights of the actor network """
        self.actor.load_state_dict(torch.load(path, map_location=self.device))

    def load_critic(self, path):
        """ Load the weights of the critic network """
        self.critic.load_state_dict(torch.load(path, map_location=self.device))
