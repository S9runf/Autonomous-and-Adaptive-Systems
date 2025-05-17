import torch
import os

from agents.ppoAgent import PPOAgent

import gymnasium as gym


def main():
    env = gym.make("CartPole-v1")
    agent = PPOAgent(env)

    agent.learn(total_timesteps=10000)

    # get the working directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # make sure the weights dierectory exists
    os.makedirs(f"{current_dir}/weights", exist_ok=True)

    # save the actor and critic models
    torch.save(agent.actor.state_dict(), f"{current_dir}/weights/cartpole.pth")
    torch.save(agent.critic.state_dict(), f"{current_dir}/weights/cartpole-critic.pth")
    
    print("Models saved.")


if __name__ == "__main__":
    main()
