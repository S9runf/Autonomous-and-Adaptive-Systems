import torch
import os

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked

from agents.ppoAgent import PPOAgent


def main():
    base_mdp = OvercookedGridworld.from_layout_name("cramped_room", old_dynamics=True)
    base_env = OvercookedEnv.from_mdp(base_mdp, horizon=400, info_level=0)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
    agent = PPOAgent(env)

    agent.learn()

    # get the working directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # make sure the weights dierectory exists
    os.makedirs(f"{current_dir}/weights", exist_ok=True)

    # save the actor and critic models
    torch.save(agent.actor.state_dict(), f"{current_dir}/weights/overcooked-ppo.pth")
    torch.save(agent.critic.state_dict(), f"{current_dir}/weights/overcooked-critic-ppo.pth")

    print("Models saved.")


if __name__ == "__main__":
    main()
