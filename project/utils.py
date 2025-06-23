from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
import torch
import random

def make_env(layout_name):
    """ Create an Overcooked environment from a layout name """

    base_mdp = OvercookedGridworld.from_layout_name(
        layout_name,
        old_dynamics=True,
    )
    base_env = OvercookedEnv.from_mdp(base_mdp, horizon=400, info_level=0)

    featurize_fn = base_env.featurize_state_mdp
    return Overcooked(
        base_env=base_env,
        featurize_fn=featurize_fn
    )


class GeneralizedOvercooked:
    def __init__(self, layouts):
        if layouts is None:
            raise ValueError("Layouts must be provided.")

        self.layouts = layouts
        self.envs = [make_env(layout) for layout in layouts]

        self.stage_idx = 0
        self.curr_env = self.envs[0]
        self.observation_space = self.curr_env.observation_space
        self.action_space = self.curr_env.action_space

    def reset(self):
        if len(self.envs) > 1:
            self.curr_env = random.choice(self.envs)

        return self.curr_env.reset()

    def step(self, *args):
        return self.curr_env.step(*args)


