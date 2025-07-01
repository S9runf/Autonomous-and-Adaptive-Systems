from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
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
    def __init__(self, layouts, randomize=True):
        if layouts is None:
            raise ValueError("Layouts must be provided.")

        self.randomize = randomize

        self.n_layouts = len(layouts)
        self.envs = [make_env(layout) for layout in layouts]

        self.env_idx = 0
        self.curr_env = self.envs[self.env_idx]
        self.layout_name = self.curr_env.base_env.mdp.layout_name

        self.observation_space = self.curr_env.observation_space
        self.action_space = self.curr_env.action_space

    def reset(self):
        if len(self.envs) > 1 and self.randomize:
            self.curr_env = random.choice(self.envs)

        return self.curr_env.reset()

    def step(self, actions):
        return self.curr_env.step(actions)

    def next_layout(self):
        """ Switch to the next layout in the list """
        if self.env_idx < len(self.envs) - 1:
            self.env_idx += 1

            self.curr_env = self.envs[self.env_idx]
            self.layout_name = self.curr_env.base_env.mdp.layout_name
        else:
            raise IndexError("No more layouts available. Reset to start over.")

    def reset_layouts(self):
        """ Reset the environment to the first layout """
        self.env_idx = 0
        self.curr_env = self.envs[self.env_idx]
        self.layout_name = self.curr_env.base_env.mdp.layout_name
