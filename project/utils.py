from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
import random

def make_env(layout_name, rew_shaping=None):
    """ Create an Overcooked environment from a layout name """

    base_mdp = OvercookedGridworld.from_layout_name(
        layout_name,
        old_dynamics=True,
        params_to_overwrite={
            "rew_shaping_params": rew_shaping
        }
    )
    base_env = OvercookedEnv.from_mdp(base_mdp, horizon=400, info_level=0)

    return Overcooked(
        base_env=base_env,
        featurize_fn=base_env.featurize_state_mdp
    )


class GeneralizedOvercooked:
    def __init__(self, layouts):
        self.envs = []

        assert isinstance(layouts, list), "Layouts should be a list of layout names."
        assert len(layouts) > 0, "At least one layout name must be provided."

        for layout in layouts:
            env = make_env(layout, rew_shaping=None)
            self.envs.append(env)
        self.cur_env = self.envs[0]
        self.observation_space = self.cur_env.observation_space
        self.action_space = self.cur_env.action_space

    def reset(self):
        idx = random.randint(0, len(self.envs)-1)
        self.cur_env = self.envs[idx]
        return self.cur_env.reset()

    def step(self, *args):
        return self.cur_env.step(*args)

    def render(self, *args):
        return self.cur_env.render(*args)