from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
import random

def make_env(layout_name, linear_features=True):
    """ Create an Overcooked environment from a layout name """

    base_mdp = OvercookedGridworld.from_layout_name(
        layout_name,
        old_dynamics=True,
    )
    base_env = OvercookedEnv.from_mdp(base_mdp, horizon=400, info_level=0)

    if linear_features:
        featurize_fn = base_env.featurize_state_mdp
    else:
        featurize_fn = base_env.lossless_state_encoding_mdp
    return Overcooked(
        base_env=base_env,
        featurize_fn=featurize_fn
    )


class CurriculumStage:
    def __init__(self, layouts, linear_features=True):
        self.envs = []

        assert isinstance(layouts, list), "Layouts should be a list of layout names."
        assert len(layouts) > 0, "At least one layout name must be provided."

        for layout in layouts:
            env = make_env(layout, linear_features=linear_features)
            self.envs.append(env)

class GeneralizedOvercooked:
    def __init__(self, curriculum=None, layouts=None, linear_features=True):
        if curriculum is None and layouts is None:
            raise ValueError("Either curriculum or layouts must be provided.")
        
        if curriculum and layouts:
            raise ValueError("Only one of curriculum or layouts should be provided.")

        if curriculum:
            self.stages = [
                CurriculumStage(**stage, linear_features=linear_features)
                  for stage in curriculum
            ]
        elif layouts:
            self.layouts = layouts
            for layout in layouts:
                self.stages = [
                    CurriculumStage([layout],    linear_features=linear_features)
                ]

        self.stage_idx = 0
        self.curr_env = self.stages[self.stage_idx].envs[0]
        self.observation_space = self.curr_env.observation_space
        self.action_space = self.curr_env.action_space

    def reset(self):
        self.curr_env = random.choice(self.stages[self.stage_idx].envs)
        return self.curr_env.reset()

    def step(self, *args):
        return self.curr_env.step(*args)

    def next_stage(self):
        if self.stage_idx < len(self.stages) - 1:
            self.stage_idx += 1
            self.curr_env = self.stages[self.stage_idx].envs[0]
            print(f"Moved to stage {self.stage_idx + 1}.")
        else:
            raise IndexError("No more stages available in the curriculum.")
