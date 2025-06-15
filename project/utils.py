from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked

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
