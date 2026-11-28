import numpy as np

from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager
from scripts.g1d import build_shapes


class G1dBuildHouseCommon:
    """Unitree G1-D, one arm at a time: pick the seven house blocks from either side of the table and build the house
    of scripts/g1d/build_shapes.HOUSE (a cube, an arch and a cube in a row, two cubes on top, a long block across them,
    a triangular roof). Success: every block within build_shapes' tolerances of its target in the layout's BuildShape
    entry, then both arms back at their start pose.

    Eval layouts come from scripts/g1d/build_house_layouts.py.
    """

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 2700  # seven pick-and-places of ~300 actions each, and the way back

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)
        self.reward_manager.func_parser.is_shape_built = self._shape_built

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def build_shape(self, env_idx):
        layout_id = self.current_env_seed_map[env_idx]
        cache = self.__dict__.setdefault("_build_shapes", {})
        if layout_id not in cache:  # the reward reads it every action
            cache[layout_id] = build_shapes.shape_entry(self.seed_manager.get_seed_scene_info(layout_id))
        return cache[layout_id]

    def block_poses(self, env_idx, labels):
        lm = self.scene_manager.layout_manager
        out = {}
        for label in labels:
            pos, quat = lm.get_instance_pose(env_idx, label=label)
            out[label] = [np.asarray(v.cpu() if hasattr(v, "cpu") else v, dtype=float).reshape(-1) for v in (pos, quat)]
        return out

    def _shape_built(self, args):
        entry = self.build_shape(args["env_idx"])
        things = [*entry["parts"].values(), *entry.get("fixed", {}).values()]  # containers: the trays, box, base cup
        poses = self.block_poses(args["env_idx"], [p["label"] for p in things])
        return float(build_shapes.built(entry, poses))

    def run_reward(self):
        self.reward_manager.check([("is_shape_built", {}), self.reward_manager.all_robot_back_to_origin()])

    def gen_instruction(self, env_idx):
        return [build_shapes.instruction(self.build_shape(env_idx))]


class g1d_build_house(G1dBuildHouseCommon, TaskEnv):
    pass
