"""Franka cabinet scene used by VLA-TidyBench.

The class extends Isaac Lab's verified DLS IK-relative drawer task with two
deployable RGB observations and a red target object.  Simulator truth remains
available to scripted teachers and metrics, but is deliberately excluded from
the policy observation group.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.ik_rel_env_cfg import (
    FrankaCabinetEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


_FRANKA_TIDYBENCH_INIT_JOINT_POS = {
    "panda_joint1": 0.0444,
    "panda_joint2": -0.1894,
    "panda_joint3": -0.1107,
    "panda_joint4": -2.5148,
    "panda_joint5": 0.0044,
    "panda_joint6": 2.3775,
    "panda_joint7": 0.6952,
    "panda_finger_joint.*": 0.0400,
}


@configclass
class TidyBenchObservationsCfg:
    """Only observations that can be reproduced at deployment time."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=stack_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=stack_mdp.joint_vel_rel)
        table_cam = ObsTerm(
            func=stack_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=stack_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TidyBenchDrawerEnvCfg(FrankaCabinetEnvCfg):
    """One-env-ready drawer scene with a graspable red target object."""

    observations: TidyBenchObservationsCfg = TidyBenchObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # The VLA data/control contract is 20 Hz, matching the validated stack
        # baseline: 100 Hz physics with five simulator steps per policy step.
        self.sim.default.dt = 0.01
        self.sim.default.render_interval = 5
        self.sim.physx.dt = 0.01
        self.sim.physx.render_interval = 5
        self.decimation = 5
        self.episode_length_s = 30.0
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.cabinet.spawn.semantic_tags = [("class", "cabinet")]
        self.scene.cabinet.init_state.pos = (0.95, 0.0, 0.4)
        self.scene.cabinet.actuators["drawers"].stiffness = 0.0
        self.scene.cabinet.actuators["drawers"].damping = 20.0
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=_FRANKA_TIDYBENCH_INIT_JOINT_POS,
            ),
        )

        target_props = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=100.0,
            max_linear_velocity=10.0,
            max_depenetration_velocity=2.0,
            disable_gravity=False,
        )
        self.scene.target_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetObject",
            # Keep the pick zone in front of the open drawer front.  Placing
            # the object farther back causes a lifted grasp to collide with
            # the top-drawer fascia when the drawer starts open.
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, -0.20, 0.020), rot=(0.0, 0.0, 0.0, 1.0)),
            spawn=sim_utils.CuboidCfg(
                size=(0.045, 0.045, 0.040),
                rigid_props=target_props,
                collision_props=CollisionPropertiesCfg(),
                mass_props=MassPropertiesCfg(mass=0.04),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.82, 0.03, 0.03),
                    roughness=0.35,
                    metallic=0.0,
                ),
                semantic_tags=[("class", "red_target_object")],
            ),
        )

        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15),
                rot=(0.03701, 0.03701, -0.70614, -0.70614),
                convention="ros",
            ),
        )
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 3.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.55, -0.85, 1.15),
                rot=(-0.494, -0.760, 0.350, 0.230),
                convention="ros",
            ),
        )

        self.num_rerenders_on_reset = 3
        self.sim.default.render.antialiasing_mode = "DLAA"
        self.sim.physx.render.antialiasing_mode = "DLAA"
        self.image_obs_list = ["table_cam", "wrist_cam"]

        # Teacher-owned episode boundaries avoid time-out auto-resets while a
        # successful trajectory is being exported.
        self.terminations.time_out = None
        self.rewards = None
        for event_cfg in (self.events.default, self.events.physx):
            event_cfg.robot_physics_material = None
            event_cfg.cabinet_physics_material = None
            event_cfg.reset_robot_joints = None
            event_cfg.reset_all.params = {"reset_joint_targets": True}
