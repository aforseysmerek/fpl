"""TwoArmLift with a textured 8-quart stockpot mesh instead of the procedural
PotWithHandles. The object XML (objs/pot/pot_8qt.xml) keeps the stock pot's
conventions -- origin at the volumetric center, `center` site at the origin,
top_offset = half height, handle sites on the graspable crossbars -- so
TwoArmLift's reward/grasp/site references and the scripted policy in
scripts/pot_trace.py work unchanged."""
import os

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

POT_8QT_XML = os.path.join(os.path.dirname(__file__), "objs", "pot", "pot_8qt.xml")


class Pot8QtObject(MujocoXMLObject):
    """8-quart stockpot with two side handles.

    duplicate_collision_geoms=False: the XML already pairs an invisible
    group-0 collision set with a textured group-1 visual mesh; robosuite's
    duplication pass would recolor the collision geoms and strip materials.
    """

    def __init__(self, name):
        super().__init__(
            POT_8QT_XML, name=name, joints="default",
            obj_type="all", duplicate_collision_geoms=False,
        )

    @property
    def important_sites(self):
        dic = super().important_sites
        dic.update({
            "handle0": self.naming_prefix + "handle0",
            "handle1": self.naming_prefix + "handle1",
            "center": self.naming_prefix + "center",
        })
        return dic

    @property
    def handle0_geoms(self):
        return self.correct_naming(["handle0_bar"])

    @property
    def handle1_geoms(self):
        return self.correct_naming(["handle1_bar"])

    @property
    def handle_geoms(self):
        return self.handle0_geoms + self.handle1_geoms

    @property
    def handle_distance(self):
        # crossbar center to crossbar center (capsules at x = +-0.1455)
        return 0.291


class TwoArmLift8QtPot(TwoArmLift):
    """TwoArmLift with the mesh stockpot swapped in. Subclassing registers the
    env with robosuite.make via robosuite's EnvMeta metaclass. _load_model is
    the stock robosuite 1.2.0 body verbatim except for the object constructor.
    """

    def _load_model(self):
        super(TwoArmLift, self)._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "bimanual":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "single-arm-parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.pot = Pot8QtObject(name="pot")

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.pot)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.pot,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                rotation=(np.pi + -np.pi / 3, np.pi + np.pi / 3),
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.pot,
        )
