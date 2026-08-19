"""
VerticalWipe — robosuite Wipe with the wiping surface rotated into a vertical
wall (whiteboard-style), reusing the stock Wipe env end to end.

Why this is a small change: Wipe's marker-wiping check is frame-invariant (it
builds the contact plane from the wiper tool's corner geoms in WORLD frame),
and WipeArena samples/resets markers in the table body's LOCAL frame. So the
whole task transfers by rotating the one table body 90 deg about y; reward,
proportion_wiped, force thresholds, observables are all inherited untouched.

What actually changes:
  - VerticalWipeArena: rotates the table slab vertical, reinterprets
    table_offset as the center of the wall SURFACE, hides the (now sideways)
    legs, and moves the agentview camera to the robot's side of the wall
    (the stock agentview at x=+0.5 would film the BACK of the wall).
  - VerticalWipe: identical to Wipe except _load_model builds the vertical
    arena and defaults to a wall-tuned task_config (wall in front of the
    robot at reachable height).

Importing this module registers the env, so:
    import envs.vertical_wipe
    robosuite.make("VerticalWipe", robots="Panda", ...)
"""
import copy
import xml.etree.ElementTree as ET

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.environments.manipulation.wipe import Wipe, DEFAULT_WIPE_CONFIG
from robosuite.models.arenas.wipe_arena import WipeArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, find_elements


# Same task as Wipe; only the surface pose differs. table_offset here means
# the center of the wall SURFACE (x = the plane the markers live on, faces the
# robot at -x). Wall extent: table_full_size[0] (0.5 m) is now the VERTICAL
# span, table_full_size[1] (0.8 m) stays the horizontal (y) span.
# Placement + init pose were solved jointly (IK with a joint-limit barrier,
# validated over the marker-coverage corners and all 4 demo-knob corners):
# this is the pair that wipes the whole board with every joint >0.1 rad from
# its limits (robosuite prints "Joint limit reached" inside that tolerance).
DEFAULT_VERTICAL_WIPE_CONFIG = copy.deepcopy(DEFAULT_WIPE_CONFIG)
DEFAULT_VERTICAL_WIPE_CONFIG["table_offset"] = [0.15, 0.0, 0.95]
# Spill kept inside the arm's flush-reach envelope (0.35 vs the tabletop's
# 0.6): a wall-facing Panda can hold the pad flush only over the central
# board region — the stock 0.6 coverage puts spill in areas that demand a
# limit-riding shoulder, which stalls high-press wipes at 40-70%. At 0.35,
# validated over 12 random spills: high-press episodes wipe 97-100/100 and
# realized force tracks the press knob 1 N -> 24 N.
DEFAULT_VERTICAL_WIPE_CONFIG["coverage_factor"] = 0.35

# Wall-facing Panda start pose (IK-solved for the wall above, pad on the wall
# center). Doing this in the ENV and not just the scripted policy matters:
# robosuite's OSC null-space pulls toward the reset qpos, so starting from the
# stock downward-facing home pose drags every wall-facing motion into the
# forearm-roll limit.
WALL_INIT_QPOS = np.array([2.011, -1.337, -2.251, -2.145, 2.541, 2.003, -0.001])


def _lookat_quat_wxyz(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera quat (w,x,y,z) for a camera at cam_pos looking at target.
    MuJoCo cameras look along their local -z with +y as image-up."""
    f = np.asarray(target, dtype=float) - np.asarray(cam_pos, dtype=float)
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, dtype=float))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    q_xyzw = T.mat2quat(np.column_stack([r, u, -f]))
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


class VerticalWipeArena(WipeArena):
    """WipeArena with the table slab rotated into a vertical wall.

    All marker sampling (sample_start_pos / sample_path_pos) and reset_arena
    run in the table body's LOCAL frame, so they are inherited unchanged: the
    local (x, y) marker plane simply maps to world (z, y) on the wall.
    """

    # -90 deg about y: local +z (surface normal) -> world -x (faces the robot),
    # local +x (marker-plane long axis) -> world +z (up the wall).
    WALL_QUAT = (0.7071068, 0.0, -0.7071068, 0.0)

    # agentview replacement: robot side of the wall, offset in +y — the
    # WALL_INIT_QPOS arm branch reaches in from the right (-y side), so this
    # side sees the board face past the arm; aimed at the wall-surface center.
    AGENTVIEW_POS = (-0.45, 0.75, 1.35)

    # Wood brown shared by the frame border and the stand posts.
    BOARD_FRAME_RGBA = (0.38, 0.24, 0.12, 1.0)

    def configure_location(self):
        # Stock setup first: sizes, friction, and the marker line appended to
        # the table body (marker poses are table-local, so they follow the
        # rotation below for free).
        super().configure_location()

        # Whiteboard look: the stock tabletop is painted by the ceramic
        # texture material and the markers by a brown "Dirt" texture, both of
        # which override plain rgba. Strip the materials so rgba shows:
        # dark chalkboard green for the board, white for the markings.
        # (reset_arena and the wipe logic only ever touch rgba's alpha, so
        # these colors survive resets.)
        self.table_visual.attrib.pop("material", None)
        self.table_visual.set("rgba", array_to_string([0.05, 0.12, 0.09, 1.0]))
        for marker in self.markers:
            marker_geom = find_elements(
                root=self.table_body, tags="geom",
                attribs={"name": marker.visual_geoms[0]}, return_first=True,
            )
            marker_geom.attrib.pop("material", None)
            marker_geom.set("rgba", array_to_string([1.0, 1.0, 1.0, 1.0]))

        # Brown frame: a visual-only box in the table body's LOCAL frame
        # (so it rotates with the wall for free), 3 cm wider than the board
        # on each side and recessed 2 mm behind the wiping face (just enough
        # to avoid z-fighting between the coplanar faces), so only a border
        # shows around the edge. contype/conaffinity 0 = no collision.
        frame = ET.Element(
            "geom",
            name="board_frame",
            type="box",
            pos=array_to_string([0.0, 0.0, -0.002]),
            size=array_to_string([
                self.table_half_size[0] + 0.03,
                self.table_half_size[1] + 0.03,
                self.table_half_size[2],
            ]),
            rgba=array_to_string(self.BOARD_FRAME_RGBA),
            contype="0", conaffinity="0", group="1",
        )
        self.table_body.append(frame)

        # Reinterpret table_offset as the center of the wall SURFACE; the slab
        # body center sits half a thickness behind it (surface faces -x).
        wall_center = np.asarray(self.table_offset, dtype=float) + np.array(
            [self.table_half_size[2], 0.0, 0.0]
        )
        self.center_pos = wall_center
        self.table_body.set("pos", array_to_string(wall_center))
        self.table_body.set("quat", array_to_string(np.asarray(self.WALL_QUAT)))

        # Legs would stick out horizontally from the wall — hide them (same
        # trick TableArena uses for has_legs=False).
        for leg in self.table_legs_visual:
            leg.set("rgba", array_to_string([1, 0, 0, 0]))
            leg.set("size", array_to_string([0.0001, 0.0001]))

        # The stock agentview (x=+0.5 looking back) films the BACK of the wall;
        # move it to the robot's side, aimed at the wall-surface center.
        self.set_camera(
            "agentview",
            pos=np.asarray(self.AGENTVIEW_POS),
            quat=_lookat_quat_wxyz(self.AGENTVIEW_POS, self.table_offset),
        )

        # Stand: two visual-only posts from the floor to the board's underside
        # so the board reads as mounted, not floating. contype/conaffinity 0 =
        # no collision, so the physics (and the arm) are untouched.
        board_bottom = float(wall_center[2]) - float(self.table_half_size[0])
        post_x = float(wall_center[0])
        for i, post_y in enumerate((-0.3, 0.3)):
            post = ET.Element(
                "geom",
                name=f"board_stand_post{i}",
                type="cylinder",
                pos=array_to_string([post_x, post_y, board_bottom / 2.0]),
                size=array_to_string([0.02, board_bottom / 2.0]),
                rgba=array_to_string(self.BOARD_FRAME_RGBA),
                contype="0", conaffinity="0", group="1",
            )
            self.worldbody.append(post)


class VerticalWipe(Wipe):
    """Wipe on a vertical wall. Same reward, observables, and gripper — only
    the arena (and the default task_config placing it) differ."""

    def __init__(self, robots, task_config=None, **kwargs):
        super().__init__(
            robots=robots,
            task_config=task_config if task_config is not None else DEFAULT_VERTICAL_WIPE_CONFIG,
            **kwargs,
        )

    def _load_model(self):
        # Mirrors Wipe._load_model with VerticalWipeArena swapped in. Robot
        # base keeps the same table-length standoff heuristic; the wall's
        # x-position (and hence the true standoff) is set by table_offset[0].
        super(Wipe, self)._load_model()  # SingleArmEnv setup, skip Wipe's arena build

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        self.robot_contact_geoms = self.robots[0].robot_model.contact_geoms

        mujoco_arena = VerticalWipeArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            table_friction_std=self.table_friction_std,
            coverage_factor=self.coverage_factor,
            num_markers=self.num_markers,
            line_width=self.line_width,
            two_clusters=self.two_clusters,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

        # Start wall-facing (see WALL_INIT_QPOS comment). Panda-calibrated;
        # other 7-dof arms would need their own solve.
        if len(self.robots[0].init_qpos) == len(WALL_INIT_QPOS):
            self.robots[0].init_qpos = WALL_INIT_QPOS.copy()
