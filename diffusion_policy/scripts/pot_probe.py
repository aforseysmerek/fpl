"""
pot_probe.py — read the absolute geometry of the TwoArmLift task so we set the
grasp / lift heights exactly (mirror of wipe_probe.py, for two arms). Prints:
table top z, both eef start positions, both handle world positions, the pot
center / top_offset, and the gripper finger reach below each eef -> the grasp z
offset that lands the fingers around a handle. Fast (no rendering).

Run in robodiff:  python scripts/pot_probe.py
"""
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import robosuite
try:
    from robosuite.controllers import load_controller_config
except ImportError:
    from robosuite.controllers.controller_factory import load_controller_config

env = robosuite.make(
    "TwoArmLift", robots=["Panda", "Panda"], env_configuration="single-arm-opposed",
    controller_configs=load_controller_config(default_controller="OSC_POSE"),
    has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    control_freq=20, horizon=100, ignore_done=True, hard_reset=False,
)
obs = env.reset()

# --- table top ---
table_z = float(env.sim.data.site_xpos[env.table_top_id][2])
print(f"TABLE top z: {table_z:.4f}")
print(f"TABLE table_offset (config): {np.round(np.asarray(env.table_offset), 4)}")

# --- pot ---
center = np.asarray(env.sim.data.site_xpos[env.pot_center_id], float)
top_off = np.asarray(env.pot.top_offset, float)
pot_bottom = center[2] - top_off[2]
print(f"POT  center: {np.round(center, 4)}   top_offset={np.round(top_off, 4)}")
print(f"POT  bottom z: {pot_bottom:.4f}   (bottom above table = {pot_bottom - table_z:.4f})")
print(f"POT  success needs bottom > table + 0.10  ->  lift bottom above {table_z + 0.10:.4f}")

# --- handles + eefs (arm0 -> handle0, arm1 -> handle1) ---
h0 = np.asarray(env._handle0_xpos, float)
h1 = np.asarray(env._handle1_xpos, float)
e0 = np.asarray(env._eef0_xpos, float)
e1 = np.asarray(env._eef1_xpos, float)
print(f"HANDLE0 (arm0/green): {np.round(h0, 4)}")
print(f"HANDLE1 (arm1/blue):  {np.round(h1, 4)}")
print(f"EEF0 start: {np.round(e0, 4)}   dist->handle0 = {np.linalg.norm(h0 - e0):.4f}")
print(f"EEF1 start: {np.round(e1, 4)}   dist->handle1 = {np.linalg.norm(h1 - e1):.4f}")

# --- gripper finger reach below each eef (top-down grasp offset) ---
def finger_reach(eef_z, prefix):
    tip = eef_z
    for i in range(env.sim.model.ngeom):
        name = env.sim.model.geom_id2name(i) or ""
        if prefix in name and any(s in name.lower() for s in ("finger", "pad", "grip")):
            tip = min(tip, float(env.sim.data.geom_xpos[i][2]))
    return eef_z - tip

pf0 = env.robots[0].robot_model.naming_prefix   # e.g. "robot0_"
pf1 = env.robots[1].robot_model.naming_prefix   # e.g. "robot1_"
r0 = finger_reach(e0[2], pf0)
r1 = finger_reach(e1[2], pf1)
print(f"\nARM0 fingers reach {r0:.4f} m below eef  -> grasp eef_z = handle0_z + {r0:.3f}")
print(f"ARM1 fingers reach {r1:.4f} m below eef  -> grasp eef_z = handle1_z + {r1:.3f}")
print("\n=> run pot_trace.py with --grasp_offset ~", f"{max(r0, r1):.3f}",
      "(subtract a little to seat fingers around the handle)")

# --- grasp orientation: straight down at each arm's own natural yaw ---
from scripts.pot_trace import down_grasp_aa
s = (h1 - h0)[:2]
pot_yaw = float(np.degrees(np.arctan2(s[1], s[0])))
print(f"\nGRASP  handle0->handle1 axis (pot yaw) = {pot_yaw:.1f} deg")
print(f"GRASP  eef axis-angle arm0 (down @ natural yaw) = {np.round(down_grasp_aa(obs['robot0_eef_quat']), 3)}")
print(f"GRASP  eef axis-angle arm1 (down @ natural yaw) = {np.round(down_grasp_aa(obs['robot1_eef_quat']), 3)}")
print("       (gripper points straight down; each arm keeps its reachable yaw)")
