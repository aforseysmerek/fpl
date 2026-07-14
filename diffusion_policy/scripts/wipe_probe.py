"""
wipe_probe.py — read the absolute geometry so we set the wipe contact height
exactly. Prints: table surface z, eef z, and the wiper-tool geom z's, then the
--wiper_offset that puts the tool tip on the surface. Fast (no rendering).

Run in robodiff:  python scripts/wipe_probe.py
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
    "Wipe", robots="Panda",
    controller_configs=load_controller_config(default_controller="OSC_POSE"),
    has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    control_freq=20, horizon=100, ignore_done=True, hard_reset=False,
)
obs = env.reset()

# --- table surface (dirt markers sit on it) ---
mz = [float(env.sim.data.body_xpos[env.sim.model.body_name2id(m.root_body)][2])
      for m in env.model.mujoco_arena.markers]
surface_z = float(np.mean(mz))
print(f"TABLE surface z (markers): {surface_z:.4f}   (min {min(mz):.4f}, max {max(mz):.4f})")
try:
    print(f"TABLE table_offset (config): {np.round(np.asarray(env.table_offset), 4)}")
except Exception:
    pass

# --- eef ---
eef = np.asarray(obs["robot0_eef_pos"], float)
print(f"EEF  robot0_eef_pos: {np.round(eef, 4)}   (eef_z={eef[2]:.4f})")

# --- wiper / gripper geoms ---
print("GRIPPER/TOOL geoms (name -> world z):")
tip_z = eef[2]
for i in range(env.sim.model.ngeom):
    name = env.sim.model.geom_id2name(i) or ""
    if any(s in name.lower() for s in ("gripper", "wip", "pad", "eef", "tool")):
        z = float(env.sim.data.geom_xpos[i][2])
        print(f"    {name:44s} z={z:.4f}")
        tip_z = min(tip_z, z)

tool_len = eef[2] - tip_z
print(f"\nlowest tool/gripper geom z (tip): {tip_z:.4f}")
print(f"tool extends {tool_len:.4f} m below the eef")
print(f"=> tip touches surface when eef_z = surface + {tool_len:.4f}")
print(f"=> run wipe_trace.py with  --wiper_offset {tool_len:.3f}   "
      f"(subtract ~0.005-0.010 for a gentle press)")
