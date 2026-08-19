"""Custom robosuite envs. Importing a submodule (or this package) registers
its envs with robosuite.make via robosuite's EnvMeta metaclass."""
from envs.vertical_wipe import VerticalWipe, VerticalWipeArena  # noqa: F401
from envs.pot_8qt import Pot8QtObject, TwoArmLift8QtPot  # noqa: F401
