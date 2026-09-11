# Corrected-scene reference

The corrected `spacerobot.usd` scene has no pre-existing success-grasp Python
script.  `scenario_reference.json` is therefore the first reproducible design
for this scene, and `run_reference.py` is the Golden Reference runner for the
new dataset adapter.  It must pass the reachability and physics checks before
it is used as a repeatability baseline.

The original USD is not copied, flattened, or edited by this project.
