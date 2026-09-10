from __future__ import annotations

from r1_bimanual_dataset.core.bimanual_controller import BimanualController


def build_plan(controller: BimanualController, scenario, nominal):
    if scenario.failure_mode == "none":
        raise ValueError("perturbed_grasp requires one primary failure_mode")
    return controller.build(scenario, nominal)
