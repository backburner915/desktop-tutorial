from __future__ import annotations

from r1_bimanual_dataset.core.bimanual_controller import BimanualController


def build_plan(controller: BimanualController, scenario, nominal):
    if scenario.failure_mode != "none":
        raise ValueError("normal_grasp only accepts failure_mode='none'")
    return controller.build(scenario, nominal)
