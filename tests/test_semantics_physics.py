import math
import unittest

from BEACON.environment.material import Material
from BEACON.environment.obstacle import Obstacle
from BEACON.interaction.interaction_cost import GRAVITY, InteractionCost
from BEACON.semantic_dstar_lite import PlannerConfig, resolve_planner_config, summarize_scene_cost_inputs


def make_obstacle(*, mass=2.0, friction=0.4, pushable=True, movable=True):
    material = Material(
        name="test",
        density=1.0,
        mass=mass,
        friction=friction,
        fragility=0.1,
        pushable=pushable,
    )
    return Obstacle(
        id=1,
        shape="polygon",
        position=(0.5, 0.5),
        size={"radius": 0.0, "vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]},
        material=material,
        movable=movable,
        class_label="movable",
        vertices=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )


class SemanticsPhysicsAlignmentTests(unittest.TestCase):
    def test_required_push_force_matches_friction_model(self):
        obstacle = make_obstacle(mass=3.0, friction=0.25)
        model = InteractionCost(push_duration=2.0)

        expected_force = 0.25 * 3.0 * GRAVITY

        self.assertAlmostEqual(
            model.required_push_force(
                obstacle,
                push_distance=0.75,
                effective_mass=3.0,
                effective_friction=0.25,
                cluster_factor=4.0,
            ),
            expected_force,
        )

    def test_push_energy_and_power_follow_semantics_formulas(self):
        obstacle = make_obstacle(mass=1.5, friction=0.5)
        model = InteractionCost(push_duration=2.0)

        expected_force = 0.5 * 1.5 * GRAVITY
        expected_energy = expected_force * 0.8
        expected_power = expected_force * (0.8 / 2.0)
        metrics = model.estimate_push_metrics(obstacle, 0.8)

        self.assertAlmostEqual(metrics["required_force"], expected_force)
        self.assertAlmostEqual(metrics["energy"], expected_energy)
        self.assertAlmostEqual(metrics["work"], expected_energy)
        self.assertAlmostEqual(metrics["average_power"], expected_power)

    def test_minimum_force_floor_is_preserved_for_near_zero_inputs(self):
        obstacle = make_obstacle(mass=0.0001, friction=0.0001)
        model = InteractionCost()

        self.assertTrue(math.isclose(model.required_push_force(obstacle, 0.0), 0.05))

    def test_scene_hyperparameters_can_resolve_cost_weightages(self):
        scene = {
            "workspace": [0.0, 2.0, 0.0, 2.0],
            "obstacles": [
                {
                    "id": 1,
                    "class_true": "fragile",
                    "occupancy_prob": 0.7,
                    "dynamic_prediction": {"1": 0.6, "2": 0.2},
                },
                {
                    "id": 2,
                    "semantic_probs": {"human": 0.8, "safe": 0.2},
                    "occupancy_prob": 0.9,
                },
            ],
        }
        config = PlannerConfig(
            lambda_occ=5.0,
            lambda_sem=2.5,
            auto_tune_weights=True,
            occupancy_weight_scale=0.8,
            semantic_weight_scale=0.2,
            dynamic_weight_scale=0.5,
            density_weight_scale=0.3,
        )

        resolved_config, weight_tuning = resolve_planner_config(scene, config)

        self.assertGreater(resolved_config.lambda_occ, config.lambda_occ)
        self.assertGreater(resolved_config.lambda_sem, config.lambda_sem)
        self.assertEqual(weight_tuning["mode"], "auto")
        self.assertIn("scene_summary", weight_tuning)

    def test_manual_weightages_stay_unchanged_without_auto_tuning(self):
        scene = {"workspace": [0.0, 1.0, 0.0, 1.0], "obstacles": []}
        config = PlannerConfig(lambda_occ=7.0, lambda_sem=3.5, auto_tune_weights=False)

        resolved_config, weight_tuning = resolve_planner_config(scene, config)

        self.assertEqual(resolved_config.lambda_occ, 7.0)
        self.assertEqual(resolved_config.lambda_sem, 3.5)
        self.assertEqual(weight_tuning["mode"], "manual")

    def test_scene_summary_handles_empty_scenes(self):
        summary = summarize_scene_cost_inputs(
            {"workspace": [0.0, 3.0, 0.0, 2.0], "obstacles": []},
        )

        self.assertEqual(summary["obstacle_count"], 0)
        self.assertEqual(summary["obstacle_density"], 0.0)
        self.assertEqual(summary["avg_semantic_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
