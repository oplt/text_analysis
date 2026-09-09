"""P2 execution, plugin registry, and workflow recipe tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.core.config import settings
from backend.modules.text_research.application.workflow_composer import (
    compile_recipe,
    describe_recipe,
    recipe_fingerprint,
)
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.domain.analysis_task import AnalysisTask, resource_class_for
from backend.modules.text_research.domain.workflow_recipe import RecipeStep, WorkflowRecipe
from backend.modules.text_research.infrastructure.execution_policy import (
    Checkpoint,
    is_idempotent_hit,
    load_checkpoint,
    retry_policy_for,
    save_checkpoint,
    timeout_for,
)
from backend.modules.text_research.infrastructure.pipeline_compiler import (
    BASE_STAGES,
    computation_identity,
)
from backend.modules.text_research.infrastructure.plugin_registry import (
    get_plugin,
    list_plugins,
    register_builtins,
    register_plugin,
    unregister_plugin,
)
from backend.modules.text_research.workers import queue_for_resource_class


class QueueMappingTests(unittest.TestCase):
    def test_queue_for_resource_class_mapping(self) -> None:
        self.assertEqual(queue_for_resource_class("research_light"), settings.RESEARCH_QUEUE_LIGHT)
        self.assertEqual(queue_for_resource_class("research_cpu"), settings.RESEARCH_QUEUE_CPU)
        self.assertEqual(queue_for_resource_class("research_io"), settings.RESEARCH_QUEUE_IO)
        self.assertEqual(queue_for_resource_class("research_nlp"), settings.RESEARCH_QUEUE_NLP)
        self.assertEqual(
            queue_for_resource_class("research_memory"), settings.RESEARCH_QUEUE_MEMORY
        )
        self.assertEqual(queue_for_resource_class("research_gpu"), settings.RESEARCH_QUEUE_GPU)

    def test_resource_class_for_analysis_types(self) -> None:
        self.assertEqual(resource_class_for("frequencies"), "research_light")
        self.assertEqual(resource_class_for("dfm"), "research_memory")
        self.assertEqual(resource_class_for("classification"), "research_cpu")
        self.assertEqual(resource_class_for("embedding"), "research_nlp")
        self.assertEqual(resource_class_for("topic_model"), "research_gpu")


class AnalysisTaskCreateTests(unittest.TestCase):
    def test_create_fills_defaults(self) -> None:
        task = AnalysisTask.create("topic_model", spec_hash="abc123")
        self.assertEqual(task.resource_class, "research_gpu")
        self.assertEqual(task.retry_policy, retry_policy_for("research_gpu"))
        self.assertEqual(task.timeout_seconds, timeout_for("research_gpu"))
        self.assertEqual(task.progress, 0.0)
        self.assertIsNone(task.checkpoint)
        self.assertEqual(task.input_artifact_ids, [])

    def test_create_accepts_input_artifact_ids(self) -> None:
        task = AnalysisTask.create(
            "classification",
            spec_hash="def456",
            input_artifact_ids=["artifact-1"],
        )
        self.assertEqual(task.resource_class, "research_cpu")
        self.assertEqual(task.input_artifact_ids, ["artifact-1"])


class CheckpointTests(unittest.TestCase):
    def test_save_and_load_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = Checkpoint.now(stage="prepare_corpus", progress=0.5, payload={"rows": 10})
            path = save_checkpoint("run-1", checkpoint, root=root)
            self.assertTrue(path.is_file())

            loaded = load_checkpoint("run-1", root=root)
            assert loaded is not None
            self.assertEqual(loaded.stage, "prepare_corpus")
            self.assertEqual(loaded.progress, 0.5)
            self.assertEqual(loaded.payload, {"rows": 10})

    def test_load_missing_checkpoint_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_checkpoint("missing", root=Path(tmp)))


class IdempotencyTests(unittest.TestCase):
    def test_is_idempotent_hit_returns_cached_stage_key(self) -> None:
        identity = computation_identity("spec", "snapshot", "engine/1")
        cache = {identity: "prepared_corpus:abc"}

        def stage_cache_get(key: str) -> str | None:
            return cache.get(key)

        hit = is_idempotent_hit(
            spec_hash="spec",
            corpus_snapshot_hash="snapshot",
            engine_version="engine/1",
            stage_cache_get=stage_cache_get,
        )
        self.assertEqual(hit, "prepared_corpus:abc")


class PluginRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        register_builtins()

    def test_register_and_get_topic_engine(self) -> None:
        factory = get_plugin("topic_engine", "sklearn_lda")
        engine = factory()
        self.assertEqual(engine.name, "sklearn_lda")

    def test_list_plugins_includes_analysis_types(self) -> None:
        plugins = list_plugins("analysis")
        self.assertIn("frequencies", plugins["analysis"])
        self.assertIn("topic_model", plugins["analysis"])

    def test_analysis_plugin_resolves_compile_plan_stage(self) -> None:
        spec = AnalysisSpecification.from_flat(corpus_id="c1", analysis_type="dfm")
        factory = get_plugin("analysis", "dfm")
        stage = factory(spec.model_dump(mode="json"))
        self.assertEqual(stage, "dfm")

    def test_unregister_plugin(self) -> None:
        register_plugin("text_transform", "test_transform", lambda: None)
        self.assertIn("test_transform", list_plugins("text_transform")["text_transform"])
        unregister_plugin("text_transform", "test_transform")
        self.assertNotIn("test_transform", list_plugins("text_transform")["text_transform"])


class WorkflowRecipeTests(unittest.TestCase):
    def _sample_recipe(self) -> WorkflowRecipe:
        return WorkflowRecipe(
            name="baseline",
            version="1.0",
            corpus_id="corpus-1",
            steps=[
                RecipeStep(name="freq", analysis_type="frequencies"),
                RecipeStep(name="dfm", analysis_type="dfm", depends_on=["freq"]),
            ],
        )

    def test_topological_order_respects_dependencies(self) -> None:
        ordered = self._sample_recipe().topological_order()
        self.assertEqual([step.name for step in ordered], ["freq", "dfm"])

    def test_cycle_detection_raises(self) -> None:
        recipe = WorkflowRecipe(
            name="cycle",
            version="1.0",
            corpus_id="corpus-1",
            steps=[
                RecipeStep(name="a", analysis_type="frequencies", depends_on=["b"]),
                RecipeStep(name="b", analysis_type="dfm", depends_on=["a"]),
            ],
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            recipe.validate_dag()

    def test_compile_recipe_includes_prepare_corpus(self) -> None:
        plans = compile_recipe(self._sample_recipe())
        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertIn("prepare_corpus", plan.stages)
            self.assertEqual(plan.stages[: len(BASE_STAGES)], list(BASE_STAGES))

    def test_describe_recipe_and_fingerprint(self) -> None:
        recipe = self._sample_recipe()
        summary = describe_recipe(recipe)
        self.assertEqual(summary["step_count"], 2)
        self.assertEqual(summary["fingerprint"], recipe_fingerprint(recipe))


if __name__ == "__main__":
    unittest.main()
