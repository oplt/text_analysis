"""Compile workflow recipes into deterministic execution plans."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.domain.workflow_recipe import WorkflowRecipe
from backend.modules.text_research.infrastructure.pipeline_compiler import (
    ExecutionPlan,
    compile_plan,
)


def compile_recipe(recipe: WorkflowRecipe) -> list[ExecutionPlan]:
    """Compile each recipe step into a pipeline execution plan."""
    plans: list[ExecutionPlan] = []
    for spec_dict in recipe.to_spec_list():
        spec = AnalysisSpecification.model_validate(spec_dict)
        plans.append(compile_plan(spec))
    return plans


def recipe_fingerprint(recipe: WorkflowRecipe) -> str:
    """Stable digest for a normalized workflow recipe."""
    payload = recipe.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def describe_recipe(recipe: WorkflowRecipe) -> dict[str, Any]:
    """Human/CLI-friendly summary of a workflow recipe."""
    ordered = recipe.topological_order()
    return {
        "name": recipe.name,
        "version": recipe.version,
        "corpus_id": recipe.corpus_id,
        "random_seed": recipe.random_seed,
        "step_count": len(ordered),
        "steps": [
            {
                "name": step.name,
                "analysis_type": step.analysis_type,
                "depends_on": list(step.depends_on),
            }
            for step in ordered
        ],
        "fingerprint": recipe_fingerprint(recipe),
    }
