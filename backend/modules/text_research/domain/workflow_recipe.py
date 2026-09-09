"""Workflow recipes — multi-step analysis DAGs over a shared corpus."""

from __future__ import annotations

from collections import deque
from typing import Any

from pydantic import BaseModel, Field

from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification


class RecipeStep(BaseModel):
    name: str
    analysis_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class WorkflowRecipe(BaseModel):
    name: str
    version: str
    corpus_id: str
    steps: list[RecipeStep]
    random_seed: int = 42

    def validate_dag(self) -> None:
        """Ensure dependencies exist and the step graph is acyclic."""
        names = {step.name for step in self.steps}
        for step in self.steps:
            for dependency in step.depends_on:
                if dependency not in names:
                    raise ValueError(
                        f"unknown dependency {dependency!r} for step {step.name!r}"
                    )

        indegree: dict[str, int] = {step.name: 0 for step in self.steps}
        adjacency: dict[str, list[str]] = {step.name: [] for step in self.steps}
        for step in self.steps:
            for dependency in step.depends_on:
                adjacency[dependency].append(step.name)
                indegree[step.name] += 1

        queue: deque[str] = deque(name for name, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for neighbor in adjacency[current]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(self.steps):
            raise ValueError("workflow recipe contains a dependency cycle")

    def topological_order(self) -> list[RecipeStep]:
        """Return steps in dependency order."""
        self.validate_dag()
        by_name = {step.name: step for step in self.steps}
        indegree: dict[str, int] = {step.name: 0 for step in self.steps}
        adjacency: dict[str, list[str]] = {step.name: [] for step in self.steps}
        for step in self.steps:
            for dependency in step.depends_on:
                adjacency[dependency].append(step.name)
                indegree[step.name] += 1

        queue: deque[str] = deque(name for name, degree in indegree.items() if degree == 0)
        ordered: list[RecipeStep] = []
        while queue:
            current = queue.popleft()
            ordered.append(by_name[current])
            for neighbor in adjacency[current]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        return ordered

    def to_spec_list(self) -> list[dict[str, Any]]:
        """Produce normalized AnalysisSpecification-like dicts per step."""
        specs: list[dict[str, Any]] = []
        for step in self.topological_order():
            spec = AnalysisSpecification.from_flat(
                corpus_id=self.corpus_id,
                analysis_type=step.analysis_type,
                analysis_parameters=step.parameters,
                random_seed=self.random_seed,
            ).normalize()
            specs.append(spec.model_dump(mode="json"))
        return specs
