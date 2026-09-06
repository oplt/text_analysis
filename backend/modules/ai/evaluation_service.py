from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.core.config import settings
from backend.core.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from backend.modules.ai.evaluation_scoring import _EvaluationCaseResult, score_evaluation_case
from backend.modules.ai.models import AiEvaluationRun, AiPromptTemplate, AiPromptVersion
from backend.modules.ai.run_service import AiRunService
from backend.modules.identity_access.models import User
from backend.modules.identity_access.repository import IdentityRepository

logger = logging.getLogger(__name__)


class AiEvaluationService(AiRunService):
    async def list_datasets(
        self,
        user: User,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ):
        return await self.repo.list_datasets_for_user(user.id, limit=limit, offset=offset)

    async def create_dataset(self, user: User, name: str, description: str | None):
        dataset = await self.repo.create_dataset(
            user_id=user.id, name=name, description=description
        )
        await self.db.commit()
        await self.db.refresh(dataset)
        return dataset

    async def update_dataset(self, user: User, dataset_id: str, updates: dict[str, Any]):
        dataset = await self.repo.get_dataset_for_user(user.id, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Evaluation dataset not found")
        for field, value in updates.items():
            setattr(dataset, field, value)
        await self.db.commit()
        await self.db.refresh(dataset)
        return dataset

    async def list_dataset_cases(
        self,
        user: User,
        dataset_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ):
        dataset = await self.repo.get_dataset_for_user(user.id, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Evaluation dataset not found")
        return await self.repo.list_dataset_cases(dataset.id, limit=limit, offset=offset)

    async def create_dataset_case(self, user: User, dataset_id: str, payload: dict[str, Any]):
        dataset = await self.repo.get_dataset_for_user(user.id, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Evaluation dataset not found")
        case = await self.repo.create_dataset_case(
            dataset_id=dataset.id,
            input_variables_json=payload["input_variables"],
            retrieval_query=payload["retrieval_query"],
            document_ids_json=payload["document_ids"],
            expected_chunk_ids_json=payload["expected_chunk_ids"],
            expected_output_text=payload["expected_output_text"],
            expected_output_json=payload["expected_output_json"],
            notes=payload["notes"],
        )
        await self.db.commit()
        await self.db.refresh(case)
        return case

    async def _list_all_dataset_cases(self, dataset_id: str):
        cases, total = await self.repo.list_dataset_cases(
            dataset_id, limit=MAX_PAGE_LIMIT, offset=0
        )
        offset = len(cases)
        while offset < total:
            page, _ = await self.repo.list_dataset_cases(
                dataset_id, limit=MAX_PAGE_LIMIT, offset=offset
            )
            cases.extend(page)
            offset += len(page)
        return cases

    async def _execute_evaluation_case(
        self,
        *,
        user: User,
        template: AiPromptTemplate,
        version: AiPromptVersion,
        dataset_id: str,
        case,
    ) -> _EvaluationCaseResult:
        from backend.db.session import SessionLocal

        async with SessionLocal() as db:
            case_service = AiRunService(db)
            ai_run = await case_service.run_prompt(
                user,
                prompt_template_key=template.key,
                prompt_version_id=version.id,
                variables=case.input_variables_json,
                retrieval_query=case.retrieval_query,
                document_ids=case.document_ids_json,
                top_k=max(4, len(case.expected_chunk_ids_json)),
                review_required=False,
                evaluation_dataset_id=dataset_id,
                evaluation_case_id=case.id,
            )
            score, passed, notes = self._score_evaluation_case(
                ai_run.output_text,
                ai_run.output_json,
                ai_run.retrieved_chunk_ids_json,
                case,
            )
            return _EvaluationCaseResult(
                case_id=case.id,
                ai_run_id=ai_run.id,
                score=score,
                passed=passed,
                notes=notes,
            )

    async def queue_evaluation(
        self, user: User, dataset_id: str, prompt_version_id: str
    ) -> AiEvaluationRun:
        dataset = await self.repo.get_dataset_for_user(user.id, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Evaluation dataset not found")
        version = await self.repo.get_prompt_version(prompt_version_id)
        if not version:
            raise HTTPException(status_code=404, detail="Prompt version not found")
        template = await self.repo.get_prompt_template_for_user(user.id, version.prompt_template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Prompt template not found")

        cases = await self._list_all_dataset_cases(dataset.id)
        evaluation_run = await self.repo.create_evaluation_run(
            dataset_id=dataset.id,
            prompt_version_id=version.id,
            user_id=user.id,
            status="running",
            total_cases=len(cases),
            passed_cases=0,
            average_score=0,
        )
        await self.db.commit()
        await self.db.refresh(evaluation_run)

        from backend.workers.evaluation import queue_evaluation_run

        queue_evaluation_run(
            evaluation_run_id=evaluation_run.id,
            user_id=user.id,
            dataset_id=dataset.id,
            prompt_version_id=version.id,
        )
        return evaluation_run

    async def execute_evaluation_run(
        self,
        *,
        evaluation_run_id: str,
        user_id: str,
        dataset_id: str,
        prompt_version_id: str,
    ) -> None:
        try:
            user = await IdentityRepository(self.db).get_user_by_id(user_id)
            if not user:
                await self._mark_evaluation_run_failed(
                    evaluation_run_id, notes="User not found for evaluation run"
                )
                return

            dataset = await self.repo.get_dataset_for_user(user_id, dataset_id)
            if not dataset:
                await self._mark_evaluation_run_failed(
                    evaluation_run_id, notes="Evaluation dataset not found"
                )
                return
            version = await self.repo.get_prompt_version(prompt_version_id)
            if not version:
                await self._mark_evaluation_run_failed(
                    evaluation_run_id, notes="Prompt version not found"
                )
                return
            template = await self.repo.get_prompt_template_for_user(
                user_id, version.prompt_template_id
            )
            if not template:
                await self._mark_evaluation_run_failed(
                    evaluation_run_id, notes="Prompt template not found"
                )
                return

            evaluation_run = await self.repo.get_evaluation_run_by_id(evaluation_run_id)
            if not evaluation_run:
                logger.warning(
                    "Evaluation run %s not found; skipping worker execution",
                    evaluation_run_id,
                )
                return
            if evaluation_run.status != "running":
                logger.info(
                    "Evaluation run %s already in status=%s; skipping",
                    evaluation_run_id,
                    evaluation_run.status,
                )
                return

            cases = await self._list_all_dataset_cases(dataset.id)
            evaluation_run.total_cases = len(cases)
            concurrency = max(1, settings.AI_EVALUATION_CONCURRENCY)
            semaphore = asyncio.Semaphore(concurrency)

            async def _run_case(case):
                async with semaphore:
                    return await self._execute_evaluation_case(
                        user=user,
                        template=template,
                        version=version,
                        dataset_id=dataset.id,
                        case=case,
                    )

            results = await asyncio.gather(*[_run_case(case) for case in cases])
            passed_cases = 0
            scores: list[float] = []
            item_payloads: list[dict] = []
            for result in results:
                scores.append(result.score)
                if result.passed:
                    passed_cases += 1
                item_payloads.append(
                    {
                        "evaluation_run_id": evaluation_run.id,
                        "evaluation_case_id": result.case_id,
                        "ai_run_id": result.ai_run_id,
                        "score": result.score,
                        "passed": result.passed,
                        "notes": result.notes,
                    }
                )
            if item_payloads:
                await self.repo.create_evaluation_run_items_batch(item_payloads)
            evaluation_run.status = "completed"
            evaluation_run.passed_cases = passed_cases
            evaluation_run.average_score = round(sum(scores) / len(scores), 4) if scores else 0.0
            evaluation_run.completed_at = datetime.now(UTC)
            await self.db.commit()
        except Exception:
            logger.exception("AI evaluation run %s failed", evaluation_run_id)
            await self._mark_evaluation_run_failed(evaluation_run_id)
            raise

    async def _mark_evaluation_run_failed(
        self, evaluation_run_id: str, *, notes: str | None = None
    ) -> None:
        if notes:
            logger.error("Evaluation run %s failed: %s", evaluation_run_id, notes)
        evaluation_run = await self.repo.get_evaluation_run_by_id(evaluation_run_id)
        if evaluation_run is None or evaluation_run.status != "running":
            return
        evaluation_run.status = "failed"
        evaluation_run.completed_at = datetime.now(UTC)
        await self.db.commit()

    def _score_evaluation_case(
        self,
        output_text: str | None,
        output_json: dict | None,
        retrieved_chunk_ids: list[str],
        case,
    ) -> tuple[float, bool, str]:
        return score_evaluation_case(output_text, output_json, retrieved_chunk_ids, case)

    async def list_evaluation_runs(
        self,
        user: User,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ):
        return await self.repo.list_evaluation_runs_for_user(user.id, limit=limit, offset=offset)
