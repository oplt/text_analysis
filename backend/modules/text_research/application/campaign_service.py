"""Annotation campaign CRUD, assignment wiring, and blind-mode policy."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.annotation_service import AnnotationService
from backend.modules.text_research.domain.enums import (
    AnnotationCampaignStatus,
    AnnotationMode,
    AnnotationTaskStatus,
)
from backend.modules.text_research.domain.models import AnnotationCampaign, dumps, loads


def resolve_annotation_mode(
    *,
    annotation_mode: str | None = None,
    blind_mode: bool | None = None,
    ai_assistance_enabled: bool | None = None,
) -> tuple[str, bool, bool]:
    """Normalize mode flags. Blind reliability and AI assistance are mutually exclusive."""
    if annotation_mode is not None:
        mode = AnnotationMode(annotation_mode)
    elif blind_mode is False or ai_assistance_enabled is True:
        mode = AnnotationMode.AI_ASSISTED
    else:
        mode = AnnotationMode.BLIND_RELIABILITY

    if mode == AnnotationMode.BLIND_RELIABILITY:
        return mode.value, True, False
    return mode.value, False, True


class AnnotationCampaignService(ResearchAccessMixin):
    async def create_campaign(
        self,
        *,
        user_id: str,
        project_id: str,
        corpus_id: str,
        name: str,
        description: str | None = None,
        codebook_id: str | None = None,
        unit_type: str = "paragraph",
        sampling_strategy: str = "random",
        assignment_strategy: str = "overlap",
        sample_size: int | None = None,
        overlap_count: int | None = None,
        overlap_percent: float | None = None,
        annotation_mode: str | None = None,
        blind_mode: bool | None = None,
        ai_assistance_enabled: bool | None = None,
        reveal_after: str = "campaign_released",
        annotator_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AnnotationCampaign:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id, project_id=project_id)
        codebook_version: str | None = None
        if codebook_id:
            codebook = await self.get_codebook_or_404(codebook_id, user_id=user_id)
            if codebook.project_id != corpus.project_id:
                raise HTTPException(
                    status_code=400, detail="Codebook must belong to the same project"
                )
            codebook_version = codebook.version

        mode, is_blind, ai_enabled = resolve_annotation_mode(
            annotation_mode=annotation_mode,
            blind_mode=blind_mode,
            ai_assistance_enabled=ai_assistance_enabled,
        )
        now = datetime.now(UTC)
        campaign = await self.repo.create_annotation_campaign(
            project_id=corpus.project_id,
            corpus_id=corpus_id,
            name=name,
            description=description,
            codebook_id=codebook_id,
            codebook_version=codebook_version,
            unit_type=unit_type,
            sampling_strategy=sampling_strategy,
            assignment_strategy=assignment_strategy,
            sample_size=sample_size,
            overlap_count=overlap_count,
            overlap_percent=overlap_percent,
            blind_mode=is_blind,
            ai_assistance_enabled=ai_enabled,
            annotation_mode=mode,
            reveal_after=reveal_after,
            status=AnnotationCampaignStatus.ACTIVE.value,
            annotator_ids=annotator_ids or [],
            created_by=user_id,
            started_at=now,
            metadata=metadata
            or {
                "provenance": {
                    "annotation_mode": mode,
                    "blind_mode": is_blind,
                    "ai_assistance_enabled": ai_enabled,
                }
            },
        )
        await self.db.flush()
        return campaign

    async def create_campaign_committed(self, **kwargs: Any) -> AnnotationCampaign:
        campaign = await self.create_campaign(**kwargs)
        await self.db.commit()
        await self.db.refresh(campaign)
        return campaign

    async def list_campaigns(
        self, project_id: str, *, user_id: str, corpus_id: str | None = None
    ) -> list[AnnotationCampaign]:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        return await self.repo.list_annotation_campaigns(project_id, corpus_id=corpus_id)

    async def get_campaign(self, campaign_id: str, *, user_id: str) -> AnnotationCampaign:
        campaign = await self.repo.get_annotation_campaign(campaign_id)
        if campaign is None:
            raise HTTPException(status_code=404, detail="Annotation campaign not found")
        await self.ensure_project_access(user_id=user_id, project_id=campaign.project_id)
        return campaign

    async def patch_campaign(
        self,
        campaign_id: str,
        *,
        user_id: str,
        name: str | None = None,
        description: str | None = None,
        status: str | None = None,
        annotation_mode: str | None = None,
        blind_mode: bool | None = None,
        ai_assistance_enabled: bool | None = None,
        reveal_after: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AnnotationCampaign:
        campaign = await self.get_campaign(campaign_id, user_id=user_id)
        if name is not None:
            campaign.name = name
        if description is not None:
            campaign.description = description
        if status is not None:
            campaign.status = AnnotationCampaignStatus(status).value
            if campaign.status == AnnotationCampaignStatus.COMPLETED.value:
                campaign.completed_at = datetime.now(UTC)
        if (
            annotation_mode is not None
            or blind_mode is not None
            or ai_assistance_enabled is not None
        ):
            mode, is_blind, ai_enabled = resolve_annotation_mode(
                annotation_mode=annotation_mode,
                blind_mode=blind_mode if annotation_mode is None else None,
                ai_assistance_enabled=ai_assistance_enabled if annotation_mode is None else None,
            )
            campaign.annotation_mode = mode
            campaign.blind_mode = is_blind
            campaign.ai_assistance_enabled = ai_enabled
            meta = loads(campaign.metadata_json, {}) or {}
            provenance = dict(meta.get("provenance") or {})
            provenance.update(
                {
                    "annotation_mode": mode,
                    "blind_mode": is_blind,
                    "ai_assistance_enabled": ai_enabled,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            meta["provenance"] = provenance
            campaign.metadata_json = dumps(meta)
        if metadata is not None:
            existing = loads(campaign.metadata_json, {}) or {}
            existing.update(metadata)
            campaign.metadata_json = dumps(existing)
        if reveal_after is not None:
            if reveal_after not in {"campaign_completed", "campaign_released"}:
                raise HTTPException(status_code=422, detail="Invalid campaign reveal policy")
            campaign.reveal_after = reveal_after
        await self.db.commit()
        await self.db.refresh(campaign)
        return campaign

    async def assign(
        self,
        campaign_id: str,
        *,
        user_id: str,
        annotator_ids: list[str] | None = None,
        sample_size: int | None = None,
        strategy: str | None = None,
        overlap_count: int | None = None,
        overlap_percent: float | None = None,
        random_seed: int | None = None,
        stratify_by: list[str] | None = None,
        stratum_mode: str = "proportional",
        sampling_level: str = "unit",
        max_units_per_document: int | None = None,
    ) -> dict[str, Any]:
        campaign = await self.get_campaign(campaign_id, user_id=user_id)
        targets = annotator_ids or loads(campaign.annotator_ids_json, []) or [user_id]
        result = await AnnotationService(self.db).assign_corpus_tasks(
            campaign.corpus_id,
            user_id=user_id,
            unit_type=campaign.unit_type,
            annotator_ids=targets,
            sample_size=sample_size or campaign.sample_size or 50,
            strategy=strategy or campaign.assignment_strategy,
            overlap_count=overlap_count if overlap_count is not None else campaign.overlap_count,
            overlap_percent=overlap_percent
            if overlap_percent is not None
            else campaign.overlap_percent,
            random_seed=random_seed,
            stratify_by=stratify_by,
            stratum_mode=stratum_mode,
            sampling_level=sampling_level,
            max_units_per_document=max_units_per_document,
            campaign_id=campaign.id,
        )
        campaign.annotator_ids_json = dumps(list(dict.fromkeys(targets)))
        if sample_size is not None:
            campaign.sample_size = sample_size
        if strategy is not None:
            campaign.assignment_strategy = strategy
        await self.db.commit()
        result["campaign_id"] = campaign.id
        result["annotation_mode"] = campaign.annotation_mode
        result["blind_mode"] = campaign.blind_mode
        return result

    async def progress(self, campaign_id: str, *, user_id: str) -> dict[str, Any]:
        campaign = await self.get_campaign(campaign_id, user_id=user_id)
        progress = await self.repo.annotation_progress_for_campaign(campaign_id)
        total_tasks = progress["total_tasks"]
        progress["completion_rate"] = (
            progress["completed_tasks"] / total_tasks if total_tasks else 0.0
        )
        progress["campaign_id"] = campaign.id
        progress["annotation_mode"] = campaign.annotation_mode
        progress["blind_mode"] = campaign.blind_mode
        released = campaign.status == AnnotationCampaignStatus.RELEASED.value or (
            campaign.reveal_after == "campaign_completed"
            and campaign.status == AnnotationCampaignStatus.COMPLETED.value
        )
        if campaign.blind_mode and not released and campaign.created_by != user_id:
            # Progress remains useful without disclosing any coder's work.
            progress["by_annotator"] = {}
        return progress

    async def campaign_payload(self, campaign: AnnotationCampaign) -> dict[str, Any]:
        return {
            "id": campaign.id,
            "project_id": campaign.project_id,
            "corpus_id": campaign.corpus_id,
            "name": campaign.name,
            "description": campaign.description,
            "codebook_id": campaign.codebook_id,
            "codebook_version": campaign.codebook_version,
            "unit_type": campaign.unit_type,
            "sampling_strategy": campaign.sampling_strategy,
            "assignment_strategy": campaign.assignment_strategy,
            "sample_size": campaign.sample_size,
            "overlap_count": campaign.overlap_count,
            "overlap_percent": campaign.overlap_percent,
            "blind_mode": campaign.blind_mode,
            "ai_assistance_enabled": campaign.ai_assistance_enabled,
            "annotation_mode": campaign.annotation_mode,
            "reveal_after": campaign.reveal_after,
            "status": campaign.status,
            "annotator_ids": loads(campaign.annotator_ids_json, []) or [],
            "created_by": campaign.created_by,
            "created_at": campaign.created_at,
            "started_at": campaign.started_at,
            "completed_at": campaign.completed_at,
            "metadata": loads(campaign.metadata_json, {}) or {},
        }

    async def blind_policy_for_annotator_unit(
        self, *, text_unit_id: str, annotator_id: str
    ) -> dict[str, Any]:
        """Return blind-coding constraints for an annotator on a unit."""
        task = await self.repo.get_task(text_unit_id, annotator_id)
        if task is None or not task.campaign_id:
            return {
                "blind_mode": False,
                "ai_assistance_enabled": True,
                "hide_peer_annotations": False,
                "hide_model_predictions": False,
                "hide_adjudications": False,
                "campaign_id": None,
                "annotation_mode": None,
            }
        campaign = await self.repo.get_annotation_campaign(task.campaign_id)
        if campaign is None:
            return {
                "blind_mode": False,
                "ai_assistance_enabled": True,
                "hide_peer_annotations": False,
                "hide_model_predictions": False,
                "hide_adjudications": False,
                "campaign_id": task.campaign_id,
                "annotation_mode": None,
            }
        campaign_status = getattr(campaign, "status", None)
        # Compatibility for pre-release persisted records/tests: only a
        # completed individual task is considered released when no campaign
        # lifecycle state exists. Real campaigns always have a status.
        released = (
            campaign_status == AnnotationCampaignStatus.RELEASED.value
            or (
                getattr(campaign, "reveal_after", "campaign_released") == "campaign_completed"
                and campaign_status == AnnotationCampaignStatus.COMPLETED.value
            )
            if campaign_status is not None
            else task.status == AnnotationTaskStatus.COMPLETED.value
        )
        hide = bool(campaign.blind_mode and not released)
        return {
            "blind_mode": campaign.blind_mode,
            "ai_assistance_enabled": campaign.ai_assistance_enabled and not campaign.blind_mode,
            "hide_peer_annotations": hide,
            "hide_model_predictions": hide or campaign.blind_mode,
            "hide_adjudications": hide,
            "campaign_id": campaign.id,
            "annotation_mode": campaign.annotation_mode,
            "task_status": task.status,
            "reveal_after": getattr(campaign, "reveal_after", "campaign_released"),
            "released": released,
        }
