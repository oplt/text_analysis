"""CRUD + apply for versioned document CleaningProfiles.

Cleaning operates on raw extracted text and rewrites only the cleaned
canonical representation. Raw extracts are never overwritten.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, CleaningProfile, dumps, loads
from backend.modules.text_research.infrastructure.canonical_text import sha256_text
from backend.modules.text_research.infrastructure.document_cleaning import (
    CleaningConfig,
    apply_cleaning,
    preview_cleaning,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CleaningProfileService(ResearchAccessMixin):
    @staticmethod
    def _resolved_config_dict(config: dict | None) -> dict:
        try:
            return CleaningConfig.from_dict(config).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    async def create_profile(
        self,
        *,
        project_id: str,
        user_id: str,
        name: str,
        description: str | None = None,
        version: str = "1.0",
        config: dict | None = None,
    ) -> CleaningProfile:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        resolved = self._resolved_config_dict(config)
        profile = await self.repo.create_cleaning_profile(
            project_id=project_id,
            name=name,
            description=description,
            version=version or "1.0",
            config_json=json.dumps(resolved, ensure_ascii=True),
            created_by=user_id,
        )
        await self.db.commit()
        return profile

    async def list_profiles(self, *, project_id: str, user_id: str) -> list[CleaningProfile]:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        return await self.repo.list_cleaning_profiles(project_id)

    async def get_profile(self, profile_id: str, *, user_id: str) -> CleaningProfile:
        return await self.get_cleaning_profile_or_404(profile_id, user_id=user_id)

    async def update_profile(
        self,
        profile_id: str,
        *,
        user_id: str,
        name: str | None = None,
        description: str | None = None,
        version: str | None = None,
        config: dict | None = None,
    ) -> CleaningProfile:
        profile = await self.get_cleaning_profile_or_404(profile_id, user_id=user_id)
        fields: dict[str, Any] = {}
        if name is not None:
            fields["name"] = name
        if description is not None:
            fields["description"] = description
        if version is not None:
            fields["version"] = version
        if config is not None:
            fields["config_json"] = json.dumps(
                self._resolved_config_dict(config), ensure_ascii=True
            )
        updated = await self.repo.update_cleaning_profile(profile, **fields)
        await self.db.commit()
        return updated

    async def delete_profile(self, profile_id: str, *, user_id: str) -> None:
        profile = await self.get_cleaning_profile_or_404(profile_id, user_id=user_id)
        await self.repo.delete_cleaning_profile(profile)
        await self.db.commit()

    async def preview(
        self,
        *,
        user_id: str,
        project_id: str | None = None,
        texts: list[str] | None = None,
        config: dict | None = None,
        cleaning_profile_id: str | None = None,
        document_id: str | None = None,
    ) -> dict:
        resolved_config: dict | None = config
        profile_name: str | None = None
        profile_version: str | None = None

        if cleaning_profile_id:
            profile = await self.get_cleaning_profile_or_404(cleaning_profile_id, user_id=user_id)
            if project_id and profile.project_id != project_id:
                raise HTTPException(status_code=404, detail="Cleaning profile not found")
            resolved_config = json.loads(profile.config_json)
            profile_name = profile.name
            profile_version = profile.version
            project_id = profile.project_id
        elif project_id:
            await self.ensure_project_access(user_id=user_id, project_id=project_id)

        sample_texts = [t for t in (texts or []) if t and t.strip()]
        if not sample_texts and document_id:
            source = await self._raw_text_for_document(document_id, user_id=user_id)
            sample_texts = [source]

        if not sample_texts:
            raise HTTPException(
                status_code=422,
                detail="Provide sample texts or a document_id to preview cleaning.",
            )

        try:
            preview = preview_cleaning(sample_texts, resolved_config)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        preview["profile_name"] = profile_name
        preview["profile_version"] = profile_version
        return preview

    async def apply_to_corpus(
        self,
        corpus_id: str,
        *,
        user_id: str,
        cleaning_profile_id: str,
        document_ids: list[str] | None = None,
    ) -> AnalysisRun:
        """Recompute canonical text from raw extracts using a cleaning profile.

        Never overwrites ``raw_extracted_text``.
        """
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        profile = await self.get_cleaning_profile_or_404(cleaning_profile_id, user_id=user_id)
        if profile.project_id != corpus.project_id:
            raise HTTPException(
                status_code=422,
                detail="Cleaning profile belongs to a different project than this corpus.",
            )

        config = CleaningConfig.from_dict(json.loads(profile.config_json))
        documents = await self.repo.list_documents(corpus_id)
        if document_ids:
            allow = set(document_ids)
            documents = [d for d in documents if d.id in allow]

        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus_id,
                run_type=AnalysisRunType.DOCUMENT_CLEANING.value,
                status=AnalysisRunStatus.RUNNING.value,
                progress_stage="cleaning",
                parameters_json=dumps(
                    {
                        "cleaning_profile_id": profile.id,
                        "cleaning_profile_name": profile.name,
                        "cleaning_profile_version": profile.version,
                        "config": config.to_dict(),
                        "document_count": len(documents),
                        "mutates_raw": False,
                    }
                ),
                created_by=user_id,
                started_at=_utcnow(),
            )
        )
        await self.db.commit()

        try:
            per_document: list[dict[str, Any]] = []
            for document in documents:
                source = await self.repo.get_canonical_source(document.id)
                if source is None:
                    per_document.append(
                        {
                            "document_id": document.id,
                            "status": "skipped",
                            "reason": "no_canonical_source",
                        }
                    )
                    continue

                raw = source.raw_extracted_text
                if raw is None:
                    # Legacy row: treat current canonical as raw once, then clean.
                    raw = source.canonical_text
                    await self.repo.update_canonical_source(
                        source,
                        raw_extracted_text=raw,
                        raw_extracted_checksum=source.canonical_text_checksum or sha256_text(raw),
                    )

                result = apply_cleaning(raw, config)
                assert result.raw_text == raw  # raw never mutated by engine

                existing_meta = loads(source.transformation_metadata_json, {}) or {}
                transformation = {
                    **existing_meta,
                    "cleaning": {
                        **result.to_transformation_metadata(),
                        "cleaning_profile_id": profile.id,
                        "cleaning_profile_name": profile.name,
                        "cleaning_profile_version": profile.version,
                        "applied_at": _utcnow().isoformat(),
                    },
                }
                await self.repo.update_canonical_source(
                    source,
                    canonical_text=result.cleaned_text,
                    canonical_text_checksum=result.cleaned_checksum,
                    cleaning_profile_id=profile.id,
                    transformation_metadata_json=dumps(transformation),
                    # raw fields intentionally untouched after backfill
                )
                per_document.append(
                    {
                        "document_id": document.id,
                        "status": "cleaned",
                        "raw_checksum": result.raw_checksum,
                        "cleaned_checksum": result.cleaned_checksum,
                        "chars_raw": len(result.raw_text),
                        "chars_cleaned": len(result.cleaned_text),
                        "steps_enabled": [s.name for s in result.steps if s.enabled],
                    }
                )

            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.COMPLETED.value,
                progress_stage="completed",
                completed_at=_utcnow(),
                metrics_json=dumps(
                    {
                        "documents_processed": len(per_document),
                        "documents_cleaned": sum(
                            1 for row in per_document if row.get("status") == "cleaned"
                        ),
                        "mutates_raw": False,
                    }
                ),
                results_json=dumps(
                    {
                        "cleaning_profile_id": profile.id,
                        "cleaning_profile_version": profile.version,
                        "config": config.to_dict(),
                        "documents": per_document,
                    }
                ),
            )
            await self.db.commit()
        except Exception as exc:  # noqa: BLE001
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.FAILED.value,
                progress_stage="failed",
                completed_at=_utcnow(),
                error_message=str(exc),
            )
            await self.db.commit()
            raise

        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def _raw_text_for_document(self, document_id: str, *, user_id: str) -> str:
        document, _ = await self.get_document_or_404(document_id, user_id=user_id)
        source = await self.repo.get_canonical_source(document.id)
        if source is None:
            raise HTTPException(status_code=422, detail="No canonical research source for document")
        if source.raw_extracted_text:
            return source.raw_extracted_text
        return source.canonical_text

    @staticmethod
    def resolve_config(profile: CleaningProfile | None) -> CleaningConfig:
        if profile is None:
            return CleaningConfig()
        return CleaningConfig.from_dict(json.loads(profile.config_json))
