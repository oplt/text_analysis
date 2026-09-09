"""Project-access enforcement helpers shared by all text_research services.

Every research entity is reachable only through a `ResearchCorpus`, which is
itself owned by a `Project`. This mixin centralizes "load X and verify the
caller has access to its owning project" so individual services never skip
the check.
"""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.lib.project_access import SqlAlchemyProjectAccessPort
from backend.modules.text_research.domain.models import CorpusDocument, ResearchCorpus, TextUnit
from backend.modules.text_research.infrastructure.repositories import ResearchRepository


class ResearchAccessMixin:
    """Base class for text_research application services.

    Subclasses get `self.db`, `self.repo`, and `self.project_access`, plus
    helpers that resolve a resource *and* raise `HTTPException` if the
    current user cannot access its owning project.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = ResearchRepository(db)
        self.project_access = SqlAlchemyProjectAccessPort(db)

    async def ensure_project_access(self, *, user_id: str, project_id: str) -> None:
        await self.project_access.ensure_project_access(user_id, project_id)

    async def get_corpus_or_404(
        self, corpus_id: str, *, user_id: str, project_id: str | None = None
    ) -> ResearchCorpus:
        corpus = await self.repo.get_corpus(corpus_id)
        if corpus is None:
            raise HTTPException(status_code=404, detail="Research corpus not found")
        if project_id is not None and corpus.project_id != project_id:
            raise HTTPException(status_code=404, detail="Research corpus not found")
        await self.ensure_project_access(user_id=user_id, project_id=corpus.project_id)
        return corpus

    async def get_document_or_404(
        self, document_id: str, *, user_id: str
    ) -> tuple[CorpusDocument, ResearchCorpus]:
        document = await self.repo.get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Corpus document not found")
        corpus = await self.get_corpus_or_404(document.corpus_id, user_id=user_id)
        return document, corpus

    async def get_text_unit_or_404(
        self, text_unit_id: str, *, user_id: str
    ) -> tuple[TextUnit, ResearchCorpus]:
        unit = await self.repo.get_text_unit(text_unit_id)
        if unit is None:
            raise HTTPException(status_code=404, detail="Text unit not found")
        corpus_id = await self.repo.get_corpus_id_for_document(unit.corpus_document_id)
        if corpus_id is None:
            raise HTTPException(status_code=404, detail="Text unit not found")
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        return unit, corpus

    async def get_codebook_or_404(self, codebook_id: str, *, user_id: str):
        codebook = await self.repo.get_codebook(codebook_id)
        if codebook is None:
            raise HTTPException(status_code=404, detail="Codebook not found")
        await self.ensure_project_access(user_id=user_id, project_id=codebook.project_id)
        return codebook

    async def get_run_or_404(self, run_id: str, *, user_id: str):
        run = await self.repo.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Analysis run not found")
        await self.ensure_project_access(user_id=user_id, project_id=run.project_id)
        return run

    async def get_model_or_404(self, model_id: str, *, user_id: str):
        model = await self.repo.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Trained model not found")
        await self.ensure_project_access(user_id=user_id, project_id=model.project_id)
        return model

    async def get_snapshot_or_404(self, snapshot_id: str, *, user_id: str):
        snapshot = await self.repo.get_snapshot(snapshot_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Training dataset snapshot not found")
        await self.ensure_project_access(user_id=user_id, project_id=snapshot.project_id)
        return snapshot

    async def get_dictionary_or_404(self, dictionary_id: str, *, user_id: str):
        dictionary = await self.repo.get_dictionary(dictionary_id)
        if dictionary is None:
            raise HTTPException(status_code=404, detail="Dictionary not found")
        await self.ensure_project_access(user_id=user_id, project_id=dictionary.project_id)
        return dictionary

    async def get_preprocessing_profile_or_404(self, profile_id: str, *, user_id: str):
        profile = await self.repo.get_preprocessing_profile(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Preprocessing profile not found")
        await self.ensure_project_access(user_id=user_id, project_id=profile.project_id)
        return profile

    async def get_cleaning_profile_or_404(self, profile_id: str, *, user_id: str):
        profile = await self.repo.get_cleaning_profile(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Cleaning profile not found")
        await self.ensure_project_access(user_id=user_id, project_id=profile.project_id)
        return profile

    async def get_contextual_dataset_or_404(self, dataset_id: str, *, user_id: str):
        dataset = await self.repo.get_contextual_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Contextual dataset not found")
        await self.ensure_project_access(user_id=user_id, project_id=dataset.project_id)
        return dataset

    async def get_prediction_set_or_404(self, prediction_set_id: str, *, user_id: str):
        prediction_set = await self.repo.get_prediction_set(prediction_set_id)
        if prediction_set is None:
            raise HTTPException(status_code=404, detail="Prediction set not found")
        await self.ensure_project_access(user_id=user_id, project_id=prediction_set.project_id)
        return prediction_set
