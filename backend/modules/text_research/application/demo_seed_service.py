"""SYNTHETIC demo-data seeding for the Policy Text Lab.

Every document, organization, and passage of text created here is
**fabricated for demonstration purposes only** — none of it represents a
real government, intergovernmental organization, NGO, or policy document.
Every seeded row is tagged so it can never be mistaken for real research
data:

* Organization names are explicitly fictional (e.g. "Fictional Global
  Council for Learning") and never reuse a real IO/NGO/government name or
  acronym.
* `CorpusDocument.research_notes` and the corpus description are prefixed
  with "SYNTHETIC DEMO DATA".
* The backing `RagDocument.metadata_json` carries `synthetic_demo: true`.

This lets a new user explore the full corpus -> segmentation -> annotation ->
classification workflow immediately, without needing to source real policy
documents first.
"""

from __future__ import annotations

from typing import Any

from backend.modules.rag.infrastructure.repositories import RagRepository
from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.models import CorpusDocument, ResearchCorpus, dumps

SYNTHETIC_TAG = "SYNTHETIC DEMO DATA"

_SYNTHETIC_DOCUMENTS: tuple[dict[str, Any], ...] = (
    {
        "organization": "Fictional Global Council for Learning (FGCL)",
        "organization_type": "intergovernmental_fictional",
        "region": "Global (fictional)",
        "cultural_sphere": "Composite/Fictional",
        "country": None,
        "language": "en",
        "publication_year": 2019,
        "publication_type": "policy_brief",
        "title": "Framework for Universal Learning Rights (SYNTHETIC)",
        "text": (
            "This fictional policy brief affirms that every learner, regardless of "
            "nationality or background, holds an inherent right to quality education. "
            "The Council calls on member states to guarantee free and compulsory primary "
            "and secondary education as an individual entitlement, while fostering "
            "personal choice in curriculum pathways and school selection. Education "
            "systems should be designed around universal standards that apply equally "
            "to all learners everywhere, independent of local context."
        ),
    },
    {
        "organization": "Alliance of Southern Community Schools (ASCS)",
        "organization_type": "ngo_fictional",
        "region": "Fictional Southern Region",
        "cultural_sphere": "Communal/Fictional",
        "country": None,
        "language": "en",
        "publication_year": 2021,
        "publication_type": "position_paper",
        "title": "Community-Rooted Curricula for Shared Futures (SYNTHETIC)",
        "text": (
            "Our alliance holds that education is best understood as a collective, "
            "community-held responsibility rather than an individual entitlement. "
            "Curricula should reflect the shared cultural heritage of the community, "
            "strengthening group identity and mutual obligation over individual "
            "achievement. Multiple cultural traditions coexisting within a single "
            "school should be actively celebrated and represented in daily practice."
        ),
    },
    {
        "organization": "Ministry of Education, Republic of Vestland (fictional state)",
        "organization_type": "national_ministry_fictional",
        "region": "Fictional Northern Region",
        "cultural_sphere": "State-centered/Fictional",
        "country": "Vestland (fictional)",
        "language": "en",
        "publication_year": 2018,
        "publication_type": "national_strategy",
        "title": "National Strategy for Standardized Assessment (SYNTHETIC)",
        "text": (
            "The Ministry establishes a single national standard of assessment "
            "applicable uniformly across all schools, prioritizing measurable, "
            "comparable outcomes. This strategy emphasizes the state's role as sole "
            "guarantor of educational quality, favoring centralized oversight over "
            "individual school autonomy or locally negotiated curricula."
        ),
    },
    {
        "organization": "Pacific Rim Indigenous Education Network (fictional)",
        "organization_type": "network_fictional",
        "region": "Fictional Pacific Region",
        "cultural_sphere": "Indigenous/Fictional",
        "country": None,
        "language": "en",
        "publication_year": 2022,
        "publication_type": "advocacy_statement",
        "title": "Reclaiming Pluralism in Curriculum Design (SYNTHETIC)",
        "text": (
            "Our network advocates for curricula that recognize and actively integrate "
            "the many distinct cultural traditions of the communities we serve. "
            "Multicultural representation, language preservation, and pluralistic "
            "values must sit at the center of any legitimate education policy, rather "
            "than being treated as optional additions to a universal template."
        ),
    },
)


class DemoSeedService(ResearchAccessMixin):
    async def seed_demo_corpus(
        self,
        *,
        project_id: str,
        user_id: str,
        corpus_name: str = "Policy Text Lab — Synthetic Demo Corpus",
    ) -> ResearchCorpus:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        rag_repo = RagRepository(self.db)

        corpus = await self.repo.create_corpus(
            project_id=project_id,
            name=corpus_name,
            description=(
                f"{SYNTHETIC_TAG}: fabricated fictional-organization policy texts used to "
                "demonstrate the Policy Text Lab workflow end to end. Not real "
                "research data — do not cite."
            ),
            created_by=user_id,
        )

        created_documents: list[CorpusDocument] = []
        for index, spec in enumerate(_SYNTHETIC_DOCUMENTS):
            rag_document = await rag_repo.create_document(
                user_id=user_id,
                filename=f"synthetic_demo_doc_{index}.txt",
                original_filename=f"{spec['title']}.txt",
                content_type="text/plain",
                storage_path=None,
                project_id=project_id,
                organization_id=None,
                source_type="synthetic_demo",
                metadata={"synthetic_demo": True, "seed_index": index},
            )
            await rag_repo.replace_chunks(
                rag_document,
                [
                    {
                        "chunk_index": 0,
                        "content": spec["text"],
                        "token_count": len(spec["text"].split()),
                    }
                ],
            )

            document = await self.repo.add_document(
                corpus_id=corpus.id,
                rag_document_id=rag_document.id,
                title=spec["title"],
                organization=spec["organization"],
                organization_type=spec["organization_type"],
                publication_year=spec["publication_year"],
                publication_type=spec["publication_type"],
                country=spec["country"],
                region=spec["region"],
                cultural_sphere=spec["cultural_sphere"],
                language=spec["language"],
                research_notes=(
                    f"{SYNTHETIC_TAG}: fabricated for demo purposes; does not represent a "
                    "real organization or document."
                ),
                metadata_json=dumps({"synthetic_demo": True}),
            )
            created_documents.append(document)

        await self.db.commit()
        return corpus
