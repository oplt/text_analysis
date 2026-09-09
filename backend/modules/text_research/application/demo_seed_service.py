"""SYNTHETIC demo-data seeding for the text research workspace.

Every document created here is **fabricated for demonstration purposes only**.
Content is deliberately topic-neutral so the platform does not prescribe a
research framework, discipline, or substantive label set.

* Organization / source names are explicitly fictional.
* `CorpusDocument.research_notes` and the corpus description are prefixed
  with "SYNTHETIC DEMO DATA".
* The backing `RagDocument.metadata_json` carries `synthetic_demo: true`.

This lets a new user explore corpus → segmentation → annotation →
classification without importing real documents first. Users still define
their own codebook labels and dictionaries.
"""

from __future__ import annotations

from typing import Any

from backend.modules.rag.infrastructure.repositories import RagRepository
from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.models import (
    CanonicalResearchSource,
    CorpusDocument,
    ResearchCorpus,
    dumps,
)
from backend.modules.text_research.infrastructure.canonical_text import build_canonical_from_full_text

SYNTHETIC_TAG = "SYNTHETIC DEMO DATA"

# Neutral synthetic passages — no theoretical constructs, policy frameworks,
# sentiment taxonomies, or geographic/cultural category systems.
_SYNTHETIC_DOCUMENTS: tuple[dict[str, Any], ...] = (
    {
        "organization": "Fictional Source Alpha",
        "organization_type": "publisher_fictional",
        "region": "Region A (fictional)",
        "cultural_sphere": None,
        "country": None,
        "language": "en",
        "publication_year": 2019,
        "publication_type": "article",
        "title": "Synthetic sample document A",
        "text": (
            "This fictional article discusses how teams coordinate shared work across "
            "time zones. Clear agendas, written summaries, and predictable handoffs "
            "reduce duplicated effort. The text is synthetic demo data only."
        ),
    },
    {
        "organization": "Fictional Source Beta",
        "organization_type": "archive_fictional",
        "region": "Region B (fictional)",
        "cultural_sphere": None,
        "country": None,
        "language": "en",
        "publication_year": 2021,
        "publication_type": "report",
        "title": "Synthetic sample document B",
        "text": (
            "This fictional report describes a laboratory notebook workflow. "
            "Observations are timestamped, materials are listed before results, and "
            "negative findings are retained. The text is synthetic demo data only."
        ),
    },
    {
        "organization": "Fictional Source Gamma",
        "organization_type": "institute_fictional",
        "region": "Region C (fictional)",
        "cultural_sphere": None,
        "country": "Fictionalia",
        "language": "en",
        "publication_year": 2018,
        "publication_type": "memo",
        "title": "Synthetic sample document C",
        "text": (
            "This fictional memo compares two scheduling options for a conference. "
            "Option one favors fewer parallel tracks; option two favors shorter talks. "
            "The text is synthetic demo data only."
        ),
    },
    {
        "organization": "Fictional Source Delta",
        "organization_type": "network_fictional",
        "region": "Region D (fictional)",
        "cultural_sphere": None,
        "country": None,
        "language": "en",
        "publication_year": 2022,
        "publication_type": "brief",
        "title": "Synthetic sample document D",
        "text": (
            "This fictional brief outlines a document-review checklist: verify citations, "
            "confirm figure captions match the body text, and archive the final PDF. "
            "The text is synthetic demo data only."
        ),
    },
)


class DemoSeedService(ResearchAccessMixin):
    async def seed_demo_corpus(
        self,
        *,
        project_id: str,
        user_id: str,
        corpus_name: str = "Synthetic demo corpus",
    ) -> ResearchCorpus:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        rag_repo = RagRepository(self.db)

        corpus = await self.repo.create_corpus(
            project_id=project_id,
            name=corpus_name,
            description=(
                f"{SYNTHETIC_TAG}: topic-neutral fabricated texts used to demonstrate "
                "the research workflow end to end. Not real research data — do not cite. "
                "Define your own codebook labels and dictionaries separately."
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
            build = build_canonical_from_full_text(
                spec["text"],
                language=spec["language"],
                source_file_reference=None,
                extra_transformation={
                    "source": "synthetic_demo_full_text",
                    "synthetic_demo": True,
                },
            )
            await self.repo.create_canonical_source(
                CanonicalResearchSource(
                    corpus_document_id=document.id,
                    canonical_text=build.text,
                    canonical_text_checksum=build.text_checksum,
                    raw_extracted_text=build.text,
                    raw_extracted_checksum=build.text_checksum,
                    original_file_checksum=build.original_file_checksum,
                    parser_name=build.parser_name,
                    parser_version=build.parser_version,
                    extracted_at=build.extracted_at,
                    source_rag_document_id=rag_document.id,
                    source_storage_path=None,
                    source_filename=f"{spec['title']}.txt",
                    language=build.language,
                    page_provenance_json=dumps(build.page_provenance),
                    transformation_metadata_json=dumps(
                        {
                            **(build.transformation_metadata or {}),
                            "raw_equals_canonical": True,
                            "cleaning_applied": False,
                        }
                    ),
                )
            )
            created_documents.append(document)

        await self.db.commit()
        return corpus
