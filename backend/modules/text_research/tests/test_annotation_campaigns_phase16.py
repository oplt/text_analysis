"""Phase 16: annotation campaign CRUD, isolation, membership, completion."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.modules.text_research.application.campaign_service import AnnotationCampaignService
from backend.modules.text_research.domain.enums import AnnotationCampaignStatus


class AnnotationCampaignServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_campaign_persists_mode_codebook_version_unit_type_annotators(
        self,
    ) -> None:
        corpus = SimpleNamespace(id="corp-1", project_id="proj-1")
        codebook = SimpleNamespace(id="cb-1", project_id="proj-1", version="3.1")
        created: dict = {}

        async def create_annotation_campaign(**kwargs):
            created.update(kwargs)
            return SimpleNamespace(id="camp-1", **kwargs, created_at=datetime.now(UTC))

        service = AnnotationCampaignService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        service.get_codebook_or_404 = AsyncMock(return_value=codebook)
        service.repo = MagicMock()
        service.repo.create_annotation_campaign = AsyncMock(side_effect=create_annotation_campaign)
        service.db = MagicMock()
        service.db.flush = AsyncMock()

        campaign = await service.create_campaign(
            user_id="user-1",
            project_id="proj-1",
            corpus_id="corp-1",
            name="Wave A",
            codebook_id="cb-1",
            unit_type="sentence",
            assignment_strategy="overlap",
            annotator_ids=["a1", "a2"],
            annotation_mode="blind_reliability",
        )

        self.assertEqual(campaign.id, "camp-1")
        self.assertEqual(created["codebook_version"], "3.1")
        self.assertEqual(created["unit_type"], "sentence")
        self.assertEqual(created["annotator_ids"], ["a1", "a2"])
        self.assertTrue(created["blind_mode"])
        self.assertFalse(created["ai_assistance_enabled"])
        self.assertEqual(created["annotation_mode"], "blind_reliability")
        self.assertEqual(created["status"], AnnotationCampaignStatus.ACTIVE.value)

    async def test_assign_passes_campaign_unit_type_and_membership(self) -> None:
        campaign = SimpleNamespace(
            id="camp-1",
            corpus_id="corp-1",
            project_id="proj-1",
            unit_type="paragraph",
            assignment_strategy="overlap",
            sample_size=40,
            overlap_count=2,
            overlap_percent=None,
            annotator_ids_json='["a1","a2"]',
            annotation_mode="blind_reliability",
            blind_mode=True,
        )
        service = AnnotationCampaignService(MagicMock())
        service.get_campaign = AsyncMock(return_value=campaign)
        service.db = MagicMock()
        service.db.commit = AsyncMock()

        with patch(
            "backend.modules.text_research.application.campaign_service.AnnotationService"
        ) as annotation_cls:
            annotation_cls.return_value.assign_corpus_tasks = AsyncMock(
                return_value={"created_tasks": 4, "annotator_ids": ["a1", "a2"]}
            )
            result = await service.assign("camp-1", user_id="owner", annotator_ids=["a1", "a2"])

        kwargs = annotation_cls.return_value.assign_corpus_tasks.await_args.kwargs
        self.assertEqual(kwargs["unit_type"], "paragraph")
        self.assertEqual(kwargs["campaign_id"], "camp-1")
        self.assertEqual(kwargs["annotator_ids"], ["a1", "a2"])
        self.assertEqual(result["campaign_id"], "camp-1")
        self.assertTrue(result["blind_mode"])

    async def test_assign_rejects_empty_non_member_override_by_requiring_targets(self) -> None:
        """When campaign has members, assign uses them unless override provided."""
        campaign = SimpleNamespace(
            id="camp-1",
            corpus_id="corp-1",
            project_id="proj-1",
            unit_type="paragraph",
            assignment_strategy="shared",
            sample_size=10,
            overlap_count=None,
            overlap_percent=None,
            annotator_ids_json='["member-1"]',
            annotation_mode="blind_reliability",
            blind_mode=True,
        )
        service = AnnotationCampaignService(MagicMock())
        service.get_campaign = AsyncMock(return_value=campaign)
        service.db = MagicMock()
        service.db.commit = AsyncMock()

        with patch(
            "backend.modules.text_research.application.campaign_service.AnnotationService"
        ) as annotation_cls:
            annotation_cls.return_value.assign_corpus_tasks = AsyncMock(
                return_value={"created_tasks": 1}
            )
            await service.assign("camp-1", user_id="owner")

        kwargs = annotation_cls.return_value.assign_corpus_tasks.await_args.kwargs
        self.assertEqual(kwargs["annotator_ids"], ["member-1"])

    async def test_patch_campaign_completed_sets_completed_at(self) -> None:
        campaign = SimpleNamespace(
            id="camp-1",
            project_id="proj-1",
            name="Wave A",
            description=None,
            status=AnnotationCampaignStatus.ACTIVE.value,
            completed_at=None,
            annotation_mode="blind_reliability",
            blind_mode=True,
            ai_assistance_enabled=False,
            metadata_json="{}",
        )
        service = AnnotationCampaignService(MagicMock())
        service.get_campaign = AsyncMock(return_value=campaign)
        service.db = MagicMock()
        service.db.commit = AsyncMock()
        service.db.refresh = AsyncMock()

        updated = await service.patch_campaign("camp-1", user_id="user-1", status="completed")
        self.assertEqual(updated.status, AnnotationCampaignStatus.COMPLETED.value)
        self.assertIsNotNone(updated.completed_at)

    async def test_create_rejects_codebook_from_other_project(self) -> None:
        corpus = SimpleNamespace(id="corp-1", project_id="proj-1")
        codebook = SimpleNamespace(id="cb-1", project_id="proj-OTHER", version="1")
        service = AnnotationCampaignService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        service.get_codebook_or_404 = AsyncMock(return_value=codebook)

        with self.assertRaises(HTTPException) as ctx:
            await service.create_campaign(
                user_id="user-1",
                project_id="proj-1",
                corpus_id="corp-1",
                name="Bad",
                codebook_id="cb-1",
            )
        self.assertEqual(ctx.exception.status_code, 400)


class UncertainQueueBlindGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_route_returns_empty_when_campaign_blind(self) -> None:
        from backend.modules.text_research.api import routes

        campaign = SimpleNamespace(id="camp-1", blind_mode=True)
        db = MagicMock()
        user = SimpleNamespace(id="user-1")

        with (
            patch.object(
                routes.AnnotationCampaignService,
                "get_campaign",
                new=AsyncMock(return_value=campaign),
            ),
            patch.object(
                routes.ActiveLearningService,
                "uncertain_queue",
                new=AsyncMock(return_value=([{"should": "not appear"}], 1)),
            ) as queue_mock,
        ):
            result = await routes.list_uncertain_predictions(
                model_id="model-1",
                campaign_id="camp-1",
                text_unit_id=None,
                limit=20,
                offset=0,
                content_mode="snippet",
                db=db,
                current_user=user,
            )
        self.assertEqual(result, {"items": [], "total": 0, "limit": 20, "offset": 0})
        queue_mock.assert_not_called()

    async def test_route_returns_empty_when_unit_policy_hides_predictions(self) -> None:
        from backend.modules.text_research.api import routes

        db = MagicMock()
        user = SimpleNamespace(id="user-1")

        with (
            patch.object(
                routes.AnnotationCampaignService,
                "blind_policy_for_annotator_unit",
                new=AsyncMock(return_value={"hide_model_predictions": True}),
            ),
            patch.object(
                routes.ActiveLearningService,
                "uncertain_queue",
                new=AsyncMock(return_value=([{"leak": True}], 1)),
            ) as queue_mock,
        ):
            result = await routes.list_uncertain_predictions(
                model_id="model-1",
                campaign_id=None,
                text_unit_id="unit-1",
                limit=20,
                offset=0,
                content_mode="snippet",
                db=db,
                current_user=user,
            )
        self.assertEqual(result, {"items": [], "total": 0, "limit": 20, "offset": 0})
        queue_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
