"""Unit tests for annotation campaigns and blind reliability coding."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.modules.text_research.application.campaign_service import (
    AnnotationCampaignService,
    resolve_annotation_mode,
)
from backend.modules.text_research.domain.enums import (
    AnnotationCampaignStatus,
    AnnotationMode,
    AnnotationTaskStatus,
)
from backend.modules.text_research.domain.models import AnnotationCampaign, AnnotationTask


class ResolveAnnotationModeTests(unittest.TestCase):
    def test_blind_reliability_default(self) -> None:
        mode, blind, ai = resolve_annotation_mode()
        self.assertEqual(mode, AnnotationMode.BLIND_RELIABILITY.value)
        self.assertTrue(blind)
        self.assertFalse(ai)

    def test_ai_assisted_mode(self) -> None:
        mode, blind, ai = resolve_annotation_mode(annotation_mode="ai_assisted")
        self.assertEqual(mode, AnnotationMode.AI_ASSISTED.value)
        self.assertFalse(blind)
        self.assertTrue(ai)

    def test_blind_flag_false_maps_to_ai(self) -> None:
        mode, blind, ai = resolve_annotation_mode(blind_mode=False)
        self.assertEqual(mode, AnnotationMode.AI_ASSISTED.value)
        self.assertFalse(blind)
        self.assertTrue(ai)

    def test_mutual_exclusion(self) -> None:
        _, blind, ai = resolve_annotation_mode(annotation_mode="blind_reliability")
        self.assertTrue(blind)
        self.assertFalse(ai)


class AnnotationCampaignModelTests(unittest.TestCase):
    def test_campaign_table_and_task_fk(self) -> None:
        campaign_cols = {column.key for column in AnnotationCampaign.__table__.columns}
        self.assertIn("blind_mode", campaign_cols)
        self.assertIn("ai_assistance_enabled", campaign_cols)
        self.assertIn("annotation_mode", campaign_cols)
        self.assertIn("annotator_ids_json", campaign_cols)
        task_cols = {column.key for column in AnnotationTask.__table__.columns}
        self.assertIn("campaign_id", task_cols)
        self.assertEqual(AnnotationCampaignStatus.ACTIVE.value, "active")


class BlindPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_incomplete_blind_task_hides_peers_and_predictions(self) -> None:
        campaign = SimpleNamespace(
            id="camp-1",
            blind_mode=True,
            ai_assistance_enabled=False,
            annotation_mode="blind_reliability",
        )
        task = SimpleNamespace(
            campaign_id="camp-1",
            status=AnnotationTaskStatus.ASSIGNED.value,
        )
        service = AnnotationCampaignService(MagicMock())
        service.repo = MagicMock()
        service.repo.get_task = AsyncMock(return_value=task)
        service.repo.get_annotation_campaign = AsyncMock(return_value=campaign)

        policy = await service.blind_policy_for_annotator_unit(
            text_unit_id="unit-1", annotator_id="user-1"
        )
        self.assertTrue(policy["blind_mode"])
        self.assertTrue(policy["hide_peer_annotations"])
        self.assertTrue(policy["hide_model_predictions"])
        self.assertTrue(policy["hide_adjudications"])
        self.assertFalse(policy["ai_assistance_enabled"])

    async def test_completed_blind_task_allows_peer_review(self) -> None:
        campaign = SimpleNamespace(
            id="camp-1",
            blind_mode=True,
            ai_assistance_enabled=False,
            annotation_mode="blind_reliability",
        )
        task = SimpleNamespace(
            campaign_id="camp-1",
            status=AnnotationTaskStatus.COMPLETED.value,
        )
        service = AnnotationCampaignService(MagicMock())
        service.repo = MagicMock()
        service.repo.get_task = AsyncMock(return_value=task)
        service.repo.get_annotation_campaign = AsyncMock(return_value=campaign)

        policy = await service.blind_policy_for_annotator_unit(
            text_unit_id="unit-1", annotator_id="user-1"
        )
        self.assertTrue(policy["blind_mode"])
        self.assertFalse(policy["hide_peer_annotations"])
        # Model suggestions stay off for the whole blind campaign.
        self.assertTrue(policy["hide_model_predictions"])

    async def test_ai_assisted_allows_predictions(self) -> None:
        campaign = SimpleNamespace(
            id="camp-2",
            blind_mode=False,
            ai_assistance_enabled=True,
            annotation_mode="ai_assisted",
        )
        task = SimpleNamespace(
            campaign_id="camp-2",
            status=AnnotationTaskStatus.ASSIGNED.value,
        )
        service = AnnotationCampaignService(MagicMock())
        service.repo = MagicMock()
        service.repo.get_task = AsyncMock(return_value=task)
        service.repo.get_annotation_campaign = AsyncMock(return_value=campaign)

        policy = await service.blind_policy_for_annotator_unit(
            text_unit_id="unit-1", annotator_id="user-1"
        )
        self.assertFalse(policy["hide_model_predictions"])
        self.assertTrue(policy["ai_assistance_enabled"])

    async def test_no_campaign_defaults_to_open(self) -> None:
        service = AnnotationCampaignService(MagicMock())
        service.repo = MagicMock()
        service.repo.get_task = AsyncMock(return_value=None)

        policy = await service.blind_policy_for_annotator_unit(
            text_unit_id="unit-1", annotator_id="user-1"
        )
        self.assertFalse(policy["blind_mode"])
        self.assertFalse(policy["hide_peer_annotations"])
        self.assertFalse(policy["hide_model_predictions"])


class ListAnnotationsBlindFilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_filters_peer_annotations_when_blind(self) -> None:
        from unittest.mock import patch

        from backend.modules.text_research.application.annotation_service import AnnotationService

        own = SimpleNamespace(annotator_id="user-1", value="yes")
        peer = SimpleNamespace(annotator_id="user-2", value="no")
        service = AnnotationService(MagicMock())
        service.get_text_unit_or_404 = AsyncMock(return_value=(MagicMock(), MagicMock()))
        service.repo = MagicMock()
        service.repo.list_annotations_for_unit = AsyncMock(return_value=[own, peer])

        with patch(
            "backend.modules.text_research.application.campaign_service.AnnotationCampaignService"
        ) as campaign_cls:
            campaign_cls.return_value.blind_policy_for_annotator_unit = AsyncMock(
                return_value={"hide_peer_annotations": True}
            )
            result = await service.list_annotations_for_unit("unit-1", user_id="user-1")
        self.assertEqual(result, [own])


class QueueBlindPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_hides_peers_predictions_and_adjudications_while_incomplete(self) -> None:
        from backend.modules.text_research.application.annotation_service import AnnotationService
        from backend.modules.text_research.domain.enums import AnnotationTaskStatus

        task = SimpleNamespace(
            id="t1",
            campaign_id="camp-1",
            text_unit_id="unit-1",
            annotator_id="user-1",
            status=AnnotationTaskStatus.ASSIGNED.value,
            assigned_at=None,
            completed_at=None,
        )
        unit = SimpleNamespace(
            id="unit-1",
            corpus_document_id="doc-1",
            unit_type="paragraph",
            position=0,
            page_number=None,
            paragraph_number=1,
            sentence_number=None,
            char_start=None,
            char_end=None,
            section_heading=None,
            text="hello",
            text_hash="h",
            source_text_hash="s",
            created_at=None,
        )
        campaign = SimpleNamespace(
            id="camp-1",
            name="Blind wave",
            annotation_mode="blind_reliability",
            blind_mode=True,
            ai_assistance_enabled=False,
        )
        service = AnnotationService(MagicMock())
        service.repo = MagicMock()
        service.repo.paginate_tasks_for_annotator = AsyncMock(return_value=([task], 1))
        service.repo.list_text_units_by_ids = AsyncMock(return_value=[unit])
        service.repo.list_campaigns_for_tasks = AsyncMock(return_value={"camp-1": campaign})

        items, total = await service.list_queue(requesting_user_id="user-1")
        self.assertEqual(total, 1)
        policy = items[0]["blind_policy"]
        self.assertTrue(policy["hide_model_predictions"])
        self.assertTrue(policy["hide_peer_annotations"])
        self.assertTrue(policy["hide_adjudications"])
        self.assertFalse(policy["ai_assistance_enabled"])


if __name__ == "__main__":
    unittest.main()
