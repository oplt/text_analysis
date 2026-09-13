"""HTTP routes for the text research workflow.

This module is a thin aggregator (LATEST-017): domain-specific endpoints now
live in dedicated router modules, each included below without changing any
public URL:

- ``corpus_routes.py`` — demo seed, corpora, corpus documents, document
  cleaning profiles, segmentation, preprocessing profiles
- ``annotation_routes.py`` — codebooks & labels, annotation, reliability &
  adjudication
- ``quantitative_routes.py`` — quantitative/statistical-model/measurement
  endpoints (TASK-022)
- ``classification_routes.py`` — dictionaries, training datasets & classifiers
- ``topic_routes.py`` — topic models
- ``run_routes.py`` — robustness/comparative/dashboard + run lifecycle
- ``export_routes.py`` — exports
- ``contextual_routes.py`` — contextual / mixed-method datasets

Shared response serializers live in ``route_serializers.py``. A handful of
names are re-exported here for backward compatibility with existing imports
(``from backend.modules.text_research.api.routes import ...``) and tests that
patch attributes on this module.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from backend.api.deps.auth import get_current_user
from backend.modules.identity_access.models import User
from backend.modules.text_research.api import (
    annotation_routes as _annotation_routes,
)
from backend.modules.text_research.api import (
    classification_routes as _classification_routes,
)
from backend.modules.text_research.api import (
    contextual_routes as _contextual_routes,
)
from backend.modules.text_research.api import corpora as _corpora
from backend.modules.text_research.api import corpus_routes as _corpus_routes
from backend.modules.text_research.api import export_routes as _export_routes
from backend.modules.text_research.api import (
    quantitative_routes as _quantitative_routes,
)
from backend.modules.text_research.api import run_routes as _run_routes
from backend.modules.text_research.api import topic_routes as _topic_routes
from backend.modules.text_research.api.assistant_routes import router as assistant_router
from backend.modules.text_research.api.corpora import router as corpora_router
from backend.modules.text_research.api.memo_routes import router as memo_router
from backend.modules.text_research.api.route_serializers import (  # noqa: F401
    _cleaning_profile_response,
    _corpus_response,
    _dictionary_response,
    _document_response,
    _label_response,
    _loads,
    _model_response,
    _prediction_item_response,
    _prediction_set_response,
    _profile_response,
    _run_event_name,
    _run_response,
    _snapshot_response,
)
from backend.modules.text_research.application.active_learning_service import (  # noqa: F401
    ActiveLearningService,
)
from backend.modules.text_research.application.campaign_service import (  # noqa: F401
    AnnotationCampaignService,
)
from backend.modules.text_research.application.corpus_service import CorpusService  # noqa: F401
from backend.modules.text_research.application.export_service import ExportService  # noqa: F401
from backend.modules.text_research.application.run_service import RunService  # noqa: F401

router = APIRouter()
router.include_router(corpora_router)
router.include_router(assistant_router)
router.include_router(memo_router)
router.include_router(_corpus_routes.router)
router.include_router(_annotation_routes.router)

# TASK-022: quantitative analysis endpoints live on quantitative_router.
_quantitative_routes.bind_run_response(_run_response)
router.include_router(_quantitative_routes.router)
# Compatibility re-export for tests and callers that still import from routes.
_analysis_filters = _quantitative_routes._analysis_filters

router.include_router(_classification_routes.router)
router.include_router(_topic_routes.router)
router.include_router(_run_routes.router)
router.include_router(_export_routes.router)
router.include_router(_contextual_routes.router)


@router.get("/analysis-capabilities")
async def analysis_capabilities(
    current_user: User = Depends(get_current_user),
) -> dict[str, dict[str, Any]]:
    """Report async support from the operation execution registry + adapters."""
    from backend.modules.text_research.application.run_adapters import (
        RUN_ADAPTERS,
        describe_operation_execution_capabilities,
    )

    return {
        "operations": describe_operation_execution_capabilities(),
        "run_types": {
            run_type: {"async": adapter.supports_async}
            for run_type, adapter in RUN_ADAPTERS.items()
        },
    }


_TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}


# ------------------------------------------------------------------
# Backward-compatible re-exports.
#
# Domain endpoint handlers live in the modules included above; the names
# below let existing imports (``from ...api.routes import X``) and module
# attribute access (``routes.X``) keep working unchanged.
# ------------------------------------------------------------------

# corpus_routes
seed_demo_corpus = _corpus_routes.seed_demo_corpus
create_corpus = _corpus_routes.create_corpus
list_corpora = _corpus_routes.list_corpora
get_corpus = _corpus_routes.get_corpus
update_corpus = _corpus_routes.update_corpus
delete_corpus = _corpus_routes.delete_corpus
add_document = _corpus_routes.add_document
list_documents = _corpus_routes.list_documents
get_document = _corpus_routes.get_document
update_document = _corpus_routes.update_document
bulk_update_metadata = _corpus_routes.bulk_update_metadata
import_metadata_csv = _corpus_routes.import_metadata_csv
delete_document = _corpus_routes.delete_document
get_source_text = _corpus_routes.get_source_text
run_ingestion_qa = _corpus_routes.run_ingestion_qa
get_document_ingestion_qa = _corpus_routes.get_document_ingestion_qa
create_cleaning_profile = _corpus_routes.create_cleaning_profile
list_cleaning_profiles = _corpus_routes.list_cleaning_profiles
get_cleaning_profile = _corpus_routes.get_cleaning_profile
update_cleaning_profile = _corpus_routes.update_cleaning_profile
delete_cleaning_profile = _corpus_routes.delete_cleaning_profile
preview_cleaning_config = _corpus_routes.preview_cleaning_config
apply_document_cleaning = _corpus_routes.apply_document_cleaning
segment_corpus = _corpus_routes.segment_corpus
create_preprocessing_profile = _corpus_routes.create_preprocessing_profile
list_preprocessing_profiles = _corpus_routes.list_preprocessing_profiles
get_preprocessing_profile = _corpus_routes.get_preprocessing_profile
update_preprocessing_profile = _corpus_routes.update_preprocessing_profile
delete_preprocessing_profile = _corpus_routes.delete_preprocessing_profile
preview_preprocessing_config = _corpus_routes.preview_preprocessing_config

# annotation_routes
create_codebook = _annotation_routes.create_codebook
list_codebooks = _annotation_routes.list_codebooks
get_codebook = _annotation_routes.get_codebook
create_codebook_version = _annotation_routes.create_codebook_version
freeze_codebook = _annotation_routes.freeze_codebook
add_label = _annotation_routes.add_label
list_labels = _annotation_routes.list_labels
update_label = _annotation_routes.update_label
assign_annotation_tasks = _annotation_routes.assign_annotation_tasks
assign_corpus_annotation_tasks = _annotation_routes.assign_corpus_annotation_tasks
create_annotation_campaign = _annotation_routes.create_annotation_campaign
list_annotation_campaigns = _annotation_routes.list_annotation_campaigns
get_annotation_campaign = _annotation_routes.get_annotation_campaign
patch_annotation_campaign = _annotation_routes.patch_annotation_campaign
assign_annotation_campaign = _annotation_routes.assign_annotation_campaign
annotation_campaign_progress = _annotation_routes.annotation_campaign_progress
text_unit_blind_policy = _annotation_routes.text_unit_blind_policy
list_annotation_queue = _annotation_routes.list_annotation_queue
save_annotations = _annotation_routes.save_annotations
annotation_progress = _annotation_routes.annotation_progress
list_unit_annotations = _annotation_routes.list_unit_annotations
list_corpus_annotations_for_units = _annotation_routes.list_corpus_annotations_for_units
get_text_unit_context = _annotation_routes.get_text_unit_context
compute_reliability = _annotation_routes.compute_reliability
compute_campaign_reliability = _annotation_routes.compute_campaign_reliability
list_disagreements = _annotation_routes.list_disagreements
save_adjudication = _annotation_routes.save_adjudication
list_adjudications = _annotation_routes.list_adjudications
list_campaign_disagreements = _annotation_routes.list_campaign_disagreements
save_campaign_adjudication = _annotation_routes.save_campaign_adjudication
list_campaign_adjudications = _annotation_routes.list_campaign_adjudications

# classification_routes
create_dictionary = _classification_routes.create_dictionary
list_dictionaries = _classification_routes.list_dictionaries
get_dictionary = _classification_routes.get_dictionary
update_dictionary = _classification_routes.update_dictionary
create_dictionary_version = _classification_routes.create_dictionary_version
dataset_preview = _classification_routes.dataset_preview
freeze_dataset = _classification_routes.freeze_dataset
list_dataset_snapshots = _classification_routes.list_dataset_snapshots
get_dataset_snapshot = _classification_routes.get_dataset_snapshot
train_classifier = _classification_routes.train_classifier
list_classifiers = _classification_routes.list_classifiers
list_models = _classification_routes.list_models
update_model_lifecycle = _classification_routes.update_model_lifecycle
list_model_lifecycle_events = _classification_routes.list_model_lifecycle_events
get_classifier = _classification_routes.get_classifier
get_classifier_coefficients = _classification_routes.get_classifier_coefficients
clone_classifier_config = _classification_routes.clone_classifier_config
predict_classifier = _classification_routes.predict_classifier
compare_classifier_drift = _classification_routes.compare_classifier_drift
list_predictions = _classification_routes.list_predictions
get_prediction_set = _classification_routes.get_prediction_set
browse_prediction_set_predictions = _classification_routes.browse_prediction_set_predictions
list_prediction_sets = _classification_routes.list_prediction_sets
list_uncertain_predictions = _classification_routes.list_uncertain_predictions
assign_uncertain_predictions = _classification_routes.assign_uncertain_predictions

# topic_routes
train_topic_model = _topic_routes.train_topic_model
topic_k_sweep = _topic_routes.topic_k_sweep
topic_seed_stability = _topic_routes.topic_seed_stability
name_topic = _topic_routes.name_topic
list_topic_labels = _topic_routes.list_topic_labels

# run_routes
run_robustness_sweep = _run_routes.run_robustness_sweep
comparative_prevalence = _run_routes.comparative_prevalence
dashboard_summary = _run_routes.dashboard_summary
list_runs = _run_routes.list_runs
get_run = _run_routes.get_run
get_run_results_artifact = _run_routes.get_run_results_artifact
stream_run_events = _run_routes.stream_run_events
clone_run_parameters = _run_routes.clone_run_parameters
get_run_provenance = _run_routes.get_run_provenance
get_run_results = _run_routes.get_run_results
download_run_results = _run_routes.download_run_results
rerun = _run_routes.rerun
cancel_run = _run_routes.cancel_run
compare_runs = _run_routes.compare_runs

# export_routes
export_run_json = _export_routes.export_run_json
export_codebook_json = _export_routes.export_codebook_json
export_model_metrics = _export_routes.export_model_metrics
export_preprocessing_profile_json = _export_routes.export_preprocessing_profile_json
export_manifest = _export_routes.export_manifest
export_quanteda_script = _export_routes.export_quanteda_script
export_units_csv = _export_routes.export_units_csv
export_annotations_csv = _export_routes.export_annotations_csv
export_predictions_csv = _export_routes.export_predictions_csv

# contextual_routes
create_contextual_dataset = _contextual_routes.create_contextual_dataset
list_contextual_datasets = _contextual_routes.list_contextual_datasets
get_contextual_dataset = _contextual_routes.get_contextual_dataset
list_contextual_observations = _contextual_routes.list_contextual_observations
delete_contextual_dataset = _contextual_routes.delete_contextual_dataset
import_contextual_csv = _contextual_routes.import_contextual_csv
link_contextual_discourse = _contextual_routes.link_contextual_discourse

# corpora.py (metadata facets) — left included as-is (see docstring above).
corpus_metadata_facets = _corpora.corpus_metadata_facets
