"""Quantitative text-analysis workflows (LATEST-017 facade).

Domain logic lives in:
- ``quantitative_support`` (selection / persistence / worker entry)
- ``lexical_analysis_service``
- ``matrix_analysis_service``
- ``similarity_analysis_service``
- ``exploratory_analysis_service``

Call sites continue to use ``QuantitativeAnalysisService`` method names.
"""

from __future__ import annotations

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.exploratory_analysis_service import (
    ExploratoryAnalysisMixin,
)
from backend.modules.text_research.application.lexical_analysis_service import (
    LexicalAnalysisMixin,
)
from backend.modules.text_research.application.matrix_analysis_service import (
    MatrixAnalysisMixin,
)
from backend.modules.text_research.application.quantitative_support import (
    QuantitativeSupportMixin,
)
from backend.modules.text_research.application.similarity_analysis_service import (
    SimilarityAnalysisMixin,
)


class QuantitativeAnalysisService(
    LexicalAnalysisMixin,
    MatrixAnalysisMixin,
    SimilarityAnalysisMixin,
    ExploratoryAnalysisMixin,
    QuantitativeSupportMixin,
    ResearchAccessMixin,
):
    """Facade retaining the historical service surface for workers and adapters."""


# Re-export helpers historically imported from this module.
# Compatibility re-exports for older tests that patch this module.
from backend.modules.text_research.application.analysis_executor import (  # noqa: E402,F401
    estimate_workload,
    execute_or_enqueue,
    run_cpu_bound,
    should_enqueue_cpu_job,
)
from backend.modules.text_research.application.quantitative_support import (  # noqa: E402,F401
    QUANT_OP_KEY,
    _apply_document_filters,
    _filter_kwargs,
    _prepare_with_identity,
    _with_frozen_preprocessing,
)
from backend.modules.text_research.infrastructure import quantitative  # noqa: E402,F401
