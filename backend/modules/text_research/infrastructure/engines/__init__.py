"""Execution-engine implementations for text research analyses."""

from backend.modules.text_research.infrastructure.engines.python_engine import PythonAnalysisEngine
from backend.modules.text_research.infrastructure.engines.r_engine import RAnalysisEngine

__all__ = ["PythonAnalysisEngine", "RAnalysisEngine"]
