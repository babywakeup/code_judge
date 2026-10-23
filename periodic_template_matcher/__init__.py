"""Public API for template matching, label transfer and evaluation.

Start with MatcherConfig and PeriodicTemplateMatcher. Use run_config for a
file-based JSON workflow, and utils for image IO. _internal is not public API.
"""
from .config import MatcherConfig
from .matcher import PeriodicTemplateMatcher
from ._internal.results import EvaluationResult, MatchResult
from ._internal.workflow import run_config

__all__ = [
    'MatcherConfig', 'PeriodicTemplateMatcher', 'MatchResult', 'EvaluationResult',
    'run_config',
]