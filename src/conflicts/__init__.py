"""Deterministic conflict-candidate generation."""

from .candidates import (
    CandidateConfig,
    CandidatePair,
    CandidateRequest,
    CandidateSignals,
    ConflictCandidateError,
    ConflictCandidateService,
    generate_candidate_pairs,
    load_candidate_config,
)
from .classifier import (
    CheckedRelation,
    ClassificationRequest,
    ConflictClassifier,
    ConflictClassifierError,
    ConflictDecision,
    classify_conflict,
    load_classifier_config,
)
from .relations import (
    ConflictRelationConflict,
    ConflictRelationError,
    ConflictRelationService,
    PersistedConflict,
)

__all__ = [
    "CandidateConfig",
    "CandidatePair",
    "CandidateRequest",
    "CandidateSignals",
    "CheckedRelation",
    "ClassificationRequest",
    "ConflictCandidateError",
    "ConflictCandidateService",
    "ConflictClassifier",
    "ConflictClassifierError",
    "ConflictDecision",
    "ConflictRelationConflict",
    "ConflictRelationError",
    "ConflictRelationService",
    "PersistedConflict",
    "classify_conflict",
    "generate_candidate_pairs",
    "load_candidate_config",
    "load_classifier_config",
]
