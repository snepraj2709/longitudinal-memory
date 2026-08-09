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

__all__ = [
    "CandidateConfig",
    "CandidatePair",
    "CandidateRequest",
    "CandidateSignals",
    "ConflictCandidateError",
    "ConflictCandidateService",
    "generate_candidate_pairs",
    "load_candidate_config",
]
