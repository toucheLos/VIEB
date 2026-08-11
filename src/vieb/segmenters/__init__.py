"""Segmenters — the slot that says how a state space is cut up."""

from .base import Segmentation, Segmenter, make_segmentation, validate_labels

__all__ = ["Segmenter", "Segmentation", "make_segmentation", "validate_labels"]
