"""Representations — the slot that says what the state space is."""

from .base import BaseRepresentation, Representation
from .delay_embed import delay_embed, embedded_length, scatter_labels

__all__ = [
    "Representation",
    "BaseRepresentation",
    "delay_embed",
    "embedded_length",
    "scatter_labels",
]
