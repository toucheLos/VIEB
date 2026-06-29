#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authoritative VIEB pipeline stage registry.

Pure data with no third-party imports so it can be loaded cheaply at startup
(unlike _utils.py, which eagerly imports matplotlib/cv2). Both _utils.py and
user_interface.py import STAGES / _STAGE_BY_ID from here so there is a single
source of truth for stage IDs, names, and descriptions.
"""

from __future__ import annotations

STAGES = [
    {
        "id": 0,
        "name": "Onboarding",
        "desc": "Choose a project, import a session-defining data source, and prepare metadata.",
        "cmd": "onboard project",
    },
    {
        "id": 1,
        "name": "Pose Estimation / DLC Analysis",
        "desc": "Run DeepLabCut analysis to generate pose CSV files for videos.",
        "cmd": "python setup_dlc_training.py --analyze",
    },
    {
        "id": 2,
        "name": "Feature Extraction",
        "desc": "Extract frame-level behavioral features from tracked keypoints.",
        "cmd": "python compare.py --extract [--no-wavelets]",
    },
    {
        "id": 3,
        "name": "Preprocessing · UMAP · Clustering · Smoothing",
        "desc": "Standardize features, reduce with UMAP, cluster with HDBSCAN, then smooth labels with HMM.",
        "cmd": "python compare.py --cluster --min-cluster-size N --umap-dims N [--hdbscan-min-samples N] [--validate]",
    },
    {
        "id": 4,
        "name": "State Collapsing (optional)",
        "desc": "Merge states whose centroids exceed a cosine similarity threshold in full feature space.",
        "cmd": "python compare.py --collapse --collapse-threshold 0.5",
    },
    {
        "id": 5,
        "name": "Report Generation",
        "desc": "Build summary tables, transition outputs, and group comparison plots.",
        "cmd": "python compare.py --report",
    },
    {
        "id": 6,
        "name": "Per-Animal Scalars",
        "desc": "Compute freeze AUC and discrimination metrics for each animal.",
        "cmd": "python compare.py --summarize",
    },
    {
        "id": 7,
        "name": "Motif Discovery",
        "desc": "Find enriched bigram/trigram motifs between contexts.",
        "cmd": "python compare.py --motifs",
    },
    {
        "id": 8,
        "name": "Generate Clips",
        "desc": "Export exemplar video clips for each behavioral state.",
        "cmd": "python generate_clips.py",
    },
    {
        "id": 9,
        "name": "Add Videos",
        "desc": "Add more videos to the active project after a first pass or when expanding the dataset.",
        "cmd": "open add videos",
    },
]

_STAGE_BY_ID: dict[int, dict] = {s["id"]: s for s in STAGES}
