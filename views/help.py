from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

_SECTIONS = [
    ("what_is_vieb",           "What is VIEB?"),
    ("how_vieb_works",         "How VIEB Works — The Big Picture"),
    ("stage_1_dlc",            "Stage 1 — DLC Setup (Pose Estimation)"),
    ("stage_2_features",       "Stage 2 — Feature Extraction"),
    ("stage_3_clustering",     "Stage 3 — Preprocessing, UMAP, Clustering & Smoothing"),
    ("stage_4_collapse",       "Stage 4 — State Collapse (optional)"),
    ("stage_5_comparison",     "Stage 5 — Comparison Report"),
    ("stage_6_clips",          "Stage 6 — Generate Clips"),
    ("stage_7_quantification", "Stage 7 — Quantification"),
    ("diagnose",               "Diagnose Clustering Parameters"),
    ("split_dominant",         "Split Dominant State"),
    ("clip_reviewer",          "Clip Reviewer"),
    ("settings",               "Settings"),
]

_CONTENT: dict[str, str] = {

"what_is_vieb": """
<p>VIEB (Video Interpreter for Experimental Behavior) is an unsupervised machine learning pipeline
for discovering behavioral states in animal behavior videos. Unlike traditional behavioral analysis
tools that require you to define behaviors before measuring them — drawing zones, setting thresholds,
or labeling example clips — VIEB finds behavioral structure in your data automatically, without any
prior assumptions about what behaviors exist.</p>

<p>The core question VIEB answers is: <b>what is this animal actually doing, and how does that change
across experimental conditions?</b></p>

<p>This matters because behavior is not a single measurement. An animal's response to fear, learning,
disease, or drug treatment reorganizes its entire behavioral repertoire in ways that no single metric
captures. VIEB characterizes that reorganization comprehensively, from raw video to publication-ready
quantitative outputs.</p>

<p>VIEB was developed at Temple University in collaboration with the Luna Lab for studying behavioral
phenotypes in Alzheimer's mouse models, and is designed to work with any rodent behavioral experiment
where DeepLabCut pose tracking output is available.</p>
""",

"how_vieb_works": """
<p>VIEB operates in two tiers:</p>

<p><b>Tier 1 — Pose tracking (DeepLabCut):</b> A deep learning model tracks the positions of defined
body keypoints (nose, ears, hips, tail, etc.) in every frame of every video. This produces a CSV file
per video containing x/y coordinates and confidence scores for each keypoint at each frame. This step
requires GPU acceleration and manual labeling of example frames to train the model. If you already have
a trained DLC model and pose CSVs, you can skip this tier entirely.</p>

<p><b>Tier 2 — Behavioral ML pipeline:</b> VIEB takes the pose CSVs and runs a fully automated
unsupervised machine learning pipeline to discover behavioral states. No human labeling required.
The pipeline consists of six stages:</p>

<ul>
<li><b>Stage 1 — DLC Setup:</b> Trains and runs DeepLabCut to generate pose CSV files.</li>
<li><b>Stage 2 — Feature Extraction:</b> Converts pose coordinates into 51–91 kinematic features per frame.</li>
<li><b>Stages 3–6 — Preprocessing, UMAP, Clustering, Smoothing:</b> Discovers behavioral states across all videos.</li>
<li><b>Stage 7 — State Collapse (optional):</b> Merges highly similar states.</li>
<li><b>Stage 8 — Comparison Report:</b> Generates state occupancy plots and tables.</li>
<li><b>Stage 11 — Generate Clips:</b> Exports exemplar video clips per state.</li>
</ul>
""",

"stage_1_dlc": """
<p><b>What it does:</b> Trains a DeepLabCut neural network to track body keypoints in your videos,
then runs pose estimation to generate one CSV per video.</p>

<p><b>When to use:</b> Only if you don't already have DLC pose tracking output. If your lab has
already run DLC on your videos, skip this stage and point VIEB to your existing CSV files in Settings.</p>

<p><b>What you need:</b> Raw .mp4 video files and enough patience to manually label body keypoints
in 20–50 example frames. VIEB guides you through this process step by step.</p>

<p><b>Output:</b> One CSV file per video alongside each .mp4, containing frame-by-frame x/y coordinates
and confidence scores for each tracked keypoint.</p>

<p><b>Time:</b> Labeling takes 2–4 hours. Training takes several hours to overnight depending on
GPU availability.</p>
""",

"stage_2_features": """
<p><b>What it does:</b> Converts raw keypoint coordinates into a rich set of behavioral features
that capture how the animal is moving and what posture it is holding at each frame.</p>

<p>For each video frame, VIEB computes up to 91 features including:</p>

<ul>
<li><b>Speed per keypoint</b> — how fast each body part is moving</li>
<li><b>Pairwise distances</b> — distances between all pairs of keypoints, normalized to body length
so results are size-independent</li>
<li><b>Centroid speed</b> — overall movement speed of the animal's center of mass</li>
<li><b>Body orientation</b> — the angle the animal's body axis makes with the horizontal</li>
<li><b>Elongation</b> — how stretched out vs. compact the animal's body is</li>
<li><b>Angular velocity</b> — how fast the animal is turning</li>
<li><b>Rearing score</b> — a measure of whether the animal is standing on its hind legs</li>
<li><b>Head angle</b> — the angle of the animal's head relative to its body</li>
<li><b>Temporal window statistics</b> — mean, variance, and peak values of movement features over
a 500 ms sliding window, capturing short-term behavioral dynamics</li>
<li><b>Morlet wavelet amplitudes</b> — frequency-domain features that capture the rhythmic structure
of movement, useful for detecting repetitive behaviors like grooming or locomotion cycles</li>
</ul>

<p><b>Why this matters:</b> Raw keypoint coordinates are not directly comparable across animals of
different sizes or across sessions where the animal starts in different positions. By computing
normalized, physics-based features, VIEB ensures that behavioral similarity is based on <i>how</i>
the animal is moving, not <i>where</i> it happens to be in the frame.</p>

<p><b>Output:</b> One .npy feature array per video in results/features/, plus an index.json tracking
all extracted videos. Feature extraction only runs on videos not already in the index, so re-running
is fast.</p>

<p><b>Parameters:</b><br>
<b>Wavelets on/off</b> — including Morlet wavelet features increases the feature vector from 51 to
91 dimensions and improves detection of rhythmic behaviors at the cost of longer extraction time.
Recommended on.</p>

<p><b>Time:</b> 5–15 minutes for 222 videos on a modern CPU.</p>
""",

"stage_3_clustering": """
<p><b>What it does:</b> This is the core discovery stage. VIEB pools all frames from all videos
into a single dataset (for 222 videos this is over 1.2 million frames), finds the underlying structure
of behavioral space, and assigns every frame to a behavioral state.</p>

<p>The stage runs four steps in sequence:</p>

<p><b>1. Standardization:</b> Each of the 91 features is scaled to have zero mean and unit variance
across all frames. This prevents features with large numerical values from dominating the analysis
over features with small values.</p>

<p><b>2. UMAP (Uniform Manifold Approximation and Projection):</b> UMAP compresses the 91-dimensional
feature space down to a lower-dimensional space (3–10 dimensions by default) while preserving the
neighborhood structure — frames that are behaviorally similar in 91 dimensions end up close together
in the reduced space. UMAP is fit on a random sample of 200,000 frames for speed, then applied to
all frames.<br><br>
<i>Why UMAP instead of PCA:</i> PCA finds linear structure. Behavioral space is highly nonlinear —
the difference between freezing and locomotion is not captured by a straight line through feature
space. UMAP finds the manifold structure of behavior, revealing clusters and transitions that PCA
would miss.</p>

<p><b>3. HDBSCAN:</b> HDBSCAN finds clusters of frames in the UMAP-reduced space. Unlike K-means,
HDBSCAN does not require you to specify the number of clusters in advance — it finds however many
dense regions exist in the data. Frames that don't belong to any dense region are labeled as noise
(−1) rather than forced into the nearest cluster. Each frame receives both a cluster label and a soft
probability (0–1) indicating confidence of cluster membership.</p>

<p><b>4. HMM Smoothing:</b> Raw HDBSCAN labels can flicker frame to frame. A Hidden Markov Model
(Viterbi algorithm) smooths the label sequence within each video, enforcing temporal continuity while
respecting the transition structure of the data.</p>

<p><b>Parameters:</b></p>

<p><b>min_cluster_size</b> — The minimum number of frames required for a group of frames to be
recognized as a behavioral state. This is the most important parameter in VIEB.<br>
• Too low (&lt; 500): produces many small noisy clusters, hard to interpret, high noise fraction<br>
• Too high (&gt; 5000): merges distinct behaviors into one cluster, loses fine-grained structure<br>
• Recommended starting point: 2000–3000 for datasets of 1M+ frames<br>
• Rule of thumb: aim for 15–25 states with less than 30% noise frames</p>

<p><b>min_samples (0 = auto)</b> — Controls the conservatism of cluster borders. 0 (recommended)
automatically uses the same value as min_cluster_size. Only change this if you have a specific reason.</p>

<p><b>UMAP dims</b> — The number of dimensions UMAP compresses the feature space to before clustering.<br>
• 3 dims: fastest, finds broad coarse behavioral categories<br>
• 10 dims (default): balanced, finds moderate detail<br>
• 15+ dims: slowest, preserves the most fine-grained behavioral structure</p>

<p><b>80/20 validation split</b> — When checked, VIEB splits your videos into a training set (80%)
and a test set (20%). A high generalization score (&gt; 0.7) means the discovered states are robust
and reproducible. Recommended when preparing results for publication.</p>

<p><b>Time:</b> 10–60 minutes depending on dataset size, GPU availability, and UMAP dimensions.
GPU acceleration via cuML reduces this to 2–10 minutes.</p>
""",

"stage_4_collapse": """
<p><b>What it does:</b> Merges highly similar behavioral states discovered by HDBSCAN. When two
states have very similar kinematic profiles and frequently occur in the same contexts, they may
represent the same underlying behavior split artificially by the clustering algorithm. Collapse
merges them into a single state.</p>

<p><b>When to use:</b> After clustering produces more states than you can meaningfully interpret
(e.g. 40+ states for a simple experiment). Start without collapse and only apply it if the number
of states is unmanageable.</p>

<p><b>Parameters:</b><br>
<b>Collapse threshold</b> — States with a cosine similarity above this threshold in feature space
are merged. 0.5 is a moderate threshold. Higher values = only very similar states are merged.
Lower values = more aggressive merging.</p>

<p><b>Warning:</b> Collapse is irreversible without re-clustering. Save your cluster run before
applying collapse.</p>
""",

"stage_5_comparison": """
<p><b>What it does:</b> Computes state occupancy fractions for every video and generates a
comprehensive set of comparison plots showing how behavioral state usage varies across experimental
conditions.</p>

<p><b>Output files:</b></p>

<ul>
<li><b>summary_table.csv</b> — One row per video with state occupancy fractions and all metadata
columns. The primary data table for downstream analysis.</li>
<li><b>transition_table.csv</b> — Summary table plus flattened transition matrices showing how often
each state transitions to each other state.</li>
<li><b>state_by_day.png</b> — How state occupancy changes across training days</li>
<li><b>state_by_context.png</b> — How state occupancy differs across experimental conditions</li>
<li><b>state_by_experiment.png</b> — Comparison across experiment types</li>
<li><b>state_by_animal.png</b> — Per-animal state occupancy profiles</li>
<li><b>animal_trajectories.png</b> — Per-animal state occupancy across days, showing individual
learning trajectories</li>
<li><b>transition_by_context.png</b> — Side-by-side transition heatmaps per condition</li>
</ul>

<p><b>Time:</b> 1–2 minutes.</p>
""",

"stage_6_clips": """
<p><b>What it does:</b> Exports short video clips illustrating each behavioral state. Two types of
clips are generated per state:</p>

<ul>
<li><b>Longest bouts</b> — the longest continuous stretches of that state, showing what the behavior
looks like when it is sustained</li>
<li><b>Typical clips</b> — bouts whose kinematics are closest to the cluster centroid, showing the
most representative examples of that state</li>
</ul>

<p><b>Parameters:</b><br>
<b>n-clips</b> — Number of clips to generate per state (default 15)<br>
<b>clip-purity</b> — Minimum HDBSCAN confidence score for frames included in clips. 0.95 means only
clips where 95% of frames have high confidence assignments. Lower values include more clips but with
less certain state assignments.</p>

<p><b>Output:</b> clips/state_{id}/ directory with .mp4 files for each state.</p>

<p><b>Time:</b> 5–20 minutes depending on number of states and video length.</p>
""",

"stage_7_quantification": """
<p><b>What it does:</b> Computes per-animal behavioral scalars that summarize each animal's
behavioral profile across all sessions. These scalars are the primary output for statistical
analysis and correlation with biological measurements.</p>

<p><b>Key outputs:</b></p>

<p><b>Contrast vector</b> — For each animal, the difference in state occupancy between condition A
sessions and condition B sessions. The contrast vector captures the full behavioral reorganization
between conditions as a single multi-dimensional object.</p>

<p><b>Contrast magnitude</b> — The L2 norm of the contrast vector, normalized to the range 0–1. A
higher contrast magnitude means the animal shows stronger behavioral differentiation between
conditions. This is the primary scalar for correlating behavioral data with molecular measurements.</p>

<p><b>Discrimination ratio</b> — For each state, (occupancy_A − occupancy_B) / (occupancy_A +
occupancy_B). Ranges from −1 to +1. Positive = state is enriched in condition A. Negative = state
is enriched in condition B.</p>

<p><b>Condition AUC</b> — Area under the condition A occupancy curve across training days. Captures
the cumulative expression of condition-specific behavior over the course of learning.</p>

<p><b>Learning rate</b> — The slope of contrast magnitude across days, computed by linear regression.
Positive slope = the animal is increasingly differentiating its behavior between conditions over time,
indicating learning.</p>

<p><b>Behavioral diversity</b> — Shannon entropy of state occupancy, normalized to 0–1. High
diversity = animal uses many states equally. Low diversity = animal spends most time in one or two
states.</p>

<p><b>Transition entropy</b> — The average entropy of each row of the transition matrix. High
transition entropy = the animal transitions unpredictably between states. Low = the animal follows
stereotyped behavioral sequences.</p>

<p><b>Output:</b> master_table.csv with one row per animal and all scalar columns.</p>

<p><b>Time:</b> 1–2 minutes.</p>
""",

"diagnose": """
<p>Sweeps a range of min_cluster_size values and recommends the best setting before you commit to
a full cluster run. For each value tested it reports number of states, noise fraction, dominant
state size, and silhouette score. Run this first if you're unsure what parameters to use.</p>

<p>Good results look like: <b>15–25 states</b>, <b>under 30% noise</b>, no single state above
<b>25% occupancy</b>.</p>

<p><b>Parameters:</b></p>
<ul>
<li><b>MCS values</b> — comma-separated values to test, e.g. 500,1000,2000,3000. Leave blank
for defaults.</li>
<li><b>UMAP sweep</b> — also tests UMAP n_neighbors values. Adds significant time. Only use if
clustering feels unstable.</li>
<li><b>HDBSCAN jobs</b> — parallel CPU cores for the sweep. Set to your core count minus one for
fastest results.</li>
</ul>
""",

"split_dominant": """
<p>When one state captures too many frames (&gt;25–30%), it likely contains multiple distinct
behaviors merged together. This tool re-clusters just the dominant state at finer resolution,
discovers sub-states within it, and rewrites all label files with the expanded state set.</p>

<p>Run this after clustering if your dominant state feels too broad. <b>Save your current cluster
run first</b> — this operation overwrites label files and cannot be undone without restoring a
saved run.</p>
""",

"clip_reviewer": """
<p><b>What it does:</b> Allows you to watch video clips from your dataset and assign them to
categories you define. This serves two purposes:</p>

<ul>
<li><b>State verification</b> — confirm that the behavioral states VIEB discovered make biological
sense by watching exemplar clips</li>
<li><b>Supervised annotation</b> — build a labeled dataset for training a classifier that can
predict behavioral categories in new, unlabeled data</li>
</ul>

<p><b>How to use:</b></p>

<ol>
<li>Define your categories by typing them in and pressing Enter (e.g. "Freezing, Locomotion,
Grooming" or "Success, Failure")</li>
<li>Press <b>Start Session</b> — clips will be shown in random order across all states</li>
<li>Watch each clip and press the corresponding category button (or keyboard shortcut 1, 2, 3…)</li>
<li>Press <b>Skip</b> to pass without labeling</li>
<li>Your progress is shown as a percentage distribution — come back across multiple sessions,
your annotations are saved automatically</li>
</ol>

<p>When you have enough annotations (at least 5 clips per category), a <b>Train Classifier</b>
button appears. This trains a Random Forest model on your annotations and can predict labels for
all unannotated clips, giving you a complete behavioral classification of your entire dataset.</p>
""",

"settings": """
<p><b>Raw videos directory:</b> The folder containing your .mp4 video files and their DLC pose
CSV files. Both must be in the same folder.</p>

<p><b>Results directory:</b> Where all pipeline output files are saved. Defaults to results/
inside the VIEB project folder.</p>

<p><b>Metadata CSV:</b> A CSV file with one row per video containing experimental metadata.
Required columns: filename (must match video filenames exactly), animal_id, day, context.
Additional columns are supported and can be mapped in Column Mapping.</p>

<p><b>DLC project directory:</b> The folder containing your DeepLabCut config.yaml and trained
model weights. Set this if you have an existing DLC project.</p>

<p><b>Condition A / B labels:</b> Human-readable names for your two experimental conditions used
in all plots and reports. Leave blank to auto-detect from your metadata.</p>

<p><b>Primary metric label:</b> The name for your primary per-animal scalar shown in Overview and
Quantification (e.g. Fear Index, Adaptation Score, Learning Index).</p>

<p><b>Arena bounds:</b> Pixel coordinates of the arena boundary. Used to compute distance-to-wall
features. Set to your full frame size if unsure.</p>

<p><b>FPS:</b> Frame rate of your videos. Used to convert frame counts to seconds in all bout
duration calculations.</p>

<p><b>UMAP dims:</b> Number of UMAP output dimensions before HDBSCAN clustering. Lower values
(3–5) run faster and produce coarser clusters. Higher values (10–15) preserve more structure.
Default: 10. See Stage 3 for a full explanation.</p>

<p><b>min_cluster_size:</b> The minimum number of frames for a group to be recognized as a
behavioral state. The most important clustering parameter. Recommended 2000–3000 for large datasets.
See Stage 3 for a full explanation.</p>

<p><b>HDBSCAN min_samples:</b> Controls cluster border conservatism. 0 = use min_cluster_size
(recommended). Only change if you have a specific reason. See Stage 3 for a full explanation.</p>
""",

}


def _nav_help_btn(section_id: str, signal) -> QPushButton:
    """Create a small circular '?' button that emits signal(section_id)."""
    b = QPushButton("?")
    b.setFixedSize(20, 20)
    b.setFlat(True)
    b.setToolTip("Open Help for this section")
    b.setCursor(Qt.PointingHandCursor)
    b.setStyleSheet(
        "QPushButton{border:1px solid #aaa;border-radius:10px;color:#555;"
        "background:#f5f5f5;font-size:10px;font-weight:bold;}"
        "QPushButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
    )
    b.clicked.connect(lambda: signal.emit(section_id))
    return b


class HelpView(QWidget):
    def __init__(self):
        super().__init__()
        self._anchors: dict[str, QLabel] = {}
        self._scroll: QScrollArea | None = None
        self._build()

    def _build(self):
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)

        root_lay = QVBoxLayout(self)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.addWidget(self._scroll)

        content = QWidget()
        self._scroll.setWidget(content)
        lay = QVBoxLayout(content)
        lay.setContentsMargins(32, 24, 32, 32)
        lay.setSpacing(0)

        # Page title
        page_title = QLabel("VIEB — Help & Documentation")
        page_title.setFont(QFont("Arial", 20, QFont.Bold))
        page_title.setStyleSheet("color:#1A1A1A; padding-bottom:4px;")
        lay.addWidget(page_title)

        sub = QLabel("Click a section to jump directly to that topic.")
        sub.setStyleSheet("color:#666; font-size:12px; padding-bottom:16px;")
        lay.addWidget(sub)

        # Table of contents
        toc_frame = QFrame()
        toc_frame.setStyleSheet(
            "QFrame{background:#f5f7ff;border:1px solid #c8d4f7;border-radius:8px;}"
        )
        toc_lay = QVBoxLayout(toc_frame)
        toc_lay.setContentsMargins(16, 12, 16, 12)
        toc_lay.setSpacing(1)
        toc_hdr = QLabel("Contents")
        toc_hdr.setFont(QFont("Arial", 11, QFont.Bold))
        toc_hdr.setStyleSheet("color:#1A1A1A; background:transparent; border:none;")
        toc_lay.addWidget(toc_hdr)
        for section_id, section_title in _SECTIONS:
            btn = QPushButton(f"  → {section_title}")
            btn.setFlat(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                "QPushButton{border:none;color:#0b57d0;background:transparent;"
                "font-size:12px;text-align:left;padding:2px 0;}"
                "QPushButton:hover{color:#1a73e8;}"
            )
            btn.clicked.connect(lambda _, sid=section_id: self.scroll_to_section(sid))
            toc_lay.addWidget(btn)
        lay.addWidget(toc_frame)
        lay.addSpacing(20)

        # Sections
        for section_id, section_title in _SECTIONS:
            self._add_section(lay, section_id, section_title, _CONTENT[section_id])

        lay.addStretch()

    def _add_section(self, lay, section_id: str, title: str, html_body: str):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(
            "background:#E5E5E5;border:none;max-height:1px;margin-top:8px;margin-bottom:0;"
        )
        lay.addWidget(sep)

        # Zero-height anchor widget used as scroll target
        anchor = QLabel()
        anchor.setFixedHeight(1)
        self._anchors[section_id] = anchor
        lay.addWidget(anchor)

        title_lbl = QLabel(title)
        title_lbl.setFont(QFont("Arial", 14, QFont.Bold))
        title_lbl.setStyleSheet("color:#1A1A1A; padding-top:10px; padding-bottom:4px;")
        lay.addWidget(title_lbl)

        body = QLabel(html_body.strip())
        body.setTextFormat(Qt.RichText)
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        body.setStyleSheet(
            "color:#333; font-size:12px; background:transparent; border:none;"
            "padding-bottom:8px;"
        )
        lay.addWidget(body)

    def scroll_to_section(self, section_id: str):
        anchor = self._anchors.get(section_id)
        if anchor and self._scroll:
            self._scroll.ensureWidgetVisible(anchor)
