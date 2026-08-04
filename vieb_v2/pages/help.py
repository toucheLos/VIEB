"""Help page -- what each stage does, what the metrics mean, CLI equivalents.

Static content, so it works with no project loaded. The metric definitions
matter more than usual here: v2 reports occupancy in two conventions and the
difference is easy to misread.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from app import theme
from app.widgets import SectionTitle, scroll_content

SECTIONS = [
    ("Pipeline stages", """
<b>Align</b> — loads DLC pose, drops <tt>tail_tip</tt>, and removes translation
and rotation by weighted Procrustes, so the same posture maps to the same
coordinates wherever it occurred in the arena. Scale is deliberately
<i>not</i> removed: for a top-down 2D view, rearing is out-of-plane motion whose
only signature is apparent foreshortening, so normalising scale would delete it.
Keypoints are weighted by their DLC likelihood.

<b>Latent space</b> — reduces aligned pose, fitted once across every recording.
Per-recording fitting is not offered: two frames with identical pose would get
different coordinates, and no state would mean the same thing across animals.

<b>Delay embedding</b> — stacks each frame with its preceding lags, so a
clustered point is a short trajectory rather than an instant. This is what lets
a static clusterer distinguish rear-up from rear-down. Lags never cross a
recording boundary.

<b>Cluster</b> — HDBSCAN over the delay-embedded coordinates. The −1 noise label
is preserved and never force-assigned.
"""),
    ("PCA or diffusion maps?", """
<b>PCA</b> is linear. Distances in the embedding are exact Euclidean distances
in pose space, and components are interpretable as postural deformations.

<b>Diffusion maps</b> are nonlinear, and unlike UMAP the distance has a
definition: Euclidean distance in the embedding is <i>diffusion distance</i> —
the probability a random walk connects two points through the data manifold
within t steps. Being an eigenproblem rather than a stochastic optimisation, it
is reproducible: two runs on the same data give the same answer.

The <b>alpha</b> parameter matters. At alpha=1 (Laplace–Beltrami) the embedding
geometry is largely independent of how densely each region was sampled. Below
that, densely sampled — that is, slow — behaviors get compressed toward a point,
because a random walk mixes quickly through a well-connected neighbourhood.
"""),
    ("Reading the metrics", """
<b>n_states</b> — number of clusters found, excluding noise.

<b>noise_frac</b> — fraction of frames HDBSCAN left unclustered (−1).

<b>largest_state_frac</b> and <b>state_entropy</b> are reported in two
conventions, and they are not interchangeable:
<br>• <i>v1 convention</i> — fractions over <i>all</i> frames including noise,
so they sum to 1 − noise_frac, and entropy is normalised by log(n_states).
<br>• <i>clustered only</i> — fractions over clustered frames, summing to 1.

<b>noise/clustered speed ratio</b> — mean step length of unclustered frames
divided by that of clustered ones. Above 1 means the frames that failed to
cluster are the fast ones, which is the signature of density-based clustering
under-detecting brief behaviors. Near 1 means that account does not explain the
result, and should stop being cited.

A large dominant state is <i>not</i> by itself evidence of a problem — it may
simply be how long the animal spent in that behavior.
"""),
    ("Command line equivalents", """
Everything the GUI does is available headless, which is how it runs on a
cluster:
<pre>python -m vieb_v2.cli doctor
python -m vieb_v2.cli tune    --pose results/pose
python -m vieb_v2.cli run     --pose results/pose --latent-method pca
python -m vieb_v2.cli align   --pose results/pose --out results/v2
python -m vieb_v2.cli latent  --out results/v2 --latent-method diffusion
python -m vieb_v2.cli embed   --out results/v2 --n-lags 4 --lag-stride 2
python -m vieb_v2.cli cluster --out results/v2 --min-cluster-size 50
python -m vieb_v2.cli sweep   --out results/v2 --min-cluster-sizes 25,50,100
python -m vieb_v2.cli compare-latents --out results/v2
python -m vieb_v2.cli benchmark --project /path/to/vieb</pre>

<tt>doctor</tt> reports the environment and whether GPU clustering is usable —
run it first on any new machine. Exit codes: 0 completed, 1 needs attention
(missing pose, only one state found), 2 hard failure.
"""),
    ("Getting pose data", """
The pipeline starts from continuous per-frame DeepLabCut output — one
<tt>.h5</tt> or <tt>.csv</tt> per recording. If the pose directory is empty,
run DLC first:
<pre>python setup_dlc_training.py --analyze</pre>
Sparse hand-labelled frames (DLC's <tt>labeled-data/</tt>) are not sufficient:
delay embedding needs consecutive frames.
"""),
]


class HelpPage(QWidget):
    TITLE = "Help"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(theme.page_background())

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.PAGE_MARGIN, 24, theme.PAGE_MARGIN, 24)

        heading = QLabel("Help")
        heading.setStyleSheet(theme.heading_style())
        outer.addWidget(heading)

        area, layout = scroll_content()
        for title, body in SECTIONS:
            layout.addWidget(SectionTitle(title))
            label = QLabel(body.strip())
            label.setWordWrap(True)
            label.setTextFormat(Qt.RichText)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setStyleSheet(
                theme.label_style(theme.TEXT, theme.FONT_SIZE_BODY))
            layout.addWidget(label)
        layout.addStretch()
        outer.addWidget(area, stretch=1)
