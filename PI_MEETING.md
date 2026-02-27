# PI Meeting Prep — VIEB Behavioral Analysis

---

## What to say in one sentence

> "I built an unsupervised pipeline that discovers behavioral states directly from body-tracking data across all 222 videos, without me telling it what to look for — and it found a behavior expressed almost exclusively in the fear context that traditional freezing scoring would miss entirely."

---

## What to show (in order)

### 1. The pipeline overview (30 seconds)

```
Raw video → DeepLabCut (8 keypoints) → 49 kinematic features per frame
→ Shared clustering across 222 videos → 5 behavioral states
→ Compare states by context (A / B / C)
```

Key point for PI: **no manual annotation of behavior categories**. The categories emerge from the data.

---

### 2. The key finding: `results/characterization/context_fractions.png`

Show this plot. Walk through it:

| State | Context A (habituation, no fear) | Context B (post-shock, fear) | Context C (new environment) | Interpretation |
|-------|----------------------------------|------------------------------|------------------------------|----------------|
| 3 | ~51% | ~40% | ~13% | **High stillness** — most common state overall; drops in novel environment |
| 4 | ~0.1% | ~17.6% | ~0.7% | **Fear-specific unknown** — almost exclusively in shock context |
| 1 | ~1% | ~8% | ~40% | **Novelty exploration** — spikes sharply in new environment |
| 0+2 | ~49% | ~35% | ~47% | Locomotion / grooming variants — broadly distributed |

**The headline number**: State 4 is 0.1% in Context A but 17.6% in Context B — a behavioral state expressed almost exclusively in the fear/shock environment. The 95% bootstrap confidence interval for the B−A difference is [+0.115, +0.245], which does not cross zero. This is not detectable by standard freezing scoring.

Note on State 3: although it has the lowest movement speed (median 9 px/s, consistent with stillness), it is *more* common in the habituation context than the fear context. This suggests mice are resting/still during habituated sessions, and are more behaviorally *active* (including State 4) in the fear context — the opposite of simple freezing.

---

### 3. The statistics: `results/characterization/context_report.csv`

For each state, you have:
- **Fractions**: time-in-state per context (A, B, C)
- **Δ fractions**: B − A, C − A, C − B
- **Bootstrap 95% confidence intervals** (1000 resamples of video-level data)
- **Cohen's d** effect size

Key numbers for State 4:
- B − A = +0.175, 95% CI [+0.115, +0.245] — **CI does not cross zero**
- Cohen's d (B vs A) = 0.843 — large effect size

Key numbers for State 3:
- A − B = +0.111, 95% CI [+0.022, +0.196]
- Cohen's d (A vs B) = 0.367 — moderate effect; State 3 is actually *higher* in habituation

---

### 4. Physical signature of State 4: `results/comparison/pose_profiles.png`

This shows the average body posture for each state, normalized to body length.

State 4 has a distinctive kinematic fingerprint in the raw data:
- **Centroid speed**: 137 px/s (active — similar to locomotion states)
- **Apparent body length**: 84.7 px mean — **the shortest of all 5 states** (State 3 = 111 px, State 1 = 177 px)
- **Angular velocity**: 3.1 (low — not turning rapidly)
- **Movement entropy**: 0.094 (low — stereotyped, repetitive pattern)

Short apparent body length + active + stereotyped = consistent with **rearing behavior** (mouse standing on hind legs appears foreshortened in top-down camera view). In fear conditioning, rearing is associated with risk assessment scanning. Worth discussing with PI.

State 3 comparison:
- Speed: 23 px/s, angular_vel: 0.77 → genuinely slow/still
- Body length: 111 px → normal extended posture
- This is behaviorally quiet but not fear-specific

---

### 5. Exemplar clips (after running `--clips`): `clips/state_4/context_B_*.mp4`

Play 2–3 clips. Let the PI identify what State 4 is. This is the most compelling moment in the meeting.

Command to generate clips if not done:
```bash
python characterize.py --clips
```

---

## The math (enough to answer questions)

**Pose → features**: Each frame is converted to 49 numbers describing the animal's motion and posture independent of position in the cage:
- *Speed* of each of the 8 keypoints
- *Pairwise distances* between all keypoints (28 pairs) → body shape
- *Angular velocity* → turning rate
- *Body elongation* → stretched vs. compact
- *Movement entropy* → how stereotyped the motion is over a 1-second window

**Clustering (K-Means)**: Find K groups in the 49-dimensional feature space such that frames within each group are as similar as possible. The algorithm minimizes within-group variance:

```
J = Σ_k  Σ_{frames in group k}  || frame_features − group_center ||²
```

K=5 was selected automatically by silhouette score (a measure of cluster separation, −1 to +1). The search runs from K=4 to K=12 and picks the peak.

**Key design choice**: one shared model trained on *all 222 videos pooled together* (1,288,650 frames). This means "State 3" in a Day 1 Context A video is the same behavioral state as "State 3" in a Day 5 Context B video — directly comparable across animals, days, and contexts.

**Comparison statistics**: Mann-Whitney U test (non-parametric, appropriate for small N mouse groups) with 1000-sample bootstrap confidence intervals and Cohen's d effect size.

---

## Anticipated PI questions

**Q: How do you know these states are real behaviors and not clustering artifacts?**

A: Three converging lines of evidence: (1) the states are stable across 222 independent videos with a shared model, (2) they differ systematically by experimental context in interpretable ways — novelty exploration spikes in the new environment, stillness drops in the novel context, and State 4 is almost exclusively in the fear context, and (3) the exemplar clips show visually distinct behaviors.

**Q: State 3 has the lowest speed — isn't that freezing? Why is it highest in habituation?**

A: State 3 is genuinely slow (median centroid speed 9 px/s), consistent with stillness. But it is most common during habituated sessions when mice know the environment is safe and rest. In the fear context, mice appear to be more behaviorally active overall — including spending ~18% of the session in State 4, the fear-specific behavior. Standard freezing scores lump together "safe stillness" and "fear freezing," which may be two different things with different underlying states.

**Q: Why not just score freezing like everyone else?**

A: This pipeline captures the *full behavioral repertoire*, not just one behavior we decided to look for. State 4 — context-specific, ~18% of fear-context sessions — would be completely invisible to freezing-only scoring. We don't know yet what it is, but it discriminates shock vs. no-shock context with a large effect size (Cohen's d = 0.84).

**Q: How much labeled data did this require?**

A: For the pose tracking model (DeepLabCut), about 40–50 manually labeled frames across ~20 videos. The behavioral clustering requires **zero** behavioral labels — it is fully unsupervised.

**Q: What's the mAP of the pose estimation?**

A: Currently 84.1% (measured on held-out frames). For detecting subtle micro-behaviors, we want ≥90%. More labeled frames will improve the downstream behavioral results.

**Q: Is this validated?**

A: Not yet formally. The immediate next step is to watch the exemplar clips for State 4 and get PI/lab agreement on what it is. Once states are labeled by observation, we can compare against traditional scoring on a subset of videos.

---

## What is NOT done yet (be honest)

1. **State 4 identity is unconfirmed** — the kinematic profile suggests rearing/risk-assessment, but need to watch clips to confirm
2. **mAP is 84%, should be ≥90%** — need ~10 more labeled videos in DLC
3. **`fear` column in metadata.csv is empty** — once filled, the fear vs. no-fear comparison (collapsing across days) runs automatically with `python compare.py --report`
4. **Context C protocol unclear** — is it re-exposure to the shock context or a genuinely different environment? This affects how to interpret the State 1 spike there
5. **Single cohort** — results should replicate in a second cohort before strong claims

---

## Files to open in the meeting

```
results/characterization/context_fractions.png   ← lead with this
results/comparison/pose_profiles.png             ← show body shapes
results/characterization/context_report.csv      ← statistics
clips/state_4/context_B_01.mp4                  ← the key mystery clip (generate with: python characterize.py --clips)
```
