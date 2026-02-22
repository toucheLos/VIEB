# VIEB — First Use Guide

You have videos. Here's everything you need to do.

---

## 0. Requirements

- Python 3.10+
- Your `.mp4` videos

---

## 1. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[tracking]"    # installs VIEB + DeepLabCut
```

---

## 2. Add Your Videos

Place all `.mp4` files into `raw_videos/`.

---

## 3. Extract Frames for Labeling

```bash
python setup_dlc_training.py
```

This registers your videos in the DLC config and opens the labeling GUI.

---

## 4. Label Frames

The napari GUI will open automatically. For each frame, click on all 8 keypoints:

`left_ear` · `right_ear` · `nose` · `center` · `lateral_left` · `lateral_right` · `tail_base` · `tail_end`

Save with **Ctrl+S**, then close the GUI.

---

## 5. Train the Model

```bash
python setup_dlc_training.py --train
```

Takes 30 min – 2 hrs. You can stop early with Ctrl+C — snapshots are saved automatically.

---

## 6. Evaluate the Model

```bash
python setup_dlc_training.py --evaluate
```

Check `VIEB-Carlos-*/evaluation-results-pytorch/` for accuracy metrics and labeled overlay images.

---

## 7. Run Pose Estimation on All Videos

```bash
caffeinate -i python setup_dlc_training.py --analyze
```

Outputs a `.csv` and `.h5` per video alongside each file in `raw_videos/`. Takes a while — `caffeinate` prevents your Mac from sleeping.

---

## 8. Run Behavioral Analysis

```bash
python main.py \
  --video raw_videos/your_video.mp4 \
  --dlc-config VIEB-Carlos-*/config.yaml \
  --output results/your_video/
```

Repeat for each video, or write a loop. Results land in `results/`.

---

## Output Files (per video)

| File | Contents |
|------|----------|
| `clusters_pca.png` | Behavioral states visualized |
| `ethogram.png` | State timeline |
| `behavior_summary.png` | Full summary figure |
| `transition_matrix.png` | State-to-state transitions |
| `anomaly_distribution.png` | Unusual behavior scores |
| `analysis_report.txt` | Human-readable summary |
| `preprocessor.pkl` | Saved normalizer (reusable) |
| `anomaly_detector.pt` | Saved anomaly model |
| `temporal_model.pt` | Saved LSTM model |

---

## Tips

- **Skip anomaly/temporal analysis** (faster): add `--no-anomaly --no-temporal`
- **Fix number of behavioral states**: add `--n-clusters 6`
- **Different frame rate**: add `--fps 25`
- Steps 3–6 only need to be done **once** per project. Step 7–8 repeat for new videos.
