# VIEB Quick Start Guide

Get started with VIEB in 5 minutes!

## Prerequisites

- Python ≥ 3.10
- A video of mouse behavior
- DeepLabCut config file (or existing pose tracking data)

## Installation

```bash
# Navigate to VIEB directory
cd /Users/carlos/Desktop/neuron/luna_lab/VIEB

# Activate virtual environment (if not already active)
source venv/bin/activate

# Install VIEB with all dependencies
pip install -e .
```

## Quick Test Run

### Option 1: Full Pipeline (if you have a video)

```bash
python main.py \
    --video path/to/your/mouse_video.mp4 \
    --dlc-config VIEB-Carlos-2026-02-11/config.yaml \
    --output test_output/
```

This will:
1. Run DeepLabCut pose tracking
2. Extract behavioral features
3. Discover behavioral states (clustering)
4. Detect anomalies (autoencoder)
5. Analyze temporal dynamics (LSTM)
6. Generate visualizations and reports

Expected runtime: 15-30 minutes

### Option 2: Just ML Pipeline (if you have pose data)

```python
from tracking.deeplabcut_backend import DeepLabCutTracker
from ml import PoseFeatureExtractor, BehaviorPreprocessor, BehaviorClusterer

# Load existing pose data
tracker = DeepLabCutTracker({"dlc_config_path": "config.yaml"})
pose_data = tracker.load_outputs("path/to/dlc_output/")

# Extract features
extractor = PoseFeatureExtractor(fps=30.0)
features = extractor.extract_features(pose_data["pose"])

# Discover behavioral states
preprocessor = BehaviorPreprocessor()
features_norm = preprocessor.fit_transform(features["flattened"])

clusterer = BehaviorClusterer(method="kmeans", auto_tune=True)
labels = clusterer.fit_predict(features_norm)

print(f"Discovered {clusterer.n_clusters} behavioral states!")
```

## Understanding the Output

After running, check the `output/` directory:

### Key Files

1. **behavior_summary.png** - Multi-panel overview figure
   - Top: Ethogram (behavioral states over time)
   - Middle: State distribution histogram
   - Bottom: Anomaly scores over time

2. **clusters_pca.png** - 2D visualization of behavioral clusters
   - Each color = different behavioral state
   - Proximity = behavioral similarity

3. **transition_matrix.png** - How behaviors transition
   - Heatmap showing transition probabilities
   - Diagonal = staying in same state
   - Off-diagonal = transitions between states

4. **analysis_report.txt** - Human-readable summary
   - Number of states discovered
   - Occurrence rates and durations
   - Anomaly statistics
   - Top transitions

5. **behavioral_states.csv** - State statistics table
   - Occurrence rates
   - Mean bout durations
   - Number of bouts

6. **analysis_results.json** - Complete results (machine-readable)

### Interpreting Results

#### Behavioral States
- **High occurrence rate** (>20%) = common behavior (e.g., exploring)
- **Short bouts** (<1 sec) = transient states
- **Long bouts** (>5 sec) = sustained behaviors (e.g., grooming, resting)

#### Anomalies
- **Low anomaly rate** (<5%) = typical
- **High scores** = unusual movements worth investigating
- Check `anomaly_bouts.csv` for specific time windows

#### Transitions
- **High self-transitions** = stable states
- **Frequent transitions** = behavioral switching patterns
- Can reveal routine behavioral sequences

## Common Use Cases

### 1. Discover behavioral repertoire
```bash
python main.py --video mouse.mp4 --dlc-config config.yaml
```
Result: Identifies all discrete behavioral states in your video

### 2. Find rare behaviors
```bash
python main.py --video mouse.mp4 --dlc-config config.yaml --n-clusters 5
```
Result: Focuses on major states, flags everything else as anomalies

### 3. Compare experimental conditions
```python
# Train on control
clusterer.fit(control_features)
control_labels = clusterer.predict(control_features)

# Apply to treatment
treatment_labels = clusterer.predict(treatment_features)

# Compare distributions
# (treatment may spend more time in certain states)
```

### 4. Track behavioral changes over time
```python
# Split video into epochs
early_features = features[:1000]
late_features = features[-1000:]

# Cluster each epoch
early_labels = clusterer.fit_predict(early_features)
late_labels = clusterer.predict(late_features)

# Compare how behavior evolved
```

## Troubleshooting

### "DeepLabCut not installed"
```bash
pip install deeplabcut
```

### "CUDA/GPU not available"
- No problem! Models will use CPU (just slower)
- To enable GPU: Install PyTorch with CUDA support

### "Too many/few clusters detected"
```bash
# Manually specify number of clusters
python main.py --video video.mp4 --dlc-config config.yaml --n-clusters 8
```

### "Out of memory"
- Reduce batch sizes in code (edit `ml/anomaly_detection.py` and `ml/sequence_models.py`)
- Skip temporal analysis: `--no-temporal`
- Process shorter video segments

### "Low confidence predictions"
- Normal! VIEB handles this automatically
- Low-confidence keypoints are interpolated
- Check `confidence` threshold in DLC config

## Next Steps

1. **Explore the code** - All modules in `ml/` are documented
2. **Customize features** - Edit `ml/feature_extraction.py`
3. **Try different clustering** - K-Means, DBSCAN, GMM in `ml/clustering.py`
4. **Adjust anomaly sensitivity** - Change `percentile` in `AnomalyDetector._compute_threshold()`
5. **Compare conditions** - See `example_usage.py`

## Getting Help

- Read `ML_PIPELINE_SUMMARY.md` for detailed documentation
- Check `example_usage.py` for code examples
- Review inline comments in source files

## What's Next?

Now that you have behavioral patterns discovered:
- **Validate** by watching videos at detected anomaly times
- **Correlate** with neural recordings or other signals
- **Compare** across experimental conditions
- **Publish** using generated figures and statistics

Happy analyzing! 🐁🔬
