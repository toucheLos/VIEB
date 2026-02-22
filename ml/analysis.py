"""
analysis.py

High-level behavior analysis and visualization tools.

Integrates all ML components to provide comprehensive behavioral insights:
- Statistical summaries of discovered patterns
- Temporal dynamics visualization
- Comparative analysis between conditions
- Export results for publication
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd


class BehaviorAnalyzer:
    """
    Comprehensive behavior analysis system.

    Combines clustering, anomaly detection, and temporal modeling
    to provide interpretable insights about mouse behavior.
    """

    def __init__(self, fps: float = 30.0):
        """
        Parameters
        ----------
        fps : float
            Frames per second of videos
        """
        self.fps = fps
        self.results = {}

    def analyze_behavioral_states(
        self,
        cluster_labels: np.ndarray,
        feature_names: List[str],
        features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze discovered behavioral states from clustering.

        Parameters
        ----------
        cluster_labels : np.ndarray
            Cluster assignments for each frame
        feature_names : list of str
            Names of features
        features : np.ndarray
            Feature matrix used for clustering

        Returns
        -------
        analysis : dict
            Statistical analysis of behavioral states
        """
        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise

        analysis = {
            "n_states": len(unique_labels),
            "states": {}
        }

        for label in unique_labels:
            mask = cluster_labels == label
            state_features = features[mask]

            # Compute bout statistics
            bouts = self._extract_bouts(mask)

            state_info = {
                "occurrence_rate": np.mean(mask),
                "total_frames": np.sum(mask),
                "total_duration_sec": np.sum(mask) / self.fps,
                "n_bouts": len(bouts),
                "mean_bout_duration_sec": np.mean([b[1] - b[0] for b in bouts]) / self.fps if bouts else 0,
                "median_bout_duration_sec": np.median([b[1] - b[0] for b in bouts]) / self.fps if bouts else 0,
                "max_bout_duration_sec": np.max([b[1] - b[0] for b in bouts]) / self.fps if bouts else 0,
                "feature_profile": {
                    name: {
                        "mean": float(np.mean(state_features[:, i])),
                        "std": float(np.std(state_features[:, i])),
                        "median": float(np.median(state_features[:, i]))
                    }
                    for i, name in enumerate(feature_names[:state_features.shape[1]])
                }
            }

            analysis["states"][f"state_{label}"] = state_info

        self.results["behavioral_states"] = analysis
        return analysis

    def analyze_anomalies(
        self,
        is_anomaly: np.ndarray,
        anomaly_scores: np.ndarray,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze detected anomalous behaviors.

        Parameters
        ----------
        is_anomaly : np.ndarray
            Boolean array indicating anomalies
        anomaly_scores : np.ndarray
            Anomaly scores (reconstruction errors)
        features : np.ndarray
            Feature matrix

        Returns
        -------
        analysis : dict
            Anomaly statistics
        """
        # Extract anomaly bouts
        bouts = self._extract_bouts(is_anomaly)

        analysis = {
            "anomaly_rate": float(np.mean(is_anomaly)),
            "total_anomalous_frames": int(np.sum(is_anomaly)),
            "total_anomalous_duration_sec": float(np.sum(is_anomaly) / self.fps),
            "n_anomaly_bouts": len(bouts),
            "mean_bout_duration_sec": float(np.mean([b[1] - b[0] for b in bouts]) / self.fps) if bouts else 0,
            "anomaly_score_stats": {
                "mean": float(np.mean(anomaly_scores)),
                "std": float(np.std(anomaly_scores)),
                "min": float(np.min(anomaly_scores)),
                "max": float(np.max(anomaly_scores)),
                "median": float(np.median(anomaly_scores)),
                "p95": float(np.percentile(anomaly_scores, 95)),
                "p99": float(np.percentile(anomaly_scores, 99))
            },
            "anomaly_bouts": [
                {
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "duration_sec": float((end - start) / self.fps),
                    "mean_score": float(np.mean(anomaly_scores[start:end]))
                }
                for start, end in bouts
            ]
        }

        self.results["anomalies"] = analysis
        return analysis

    def analyze_temporal_transitions(
        self,
        cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze transitions between behavioral states.

        Parameters
        ----------
        cluster_labels : np.ndarray
            Cluster assignments over time

        Returns
        -------
        analysis : dict
            Transition statistics and matrix
        """
        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]
        n_states = len(unique_labels)

        # Build transition matrix
        transition_matrix = np.zeros((n_states, n_states))
        transition_counts = np.zeros((n_states, n_states))

        for i in range(len(cluster_labels) - 1):
            current = cluster_labels[i]
            next_state = cluster_labels[i + 1]

            if current != -1 and next_state != -1:
                curr_idx = np.where(unique_labels == current)[0][0]
                next_idx = np.where(unique_labels == next_state)[0][0]
                transition_counts[curr_idx, next_idx] += 1

        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_counts / row_sums

        # Find most common transitions
        transitions = []
        for i in range(n_states):
            for j in range(n_states):
                if i != j and transition_counts[i, j] > 0:
                    transitions.append({
                        "from_state": int(unique_labels[i]),
                        "to_state": int(unique_labels[j]),
                        "count": int(transition_counts[i, j]),
                        "probability": float(transition_matrix[i, j])
                    })

        # Sort by count
        transitions.sort(key=lambda x: x["count"], reverse=True)

        analysis = {
            "transition_matrix": transition_matrix.tolist(),
            "transition_counts": transition_counts.tolist(),
            "state_labels": unique_labels.tolist(),
            "top_transitions": transitions[:10]  # Top 10 transitions
        }

        self.results["transitions"] = analysis
        return analysis

    def _extract_bouts(self, binary_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract continuous bouts from binary mask.

        Parameters
        ----------
        binary_mask : np.ndarray
            Boolean or binary array

        Returns
        -------
        bouts : list of (start, end) tuples
        """
        diff = np.diff(np.concatenate([[0], binary_mask.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        return list(zip(starts, ends))

    def plot_behavior_summary(
        self,
        cluster_labels: np.ndarray,
        anomaly_scores: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive summary visualization.

        Parameters
        ----------
        cluster_labels : np.ndarray
            Behavioral state labels over time
        anomaly_scores : np.ndarray, optional
            Anomaly scores over time
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        time = np.arange(len(cluster_labels)) / self.fps

        # 1. Ethogram (behavioral states over time)
        axes[0].plot(time, cluster_labels, linewidth=0.5, color='black')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Behavioral State')
        axes[0].set_title('Ethogram: Behavioral States Over Time')
        axes[0].grid(True, alpha=0.3)

        # 2. State distribution
        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]

        state_counts = [np.sum(cluster_labels == label) for label in unique_labels]
        axes[1].bar(unique_labels, state_counts, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Behavioral State')
        axes[1].set_ylabel('Number of Frames')
        axes[1].set_title('Distribution of Behavioral States')
        axes[1].grid(True, alpha=0.3, axis='y')

        # 3. Anomaly scores (if available)
        if anomaly_scores is not None:
            axes[2].plot(time, anomaly_scores, linewidth=0.5, color='red', alpha=0.7)
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Anomaly Score')
            axes[2].set_title('Anomaly Detection Scores Over Time')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No anomaly scores provided',
                        ha='center', va='center', transform=axes[2].transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_transition_matrix(
        self,
        transition_matrix: np.ndarray,
        state_labels: List[int],
        save_path: Optional[str] = None
    ):
        """
        Visualize state transition matrix as heatmap.

        Parameters
        ----------
        transition_matrix : np.ndarray
            Transition probability matrix
        state_labels : list of int
            State labels
        save_path : str, optional
            Path to save figure
        """
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            transition_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=[f'State {l}' for l in state_labels],
            yticklabels=[f'State {l}' for l in state_labels],
            cbar_kws={'label': 'Transition Probability'}
        )

        plt.xlabel('To State')
        plt.ylabel('From State')
        plt.title('Behavioral State Transition Matrix')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def export_results(self, output_dir: str):
        """
        Export all analysis results to files.

        Parameters
        ----------
        output_dir : str
            Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export as JSON
        json_path = output_path / "analysis_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results exported to {output_dir}")

        # Export behavioral states as CSV
        if "behavioral_states" in self.results:
            states_data = []
            for state_name, state_info in self.results["behavioral_states"]["states"].items():
                states_data.append({
                    "state": state_name,
                    "occurrence_rate": state_info["occurrence_rate"],
                    "total_duration_sec": state_info["total_duration_sec"],
                    "n_bouts": state_info["n_bouts"],
                    "mean_bout_duration_sec": state_info["mean_bout_duration_sec"]
                })

            df = pd.DataFrame(states_data)
            csv_path = output_path / "behavioral_states.csv"
            df.to_csv(csv_path, index=False)
            print(f"Behavioral states exported to {csv_path}")

        # Export anomalies as CSV
        if "anomalies" in self.results and "anomaly_bouts" in self.results["anomalies"]:
            df = pd.DataFrame(self.results["anomalies"]["anomaly_bouts"])
            csv_path = output_path / "anomaly_bouts.csv"
            df.to_csv(csv_path, index=False)
            print(f"Anomaly bouts exported to {csv_path}")

        # Export transition matrix as CSV
        if "transitions" in self.results:
            transition_matrix = np.array(self.results["transitions"]["transition_matrix"])
            state_labels = self.results["transitions"]["state_labels"]

            df = pd.DataFrame(
                transition_matrix,
                index=[f"State_{l}" for l in state_labels],
                columns=[f"State_{l}" for l in state_labels]
            )
            csv_path = output_path / "transition_matrix.csv"
            df.to_csv(csv_path)
            print(f"Transition matrix exported to {csv_path}")

    def generate_report(self, output_path: str):
        """
        Generate a human-readable text report.

        Parameters
        ----------
        output_path : str
            Path to save report
        """
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VIEB Behavioral Analysis Report\n")
            f.write("=" * 80 + "\n\n")

            # Behavioral states summary
            if "behavioral_states" in self.results:
                f.write("BEHAVIORAL STATES\n")
                f.write("-" * 80 + "\n")
                states = self.results["behavioral_states"]
                f.write(f"Number of discovered states: {states['n_states']}\n\n")

                for state_name, state_info in states["states"].items():
                    f.write(f"{state_name.upper()}:\n")
                    f.write(f"  Occurrence rate: {state_info['occurrence_rate']:.2%}\n")
                    f.write(f"  Total duration: {state_info['total_duration_sec']:.2f} seconds\n")
                    f.write(f"  Number of bouts: {state_info['n_bouts']}\n")
                    f.write(f"  Mean bout duration: {state_info['mean_bout_duration_sec']:.2f} seconds\n")
                    f.write("\n")

            # Anomalies summary
            if "anomalies" in self.results:
                f.write("\nANOMALY DETECTION\n")
                f.write("-" * 80 + "\n")
                anomalies = self.results["anomalies"]
                f.write(f"Anomaly rate: {anomalies['anomaly_rate']:.2%}\n")
                f.write(f"Total anomalous duration: {anomalies['total_anomalous_duration_sec']:.2f} seconds\n")
                f.write(f"Number of anomaly bouts: {anomalies['n_anomaly_bouts']}\n")
                f.write(f"Mean bout duration: {anomalies['mean_bout_duration_sec']:.2f} seconds\n")
                f.write("\n")

            # Transitions summary
            if "transitions" in self.results:
                f.write("\nBEHAVIORAL TRANSITIONS\n")
                f.write("-" * 80 + "\n")
                transitions = self.results["transitions"]["top_transitions"]
                f.write("Top 5 most common transitions:\n")
                for i, trans in enumerate(transitions[:5], 1):
                    f.write(f"  {i}. State {trans['from_state']} → State {trans['to_state']}: "
                           f"{trans['count']} times ({trans['probability']:.2%})\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Report generated: {output_path}")
