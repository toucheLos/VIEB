"""
anomaly_detection.py

Deep learning models for detecting unusual/rare behavioral patterns.

Uses autoencoders to learn a compressed representation of normal behavior,
then identifies anomalies as samples with high reconstruction error.

This is particularly powerful for discovering subtle, previously unknown
behavioral patterns that deviate from typical mouse behavior.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class BehaviorAutoencoder(nn.Module):
    """
    Variational Autoencoder for behavioral feature learning.

    The encoder compresses behavioral features into a low-dimensional
    latent space, and the decoder reconstructs the original features.

    Anomalies are detected as samples with high reconstruction error.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: List[int] = [128, 64, 32]
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input feature vectors
        latent_dim : int
            Dimension of latent bottleneck layer
        hidden_dims : list of int
            Hidden layer dimensions for encoder
        """
        super(BehaviorAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Forward pass: encode then decode.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim)

        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed features
        latent : torch.Tensor
            Latent representation
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent representation to reconstruction."""
        return self.decoder(z)


class AnomalyDetector:
    """
    Anomaly detection system for behavioral patterns.

    Trains an autoencoder on normal behavior and detects anomalies
    as samples with reconstruction errors above a threshold.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: List[int] = [128, 64, 32],
        learning_rate: float = 1e-3,
        device: str = "auto"
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input features
        latent_dim : int
            Dimension of latent bottleneck
        hidden_dims : list of int
            Hidden layer dimensions
        learning_rate : float
            Learning rate for Adam optimizer
        device : str
            Device to use: "auto", "cuda", "mps", or "cpu"
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = BehaviorAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.threshold = None
        self.trained = False

    def train(
        self,
        features: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.2,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the autoencoder on behavioral features.

        Parameters
        ----------
        features : np.ndarray
            Training features of shape (T, F)
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        validation_split : float
            Fraction of data for validation
        patience : int
            Early stopping patience
        verbose : bool
            Print training progress

        Returns
        -------
        history : dict
            Training history with losses
        """
        # Split into train/validation
        n_samples = len(features)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_features = features[train_indices]
        val_features = features[val_indices]

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_features)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(val_features)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # Training loop
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                x = batch[0].to(self.device)

                self.optimizer.zero_grad()
                reconstruction, _ = self.model(x)
                loss = self.criterion(reconstruction, x)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(x)

            train_loss /= len(train_dataset)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(self.device)
                    reconstruction, _ = self.model(x)
                    loss = self.criterion(reconstruction, x)
                    val_loss += loss.item() * len(x)

            val_loss /= len(val_dataset)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(self.best_model_state)

        # Compute anomaly threshold from validation set
        self._compute_threshold(val_features)

        self.trained = True

        return history

    def _compute_threshold(self, features: np.ndarray, percentile: float = 95):
        """
        Compute anomaly threshold based on reconstruction errors.

        Parameters
        ----------
        features : np.ndarray
            Features to compute threshold from
        percentile : float
            Percentile for threshold (e.g., 95 = top 5% are anomalies)
        """
        reconstruction_errors = self.compute_reconstruction_error(features)
        self.threshold = np.percentile(reconstruction_errors, percentile)

    def compute_reconstruction_error(self, features: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for each sample.

        Parameters
        ----------
        features : np.ndarray
            Features of shape (T, F)

        Returns
        -------
        errors : np.ndarray
            Reconstruction errors of shape (T,)
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before computing errors")

        self.model.eval()
        errors = []

        with torch.no_grad():
            # Process in batches
            batch_size = 256
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size]
                x = torch.FloatTensor(batch).to(self.device)
                reconstruction, _ = self.model(x)

                # MSE per sample
                error = torch.mean((reconstruction - x) ** 2, dim=1)
                errors.append(error.cpu().numpy())

        return np.concatenate(errors)

    def detect_anomalies(
        self,
        features: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalous behavioral patterns.

        Parameters
        ----------
        features : np.ndarray
            Features of shape (T, F)
        threshold : float, optional
            Custom threshold (uses learned threshold if None)

        Returns
        -------
        is_anomaly : np.ndarray
            Boolean array indicating anomalies
        anomaly_scores : np.ndarray
            Anomaly scores (reconstruction errors)
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before detecting anomalies")

        if threshold is None:
            threshold = self.threshold

        anomaly_scores = self.compute_reconstruction_error(features)
        is_anomaly = anomaly_scores > threshold

        return is_anomaly, anomaly_scores

    def get_latent_representation(self, features: np.ndarray) -> np.ndarray:
        """
        Encode features to latent space.

        Useful for visualization and further analysis.

        Parameters
        ----------
        features : np.ndarray
            Features of shape (T, F)

        Returns
        -------
        latent : np.ndarray
            Latent representations of shape (T, latent_dim)
        """
        if not self.trained:
            raise RuntimeError("Model must be trained first")

        self.model.eval()
        latent_list = []

        with torch.no_grad():
            batch_size = 256
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size]
                x = torch.FloatTensor(batch).to(self.device)
                latent = self.model.encode(x)
                latent_list.append(latent.cpu().numpy())

        return np.concatenate(latent_list)

    def save(self, filepath: str):
        """
        Save trained model to disk.

        Parameters
        ----------
        filepath : str
            Path to save model
        """
        if not self.trained:
            raise RuntimeError("Cannot save untrained model")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'learning_rate': self.learning_rate
        }, filepath)

    def load(self, filepath: str):
        """
        Load trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Recreate model with saved architecture
        self.input_dim = checkpoint['input_dim']
        self.latent_dim = checkpoint['latent_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.learning_rate = checkpoint['learning_rate']

        self.model = BehaviorAutoencoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.threshold = checkpoint['threshold']

        self.trained = True

    def plot_reconstruction_errors(
        self,
        features: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of reconstruction errors.

        Parameters
        ----------
        features : np.ndarray
            Features to analyze
        save_path : str, optional
            Path to save figure
        """
        import matplotlib.pyplot as plt

        errors = self.compute_reconstruction_error(features)

        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(self.threshold, color='r', linestyle='--',
                   label=f'Anomaly Threshold ({self.threshold:.4f})')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
