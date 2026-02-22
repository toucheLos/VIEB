"""
sequence_models.py

Temporal sequence models for understanding behavioral dynamics.

Uses recurrent neural networks (LSTMs) to capture temporal dependencies
and predict future behaviors from past sequences. This enables detection
of complex temporal patterns that simple frame-by-frame analysis misses.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class BehaviorLSTM(nn.Module):
    """
    LSTM-based model for behavioral sequence analysis.

    Can be used for:
    - Sequence classification (identify behavioral motifs)
    - Sequence prediction (predict next behavior)
    - Sequence embedding (learn temporal representations)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: Optional[int] = None,
        bidirectional: bool = True,
        dropout: float = 0.2,
        task: str = "embedding"
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input features per timestep
        hidden_dim : int
            Hidden dimension of LSTM
        num_layers : int
            Number of LSTM layers
        output_dim : int, optional
            Output dimension (for classification/prediction tasks)
        bidirectional : bool
            Whether to use bidirectional LSTM
        dropout : float
            Dropout probability
        task : str
            Task type: "embedding", "classification", or "prediction"
        """
        super(BehaviorLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.task = task

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output dimension accounting for bidirectionality
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Task-specific heads
        if task == "classification":
            if output_dim is None:
                raise ValueError("output_dim required for classification task")
            self.fc = nn.Sequential(
                nn.Linear(lstm_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        elif task == "prediction":
            self.fc = nn.Sequential(
                nn.Linear(lstm_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim)
            )
        elif task == "embedding":
            self.fc = nn.Linear(lstm_output_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (batch_size, seq_len, input_dim)
        hidden : tuple, optional
            Initial hidden state

        Returns
        -------
        output : torch.Tensor
            Task-specific output
        hidden : tuple
            Final hidden state
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)

        # For sequence-to-one tasks, use last timestep
        if self.task in ["classification", "embedding"]:
            # Take last output (or mean/max pooling)
            last_output = lstm_out[:, -1, :]
            output = self.fc(last_output)
        # For sequence-to-sequence tasks, use all timesteps
        elif self.task == "prediction":
            # Apply fc to all timesteps
            batch_size, seq_len, _ = lstm_out.shape
            lstm_out_flat = lstm_out.reshape(-1, lstm_out.size(2))
            output_flat = self.fc(lstm_out_flat)
            output = output_flat.reshape(batch_size, seq_len, -1)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return output, hidden


class TemporalBehaviorModel:
    """
    High-level interface for temporal behavior analysis.

    Trains LSTM models to learn temporal patterns and predict behaviors.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: Optional[int] = None,
        task: str = "embedding",
        learning_rate: float = 1e-3,
        device: str = "auto"
    ):
        """
        Parameters
        ----------
        input_dim : int
            Feature dimension per timestep
        hidden_dim : int
            LSTM hidden dimension
        num_layers : int
            Number of LSTM layers
        output_dim : int, optional
            Output dimension for classification
        task : str
            "embedding", "classification", or "prediction"
        learning_rate : float
            Learning rate
        device : str
            Device: "auto", "cuda", "mps", or "cpu"
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.task = task
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
        self.model = BehaviorLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            task=task
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Task-specific loss
        if task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        self.trained = False

    def train(
        self,
        sequences: np.ndarray,
        labels: Optional[np.ndarray] = None,
        n_epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the temporal model.

        Parameters
        ----------
        sequences : np.ndarray
            Input sequences of shape (N, seq_len, input_dim)
        labels : np.ndarray, optional
            Labels for supervised tasks (classification)
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        validation_split : float
            Fraction for validation
        patience : int
            Early stopping patience
        verbose : bool
            Print progress

        Returns
        -------
        history : dict
            Training history
        """
        # Validation
        if self.task == "classification" and labels is None:
            raise ValueError("Labels required for classification task")

        if self.task == "prediction":
            # For prediction, labels are next timestep
            if labels is None:
                labels = sequences[:, 1:, :]
                sequences = sequences[:, :-1, :]

        # Split data
        n_samples = len(sequences)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_sequences = sequences[train_idx]
        val_sequences = sequences[val_idx]

        if labels is not None:
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]
        else:
            train_labels = None
            val_labels = None

        # Create data loaders
        if train_labels is not None:
            train_dataset = TensorDataset(
                torch.FloatTensor(train_sequences),
                torch.FloatTensor(train_labels) if self.task == "prediction"
                else torch.LongTensor(train_labels)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(val_sequences),
                torch.FloatTensor(val_labels) if self.task == "prediction"
                else torch.LongTensor(val_labels)
            )
        else:
            # For embedding task (unsupervised)
            train_dataset = TensorDataset(torch.FloatTensor(train_sequences))
            val_dataset = TensorDataset(torch.FloatTensor(val_sequences))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                x = batch[0].to(self.device)

                if len(batch) > 1:
                    y = batch[1].to(self.device)
                else:
                    # Unsupervised: reconstruct input
                    y = x

                self.optimizer.zero_grad()
                output, _ = self.model(x)

                if self.task == "prediction":
                    loss = self.criterion(output, y)
                elif self.task == "classification":
                    loss = self.criterion(output, y)
                else:
                    # Embedding: reconstruction loss
                    loss = self.criterion(output, torch.mean(x, dim=1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_loss += loss.item() * len(x)

            train_loss /= len(train_dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(self.device)

                    if len(batch) > 1:
                        y = batch[1].to(self.device)
                    else:
                        y = x

                    output, _ = self.model(x)

                    if self.task == "prediction":
                        loss = self.criterion(output, y)
                    elif self.task == "classification":
                        loss = self.criterion(output, y)
                    else:
                        loss = self.criterion(output, torch.mean(x, dim=1))

                    val_loss += loss.item() * len(x)

            val_loss /= len(val_dataset)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(self.best_model_state)
        self.trained = True

        return history

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        Make predictions on sequences.

        Parameters
        ----------
        sequences : np.ndarray
            Input sequences of shape (N, seq_len, input_dim)

        Returns
        -------
        predictions : np.ndarray
            Predictions (task-dependent shape)
        """
        if not self.trained:
            raise RuntimeError("Model must be trained first")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                x = torch.FloatTensor(batch).to(self.device)
                output, _ = self.model(x)
                predictions.append(output.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

        # For classification, return class predictions
        if self.task == "classification":
            predictions = np.argmax(predictions, axis=1)

        return predictions

    def get_embeddings(self, sequences: np.ndarray) -> np.ndarray:
        """
        Get learned embeddings for sequences.

        Parameters
        ----------
        sequences : np.ndarray
            Input sequences of shape (N, seq_len, input_dim)

        Returns
        -------
        embeddings : np.ndarray
            Sequence embeddings of shape (N, hidden_dim)
        """
        if not self.trained:
            raise RuntimeError("Model must be trained first")

        # Temporarily switch to embedding mode
        original_task = self.model.task
        self.model.task = "embedding"

        embeddings = self.predict(sequences)

        self.model.task = original_task

        return embeddings

    def save(self, filepath: str):
        """Save model to disk."""
        if not self.trained:
            raise RuntimeError("Cannot save untrained model")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'task': self.task,
            'learning_rate': self.learning_rate
        }, filepath)

    def load(self, filepath: str):
        """Load model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)

        # Recreate model
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.output_dim = checkpoint['output_dim']
        self.task = checkpoint['task']
        self.learning_rate = checkpoint['learning_rate']

        self.model = BehaviorLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            task=self.task
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.trained = True
