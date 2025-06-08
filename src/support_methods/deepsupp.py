import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
from hmmlearn import hmm
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.support_methods.percentile import percentile_support


class MultiHeadAttentionAutoencoder(nn.Module):
    """
    A lighter version of a multi-head attention autoencoder for support level detection.
    """

    def __init__(
        self, sequence_length, num_heads=4, hidden_dim=32
    ):  # Reduced hidden_dim from 64 to 32
        super(MultiHeadAttentionAutoencoder, self).__init__()
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Ensure embed_dim is divisible by num_heads
        assert (
            sequence_length % num_heads == 0
        ), f"embed_dim ({sequence_length}) must be divisible by num_heads ({num_heads})"

        # LIGHTER Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=sequence_length,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,  # Removed dropout for speed
            bias=False,  # Remove bias for fewer parameters
            add_zero_attn=False,  # Disable zero attention
            kdim=None,  # Use same dim for K, V, Q
            vdim=None,
        )

        # LIGHTER Layer normalization (optional, can be removed)
        self.layer_norm = nn.LayerNorm(
            sequence_length, elementwise_affine=False
        )  # No learnable params

        # LIGHTER Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim, bias=False),  # No bias
            nn.ReLU(inplace=True),  # In-place operation
            nn.Linear(
                hidden_dim, hidden_dim // 2, bias=False
            ),  # No bias, smaller dimension
            nn.ReLU(inplace=True),
        )

        # LIGHTER Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim, bias=False),  # No bias
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, sequence_length, bias=False),  # No bias
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Apply multi-head attention (SAME LOGIC)
        # x shape: (batch_size, sequence_length, sequence_length)
        attn_output, _ = self.multihead_attn(
            x, x, x, need_weights=False
        )  # Don't compute weights

        # Add residual connection and layer norm (SAME LOGIC)
        attn_output = self.layer_norm(attn_output + x)

        # Apply encoder to each time step (SAME LOGIC)
        encoded = []
        for i in range(self.sequence_length):
            enc_step = self.encoder(attn_output[:, i, :])
            encoded.append(enc_step)

        encoded = torch.stack(encoded, dim=1)  # (batch_size, seq_len, hidden_dim//2)

        # Apply decoder to reconstruct (SAME LOGIC)
        decoded = []
        for i in range(self.sequence_length):
            dec_step = self.decoder(encoded[:, i, :])
            decoded.append(dec_step)

        reconstruction = torch.stack(decoded, dim=1)  # (batch_size, seq_len, seq_len)

        return reconstruction, encoded

    def get_embeddings(self, x):
        with torch.no_grad():
            _, encoded = self.forward(x)
            # Global average pooling across sequence dimension (SAME LOGIC)
            embeddings = encoded.mean(dim=1)  # (batch_size, hidden_dim//2)
            return embeddings


def deepsupp(price_data, window=20, num_levels=7):
    """
    Identify support levels using deep learning autoencoder with correlation analysis.
    Uses lighter PyTorch multi-head attention with same logic.
    """
    try:

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert price_data to DataFrame format expected by the algorithm
        if isinstance(price_data, pd.Series):
            data = pd.DataFrame({"Close": price_data})
        else:
            data = price_data.copy()

        # Calculate required features exactly as in original implementation
        if "Volume" not in data.columns:
            data["Volume"] = 100000  # Default volume if not available

        data["VWAP"] = (data["Close"] * data["Volume"]).cumsum() / data[
            "Volume"
        ].cumsum()
        data["PriceChangeVolume"] = (
            data["Close"].pct_change().fillna(0) * data["Volume"]
        )
        data["VolumeRatio"] = data["Volume"] / data["Volume"].rolling(20).mean()
        data.dropna(inplace=True)

        features = ["Close", "VWAP", "Volume", "PriceChangeVolume", "VolumeRatio"]

        # Create and fit scaler with the data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[features])

        # Compute correlation matrix (Spearman) for the data
        corr_matrix, _ = spearmanr(data_scaled, axis=1)

        # Fix sequence length to be divisible by num_heads
        sequence_length = 32  # Changed from 30 to 32 (divisible by 4)
        num_heads = 4

        X = []
        for i in range(len(corr_matrix) - sequence_length):
            # Pad or trim correlation matrix to match sequence_length
            corr_slice = corr_matrix[i : i + sequence_length, i : i + sequence_length]
            if corr_slice.shape[0] < sequence_length:
                # Pad with zeros if too small
                padded = np.zeros((sequence_length, sequence_length))
                padded[: corr_slice.shape[0], : corr_slice.shape[1]] = corr_slice
                X.append(padded)
            else:
                # Trim if too large
                X.append(corr_slice[:sequence_length, :sequence_length])

        if len(X) == 0:
            return percentile_support(price_data, num_levels=num_levels)

        X = np.array(X)
        X_tensor = torch.FloatTensor(X).to(device)

        # Initialize model with compatible dimensions (LIGHTER)
        model = MultiHeadAttentionAutoencoder(sequence_length, num_heads).to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=0.01, weight_decay=0
        )  # Higher LR, no weight decay
        criterion = nn.MSELoss()

        # Create DataLoader (SAME LOGIC)
        dataset = TensorDataset(
            X_tensor, X_tensor
        )  # Input and target are the same (autoencoder)
        dataloader = DataLoader(
            dataset, batch_size=min(32, len(X_tensor)), shuffle=False
        )  # Larger batch, no shuffle

        # Train the autoencoder (LIGHTER but SAME LOGIC)
        model.train()
        epochs = 10  # Reduced epochs from 20 to 10
        for epoch in range(epochs):
            for batch_data, batch_target in dataloader:
                optimizer.zero_grad()

                reconstruction, _ = model(batch_data)
                loss = criterion(reconstruction, batch_target)

                loss.backward()
                optimizer.step()

        # Extract embeddings from trained model (SAME LOGIC)
        model.eval()
        with torch.no_grad():
            embeddings = model.get_embeddings(X_tensor)
            embeddings_np = embeddings.cpu().numpy()

        # Clustering embeddings (SAME LOGIC)
        if len(embeddings_np) < 3:
            return percentile_support(price_data, num_levels=num_levels)

        clustering = DBSCAN(eps=0.1, min_samples=max(2, len(embeddings_np) // 10)).fit(
            embeddings_np
        )

        # Count the number of clusters found (SAME LOGIC)
        unique_labels = np.unique(clustering.labels_)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        # For each cluster, calculate the median price level (SAME LOGIC)
        support_levels = []
        for cluster_label in np.unique(clustering.labels_):
            if cluster_label == -1:
                continue  # Skip outliers

            # Get points in this cluster
            cluster_indices = np.where(clustering.labels_ == cluster_label)[0]

            # Map back to original data indices
            valid_indices = []
            for idx in cluster_indices:
                if idx < len(data) - sequence_length:
                    valid_indices.append(idx + sequence_length)

            if valid_indices:
                cluster_prices = data["Close"].iloc[valid_indices]
                if len(cluster_prices) > 0:
                    support_level = np.median(cluster_prices)
                    support_levels.append(support_level)

        # Add nearest support level info (SAME LOGIC)
        if support_levels:
            support_levels = sorted(support_levels)
            # Ensure we have enough levels
            while len(support_levels) < num_levels:
                # Add levels based on percentiles if needed
                percentile_val = np.percentile(
                    data["Close"], 5 + len(support_levels) * 5
                )
                support_levels.append(percentile_val)
        else:
            # If no clusters found, fall back to percentile method
            return percentile_support(price_data, num_levels=num_levels)

        return [round(level, 2) for level in support_levels[:num_levels]]

    except Exception as e:
        print(f"Deep support method failed, falling back to percentile method: {e}")
        return percentile_support(price_data, num_levels=num_levels)
