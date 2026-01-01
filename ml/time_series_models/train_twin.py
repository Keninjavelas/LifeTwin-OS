"""
Enhanced time-series forecasting model for the digital twin.

This script trains advanced time-series models to predict multiple behavioral metrics:
- Screen time patterns
- Energy/focus/mood levels
- App usage intensity
- Notification patterns

Features:
- Multi-target forecasting
- Feature engineering from raw behavioral data
- Advanced models (LSTM, GRU, Transformer)
- Comprehensive evaluation metrics
- Model export for on-device inference
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import argparse
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Using scikit-learn models only.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngineer:
    """Advanced feature engineering for time-series behavioral data."""
    
    def __init__(self):
        self.scalers = {}
    
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """Create temporal features from date column."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic temporal features
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for temporal features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_cols: List[str], 
                           lags: List[int] = [1, 2, 3, 7, 14]) -> pd.DataFrame:
        """Create lagged features for target variables."""
        df = df.copy()
        
        for col in target_cols:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_cols: List[str],
                               windows: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        """Create rolling window features."""
        df = df.copy()
        
        for col in target_cols:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
        
        return df
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral pattern features."""
        df = df.copy()
        
        # Screen time patterns
        if 'total_screen_time' in df.columns:
            df['screen_time_normalized'] = df['total_screen_time'] / (24 * 60)  # Normalize by max possible minutes
            df['screen_time_category'] = pd.cut(df['total_screen_time'], 
                                               bins=[0, 120, 300, 480, float('inf')], 
                                               labels=['low', 'medium', 'high', 'very_high'])
        
        # Interaction intensity patterns
        if 'interaction_intensity' in df.columns:
            df['interaction_high'] = (df['interaction_intensity'] > df['interaction_intensity'].quantile(0.75)).astype(int)
            df['interaction_low'] = (df['interaction_intensity'] < df['interaction_intensity'].quantile(0.25)).astype(int)
        
        # Notification patterns
        if 'notification_count' in df.columns:
            df['notification_rate'] = df['notification_count'] / df.get('total_screen_time', 1).replace(0, 1)
            df['high_notification_day'] = (df['notification_count'] > df['notification_count'].quantile(0.8)).astype(int)
        
        # Energy/focus/mood stability
        for metric in ['energy_level', 'focus_level', 'mood_level']:
            if metric in df.columns:
                df[f'{metric}_stable'] = (abs(df[metric] - df[metric].shift(1)) < 0.1).astype(int)
                df[f'{metric}_trend'] = df[metric] - df[metric].shift(1)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        logger.info("Starting feature engineering...")
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create lag features
        df = self.create_lag_features(df, target_cols)
        
        # Create rolling features
        df = self.create_rolling_features(df, target_cols)
        
        # Create behavioral features
        df = self.create_behavioral_features(df)
        
        # Drop rows with too many NaN values (from lag/rolling features)
        df = df.dropna(thresh=len(df.columns) * 0.7)  # Keep rows with at least 70% non-null values
        
        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        return df
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray, 
                      feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['features'] = scaler
        return X_train_scaled, X_test_scaled
    
    def scale_targets(self, y_train: np.ndarray, y_test: np.ndarray, 
                     target_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Scale target variables."""
        scaler = MinMaxScaler()
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, len(target_names)))
        y_test_scaled = scaler.transform(y_test.reshape(-1, len(target_names)))
        
        self.scalers['targets'] = scaler
        return y_train_scaled, y_test_scaled


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time-series data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 14):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.sequence_length],
            self.y[idx + self.sequence_length - 1]
        )


class LSTMForecaster(nn.Module):
    """LSTM-based multi-target forecasting model."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last time step
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and final linear layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class TransformerForecaster(nn.Module):
    """Transformer-based forecasting model."""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int,
                 output_size: int, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer encoding
        encoded = self.transformer(x)
        
        # Use the last time step
        last_output = encoded[:, -1, :]
        
        # Output projection
        output = self.dropout(last_output)
        output = self.output_projection(output)
        
        return output


def load_timeseries_data(data_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load time-series data from comprehensive export."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return pd.DataFrame(), {}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract daily summaries
    if 'daily_summaries' in data:
        summaries = data['daily_summaries']
    elif 'summaries' in data:
        summaries = data['summaries']
    else:
        logger.error("No daily summaries found in export file")
        return pd.DataFrame(), {}
    
    if not summaries:
        logger.error("No summary data available")
        return pd.DataFrame(), {}
    
    # Convert to DataFrame
    df = pd.DataFrame(summaries)
    
    # Ensure required columns exist with defaults
    required_cols = {
        'total_screen_time': 0,
        'total_unlocks': 0,
        'notification_count': 0,
        'interaction_intensity': 0.5,
        'energy_level': 0.5,
        'focus_level': 0.5,
        'mood_level': 0.5
    }
    
    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val
    
    # Convert timestamp to date
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], unit='ms')
    else:
        # Create synthetic dates
        df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    metadata = {
        'total_days': len(df),
        'date_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat()
        },
        'export_timestamp': data.get('export_timestamp'),
        'available_metrics': list(df.columns)
    }
    
    logger.info(f"Loaded {len(df)} days of time-series data")
    return df, metadata


def evaluate_forecasting_model(y_true: np.ndarray, y_pred: np.ndarray, 
                              target_names: List[str]) -> Dict[str, float]:
    """Comprehensive evaluation of forecasting model."""
    metrics = {}
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    for i, target in enumerate(target_names):
        true_vals = y_true[:, i] if y_true.shape[1] > i else y_true[:, 0]
        pred_vals = y_pred[:, i] if y_pred.shape[1] > i else y_pred[:, 0]
        
        # Remove any NaN values
        mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
        true_vals = true_vals[mask]
        pred_vals = pred_vals[mask]
        
        if len(true_vals) > 0:
            metrics[f'{target}_mae'] = mean_absolute_error(true_vals, pred_vals)
            metrics[f'{target}_mse'] = mean_squared_error(true_vals, pred_vals)
            metrics[f'{target}_rmse'] = np.sqrt(metrics[f'{target}_mse'])
            metrics[f'{target}_r2'] = r2_score(true_vals, pred_vals)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((true_vals - pred_vals) / np.maximum(np.abs(true_vals), 1e-8))) * 100
            metrics[f'{target}_mape'] = mape
    
    # Overall metrics
    overall_mae = np.mean([metrics[k] for k in metrics.keys() if k.endswith('_mae')])
    overall_r2 = np.mean([metrics[k] for k in metrics.keys() if k.endswith('_r2')])
    
    metrics['overall_mae'] = overall_mae
    metrics['overall_r2'] = overall_r2
    
    return metrics


def train_sklearn_model(X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray,
                       target_names: List[str], model_type: str = 'rf') -> Tuple[Any, Dict[str, float]]:
    """Train scikit-learn model for forecasting."""
    logger.info(f"Training {model_type} model...")
    
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_forecasting_model(y_test, y_pred, target_names)
    
    return model, metrics


def train_pytorch_model(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       target_names: List[str], model_type: str = 'lstm',
                       epochs: int = 100, batch_size: int = 32,
                       sequence_length: int = 14) -> Tuple[Any, Dict[str, float]]:
    """Train PyTorch model for forecasting."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available for deep learning models")
    
    logger.info(f"Training {model_type} model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
    test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[1]
    output_size = len(target_names)
    
    if model_type == 'lstm':
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            output_size=output_size,
            dropout=0.2
        )
    elif model_type == 'transformer':
        model = TransformerForecaster(
            input_size=input_size,
            d_model=64,
            nhead=4,
            num_layers=2,
            output_size=output_size,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    model.eval()
    
    # Make predictions on test set
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    y_pred = np.vstack(all_predictions)
    y_true = np.vstack(all_targets)
    
    # Evaluate
    metrics = evaluate_forecasting_model(y_true, y_pred, target_names)
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train enhanced time-series forecasting model")
    parser.add_argument("--data-path", default="ml/data/comprehensive_export.json",
                       help="Path to exported Android data")
    parser.add_argument("--model-type", choices=['rf', 'lstm', 'transformer'], default='rf',
                       help="Type of model to train")
    parser.add_argument("--target-metrics", nargs='+', 
                       default=['total_screen_time', 'energy_level', 'focus_level', 'mood_level'],
                       help="Target metrics to forecast")
    parser.add_argument("--sequence-length", type=int, default=14,
                       help="Sequence length for deep learning models")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs for deep learning models")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for deep learning models")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proportion of data for testing")
    parser.add_argument("--model-dir", default="ml/models",
                       help="Directory to save models")
    
    args = parser.parse_args()
    
    # Load data
    df, metadata = load_timeseries_data(args.data_path)
    if df.empty:
        logger.error("No data available for training")
        return
    
    # Feature engineering
    feature_engineer = TimeSeriesFeatureEngineer()
    df_features = feature_engineer.prepare_features(df, args.target_metrics)
    
    if df_features.empty:
        logger.error("No features available after preprocessing")
        return
    
    # Prepare features and targets
    feature_cols = [col for col in df_features.columns 
                   if col not in ['date'] + args.target_metrics]
    
    X = df_features[feature_cols].values
    y = df_features[args.target_metrics].values
    
    # Remove any remaining NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    X = X[mask]
    y = y[mask]
    
    if len(X) < 20:
        logger.error("Insufficient data for training after preprocessing")
        return
    
    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Train/test split
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    X_train_scaled, X_test_scaled = feature_engineer.scale_features(X_train, X_test, feature_cols)
    
    # Scale targets for deep learning models
    if args.model_type in ['lstm', 'transformer']:
        y_train_scaled, y_test_scaled = feature_engineer.scale_targets(y_train, y_test, args.target_metrics)
    else:
        y_train_scaled, y_test_scaled = y_train, y_test
    
    # Train model
    if args.model_type == 'rf':
        model, metrics = train_sklearn_model(
            X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, args.target_metrics, args.model_type
        )
    else:
        model, metrics = train_pytorch_model(
            X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, args.target_metrics,
            args.model_type, args.epochs, args.batch_size, args.sequence_length
        )
    
    # Save model
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model_type == 'rf':
        # Save scikit-learn model
        model_path = model_dir / "time_series_twin.joblib"
        model_obj = {
            'model': model,
            'feature_engineer': feature_engineer,
            'target_metrics': args.target_metrics,
            'feature_columns': feature_cols,
            'model_type': args.model_type
        }
        joblib.dump(model_obj, model_path)
        logger.info(f"Model saved to {model_path}")
    else:
        # Save PyTorch model
        model_path = model_dir / f"time_series_twin_{args.model_type}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'model_type': args.model_type,
                'input_size': X_train_scaled.shape[1],
                'output_size': len(args.target_metrics),
                'sequence_length': args.sequence_length
            },
            'feature_engineer': feature_engineer,
            'target_metrics': args.target_metrics,
            'feature_columns': feature_cols
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    
    # Save metadata
    metadata_path = model_dir / "time_series_twin.json"
    training_metadata = {
        'model_name': 'time_series_twin',
        'model_type': args.model_type,
        'framework': 'pytorch' if args.model_type in ['lstm', 'transformer'] else 'scikit-learn',
        'target_metrics': args.target_metrics,
        'feature_count': len(feature_cols),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'sequence_length': args.sequence_length if args.model_type in ['lstm', 'transformer'] else None,
        'metrics': {k: float(v) for k, v in metrics.items()},
        'training_args': vars(args),
        'data_metadata': metadata,
        'saved_at': datetime.utcnow().isoformat() + 'Z'
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(training_metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Save metrics
    metrics_path = model_dir / "time_series_twin.metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'overall_mae': float(metrics.get('overall_mae', 0)),
            'overall_r2': float(metrics.get('overall_r2', 0)),
            'detailed_metrics': {k: float(v) for k, v in metrics.items()},
            'training_metadata': training_metadata
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Print results
    logger.info("Training completed successfully!")
    logger.info(f"Overall MAE: {metrics.get('overall_mae', 0):.4f}")
    logger.info(f"Overall R²: {metrics.get('overall_r2', 0):.4f}")
    
    for target in args.target_metrics:
        mae = metrics.get(f'{target}_mae', 0)
        r2 = metrics.get(f'{target}_r2', 0)
        logger.info(f"{target}: MAE={mae:.4f}, R²={r2:.4f}")


if __name__ == "__main__":
    main()
