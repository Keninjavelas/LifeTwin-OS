"""Enhanced training script for the sequence behavior model (next-app predictor).

Features:
- Loads real Android data from comprehensive export
- Advanced preprocessing with app categorization
- Improved model architecture with attention mechanism
- Comprehensive evaluation metrics
- ONNX export with quantization
- Model versioning and metadata tracking
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Any
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppCategorizer:
    """Categorizes apps into semantic groups for better generalization."""
    
    CATEGORIES = {
        'social': ['com.facebook', 'com.instagram', 'com.twitter', 'com.snapchat', 
                  'com.whatsapp', 'com.telegram', 'com.discord'],
        'productivity': ['com.microsoft.office', 'com.google.android.apps.docs',
                        'com.adobe.reader', 'com.evernote', 'com.todoist'],
        'entertainment': ['com.netflix', 'com.youtube', 'com.spotify', 'com.amazon.avod',
                         'com.hulu.plus', 'com.disney.disneyplus'],
        'games': ['com.king.candycrushsaga', 'com.supercell.clashofclans',
                 'com.rovio.angrybirdsstella', 'com.mojang.minecraftpe'],
        'shopping': ['com.amazon.mshop.android.shopping', 'com.ebay.mobile',
                    'com.etsy.android', 'com.shopify.mobile'],
        'news': ['com.cnn.mobile.android.phone', 'com.nytimes.android',
                'com.bbc.news', 'com.reddit.frontpage'],
        'finance': ['com.paypal.android.p2pmobile', 'com.chase.sig.android',
                   'com.bankofamerica.digitalwallet', 'com.robinhood.android'],
        'health': ['com.myfitnesspal.android', 'com.fitbit.FitbitMobile',
                  'com.nike.plusgps', 'com.headspace.android'],
        'travel': ['com.google.android.apps.maps', 'com.uber.m',
                  'com.airbnb.android', 'com.booking'],
        'system': ['com.android.settings', 'com.android.systemui',
                  'com.google.android.gms', 'com.android.chrome']
    }
    
    def categorize_app(self, package_name: str) -> str:
        """Categorize an app by its package name."""
        for category, patterns in self.CATEGORIES.items():
            for pattern in patterns:
                if pattern in package_name:
                    return category
        
        # Fallback categorization based on common patterns
        if 'game' in package_name.lower():
            return 'games'
        elif 'music' in package_name.lower() or 'video' in package_name.lower():
            return 'entertainment'
        elif 'bank' in package_name.lower() or 'pay' in package_name.lower():
            return 'finance'
        elif 'news' in package_name.lower():
            return 'news'
        else:
            return 'other'


class AttentionNextAppModel(nn.Module):
    """Enhanced next-app prediction model with attention mechanism."""
    
    def __init__(self, num_apps: int, embed_dim: int = 64, hidden_dim: int = 128, 
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_apps = num_apps
        
        # Embedding layer
        self.embed = nn.Embedding(num_apps, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(100, embed_dim)  # Positional embedding
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_apps)
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Create positional indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.embed(x)
        pos_emb = self.pos_embed(positions)
        embeddings = token_emb + pos_emb
        
        # Create attention mask for padding
        if mask is None:
            mask = (x == 0)  # Padding mask
        
        # Transformer encoding
        encoded = self.transformer(embeddings, src_key_padding_mask=mask)
        
        # Use the last non-padded token for prediction
        if mask is not None:
            # Find the last non-padded position for each sequence
            seq_lengths = (~mask).sum(dim=1) - 1
            last_hidden = encoded[torch.arange(batch_size), seq_lengths]
        else:
            last_hidden = encoded[:, -1, :]
        
        # Output projection
        output = self.layer_norm(last_hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        
        return logits


class SequenceDataset:
    """Enhanced dataset class for sequence data with preprocessing."""
    
    def __init__(self, sequences: List[List[str]], app_to_id: Dict[str, int], 
                 max_seq_length: int = 50, use_categories: bool = False):
        self.sequences = sequences
        self.app_to_id = app_to_id
        self.max_seq_length = max_seq_length
        self.use_categories = use_categories
        self.categorizer = AppCategorizer() if use_categories else None
        
        self.X, self.y, self.masks = self._prepare_data()
    
    def _prepare_data(self):
        X, y, masks = [], [], []
        
        for sequence in self.sequences:
            if len(sequence) < 2:
                continue
                
            # Convert apps to IDs
            if self.use_categories:
                # Use app categories instead of specific apps
                app_ids = []
                for app in sequence:
                    category = self.categorizer.categorize_app(app)
                    if category in self.app_to_id:
                        app_ids.append(self.app_to_id[category])
                    else:
                        app_ids.append(self.app_to_id.get('other', 1))
            else:
                app_ids = [self.app_to_id.get(app, 1) for app in sequence]  # 1 for UNK
            
            if len(app_ids) < 2:
                continue
            
            # Create input-target pairs
            for i in range(1, len(app_ids)):
                input_seq = app_ids[:i]
                target = app_ids[i]
                
                # Pad or truncate sequence
                if len(input_seq) > self.max_seq_length:
                    input_seq = input_seq[-self.max_seq_length:]
                
                # Create padding mask
                mask = [False] * len(input_seq) + [True] * (self.max_seq_length - len(input_seq))
                
                # Pad sequence
                input_seq = input_seq + [0] * (self.max_seq_length - len(input_seq))
                
                X.append(input_seq)
                y.append(target)
                masks.append(mask)
        
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def load_real_data(data_path: str) -> Tuple[List[List[str]], Dict[str, Any]]:
    """Load real Android data from comprehensive export."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return [], {}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract sequences from the comprehensive export
    if 'sequence_data' in data and 'sequences' in data['sequence_data']:
        sequences = data['sequence_data']['sequences']
        metadata = {
            'total_sequences': len(sequences),
            'vocabulary_size': len(data['sequence_data'].get('vocabulary', {})),
            'export_timestamp': data.get('export_timestamp'),
            'days_back': data.get('days_back', 30)
        }
    elif 'sequences' in data:
        # Fallback to legacy format
        sequences = data['sequences']
        metadata = {
            'total_sequences': len(sequences),
            'vocabulary_size': len(data.get('vocabulary', {})),
            'device_id': data.get('device_id')
        }
    else:
        logger.error("No sequence data found in export file")
        return [], {}
    
    logger.info(f"Loaded {len(sequences)} sequences from {data_path}")
    return sequences, metadata


def build_vocabulary(sequences: List[List[str]], min_frequency: int = 3, 
                    use_categories: bool = False) -> Dict[str, int]:
    """Build vocabulary from sequences with frequency filtering."""
    if use_categories:
        categorizer = AppCategorizer()
        # Count categories instead of individual apps
        category_counts = Counter()
        for sequence in sequences:
            for app in sequence:
                category = categorizer.categorize_app(app)
                category_counts[category] += 1
        
        # Build category vocabulary
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for category, count in category_counts.items():
            if count >= min_frequency:
                vocab[category] = len(vocab)
        
        logger.info(f"Built category vocabulary with {len(vocab)} categories")
        return vocab
    else:
        # Count individual apps
        app_counts = Counter()
        for sequence in sequences:
            for app in sequence:
                app_counts[app] += 1
        
        # Build app vocabulary
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for app, count in app_counts.items():
            if count >= min_frequency:
                vocab[app] = len(vocab)
        
        logger.info(f"Built app vocabulary with {len(vocab)} apps")
        return vocab


def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_x, batch_y, batch_mask in test_loader:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
    avg_loss = total_loss / len(test_loader)
    
    # Top-k accuracy
    top3_correct = 0
    top5_correct = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_mask in test_loader:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            logits = model(batch_x, batch_mask)
            
            # Top-3 accuracy
            _, top3_preds = torch.topk(logits, 3, dim=1)
            top3_correct += (top3_preds == batch_y.unsqueeze(1)).any(dim=1).sum().item()
            
            # Top-5 accuracy
            _, top5_preds = torch.topk(logits, min(5, logits.size(1)), dim=1)
            top5_correct += (top5_preds == batch_y.unsqueeze(1)).any(dim=1).sum().item()
    
    total_samples = len(all_targets)
    top3_accuracy = top3_correct / total_samples if total_samples > 0 else 0.0
    top5_accuracy = top5_correct / total_samples if total_samples > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': avg_loss,
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy
    }


def export_to_onnx(model, vocab_size, max_seq_length, output_path, quantize=False):
    """Export model to ONNX format with optional quantization."""
    try:
        model.eval()
        
        # Create dummy input
        dummy_input = torch.zeros(1, max_seq_length, dtype=torch.long)
        dummy_mask = torch.zeros(1, max_seq_length, dtype=torch.bool)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input, dummy_mask),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
        
        # Quantization
        if quantize:
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                quantized_path = output_path.with_suffix('.quantized.onnx')
                quantize_dynamic(
                    str(output_path),
                    str(quantized_path),
                    weight_type=QuantType.QUInt8
                )
                logger.info(f"Quantized model saved to: {quantized_path}")
                return quantized_path
            except ImportError:
                logger.warning("ONNX quantization not available. Install onnxruntime-tools for quantization.")
        
        return output_path
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train enhanced next-app prediction model")
    parser.add_argument("--data-path", default="ml/data/comprehensive_export.json",
                       help="Path to exported Android data")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=64,
                       help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128,
                       help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=4,
                       help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--max-seq-length", type=int, default=50,
                       help="Maximum sequence length")
    parser.add_argument("--min-frequency", type=int, default=3,
                       help="Minimum app frequency for vocabulary")
    parser.add_argument("--use-categories", action="store_true",
                       help="Use app categories instead of specific apps")
    parser.add_argument("--export-onnx", action="store_true",
                       help="Export model to ONNX format")
    parser.add_argument("--quantize", action="store_true",
                       help="Quantize ONNX model")
    parser.add_argument("--model-dir", default="ml/models",
                       help="Directory to save models")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    sequences, metadata = load_real_data(args.data_path)
    if not sequences:
        logger.error("No training data available")
        return
    
    # Build vocabulary
    vocab = build_vocabulary(sequences, args.min_frequency, args.use_categories)
    if len(vocab) < 3:  # At least PAD, UNK, and one real token
        logger.error("Vocabulary too small for training")
        return
    
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset
    dataset = SequenceDataset(sequences, vocab, args.max_seq_length, args.use_categories)
    
    if len(dataset.X) == 0:
        logger.error("No training examples created")
        return
    
    logger.info(f"Created {len(dataset.X)} training examples")
    
    # Train/test split
    total_size = len(dataset.X)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    train_dataset = TensorDataset(dataset.X[:train_size], dataset.y[:train_size], dataset.masks[:train_size])
    test_dataset = TensorDataset(dataset.X[train_size:], dataset.y[train_size:], dataset.masks[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = AttentionNextAppModel(
        num_apps=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    best_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        
        # Evaluation
        eval_metrics = evaluate_model(model, test_loader, device)
        
        # Learning rate scheduling
        scheduler.step(eval_metrics['loss'])
        
        # Save best model
        if eval_metrics['accuracy'] > best_accuracy:
            best_accuracy = eval_metrics['accuracy']
            best_model_state = model.state_dict().copy()
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {eval_metrics['loss']:.4f} | "
            f"Accuracy: {eval_metrics['accuracy']:.3f} | "
            f"Top-3: {eval_metrics['top3_accuracy']:.3f} | "
            f"Top-5: {eval_metrics['top5_accuracy']:.3f} | "
            f"F1: {eval_metrics['f1_score']:.3f}"
        )
    
    # Save best model
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    # Save PyTorch model
    model_path = model_dir / "next_app_model.pt"
    torch.save({
        'model_state_dict': best_model_state,
        'model_config': {
            'num_apps': len(vocab),
            'embed_dim': args.embed_dim,
            'hidden_dim': args.hidden_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'max_seq_length': args.max_seq_length
        },
        'vocabulary': vocab,
        'training_args': vars(args),
        'metadata': metadata
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Save vocabulary
    vocab_path = model_dir / "vocab.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    logger.info(f"Vocabulary saved to {vocab_path}")
    
    # Save training metadata
    metadata_path = model_dir / "next_app_model.json"
    training_metadata = {
        'model_name': 'next_app_model',
        'model_type': 'sequence_prediction',
        'framework': 'pytorch',
        'architecture': 'transformer',
        'vocab_size': len(vocab),
        'max_seq_length': args.max_seq_length,
        'use_categories': args.use_categories,
        'training_sequences': len(sequences),
        'training_examples': len(dataset.X),
        'best_accuracy': float(best_accuracy),
        'final_metrics': {k: float(v) for k, v in eval_metrics.items()},
        'training_args': vars(args),
        'data_metadata': metadata,
        'saved_at': __import__('datetime').datetime.utcnow().isoformat() + 'Z'
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(training_metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Save metrics
    metrics_path = model_dir / "next_app_model.metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_accuracy': float(best_accuracy),
            'final_metrics': {k: float(v) for k, v in eval_metrics.items()},
            'training_metadata': training_metadata
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # ONNX export
    if args.export_onnx:
        onnx_path = model_dir / "next_app_model.onnx"
        exported_path = export_to_onnx(model, len(vocab), args.max_seq_length, onnx_path, args.quantize)
        if exported_path:
            logger.info(f"ONNX model exported successfully")
    
    logger.info("Training completed successfully!")
    logger.info(f"Best accuracy: {best_accuracy:.3f}")


if __name__ == "__main__":
    main()
