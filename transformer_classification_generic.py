import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from tqdm import tqdm
from torchinfo import summary
import warnings

# Configuration dictionary - modify these parameters as needed
CONFIG = {
    # Data Configuration (REQUIRED - modify these for your dataset)
    'data_path': 'transactions.xlsx',  # Path to your data file (.xlsx, .xls, or .csv)
    'num_numerical_features': 5,       # Number of numerical features (should be at the beginning)
    'num_categorical_features': 6,     # Number of categorical features (should come after numerical)
    
    # Model Architecture (can be modified for different model sizes)
    # embedding_dim: Controls embedding size for categorical features
    #   - Values: 4, 8, 16, 32, 64
    #   - Smaller (4-8): Faster training, less memory, may underfit
    #   - Larger (16-64): Better representation, more memory, may overfit
    'embedding_dim': 8,
    
    # num_heads: Number of attention heads in transformer
    #   - Values: 2, 4, 8, 16, 32
    #   - Must be divisible by embedding_dim * num_categorical_features
    #   - Smaller (2-4): Faster, less complex attention
    #   - Larger (8-16): Better attention patterns, more parameters
    'num_heads': 8,
    
    # num_layers: Number of transformer layers
    #   - Values: 1, 2, 3, 6, 9, 12, 18
    #   - Smaller (1-3): Fast training, may underfit
    #   - Medium (6-9): Good balance of performance and speed
    #   - Larger (12-18): Better performance, slower training, may overfit
    'num_layers': 6,
    
    # dropout: Dropout rate for regularization
    #   - Values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
    #   - 0.0: No regularization, may overfit
    #   - 0.1-0.2: Light regularization, good for most cases
    #   - 0.3-0.5: Heavy regularization, use if overfitting
    'dropout': 0.2,
    
    # Training Parameters (can be modified for different training strategies)
    # learning_rate: Step size for gradient descent
    #   - Values: 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01
    #   - Smaller (0.0001-0.0005): Slow convergence, stable training
    #   - Medium (0.001-0.002): Good balance, recommended starting point
    #   - Larger (0.005-0.01): Fast convergence, may be unstable
    'learning_rate': 0.001,
    
    # num_epochs: Number of complete passes through training data
    #   - Values: 5, 10, 20, 30, 50, 75, 100, 150
    #   - Smaller (5-20): Fast training, may underfit
    #   - Medium (30-75): Good balance, monitor for overfitting
    #   - Larger (100+): Thorough training, risk of overfitting
    'num_epochs': 5,
    
    # batch_size: Number of samples per training batch
    #   - Values: 16, 32, 64, 128, 256, 512
    #   - Smaller (16-32): More stable gradients, slower training
    #   - Medium (64-128): Good balance, recommended
    #   - Larger (256+): Faster training, may need higher learning rate
    'batch_size': 64,
    
    # weight_decay: L2 regularization strength
    #   - Values: 0.0, 0.0001, 0.001, 0.01, 0.1, 1.0
    #   - 0.0: No regularization
    #   - 0.0001-0.001: Light regularization
    #   - 0.01-0.1: Medium regularization, good for most cases
    #   - 1.0+: Heavy regularization, use if overfitting
    'weight_decay': 0.01,
    
    # Data Processing (can be modified for different splitting/resampling strategies)
    # test_size: Fraction of data used for testing
    #   - Values: 0.1, 0.15, 0.2, 0.25, 0.3
    #   - Smaller (0.1-0.15): More training data, less reliable test
    #   - Medium (0.2-0.25): Good balance, recommended
    #   - Larger (0.3): Less training data, more reliable test
    'test_size': 0.2,
    
    # val_size: Fraction of remaining data used for validation
    #   - Values: 0.2, 0.25, 0.3, 0.4
    #   - Smaller (0.2-0.25): More training data, less validation
    #   - Larger (0.3-0.4): More validation, less training data
    'val_size': 0.25,
    
    # smote_ratio: SMOTE oversampling ratio for minority class
    #   - Values: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    #   - 0.1-0.3: Light oversampling, good for slightly imbalanced data
    #   - 0.4-0.6: Medium oversampling, good for moderately imbalanced data
    #   - 0.7-1.0: Heavy oversampling, use for highly imbalanced data
    'smote_ratio': 0.3,
    
    # undersample_ratio: Undersampling ratio for majority class
    #   - Values: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    #   - 0.3-0.5: Heavy undersampling, reduces majority class significantly
    #   - 0.6-0.7: Medium undersampling, good balance
    #   - 0.8-0.9: Light undersampling, keeps most majority samples
    'undersample_ratio': 0.5,
    
    # Loss Function (can be modified for different class imbalance strategies)
    # loss_function: Type of loss function to use
    #   - Values: 'focal', 'bce', 'weighted_bce', 'dice', 'combo'
    #   - 'focal': Focal Loss - focuses on hard examples, good for imbalanced data
    #   - 'bce': Binary Cross Entropy - standard loss function
    #   - 'weighted_bce': Weighted BCE - gives more weight to minority class
    #   - 'dice': Dice Loss - good for imbalanced data, focuses on overlap
    #   - 'combo': Combination of BCE and Dice Loss
    'loss_function': 'focal',
    
    # focal_alpha: Weight for positive class in focal loss
    #   - Values: 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99
    #   - Smaller (0.7-0.8): Less focus on positive class
    #   - Medium (0.85-0.9): Balanced focus, good for most cases
    #   - Larger (0.95-0.99): Heavy focus on positive class, use for severe imbalance
    'focal_alpha': 0.9,
    
    # focal_gamma: Focusing parameter in focal loss
    #   - Values: 0, 1, 2, 3, 4, 5, 6, 7, 8
    #   - 0: No focusing (equivalent to BCE loss)
    #   - 1-3: Light focusing on hard examples
    #   - 4-6: Medium focusing, good for most cases
    #   - 7-8: Heavy focusing, use for very hard classification tasks
    'focal_gamma': 5,
    
    # class_weight: Weight for positive class in weighted BCE
    #   - Values: 0.1, 0.5, 1.0, 2.0, 5.0, 10.0
    #   - 0.1-0.5: Less weight on positive class
    #   - 1.0: Equal weights (standard BCE)
    #   - 2.0-10.0: More weight on positive class, use for imbalanced data
    'class_weight': 2.0,
    
    # Optimizer Configuration
    # optimizer: Type of optimizer to use
    #   - Values: 'adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'adamax'
    #   - 'adam': Adaptive learning rate, good for most cases
    #   - 'adamw': Adam with weight decay, better regularization
    #   - 'sgd': Stochastic Gradient Descent, more stable but slower
    #   - 'rmsprop': Good for non-stationary problems
    #   - 'adagrad': Good for sparse data
    #   - 'adamax': Variant of Adam, sometimes better convergence
    'optimizer': 'adamw',
    
    # learning_rate: Step size for gradient descent
    #   - Values: 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01
    #   - Smaller (0.0001-0.0005): Slow convergence, stable training
    #   - Medium (0.001-0.002): Good balance, recommended starting point
    #   - Larger (0.005-0.01): Fast convergence, may be unstable
    'learning_rate': 0.001,
    
    # weight_decay: L2 regularization strength
    #   - Values: 0.0, 0.0001, 0.001, 0.01, 0.1, 1.0
    #   - 0.0: No regularization
    #   - 0.0001-0.001: Light regularization
    #   - 0.01-0.1: Medium regularization, good for most cases
    #   - 1.0+: Heavy regularization, use if overfitting
    'weight_decay': 0.01,
    
    # momentum: Momentum for SGD optimizer
    #   - Values: 0.0, 0.5, 0.7, 0.9, 0.95, 0.99
    #   - 0.0: No momentum
    #   - 0.5-0.7: Light momentum
    #   - 0.9-0.95: Standard momentum, good for most cases
    #   - 0.99: High momentum, may overshoot
    'momentum': 0.9,
    
    # beta1: Beta1 parameter for Adam-based optimizers
    #   - Values: 0.8, 0.9, 0.95, 0.99
    #   - 0.8-0.9: Less smoothing of gradients
    #   - 0.9-0.95: Standard values, good for most cases
    #   - 0.99: Heavy smoothing, more stable but slower
    'beta1': 0.9,
    
    # beta2: Beta2 parameter for Adam-based optimizers
    #   - Values: 0.9, 0.95, 0.99, 0.999
    #   - 0.9-0.95: Less smoothing of squared gradients
    #   - 0.99-0.999: Standard values, good for most cases
    #   - 0.999: Heavy smoothing, very stable
    'beta2': 0.999,
    
    # Scheduler Configuration
    # scheduler: Type of learning rate scheduler to use
    #   - Values: 'onecycle', 'step', 'cosine', 'exponential', 'plateau', 'none'
    #   - 'onecycle': One-cycle policy, often best performance
    #   - 'step': Step decay, simple and effective
    #   - 'cosine': Cosine annealing, smooth decay
    #   - 'exponential': Exponential decay, aggressive
    #   - 'plateau': Reduce on plateau, adaptive
    #   - 'none': No scheduler, constant learning rate
    'scheduler': 'onecycle',
    
    # scheduler_max_lr: Maximum learning rate for schedulers
    #   - Values: 0.0005, 0.001, 0.002, 0.005, 0.01
    #   - Usually 1-2x the base learning_rate
    #   - Higher for onecycle, lower for others
    'scheduler_max_lr': 0.001,
    
    # scheduler_step_size: Step size for step scheduler
    #   - Values: 5, 10, 15, 20, 30
    #   - Smaller: More frequent decay
    #   - Larger: Less frequent decay
    'scheduler_step_size': 15,
    
    # scheduler_gamma: Decay factor for step/exponential schedulers
    #   - Values: 0.1, 0.3, 0.5, 0.7, 0.9
    #   - Smaller: More aggressive decay
    #   - Larger: Gentler decay
    'scheduler_gamma': 0.5,
    
    # scheduler_patience: Patience for plateau scheduler
    #   - Values: 3, 5, 7, 10, 15
    #   - Smaller: More sensitive to plateaus
    #   - Larger: More tolerant of plateaus
    'scheduler_patience': 7,
    
    # scheduler_factor: Factor for plateau scheduler
    #   - Values: 0.1, 0.3, 0.5, 0.7, 0.9
    #   - Smaller: More aggressive reduction
    #   - Larger: Gentler reduction
    'scheduler_factor': 0.5,
    
    # Threshold Optimization (can be modified for different precision/recall trade-offs)
    # min_recall: Minimum recall threshold for model selection
    #   - Values: 0.8, 0.85, 0.9, 0.95, 0.98, 0.99
    #   - Smaller (0.8-0.85): Higher precision, lower recall
    #   - Medium (0.9-0.95): Balanced precision/recall, recommended
    #   - Larger (0.98-0.99): Higher recall, lower precision
    'min_recall': 0.95,
    
    # threshold_range: Range for threshold search (min, max)
    #   - Values: (0.01, 0.5), (0.01, 0.7), (0.01, 0.81), (0.1, 0.9), (0.2, 0.8)
    #   - Lower range (0.01, 0.5): Focus on high recall
    #   - Medium range (0.01, 0.7-0.81): Balanced search
    #   - Higher range (0.1, 0.9): Focus on high precision
    'threshold_range': (0.01, 0.81),
    
    # threshold_step: Step size for threshold search
    #   - Values: 0.005, 0.01, 0.02, 0.05
    #   - Smaller (0.005-0.01): Fine-grained search, more precise
    #   - Larger (0.02-0.05): Coarse search, faster computation
    'threshold_step': 0.01,
    
    # Reproducibility
    # random_state: Random seed for reproducibility
    #   - Values: Any integer (42, 123, 2024, etc.)
    #   - Use same value for reproducible results
    #   - Change for different random initializations
    'random_state': 42
}

# Set random seeds for reproducibility
torch.manual_seed(CONFIG['random_state'])
np.random.seed(CONFIG['random_state'])

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=None, reduction='mean'):
        """
        Focal Loss: Focuses more on hard-to-classify examples
        alpha: weight for positive class (higher means more focus on positives)
        gamma: focusing parameter (higher means more focus on hard examples)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else CONFIG['focal_alpha']
        self.gamma = gamma if gamma is not None else CONFIG['focal_gamma']
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        
        # Focusing factor: (1-pt)^gamma where pt is the predicted probability of the true class
        pt = torch.exp(-BCE_loss)
        
        # Apply alpha-weighting (giving more weight to positive examples)
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine for final focal loss
        focal_loss = alpha_weight * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Dice Loss: Good for imbalanced data, focuses on overlap between predictions and targets
        smooth: Smoothing factor to prevent division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection
        intersection = (inputs * targets).sum()
        
        # Calculate dice coefficient
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # Return dice loss (1 - dice coefficient)
        return 1 - dice

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        """
        Weighted Binary Cross Entropy Loss
        pos_weight: Weight for positive class
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight if pos_weight is not None else CONFIG['class_weight']

    def forward(self, inputs, targets):
        # Create weight tensor
        weights = torch.ones_like(targets)
        weights[targets == 1] = self.pos_weight
        
        # Calculate weighted BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        weighted_loss = weights * bce_loss
        
        return weighted_loss.mean()

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        Combination of BCE and Dice Loss
        alpha: Weight for BCE loss
        beta: Weight for Dice loss
        """
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        
        return self.alpha * bce_loss + self.beta * dice_loss

def create_loss_function():
    """
    Create loss function based on config
    """
    loss_type = CONFIG['loss_function'].lower()
    
    if loss_type == 'focal':
        return FocalLoss()
    elif loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'weighted_bce':
        return WeightedBCELoss()
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'combo':
        return ComboLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_type}. Available: focal, bce, weighted_bce, dice, combo")

def calculate_categorical_dimensions(X, categorical_features):
    """
    Calculate the number of unique values for each categorical feature.
    This handles cases where categorical values might not start from 0.
    
    Args:
        X (np.ndarray): Feature matrix
        categorical_features (list): List of categorical feature indices
    
    Returns:
        list: List of dimensions for each categorical feature
    """
    cat_dims = []
    
    for col_idx in categorical_features:
        unique_values = np.unique(X[:, col_idx])
        min_val = np.min(unique_values)
        max_val = np.max(unique_values)
        
        # Calculate dimension as (max_val - min_val + 1)
        # This handles cases like [4,8] -> dimension 5 (values 0,1,2,3,4)
        dimension = int(max_val - min_val + 1)
        cat_dims.append(dimension)
        
        print(f"Categorical feature {col_idx}: values {min_val} to {max_val}, dimension = {dimension}")
    
    return cat_dims

def preprocess_categorical_features(features, categorical_features):
    """
    Preprocess categorical features to map them to 0-based indices.
    
    Args:
        features (np.ndarray): Feature matrix
        categorical_features (list): List of categorical feature indices
    
    Returns:
        np.ndarray: Preprocessed feature matrix with categorical features mapped to 0-based indices
    """
    processed_features = features.copy()
    
    for col_idx in categorical_features:
        unique_values = np.unique(features[:, col_idx])
        min_val = np.min(unique_values)
        
        # Map values to 0-based indices
        processed_features[:, col_idx] = features[:, col_idx] - min_val
    
    return processed_features

class TransactionDataset(Dataset):
    def __init__(self, features, labels, num_numerical=5, num_categorical=6):
        """
        Generic dataset class for mixed numerical and categorical features
        
        Args:
            features (np.ndarray): Feature matrix
            labels (np.ndarray): Target labels
            num_numerical (int): Number of numerical features (should be at the beginning)
            num_categorical (int): Number of categorical features (should come after numerical)
        """
        # Validate input
        total_features = features.shape[1]
        expected_total = num_numerical + num_categorical
        
        if total_features != expected_total:
            raise ValueError(f"Expected {expected_total} features, but found {total_features}")
        
        # Split features into numerical and categorical
        self.num_features = torch.FloatTensor(features[:, :num_numerical])
        
        # Process categorical features - map to 0-based indices
        cat_features = features[:, num_numerical:].copy()
        
        # Preprocess categorical features to ensure they start from 0
        cat_features = preprocess_categorical_features(cat_features, list(range(num_categorical)))
        
        self.cat_features = torch.LongTensor(cat_features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.num_features[idx], self.cat_features[idx], self.labels[idx]

class TransformerClassifier(nn.Module):
    def __init__(self, num_input_dim=None, cat_input_dims=None, 
                 embedding_dim=None, num_heads=None, num_layers=None, dropout=None):
        super().__init__()
        
        # Use config values if not provided
        self.embedding_dim = embedding_dim if embedding_dim is not None else CONFIG['embedding_dim']
        self.num_heads = num_heads if num_heads is not None else CONFIG['num_heads']
        self.num_layers = num_layers if num_layers is not None else CONFIG['num_layers']
        self.dropout = dropout if dropout is not None else CONFIG['dropout']
        
        # Use provided dimensions or defaults
        self.num_input_dim = num_input_dim if num_input_dim is not None else CONFIG['num_numerical_features']
        self.cat_input_dims = cat_input_dims if cat_input_dims is not None else [2, 4, 3, 4, 2, 3]
        
        # Print information for debugging
        print(f"Categorical input dimensions: {self.cat_input_dims}")
        
        # Embedding layers for categorical features with larger embedding dim
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(dim, self.embedding_dim) 
            for dim in self.cat_input_dims
        ])
        
        # Calculate total dimension after embeddings
        total_cat_dim = len(self.cat_input_dims) * self.embedding_dim
        raw_input_dim = self.num_input_dim + total_cat_dim
        
        # Ensure total dimension is divisible by num_heads
        # We'll project to the nearest multiple of num_heads that's >= raw_input_dim
        self.total_input_dim = ((raw_input_dim + self.num_heads - 1) // self.num_heads) * self.num_heads
        
        print(f"Raw input dim: {raw_input_dim}, Adjusted to: {self.total_input_dim}")
        
        # Initial projection to make dimensions compatible
        self.initial_projection = nn.Linear(raw_input_dim, self.total_input_dim)
        
        # Input projection
        self.input_projection = nn.Linear(self.total_input_dim, self.total_input_dim)
        
        # Deeper stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.total_input_dim, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Classification head with more layers for better generalization
        self.classifier = nn.Sequential(
            nn.Linear(self.total_input_dim, self.total_input_dim),
            nn.LayerNorm(self.total_input_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.total_input_dim, self.total_input_dim // 2),
            nn.LayerNorm(self.total_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.total_input_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, num_x, cat_x):
        # Process categorical features through embeddings
        embedded_cats = []
        for i, embedding_layer in enumerate(self.embedding_layers):
            embedded_cats.append(embedding_layer(cat_x[:, i]))
        
        # Concatenate all categorical embeddings
        cat_features = torch.cat(embedded_cats, dim=1)
        
        # Concatenate with numerical features
        x = torch.cat([num_x, cat_features], dim=1)
        
        # Project to make divisible by num_heads
        x = self.initial_projection(x)
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        
        # Apply transformer blocks with residual connections
        feature_maps = []
        for block in self.transformer_blocks:
            x = block(x)
            feature_maps.append(x)
        
        # Use last transformer output with global average pooling
        # This avoids dimension issues when concatenating features
        final_features = x.mean(dim=1)  # Global average pooling on sequence dimension
        
        # Classification head
        x = self.classifier(final_features)
        return x.squeeze()

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads=None, dropout=None):
        super().__init__()
        self.num_heads = num_heads if num_heads is not None else CONFIG['num_heads']
        self.dropout = dropout if dropout is not None else CONFIG['dropout']
        
        self.attention = nn.MultiheadAttention(input_dim, self.num_heads, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),  # GELU instead of ReLU for better performance
            nn.Dropout(self.dropout),
            nn.Linear(input_dim * 4, input_dim)
        )
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, x):
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        x = residual + self.dropout(attn_output)
        
        # Feed forward with pre-norm
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + self.dropout(ff_output)
        return x

def find_optimal_threshold_from_arrays(outputs, labels):
    """
    Find the optimal threshold from arrays of outputs and labels.
    This function searches thresholds in range [0.01, 0.81] to find the one with highest precision
    while maintaining recall >= 0.95.
    """
    outputs = np.array(outputs)
    labels = np.array(labels)
    
    # Check if we have any positive examples
    num_positives = np.sum(labels)
    if num_positives == 0:
        print("Warning: No positive examples in validation set")
        return CONFIG['threshold_range'][0]  # Default threshold
    
    # Evaluate thresholds in the range [0.01, 0.81] with step 0.01
    best_threshold = CONFIG['threshold_range'][0]  # Start with lowest
    best_precision = 0.0
    
    # Search thresholds from low to high
    thresholds = np.arange(CONFIG['threshold_range'][0], CONFIG['threshold_range'][1], CONFIG['threshold_step'])
    results = []
    
    print("\nThreshold search results:")
    print("Threshold | Precision | Recall")
    print("-" * 35)
    
    for threshold in thresholds:
        # Make predictions with current threshold
        predictions = outputs >= threshold
        
        # Calculate metrics
        recall = recall_score(labels, predictions, zero_division=0)
        
        # If recall drops below minimum, we can stop
        if recall < CONFIG['min_recall']:
            break
        
        precision = precision_score(labels, predictions, zero_division=0)
        results.append((threshold, precision, recall))
        
        print(f"{threshold:.2f}     | {precision:.4f}    | {recall:.4f}")
        
        # Update best if this gives better precision with sufficient recall
        if precision > best_precision:
            best_precision = precision
            best_threshold = threshold
    
    if not results:
        print(f"No threshold found with recall >= {CONFIG['min_recall']}. Using lowest threshold ({CONFIG['threshold_range'][0]})")
        return CONFIG['threshold_range'][0]
    
    print(f"\nSelected threshold {best_threshold:.2f} with precision {best_precision:.4f}")
    return best_threshold

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=None, device='cuda'):
    best_val_precision = 0
    best_model = None
    
    # Use config value if not provided
    num_epochs = num_epochs if num_epochs is not None else CONFIG['num_epochs']
    
    print(f"Training set size: {len(train_loader.dataset)}, Validation set size: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for num_features, cat_features, labels in train_loop:
            num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(num_features, cat_features)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=train_loss / (train_loop.n + 1))
        
        # Step the scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Plateau scheduler needs validation loss
                scheduler.step(train_loss / len(train_loader))
            else:
                # Other schedulers step every epoch
                scheduler.step()
            
            # Print current learning rate
            if hasattr(scheduler, 'get_last_lr'):
                current_lr = scheduler.get_last_lr()[0]
                print(f"Current learning rate: {current_lr:.6f}")
            elif hasattr(optimizer, 'param_groups'):
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.6f}")
        
        # Validation phase - first collect all predictions
        model.eval()
        val_outputs = []
        val_labels = []
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val Predictions]")
            for num_features, cat_features, labels in val_loop:
                num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
                outputs = model(num_features, cat_features)
                val_outputs.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Find optimal threshold using validation set
        optimal_threshold = find_optimal_threshold_from_arrays(val_outputs, val_labels)
        
        # Compute validation metrics with the optimal threshold
        val_preds = (np.array(val_outputs) > optimal_threshold).astype(int)
        val_precision = precision_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        
        # Print validation confusion matrix
        val_cm = confusion_matrix(val_labels, val_preds)
        print(f'Validation Confusion Matrix:\n{val_cm}')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Optimal Threshold: {optimal_threshold:.4f}')
        print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
        
        # Save model if it improves precision score with sufficient recall
        if val_precision > best_val_precision and val_recall >= CONFIG['min_recall']:
            print(f"New best model with Precision: {val_precision:.4f} and Recall: {val_recall:.4f}")
            best_val_precision = val_precision
            best_model = model.state_dict().copy()
            best_threshold = optimal_threshold
        
        # If no model with sufficient recall found after half the epochs, save best model anyway
        if epoch == num_epochs // 2 and best_model is None:
            print("No model with recall >= {CONFIG['min_recall']} found yet. Saving current model.")
            best_model = model.state_dict().copy()
            best_threshold = optimal_threshold
    
    # If we never found a model with good recall, use the last one
    if best_model is None:
        print("No model with sufficient recall found. Using final model.")
        best_model = model.state_dict().copy()
        best_threshold = optimal_threshold
    
    return best_model, best_threshold

def load_data(data_path):
    """
    Load data from Excel or CSV file based on file extension
    
    Args:
        data_path (str): Path to the data file
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    print(f"Loading data from: {data_path}")
    
    if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format. Please use .xlsx, .xls, or .csv files")
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return X, y

def create_optimizer(model):
    """
    Create optimizer based on config
    """
    optimizer_type = CONFIG['optimizer'].lower()
    lr = CONFIG['learning_rate']
    weight_decay = CONFIG['weight_decay']
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(CONFIG['beta1'], CONFIG['beta2'])
        )
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(CONFIG['beta1'], CONFIG['beta2'])
        )
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=CONFIG['momentum'],
            weight_decay=weight_decay
        )
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adagrad':
        return torch.optim.Adagrad(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamax':
        return torch.optim.Adamax(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(CONFIG['beta1'], CONFIG['beta2'])
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}. Available: adam, adamw, sgd, rmsprop, adagrad, adamax")

def create_scheduler(optimizer, num_epochs, steps_per_epoch):
    """
    Create scheduler based on config
    """
    scheduler_type = CONFIG['scheduler'].lower()
    
    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=CONFIG['scheduler_max_lr'], 
            epochs=num_epochs, 
            steps_per_epoch=steps_per_epoch
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=CONFIG['scheduler_step_size'], 
            gamma=CONFIG['scheduler_gamma']
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs
        )
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=CONFIG['scheduler_gamma']
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=CONFIG['scheduler_factor'],
            patience=CONFIG['scheduler_patience']
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}. Available: none, onecycle, step, cosine, exponential, plateau")

def main():
    """
    Config-driven main function for transformer classification.
    All parameters are controlled by the CONFIG dictionary at the top of the file.
    """
    # Load data using config
    X, y = load_data(CONFIG['data_path'])
    
    # Validate input parameters
    total_features = X.shape[1]
    expected_total = CONFIG['num_numerical_features'] + CONFIG['num_categorical_features']
    
    if total_features != expected_total:
        raise ValueError(f"Expected {expected_total} features (numerical: {CONFIG['num_numerical_features']}, categorical: {CONFIG['num_categorical_features']}), but found {total_features}")
    
    # Define feature ranges
    numerical_start = 0
    numerical_end = CONFIG['num_numerical_features']
    categorical_start = CONFIG['num_numerical_features']
    categorical_end = CONFIG['num_numerical_features'] + CONFIG['num_categorical_features']
    
    # Print ranges of categorical features for verification
    print(f"\nFeature ranges:")
    print(f"Numerical features: columns {numerical_start} to {numerical_end-1}")
    print(f"Categorical features: columns {categorical_start} to {categorical_end-1}")
    
    for i in range(categorical_start, categorical_end):
        feature_idx = i - categorical_start
        min_val = np.min(X[:, i])
        max_val = np.max(X[:, i])
        print(f"Categorical feature {feature_idx+1} (column {i}): {min_val:.0f} to {max_val:.0f}")
    
    # Split data using config
    print("Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=CONFIG['val_size'], random_state=CONFIG['random_state'], stratify=y_temp)
    
    # Define categorical features indices
    categorical_features = list(range(categorical_start, categorical_end))
    
    # Create sampling pipeline using config
    print("Resampling training data...")
    sampling_pipeline = Pipeline([
        ('smote', SMOTENC(categorical_features=categorical_features, sampling_strategy=CONFIG['smote_ratio'], random_state=CONFIG['random_state'])),
        ('under', RandomUnderSampler(sampling_strategy=CONFIG['undersample_ratio'], random_state=CONFIG['random_state']))
    ])
    
    # Apply sampling
    X_train_resampled, y_train_resampled = sampling_pipeline.fit_resample(X_train, y_train)
    
    # Print class distribution after resampling
    pos_count = np.sum(y_train_resampled)
    total_count = len(y_train_resampled)
    print(f"Class distribution after resampling: {pos_count} positives, {total_count-pos_count} negatives")
    print(f"Positive percentage: {100*pos_count/total_count:.2f}%")
    
    # Create datasets and dataloaders using config
    print("Creating datasets and dataloaders...")
    train_dataset = TransactionDataset(X_train_resampled, y_train_resampled, 
                                     num_numerical=CONFIG['num_numerical_features'], 
                                     num_categorical=CONFIG['num_categorical_features'])
    val_dataset = TransactionDataset(X_val, y_val, 
                                   num_numerical=CONFIG['num_numerical_features'], 
                                   num_categorical=CONFIG['num_categorical_features'])
    test_dataset = TransactionDataset(X_test, y_test, 
                                    num_numerical=CONFIG['num_numerical_features'], 
                                    num_categorical=CONFIG['num_categorical_features'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Calculate categorical feature dimensions dynamically
    cat_input_dims = calculate_categorical_dimensions(X, categorical_features)
    print(f"Calculated categorical dimensions: {cat_input_dims}")
    
    # Initialize model using config
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = TransformerClassifier(num_input_dim=CONFIG['num_numerical_features'], 
                                 cat_input_dims=cat_input_dims).to(device)
    
    # Print model summary
    print("\nModel Summary:")
    
    # Create proper input tensors for the model summary
    input_data = (
        torch.zeros(CONFIG['batch_size'], CONFIG['num_numerical_features'], dtype=torch.float32, device=device),
        torch.zeros(CONFIG['batch_size'], CONFIG['num_categorical_features'], dtype=torch.long, device=device)
    )
    
    # Use torchinfo's summary with input_data for detailed layer-by-layer breakdown
    summary(
        model, 
        input_data=input_data,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        col_width=20,
        row_settings=["var_names"]
    )
    
    # Calculate and print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training setup using config
    criterion = create_loss_function()
    optimizer = create_optimizer(model)
    
    # Learning rate scheduler using config
    scheduler = create_scheduler(optimizer, CONFIG['num_epochs'], len(train_loader))
    
    # Train model using config
    print("Starting training...")
    best_model, best_threshold = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        scheduler=scheduler, 
        device=device
    )
    
    # Load best model and evaluate on test set
    print("Evaluating on test set...")
    model.load_state_dict(best_model)
    model.eval()
    
    test_outputs = []
    test_labels = []
    
    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing")
        for num_features, cat_features, labels in test_loop:
            num_features, cat_features, labels = num_features.to(device), cat_features.to(device), labels.to(device)
            outputs = model(num_features, cat_features)
            test_outputs.extend(outputs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Apply best threshold to test predictions
    test_preds = (np.array(test_outputs) > best_threshold).astype(int)
    
    # Print final metrics
    print("\nTest Set Results:")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Precision: {precision_score(test_labels, test_preds):.4f}")
    print(f"Recall: {recall_score(test_labels, test_preds):.4f}")
    print(f"F1 Score: {f1_score(test_labels, test_preds):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

if __name__ == "__main__":
    # Example usage with default config (original dataset)
    main()
    
    # To use with a different dataset, modify the CONFIG dictionary at the top:
    # CONFIG['data_path'] = 'your_data.csv'
    # CONFIG['num_numerical_features'] = 3
    # CONFIG['num_categorical_features'] = 4
    # Then call main()
    
    # Example for a different dataset:
    # CONFIG.update({
    #     'data_path': 'test_data_imbalanced.csv',
    #     'num_numerical_features': 3,
    #     'num_categorical_features': 4,
    #     'batch_size': 32,  # Smaller batch size for smaller dataset
    #     'num_epochs': 30   # Fewer epochs for faster training
    # })
    # main() 