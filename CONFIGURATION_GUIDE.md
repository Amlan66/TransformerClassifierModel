# Configuration Guide for Transformer Classification

This guide provides detailed information about all configurable parameters in the transformer classification system.

## Quick Configuration Examples

### Basic Setup
```python
CONFIG = {
    'data_path': 'your_data.csv',
    'num_numerical_features': 5,
    'num_categorical_features': 6,
    # All other parameters use defaults
}
```

### High Precision Setup
```python
CONFIG.update({
    'loss_function': 'weighted_bce',
    'class_weight': 5.0,
    'optimizer': 'adamw',
    'learning_rate': 0.0005,
    'scheduler': 'plateau',
    'min_recall': 0.85
})
```

### Fast Training Setup
```python
CONFIG.update({
    'loss_function': 'focal',
    'optimizer': 'adam',
    'learning_rate': 0.002,
    'scheduler': 'onecycle',
    'batch_size': 128,
    'num_epochs': 20
})
```

## Loss Functions

### 1. Focal Loss (`'focal'`)
**Best for**: Highly imbalanced datasets
```python
'loss_function': 'focal',
'focal_alpha': 0.9,    # Weight for positive class (0.7-0.99)
'focal_gamma': 5       # Focusing parameter (0-8)
```

**Parameters**:
- `focal_alpha`: Higher values (0.95-0.99) give more weight to positive class
- `focal_gamma`: Higher values (5-8) focus more on hard examples

### 2. Binary Cross Entropy (`'bce'`)
**Best for**: Balanced datasets
```python
'loss_function': 'bce'
```

**Parameters**: None (standard BCE loss)

### 3. Weighted BCE (`'weighted_bce'`)
**Best for**: Moderately imbalanced datasets
```python
'loss_function': 'weighted_bce',
'class_weight': 2.0    # Weight for positive class (0.1-10.0)
```

**Parameters**:
- `class_weight`: Values > 1.0 give more weight to positive class

### 4. Dice Loss (`'dice'`)
**Best for**: Imbalanced datasets, focuses on overlap
```python
'loss_function': 'dice'
```

**Parameters**: None (uses default smoothing)

### 5. Combo Loss (`'combo'`)
**Best for**: Balanced approach combining BCE and Dice
```python
'loss_function': 'combo'
```

**Parameters**: None (uses equal weights for BCE and Dice)

## Optimizers

### 1. Adam (`'adam'`)
**Best for**: Most cases, adaptive learning rate
```python
'optimizer': 'adam',
'learning_rate': 0.001,
'beta1': 0.9,
'beta2': 0.999
```

### 2. AdamW (`'adamw'`)
**Best for**: Better regularization, weight decay
```python
'optimizer': 'adamw',
'learning_rate': 0.001,
'weight_decay': 0.01,
'beta1': 0.9,
'beta2': 0.999
```

### 3. SGD (`'sgd'`)
**Best for**: Stable training, momentum
```python
'optimizer': 'sgd',
'learning_rate': 0.01,
'momentum': 0.9,
'weight_decay': 0.01
```

### 4. RMSprop (`'rmsprop'`)
**Best for**: Non-stationary problems
```python
'optimizer': 'rmsprop',
'learning_rate': 0.001,
'weight_decay': 0.01
```

### 5. Adagrad (`'adagrad'`)
**Best for**: Sparse data
```python
'optimizer': 'adagrad',
'learning_rate': 0.01,
'weight_decay': 0.01
```

### 6. Adamax (`'adamax'`)
**Best for**: Sometimes better convergence than Adam
```python
'optimizer': 'adamax',
'learning_rate': 0.001,
'beta1': 0.9,
'beta2': 0.999
```

## Learning Rate Schedulers

### 1. OneCycle (`'onecycle'`)
**Best for**: Often best performance, fast convergence
```python
'scheduler': 'onecycle',
'scheduler_max_lr': 0.002  # Usually 1-2x base learning rate
```

### 2. Step (`'step'`)
**Best for**: Simple and effective
```python
'scheduler': 'step',
'scheduler_step_size': 15,  # Epochs between decay
'scheduler_gamma': 0.5      # Decay factor
```

### 3. Cosine (`'cosine'`)
**Best for**: Smooth decay, good for long training
```python
'scheduler': 'cosine'
```

### 4. Exponential (`'exponential'`)
**Best for**: Aggressive decay
```python
'scheduler': 'exponential',
'scheduler_gamma': 0.95     # Decay factor
```

### 5. Plateau (`'plateau'`)
**Best for**: Adaptive decay based on validation loss
```python
'scheduler': 'plateau',
'scheduler_patience': 7,    # Epochs to wait
'scheduler_factor': 0.5     # Reduction factor
```

### 6. None (`'none'`)
**Best for**: Constant learning rate
```python
'scheduler': 'none'
```

## Recommended Configurations

### For Highly Imbalanced Data (e.g., 1% positive)
```python
CONFIG.update({
    'loss_function': 'focal',
    'focal_alpha': 0.95,
    'focal_gamma': 7,
    'smote_ratio': 0.7,
    'undersample_ratio': 0.3,
    'optimizer': 'adamw',
    'learning_rate': 0.0005,
    'scheduler': 'plateau'
})
```

### For Balanced Data
```python
CONFIG.update({
    'loss_function': 'bce',
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'scheduler': 'onecycle',
    'smote_ratio': 0.2,
    'undersample_ratio': 0.8
})
```

### For Fast Experimentation
```python
CONFIG.update({
    'num_epochs': 10,
    'batch_size': 128,
    'learning_rate': 0.002,
    'scheduler': 'onecycle',
    'embedding_dim': 4,
    'num_layers': 3
})
```

### For High Precision Requirements
```python
CONFIG.update({
    'loss_function': 'weighted_bce',
    'class_weight': 5.0,
    'min_recall': 0.85,
    'threshold_range': (0.1, 0.9),
    'optimizer': 'adamw',
    'weight_decay': 0.1
})
```

### For Conservative Training
```python
CONFIG.update({
    'loss_function': 'dice',
    'optimizer': 'sgd',
    'learning_rate': 0.0001,
    'momentum': 0.95,
    'scheduler': 'cosine',
    'dropout': 0.3,
    'num_epochs': 100
})
```

## Parameter Tuning Guidelines

### Learning Rate
- **Start with**: 0.001
- **If training is slow**: Increase to 0.002-0.005
- **If training is unstable**: Decrease to 0.0001-0.0005

### Batch Size
- **Small dataset**: 16-32
- **Medium dataset**: 64-128
- **Large dataset**: 128-256

### Model Architecture
- **Fast training**: `embedding_dim=4`, `num_layers=3`, `num_heads=4`
- **Balanced**: `embedding_dim=8`, `num_layers=6`, `num_heads=8`
- **High performance**: `embedding_dim=16`, `num_layers=9`, `num_heads=16`

### Regularization
- **If overfitting**: Increase `dropout` (0.3-0.5), increase `weight_decay` (0.1-1.0)
- **If underfitting**: Decrease `dropout` (0.1-0.2), decrease `weight_decay` (0.001-0.01)

## Troubleshooting

### Training is too slow
- Increase `learning_rate`
- Increase `batch_size`
- Use `scheduler='onecycle'`
- Reduce model size (`embedding_dim`, `num_layers`)

### Training is unstable
- Decrease `learning_rate`
- Use `optimizer='sgd'` with `momentum=0.9`
- Increase `batch_size`
- Use `scheduler='plateau'`

### Model is overfitting
- Increase `dropout`
- Increase `weight_decay`
- Reduce model size
- Use `scheduler='plateau'`

### Model is underfitting
- Decrease `dropout`
- Decrease `weight_decay`
- Increase model size
- Increase `num_epochs`
- Use `scheduler='onecycle'`

### Poor performance on imbalanced data
- Use `loss_function='focal'` or `'weighted_bce'`
- Increase `smote_ratio`
- Decrease `undersample_ratio`
- Adjust `class_weight` or `focal_alpha` 