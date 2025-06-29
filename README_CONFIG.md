 # Config-Driven Transformer Classification

This is a config-driven version of the transformer classification system that can be easily adapted to different datasets by modifying a single configuration dictionary.

## Quick Start

### 1. Basic Usage

To use with your dataset, simply modify the `CONFIG` dictionary at the top of `transformer_classification_generic.py`:

```python
CONFIG = {
    # REQUIRED: Modify these for your dataset
    'data_path': 'your_data.csv',           # Path to your data file
    'num_numerical_features': 3,            # Number of numerical features (at the beginning)
    'num_categorical_features': 4,          # Number of categorical features (after numerical)
    
    # OPTIONAL: Other parameters can be left as default
    'embedding_dim': 8,
    'num_heads': 8,
    'num_layers': 6,
    # ... other parameters
}
```

Then run:
```bash
python transformer_classification_generic.py
```

### 2. Data Format Requirements

Your data file should have:
- **Numerical features first**: Columns 0 to `num_numerical_features-1`
- **Categorical features next**: Columns `num_numerical_features` to `num_numerical_features + num_categorical_features - 1`
- **Target column last**: The final column should be your binary target (0/1)

Example:
```
| num_0 | num_1 | num_2 | cat_0 | cat_1 | cat_2 | target |
|-------|-------|-------|-------|-------|-------|--------|
| 1.2   | 3.4   | 5.6   | 0     | 2     | 1     | 0      |
| 2.1   | 4.2   | 6.7   | 1     | 3     | 0     | 1      |
```

### 3. Supported File Formats

- **Excel**: `.xlsx`, `.xls`
- **CSV**: `.csv`

## Configuration Parameters

### Required Parameters (Must Modify)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `data_path` | Path to your data file | `'my_data.csv'` |
| `num_numerical_features` | Number of numerical features | `5` |
| `num_categorical_features` | Number of categorical features | `6` |

### Model Architecture (Optional)

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `embedding_dim` | Embedding dimension for categorical features | `8` | `4-32` |
| `num_heads` | Number of attention heads | `8` | `4-16` |
| `num_layers` | Number of transformer layers | `6` | `3-12` |
| `dropout` | Dropout rate | `0.2` | `0.1-0.5` |

### Training Parameters (Optional)

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `learning_rate` | Learning rate | `0.001` | `0.0001-0.01` |
| `num_epochs` | Number of training epochs | `50` | `10-100` |
| `batch_size` | Batch size | `64` | `32-256` |
| `weight_decay` | L2 regularization | `0.01` | `0.001-0.1` |

### Data Processing (Optional)

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `test_size` | Test set ratio | `0.2` | `0.1-0.3` |
| `val_size` | Validation set ratio | `0.25` | `0.2-0.3` |
| `smote_ratio` | SMOTE oversampling ratio | `0.3` | `0.1-0.5` |
| `undersample_ratio` | Undersampling ratio | `0.5` | `0.3-0.7` |

### Loss Function (Optional)

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `focal_alpha` | Focal loss alpha (class weight) | `0.9` | `0.7-0.95` |
| `focal_gamma` | Focal loss gamma (focusing) | `5` | `2-7` |

### Threshold Optimization (Optional)

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `min_recall` | Minimum recall threshold | `0.95` | `0.8-0.99` |
| `threshold_range` | Threshold search range | `(0.01, 0.81)` | - |
| `threshold_step` | Threshold search step | `0.01` | `0.01-0.05` |

## Examples

### Example 1: Different Dataset

```python
CONFIG.update({
    'data_path': 'my_dataset.csv',
    'num_numerical_features': 7,
    'num_categorical_features': 3,
    'batch_size': 32,  # Smaller batch for smaller dataset
    'num_epochs': 30   # Fewer epochs for faster training
})
```

### Example 2: High Precision Configuration

```python
CONFIG.update({
    'min_recall': 0.85,           # Lower recall threshold
    'threshold_range': (0.1, 0.9), # Higher threshold range
    'focal_alpha': 0.8,           # Less focus on positive class
    'focal_gamma': 3              # Less focus on hard examples
})
```

### Example 3: Fast Training

```python
CONFIG.update({
    'batch_size': 128,            # Larger batch size
    'num_epochs': 10,             # Very few epochs
    'learning_rate': 0.002,       # Higher learning rate
    'embedding_dim': 4,           # Smaller embeddings
    'num_heads': 4,               # Fewer attention heads
    'num_layers': 3               # Fewer transformer layers
})
```

## Running Examples

See `example_usage.py` for complete examples of different configurations.

```bash
# Run with default config
python transformer_classification_generic.py

# Run examples
python example_usage.py
```

## Key Features

### 1. Automatic Categorical Dimension Calculation
The system automatically calculates the number of unique values for each categorical feature, handling cases where values don't start from 0.

### 2. Dynamic Feature Preprocessing
Categorical features are automatically mapped to 0-based indices, removing the need for hardcoded mappings.

### 3. Flexible Data Loading
Supports both Excel and CSV files automatically based on file extension.

### 4. Config-Driven Everything
All hyperparameters, model architecture, and training settings are controlled by the CONFIG dictionary.

### 5. Imbalanced Dataset Handling
Built-in SMOTE oversampling and undersampling with configurable ratios.

## Troubleshooting

### Common Issues

1. **Wrong number of features**: Make sure `num_numerical_features + num_categorical_features` equals the total number of feature columns (excluding target).

2. **File not found**: Check that the `data_path` points to an existing file.

3. **Memory issues**: Reduce `batch_size` or `embedding_dim` for large datasets.

4. **Training too slow**: Reduce `num_epochs`, increase `learning_rate`, or use the fast training configuration.

### Validation

The system validates your configuration and provides helpful error messages:
- Checks that the total number of features matches your specification
- Verifies that the data file exists and can be loaded
- Ensures categorical features are properly formatted

## Performance Tips

1. **For large datasets**: Use larger `batch_size` and fewer `epochs`
2. **For imbalanced datasets**: Increase `smote_ratio` and adjust `focal_alpha`
3. **For high precision**: Lower `min_recall` and adjust `threshold_range`
4. **For fast training**: Use smaller model architecture (`embedding_dim`, `num_heads`, `num_layers`)