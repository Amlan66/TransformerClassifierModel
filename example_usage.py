"""
Example usage of the config-driven transformer classification system.

This script demonstrates how to:
1. Use the default configuration
2. Modify the configuration for different datasets
3. Run the model with custom settings
"""

from transformer_classification_generic import main, CONFIG

def example_1_default_config():
    """Example 1: Use default configuration (original dataset)"""
    print("=" * 60)
    print("EXAMPLE 1: Using default configuration")
    print("=" * 60)
    
    # Use default config (transactions.xlsx with 5 numerical + 6 categorical features)
    main()

def example_2_custom_dataset():
    """Example 2: Use custom configuration for a different dataset"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Using custom configuration")
    print("=" * 60)
    
    # Modify config for a different dataset
    CONFIG.update({
        'data_path': 'test_data_imbalanced.csv',  # Different data file
        'num_numerical_features': 3,              # 3 numerical features
        'num_categorical_features': 4,            # 4 categorical features
        'batch_size': 32,                         # Smaller batch size
        'num_epochs': 30,                         # Fewer epochs for faster training
        'learning_rate': 0.0005,                  # Lower learning rate
        'smote_ratio': 0.5,                       # More aggressive oversampling
        'undersample_ratio': 0.3                  # Less aggressive undersampling
    })
    
    print("Modified CONFIG for custom dataset:")
    print(f"  Data path: {CONFIG['data_path']}")
    print(f"  Numerical features: {CONFIG['num_numerical_features']}")
    print(f"  Categorical features: {CONFIG['num_categorical_features']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Epochs: {CONFIG['num_epochs']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    
    # Run with custom config
    main()

def example_3_high_precision_config():
    """Example 3: Configuration optimized for high precision"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: High precision configuration")
    print("=" * 60)
    
    # Modify config for high precision (lower recall threshold)
    CONFIG.update({
        'data_path': 'transactions.xlsx',         # Back to original dataset
        'num_numerical_features': 5,              # Original features
        'num_categorical_features': 6,
        'min_recall': 0.85,                       # Lower recall threshold (was 0.95)
        'threshold_range': (0.1, 0.9),           # Higher threshold range
        'threshold_step': 0.02,                   # Coarser threshold steps
        'focal_alpha': 0.8,                       # Less focus on positive class
        'focal_gamma': 3,                         # Less focus on hard examples
        'num_epochs': 40,                         # Fewer epochs
        'learning_rate': 0.0008                   # Lower learning rate
    })
    
    print("Modified CONFIG for high precision:")
    print(f"  Min recall threshold: {CONFIG['min_recall']}")
    print(f"  Threshold range: {CONFIG['threshold_range']}")
    print(f"  Focal alpha: {CONFIG['focal_alpha']}")
    print(f"  Focal gamma: {CONFIG['focal_gamma']}")
    
    # Run with high precision config
    main()

def example_4_fast_training_config():
    """Example 4: Configuration for fast training/testing"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Fast training configuration")
    print("=" * 60)
    
    # Modify config for fast training
    CONFIG.update({
        'data_path': 'transactions.xlsx',
        'num_numerical_features': 5,
        'num_categorical_features': 6,
        'batch_size': 128,                        # Larger batch size
        'num_epochs': 10,                         # Very few epochs
        'learning_rate': 0.002,                   # Higher learning rate
        'embedding_dim': 4,                       # Smaller embeddings
        'num_heads': 4,                           # Fewer attention heads
        'num_layers': 3,                          # Fewer transformer layers
        'dropout': 0.1,                           # Less dropout
        'smote_ratio': 0.2,                       # Less aggressive resampling
        'undersample_ratio': 0.7                  # More aggressive undersampling
    })
    
    print("Modified CONFIG for fast training:")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Epochs: {CONFIG['num_epochs']}")
    print(f"  Embedding dim: {CONFIG['embedding_dim']}")
    print(f"  Num heads: {CONFIG['num_heads']}")
    print(f"  Num layers: {CONFIG['num_layers']}")
    
    # Run with fast training config
    main()

if __name__ == "__main__":
    print("CONFIG-DRIVEN TRANSFORMER CLASSIFICATION EXAMPLES")
    print("=" * 60)
    print("This script demonstrates different ways to use the config-driven system.")
    print("Each example shows how to modify the CONFIG dictionary for different use cases.")
    print()
    
    # Uncomment the example you want to run:
    
    # Example 1: Default configuration
    # example_1_default_config()
    
    # Example 2: Custom dataset (requires test_data_imbalanced.csv)
    # example_2_custom_dataset()
    
    # Example 3: High precision configuration
    # example_3_high_precision_config()
    
    # Example 4: Fast training configuration
    example_4_fast_training_config()
    
    print("\nTo run an example, uncomment the corresponding function call above.")
    print("Make sure you have the required data files before running the examples.") 