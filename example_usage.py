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

def example_5_different_loss_functions():
    """Example 5: Different loss function configurations"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Different loss function configurations")
    print("=" * 60)
    
    # Test different loss functions
    loss_functions = ['focal', 'bce', 'weighted_bce', 'dice', 'combo']
    
    for loss_func in loss_functions:
        print(f"\n--- Testing {loss_func.upper()} Loss ---")
        
        CONFIG.update({
            'data_path': 'transactions.xlsx',
            'num_numerical_features': 5,
            'num_categorical_features': 6,
            'loss_function': loss_func,
            'num_epochs': 3,  # Quick test
            'batch_size': 32
        })
        
        print(f"Loss function: {CONFIG['loss_function']}")
        if loss_func == 'weighted_bce':
            print(f"Class weight: {CONFIG['class_weight']}")
        elif loss_func == 'focal':
            print(f"Focal alpha: {CONFIG['focal_alpha']}, gamma: {CONFIG['focal_gamma']}")
        
        # Note: In practice, you'd run main() here
        main()

def example_6_different_optimizers():
    """Example 6: Different optimizer configurations"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Different optimizer configurations")
    print("=" * 60)
    
    # Test different optimizers
    optimizers = [
        ('adam', {'learning_rate': 0.001}),
        ('adamw', {'learning_rate': 0.001, 'weight_decay': 0.01}),
        ('sgd', {'learning_rate': 0.01, 'momentum': 0.9}),
        ('rmsprop', {'learning_rate': 0.001}),
        ('adagrad', {'learning_rate': 0.01}),
        ('adamax', {'learning_rate': 0.001})
    ]
    
    for opt_name, opt_params in optimizers:
        print(f"\n--- Testing {opt_name.upper()} Optimizer ---")
        
        CONFIG.update({
            'data_path': 'transactions.xlsx',
            'num_numerical_features': 5,
            'num_categorical_features': 6,
            'optimizer': opt_name,
            'num_epochs': 3,  # Quick test
            'batch_size': 32,
            **opt_params
        })
        
        print(f"Optimizer: {CONFIG['optimizer']}")
        print(f"Learning rate: {CONFIG['learning_rate']}")
        if opt_name == 'sgd':
            print(f"Momentum: {CONFIG['momentum']}")
        elif opt_name in ['adam', 'adamw', 'adamax']:
            print(f"Beta1: {CONFIG['beta1']}, Beta2: {CONFIG['beta2']}")
        
        # Note: In practice, you'd run main() here
        main()

def example_7_different_schedulers():
    """Example 7: Different scheduler configurations"""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Different scheduler configurations")
    print("=" * 60)
    
    # Test different schedulers
    schedulers = [
        ('none', {}),
        ('onecycle', {'scheduler_max_lr': 0.002}),
        ('step', {'scheduler_step_size': 10, 'scheduler_gamma': 0.5}),
        ('cosine', {}),
        ('exponential', {'scheduler_gamma': 0.95}),
        ('plateau', {'scheduler_patience': 5, 'scheduler_factor': 0.5})
    ]
    
    for sched_name, sched_params in schedulers:
        print(f"\n--- Testing {sched_name.upper()} Scheduler ---")
        
        CONFIG.update({
            'data_path': 'transactions.xlsx',
            'num_numerical_features': 5,
            'num_categorical_features': 6,
            'scheduler': sched_name,
            'num_epochs': 3,  # Quick test
            'batch_size': 32,
            **sched_params
        })
        
        print(f"Scheduler: {CONFIG['scheduler']}")
        if sched_name == 'onecycle':
            print(f"Max LR: {CONFIG['scheduler_max_lr']}")
        elif sched_name == 'step':
            print(f"Step size: {CONFIG['scheduler_step_size']}, Gamma: {CONFIG['scheduler_gamma']}")
        elif sched_name == 'exponential':
            print(f"Gamma: {CONFIG['scheduler_gamma']}")
        elif sched_name == 'plateau':
            print(f"Patience: {CONFIG['scheduler_patience']}, Factor: {CONFIG['scheduler_factor']}")
        
        # Note: In practice, you'd run main() here
        main()

def example_8_advanced_configurations():
    """Example 8: Advanced configurations for specific use cases"""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Advanced configurations for specific use cases")
    print("=" * 60)
    
    # Configuration 1: High precision setup
    print("\n--- High Precision Configuration ---")
    CONFIG.update({
        'data_path': 'transactions.xlsx',
        'num_numerical_features': 5,
        'num_categorical_features': 6,
        'loss_function': 'weighted_bce',
        'class_weight': 5.0,
        'optimizer': 'adamw',
        'learning_rate': 0.0005,
        'weight_decay': 0.1,
        'scheduler': 'plateau',
        'scheduler_patience': 10,
        'scheduler_factor': 0.3,
        'min_recall': 0.85,
        'threshold_range': (0.1, 0.9),
        'num_epochs': 50
    })
    print("High precision setup with weighted BCE, AdamW, and plateau scheduler")
    main()
    
    # Configuration 2: Fast convergence setup
    print("\n--- Fast Convergence Configuration ---")
    CONFIG.update({
        'data_path': 'transactions.xlsx',
        'num_numerical_features': 5,
        'num_categorical_features': 6,
        'loss_function': 'focal',
        'focal_alpha': 0.95,
        'focal_gamma': 7,
        'optimizer': 'adam',
        'learning_rate': 0.002,
        'scheduler': 'onecycle',
        'scheduler_max_lr': 0.005,
        'batch_size': 128,
        'num_epochs': 20
    })
    print("Fast convergence setup with focal loss, Adam, and onecycle scheduler")
    main()
    
    # Configuration 3: Conservative training setup
    print("\n--- Conservative Training Configuration ---")
    CONFIG.update({
        'data_path': 'transactions.xlsx',
        'num_numerical_features': 5,
        'num_categorical_features': 6,
        'loss_function': 'dice',
        'optimizer': 'sgd',
        'learning_rate': 0.0001,
        'momentum': 0.95,
        'scheduler': 'cosine',
        'batch_size': 32,
        'num_epochs': 50,
        'dropout': 0.3
    })
    print("Conservative training setup with dice loss, SGD, and cosine scheduler")
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
    #example_4_fast_training_config()
    
    # Example 5: Different loss functions
    # example_5_different_loss_functions()
    
    # Example 6: Different optimizers
    # example_6_different_optimizers()
    
    # Example 7: Different schedulers
    # example_7_different_schedulers()
    
    # Example 8: Advanced configurations
    example_8_advanced_configurations()
    
    print("\nTo run an example, uncomment the corresponding function call above.")
    print("Make sure you have the required data files before running the examples.") 