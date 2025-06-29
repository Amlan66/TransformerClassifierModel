import pandas as pd
import numpy as np
from transformer_classification_generic import main, calculate_categorical_dimensions, preprocess_categorical_features

def create_test_data():
    """Create a test dataset with different numerical and categorical feature configurations"""
    
    # Create sample data with 3 numerical and 4 categorical features
    np.random.seed(42)
    n_samples = 1000
    n_positive = 20  # Only 20 positive samples for imbalanced dataset
    
    # Numerical features (3 features)
    num_features = np.random.randn(n_samples, 3)
    
    # Categorical features (4 features) with different ranges
    cat_features = np.column_stack([
        np.random.randint(0, 3, n_samples),      # 0, 1, 2
        np.random.randint(4, 8, n_samples),      # 4, 5, 6, 7 -> should become 0, 1, 2, 3
        np.random.randint(10, 13, n_samples),    # 10, 11, 12 -> should become 0, 1, 2
        np.random.randint(1, 4, n_samples)       # 1, 2, 3 -> should become 0, 1, 2
    ])
    
    # Combine features
    X = np.column_stack([num_features, cat_features])
    
    # Create imbalanced target (binary classification)
    # First create all zeros
    y = np.zeros(n_samples, dtype=int)
    
    # Randomly select 20 indices to be positive (1)
    positive_indices = np.random.choice(n_samples, size=n_positive, replace=False)
    y[positive_indices] = 1
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'num_{i}' for i in range(3)] + [f'cat_{i}' for i in range(4)])
    df['target'] = y
    
    # Print class distribution
    positive_count = np.sum(y)
    negative_count = n_samples - positive_count
    print(f"Created imbalanced dataset:")
    print(f"  Total samples: {n_samples}")
    print(f"  Positive samples (1): {positive_count} ({100*positive_count/n_samples:.2f}%)")
    print(f"  Negative samples (0): {negative_count} ({100*negative_count/n_samples:.2f}%)")
    
    return df

def test_categorical_dimensions():
    """Test the categorical dimension calculation function"""
    print("Testing categorical dimension calculation...")
    
    # Create test data
    df = create_test_data()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Test with 3 numerical and 4 categorical features
    categorical_features = list(range(3, 7))  # Columns 3, 4, 5, 6 are categorical
    
    cat_dims = calculate_categorical_dimensions(X, categorical_features)
    print(f"Calculated dimensions: {cat_dims}")
    
    # Expected dimensions based on our test data
    expected_dims = [3, 4, 3, 3]  # Based on the ranges we created
    
    assert cat_dims == expected_dims, f"Expected {expected_dims}, got {cat_dims}"
    print("✅ Categorical dimension calculation test passed!")

def test_preprocessing():
    """Test the categorical feature preprocessing function"""
    print("\nTesting categorical feature preprocessing...")
    
    # Create test data
    df = create_test_data()
    X = df.iloc[:, :-1].values
    
    # Test preprocessing
    categorical_features = list(range(3, 7))
    processed_X = preprocess_categorical_features(X, categorical_features)
    
    # Check that categorical features now start from 0
    for i, col_idx in enumerate(categorical_features):
        min_val = np.min(processed_X[:, col_idx])
        max_val = np.max(processed_X[:, col_idx])
        print(f"Feature {i}: original range {np.min(X[:, col_idx])}-{np.max(X[:, col_idx])}, "
              f"processed range {min_val}-{max_val}")
        
        assert min_val == 0, f"Feature {i} should start from 0, but starts from {min_val}"
    
    print("✅ Categorical preprocessing test passed!")

def test_generic_main():
    """Test the generic main function with our test data"""
    print("\nTesting generic main function...")
    
    # Create and save test data
    df = create_test_data()
    test_file = 'test_data.csv'
    df.to_csv(test_file, index=False)
    
    try:
        # Test with 3 numerical and 4 categorical features
        print("Running main function with test data...")
        main(data_path=test_file, num_numerical_features=3, num_categorical_features=4)
        print("✅ Generic main function test completed!")
        
    except Exception as e:
        print(f"❌ Error in main function: {e}")
        raise
    finally:
        # Clean up test file
        import os
        if os.path.exists(test_file):
            os.remove(test_file)

def create_imbalanced_test_dataset():
    """Create and save an imbalanced test dataset as CSV file"""
    print("Creating imbalanced test dataset...")
    
    # Create the imbalanced dataset
    df = create_test_data()
    
    # Save to CSV
    test_file = 'test_data_imbalanced.csv'
    df.to_csv(test_file, index=False)
    
    print(f"✅ Imbalanced test dataset saved as '{test_file}'")
    print(f"   File contains {len(df)} records with {df['target'].sum()} positive samples")
    
    return test_file

if __name__ == "__main__":
    print("Running tests for generic transformer classification...")
    
    # Create imbalanced test dataset
    test_file = create_imbalanced_test_dataset()
    
    # Run tests
    test_categorical_dimensions()
    test_preprocessing()
    
    # Test with the imbalanced dataset
    print(f"\nTesting generic main function with imbalanced dataset...")
    try:
        main(data_path=test_file, num_numerical_features=3, num_categorical_features=4)
        print("✅ Generic main function test completed with imbalanced dataset!")
        
    except Exception as e:
        print(f"❌ Error in main function: {e}")
        raise
    
    print("\n🎉 All tests passed! The generic version is working correctly.")
    print(f"📁 You can now use '{test_file}' for further testing with your imbalanced dataset.") 