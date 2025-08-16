#!/usr/bin/env python3
"""
Simple test script to verify the pipeline components work correctly
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Application'))

try:
    from Experiment4functions import create_3d_sphere_data, SEEDS
    print("✓ Successfully imported Experiment4functions")
except ImportError as e:
    print(f"✗ Failed to import Experiment4functions: {e}")
    sys.exit(1)

try:
    from proxUtil import getOutlierScores, SymmetrizeProx
    print("✓ Successfully imported proxUtil functions")
except ImportError as e:
    print(f"✗ Failed to import proxUtil: {e}")
    sys.exit(1)

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    print("✓ Successfully imported scikit-learn components")
except ImportError as e:
    print(f"✗ Failed to import scikit-learn: {e}")
    sys.exit(1)

def test_data_generation():
    """Test that we can generate 3D sphere data"""
    print("\nTesting data generation...")
    
    try:
        data, labels, sphere = create_3d_sphere_data(SEEDS[0])
        print(f"✓ Generated data with shape: {data.shape}")
        print(f"✓ Generated labels with shape: {labels.shape}")
        print(f"✓ Unique labels: {np.unique(labels)}")
        
        # Test train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.3, random_state=SEEDS[0], stratify=labels
        )
        print(f"✓ Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False

def test_knn():
    """Test that KNN works with our data"""
    print("\nTesting KNN classification...")
    
    try:
        data, labels, sphere = create_3d_sphere_data(SEEDS[0])
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.3, random_state=SEEDS[0], stratify=labels
        )
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✓ KNN accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ KNN test failed: {e}")
        return False

def test_directory_creation():
    """Test that we can create the results directories"""
    print("\nTesting directory creation...")
    
    try:
        results_dir = Path(__file__).parent / "results"
        metrics_dir = results_dir / "metrics"
        proximity_dir = results_dir / "proximity_matrices"
        outlier_dir = results_dir / "outlier_scores"
        
        for dir_path in [results_dir, metrics_dir, proximity_dir, outlier_dir]:
            dir_path.mkdir(exist_ok=True)
            print(f"✓ Created/verified directory: {dir_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Directory creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running simple tests for PFGAP vs KNN pipeline...")
    print("=" * 50)
    
    tests = [
        test_data_generation,
        test_knn,
        test_directory_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The pipeline should work correctly.")
        print("\nYou can now run the full pipeline with:")
        print("python test_pipeline.py")
    else:
        print("✗ Some tests failed. Please fix the issues before running the full pipeline.")
        sys.exit(1)

if __name__ == "__main__":
    main()
