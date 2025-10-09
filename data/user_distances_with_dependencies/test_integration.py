#!/usr/bin/env python3
"""
Simple test script to verify the molhiv test pipeline integration
"""

import os
import sys
import numpy as np
from pathlib import Path

def test_data_loading():
    """Test if we can load the molhiv dataset"""
    print("Testing data loading...")
    
    data_dir = Path(__file__).parent / "Data"
    train_data_file = data_dir / "molhiv_data_train.txt"
    train_labels_file = data_dir / "molhiv_labels_train.txt"
    
    if not train_data_file.exists():
        print(f"❌ Training data file not found: {train_data_file}")
        return False
    
    if not train_labels_file.exists():
        print(f"❌ Training labels file not found: {train_labels_file}")
        return False
    
    # Load a few samples
    with open(train_data_file, 'r') as f:
        train_data = [line.strip() for line in f.readlines()[:5]]
    
    with open(train_labels_file, 'r') as f:
        train_labels = [int(line.strip()) for line in f.readlines()[:5]]
    
    print(f"✅ Loaded {len(train_data)} training samples")
    print(f"✅ Loaded {len(train_labels)} training labels")
    
    return True

def test_graph_parsing():
    """Test if we can parse graph data"""
    print("\nTesting graph parsing...")
    
    # Sample graph string from the dataset
    sample_graph = "0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,0.0,3.0,5.0,2.0,0.0,1.0,0.0,0.0:1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,5.0,0.0,3.0,5.0,0.0,0.0,1.0,1.0,1.0"
    
    try:
        # Parse the graph
        parts = sample_graph.split(':')
        adjacency_part = parts[0]
        
        adj_values = [float(x) for x in adjacency_part.split(',')]
        
        # The adjacency matrix is not necessarily square
        # Let's assume it's a flattened representation of a square matrix
        # Try to find the square root first
        n_values = len(adj_values)
        n = int(np.sqrt(n_values))
        
        if n * n == n_values:
            # It's a perfect square
            adjacency_matrix = np.array(adj_values).reshape(n, n)
            print(f"✅ Parsed square adjacency matrix: {adjacency_matrix.shape}")
        else:
            # It's not a perfect square, treat as a vector
            adjacency_matrix = np.array(adj_values)
            print(f"✅ Parsed adjacency vector: {adjacency_matrix.shape}")
        
        if len(parts) > 1:
            feature_values = []
            for part in parts[1:]:
                feature_values.extend([float(x) for x in part.split(',')])
            print(f"✅ Parsed node features: {len(feature_values)} values")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph parsing failed: {e}")
        return False

def test_pfgap_wrapper():
    """Test if PF_wrapper can be imported"""
    print("\nTesting PF_wrapper import...")
    
    try:
        import PF_wrapper as PF
        print("✅ PF_wrapper imported successfully")
        return True
    except ImportError as e:
        print(f"❌ PF_wrapper import failed: {e}")
        return False

def test_java_distance():
    """Test if the Java distance JAR exists"""
    print("\nTesting Java distance JAR...")
    
    jar_file = Path(__file__).parent / "userdistances-1.0.jar"
    if jar_file.exists():
        print(f"✅ Java distance JAR found: {jar_file}")
        return True
    else:
        print(f"❌ Java distance JAR not found: {jar_file}")
        return False

def test_dependencies():
    """Test if required Python packages are available"""
    print("\nTesting Python dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'joblib', 'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 Testing MolHIV Pipeline Integration")
    print("=" * 50)
    
    tests = [
        test_dependencies,
        test_data_loading,
        test_graph_parsing,
        test_pfgap_wrapper,
        test_java_distance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline should work correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
