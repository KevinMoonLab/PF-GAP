# MolHIV Dataset Test Pipeline

This directory contains a comprehensive test pipeline for comparing PFGAP with KNN on the ogbg-molhiv dataset using Graph Edit Distance (GED).

## Files

- `test_pipeline_molhiv.py` - Main test pipeline script
- `test_integration.py` - Integration test script to verify setup
- `GraphEditDistance.java` - Java implementation of Graph Edit Distance
- `HungarianAlgorithm.java` - Hungarian algorithm implementation for optimal node matching
- `PF_wrapper.py` - Python wrapper for PFGAP
- `Data/` - Directory containing molhiv dataset files

## Dataset Format

The molhiv dataset contains molecular graphs represented as:
- **Adjacency matrices**: Flattened comma-separated values representing graph structure
- **Node features**: Additional features separated by `:` from adjacency data
- **Labels**: Binary classification (0/1) for HIV inhibition prediction

Example format:
```
0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,0.0,3.0,5.0,2.0,0.0,1.0,0.0,0.0:1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,5.0,0.0,3.0,5.0,0.0,0.0,1.0,1.0,1.0
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install numpy pandas scikit-learn joblib psutil
   ```

2. **Verify setup**:
   ```bash
   python test_integration.py
   ```

3. **Run the test pipeline**:
   ```bash
   python test_pipeline_molhiv.py
   ```

## Test Pipeline Features

### 1. Accuracy Comparison
- **KNN with Euclidean distance**: Standard vectorized approach
- **KNN with GED approximation**: Using structural similarity
- **PFGAP with Graph Edit Distance**: Using the Java GED implementation

### 2. Performance Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1 score
- **Precision/Recall**: Per-class metrics
- **ROC-AUC**: Area under ROC curve (when probabilities available)
- **Runtime**: Execution time comparison
- **Memory Usage**: Memory consumption analysis

### 3. Outlier Detection
- **KNN-based outlier detection**: Using distance-based methods
- **PFGAP outlier detection**: Using proximity matrices

### 4. Reproducibility
- **Consistent seeds**: Uses fixed random seeds for reproducible results
- **Multiple runs**: Tests across multiple seeds for statistical significance
- **Comprehensive logging**: Detailed results saved to JSON files

## Configuration

Key parameters in `test_pipeline_molhiv.py`:

```python
K_VALUES = [1, 3, 5, 7, 10, 15, 20]  # K values for KNN
TRAIN_TEST_SPLIT = 0.8               # Training set proportion
PFGAP_NUM_TREES = 11                 # Number of trees in PFGAP
PFGAP_R = 5                          # R parameter for PFGAP
SEEDS = [42, 123, 456, 789, 101112] # Random seeds for reproducibility
```

## Results

Results are saved in the `molhiv_results/` directory:

- `metrics/` - Individual metric files for each test
- `proximity_matrices/` - PFGAP proximity matrices
- `outlier_scores/` - Outlier detection scores
- `runtime_analysis/` - Performance analysis
- `molhiv_pipeline_summary.json` - Overall summary

## Graph Edit Distance Implementation

The GED implementation uses:
- **Hungarian Algorithm**: For optimal node matching
- **Node substitution costs**: Based on feature similarity
- **Edge edit costs**: Based on structural differences
- **Size penalties**: For graphs of different sizes

## Usage Example

```python
# Run a single test
from test_pipeline_molhiv import run_single_test
result = run_single_test(seed=42)

# Load results
import json
with open('molhiv_results/42_comprehensive_results.json', 'r') as f:
    results = json.load(f)
```

## Troubleshooting

1. **Java JAR not found**: Ensure `userdistances-1.0.jar` is in the directory
2. **Memory issues**: Reduce dataset size or increase system memory
3. **Import errors**: Check that all Python dependencies are installed
4. **Data format issues**: Verify molhiv dataset files are in correct format

## Performance Notes

- **KNN**: Fast but limited by vectorization of graph data
- **PFGAP**: Slower but preserves graph structure through GED
- **Memory**: PFGAP may use more memory due to proximity matrix computation
- **Scalability**: Consider subset sizes for large datasets

## Future Improvements

- [ ] Implement true GED distance function calling Java implementation
- [ ] Add support for probability outputs from PFGAP
- [ ] Implement more sophisticated graph kernels
- [ ] Add visualization of results
- [ ] Support for other graph datasets
