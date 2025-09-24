# PF-GAP

[![Java](https://img.shields.io/badge/Java-17%2B-blue.svg)](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

PF-GAP is a flexible, extensible framework for proximity-based learning on time series and structured data. It builds on the original [Proximity Forest (PF)](https://github.com/fpetitjean/ProximityForest) model and introduces:

- **GAP proximities**
    - Supervised imputation (test and train sets)
    - Intra-class outlier scores
    - Returnable for visualization, SVM kernel, etc.
- **Custom distance functions** (in Python, Maple, or Java)
- **Support for multivariate and variable-length time series**
- **Parallel training and proximity computation**
- **Flexible data formatting and imputation options**
- **Regression (Extrinsic)**
- **Customizable node purity measures and aggregation schemes**

---
## 📚 Table of Contents

1. [Installation](#installation)
2. [Repository Structure](#repository-structure)
3. [Quickstart](#quickstart)
4. [Usage](#usage)
   - [Training](#training)
   - [Prediction](#prediction)
   - [Imputation](#imputation)
   - [Custom Distances](#custom-distances)
5. [Demo Notebooks](#demo-notebooks)
6. [Data Format](#data-format)
7. [Output Files](#output-files)
8. [Citation](#citation)

---


## 🛠 Installation

### Requirements

- **Java 17+**
- **Python 3.8+** (tested with Python 3.13)
- Python packages (for running the demo files):

```bash
  pip install numpy pandas matplotlib scikit-learn aeon
```
- Optional: **Maple 2016+** (for Maple-based distance functions)

## 📂 Repository Structure

```bash
PF-GAP/
├── PFGAP/
│   └── PFGAP/              # Java source code
├── Application/
│   ├── PFGAP.jar           # Compiled Java executable
│   ├── PF_wrapper.py       # Python interface to PFGAP.jar
│   ├── demo_*.py           # Demo scripts (converted from notebooks)
│   ├── Data/               # Sample datasets
│   ├── PythonDistance.py   # Example Python distance
│   └── MapleDistance.mpl   # Example Maple distance
└── README.md
```


## ⚡ Quickstart

```bash
# From the Application directory
python demo_gunpoint.py
```

# 🚀 Usage

# Training
Use PF_wrapper.train() to train a proximity forest:

```python
import PF_wrapper as PF

PF.train(
    train_file="Data/GunPoint_TRAIN.tsv",
    model_name="Spartacus",
    return_proximities=True,
    output_directory="training_output",
    entry_separator="\t"
)
```

# Prediction
Use PF_wrapper.predict() to evaluate a saved model on a test set:

```python
PF.predict(
    model_name="training_output/Spartacus",
    testfile="Data/GunPoint_TEST.tsv",
    entry_separator="\t"
)
```

# Imputation
PF-GAP supports iterative imputation for both training and test sets:

```python
PF.train(
    train_file="Data/differentlengths.txt",
    test_file="Data/differentlengths_test.txt",
    train_labels="Data/differentlabels.txt",
    test_labels="Data/differentlabels_test.txt",
    impute_training_data=True,
    return_imputed_training=True,
    impute_testing_data=True,
    return_imputed_testing=True,
    impute_iterations=5,
    data_dimension=2,
    entry_separator=",",
    array_separator=":"
)
```

Custom Distances
You can define your own distance function in:

- *Python:* PythonDistance.py with a function Distance(list1, list2)
- *Maple:* MapleDistance.mpl with a function Distance(list1, list2)

Specify the distance source using:

```python
distances=["python"]  # or ["maple"]
```


## 📊 Demo Notebooks

| Demo | Description |
|------|-------------|
| `demo_gunpoint.py` | Classic PF classification on UCR GunPoint dataset |
| `demo_multi_impute.py` | Imputation on multivariate time series with missing values |
| `demo_load_japanese.py` | Large-scale multivariate classification with variable-length sequences |
| `demo_regression.py` | Time Series Extrinsic Regression on the FloodModeling1 dataset |

### Example MDS Visualization

![Demo MDS GunPoint Train](PFGAP/Application/Demo_MDS_GunPointTrain.pdf)

---

## 📄 Data Format

PF-GAP supports flexible input formats:

- **UCR-style `.tsv` files** (label + data in one file)
- **Custom delimited files** with:
  - `entry_separator` (e.g., `","`, `"	"`)
  - `array_separator` (e.g., `":"` for 2D arrays)

For multivariate or 3D data, use `data_dimension=2`.

---

## 📂 Output Files

Depending on options, PF-GAP may generate:

| File | Description |
|------|-------------|
| `Predictions.txt` | Predictions on test set |
| `Predictions_saved.txt` | Predictions from a saved model |
| `TrainingProximities.txt` | Proximity matrix for training set |
| `TestTrainProximities.txt` | Proximities between test and train |
| `outlier_scores.txt` | Intra-class outlier scores |
| `imputed_train.txt` | Imputed training data (if requested) |
| `imputed_test.txt` | Imputed test data (if requested) |

Use `PF_wrapper.getArray(filename)` to load proximity or outlier arrays.

🔹 **Outlier Scores**

- Set return_training_outlier_scores=True to compute intra-class outlier scores for the training set.
- These are saved to outlier_scores.txt in the output directory.
- Use PF_wrapper.getArray(output_directory + "outlier_scores.txt") to load them as a NumPy array.
- Note that outlier scores are not supported for regression.

🔹 **Imputed Data**

- If impute_training_data=True and return_imputed_training=True, the imputed training set is saved to:

```bash
[output_directory]/[train_file].txt
```

- Similarly, return_imputed_testing=True saves:

```bash
[output_directory]/[test_file].txt
```

- These files preserve the original format and delimiters.

🔹 **Proximity Matrices**

- If return_proximities=True, proximity matrices are saved to:

    - TrainingProximities.txt (train vs. train)
    - TestTrainProximities.txt (test vs. train)


- These are used internally for imputation and outlier detection, but can also be used for:

    - **MDS or PHATE visualization**
    - **Clustering**
    - **Custom analysis**


Load them with:

```python
p = PF_wrapper.getArray(str(output_directory) + "TrainingProximities.txt")
```

or:

```python
pt = PF_wrapper.getArray(str(output_directory) + "TestTrainProximities.txt")
```

---

## 📖 Citation

If you use PF-GAP in your work, please cite the appropriate paper(s) from the following list:

> Ben Shaw, Jake Rhodes, Soukaina Filali Boubrahimi, and Kevin R. Moon.
> **Forest Proximities for Time Series**, IntelliSys 2025  
> [arXiv preprint](https://arxiv.org/abs/2410.03098)
