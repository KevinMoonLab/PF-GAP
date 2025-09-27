# PFGAP Data Format Documentation

This document describes the expected data format for using the PFGAP system via the PF_wrapper.py interface.

## Supported File Types
PFGAP expects delimited text files. Commonly supported file extensions include:
- `.csv`
- `.tsv`
- `.txt`

The file extension itself is not a barrier, but the contents must be properly delimited.

## Delimiters
Two delimiters are used to parse the data:
- `entry_separator`: separates entries within a row (default: `,`)
- `array_separator`: separates arrays within a row for 2D data (default: `:`)

These can be customized via arguments to `PF_wrapper.py`.

## Data Dimensionality
Each row in the input file represents a single data instance.

- **1D Data**: Entries in each row are separated by `entry_separator`.
  Example: `1.2,3.4,5.6`

- **2D Data**: Each row contains multiple arrays separated by `array_separator`, and entries within each array are separated by `entry_separator`.
  Example: `1.2,3.4:5.6,7.8`

To specify 2D data, use the argument `data_dimension=2`.

## Label Handling
Labels can be provided in two ways:
- **Inline**: Labels are embedded in the training or testing file. Use `target_column="first"` or `"last"` to indicate label position.
- **Separate File**: Labels are provided via `train_labels` or `test_labels` arguments.

Note:
- For 2D data, labels **cannot** be inferred from the data file and must be provided separately.
- In future versions, users will be able to specify whether test labels are present at all.

## Missing Values
The following strings are interpreted as missing values:
- `""` (empty string)
- `"NA"`
- `"N/A"`
- `"null"`

In future versions, users will be able to customize the list of missing value indicators.

## Notes

- To enable missing value handling, users should set `impute_training_data=True` and/or `impute_testing_data=True` in the `train` method, or `impute_testing_data=True` in the `predict` method. This informs PFGAP to treat recognized missing value strings appropriately. Even if no missing values are present, setting these flags can improve performance by skipping unnecessary checks.

- For numeric datasets, users should set `numeric_data=True` (this is the default). This setting is compatible with missing data. If the dataset contains non-numeric types such as strings, dates, or booleans, users should set `numeric_data=False`.

## Summary
| Feature            | Description |
|--------------------|-------------|
| File Types         | `.csv`, `.tsv`, `.txt` |
| Delimiters         | `entry_separator` (`,`), `array_separator` (`:`) |
| Data Dimensionality| 1D or 2D (`data_dimension=1` or `2`) |
| Label Handling     | Inline or separate file |
| Missing Values     | `""`, `NA`, `N/A`, `null` |

This format ensures compatibility with the DelimitedFileReader class in PFGAP and allows flexible configuration via PF_wrapper.py.
