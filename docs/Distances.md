# Distance Functions in PFGAP

This document describes the supported distance functions in PFGAP and how to configure them.

## Built-in Distances
The following built-in distances are supported:

- erp
- lcss
- msm
- twe
- wdtw
- dtw
- wddtw
- ddtw
- shifazDTW
- shifazDDTW
- shifazWDTW
- shifazWDDTW
- shifazERP
- shifazMSM
- shifazLCSS
- shifazTWE
- shapeHoG1dDTW

- euclidean
- manhattan

- dtw_i
- ddtw_i
- wdtw_i
- wddtw_i
- twe_i
- erp_i
- euclidean_i
- lcss_i
- msm_i
- manhattan_i
- cid_i
- sbd_i

- dtw_d
- ddtw_d
- wdtw_d
- wddtw_d
- shapeHoGdtw_d
- shapeHoGdtw

To use these distances, specify them in the `distances` argument of the `train` method:
distances = ["dtw", "euclidean", "manhattan"]

## Custom Java Distances
For instructions on using custom Java distances, see the separate documentation file:
`Custom_Java_Distances.md`

## Interoperability with Python and Maple

PFGAP supports limited interop with external languages for distance computation.

### Python Interop
To use a Python-defined distance function, specify `"python"` in the `distances` list.
The Python function must be defined in a file named `PythonDistance.py` located in the same directory as `PF_wrapper.py`.
The function should be named `Distance` and accept two arguments: `x` and `y`.

Example usage:
distances = ["python"]

### Maple Interop
To use a Maple-defined distance function, specify `"maple"` in the `distances` list.
The Maple function must be defined in a file named `MapleDistance.mpl` located in the same directory as `PF_wrapper.py`.
The function should be named `Distance` and accept two arguments: `x` and `y`.

Example usage:
distances = ["maple"]
