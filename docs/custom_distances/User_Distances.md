# Custom Distance Integration for PFGAP

This guide explains how to define, compile, and integrate custom distance functions for use with the `PFGAP.jar` application. It includes instructions for compiling `.java` files and optionally packaging them into a `.jar` file. To pack multiple distances into the same `.jar` file, or to incorporate external libraries, see `User_Distances_with_Dependencies.md`.

---

## Overview

PFGAP supports user-defined distance functions written in Java. These functions must implement the following interface:

```java
package distance.api;

public interface DistanceFunction {
    double compute(Object t1, Object t2);
}
```

## Example Distance file

An example of a Euclidean distance implementation looks like

```java
import distance.api.DistanceFunction;

public class MyEuclideanDistance implements DistanceFunction {
    @Override
    public double compute(Object t1, Object t2) {
        double[] a = (double[]) t1;
        double[] b = (double[]) t2;
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
}
```

## Compilation

To use a custom `.java` file defining a distance, it can be compiled into a `.class` file or `.jar` file. Start with (assuming `PFGAP.jar` is in the same directory--adjust paths if needed).

```bash
javac -cp PFGAP.jar MyEuclideanDistance.java
```
This will generate `MyEuclideanDistance.class`. You may then optionally package it as a `.jar` file:

```bash
jar cf userdistances.jar MyEuclideanDistance.class
```

## Runtime Usage

When calling `PFGAP.jar` using the PF_wrapper.py file, use

```python
distance = ["javadistance:/full/path/to/MyEuclideanDistance.class"]
```
for a `.class` file (note that multiple distances can be used). For a `.jar` file, use

```python
distance = ["javadistance:/full/path/to/userdistances.jar:MyEuclideanDistance"]
```

Note that multiple distances can be contained in the same `.jar` file.
