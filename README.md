# PF-GAP

This repository provides an implementation of PF-GAP. This implementation makes extensive use of the java-based implementation of proximity forests (referred to here as the PF model, https://github.com/fpetitjean/ProximityForest/tree/master).

PF-GAP is an extension of the original PF model. First, PF-GAP extends PF by the inclusion of RF-GAP proximities (https://github.com/jakerhodes/RF-GAP-Python), useful in particular for outlier detection, imputation, and visualization. Second, PF-GAP allows for distance customization, extending the PF model to any data type for which a distance measure can be defined (and where class labels are given). Note that in order to use GAP proximities, PF-GAP also uses bootstrap sampling, though the original PF model does not.

## Requirements

Java 17. Recommended: python 3 with packages "numpy", "subprocess", and "os".
Optional: Maple (2016+). This only applies if you want to use Maple to compute distances.

## Using PF-GAP

It is not necessary to clone this repository to use PF-GAP. One only needs three files in the Application folder, namely the .jar files and the proxUtil.py file. In the Application folder, the "PythonDistance.py" and "MapleDistance.mpl" files are provided as examples if a user wants to define a distance measure using either python or Maple. The "Demo.ipynb" file illustrates the use of PFGAP, with minimal data provided in the "Data/" folder in order to run the notebook. 

The proxUtil.py file is not strictly necessary, but rather provides a convenient mechanism for calling the .jar files in python. The proxUtil.py file contains the function getProx, which calls the PFGAP.jar file, building and training a Proximity Forest using the training data, then applying Proximity Forest on the test data, optionally computing proximities for the training samples. The training/test data files are specified in the function arguments. Other parameters may be passed to this function as well, including the desired number of trees and r parameter. By default, the PFGAP.jar file creates a "Predictions.txt" file containing the predictions on the test dataset, a "ForestProximities.txt" file containing the array of forest proximities, and a "ytrain.txt" file containing the ground-truth class labels (currently re-mapped) from the training dataset. 

The output of the getProxArrays() is twofold: the (numpy) array of proximities read from the "Proximities.txt" file, and the (numpy) array of training labels read from the "ytrain.txt" file. The proximity array can be symmetrized with the SymmetrizeProx(ProximityArray) function (not in-place). The getOutlierScores(ProximityArray,ytrain) function is used to compute within-class outlier scores: it returns a list.

By Default, the getProx() function also creates a modelname.ser file of the serialized trained proximity forest. This can be used to evaluate additional test data using the evalPF(testdata, modelname="PF") function, which function calls the PFGAP_eval.jar file to perform the evaluation. The PFGAP_eval.jar file also creates a "Predictions_saved.txt" file containing the model predictions on the evaluated data.

## Custom Distances in Python or Maple

The "distances" keyword can be used to specify which distances are to be used (using proxUtil.py, this is passed using a list of strings). For a Python-defined distance, the keyword argument is ["python"], while for a Maple-defined distance, the keyword argument is ["maple"]. In either case, a file, "PythonDistance.py" or "MapleDistance.mpl", respectively, must be provided.

At the time of writing, if a python- or maple-defined distance is used, only a single distance should be used at a time, meaning the keyword argument "distances" should contain a list of only a single element, which element is either "python" or "maple", respectively. Currently, python (or Maple) is called in Java via the command terminal, and the output is then read back into Java. Future improvements to this interface (using Jython and/or OpenMaple) may be considered in the future. (Of course, most distances can also just be written in Java, which would be the faster option.)

When creating a "PythonDistance.py" or "MapleDistance.mpl" file, the name of the function/procedure to compute distances must be named "Distance" and must take, as input, two lists. See the provided PythonDistance.py and MapleDistance.mpl files in the Application folder for examples.

## Data format

The program is designed to be compatible with .tsv files formatted in the same way as files from the UCR 2018 repository (https://www.cs.ucr.edu/~eamonn/time_series_data_2018/). In particular, the class label and the data is given in the same file.

For more generic data types with user-defined distances, it is possible to pass a .tsv file into PF-GAP containing the training labels and indices corresponding to the training instances. Then, a user-created distance can reference another datafile using the indices which are passed from PF-GAP to the custom distance.

## Associsted Papers

"Forest Proximities for Time Series," IntelliSys 2025. Project page: https://sites.google.com/view/forest-proximities
