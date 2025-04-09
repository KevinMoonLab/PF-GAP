# PF-GAP

This repository provides an implementation of PF-GAP. This implementation makes extensive use of the java-based implementation of proximity forests (https://github.com/fpetitjean/ProximityForest/tree/master).

## Requirements

Java 17, Python 3.

## Using PF-GAP

It is not necessary to clone this repository to use PF-GAP. One only needs the files in the Application folder, namely the .jar files and the proxUtil.py file. The proxUtil.py file is not strictly necessary, but rather provides a convenient mechanism for calling the .jar files in python. The proxUtil.py file contains the function getProx, which calls the PFGAP.jar file, building and training a Proximity Forest using the training data, then applying Proximity Forest on the test data. The training/test data files are specified in the function arguments. Other parameters may be passed to this function as well, including the desired number of trees and r parameter. By default, the PFGAP.jar file creates a "Predictions.txt" file containing the predictions on the test dataset, a "ForestProximities.txt" file containing the array of forest proximities, and a "ytrain.txt" file containing the ground-truth class labels from the training dataset.

The output of the getProxArrays() is twofold: the (numpy) array of proximities read from the "Proximities.txt" file, and the (numpy) array of training labels read from the "ytrain.txt" file. The proximity array can be symmetrized with the SymmetrizeProx(ProximityArray) function (not in-place). The getOutlierScores(ProximityArray,ytrain) function is used to compute within-class outlier scores: it returns a list.

By Default, the getProx function also creates a modelname.ser file of the serialized trained proximity forest. This can be used to evaluate additional test data using the evalPF(testdata, modelname="PF") function, which function calls the PFGAP_eval.jar file to perform the evaluation. The PFGAP_eval.jar file also creates a "Predictions_saved.txt" file containing the model predictions on the evaluated data.

## Data format

The program is designed to be compatible with .tsv files formatted in the same way as files from the UCR 2018 repository (https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).

## Associated Papers

"Forest Proximities for Time Series," IntelliSys 2025. Project page: https://sites.google.com/view/forest-proximities
