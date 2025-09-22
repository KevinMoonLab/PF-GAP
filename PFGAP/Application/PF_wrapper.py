import subprocess
import numpy as np
import os

def train(train_file, test_file=None, train_labels=None, test_labels=None, return_proximities=False, save_model=True, model_name="PF", output_directory="", repeats=1, num_trees=11, r=5, on_tree=True, max_depth=0, shuffle=False, export=1, verbosity=1, file_has_header=False, target_column="first", distances=None, memory='1g', parallel_train=False, parallel_prox=False, impute_training_data=False, impute_testing_data=False, impute_iterations=5, return_imputed_training=False, return_imputed_testing=False, data_dimension=1, numeric_data=True, entry_separator=",", array_separator=":", return_training_outlier_scores=False):
    
    TFdict = {True:"true", False:"false"}
    if (data_dimension not in [1,2]):
        print("Keyword argument 'data_dimension' must be 1 or 2.")
        return
    else:
        if data_dimension==1:
            is2D = False
        else:
            is2D = True
    
    msgList = ['java', '-jar'] #, '-Xmx1g', 'PFGAP.jar']
    msgList.extend(['-Xmx' + memory])
    msgList.extend(['PFGAP.jar', '-eval=false'])
    msgList.extend(["-train=" + train_file])
    msgList.extend(["-train_labels=" + str(train_labels)])
    msgList.extend(["-test=" + str(test_file)])
    msgList.extend(["-test_labels=" + str(test_labels)])
    msgList.extend(["-repeats=" + str(repeats)])
    msgList.extend(["-trees=" + str(num_trees)])
    msgList.extend(["-r=" + str(r)])
    msgList.extend(["-on_tree=" + TFdict[on_tree]])
    msgList.extend(["-max_depth=" + str(max_depth)]) #max_depth=0 means no max depth.
    msgList.extend(["-shuffle=" + TFdict[shuffle]])
    msgList.extend(["-export=" + str(export)])
    msgList.extend(["-verbosity=" + str(verbosity)])
    msgList.extend(["-csv_has_header=" + TFdict[file_has_header]])
    msgList.extend(["-target_column=" + target_column])
    msgList.extend(["-getprox=" + TFdict[return_proximities]])
    msgList.extend(["-savemodel=" + TFdict[save_model]])
    msgList.extend(["-modelname=" + model_name])
    msgList.extend(["-parallelTrees=" + TFdict[parallel_train]])
    msgList.extend(["-parallelProx=" + TFdict[parallel_prox]])    
    msgList.extend(["-hasMissingValues=" + TFdict[impute_training_data]])
    msgList.extend(["-numImputes=" + str(impute_iterations)])
    msgList.extend(["-impute_train=" + TFdict[return_imputed_training]])
    msgList.extend(["-impute_test=" + TFdict[return_imputed_testing]])
    msgList.extend(["-is2D=" + TFdict[is2D]])
    msgList.extend(["-isNumeric=" + TFdict[numeric_data]])
    msgList.extend(["-get_training_outlier_scores=" + TFdict[return_training_outlier_scores]])
    
    if entry_separator=="\t":
        entry_separator = "\\t"
    if array_separator=="\t":
        array_separator = "\\t"
    
    msgList.extend(["-entry_separator=" + entry_separator])
    msgList.extend(["-array_separator=" + array_separator])
    
    if output_directory=="":
        out = os.getcwd() + "/"
        msgList.extend(["-out=" + out])
    else:
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)
        msgList.extend(["-out=" + output_directory + "/"])
    
    if distances==None:
        distances="[]"
    else:
        distances = "[" + ",".join(distances) + "]"
        
    msgList.extend(["-distances=" + distances])    

    process = subprocess.call(msgList)
    return


def predict(model_name, testfile, test_labels=None, output_directory="", shuffle=False, export=1, verbosity=1, file_has_header=False, target_column="first", parallel_train=False, memory='1g', data_dimension=1, numeric_data=True, entry_separator=",", array_separator=":"):
    
    TFdict = {True:"true", False:"false"}
    if (data_dimension not in [1,2]):
        print("Keyword argument 'data_dimension' must be 1 or 2.")
        return
    else:
        if data_dimension==1:
            is2D = False
        else:
            is2D = True
    
    if entry_separator=="\t":
        entry_separator = "\\t"
    if array_separator=="\t":
        array_separator = "\\t"
    
    msgList = ['java', '-jar']
    msgList.extend(['-Xmx' + memory])
    msgList.extend(['PFGAP.jar', '-eval=true'])
    # Mostly, trainfile, testfile, num_trees, and r are what will be tampered with.
    msgList.extend(["-train=" + testfile])
    msgList.extend(["-test=" + testfile])
    msgList.extend(["-test_labels=" + str(test_labels)])
    msgList.extend(["-shuffle=" + TFdict[shuffle]])
    msgList.extend(["-export=" + str(export)])
    msgList.extend(["-verbosity=" + str(verbosity)])
    msgList.extend(["-csv_has_header=" + TFdict[file_has_header]])
    msgList.extend(["-target_column=" + target_column])
    msgList.extend(["-modelname=" + model_name])
    msgList.extend(["-parallelTrees=" + TFdict[parallel_train]])
    msgList.extend(["-is2D=" + TFdict[is2D]])
    msgList.extend(["-isNumeric=" + TFdict[numeric_data]])
    msgList.extend(["-entry_separator=" + entry_separator])
    msgList.extend(["-array_separator=" + array_separator])
    
    if output_directory=="":
        out = os.getcwd() + "/"
        msgList.extend(["-out=" + out])
    else:
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)
        msgList.extend(["-out=" + output_directory + "/"])

    subprocess.call(msgList)
    return



def getArray(filename):
    # this simply reads the Java arrays as numpy arrays.
    # Intended for outlier scores and proximities.
    f1 = open(filename)
    f2 = f1.read()
    f2 = f2.replace("{","[")
    f2 = f2.replace("}","]")
    #exec("Arr = np.array(" + f2 + ")")
    Arr = eval("np.array(" + f2 + ")")
    return Arr

