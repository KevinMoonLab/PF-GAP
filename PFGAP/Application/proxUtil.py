import subprocess
import numpy as np
import os

def PF_Train(train_file, test_file=None, train_labels=None, test_labels=None, return_proximities=False, save_model=True, model_name="PF", output_directory="", repeats=1, num_trees=11, r=5, on_tree=True, max_depth=0, shuffle=False, export=1, verbosity=1, file_has_header=False, target_column="first", distances=None, memory='1g', parallel_train=False, parallel_prox=False, impute_training_data=False, impute_testing_data=False, impute_iterations=5, return_imputed_training=False, return_imputed_testing=False, data_dimension=1, numeric_data=True, row_separator=",", column_separator=":"):
    
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
    
    if row_separator=="\t":
        row_separator = "\\t"
    if column_separator=="\t":
        column_separator = "\\t"
    
    msgList.extend(["-firstSeparator=" + row_separator])
    msgList.extend(["-secondSeparator=" + column_separator])
    
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


def PF_Predict(model_name, testfile, test_labels=None, output_directory="", shuffle=False, export=1, verbosity=1, file_has_header=False, target_column="first", parallel_train=False, memory='1g', data_dimension=1, numeric_data=True, row_separator=",", column_separator=":"):

    # TODO: implement test proximities and test imputation.
    
    TFdict = {True:"true", False:"false"}
    if (data_dimension!=1 or data_dimension!=2):
        print("Keyword argument 'data_dimension' must be 1 or 2.")
        return
    else:
        if data_dimension==1:
            is2D = False
        else:
            is2D = True
    
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
    msgList.extend(["-csv_has_header=" + file_has_header])
    msgList.extend(["-target_column=" + target_column])
    msgList.extend(["-modelname=" + model_name])
    msgList.extend(["-parallelTrees=" + TF[parallel_train]])
    msgList.extend(["-is2D=" + TFdict[is2D]])
    msgList.extend(["-isNumeric=" + TFdict[numeric_data]])
    msgList.extend(["-firstSeparator=" + row_separator])
    msgList.extend(["-secondSeparator=" + column_separator])
    
    if output_directory=="":
        out = os.getcwd() + "/"
        msgList.extend(["-out=" + out])
    else:
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)
        msgList.extend(["-out=" + output_directory + "/"])

    subprocess.call(msgList)
    return


def getProx(trainfile, testfile=None, getprox="true", savemodel="true", modelname="PF", out="", repeats=1, num_trees=11, r=5, on_tree="true", max_depth=0, shuffle="false", export=1, verbosity=1, csv_has_header="false", target_column="first", distances=None, memory='1g', parallelTrees="false", parallelProx="false"):
    #msgList = ['java', '-jar', '-Xmx1g', 'PFGAP.jar']
    msgList = ['java', '-jar'] #, '-Xmx1g', 'PFGAP.jar']
    msgList.extend(['-Xmx' + memory])
    msgList.extend(['PFGAP.jar', '-eval=false'])
    # Mostly, trainfile, testfile, num_trees, and r are what will be tampered with.
    msgList.extend(["-train=" + trainfile])
    
    if testfile==None:
        msgList.extend(["-test=" + "littleblackraincloud"])
    else:
        msgList.extend(["-test=" + testfile])
    
    #msgList.extend(["-out=" + out])
    msgList.extend(["-repeats=" + str(repeats)])
    msgList.extend(["-trees=" + str(num_trees)])
    msgList.extend(["-r=" + str(r)])
    msgList.extend(["-on_tree=" + on_tree])
    msgList.extend(["-max_depth=" + str(max_depth)]) #max_depth=0 means no max depth.
    msgList.extend(["-shuffle=" + shuffle])
    msgList.extend(["-export=" + str(export)])
    msgList.extend(["-verbosity=" + str(verbosity)])
    msgList.extend(["-csv_has_header=" + csv_has_header]) # we mean this to work primarily with tsv files, actually.
    msgList.extend(["-target_column=" + target_column])
    msgList.extend(["-getprox=" + getprox])
    msgList.extend(["-savemodel=" + savemodel])
    msgList.extend(["-modelname=" + modelname])
    msgList.extend(["-parallelTrees=" + parallelTrees])
    msgList.extend(["-parallelProx=" + parallelProx])
    
    if out=="":
        out = os.getcwd() + "/"
        msgList.extend(["-out=" + out])
    else:
        if not os.path.isdir(out):
            os.mkdir(out)
        msgList.extend(["-out=" + out + "/"])
    
    if distances==None:
        distances="[]"
    else:
        distances = "[" + ",".join(distances) + "]"
        
    msgList.extend(["-distances=" + distances])    

    process = subprocess.call(msgList)
    return
    

def evalPF(testfile, modelname="PF", out="", shuffle="false", export=1, verbosity=1, csv_has_header="false", target_column="first", parallelTrees="false", memory='1g'):
    msgList = ['java', '-jar']
    msgList.extend(['-Xmx' + memory])
    msgList.extend(['PFGAP.jar', '-eval=true'])
    # Mostly, trainfile, testfile, num_trees, and r are what will be tampered with.
    msgList.extend(["-train=" + testfile])
    msgList.extend(["-test=" + testfile])
    #msgList.extend(["-out=" + out])
    msgList.extend(["-shuffle=" + shuffle])
    msgList.extend(["-export=" + str(export)])
    msgList.extend(["-verbosity=" + str(verbosity)])
    msgList.extend(["-csv_has_header=" + csv_has_header]) # we mean this to work primarily with tsv files, actually.
    msgList.extend(["-target_column=" + target_column])
    msgList.extend(["-modelname=" + modelname])
    msgList.extend(["-parallelTrees=" + parallelTrees])
    
    if out=="":
        out = os.getcwd() + "/"
        msgList.extend(["-out=" + out])
    else:
        if not os.path.isdir(out):
            os.mkdir(out)
        msgList.extend(["-out=" + out + "/"])

    subprocess.call(msgList)
    return


def getProxArrays(proxfile="ForestProximities.txt", yfile="ytrain.txt"):
    f1 = open(proxfile)
    f2 = f1.read()
    f2 = f2.replace("{","[")
    f2 = f2.replace("}","]")
    #exec("Arr = np.array(" + f2 + ")")
    proxArr = eval("np.array(" + f2 + ")")
    #f1.close()
    f1 = open(yfile)
    f2 = f1.read()
    f2 = f2.replace("{", "[")
    f2 = f2.replace("}", "]")
    yArr = eval("np.array(" + f2 + ")")
    return proxArr, yArr


def SymmetrizeProx(Pmat):
    PMat = (Pmat + Pmat.transpose()) / 2
    return PMat


# Here is a function to compute within-class outliers.
def getOutlierScores(proxArray, ytrain):
    # the proxArray should be symmetrized first.
    uniqueLabels = np.unique(ytrain)
    mydict = {label:[] for label in uniqueLabels}
    # find out which indices have which class labels:
    for i in range(ytrain.shape[0]):
        label = ytrain[i]
        mydict[label].extend([i])
    # now find the outlier score for each datapoint
    scores = []
    for i in range(ytrain.shape[0]):
        label = ytrain[i]
        PiList = [proxArray[i][k]**2 for k in mydict[label]]
        Pn = np.sum(PiList)
        if Pn == 0:
            print("Warning: index " + str(i) + " has within-class proximity of 0. Changing to 1e-6.")
            Pn = 1e-6
        scores.extend([ytrain.shape[0]/Pn])
    # now we have the raw outlier scores.
    # now let's normalize them.
    medians = {label:0 for label in uniqueLabels}
    mads = {label:0 for label in uniqueLabels}
    for uniquelabel in uniqueLabels:
        proxes = [scores[i] for i in mydict[uniquelabel]]
        medians[uniquelabel] = np.median(proxes)
        mean = np.mean(proxes)
        tosum = [np.abs(x-medians[uniquelabel]) for x in proxes] #[np.abs(x-mean) for x in proxes]
        mads[uniquelabel] = sum(tosum)/len(tosum)
        
    Scores = [np.abs(scores[i]-medians[ytrain[i]])/mads[ytrain[i]] for i in range(len(scores))]
    
    #raw scores are scores.
    return Scores



# example use:
'''mytrain = "/home/ben/Documents/classes/CS7675/Project/UCRArchive_2018/ArrowHead/ArrowHead_TRAIN.tsv"
mytest = "/home/ben/Documents/classes/CS7675/Project/UCRArchive_2018/ArrowHead/ArrowHead_TEST.tsv"
getProx(mytrain, mytest, num_trees=18, r=5)
prox,labels = getProxArrays()
prox = SymmetrizeProx(prox)
getRawOutlierScores(prox,labels)'''
