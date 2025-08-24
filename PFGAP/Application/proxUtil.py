import subprocess
import numpy as np

def getProx(trainfile, testfile, getprox="true", savemodel="true", modelname="PF", out="output", repeats=1, num_trees=11, r=5, on_tree="true", shuffle="false", export=1, verbosity=1, csv_has_header="false", target_column="first", distances=None, memory='1g', parallelTrees="false", parallelProx="false"):
    #msgList = ['java', '-jar', '-Xmx1g', 'PFGAP.jar']
    msgList = ['java', '-jar'] #, '-Xmx1g', 'PFGAP.jar']
    msgList.extend(['-Xmx' + memory])
    msgList.extend(['PFGAP.jar'])
    # Mostly, trainfile, testfile, num_trees, and r are what will be tampered with.
    msgList.extend(["-train=" + trainfile])
    msgList.extend(["-test=" + testfile])
    msgList.extend(["-out=" + out])
    msgList.extend(["-repeats=" + str(repeats)])
    msgList.extend(["-trees=" + str(num_trees)])
    msgList.extend(["-r=" + str(r)])
    msgList.extend(["-on_tree=" + on_tree])
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
    
    if distances==None:
        distances="[]"
    else:
        distances = "[" + ",".join(distances) + "]"
        
    msgList.extend(["-distances=" + distances])    

    process = subprocess.call(msgList)
    return
    

def evalPF(testfile, modelname="PF", out="output", shuffle="false", export=1, verbosity=1, csv_has_header="false", target_column="first", parallelTrees="false"):
    msgList = ['java', '-jar', '-Xmx1g', 'PFGAP_eval.jar']
    # Mostly, trainfile, testfile, num_trees, and r are what will be tampered with.
    msgList.extend(["-train=" + testfile])
    msgList.extend(["-test=" + testfile])
    msgList.extend(["-out=" + out])
    msgList.extend(["-shuffle=" + shuffle])
    msgList.extend(["-export=" + str(export)])
    msgList.extend(["-verbosity=" + str(verbosity)])
    msgList.extend(["-csv_has_header=" + csv_has_header]) # we mean this to work primarily with tsv files, actually.
    msgList.extend(["-target_column=" + target_column])
    msgList.extend(["-modelname=" + modelname])
    msgList.extend(["-parallelTrees=" + parallelTrees])

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
