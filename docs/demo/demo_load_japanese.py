#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PF_wrapper as PF
import numpy as np
import pandas as pd

from aeon.datasets import load_japanese_vowels


# In[2]:


X3, y3 = load_japanese_vowels(split="TRAIN")
X3t, y3t = load_japanese_vowels(split="TEST")


# In[3]:


def aeonToFile(X3, filename, entry_separator = ',', array_separator = ':'):

    with open(filename, 'w') as f:
        # Loop over the first dimension (the "planes")
        for plane in X3:
            # Loop over the second dimension (the "rows")
            for i, row in enumerate(plane):
                # Convert the row to a string, joining with the item separator
                row_str = entry_separator.join(str(item) for item in row)

                # Write the row string to the file
                f.write(row_str)

                # Add the sub-array separator if not the last row
                if i < len(plane) - 1:
                    f.write(array_separator)

            # Add a newline after each plane
            f.write('\n')


# In[4]:


aeonToFile(X3, 'Data/ThreeDdata.txt')
aeonToFile(X3t, 'Data/ThreeDdata_test.txt')


# In[5]:


y3df = pd.DataFrame(y3)
y3df.to_csv("Data/ThreeDlabels.csv", index=False, header=None)

y3tdf = pd.DataFrame(y3t)
y3tdf.to_csv("Data/ThreeDlabels_test.csv", index=False, header=None)


# In[6]:


dir1 = "training_output"
dir2 = "training_predictions"


# In[7]:


# First, we train a PF model and give it the name 'Spartacus'

PF.train("Data/ThreeDdata.txt", test_file="Data/ThreeDdata_test.txt", distances=['dtw_i', 'dtw_d'], 
                  train_labels="Data/ThreeDlabels.csv", test_labels="Data/ThreeDlabels_test.csv",
                  return_proximities=True, output_directory=dir1, array_separator=":", entry_separator=",", 
                  model_name="Spartacus", data_dimension=2, return_training_outlier_scores=True,
           num_trees=100, parallel_train=True, r=5, parallel_prox=True)


# In[8]:


# Here are the predictions on the provided test set.
f0 = open(dir1 + "/Validation_Predictions.txt")
f1 = f0.read()
preds = eval("np.array(" + f1 + ")")
f0.close()


# In[9]:


# We can now read a model by name and obtain predictions on another dataset. This creates:
    #1. Predictions_saved.txt: the predicted labels of the read-in model.
# Let's get predictions on the training set to illustrate.
PF.predict(dir1 + "/Spartacus", "Data/ThreeDdata.txt", test_labels="Data/ThreeDlabels.csv",
           output_directory=dir2, entry_separator=",", array_separator=":", data_dimension=2)
# Now let's get predictions on the test set.
PF.predict(dir1 + "/Spartacus", "Data/ThreeDdata_test.txt", test_labels="Data/ThreeDlabels_test.csv",
           entry_separator=",", array_separator=":", data_dimension=2)


# In[10]:


# Here are the predictions (of the saved model) on the training set.
f0 = open(dir2 + "/Predictions_saved.txt")
f1 = f0.read()
train_preds_saved = eval("np.array(" + f1 + ")")
f0.close()


# In[11]:


# Here are the predictions (of the saved model) on the test set.
f0 = open("Predictions_saved.txt")
f1 = f0.read()
preds_saved = eval("np.array(" + f1 + ")")
f0.close()


# In[12]:


print(len(train_preds_saved))
print(len(preds_saved))
print(len(preds))


# In[13]:


# Just checking: are the outputs of the saved model equal to the original predictions?
np.unique([preds[i]-preds_saved[i] for i in range(len(preds))])


# In[14]:


# the following can be used to obtain the training proximities
p=PF.getArray(dir1 + "/TrainingProximities.txt")


# In[15]:


p.shape


# In[16]:


# We can also access the test/train proximities
pt=PF.getArray(dir1 + "/TestTrainProximities.txt")


# In[17]:


pt.shape


# In[18]:


# The raw proximities are not symmetric. But in some applications, one desires symmetry.
p = 0.5*(p + p.transpose())


# In[19]:


# The following can be used to obtain outlier scores for the training set.
# Note that these are intra-class outlier scores.
outlier_scores = PF.getArray(dir1 + "/outlier_scores.txt")
outlier_scores.shape


# In[ ]:




