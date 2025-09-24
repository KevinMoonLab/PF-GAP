#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PF_wrapper as PF
import numpy as np
import pandas as pd

from aeon.datasets import load_regression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[2]:


X_train, y_train = load_regression("FloodModeling1",split="TRAIN")
X_test, y_test = load_regression("FloodModeling1",split="TEST")


# In[3]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[4]:


X_train = X_train.reshape((X_train.shape[0],X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[2]))


# In[5]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[6]:


Xdf = pd.DataFrame(X_train)
Xdf.to_csv("Data/FloodTrain.csv", index=False, header=None)

ydf = pd.DataFrame(y_train)
ydf.to_csv("Data/FloodTrain_targets.csv", index=False, header=None)

Xtdf = pd.DataFrame(X_test)
Xtdf.to_csv("Data/FloodTest.csv", index=False, header=None)

ytdf = pd.DataFrame(y_test)
ytdf.to_csv("Data/FloodTest_targets.csv", index=False, header=None)


# In[7]:


dir1 = "training_output"
dir2 = "training_predictions"


# In[8]:


# First, we train a PF model and give it the name 'Spartacus'

PF.train("Data/FloodTrain.csv", test_file="Data/FloodTest.csv", 
                  train_labels="Data/FloodTrain_targets.csv", test_labels="Data/FloodTest_targets.csv",
                  return_proximities=True, output_directory=dir1, entry_separator=",", distances=['shifazDTW'],
                  model_name="Spartacus", num_trees=100, parallel_prox=True, purity="mad", purity_threshold=1e-8,
        parallel_train=True, r=5, regressor=True)


# In[9]:


# Here are the predictions on the provided test set.
f0 = open(dir1 + "/Validation_Predictions.txt")
f1 = f0.read()
preds = eval("np.array(" + f1 + ")")
f0.close()


# In[10]:


print(r2_score(y_test,preds))


# In[11]:


print(mean_absolute_error(y_test,preds))


# In[ ]:




