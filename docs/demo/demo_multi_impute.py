#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PF_wrapper as PF
import numpy as np
import pandas as pd


# In[2]:


dir1 = "training_output"
dir2 = "training_predictions"


# In[3]:


# The tiny, made-up dataset is just for testing purposes.


# In[4]:


PF.train("Data/differentlengths.txt", test_file="Data/differentlengths_test.txt", distances=['dtw_d', 'dtw_i'], 
            train_labels="Data/differentlabels.txt", test_labels="Data/differentlabels_test.txt",
            output_directory=dir1, array_separator=":", entry_separator=",", data_dimension=2,
           impute_training_data=True, return_imputed_training=True, impute_testing_data=True,
           return_imputed_testing=True, impute_iterations=5)


# In[5]:


PF.predict(dir1 + "/PF", "Data/differentlengths_test.txt", output_directory="test_output",
           return_predictions=True, return_proximities=True, impute_testing_data=True,
           return_imputed_testing=True, initial_imputer="linear")

