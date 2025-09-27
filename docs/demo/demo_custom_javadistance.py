#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PF_wrapper as PF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


dir1 = "training_output"
dir2 = "training_predictions"


# In[3]:


#distances=["javadistance:userdistances.jar:MyEuclideanDistance"]


# In[7]:


PF.train("Data/GunPoint_TRAIN.tsv", test_file="Data/GunPoint_TEST.tsv", model_name="Spartacus", 
         distances=["javadistance:userdistances.jar:MyEuclideanDistance"],
                  return_proximities=True, output_directory=dir1, entry_separator="\t")


# In[ ]:




