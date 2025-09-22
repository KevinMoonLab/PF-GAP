#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PF_wrapper as PF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS


# In[2]:


dir1 = "training_output"
dir2 = "training_predictions"


# In[3]:


# First, we train a PF model and give it the name 'Spartacus'

PF.train("Data/GunPoint_TRAIN.tsv", model_name="Spartacus", 
                  return_proximities=True, output_directory=dir1, entry_separator="\t")


# In[4]:


# We can now read a model by name and obtain predictions on another dataset. This creates:
    #1. Predictions_saved.txt: the predicted labels of the read-in model.
# Let's get predictions on the training set to illustrate.
PF.predict(dir1 + "/Spartacus", "Data/GunPoint_TRAIN.tsv", entry_separator="\t", output_directory=dir2)
# Now let's get predictions on the test set.
PF.predict(dir1 + "/Spartacus", "Data/GunPoint_TEST.tsv", entry_separator="\t")


# In[5]:


# Here are the predictions (of the saved model) on the training set.
f0 = open(dir2 + "/Predictions_saved.txt")
f1 = f0.read()
train_preds_saved = eval("np.array(" + f1 + ")")
f0.close()


# In[6]:


# Here are the predictions (of the saved model) on the test set.
f0 = open("Predictions_saved.txt")
f1 = f0.read()
preds_saved = eval("np.array(" + f1 + ")")
f0.close()


# In[7]:


print(len(train_preds_saved))
print(len(preds_saved))


# In[8]:


# the following can be used to obtain the proximities (p) and true training labels (y)
p=PF.getArray(dir1 + "/TrainingProximities.txt")
p.shape


# In[9]:


# The raw proximities are not symmetric. But in some applications, one desires symmetry.
p = 0.5*(p+p.transpose())


# In[10]:


# The proximities can be used to obtain a vector embedding.
# we will prepare a visual, using size to indicate intra-class outlier-ness.
embed = MDS(n_components=2, random_state=0, dissimilarity='precomputed')
dis = (np.ones(p.shape) - p)**4
x_trans = embed.fit_transform(dis)
xt = x_trans.transpose()
#sizes = [x*100 for x in outlier_scores]


# In[11]:


#plt.scatter(xt[0],xt[1],c=train_preds_saved)
#plt.title("MDS Embedding of the GunPoint dataset PF Proximities")
#plt.xlabel("MDS 1")
#plt.ylabel("MDS 2")
#plt.show()


# In[12]:


X1 = np.array([x_trans[i] for i in range(x_trans.shape[0]) if train_preds_saved[i]==1]).transpose()
X2 = np.array([x_trans[i] for i in range(x_trans.shape[0]) if train_preds_saved[i]==2]).transpose()
eX = [X1,X2]

#outs1 = np.array([sizes[i] for i in range(x_trans.shape[0]) if train_preds_saved[i]==1]).transpose()
#outs2 = np.array([sizes[i] for i in range(x_trans.shape[0]) if train_preds_saved[i]==2]).transpose()
#Outs = [outs1,outs2]


# In[13]:


cmap = plt.cm.Paired
#plt.figure(figsize = (15,10))
for i in range(1,3):
    cmap1 = [i-1 for j in range(eX[i-1].shape[1])]
    #if y[np.argmax(sizes)] == i:
     #   cmap1[np.argmax(eval("outs" + str(i)))] = 5
    #plt.scatter(eX[i-1][0], eX[i-1][1], c=cmap(cmap1), s=Outs[i-1],
     #           label="Class {:g}".format(i-1))
    plt.scatter(eX[i-1][0], eX[i-1][1], c=cmap(cmap1),
                label="Class {:g}".format(i-1))

plt.legend()
plt.xlabel('MDS component 1')
plt.ylabel('MDS component 2')
plt.title('MDS embedding using PFGAP')
#plt.savefig("Demo_MDS_GunPointTrain.pdf")
#plt.show()
plt.close()


# In[ ]:




