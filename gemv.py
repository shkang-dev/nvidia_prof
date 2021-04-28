#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.cuda.profiler as profiler
import pyprof
pyprof.init()

# In[2]:


mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 3)
torch.mm(mat1, mat2)


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[4]:

import torch.nn as nn

fc1 = nn.Linear(1024, 1024).to(device)
mat1 = torch.randn(128, 1024).to(device)

profiler.start()
fc1.forward(mat1)
profiler.stop()

# In[ ]:




