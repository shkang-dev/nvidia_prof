#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.cuda.profiler as profiler
import pyprof
pyprof.init()

# In[2]:



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[4]:

import torch.nn as nn

mat1 = torch.randn(1024* 1024).to(device)
mat2 = torch.randn(1024* 1024).to(device)
profiler.start()
mat1+mat2
profiler.stop()

# In[ ]:




