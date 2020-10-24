#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import random
from IPython.display import display
pd.set_option('mode.chained_assignment', None)      # To suppress pandas warnings.
pd.set_option('display.max_colwidth', -1)           # To display all the data in each column
pd.options.display.max_columns = 50                 # To display every column of the dataset in head()


import warnings
warnings.filterwarnings('ignore')                   # To suppress all the warnings in the notebook.

import seaborn as sns
sns.set(style='whitegrid', font_scale=1.3, color_codes=True)      # To apply seaborn styles to the plots.

import pandas_profiling
from pandas_profiling import profile_report

from sklearn.preprocessing import LabelEncoder # For encoding Gender column


# In[16]:


data = pd.read_csv("D:\\DataBases\\Facebook_data\\facebookdata.csv")


# In[7]:


data.head(5)


# In[8]:


data.info()
data.describe()


# In[21]:


profile = data.profile_report(title='Facebook Data Profiling',html={'style':{'full_width':True}})
profile.to_file(output_file="D:\\dev\\DataAnalysis\\FBAnalysis\\profiling_facebook_dataset.html")
data.profile_report(title='Facebook Data Profiling',html={'style':{'full_width':True}})


# In[15]:


data.hist(figsize=(20,15),color='red')
plt.show()


# In[19]:


le = LabelEncoder()
data['gender']= le.fit_transform(data['gender'].astype("|S"))


# In[20]:


sns.set(rc={'figure.figsize':(5,4)})
for idx in data.columns:
    sns.distplot(data[idx])
    plt.show()


# In[ ]:




