#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

#import sys
#!{sys.executable} -m pip install pandas-profiling 
#!{sys.executable} -m pip install -U phik 
#!{sys.executable} -m pip install geopandas

import pandas_profiling
from pandas_profiling import profile_report

#import geopandas as gp


# In[3]:


data = pd.read_csv("D:\\DataBases\\SolarRadiationPrediction\\SolarPrediction.csv")


# In[5]:


data.head(5)


# In[14]:


data.describe()


# In[15]:


data.info()


# In[6]:


data.profile_report(title='Pandas Profiling of SolarRadiation Dataset',html={'style':{'full_width':True}})


# In[14]:


sns.set(rc={'figure.figsize':(10,10)})
for idx in data.columns:
    dataTypeObj = data.dtypes[idx]
    if dataTypeObj == np.object:
        continue
    sns.distplot(data[idx])
    path = 'D:\\dev\\DataAnalysis\\DataAnalysis\\' + idx + '.png' #concatenating the path '//' is required for escaping single '/' 
    plt.savefig(path)                                             #printing to file
    plt.show()


# In[19]:


data.head(5)


# In[18]:


sns.jointplot(data=data, x=data['Temperature'], y=data['Radiation'])


# In[6]:


plt.figure(figsize=(50,50))
sns.jointplot(
    data=data,
    x=data['Temperature'], y=data['Radiation'],
    kind="kde"
)


# In[13]:


plt.figure(figsize=(50,50))
sns.scatterplot(data=data, x=data['Temperature'], y=data['Radiation'], hue=data['Humidity'])
plt.savefig("D:\\dev\\DataAnalysis\\DataAnalysis\\Temp-Rad-HUE-Humidity.png") 
plt.show()


# In[30]:


plt.figure(figsize=(50,50))
sns.jointplot(
    data=data,
    x=data['Temperature'], y=data['Pressure'], hue=data['Humidity'],
    kind="kde", height=50
)


# In[15]:


plt.figure(figsize=(25,25))
sns.scatterplot(data=data, x=data['Temperature'], y=data['Pressure'], hue=data['Humidity'])
plt.savefig("D:\\dev\\DataAnalysis\\DataAnalysis\\Temp-Pressure-HUE-Humidity.png") 
plt.show()


# In[11]:


plt.figure(figsize=(50,50))
sns.jointplot(
    data=data,
    x=data['Temperature'], y=data['Pressure'],
    kind="kde"
)


# In[16]:


data.head(5)


# In[17]:


plt.figure(figsize=(25,25))
sns.scatterplot(data=data, x=data['WindDirection(Degrees)'], y=data['Speed'], hue=data['Humidity'])
plt.savefig("D:\\dev\\DataAnalysis\\DataAnalysis\\WindDirection(Degrees)-Speed-HUE-Humidity.png") 
plt.show()


# In[20]:


plt.figure(figsize=(25,25))
sns.regplot(x=data['WindDirection(Degrees)'], y=data['Speed'], data=data);


# In[21]:


plt.figure(figsize=(25,25))
sns.regplot(x=data['Radiation'], y=data['Temperature'], data=data);


# In[22]:


plt.figure(figsize=(25,25))
sns.regplot(x=data['Temperature'],y=data['Pressure'],data=data)


# In[23]:


data.head(5)


# In[27]:


plt.figure(figsize=(25,25))
sns.regplot(x=data['Pressure'],y=data['Humidity'],data=data)


# In[28]:


plt.figure(figsize=(25,25))
sns.regplot(x=data['Temperature'],y=data['Humidity'],data=data)


# In[29]:


plt.figure(figsize=(25,25))
sns.regplot(x=data['Humidity'],y=data['WindDirection(Degrees)'],data=data)


# In[37]:


sns.jointplot(
    data=data,
    x=data['Humidity'], y=data['Speed'], hue=data['Temperature'],
    kind="kde", height=25
)


# In[38]:


sns.jointplot(
    data=data,
    x=data['Humidity'], y=data['Speed'],
    kind="kde", height=25
)


# In[ ]:




