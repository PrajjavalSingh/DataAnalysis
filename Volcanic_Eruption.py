#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import random


# In[44]:


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


# In[162]:


data = pd.read_csv('D:\DataBases\Smithsonians_VolcanicEruptions\database.csv')


# In[172]:


data.head(5)  ### Displaying the First Five Rows of the data.


# In[164]:


data.info()


# In[165]:


data.describe()


# In[ ]:





# In[ ]:





# In[166]:


data.profile_report(title='Pandas Profiling of HoloceneVolcanism Dataset',html={'style':{'full_width':True}})  # To display in notebook.
#,html={'style':{'full_width':True}}
#profile=data.profile_report(title='Pandas Profiling of Smithsonian Volcanic Dataset')
#profile.to_file(output_file="profiling_HoloceneVolcanism_dataset.html") # Storing the Profile in separate HTML File 


# In[72]:


#data.rename(columns={'Number': 'Number','Name':'Name','Country':'Country','Region':'Region','Type':'Type','Activity Evidence':'Activity Evidence','Last Known Eruption':'Last Known Eruption','Latitude':'Latitude','Longitude':'Longitude','Elevation (Meters)':'Elevation (Meters)','Dominant':'Dominant','Rock Type':'Rock Type','Tectonic Setting':'Tectonic Setting'}, inplace=True)


# In[73]:


data.hist(figsize=(20,15),color='skyblue')
plt.show()


# In[74]:


#data.plot()  # plots all columns against index
data.plot(kind='scatter',x='Longitude',y='Latitude') # scatter plot
#data.plot(kind='density')  # estimate density function
#data.plot(kind='hist')  # histogram


# In[75]:


import sys
'geopandas' in sys.modules


# In[77]:


data.plot(x='Tectonic_Setting',y='Elevation (Meters)') 


# In[185]:


tectsett_numval = []
for ind in data['Tectonic_Setting']:
    if ind == "Rift Zone / Continental Crust (>25 km)":
        tectsett_numval.append(1)
    elif ind == "Subduction Zone / Continental Crust (>25 km)":
        tectsett_numval.append(2)
    elif ind == "Intraplate / Continental Crust (>25 km)":
        tectsett_numval.append(3)
    elif ind == "Rift Zone / Oceanic Crust (< 15 km)":
        tectsett_numval.append(4)
    elif ind == "Rift Zone / Intermediate Crust (15-25 km)":
        tectsett_numval.append(5)
    elif ind == "Intraplate / Oceanic Crust (< 15 km)":
        tectsett_numval.append(6)
    elif ind == "Subduction Zone / Oceanic Crust (< 15 km)":
        tectsett_numval.append(7)
    elif ind == "Subduction Zone / Crust Thickness Unknown":
        tectsett_numval.append(8)
    elif ind == "Subduction Zone / Intermediate Crust (15-25 km)":
        tectsett_numval.append(9)
    elif ind == "Unknown":
        tectsett_numval.append(10)
    else:
        tectsett_numval.append(0)
    
data['Tectonic Setting [Numerical Value]'] = tectsett_numval


# In[186]:


data.head(5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[187]:


#u, ind = np.unique(data['Tectonic_Setting'], return_inverse=True)
#plt.scatter(ind, data['Dominant_Rock_Type'], s=data['Elevation (Meters)'])
#plt.xticks(range(len(u)), u)

#plt.show()

#data.plot(x='Tectonic Setting',y='Elevation (Meters)') 


# In[188]:


data.plot(x='Tectonic_Setting',y='Elevation (Meters)') 


# In[189]:


data.plot(x='Tectonic_Setting',y='Elevation (Meters)')
figure(num=None, figsize=(10, 12), dpi=120, facecolor='w', edgecolor='k')


# In[190]:


#plt.matshow(data.corr())
#plt.show()


# In[191]:


#f = plt.figure(figsize=(19, 15))
#plt.matshow(data.corr(), fignum=f.number)
#plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)
#plt.yticks(range(data.shape[1]), data.columns, fontsize=14)
#cb = plt.colorbar()
#cb.ax.tick_params(labelsize=14)
#plt.title('Correlation Matrix', fontsize=16);


# In[192]:


#data['Dominant_Rock_Type']=data['Dominant_Rock_Type'].astype('category').cat.codes
#data['Tectonic_Setting']=data['Tectonic_Setting'].astype('category').cat.codes
#data.corr()
#plt.matshow(data.corr())
#plt.show()
#f = plt.figure(figsize=(19, 15))
#plt.matshow(data.corr(), fignum=f.number)
#plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)
#plt.yticks(range(data.shape[1]), data.columns, fontsize=14)
#cb = plt.colorbar()
#cb.ax.tick_params(labelsize=14)
#plt.title('Correlation Matrix', fontsize=16);


# In[193]:


data.head(10)
#rocktyp_tectsett = data['Dominant_Rock_Type','Tectonic_Setting']
#rocktyp_tectsett.head(5)


# In[194]:


rocktyp_tectsett = data.iloc[:,10:12]
rocktyp_tectsett.head(5)


# In[206]:


sns.catplot(x="Tectonic_Setting", y="Dominant_Rock_Type", jitter=False, data=rocktyp_tectsett,height=8.27, aspect=11.7/8.27)


# In[210]:


sns.catplot(x="Tectonic_Setting", y="Dominant_Rock_Type", kind="swarm", data=data,height=20,aspect=60/20)


# In[ ]:





# In[ ]:





# In[197]:


rocktyp_tectsett_onehot = data[['Dominant_Rock_Type','Tectonic Setting [Numerical Value]']]


# In[198]:


from sklearn.preprocessing import OrdinalEncoder


# In[199]:


enc = OrdinalEncoder()
enc.fit(rocktyp_tectsett_onehot)


# In[200]:


rocktyp_tectsett_onehot.head(5)


# In[205]:


sns.catplot(x="Tectonic Setting [Numerical Value]", y="Dominant_Rock_Type", kind="swarm", data=data, height=8.27, aspect=11.7/8.27)


# In[222]:


g = sns.FacetGrid(data, col="Dominant_Rock_Type", hue="Tectonic_Setting",height=8, col_wrap=2)
g.map(sns.scatterplot, "Longitude","Latitude", alpha=.7)
g.add_legend()


# In[ ]:




