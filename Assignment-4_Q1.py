#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[2]:


# import dataset
dataset=pd.read_csv('delivery_time.csv')
dataset


# In[3]:


dataset.info()


# In[4]:


sns.distplot(dataset['Delivery Time'])


# In[5]:


sns.distplot(dataset['Sorting Time'])


# In[6]:


# Renaming Columns
dataset=dataset.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
dataset


# In[7]:


dataset.corr()


# In[8]:


sns.regplot(x=dataset['sorting_time'],y=dataset['delivery_time'])


# In[9]:


model=smf.ols("delivery_time~sorting_time",data=dataset).fit()


# In[10]:


# Finding Coefficient parameters
model.params


# In[11]:


# Finding tvalues and pvalues
model.tvalues , model.pvalues


# In[12]:


# Finding Rsquared Values
model.rsquared , model.rsquared_adj


# In[13]:


# Manual prediction for say sorting time 5
delivery_time = (6.582734) + (1.649020)*(5)
delivery_time


# In[14]:


# Automatic Prediction for say sorting time 5, 8
new_data=pd.Series([5,8])
new_data


# In[15]:


data_pred=pd.DataFrame(new_data,columns=['sorting_time'])
data_pred


# In[16]:


model.predict(data_pred)


# In[ ]:




