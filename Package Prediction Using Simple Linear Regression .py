#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[4]:


df = pd.read_csv('placement.csv')


# In[5]:


df.head()


# In[9]:


plt.scatter(df['cgpa'],df['package'])
plt.xlabel("CGPA")
plt.ylabel("Package (in LPA)")


# In[10]:


x = df.iloc[:,:1]
y = df.iloc[:,-1]


# In[12]:


x.head()


# In[13]:


y.head()


# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)


# In[33]:


from sklearn.linear_model import LinearRegression


# In[37]:


lr = LinearRegression()


# In[39]:


lr.fit(x_train,y_train)


# In[40]:


x_test


# In[41]:


y_test


# In[42]:


lr.predict(x_test.iloc[0].values.reshape(1,1))


# In[47]:


plt.scatter(df['cgpa'],df['package'])
plt.plot(x_train,lr.predict(x_train),color='black')
plt.xlabel('CGPA')
plt.ylabel('Package in LPA')


# In[48]:


m = lr.coef_


# In[49]:


m


# In[52]:


b = lr.intercept_


# In[53]:


b

