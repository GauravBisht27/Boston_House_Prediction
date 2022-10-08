#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_csv("C:\\Users\\Gaurav singh\\Downloads\\Boston.csv")
df


# # Data Exploration

# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum().sum()


# In[7]:


df['chas'].value_counts()


# In[8]:


df['zn'].value_counts()


# In[9]:


#df=df.drop(["chas"],axis=1)
df.dtypes


# In[10]:


df.corr()
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[11]:


y3=df['medv']
for i in df.columns:
    x3=df[i]
    plt.scatter(x3,y3)
    plt.show()


# # Data Preprocessing
# 

# In[12]:


x_train=df.drop(['medv'],axis=1)
y_train=df['medv']
x_train


# # Data Modeling

# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


lr=LinearRegression()
lr.fit(x_train,y_train)


# In[15]:


df2=pd.read_csv("C:\\Users\\Dipesh Singh\\Downloads\\Boston_Test.csv")
#df2=df2.drop(['chas'],axis=1)
df2


# In[16]:


x_test=df2.drop(['medv'],axis=1)
y_test=df2['medv']
x_test


# In[17]:


y_pred=lr.predict(x_test)
plt.scatter(y_test,y_pred)
plt.show()


# In[18]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
mse=mean_squared_error(y_test,y_pred)
ae=mean_absolute_error(y_test,y_pred)
re=r2_score(y_test,y_pred)
print(mse)
print(ae)
print(re)


# Score By Using Linear Regression 

# In[19]:


lr.score(x_test,y_test)


# # Random Forest

# In[20]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(x_train, y_train) 


# In[21]:


y_pred=lr.predict(x_test)
plt.scatter(y_test,y_pred)
plt.show()


# Score by using Random Forest 

# In[22]:


regressor.score(x_test,y_test)


# In[ ]:




