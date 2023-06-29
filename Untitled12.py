#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# In[7]:


bank=pd.read_csv('bank-full.csv')


# In[8]:


bank.head()


# In[10]:


bank.shape


# In[82]:


sns.histplot(bank['age'])
plt.show()


# In[12]:


bank.describe()


# In[16]:


bank['job'].value_counts(),bank.shape


# In[20]:


bank['marital'].value_counts(),bank.shape


# In[21]:


bank['education'].value_counts(),bank.shape


# In[22]:


bank['job'].value_counts().keys()


# In[23]:


bank['job'].value_counts().values


# In[49]:


plt.figure(figsize=(10,5))
plt.bar(list(bank['job'].value_counts().keys()[0:5]),list(bank['job'].value_counts()[0:5]),color=["blue","red","orange","yellow","green"])
plt.show()


# In[50]:


bank['marital'].value_counts().keys()


# In[53]:


bank['marital'].value_counts().values


# In[63]:


plt.bar(list(bank['marital'].value_counts().keys()),list(bank['marital'].value_counts().values),color=["red","green","orange"])
plt.show()                                                             


# In[64]:


bank.head()


# In[67]:


bank['education'].value_counts()


# In[68]:


bank['education'].value_counts().keys()


# In[69]:


bank['education'].value_counts().values


# In[71]:


plt.bar(list(bank['education'].value_counts().keys()),list(bank['education'].value_counts().values),color=["yellow","green","orange","red"])
plt.show()


# In[72]:


bank['default'].value_counts()


# In[74]:


bank['default'].value_counts().keys()


# In[75]:


bank['default'].value_counts().values


# In[77]:


plt.bar(list(bank['default'].value_counts().keys()),list(bank['default'].value_counts().values),color=["red","green"])
plt.show()


# In[81]:


sns.histplot(bank['balance'])


# In[84]:


sns.displot(bank['balance'])


# In[90]:


plt.figure(figsize=(10,5))
plt.hist(bank['balance'],bins=5)
plt.show()


# In[91]:


bank.head()


# In[92]:


bank['loan'].value_counts()


# In[93]:


bank['loan'].value_counts().keys()


# In[94]:


bank['loan'].value_counts().values


# In[95]:


plt.bar(list(bank['loan'].value_counts().keys()),list(bank['loan'].value_counts().values),color=["red","green"])
plt.show()


# In[102]:


x=bank[['age']]
y=bank[['balance']] 


# In[103]:


from sklearn.model_selection import train_test_split


# In[104]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[105]:


from sklearn.ensemble import RandomForestRegressor


# In[109]:


rfg=RandomForestRegressor()


# In[110]:


rfg.fit(x_train,y_train)


# In[112]:


y_pred=rfg.predict(x_test)


# In[114]:


y_test.head(),y_pred[0:5]


# In[118]:


from sklearn.metrics import mean_squared_error


# In[119]:


mean_squared_error(y_test,y_pred)


# In[ ]:




