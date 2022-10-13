#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix


# In[3]:


data= pd.read_csv("C:/Users/lenovo/Downloads/Iris.csv")


# In[5]:


data.shape


# In[6]:


data.describe()


# In[7]:


data.head()


# In[10]:


data.columns


# In[11]:


data.tail()


# In[13]:


data.isnull().any()


# In[15]:


sns.boxplot(y=data['SepalLengthCm'])


# In[16]:


sns.boxplot(y=data['SepalWidthCm'])


# In[17]:


sns.boxplot(y=data['PetalLengthCm'])


# In[18]:


sns.boxplot(y=data['PetalWidthCm'])


# In[20]:


data['Species'].value_counts()


# In[21]:


sns.pairplot(data, hue='Species')


# In[25]:


from pandas.plotting import parallel_coordinates
plt.figure(figsize=(15,10))
parallel_coordinates(data.drop("Id", axis=1), "Species")
plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
plt.xlabel('Features', fontsize=15)
plt.ylabel('Features values', fontsize=15)
plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
plt.show()


# In[26]:


from pandas.plotting import andrews_curves
plt.figure(figsize=(15,10))
andrews_curves(data.drop("Id", axis=1), "Species")
plt.title('curve plot', fontsize=20, fontweight='bold')
plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
plt.show()


# In[27]:


from sklearn.preprocessing import StandardScaler
features = data.drop('Species', axis=1)
target = data['Species']


# In[28]:


scale = StandardScaler()


# In[29]:


scale.fit(features)


# In[30]:


scaled_features=scale.transform(features)


# In[31]:


data_new = pd.DataFrame(scaled_features)
data_new.head(3)


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(data_new, target, test_size=0.25, random_state=45)


# In[33]:


x_train.shape


# In[34]:


x_train.head()


# In[35]:


model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)


# In[36]:


pred = model.predict(x_test)
pred


# In[37]:


print(classification_report(y_test, pred))


# In[38]:


accuracy = model.score(x_test, y_test)
print(accuracy*100,'%')


# In[ ]:





# In[ ]:





# In[ ]:




