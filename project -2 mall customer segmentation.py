#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(r"C:\Users\SWADHINKETAN\Pictures\Saved Pictures\WhatsApp Image 2023-08-22 at 20.24.58.jpg"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data = pd.read_csv(r"C:\Users\SWADHINKETAN\Downloads\archive\Mall_Customers.csv")


# In[6]:


data.head()


# In[7]:


data.info()


# In[8]:


data.shape


# In[9]:


data.describe()


# In[10]:


print(data['Gender'].value_counts())
sns.countplot(x='Gender', data=data)


# In[11]:


# Create a histogram for 'Annual Income (k$)'
plt.hist(data['Annual Income (k$)'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.title('Distribution of Annual Income')
plt.show()


# In[12]:


# Create a histogram for 'Spending Score (1-100)'
plt.hist(data['Spending Score (1-100)'], bins=10, color='lightgreen', edgecolor='black')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')
plt.title('Distribution of Spending Score')
plt.show()


# In[13]:


# Create a histogram for 'Age'
plt.hist(data['Age'], bins=10, color='pink', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()


# In[14]:


# Drop the 'CustomerID' column
data.drop('CustomerID', axis=1, inplace=True)


# In[15]:


# Label Encoding for the 'Gender' column
data['Gender_encoded'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Drop the original 'Gender' column
data.drop('Gender', axis=1, inplace=True)


# In[16]:


data.head()


# In[17]:


print(data.isnull().sum())


# In[19]:


# Create a pair plot of your data
with pd.option_context('mode.use_inf_as_na', True):
    sns.pairplot(data)


# In[20]:


from sklearn.cluster import KMeans
KM = KMeans(n_clusters=5).fit(data)


# In[21]:


KM.labels_


# In[22]:


data['KM_result'] = KM.labels_
data.sort_values(by=['KM_result'])


# In[23]:


data.groupby('KM_result').size()


# In[24]:


data.groupby('KM_result').mean()


# In[25]:


colors = np.array(['red', 'green', 'blue', 'yellow', 'grey'])
from pandas.plotting import scatter_matrix
scatter_matrix(data, s=100, alpha=1, c=colors[data['KM_result']], figsize=(10, 10))


# In[27]:


from sklearn.cluster import MeanShift, estimate_bandwidth

# Estimate bandwidth
bandwidth = estimate_bandwidth(data, quantile=0.1)
print(bandwidth)


# In[28]:


# Create Mean Shift clustering instance with estimated bandwidth
MS = MeanShift(bandwidth=bandwidth)

# Fit the model to your data
MS.fit(data)


# In[29]:


MS.labels_


# In[30]:


# Assign labels to new columns in the DataFrame
data['MS_result'] = MS.labels_
data.sort_values(by=['MS_result'])


# In[31]:


data.groupby('MS_result').size()


# In[32]:


data.groupby('MS_result').mean()


# In[35]:


scatter_matrix(data, s=100, alpha=1, c=colors[data['MS_result']], figsize=(10, 10))


# In[ ]:




