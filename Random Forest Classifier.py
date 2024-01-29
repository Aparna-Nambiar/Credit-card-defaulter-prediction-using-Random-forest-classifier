#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df= pd.read_csv(r"C:\Users\aparn\OneDrive\Desktop\SET\Default of credit card clients.csv.csv")


# In[3]:


df


# In[9]:


missing_values = df.isnull().sum().sum()
missing_values


# In[47]:


print(df.columns)


# In[40]:


Q1 = df['AGE'].quantile(0.25)
Q3 = df['AGE'].quantile(0.75)
IQR = Q3 - Q1

outliers = (df['AGE'] < Q1 - 1.5 * IQR) | (df['AGE'] > Q3 + 1.5 * IQR)
outliers


# In[42]:


correlation_matrix = df.corr()
correlation_matrix


# In[43]:


# defaulters is somewhat correlated to limit_bal , payments


# In[4]:


# Putting feature variable to X
X = df.drop('default payment next month',axis=1)
X


# In[5]:


# Putting response variable to y
y = df['default payment next month']
y


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[ ]:





# In[10]:


from sklearn.ensemble import RandomForestClassifier


# In[11]:


# Running the random forest with default parameters
rfc = RandomForestClassifier()


# In[12]:


# fit
rfc.fit(X_train,y_train)


# In[13]:


# Making predictions
predictions = rfc.predict(X_test)


# In[15]:


predictions


# In[17]:


from sklearn.metrics import classification_report


# In[18]:


# Let's check the report of our default model
print(classification_report(y_test,predictions))


# In[20]:


from sklearn.metrics import confusion_matrix


# In[21]:


# Printing confusion matrix
print(confusion_matrix(y_test,predictions))



# In[23]:


from sklearn.metrics import accuracy_score


# In[24]:


print(accuracy_score(y_test,predictions))


# In[ ]:




