#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[3]:


diabetes_dataset = pd.read_csv("C:\\Users\\91762\\Downloads\\diabetes (1).csv")


# In[10]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[8]:


diabetes_dataset.head()


# In[9]:


diabetes_dataset.shape


# In[12]:


diabetes_dataset.describe()


# In[13]:


diabetes_dataset['Outcome'].value_counts()


# In[14]:


diabetes_dataset.groupby('Outcome').mean()


# In[15]:


X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[16]:


print(X)


# In[17]:


print(Y)


# In[18]:


scaler = StandardScaler()


# In[19]:


scaler.fit(X)


# In[20]:


standardized_data = scaler.transform(X)


# In[21]:


print(standardized_data)


# In[22]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[23]:


print(X)
print(Y)


# In[26]:


#Training the Model using Logistic Regression 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[28]:


print(X.shape, X_train.shape, X_test.shape)


# In[32]:


classifier = LogisticRegression()


# In[33]:


classifier.fit(X_train, Y_train)


# In[34]:


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[35]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[36]:


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[37]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[38]:


#Making a Predictive System
input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




