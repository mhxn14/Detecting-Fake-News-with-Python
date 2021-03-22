#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This project will detect fake news using Term Frequency (TF) & Inverse Document Frequency (IDF)


# In[4]:


pip install numpy pandas sklearn


# In[3]:


import pandas as pd 
import pandas as pd
import itertools


# In[7]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[12]:


# READ THE DATA

df = pd.read_csv('news.csv')
df.shape
df.head()


# In[13]:


# GET THE LABELS FROM DATAFRAME

labels = df.label
labels.head()


# In[15]:


#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[16]:


# Now initialize a TFIDVECTORIZER 

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english' , max_df = 0.7)


# In[18]:


# Fit and Transform train sets

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


# In[20]:


# Initialize a Passive Aggressive Classifier

pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)




# In[22]:


# Predict on the test set and calculate accuracy

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[24]:


# We got an accuracy of 92.82%. Now print confusion matrix to gain insight into num of False and True negatives and positives


confusion_matrix(y_test, y_pred, labels = ['FAKE', 'REAL'])




# we learned to detect fake news with Python. We took a political dataset, implemented a TfidfVectorizer, initialized a PassiveAggressiveClassifier, and fit our model. We ended up obtaining an accuracy of 92.82% in magnitude.


# In[ ]:




