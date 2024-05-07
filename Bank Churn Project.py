#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Bank%20Churn%20Modelling.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.duplicated('CustomerId')


# In[5]:


df.duplicated('CustomerId').sum()


# In[6]:


df = df.set_index('CustomerId')


# In[7]:


df.info()


# # Encoding

# In[8]:


df['Geography'].value_counts()


# In[9]:


df.replace({'Geography': {'France':2, 'Germany':1, 'Spain':0}}, inplace = True)


# In[10]:


df


# In[11]:


df.replace({'Gender': {'Female':1, 'Male':0}}, inplace = True)


# In[12]:


df['Num Of Products'].value_counts()


# In[13]:


df.replace({'Num Of Products': {1:0, 2:1, 3:1, 4:1}}, inplace = True)


# In[14]:


df


# In[15]:


df['Has Credit Card'].value_counts() 


# In[16]:


df['Is Active Member'].value_counts()


# In[17]:


df.loc[(df['Balance']==0), 'Churn'].value_counts()


# In[18]:


import numpy as np


# In[19]:


df['Zero Balance'] = np.where(df['Balance']>0, 1, 0)


# In[20]:


df['Zero Balance'].hist()


# In[21]:


df.groupby (['Churn','Geography']).count()


# # Define Label and Features

# In[22]:


df.columns


# In[23]:


X = df.drop(['Surname','Churn'], axis = 1)


# In[24]:


Y = df['Churn']


# In[25]:


X.shape , Y.shape


# # Handling Imbalance Data

# In[26]:


df['Churn'].value_counts()


# In[27]:


import seaborn as sns


# In[28]:


sns.countplot(x = 'Churn', data = df)


# # Random Under Sampling

# In[29]:


from imblearn.under_sampling import RandomUnderSampler


# In[30]:


rus = RandomUnderSampler(random_state = 2529)


# In[31]:


X_rus,Y_rus = rus.fit_resample(X,Y)


# In[32]:


X_rus.shape, Y_rus.shape, X.shape, Y.shape 


# In[33]:


Y.value_counts()


# In[34]:


Y_rus.value_counts()


# In[35]:


import matplotlib.pyplot as plt


# In[36]:


Y_rus.plot(kind = 'hist')


# # Random Over Sampling

# In[37]:


from imblearn.over_sampling import RandomOverSampler


# In[38]:


ros = RandomOverSampler(random_state = 2529)


# In[39]:


X_ros,Y_ros = ros.fit_resample(X,Y)


# In[40]:


X_ros.shape, Y_ros.shape, X.shape, Y.shape 


# In[41]:


Y.value_counts()


# In[42]:


Y_ros.value_counts()


# In[43]:


Y_ros.plot(kind = 'hist')


# # Test Train Split

# # Split Orignal Data

# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3 , random_state = 25)


# # Split Random Under Sample Data

# In[45]:


X_train_rus, X_test_rus, Y_train_rus, Y_test_rus = train_test_split(X_rus, Y_rus, test_size = 0.3 , random_state = 25)


# # Split Random Over Sample Data

# In[46]:


X_train_ros, X_test_ros, Y_train_ros, Y_test_ros = train_test_split(X_ros, Y_ros, test_size = 0.3 , random_state = 25)


# # Standardize Features

# In[47]:


from sklearn.preprocessing import StandardScaler


# In[48]:


sc = StandardScaler()


# # Standardize Orignal Data

# In[49]:


X_train[['CreditScore', 'Age','Tenure','Balance','Estimated Salary']]  = sc.fit_transform(X_train[['CreditScore', 'Age','Tenure','Balance','Estimated Salary']])


# In[50]:


X_test[['CreditScore', 'Age','Tenure','Balance','Estimated Salary']]  = sc.fit_transform(X_test[['CreditScore', 'Age','Tenure','Balance','Estimated Salary']])


# # Standardize Random Under Sample Data

# In[51]:


X_train_rus[['CreditScore', 'Age','Tenure','Balance','Estimated Salary']]  = sc.fit_transform(X_train_rus[['CreditScore', 'Age','Tenure','Balance','Estimated Salary']])


# In[52]:


X_test_rus[['CreditScore', 'Age','Tenure','Balance','Estimated Salary']]  = sc.fit_transform(X_test_rus[['CreditScore', 'Age','Tenure','Balance','Estimated Salary']])


# # Standardize Random Over Sample Data

# In[53]:


X_train_ros[['CreditScore', 'Age','Tenure','Balance','Estimated Salary']]  = sc.fit_transform(X_train_ros[['CreditScore', 'Age','Tenure','Balance','Estimated Salary']])


# In[54]:


X_test_ros[['CreditScore', 'Age','Tenure','Balance','Estimated Salary']]  = sc.fit_transform(X_test_ros[['CreditScore', 'Age','Tenure','Balance','Estimated Salary']])


# # Modelling

# In[55]:


from sklearn.svm import SVC


# In[56]:


model = SVC()


# In[57]:


model.fit (X_train,Y_train)


# In[58]:


Y_pred = model.predict(X_test)


# # Model Accuracy

# In[59]:


from sklearn.metrics import confusion_matrix, classification_report 


# In[60]:


confusion_matrix(Y_test,Y_pred)


# In[61]:


print(classification_report(Y_test, Y_pred))


# In[62]:


from sklearn.model_selection import GridSearchCV


# In[63]:


param_grid = {'C': [0.1,1,10],
               'gamma': [1,0.1,0.01],
               'kernel': ['rbf'],
               'class_weight': ['balanced']}


# In[64]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 2, cv = 2)


# In[65]:


grid.fit(X_train, Y_train)


# In[69]:


print(grid.best_estimator_)


# In[70]:


grid_predictions = grid.predict(X_test)


# In[71]:


confusion_matrix(Y_test, grid_predictions)


# In[72]:


print(classification_report(Y_test, grid_predictions))


# # Model with Random Under Sampling

# In[73]:


svc_rus = SVC()


# In[74]:


svc_rus.fit(X_train_rus,Y_train_rus)


# In[75]:


Y_pred_rus = svc_rus.predict(X_test_rus)


# # Model Accuracy

# In[76]:


confusion_matrix(Y_test_rus, Y_pred_rus)


# In[77]:


print(classification_report(Y_test_rus, Y_pred_rus))


# # Hyperparameter Tunning

# In[78]:


param_grid = {'C': [0.1,1,10],
               'gamma': [1,0.1,0.01],
               'kernel': ['rbf'],
               'class_weight': ['balanced']}


# In[79]:


grid_rus= GridSearchCV(SVC(), param_grid, refit = True, verbose = 2, cv = 2)
grid_rus.fit(X_train_rus, Y_train_rus)


# In[80]:


print(grid_rus.best_estimator_)


# In[81]:


grid_predictions_rus = grid_rus.predict(X_test_rus)


# In[82]:


confusion_matrix(Y_test_rus,grid_predictions_rus)


# In[83]:


print(classification_report(Y_test_rus,grid_predictions_rus))


# # Model with Random Over Sampling

# In[84]:


svc_ros = SVC()


# In[85]:


svc_ros.fit(X_train_ros,Y_train_ros)


# In[86]:


Y_pred_ros = svc_ros.predict(X_test_ros)


# # Model Accuracy

# In[87]:


confusion_matrix(Y_test_ros, Y_pred_ros)


# In[88]:


print(classification_report(Y_test_ros, Y_pred_ros))


# # Hyperparameter Tunning

# In[89]:


param_grid = {'C': [0.1,1,10],
               'gamma': [1,0.1,0.01],
               'kernel': ['rbf'],
               'class_weight': ['balanced']}


# In[90]:


grid_ros= GridSearchCV(SVC(), param_grid, refit = True, verbose = 2, cv = 2)
grid_ros.fit(X_train_ros, Y_train_ros)


# In[91]:


print(grid_ros.best_estimator_)


# In[92]:


grid_predictions_ros = grid_ros.predict(X_test_ros)


# In[93]:


confusion_matrix(Y_test_ros,grid_predictions_ros)


# In[94]:


print(classification_report(Y_test_ros,grid_predictions_ros))


# In[ ]:


# ROS model is the best among all 3 models

