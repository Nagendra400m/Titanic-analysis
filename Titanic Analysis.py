#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()


# In[3]:


test.head()


# In[4]:


train.describe()


# In[5]:


test.describe()


# In[6]:


train.info()


# In[7]:


test.info()


# In[8]:


train.isnull().sum()


# In[9]:


test.isnull().sum()


# In[10]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[11]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[12]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[13]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[14]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[15]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[16]:


train['Age'].hist(bins=30,color='darkred',alpha=0.3)


# In[17]:


sns.countplot(x='SibSp',data=train)


# In[18]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[19]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[20]:


train['Age'].fillna(train['Age'].mean(),inplace=True)


# In[21]:


train.drop(['PassengerId','Name','Ticket','Cabin','Embarked','SibSp','Parch'], axis=1, inplace=True )


# In[22]:


train.isnull().sum()


# In[23]:


sex_dummies=pd.get_dummies(train['Sex'],drop_first=True)


# In[24]:


train= pd.concat([train,sex_dummies],axis=1)
train.head()


# In[25]:


train.drop(['Sex'], axis=1, inplace=True )


# In[26]:


train.head()


# In[27]:


from sklearn.preprocessing import StandardScaler
sts =StandardScaler()


# In[28]:


feature_scale = ['Age','Fare']
train[feature_scale] = sts.fit_transform(train[feature_scale])


# In[29]:


train.head()


# In[30]:


X=train.drop(['Survived'],axis=1)
y=train['Survived']


# In[31]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[32]:


#create param
model_param = {
    'DecisionTreeClassifier':{
        'model':DecisionTreeClassifier(),
        'param':{
            'criterion': ['gini','entropy']
        }
    },
        'KNeighborsClassifier':{
        'model':KNeighborsClassifier(),
        'param':{
            'n_neighbors': [5,10,15,20,25]
        }
    },
        'SVC':{
        'model':SVC(),
        'param':{
            'kernel':['rbf','linear','sigmoid'],
            'C': [0.1, 1, 10, 100]
         
        }
    }
}


# In[33]:


scores =[]
for model_name, mp in model_param.items():
    model_selection = GridSearchCV(estimator=mp['model'],param_grid=mp['param'],cv=5,return_train_score=False)
    model_selection.fit(X,y)
    scores.append({
        'model': model_name,
        'best_score': model_selection.best_score_,
        'best_params': model_selection.best_params_
    })


# In[34]:


df_model_score = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df_model_score


# In[35]:


model_svc = SVC( C= 100,kernel='rbf')


# In[36]:


model_svc.fit(X, y)


# In[37]:


test.head()


# In[38]:


df1=test.drop(['PassengerId','Name','Ticket','Cabin','Embarked','SibSp','Parch'], axis=1 )


# In[39]:


df1.head()


# In[40]:


df1.isnull().sum()


# In[41]:


df1['Age'].fillna(df1['Age'].mean(),inplace=True)
df1['Fare'].fillna(df1['Fare'].mean(),inplace=True)


# In[42]:


l_sex_dummies=pd.get_dummies(df1['Sex'],drop_first=True)
df1= pd.concat([df1,l_sex_dummies],axis=1)
df1.drop(['Sex'], axis=1, inplace=True )


# In[43]:


df1.head()


# In[44]:


df1[feature_scale] = sts.fit_transform(df1[feature_scale])


# In[45]:


df1.head()


# In[46]:


y_predicted = model_svc.predict(df1)


# In[47]:


submission = pd.DataFrame({
        "PassengerId": test['PassengerId'],
        "Survived": y_predicted
    })


# In[48]:


submission.to_csv('titanic_submission_v02.csv', index=False)

