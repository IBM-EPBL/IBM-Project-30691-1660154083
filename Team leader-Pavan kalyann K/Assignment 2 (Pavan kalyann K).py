#!/usr/bin/env python
# coding: utf-8

# # 1.Download the dataset: Dataset

# # 2. Load the dataset.

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


file=pd.read_csv("C:\Users\PAVAN KALYANN\Downloads\Churn_Modelling (1).csv")
df=pd.DataFrame(file)
df.head()


# In[4]:


df['HasCrCard'] = df['HasCrCard'].astype('category')


# In[5]:


df['IsActiveMember'] = df['IsActiveMember'].astype('category')
df['Exited'] = df['Exited'].astype('category')


# In[6]:


df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])


# In[7]:


df.head()


# # 3. Perform Below Visualizations.
# 

# ● Univariate Analysis

# ● Bi - Variate Analysis

# ● Multi - Variate Analysis

# In[9]:


import seaborn as sns
density = df['Exited'].value_counts(normalize=True).reset_index()
sns.barplot(data=density, x='index', y='Exited', );
density


# the data is significantly imbalanced

# In[12]:


import matplotlib.pyplot as plt


# In[13]:


categorical = df.drop(columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'])
rows = int(np.ceil(categorical.shape[1] / 2)) - 1

# create sub-plots anf title them
fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(10,6))
axes = axes.flatten()

for row in range(rows):
    cols = min(2, categorical.shape[1] - row*2)
    for col in range(cols):
        col_name = categorical.columns[2 * row + col]
        ax = axes[row*2 + col]       

        sns.countplot(data=categorical, x=col_name, hue="Exited", ax=ax);
        
plt.tight_layout()


# # 4. Perform descriptive statistics on the dataset

# In[15]:


df.info()


# In[16]:


df.describe()


# # 5. Handle the Missing values.

# In[18]:


df.isna().sum()


# there is no missing values in dataset

# In[20]:


for i in df:
    if df[i].dtype=='object' or df[i].dtype=='category':
        print("unique of "+i+" is "+str(len(set(df[i])))+" they are "+str(set(df[i])))


# # 6. Find the outliers and replace the outliers
# Checking for outliers

# In[22]:


def box_scatter(data, x, y):    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16,6))
    sns.boxplot(data=data, x=x, ax=ax1)
    sns.scatterplot(data=data, x=x,y=y,ax=ax2)


# In[23]:


box_scatter(df,'CreditScore','Exited');
plt.tight_layout()
print(f"# of Bivariate Outliers: {len(df.loc[df['CreditScore'] < 400])}")


# In[24]:


box_scatter(df,'Age','Exited');
plt.tight_layout()
print(f"# of Bivariate Outliers: {len(df.loc[df['Age'] > 87])}")


# In[25]:


box_scatter(df,'Balance','Exited');
plt.tight_layout()
print(f"# of Bivariate Outliers: {len(df.loc[df['Balance'] > 220000])}")


# In[26]:


box_scatter(df,'EstimatedSalary','Exited');
plt.tight_layout()


# Removing outliers

# In[28]:


for i in df:
    if df[i].dtype=='int64' or df[i].dtypes=='float64':
        q1=df[i].quantile(0.25)
        q3=df[i].quantile(0.75)
        iqr=q3-q1
        upper=q3+1.5*iqr
        lower=q1-1.5*iqr
        df[i]=np.where(df[i] >upper, upper, df[i])
        df[i]=np.where(df[i] <lower, lower, df[i])


# After removing outliers, boxplot will be like

# In[30]:


box_scatter(df,'CreditScore','Exited');
plt.tight_layout()
print(f"# of Bivariate Outliers: {len(df.loc[df['CreditScore'] < 400])}")


# In[31]:


box_scatter(df,'Age','Exited');
plt.tight_layout()
print(f"# of Bivariate Outliers: {len(df.loc[df['Age'] > 87])}")


# In[32]:


box_scatter(df,'Balance','Exited');
plt.tight_layout()
print(f"# of Bivariate Outliers: {len(df.loc[df['Balance'] > 220000])}")


# # 7. Check for Categorical columns and perform encoding.

# In[34]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
for i in df:
    if df[i].dtype=='object' or df[i].dtype=='category':
        df[i]=encoder.fit_transform(df[i])


# # 8. Split the data into dependent and independent variables.

# In[36]:


x=df.iloc[:,:-1]
x.head()


# In[37]:


y=df.iloc[:,-1]
y.head()


# # 9. Scale the independent variables

# In[39]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)


# In[40]:


x


# # 10. Split the data into training and testing

# In[42]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)


# In[43]:


x_train.shape


# In[44]:


x_test.shape


# In[45]:


y_train.shape


# In[46]:


y_test.shape


# In[ ]:




