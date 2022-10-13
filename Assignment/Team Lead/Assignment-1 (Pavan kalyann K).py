#!/usr/bin/env python
# coding: utf-8

# # Basic Python

# ## 1. Split this string

# In[ ]:


s = "Hi there Sam!"


# In[3]:


s = 'Hi there Sam!'
s.split()


# *`italicized text`*## 2. Use .format() to print the following string. 
# 
# ### Output should be: The diameter of Earth is 12742 kilometers.

# In[ ]:


planet = "Earth"
diameter = 12742


# In[2]:



planet = "Earth"
diameter = 12742
print("The diameter of {} is {} kilometers.".format(planet,diameter))


# ## 3. In this nest dictionary grab the word "hello"

# In[ ]:


d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}


# In[7]:


d['k1'][3]['tricky'][3]['target'][3]


# # Numpy

# In[8]:


import numpy as np


# ## 4.1 Create an array of 10 zeros? 
# ## 4.2 Create an array of 10 fives?

# In[11]:


np.zeros(10)


# In[13]:


np.ones(10) * 5


# ## 5. Create an array of all the even integers from 20 to 35

# In[15]:


import numpy as np
array=np.arange(20,35)
print(array)


# ## 6. Create a 3x3 matrix with values ranging from 0 to 8

# In[ ]:


np.arange(0,9).reshape((3,3))


# ## 7. Concatinate a and b 
# ## a = np.array([1, 2, 3]), b = np.array([4, 5, 6])

# In[18]:


a= np.array([[1, 2, 3], [4, 5, 6]])


# # Pandas

# ## 8. Create a dataframe with 3 rows and 2 columns

# In[ ]:


import pandas as pd


# In[20]:


import numpy as np
A = np.random.randint(10, size=(3,2))


# ## 9. Generate the series of dates from 1st Jan, 2023 to 10th Feb, 2023

# In[21]:


import pandas
pandas.date_range(sdate,edate-timedelta(days=1),freq='d')


# ## 10. Create 2D list to DataFrame
# 
# lists = [[1, 'aaa', 22],
#          [2, 'bbb', 25],
#          [3, 'ccc', 24]]

# In[ ]:


lists = [[1, 'aaa', 22], [2, 'bbb', 25], [3, 'ccc', 24]]


# In[22]:


import pandas as pd.
import numpy as np.
arr= np.array([[1, 'aaa', 22], [2, 'bbb', 25], [3, 'ccc', 24]])
df=pd.DataFrame(arr)
print(df)


# In[ ]:




