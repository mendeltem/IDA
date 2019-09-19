
# coding: utf-8

# # Linear Regression
# 
# In this tutorial we will implement a linear regression model. We will also implement a function that splits the available data into a training and a testting part.
# 
# ## Problem Setting
# 
# We will use the Boston Housing Dataset. This dataset contains information collected by the U.S Census Service concerning housing in the city of Boston in the state of Massachusetts in 1978. Our goal is to predict the median value of the houses in a particular town in the city of Boston given its attributes. Check the file ’housing.names’ for more information on the attributes.

# In[ ]:


import urllib
import pandas as pd
import numpy as np
# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

from sklearn.datasets import load_boston
boston=load_boston()
testfile = urllib.URLopener()
testfile.retrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names", "housing.names")
df=pd.DataFrame(boston.data)
df.columns=['crime_rate','res_land_zoned','industry','charles_river','nox','avg_num_rooms','prop_bf_1940','dst_emply_center','rd_highway_idx','tax_rate','stdnt_tchr_ratio','prop_blacks','low_status_pct']
X=boston.data
y=boston.target


# In[ ]:


df.head(10)


# ### Exercise 1
# 
# Write the *split_train_test(X,y,split,seed)*, given an instance matrix $X \in \mathbb{R}^{N \times D}$, labels $y \in Y^N$, a split ratio in $[0, 1]$ and a random seed $\in \mathbb{Z}$. Split the dataset in $(split×100)\%$ of the instances for training our model and the rest for testing, i.e. 
# 
# $$ \left|X_{\text{train}}\right| = \lceil \text{split} \cdot N \rceil, \qquad |X_{\text{train}}| + |X_{\text{test}}| = N. $$
# Make sure you use the given random number generator seed so we all get the same results. The function is supposed to return:
# 
# - X_train, y_train: the training instances and labels;
# - X_test, y_test: the test instances and labels,
# 
# in the same order as was mentioned.
# 
# Hint: It may be helpful to use shuffling functionality (e.g. np.random.shuffle).

# In[ ]:


def split_train_test(X,y,split,seed):
    ##################
    #INSERT CODE HERE#
    ##################
    return None # X_train, y_train, X_test, y_test


# ### Exercise 2
# 
# Write the function *train_linear_reg(X_train,y_train,lmbd)*.
# Implement the ridge regression model (slide 24). The function should output the learned weight vector $\theta \in \mathbb{R}^D$ or $\mathbb{R}^{D+1}$ (depending on whether you are adding *bias*).

# In[ ]:


def train_linear_reg(X, y, lmbd):
    ##################
    #INSERT CODE HERE#
    ##################
    return None # theta


# ### Exercise 3
# 
# Write the function *predict(X,theta)* which predicts housing values vector pred for a dataset X and a previously trained parameter vector $\theta$.

# In[ ]:


def predict(X, theta):
    ##################
    #INSERT CODE HERE#
    ##################
    return None # y_pred


# ### Exercise 4
# 
# Write the function *mean_abs_loss(y_true,y_pred)* which computes the mean of the absolute differences between our prediction vector $y\_pred$ and the real housing values $y\_true$.

# In[ ]:


def mean_abs_loss(y_true,y_pred):
    ##################
    #INSERT CODE HERE#
    ##################
    return 0


# ### Exercise 5
# 
# Evaluate your solutions by running the following code. 
# 
# Moreover, answer the following questions: What is the most important feature in your model? Are there features that are not so important? What happens if you remove them? Are there outliers with a high absolute loss?

# In[ ]:


seed = 3
lmbd=1
split=0.7
X_train,y_train,X_test,y_test=split_train_test(X,y,split,seed)
theta=train_linear_reg(X_train,y_train,lmbd)
y_pred=predict(X_test,theta)
mae=mean_abs_loss(y_test,y_pred)
print 'The mean absolute loss is {loss:0.3f}'.format(loss=mae*1000)

