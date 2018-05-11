
# coding: utf-8

# In[2]:


get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
names = ("age, workclass, fnlwgt, education, education-num, "
         "marital-status, occupation, relationship, race, sex, "
         "capital-gain, capital-loss, hours-per-week, "
         "native-country, income").split(', ')    
data = pd.read_csv('d:/data/adult.data',names = names)
data = data.dropna() # pandas.DataFrame.dropna 可以删除数据框中的缺失值
target_names = data['income'].unique()
target = data['income']
features_data = data.drop('income', axis=1)
numeric_features = [c for c in features_data if features_data[c].dtype.kind in ('i', 'f')] # 提取数值类型为整数或浮点数的变量
numeric_data = features_data[numeric_features]
categorical_data = features_data.drop(numeric_features, 1)
categorical_data_encoded = categorical_data.apply(lambda x: pd.factorize(x)[0]) # pd.factorize即可将分类变量转换为数值表示
features = pd.concat([numeric_data, categorical_data_encoded], axis=1)
X = features.values.astype(np.float32) # 转换数据类型
y = (target.values == ' >50K').astype(np.int32) # 收入水平 ">50K" 记为1，“<=50K” 记为0
from sklearn.cross_validation import train_test_split # sklearn库中train_test_split函数可实现该划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0) # 参数test_size设置训练集占比


# In[3]:


from sklearn.cross_validation import cross_val_score
#高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d"
#  % (iris.data.shape[0],(iris.target != y_pred).sum()))
right = 0
wrong = 0
i = 0
for single in y_pred:
    if single == y_test[i]: 
        right = right + 1
    else:
        wrong = wrong + 1
    i = i + 1
print(right + wrong)
print(right/(right + wrong))
print(wrong/(right + wrong))


# In[8]:


# print(y_pred)
y_test.shape


# In[4]:


#多项分布朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
gnb = MultinomialNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d"
#  % (iris.data.shape[0],(iris.target != y_pred).sum()))
right = 0
wrong = 0
i = 0
for single in y_pred:
    if single == y_test[i]: 
        right = right + 1
    else:
        wrong = wrong + 1
    i = i + 1
print(right + wrong)
print(right/(right + wrong))
print(wrong/(right + wrong))


# In[13]:


#伯努利朴素贝叶斯
from sklearn.naive_bayes import BernoulliNB
gnb = BernoulliNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d"
#  % (iris.data.shape[0],(iris.target != y_pred).sum()))
right = 0
wrong = 0
i = 0
for single in y_pred:
    if single == y_test[i]: 
        right = right + 1
    else:
        wrong = wrong + 1
    i = i + 1
print(right + wrong)
print(right/(right + wrong))
print(wrong/(right + wrong))


# In[14]:


X_train.shape


# In[5]:


#ComplementNB
from sklearn.naive_bayes import ComplementNB
gnb = ComplementNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d"
#  % (iris.data.shape[0],(iris.target != y_pred).sum()))
right = 0
wrong = 0
i = 0
for single in y_pred:
    if single == y_test[i]: 
        right = right + 1
    else:
        wrong = wrong + 1
    i = i + 1
print(right + wrong)
print(right/(right + wrong))
print(wrong/(right + wrong))

