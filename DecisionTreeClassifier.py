
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


plt.style.use('ggplot')


# In[4]:


names = ("age, workclass, fnlwgt, education, education-num, "
         "marital-status, occupation, relationship, race, sex, "
         "capital-gain, capital-loss, hours-per-week, "
         "native-country, income").split(', ')    
data = pd.read_csv('d:/data/adult.data',names = names)


# In[5]:


data = data.dropna() # pandas.DataFrame.dropna 可以删除数据框中的缺失值


# In[6]:


target_names = data['income'].unique()
target_names


# In[7]:


target = data['income']
features_data = data.drop('income', axis=1)


# In[8]:


numeric_features = [c for c in features_data if features_data[c].dtype.kind in ('i', 'f')] # 提取数值类型为整数或浮点数的变量
numeric_features


# In[9]:


numeric_data = features_data[numeric_features]
numeric_data.head(5)


# In[10]:


categorical_data = features_data.drop(numeric_features, 1)
categorical_data.head(5)


# In[11]:


categorical_data_encoded = categorical_data.apply(lambda x: pd.factorize(x)[0]) # pd.factorize即可将分类变量转换为数值表示
                                                                                # apply运算将转换函数应用到每一个变量维度
categorical_data_encoded.head(5)


# In[12]:


features = pd.concat([numeric_data, categorical_data_encoded], axis=1)
features.head()


# In[13]:


X = features.values.astype(np.float32) # 转换数据类型
y = (target.values == ' >50K').astype(np.int32) # 收入水平 ">50K" 记为1，“<=50K” 记为0


# In[14]:


X.shape


# In[15]:


y


# In[16]:


from sklearn.cross_validation import train_test_split # sklearn库中train_test_split函数可实现该划分

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0) # 参数test_size设置训练集占比


# In[21]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.cross_validation import cross_val_score
import graphviz # doctest: +SKIP

clf = DecisionTreeClassifier(max_depth=12) # 参数max_depth设置树最大深度

# 交叉验证，评价分类器性能，此处选择的评分标准是ROC曲线下的AUC值，对应AUC更大的分类器效果更好
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc') 
print("ROC AUC Decision Tree: {:.4f} +/-{:.4f}".format(
    np.mean(scores), np.std(scores)))
clf = clf.fit(X_train, y_train)
plot_learning_curve(clf, X_train, y_train, scoring='roc_auc')




# In[22]:


clf = DecisionTreeClassifier(max_depth=10) # 参数max_depth设置树最大深度

# 交叉验证，评价分类器性能，此处选择的评分标准是ROC曲线下的AUC值，对应AUC更大的分类器效果更好
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc') 
print("ROC AUC Decision Tree: {:.4f} +/-{:.4f}".format(
    np.mean(scores), np.std(scores)))
clf = clf.fit(X_train, y_train)
plot_learning_curve(clf, X_train, y_train, scoring='roc_auc')


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.cross_validation import cross_val_score
import graphviz # doctest: +SKIP

clf = DecisionTreeClassifier(max_depth=8) # 参数max_depth设置树最大深度

# 交叉验证，评价分类器性能，此处选择的评分标准是ROC曲线下的AUC值，对应AUC更大的分类器效果更好
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc') 
print("ROC AUC Decision Tree: {:.4f} +/-{:.4f}".format(
    np.mean(scores), np.std(scores)))
clf = clf.fit(X_train, y_train)
dot_data = tree.export_graphviz(clf, out_file=None, # doctest: +SKIP
                            feature_names=features.head(0).columns.values.tolist(),  # doctest: +SKIP
                            #class_names=["MORE THAN", "NO MORE THAN"],  # doctest: +SKIP
                            filled=True, rounded=True,  # doctest: +SKIP
                            special_characters=True)  # doctest: +SKIP
graph = graphviz.Source(dot_data)  # doctest: +SKIP
graph


# In[19]:


from sklearn.learning_curve import learning_curve


def plot_learning_curve(estimator, X, y, ylim=(0, 1.1), cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring=None):
    plt.title("Learning curves for %s" % type(estimator).__name__)
    plt.ylim(*ylim); plt.grid()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    print("Best validation score: {:.4f}".format(validation_scores_mean[-1]))


# In[ ]:


clf = DecisionTreeClassifier(max_depth=None)
plot_learning_curve(clf, X_train, y_train, scoring='roc_auc')
# 可以注意到训练数据和交叉验证数据的得分有很大的差距，意味着可能过度拟合训练数据了


# In[ ]:


clf = tree.DecisionTreeClassifier(max_depth=2)
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc') 
print("ROC AUC Decision Tree: {:.4f} +/-{:.4f}".format(
    np.mean(scores), np.std(scores)))
clf = clf.fit(X_train, y_train)
plot_learning_curve(clf, X_train, y_train, scoring='roc_auc')
clf = clf.fit(X_train, y_train)
dot_data = tree.export_graphviz(clf, out_file=None, # doctest: +SKIP
                            feature_names=features.head(0).columns.values.tolist(),  # doctest: +SKIP
                            #class_names=["MORE THAN", "NO MORE THAN"],  # doctest: +SKIP
                            filled=True, rounded=True,  # doctest: +SKIP
                            special_characters=True)  # doctest: +SKIP
graph = graphviz.Source(dot_data)  # doctest: +SKIP
graph
graph.render("adultTreefor2")


# In[ ]:


clf = DecisionTreeClassifier(max_depth=8)
plot_learning_curve(clf, X_train, y_train, scoring='roc_auc')


# In[ ]:


clf = DecisionTreeClassifier(max_depth=1)
plot_learning_curve(clf, X_train, y_train, scoring='roc_auc')


# In[ ]:


#截止以上，决策树实验已经做得完成了
#下面准备可视化决策树


# In[ ]:


graph = graphviz.Source(dot_data) # doctest: +SKIP


# In[ ]:


graph.render("adultTree") # doctest: +SKIP


# In[ ]:


dot_data = tree.export_graphviz(clf, out_file=None, # doctest: +SKIP
                            feature_names=features.head(0).columns.values.tolist(),  # doctest: +SKIP
                            #class_names=["MORE THAN", "NO MORE THAN"],  # doctest: +SKIP
                            filled=True, rounded=True,  # doctest: +SKIP
                            special_characters=True)  # doctest: +SKIP
graph = graphviz.Source(dot_data)  # doctest: +SKIP
graph
graph.render("adultTree1")


# In[ ]:


# X = features.values.astype(np.float32) # 转换数据类型
# y = (target.values == ' >50K').astype(np.int32) # 收入水平 ">50K" 记为1，“<=50K” 记为0
# target
graph = graphviz.Source(dot_data)  # doctest: +SKIP

# from IPython.display import Image
# Image(graph.create_png())


# In[ ]:


# a99 = np.array(features.head(0))
a99 = features.head(0).columns.values.tolist()
# type(a99)
print(a99)
# type(features.head(0))


# In[ ]:


type(target)


# In[ ]:


target.values


# In[ ]:


# 下方是svm
y_train[:50]


# In[20]:


from sklearn import svm
from sklearn.cross_validation import cross_val_score
clf = svm.SVC()
# 交叉验证，评价分类器性能，此处选择的评分标准是ROC曲线下的AUC值，对应AUC更大的分类器效果更好
scores = cross_val_score(clf, X_train[:10], y_train[:10], cv=5, scoring='roc_auc') 
print("ROC AUC Decision Tree: {:.4f} +/-{:.4f}".format(
    np.mean(scores), np.std(scores)))
# clf = clf.fit(X_train[:10], y_train[:10])

