
# coding: utf-8

# In[5]:


get_ipython().magic('matplotlib inline')


# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[7]:


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


# In[17]:


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


# In[18]:


# 下方是svm
y_train[:50]


# In[19]:


from sklearn import svm
from sklearn.cross_validation import cross_val_score
C = 1  # SVM regularization parameter
clf = svm.SVC(kernel='rbf', gamma=0.07, C=C)

# 交叉验证，评价分类器性能，此处选择的评分标准是ROC曲线下的AUC值，对应AUC更大的分类器效果更好
scores = cross_val_score(clf, X_train[:1000], y_train[:1000], cv=5, scoring='roc_auc') 
print("ROC AUC Decision Tree: {:.4f} +/-{:.4f}".format(
    np.mean(scores), np.std(scores)))
# clf = clf.fit(X_train[:500], y_train[:500])


# In[ ]:


#不要做这个单元格的运算
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X_train[:200], y_train[:200]) for clf in models)
# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
# xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    #画出预测结果
    # plot_contours(ax, clf, xx, yy,
    #               cmap=plt.cm.coolwarm, alpha=0.8)
    #把原始点画上去                  
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()


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


def plot_point(dataArr,labelArr,Support_vector_index):  
    for i in range(np.shape(dataArr)[0]):  
        if labelArr[i] == 1:  
            plt.scatter(dataArr[i][0],dataArr[i][1],c='b',s=20)  
        else:  
            plt.scatter(dataArr[i][0],dataArr[i][1],c='y',s=20)  
      
    for j in Support_vector_index:  
        plt.scatter(dataArr[j][0],dataArr[j][1], s=100, c = '', alpha=0.5, linewidth=1.5, edgecolor='red')  
    plt.show()


# In[8]:


from sklearn import cross_validation,metrics
from sklearn import svm
# clf = svm.SVC(cache_size=200, class_weight=None, coef0=0.0,C=1.0,  
#         decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',  
#         max_iter=-1, probability=True, random_state=None, shrinking=True,  
#         tol=0.001, verbose=False) 
clf = svm.SVC(kernel="linear", C=0.025)
clf = clf.fit(X_train[:1000], y_train[:1000])
#预测X_test  
predict_list = clf.predict(X_train[1001:2000])  
#预测精度  
precition = clf.score(X_train[1001:2000],y_train[1001:2000])  
print('precition is : ',precition*100,"%")  
#获取模型返回值  
n_Support_vector = clf.n_support_#支持向量个数  
print("支持向量个数为： ",n_Support_vector)  
Support_vector_index = clf.support_#支持向量索引  

predict_prob_y = clf.predict_proba(X_train[1001:2000])#基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
result_y = np.array(y_train[1001:2000])
i = 0
for single in predict_prob_y:
#     print(single)
#     break
    if single[0] > single[1]:
        result_y[i] = 1
    else:
        result_y[i] = 0
    i = i + 1
print(y_train[1001:2000].shape)
print(predict_prob_y.shape)
print(result_y.shape)

#end svm ,start metrics 
test_auc = metrics.roc_auc_score(y_train[1001:2000],result_y)#验证集上的auc值

print (test_auc)

plot_point(X_train[:1000], y_train[:1000], Support_vector_index) 

#那么AUC应该接近0.5。


# In[52]:


# 交叉验证，评价分类器性能，此处选择的评分标准是ROC曲线下的AUC值，对应AUC更大的分类器效果更好
scores = cross_val_score(clf, X_train[:2000], y_train[:2000], cv=5, scoring='roc_auc') 
print("ROC AUC Decision Tree: {:.4f} +/-{:.4f}".format(
    np.mean(scores), np.std(scores)))


# In[53]:


from sklearn import cross_validation,metrics
clf = svm.SVC(cache_size=200, class_weight=None, coef0=0.0,C=1.0,  
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',  
        max_iter=-1, probability=True, random_state=None, shrinking=True,  
        tol=0.001, verbose=False) 
clf = clf.fit(X_train[:2000], y_train[:2000])
#预测X_test  
predict_list = clf.predict(X_train[2001:3000])  
#预测精度  
precition = clf.score(X_train[2001:3000],y_train[2001:3000])  
print('precition is : ',precition*100,"%")  
#获取模型返回值  
n_Support_vector = clf.n_support_#支持向量个数  
print("支持向量个数为： ",n_Support_vector)  
Support_vector_index = clf.support_#支持向量索引  

predict_prob_y = clf.predict_proba(X_train[2001:3000])#基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
result_y = np.array(y_train[2001:3000])
i = 0
for single in predict_prob_y:
#     print(single)
#     break
    if single[0] > single[1]:
        result_y[i] = 1
    else:
        result_y[i] = 0
    i = i + 1
print(y_train[2001:3000].shape)
print(predict_prob_y.shape)
print(result_y.shape)

#end svm ,start metrics 
test_auc = metrics.roc_auc_score(y_train[2001:3000],result_y)#验证集上的auc值

print (test_auc)

plot_point(X_train[:2000], y_train[:2000], Support_vector_index) 

#完全随机的样本，那么AUC应该接近0.5。


# In[55]:


from sklearn import cross_validation,metrics
clf = svm.SVC(cache_size=200, class_weight=None, coef0=0.0,C=1.0,  
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',  
        max_iter=-1, probability=True, random_state=None, shrinking=True,  
        tol=0.001, verbose=False) 
clf = clf.fit(X_train[:3000], y_train[:3000])
#预测X_test  
predict_list = clf.predict(X_train[3001:4000])  
#预测精度  
precition = clf.score(X_train[3001:4000],y_train[3001:4000])  
print('precition is : ',precition*100,"%")  
#获取模型返回值  
n_Support_vector = clf.n_support_#支持向量个数  
print("支持向量个数为： ",n_Support_vector)  
Support_vector_index = clf.support_#支持向量索引  

predict_prob_y = clf.predict_proba(X_train[3001:4000])#基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
result_y = np.array(y_train[3001:4000])
i = 0
for single in predict_prob_y:
#     print(single)
#     break
    if single[0] > single[1]:
        result_y[i] = 1
    else:
        result_y[i] = 0
    i = i + 1
print(y_train[3001:4000].shape)
print(predict_prob_y.shape)
print(result_y.shape)

#end svm ,start metrics 
test_auc = metrics.roc_auc_score(y_train[3001:4000],result_y)#验证集上的auc值

print (test_auc)

plot_point(X_train[:3000], y_train[:3000], Support_vector_index) 

#完全随机的样本，那么AUC应该接近0.5。

