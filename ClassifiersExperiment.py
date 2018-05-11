
# coding: utf-8

# In[ ]:


get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
names = ("age, workclass, fnlwgt, education, education-num, "
         "marital-status, occupation, relationship, race, sex, "
         "capital-gain, capital-loss, hours-per-week, "
         "native-country, income").split(', ')   



# 下面开始对样本进行处理
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
X_train = features.values.astype(np.float32) # 转换数据类型
y_train = (target.values == ' >50K').astype(np.int32) # 收入水平 ">50K" 记为1，“<=50K” 记为0


# 下面开始对测试集进行处理
data2 = pd.read_csv('d:/data/adult.test',names = names)
data2 = data2.dropna() # pandas.DataFrame.dropna 可以删除数据框中的缺失值
target_names2 = data2['income'].unique()
target2 = data2['income']
features_data2 = data2.drop('income', axis=1)
numeric_features2 = [c2 for c2 in features_data2 if features_data2[c2].dtype.kind in ('i', 'f')] # 提取数值类型为整数或浮点数的变量
numeric_data2 = features_data2[numeric_features2]
categorical_data2 = features_data2.drop(numeric_features2, 1)
categorical_data_encoded2 = categorical_data2.apply(lambda x: pd.factorize(x)[0]) # pd.factorize即可将分类变量转换为数值表示
features2 = pd.concat([numeric_data2, categorical_data_encoded2], axis=1)
X_test = features2.values.astype(np.float32) # 转换数据类型
y_test = (target2.values == ' >50K').astype(np.int32) # 收入水平 ">50K" 记为1，“<=50K” 记为0





# from sklearn.cross_validation import train_test_split # sklearn库中train_test_split函数可实现该划分
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=0) # 参数test_size设置训练集占比


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


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#from sklearn.model_selection import train_test_split #废弃！！
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import BernoulliRBM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcess
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

h = .02  # step size in the mesh

clf_names = ["Nearest Neighbors", 
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes", "QDA", "Neural Net", ]

clf_names_svm = ["Linear SVM", "RBF SVM", ]

clf_names_no_use = ["Gaussian Process",]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=8),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
#     GaussianNB(),
    BernoulliNB(),
    QuadraticDiscriminantAnalysis(),
    #BernoulliRBM(),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    ]

classifiers_svm = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    ]

classifiers_no_use = [
    GaussianProcess(),
]

for clf_name, clf in zip(clf_names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(clf_name)
    print(score)
    
for clf_name, clf in zip(clf_names_svm, classifiers_svm):
    clf.fit(X_train[:100], y_train[:100])
    score = clf.score(X_test, y_test)
    print(clf_name)
    print(score)
    
# for clf_name, clf in zip(clf_names_no_use, classifiers_no_use):
#     clf.fit(X_train[:1000], y_train[:1000])
#     score = clf.score(X_test, y_test)
#     print(clf_name)
#     print(score)
    
# plt.tight_layout()
# plt.show()


# In[ ]:


X_test.shape


# In[2]:


def plot_point(dataArr,labelArr,Support_vector_index):  
    for i in range(np.shape(dataArr)[0]):  
        if labelArr[i] == 1:  
            plt.scatter(dataArr[i][0],dataArr[i][1],c='b',s=20)  
        else:  
            plt.scatter(dataArr[i][0],dataArr[i][1],c='y',s=20)  
      
    for j in Support_vector_index:  
        plt.scatter(dataArr[j][0],dataArr[j][1], s=100, c = '', alpha=0.5, linewidth=1.5, edgecolor='red')  
    plt.show()


# In[ ]:


from sklearn.svm import SVC
clf_names_svm2 = ["Linear SVM", "RBF SVM default", "sigmoid SVM",]
classifiers_svm2 = [
    SVC(kernel="linear"),
    SVC(kernel="rbf"),
    SVC(kernel="sigmoid"),
    ]
for clf_name, clf in zip(clf_names_svm2, classifiers_svm2):
    clf.fit(X_train[:1000], y_train[:1000])
    score = clf.score(X_test, y_test)
    print(clf_name)
    print(score)
    n_Support_vector = clf.n_support_#支持向量个数  
    print("支持向量个数为： ",n_Support_vector)  
    Support_vector_index = clf.support_#支持向量索引 
    plot_point(X_train[:1000], y_train[:1000], Support_vector_index) 


# In[3]:


from sklearn.svm import SVC
clf_names_svm2 = [ "RBF SVM default", "sigmoid SVM",]
classifiers_svm2 = [
    SVC(kernel="rbf"),
    SVC(kernel="sigmoid"),
    ]
for clf_name, clf in zip(clf_names_svm2, classifiers_svm2):
    clf.fit(X_train[:1000], y_train[:1000])
    score = clf.score(X_test, y_test)
    print(clf_name)
    print(score)
    n_Support_vector = clf.n_support_#支持向量个数  
    print("支持向量个数为： ",n_Support_vector)  
    Support_vector_index = clf.support_#支持向量索引 
    plot_point(X_train[:1000], y_train[:1000], Support_vector_index) 


# In[4]:


X_test.shape


# In[3]:


X_train.shape

