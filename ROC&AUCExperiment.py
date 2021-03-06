import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
parameter = 50
data=pd.DataFrame(index=range(0,parameter),columns=('probability','The true label'))
data['The true label']=np.random.randint(0,2,size=len(data))
data['probability']=np.random.choice(np.arange(0.1,1,0.1),len(data['probability']))
print(data)
cm=np.arange(4).reshape(2,2)
cm[0,0]=len(data[data['The true label']==0][data['probability']<0.5])#TN
cm[0,1]=len(data[data['The true label']==0][data['probability']>=0.5])#FP
cm[1,0]=len(data[data['The true label']==1][data['probability']<0.5]) #FN
cm[1,1]=len(data[data['The true label']==1][data['probability']>=0.5])#TP
print(cm)
import itertools
classes = [0,1]
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
print(thresh)
for i, j in  itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    print(cm[i, j])
    print(thresh)
    #print("white" if (cm[i, j] <= thresh) else "black")
    #print("white" if (16 >= 8) else "black")
    print("white" if (cm[i, j] > thresh) else "black")
    plt.text(j, i, cm[i, j],
    horizontalalignment="center",
    color="red" if (cm[i, j] > thresh) else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
data.sort_values('probability',inplace=True,ascending=False)
print(data)
TPRandFPR=pd.DataFrame(index=range(len(data)),columns=('TP','FP'))
for j in range(len(data)):
    data1=data.head(n=j+1)
    FP=len(data1[data1['The true label']==0] [data1['probability']>=data1.head(len(data1))['probability']])/float(len(data[data['The true label']==0]))
    TP=len(data1[data1['The true label']==1][data1['probability']>=data1.head(len(data1))['probability']])/float(len(data[data['The true label']==1])) 
    TPRandFPR.iloc[j]=[TP,FP]
from sklearn.metrics import auc
AUC= auc(TPRandFPR['FP'],TPRandFPR['TP'])
plt.scatter(x=TPRandFPR['FP'],y=TPRandFPR['TP'],label='(FPR,TPR)',color='k')
plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], 'k',label='AUC = %0.2f'% AUC)
plt.legend(loc='lower right')
plt.title('Receiver Operating Characteristic')
plt.plot([(0,0),(1,1)],'r--')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,01.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
