import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt


# 读入数据
df = pd.read_csv("https://raw.githubusercontent.com/niyanchun/AI_Learning/master/SampleData/diabetes.csv")

X = df.values[:, :-1]
y = df.values[:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1024)

# 创建模型并训练
lr = LogisticRegression()
lr.fit(X_train, y_train)

# 在测试集上面预测
predictions = lr.predict(X_test)

# 求混淆矩阵、准确率、精确率、召回率
print("Confusion Matrix: ", metrics.confusion_matrix(y_test, predictions))
print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
print("Precision: ", metrics.precision_score(y_test,predictions))
print("Recall: ", metrics.recall_score(y_test, predictions))

# 画ROC图
pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, pred_proba)
auc = metrics.roc_auc_score(y_test, pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
