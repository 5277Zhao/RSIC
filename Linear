from pandas import DataFrame, ExcelFile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold,LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from sklearn.metrics import explained_variance_score,accuracy_score,precision_score,recall_score,f1_score
import sklearn.model_selection as ms
import seaborn as sns
from joblib import dump,load
import itertools


#打印当前工作目录
print("当前目录：",os.getcwd())
#切换到指定目录
os.chdir('E:\data')
#打印切换后的工作目录
print("切换后的工作目录：",os.getcwd())
# ==================== ==================== 机器学习部分 ==================== ====================
# 1.获取数据
data = pd.read_excel(r'E:\data\grain composition.xlsx')
X = data.iloc[:, 1:7]
Y = data.iloc[:, 8:9]

# 2.数据集划分
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=3)

# 3.特征工程-标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)



# 4.机器学习-线性回归(正规方程)
#4.1实例化一个估计器
estimator = LinearRegression()

"""
#4.2 网格搜索交叉验证

gird = {"random_state":range(1,100)}
estimator = GridSearchCV(estimator,param_grid=gird,n_jobs=1,refit=True,cv=10)
"""
#4.3模型训练
estimator.fit(x_train, y_train)


# 5.模型评估
# 5.1 获取系数等值
y_pre = estimator.predict(x_test)
print("预测值为:\n", y_pre)
#print("预测值和真实值的对比：\n",y_pre==y_test)
print("模型中的系数为:\n", estimator.coef_)
print("模型中的偏置为:\n", estimator.intercept_)
#net = estimator.score(x_test,y_test)
#print("准确率是：\n",net)

#模型的保存//模型保存到data所属文件夹中 记得复制到python文件所属文件夹或者保存模型的时候指定路径
dump(estimator,'estimator.joblib')

# 5.2 评价
# 均方误差
MSE = mean_squared_error(y_test, y_pre)
print("均方误差为:\n", MSE)
#平均绝对误差
MAE = mean_absolute_error(y_test, y_pre)
print("平均绝对误差为:\n", MAE)

#均方根误差
RMSE_test = MSE**0.5
print("均方根误差为:\n", RMSE_test)
#决定系数
R2 = r2_score(y_test,y_pre)
print("决定系数为:\n", R2)
"""
# 6.预测未知数据
data_p = pd.read_excel(r'E:\data\pred.xlsx')
X1 = data_p.iloc[:, 1:7]
Y2 = data_p.iloc[:, 8:9]
xtest = transfer.fit_transform(X1)
y_pred = estimator.predict(xtest)
print("预测结果为：\n",y_pred)
"""
"""
#绘制相对误差图
error =abs(y_pre-y_test)
print("误差为：\n",error)
fig, ax = plt.subplots()
sns.distplot(error, ax=ax, bins=10, kde=True, hist=True)
ax.set_xlabel('Error')
ax.set_ylabel('Density')
ax.set_title('Error Distribution')
plt.show()
"""
"""
# ====================使用 scatter 方法绘制实际值与预测值的散点图，以便直观地比较模型的预测效果 ====================
#将数字转化为数组，该数组包含从1~最大样本编号的所有值
#num_pre = np.arange(1, y_pre.shape[0]+1, 1)
#num_test = np.arange(1, y_test.shape[0]+1, 1)
# 裁剪y数组，使其长度与x数组一致
y_pre = y_pre[:len(y_test)]

plt.scatter(y_test, y_pre,c = 'yellowgreen')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Porosity:Actual vs Predicted')
plt.plot([0.2, 0.45], [0.2, 0.45], color='red', linestyle='--')
plt.show()
"""
"""

# 2.划分（加交叉验证取最佳随机种子）
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=22)
#交叉验证
cv_score = cross_val_score(train_test_split,x_train,y_train,cv=10,scoring="accuracy")
print("交叉验证分数:",cv_score)
print("平均准确率:%.5f"%(cv_score.mean()))
#print("最好的模型:\n",estimator.get_params)
#print("最好的结果:\n",estimator._get_scorers)
#print("整体模型结果:\n",estimator.cv_results_)
"""
