import pandas as pd  # load csv's (pd.read_csv)
import numpy as np   # math (lin. algebra)

import sklearn as skl   # machine learning
from sklearn.ensemble import RandomForestClassifier
# from plotnine import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def get_train_data():
#下面两个文件路径替换为你电脑上该文件的路径
    train_path = r'E:\\HZNU_Study\\PyCharmProjects\\RandomForest1\\adult.data'
    test_path = r'E:\\HZNU_Study\\PyCharmProjects\\RandomForest1\\adult.test'
    columns = ['Age','Workclass','fnlgwt','Education','EdNum','MaritalStatus','Occupation',
               'Relationship','Race','Sex','CapitalGain','CapitalLoss','HoursPerWeek','Country','Income']
    df_train_set = pd.read_csv(train_path, names=columns)
    print(df_train_set.head())
    return df_train_set

def get_test_data():
    test_path = r'E:\\HZNU_Study\\PyCharmProjects\\RandomForest1\\adult.test'
    columns = ['Age','Workclass','fnlgwt','Education','EdNum','MaritalStatus','Occupation',
               'Relationship','Race','Sex','CapitalGain','CapitalLoss','HoursPerWeek','Country','Income']
    df_test_set = pd.read_csv(test_path, names=columns)
    print(df_test_set.head())
    return df_test_set

df_train_set=get_train_data()
df_test_set=get_test_data()

df_train_set.drop('fnlgwt', axis=1, inplace=True) # inplace=True：不创建新的对象，直接对原始对象进行修改；
                                                  # inplace=False：对数据进行修改，创建并返回新的对象承载其修改结果。
df_test_set.drop('fnlgwt', axis=1, inplace=True)

print(df_train_set.replace(' ?', np.nan).shape)
#(32561, 14)
print(df_train_set.replace(' ?', np.nan).dropna().shape)
# (30162, 15)
print(df_test_set.replace(' ?', np.nan).shape)
#(16282, 14)
print(df_test_set.replace(' ?', np.nan).dropna().shape)
# (15060, 15)

# 删除含有?（缺失行）
train_set = df_train_set.replace(' ?', np.nan).dropna()

test_set = df_test_set.replace(' ?', np.nan).dropna()

# 把测试语料集中的 Income 归一化
test_set['Income'] = test_set.Income.replace({' <=50K.': ' <=50K', ' >50K.': ' >50K'})
print(test_set.Income.unique())
# [' <=50K' ' >50K']
print(df_train_set.Income.unique())
# [' <=50K' ' >50K']

# 因为有 受教育的年数，所以这里不需要教育这一列
train_set.drop(["Education"], axis=1, inplace=True)
test_set.drop(["Education"], axis=1, inplace=True)


combined_set = pd.concat([train_set, test_set], axis=0)
for feature in combined_set.columns:
    if combined_set[feature].dtype == 'object':
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes

train_set = combined_set[:train_set.shape[0]]
test_set = combined_set[test_set.shape[0]:]
print(train_set.Workclass.unique())
print(test_set.Income.unique())

cols = list(train_set.columns)
cols.remove("Income")

x_train, y_train = train_set[cols].values, train_set["Income"].values
#测试集的输入和输出结果分开，用来校验模型的准确率
x_test, y_test = test_set[cols].values, test_set["Income"].values


treeClassifier = DecisionTreeClassifier()
treeClassifier.fit(x_train, y_train) # 训练模型
treeClassifier.score(x_test, y_test)
print(treeClassifier.score(x_test, y_test))
# 输出为：0.8700351435581195
y_pred = treeClassifier.predict(x_test)  # 用测试集做预测
print(classification_report(y_test, y_pred)) # 查看模型的预测值与真实值
